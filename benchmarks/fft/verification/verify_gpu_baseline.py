from __future__ import annotations

"""Algorithm-fidelity tests for the three GPU-derived BASELINE planners
(planning/gpu_baseline/{clfft,rocfft,vkfft}.py) -- section 5 of the task
these were built from: "Create tests that test ALGORITHM FIDELITY, not
merely correctness." These check that each baseline reproduces its real
GPU source's own exact tables/formulas/pruning rules (verified against
the research reports each module cites), not merely that a plan gets
built.

A second pass (`verify_numeric_roundtrip`) spot-checks that at least one
OK-mapped candidate per baseline actually computes a correct FFT once
lowered through this project's own already-trusted execution harness
(verification/verify_fft_harness.py) -- i.e. that a baseline's own radix
sequence, once accepted by the M2NDP translation layer, is genuinely
correct math, not just a plan object that happens to construct without
raising.

Run directly: `python3 verify_gpu_baseline.py` from benchmarks/fft/.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from planning.strategies.fft_plan_recursive import FFTLeafPlan
from planning.gpu_baseline import clfft, rocfft, rocfft_default, vkfft
from planning.gpu_baseline.common import BaselineStatus, GPUKernelConfig, map_cooperative_kernel
from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from verification.verify_fft_harness import Ptr, run_kernel

_FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        _FAILURES.append(message)
        print(f"  FAIL: {message}")


# ============================================================================
# clFFT
# ============================================================================

# (length): (radices, workgroup_size, num_transforms) -- from generator.
# stockham.cpp's real single-precision KernelCoreSpecs table, verified
# against the research report's own source citations.
_CLFFT_EXPECTED_TABLE = {
    2: ((2,), 64, 64),
    4: ((2, 2), 64, 32),
    8: ((4, 2), 64, 32),
    16: ((4, 4), 64, 16),
    32: ((8, 4), 64, 16),
    64: ((4, 4, 4), 64, 4),
    128: ((8, 4, 4), 64, 4),
    256: ((4, 4, 4, 4), 64, 1),
    512: ((8, 8, 8), 64, 1),
    1024: ((8, 8, 4, 4), 128, 1),
    2048: ((8, 8, 8, 4), 256, 1),
    4096: ((8, 8, 8, 8), 256, 1),
}

# (length): (leastNumPerWI, maxWorkGroupSize) -- DetermineSizes's mixed-
# prime branch, every case from the research report's own worked table.
_CLFFT_DETERMINE_SIZES_MIXED_CASES = {
    12: (12, 128), 24: (12, 128),        # 2*3, divisible by 12
    18: (6, 256),                         # 2*3, not divisible by 12
    20: (20, 64), 40: (20, 64),          # 2*5, divisible by 20
    10: (10, 128),                        # 2*5, not divisible by 20
    14: (14, 64),                         # 2*7
    15: (15, 128),                        # 3*5
    21: (21, 128),                        # 3*7
    35: (35, 64),                         # 5*7
    30: (30, 64),                         # 2*3*5
    42: (42, 60),                         # 2*3*7
    70: (70, 36),                         # 2*5*7
    105: (105, 24),                       # 3*5*7
    22: (22, 128),                        # 2*11
    26: (26, 128),                        # 2*13
    2 * 3 * 11: (210, 12),                # fallback (no explicit rule matches)
}


def verify_clfft_table() -> None:
    print("clFFT: exact KernelCoreSpecs table selections")
    for length, (expected_radices, expected_wgs, expected_nt) in _CLFFT_EXPECTED_TABLE.items():
        choice = clfft.get_radices(length)
        check(choice.source == "table", f"clfft N={length}: expected table source, got {choice.source}")
        check(
            choice.radices == expected_radices,
            f"clfft N={length}: radices {choice.radices} != expected {expected_radices}",
        )
        check(
            choice.workgroup_size == expected_wgs and choice.num_transforms == expected_nt,
            f"clfft N={length}: (wgs,nt)=({choice.workgroup_size},{choice.num_transforms}) "
            f"!= expected ({expected_wgs},{expected_nt})",
        )
    print(f"  {len(_CLFFT_EXPECTED_TABLE)} table entries checked")


def verify_clfft_determine_sizes() -> None:
    print("clFFT: DetermineSizes mixed-prime branch")
    for length, (expected_lnpi, expected_max_wgs) in _CLFFT_DETERMINE_SIZES_MIXED_CASES.items():
        least_num_per_wi, max_wgs = clfft._mixed_prime_least_num_per_wi(length, clfft._factor_supported_primes(length))
        check(
            (least_num_per_wi, max_wgs) == (expected_lnpi, expected_max_wgs),
            f"clfft DetermineSizes N={length}: got ({least_num_per_wi},{max_wgs}), "
            f"expected ({expected_lnpi},{expected_max_wgs})",
        )
    print(f"  {len(_CLFFT_DETERMINE_SIZES_MIXED_CASES)} mixed-prime cases checked")

    # Unsupported-prime length must raise, not silently produce a bogus answer.
    try:
        clfft.determine_sizes(2 * 19)
        check(False, "clfft DetermineSizes: length with unsupported prime 19 should raise")
    except clfft.ClfftUnsupportedLengthError:
        pass


def verify_clfft_large1d_split_exact() -> None:
    """Golden values from docs/gpu_baseline_clfft_large1d_split_deepdive.md's
    own worked examples -- an EXACT_SOURCE_SELECTION port (not merely
    ALGORITHM_EQUIVALENT), confirmed byte-for-byte against the real clFFT
    plan.cpp large-1D split logic (BitScanF bit-balancing + the literal
    490-entry non-po2 table)."""
    print("clFFT: EXACT large-1D split selection (BitScanF + 490-entry table)")
    threshold = clfft.get_max_1d_length()
    check(threshold == 4096, f"clFFT single-precision Large1DThreshold should be 4096, got {threshold}")
    for length, expected_a, expected_b in (
        (8192, 128, 64),
        (16384, 256, 64),
        (100000, 200, 500),
        (5040, 40, 126),
        (45360, 180, 252),
        (262144, 4096, 64),
        (1048576, 1024, 1024),
    ):
        a, b = clfft.choose_large1d_split(length, threshold)
        check(
            (a, b) == (expected_a, expected_b),
            f"clfft.choose_large1d_split({length}) = ({a},{b}), expected ({expected_a},{expected_b})",
        )
    # A large prime: real clFFT itself has no valid split (recurses forever
    # in the real source) -- this baseline must raise, not silently pick
    # a "helpful" M2NDP-friendly answer clFFT itself never produces.
    try:
        clfft.choose_large1d_split(1000003, threshold)
        check(False, "choose_large1d_split(1000003) should raise (clFFT itself has no valid split here)")
    except clfft.ClfftUnsupportedLengthError:
        pass


def verify_clfft_mapping_findings() -> None:
    """Not a pass/fail check -- a documented, printed record of WHICH
    lengths map onto M2NDP at all, per section 5's "do NOT hide failed
    GPU-baseline configurations. A failure itself is a scientifically
    useful baseline result." """
    print("clFFT: M2NDP mapping outcome per table length (informational)")
    for length in sorted(_CLFFT_EXPECTED_TABLE):
        result = clfft.plan_single_kernel(length, total_ffts=4)
        print(f"  N={length:5d}: {result.status.value}")


# ============================================================================
# rocFFT
# ============================================================================


def verify_rocfft_factorize() -> None:
    print("rocFFT: Factorize / GetMaxRadicesSize / SupportedThreadsPerTransform")
    factors_24 = rocfft.factorize(24)
    expected_24 = {(2, 2, 2, 3), (2, 2, 6), (2, 3, 4), (3, 8), (4, 6)}
    check(set(factors_24) == expected_24, f"Factorize(24) = {set(factors_24)}, expected {expected_24}")

    check(
        rocfft.get_max_radices_size(rocfft.factorize(24), length=24) == 4,
        "GetMaxRadicesSize(24): min factor count is 2 (e.g. (4,6)), +2 = 4",
    )
    # length==336's hardcoded -1 exception (tuning_kernel_tuner.cpp lines 499-500).
    normal = rocfft.get_max_radices_size(rocfft.factorize(300), length=300)
    special = rocfft.get_max_radices_size(rocfft.factorize(336), length=336)
    min_336 = min(len(f) for f in rocfft.factorize(336))
    check(special == min_336 + 2 - 1, f"GetMaxRadicesSize(336) should be (min+2)-1={min_336 + 1}, got {special}")

    tpts = rocfft.supported_threads_per_transform((2, 3, 4))
    check(tpts == [2, 3, 4, 6, 8, 12, 24], f"SupportedThreadsPerTransform((2,3,4)) = {tpts}")


def verify_rocfft_utilization() -> None:
    print("rocFFT: GetUtilizationRate rejection rule (avg<1.0 or max(heights)>8.0)")
    # length=24, factors=(2,3,4), tpt=4: heights = [3.0, 2.0, 1.5], avg=2.1667 -- not bad.
    check(not rocfft.is_bad_utilization(24, (2, 3, 4), 4), "24/(2,3,4)/tpt=4 should NOT be bad utilization")
    # A very large tpt relative to length drives avg well under 1.0.
    check(rocfft.is_bad_utilization(24, (2, 3, 4), 24), "24/(2,3,4)/tpt=24 (avg<1.0) should be bad utilization")
    # A tiny tpt on a large length drives some height over 8.0.
    check(rocfft.is_bad_utilization(1024, (2, 2, 2, 2, 2, 2, 2, 2, 2, 2), 1), "tpt=1 on N=1024 should exceed height=8.0")


def verify_rocfft_tpb() -> None:
    print("rocFFT: DeriveMaxTPB / ConservativeMaxTPB")
    # length=64, half_lds=False: bytes_per_batch=64*8=512, tpb=32768/512=64,
    # then clamp to wgs_bound.
    tpb = rocfft.derive_max_tpb(64, half_lds=False, tpt=4, wgs_bound=256)
    check(tpb * 4 <= 256, f"derive_max_tpb: tpt*tpb={tpb * 4} must not exceed wgs_bound=256")
    tpb_half = rocfft.derive_max_tpb(64, half_lds=True, tpt=4, wgs_bound=256)
    check(tpb_half >= tpb, "half_lds=True should never derive a SMALLER max tpb than half_lds=False")
    conservative = rocfft.conservative_max_tpb(1024)
    conservative_small = rocfft.conservative_max_tpb(64)
    check(
        conservative == (32768 // (1024 * 8)) + 1,
        f"ConservativeMaxTPB(1024) should include the +1 fudge for length>=1024, got {conservative}",
    )
    check(
        conservative_small == 32768 // (64 * 8),
        f"ConservativeMaxTPB(64) should NOT include the +1 fudge, got {conservative_small}",
    )


def verify_rocfft_phase1_permutations() -> None:
    print("rocFFT: Phase-1 permutation generation (cyclic-shift fallback above 6)")
    small = rocfft.get_all_factorizations_for_phase1((2, 3, 4))
    check(len(small) == 5, f"(2,3,4) has 3!=6 total perms, minus the original = 5, got {len(small)}")
    check((2, 3, 4) not in small, "the original ascending order must not reappear in the plain-permutation branch")

    big = (2, 2, 3, 5, 7)  # 5!/2! = 60 distinct perms > 6 -> cyclic-shift fallback
    shifted = rocfft.get_all_factorizations_for_phase1(big)
    check(len(shifted) == 2 * len(big), f"cyclic-shift fallback should produce 2*len={2 * len(big)}, got {len(shifted)}")
    check(shifted[0] == big, "cyclic-shift fallback's first entry is shift-0 of the original sequence itself")
    for perm in shifted:
        check(sorted(perm) == sorted(big), f"cyclic-shifted {perm} must be a permutation of {big}")


def verify_rocfft_winner_selection() -> None:
    print("rocFFT: winner selection uses real measured time, never estimate_cost")

    def fake_bench(plan, target):
        # Deterministic synthetic "measurement" -- exercises the full
        # phase0 -> propagate-best-3 -> phase1 -> winner pipeline without
        # needing the real M2NDP toolchain present in this environment.
        return rocfft.BenchmarkOutcome(ok=True, ndp_cycles=plan.length * 7 + len(plan.stages) * 13, spill_free=True)

    result, outcomes = rocfft.tune(24, total_ffts=4, benchmark_fn=fake_bench)
    check(result.status is BaselineStatus.OK, f"rocFFT tune(24) should find an OK winner, got {result.status}")
    check(len(outcomes) > 0, "rocFFT tune(24) should record every candidate considered, not just the winner")
    ok_mapped = [o for o in outcomes if o.mapping.status is BaselineStatus.OK]
    check(len(ok_mapped) > 0, "rocFFT tune(24) should have at least one OK-mapped-onto-M2NDP candidate")
    if result.status is BaselineStatus.OK:
        winner_cycles = result.ndp_cycles
        for o in ok_mapped:
            if o.benchmark is not None and o.benchmark.ok:
                check(
                    winner_cycles <= o.benchmark.ndp_cycles,
                    "rocFFT winner must have the lowest real measured ndp_cycles among all benchmarked candidates",
                )


# ============================================================================
# VkFFT
# ============================================================================


def verify_vkfft_register_classification() -> None:
    print("VkFFT: register-per-thread good/bad sequence classification")
    max_r, min_r, is_good = vkfft.registers_per_thread_for({2: 1, 3: 1, 5: 1, 7: 1})
    check(max_r == 7 and min_r == 5, f"2&3&5&7 branch: expected max=7 min=5, got max={max_r} min={min_r}")
    check(is_good == (not (max_r > 16 or max_r >= 2 * min_r)), "isGoodSequence must match the exact source rule")
    check(is_good is True, f"7<=16 and 7<2*5=10, so this sequence should be classified good, got {is_good}")


def verify_vkfft_pow2_grouping() -> None:
    print("VkFFT: power-of-two stage/radix grouping determinism")
    for length in (64, 256, 1024, 4096, 65536):
        radix = vkfft.choose_pow2_grouping_radix(length, max_rhs=4)
        check(radix in (2, 4, 8), f"grouping radix for N={length} must be 2, 4, or 8 (fixMaxCheckRadix2=3), got {radix}")
        seq = vkfft.pow2_radix_sequence(length, max_rhs=4)
        product = 1
        for r in seq:
            product *= r
        check(product == length, f"pow2_radix_sequence({length}) = {seq} does not multiply back to {length}")
        for r in seq:
            check(r in {2, 4, 8}, f"pow2_radix_sequence({length}) used radix {r} outside {{2,4,8}}")


def verify_vkfft_axis_splitting() -> None:
    print("VkFFT: axis splitting (pow2 power-of-8 preference, non-pow2 sqrt/cbrt search)")
    target = vkfft.DEFAULT_TARGET_PROFILE
    a, b = vkfft.split_pow2_2pass(8192, target)
    check(a * b == 8192, f"split_pow2_2pass(8192) = ({a},{b}) does not multiply back to 8192")
    check(a >= b, "split_pow2_2pass must return the larger factor first")

    split = vkfft.split_non_pow2_2pass(8 * 3 * 5 * 7, target)
    check(split is not None, "split_non_pow2_2pass(840) should find a legal divisor pair")
    if split is not None:
        a2, b2 = split
        check(a2 * b2 == 8 * 3 * 5 * 7, f"split_non_pow2_2pass(840) = {split} does not multiply back to 840")


def verify_vkfft_direct_radix_and_bluestein_boundary() -> None:
    print("VkFFT: direct-radix decomposition + Rader/Bluestein boundary")
    seq = vkfft.direct_radix_sequence(2 * 3 * 5 * 7)
    product = 1
    for r in seq:
        product *= r
    check(product == 2 * 3 * 5 * 7, f"direct_radix_sequence(210) = {seq} does not multiply back to 210")

    try:
        vkfft.direct_radix_sequence(23)  # a prime outside the direct-radix vocabulary
        check(False, "direct_radix_sequence(23) should raise (needs Rader/Bluestein, out of scope)")
    except vkfft.VkfftUnsupportedError:
        pass


# ============================================================================
# Numeric round-trip (spot check): an OK-mapped single-leaf candidate from
# each baseline must compute a genuinely correct FFT, not just construct.
# ============================================================================


def _run_reference_check(plan_root_leaf: FFTLeafPlan, *, label: str) -> None:
    """`total` here is the number of INDEPENDENT logical FFTs the DRAM
    buffer actually holds -- for a cooperative plan (see CooperationPlan),
    that is `kernel.total_uthreads // workers_per_fft` (many physical
    microthreads share one logical FFT's row via `logical_fft_id =
    global_uthread_id() // workers_per_fft`), NOT `kernel.total_uthreads`
    itself (the physical microthread count) -- mirroring verification.
    verify_fft_cooperative.verify_cooperative_leaf's own buffer sizing
    convention (`n = length * total_ffts`, not `length * total_uthreads`).
    """
    kernel = plan_root_leaf.kernel
    n = kernel.length
    workers_per_fft = kernel.cooperation.workers_per_fft if kernel.cooperation is not None else 1
    total = kernel.total_uthreads // workers_per_fft
    rng = np.random.default_rng(0)
    x_real = rng.standard_normal((total, n))
    x_imag = rng.standard_normal((total, n))

    input_real, input_imag = Ptr(total * n), Ptr(total * n)
    output_real, output_imag = Ptr(total * n), Ptr(total * n)
    input_real.arr[:] = x_real.reshape(-1)
    input_imag.arr[:] = x_imag.reshape(-1)

    run_kernel(kernel, input_real=input_real, input_imag=input_imag, output_real=output_real, output_imag=output_imag)

    got = (output_real.arr + 1j * output_imag.arr).reshape(total, n)
    signal = x_real + 1j * x_imag
    expected = np.fft.ifft(signal, axis=1) * n if kernel.inverse else np.fft.fft(signal, axis=1)
    max_err = np.max(np.abs(got - expected))
    check(max_err < 1e-2, f"{label}: numeric round-trip max error {max_err} too large")
    if max_err < 1e-2:
        print(f"  {label}: OK (max error {max_err:.2e})")


def verify_numeric_roundtrip() -> None:
    print("Numeric round-trip: one OK-mapped candidate per baseline vs. numpy.fft")

    clfft_result = clfft.plan_single_kernel(16, total_ffts=4)
    check(clfft_result.status is BaselineStatus.OK, "clfft N=16 should map OK for the round-trip check")
    if clfft_result.status is BaselineStatus.OK:
        _run_reference_check(clfft_result.plan.root, label="clfft N=16")

    vkfft_result = vkfft.plan(64, batch=4)
    check(vkfft_result.status is BaselineStatus.OK, "vkfft N=64 should map OK for the round-trip check")
    if vkfft_result.status is BaselineStatus.OK and isinstance(vkfft_result.plan.root, FFTLeafPlan):
        _run_reference_check(vkfft_result.plan.root, label="vkfft N=64")

    def fake_bench(plan, target):
        return rocfft.BenchmarkOutcome(ok=True, ndp_cycles=plan.length * 7 + len(plan.stages) * 13, spill_free=True)

    # N=24, not 16: a genuine, confirmed finding (not a bug) is that
    # rocFFT's own real MIN_WGS=64 floor, combined with the ported
    # "power-of-two length requires the workgroup size to evenly divide
    # it" rule, produces ZERO viable KernelConfigs for a power-of-2 length
    # as small as 16 -- real rocFFT almost certainly routes such tiny
    # transforms through a separate, hand-written "builtin kernel" path
    # outside SupportedKernelConfigs's own tuning space entirely (out of
    # scope for this baseline -- see module docstring). A non-power-of-2
    # length has no such divisibility constraint, so N=24 exercises the
    # winner-selection pipeline without hitting that separate boundary.
    rocfft_result, _ = rocfft.tune(24, total_ffts=4, benchmark_fn=fake_bench)
    check(rocfft_result.status is BaselineStatus.OK, "rocfft N=24 should find an OK winner for the round-trip check")
    if rocfft_result.status is BaselineStatus.OK and isinstance(rocfft_result.plan.root, FFTLeafPlan):
        _run_reference_check(rocfft_result.plan.root, label="rocfft N=24")


# ============================================================================
# Regression tests for findings from the Phase-1-through-6 baseline
# fidelity/mapping audit (2026-09-08) -- see docs/
# gpu_baseline_hardware_mapping_audit.md for the full derivation.
# ============================================================================


def verify_provenance_metadata() -> None:
    print("Provenance: every baseline module records immutable upstream source metadata")
    from planning.gpu_baseline import rocfft_default

    for module, expected_library in (
        (clfft, "clFFT"), (rocfft, "rocFFT"), (rocfft_default, "rocFFT"), (vkfft, "VkFFT"),
    ):
        prov = getattr(module, "PROVENANCE", None)
        check(prov is not None, f"{module.__name__} must define a module-level PROVENANCE")
        if prov is None:
            continue
        check(prov.library == expected_library, f"{module.__name__}.PROVENANCE.library={prov.library!r}")
        check(bool(prov.upstream_repository), f"{module.__name__}.PROVENANCE.upstream_repository is empty")
        check(bool(prov.upstream_commit), f"{module.__name__}.PROVENANCE.upstream_commit is empty")
        check(len(prov.source_files) > 0, f"{module.__name__}.PROVENANCE.source_files is empty")
        check(prov.baseline_version == "gpu-baseline-v1", f"{module.__name__}.PROVENANCE.baseline_version={prov.baseline_version!r}")
    # rocfft.py and rocfft_default.py may both cite a shared file (e.g.
    # plan.cpp, relevant to both for different functions), but must NOT cite
    # the exact same FILE SET or FUNCTION SET -- that would mean they are
    # not actually two separate mechanisms, just one relabeled.
    check(
        set(rocfft.PROVENANCE.source_files) != set(rocfft_default.PROVENANCE.source_files),
        "rocfft.py and rocfft_default.py must not cite the identical source-file set",
    )
    check(
        set(rocfft.PROVENANCE.source_functions) != set(rocfft_default.PROVENANCE.source_functions),
        "rocfft.py and rocfft_default.py must not cite the identical source-function set",
    )


def verify_three_way_mapping_classification() -> None:
    print("Hardware mapping: 3-way workers_per_fft classification (OK / current-codegen / hardware-mapping)")
    chunk = DEFAULT_TARGET_PROFILE.interleave_chunk_uthreads  # 8 on this target
    gpu_config = GPUKernelConfig(source="test", length=64, radices=(8, 8))

    # Divides the chunk evenly -> OK.
    r = map_cooperative_kernel(
        length=64, radices=(8, 8), workers_per_fft=chunk // 2, fft_slots_wanted=1,
        total_ffts=4, inverse=False, inverse_scale=None, kernel_name="T1",
        target=DEFAULT_TARGET_PROFILE, gpu_config=gpu_config,
    )
    check(r.status is BaselineStatus.OK, f"workers_per_fft={chunk // 2} (divides chunk={chunk}) should be OK, got {r.status}")

    # An exact multiple of the chunk -> architecturally reachable via
    # striping, but not implemented -> UNSUPPORTED_CURRENT_CODEGEN, NOT a
    # hardware wall.
    r = map_cooperative_kernel(
        length=1024, radices=(8, 8, 4, 4), workers_per_fft=chunk * 2, fft_slots_wanted=1,
        total_ffts=4, inverse=False, inverse_scale=None, kernel_name="T2",
        target=DEFAULT_TARGET_PROFILE, gpu_config=gpu_config,
    )
    check(
        r.status is BaselineStatus.UNSUPPORTED_CURRENT_CODEGEN,
        f"workers_per_fft={chunk * 2} (exact multiple of chunk={chunk}) should be "
        f"UNSUPPORTED_CURRENT_CODEGEN (proven architecturally reachable, just not "
        f"implemented), got {r.status}",
    )

    # Neither a divisor nor a multiple -> genuinely impossible.
    r = map_cooperative_kernel(
        length=105, radices=(7, 15), workers_per_fft=15, fft_slots_wanted=1,
        total_ffts=4, inverse=False, inverse_scale=None, kernel_name="T3",
        target=DEFAULT_TARGET_PROFILE, gpu_config=gpu_config,
    )
    check(
        r.status is BaselineStatus.UNSUPPORTED_HARDWARE_MAPPING,
        f"workers_per_fft=15 (neither divides nor is a multiple of chunk={chunk}) "
        f"should be UNSUPPORTED_HARDWARE_MAPPING, got {r.status}",
    )


def verify_clfft_multiple_of_chunk_is_current_codegen_not_hardware() -> None:
    print("clFFT: N=64..4096 (workers_per_fft always a multiple of 8) reclassified as UNSUPPORTED_CURRENT_CODEGEN")
    for length in (64, 128, 256, 512, 1024, 2048, 4096):
        result = clfft.plan_single_kernel(length, total_ffts=4)
        check(
            result.status is BaselineStatus.UNSUPPORTED_CURRENT_CODEGEN,
            f"clfft N={length} should be UNSUPPORTED_CURRENT_CODEGEN post-audit, got {result.status}",
        )


def verify_rocfft_min_wgs_fix() -> None:
    print("rocFFT: MIN_WGS fix -- N=8/N=16 now produce surviving KernelConfig candidates")
    for length in (8, 16):
        configs = rocfft.phase0_candidates(length)
        check(len(configs) > 0, f"rocfft phase0_candidates({length}) should be non-empty after the MIN_WGS fix")

        def fake_bench(plan, target):
            return rocfft.BenchmarkOutcome(ok=True, ndp_cycles=plan.length * 7 + len(plan.stages) * 13, spill_free=True)

        result, _ = rocfft.tune(length, total_ffts=4, benchmark_fn=fake_bench)
        check(
            result.status is BaselineStatus.OK,
            f"rocfft.tune({length}) should find an OK winner after the MIN_WGS fix, got {result.status}",
        )


def verify_rocfft_default_table() -> None:
    print("rocFFT-default: compiled-in small-kernel table (config_sbrr.py, full 2..4096+)")
    for length, expected_wgs, expected_tpt, expected_factors in (
        (8, 64, 4, (4, 2)),
        (16, 64, 4, (4, 4)),
        (1024, 128, 128, (8, 8, 4, 4)),
        (105, 256, 21, (7, 3, 5)),  # mixed-radix, real table row (deep-dive-confirmed)
    ):
        d = rocfft_default.decide_scheme(length, batch=1)  # batch=1 avoids the >4096 occupancy branch
        check(d.scheme == "CS_KERNEL_STOCKHAM", f"rocfft-default N={length}: expected CS_KERNEL_STOCKHAM, got {d.scheme}")
        if d.single_kernel is not None:
            check(
                d.single_kernel.workgroup_size == expected_wgs and d.single_kernel.threads_per_transform == expected_tpt,
                f"rocfft-default N={length}: got wgs={d.single_kernel.workgroup_size} "
                f"tpt={d.single_kernel.threads_per_transform}, expected wgs={expected_wgs} tpt={expected_tpt}",
            )
            check(
                d.single_kernel.factors == expected_factors,
                f"rocfft-default N={length}: factors={d.single_kernel.factors} != {expected_factors}",
            )

    # A length genuinely outside the compiled single-kernel table (but
    # power-of-2, so it routes through CS_L1D_CC via map1DLengthSingle) --
    # decided faithfully (real scheme + real divLength1), never fabricated,
    # and never silently mapped onto this repo's unrelated six-step shape.
    d16384 = rocfft_default.decide_scheme(16384, batch=1)
    check(d16384.scheme == "CS_L1D_CC", f"rocfft-default N=16384: expected CS_L1D_CC, got {d16384.scheme}")
    check(d16384.div_length1 == 64, f"rocfft-default N=16384: expected divLength1=64, got {d16384.div_length1}")
    result = rocfft_default.plan(16384, batch=4)
    check(
        result.status is BaselineStatus.UNSUPPORTED_CURRENT_CODEGEN,
        f"rocfft-default N=16384 (CS_L1D_CC, no SBCC/SBRC codegen mechanism) should be "
        f"UNSUPPORTED_CURRENT_CODEGEN, got {result.status}",
    )
    check(
        result.gpu_config.extra.get("scheme") == "CS_L1D_CC" and result.gpu_config.extra.get("div_length1") == 64,
        f"rocfft-default N=16384 diagnostics must preserve the real decided scheme/divLength1, got {result.gpu_config.extra}",
    )

    # A length with no decomposition at all under this chain -> Bluestein.
    d_prime = rocfft_default.decide_scheme(1000003, batch=1)
    check(
        d_prime.scheme == "CS_BLUESTEIN",
        f"rocfft-default N=1000003 (large prime, no factor in the compiled table) should "
        f"resolve to CS_BLUESTEIN, got {d_prime.scheme}",
    )


def verify_vkfft_four_step_reordering() -> None:
    print("VkFFT: four-step reordering (%2/%4/%8 preference on locAxisSplit[0])")
    # Same factor multiset, different starting order -> the source rule is
    # deterministic and order-independent: both must converge to the same
    # canonical (source-mandated) arrangement.
    check(
        vkfft.apply_four_step_reordering((6, 8)) == vkfft.apply_four_step_reordering((8, 6)),
        "reordering (6,8) and (8,6) must converge to the same canonical order",
    )
    check(
        vkfft.apply_four_step_reordering((6, 8)) == (8, 6),
        f"(6,8): position 0 (6) isn't %4-divisible, position 1 (8) is -> must swap to (8,6), "
        f"got {vkfft.apply_four_step_reordering((6, 8))}",
    )
    check(
        vkfft.apply_four_step_reordering((3, 16, 5)) == vkfft.apply_four_step_reordering((5, 3, 16)),
        "reordering (3,16,5) and (5,3,16) (same multiset) must converge to the same canonical order",
    )
    check(
        vkfft.apply_four_step_reordering((3, 16, 5))[0] % 8 == 0,
        "position 0 must end up %8-divisible when any position in the multiset is (16 is)",
    )
    # A tuple already starting with the most-divisible value is a no-op.
    check(
        vkfft.apply_four_step_reordering((8, 6)) == (8, 6),
        "an already-canonical order must be a no-op",
    )


def verify_vkfft_num_passes_fix() -> None:
    print("VkFFT: choose_num_passes fidelity fix (reorderFourStep=True default formula)")
    target = DEFAULT_TARGET_PROFILE
    max_strided = vkfft.max_sequence_length_shared_memory_strided(target)
    # Real formula (reorderFourStep default): ceil(log2(N) / log2(maxSingleSizeStrided)).
    for length in (1 << 20, 1 << 24, 1 << 28):
        expected = math.ceil(math.log2(length) / math.log2(max_strided))
        if expected > 3:
            continue
        got = vkfft.choose_num_passes(length, non_strided=True, target=target)
        check(got == expected, f"choose_num_passes({length}) = {got}, expected {expected} (reorderFourStep formula)")


def main() -> None:
    verify_clfft_table()
    verify_clfft_determine_sizes()
    verify_clfft_large1d_split_exact()
    verify_clfft_mapping_findings()
    verify_rocfft_factorize()
    verify_rocfft_utilization()
    verify_rocfft_tpb()
    verify_rocfft_phase1_permutations()
    verify_rocfft_winner_selection()
    verify_vkfft_register_classification()
    verify_vkfft_pow2_grouping()
    verify_vkfft_axis_splitting()
    verify_vkfft_direct_radix_and_bluestein_boundary()
    verify_numeric_roundtrip()
    verify_provenance_metadata()
    verify_three_way_mapping_classification()
    verify_clfft_multiple_of_chunk_is_current_codegen_not_hardware()
    verify_rocfft_min_wgs_fix()
    verify_rocfft_default_table()
    verify_vkfft_four_step_reordering()
    verify_vkfft_num_passes_fix()

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S):")
        for f in _FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
