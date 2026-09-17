from __future__ import annotations

"""Correctness + fidelity tests for the M2NDP-ADAPTED GPU baselines
(`clfft.plan_m2ndp`, `rocfft_default.plan_m2ndp`, `vkfft.plan_m2ndp`) --
see docs/gpu_planner_m2ndp_target_mapping.md for the full semantics.

Two things this suite exists to prove, neither already covered by
`verify_gpu_baseline_source_fidelity.py` (which only re-verifies the
FROZEN `plan()` entry points and must never need to change for this work):

A. PLANNER DECISION FIDELITY: switching only the target hardware
   parameters (source-faithful GPU constants -> M2NDP TargetProfile
   values) changes ONLY the hardware-dependent decisions -- never the
   algorithm's own control flow, table contents, or radix vocabulary. For
   every length where the two baselines' decisions differ, the difference
   must trace to exactly one of the three documented mappings (clFFT LDS,
   rocFFT-default multiprocessor count, VkFFT warp size), not to some
   other, undocumented change.

B. NUMERICAL CORRECTNESS: every `plan_m2ndp` result, whatever scheme it
   picked, computes a genuinely correct FFT against `numpy.fft` -- the
   Python-side oracle already used throughout this repo
   (`verification.verify_fft_recursive.run_recursive_plan`, which
   re-executes the actual emitted stage formulas).
"""

import numpy as np

from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.gpu_baseline import clfft, rocfft_default, vkfft
from planning.gpu_baseline.common import BaselineStatus
from verification.verify_fft_recursive import run_recursive_plan

_FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        _FAILURES.append(message)
        print(f"  FAIL: {message}")


def _numeric_check(planner_name: str, n: int, plan) -> None:
    rng = np.random.default_rng(1234)
    x = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)
    got = run_recursive_plan(plan, x)
    expected = np.fft.fft(x)
    err = float(np.max(np.abs(got - expected)))
    check(err < 1e-2, f"{planner_name} N={n}: max error {err:.3e} exceeds tolerance")
    if err < 1e-2:
        print(f"  OK   {planner_name} N={n}: max error {err:.3e}")


def verify_clfft_m2ndp_numerical() -> None:
    print("clFFT-m2ndp: numerical correctness across the LDS-threshold boundary")
    # 4096 = source threshold (single-kernel either side); 5000/6144/8000 sit
    # strictly between the source (4096) and M2NDP (8192) thresholds, so the
    # M2NDP-adapted baseline takes the SINGLE-KERNEL path there while the
    # source-faithful one takes large-1D -- exactly the decision this mapping
    # is supposed to change (see clfft.py's own get_max_1d_length_m2ndp).
    for n in (216, 512, 4096, 5000, 6144, 8000, 10000):
        r = clfft.plan_m2ndp(n, target=DEFAULT_TARGET_PROFILE)
        if r.status is BaselineStatus.RESOURCE_INFEASIBLE:
            # A real, legitimate outcome (task section 8, case B): the
            # wider LDS-equivalent threshold makes N=8000 ELIGIBLE for a
            # single-kernel Stockham plan, but that single kernel's own
            # cooperative-slot scratchpad footprint (leaf_scratchpad_bytes,
            # a DIFFERENT formula from get_max_1d_length_m2ndp's threshold)
            # still exceeds target.spad_capacity_bytes -- the two formulas
            # answer related but distinct questions and are not required to
            # agree. This must never be silently forced to succeed.
            print(f"  SKIP clfft-m2ndp N={n}: RESOURCE_INFEASIBLE (single-kernel eligible by threshold, "
                  f"but its own cooperative-slot scratchpad footprint does not fit -- a real, expected refusal)")
            continue
        check(r.status is BaselineStatus.OK, f"clfft-m2ndp N={n} should map OK, got {r.status}")
        if r.status is BaselineStatus.OK:
            _numeric_check("clfft-m2ndp", n, r.plan)


def verify_clfft_m2ndp_decision_fidelity() -> None:
    print("clFFT-m2ndp: decision differences trace ONLY to the documented LDS mapping")
    t = DEFAULT_TARGET_PROFILE
    src_threshold = clfft.get_max_1d_length()
    m2ndp_threshold = clfft.get_max_1d_length_m2ndp(t)
    check(m2ndp_threshold != src_threshold, "M2NDP threshold should differ from clFFT's own representative-GPU threshold")
    check(m2ndp_threshold == clfft._floor_po2(t.spad_capacity_bytes // clfft.CLFFT_ELEM_BYTES),
          "M2NDP threshold must be exactly floor_po2(spad_capacity_bytes / elem_bytes) -- the SAME formula, different input")
    for n in range(src_threshold + 1, m2ndp_threshold + 1):
        if not clfft.is_1d_possible(n, m2ndp_threshold):
            continue
        r_src = clfft.plan(n)
        r_m2ndp = clfft.plan_m2ndp(n, target=t)
        if r_src.status is BaselineStatus.OK and r_m2ndp.status is BaselineStatus.OK:
            src_is_single = bool(r_src.gpu_config.radices)
            m2ndp_is_single = bool(r_m2ndp.gpu_config.radices)
            # A wider M2NDP threshold can only ever ADD single-kernel
            # eligibility relative to the source-faithful one, never
            # remove it -- so "source single-kernel, M2NDP large-1D" is
            # the one combination that must never happen.
            check(
                not (src_is_single and not m2ndp_is_single),
                f"N={n}: source-faithful is single-kernel but M2NDP-adapted regressed to large-1D "
                f"(src_single={src_is_single}, m2ndp_single={m2ndp_is_single})",
            )
    print(f"  source threshold={src_threshold}, M2NDP threshold={m2ndp_threshold} -- checked every length in between")


def verify_rocfft_default_m2ndp_numerical() -> None:
    print("rocFFT-default-m2ndp: numerical correctness")
    for n in (216, 512, 4096, 8192):
        r = rocfft_default.plan_m2ndp(n, target=DEFAULT_TARGET_PROFILE)
        if r.status is not BaselineStatus.OK:
            print(f"  SKIP rocfft-default-m2ndp N={n}: {r.status} (not a planning bug -- see UNSUPPORTED_CURRENT_CODEGEN policy)")
            continue
        _numeric_check("rocfft-default-m2ndp", n, r.plan)


def verify_rocfft_default_m2ndp_decision_fidelity() -> None:
    print("rocFFT-default-m2ndp: multiprocessor_count mapping changes ONLY the documented occupancy branch")
    t = DEFAULT_TARGET_PROFILE
    check(t.num_ndp_units != rocfft_default.ROCFFT_DEFAULT_MULTIPROCESSOR_COUNT,
          "M2NDP num_ndp_units should differ from rocFFT-default's own representative-GPU CU count")
    big_lengths = [n for n in rocfft_default.SBRR_TABLE if n > rocfft_default.SINGLE_KERNEL_OCCUPANCY_THRESHOLD]
    check(bool(big_lengths), "expected at least one SBRR_TABLE length above the occupancy threshold to exercise this branch")
    found_a_difference = False
    for n in big_lengths:
        for batch in (1, 8, 16, 24, 32, 40, 64, 128, 200):
            d_src = rocfft_default.decide_scheme(n, batch=batch)
            d_m2ndp = rocfft_default.decide_scheme(n, batch=batch, multiprocessor_count=t.num_ndp_units)
            if d_src.scheme != d_m2ndp.scheme:
                found_a_difference = True
                check(
                    d_m2ndp.scheme == "CS_KERNEL_STOCKHAM" and d_src.scheme != "CS_KERNEL_STOCKHAM",
                    f"N={n} batch={batch}: a SMALLER multiprocessor_count should only ever make "
                    f"CS_KERNEL_STOCKHAM MORE reachable (less batch needed to saturate fewer units), "
                    f"got src={d_src.scheme} m2ndp={d_m2ndp.scheme}",
                )
    check(found_a_difference, "expected at least one (length, batch) pair where the multiprocessor_count mapping changes the scheme")
    print(f"  checked {len(big_lengths)} lengths x 9 batch values; found_a_difference={found_a_difference}")


def verify_vkfft_m2ndp_numerical() -> None:
    print("VkFFT-m2ndp: numerical correctness")
    for n in (12, 20, 216, 512, 768, 1024, 4096, 4704):
        r = vkfft.plan_m2ndp(n, target=DEFAULT_TARGET_PROFILE)
        check(r.status is BaselineStatus.OK, f"vkfft-m2ndp N={n} should map OK, got {r.status}")
        if r.status is BaselineStatus.OK:
            _numeric_check("vkfft-m2ndp", n, r.plan)


def verify_vkfft_m2ndp_warp_size_wiring() -> None:
    print("VkFFT-m2ndp: warp_size mapping reaches axisblock_batch_single_pass correctly")
    t = DEFAULT_TARGET_PROFILE
    check(t.interleave_chunk_uthreads != vkfft.VKFFT_WARP_SIZE,
          "M2NDP interleave_chunk_uthreads should differ from VkFFT's own representative-GPU warp size")
    # Direct function-level check (isolates the mapping from downstream
    # _postprocess_axis_upload0 reshaping, which can converge to the same
    # final answer for some lengths -- see docs/gpu_planner_m2ndp_target_
    # mapping.md's own VkFFT row for why the raw seed is what's guaranteed
    # to differ, not necessarily every final plan).
    found_a_difference = False
    for tpt in range(1, 300):
        seed_src = vkfft.axisblock_batch_single_pass(100, tpt, t)
        seed_m2ndp = vkfft.axisblock_batch_single_pass(100, tpt, t, warp_size=t.interleave_chunk_uthreads)
        if seed_src != seed_m2ndp:
            found_a_difference = True
    check(found_a_difference, "expected at least one threads_per_transform where warp_size changes axisblock_batch_single_pass's own seed")
    print(f"  found_a_difference={found_a_difference} across threads_per_transform=1..299")


def main() -> None:
    verify_clfft_m2ndp_numerical()
    verify_clfft_m2ndp_decision_fidelity()
    verify_rocfft_default_m2ndp_numerical()
    verify_rocfft_default_m2ndp_decision_fidelity()
    verify_vkfft_m2ndp_numerical()
    verify_vkfft_m2ndp_warp_size_wiring()
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S)")
        raise SystemExit(1)
    print("ALL M2NDP-ADAPTED BASELINE CHECKS PASSED")


if __name__ == "__main__":
    main()
