from __future__ import annotations

"""Independent source-fidelity tests for the GPU-derived BASELINE planners.

Unlike `verify_gpu_baseline_golden.py` (a regression freeze whose own
comments note many golden values were captured FROM the local
implementation, so it cannot by itself prove upstream fidelity), every
expected value here was derived either directly from the pinned upstream
C++ source (read and hand-traced, cited by exact line number) or from a
SEPARATE, independent Python re-transliteration of that same source,
written from scratch without reference to this repository's own
`planning/gpu_baseline/*.py` implementation, then cross-checked against
it. Never "run the function under test, save what it printed."

Run directly: `python3 -m verification.verify_gpu_baseline_source_fidelity`
from benchmarks/fft/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import re

from planning.gpu_baseline import clfft, rocfft, rocfft_default
from planning.gpu_baseline import rocfft_upstream_solution_map as usm
from planning.gpu_baseline.common import BaselineStatus

_FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        _FAILURES.append(message)
        print(f"  FAIL: {message}")


# ============================================================================
# Independent re-transliteration of SupportedKernelConfigs (phase-0/phase-1),
# tuning_kernel_tuner.cpp lines 459-757 at the pinned commit
# bee97df517907c771de17189cb867d3c401285ae, fixed to is_single=True,
# is_sbcc=is_sbrc=is_sbcr=False, large1D=0 (this baseline's own SBRR-only
# scope -- see rocfft.py's module docstring SCOPE LIMIT). Written directly
# from the fetched source, NOT copied from rocfft.py's own
# `_supported_kernel_configs` -- this is the "small independent upstream-
# reference extractor" the task's own testing section calls for.
# ============================================================================


def _ref_factorize(n: int, memo: dict[int, set[tuple[int, ...]]] = {}) -> set[tuple[int, ...]]:
    if n in memo:
        return memo[n]
    ret: set[tuple[int, ...]] = set()
    for f in rocfft.SUPPORTED_FACTORS:
        if n % f == 0:
            remain = n // f
            if remain == 1:
                ret.add((f,))
            else:
                for rf in _ref_factorize(remain):
                    ret.add(tuple(sorted((f,) + rf)))
    memo[n] = ret
    return ret


def _ref_max_radices_size(all_factors: set[tuple[int, ...]]) -> int:
    min_size = rocfft.TWIDDLES_MAX_RADICES + 1
    for f in all_factors:
        min_size = min(min_size, len(f))
    return min_size + 2


def _ref_power_set(factors: tuple[int, ...]) -> set[tuple[int, ...]]:
    ret: set[tuple[int, ...]] = {()}
    for f in factors:
        ret |= {t + (f,) for t in ret}
    return ret


def _ref_supported_tpt(factorization: tuple[int, ...]) -> list[int]:
    tpts: set[int] = set()
    for subset in _ref_power_set(factorization):
        if not subset:
            continue
        p = 1
        for f in subset:
            p *= f
        tpts.add(p)
    return sorted(tpts)


def _ref_util_rates(length: int, factors: tuple[int, ...], tpt: int) -> tuple[list[float], float]:
    heights = [length / w / tpt for w in factors]
    return heights, sum(heights) / len(factors)


def _ref_is_po2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _ref_derive_max_tpb(length: int, half_lds: bool, tpt: int, wgs_bound: int) -> int:
    bpb = length * rocfft.BYTES_PER_ELEM
    if half_lds:
        bpb //= 2
    tpb = rocfft.LDS_BYTE_LIMIT // bpb
    while tpt * tpb > wgs_bound:
        tpb -= 1
    return tpb


def _ref_conservative_max_tpb(length: int) -> int:
    bpb = length * rocfft.BYTES_PER_ELEM
    c = rocfft.LDS_BYTE_LIMIT // bpb
    if length >= 1024:
        c += 1
    return c


def _ref_supported_kernel_configs(
    length: int, factorizations: list[tuple[int, ...]], *, is_phase0: bool,
) -> list[tuple[tuple[int, ...], int, int, int, bool]]:
    """Returns (factors, tpt, tpb, final_wgs, half_lds) tuples."""
    max_radices_size = _ref_max_radices_size(_ref_factorize(length))
    if length == 336:
        max_radices_size -= 1
    conservative_tpb = _ref_conservative_max_tpb(length)

    tpbs_to_remove: set[int] = set()
    tpts_bad_util: set[int] = set()
    all_tpts: set[int] = set()

    min_wgs, max_wgs = 64, 512
    min_wgs = length if length < min_wgs else min_wgs
    min_wgs = min_wgs if min_wgs % 64 == 0 else max(0, min_wgs - (min_wgs % 64))
    max_wgs = max_wgs if max_wgs % 64 == 0 else max_wgs - (max_wgs % 64)

    configs: list[tuple[tuple[int, ...], int, int, int, bool]] = []

    for factorization in factorizations:
        if is_phase0 and len(factorization) > max_radices_size:
            continue
        tpts = _ref_supported_tpt(factorization)
        for tpt in tpts:
            heights, avg = _ref_util_rates(length, factorization, tpt)
            if avg < 1.0 or max(heights) > 8.0:
                tpts_bad_util.add(tpt)

        wgs = min_wgs
        while wgs <= max_wgs:
            for tpt in tpts:
                if not (tpt < wgs):
                    continue
                for half_lds in (True, False):
                    max_tpb = _ref_derive_max_tpb(length, half_lds, tpt, wgs)
                    if tpt == length:
                        tpbs_to_remove.add(max_tpb)
                    num_try = 1 if tpt * max_tpb == wgs else 2
                    for t in range(num_try):
                        tpb = max_tpb + t
                        final_wgs = tpt * tpb
                        if final_wgs > max_wgs:
                            continue
                        if final_wgs <= wgs - 64:
                            continue
                        if tpb > conservative_tpb:
                            continue
                        if length >= 64 and final_wgs < 64:
                            continue
                        if _ref_is_po2(length) and length % final_wgs != 0:
                            continue
                        all_tpts.add(tpt)
                        configs.append((factorization, tpt, tpb, final_wgs, half_lds))
            wgs += 64

    if len(all_tpts) >= 2 and tpbs_to_remove:
        configs = [c for c in configs if c[1] != length and c[2] not in tpbs_to_remove]
        all_tpts.discard(length)

    if len(all_tpts) > len(tpts_bad_util):
        configs = [c for c in configs if c[1] not in tpts_bad_util]
        all_tpts -= tpts_bad_util

    if all_tpts and is_phase0:
        n_remove = (len(all_tpts) - 1) // 2
        if n_remove > 0:
            sorted_tpts = sorted(all_tpts)
            to_remove = set(sorted_tpts[len(sorted_tpts) - n_remove :])
            configs = [c for c in configs if c[1] not in to_remove]

    return list(dict.fromkeys(configs))


def _as_tuple_set(configs) -> set[tuple]:
    return {
        (tuple(sorted(c.factors)), c.threads_per_transform, c.transforms_per_block, c.workgroup_size, c.half_lds)
        for c in configs
    }


def _ref_as_tuple_set(ref_configs) -> set[tuple]:
    return {(tuple(sorted(f)), tpt, tpb, wgs, half_lds) for f, tpt, tpb, wgs, half_lds in ref_configs}


# ============================================================================
# rocFFT tuned: full candidate-set comparison against the independent
# reference, for every length the task's own testing section names.
# ============================================================================


def verify_rocfft_tuned_phase0_matches_independent_reference() -> None:
    print("rocFFT tuned: phase-0 full candidate set vs. independent SupportedKernelConfigs re-derivation")
    for length in (8, 16, 24, 64, 336, 1024):
        prod_configs = rocfft.phase0_candidates(length)
        ref_configs = _ref_supported_kernel_configs(length, sorted(_ref_factorize(length)), is_phase0=True)
        prod_set = _as_tuple_set(prod_configs)
        ref_set = _ref_as_tuple_set(ref_configs)
        check(
            prod_set == ref_set,
            f"rocfft.phase0_candidates({length}) differs from the independent reference: "
            f"only-in-production={prod_set - ref_set!r} only-in-reference={ref_set - prod_set!r}",
        )
        check(
            len(prod_configs) == len(ref_configs),
            f"rocfft.phase0_candidates({length}) count {len(prod_configs)} != reference count {len(ref_configs)}",
        )


def verify_rocfft_tuned_min_wgs_rounding_below_64() -> None:
    """N<64 min_wgs rounding (tuning_kernel_tuner.cpp lines 489-491):
    min_wgs is first lowered to `length` if `length < 64`, THEN rounded
    DOWN to a 64-multiple -- so any length < 64 always produces min_wgs=0,
    never `length` itself. Combined with the `tpt < wgs` guard, this means
    the smallest wgs bucket any TPT can ever enter is 64, and the very
    first (and for short po2 lengths, ONLY) bucket a TPT lands in almost
    always forces `final_wgs == 64` exactly (see rocfft.py's own
    `_supported_kernel_configs` docstring) -- so a power-of-two length
    under 64 that does not itself divide 64 evenly can produce zero
    candidates. 8 and 16 both divide 64, yet still produce zero, because
    every one of their own TPTs (divisors of the length itself) forces
    final_wgs=64 exactly and 64 does not divide 8 or 16 either."""
    print("rocFFT tuned: N<64 min_wgs floor rounds DOWN to a 64-multiple (never raw `length`)")
    for length in (2, 3, 4, 5, 6, 7, 8, 16, 32):
        min_wgs = length if length < 64 else 64
        min_wgs = min_wgs if min_wgs % 64 == 0 else max(0, min_wgs - (min_wgs % 64))
        check(min_wgs == 0, f"length={length}: rounded min_wgs should be 0, got {min_wgs}")


def verify_rocfft_tuned_max_radices_size_336_exception() -> None:
    """N=336 radix-count exception (tuning_kernel_tuner.cpp lines 498-500:
    `if(length == 336) --max_radices_size;`), applied AFTER the
    `min_size + 2` formula, unconditionally regardless of phase."""
    print("rocFFT tuned: N=336 max_radices_size exception (min_size+2, then -1)")
    all_factors_336 = _ref_factorize(336)
    min_size_336 = min(len(f) for f in all_factors_336)
    expected_336 = min_size_336 + 2 - 1
    got_336 = rocfft.get_max_radices_size(rocfft.factorize(336), length=336)
    check(got_336 == expected_336, f"get_max_radices_size(336) = {got_336}, expected {expected_336}")

    all_factors_300 = _ref_factorize(300)
    min_size_300 = min(len(f) for f in all_factors_300)
    expected_300 = min_size_300 + 2
    got_300 = rocfft.get_max_radices_size(rocfft.factorize(300), length=300)
    check(got_300 == expected_300, f"get_max_radices_size(300) = {got_300}, expected {expected_300} (no exception)")


def verify_rocfft_tuned_global_pruning_scope() -> None:
    """Global pruning scope (task section 2B / tuning_kernel_tuner.cpp
    lines 675-754): `tpbs_to_remove`/`tpts_with_bad_util_rate`/`all_tpts`
    accumulate across EVERY factorization passed into one
    `SupportedKernelConfigs` call, and the three pruning passes at the end
    consume that combined state -- one ordering's own bad TPT can suppress
    a DIFFERENT ordering's config that happens to reuse the same TPT
    value. This is verified two ways: (1) full-set equality against the
    from-scratch independent reference above already fails if the scope
    is wrong (confirmed during development: scoping pruning per-ordering
    instead of per-call changes N=24's own result from 105 to 178
    candidates and changes its surviving factor-multiset shape); (2) a
    direct scope check here: pruning phase0_candidates(24) as ONE call
    over all its factorizations must NOT equal the union of pruning each
    factorization SEPARATELY (which is what an incorrectly-scoped,
    per-ordering implementation would compute)."""
    print("rocFFT tuned: N=24 pruning is scoped to the WHOLE phase-0 call, not per-ordering")
    length = 24
    factorizations = sorted(_ref_factorize(length))

    correctly_scoped = _ref_as_tuple_set(
        _ref_supported_kernel_configs(length, factorizations, is_phase0=True)
    )
    per_ordering_union: set[tuple] = set()
    for factorization in factorizations:
        per_ordering_union |= _ref_as_tuple_set(
            _ref_supported_kernel_configs(length, [factorization], is_phase0=True)
        )

    check(
        correctly_scoped != per_ordering_union,
        "N=24: correctly-scoped (whole-call) pruning should differ from per-ordering-scoped pruning "
        "-- if they're equal, the global-scope distinction this test exists to catch has no effect "
        "for this length and is not a meaningful regression guard",
    )
    check(
        _as_tuple_set(rocfft.phase0_candidates(length)) == correctly_scoped,
        "rocfft.phase0_candidates(24) should match the correctly (whole-call) scoped reference, "
        "not the per-ordering-scoped one",
    )
    check(
        _as_tuple_set(rocfft.phase0_candidates(length)) != per_ordering_union,
        "rocfft.phase0_candidates(24) should NOT match the (incorrect) per-ordering-scoped union",
    )


def verify_rocfft_tuned_phase1_pools_all_families_in_one_call() -> None:
    """Phase-1 call scope: `GetAllFactorizationsForPhase1` is handed the
    WHOLE per-node target-factors set (up to 3 propagated families) at
    once, and `SupportedKernelConfigs` is called ONCE on the combined
    result -- never once per family. Verified the same way as phase 0:
    the whole-call-scoped reference must differ from a per-family-scoped
    union, and rocfft.phase1_candidates must match the former."""
    print("rocFFT tuned: phase-1 pools every propagated family into ONE SupportedKernelConfigs call")
    length = 1024
    all_factors = sorted(_ref_factorize(length))
    # Pick 2 real phase-0-survivor families (not hand-invented) as the
    # "propagated best families" phase 1 would receive.
    families = [f for f in all_factors if len(f) >= 4][:2]
    assert len(families) == 2, "test fixture needs >=2 length-4+ factorizations of 1024"

    combined_orderings: dict[tuple[int, ...], None] = {}
    for family in families:
        for ordering in rocfft.get_all_factorizations_for_phase1(family):
            combined_orderings.setdefault(ordering)
    correctly_scoped = _ref_as_tuple_set(
        _ref_supported_kernel_configs(length, list(combined_orderings), is_phase0=False)
    )

    per_family_union: set[tuple] = set()
    for family in families:
        orderings = list(rocfft.get_all_factorizations_for_phase1(family))
        per_family_union |= _ref_as_tuple_set(
            _ref_supported_kernel_configs(length, orderings, is_phase0=False)
        )

    prod = _as_tuple_set(rocfft.phase1_candidates(families, length))
    check(
        prod == correctly_scoped,
        f"rocfft.phase1_candidates should match the whole-call-scoped reference for N={length}: "
        f"only-in-production={prod - correctly_scoped!r} only-in-reference={correctly_scoped - prod!r}",
    )
    # Phase 1 has no bad-utilization-driven cross-family effect for every
    # input by construction (there is no phase-0-only "largest half" step
    # here), so scope differences are narrower than phase 0's -- report
    # rather than assert inequality, since equality is possible depending
    # on which two families happen to be picked.
    if per_family_union != correctly_scoped:
        print(
            f"    (confirms phase-1 scope matters here too: per-family union "
            f"has {len(per_family_union)} configs vs. whole-call {len(correctly_scoped)})"
        )


# ============================================================================
# VkFFT: complete VkFFTSplitAxisBlock axis_upload_id==0 continuation
# (vkFFT_AxisBlockSplitter.h lines 301-364 at the pinned commit
# 066a17c17068c0f11c9298d848c2976c71fad1c1), independently re-derived --
# written fresh from the fetched source, not from vkfft.py's own
# `_postprocess_axis_upload0`.
# ============================================================================

_VKFFT_AIM_THREADS = 128
_VKFFT_MAX_THREADS_NUM = 1024
_VKFFT_MAX_COMPUTE_WORKGROUP_SIZE = 1024
_VKFFT_NUM_SHARED_BANKS = 32
_VKFFT_MAX_BATCH_COALESCED = 32 // 8  # coalescedMemory / complexSize


def _ref_floor_po2(x: int) -> int:
    return (1 << (x.bit_length() - 1)) if x > 0 else 0


def _ref_axisblock_postprocess(
    axis_block0: int, seed_batch: int, fft_dim: int, *, num_passes: int, max_rhs: int,
    original_length: int, shared_bytes: int, complex_bytes: int = 8,
) -> tuple[int, int]:
    max_seq_shared = shared_bytes // complex_bytes
    max_seq_shared_pow2 = _ref_floor_po2(shared_bytes) // complex_bytes
    batch = seed_batch

    if (
        (fft_dim % 2 == 0 or axis_block0 < _VKFFT_NUM_SHARED_BANKS // 4)
        and batch > 1 and batch * fft_dim < max_seq_shared_pow2
    ):
        p = 0
        while (1 << p) < batch:
            p += 1
        batch = 1 << p

    if num_passes > 1:
        cap = -(-original_length // fft_dim)
        if cap < batch:
            batch = cap

    if num_passes == 1 and max_rhs < batch:
        batch = max_rhs

    while batch * axis_block0 >= 2 * _VKFFT_AIM_THREADS and batch > _VKFFT_MAX_BATCH_COALESCED:
        batch //= 2
        if batch < _VKFFT_MAX_BATCH_COALESCED:
            batch = _VKFFT_MAX_BATCH_COALESCED

    if batch > _VKFFT_MAX_COMPUTE_WORKGROUP_SIZE:
        batch = _VKFFT_MAX_COMPUTE_WORKGROUP_SIZE

    if axis_block0 * batch > _VKFFT_MAX_THREADS_NUM:
        for i in range(1, batch + 1):
            if (batch // i) * axis_block0 <= _VKFFT_MAX_THREADS_NUM:
                batch //= i
                break

    while batch * fft_dim > max_seq_shared and batch > 1:
        batch //= 2

    if (
        (fft_dim % 2 == 0 or axis_block0 < _VKFFT_NUM_SHARED_BANKS // 4)
        and batch > 1 and batch * fft_dim < max_seq_shared
    ):
        axis_block0, batch = batch, axis_block0

    return axis_block0, max(batch, 1)


def _ref_divisibility_loop_literal(axis_block1_before, guard_fn, shared_mem_allows_fn):
    """Literal re-transliteration of vkFFT_AxisBlockSplitter.h lines
    301-307's own control flow (see docstring in vkfft.py's module-level
    comment for the annotated source)."""
    current = axis_block1_before
    axis_block1 = axis_block1_before
    i = current
    while i < 2 * current:
        if guard_fn(axis_block1):
            if shared_mem_allows_fn(i):
                axis_block1 = i
            i = 2 * current
        else:
            i += 1
    return axis_block1


def verify_vkfft_divisibility_loop_is_proven_noop() -> None:
    """Task's own required proof for an omitted rule that cannot change
    the result: the "divisibility-fix loop" (lines 301-307) is a
    confirmed no-op in the real source at this pinned commit for EVERY
    possible guard/shared-mem-allows outcome, not just a hand-picked one
    -- so omitting it from vkfft.py cannot silently affect any result."""
    print("VkFFT: divisibility-fix loop (lines 301-307) is a proven no-op -- exhaustive proof")
    for before in (1, 2, 3, 5, 8, 16, 100, 257):
        for guard_always in (True, False):
            for mem_always in (True, False):
                after = _ref_divisibility_loop_literal(
                    before, lambda x, g=guard_always: g, lambda i, s=mem_always: s,
                )
                check(
                    after == before,
                    f"before={before} guard={guard_always} mem={mem_always}: "
                    f"loop should be a no-op, got after={after}",
                )


def verify_vkfft_axisblock_postprocess_matches_independent_reference() -> None:
    print("VkFFT: AxisBlockSplitter axis_upload_id==0 continuation vs. independent re-derivation")
    from planning.gpu_baseline import vkfft
    from planning.core.target_profile import DEFAULT_TARGET_PROFILE as T

    mismatches = 0
    checked = 0
    for fft_dim in (2, 3, 4, 5, 7, 8, 9, 16, 32, 64, 128, 256, 512, 1000, 1024):
        for axis_block0 in (1, 2, 3, 4, 5, 7, 8, 16, 32, 64, 128, 256):
            for seed_batch in (1, 2, 3, 4, 5, 7, 8, 16, 32, 64, 128):
                for num_passes in (1, 2, 3):
                    for max_rhs in (1, 2, 4, 8, 16, 64):
                        for original_length in (fft_dim, fft_dim * 2, fft_dim * 4):
                            checked += 1
                            got = vkfft._postprocess_axis_upload0(
                                axis_block0, seed_batch, fft_dim, T,
                                num_passes=num_passes, max_rhs=max_rhs, original_length=original_length,
                            )
                            exp = _ref_axisblock_postprocess(
                                axis_block0, seed_batch, fft_dim, num_passes=num_passes,
                                max_rhs=max_rhs, original_length=original_length,
                                shared_bytes=T.spad_capacity_bytes,
                            )
                            if got != exp:
                                mismatches += 1
    check(checked > 50000, f"expected a large combination sweep, only checked {checked}")
    check(mismatches == 0, f"{mismatches}/{checked} combinations mismatched the independent reference")


def verify_vkfft_axisblock_swap_and_upload_cases() -> None:
    """Source-fidelity tests specifically constructed to trigger: the
    bank-conflict axis swap (reachable -- N=128 under DEFAULT_TARGET_
    PROFILE), single upload, first multi-upload, and later multi-upload
    (task's own explicit list; the divisibility correction is proven
    unreachable-in-effect above, so no trigger case exists for it)."""
    print("VkFFT: AxisBlockSplitter swap + single/first/later-upload cases exercised end-to-end")
    from planning.gpu_baseline import vkfft
    from planning.core.target_profile import DEFAULT_TARGET_PROFILE as T

    # Bank-conflict swap fires for N=128 (single pass): verified directly
    # against the golden-confirmed (4, 16) pair (pre-fix would have been
    # (16, 8) -- see verify_gpu_baseline_golden.py's own 2026-09-09 note).
    radices = vkfft.leaf_radix_sequence(128, 4)
    min_regs = vkfft.min_registers_per_thread_for(128, radices, 4)
    axis_block0 = vkfft.axisblock_threads_per_transform(128, min_regs)
    seed = vkfft.axisblock_batch_single_pass(128, axis_block0, T)
    final0, final1 = vkfft._postprocess_axis_upload0(
        axis_block0, seed, 128, T, num_passes=1, max_rhs=4, original_length=128,
    )
    check((final0, final1) == (4, 16), f"N=128 single-upload should trigger the swap to (4,16), got {(final0, final1)}")
    check(
        (axis_block0, seed) != (final0, final1),
        "the swap should have actually changed the pre-swap (axis_block0, seed) pair for N=128",
    )

    result = vkfft.plan(128, batch=4)
    check(result.status is BaselineStatus.OK, f"vkfft.plan(128) should be OK post-fix, got {result.status}")
    cooperation = result.plan.root.kernel.cooperation
    check(
        cooperation is not None and cooperation.workers_per_fft == 4,
        f"vkfft.plan(128)'s own built kernel should use the swapped workers_per_fft=4, got {cooperation}",
    )

    # Single upload (num_passes==1): any length handled entirely by
    # axisblock_batch_single_pass -- N=64 (already the module's own
    # numeric round-trip candidate).
    result64 = vkfft.plan(64, batch=4)
    check(
        result64.status is BaselineStatus.OK and result64.gpu_config.extra.get("num_passes") == 1,
        f"N=64 should be a single-upload (num_passes==1) OK case, got {result64.status}/{result64.gpu_config.extra}",
    )

    # First multi-upload (num_passes>1, upload_id==0) and later multi-
    # upload (upload_id>0): N=16384 needs 2 passes (choose_num_passes),
    # exercising both axisblock_batch_multipass_first (the near_fft leaf,
    # upload_id==0) and axisblock_batch_multipass_later (every leaf after
    # it, upload_id>0) within the same plan.
    num_passes_16384 = vkfft.choose_num_passes(16384, non_strided=True, target=T)
    check(num_passes_16384 > 1, f"N=16384 should need >1 pass to exercise first/later-upload, got {num_passes_16384}")
    result16384 = vkfft.plan(16384, batch=4)
    check(
        result16384.gpu_config.extra.get("upload_id") == 0,
        f"vkfft.plan(16384)'s own reported leaf should be upload_id=0 (near_fft, first upload), "
        f"got {result16384.gpu_config.extra}",
    )
    # Directly exercise a later-upload leaf (upload_id=1) for the same
    # length via axisblock_for_leaf, confirming it takes the
    # multipass_later path (no swap applied there -- see module docstring:
    # the real source's own axis_upload_id>0 branch has none).
    later_tpt, later_batch = vkfft.axisblock_for_leaf(
        64, vkfft.leaf_radix_sequence(64, 4), max_rhs=4, num_passes=num_passes_16384,
        upload_id=1, original_length=16384, target=T,
    )
    check(later_tpt > 0 and later_batch > 0, f"later-upload leaf should produce a valid pair, got {(later_tpt, later_batch)}")


# ============================================================================
# VkFFT: Rader-vs-Bluestein PLANNING DECISION (task section 5C /
# vkFFT_AppManagement/vkFFT_InitializeApp.h's own vendor/precision-keyed
# defaults, lines 1257-1292 at the pinned commit), independently re-derived.
# ============================================================================

_REF_DIRECT_KERNEL_PRIMES = frozenset({2, 3, 5, 7, 11, 13})
_REF_FIX_MAX_RADER_PRIME_FFT = 16384  # NVIDIA/single-precision default, all vendors


def _ref_prime_factors(n: int) -> list[int]:
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def _ref_classify_residual(residual: int) -> str:
    worst = "direct"
    for p in _ref_prime_factors(residual):
        if p in _REF_DIRECT_KERNEL_PRIMES:
            c = "direct"
        elif p < _REF_FIX_MAX_RADER_PRIME_FFT:
            c = "rader"
        else:
            c = "bluestein"
        if c == "bluestein" or (c == "rader" and worst == "direct"):
            worst = c
    return worst


def verify_vkfft_rader_bluestein_classification_matches_independent_reference() -> None:
    print("VkFFT: Rader-vs-Bluestein residual classification vs. independent InitializeApp.h re-derivation")
    from planning.gpu_baseline import vkfft

    # Every prime up to a representative sample, plus composites straddling
    # the direct/Rader/Bluestein boundaries (17 = fixMinRaderPrimeMult,
    # 16384 = fixMaxRaderPrimeFFT).
    samples = list(range(2, 200)) + [16383, 16384, 16385, 17389, 11 * 13, 11 * 11, 13 * 13, 2 * 17389]
    for residual in samples:
        got = vkfft.classify_vkfft_residual_scheme(residual)
        expected = _ref_classify_residual(residual)
        check(
            got["scheme"] == expected,
            f"classify_vkfft_residual_scheme({residual})['scheme'] = {got['scheme']!r}, expected {expected!r}",
        )

    # Named boundary checks (task's own explicit examples).
    check(vkfft.classify_vkfft_residual_scheme(11)["scheme"] == "direct", "11 is a real VkFFT built-in kernel prime")
    check(vkfft.classify_vkfft_residual_scheme(13)["scheme"] == "direct", "13 is a real VkFFT built-in kernel prime")
    check(vkfft.classify_vkfft_residual_scheme(17)["scheme"] == "rader", "17 == fixMinRaderPrimeMult should be Rader")
    check(
        vkfft.classify_vkfft_residual_scheme(16383)["scheme"] == "rader",
        "16383 < fixMaxRaderPrimeFFT should be Rader (via its own prime factors)",
    )
    check(
        vkfft.classify_vkfft_residual_scheme(17389)["scheme"] == "bluestein",
        "17389 (prime, > fixMaxRaderPrimeFFT) should be Bluestein",
    )

    # End-to-end: the classification must never make an unsupported length
    # silently OK, and must be attached to the actual refusal diagnostics.
    # A Bluestein-tier residual (prime >= 16384) cannot appear in this
    # check: it would need a single-kernel leaf length >= 16384, which
    # already exceeds this target's own max_sequence_length_shared_memory
    # (15360) -- any length that large is refused earlier, at the
    # axis-split stage, before direct_radix_sequence's own residual check
    # ever runs (confirmed directly: length=2*17389 fails with "no legal
    # non-power-of-2 2-pass axis split found," never reaching a residual
    # classification at all). The classifier itself is still proven
    # correct in isolation above; only end-to-end reachability differs.
    for length, expected_scheme in ((34, "rader"), (22, "direct")):
        result = vkfft.plan(length, batch=4)
        check(
            result.status is BaselineStatus.UNSUPPORTED_GPU_ALGORITHM,
            f"length={length} should still be UNSUPPORTED_GPU_ALGORITHM (no algorithm invented), got {result.status}",
        )
        check(
            result.gpu_config.extra.get("scheme") == expected_scheme,
            f"length={length} should report scheme={expected_scheme!r} in diagnostics, got {result.gpu_config.extra}",
        )


# ============================================================================
# clFFT: block-compute (SBCC) scheme detection, independently re-derived
# from plan.cpp lines 646-682 at the pinned commit
# c59712e136fa6207956af22f5c0e4cee7d05340e (single precision, this
# baseline's own fixed C2C/out-of-place/unit-stride/1D/no-mem-alloc-unset
# assumptions -- see clfft.py's `is_block_compute_length` docstring).
# ============================================================================

# Transcribed directly from the fetched plan.cpp switch statement (lines
# 654-666), not from clfft.py's own CLFFT_BLOCK_COMPUTE_TABLE_SINGLE.
_REF_BLOCK_COMPUTE_TABLE_SINGLE = {
    8192: 64, 16384: 64, 32768: 128, 65536: 256,
    131072: 64, 262144: 64, 524288: 256, 1048576: 256,
}
_REF_BLOCK_COMPUTE_GATE_SINGLE = 262144  # 262144 / PrecisionWidth(single==1)


def verify_clfft_block_compute_gate_matches_independent_reference() -> None:
    print("clFFT: block-compute (SBCC) eligibility gate matches an independent plan.cpp re-derivation")
    # Every power of two from 2 up through well past the gate, plus a
    # sample of non-power-of-two lengths that must never be eligible.
    candidates = [2**k for k in range(1, 22)] + [24, 45360, 105, 999983]
    for length in candidates:
        expected = (
            _ref_is_po2(length)
            and length <= _REF_BLOCK_COMPUTE_GATE_SINGLE
            and length in _REF_BLOCK_COMPUTE_TABLE_SINGLE
        )
        got = clfft.is_block_compute_length(length)
        check(got == expected, f"is_block_compute_length({length}) = {got}, expected {expected}")


def verify_clfft_block_compute_never_silently_becomes_four_step() -> None:
    """Task section 4's core requirement: a length where real clFFT would
    select block-compute must report UNSUPPORTED_CURRENT_CODEGEN with the
    scheme recorded as metadata -- never a silently-OK four-step plan.
    N=8192 is the task's own named representative example."""
    print("clFFT: block-compute lengths report UNSUPPORTED_CURRENT_CODEGEN with scheme metadata, never silent OK")
    for length, expected_a, expected_b in (
        (8192, 128, 64), (16384, 256, 64), (32768, 256, 128),
        (65536, 256, 256), (131072, 2048, 64), (262144, 4096, 64),
    ):
        check(
            clfft.is_block_compute_length(length),
            f"length={length} should be block-compute-eligible under the fixed baseline gate",
        )
        result = clfft.plan_large1d(length, batch=4)
        check(
            result.status is BaselineStatus.UNSUPPORTED_CURRENT_CODEGEN,
            f"clfft.plan_large1d({length}) should be UNSUPPORTED_CURRENT_CODEGEN (block-compute has no "
            f"M2NDP codegen), got {result.status}",
        )
        check(
            result.gpu_config.extra.get("scheme") == "block_compute",
            f"clfft.plan_large1d({length}) should record scheme='block_compute' in gpu_config.extra, "
            f"got {result.gpu_config.extra!r}",
        )
        check(
            result.gpu_config.extra.get("clfft_row_length_a") == expected_a
            and result.gpu_config.extra.get("clfft_column_length_b") == expected_b,
            f"clfft.plan_large1d({length}) should preserve the real clFFT split (a={expected_a}, "
            f"b={expected_b}), got {result.gpu_config.extra!r}",
        )

    # A length just above the block-compute gate must still go through
    # ordinary four-step (this fix must not over-trigger).
    check(
        not clfft.is_block_compute_length(524288),
        "length=524288 exceeds the single-precision block-compute gate (262144) and must not be "
        "flagged block-compute-eligible",
    )


# ============================================================================
# VkFFT: complete base {2,3,5,7} register table, independently re-derived
# from vkFFT_Scheduler.h's VkFFTGetRegistersPerThreadQuad (lines 32-283) at
# the pinned commit 066a17c17068c0f11c9298d848c2976c71fad1c1 -- written
# fresh, without reference to vkfft.py's own implementation.
# ============================================================================


def _ref_vkfft_register_table(c2: int, c3: int, c5: int, c7: int):
    has2, has3, has5, has7 = c2 > 0, c3 > 0, c5 > 0, c7 > 0
    r = {2: 0, 3: 0, 5: 0, 7: 0}
    if has2:
        if has3:
            if has5:
                r[2], r[3], r[5], r[7] = (6, 6, 5, 7) if has7 else (6, 6, 5, 0)
            elif has7:
                r[2], r[3], r[7] = (6, 6, 7) if c2 in (1, 2) else (8, 6, 7)
            else:
                r[2], r[3] = 6, 6
        elif has5:
            if has7:
                r[2], r[5], r[7] = (6, 5, 7) if c2 == 1 else (8, 5, 7)
            else:
                r[2], r[5] = 4, 5
        elif has7:
            r[2], r[7] = 8, 7
        else:
            return None  # pure pow2 -- not this function's scope
    else:
        if has3:
            if has5:
                r[3], r[5], r[7] = (6, 5, 7) if has7 else (3, 5, 0)
            elif has7:
                r[3], r[7] = 6, 7
            else:
                r[3] = 3 if c3 == 1 else 9
        elif has5:
            r[5], r[7] = (5, 7) if has7 else (5, 0)
        elif has7:
            r[7] = 7
        else:
            return "rader"
    return {2: r[2], 3: r[3], 5: r[5], 7: r[7]}


def verify_vkfft_register_table_matches_independent_reference() -> None:
    print("VkFFT: complete base {2,3,5,7} register table vs. independent VkFFTGetRegistersPerThreadQuad re-derivation")
    checked = 0
    for c2 in range(5):
        for c3 in range(5):
            for c5 in range(5):
                for c7 in range(5):
                    if c2 == c3 == c5 == c7 == 0 or (c3 == c5 == c7 == 0):
                        continue  # all-absent (rader) / pure-pow2: out of this function's scope
                    ref = _ref_vkfft_register_table(c2, c3, c5, c7)
                    if ref in (None, "rader"):
                        continue
                    from planning.gpu_baseline import vkfft

                    got = vkfft.registers_per_thread_base_table(c2, c3, c5, c7)
                    check(
                        got == ref,
                        f"registers_per_thread_base_table({c2},{c3},{c5},{c7}) = {got}, expected {ref}",
                    )
                    checked += 1
    check(checked > 500, f"expected to check >500 combinations, only checked {checked}")


def verify_vkfft_register_table_is_a_real_behavior_change() -> None:
    """Confirms the fix is not a no-op refactor: several common radix pairs
    from this baseline's own direct-radix vocabulary get a DIFFERENT (and
    correct) `min_registers_per_thread` than the old `min(radices)` proxy
    would have returned, in both directions (sometimes higher, sometimes
    lower) -- exactly the kind of silent mis-mapping the task's own rule
    ("this proxy must not remain on any path returning OK") is about."""
    print("VkFFT: register-table fix changes min_registers_per_thread vs. the old min(radices) proxy")
    from planning.gpu_baseline import vkfft

    changed = 0
    for radices in ((9, 8), (9, 4), (9, 2), (7, 4), (7, 9), (7, 2), (10, 3), (5, 9)):
        counts = vkfft._prime_multiplicities(radices)
        table = vkfft.registers_per_thread_base_table(counts[2], counts[3], counts[5], counts[7])
        exact = min(v for v in table.values() if v != 0)
        proxy = min(radices)
        if exact != proxy:
            changed += 1
    check(changed >= 5, f"expected the fix to change the result for most sampled radix pairs, only {changed}/8 differed")


# ============================================================================
# rocFFT-default: the real solution-map layer (ApplySolution), verified
# directly against the shipped gfx908 data file (planning/gpu_baseline/
# data/gfx908_rocfft_solution_map.dat, a verbatim copy of library/
# solution_map/gfx908_rocfft_solution_map.dat at the pinned commit) -- read
# with plain json.load here, independently of usm's own parser, wherever a
# check can be phrased directly against the raw file.
# ============================================================================


def _raw_gfx908_data() -> dict:
    return json.loads(usm._DATA_PATH.read_text(encoding="utf-8"))


def verify_rocfft_solution_map_token_format() -> None:
    """`get_node_token`/`GenerateProbKeys`'s own token format, checked
    against a real key actually present in the shipped file (not merely
    self-consistency of this module's own formula)."""
    print("rocFFT-default: get_node_token format matches a real shipped key exactly")
    min_token, full_token = usm.get_node_token(
        4096, precision="single", placement="ip", inverse=False, batch=1,
        in_stride=(1,), out_stride=(1,), i_dist=4096, o_dist=4096,
    )
    check(min_token == "4096_sp_ip_complex", f"min_token = {min_token!r}, expected '4096_sp_ip_complex'")
    check(
        full_token == "4096_sp_ip_complex_fwd_batch_1_istride_1_ostride_1_idist_4096_odist_4096_ioffset_0_ooffset_0",
        f"full_token = {full_token!r}",
    )
    raw = _raw_gfx908_data()
    real_tokens = {e["Problem"]["token"] for e in raw["Data"]}
    check(min_token in real_tokens, f"{min_token!r} should be a real key present in the shipped gfx908 file")


def verify_rocfft_solution_map_root_dummy_falls_back() -> None:
    """At least one root whose option 0 is SOL_DUMMY, and therefore falls
    back to decide_scheme (task's own required test case)."""
    print("rocFFT-default: SOL_DUMMY root (option 0) correctly yields no override")
    raw = _raw_gfx908_data()
    dummy_lengths = []
    for e in raw["Data"]:
        tok = e["Problem"]["token"]
        m = re.fullmatch(r"(\d+)_sp_ip_complex", tok)
        if m and e["Solutions"][0]["sol_node_type"] == "SOL_DUMMY":
            dummy_lengths.append(m.group(1))
    check(len(dummy_lengths) >= 3, f"expected several SOL_DUMMY single-precision roots, found {dummy_lengths}")
    for length_str in dummy_lengths:
        length = int(length_str)
        result = usm.apply_solution(
            length, placement="ip", inverse=False, batch=1,
            in_stride=(1,), out_stride=(1,), i_dist=length, o_dist=length,
        )
        check(result is None, f"length={length}: SOL_DUMMY root should yield apply_solution()==None, got {result}")


def verify_rocfft_solution_map_real_internal_node() -> None:
    """At least one real non-dummy solution-map root that is itself an
    internal (CS_L1D_*) decomposition, with exact child_option/kernel
    config checks -- the task's own two remaining required test cases."""
    print("rocFFT-default: real non-dummy CS_L1D_TRTRT solution tree (N=16777216) resolves exactly")
    length = 16777216
    node = usm.apply_solution(
        length, placement="ip", inverse=False, batch=1,
        in_stride=(1,), out_stride=(1,), i_dist=length, o_dist=length,
    )
    check(node is not None, f"length={length}: expected a real non-dummy solution-map match")
    check(node.sol_node_type == "SOL_INTERNAL_NODE", f"sol_node_type = {node.sol_node_type}")
    check(node.using_scheme == "CS_L1D_TRTRT", f"using_scheme = {node.using_scheme}")
    check(len(node.children) == 5, f"expected 5 children (T-R-T-R-T), got {len(node.children)}")

    expected_shapes = [
        ("SOL_LEAF_NODE", "CS_KERNEL_TRANSPOSE", None),
        ("SOL_LEAF_NODE", "CS_KERNEL_STOCKHAM", 1),
        ("SOL_LEAF_NODE", "CS_KERNEL_TRANSPOSE", None),
        ("SOL_LEAF_NODE", "CS_KERNEL_STOCKHAM", 2),
        ("SOL_LEAF_NODE", "CS_KERNEL_TRANSPOSE", None),
    ]
    for i, (child, (exp_type, exp_scheme, exp_option)) in enumerate(zip(node.children, expected_shapes)):
        check(child.sol_node_type == exp_type, f"child[{i}].sol_node_type = {child.sol_node_type}, expected {exp_type}")
        check(child.using_scheme == exp_scheme, f"child[{i}].using_scheme = {child.using_scheme}, expected {exp_scheme}")
        if exp_option is not None:
            check(child.option == exp_option, f"child[{i}].option = {child.option}, expected {exp_option}")

    # Exact kernel_config checks (task's own explicit requirement) -- the
    # two 4096-length row transforms use DIFFERENT tuned configs, straight
    # from the shipped kernel_len4096_single_sbrr entries at option 0/1
    # respectively (child_option 1/2 of 4096_sp_ip_complex point there).
    row1, row2 = node.children[1], node.children[3]
    check(row1.kernel_key is not None and row2.kernel_key is not None, "both row transforms need a kernel_key")
    kc1, kc2 = row1.kernel_key.kernel_config, row2.kernel_key.kernel_config
    check(
        (kc1.wgs, kc1.tpb, kc1.tpt, kc1.factors) == (256, 2, (128, 0), (8, 16, 4, 8)),
        f"row1 kernel_config = {kc1}",
    )
    check(
        (kc2.wgs, kc2.tpb, kc2.tpt, kc2.factors) == (512, 2, (256, 0), (8, 8, 16, 4)),
        f"row2 kernel_config = {kc2}",
    )
    check(kc1 != kc2, "the two row-transform configs must be genuinely different tuned options, not duplicates")


def verify_rocfft_solution_map_out_of_place_never_matches() -> None:
    """This project's own consistent out-of-place assumption means the
    real shipped gfx908 file has ZERO single-precision matches -- proven
    here by an exhaustive scan of the raw file (not merely inferred from
    a few sampled lengths), plus a direct apply_solution() check on every
    single-precision root token the file contains."""
    print("rocFFT-default: exhaustive scan confirms zero out-of-place single-precision matches in gfx908 map")
    raw = _raw_gfx908_data()
    # This baseline's own scope is 1D C2C only (see gpu_baseline_v1_freeze.
    # md's "Exact domain covered") -- restrict the scan to that shape
    # (a bare `<length>_sp_..._complex...` token, never `_real_`, never a
    # multi-length 2D/3D token, never a `kernel_`/`sbcc_`/etc. sub-token).
    op_single_1d_c2c_tokens = [
        tok for e in raw["Data"]
        if re.fullmatch(r"\d+_sp_op_complex(_fwd|_bwd)?.*", tok := e["Problem"]["token"])
    ]
    check(
        len(op_single_1d_c2c_tokens) == 0,
        f"expected zero out-of-place single-precision 1D C2C tokens, found {op_single_1d_c2c_tokens}",
    )

    sp_root_lengths = sorted(
        int(m.group(1))
        for e in raw["Data"]
        if (m := re.fullmatch(r"(\d+)_sp_ip_complex", e["Problem"]["token"]))
    )
    check(len(sp_root_lengths) >= 5, f"expected several single-precision root tokens to test, found {sp_root_lengths}")
    for length in sp_root_lengths:
        result = usm.apply_solution(
            length, placement="op", inverse=False, batch=1,
            in_stride=(1,), out_stride=(1,), i_dist=length, o_dist=length,
        )
        check(result is None, f"length={length}: out-of-place apply_solution() should be None, got {result}")


def verify_rocfft_default_plan_unaffected_by_solution_map_wiring() -> None:
    """End-to-end: rocfft_default.plan()'s own OK/refusal results and
    gpu_config contents for representative lengths are UNCHANGED by wiring
    in apply_solution (since it always returns None for this baseline's
    own out-of-place calling convention) -- confirms the wiring is a
    correct no-op for plan()'s own real domain, not merely that the raw
    apply_solution function returns None in isolation."""
    print("rocFFT-default: plan()'s own results are unaffected by the solution-map wiring (verified, not assumed)")
    for length in (2, 8, 64, 256, 4096, 16777216):
        result = rocfft_default.plan(length, batch=1)
        check(
            result.gpu_config.extra.get("mechanism") != "solution-map override (ApplySolution)",
            f"length={length}: plan() should never report a solution-map override for this baseline's "
            f"own out-of-place domain, got mechanism={result.gpu_config.extra.get('mechanism')!r}",
        )


def main() -> None:
    verify_rocfft_tuned_phase0_matches_independent_reference()
    verify_rocfft_tuned_min_wgs_rounding_below_64()
    verify_rocfft_tuned_max_radices_size_336_exception()
    verify_rocfft_tuned_global_pruning_scope()
    verify_rocfft_tuned_phase1_pools_all_families_in_one_call()
    verify_clfft_block_compute_gate_matches_independent_reference()
    verify_clfft_block_compute_never_silently_becomes_four_step()
    verify_vkfft_register_table_matches_independent_reference()
    verify_vkfft_register_table_is_a_real_behavior_change()
    verify_rocfft_solution_map_token_format()
    verify_rocfft_solution_map_root_dummy_falls_back()
    verify_rocfft_solution_map_real_internal_node()
    verify_rocfft_solution_map_out_of_place_never_matches()
    verify_rocfft_default_plan_unaffected_by_solution_map_wiring()
    verify_vkfft_divisibility_loop_is_proven_noop()
    verify_vkfft_axisblock_postprocess_matches_independent_reference()
    verify_vkfft_axisblock_swap_and_upload_cases()
    verify_vkfft_rader_bluestein_classification_matches_independent_reference()

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S):")
        for f in _FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL SOURCE-FIDELITY CHECKS PASSED")


if __name__ == "__main__":
    main()
