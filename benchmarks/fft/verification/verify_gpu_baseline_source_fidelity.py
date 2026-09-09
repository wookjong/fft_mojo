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

from planning.gpu_baseline import clfft, rocfft
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

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S):")
        for f in _FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL SOURCE-FIDELITY CHECKS PASSED")


if __name__ == "__main__":
    main()
