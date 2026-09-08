from __future__ import annotations

"""rocFFT-style BASELINE planner.

Ported from the REAL, public rocFFT source, now living in the ROCm/
rocm-libraries monorepo under `projects/rocfft/` (the standalone `ROCm/
rocFFT` repo is deprecated), fetched and read directly (not recalled from
memory) at commit `bee97df517907c771de17189cb867d3c401285ae` of the
`develop` branch. Every table entry, formula, and threshold below carries
a citation into that source. See `docs/gpu_baseline_rocfft_research.md`
(a copy of the research report this module was built from) for the full
verification trail, including two confirmed comment-vs-code discrepancies
in the real source (reported here, never silently "corrected" to match
the comment instead of the executable code -- section 8 of the task this
package was built from is explicit that the source comment and
implementation for "largest 33% TPT" are inconsistent, and instructs
treating the executable source as authoritative):

* The "remove the largest 33% tpt" comment (tuning_kernel_tuner.cpp) is
  computed as `num_tpts_to_remove = (n-1)//2`, which is 33% only at n=3
  and approaches 50% as n grows. This module implements `(n-1)//2` (the
  real code), not literally 33%.
* The utilization-rate comment says "average rate < 1.0 or > 8.0"; the
  actual code never checks `avg_rate > 8.0` directly, only
  `max(heights) > 8.0` (which happens to bound `avg_rate` too, since a
  mean cannot exceed its own max, but is not the literal test described).

SCOPE LIMIT (explicitly marked, not silently overreached -- section 5 of
the task): rocFFT's own kernel-tuning system (the object of this port)
only ever tunes `KernelConfig` values at the LEAF nodes of ONE, already
fixed, non-tuned decomposition tree -- confirmed via a verbatim, still-
unresolved `// TODO- plan-tuning: build tree several times to generate
different trees` in `tuning_plan_tuner.cpp::EnumerateTrees`. This baseline
therefore only replicates rocFFT's SINGLE-KERNEL leaf-tuning search
(`SupportedKernelConfigs`'s 1D path) -- a length needing more than one
kernel (rocFFT's own multi-node decomposition trees, e.g. its large-1D/
SBRC/SBCC schemes) is reported UNSUPPORTED_GPU_ALGORITHM, exactly mirroring
rocFFT's own real scope, not a gap this port invented. The 2D-single
kernel path (`Supported2DKernelConfigs`, a materially different, simpler
uwide/wide TPT scheme) is likewise out of scope -- this repository only
ever plans 1D FFTs.

NON-NEGOTIABLE (see gpu_baseline/common.py's own module docstring): no
import from planning.search.fft_cost_model / planning.execution.fft_plan_cooperative's own
worker-count heuristic / planning.execution.fft_plan_persistent / planning.
fft_plan_lanes. The winner among candidates is chosen by REAL measured
M2NDP execution time (via planning.diagnostics.spill_probe's real build/run
infrastructure, reused exactly as the task instructs -- "reuse the
existing real build/run infrastructure where possible"), never by
planning.search.fft_cost_model.estimate_cost.
"""

import itertools
from dataclasses import dataclass, field, replace
from typing import Callable

from planning.core.fft_plan_core import FFTCodegenPlan
from planning.core.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile
from radix_spec import SUPPORTED_RADICES

from .common import (
    BaselineProvenance,
    BaselineResult,
    BaselineStatus,
    GPUKernelConfig,
    map_cooperative_kernel,
    unsupported,
    wrap_leaf_as_recursive_plan,
)

# ---------------------------------------------------------------------------
# Source provenance (section 7 of the baseline-freeze task this was built
# from). This is the ONE baseline module whose research pass captured a
# real, pinned commit SHA (via the GitHub Contents API alongside the raw-
# content fetch) -- see BaselineProvenance's own docstring for why clFFT/
# VkFFT's own provenance records instead carry a floating-branch-plus-date.
# ---------------------------------------------------------------------------
PROVENANCE = BaselineProvenance(
    library="rocFFT",
    upstream_repository="https://github.com/ROCm/rocm-libraries",
    upstream_commit="bee97df517907c771de17189cb867d3c401285ae (develop, projects/rocfft/)",
    source_files=(
        "projects/rocfft/library/src/tuning_kernel_tuner.cpp",
        "projects/rocfft/library/src/tuning_plan_tuner.cpp",
        "projects/rocfft/library/src/tuning_helper.cpp",
        "projects/rocfft/library/src/rocfft_offline_tuner.cpp",
        "projects/rocfft/library/src/rocfft_kernel_config_search.cpp",
        "projects/rocfft/library/src/include/function_map_key.h",
        "projects/rocfft/library/src/include/solution_map.h",
        "projects/rocfft/library/src/include/twiddles.h",
        "projects/rocfft/library/src/plan.cpp",
    ),
    baseline_version="gpu-baseline-v1",
    source_functions={
        "factorize": "tuning_kernel_tuner.cpp: Factorize",
        "get_max_radices_size": "tuning_kernel_tuner.cpp: GetMaxRadicesSize",
        "supported_threads_per_transform": "tuning_kernel_tuner.cpp: PowerSet + SupportedThreadsPerTransform",
        "is_bad_utilization": "tuning_kernel_tuner.cpp: GetUtilizationRate + SupportedKernelConfigs's own rejection site",
        "derive_max_tpb": "tuning_kernel_tuner.cpp: DeriveMaxTPB",
        "conservative_max_tpb": "tuning_kernel_tuner.cpp: ConservativeMaxTPB",
        "_configs_for_ordering": "tuning_kernel_tuner.cpp: SupportedKernelConfigs",
        "get_all_factorizations_for_phase1": "tuning_kernel_tuner.cpp: GetAllFactorizationsForPhase1",
        "propagate_best_factors_to_next_phase": "tuning_helper.cpp: TuningBenchmarker::PropagateBestFactorsToNextPhase",
        "tune": "rocfft_offline_tuner.cpp: offline_tune_problems + tuning_helper.cpp: FindWinnerForCurrNode",
    },
    notes=(
        "This module ports the OFFLINE TUNER's own search space only -- see "
        "gpu-rocfft-default (rocfft_default.py) for the SEPARATE production "
        "default-planning mechanism; the two are not alternate views of the "
        "same code path (confirmed via call-graph grep across the entire "
        "production surface, see docs/gpu_baseline_rocfft_default_research.md)."
    ),
)

# ---------------------------------------------------------------------------
# Constants -- library/src/tuning_kernel_tuner.cpp line 39 and
# library/src/include/twiddles.h line 30.
# ---------------------------------------------------------------------------
SUPPORTED_FACTORS: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17)
TWIDDLES_MAX_RADICES = 8
LDS_BYTE_LIMIT = 32 * 1024
BYTES_PER_ELEM = 8  # sizeof(float)*2 -- this project is single-precision (FP32) only

assert set(SUPPORTED_FACTORS) <= SUPPORTED_RADICES, (
    "every rocFFT supported_factors entry must be one this M2NDP butterfly "
    "generator implements -- if this ever fires, rocFFT's own factor set has "
    "grown past radix_spec.SUPPORTED_RADICES and needs an unsupported_radix "
    "path, not a silent substitution"
)

# ---------------------------------------------------------------------------
# Fixed baseline-wide search-space bounds (section 3: "fixed implementation
# parameters ... must remain FIXED for all GPU baseline candidates and must
# be documented"). rocFFT's own MIN_WGS/MAX_WGS default to 64/512 (tunable
# via env vars in the real source, per the research report's section 8e) --
# held fixed here, identically, for every rocFFT candidate this module ever
# builds.
# ---------------------------------------------------------------------------
MIN_WGS = 64
MAX_WGS = 512


class RocfftUnsupportedLengthError(Exception):
    """Raised when `length` has no factorization at all into
    `SUPPORTED_FACTORS` (rocFFT's own "Prime number" case)."""


# ---------------------------------------------------------------------------
# Factorize / GetMaxRadicesSize -- tuning_kernel_tuner.cpp lines 86-148.
# ---------------------------------------------------------------------------


def factorize(length: int) -> frozenset[tuple[int, ...]]:
    """`Factorize(length)` (tuning_kernel_tuner.cpp lines 108-134), ported
    exactly: recursive/exhaustive over `SUPPORTED_FACTORS`, every
    resulting factor list sorted ascending before being added to the
    result set -- so this returns SORTED MULTISETS (order stripped), never
    orderings; those are reintroduced later, in phase-0's own un-permuted
    use and phase-1's `get_all_factorizations_for_phase1`."""
    memo: dict[int, frozenset[tuple[int, ...]]] = {}

    def go(n: int) -> frozenset[tuple[int, ...]]:
        if n in memo:
            return memo[n]
        results: set[tuple[int, ...]] = set()
        for factor in SUPPORTED_FACTORS:
            if n % factor != 0:
                continue
            remain = n // factor
            if remain == 1:
                results.add((factor,))
            else:
                for remain_factors in go(remain):
                    combined = tuple(sorted((factor,) + remain_factors))
                    results.add(combined)
        frozen = frozenset(results)
        memo[n] = frozen
        return frozen

    return go(length)


def get_max_radices_size(all_factors_set: frozenset[tuple[int, ...]], *, length: int) -> int:
    """`GetMaxRadicesSize` (tuning_kernel_tuner.cpp lines 136-148): `(the
    shortest factorization's own factor count) + 2`, seeded at
    `TWIDDLES_MAX_RADICES + 1 = 9` before being pulled down by the real
    minimum. Includes the source's own hardcoded `length==336` exception
    (`--max_radices_size`, lines 499-500: "to avoid 336 from expanding to
    5 factors")."""
    min_size = TWIDDLES_MAX_RADICES + 1
    for factors in all_factors_set:
        min_size = min(min_size, len(factors))
    max_radices_size = min_size + 2
    if length == 336:
        max_radices_size -= 1
    return max_radices_size


# ---------------------------------------------------------------------------
# SupportedThreadsPerTransform -- tuning_kernel_tuner.cpp lines 150-186.
# ---------------------------------------------------------------------------


def power_set(factors: tuple[int, ...]) -> frozenset[tuple[int, ...]]:
    """`PowerSet` ported directly: every subset (including the empty one)
    of `factors`, as a sorted-order-preserved sub-tuple."""
    result: set[tuple[int, ...]] = {()}
    for f in factors:
        result |= {subset + (f,) for subset in result}
    return frozenset(result)


def supported_threads_per_transform(factorization: tuple[int, ...]) -> list[int]:
    """`SupportedThreadsPerTransform`: the power set's non-empty subsets,
    each reduced to its own product, deduped -- returned sorted ascending
    (the real source uses a `std::set`, which is already sorted; this
    matches that iteration order for downstream determinism)."""
    tpts: set[int] = set()
    for subset in power_set(factorization):
        if not subset:
            continue
        product = 1
        for f in subset:
            product *= f
        tpts.add(product)
    return sorted(tpts)


# ---------------------------------------------------------------------------
# GetUtilizationRate + rejection -- tuning_kernel_tuner.cpp lines 188-216.
# ---------------------------------------------------------------------------


def get_utilization_rate(length: int, factors: tuple[int, ...], tpt: int) -> tuple[list[float], float]:
    """`GetUtilizationRate`: `height_i = length / factor_i / tpt` for each
    factor, `avg = mean(heights)`. Returns `(heights, avg)`."""
    heights = [length / width / tpt for width in factors]
    avg = sum(heights) / len(factors)
    return heights, avg


def is_bad_utilization(length: int, factors: tuple[int, ...], tpt: int) -> bool:
    """The real rejection test (lines 533-543, `SupportedKernelConfigs`):
    `avg_rate < 1.0 || max(heights) > 8.0` -- ported using `max(heights)`
    directly (not `max(heights + [avg])`) since the two are behaviorally
    identical (a mean cannot exceed its own inputs' max), per the research
    report's own note. The comment's separate "> 8.0" clause on `avg_rate`
    is NOT a real, distinct check in the source -- see module docstring."""
    heights, avg = get_utilization_rate(length, factors, tpt)
    return avg < 1.0 or max(heights) > 8.0


# ---------------------------------------------------------------------------
# DeriveMaxTPB / ConservativeMaxTPB -- tuning_kernel_tuner.cpp lines 50-103.
#
# This baseline never sets use_3steps_large_twd (out of scope -- see module
# docstring's own SCOPE LIMIT: that flag only matters for rocFFT's own
# large-1D multi-kernel tree, which this single-leaf-kernel-only port does
# not build), so `get_large_twd_base_steps` is ported for documentation
# completeness but DeriveMaxTPB below always takes the `ltwd_base == 8`
# (i.e. "no charge") path, exactly the real formula's own behavior in that
# case.
# ---------------------------------------------------------------------------


def get_large_twd_base_steps(large1d_len: int, use_3steps: bool) -> tuple[int, int]:
    """`get_large_twd_base_steps` (plan.cpp lines 5214-5234): `base =
    use3steps ? clamp(4, 6, (ceil_log2(large1DLen)+2)//3) : 8`. Unused by
    this baseline's own `derive_max_tpb` (always called with
    `use_3steps_large_twd=False` -- see module docstring), ported for
    completeness/documentation only."""
    if not use_3steps:
        return 8, 1
    bits = (large1d_len - 1).bit_length() if large1d_len > 1 else 0
    base = max(4, min(6, (bits + 2) // 3))
    return base, 1


def derive_max_tpb(
    length: int, *, half_lds: bool, tpt: int, wgs_bound: int,
) -> int:
    """`DeriveMaxTPB`, single precision, `use_ltwd_3steps=False` always
    (see module docstring)."""
    bytes_per_batch = length * BYTES_PER_ELEM
    if half_lds:
        bytes_per_batch //= 2
    tpb = LDS_BYTE_LIMIT // bytes_per_batch
    while tpt * tpb > wgs_bound:
        tpb -= 1
    return tpb


def conservative_max_tpb(length: int) -> int:
    """`ConservativeMaxTPB`, single precision."""
    bytes_per_batch = length * BYTES_PER_ELEM
    conservative = LDS_BYTE_LIMIT // bytes_per_batch
    if length >= 1024:
        conservative += 1
    return conservative


def _is_po2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


# ---------------------------------------------------------------------------
# KernelConfig + per-ordering candidate generation --
# tuning_kernel_tuner.cpp's SupportedKernelConfigs (lines ~460-760).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelConfig:
    """Mirrors the real `struct KernelConfig` (function_map_key.h lines
    34-39) field-for-field for the tunable subset this baseline's single-
    kernel-leaf scope actually searches: `factors` (ORDERED -- Stockham
    pass order, not a multiset), `threads_per_transform` (kept as a plain
    int here, not the real `std::array<int,2>`, since this baseline never
    plans the 2D-single scheme that uses the second slot -- see module
    docstring's SCOPE LIMIT), `transforms_per_block`, `workgroup_size`,
    `half_lds`.

    `direct_to_from_reg`/`intrinsic_buffer_inst`/`use_3steps_large_twd`
    are recorded for fidelity (real rocFFT's current source pins
    `direct_to_from_reg` to `True` unconditionally -- `False` is
    literally commented out, "from current benchmark result, dir-reg mode
    always ranks high" -- so this baseline mirrors that pin) but have NO
    EFFECT on the resulting M2NDP plan: M2NDP's own load/store model has
    no "read straight into registers, bypassing the LDS-equivalent
    scratchpad" mode at all (see the project audit's own section E), and
    unlike `half_lds` (which measurably changes the scratchpad byte
    budget `derive_max_tpb` checks), `direct_to_from_reg` is not even a
    parameter to the real `DeriveMaxTPB` -- it is a pure GPU code-
    generation/instruction-selection strategy with no scratchpad-sizing
    consequence in rocFFT's OWN formula either. Carrying it as inert
    metadata (never as a M2NDP-mapping gate) is the accurate reflection of
    that: it is not silently "ported," and it does not manufacture a
    false UNSUPPORTED_HARDWARE_MAPPING for every single candidate either.
    """

    factors: tuple[int, ...]
    threads_per_transform: int
    transforms_per_block: int
    workgroup_size: int
    half_lds: bool
    direct_to_from_reg: bool = True
    intrinsic_buffer_inst: bool = False
    use_3steps_large_twd: bool = False


def _configs_for_ordering(
    length: int, factors: tuple[int, ...], *, is_phase0: bool,
) -> list[KernelConfig]:
    """One ordering's worth of `KernelConfig`s -- `SupportedKernelConfigs`'s
    body for a single already-decided `factors` sequence: TPT candidates
    (power set), utilization-rate pruning, then for each surviving TPT a
    workgroup-size-bucket sweep deriving TPB (section 8's items a-e; the
    exact wgs-bucket loop is reconstructed from the research report's own
    documented bullet points -- max_tpb via DeriveMaxTPB, reject if it
    exceeds ConservativeMaxTPB, try at most `max_tpb` and `max_tpb+1`,
    accept only if the resulting `final_wgs` lands within the current
    64-wide bucket, respects `length>=64 => final_wgs>=64`, and a
    power-of-2 length requires `length % final_wgs == 0`).

    `is_phase0`: gates the radix-count cap (`get_max_radices_size`, only
    applied in phase 0 -- see that function's own docstring) and the
    "largest ~half of distinct TPTs" pruning (section 8d, phase-0 only).
    """
    if is_phase0:
        all_factors = factorize(length)
        max_radices = get_max_radices_size(all_factors, length=length)
        if len(factors) > max_radices:
            return []

    tpts = supported_threads_per_transform(factors)
    bad = {tpt for tpt in tpts if is_bad_utilization(length, factors, tpt)}
    if bad and len(tpts) > len(bad):
        tpts = [t for t in tpts if t not in bad]

    if is_phase0 and len(tpts) > 0:
        num_to_remove = (len(tpts) - 1) // 2
        if num_to_remove > 0:
            sorted_tpts = sorted(tpts)
            keep = set(sorted_tpts[: len(sorted_tpts) - num_to_remove])
            tpts = [t for t in tpts if t in keep]

    configs: list[KernelConfig] = []
    tpb_to_remove: set[int] = set()
    tpt_is_length: set[int] = set()

    # FIDELITY FIX (found during the Phase-1-through-6 baseline audit,
    # 2026-09-08): the first version of this loop always started the
    # wgs-bucket sweep at the fixed MIN_WGS=64 -- this produced ZERO
    # surviving candidates for short power-of-two lengths (8, 16), which
    # was originally (wrongly) reported as evidence that real rocFFT must
    # route such lengths through a separate hand-written kernel path. A
    # dedicated source re-check (docs/gpu_baseline_rocfft_default_research
    # .md) found real rocFFT's own tuner (tuning_kernel_tuner.cpp:490-491)
    # actually LOWERS min_wgs for a length smaller than it: `min_wgs =
    # (length < min_wgs) ? length : min_wgs`. The exact 64-rounding order
    # around that line was not pinned down by that research pass, so this
    # is a best-effort, explicitly-flagged reconstruction of the
    # confirmed DIRECTION (the floor shrinks for short lengths), not a
    # byte-for-byte port of an unseen exact formula.
    effective_min_wgs = length if length < MIN_WGS else MIN_WGS
    for tpt in tpts:
        if tpt == length:
            tpt_is_length.add(tpt)
        for half_lds in (False, True):
            for wgs_bucket in range(effective_min_wgs, MAX_WGS + 1, 64):
                max_tpb = derive_max_tpb(length, half_lds=half_lds, tpt=tpt, wgs_bound=wgs_bucket)
                if max_tpb < 1:
                    continue
                if max_tpb > conservative_max_tpb(length):
                    continue
                num_tpb_try = 1 if tpt * max_tpb == wgs_bucket else 2
                for delta in range(num_tpb_try):
                    tpb = max_tpb + delta
                    if tpb < 1:
                        continue
                    final_wgs = tpt * tpb
                    if final_wgs <= wgs_bucket - 64:
                        continue
                    if final_wgs > MAX_WGS:
                        continue
                    if length >= 64 and final_wgs < 64:
                        continue
                    if _is_po2(length) and length % final_wgs != 0:
                        continue
                    if tpt == length:
                        tpb_to_remove.add(tpb)
                    configs.append(
                        KernelConfig(
                            factors=factors, threads_per_transform=tpt,
                            transforms_per_block=tpb, workgroup_size=final_wgs,
                            half_lds=half_lds,
                        )
                    )

    # Section 8b: if there is at least one TPT choice besides tpt==length,
    # drop every tpt==length config AND every config elsewhere that shares
    # one of those "bad" TPB values.
    if tpt_is_length and len(tpts) >= 2 and tpb_to_remove:
        configs = [
            c for c in configs
            if c.threads_per_transform != length and c.transforms_per_block not in tpb_to_remove
        ]

    return configs


# ---------------------------------------------------------------------------
# Phase 0 / Phase 1 -- tuning_helper.cpp / tuning_kernel_tuner.cpp.
# ---------------------------------------------------------------------------


def phase0_candidates(length: int) -> list[KernelConfig]:
    """Phase 0: every un-permuted (ascending, as `factorize` returns it)
    factorization of `length`, each producing its own `_configs_for_
    ordering` candidates."""
    configs: list[KernelConfig] = []
    for factors in sorted(factorize(length)):
        configs.extend(_configs_for_ordering(length, factors, is_phase0=True))
    return configs


def get_all_factorizations_for_phase1(good_factors: tuple[int, ...]) -> list[tuple[int, ...]]:
    """`GetAllFactorizationsForPhase1` (tuning_kernel_tuner.cpp lines
    241-299), ported exactly: every distinct permutation OTHER than the
    given (ascending) one via `next_permutation`-equivalent enumeration;
    if that count exceeds 6, throw it away and instead use `2*len`
    cyclic shifts of the original sequence and of its reverse (the
    sliding-double-array trick) -- note this fallback DOES re-include the
    original ascending order as its own shift-0 entry, exactly as the
    real source does (not "fixed" here)."""
    n = len(good_factors)
    seen: set[tuple[int, ...]] = {good_factors}
    permutations: list[tuple[int, ...]] = []
    for p in sorted(set(itertools.permutations(good_factors))):
        if p in seen:
            continue
        permutations.append(p)

    if len(permutations) > 6:
        permutations = []
        reversed_factors = tuple(reversed(good_factors))
        doubled = good_factors * 2
        doubled_rev = reversed_factors * 2
        for i in range(n):
            permutations.append(doubled[i : i + n])
            permutations.append(doubled_rev[i : i + n])
    return permutations


def propagate_best_factors_to_next_phase(
    phase0_results: list[tuple[KernelConfig, float]],
) -> list[tuple[int, ...]]:
    """`TuningBenchmarker::PropagateBestFactorsToNextPhase` (tuning_helper.
    cpp lines 277-306): sort phase-0 results by measured time ascending,
    de-duplicate by factor multiset in first-seen (== best-time) order,
    keep at most the best 3."""
    ranked = sorted(phase0_results, key=lambda pair: pair[1])
    seen: set[tuple[int, ...]] = set()
    best: list[tuple[int, ...]] = []
    for config, _time in ranked:
        key = tuple(sorted(config.factors))
        if key in seen:
            continue
        seen.add(key)
        best.append(key)
        if len(best) >= 3:
            break
    return best


def phase1_candidates(good_factor_families: list[tuple[int, ...]], length: int) -> list[KernelConfig]:
    """Phase 1: for each of the (up to 3) best un-ordered factorizations
    propagated from phase 0, every permutation `get_all_factorizations_
    for_phase1` returns, each producing its own `_configs_for_ordering`
    candidates (`is_phase0=False` -- no radix-count cap, no "largest half"
    pruning, matching the real source's own `no_permutation` gating)."""
    configs: list[KernelConfig] = []
    for family in good_factor_families:
        for ordering in get_all_factorizations_for_phase1(family):
            configs.extend(_configs_for_ordering(length, ordering, is_phase0=False))
    return configs


# ---------------------------------------------------------------------------
# M2NDP mapping + real-measurement winner selection.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkOutcome:
    """One real (or simulated-real) measurement of one M2NDP-mapped
    candidate -- mirrors the real driver's own occupancy<0/occupancy==1
    skip conditions (section 10 of the research report) as `ok=False`,
    `ndp_cycles=None`."""

    ok: bool
    ndp_cycles: int | None
    spill_free: bool | None = None
    log: str = ""


BenchmarkFn = Callable[[FFTCodegenPlan, TargetProfile], BenchmarkOutcome]


def default_benchmark(plan: FFTCodegenPlan, target: TargetProfile) -> BenchmarkOutcome:
    """The default `BenchmarkFn`: reuses planning.diagnostics.spill_probe's real
    Mojo -> llc -> M2NDP-Detour build+run infrastructure exactly as
    instructed ("For M2NDP, reuse the existing real build/run
    infrastructure where possible") -- this is the M2NDP equivalent of
    rocFFT's own `hipEvent`-based real GPU timing (research report
    section 10), never a static cost-model estimate. Needs the real
    toolchain present; import is local so every pure candidate-generation
    function above stays importable/testable without it (mirroring
    planning.diagnostics.spill_probe's own "toolchain-free unless a caller opts in"
    discipline)."""
    from planning.diagnostics.spill_probe import probe_spill_free

    from .common import wrap_leaf_as_recursive_plan

    recursive_plan = wrap_leaf_as_recursive_plan(
        length=plan.length, total_ffts=plan.replicas_per_round(), inverse=plan.inverse, built_plan=plan,
    )
    result = probe_spill_free(recursive_plan, target=target)
    if not (result.build_ok and result.run_ok):
        return BenchmarkOutcome(ok=False, ndp_cycles=None, spill_free=None, log=result.log)
    # A real spill is a correctness failure, not merely "slow" -- treated as
    # a rejection here (occupancy==1-equivalent: this candidate is not a
    # viable winner), never averaged into a cost score. See [[fft-spill-
    # hard-filter]] and this project's own spill_probe discipline.
    if not result.spill_free:
        return BenchmarkOutcome(ok=False, ndp_cycles=result.ndp_cycles, spill_free=False, log=result.log)
    return BenchmarkOutcome(ok=True, ndp_cycles=result.ndp_cycles, spill_free=True, log=result.log)


@dataclass(frozen=True)
class RocfftCandidateOutcome:
    """One rocFFT candidate's own full, honest outcome -- kept distinct
    from `BaselineResult` (which is this module's own eventual return
    type) because section 5 of the task requires recording EVERY
    candidate the staged search considered, not just the final winner:
    "Do NOT hide failed GPU-baseline configurations. A failure itself is a
    scientifically useful baseline result." """

    config: KernelConfig
    mapping: BaselineResult  # OK, or UNSUPPORTED_HARDWARE_MAPPING/CURRENT_CODEGEN/RESOURCE_INFEASIBLE
    benchmark: BenchmarkOutcome | None = None  # None: never reached real measurement


def _map_config(
    length: int, radices: tuple[int, ...], config: KernelConfig, *, total_ffts: int,
    inverse: bool, kernel_name: str, target: TargetProfile,
) -> BaselineResult:
    gpu_config = GPUKernelConfig(
        source="rocfft", length=length, radices=radices,
        extra={
            "threads_per_transform": config.threads_per_transform,
            "transforms_per_block": config.transforms_per_block,
            "workgroup_size": config.workgroup_size,
            "half_lds": config.half_lds,
            "direct_to_from_reg": config.direct_to_from_reg,
            "intrinsic_buffer_inst": config.intrinsic_buffer_inst,
            "use_3steps_large_twd": config.use_3steps_large_twd,
        },
    )
    inverse_scale = (1.0 / length) if inverse else None
    return map_cooperative_kernel(
        length=length, radices=radices, workers_per_fft=config.threads_per_transform,
        fft_slots_wanted=config.transforms_per_block, total_ffts=total_ffts,
        inverse=inverse, inverse_scale=inverse_scale, kernel_name=kernel_name,
        target=target, gpu_config=gpu_config,
    )


def tune(
    length: int,
    *,
    total_ffts: int = 1,
    inverse: bool = False,
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
    benchmark_fn: BenchmarkFn = default_benchmark,
    kernel_name: str = "FFTRocfft",
) -> tuple[BaselineResult, list[RocfftCandidateOutcome]]:
    """The full two-phase rocFFT-style tuning search for one single-kernel-
    representable length, returning `(winner, every_candidate_outcome)` --
    the second element exists precisely so a caller can report every
    considered candidate (mapped-but-rejected, mapping-infeasible, or
    actually benchmarked), never just the winner.

    Phase 0: every un-permuted factorization's own candidates, each mapped
    onto M2NDP and -- only for those that map OK -- benchmarked via
    `benchmark_fn`. Best-3 factor families propagated by measured time
    (`propagate_best_factors_to_next_phase`).

    Phase 1: every permutation (or cyclic-shift fallback) of those 3
    families' own candidates, mapped and benchmarked the same way.

    Winner: the lowest real `ndp_cycles` among every OK-mapped,
    successfully-benchmarked (`BenchmarkOutcome.ok`) candidate across
    BOTH phases -- never `fft_cost_model.estimate_cost`. If nothing ever
    reaches a successful benchmark, the returned `BaselineResult` carries
    whichever status best characterizes why (UNSUPPORTED_GPU_ALGORITHM if
    `length` doesn't factor into `SUPPORTED_FACTORS` at all;
    UNSUPPORTED_HARDWARE_MAPPING/UNSUPPORTED_CURRENT_CODEGEN/
    RESOURCE_INFEASIBLE if every mapped candidate was infeasible;
    COMPILE_FAILURE/SPILLING if mapping succeeded but every real probe
    failed).
    """
    try:
        all_factors = factorize(length)
    except RecursionError:
        all_factors = frozenset()
    if not all_factors:
        gpu_config = GPUKernelConfig(source="rocfft", length=length, radices=())
        return unsupported(
            BaselineStatus.UNSUPPORTED_GPU_ALGORITHM, gpu_config,
            f"length={length} has no factorization into rocFFT's own supported "
            f"factor set {SUPPORTED_FACTORS} (a 'Prime number' case in the real "
            f"source's own terms)",
        ), []

    outcomes: list[RocfftCandidateOutcome] = []
    phase0_measured: list[tuple[KernelConfig, float]] = []

    for config in phase0_candidates(length):
        mapping = _map_config(
            length, config.factors, config, total_ffts=total_ffts, inverse=inverse,
            kernel_name=kernel_name, target=target,
        )
        if mapping.status is not BaselineStatus.OK:
            outcomes.append(RocfftCandidateOutcome(config=config, mapping=mapping))
            continue
        bench = benchmark_fn(mapping.plan, target)
        outcomes.append(RocfftCandidateOutcome(config=config, mapping=mapping, benchmark=bench))
        if bench.ok and bench.ndp_cycles is not None:
            phase0_measured.append((config, float(bench.ndp_cycles)))

    best_families = propagate_best_factors_to_next_phase(phase0_measured)

    for config in phase1_candidates(best_families, length):
        mapping = _map_config(
            length, config.factors, config, total_ffts=total_ffts, inverse=inverse,
            kernel_name=kernel_name, target=target,
        )
        if mapping.status is not BaselineStatus.OK:
            outcomes.append(RocfftCandidateOutcome(config=config, mapping=mapping))
            continue
        bench = benchmark_fn(mapping.plan, target)
        outcomes.append(RocfftCandidateOutcome(config=config, mapping=mapping, benchmark=bench))

    successful = [
        o for o in outcomes
        if o.benchmark is not None and o.benchmark.ok and o.benchmark.ndp_cycles is not None
    ]
    if not successful:
        # Characterize the aggregate failure honestly rather than picking an
        # arbitrary single diagnostic -- see this function's own docstring.
        # This is NOT an unsupported-algorithm case (the length DID factor
        # into SUPPORTED_FACTORS -- checked above). FIDELITY UPDATE
        # (2026-09-08 audit): this branch used to claim (as a plausibility
        # guess) that a length hitting it "almost certainly" has a separate
        # production kernel path in real rocFFT -- that guess is now
        # CONFIRMED by direct source tracing (docs/
        # gpu_baseline_rocfft_default_research.md): production rocFFT never
        # runs this tuning search for such lengths at all; it uses a
        # compiled-in per-length KernelConfig table instead (see
        # rocfft_default.py, the separate `gpu-rocfft-default` baseline this
        # audit added specifically to represent that mechanism). This
        # baseline (`gpu-rocfft`/`gpu-rocfft-tuned`) is a faithful port of
        # the TUNER's own search space specifically, so an empty result
        # here means only that OUR reconstruction of that search space
        # still produced nothing survivable for this length after the
        # min-wgs fix above -- a limitation of this port's own
        # completeness, not a proven M2NDP hardware wall or a claim about
        # what real rocFFT does in production.
        if not outcomes:
            gpu_config = GPUKernelConfig(source="rocfft", length=length, radices=())
            return unsupported(
                BaselineStatus.UNSUPPORTED_CURRENT_CODEGEN, gpu_config,
                f"length={length}: factored successfully into {SUPPORTED_FACTORS}, but "
                f"this baseline's own ported reconstruction of rocFFT's tuning search "
                f"space (SupportedKernelConfigs) produced zero surviving KernelConfig "
                f"candidates -- see gpu-rocfft-default for the SEPARATE, confirmed-real "
                f"production mechanism (a compiled-in per-length table) rocFFT actually "
                f"uses for such lengths instead of this tuner search",
            ), outcomes
        mapping_failed = [o for o in outcomes if o.mapping.status is not BaselineStatus.OK]
        if len(mapping_failed) == len(outcomes):
            worst = mapping_failed[0].mapping
            return worst, outcomes
        bench_failed = [o for o in outcomes if o.benchmark is not None and not o.benchmark.ok]
        spill_failed = [o for o in bench_failed if o.benchmark.spill_free is False]
        if spill_failed:
            gpu_config = spill_failed[0].mapping.gpu_config
            return unsupported(
                BaselineStatus.SPILLING, gpu_config,
                f"length={length}: every real-measured candidate that mapped onto "
                f"M2NDP spilled (checked {len(spill_failed)} of {len(outcomes)})",
            ), outcomes
        gpu_config = bench_failed[0].mapping.gpu_config if bench_failed else outcomes[0].mapping.gpu_config
        return unsupported(
            BaselineStatus.COMPILE_FAILURE, gpu_config,
            f"length={length}: every mapped candidate's real build/run probe failed "
            f"(checked {len(outcomes)} candidates)",
        ), outcomes

    winner = min(successful, key=lambda o: o.benchmark.ndp_cycles)
    recursive_plan = wrap_leaf_as_recursive_plan(
        length=length, total_ffts=total_ffts, inverse=inverse, built_plan=winner.mapping.plan,
    )
    result = BaselineResult(
        status=BaselineStatus.OK,
        gpu_config=winner.mapping.gpu_config,
        plan=recursive_plan,
        ndp_cycles=winner.benchmark.ndp_cycles,
        spill_free=winner.benchmark.spill_free,
    )
    return result, outcomes


def plan(
    length: int,
    *,
    batch: int = 1,
    inverse: bool = False,
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
    benchmark_fn: BenchmarkFn = default_benchmark,
) -> BaselineResult:
    """Top-level rocFFT-style baseline entry point -- see `tune` for the
    full two-phase search and every candidate it considered."""
    result, _outcomes = tune(length, total_ffts=batch, inverse=inverse, target=target, benchmark_fn=benchmark_fn)
    return result
