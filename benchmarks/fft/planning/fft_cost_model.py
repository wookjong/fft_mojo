from __future__ import annotations

"""Approximate, transparent cost metrics for a already-built RecursiveFFTPlan
-- for ranking/debugging candidates from fft_plan_search.py, not a precise
M2NDP performance model (real behavior depends on DRAM access pattern,
uthread scheduling, register spills, and simulator scheduling details this
module does not attempt to model). Every term below is a documented
heuristic over the plan's own already-decided structure, not a fitted
formula -- see PlanMetrics' own field comments for what each one means and
why, and CostWeights for how they combine. Nothing here makes a planning
decision; it only measures one already-built plan.
"""

from dataclasses import dataclass

from planning.fft_plan_recursive import (
    FFTLeafPlan,
    FFTNode,
    FFTRecursiveNodePlan,
    PhysicalTransposePlan,
    RecursiveFFTPlan,
    flatten_recursive_node,
)
from planning.fft_plan_core import FFTCodegenPlan
from planning.target_profile import TargetProfile

# Composite/prime radices confirmed clean as a leaf's *first* stage
# (reading straight from DRAM) but not as a later stage (reading its
# operands out of scratchpad instead):
#
# * (6, 9) as an *adjacent stage pair*: fft_plan_core._prime_factors_
#   supported's own comment documents the concrete N=54=(6,9) spill this
#   was originally based on (radix-9 stage, immediately preceded by
#   radix-6). Not universal: a direct (4, 9) chain (N=36, forced via
#   allowed_radix_composites) built and ran clean, zero spill, and 6/9
#   standalone (N=6, N=9) are clean too -- so whatever makes this pairing
#   risky depends on the specific (6, 9) sequence, not "radix 6 or 9 in a
#   non-first position" alone. Scored here by adjacent-pair membership in
#   _RISKY_RADIX_PAIRS below, kept in sync by hand with codegen.fft_codegen.
#   _RISKY_RADIX_PAIRS (not imported -- this module is deliberately
#   toolchain/codegen-free, see this module's own docstring, and planning
#   code does not import from codegen) -- the same set codegen's own
#   _stage_compute_lanes floors to compute_lanes=1 for. This used to be a
#   standalone-radix set (_NON_FIRST_STAGE_RISKY_RADICES
#   = {6, 9}) that over-counted risk for the confirmed-safe (4, 9) shape;
#   fixed 2026-08-27 alongside codegen's own sequence-aware fix so this
#   score and the actual narrowing decision agree on what is risky.
#   Root cause confirmed 2026-08-27 (see codegen.fft_codegen.
#   _ALWAYS_NARROW_RADICES's own comment): the same M2NDP-Detour `ReadCsr`
#   vlenb gap as 10/11/13/17 below -- compute_lanes=1 (not codegen's own
#   narrow_middle_stages default, which only halves, and only for a
#   genuine middle stage; N=54=(6,9) is a 2-stage kernel, so its radix-9
#   stage is technically "last") confirmed clean by direct probe.
# * 11, 13, 17: confirmed by direct probe (each forced into a (4, r)
#   chain -- N=44, N=52, N=68 respectively, r as the second/scratchpad-
#   reading stage): all three spill *and* silently produce an all-zero
#   (wrong, not just slow) result -- and, unlike 6/9, confirmed risky even
#   completely alone (N=11/13/17 as a standalone single-stage kernel,
#   2026-08-27), so reclassified here from _NON_FIRST_STAGE_RISKY_RADICES
#   to _ALWAYS_RISKY_RADICES: the earlier classification undersold the
#   real risk, since it was only ever probed embedded in a (4, r) chain.
#   Fixed at the codegen level by `narrow_middle_stages`'s own unconditional
#   floor-to-1 for these radices (see codegen.fft_codegen.
#   _ALWAYS_NARROW_RADICES) -- this risk_score term stays anyway as a
#   cost-model-visible signal independent of whether that codegen fix is
#   in effect for a given render.
#
# `10` fails outright at *any* stage position (N=160/320, and reconfirmed
# by the same probe as (4, 10) = N=40 -- spills and mismatches even
# directly after a radix-4 first stage) -- see the same fft_plan_core
# comment. Same fix and same root cause as 11/13/17 above.
_RISKY_RADIX_PAIRS = frozenset({(6, 9)})
_ALWAYS_RISKY_RADICES = frozenset({10, 11, 13, 17})

# Radix 5 as a *genuine* middle stage (neither first nor last -- has a
# preceding stage AND a following one, e.g. the 5 in a 3-stage (3, 5, 7)
# leaf) spills on real hardware in a way `narrow_middle_stages` cannot
# fix, unlike every other middle-stage liability this module tracks.
# Confirmed 2026-08-28 by direct probe of N=105 (a single fused (3, 5, 7)
# leaf, isolated -- no transpose stage, no other kernel) at every
# compute_lanes this project ships or could plausibly ship:
#
#   compute_lanes=4, narrow_middle_stages=False (no narrowing at all):
#     stage_1 (radix 5) spills, 128-byte frame
#   compute_lanes=4, narrow_middle_stages=True (stage_1 renders at
#     narrowed width 2, this project's own real default combination):
#     stage_1 spills, 128-byte frame -- identical to the unnarrowed case
#   compute_lanes=2 (stage_1 narrows further to 1, the floor):
#     stage_1 spills, 1296-byte frame -- WORSE, not better
#   compute_lanes=1 (already at the floor, narrowing is a no-op):
#     stage_1 AND stage_2 (radix 7, the *last* stage) both spill,
#     1296- and 64-byte frames respectively
#   loop_stages=True at compute_lanes=4/narrow_middle_stages=True:
#     stage_1 still spills, same 128-byte frame -- rules out per-batch
#     unrolling/function size as the mechanism too
#
# So compute_lanes narrowing -- the fix for every *other* middle-stage
# liability this module and codegen.fft_codegen track -- does not apply
# here at any width, and the one lever that does exist (narrower) makes
# it worse past width 2. This also retroactively explains why N=630's own
# FFTRecNear0::stage_1() (the exact case narrow_middle_stages was
# originally built around) was previously reported "confirmed clean at
# the halved width": that claim was checking run_fft_test.sh's PASS/FAIL
# reference check, not the spill warning specifically -- and N=630's
# default-tile plan does print `[PASS] (spill warning!)`, i.e. it was
# never actually spill-free, just not (this time) numerically wrong,
# exactly the "it happened to pass its reference check" case [[fft-spill-
# hard-filter]] warns against trusting.
#
# No codegen-level fix exists yet (nothing here floors or gates
# compute_lanes for this -- there is no width that helps), so this is a
# *cost-model-only* signal for now: it steers estimated_cost away from
# radix-5-as-middle and, more importantly, means planning.spill_probe.
# probe_and_rerank_candidates' own hard exclusion actually has somewhere
# else to fall back to (e.g. N=105's own (5, 7)+(3,) or (7,)+(3, 5) split
# candidates, which put radix 5 first or last instead of in the middle --
# both confirmed clean positions) rather than exhausting every candidate
# and raising NoSpillFreeCandidateError. A plain default (no --verify-
# spill-free) build of an N whose only decomposition has radix 5 in a
# genuine middle position has no protection from this yet.
_RISKY_AS_MIDDLE_RADICES = frozenset({5})

# Real M2NDP runs (benchmark_fft_candidates.sh, --batch sweep at N=1024 plus
# one N=16384 point) of a transpose stage's own total_uthreads vs. whether a
# smaller tile (more, smaller tiles) still wins over a bigger one:
#
#   1024 uthreads (N=1024, batch=1): tile=1x1 wins (1252 vs. 1516 cycles)
#   2048 uthreads (N=1024, batch=2): tile=1x1 wins (1355 vs. 1471 cycles)
#   4096 uthreads (N=1024, batch=4): tile=1x1 LOSES (1786 vs. 1560 cycles)
#  16384 uthreads (N=16384, batch=1): tile=1x1 LOSES (2258 vs. 1787 cycles)
#
# So CostWeights.transpose_tile_count's own "more/smaller tiles is usually
# faster" (see its own comment) only holds up to some point -- past it, the
# same extra tiles apparently cost more in launch/scheduling overhead than
# they gain in parallelism. Only 4 points, all at one of two N -- nowhere
# near enough to fit *where* the crossover really sits, so this uses the
# last CONFIRMED-still-winning point (2048) as the threshold, the same
# "last confirmed-safe point, not a guessed midpoint" discipline target_
# profile.DEFAULT_TARGET_PROFILE's own max_concurrent_scratchpad_bytes
# comment already uses. Reread with more benchmark_fft_candidates.sh data
# before trusting this far from N=1024/16384.
_TILE_PARALLELISM_SATURATION_UTHREADS = 2048


@dataclass(frozen=True)
class PlanMetrics:
    recursion_depth: int
    leaf_kernel_count: int
    transpose_kernel_count: int
    total_leaf_stage_count: int
    total_transpose_tiles: int
    # The single busiest PRE/MIDDLE/POST transpose stage's own total_uthreads
    # (0 if this plan has no transpose stage at all, i.e. a single fused
    # leaf) -- see CostWeights.tile_oversaturation_penalty's own comment for
    # why this, not total_transpose_tiles (summed across stages), is the
    # quantity the "smaller tile stops helping" crossover was measured
    # against.
    max_transpose_stage_uthreads: int
    estimated_dram_bytes: int
    max_scratchpad_bytes: int
    worst_worker_utilization: float  # 1.0 = no cooperative leaf ever idles a worker
    radix_risk_score: float          # 0.0 = every leaf's radix sequence is the confirmed-safe kind
    # How many PRE/MIDDLE/POST transpose stages have a tile that does NOT
    # divide `rows`/`cols` exactly (`rows % tile_rows != 0 or cols %
    # tile_cols != 0` -- see fft_plan_search.generate_tile_candidates' own
    # comment for the real-hardware finding this is based on). 0 means
    # every transpose stage's own tile evenly divides both dimensions --
    # confirmed real-hardware clean at every no-tail size tried (N=630's
    # PRE transpose, gcd(105,6)=3's own divisors 1 and 3). A stage WITH a
    # tail is not confirmed unsafe (tile=2x2/5x5 also had one and stayed
    # clean) -- only a weak risk signal, not a certainty, which is why
    # transpose_tail_risk_penalty (below) stays small relative to
    # radix_risk_penalty.
    transpose_tail_tile_count: int
    # `None` (the default, and the only value estimate_metrics itself ever
    # produces) means "not probed" -- unlike every other field above, this
    # one cannot be computed from the plan alone (see planning/spill_probe.
    # py's own module docstring: it needs a real Mojo -> llc -> M2NDP-
    # Detour build+run, tens of seconds to minutes, not something
    # estimate_metrics can afford to do for every candidate). A caller who
    # wants this signal calls spill_probe.probe_spill_free explicitly and
    # folds the result in via spill_probe.apply_spill_probe -- estimate_cost
    # then only applies spill_penalty once this is actually `True`/`False`,
    # never for the untouched `None` default, so every existing caller's
    # ranking is bit-for-bit unchanged unless it opts in. See CostWeights.
    # spill_penalty's own comment and [[fft-spill-hard-filter]] for why
    # this soft weighting is a diagnostic signal only, never the actual
    # mechanism that keeps a confirmed-spilling candidate out of a final
    # pick -- planning.spill_probe.probe_and_rerank_candidates' own hard
    # exclusion is.
    spill_free: bool | None = None
    # `None` (the default) means "not probed," the same discipline as
    # `spill_free` above and for the same reason -- a real measured cycle
    # count (planning.spill_probe.SpillProbeResult.ndp_cycles, the
    # simulator's own Gantt-log total, not an estimate) only exists once a
    # caller has actually built+run this plan. Folded in by spill_probe.
    # apply_spill_probe alongside spill_free, from the same probe -- no
    # separate probe call needed. Used by planning.spill_probe.
    # probe_and_rerank_candidates' own `rank_by_cycles` to pick the
    # fastest *measured* candidate among several already-confirmed-
    # spill-free ones, rather than trusting estimated_cost's static
    # ranking to have picked the fastest one first (that ranking is a
    # cheap pre-filter for which candidates are worth a real probe at
    # all, not a claim that its own order matches real hardware speed --
    # see this project's own N=16384 tile=(4,4) vs (2,2) mismatch).
    ndp_cycles: int | None = None
    estimated_cost: float = 0.0      # filled in by estimate_cost, 0.0 until then


@dataclass(frozen=True)
class CostWeights:
    """One documented per-unit weight per PlanMetrics term -- not a fitted
    model (see this module's own docstring). Tune here; candidate
    generation never needs to change when these do. Ranking matters most
    *within* one generate_candidates(n) call, where most of these terms
    move independently of each other (e.g. a worker-count sweep holds
    estimated_dram_bytes essentially constant while worst_worker_utilization
    varies) -- see each field's own comment for why its magnitude was
    picked relative to the others.
    """

    memory_traffic: float = 1.0
    # Each transpose kernel is a full extra launch + DRAM round trip beyond
    # what memory_traffic already counts per stage -- a per-kernel
    # fixed-overhead term on top of the raw byte count.
    transpose_passes: float = 50.0
    stage_work: float = 0.1
    # A spill on real hardware is a correctness failure (the kernel panics
    # or silently zeroes its output), not a slowdown -- this must dominate
    # every other term whenever radix_risk_score is nonzero.
    idle_worker_penalty: float = 200.0
    radix_risk_penalty: float = 5000.0
    # NOT the authoritative spill policy -- see [[fft-spill-hard-filter]]
    # (a confirmed real spill must disqualify a candidate outright, never
    # just adjust a score) -- that's enforced by planning.spill_probe.
    # probe_and_rerank_candidates hard-excluding any candidate whose
    # PlanMetrics.spill_free a real probe confirmed False, before this
    # weight ever gets a chance to matter for it. This term only fires
    # once spill_free has actually been probed (see that field's own
    # docstring -- estimate_cost skips it entirely for the untouched
    # `None` default) and exists purely as a *diagnostic* signal for a
    # caller reading estimated_cost directly (format_plan_summary, ad hoc
    # debugging) without going through the hard-exclusion path -- e.g. so
    # a spilling candidate's own cost still visibly reflects that DRAM-
    # spill traffic is real cost on top of whatever else is wrong with it,
    # rather than looking identical to a spill-free sibling. Never rely on
    # this weight alone to keep a confirmed-spilling candidate out of a
    # final recommendation; only probe_and_rerank_candidates' own hard
    # exclusion does that reliably.
    spill_penalty: float = 2000.0
    # A weak, pre-probe nudge toward tile choices that structurally can't
    # hit the tail-branch liability transpose_tail_tile_count flags (see
    # that field's own comment) -- deliberately much smaller than
    # radix_risk_penalty/spill_penalty, since a tail tile is only a risk
    # signal, not a confirmed one (tile=2x2/5x5 in the real N=630 case
    # this is based on both had a tail and stayed clean). Sized just large
    # enough to break a near-tie in favor of the no-tail option (e.g. two
    # tile sizes with otherwise-similar transpose_tile_count/DRAM cost),
    # not to override a genuinely cheaper has-tail candidate outright --
    # a caller who wants the stronger guarantee should reach for
    # planning.spill_probe's real probe instead, same as radix_risk_score
    # vs. a confirmed spill_free=False.
    transpose_tail_risk_penalty: float = 20.0
    recursion_depth_penalty: float = 100.0
    # Real M2NDP runs (this project's own benchmark_fft_candidates.sh
    # sweeps at N=1024/960/630) show MORE, SMALLER transpose tiles usually
    # finishing in *fewer* ndp cycles, not more -- e.g. N=1024: 192 tiles
    # -> 2111 cycles vs. 48 tiles -> 7421-7871 cycles; N=630: 210 tiles ->
    # 1867 cycles (the fastest of 6 real candidates) vs. the 54-tile
    # baseline's 4426. The opposite of the usual "more kernel launches =
    # more overhead" assumption, plausibly because a smaller tile fits
    # this target's own SIMD/register width more cleanly -- but NOT
    # cleanly monotonic (N=630's own 46-tile candidate ran *slower*, 5561
    # cycles, than its 54-tile baseline), so this weight is deliberately
    # small: before it existed, every split/tile candidate for one N was
    # an exact estimated_cost tie (total_transpose_tiles was computed in
    # PlanMetrics but never read here), so ranking among them fell back to
    # generation-order luck. This only needs to break that exact tie in
    # the right direction, not carry serious absolute weight -- treat any
    # single ranking decision it flips as a hint to verify with
    # benchmark_fft_candidates.sh, not a settled answer.
    transpose_tile_count: float = -2.0
    # Counteracts transpose_tile_count once a candidate's own busiest
    # transpose stage passes _TILE_PARALLELISM_SATURATION_UTHREADS -- see
    # that constant's own comment for the 4 real data points this is based
    # on. Sized so that N=1024/batch=4's tile=1x1 candidate (4096 uthreads,
    # 2048 over threshold) actually ranks behind its own tile=2x2 sibling
    # (real cycles: 1786 vs. 1560) -- picked as the smallest multiple of 10
    # that does, not a fitted rate; recheck this weight if more benchmark_
    # fft_candidates.sh data at other N/batch combinations disagrees.
    tile_oversaturation_penalty: float = 10.0


DEFAULT_COST_WEIGHTS = CostWeights()


def _tree_depth(node: FFTNode) -> int:
    if isinstance(node, FFTLeafPlan):
        return 0
    assert isinstance(node, FFTRecursiveNodePlan)
    return 1 + _tree_depth(node.far_child)


def _leaf_radix_risk(codegen_plan: FFTCodegenPlan) -> float:
    risk = 0.0
    prev_radix: int | None = None
    n_stages = len(codegen_plan.stages)
    for stage in codegen_plan.stages:
        is_middle = 0 < stage.stage_id < n_stages - 1
        if stage.radix in _ALWAYS_RISKY_RADICES:
            risk += 1.0
        elif prev_radix is not None and (prev_radix, stage.radix) in _RISKY_RADIX_PAIRS:
            risk += 1.0
        elif is_middle and stage.radix in _RISKY_AS_MIDDLE_RADICES:
            risk += 1.0
        prev_radix = stage.radix
    return risk


def _leaf_worker_utilization(codegen_plan: FFTCodegenPlan) -> float | None:
    """`None` if this leaf isn't cooperative (nothing to measure); else the
    worst (lowest) fraction of `workers_per_fft` actually busy across this
    leaf's own stages -- see FFTStagePlan.worker_batches' own docstring:
    `_partition_batches` always returns exactly `workers_per_fft` buckets,
    some possibly empty on a thin stage."""
    if codegen_plan.cooperation is None:
        return None
    worst = 1.0
    for stage in codegen_plan.stages:
        if not stage.worker_batches:
            continue
        active = sum(1 for wb in stage.worker_batches if wb)
        worst = min(worst, active / len(stage.worker_batches))
    return worst


def estimate_metrics(plan: RecursiveFFTPlan, target: TargetProfile) -> PlanMetrics:
    stages = flatten_recursive_node(plan.root)

    leaf_kernel_count = 0
    transpose_kernel_count = 0
    total_leaf_stage_count = 0
    total_transpose_tiles = 0
    max_transpose_stage_uthreads = 0
    max_scratchpad_bytes = 0
    radix_risk_score = 0.0
    transpose_tail_tile_count = 0
    utilizations: list[float] = []

    for stage in stages:
        if isinstance(stage, PhysicalTransposePlan):
            transpose_kernel_count += 1
            total_transpose_tiles += stage.total_uthreads
            max_transpose_stage_uthreads = max(max_transpose_stage_uthreads, stage.total_uthreads)
            max_scratchpad_bytes = max(max_scratchpad_bytes, stage.scratchpad_elements * 4)
            if stage.rows % stage.tile_rows != 0 or stage.cols % stage.tile_cols != 0:
                transpose_tail_tile_count += 1
        else:
            leaf_kernel_count += 1
            total_leaf_stage_count += len(stage.stages)
            radix_risk_score += _leaf_radix_risk(stage)
            leaf_bytes = sum(buf.elements for buf in stage.scratchpad_buffers) * 4
            max_scratchpad_bytes = max(max_scratchpad_bytes, leaf_bytes)
            util = _leaf_worker_utilization(stage)
            if util is not None:
                utilizations.append(util)

    # One full real+imag read + write per stage in the chain, times how many
    # independent batch transforms actually move through it -- see the
    # module design writeup's own "DRAM full-array passes" accounting
    # (fft_plan_recursive's own docstrings describe every stage as touching
    # the whole N-element buffer once each way) and make_recursive_transpose
    # _plan's own `batch` docstring (every stage's total_uthreads already
    # scales with it, so the DRAM traffic estimate must too or every batch
    # sweep would under-count it identically regardless of batch).
    estimated_dram_bytes = len(stages) * plan.n * plan.batch * 2 * 2 * 4

    return PlanMetrics(
        recursion_depth=_tree_depth(plan.root),
        leaf_kernel_count=leaf_kernel_count,
        transpose_kernel_count=transpose_kernel_count,
        total_leaf_stage_count=total_leaf_stage_count,
        total_transpose_tiles=total_transpose_tiles,
        max_transpose_stage_uthreads=max_transpose_stage_uthreads,
        estimated_dram_bytes=estimated_dram_bytes,
        max_scratchpad_bytes=max_scratchpad_bytes,
        worst_worker_utilization=min(utilizations) if utilizations else 1.0,
        radix_risk_score=radix_risk_score,
        transpose_tail_tile_count=transpose_tail_tile_count,
    )


def estimate_cost(metrics: PlanMetrics, weights: CostWeights = DEFAULT_COST_WEIGHTS) -> float:
    oversaturation = max(
        0, metrics.max_transpose_stage_uthreads - _TILE_PARALLELISM_SATURATION_UTHREADS
    )
    # metrics.spill_free is None until a caller explicitly probes it (see
    # that field's own docstring) -- only apply the penalty once it's
    # actually been measured `False`, never for "not probed" or "probed
    # spill-free".
    spill_term = weights.spill_penalty if metrics.spill_free is False else 0.0
    return (
        weights.memory_traffic * metrics.estimated_dram_bytes
        + weights.transpose_passes * metrics.transpose_kernel_count
        + weights.stage_work * metrics.total_leaf_stage_count
        + weights.idle_worker_penalty * (1.0 - metrics.worst_worker_utilization)
        + weights.radix_risk_penalty * metrics.radix_risk_score
        + weights.recursion_depth_penalty * metrics.recursion_depth
        + weights.transpose_tile_count * metrics.total_transpose_tiles
        + weights.tile_oversaturation_penalty * oversaturation
        + weights.transpose_tail_risk_penalty * metrics.transpose_tail_tile_count
        + spill_term
    )
