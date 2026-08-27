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
# * 6, 9: fft_plan_core._prime_factors_supported's own comment documents
#   the concrete N=54=(6,9) spill this was originally based on (radix-9
#   stage, preceded by radix-6). Note this isn't universal, though: a
#   direct (4, 9) chain (N=36, forced via allowed_radix_composites) built
#   and ran clean, zero spill -- so whatever makes radix-9 risky here
#   depends on more than "radix 9 in a non-first position" alone (likely
#   something about what precedes it, e.g. radix-6's own address/register
#   shape specifically), which this per-stage-radix-only model has no way
#   to represent. Kept flagged anyway since the known-bad N=54 combination
#   is real and this heuristic can only be conservative, not precise.
# * 11, 13, 17: confirmed by direct probe (each forced into a (4, r)
#   chain -- N=44, N=52, N=68 respectively, r as the second/scratchpad-
#   reading stage): all three spill *and* silently produce an all-zero
#   (wrong, not just slow) result. Previously unflagged here entirely
#   (radix_risk_score reported 0.0 for any N landing one of these in a
#   non-first stage) -- a real correctness gap, not just a missed
#   optimization.
#
# `10` fails outright at *any* stage position (N=160/320, and reconfirmed
# by the same probe as (4, 10) = N=40 -- spills and mismatches even
# directly after a radix-4 first stage, unlike 6/9's apparently
# context-dependent failure) -- see the same fft_plan_core comment.
_NON_FIRST_STAGE_RISKY_RADICES = frozenset({6, 9, 11, 13, 17})
_ALWAYS_RISKY_RADICES = frozenset({10})


@dataclass(frozen=True)
class PlanMetrics:
    recursion_depth: int
    leaf_kernel_count: int
    transpose_kernel_count: int
    total_leaf_stage_count: int
    total_transpose_tiles: int
    estimated_dram_bytes: int
    max_scratchpad_bytes: int
    worst_worker_utilization: float  # 1.0 = no cooperative leaf ever idles a worker
    radix_risk_score: float          # 0.0 = every leaf's radix sequence is the confirmed-safe kind
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
    recursion_depth_penalty: float = 100.0


DEFAULT_COST_WEIGHTS = CostWeights()


def _tree_depth(node: FFTNode) -> int:
    if isinstance(node, FFTLeafPlan):
        return 0
    assert isinstance(node, FFTRecursiveNodePlan)
    return 1 + _tree_depth(node.far_child)


def _leaf_radix_risk(codegen_plan: FFTCodegenPlan) -> float:
    risk = 0.0
    for stage in codegen_plan.stages:
        if stage.radix in _ALWAYS_RISKY_RADICES:
            risk += 1.0
        elif stage.stage_id > 0 and stage.radix in _NON_FIRST_STAGE_RISKY_RADICES:
            risk += 1.0
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
    max_scratchpad_bytes = 0
    radix_risk_score = 0.0
    utilizations: list[float] = []

    for stage in stages:
        if isinstance(stage, PhysicalTransposePlan):
            transpose_kernel_count += 1
            total_transpose_tiles += stage.total_uthreads
            max_scratchpad_bytes = max(max_scratchpad_bytes, stage.scratchpad_elements * 4)
        else:
            leaf_kernel_count += 1
            total_leaf_stage_count += len(stage.stages)
            radix_risk_score += _leaf_radix_risk(stage)
            leaf_bytes = sum(buf.elements for buf in stage.scratchpad_buffers) * 4
            max_scratchpad_bytes = max(max_scratchpad_bytes, leaf_bytes)
            util = _leaf_worker_utilization(stage)
            if util is not None:
                utilizations.append(util)

    # One full real+imag read + write per stage in the chain -- see the
    # module design writeup's own "DRAM full-array passes" accounting
    # (fft_plan_recursive's own docstrings describe every stage as
    # touching the whole N-element buffer once each way).
    estimated_dram_bytes = len(stages) * plan.n * 2 * 2 * 4

    return PlanMetrics(
        recursion_depth=_tree_depth(plan.root),
        leaf_kernel_count=leaf_kernel_count,
        transpose_kernel_count=transpose_kernel_count,
        total_leaf_stage_count=total_leaf_stage_count,
        total_transpose_tiles=total_transpose_tiles,
        estimated_dram_bytes=estimated_dram_bytes,
        max_scratchpad_bytes=max_scratchpad_bytes,
        worst_worker_utilization=min(utilizations) if utilizations else 1.0,
        radix_risk_score=radix_risk_score,
    )


def estimate_cost(metrics: PlanMetrics, weights: CostWeights = DEFAULT_COST_WEIGHTS) -> float:
    return (
        weights.memory_traffic * metrics.estimated_dram_bytes
        + weights.transpose_passes * metrics.transpose_kernel_count
        + weights.stage_work * metrics.total_leaf_stage_count
        + weights.idle_worker_penalty * (1.0 - metrics.worst_worker_utilization)
        + weights.radix_risk_penalty * metrics.radix_risk_score
        + weights.recursion_depth_penalty * metrics.recursion_depth
    )
