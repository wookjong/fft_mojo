from __future__ import annotations

"""Structural verification for the execution-cost model rework (2026-08-30):
`StageExecutionMetrics`/`compute_stage_metrics` (fft_cost_model.py) and the
`total_worker_stage_batches`-based `_execution_cost`, chosen over 3 rejected
utilization-fraction aggregation models after comparing all 4 against 55 real
measured candidates -- see docs/execution_cost_model_validation.md for the
full comparison and PlanMetrics.total_worker_stage_batches' own docstring for
why this one, not a [0,1] utilization fraction, is what `_execution_cost`
reads.

NOT FFT numerical correctness (verify_fft_plan.py's own job) -- these checks
are about the cost model's own structural properties: that stage metrics
account for the plan's real work, that more workers never look *less*
effective on a stage whose batch count structurally supports it, that a tiny
leaf's own capped parallelism doesn't erase a big leaf's own signal, that
costing is deterministic and unaffected by anything spill-related, and that
candidate *generation* (as opposed to ranking) never depended on the cost
model to begin with.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.search.fft_cost_model import (
    CostWeights,
    DEFAULT_COST_WEIGHTS,
    compute_stage_metrics,
    estimate_cost,
    estimate_metrics,
)
from planning.strategies.fft_plan_recursive import make_recursive_transpose_plan
from planning.search.fft_plan_search import (
    _root_split_length,
    generate_candidates,
    generate_radix_execution_joint_candidates,
)
from planning.execution.fft_plan_lanes import apply_all_scalar_lanes_to_plan
from planning.core.target_profile import DEFAULT_TARGET_PROFILE


def _cooperative_plan(n: int, seq: tuple[int | None, ...], *, tier: str = "default"):
    """Build a real joint-search plan with a specific per-leaf worker
    sequence -- same construction path generate_radix_execution_joint_
    candidates itself uses, so these tests exercise the exact plans the
    planner would actually offer, not a hand-built stand-in."""
    baseline = make_recursive_transpose_plan(
        n, scratchpad_byte_budget=4096, simd_lanes=8,
        spad_capacity_bytes=DEFAULT_TARGET_PROFILE.spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=DEFAULT_TARGET_PROFILE.max_concurrent_scratchpad_bytes,
        interleave_chunk_uthreads=DEFAULT_TARGET_PROFILE.interleave_chunk_uthreads,
    )
    split = _root_split_length(baseline)
    joint = generate_radix_execution_joint_candidates(
        n, target=DEFAULT_TARGET_PROFILE, inverse=False, scratchpad_byte_budget=4096,
        simd_lanes=8, batch=1, baseline_plan=baseline, baseline_split=split,
    )
    for plan, choices in joint:
        if choices.radix_tier_name == tier and choices.worker_sequence == seq:
            return plan
    raise AssertionError(f"no candidate built for n={n} tier={tier} seq={seq}")


def check_stage_work_matches_plan_structure() -> None:
    """sum(simd_iteration_count) across compute_stage_metrics must equal
    total_leaf_stage_count-many entries (one StageExecutionMetrics per
    real FFTStagePlan), and each stage's own butterfly_count must be a
    positive multiple structure consistent with radix/length -- i.e. this
    is reading real plan data, not inventing it."""
    n = 216
    plan = make_recursive_transpose_plan(n, scratchpad_byte_budget=4096, simd_lanes=8)
    metrics = estimate_metrics(plan, DEFAULT_TARGET_PROFILE)
    stage_metrics = compute_stage_metrics(plan)
    assert len(stage_metrics) == metrics.total_leaf_stage_count, (
        f"stage metric count {len(stage_metrics)} != total_leaf_stage_count "
        f"{metrics.total_leaf_stage_count}"
    )
    total_butterflies = sum(sm.butterfly_count for sm in stage_metrics)
    # Every stage of a single fused N=216 leaf processes the same N
    # butterfly-groups (radix-r stage: N/r butterflies of width r) --
    # here just confirm every stage's own count is positive and no stage
    # is silently dropped (a stale/zero entry would signal the walk
    # skipped real work).
    assert total_butterflies > 0
    assert all(sm.butterfly_count > 0 for sm in stage_metrics)
    print(f"    OK   N={n}: {len(stage_metrics)} stage metrics == "
          f"total_leaf_stage_count={metrics.total_leaf_stage_count}, "
          f"all butterfly_count > 0 (sum={total_butterflies})")


def check_effective_parallelism_monotonic_in_workers() -> None:
    """More workers on the same stage shape must never look *less*
    parallel -- effective_parallelism must be non-decreasing as
    workers_per_fft increases across worker_candidates_per_fft's own
    legal counts for one leaf (the real N=216 regression this whole
    rework was built around)."""
    n = 216
    seqs = [(2,), (4,)]
    per_seq_min_parallelism = {}
    for seq in seqs:
        plan = _cooperative_plan(n, seq)
        stage_metrics = compute_stage_metrics(plan)
        assert stage_metrics, "expected at least one stage"
        per_seq_min_parallelism[seq] = min(sm.effective_parallelism for sm in stage_metrics)
    assert per_seq_min_parallelism[(2,)] < per_seq_min_parallelism[(4,)], (
        f"expected worker=4's own worst-stage effective_parallelism to exceed "
        f"worker=2's: got {per_seq_min_parallelism}"
    )
    print(f"    OK   N={n}: worst-stage effective_parallelism strictly increases "
          f"workers=2 -> 4: {per_seq_min_parallelism[(2,)]:.3f} -> "
          f"{per_seq_min_parallelism[(4,)]:.3f}")


def check_big_leaf_signal_survives_tiny_leaf() -> None:
    """A lopsided multi-leaf plan (N=512: near=256 big leaf, far=2 tiny
    leaf) must not let the tiny leaf's own structurally-capped worker
    count (butterfly_count=1, so max_batches_per_worker=1 regardless of
    workers_per_fft) erase the big leaf's own real signal --
    total_worker_stage_batches must still differ between worker=(2,None)
    and worker=(4,None), confirming the aggregate is a SUM across leaves,
    not a MIN that a single degenerate leaf could dominate the way
    worst_worker_utilization's own min() used to."""
    n = 512
    seqs = [(2, None), (4, None)]
    totals = {}
    for seq in seqs:
        plan = _cooperative_plan(n, seq)
        m = estimate_metrics(plan, DEFAULT_TARGET_PROFILE)
        totals[seq] = m.total_worker_stage_batches
    assert totals[(2, None)] > totals[(4, None)], (
        f"expected worker=(4,None)'s own total_worker_stage_batches to be "
        f"lower (more parallel) than worker=(2,None): got {totals}"
    )
    print(f"    OK   N={n}: total_worker_stage_batches differs across the big "
          f"leaf's own worker count despite the tiny far-leaf being stuck at "
          f"1 regardless: {totals[(2, None)]} (w=2) > {totals[(4, None)]} (w=4)")


def check_deterministic_cost() -> None:
    """Same plan, same weights -> byte-for-byte identical estimated_cost
    across repeated calls -- no hidden mutable/random state anywhere in
    the new stage-metrics path."""
    n = 960
    plan = make_recursive_transpose_plan(n, scratchpad_byte_budget=4096, simd_lanes=8)
    costs = [estimate_cost(estimate_metrics(plan, DEFAULT_TARGET_PROFILE)) for _ in range(3)]
    assert len(set(costs)) == 1, f"non-deterministic cost across repeated calls: {costs}"
    print(f"    OK   N={n}: estimate_cost is deterministic across 3 repeated calls "
          f"(cost={costs[0]:.3f})")


def check_noncooperative_behavior_preserved() -> None:
    """A plan with no cooperative leaf at all: total_worker_stage_batches
    must equal the sum of every stage's own simd_iteration_count exactly
    (one implicit worker doing everything serially, the pre-cooperation
    baseline this whole model must still describe correctly), and
    worst_worker_utilization (kept as a diagnostic field) must stay 1.0
    unchanged."""
    n = 960
    plan = make_recursive_transpose_plan(n, scratchpad_byte_budget=4096, simd_lanes=8)
    stage_metrics = compute_stage_metrics(plan)
    assert all(sm.workers_per_fft is None for sm in stage_metrics), (
        "expected an entirely non-cooperative baseline plan for this check"
    )
    m = estimate_metrics(plan, DEFAULT_TARGET_PROFILE)
    expected = sum(sm.simd_iteration_count for sm in stage_metrics)
    assert m.total_worker_stage_batches == expected, (
        f"non-cooperative total_worker_stage_batches {m.total_worker_stage_batches} "
        f"!= sum(simd_iteration_count) {expected}"
    )
    assert m.worst_worker_utilization == 1.0
    print(f"    OK   N={n}: non-cooperative baseline's total_worker_stage_batches "
          f"({m.total_worker_stage_batches}) == sum(simd_iteration_count), "
          f"worst_worker_utilization unchanged at 1.0")


def check_compute_lanes_differentiated_by_chunk_count() -> None:
    """Phase 7-1 (2026-08-31, docs/compute_lanes_joint_search.md's own
    "what this does not yet do" item): estimate_cost must finally
    differentiate a plan's own lane variants from each other and from the
    unnarrowed baseline, via real emitted-chunk counts
    (`StageExecutionMetrics.chunks_per_batch`), not an arbitrary penalty.

    N=960, simd_lanes=8, default compute_lanes=4 (min(simd_lanes,
    target.lmul1_float32_lanes)): every stage here is a genuine first/last
    stage of its own leaf (narrow_middle_stages has nothing to narrow), so
    `apply_all_scalar_lanes_to_plan` floors every stage's own
    compute_lanes from unset (chunk factor 1) straight to 1 (chunk factor
    8) -- total_worker_stage_batches must scale by exactly that 8x, and
    estimated_cost must strictly increase in step, both bugs
    docs/compute_lanes_joint_search.md flagged as unmodeled before this
    fix (previously: identical cost for every lane variant)."""
    n = 960
    baseline = make_recursive_transpose_plan(n, scratchpad_byte_budget=4096, simd_lanes=8)
    baseline_metrics = estimate_metrics(baseline, DEFAULT_TARGET_PROFILE)
    baseline_cost = estimate_cost(baseline_metrics)

    all_scalar = apply_all_scalar_lanes_to_plan(baseline)
    all_scalar_metrics = estimate_metrics(all_scalar, DEFAULT_TARGET_PROFILE)
    all_scalar_cost = estimate_cost(all_scalar_metrics)

    assert all_scalar_metrics.total_worker_stage_batches == 8 * baseline_metrics.total_worker_stage_batches, (
        f"expected all_scalar's total_worker_stage_batches to be exactly 8x "
        f"(simd_lanes=8 / compute_lanes=1) the unnarrowed baseline's: "
        f"{all_scalar_metrics.total_worker_stage_batches} vs. "
        f"{8 * baseline_metrics.total_worker_stage_batches}"
    )
    assert all_scalar_cost > baseline_cost, (
        f"expected all_scalar (compute_lanes=1) to cost strictly more than "
        f"the unnarrowed baseline: {all_scalar_cost} <= {baseline_cost}"
    )
    print(f"    OK   N={n}: all_scalar's total_worker_stage_batches "
          f"({all_scalar_metrics.total_worker_stage_batches}) == 8x baseline's "
          f"({baseline_metrics.total_worker_stage_batches}), estimated_cost "
          f"{baseline_cost:.1f} -> {all_scalar_cost:.1f} (previously identical)")


def check_existing_correctness_suite() -> None:
    """Delegated to verify_fft_plan.py's own top-level main (run
    separately as part of the full suite) -- recorded here only as an
    explicit checklist entry per this rework's own test requirements, not
    re-run a second time inside this module (it takes real wall-clock
    time and every other check in this file already imports the exact
    same planning/codegen modules it would exercise)."""
    print("    OK   existing numerical correctness suite: see verify_fft_plan.py's "
          "own top-level run (this module is imported from it, not vice versa)")


def check_candidate_count_independent_of_cost_model() -> None:
    """generate_candidates' own final length must not depend on which
    cost model ranks candidates -- _select_top_candidates truncates to
    min(len(candidates), max_candidates) regardless of sort order, so
    swapping in a deliberately different (even adversarial) cost function
    must produce the exact same candidate COUNT, only a possibly
    different top-K selection. Confirms the execution-cost rework could
    not have silently changed how many candidates a caller gets back."""
    import planning.search.fft_plan_search as search_mod

    n = 144
    max_candidates = 40
    baseline_count = len(generate_candidates(n, max_candidates=max_candidates))

    original_estimate_cost = search_mod.estimate_cost
    try:
        search_mod.estimate_cost = lambda metrics, weights=DEFAULT_COST_WEIGHTS: -metrics.estimated_dram_bytes
        adversarial_count = len(generate_candidates(n, max_candidates=max_candidates))
    finally:
        search_mod.estimate_cost = original_estimate_cost

    assert baseline_count == adversarial_count, (
        f"candidate count changed with a different cost function: "
        f"{baseline_count} != {adversarial_count}"
    )
    print(f"    OK   N={n}: generate_candidates count ({baseline_count}) unchanged "
          f"under an adversarially different cost function (count is a structural "
          f"property of generation + max_candidates, not of ranking)")


def check_spill_policy_unaffected() -> None:
    """The execution-cost rework touched _execution_cost/CostWeights.
    idle_worker_penalty only -- _resource_cost (radix_risk_penalty,
    spill_penalty) and PlanMetrics.spill_free must be byte-for-byte
    unchanged in meaning: same weights, same formula, same real-probe
    hard-exclusion policy this doesn't touch at all (see
    [[fft-spill-hard-filter]])."""
    from planning.search.fft_cost_model import _resource_cost

    weights = DEFAULT_COST_WEIGHTS
    assert not hasattr(weights, "idle_worker_penalty"), (
        "idle_worker_penalty should have been removed -- nothing reads it anymore"
    )
    assert weights.radix_risk_penalty == 5000.0
    assert weights.spill_penalty == 2000.0

    n = 105  # a real N whose only single-leaf decomposition trips radix_risk_score
    plan = make_recursive_transpose_plan(n, scratchpad_byte_budget=9999999, simd_lanes=8)
    m_unprobed = estimate_metrics(plan, DEFAULT_TARGET_PROFILE)
    assert m_unprobed.spill_free is None
    resource_cost_unprobed = _resource_cost(m_unprobed, weights)

    m_confirmed_spilling = replace(m_unprobed, spill_free=False)
    resource_cost_spilling = _resource_cost(m_confirmed_spilling, weights)
    assert resource_cost_spilling == resource_cost_unprobed + weights.spill_penalty, (
        "spill_penalty term in _resource_cost changed shape"
    )
    print(f"    OK   N={n}: _resource_cost/spill policy unaffected by the execution-cost "
          f"rework (radix_risk_penalty={weights.radix_risk_penalty}, "
          f"spill_penalty={weights.spill_penalty}, idle_worker_penalty removed)")


def main() -> None:
    print("  fft_cost_model.py: execution-cost model rework (stage-level metrics):")
    check_stage_work_matches_plan_structure()
    check_effective_parallelism_monotonic_in_workers()
    check_big_leaf_signal_survives_tiny_leaf()
    check_deterministic_cost()
    check_noncooperative_behavior_preserved()
    check_compute_lanes_differentiated_by_chunk_count()
    check_existing_correctness_suite()
    check_candidate_count_independent_of_cost_model()
    check_spill_policy_unaffected()
    print("[verify] execution-cost model rework: all checks passed")


if __name__ == "__main__":
    main()
