from __future__ import annotations

"""Cost-model regression tests for the logical/physical correction in
`planning.search.fft_cost_model` (see docs/logical_vs_physical_cost_model.md
and `StageExecutionMetrics.physical_max_batches_per_lane`'s own docstring).

Core invariant under test: M2NDP's real physical concurrency ceiling for a
persistent leaf is `target.interleave_chunk_uthreads` (8) physical lanes,
regardless of how wide a GPU planner's own `workers_per_fft` choice is.
`estimate_cost` must not predict unbounded speedup past that ceiling --
concretely:

1. Same total workload, `workers_per_fft` a clean multiple of the physical
   lane count (8, 16, 24, 32, 64): `physical_total_worker_stage_batches`
   and `estimated_cost` are IDENTICAL across every one of those worker
   counts (a perfectly balanced 8-way split is a perfectly balanced
   8-way split, how many logical workers it came from is invisible to the
   real hardware).
2. The OLD, logical-only field (`total_worker_stage_batches`, kept
   unchanged for backward compatibility -- see that field's own
   docstring) DOES keep shrinking as `workers_per_fft` grows past 8 --
   demonstrated here as a contrast, not a requirement: this is exactly
   the pre-fix behavior this correction targets, kept visible so a reader
   of this test can see what `_execution_cost` no longer trusts.
3. `physical_total_worker_stage_batches` (and therefore `estimated_cost`)
   is NEVER lower, for any `workers_per_fft > 8`, than its own value at
   the balanced `workers_per_fft == 8` baseline -- i.e. adding more
   LOGICAL cooperation width past the real physical ceiling can only ever
   match or hurt (via lane imbalance), never beat, the perfectly-balanced
   8-way split.
4. A ragged `workers_per_fft` (one that leaves physical lanes unevenly
   loaded, e.g. 9 against 8 physical lanes) costs MORE than the balanced
   neighbor (8) -- `physical_lane_imbalance` (`max/avg` work per lane)
   going up moves `estimated_cost` up, not down.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.execution.fft_plan_persistent import make_persistent_leaf_plan
from planning.gpu_baseline.common import wrap_leaf_as_recursive_plan
from planning.search.fft_cost_model import estimate_cost, estimate_metrics

_LENGTH = 2048
_RADICES = (4, 4, 4, 4, 4, 2)


def _cost_and_metrics(workers_per_fft: int):
    plan = make_persistent_leaf_plan(
        _LENGTH, _RADICES, num_logical_blocks=1, workers_per_fft=workers_per_fft,
        target=DEFAULT_TARGET_PROFILE,
    )
    recursive_plan = wrap_leaf_as_recursive_plan(
        length=_LENGTH, total_ffts=1, inverse=False, built_plan=plan,
    )
    metrics = estimate_metrics(recursive_plan, DEFAULT_TARGET_PROFILE)
    return metrics, estimate_cost(metrics)


def check_no_unbounded_speedup_past_physical_ceiling() -> None:
    baseline_metrics, baseline_cost = _cost_and_metrics(8)
    assert baseline_metrics.physical_lanes == 8 if hasattr(baseline_metrics, "physical_lanes") else True

    # 1. Clean multiples of 8: identical physical totals and identical cost.
    clean_multiples = (8, 16, 24, 32, 64)
    physical_totals = {}
    costs = {}
    for w in clean_multiples:
        metrics, cost = _cost_and_metrics(w)
        physical_totals[w] = metrics.physical_total_worker_stage_batches
        costs[w] = cost
    first = physical_totals[clean_multiples[0]]
    for w in clean_multiples[1:]:
        assert physical_totals[w] == first, (
            f"W={w}: physical_total_worker_stage_batches={physical_totals[w]} != "
            f"W={clean_multiples[0]}'s {first} -- a clean multiple of the physical "
            f"lane count should flatten to the identical balanced 8-way split"
        )
        assert costs[w] == costs[clean_multiples[0]], (
            f"W={w}: estimated_cost={costs[w]} != W={clean_multiples[0]}'s "
            f"{costs[clean_multiples[0]]} for an identical physical workload"
        )
    print(
        f"    OK   W in {clean_multiples}: physical_total_worker_stage_batches and "
        f"estimated_cost are IDENTICAL across every clean multiple of 8 "
        f"(physical_total={first}, cost={costs[clean_multiples[0]]:.2f})"
    )

    # 2. Contrast: the OLD logical-only field keeps shrinking (this is
    # exactly the bug this correction targets -- shown here, not asserted
    # as desirable, so a reader can see what estimate_cost no longer uses).
    logical_totals = []
    for w in clean_multiples:
        metrics, _ = _cost_and_metrics(w)
        logical_totals.append(metrics.total_worker_stage_batches)
    assert logical_totals == sorted(logical_totals, reverse=True), (
        f"expected the OLD logical field to keep shrinking as W grows (demonstrating "
        f"the pre-fix bug pattern): {list(zip(clean_multiples, logical_totals))}"
    )
    assert logical_totals[-1] < logical_totals[0], (
        "expected the logical-only metric to differ substantially by W=64 -- if it "
        "doesn't, this test's own N/radix choice no longer demonstrates the bug "
        "pattern and should be revisited"
    )
    print(
        f"    OK   contrast: OLD logical total_worker_stage_batches DOES keep "
        f"shrinking with W ({dict(zip(clean_multiples, logical_totals))}) -- exactly "
        f"the pattern _execution_cost no longer trusts"
    )

    # 3. No ragged/wide W ever beats the balanced W=8 baseline.
    all_tested_w = (8, 9, 10, 12, 14, 16, 20, 24, 32, 36, 64)
    for w in all_tested_w:
        metrics, cost = _cost_and_metrics(w)
        assert metrics.physical_total_worker_stage_batches >= baseline_metrics.physical_total_worker_stage_batches, (
            f"W={w}: physical_total_worker_stage_batches="
            f"{metrics.physical_total_worker_stage_batches} < the balanced W=8 "
            f"baseline's {baseline_metrics.physical_total_worker_stage_batches} -- "
            f"no workers_per_fft should ever beat a perfectly balanced 8-way split"
        )
        assert cost >= baseline_cost - 1e-6, (
            f"W={w}: estimated_cost={cost} < balanced W=8 baseline's {baseline_cost} "
            f"-- the cost model must never predict a plan wider than the physical "
            f"ceiling as CHEAPER than the perfectly balanced ceiling-width plan"
        )
    print(
        f"    OK   no W in {all_tested_w} ever beats the balanced W=8 baseline "
        f"(cost={baseline_cost:.2f}) -- physical concurrency ceiling=8 respected"
    )


def check_imbalance_increases_cost() -> None:
    """W=9 against 8 physical lanes is maximally ragged for this shape (one
    lane owns 2 logical workers' worth of batches, the other 7 own 1) --
    must cost strictly MORE than the balanced W=8 neighbor."""
    metrics8, cost8 = _cost_and_metrics(8)
    metrics9, cost9 = _cost_and_metrics(9)
    assert metrics9.physical_total_worker_stage_batches > metrics8.physical_total_worker_stage_batches, (
        f"W=9 physical_total={metrics9.physical_total_worker_stage_batches} should exceed "
        f"W=8's {metrics8.physical_total_worker_stage_batches} (W=9 is ragged against 8 "
        f"physical lanes)"
    )
    assert cost9 > cost8, (
        f"W=9 cost={cost9} should exceed W=8's balanced cost={cost8} -- lane imbalance "
        f"must move estimated_cost UP, not down"
    )
    print(
        f"    OK   ragged W=9 (imbalanced across 8 physical lanes) costs more than "
        f"balanced W=8: {cost8:.2f} -> {cost9:.2f}"
    )


def main() -> None:
    print("  Physical-lane cost model: no unbounded speedup past the physical concurrency ceiling:")
    check_no_unbounded_speedup_past_physical_ceiling()
    check_imbalance_increases_cost()
    print("[verify] physical-lane cost model: all checks passed")


if __name__ == "__main__":
    main()
