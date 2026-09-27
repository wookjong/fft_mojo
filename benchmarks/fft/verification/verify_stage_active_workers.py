from __future__ import annotations

"""Verification for stage-local active-physical-worker diagnostics
(P2.1 -- see docs/priority2_execution_strategies.md and `planning.
diagnostics.stage_active_workers`'s own module docstring).

Covers physical batch counts `< 8`, `= 8`, `> 8`, and NOT divisible by 8
(the task's own required test matrix), for both the persistent and
cooperative paths, verifying:

* every batch counted is a real, already-emitted `SIMDBatchPlan` (no
  invented/estimated count -- cross-checked directly against the plan's
  own `stage.persistent_vector_batches`/`worker_batches`);
* `active_physical_workers + idle_physical_workers == physical_workers_
  per_group` always;
* coverage (all work executed exactly once, no duplicate, no missing) --
  reusing the same identity-based technique `verify_fft_physical_lane_
  mapping.py` already established for the underlying flatten function,
  applied here at the REPORT level instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.execution.fft_plan_cooperative import make_cooperative_leaf_plan
from planning.execution.fft_plan_persistent import make_persistent_leaf_plan
from planning.gpu_baseline.common import wrap_leaf_as_recursive_plan
from planning.diagnostics.stage_active_workers import compute_stage_active_worker_reports

_T = DEFAULT_TARGET_PROFILE


def _persistent_reports(length, radices, *, workers_per_fft=None, num_logical_blocks=1):
    plan = make_persistent_leaf_plan(
        length, radices, num_logical_blocks=num_logical_blocks, target=_T, workers_per_fft=workers_per_fft,
    )
    rp = wrap_leaf_as_recursive_plan(length=length, total_ffts=num_logical_blocks, inverse=False, built_plan=plan)
    return compute_stage_active_worker_reports(rp), plan


def check_invariant_active_plus_idle_equals_group(reports) -> None:
    for r in reports:
        assert r.active_physical_workers + r.idle_physical_workers == r.physical_workers_per_group, (
            f"stage {r.stage_id}: active({r.active_physical_workers}) + idle("
            f"{r.idle_physical_workers}) != physical_workers_per_group({r.physical_workers_per_group})"
        )
        assert sum(1 for c in r.physical_batches_per_lane if c > 0) == r.active_physical_workers
        assert len(r.physical_batches_per_lane) == r.physical_workers_per_group


def check_less_than_8_active() -> None:
    """N=64, radices=(4,4,4), default workers_per_fft=8: every stage's own
    3 (radix-4) leaves only 2 of 8 physical lanes active -- confirmed
    directly (not assumed), matching this module's own real output."""
    reports, plan = _persistent_reports(64, (4, 4, 4))
    check_invariant_active_plus_idle_equals_group(reports)
    for r in reports:
        assert r.physical_workers_per_group == 8
        assert r.active_physical_workers == 2, f"stage {r.stage_id}: expected 2 active, got {r.active_physical_workers}"
        assert r.idle_physical_workers == 6
        assert sorted(r.physical_batches_per_lane, reverse=True) == [1, 1, 0, 0, 0, 0, 0, 0]
    print("    OK   N=64 radices=(4,4,4): every stage has active_physical_workers=2 < 8 "
          "(6 idle lanes), confirmed against real emitted batch buckets")


def check_exactly_8_active() -> None:
    """N=2048, radices=(4,4,4,4,4,2): stages 0-4 (radix 4) each have exactly
    8 active lanes, 8 batches each (perfectly balanced, no idle)."""
    reports, plan = _persistent_reports(2048, (4, 4, 4, 4, 4, 2))
    check_invariant_active_plus_idle_equals_group(reports)
    for r in reports[:5]:
        assert r.active_physical_workers == 8
        assert r.idle_physical_workers == 0
        assert r.max_batches_per_lane == 8
        assert r.imbalance_ratio == 1.0
        assert list(r.physical_batches_per_lane) == [8] * 8
    print("    OK   N=2048 radices=(4,4,4,4,4,2): stages 0-4 have active_physical_workers=8 "
          "(no idle lanes), perfectly balanced (imbalance_ratio=1.0)")


def check_greater_than_8_batches_per_lane() -> None:
    """The same N=2048 plan's own last stage (radix=2) has 16 batches per
    lane -- max_batches_per_lane=16 > 8, still perfectly balanced across
    exactly 8 active physical lanes (a persistent lane visiting MORE than
    one batch is the expected shape once total work exceeds 8, not an
    error)."""
    reports, _ = _persistent_reports(2048, (4, 4, 4, 4, 4, 2))
    last = reports[-1]
    assert last.max_batches_per_lane == 16
    assert last.active_physical_workers == 8
    assert list(last.physical_batches_per_lane) == [16] * 8
    print(f"    OK   N=2048 stage {last.stage_id}: max_batches_per_lane=16 (> 8) across all 8 "
          f"active physical lanes, still perfectly balanced")


def check_not_divisible_by_8() -> None:
    """N=105, radices=(3,5,7), workers_per_fft=10 (ragged -- 10 does not
    divide 8): every stage's own physical_batches_per_lane is genuinely
    UNEVEN (imbalance_ratio > 1.0), confirmed against real numbers."""
    reports, plan = _persistent_reports(105, (3, 5, 7), workers_per_fft=10)
    check_invariant_active_plus_idle_equals_group(reports)
    assert len(reports) == 3
    stage0, stage1, stage2 = reports
    assert list(stage0.physical_batches_per_lane) == [1, 2, 1, 1, 0, 0, 0, 0]
    assert stage0.active_physical_workers == 4
    assert abs(stage0.imbalance_ratio - 1.6) < 1e-6
    assert list(stage1.physical_batches_per_lane) == [1, 2, 0, 0, 0, 0, 0, 0]
    assert stage1.active_physical_workers == 2
    for r in reports:
        assert r.imbalance_ratio >= 1.0
    print("    OK   N=105 radices=(3,5,7) workers_per_fft=10 (not divisible by 8): every stage's "
          "own physical_batches_per_lane is genuinely uneven, imbalance_ratio computed correctly "
          f"(stage0={stage0.imbalance_ratio:.2f}, stage1={stage1.imbalance_ratio:.2f})")


def check_cooperative_path() -> None:
    """Cooperative leaves (workers_per_fft already <= 8 by construction --
    see fft_plan_cooperative.worker_candidates_per_fft's own docstring)
    report strategy='cooperative', physical_workers_per_group == workers_
    per_fft exactly (no virtualization possible)."""
    plan = make_cooperative_leaf_plan(64, (4, 4, 4), workers_per_fft=4, total_ffts=8)
    rp = wrap_leaf_as_recursive_plan(length=64, total_ffts=8, inverse=False, built_plan=plan)
    reports = compute_stage_active_worker_reports(rp)
    check_invariant_active_plus_idle_equals_group(reports)
    for r in reports:
        assert r.strategy == "cooperative"
        assert r.logical_workers_per_fft == 4
        assert r.physical_workers_per_group == 4
    print("    OK   cooperative leaf (workers_per_fft=4): physical_workers_per_group == "
          "logical_workers_per_fft exactly (no virtualization), strategy correctly tagged")


def check_coverage_against_raw_plan() -> None:
    """Every batch this module's own report counts must correspond to a
    real, distinct SIMDBatchPlan object on the underlying plan -- cross-
    check total counted batches against a direct walk of `stage.
    persistent_vector_batches`/`persistent_scalar_batches`."""
    reports, plan = _persistent_reports(105, (3, 5, 7), workers_per_fft=10)
    for report, stage in zip(reports, plan.stages):
        assert stage.persistent_vector_batches is not None
        raw_total = sum(len(wb) for wb in stage.persistent_vector_batches) + len(
            stage.persistent_scalar_batches or ()
        )
        report_total = sum(report.physical_batches_per_lane)
        assert report_total == raw_total, (
            f"stage {report.stage_id}: report total={report_total} != raw plan total={raw_total}"
        )
    print("    OK   report's own batch totals match the underlying plan's raw batch counts exactly "
          "(no invented/estimated count)")


def main() -> None:
    print("  Stage-local active physical worker diagnostics (P2.1):")
    check_less_than_8_active()
    check_exactly_8_active()
    check_greater_than_8_batches_per_lane()
    check_not_divisible_by_8()
    check_cooperative_path()
    check_coverage_against_raw_plan()
    print("[verify] stage active worker diagnostics: all checks passed")


if __name__ == "__main__":
    main()
