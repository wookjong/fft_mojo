from __future__ import annotations

"""Priority-2 controlled A/B experiment harness -- see docs/
priority2_execution_strategies.md's own "A/B experiment harness" section
and the task this script was built from ("PRIORITY-2 EXPERIMENT DESIGN":
Experiments 0-5, one mechanism at a time before any combination).

For each experiment, the GPU LOGICAL plan (length, radices, `workers_per_
fft`) is HELD FIXED -- only the M2NDP EXECUTION mechanism varies, per this
task's own "the GPU logical plan must remain fixed when evaluating these
changes" instruction. Two fixed shapes are used (see module docstring
below for why one shape cannot exercise all four mechanisms):

* `_PERSISTENT_SHAPE`: `length=64, radices=(4,4,4), workers_per_fft=16` --
  16 does not divide the physical lane count (8), so this leaf lowers via
  the persistent/worker-wave path, exercising P2.1 (stage diagnostics),
  P2.3 (copy mode), and (via a separate replica-count axis) P2.4 (tail
  hybrid).
* `_COOPERATIVE_SHAPE`: `length=64, radices=(4,4,4), workers_per_fft=4` --
  4 divides 8, so this leaf lowers via the ordinary cooperative path,
  the ONLY path P2.2 (batch partition) applies to at all (persistent
  leaves have no `partition_mode` concept -- see `CooperationPlan.
  partition_mode`'s own docstring).

MEASUREMENT POLICY (this task's own section): every experiment reports
`spill_free=None` and `ndp_cycles=None` -- UNMEASURED, since the real
Mojo/M2NDP-Detour toolchain is confirmed absent in this environment (see
docs/transpose_cost_model_audit.md's own toolchain-availability check).
NO timing number is fabricated anywhere in this script. What IS reported,
honestly, for every experiment: `estimated_cost` under BOTH `DEFAULT_
COST_WEIGHTS` (legacy, includes the unverified transpose-tile term -- N/A
here anyway, no transpose stage in a single leaf) and `PRIORITY2_COST_
WEIGHTS` (see fft_cost_model.py's own module-level constant), plus every
STRUCTURAL diagnostic this section's own mechanisms newly expose (active/
idle physical workers, batches-per-lane, imbalance ratio, partition mode,
copy mode, tail-round split). Per this task's own "if the existing cost
model cannot rank a new execution candidate reliably, say so explicitly"
instruction: `estimated_cost` is IDENTICAL across P2.1/P2.2/P2.3/P2.4
candidates and their own baseline in every experiment below, because none
of these new structural fields are wired into the cost-model ranking
formula (P2's own "do not broadly refit the cost model" scope limit) --
this harness does NOT claim the cost model can distinguish these
candidates; only real measured cycles could, and those are UNMEASURED.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.diagnostics.stage_active_workers import (
    compute_stage_active_worker_reports,
    format_stage_active_worker_report,
)
from planning.execution.fft_plan_cooperative import make_cooperative_leaf_plan
from planning.execution.fft_plan_persistent import (
    make_persistent_leaf_plan,
    make_persistent_tail_hybrid_plan,
    num_rounds,
    round_active_groups,
)
from planning.gpu_baseline.common import wrap_leaf_as_recursive_plan
from planning.search.fft_cost_model import DEFAULT_COST_WEIGHTS, PRIORITY2_COST_WEIGHTS, estimate_cost, estimate_metrics

_T = DEFAULT_TARGET_PROFILE
_LENGTH, _RADICES = 64, (4, 4, 4)
_PERSISTENT_WORKERS = 16
_COOPERATIVE_WORKERS = 4


def _report_candidate(label: str, plan) -> None:
    recursive = wrap_leaf_as_recursive_plan(length=_LENGTH, total_ffts=1, inverse=False, built_plan=plan)
    metrics = estimate_metrics(recursive, _T)
    legacy_cost = estimate_cost(metrics, DEFAULT_COST_WEIGHTS)
    p2_cost = estimate_cost(metrics, PRIORITY2_COST_WEIGHTS)
    print(f"  [{label}]")
    print(f"    estimated_cost (legacy weights)      = {legacy_cost:.2f}")
    print(f"    estimated_cost (PRIORITY2 weights)   = {p2_cost:.2f}  (identical: no new P2 "
          f"structural field is wired into the ranking formula)")
    print(f"    spill_free = None (UNMEASURED)   ndp_cycles = None (UNMEASURED)")
    for report in compute_stage_active_worker_reports(recursive):
        print(f"    {format_stage_active_worker_report(report)}")


def experiment0_baseline() -> None:
    print("Experiment 0: baseline M2NDP physical execution")
    plan = make_persistent_leaf_plan(
        _LENGTH, _RADICES, num_logical_blocks=1, target=_T, workers_per_fft=_PERSISTENT_WORKERS,
    )
    assert plan.persistent is not None
    assert plan.persistent.lowering_mode == "physical" and plan.persistent.copy_mode == "scalar"
    _report_candidate("baseline: persistent, lowering_mode=physical, copy_mode=scalar", plan)


def experiment1_stage_active_workers() -> None:
    print("\nExperiment 1: baseline + stage-specific active-worker diagnostics")
    print("  (P2.1 is a REPORTING mechanism only -- codegen already skips idle-lane work, see "
          "docs/priority2_execution_strategies.md's own P2.1 finding. Same plan as Experiment 0, "
          "diagnostics now shown explicitly.)")
    plan = make_persistent_leaf_plan(
        _LENGTH, _RADICES, num_logical_blocks=1, target=_T, workers_per_fft=_PERSISTENT_WORKERS,
    )
    _report_candidate("baseline + stage diagnostics (same plan, no codegen change)", plan)


def experiment2_batch_partition() -> None:
    print("\nExperiment 2: baseline + alternative cooperative batch partition")
    print(f"  (uses _COOPERATIVE_SHAPE -- workers_per_fft={_COOPERATIVE_WORKERS} divides the "
          f"physical lane count, the cooperative path P2.2 applies to)")
    for mode in ("round_robin", "contiguous", "balanced_contiguous"):
        plan = make_cooperative_leaf_plan(
            _LENGTH, _RADICES, workers_per_fft=_COOPERATIVE_WORKERS, total_ffts=8, partition_mode=mode,
        )
        assert plan.cooperation is not None and plan.cooperation.partition_mode == mode
        _report_candidate(f"partition_mode={mode}", plan)


def experiment3_vectorized_copy() -> None:
    print("\nExperiment 3: baseline + improved persistent preload/writeback")
    for copy_mode in ("scalar", "vectorized_contiguous"):
        plan = make_persistent_leaf_plan(
            _LENGTH, _RADICES, num_logical_blocks=1, target=_T, workers_per_fft=_PERSISTENT_WORKERS,
            copy_mode=copy_mode,
        )
        assert plan.persistent is not None and plan.persistent.copy_mode == copy_mode
        _report_candidate(f"copy_mode={copy_mode}", plan)


def experiment4_tail_hybrid() -> None:
    print("\nExperiment 4: baseline + hybrid persistent tail")
    print("  num_logical_blocks=33 (target.num_ndp_units=32 -> full_blocks=32, tail_blocks=1)")
    for tail_strategy in ("all_persistent", "noncoop_tail", "cooperative_tail"):
        hybrid = make_persistent_tail_hybrid_plan(
            _LENGTH, _RADICES, num_logical_blocks=33, target=_T, tail_strategy=tail_strategy,
            workers_per_fft=_PERSISTENT_WORKERS,
        )
        print(f"  [tail_strategy={tail_strategy}]")
        print(f"    full_blocks={hybrid.full_blocks} tail_blocks={hybrid.tail_blocks}")
        if tail_strategy == "all_persistent":
            rounds = num_rounds(hybrid.full_blocks, _T.num_ndp_units)
            last_round_active = round_active_groups(rounds - 1, hybrid.full_blocks, _T.num_ndp_units)
            print(f"    persistent rounds={rounds}, active NDP groups in the LAST round="
                  f"{last_round_active} of {_T.num_ndp_units} (this is the under-utilization P2.4 targets)")
        else:
            print(f"    persistent portion: full rounds only (every round uses all "
                  f"{_T.num_ndp_units} NDP groups); tail_blocks={hybrid.tail_blocks} replicas run "
                  f"via a SEPARATE {tail_strategy} kernel instead of an under-utilized persistent round")
        print(f"    ndp_cycles = None (UNMEASURED)")


def experiment5_combined() -> None:
    print("\nExperiment 5: combined Priority-2 execution strategy (only after individual effects "
          "are understood)")
    hybrid = make_persistent_tail_hybrid_plan(
        _LENGTH, _RADICES, num_logical_blocks=33, target=_T, tail_strategy="noncoop_tail",
        workers_per_fft=_PERSISTENT_WORKERS, copy_mode="vectorized_contiguous",
    )
    assert hybrid.persistent_plan is not None and hybrid.persistent_plan.persistent.copy_mode == "vectorized_contiguous"
    print("  combined: persistent portion uses copy_mode=vectorized_contiguous (P2.3) + "
        "tail_strategy=noncoop_tail (P2.4)")
    _report_candidate("combined P2.3+P2.4 (persistent portion)", hybrid.persistent_plan)
    print("  ndp_cycles = None (UNMEASURED) -- combined-mechanism cycle comparison requires the "
          "real toolchain")


def main() -> None:
    experiment0_baseline()
    experiment1_stage_active_workers()
    experiment2_batch_partition()
    experiment3_vectorized_copy()
    experiment4_tail_hybrid()
    experiment5_combined()
    print("\n[priority2_ab_harness] all experiments built and structurally reported "
          "(cycle/spill columns UNMEASURED -- no real M2NDP-Detour toolchain in this environment)")


if __name__ == "__main__":
    main()
