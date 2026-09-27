from __future__ import annotations

"""Stage-local ACTIVE PHYSICAL WORKER diagnostics (P2.1 -- see docs/
priority2_execution_strategies.md).

WHAT THIS IS: a caller-facing reporting layer, built entirely from data
`_partition_batches`/`_partition_vector_scalar` (planning) and `fft_plan_
persistent.physical_lane_workload` (the same shared function `codegen.
fft_persistent_codegen`'s default lowering and `fft_cost_model.compute_
stage_metrics` both already use, see docs/logical_vs_physical_cost_
model.md) already compute -- never a new formula, never a guess. It
answers, per stage, "how many of the physical lanes that EXIST actually
do useful work this stage, and how much."

WHAT THIS IS NOT: a codegen change. Code inspection (not assumption)
confirmed BOTH `codegen.fft_cooperative_codegen._emit_cooperative_stage`
(`if not batches: continue`) and `codegen.fft_persistent_codegen.
_emit_worker_dispatch`/`_emit_physical_lane_dispatch` (same idiom) ALREADY
skip emitting a dispatch branch for an empty-bucket lane -- an idle
physical lane touches zero FFT arithmetic and zero scratchpad addresses
for that stage, before this diagnostics module ever existed. There is
therefore no "baseline vs. stage-active-worker" CODEGEN toggle to build
here (nothing was inefficient at the codegen level to begin with) -- this
module's own value is making that already-correct behavior VISIBLE and
MEASURABLE per stage, not changing it.

Does NOT change `workers_per_fft` (logical) or physical lane count --
per this task's own instruction, this module only reports on an already-
built plan's own already-decided batch partition.
"""

from dataclasses import dataclass

from planning.core.fft_plan_core import FFTCodegenPlan
from planning.execution.fft_plan_persistent import physical_lane_workload
from planning.strategies.fft_plan_recursive import PhysicalTransposePlan, RecursiveFFTPlan, flatten_recursive_node


@dataclass(frozen=True)
class StageActiveWorkerReport:
    """One stage's own active/idle physical-worker shape -- see this
    module's own docstring for provenance. `strategy` is one of
    `"non_cooperative"` | `"cooperative"` | `"persistent"`, read straight
    off which of `FFTCodegenPlan.cooperation`/`.persistent` is set (never
    guessed).

    `logical_workers_per_fft`: the GPU-planner-facing cooperation width --
    `None` for a non-cooperative stage (no such concept applies).
    `physical_workers_per_group`: the REAL physical lane count this
    stage's own batches were flattened onto -- always `<=
    target.interleave_chunk_uthreads` (8) for cooperative (no
    virtualization possible there by construction -- see
    `fft_plan_cooperative.worker_candidates_per_fft`'s own docstring),
    always exactly `target.interleave_chunk_uthreads` for persistent, and
    `1` for non-cooperative (one implicit worker owns the whole stage).

    `physical_batches_per_lane`: one entry per PHYSICAL lane, in lane-
    index order -- `len()` of each lane's own already-emitted batch
    bucket. `active_physical_workers`/`idle_physical_workers` partition
    `physical_workers_per_group` by whether that count is `> 0`.
    `imbalance_ratio = max_batches_per_lane / avg_batches_per_active_lane`
    (`1.0` when there is no active lane at all, or when `max == 0` --
    never a division by zero).
    """

    leaf_index: int
    stage_id: int
    radix: int
    strategy: str
    logical_workers_per_fft: int | None
    physical_workers_per_group: int
    active_physical_workers: int
    idle_physical_workers: int
    physical_batches_per_lane: tuple[int, ...]
    max_batches_per_lane: int
    avg_batches_per_active_lane: float
    imbalance_ratio: float


def _report_for_cooperative_stage(
    *, leaf_index: int, stage, logical_workers_per_fft: int,
) -> StageActiveWorkerReport:
    assert stage.worker_batches is not None
    counts = tuple(len(wb) for wb in stage.worker_batches)
    return _build_report(
        leaf_index=leaf_index, stage_id=stage.stage_id, radix=stage.radix,
        strategy="cooperative", logical_workers_per_fft=logical_workers_per_fft,
        physical_workers_per_group=len(counts), counts=counts,
    )


def _report_for_persistent_stage(
    *, leaf_index: int, stage, workers_per_fft: int, workers_per_group: int,
) -> StageActiveWorkerReport:
    assert stage.persistent_vector_batches is not None
    workload = physical_lane_workload(
        stage.persistent_vector_batches, stage.persistent_scalar_batches or (),
        workers_per_fft=workers_per_fft, workers_per_group=workers_per_group,
    )
    counts = tuple(len(lane) for lane in workload.physical_lane_batches)
    return _build_report(
        leaf_index=leaf_index, stage_id=stage.stage_id, radix=stage.radix,
        strategy="persistent", logical_workers_per_fft=workers_per_fft,
        physical_workers_per_group=workers_per_group, counts=counts,
    )


def _report_for_noncooperative_stage(*, leaf_index: int, stage) -> StageActiveWorkerReport:
    counts = (len(stage.batches),)
    return _build_report(
        leaf_index=leaf_index, stage_id=stage.stage_id, radix=stage.radix,
        strategy="non_cooperative", logical_workers_per_fft=None,
        physical_workers_per_group=1, counts=counts,
    )


def _build_report(
    *, leaf_index: int, stage_id: int, radix: int, strategy: str,
    logical_workers_per_fft: int | None, physical_workers_per_group: int,
    counts: tuple[int, ...],
) -> StageActiveWorkerReport:
    active = sum(1 for c in counts if c > 0)
    idle = physical_workers_per_group - active
    max_c = max(counts, default=0)
    avg_active = (sum(counts) / active) if active > 0 else 0.0
    imbalance = (max_c / avg_active) if (active > 0 and max_c > 0) else 1.0
    return StageActiveWorkerReport(
        leaf_index=leaf_index, stage_id=stage_id, radix=radix, strategy=strategy,
        logical_workers_per_fft=logical_workers_per_fft,
        physical_workers_per_group=physical_workers_per_group,
        active_physical_workers=active, idle_physical_workers=idle,
        physical_batches_per_lane=counts, max_batches_per_lane=max_c,
        avg_batches_per_active_lane=avg_active, imbalance_ratio=imbalance,
    )


def compute_stage_active_worker_reports(
    plan: RecursiveFFTPlan,
) -> list[StageActiveWorkerReport]:
    """Every FFT-leaf stage's own `StageActiveWorkerReport`, in execution
    order (`flatten_recursive_node`'s own PRE -> near -> MIDDLE -> far ->
    POST ordering; transpose stages are skipped -- no per-worker-
    cooperation concept applies to them, same scoping `fft_cost_model.
    compute_stage_metrics` already uses)."""
    reports: list[StageActiveWorkerReport] = []
    leaf_index = 0
    for node in flatten_recursive_node(plan.root):
        if isinstance(node, PhysicalTransposePlan):
            continue
        assert isinstance(node, FFTCodegenPlan)
        for stage in node.stages:
            if stage.persistent_vector_batches is not None:
                assert node.persistent is not None
                reports.append(
                    _report_for_persistent_stage(
                        leaf_index=leaf_index, stage=stage,
                        workers_per_fft=node.persistent.workers_per_fft,
                        workers_per_group=node.persistent.workers_per_group,
                    )
                )
            elif stage.worker_batches is not None:
                assert node.cooperation is not None
                reports.append(
                    _report_for_cooperative_stage(
                        leaf_index=leaf_index, stage=stage,
                        logical_workers_per_fft=node.cooperation.workers_per_fft,
                    )
                )
            else:
                reports.append(_report_for_noncooperative_stage(leaf_index=leaf_index, stage=stage))
        leaf_index += 1
    return reports


def format_stage_active_worker_report(report: StageActiveWorkerReport) -> str:
    return (
        f"leaf {report.leaf_index} stage {report.stage_id} (radix={report.radix}, "
        f"strategy={report.strategy}): logical_workers_per_fft={report.logical_workers_per_fft} "
        f"physical_workers_per_group={report.physical_workers_per_group} "
        f"active={report.active_physical_workers} idle={report.idle_physical_workers} "
        f"batches_per_lane={list(report.physical_batches_per_lane)} "
        f"max={report.max_batches_per_lane} avg_active={report.avg_batches_per_active_lane:.2f} "
        f"imbalance_ratio={report.imbalance_ratio:.2f}"
    )
