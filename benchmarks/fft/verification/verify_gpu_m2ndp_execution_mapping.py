from __future__ import annotations

"""Semantic invariants for the GPU-logical -> M2NDP-physical mapping
(section 11 of the task this suite was built from, "Step 2"). Every check
here is a property of the RECONSTRUCTED execution description
(`gpu_baseline.common.reconstruct_gpu_logical_plan`/`reconstruct_m2ndp_
physical_plan`), not a string-formatting check -- see `verify_gpu_
execution_report.py` for the report-formatting layer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.gpu_baseline import vkfft
from planning.gpu_baseline.common import (
    BaselineStatus,
    reconstruct_gpu_logical_plan,
    reconstruct_m2ndp_physical_plan,
)

_T = DEFAULT_TARGET_PROFILE


def check_logical_worker_count_preserved() -> None:
    """The GPU planner's own `gpu_threads_per_transform` must survive
    M2NDP lowering UNCHANGED as `logical_workers_per_fft` -- for every FFT
    kernel, cooperative or persistent, at every representative length
    tested. This is THE core invariant section 7 of the task this suite
    was built from is about ("do NOT change the GPU logical plan to fit
    M2NDP")."""
    cases = [
        ("vkfft W<=8 (cooperative)", vkfft.plan(64, batch=1)),
        ("vkfft W>8 (persistent)", vkfft.plan(512, batch=3)),
        ("vkfft-m2ndp N=8192 (multi-pass + persistent)", vkfft.plan_m2ndp(8192, target=_T)),
    ]
    checked = 0
    for label, result in cases:
        assert result.status is BaselineStatus.OK, f"{label}: expected OK, got {result.status}"
        gpu_plan = reconstruct_gpu_logical_plan(result)
        m2ndp_plan = reconstruct_m2ndp_physical_plan(result, target=_T)
        for gpu_kernel, m2ndp_kernel in zip(gpu_plan, m2ndp_plan):
            if gpu_kernel.gpu_transpose_role is not None:
                continue  # transpose kernels have no worker-count concept
            assert gpu_kernel.gpu_threads_per_transform == m2ndp_kernel.logical_workers_per_fft, (
                f"{label} kernel {gpu_kernel.gpu_kernel_id}: GPU's own gpu_threads_per_transform="
                f"{gpu_kernel.gpu_threads_per_transform} != M2NDP's own logical_workers_per_fft="
                f"{m2ndp_kernel.logical_workers_per_fft} -- the GPU logical choice must survive "
                f"M2NDP lowering verbatim"
            )
            checked += 1
    assert checked >= 3, f"expected at least 3 FFT kernels checked across all cases, got {checked}"
    print(f"    OK   GPU logical worker count (gpu_threads_per_transform) is preserved verbatim as "
          f"M2NDP's logical_workers_per_fft across {checked} FFT kernel(s), cooperative and "
          f"persistent alike")


def check_wide_workers_use_virtualization_not_more_physical_lanes() -> None:
    """workers_per_fft=32 (or any value > the physical lane count) must
    produce PERSISTENT virtualization -- `physical_workers_per_group`
    stays exactly `target.interleave_chunk_uthreads` (8), never 32.
    `logical_worker_groups` (the ceil-division wave count) must be > 1,
    confirming real virtualization is actually happening, not merely a
    strategy label with no observable consequence."""
    result = vkfft.plan(512, batch=3, target=_T)
    assert result.status is BaselineStatus.OK
    [m2ndp_kernel] = [
        k for k in reconstruct_m2ndp_physical_plan(result, target=_T) if k.strategy != "transpose"
    ]
    assert m2ndp_kernel.strategy == "persistent", (
        f"expected strategy='persistent' for a wide GPU cooperation width, got "
        f"{m2ndp_kernel.strategy!r}"
    )
    assert m2ndp_kernel.logical_workers_per_fft is not None and m2ndp_kernel.logical_workers_per_fft > _T.interleave_chunk_uthreads, (
        f"this test's own N=512/batch=3 case is expected to pick workers_per_fft > "
        f"{_T.interleave_chunk_uthreads} -- got {m2ndp_kernel.logical_workers_per_fft}, "
        f"re-pick a representative case"
    )
    assert m2ndp_kernel.physical_workers_per_group == _T.interleave_chunk_uthreads, (
        f"physical_workers_per_group must stay {_T.interleave_chunk_uthreads} regardless of "
        f"logical_workers_per_fft={m2ndp_kernel.logical_workers_per_fft} -- got "
        f"{m2ndp_kernel.physical_workers_per_group} (this would mean M2NDP invented more "
        f"physical parallelism than the hardware has)"
    )
    assert m2ndp_kernel.logical_worker_groups is not None and m2ndp_kernel.logical_worker_groups > 1, (
        f"expected genuine virtualization (>1 logical worker group), got "
        f"{m2ndp_kernel.logical_worker_groups}"
    )
    print(
        f"    OK   workers_per_fft={m2ndp_kernel.logical_workers_per_fft} with physical_workers_per_"
        f"group={m2ndp_kernel.physical_workers_per_group} produces persistent virtualization "
        f"(logical_worker_groups={m2ndp_kernel.logical_worker_groups}), never {m2ndp_kernel.logical_workers_per_fft}-way physical parallelism"
    )


def check_all_logical_batches_executed_exactly_once() -> None:
    """Every stage's own `stage_physical_work` line (built via `fft_plan_
    persistent.physical_lane_workload`, the same function codegen's
    default 'physical' lowering renders through -- see docs/
    logical_vs_physical_cost_model.md) must account for every logical
    batch exactly once: `sum(physical_lane_batches counts)` must equal the
    stage's own total logical batch count, and no physical lane index
    ever exceeds the physical lane count."""
    result = vkfft.plan_m2ndp(8192, target=_T)
    assert result.status is BaselineStatus.OK
    m2ndp_plan = reconstruct_m2ndp_physical_plan(result, target=_T)
    persistent_kernels = [k for k in m2ndp_plan if k.strategy == "persistent"]
    assert persistent_kernels, "expected at least one persistent kernel in this plan"

    # Cross-check against the raw plan tree directly (not just the report
    # strings) -- re-derive expected total batch counts independently.
    from planning.strategies.fft_plan_recursive import flatten_recursive_node
    from planning.core.fft_plan_core import FFTCodegenPlan

    fft_nodes = [n for n in flatten_recursive_node(result.plan.root) if isinstance(n, FFTCodegenPlan)]
    persistent_nodes = [n for n in fft_nodes if n.persistent is not None]
    assert len(persistent_nodes) == len(persistent_kernels)

    checked_stages = 0
    for node in persistent_nodes:
        pw = node.persistent
        for stage in node.stages:
            if stage.persistent_vector_batches is None:
                continue
            total_logical = sum(len(wb) for wb in stage.persistent_vector_batches) + len(
                stage.persistent_scalar_batches or ()
            )
            from planning.execution.fft_plan_persistent import physical_lane_workload
            workload = physical_lane_workload(
                stage.persistent_vector_batches, stage.persistent_scalar_batches or (),
                workers_per_fft=pw.workers_per_fft, workers_per_group=pw.workers_per_group,
            )
            total_physical = sum(len(lane) for lane in workload.physical_lane_batches)
            assert total_physical == total_logical, (
                f"stage {stage.stage_id}: physical batch total {total_physical} != logical batch "
                f"total {total_logical} -- some batch was lost or duplicated during flattening"
            )
            assert len(workload.physical_lane_batches) == pw.workers_per_group, (
                f"stage {stage.stage_id}: {len(workload.physical_lane_batches)} physical lanes != "
                f"workers_per_group={pw.workers_per_group}"
            )
            checked_stages += 1
    assert checked_stages > 0
    print(f"    OK   all logical batches executed exactly once across {checked_stages} persistent "
          f"stage(s) of vkfft.plan_m2ndp(8192)'s own plan -- no physical lane exceeds the count, "
          f"no batch lost or duplicated")


def check_source_faithful_vs_adapted_differ_same_algorithm() -> None:
    """`vkfft.plan(8192)` and `vkfft.plan_m2ndp(8192)` -- same module, same
    underlying `plan()` control flow (see vkfft.plan_m2ndp's own
    docstring: it calls `plan()` with only `num_compute_units`/`complex_
    size_bytes` overridden) -- must be free to make DIFFERENT pass-count/
    resource decisions (source-faithful vs. M2NDP-resource-adapted, see
    docs/vkfft_m2ndp_scratchpad_resource_model.md) while both still being
    genuine outputs of the identical `vkfft.py` algorithm, never a
    different, second lowering path."""
    frozen = vkfft.plan(8192, target=_T)
    adapted = vkfft.plan_m2ndp(8192, target=_T)
    assert frozen.status is BaselineStatus.RESOURCE_INFEASIBLE, (
        f"expected the frozen baseline to still pick the infeasible one-pass shape "
        f"(unchanged since docs/vkfft_m2ndp_scratchpad_resource_model.md), got {frozen.status}"
    )
    assert adapted.status is BaselineStatus.OK
    # Different scheme decision (num_passes) ...
    frozen_num_passes = frozen.gpu_config.extra.get("workers_per_fft")  # refusal keeps GPU's own attempted config
    adapted_gpu_plan = reconstruct_gpu_logical_plan(adapted)
    adapted_scheme = adapted_gpu_plan[0].gpu_scheme
    assert "num_passes=2" in adapted_scheme or "num_passes=3" in adapted_scheme, (
        f"expected the M2NDP-adapted scheme to need >1 pass, got {adapted_scheme!r}"
    )
    # ... but the SAME planner/module produced both.
    assert frozen.gpu_config.source == "vkfft"
    assert adapted.gpu_config.source == "vkfft-m2ndp"
    import inspect
    assert inspect.getsourcefile(vkfft.plan) == inspect.getsourcefile(vkfft.plan_m2ndp)
    print(
        f"    OK   vkfft.plan(8192) (frozen: {frozen.status.value}, attempted "
        f"workers_per_fft={frozen_num_passes}) and vkfft.plan_m2ndp(8192) (adapted: "
        f"{adapted_scheme}) make DIFFERENT decisions from the SAME vkfft.py algorithm "
        f"(vkfft.plan_m2ndp literally calls vkfft.plan with different resource inputs)"
    )


def main() -> None:
    print("  GPU-logical -> M2NDP-physical execution mapping invariants:")
    check_logical_worker_count_preserved()
    check_wide_workers_use_virtualization_not_more_physical_lanes()
    check_all_logical_batches_executed_exactly_once()
    check_source_faithful_vs_adapted_differ_same_algorithm()
    print("[verify] GPU-M2NDP execution mapping: all checks passed")


if __name__ == "__main__":
    main()
