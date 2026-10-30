"""Distinct device functions registered for one task, independent of rounds.

Loop/tail decisions use codegen's existing lowering analysis; no Mojo parsing
or stage-count approximation. Callers must use the rendering loop settings.
"""
from codegen.lowering import LOOP_MIN_FULL_BATCHES, try_build_loop_stage
from planning.core.target_profile import DEFAULT_TARGET_PROFILE


class KernelRegistrationError(ValueError):
    pass


def registered_kernel_names(plan, *, loop_stages=False,
                            min_loop_batches=LOOP_MIN_FULL_BATCHES,
                            compute_lanes=None, narrow_middle_stages=False):
    names = []
    if plan.distributed is not None:
        from planning.execution.distributed_phases import resolve_distributed_phases
        return tuple(p.name for p in resolve_distributed_phases(plan,
                     compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages))
    if plan.persistent is not None:
        if plan.persistent.storage_mode == "global":
            return (*(f"stage_{s.stage_id}" for s in plan.stages), "writeback")
        return ("preload", *(f"stage_{s.stage_id}" for s in plan.stages), "writeback")
    twiddle_table = []
    for stage in plan.stages:
        names.append(f"stage_{stage.stage_id}")
        if not loop_stages:
            continue
        batches_by_worker = (stage.worker_batches if plan.cooperation is not None
                             else (stage.batches,))
        has_tail = False
        for batches in batches_by_worker:
            if not batches:
                continue
            loop = try_build_loop_stage(batches, simd_lanes=plan.simd_lanes,
                                        min_full_batches=min_loop_batches,
                                        twiddle_table=twiddle_table)
            has_tail |= loop is not None and loop.tail_batch is not None
        if has_tail:
            # Cooperative workers share one tail function for this stage.
            names.append(f"stage_{stage.stage_id}_tail")
    return tuple(names)


def validate_kernel_registration(plan, *, max_kernel_register=DEFAULT_TARGET_PROFILE.max_kernel_register,
                                 loop_stages=False, min_loop_batches=LOOP_MIN_FULL_BATCHES,
                                 compute_lanes=None, narrow_middle_stages=False):
    names = registered_kernel_names(plan, loop_stages=loop_stages,
                                    min_loop_batches=min_loop_batches, compute_lanes=compute_lanes,
                                    narrow_middle_stages=narrow_middle_stages)
    if len(names) > max_kernel_register:
        raise KernelRegistrationError(
            f"{plan.kernel_name} needs {len(names)} distinct registered kernel functions "
            f"({', '.join(names)}), but max_kernel_register={max_kernel_register}; "
            "stage tails/preload/writeback count separately, repeated rounds do not"
        )
    return names


def validate_recursive_kernel_registration(plan, *, target=DEFAULT_TARGET_PROFILE,
                                           loop_stages=True, compute_lanes=None,
                                           narrow_middle_stages=False):
    from planning.strategies.fft_plan_recursive import flatten_recursive_node, PhysicalTransposePlan
    hybrid = getattr(plan.root, "tail_hybrid", None)
    for node in flatten_recursive_node(plan.root):
        if isinstance(node, PhysicalTransposePlan):
            if target.max_kernel_register < 1:
                raise KernelRegistrationError(f"{node.kernel_name} needs one registered transpose function")
            continue
        validate_kernel_registration(node, max_kernel_register=target.max_kernel_register,
                                     loop_stages=(False if hybrid is not None else loop_stages),
                                     compute_lanes=compute_lanes,
                                     narrow_middle_stages=narrow_middle_stages)
