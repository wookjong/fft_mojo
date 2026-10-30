"""Cross-unit FFT stages exchanging results through shared CXL SoA banks.

Every launch_parallel is a completion barrier across all participating units.
Writers own disjoint butterfly outputs; the following phase reads the complete
previous bank. No device spin barrier or remote scratchpad access is needed.
"""
from dataclasses import replace

from codegen.common import Emitter, emit_execution_record
from codegen.fft_codegen import (
    _emit_batch, _emit_loop_chunk, _emit_loop_stage, _emit_params_struct, _mapping_base_expr,
    _residue_pattern, _stage_compute_lanes,
)
from codegen.lowering import chunk_batch, try_build_loop_stage
from planning.execution.distributed_phases import resolve_distributed_phases


def _nest(e, body, spaces):
    for line in body.lines:
        e.add(" " * spaces + line if line else "")


def emit_distributed_stage(e, *, plan, stage, compute_lanes=4,
                           narrow_middle_stages=False, twiddle_table=None, binding=None):
    if plan.leaf_ownership is not None:
        from codegen.fft_boundary_simd_codegen import emit_grouped_stage
        return emit_grouped_stage(e,plan=plan,stage=stage,compute_lanes=compute_lanes,
            narrow_middle_stages=narrow_middle_stages,twiddle_table=twiddle_table,binding=binding)
    distribution = plan.distributed
    if distribution is None:
        raise ValueError("distributed stage needs a distributed execution plan")
    width = stage.compute_lanes or _stage_compute_lanes(
        compute_lanes=compute_lanes, is_first=stage.stage_id == 0,
        is_last=stage.stage_id == len(plan.stages) - 1, radix=stage.radix,
        prev_radix=plan.stages[stage.stage_id - 1].radix if stage.stage_id else None,
        narrow_middle_stages=narrow_middle_stages) or plan.simd_lanes
    if width < 1 or width > plan.simd_lanes or plan.simd_lanes % width:
        raise ValueError("distributed compute lanes must divide the launch SIMD width")
    table = twiddle_table if twiddle_table is not None else []
    table_start = len(table)
    loop = try_build_loop_stage(stage.batches, simd_lanes=plan.simd_lanes,
                               min_full_batches=2, twiddle_table=table)
    # Non-power-of-two division can lower to unsupported mulhsu on this
    # simulator. Unroll those proven plans instead of emitting that opcode.
    if loop is not None and loop.period & (loop.period - 1):
        loop = None
        del table[table_start:]
    e.add("    @staticmethod")
    e.add(f"    def {binding.function_name if binding else f'stage_{stage.stage_id}'}():")
    if binding is None:
        e.add(f"        ref p = {plan.kernel_name}.params[]")
        e.add("        var gid = global_uthread_id()")
    else:
        e.add("        var gid = global_uthread_id()")
        e.add(f"        if gid >= {binding.active_uthreads}:")
        e.add("            return")
        e.add(f"        ref p = {binding.params_expr}")
        binding.emit_window_prelude(e)
    _emit_dispatch(e, plan)
    replicas = plan.host.total_elems // plan.length
    e.add("        var fft_index = leaf_group")
    has_tail = binding and binding.window_batch and binding.total_batch % binding.window_batch
    limit = f'live_batch * {binding.window_n // plan.length}' if has_tail else str(replicas)
    e.add(f"        while fft_index < {limit}:")
    body = Emitter()
    for name, mapping in (("in_batch_base", plan.input_mapping), ("out_batch_base", plan.output_mapping)):
        expr = _mapping_base_expr(mapping, plan.length, row_index="fft_index")
        if binding and binding.window_batch:
            expr = f'window_base + ({expr})'
        body.add(f"        var {name} = {expr}")
    if plan.leaf_storage is not None:
        # Every worker assigned to this leaf (same fft_index) must land on
        # the same resident slot across every stage barrier -- see
        # LeafResidentSpadPlan -- not the per-replica CXL offset below.
        body.add(f"        var spad_base = (fft_index // {distribution.leaf_groups}) * {2 * plan.length}")
    else:
        body.add(f"        var spad_base = fft_index * {2 * plan.length}")
    chunks = plan.simd_lanes // width
    if loop is not None:
        jobs = loop.outer_iters * loop.period * chunks
        body.add("        var work_index = work_slot")
        body.add(f"        while work_index < {jobs}:")
        body.add(f"            var batch_it = work_index // {chunks}")
        body.add(f"            var chunk_it = work_index % {chunks}")
        body.add(f"            var outer_it = batch_it // {loop.period}")
        body.add(f"            var residue_it = batch_it % {loop.period}")
        arrays = [load.bases for load in loop.loads]
        for store in loop.stores:
            arrays.extend((store.bases,) if store.mode == "vector" else store.lane_bases)
        compact = all(_residue_pattern(values) is not None for values in arrays)
        for c in range(chunks):
            body.add(f"            if chunk_it == {c}:")
            if compact:
                sub = Emitter()
                _emit_loop_chunk(sub, plan=plan, stage=stage, loop_plan=loop,
                                 residue="residue_it", chunk_offset=c * width, width=width)
                _nest(body, sub, 8)
            else:
                for residue in range(loop.period):
                    body.add(f"                if residue_it == {residue}:")
                    sub = Emitter()
                    _emit_loop_chunk(sub, plan=plan, stage=stage, loop_plan=loop,
                                     residue=residue, chunk_offset=c * width, width=width)
                    _nest(body, sub, 12)
        body.add(f"            work_index += {distribution.workers_per_leaf}")
        tails = (loop.tail_batch,) if loop.tail_batch is not None else ()
        first_job = jobs
    else:
        tails = stage.batches
        first_job = 0
    job = first_job
    for batch in tails:
        for chunk_width, chunk in chunk_batch(batch, compute_lanes=width, simd_lanes=plan.simd_lanes):
            if not chunk.valid_lanes:
                continue
            body.add(f"        if work_slot == {job % distribution.workers_per_leaf}:")
            sub = Emitter()
            _emit_batch(sub, plan=plan, stage=stage, batch=chunk, width=chunk_width)
            _nest(body, sub, 4)
            job += 1
    body.add(f"        fft_index += {distribution.leaf_groups}")
    _nest(e, body, 4)
    e.add()


def _emit_dispatch(e, plan):
    distribution = plan.distributed
    e.add(f"        var unit_slot = gid // {distribution.workers_per_unit}")
    if distribution.leaf_units == distribution.num_units:
        e.add("        var leaf_group = 0")
        e.add(f"        var work_slot = unit_slot + (gid % {distribution.workers_per_unit}) * {distribution.num_units}")
        return
    e.add(f"        var leaf_group = unit_slot // {distribution.leaf_units}")
    e.add(f"        if leaf_group >= {distribution.leaf_groups}:")
    e.add("            return")
    e.add(f"        var work_slot = unit_slot % {distribution.leaf_units} + (gid % {distribution.workers_per_unit}) * {distribution.leaf_units}")


def _emit_local_stage(e, *, plan, stage, chunks, twiddle_table):
    """Reuse proven loop lowering instead of unrolling a whole local tile."""
    width = chunks[0].width
    batches = tuple(chunk.batch for chunk in chunks)
    table_start = len(twiddle_table)
    loop = try_build_loop_stage(batches, simd_lanes=width, min_full_batches=2,
                                twiddle_table=twiddle_table)
    if loop is not None and loop.period & (loop.period - 1):
        loop = None
        del twiddle_table[table_start:]
    if loop is not None:
        # Only the local instruction/twiddle rows use this width. The host
        # launch remains SIMD 8 and retains the physical decoder period.
        sub = Emitter()
        _emit_loop_stage(sub, plan=replace(plan, simd_lanes=width), stage=stage,
                         loop_plan=loop, compute_lanes=width, compact_residues=True)
        e.add("                if True:")
        _nest(e, sub, 12)
        tails = (loop.tail_batch,) if loop.tail_batch is not None else ()
    else:
        tails = batches
    for batch in tails:
        sub = Emitter()
        _emit_batch(sub, plan=plan, stage=stage, batch=batch, width=width)
        e.add("                if True:")
        _nest(e, sub, 12)


def emit_distributed_phase(e, *, plan, phase, compute_lanes=4,
                           narrow_middle_stages=False, twiddle_table=None):
    if not phase.components:
        emit_distributed_stage(e, plan=plan, stage=plan.stages[phase.stage_ids[0]],
                               compute_lanes=compute_lanes,
                               narrow_middle_stages=narrow_middle_stages,
                               twiddle_table=twiddle_table)
        return
    distribution = plan.distributed
    table = twiddle_table if twiddle_table is not None else []
    local_plan = replace(plan, scratchpad_uthread_stride=2 * phase.local_elements)
    e.add("    @staticmethod")
    e.add(f"    def {phase.name}():")
    e.add(f"        ref p = {plan.kernel_name}.params[]")
    e.add("        var gid = global_uthread_id()")
    _emit_dispatch(e, plan)
    e.add(f"        var local_base = (gid % {distribution.workers_per_unit}) * {2 * phase.local_elements}")
    e.add("        var fft_index = leaf_group")
    e.add(f"        while fft_index < {plan.host.total_elems // plan.length}:")
    for name, mapping in (("in_batch_base", plan.input_mapping), ("out_batch_base", plan.output_mapping)):
        expr = _mapping_base_expr(mapping, plan.length).replace("global_uthread_id()", "fft_index")
        e.add(f"            var {name} = {expr}")
    e.add(f"            var spad_base = fft_index * {2 * plan.length}")
    for job, component in enumerate(phase.components):
        e.add(f"            if work_slot == {job % distribution.workers_per_leaf}:")
        for stage_id in phase.stage_ids:
            chunks = tuple(c for c in component.chunks if c.stage_id == stage_id)
            _emit_local_stage(e, plan=local_plan, stage=plan.stages[stage_id],
                              chunks=chunks, twiddle_table=table)
    e.add(f"            fft_index += {distribution.leaf_groups}")
    e.add()


def emit_distributed_kernel_struct(e, *, plan, compute_lanes=4,
                                    narrow_middle_stages=False, max_kernel_register=16):
    from planning.execution.kernel_registration import validate_kernel_registration
    phases = resolve_distributed_phases(plan, compute_lanes=compute_lanes,
                                        narrow_middle_stages=narrow_middle_stages)
    validate_kernel_registration(plan, max_kernel_register=max_kernel_register,
                                 compute_lanes=compute_lanes,
                                 narrow_middle_stages=narrow_middle_stages)
    local_elements = max(p.local_elements for p in phases)
    emit_execution_record(e, plan=plan, compute_lanes=compute_lanes,
                          narrow_middle_stages=narrow_middle_stages,
                          strategy="distributed", synchronization="kernel_completion",
                          exchange="cxl_pingpong", workspace_bytes=sum(b.elements * 4 for b in plan.workspace_buffers),
                          units_per_leaf=plan.distributed.leaf_units,
                          leaf_groups=plan.distributed.leaf_groups,
                          phase_stage_ids=[list(p.stage_ids) for p in phases],
                          local_scratchpad_bytes=2 * local_elements * 4 * plan.distributed.workers_per_unit,
                          estimated_array_bytes=16 * plan.host.total_elems * len(phases))
    _emit_params_struct(e, plan=plan, add_loop_twiddle=True)
    e.add(f"struct {plan.kernel_name}(NDPTask):")
    e.add(f"    comptime Params = {plan.kernel_name}Params")
    e.add()
    if local_elements:
        elements = 2 * local_elements * plan.distributed.workers_per_unit
        e.add(f'    comptime phase_scratch = scratchpad[{elements}, Float32, name="{plan.kernel_name.lower()}_phase_scratch"]()')
        e.add()
    table = []
    for phase in phases:
        emit_distributed_phase(e, plan=plan, phase=phase, compute_lanes=compute_lanes,
                               narrow_middle_stages=narrow_middle_stages, twiddle_table=table)
    e.add("    @staticmethod")
    e.add("    def device_main():")
    for phase in phases:
        e.add("        # Complete every unit's writes before the next CXL exchange.")
        e.add(f"        launch_parallel[{plan.kernel_name}.{phase.name}]()")
    e.add()
    return table
