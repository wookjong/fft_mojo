"""Render a planned task grouping; reuse existing phase bodies and FFT math."""
from dataclasses import asdict, replace
import json
from codegen.common import Emitter, TaskKernelBinding, emit_aligned_pool, emit_workspace_buffers, emit_execution_record
from codegen.fft_codegen import _emit_params_struct
from codegen.fft_distributed_codegen import emit_distributed_stage
from planning.strategies.fft_plan_recursive import PhysicalTransposePlan


def emit_fused_task_structs(e, *, nodes, fusion, compute_lanes, narrow_middle_stages, target):
    from codegen.fft_transpose_codegen import _emit_physical_transpose_params_struct, _emit_physical_transpose_stage
    record = asdict(fusion)
    del record['private_phases']
    if fusion.far12_codelet == 'off': del record['far12_codelet']
    if fusion.spad_allocation == 'declared':
        del record['spad_allocation']; del record['memory_allocation']
    if fusion.leaf_phase_policy == 'stages': del record['leaf_phase_policy']
    if fusion.leaf_codegen_schedule == 'existing': del record['leaf_codegen_schedule']
    if fusion.private_phases:
        record['private_phase_resources'] = [None if p is None else dict(stage_ids=p.phase.stage_ids,
            scratch_offset=p.scratch_offset,scratch_elements=p.scratch_elements,
            components=len(p.phase.components)) for p in fusion.private_phases]
    if fusion.far12_codelet != 'off':
        record['register_codelets']=[None if p is None else dict(mode=p.codelet_mode,
            edges=[asdict(edge) for edge in p.codelets],private_allocation='unchanged/conservative')
            for p in fusion.private_phases]
    if fusion.boundary_layout == "legacy":
        for key in ("boundary_layout","near_group","far_group"):
            del record[key]
    if not fusion.semantic_node_ids:
        del record["semantic_node_ids"]
    if fusion.boundary_fusion == 'off':
        del record['boundary_fusion']
    if not fusion.window_batch:
        for key in ('window_batch', 'total_batch', 'window_n'):
            del record[key]
    if not fusion.control_pool_ids:
        del record['control_pool_ids']  # Preserve regular fused source verbatim.
    if not any(fusion.leaf_storage):
        del record['leaf_storage']  # Preserve regular fused source verbatim.
    e.add('# FFT_TASK_FUSION ' + json.dumps(record, sort_keys=True))
    # Bind each leaf-resident node's own Q/bank layout (see
    # task_fusion.resolve_leaf_resident_storage); `arena_ref` is filled in
    # below, once each node's own fused group -- hence its own `comptime
    # arena` -- is known.
    leaf_storage = fusion.leaf_storage or (None,) * len(nodes)
    nodes = tuple(node if resident is None else replace(node, leaf_storage=resident)
                 for node, resident in zip(nodes, leaf_storage))
    for node in nodes:
        if isinstance(node, PhysicalTransposePlan):
            _emit_physical_transpose_params_struct(e, plan=node)
        else:
            _emit_params_struct(e, plan=node, add_loop_twiddle=True)
    e.add('@fieldwise_init')
    e.add('struct FFTFusedParams(Movable):')
    for index, node in enumerate(nodes):
        e.add(f'    var node_{index}: {node.kernel_name}Params')
    e.add()
    tables = {}
    period = target.num_ndp_units * target.interleave_chunk_uthreads
    slot = f'(gid // {period}) * {target.interleave_chunk_uthreads} + gid % {target.interleave_chunk_uthreads}'
    for group in fusion.groups:
        e.add(f'struct {group.name}(NDPTask):')
        e.add('    comptime Params = FFTFusedParams')
        e.add(f'    comptime arena = scratchpad[{fusion.arena_elements}, Float32, name="{group.name.lower()}_arena"]()')
        if fusion.window_batch:
            e.add(f'    comptime batch_base = scratchpad[1, Int32, name="{group.name.lower()}_batch_base"]()')
        e.add()
        for index in group.node_ids:
            node = nodes[index]
            if isinstance(node, PhysicalTransposePlan):
                binding = TaskKernelBinding(group.name, f'node_{index}', fusion.node_functions[index][0],
                                            node.total_uthreads, slot, window_batch=fusion.window_batch,
                                            total_batch=fusion.total_batch, window_n=fusion.window_n)
                _emit_physical_transpose_stage(e, plan=node, binding=binding)
                e.add()
            else:
                if node.leaf_storage is not None:
                    node = replace(node, leaf_storage=replace(node.leaf_storage, arena_ref=f'{group.name}.arena'))
                table = []
                private = fusion.private_phases[index] if fusion.private_phases else None
                semantic = fusion.semantic_node_ids[index] if fusion.semantic_node_ids else index
                from planning.execution.fft_worker_private import node_phase_ids
                phases = node_phase_ids(node,private)
                for ids, name in zip(phases, fusion.node_functions[index]):
                    stage = node.stages[ids[0]]
                    binding = TaskKernelBinding(group.name, f'node_{index}', name, node.total_uthreads, slot,
                        window_batch=fusion.window_batch, total_batch=fusion.total_batch, window_n=fusion.window_n)
                    if len(ids) > 1:
                        from codegen.fft_worker_private_codegen import emit_worker_private_phase
                        emit_worker_private_phase(e,node=node,private=private,binding=binding,twiddle_table=table)
                    elif semantic == 3 and stage.stage_id == 0 and fusion.leaf_codegen_schedule == 'far0-short-live':
                        from codegen.fft_worker_private_codegen import emit_short_live_far0
                        emit_short_live_far0(e,node=node,binding=binding,twiddle_table=table)
                    else:
                        emit_distributed_stage(e, plan=node, stage=stage, compute_lanes=compute_lanes,
                                               narrow_middle_stages=narrow_middle_stages,
                                               twiddle_table=table, binding=binding)
                tables[index] = table
                emit_execution_record(e, plan=node, compute_lanes=compute_lanes,
                                      narrow_middle_stages=narrow_middle_stages,
                                      strategy='distributed', synchronization='kernel_completion',
                                      exchange='unit_spad_pingpong' if node.leaf_storage is not None else 'cxl_pingpong',
                                      units_per_leaf=node.distributed.leaf_units,
                                      leaf_groups=node.distributed.leaf_groups,
                                      phase_stage_ids=[list(ids) for ids in phases],
                                      task_owner=group.name,
                                      registered_functions=list(fusion.node_functions[index]),
                                      estimated_array_bytes=node.estimated_dram_bytes)
        e.add('    @staticmethod')
        e.add('    def device_main():')
        if (fusion.reason == 'layout-matched control'
                and not any(isinstance(nodes[i], PhysicalTransposePlan) for i in group.node_ids)):
            # A leaf-only control otherwise never references the arena and the
            # compiler removes its reservation, changing adaptive L1D capacity.
            # One controller store retains it; report this control-only cost.
            e.add(f'        var reserved_arena = spad_addr({group.name}.arena, 0)')
            e.add(f'        reserved_arena.store({fusion.arena_elements - 1}, Float32(0))')
        if fusion.window_batch:
            e.add(f'        for batch_base in range(0, {fusion.total_batch}, {fusion.window_batch}):')
            e.add(f'            for unit in range({target.num_ndp_units}):')
            e.add(f'                spad_addr({group.name}.batch_base, unit).store(0, Int32(batch_base))')
        indent = '            ' if fusion.window_batch else '        '
        for name in group.functions:
            e.add(f'{indent}launch_parallel[{group.name}.{name}]()')
        e.add()
    return tables


def emit_fused_pools(e, *, fusion, simd_lanes, target, debug=False):
    pool_ids = fusion.control_pool_ids or tuple(range(len(fusion.groups)))
    for index in dict.fromkeys(pool_ids):
        e.add(f'    var fusion_pool{index}_elems = {fusion.common_uthreads * simd_lanes}')
        emit_aligned_pool(e, name=f'fusion_pool{index}', elems=f'fusion_pool{index}_elems',
                          alignment_bytes=target.mapping_stride_bytes)
        if debug:
            e.add(f'    print("[pool_align] fused task {index} addr", Int(fusion_pool{index}))')


def emit_fused_host_launches(e, *, nodes, fusion, buffer_names, twiddle_names,
                             tile_coord_names, loop_twiddle_names, buffer_bindings=None):
    # A leaf-resident node (see task_fusion.resolve_leaf_resident_storage)
    # needs neither a host CXL allocation nor a Params argument for its own
    # intermediate banks below -- match emit_fused_task_structs's own
    # node-by-node leaf_storage binding so the two argument lists agree.
    leaf_storage = fusion.leaf_storage or (None,) * len(nodes)
    nodes = tuple(node if resident is None else replace(node, leaf_storage=resident)
                 for node, resident in zip(nodes, leaf_storage))
    workspace = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, PhysicalTransposePlan):
            workspace[index] = emit_workspace_buffers(e, plan=node, prefix=f'fusion_node{index}_')
    args = []
    for index, node in enumerate(nodes):
        values = [buffer_names(index - 1, 'real'), buffer_names(index - 1, 'imag'),
                  buffer_names(index, 'real'), buffer_names(index, 'imag')]
        if buffer_bindings is not None:
            binding = buffer_bindings[index]
            values = [f'{binding.input_buffer}_real', f'{binding.input_buffer}_imag',
                      f'{binding.output_buffer}_real', f'{binding.output_buffer}_imag']
        if isinstance(node, PhysicalTransposePlan):
            values.extend(twiddle_names.get(index, ()))
            values.extend(name for name in tile_coord_names.get(index, ()) if name is not None)
            if node.needs_round_split:
                values.append('0')  # Common launch covers the complete tile grid.
        else:
            if node.boundary_twiddle:
                values.extend(twiddle_names[index])
            values.extend(loop_twiddle_names[index])
            values.extend(workspace[index])
        args.append(f'{node.kernel_name}Params({", ".join(values)})')
    for index, group in enumerate(fusion.groups):
        pool_id = fusion.control_pool_ids[index] if fusion.control_pool_ids else index
        e.add(f'    var fusion_rc{index} = {group.name}.launch(')
        e.add(f'        PooledRange.over(fusion_pool{pool_id}, fusion_pool{pool_id}_elems),')
        e.add(f'        FFTFusedParams({", ".join(args)}),')
        e.add('    )')
        e.add(f'    if fusion_rc{index} != 0:')
        e.add(f'        print("[host] fused FFT task {index} failed, exit", fusion_rc{index})')
        e.add('        return')
        e.add()
