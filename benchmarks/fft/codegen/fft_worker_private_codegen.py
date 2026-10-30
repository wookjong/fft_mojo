"""Worker-private stage bodies; no inter-worker synchronization inside a phase."""
from dataclasses import replace
from codegen.common import Emitter
from codegen.fft_codegen import _mapping_base_expr, _emit_loop_chunk
from codegen.fft_distributed_codegen import _emit_local_stage
from codegen.lowering import chunk_batch, try_build_loop_stage


def _prelude(e, node, binding):
    e.add('    @staticmethod')
    e.add(f'    def {binding.function_name}():')
    e.add('        var gid = global_uthread_id()')
    e.add(f'        if gid >= {binding.active_uthreads}:')
    e.add('            return')
    e.add(f'        ref p = {binding.params_expr}')
    binding.emit_window_prelude(e)
    e.add('        var work_slot = gid % 8')
    e.add('        var unit_slot = gid // 8')


def _bases(e, node, binding, indent):
    prefix='window_base + ' if binding.window_batch else ''
    for name,mapping in (('in_batch_base',node.input_mapping),('out_batch_base',node.output_mapping)):
        expr=_mapping_base_expr(mapping,node.length,row_index='fft_index')
        e.add(' '*indent+f'var {name} = {prefix}({expr})')
    e.add(' '*indent+f'var spad_base = (fft_index // {node.distributed.num_units}) * {2*node.length}')


def emit_worker_private_phase(e, *, node, private, binding, twiddle_table):
    if node.leaf_storage is None:
        raise ValueError('private phase requires bound resident storage')
    phase=private.phase; count=len(phase.components); units=node.distributed.num_units
    local=replace(node,phase_scratch_ref=node.leaf_storage.arena_ref,
                  scratchpad_uthread_stride=2*phase.local_elements)
    _prelude(e,node,binding)
    e.add(f'        var local_base = {private.scratch_offset} + work_slot * {2*phase.local_elements}')
    replicas=node.host.total_elems//node.length
    tail=binding.window_batch and binding.total_batch%binding.window_batch
    limit=f'live_batch * {binding.window_n//node.length}' if tail else str(replicas)
    # Unit-local slots are traversed together, so independent components in
    # different resident leaves can occupy different workers in the same phase.
    e.add(f'        var component_job = work_slot')
    e.add(f'        while component_job < {node.leaf_storage.slots*count}:')
    e.add(f'            var leaf_slot = component_job // {count}')
    e.add(f'            var component_index = component_job % {count}')
    e.add(f'            var fft_index = unit_slot + leaf_slot * {units}')
    e.add(f'            if fft_index < {limit}:')
    _bases(e,node,binding,16)
    reference_table=None
    if private.codelets:
        reference_table=list(twiddle_table)
        for component in phase.components:
            for sid in phase.stage_ids:
                chunks=tuple(c for c in component.chunks if c.stage_id==sid)
                _emit_local_stage(Emitter(),plan=local,stage=node.stages[sid],chunks=chunks,twiddle_table=reference_table)
    for index,component in enumerate(phase.components):
        e.add(f'                if component_index == {index}:')
        # The existing local stage lowering emits at indentation 16/20.
        body=Emitter()
        if private.codelets:
            if private.codelet_mode == 'native4-latch-forward':
                from codegen.fft_latch_handoff_codegen import emit_latch_component
                if len(private.handoffs)!=len(phase.components):raise ValueError('missing physical latch proof')
                emit_latch_component(body,node=node,local=local,component=component,
                    handoff=private.handoffs[index],mode=private.codelet_mode,twiddle_table=twiddle_table)
            elif private.codelet_mode in ('native4-compact-control','native4-compact-forward'):
                from codegen.fft_compact_handoff_codegen import emit_compact_component
                if len(private.handoffs)!=len(phase.components):raise ValueError('missing physical compact proof')
                emit_compact_component(body,node=node,local=local,component=component,
                    handoff=private.handoffs[index],mode=private.codelet_mode,twiddle_table=twiddle_table)
            elif private.codelet_mode in ('native4-static-control','native4-static-forward'):
                from codegen.fft_static_handoff_codegen import emit_static_component
                if len(private.handoffs)!=len(phase.components):raise ValueError('missing physical static proof')
                emit_static_component(body,node=node,local=local,component=component,
                    handoff=private.handoffs[index],mode=private.codelet_mode,twiddle_table=twiddle_table)
            elif private.codelet_mode in ('loop-forward1','loop-forward-packed','native4-forward'):
                from codegen.fft_register_loop_codegen import emit_loop_component
                emit_loop_component(body,node=node,local=local,component=component,
                    edge=private.codelets[index],twiddle_table=twiddle_table,packed=private.codelet_mode in ('loop-forward-packed','native4-forward'))
            else:
                from codegen.fft_register_codelet_codegen import emit_component
                emit_component(body,node=node,local=local,component=component,edge=private.codelets[index],
                    mode=private.codelet_mode,twiddle_table=twiddle_table)
        else:
            for sid in phase.stage_ids:
                chunks=tuple(c for c in component.chunks if c.stage_id==sid)
                _emit_local_stage(body,plan=local,stage=node.stages[sid],chunks=chunks,twiddle_table=twiddle_table)
        for line in body.lines:e.add('    '+line if line else '')
    if reference_table is not None:
        if len(twiddle_table)>len(reference_table):raise ValueError('codelet expands the original twiddle pool')
        twiddle_table.extend(reference_table[len(twiddle_table):])
    e.add('            component_job += 8')
    e.add()


def emit_short_live_far0(e, *, node, binding, twiddle_table):
    """Lower SIMD4 chunks directly, eliminating SIMD8 chunk dispatch branches."""
    if node.leaf_storage is None or node.leaf_ownership is not None:
        raise ValueError('short-live far0 requires far C1 resident storage')
    stage=node.stages[0];width=4
    batches=tuple(b for original in stage.batches for w,b in chunk_batch(original,
        simd_lanes=node.simd_lanes,compute_lanes=width) if b.valid_lanes)
    loop=try_build_loop_stage(batches,simd_lanes=width,min_full_batches=2,twiddle_table=twiddle_table)
    if loop is None or loop.tail_batch is not None or loop.period & (loop.period-1):
        raise ValueError('short-live far0 requires a regular power-of-two SIMD4 loop')
    local=replace(node,simd_lanes=width)
    _prelude(e,node,binding)
    e.add('        var fft_index = unit_slot')
    replicas=node.host.total_elems//node.length
    tail=binding.window_batch and binding.total_batch%binding.window_batch
    limit=f'live_batch * {binding.window_n//node.length}' if tail else str(replicas)
    e.add(f'        while fft_index < {limit}:')
    _bases(e,node,binding,12)
    e.add('            var stage_job = work_slot')
    e.add(f'            while stage_job < {loop.outer_iters*loop.period}:')
    e.add(f'                var outer_it = stage_job // {loop.period}')
    e.add(f'                var residue_it = stage_job % {loop.period}')
    body=Emitter()
    _emit_loop_chunk(body,plan=local,stage=stage,loop_plan=loop,residue='residue_it',chunk_offset=0,width=width)
    for line in body.lines:e.add('        '+line if line else '')
    e.add('                stage_job += 8')
    e.add(f'            fft_index += {node.distributed.num_units}')
    e.add()
