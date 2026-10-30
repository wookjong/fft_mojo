"""Grouped stage dispatcher and cross-leaf PRE/POST IO, preserving radix barriers."""
from codegen.common import Emitter
from codegen.fft_codegen import (_emit_batch,_emit_loop_chunk,_mapping_base_expr,
    _residue_pattern,_stage_compute_lanes,_f32,emit_butterfly)
from codegen.lowering import chunk_batch,try_build_loop_stage


def _copy(e,sub,spaces):
    for line in sub.lines:e.add(' '*spaces+line if line else '')


def _affine(values):
    start=values[0];stride=values[1]-start if len(values)>1 else 0
    if any(v!=start+i*stride for i,v in enumerate(values)):
        raise ValueError('initial grouped emitter requires proven affine scalar offsets')
    return start,stride


def _expr(values,index):
    base,stride=_affine(values)
    return f'({base} + ({index}) * {stride})'


def emit_grouped_stage(e,*,plan,stage,compute_lanes,narrow_middle_stages,twiddle_table,binding):
    if binding is None or plan.leaf_storage is None:
        raise ValueError('grouped emitter requires a fused task and resident SPAD')
    owner=plan.leaf_ownership
    c=owner.group_leaves;u=owner.num_units;length=plan.length
    special=plan.boundary_simd
    special=special if special is not None and special.stage.stage_id==stage.stage_id else None
    table=twiddle_table if twiddle_table is not None else []
    e.add('    @staticmethod')
    e.add(f'    def {binding.function_name}():')
    e.add('        var gid = global_uthread_id()')
    e.add(f'        if gid >= {binding.active_uthreads}:')
    e.add('            return')
    e.add(f'        ref p = {binding.params_expr}')
    binding.emit_window_prelude(e)
    e.add(f'        var work_slot = gid % {owner.workers_per_unit}')
    e.add(f'        var tile = gid // {owner.workers_per_unit}')
    tail=binding.window_batch and binding.total_batch%binding.window_batch
    limit=f'live_batch * {owner.tiles_per_batch}' if tail else str(owner.max_tiles)
    e.add(f'        while tile < {limit}:')
    e.add(f'            var tile_slot = tile // {u}')
    e.add(f'            var tile_batch = tile // {owner.tiles_per_batch}')
    e.add(f'            var leaf_base = (tile % {owner.tiles_per_batch}) * {c}')
    global_base='window_base' if binding.window_batch else '0'
    if special is not None:
        _emit_boundary(e,plan,stage,special,table,global_base)
    else:
        _emit_regular(e,plan,stage,table,compute_lanes,narrow_middle_stages,global_base)
    e.add(f'            tile += {u}')
    e.add()


def _emit_regular(e,plan,stage,table,compute_lanes,narrow,global_base):
    owner=plan.leaf_ownership;c=owner.group_leaves
    width=stage.compute_lanes or _stage_compute_lanes(compute_lanes=compute_lanes,
        is_first=stage.stage_id==0,is_last=stage.stage_id==len(plan.stages)-1,radix=stage.radix,
        prev_radix=plan.stages[stage.stage_id-1].radix if stage.stage_id else None,
        narrow_middle_stages=narrow) or plan.simd_lanes
    table_start=len(table)
    loop=try_build_loop_stage(stage.batches,simd_lanes=plan.simd_lanes,min_full_batches=2,twiddle_table=table)
    if loop is not None and loop.period & (loop.period-1):
        loop=None;del table[table_start:]
    chunks=plan.simd_lanes//width
    if loop is not None:
        if loop.tail_batch is not None:raise ValueError('grouped loop tails are outside initial support')
        per_leaf=loop.outer_iters*loop.period*chunks
    else:
        packed=[(w,b) for batch in stage.batches for w,b in chunk_batch(batch,
            compute_lanes=width,simd_lanes=plan.simd_lanes) if b.valid_lanes]
        per_leaf=len(packed)
    e.add('            var tile_work = work_slot')
    e.add(f'            while tile_work < {c*per_leaf}:')
    e.add(f'                var leaf_lane = tile_work // {per_leaf}')
    e.add(f'                var work_index = tile_work % {per_leaf}')
    e.add(f'                var fft_index = tile_batch * {owner.leaves_per_batch} + leaf_base + leaf_lane')
    for name,mapping in (('in_batch_base',plan.input_mapping),('out_batch_base',plan.output_mapping)):
        expr=_mapping_base_expr(mapping,plan.length,row_index='fft_index')
        e.add(f'                var {name} = {global_base} + ({expr})')
    e.add(f'                var spad_base = ({c} * tile_slot + leaf_lane) * {2*plan.length}')
    if loop is not None:
        e.add(f'                var batch_it = work_index // {chunks}')
        e.add(f'                var chunk_it = work_index % {chunks}')
        e.add(f'                var outer_it = batch_it // {loop.period}')
        e.add(f'                var residue_it = batch_it % {loop.period}')
        arrays=[x.bases for x in loop.loads]
        for s in loop.stores:arrays.extend((s.bases,) if s.mode=='vector' else s.lane_bases)
        compact=all(_residue_pattern(v) is not None for v in arrays)
        for chunk in range(chunks):
            e.add(f'                if chunk_it == {chunk}:')
            if compact:
                sub=Emitter();_emit_loop_chunk(sub,plan=plan,stage=stage,loop_plan=loop,
                    residue='residue_it',chunk_offset=chunk*width,width=width);_copy(e,sub,12)
            else:
                for residue in range(loop.period):
                    e.add(f'                    if residue_it == {residue}:')
                    sub=Emitter();_emit_loop_chunk(sub,plan=plan,stage=stage,loop_plan=loop,
                        residue=residue,chunk_offset=chunk*width,width=width);_copy(e,sub,16)
    else:
        for index,(w,b) in enumerate(packed):
            e.add(f'                if work_index == {index}:')
            sub=Emitter();_emit_batch(sub,plan=plan,stage=stage,batch=b,width=w);_copy(e,sub,12)
    e.add(f'                tile_work += {owner.workers_per_unit}')


def _emit_boundary(e,plan,stage,special,table,global_base):
    owner=plan.leaf_ownership;c=owner.group_leaves;d=special.butterflies_per_vector
    original=special.stage.butterflies
    pre=special.boundary=='pre'
    table_base=len(table)
    has_twiddle=any(o.twiddle is not None for bf in original for o in bf.outputs)
    if has_twiddle:
        for bf in original:
            for out in bf.outputs:table.append(out.twiddle or (1.0,0.0))
    e.add('            var work_index = work_slot')
    e.add(f'            while work_index < {len(special.jobs)}:')
    e.add(f'                var scalar_bfly = work_index * {d}')
    body=Emitter()
    arena=plan.leaf_storage.arena_ref
    def local(bank,plane,offset,ell):
        return f'{plan.leaf_storage.bank_offset(bank)} + ({c} * tile_slot + {ell}) * {2*plan.length} + {plane*plan.length} + {offset}'
    for operand in range(stage.radix):
        for part,plane,prefix in (('real',0,'rr'),('imag',1,'ii')):
            if pre:
                off=_expr([bf.loads[operand].offset for bf in original],'scalar_bfly')
                addr=f'{global_base} + tile_batch * {special.n} + ({off}) * {special.a} + leaf_base'
                body.add(f'        var {prefix}{operand} = p.input_{part}_base.load[width=4]({addr})')
            else:
                vals=[]
                bank=original[0].loads[operand].buffer_name
                for delta in range(d):
                    off=_expr([bf.loads[operand].offset for bf in original],f'scalar_bfly + {delta}')
                    for ell in range(c):
                        vals.append(f'{arena}.load[DType.float32, 1]({local(bank,plane,off,ell)})[0]')
                body.add(f'        var {prefix}{operand} = SIMD[DType.float32, 4]('+', '.join(vals)+')')
    def output(k):
        if any(bf.outputs[k].twiddle is not None for bf in original):
            if d!=1:raise ValueError('initial stage-twiddle broadcast requires D1')
            addr=f'{table_base} + scalar_bfly * {stage.radix} + {k}'
            for part in ('real','imag'):
                body.add(f'        var tw_{part}{k} = p.loop_twiddle_{part}_base.load[width=1]({addr})[0]')
            body.add(f'        var tr{k} = or{k} * tw_real{k} - oi{k} * tw_imag{k}')
            body.add(f'        var ti{k} = or{k} * tw_imag{k} + oi{k} * tw_real{k}')
            body.add(f'        or{k} = tr{k}');body.add(f'        oi{k} = ti{k}')
        if special.inverse_scale is not None:
            body.add(f'        or{k} *= {_f32(special.inverse_scale)}')
            body.add(f'        oi{k} *= {_f32(special.inverse_scale)}')
        if pre:
            off=_expr([bf.outputs[k].offset for bf in original],'scalar_bfly')
            bank=original[0].outputs[k].buffer_name
            for plane,prefix in ((0,'or'),(1,'oi')):
                for ell in range(c):body.add(f'        {arena}.store({local(bank,plane,off,ell)}, {prefix}{k}[{ell}])')
        else:
            for delta in range(d):
                off=_expr([bf.outputs[k].offset for bf in original],f'scalar_bfly + {delta}')
                addr=f'{global_base} + tile_batch * {special.n} + ({off}) * {special.b} + leaf_base'
                for part,prefix in (('real','or'),('imag','oi')):
                    val=f'{prefix}{k}' if c==4 else 'SIMD[DType.float32, 2]('+', '.join(f'{prefix}{k}[{delta*c+ell}]' for ell in range(c))+')'
                    body.add(f'        p.output_{part}_base.store({addr}, {val})')
    emit_butterfly(body,indent='        ',radix=stage.radix,inverse=stage.inverse,on_output=output)
    _copy(e,body,8)
    e.add(f'                work_index += {owner.workers_per_unit}')
