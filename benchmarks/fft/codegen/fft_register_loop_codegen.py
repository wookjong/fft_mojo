"""Keep producer/consumer loops compact while carrying one SIMD4 edge."""
from dataclasses import replace
from codegen.common import Emitter
from codegen.lowering import try_build_loop_stage
from codegen.fft_codegen import (_emit_load_loop,_emit_store_loop,_emit_twiddle_loop,
    _residue_base,_residue_pattern,_f32)
from codegen.fft_butterflies import emit_butterfly

def _nest(e,sub,spaces):
    for line in sub.lines:e.add(' '*spaces+line if line else '')

def _stage(e,*,node,local,chunks,edge,producer,twiddle_table,packed=False):
    width=chunks[0].width;stage=node.stages[chunks[0].stage_id]
    loop=try_build_loop_stage(tuple(c.batch for c in chunks),simd_lanes=width,
        min_full_batches=2,twiddle_table=twiddle_table)
    if loop is None or loop.tail_batch is not None or loop.period&(loop.period-1):
        raise ValueError('loop register edge requires a complete power-of-two loop')
    arrays=[l.bases for l in loop.loads]
    for store in loop.stores:arrays.extend((store.bases,) if store.mode=='vector' else store.lane_bases)
    if any(_residue_pattern(v) is None for v in arrays):raise ValueError('non-affine register loop')
    plan=replace(local,simd_lanes=width)
    body=Emitter()
    body.add('        var codelet_it = 0')
    body.add(f'        while codelet_it < {len(chunks)}:')
    body.add(f'            var outer_it = codelet_it // {loop.period}')
    body.add(f'            var residue_it = codelet_it % {loop.period}')
    arithmetic=Emitter()
    for load in loop.loads:
        base=_residue_base(load.bases,'residue_it',0)
        expr=f'({base} + outer_it * {load.stride})'
        if not producer and load.operand==edge.consumer_operand:
            for prefix in ('rr','ii'):arithmetic.add(f'        var {prefix}{load.operand} = SIMD[DType.float32, 4](0)')
            arithmetic.add(f'        if codelet_it == {edge.consumer_chunk}:')
            for prefix,part in (('rr','real'),('ii','imag')):
                if packed:arithmetic.add(f'            {prefix}{load.operand} = codelet_{part}')
                else:
                    vals=', '.join(f'codelet_{i}_{part}[{lane}]' for i in range(2) for lane in range(2))
                    arithmetic.add(f'            {prefix}{load.operand} = SIMD[DType.float32, 4]({vals})')
            arithmetic.add('        else:')
            sub=Emitter();_emit_load_loop(sub,plan=plan,load=load,offset_expr=expr,width=width)
            for line in sub.lines:
                line=line.replace(f'var rr{load.operand} =',f'rr{load.operand} =').replace(f'var ii{load.operand} =',f'ii{load.operand} =')
                arithmetic.add('    '+line)
        else:_emit_load_loop(arithmetic,plan=plan,load=load,offset_expr=expr,width=width)
    stores={s.output:s for s in loop.stores}
    def output(k):
        tw=loop.twiddles.get(k)
        if tw is not None:
            expr=f'({tw.table_offset} + residue_it * {width} + outer_it * {loop.period*width})'
            _emit_twiddle_loop(arithmetic,output=k,table_offset_expr=expr,width=width)
        scale=loop.scales.get(k)
        if scale is not None:
            arithmetic.add(f'        or{k} *= {_f32(scale)}');arithmetic.add(f'        oi{k} *= {_f32(scale)}')
        if producer and k==edge.producer_outputs[0]:
            assert len(set(edge.producer_outputs))==1
            for i,index in enumerate(edge.producer_chunks):
                arithmetic.add(f'        {"if" if i==0 else "elif"} codelet_it == {index}:')
                if packed and edge.producer_width==4:
                    for part,prefix in (('real','or'),('imag','oi')):
                        arithmetic.add(f'            codelet_{part} = {prefix}{k}')
                elif packed:
                    for part,prefix in (('real','or'),('imag','oi')):
                        vals=f'{prefix}{k}[0], {prefix}{k}[1], Float32(0), Float32(0)' if i==0 else f'codelet_{part}[0], codelet_{part}[1], {prefix}{k}[0], {prefix}{k}[1]'
                        arithmetic.add(f'            codelet_{part} = SIMD[DType.float32, 4]({vals})')
                else:
                    arithmetic.add(f'            codelet_{i}_real = or{k}')
                    arithmetic.add(f'            codelet_{i}_imag = oi{k}')
            arithmetic.add('        else:')
            sub=Emitter();_emit_store_loop(sub,plan=plan,output=k,store=stores[k],residue='residue_it',chunk_offset=0,width=width)
            _nest(arithmetic,sub,4)
        else:_emit_store_loop(arithmetic,plan=plan,output=k,store=stores[k],residue='residue_it',chunk_offset=0,width=width)
    emit_butterfly(arithmetic,indent='        ',radix=stage.radix,inverse=stage.inverse,on_output=output)
    _nest(body,arithmetic,4)
    body.add('            codelet_it += 1')
    e.add('                if True:')
    _nest(e,body,12)

def emit_loop_component(e,*,node,local,component,edge,twiddle_table,packed=False):
    if packed:
        for part in ('real','imag'):e.add(f'                var codelet_{part} = SIMD[DType.float32, 4](0)')
    else:
        for i in range(2):
            e.add(f'                var codelet_{i}_real = SIMD[DType.float32, 2](0)')
            e.add(f'                var codelet_{i}_imag = SIMD[DType.float32, 2](0)')
    for sid in (1,2):
        chunks=tuple(c for c in component.chunks if c.stage_id==sid)
        _stage(e,node=node,local=local,chunks=chunks,edge=edge,producer=sid==1,twiddle_table=twiddle_table,packed=packed)
