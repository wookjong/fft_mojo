"""One body per stage, with the proved final consumer visited first.

Unlike static peeling, no constant producer/consumer butterfly is duplicated.
The producer captures its last output on every iteration, so the final value
is available without a conditional assignment or a growing vector phi state.
Only the final store and the first consumer load need an edge predicate.
"""
from dataclasses import replace
from codegen.common import Emitter
from codegen.fft_loop_ranges import build_full_loop, emit_range
from codegen.fft_codegen import (_emit_load_loop, _emit_store_loop,
    _emit_twiddle_loop, _residue_base, _f32)
from codegen.fft_butterflies import emit_butterfly


def nest(e, body, spaces):
    for line in body.lines:
        e.add(' '*spaces+line if line else '')


def emit_compact_component(e, *, node, local, component, handoff, mode, twiddle_table):
    producers=tuple(c for c in component.chunks if c.stage_id==1)
    consumers=tuple(c for c in component.chunks if c.stage_id==2)
    left=build_full_loop(producers,twiddle_table)
    right=build_full_loop(consumers,twiddle_table)
    count=len(consumers)
    if (not handoff.adjacent or handoff.producer_chunk!=len(producers)-1
        or handoff.consumer_chunk!=count-1 or count&(count-1)
        or left.loop.period!=1 or right.loop.period!=1):
        raise ValueError('compact handoff requires regular loops and a proved last-to-last edge')
    if handoff.consumer_order!=(count-1,)+tuple(range(count-1)):
        raise ValueError('compact handoff order differs from its physical proof')
    forward=mode=='native4-compact-forward'
    plan=replace(local,simd_lanes=4)
    if forward:
        e.add('                var handoff_real = SIMD[DType.float32, 4](0)')
        e.add('                var handoff_imag = SIMD[DType.float32, 4](0)')

    for producer, full, chunks in ((True,left,producers),(False,right,consumers)):
        if producer and not forward:
            emit_range(e,plan=plan,stage=node.stages[1],full=full,begin=0,end=len(producers))
            continue
        loop=full.loop
        stage=node.stages[1 if producer else 2]
        e.add('                if True:')
        e.add('                    var compact_it = 0')
        e.add(f'                    while compact_it < {len(chunks)}:')
        expr='compact_it' if producer else f'(compact_it + {count-1}) % {count}'
        e.add(f'                        var outer_it = {expr}')
        body=Emitter()
        for load in loop.loads:
            base=_residue_base(load.bases,0,0)
            offset=f'({base} + outer_it * {load.stride})'
            if forward and not producer and load.operand==handoff.consumer_operand:
                body.add(f'        var rr{load.operand} = handoff_real')
                body.add(f'        var ii{load.operand} = handoff_imag')
                body.add('        if compact_it != 0:')
                sub=Emitter()
                _emit_load_loop(sub,plan=plan,load=load,offset_expr=offset,width=4)
                for line in sub.lines:
                    body.add('    '+line.replace(f'var rr{load.operand} =',f'rr{load.operand} =')
                        .replace(f'var ii{load.operand} =',f'ii{load.operand} ='))
            else:
                _emit_load_loop(body,plan=plan,load=load,offset_expr=offset,width=4)
        stores={s.output:s for s in loop.stores}

        def output(k):
            tw=loop.twiddles.get(k)
            if tw is not None:
                _emit_twiddle_loop(body,output=k,
                    table_offset_expr=f'({tw.table_offset} + outer_it * 4)',width=4)
            scale=loop.scales.get(k)
            if scale is not None:
                body.add(f'        or{k} *= {_f32(scale)}')
                body.add(f'        oi{k} *= {_f32(scale)}')
            sub=Emitter()
            _emit_store_loop(sub,plan=plan,output=k,store=stores[k],residue=0,chunk_offset=0,width=4)
            if forward and producer and k==handoff.producer_output:
                body.add(f'        handoff_real = or{k}')
                body.add(f'        handoff_imag = oi{k}')
                body.add(f'        if compact_it != {handoff.producer_chunk}:')
                nest(body,sub,4)
            else:
                nest(body,sub,0)

        emit_butterfly(body,indent='        ',radix=stage.radix,inverse=stage.inverse,on_output=output)
        nest(e,body,16)
        e.add('                        compact_it += 1')
