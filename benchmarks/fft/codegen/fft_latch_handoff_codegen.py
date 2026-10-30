"""Consume the carried vector and replace it in each loop latch.

Both stages have one shared butterfly body. The producer exits before the
last edge store, and the consumer exits before loading another edge. These
are the ordinary loop exit tests, not extra predicates inside the body.
"""
from dataclasses import replace
from codegen.common import Emitter
from codegen.fft_loop_ranges import build_full_loop
from codegen.fft_codegen import (_emit_load_loop, _emit_store_loop,
    _emit_twiddle_loop, _residue_base, _f32)
from codegen.fft_butterflies import emit_butterfly


def nest(e,body,spaces):
    for line in body.lines:e.add(' '*spaces+line if line else '')


def emit_latch_component(e,*,node,local,component,handoff,mode,twiddle_table):
    producers=tuple(c for c in component.chunks if c.stage_id==1)
    consumers=tuple(c for c in component.chunks if c.stage_id==2)
    left=build_full_loop(producers,twiddle_table)
    right=build_full_loop(consumers,twiddle_table)
    count=len(consumers)
    if (mode!='native4-latch-forward' or not handoff.adjacent
        or handoff.producer_chunk!=len(producers)-1 or handoff.consumer_chunk!=count-1
        or count&(count-1) or left.loop.period!=1 or right.loop.period!=1):
        raise ValueError('latch handoff requires regular loops and a proved last-to-last edge')
    if handoff.consumer_order!=(count-1,)+tuple(range(count-1)):
        raise ValueError('latch handoff order differs from physical proof')
    plan=replace(local,simd_lanes=4)
    e.add('                var handoff_real: SIMD[DType.float32, 4]')
    e.add('                var handoff_imag: SIMD[DType.float32, 4]')
    for producer,full,chunks in ((True,left,producers),(False,right,consumers)):
        loop=full.loop
        stage=node.stages[1 if producer else 2]
        e.add('                if True:')
        e.add('                    var latch_it = 0')
        e.add('                    while True:')
        expr='latch_it' if producer else f'(latch_it + {count-1}) % {count}'
        e.add(f'                        var outer_it = {expr}')
        body=Emitter()
        for load in loop.loads:
            if not producer and load.operand==handoff.consumer_operand:
                body.add(f'        var rr{load.operand} = handoff_real')
                body.add(f'        var ii{load.operand} = handoff_imag')
            else:
                base=_residue_base(load.bases,0,0)
                _emit_load_loop(body,plan=plan,load=load,
                    offset_expr=f'({base} + outer_it * {load.stride})',width=4)
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
            if producer and k==handoff.producer_output:
                body.add(f'        handoff_real = or{k}')
                body.add(f'        handoff_imag = oi{k}')
            else:
                _emit_store_loop(body,plan=plan,output=k,store=stores[k],residue=0,chunk_offset=0,width=4)

        emit_butterfly(body,indent='        ',radix=stage.radix,inverse=stage.inverse,on_output=output)
        # Keep the body output and address tied to this iteration until the
        # producer's deferred store. Only the integer traversal index advances.
        body.add('        latch_it += 1')
        body.add(f'        if latch_it == {len(chunks)}:')
        body.add('            break')
        if producer:
            _emit_store_loop(body,plan=plan,output=handoff.producer_output,
                store=stores[handoff.producer_output],residue=0,chunk_offset=0,width=4)
        else:
            # logical next index is 0,1,...,count-2. Exit above prevents any
            # speculative final read of the omitted producer SPAD edge.
            load=next(l for l in loop.loads if l.operand==handoff.consumer_operand)
            base=_residue_base(load.bases,0,0)
            sub=Emitter()
            _emit_load_loop(sub,plan=plan,load=load,
                offset_expr=f'({base} + (latch_it - 1) * {load.stride})',width=4)
            for line in sub.lines:
                line=line.replace(f'var rr{load.operand} =','handoff_real =')
                line=line.replace(f'var ii{load.operand} =','handoff_imag =')
                body.add(line)
        nest(e,body,16)
