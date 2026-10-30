"""Peel one producer/consumer pair and optionally forward one SIMD4 value."""
from codegen.common import Emitter
from codegen.fft_codegen import _emit_load,_emit_output,_emit_twiddle,_f32
from codegen.fft_butterflies import emit_butterfly
from codegen.fft_distributed_codegen import _emit_local_stage

def _batch(e,*,plan,stage,chunk,edge,carry_index,forward):
    b=chunk.batch;callbacks=[]
    producer=carry_index is not None
    for load in b.loads:
        if forward and not producer and load.operand==edge.consumer_operand:
            for part,prefix in (('real','rr'),('imag','ii')):
                values=', '.join(f'codelet_{i}_{part}[{lane}]' for i in range(2) for lane in range(2))
                e.add(f'        var {prefix}{load.operand} = SIMD[DType.float32, 4]({values})')
        else:_emit_load(e,plan=plan,load=load,width=chunk.width)
    def output(k):
        callbacks.append(k);out=b.outputs[k]
        if forward and producer and k==edge.producer_outputs[carry_index]:
            if out.large_twiddle:raise ValueError('codelet large twiddle not supported')
            if out.twiddle is not None:_emit_twiddle(e,output=k,twiddle=out.twiddle,width=chunk.width)
            if out.scale is not None:
                e.add(f'        or{k} *= {_f32(out.scale)}');e.add(f'        oi{k} *= {_f32(out.scale)}')
            e.add(f'        codelet_{carry_index}_real = or{k}');e.add(f'        codelet_{carry_index}_imag = oi{k}')
        else:_emit_output(e,plan=plan,output_plan=out,width=chunk.width)
    emit_butterfly(e,indent='        ',radix=stage.radix,inverse=stage.inverse,on_output=output)
    if producer and callbacks[-1]!=edge.producer_outputs[carry_index]:raise ValueError('carry output is not last in butterfly schedule')

def emit_component(e,*,node,local,component,edge,mode,twiddle_table):
    producers=tuple(c for c in component.chunks if c.stage_id==1)
    consumers=tuple(c for c in component.chunks if c.stage_id==2)
    forward=mode=='forward1'
    if forward:
        for i in range(2):
            e.add(f'                var codelet_{i}_real = SIMD[DType.float32, 2](0)')
            e.add(f'                var codelet_{i}_imag = SIMD[DType.float32, 2](0)')
    if producers[:-2]:
        _emit_local_stage(e,plan=local,stage=node.stages[1],chunks=producers[:-2],twiddle_table=twiddle_table)
    for chunk,carry_index in ((producers[-2],0),(producers[-1],1),(consumers[edge.consumer_chunk],None)):
        body=Emitter();_batch(body,plan=local,stage=node.stages[chunk.stage_id],chunk=chunk,
            edge=edge,carry_index=carry_index,forward=forward)
        e.add('                if True:')
        for line in body.lines:e.add(' '*12+line if line else '')
    rest=tuple(c for i,c in enumerate(consumers) if i!=edge.consumer_chunk)
    if rest:_emit_local_stage(e,plan=local,stage=node.stages[2],chunks=rest,twiddle_table=twiddle_table)
