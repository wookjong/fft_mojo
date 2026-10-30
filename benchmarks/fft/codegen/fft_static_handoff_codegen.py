"""Fixed producer/consumer blocks, with no per-iteration handoff selection."""
from dataclasses import replace
from codegen.fft_loop_ranges import build_full_loop,emit_range,emit_at

def emit_static_component(e,*,node,local,component,handoff,mode,twiddle_table):
    producers=tuple(c for c in component.chunks if c.stage_id==1)
    consumers=tuple(c for c in component.chunks if c.stage_id==2)
    # Allocate all twiddle rows once, in the original stage/chunk order.
    left=build_full_loop(producers,twiddle_table)
    right=build_full_loop(consumers,twiddle_table)
    if not handoff.adjacent or handoff.producer_chunk!=len(producers)-1:
        raise ValueError('static codegen requires a proved adjacent handoff')
    selected=handoff.consumer_chunk
    expected=(selected,)+tuple(i for i in range(len(consumers)) if i!=selected)
    if handoff.consumer_order!=expected:raise ValueError('unproved static schedule')
    forward=mode=='native4-static-forward'
    plan=replace(local,simd_lanes=4)
    emit_range(e,plan=plan,stage=node.stages[1],full=left,begin=0,end=handoff.producer_chunk)
    if forward:
        # Assigned once in a guaranteed fixed block; neither a zero value nor
        # a loop-carried phi is needed. Separate blocks scope butterfly temps.
        e.add('                var handoff_real: SIMD[DType.float32, 4]')
        e.add('                var handoff_imag: SIMD[DType.float32, 4]')
    emit_at(e,plan=plan,stage=node.stages[1],full=left,index=handoff.producer_chunk,
        store_values={handoff.producer_output:('handoff_real','handoff_imag')} if forward else None)
    emit_at(e,plan=plan,stage=node.stages[2],full=right,index=selected,
        load_values={handoff.consumer_operand:('handoff_real','handoff_imag')} if forward else None)
    emit_range(e,plan=plan,stage=node.stages[2],full=right,begin=0,end=selected)
    emit_range(e,plan=plan,stage=node.stages[2],full=right,begin=selected+1,end=len(consumers))
