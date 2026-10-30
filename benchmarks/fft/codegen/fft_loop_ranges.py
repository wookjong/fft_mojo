"""Render intervals of one complete loop plan without re-lowering a subset."""
from dataclasses import dataclass
from codegen.common import Emitter
from codegen.lowering import try_build_loop_stage
from codegen.fft_codegen import _emit_loop_chunk,_residue_pattern

def nest(e,sub,spaces):
    for line in sub.lines:e.add(' '*spaces+line if line else '')

@dataclass(frozen=True)
class FullLoop:
    loop: object
    count: int
    width: int

def build_full_loop(chunks,twiddle_table):
    if not chunks or any(c.width!=4 or c.batch.valid_lanes!=4 for c in chunks):
        raise ValueError('static ranges require complete SIMD4 chunks')
    loop=try_build_loop_stage(tuple(c.batch for c in chunks),simd_lanes=4,
        min_full_batches=2,twiddle_table=twiddle_table)
    if loop is None or loop.tail_batch is not None or loop.outer_iters*loop.period!=len(chunks):
        raise ValueError('static ranges need a complete regular loop plan')
    arrays=[l.bases for l in loop.loads]
    for s in loop.stores:arrays.extend((s.bases,) if s.mode=='vector' else s.lane_bases)
    if any(_residue_pattern(a) is None for a in arrays):raise ValueError('non-affine static range')
    return FullLoop(loop,len(chunks),4)

def emit_range(e,*,plan,stage,full,begin,end):
    """General period-aware slice; partial blocks keep bounded residue loops."""
    if not 0<=begin<=end<=full.count:raise ValueError('loop slice out of bounds')
    if begin==end:return
    period=full.loop.period
    def block(first,last,outer_first,outer_last):
        body=Emitter()
        if period==1:
            _emit_loop_chunk(body,plan=plan,stage=stage,loop_plan=full.loop,residue=0,chunk_offset=0,width=4)
        else:
            body.add(f'        var residue_it = {first}')
            body.add(f'        while residue_it < {last}:')
            sub=Emitter();_emit_loop_chunk(sub,plan=plan,stage=stage,loop_plan=full.loop,
                residue='residue_it',chunk_offset=0,width=4)
            nest(body,sub,4);body.add('            residue_it += 1')
        e.add('                if True:')
        e.add(f'                    var outer_it = {outer_first}')
        e.add(f'                    while outer_it < {outer_last}:')
        nest(e,body,16)
        e.add('                        outer_it += 1')
    first_outer,first_r=divmod(begin,period)
    last_outer,last_r=divmod(end,period)
    if first_outer==last_outer:
        block(first_r,last_r,first_outer,first_outer+1);return
    if first_r:
        block(first_r,period,first_outer,first_outer+1);first_outer+=1
    if first_outer<last_outer:block(0,period,first_outer,last_outer)
    if last_r:block(0,last_r,last_outer,last_outer+1)

def emit_at(e,*,plan,stage,full,index,load_values=None,store_values=None):
    if not 0<=index<full.count:raise ValueError('static chunk out of bounds')
    outer,residue=divmod(index,full.loop.period)
    e.add('                if True:')
    e.add(f'                    var outer_it = {outer}')
    body=Emitter();_emit_loop_chunk(body,plan=plan,stage=stage,loop_plan=full.loop,
        residue=residue,chunk_offset=0,width=4,load_values=load_values,store_values=store_values)
    nest(e,body,12)
