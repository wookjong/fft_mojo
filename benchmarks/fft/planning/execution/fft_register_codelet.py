"""A proven SIMD vector edge forwarded across the far1/2 boundary."""
from dataclasses import dataclass
from planning.execution.distributed_phases import access_offsets

CODELET_MODES=('off','peel','forward1','loop-forward1','loop-forward-packed',
               'native4-control','native4-forward','native4-static-control','native4-static-forward','native4-compact-control','native4-compact-forward','native4-latch-forward')

@dataclass(frozen=True)
class RegisterEdge:
    producer_chunks: tuple[int,...]
    producer_outputs: tuple[int,...]
    consumer_chunk: int
    consumer_operand: int
    offsets: tuple[int,...]
    width: int
    producer_width: int

def prove_edges(phase):
    edges=[]
    if phase.stage_ids!=(1,2):raise ValueError('register edge requires far1/2')
    for component in phase.components:
        producers=tuple(c for c in component.chunks if c.stage_id==1)
        consumers=tuple(c for c in component.chunks if c.stage_id==2)
        if not producers or not consumers:raise ValueError('incomplete codelet component')
        if len(producers)<2:raise ValueError('register pair needs two producer chunks')
        last=producers[-1]
        # Radix8's factorized emitter emits outputs 0,4,1,5,2,6,3,7.
        # Codegen independently checks this callback order before using carry.
        selected=producers[-2:];outputs=[];offsets=()
        for chunk in selected:
            if chunk.width!=2 or len(chunk.batch.outputs) not in (2,4,8):
                raise ValueError('initial codelet requires complete SIMD2 radix2/4/8 producers')
            out=chunk.batch.outputs[-1];output=len(chunk.batch.outputs)-1
            if out.output!=output or out.store.destination!='scratchpad' or out.store.buffer_name!='phase_scratch':
                raise ValueError('register producer is not the final local output')
            outputs.append(output);offsets+=access_offsets(out.store,chunk.width)
        if len(offsets)!=4 or len(set(offsets))!=4:raise ValueError('invalid register edge lanes')
        writers=[(i,o.output) for i,c in enumerate(producers) for o in c.batch.outputs
            if set(access_offsets(o.store,c.width))&set(offsets)]
        expected=list(zip((len(producers)-2,len(producers)-1),outputs))
        if writers!=expected:raise ValueError('register edge has overlapping producers')
        readers=[]
        for index,c in enumerate(consumers):
            for load in c.batch.loads:
                loaded=access_offsets(load,c.width)
                if load.source=='scratchpad' and set(loaded)&set(offsets):
                    if load.buffer_name!='phase_scratch' or c.width!=4 or loaded!=offsets:
                        raise ValueError('register edge has mixed/reordered consumer lanes')
                    readers.append((index,load.operand))
        if len(readers)!=1:raise ValueError('register edge requires exactly one complete consumer')
        index,operand=readers[0]
        edges.append(RegisterEdge((len(producers)-2,len(producers)-1),tuple(outputs),index,operand,offsets,4,2))
    return tuple(edges)


def prove_native_edges(phase):
    """One complete producer vector must equal one uniquely consumed vector."""
    if phase.stage_ids != (1,2):raise ValueError('native edge requires far1/2')
    edges=[]
    for component in phase.components:
        producers=tuple(c for c in component.chunks if c.stage_id==1)
        consumers=tuple(c for c in component.chunks if c.stage_id==2)
        if not producers or not consumers or any(c.width!=4 for c in producers+consumers):
            raise ValueError('native edge requires complete SIMD4 stages')
        last=producers[-1];out=last.batch.outputs[-1]
        if out.output!=len(last.batch.outputs)-1 or out.store.destination!='scratchpad' or out.store.buffer_name!='phase_scratch':
            raise ValueError('native edge requires the final private output')
        offsets=access_offsets(out.store,4)
        if len(offsets)!=4 or len(set(offsets))!=4:raise ValueError('incomplete native vector')
        writers=[(i,o.output) for i,c in enumerate(producers) for o in c.batch.outputs
                 if set(access_offsets(o.store,c.width))&set(offsets)]
        if writers!=[(len(producers)-1,out.output)]:raise ValueError('overlapping native producers')
        readers=[]
        for i,c in enumerate(consumers):
            for load in c.batch.loads:
                if load.source=='scratchpad' and set(access_offsets(load,4))&set(offsets):
                    if load.buffer_name!='phase_scratch' or access_offsets(load,4)!=offsets:
                        raise ValueError('mixed or reordered native lanes')
                    readers.append((i,load.operand))
        if len(readers)!=1:raise ValueError('native vector must have one consumer')
        i,operand=readers[0]
        edges.append(RegisterEdge((len(producers)-1,),(out.output,),i,operand,offsets,4,4))
    return tuple(edges)
