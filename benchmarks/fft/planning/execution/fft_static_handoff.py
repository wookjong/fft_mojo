"""Prove a native-vector handoff after the physical SPAD allocation is bound."""
from dataclasses import dataclass,replace
from planning.execution.distributed_phases import access_offsets

STATIC_MODES=('native4-static-control','native4-static-forward','native4-compact-control','native4-compact-forward','native4-latch-forward')

@dataclass(frozen=True)
class StaticHandoff:
    producer_chunk: int
    producer_output: int
    consumer_chunk: int
    consumer_operand: int
    consumer_order: tuple[int,...]
    offsets: tuple[int,...]
    checked_slot_workers: int
    crossed_consumers: int
    physical_read_write_disjoint: bool = True
    adjacent: bool = True

def physical_access(node,private,storage,access,width,*,leaf_slot,worker,write=False):
    """Resolve float32 addresses in the shared arena, including both planes."""
    kind=access.destination if write else access.source
    offsets=access_offsets(access,width)
    if len(offsets)!=width or len(set(offsets))!=width:
        raise ValueError('static handoff requires complete, unique SIMD lanes')
    if kind=='output':
        # Consumer chunks of this leaf use the same out_batch_base expression.
        # Keep that common affine term symbolic; the exact lowered offsets
        # decide overlap. CXL output and local SPAD are distinct address spaces.
        if not write:raise ValueError('unexpected output read')
        return frozenset(('global_output',leaf_slot,plane,o) for plane in (0,1) for o in offsets)
    if kind=='scratchpad':
        if access.buffer_name!='phase_scratch':raise ValueError('unknown private scratch bank')
        if any(not 0<=o<private.phase.local_elements for o in offsets):
            raise ValueError('private scratch address out of bounds')
        base=private.scratch_offset+worker*2*private.phase.local_elements
        stride=private.phase.local_elements
    elif kind=='workspace':
        base=storage.bank_offset(access.buffer_name)+leaf_slot*storage.slot_elements
        stride=node.length
        if any(not 0<=o<stride for o in offsets):raise ValueError('resident address out of bounds')
    else:
        raise ValueError('static reordering requires fully resolved resident/private addresses')
    return frozenset(('arena',base+plane*stride+o) for plane in (0,1) for o in offsets)

def chunk_accesses(node,private,storage,chunk,*,leaf_slot,worker):
    reads=set();writes=set()
    for load in chunk.batch.loads:
        reads.update(physical_access(node,private,storage,load,chunk.width,leaf_slot=leaf_slot,worker=worker))
    for output in chunk.batch.outputs:
        writes.update(physical_access(node,private,storage,output.store,chunk.width,
            leaf_slot=leaf_slot,worker=worker,write=True))
    return reads,writes

def bind_static_handoffs(node,private,storage):
    if private.codelet_mode not in STATIC_MODES:return private
    from planning.execution.fft_register_codelet import prove_native_edges
    if private.codelets!=prove_native_edges(private.phase):raise ValueError('stale static edge proof')
    if storage is None or storage.slot_elements!=2*node.length:
        raise ValueError('static handoff needs a bound, canonical resident slot')
    if len(private.codelets)!=len(private.phase.components):raise ValueError('missing static edge proof')
    proofs=[]
    # All private worker slots must be physically separate from resident banks.
    # This is deliberately checked after phase-packed/live-bank rebinding.
    start=private.scratch_offset;end=start+private.scratch_elements
    for name,base in storage.bank_base:
        if start<base+storage.slots*storage.slot_elements and base<end:
            raise ValueError('physical private/resident SPAD alias: '+name)
    for component,edge in zip(private.phase.components,private.codelets):
        producers=tuple(c for c in component.chunks if c.stage_id==1)
        consumers=tuple(c for c in component.chunks if c.stage_id==2)
        if edge.producer_width!=4 or edge.width!=4 or edge.producer_chunks!=(len(producers)-1,):
            raise ValueError('static handoff needs the last native SIMD4 producer')
        selected=edge.consumer_chunk
        if not 0<=selected<len(consumers):raise ValueError('invalid static consumer')
        for slot in range(storage.slots):
            for worker in range(private.workers):
                accesses=[chunk_accesses(node,private,storage,c,leaf_slot=slot,worker=worker) for c in consumers]
                r,w=accesses[selected]
                for j in range(selected):
                    rj,wj=accesses[j]
                    if w&(rj|wj) or r&wj:
                        raise ValueError('consumer reordering has a physical RAW/WAR/WAW dependency')
                output=producers[-1].batch.outputs[edge.producer_outputs[0]].store
                load=consumers[selected].batch.loads[edge.consumer_operand]
                src=physical_access(node,private,storage,output,4,leaf_slot=slot,worker=worker,write=True)
                dst=physical_access(node,private,storage,load,4,leaf_slot=slot,worker=worker)
                if src!=dst:raise ValueError('physical handoff edge is not identical')
                if any(src&wj for _,wj in accesses):raise ValueError('consumer overwrites the register edge')
        order=(selected,)+tuple(i for i in range(len(consumers)) if i!=selected)
        proofs.append(StaticHandoff(len(producers)-1,edge.producer_outputs[0],selected,
            edge.consumer_operand,order,edge.offsets,storage.slots*private.workers,selected))
    return replace(private,handoffs=tuple(proofs))
