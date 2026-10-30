"""Physical, phase-conservative SPAD allocation after effective FFT lowering.

All offsets/sizes are Float32 elements. Phase intervals include both endpoints:
resident input and private output in the same phase must never alias.
"""
from dataclasses import dataclass,replace
from collections import defaultdict
from planning.strategies.fft_plan_recursive import PhysicalTransposePlan
from planning.execution.fft_worker_private import node_phase_ids

SPAD_ALLOCATION_MODES=('declared','live-banks','phase-packed')
ALIGNMENT=8  # 32 bytes, independent of arithmetic SIMD width.


@dataclass(frozen=True)
class SpadRegion:
    key: str
    node: int
    kind: str
    bank: str
    elements: int
    reads: tuple[int,...]
    writes: tuple[int,...]
    offset: int=0
    worker_stride: int=0

    @property
    def first(self):return min(self.reads+self.writes)
    @property
    def last(self):return max(self.reads+self.writes)
    @property
    def end(self):return self.offset+self.elements


@dataclass(frozen=True)
class SpadPhase:
    name: str
    node: int
    stage_ids: tuple[int,...]


@dataclass(frozen=True)
class TaskSpadAllocation:
    policy: str
    phases: tuple[SpadPhase,...]
    regions: tuple[SpadRegion,...]
    unused_banks: tuple[tuple[int,str],...]
    arena_elements: int

    def __post_init__(self):
        if self.policy not in SPAD_ALLOCATION_MODES[1:]:raise ValueError('unknown compact allocation policy')
        if len({r.key for r in self.regions})!=len(self.regions):raise ValueError('duplicate region key')
        if self.arena_elements<1:raise ValueError('empty arena')
        for i,a in enumerate(self.regions):
            if a.elements<1 or a.offset<0 or a.offset%ALIGNMENT or a.end>self.arena_elements:
                raise ValueError('SPAD region alignment/bounds violation: '+a.key)
            if not a.writes or min(a.writes)>min(a.reads,default=min(a.writes)):
                raise ValueError('read precedes initial write: '+a.key)
            if a.kind=='resident' and a.reads and min(a.writes)>=min(a.reads):
                raise ValueError('resident read/write cannot initialize in one parallel phase')
            if not 0<=a.first<=a.last<len(self.phases):raise ValueError('invalid region lifetime')
            if a.kind=='private' and (a.worker_stride<1 or a.elements!=8*a.worker_stride):
                raise ValueError('invalid worker-private extent')
            for b in self.regions[:i]:
                live=a.first<=b.last and b.first<=a.last
                overlap=a.offset<b.end and b.offset<a.end
                if live and overlap:raise ValueError('simultaneously live SPAD regions overlap: '+a.key+'/'+b.key)

    def region(self,node,kind,bank=''):
        for region in self.regions:
            if (region.node,region.kind,region.bank)==(node,kind,bank):return region
        raise ValueError(f'no physical SPAD binding: {node}/{kind}/{bank}')

    def resident_index(self,node,owner,replica,element,*,bank,plane,live_batch=None):
        # Validate logical coordinates with the canonical resolver, then bind
        # its owner/slot to a physical bank offset. No window-global Q here.
        owner.resident_index(replica,element,bank=bank,plane=plane,live_batch=live_batch)
        slot=owner.resolve(replica,live_batch=live_batch).leaf_slot
        region=self.region(node,'resident',bank)
        value=region.offset+slot*2*owner.leaf_length+plane*owner.leaf_length+element
        if not region.offset<=value<region.end:raise ValueError('resident binding out of bounds')
        return value


def collect_regions(nodes,resident,private,semantic_ids,common_slots):
    """Consume the emitted boundary adapter and lowered private chunks."""
    phases=[];regions=[];unused=[]
    for index,node in enumerate(nodes):
        semantic=semantic_ids[index]
        if isinstance(node,PhysicalTransposePlan):
            p=len(phases);phases.append(SpadPhase(f'node_{semantic}_transpose',index,()))
            regions.append(SpadRegion(f'transpose:{index}',index,'transpose','',
                common_slots*node.scratchpad_elements,(p,),(p,)))
            continue
        storage=resident[index]
        if storage is None:raise ValueError('compact allocation requires resident leaves')
        used=defaultdict(lambda:[set(),set()])
        proof=private[index]
        for ids in node_phase_ids(node,proof):
            p=len(phases);phases.append(SpadPhase(f'node_{semantic}_stage_'+'_'.join(map(str,ids)),index,ids))
            special=node.boundary_simd
            if len(ids)==1 and special is not None and special.stage.stage_id==ids[0]:
                reads={a.buffer_name for bf in special.stage.butterflies for a in bf.loads if a.source=='workspace'}
                writes={a.buffer_name for bf in special.stage.butterflies for a in bf.outputs if a.destination=='workspace'}
            else:
                batches=([c.batch for component in proof.phase.components for c in component.chunks]
                    if len(ids)>1 else node.stages[ids[0]].batches)
                reads={a.buffer_name for batch in batches for a in batch.loads if a.source=='workspace'}
                writes={a.store.buffer_name for batch in batches for a in batch.outputs if a.store.destination=='workspace'}
            for name in reads:used[name][0].add(p)
            for name in writes:used[name][1].add(p)
            if len(ids)>1:
                regions.append(SpadRegion(f'private:{index}',index,'private','',proof.scratch_elements,
                    (p,),(p,),worker_stride=2*proof.phase.local_elements))
        declared=tuple(name for name,_ in storage.bank_base)
        if set(used)-set(declared):raise ValueError('physical phase refers to undeclared bank')
        for name in declared:
            if name not in used:unused.append((index,name));continue
            reads,writes=used[name]
            regions.append(SpadRegion(f'bank:{index}:{name}',index,'resident',name,
                storage.slots*storage.slot_elements,tuple(sorted(reads)),tuple(sorted(writes))))
    return tuple(phases),tuple(regions),tuple(unused)


def _align(value):return -(-value//ALIGNMENT)*ALIGNMENT


def allocate_spad(nodes,resident,private,semantic_ids,common_slots,policy):
    phases,regions,unused=collect_regions(nodes,resident,private,semantic_ids,common_slots)
    if policy=='live-banks':
        sizes=defaultdict(int)
        for r in regions:
            if r.kind=='resident':sizes[r.node]+=r.elements
        base=max([0]+list(sizes.values())+[r.elements for r in regions if r.kind=='transpose'])
        cursors=defaultdict(int);placed=[]
        for r in regions:
            offset=base if r.kind=='private' else cursors[r.node] if r.kind=='resident' else 0
            placed.append(replace(r,offset=offset))
            if r.kind=='resident':cursors[r.node]=_align(offset+r.elements)
    elif policy=='phase-packed':
        placed=[]
        for r in sorted(regions,key=lambda r:(r.first,-r.elements,r.key)):
            conflicts=sorted((p for p in placed if r.first<=p.last and p.first<=r.last),key=lambda p:p.offset)
            offset=0
            for p in conflicts:
                if offset+r.elements<=p.offset:break
                offset=max(offset,_align(p.end))
            placed.append(replace(r,offset=offset))
    else:raise ValueError('unknown compact allocation policy')
    result=TaskSpadAllocation(policy,phases,tuple(placed),unused,max(p.end for p in placed))
    # The existing transpose emitter uses a common worker-slot arena at base0.
    # Refuse any future schedule needing a different base, rather than silently
    # emitting the old address. Current supported FS phases all satisfy this.
    if any(r.offset for r in result.regions if r.kind=='transpose'):
        raise ValueError('compact transpose requires arena base zero')
    rebound=[]
    for index,storage in enumerate(resident):
        if storage is None:rebound.append(None);continue
        bank_base=tuple((name,result.region(index,'resident',name).offset)
            for name,_ in storage.bank_base if (index,name) not in result.unused_banks)
        rebound.append(replace(storage,bank_base=bank_base))
    private=tuple(replace(p,scratch_offset=result.region(index,'private').offset) if p else None
        for index,p in enumerate(private))
    return result,tuple(rebound),private
