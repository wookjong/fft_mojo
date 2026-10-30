"""Typed scalar FFT IR and cross-leaf boundary jobs; no emitted code yet.

Always consume canonical stage offsets BEFORE legacy strided rebinding.
Global segments describe both SoA planes; local accesses explicitly name a plane.
"""
from dataclasses import dataclass
from math import isfinite, prod
from planning.core.fft_plan_core import AddressMapping
from planning.execution.fft_leaf_ownership import LeafOwnershipPlan, _integer


@dataclass(frozen=True)
class ScalarInputPlan:
    operand: int
    source: str
    buffer_name: str | None
    offset: int


@dataclass(frozen=True)
class ScalarOutputPlan:
    output: int
    destination: str
    buffer_name: str | None
    offset: int
    twiddle: tuple[float, float] | None
    scale: float | None


@dataclass(frozen=True)
class ScalarButterflyPlan:
    butterfly_id: int
    original_batch: int
    original_lane: int
    loads: tuple[ScalarInputPlan, ...]
    outputs: tuple[ScalarOutputPlan, ...]


@dataclass(frozen=True)
class ScalarStagePlan:
    stage_id: int
    length: int
    radix: int
    inverse: bool
    butterflies: tuple[ScalarButterflyPlan, ...]


def extract_scalar_stages(kernel):
    """Validate and scalarize ordinary distributed, canonical multi-stage leaves."""
    d = kernel.distributed
    if d is None or d.leaf_units != 1 or d.fuse_stages != 1:
        raise ValueError('scalar boundary IR requires distributed unfused single-unit leaves')
    if kernel.simd_lanes != 8 or len(kernel.stages) < 2:
        raise ValueError('scalar boundary IR requires SIMD-8 and at least two stages')
    if (kernel.large_twiddle is not None or kernel.boundary_twiddle is not None
            or kernel.persistent is not None or kernel.cooperation is not None
            or kernel.launch_mapping is not None or kernel.scratchpad_buffers
            or kernel.leaf_storage is not None):
        raise ValueError('scalar boundary IR requires ordinary canonical distributed leaves')
    if (kernel.input_mapping != AddressMapping.contiguous(kernel.length)
            or kernel.output_mapping != AddressMapping.contiguous(kernel.length)):
        raise ValueError('extract canonical offsets before boundary rebinding')
    bank_names = tuple(b.name for b in kernel.workspace_buffers)
    if kernel.total_uthreads != d.launch_uthreads:
        raise ValueError('leaf launch extent disagrees with its physical distribution')
    if prod(s.radix for s in kernel.stages) != kernel.length:
        raise ValueError('stage radices do not factor the leaf length')
    result = []
    previous_bank = None
    for index, stage in enumerate(kernel.stages):
        if (stage.stage_id != index or stage.inverse != kernel.inverse or stage.worker_batches is not None
                or stage.persistent_vector_batches is not None
                or stage.persistent_scalar_batches is not None):
            raise ValueError('stage IDs/worker partitions are not canonical')
        _integer('radix',stage.radix,2)
        jobs = []
        batch_ids = set()
        for batch in stage.batches:
            if batch.batch_id in batch_ids or not 1 <= batch.valid_lanes <= kernel.simd_lanes:
                raise ValueError('invalid SIMD batch ID/valid lanes')
            batch_ids.add(batch.batch_id)
            if ([x.operand for x in batch.loads] != list(range(stage.radix))
                    or [x.output for x in batch.outputs] != list(range(stage.radix))):
                raise ValueError('radix operands/outputs must be complete and ordered')
            for lane in range(batch.valid_lanes):
                loads = []
                for x in batch.loads:
                    expected = 'input' if index == 0 else 'workspace'
                    if (x.source != expected or (expected == 'workspace' and x.buffer_name != previous_bank)
                            or (expected == 'input' and x.buffer_name is not None)):
                        raise ValueError('stage input source/bank is not canonical')
                    if x.mode == 'vector' and x.base_offset is not None:
                        offset = x.base_offset + lane
                    elif x.mode == 'scalar_pack' and len(x.packed_lane_offsets) > lane:
                        offset = x.packed_lane_offsets[lane]
                    else:
                        raise ValueError('invalid original load')
                    _integer('load offset', offset)
                    loads.append(ScalarInputPlan(x.operand,x.source,x.buffer_name,offset))
                outputs = []
                for x in batch.outputs:
                    s = x.store
                    expected = 'output' if index == len(kernel.stages)-1 else 'workspace'
                    if (s.destination != expected or x.large_twiddle
                            or (expected == 'workspace' and s.buffer_name not in bank_names)
                            or (expected == 'output' and s.buffer_name is not None)):
                        raise ValueError('stage output destination/bank is not canonical')
                    if s.mode == 'vector' and s.base_offset is not None:
                        offset = s.base_offset + lane
                    elif s.mode == 'scalar_lanes' and len(s.lane_offsets) > lane:
                        offset = s.lane_offsets[lane]
                    else:
                        raise ValueError('invalid original store')
                    _integer('store offset', offset)
                    if x.twiddle is not None and (len(x.twiddle.real) <= lane or len(x.twiddle.imag) <= lane):
                        raise ValueError('incomplete original twiddle lanes')
                    tw = None if x.twiddle is None else (x.twiddle.real[lane],x.twiddle.imag[lane])
                    if (tw is not None and not all(isfinite(v) for v in tw)) or (x.scale is not None and not isfinite(x.scale)):
                        raise ValueError('nonfinite coefficient/scale in scalar IR')
                    outputs.append(ScalarOutputPlan(x.output,s.destination,s.buffer_name,offset,tw,x.scale))
                jobs.append(ScalarButterflyPlan(len(jobs),batch.batch_id,lane,tuple(loads),tuple(outputs)))
        reads = [x.offset for j in jobs for x in j.loads]
        writes = [x.offset for j in jobs for x in j.outputs]
        if sorted(reads) != list(range(kernel.length)) or sorted(writes) != list(range(kernel.length)):
            raise ValueError('scalar stage offsets must cover the leaf exactly once')
        write_banks={x.buffer_name for j in jobs for x in j.outputs}
        if len(write_banks)!=1:
            raise ValueError('ordinary stage must write a single resident bank')
        next_bank=next(iter(write_banks))
        if index>0 and index<len(kernel.stages)-1 and next_bank==previous_bank:
            raise ValueError('ordinary stages require separate read/write banks')
        previous_bank=next_bank
        result.append(ScalarStagePlan(stage.stage_id,kernel.length,stage.radix,stage.inverse,tuple(jobs)))
    return tuple(result)


@dataclass(frozen=True)
class BoundarySIMDLane:
    vector_lane: int
    leaf_lane: int
    butterfly: ScalarButterflyPlan


@dataclass(frozen=True)
class BoundarySIMDJob:
    job_id: int
    worker: int
    lanes: tuple[BoundarySIMDLane, ...]


@dataclass(frozen=True)
class GlobalSegmentPlan:
    direction: str
    value_id: int
    butterfly_id: int
    lane_start: int
    width: int
    base_offset: int

    @property
    def offsets(self):
        return tuple(range(self.base_offset,self.base_offset+self.width))


@dataclass(frozen=True)
class LocalLaneAccessPlan:
    direction: str
    vector_lane: int
    value_id: int
    replica: int
    bank_name: str
    plane: int
    element: int
    resident_offset: int


@dataclass(frozen=True)
class ResolvedBoundaryJob:
    tile: int
    unit: int
    worker: int
    job_id: int
    global_segments: tuple[GlobalSegmentPlan, ...]
    local_accesses: tuple[LocalLaneAccessPlan, ...]


@dataclass(frozen=True)
class BoundarySIMDPlan:
    boundary: str
    a: int
    b: int
    total_batch: int
    ownership: LeafOwnershipPlan
    stage: ScalarStagePlan
    jobs: tuple[BoundarySIMDJob, ...]
    arithmetic_width: int
    butterflies_per_vector: int
    inverse_scale: float | None
    physical_allocation: object | None = None
    physical_node: int = -1

    @property
    def n(self):
        return self.a*self.b

    def _live(self, window_base, live_batch):
        _integer('window_base',window_base)
        if window_base >= self.total_batch:
            raise ValueError('window starts outside the total batch')
        live = min(self.ownership.window_batch,self.total_batch-window_base) if live_batch is None else live_batch
        self.ownership.live_tiles(live)
        if window_base+live > self.total_batch:
            raise ValueError('live window exceeds the total batch')
        return live

    def resolve_job(self, tile, job_id, *, window_base=0, live_batch=None):
        live = self._live(window_base,live_batch)
        _integer('job_id',job_id)
        if job_id >= len(self.jobs):
            raise ValueError('boundary job is outside the stage')
        replicas = self.ownership.replicas_for_tile(tile,live_batch=live)
        first = self.ownership.resolve(replicas[0],live_batch=live)
        job = self.jobs[job_id]
        c = self.ownership.group_leaves
        segments = []
        for delta in range(self.butterflies_per_vector):
            lane_start = delta*c
            original = job.lanes[lane_start].butterfly
            values = original.loads if self.boundary == 'pre' else original.outputs
            for value in values:
                stride = self.a if self.boundary == 'pre' else self.b
                base = (window_base+first.batch)*self.n + value.offset*stride + first.leaf
                value_id = value.operand if self.boundary == 'pre' else value.output
                segments.append(GlobalSegmentPlan('load' if self.boundary=='pre' else 'store',
                    value_id,original.butterfly_id,lane_start,c,base))
        local = []
        for lane in job.lanes:
            replica = replicas[lane.leaf_lane]
            values = lane.butterfly.outputs if self.boundary=='pre' else lane.butterfly.loads
            for value in values:
                value_id = value.output if self.boundary=='pre' else value.operand
                for plane in (0,1):
                    offset = self.ownership.resident_index(replica,value.offset,
                        bank=value.buffer_name,plane=plane,live_batch=live)
                    if self.physical_allocation is not None:
                        offset=self.physical_allocation.resident_index(self.physical_node,self.ownership,
                            replica,value.offset,bank=value.buffer_name,plane=plane,live_batch=live)
                    local.append(LocalLaneAccessPlan('store' if self.boundary=='pre' else 'load',
                        lane.vector_lane,value_id,replica,value.buffer_name,plane,value.offset,offset))
        return ResolvedBoundaryJob(tile,first.unit,job.worker,job_id,tuple(segments),tuple(local))

    def iter_resolved_jobs(self, *, window_base=0, live_batch=None):
        live = self._live(window_base,live_batch)
        for tile in range(self.ownership.live_tiles(live)):
            for job in self.jobs:
                yield self.resolve_job(tile,job.job_id,window_base=window_base,live_batch=live)


def make_boundary_simd_plan(kernel, *, boundary, a, b, total_batch, group_leaves, window_batch=1, defer_leaf_budget=False):
    """Build an opt-in executable planning IR, without changing legacy kernels.

    `defer_leaf_budget`: the resident estimate below sums every declared bank
    of the ownership. A compact physical allocator (live-banks/phase-packed)
    later drops unused banks, reuses arena across phases and checks the real
    reservation against the planner and concurrent SPAD budgets
    (task_fusion.make_task_fusion_plan). When that allocator will run, the
    caller passes True and this declared-bank estimate is not a rejection."""
    if boundary not in ('pre','post'):
        raise ValueError('boundary must be pre or post')
    for name,value in (('a',a),('b',b),('total_batch',total_batch),('window_batch',window_batch)):
        _integer(name,value,1)
        if name in ('a','b') and value & (value-1):
            raise ValueError('initial boundary SIMD plans require power-of-two a and b')
    if group_leaves not in ((4,) if boundary=='pre' else (2,4)) or isinstance(group_leaves,bool):
        raise ValueError('PRE requires C4; POST requires C2 or C4')
    scalar = extract_scalar_stages(kernel)
    length,m = (b,a) if boundary=='pre' else (a,b)
    if kernel.length != length or kernel.host.total_elems != total_batch*a*b:
        raise ValueError('leaf length/replicas disagree with the recursive shape')
    if not 1 <= window_batch <= total_batch:
        raise ValueError('window must fit the total batch')
    distribution = kernel.distributed
    if distribution.workers_per_unit != 8:
        raise ValueError('grouped boundary plans require eight worker slots per unit')
    ownership = LeafOwnershipPlan(m,length,group_leaves,window_batch,
        distribution.num_units,distribution.workers_per_unit,
        tuple(x.name for x in kernel.workspace_buffers))
    if not defer_leaf_budget and ownership.arena_bytes > distribution.scratchpad_budget_bytes:
        raise ValueError('grouped resident banks exceed the leaf scratchpad budget')
    stage = scalar[0] if boundary=='pre' else scalar[-1]
    if any(x.scale is not None for s in scalar for j in s.butterflies for x in j.outputs):
        raise ValueError('root normalization must be separate from canonical scalar IR')
    v = 4
    d = v//group_leaves
    if len(stage.butterflies) % d:
        raise ValueError('boundary butterfly count does not fill CxD SIMD jobs')
    jobs = []
    for job_id in range(len(stage.butterflies)//d):
        lanes = tuple(BoundarySIMDLane(delta*group_leaves+ell,ell,
            stage.butterflies[job_id*d+delta])
            for delta in range(d) for ell in range(group_leaves))
        jobs.append(BoundarySIMDJob(job_id,job_id%ownership.workers_per_unit,lanes))
    scale = 1.0/(a*b) if boundary=='post' and kernel.inverse else None
    return BoundarySIMDPlan(boundary,a,b,total_batch,ownership,stage,tuple(jobs),v,d,scale)
