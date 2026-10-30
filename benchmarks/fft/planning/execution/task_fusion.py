"""Resource-checked grouping of recursive FFT nodes into synchronous tasks.

By default this changes execution ownership only. FFT nodes, radix math,
mappings, and global workspace boundaries remain the supplied planner's
decisions. `leaf_storage="spad"/"auto"` (default "cxl", unchanged
behavior) additionally places a leaf's own intermediate ping-pong banks in
the fused task's shared arena instead of CXL (see LeafResidentSpadPlan).
Only the *placement* of those middle-stage values changes; radix, boundary
mappings, twiddles and scale are untouched, and the first/last stage still
read/write the true global input/output arrays exactly as before.

The independent opt-in `boundary_fusion="post"` resolves an effective final
output mapping and removes the root POST from the physical schedule. The
semantic tree, radix math and all inter-stage completion barriers remain.
"""
from dataclasses import dataclass, replace
from itertools import product
from planning.core.fft_plan_core import LeafResidentSpadPlan
from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.execution.kernel_registration import registered_kernel_names
from planning.strategies.fft_plan_recursive import (
    FFTRecursiveNodePlan, FFTLeafPlan, PhysicalTransposePlan, flatten_recursive_node,
)

LEAF_STORAGE_MODES = ("cxl", "spad", "auto")


@dataclass(frozen=True)
class FFTTaskGroup:
    name: str
    node_ids: tuple[int, ...]
    functions: tuple[str, ...]


@dataclass(frozen=True)
class FFTTaskFusionPlan:
    supported: bool
    reason: str
    common_uthreads: int = 0
    slots_per_unit: int = 0
    arena_elements: int = 0
    params_bytes: int = 0
    reserved_bytes: int = 0
    groups: tuple[FFTTaskGroup, ...] = ()
    node_functions: tuple[tuple[str, ...], ...] = ()
    control_pool_ids: tuple[int, ...] = ()
    # One entry per flattened node (None for a transpose, or a leaf kept in
    # CXL); `arena_ref` is still empty here -- emit_fused_task_structs fills
    # it in once each leaf's own fused group (hence its own `comptime arena`)
    # is known. See LeafResidentSpadPlan / resolve_leaf_resident_storage.
    leaf_storage: tuple[LeafResidentSpadPlan | None, ...] = ()
    window_batch: int = 0
    total_batch: int = 0
    window_n: int = 0
    boundary_fusion: str = "off"
    semantic_node_ids: tuple[int, ...] = ()
    boundary_layout: str = "legacy"
    near_group: int = 1
    far_group: int = 1
    leaf_phase_policy: str = "stages"
    leaf_codegen_schedule: str = "existing"
    private_phases: tuple = ()
    far12_codelet: str = "off"
    spad_allocation: str = "declared"
    memory_allocation: object | None = None


def resolve_task_execution(plan, fusion):
    """Shared immutable boundary/buffer resolution for planner, emitter and checker."""
    from planning.execution.fft_boundary_fusion import make_boundary_fusion_plan
    execution = window_execution_plan(plan, fusion.window_batch) if fusion.window_batch else plan
    boundary = make_boundary_fusion_plan(execution, mode=fusion.boundary_fusion,
        boundary_layout=fusion.boundary_layout,near_group=fusion.near_group,far_group=fusion.far_group,
        defer_leaf_budget=fusion.spad_allocation != 'declared')
    if not boundary.supported:
        raise ValueError(boundary.reason)
    if fusion.memory_allocation is not None:
        nodes=[]
        for index,node in enumerate(boundary.nodes):
            if not isinstance(node,PhysicalTransposePlan) and node.boundary_simd is not None:
                special=replace(node.boundary_simd,physical_allocation=fusion.memory_allocation,
                    physical_node=index,total_batch=plan.batch)
                node=replace(node,boundary_simd=special)
            nodes.append(node)
        boundary=replace(boundary,nodes=tuple(nodes))
    return boundary


def window_execution_plan(plan, batch):
    """Resize execution extents only; callers retain full host buffers."""
    if not 1 <= batch <= plan.batch:
        raise ValueError('window batch must be between 1 and the full batch')
    full = plan.batch
    def resize(node):
        if isinstance(node, PhysicalTransposePlan):
            count = node.total_uthreads // full * batch
            return replace(node, replica_count=node.replica_count // full * batch,
                           total_uthreads=count, needs_round_split=count > node.max_uthread)
        if isinstance(node, FFTLeafPlan):
            k = node.kernel
            return replace(node, r=node.r // full * batch, kernel=replace(k,
                host=replace(k.host, total_elems=k.host.total_elems // full * batch),
                workspace_buffers=tuple(replace(b, elements=b.elements // full * batch)
                                        for b in k.workspace_buffers)))
        return replace(node, r=node.r // full * batch,
            pre_transpose=resize(node.pre_transpose), near_fft=resize(node.near_fft),
            middle_transpose=resize(node.middle_transpose), far_child=resize(node.far_child),
            post_transpose=resize(node.post_transpose))
    return replace(plan, batch=batch, root=resize(plan.root))


def resolve_leaf_resident_storage(node, *, leaf_storage):
    """Q/bank layout for one leaf's own LeafResidentSpadPlan, or (None, reason).

    Only `node.distributed.leaf_units == 1` makes this sound: one unit must
    own a whole leaf across every stage for "the same resident slot" to
    mean the same FFT replica throughout (see fft_distributed_codegen.
    emit_distributed_stage's fft_index // leaf_groups rotation). `reason`
    is set exactly when that does not hold -- the caller decides whether
    that is a hard failure (`leaf_storage == "spad"`) or a silent fallback
    to the existing CXL placement (`"auto"`).
    """
    buffer_names = tuple(b.name for b in node.workspace_buffers)
    if leaf_storage == "cxl" or not buffer_names:
        return None, None
    distribution = node.distributed
    if distribution is None or distribution.leaf_units != 1:
        leaf_units = None if distribution is None else distribution.leaf_units
        return None, f"leaf-resident SPAD requires units_per_leaf=1, not {leaf_units!r}"
    replicas = node.host.total_elems // node.length
    slots = (node.leaf_ownership.slots_per_unit if node.leaf_ownership is not None
             else -(-replicas // distribution.leaf_groups))  # ceil division
    slot_elements = 2 * node.length
    bank_base = tuple((name, index * slots * slot_elements) for index, name in enumerate(buffer_names))
    return LeafResidentSpadPlan(slots=slots, slot_elements=slot_elements, bank_base=bank_base), None


def node_parameter_bytes(node, *, leaf_resident=False):
    """RV64 fields in the current emitted ABI; no dependency on codegen."""
    if isinstance(node, PhysicalTransposePlan):
        return 8 * (4 + 2 * (node.twiddle_modulus is not None)
                    + node.needs_q_table + node.needs_tr_table + node.needs_round_split)
    # distributed emitter always declares pooled loop-twiddle pointers.
    # A leaf-resident buffer drops its CXL pointer field entirely (see
    # LeafResidentSpadPlan) -- it is addressed through the shared arena.
    buffers = (0 if leaf_resident else len(node.workspace_buffers)) + (2 if node.boundary_twiddle else 0)
    return 8 * (6 + 2 * (node.large_twiddle is not None) + buffers)


def make_task_fusion_plan(plan, *, target=DEFAULT_TARGET_PROFILE, compute_lanes=4,
                          narrow_middle_stages=False, control=False, leaf_storage="cxl",
                          batch_window="off", boundary_fusion="off", boundary_layout="legacy", near_group=1, far_group=1, leaf_phase_policy="stages", leaf_codegen_schedule="existing", spad_allocation="declared", far12_codelet="off"):
    def unsupported(reason):
        return FFTTaskFusionPlan(False, reason)

    if leaf_phase_policy not in ('stages','far12-private') or leaf_codegen_schedule not in ('existing','far0-short-live'):
        return unsupported('unknown leaf phase policy or codegen schedule')
    from planning.execution.fft_spad_liveness import SPAD_ALLOCATION_MODES
    if spad_allocation not in SPAD_ALLOCATION_MODES:
        return unsupported('unknown SPAD allocation policy')
    from planning.execution.fft_register_codelet import CODELET_MODES
    if far12_codelet not in CODELET_MODES:return unsupported('unknown far12 codelet mode')
    if far12_codelet!='off' and (leaf_phase_policy!='far12-private' or leaf_codegen_schedule!='far0-short-live'):
        return unsupported('far12 codelet requires the complete FS phase/schedule')
    compact = spad_allocation != 'declared'
    if compact and (leaf_phase_policy != 'far12-private' or leaf_codegen_schedule != 'far0-short-live'):
        return unsupported('compact SPAD currently requires the complete FS phase/schedule')
    custom = leaf_phase_policy != 'stages' or leaf_codegen_schedule != 'existing'
    if custom and (boundary_layout != 'grouped' or boundary_fusion != 'pre' or near_group != 4
                   or far_group != 1 or leaf_storage != 'spad' or control
                   or compute_lanes != 4 or not narrow_middle_stages):
        return unsupported('private/scheduled leaves require PRE-only, near C4, far C1, SPAD and SIMD4 narrowing')
    if boundary_layout not in ("legacy","grouped") or (boundary_layout == "legacy" and (near_group,far_group)!=(1,1)):
        return unsupported("nondefault leaf groups require boundary_layout=grouped")
    if boundary_layout == "grouped" and (compute_lanes != 4 or not narrow_middle_stages):
        return unsupported("initial grouped ownership requires compute_lanes=4 and middle-stage narrowing")
    if boundary_layout == "grouped" and (leaf_storage != "spad" or control):
        return unsupported("grouped ownership requires leaf SPAD and a regular fused task")
    if leaf_storage not in LEAF_STORAGE_MODES:
        raise ValueError(f"unknown leaf_storage {leaf_storage!r}; choose {LEAF_STORAGE_MODES}")
    from planning.execution.fft_boundary_fusion import BOUNDARY_FUSION_MODES, make_boundary_fusion_plan
    if boundary_fusion not in BOUNDARY_FUSION_MODES:
        raise ValueError(f'boundary_fusion must be one of {BOUNDARY_FUSION_MODES}')
    if boundary_fusion != 'off' and control:
        return unsupported('POST boundary fusion does not support layout-matched task controls')
    root = plan.root
    if not isinstance(root, FFTRecursiveNodePlan):
        return unsupported("single leaf has no recursive task boundaries to fuse")
    if not isinstance(root.near_fft, FFTLeafPlan) or not isinstance(root.far_child, FFTLeafPlan):
        return unsupported("initial task fusion supports one recursive level")
    if batch_window != "off":
        if plan.batch > 2147483647:
            return unsupported('batch window controller state requires batch <= Int32 max')
        if batch_window == "auto":
            candidates = [1 << i for i in range(plan.batch.bit_length() - 1, -1, -1)]
        else:
            try:
                requested = int(batch_window)
            except (TypeError, ValueError):
                raise ValueError('batch_window must be off, auto, or a positive integer') from None
            if requested < 1:
                raise ValueError('batch_window must be positive')
            candidates = [min(requested, plan.batch)]
        for width in candidates:
            candidate = make_task_fusion_plan(window_execution_plan(plan, width), target=target,
                compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
                control=control, leaf_storage=leaf_storage, boundary_fusion=boundary_fusion,boundary_layout=boundary_layout,near_group=near_group,far_group=far_group,leaf_phase_policy=leaf_phase_policy,leaf_codegen_schedule=leaf_codegen_schedule,spad_allocation=spad_allocation,far12_codelet=far12_codelet)
            extra = 8 if width < plan.batch else 0
            # Prefer a 32 KiB SPAD tier, leaving L1D space; explicit widths
            # may use the full existing concurrent arena budget.
            if (candidate.supported and candidate.reserved_bytes + extra <= target.spad_capacity_bytes
                    and (batch_window != 'auto' or width == 1 or candidate.reserved_bytes + extra <= 32768)):
                return replace(candidate, reserved_bytes=candidate.reserved_bytes + extra,
                    window_batch=width if width < plan.batch else 0,
                    total_batch=plan.batch if width < plan.batch else 0,
                    window_n=plan.n if width < plan.batch else 0)
        return unsupported('no batch window fits the task/SPAD constraints'
                           + (': ' + candidate.reason if boundary_fusion != 'off' else ''))
    nodes = flatten_recursive_node(root)
    if len(nodes) != 5 or any(getattr(child, 'tail_hybrid', None)
                              for child in (root.near_fft, root.far_child)):
        return unsupported("tail hybrids are not supported by task fusion")
    if boundary_fusion != 'off' or boundary_layout == 'grouped':
        # A compact allocation re-checks the real reservation below, so the
        # declared-bank estimate of the grouped ownership is deferred to it.
        boundary = make_boundary_fusion_plan(plan, mode=boundary_fusion,
            boundary_layout=boundary_layout,near_group=near_group,far_group=far_group,
            defer_leaf_budget=spad_allocation != 'declared')
        if not boundary.supported:
            return unsupported(boundary.reason)
        nodes = boundary.nodes
    if target.num_ndp_units < 1 or target.interleave_chunk_uthreads != 8 or target.uthread_bytes != 32:
        return unsupported("task fusion needs the SIMD-8 / 32-byte worker mapping")
    if target.mapping_stride_bytes != 256:
        return unsupported("task fusion needs 256-byte stripe alignment")
    if plan.n & (plan.n - 1):
        return unsupported("initial task fusion supports power-of-two lengths")
    semantic_ids = tuple(b.semantic_node_id for b in boundary.bindings) if boundary_layout == 'grouped' or boundary_fusion not in ('off','post') else ()
    functions = []
    private = [None] * len(nodes)
    resident = [None] * len(nodes)
    for index, node in enumerate(nodes):
        label = semantic_ids[index] if semantic_ids else index
        if isinstance(node, PhysicalTransposePlan):
            if (node.rows % node.tile_rows or node.cols % node.tile_cols
                    or node.needs_q_table or node.needs_tr_table):
                return unsupported("ragged/lookup transpose needs a separate fusion lowering")
            functions.append((f"node_{label}_transpose",))
        else:
            distribution = node.distributed
            if (distribution is None or distribution.fuse_stages != 1
                    or node.scratchpad_buffers or node.launch_mapping is not None
                    or node.large_twiddle is not None):
                return unsupported("task fusion requires distributed, unfused, global-workspace leaves")
            if (distribution.num_units != target.num_ndp_units
                    or distribution.workers_per_unit != target.interleave_chunk_uthreads
                    or node.total_uthreads != target.num_ndp_units * 8):
                return unsupported("leaf dispatch disagrees with the physical target")
            resident[index], reason = resolve_leaf_resident_storage(node, leaf_storage=leaf_storage)
            if reason is not None and leaf_storage == "spad":
                return unsupported(reason)
            names = registered_kernel_names(node, compute_lanes=compute_lanes,
                                            narrow_middle_stages=narrow_middle_stages)
            if label == 3 and leaf_phase_policy == 'far12-private':
                from planning.execution.fft_worker_private import prove_far12
                try:
                    private[index] = prove_far12(node,compute_lanes=compute_lanes,narrow_middle_stages=narrow_middle_stages,far12_codelet=far12_codelet)
                except ValueError as error:
                    return unsupported(str(error))
                names = ('stage_0','stage_1_2')
            functions.append(tuple(f"node_{label}_{name}" for name in names))
        if len(functions[-1]) > target.max_kernel_register:
            return unsupported("a single node exceeds the registered function limit")
    period = target.num_ndp_units * 8
    extent = max([period] + [node.total_uthreads for node in nodes if isinstance(node, PhysicalTransposePlan)])
    common = ((extent + period - 1) // period) * period
    slots = common // target.num_ndp_units
    elements = slots * max((node.scratchpad_elements for node in nodes if isinstance(node, PhysicalTransposePlan)), default=0)
    leaf_bank_elements = [r.total_elements for r in resident if r is not None]
    if leaf_bank_elements:
        elements = max(elements, *leaf_bank_elements)
    if any(private):
        offset = elements
        private = [replace(p,scratch_offset=offset) if p else None for p in private]
        elements += max(p.scratch_elements for p in private if p)
    memory = None
    if compact:
        from planning.execution.fft_spad_liveness import allocate_spad
        try:
            memory,resident,private=allocate_spad(nodes,resident,private,semantic_ids,slots,spad_allocation)
            elements=memory.arena_elements
        except ValueError as error:
            return unsupported(str(error))
    if far12_codelet in ('native4-static-control','native4-static-forward','native4-compact-control','native4-compact-forward','native4-latch-forward'):
        from planning.execution.fft_static_handoff import bind_static_handoffs
        try:
            private=[bind_static_handoffs(nodes[i],p,resident[i]) if p else None
                     for i,p in enumerate(private)]
        except (ValueError,KeyError) as error:
            return unsupported(str(error))
    params_bytes = sum(node_parameter_bytes(node, leaf_resident=resident[index] is not None)
                       for index, node in enumerate(nodes))
    reserved = elements * 4 + params_bytes + 8
    if reserved > target.spad_capacity_bytes or elements * 4 > target.max_concurrent_scratchpad_bytes:
        if leaf_storage == "auto" and leaf_bank_elements:
            # Storage placement can fail while CXL task fusion still fits.
            # Recompute Params, arena and control pools for the CXL layout.
            return make_task_fusion_plan(plan, target=target, compute_lanes=compute_lanes,
                narrow_middle_stages=narrow_middle_stages, control=control, leaf_storage="cxl",
                boundary_fusion=boundary_fusion,boundary_layout=boundary_layout,near_group=near_group,far_group=far_group,leaf_phase_policy=leaf_phase_policy,leaf_codegen_schedule=leaf_codegen_schedule,spad_allocation=spad_allocation,far12_codelet=far12_codelet)
        return unsupported("common launch arena plus Params exceeds the target scratchpad budget")

    # Partition active nodes only (five normally, four with the root POST absorbed).
    # Minimize tasks, largest registry, imbalance, then prefer a fuller prefix.
    candidates = []
    for cuts in product((False, True), repeat=len(nodes) - 1):
        ids, start = [], 0
        for index, cut in enumerate((*cuts, True)):
            if cut:
                ids.append(tuple(range(start, index + 1)))
                start = index + 1
        counts = tuple(sum(len(functions[i]) for i in group) for group in ids)
        if max(counts) <= target.max_kernel_register:
            candidates.append(((len(counts), max(counts), sum(c*c for c in counts),
                                tuple(-count for count in counts)), ids))
    if not candidates:
        return unsupported("no consecutive grouping fits the registered function limit")
    _, selected = min(candidates, key=lambda value: value[0])
    if compact and len(selected)!=1:
        return unsupported("compact SPAD currently requires one fused task")
    control_pool_ids = ()
    if control:
        # Reuse exactly the regular fused plan's launch pools. Allocating one
        # extra pool per control node shifts the FFT workspace addresses and
        # changes unit/channel placement, invalidating a matched comparison.
        control_pool_ids = tuple(next(i for i, group in enumerate(selected) if node in group)
                                 for node in range(len(nodes)))
        selected = tuple((node,) for node in range(len(nodes)))
    groups = tuple(FFTTaskGroup(f"FFTFusedTask{index}", ids,
                   tuple(name for node_id in ids for name in functions[node_id]))
                   for index, ids in enumerate(selected))
    return FFTTaskFusionPlan(True, "layout-matched control" if control else "resource-checked grouping",
                             common, slots, elements, params_bytes, reserved,
                             groups, tuple(functions), control_pool_ids, tuple(resident),
                             boundary_fusion=boundary_fusion, semantic_node_ids=semantic_ids,
                             boundary_layout=boundary_layout,near_group=near_group,far_group=far_group,leaf_phase_policy=leaf_phase_policy,leaf_codegen_schedule=leaf_codegen_schedule,private_phases=tuple(private) if any(private) else (),spad_allocation=spad_allocation,memory_allocation=memory,far12_codelet=far12_codelet)
