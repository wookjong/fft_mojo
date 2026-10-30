"""Resolve boundary addresses and buffers without changing the semantic FFT tree.

PRE, MIDDLE and POST are independently absorbed at a leaf input or output.
Each radix stage keeps its synchronous kernel completion boundary; resident
ownership is not a worker barrier. A load suffix identifies MIDDLE placement.
"""
from dataclasses import asdict, dataclass, replace

from planning.core.fft_plan_core import AddressMapping, AddressMappingKind, rebind_final_output, rebind_first_input
from planning.strategies.fft_plan_recursive import (
    FFTLeafPlan, FFTRecursiveNodePlan, PhysicalTransposePlan, flatten_recursive_node,
)

BOUNDARY_FUSION_MODES = ('off', 'pre', 'middle', 'post', 'pre-middle', 'pre-post', 'middle-post', 'all', 'middle-load', 'pre-middle-load', 'middle-post-load', 'all-load')


def normalize_boundary_mode(mode, middle_placement=None):
    """Use one canonical mode in planner, generated source and resume keys."""
    if mode not in BOUNDARY_FUSION_MODES:
        raise ValueError(f'unknown boundary mode: {mode}')
    if middle_placement is None:
        return mode
    if middle_placement not in ('store', 'load'):
        raise ValueError('middle placement must be store or load')
    if 'middle' not in mode and not mode.startswith('all'):
        raise ValueError('--middle-placement requires a mode containing MIDDLE')
    base = mode.removesuffix('-load')
    return base + ('-load' if middle_placement == 'load' else '')


@dataclass(frozen=True)
class BoundaryBufferBinding:
    semantic_node_id: int
    input_buffer: str
    output_buffer: str


@dataclass(frozen=True)
class FFTBoundaryFusionPlan:
    mode: str
    supported: bool
    reason: str
    nodes: tuple = ()
    bindings: tuple[BoundaryBufferBinding, ...] = ()
    removed_node_ids: tuple[int, ...] = ()
    output_mapping: AddressMapping | None = None
    inverse_scale: float | None = None

    def metadata(self):
        return dict(mode=self.mode, supported=self.supported, reason=self.reason,
                    logical_node_count=len(self.nodes) + len(self.removed_node_ids),
                    active_node_ids=[b.semantic_node_id for b in self.bindings],
                    removed_node_ids=list(self.removed_node_ids),
                    buffers=[asdict(b) for b in self.bindings],
                    output_mapping=(dict(kind=self.output_mapping.kind.value,
                        rows_per_batch=self.output_mapping.batch_rows,
                        batch_stride=self.output_mapping.batch_stride,
                        row_stride=self.output_mapping.row_stride,
                        elem_stride=self.output_mapping.elem_stride,
                        base=self.output_mapping.base) if self.output_mapping else None),
                    inverse_scale=self.inverse_scale,
                    synchronization='kernel_completion')


def make_boundary_fusion_plan(plan, *, mode='off', boundary_layout='legacy', near_group=1, far_group=1):
    if boundary_layout == 'grouped':
        from planning.execution.fft_grouped_boundary import make_grouped_boundary_plan
        return make_grouped_boundary_plan(plan,mode=mode,near_group=near_group,far_group=far_group)
    if boundary_layout != 'legacy' or near_group != 1 or far_group != 1:
        raise ValueError('nondefault leaf groups require boundary_layout=grouped')
    if mode not in BOUNDARY_FUSION_MODES:
        raise ValueError(f'boundary_fusion must be one of {BOUNDARY_FUSION_MODES}')

    def unsupported(reason):
        return FFTBoundaryFusionPlan(mode, False, reason)

    semantic = tuple(flatten_recursive_node(plan.root))
    if mode != 'off':
        root = plan.root
        if (not isinstance(root, FFTRecursiveNodePlan)
                or not isinstance(root.near_fft, FFTLeafPlan)
                or not isinstance(root.far_child, FFTLeafPlan) or len(semantic) != 5):
            return unsupported('POST boundary fusion requires one recursive level with two leaves')
        if plan.n & (plan.n - 1):
            return unsupported('POST boundary fusion requires a power-of-two length')
        for node in semantic:
            if isinstance(node, PhysicalTransposePlan):
                if (node.rows % node.tile_rows or node.cols % node.tile_cols
                        or node.needs_q_table or node.needs_tr_table):
                    return unsupported('POST boundary fusion does not support ragged/lookup transposes')
                continue
            if (node.distributed is None or node.distributed.fuse_stages != 1
                    or node.distributed.leaf_units != 1):
                return unsupported('POST boundary fusion requires distributed unfused leaves with units_per_leaf=1')
            if (node.large_twiddle is not None or node.scratchpad_buffers
                    or node.persistent is not None or node.cooperation is not None
                    or node.launch_mapping is not None):
                return unsupported('POST boundary fusion requires ordinary distributed leaves')
            if node.input_mapping != AddressMapping.contiguous(node.length) or node.output_mapping != AddressMapping.contiguous(node.length):
                return unsupported('POST boundary fusion requires canonical leaf boundary mappings')
            if any(o.scale is not None for s in node.stages for b in s.batches for o in b.outputs):
                return unsupported('POST boundary fusion requires normalization owned by the root POST')
        enabled = ({'pre','middle','post'} if mode.startswith('all') else set(mode.split('-'))) - {'load'}
        placement = 'load' if mode.endswith('-load') else 'store'
        effective = list(semantic)
        mapping = None
        scale = None
        if 'pre' in enabled:
            im = AddressMapping.batched_strided(rows_per_batch=root.a, batch_stride=plan.n,
                                               row_stride=1, elem_stride=root.a)
            effective[1] = rebind_first_input(effective[1], input_mapping=im)
        if 'middle' in enabled:
            if placement == 'store':
                mm = AddressMapping.batched_strided(rows_per_batch=root.a, batch_stride=plan.n,
                                                   row_stride=1, elem_stride=root.a)
                effective[1] = replace(rebind_final_output(effective[1], output_mapping=mm,
                    inverse_scale=None), boundary_twiddle=('store', root.a, root.b))
            else:
                mm = AddressMapping.batched_strided(rows_per_batch=root.b, batch_stride=plan.n,
                                                   row_stride=1, elem_stride=root.b)
                effective[3] = replace(rebind_first_input(effective[3], input_mapping=mm),
                                      boundary_twiddle=('load', root.a, root.b))
        if 'post' in enabled:
            post = root.post_transpose
            if (post.rows != root.b or post.cols != root.a or post.replica_count != plan.batch
                    or post.twiddle_modulus is not None or post.apply_inverse_scale != plan.inverse):
                return unsupported('root POST shape/scale contract disagrees with the FFT plan')
            mapping = AddressMapping.batched_strided(rows_per_batch=root.b, batch_stride=plan.n,
                                                    row_stride=1, elem_stride=root.b)
            scale = 1.0 / plan.n if plan.inverse else None
            effective[3] = rebind_final_output(effective[3], output_mapping=mapping, inverse_scale=scale)
        removed = tuple(i for i,name in ((0,'pre'),(2,'middle'),(4,'post')) if name in enabled)
        ids = tuple(i for i in range(5) if i not in removed)
        nodes = tuple(effective[i] for i in ids)
    else:
        nodes, removed, mapping, scale = semantic, (), None, None
        ids = tuple(range(len(nodes)))
    bindings = tuple(BoundaryBufferBinding(semantic_id,
                        'input' if i == 0 else f'work{(i-1)%2}',
                        'output' if i == len(nodes)-1 else f'work{i%2}')
                     for i,semantic_id in enumerate(ids))
    return FFTBoundaryFusionPlan(mode, True, 'POST absorbed by far final store' if mode == 'post'
        else 'unchanged boundaries' if mode == 'off' else 'explicit boundary transforms',
        nodes, bindings, removed, mapping, scale)
