"""Exact chunk dependency components for an opt-in resident far1/2 phase."""
from dataclasses import dataclass, replace
from planning.execution.distributed_phases import (
    DistributedPhase, _pair_components, _lower_components, access_offsets)


@dataclass(frozen=True)
class WorkerPrivatePhase:
    phase: DistributedPhase
    scratch_offset: int
    workers: int
    codelets: tuple = ()
    codelet_mode: str = "off"
    handoffs: tuple = ()

    @property
    def scratch_elements(self):
        return self.workers * 2 * self.phase.local_elements


def prove_far12(node, *, compute_lanes, narrow_middle_stages, scratch_offset=0, far12_codelet="off"):
    d = node.distributed
    if d is None or d.leaf_units != 1 or len(node.stages) != 3:
        raise ValueError('far12-private requires a three-stage, single-unit resident leaf')
    if node.leaf_ownership is not None or node.boundary_simd is not None or node.boundary_twiddle:
        raise ValueError('far12-private requires far C1 and unchanged boundary IO')
    first, second = node.stages[1:]
    components = _pair_components(node, first, second, compute_lanes, narrow_middle_stages)
    if far12_codelet in ('native4-control','native4-forward','native4-static-control','native4-static-forward','native4-compact-control','native4-compact-forward','native4-latch-forward'):
        if node.simd_lanes % 4 or first.radix not in (2,4,8):
            raise ValueError('native4 requires a SIMD4-compatible small-radix leaf')
        native = _pair_components(node, replace(first,compute_lanes=4),
                                 replace(second,compute_lanes=4),compute_lanes,narrow_middle_stages)
        if tuple(c.offsets for c in native)!=tuple(c.offsets for c in components):
            raise ValueError('native4 changes component order or worker ownership')
        components=native
    if not components:
        raise ValueError('far12-private has no complete dependency components')
    # Boundary addresses are checked by bank and exact offset. The canonical
    # graph joins RAW and WAR; the independent sets below reject any remaining
    # cross-component writes/reads, including a future layout change.
    reads, writes = [], []
    for component in components:
        r, w = set(), set()
        for chunk in component.chunks:
            for load in chunk.batch.loads:
                r.update((load.source, load.buffer_name, o) for o in access_offsets(load, chunk.width))
            for output in chunk.batch.outputs:
                s = output.store
                for o in access_offsets(s, chunk.width):
                    key = (s.destination, s.buffer_name, o)
                    if key in w:
                        raise ValueError('duplicate writer in a private component')
                    w.add(key)
        reads.append(r); writes.append(w)
    for i in range(len(components)):
        for j in range(i):
            if writes[i] & (reads[j] | writes[j]) or writes[j] & reads[i]:
                raise ValueError('cross-component dependency remains')
    elements = max(len(c.offsets) for c in components)
    if elements > d.fusion_max_elements:
        raise ValueError('private component exceeds existing element cap')
    phase = DistributedPhase((first.stage_id, second.stage_id), _lower_components(components, first.stage_id))
    from planning.execution.fft_register_codelet import CODELET_MODES,prove_edges,prove_native_edges
    if far12_codelet not in CODELET_MODES:raise ValueError('unknown far12 codelet mode')
    edges=(prove_native_edges(phase) if far12_codelet in ('native4-forward','native4-static-control','native4-static-forward','native4-compact-control','native4-compact-forward','native4-latch-forward') else
           () if far12_codelet in ('off','native4-control') else prove_edges(phase))
    return WorkerPrivatePhase(phase, scratch_offset, d.workers_per_unit, edges, far12_codelet)


def node_phase_ids(node, private):
    return ((0,), private.phase.stage_ids) if private else tuple((s.stage_id,) for s in node.stages)
