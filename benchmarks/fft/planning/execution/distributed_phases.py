"""Prove local two-stage components from the lowered FFT's exact addresses.

Only an entire dependency component may be assigned to one worker. There is
no inter-worker scratchpad communication and no assumption about radix shape.
SIMD widths are resolved before the proof, using the same policy as codegen.
"""
from dataclasses import dataclass, replace

from codegen.lowering import chunk_batch
from planning.core.fft_plan_core import SIMDBatchPlan
from planning.execution.fft_plan_lanes import resolve_stage_compute_lanes


@dataclass(frozen=True)
class PhaseChunk:
    stage_id: int
    width: int
    batch: SIMDBatchPlan


@dataclass(frozen=True)
class LocalComponent:
    chunks: tuple[PhaseChunk, ...]
    offsets: tuple[int, ...]


@dataclass(frozen=True)
class DistributedPhase:
    stage_ids: tuple[int, ...]
    components: tuple[LocalComponent, ...] = ()

    @property
    def name(self):
        return "stage_" + "_".join(map(str, self.stage_ids))

    @property
    def local_elements(self):
        return max((len(c.offsets) for c in self.components), default=0)


def stage_width(plan, stage, compute_lanes, narrow_middle_stages):
    return stage.compute_lanes or resolve_stage_compute_lanes(
        compute_lanes=compute_lanes, is_first=stage.stage_id == 0,
        is_last=stage.stage_id == len(plan.stages) - 1, radix=stage.radix,
        prev_radix=plan.stages[stage.stage_id - 1].radix if stage.stage_id else None,
        narrow_middle_stages=narrow_middle_stages) or plan.simd_lanes


def access_offsets(access, width):
    if access.mode == "vector":
        return tuple(range(access.base_offset, access.base_offset + width))
    offsets = (access.packed_lane_offsets if hasattr(access, "packed_lane_offsets")
               else access.lane_offsets)
    return tuple(offset for offset in offsets if offset is not None)


def _chunks(plan, stage, compute_lanes, narrow_middle_stages):
    width = stage_width(plan, stage, compute_lanes, narrow_middle_stages)
    if width < 1 or width > plan.simd_lanes or plan.simd_lanes % width:
        raise ValueError("distributed compute lanes must divide the launch SIMD width")
    return tuple(PhaseChunk(stage.stage_id, w, chunk)
                 for batch in stage.batches
                 for w, chunk in chunk_batch(batch, simd_lanes=plan.simd_lanes,
                                             compute_lanes=width) if chunk.valid_lanes)


def _pair_components(plan, first, second, compute_lanes, narrow_middle_stages):
    left = _chunks(plan, first, compute_lanes, narrow_middle_stages)
    right = _chunks(plan, second, compute_lanes, narrow_middle_stages)
    chunks = left + right
    parent = list(range(len(chunks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    writers = {}
    for i, chunk in enumerate(left):
        for output in chunk.batch.outputs:
            store = output.store
            if store.destination != "workspace":
                return ()
            for offset in access_offsets(store, chunk.width):
                key = (store.buffer_name, offset)
                if key in writers:
                    raise ValueError("duplicate producer in distributed phase")
                writers[key] = i
    if len({name for name, _ in writers}) != 1:
        return ()
    readers = set()
    for i, chunk in enumerate(right, len(left)):
        for load in chunk.batch.loads:
            if load.source != "workspace":
                return ()
            for offset in access_offsets(load, chunk.width):
                key = (load.buffer_name, offset)
                if key not in writers:
                    raise ValueError("distributed phase reads an unproduced element")
                parent[find(i)] = find(writers[key])
                readers.add(key)
    if readers != set(writers) or len(writers) != plan.length:
        raise ValueError("distributed phase does not cover the full intermediate bank")
    # Two ping-pong stages can end in their own input bank. A component must
    # also contain every old-bank reader that its final stores would overwrite.
    # Otherwise another worker/component could still be reading that value.
    boundary_readers = {}
    for i, chunk in enumerate(left):
        for load in chunk.batch.loads:
            if load.source == "workspace":
                for offset in access_offsets(load, chunk.width):
                    boundary_readers.setdefault((load.buffer_name, offset), []).append(i)
    for i, chunk in enumerate(right, len(left)):
        for output in chunk.batch.outputs:
            store = output.store
            if store.destination == "workspace":
                for offset in access_offsets(store, chunk.width):
                    for reader in boundary_readers.get((store.buffer_name, offset), ()):
                        parent[find(i)] = find(reader)
    groups = {}
    for i, chunk in enumerate(chunks):
        groups.setdefault(find(i), []).append(chunk)
    components = []
    for group in groups.values():
        offsets = sorted({offset for chunk in group if chunk.stage_id == first.stage_id
                          for output in chunk.batch.outputs
                          for offset in access_offsets(output.store, chunk.width)})
        components.append(LocalComponent(tuple(group), tuple(offsets)))
    return tuple(components)


def _local_access(access, offsets):
    if access.mode == "vector":
        return replace(access, buffer_name="phase_scratch",
                       base_offset=offsets[access.base_offset])
    if hasattr(access, "packed_lane_offsets"):
        return replace(access, buffer_name="phase_scratch", packed_lane_offsets=tuple(
            None if i is None else offsets[i] for i in access.packed_lane_offsets))
    return replace(access, buffer_name="phase_scratch",
                   lane_offsets=tuple(offsets[i] for i in access.lane_offsets))


def _lower_components(components, first_stage_id):
    """Resolve compact local addresses here; codegen only renders this IR."""
    lowered = []
    for component in components:
        offsets = {offset: i for i, offset in enumerate(component.offsets)}
        chunks = []
        for chunk in component.chunks:
            batch = chunk.batch
            if chunk.stage_id == first_stage_id:
                batch = replace(batch, outputs=tuple(replace(output, store=replace(
                    _local_access(output.store, offsets), destination="scratchpad"))
                    for output in batch.outputs))
            else:
                batch = replace(batch, loads=tuple(replace(
                    _local_access(load, offsets), source="scratchpad") for load in batch.loads))
            chunks.append(replace(chunk, batch=batch))
        lowered.append(replace(component, chunks=tuple(chunks)))
    return tuple(lowered)


def resolve_distributed_phases(plan, *, compute_lanes=4, narrow_middle_stages=False):
    distribution = plan.distributed
    if distribution is None:
        raise ValueError("phase resolution requires distributed execution")
    phases = []
    index = 0
    while index < len(plan.stages):
        first = plan.stages[index]
        if distribution.fuse_stages == 2 and index + 1 < len(plan.stages):
            second = plan.stages[index + 1]
            components = _pair_components(plan, first, second,
                                          compute_lanes, narrow_middle_stages)
            elements = max((len(c.offsets) for c in components), default=0)
            local_bytes = 2 * elements * 4 * distribution.workers_per_unit
            # Do not reduce the requested unit parallelism just to save a launch.
            if (len(components) >= distribution.leaf_units and elements > 0
                    and elements <= distribution.fusion_max_elements
                    and local_bytes <= distribution.scratchpad_budget_bytes):
                phases.append(DistributedPhase((first.stage_id, second.stage_id),
                              _lower_components(components, first.stage_id)))
                index += 2
                continue
        phases.append(DistributedPhase((first.stage_id,)))
        index += 1
    return tuple(phases)
