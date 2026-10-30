"""Explicit execution alternatives for GPU plans on multiple NDP units."""
from dataclasses import replace

from planning.core.fft_plan_core import (
    DistributedExecutionPlan, WorkspaceBufferPlan, _build_plan, layouts_for_radices,
)
from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.strategies.fft_plan_recursive import (
    FFTLeafPlan, make_recursive_transpose_plan,
)

GPU_EXECUTION_MODES = ("worker-wave", "distributed", "split-leaves")


def _map_tree(node, leaf_mapper):
    if isinstance(node, FFTLeafPlan):
        return leaf_mapper(node)
    if getattr(node, "tail_hybrid", None) is not None:
        raise ValueError("unit-parallel execution does not support hybrid tails")
    return replace(node, near_fft=_map_tree(node.near_fft, leaf_mapper),
                   far_child=_map_tree(node.far_child, leaf_mapper))


def distribute_leaf(kernel, *, target=DEFAULT_TARGET_PROFILE, units_per_leaf=None,
                    fuse_stages=1, fusion_max_elements=256):
    """Keep radix, boundary mappings, twiddles and scaling; change placement.

    Rebuild canonical batches because persistent preload plans read their
    first stage from scratchpad. Intermediate SoA banks become shared CXL
    allocations, with disjoint per-FFT slices and synchronous stage launches.
    """
    if kernel.simd_lanes != 8 or target.uthread_bytes != 32:
        raise ValueError("distributed FFT requires a 32-byte launch granule (SIMD 8)")
    if target.mapping_stride_bytes != target.interleave_chunk_uthreads * target.uthread_bytes:
        raise ValueError("mapping stride must match workers per unit and microthread bytes")
    if target.num_ndp_units < 1 or target.interleave_chunk_uthreads < 1:
        raise ValueError("distributed FFT needs positive unit and worker counts")
    replicas = kernel.host.total_elems // kernel.length
    if units_per_leaf == "auto":
        # Throughput heuristic, not a measured latency optimum. Preserve the
        # physical decoder period; only partition its units into leaf groups.
        units_per_leaf = max(1, target.num_ndp_units // min(replicas, target.num_ndp_units))
    radices = tuple(s.radix for s in kernel.stages)
    scale = kernel.stages[-1].batches[0].outputs[0].scale
    plain = _build_plan(length=kernel.length, inverse=kernel.inverse,
                        total_uthreads=replicas, simd_lanes=kernel.simd_lanes,
                        use_pingpong=True, layouts=layouts_for_radices(kernel.length, radices, kernel.simd_lanes),
                        kernel_name=kernel.kernel_name, input_mapping=kernel.input_mapping,
                        output_mapping=kernel.output_mapping, large_twiddle=kernel.large_twiddle,
                        inverse_scale=scale)
    stages = []
    for original, stage in zip(kernel.stages, plain.stages):
        batches = []
        for batch in stage.batches:
            loads = tuple(replace(load, source="workspace") if load.source == "scratchpad" else load
                          for load in batch.loads)
            outputs = tuple(replace(output, store=replace(output.store, destination="workspace"))
                            if output.store.destination == "scratchpad" else output
                            for output in batch.outputs)
            batches.append(replace(batch, loads=loads, outputs=outputs))
        stages.append(replace(stage, batches=tuple(batches), compute_lanes=original.compute_lanes))
    distribution = DistributedExecutionPlan(
        target.num_ndp_units, target.interleave_chunk_uthreads,
        units_per_leaf=units_per_leaf, fuse_stages=fuse_stages,
        fusion_max_elements=fusion_max_elements,
        scratchpad_budget_bytes=min(target.spad_capacity_bytes,
                                    target.max_concurrent_scratchpad_bytes))
    buffers = tuple(WorkspaceBufferPlan(b.name, 2 * kernel.length * replicas)
                    for b in plain.scratchpad_buffers)
    return replace(plain, total_uthreads=distribution.launch_uthreads,
                   max_uthread=distribution.workers_per_unit, scratchpad_buffers=(),
                   scratchpad_uthread_stride=0, stages=tuple(stages), workspace_buffers=buffers,
                   host=replace(plain.host, total_uthreads=distribution.launch_uthreads,
                                pool_elems=distribution.launch_uthreads * plain.simd_lanes),
                   distributed=distribution)


def distribute_plan(plan, *, target=DEFAULT_TARGET_PROFILE, units_per_leaf=None,
                    fuse_stages=1, fusion_max_elements=256):
    return replace(plan, root=_map_tree(plan.root, lambda leaf:
                   replace(leaf, kernel=distribute_leaf(leaf.kernel, target=target,
                           units_per_leaf=units_per_leaf, fuse_stages=fuse_stages,
                           fusion_max_elements=fusion_max_elements))))


def _set_scale(kernel, scale):
    """Apply the enclosing recursive FFT's normalization exactly once."""
    last = kernel.stages[-1]
    batches = tuple(replace(batch, outputs=tuple(replace(output, scale=scale)
                    for output in batch.outputs)) for batch in last.batches)
    by_id = {batch.batch_id: batch for batch in batches}
    def partition(groups):
        return None if groups is None else tuple(tuple(by_id[b.batch_id] for b in group) for group in groups)
    last = replace(last, batches=batches, worker_batches=partition(last.worker_batches),
                   persistent_vector_batches=partition(last.persistent_vector_batches),
                   persistent_scalar_batches=None if last.persistent_scalar_batches is None else
                   tuple(by_id[b.batch_id] for b in last.persistent_scalar_batches))
    return replace(kernel, stages=(*kernel.stages[:-1], last))


def make_split_leaf_plan(n, *, leaf_planner, inverse=False, batch=1,
                         leaf_byte_budget=4096, target=DEFAULT_TARGET_PROFILE):
    """Reuse six-step exchange, but let the selected GPU planner build leaves.

    The leaf budget deliberately changes the original GPU decomposition.
    Each leaf's independent replicas retain the GPU worker/radix choice and
    are distributed by the existing cooperative/persistent launch engine.
    """
    plan = make_recursive_transpose_plan(n, inverse=inverse, batch=batch,
             scratchpad_byte_budget=leaf_byte_budget, target=target,
             simd_lanes=8, spread_across_units=True)
    def map_leaf(leaf):
        result = leaf_planner(leaf.m, batch=leaf.r, inverse=inverse, target=target)
        if result.status.value != "ok":
            raise ValueError(f"GPU leaf {leaf.m} x {leaf.r}: {result.diagnostics}")
        root = result.plan.root
        if not isinstance(root, FFTLeafPlan):
            raise ValueError(f"GPU planner further splits leaf {leaf.m}; choose a smaller leaf budget")
        scale = leaf.kernel.stages[-1].batches[0].outputs[0].scale
        return replace(leaf, kernel=replace(_set_scale(root.kernel, scale),
                                           kernel_name=leaf.kernel.kernel_name))
    return replace(plan, root=_map_tree(plan.root, map_leaf))


def apply_gpu_execution(plan, *, mode="worker-wave", leaf_planner=None,
                        leaf_byte_budget=4096, target=DEFAULT_TARGET_PROFILE,
                        units_per_leaf=None, fuse_stages=1, fusion_max_elements=256):
    if mode != "distributed" and (units_per_leaf is not None or fuse_stages != 1
                                   or fusion_max_elements != 256):
        raise ValueError("unit and fusion options require distributed execution")
    if mode == "worker-wave":
        return plan
    if mode == "distributed":
        return distribute_plan(plan, target=target, units_per_leaf=units_per_leaf,
                               fuse_stages=fuse_stages, fusion_max_elements=fusion_max_elements)
    if mode == "split-leaves":
        if leaf_planner is None:
            raise ValueError("split-leaves requires the selected GPU leaf planner")
        return make_split_leaf_plan(plan.n, batch=plan.batch, inverse=plan.inverse,
                    leaf_planner=leaf_planner, leaf_byte_budget=leaf_byte_budget, target=target)
    raise ValueError(f"unknown GPU execution mode {mode!r}; choose {GPU_EXECUTION_MODES}")
