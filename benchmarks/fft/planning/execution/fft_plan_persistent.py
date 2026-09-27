from __future__ import annotations

"""Persistent-software-workgroup FFT leaf: a fixed number of software
groups (`workers_per_group` workers each, one group per physical NDP
unit) that each own one physical scratchpad region and process many
logical FFT blocks sequentially, across rounds, inside one host-level
launch -- instead of today's pattern (every other planner in this
package) where physical launch width scales with logical demand.

Full design: docs/persistent_leaf_design.md. Standalone and opt-in: this
module is never imported by make_fft_kernel.py or fft_plan_recursive.py,
and nothing here changes their behavior for a plan that isn't itself
persistent. UPDATE 2026-09-22 (logical/physical cost-model correction --
see docs/logical_vs_physical_cost_model.md): `fft_plan_search.py` and
`fft_cost_model.py` now both import from here too -- `num_rounds` (cost
model, pre-existing) and `flatten_logical_workers_to_physical_lanes`/
`persistent_leaf_scratchpad_bytes` (cost model and gpu_baseline/common.py,
new) -- so a cost estimate and a VkFFT-M2NDP resource-feasibility check
can each read the exact same physical-lane/scratchpad-byte formulas
`codegen.fft_persistent_codegen.py`/`make_persistent_leaf_plan` itself
already uses, instead of a second, silently-driftable copy. Importing
these pure, side-effect-free helpers changes nothing about any NON-
persistent plan's behavior -- the "nothing here changes their behavior"
invariant above still holds for that case; it now only describes plans
this module doesn't itself build. Three entry points a caller
(codegen/fft_persistent_codegen.py, planning/search/fft_cost_model.py,
planning/gpu_baseline/common.py, or a standalone script) uses:

* `make_persistent_leaf_plan(...)` -- the planner, this module.
* `generate_persistent_fft_kernel(...)` -- the codegen, a sibling module.
* `flatten_logical_workers_to_physical_lanes(...)`/`physical_lane_workload(...)`
  -- the shared logical-worker -> physical-lane mapping both codegen and
  the cost model now render/measure through (see their own docstrings).

Core invariant, unchanged from every other planner in this package:
radix decomposition, butterfly arithmetic, twiddle exponents, the
Stockham permutation, first-stage indexing, final output order, and
inverse normalization are exactly `layouts_for_radices`/`_lower_stages`'s
own math -- see `_lower_persistent_stages` below, which mirrors
`_lower_stages` structurally and reuses `_make_load`/`_make_twiddle`/
`_make_store` outright. What differs is only *where* stage 0 reads from
and the last stage writes to (scratchpad, not DRAM -- see
`_make_load`/`_make_store`'s own `force_scratchpad`), and how work
within a stage is partitioned across `workers_per_group` cooperating
workers instead of running as one implicit worker (see
`_partition_vector_scalar`).
"""

from dataclasses import dataclass, replace
from typing import Literal

from planning.core.fft_plan_core import (
    _DEFAULT,
    AddressMapping,
    FFTCodegenPlan,
    FFTStagePlan,
    HostPlan,
    OutputPlan,
    PersistentWorkgroupPlan,
    ScratchpadBufferPlan,
    SIMDBatchPlan,
    _Default,
    _StageLayout,
    _build_plan,
    _make_load,
    _make_store,
    _make_twiddle,
    layouts_for_radices,
    pingpong_needed,
)
from planning.core.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile
from planning.execution.fft_plan_cooperative import choose_workers_per_fft, make_cooperative_leaf_plan


def persistent_leaf_scratchpad_bytes(length: int) -> int:
    """The ONE authoritative formula for how many scratchpad bytes a
    persistent-software-workgroup leaf of FFT length `length` needs --
    `16 * length` (two full ping-pong banks, each `2 * length` Float32
    elements -- split-complex real+imag -- times 4 bytes, unconditionally,
    regardless of stage count: see `PersistentWorkgroupPlan`'s own module
    docstring, "Always exactly two scratchpad buffers regardless of stage_
    count"). Factored out of `make_persistent_leaf_plan`'s own inline
    `required_scratchpad_bytes = 16 * length` (still the same formula,
    unchanged) so a caller OUTSIDE this module that needs to know "how
    many scratchpad bytes will a persistent leaf of this length actually
    require" -- `planning.gpu_baseline.common`'s own M2NDP resource
    adapter for `vkfft.py`'s `plan_m2ndp` (see that module's own docstring
    for why a VkFFT-style shared-memory-sizing formula must budget against
    THIS number, not the raw single-buffer GPU-LDS convention, once a
    leaf's cooperation width needs worker-wave virtualization) -- reads
    the exact same number `make_persistent_leaf_plan`'s own feasibility
    check enforces, instead of a second, independently-typed `16 *
    length` that could silently drift out of sync with it.
    """
    return 16 * length


def _persistent_read_buffer(stage_id: int, buffer_names: tuple[str, str]) -> str:
    """Stage `stage_id` reads `buffer_names[stage_id % 2]` -- see
    `_lower_persistent_stages`'s own docstring for why this indexing
    (not `_write_buffer_for_stage`/`_read_buffer_for_stage`'s existing
    formula, which assumes stage 0 never reads scratchpad at all) is the
    one that keeps preload -> stage 0 -> ... -> writeback a single
    consistent ping-pong chain, including a 1-stage leaf."""
    return buffer_names[stage_id % 2]


def _persistent_write_buffer(stage_id: int, buffer_names: tuple[str, str]) -> str:
    """Stage `stage_id` writes `buffer_names[(stage_id + 1) % 2]` --
    the bank the *next* stage (or writeback, for the last stage) reads.
    See `_persistent_read_buffer`."""
    return buffer_names[(stage_id + 1) % 2]


def _lower_persistent_stages(
    *,
    length: int,
    inverse: bool,
    simd_lanes: int,
    layouts: tuple[_StageLayout, ...],
    buffer_names: tuple[str, str],
    inverse_scale: float | None,
) -> tuple[FFTStagePlan, ...]:
    """Structurally mirrors `fft_plan_core._lower_stages` (same per-batch
    loop, same `_make_load`/`_make_twiddle`/`_make_store` calls, same
    mathematical stage layout from `layouts`) but every stage reads and
    writes scratchpad (`force_scratchpad=True`), never DRAM -- `_lower_
    stages` itself is untouched and not called from here. Preload/
    writeback are separate bulk-copy phases (codegen/fft_persistent_
    codegen.py), not FFT stages, and never go through `_make_load`/
    `_make_store` at all.

    Always exactly two scratchpad buffers regardless of `stage_count`
    (see PersistentWorkgroupPlan's own module docstring and docs/
    persistent_leaf_design.md's "Buffer count -- always two banks") --
    `buffer_names` must already be a 2-tuple; there is no single-buffer
    or in-place optimization here.
    """
    stages: list[FFTStagePlan] = []
    stage_count = len(layouts)

    for stage_id, layout in enumerate(layouts):
        first_stage = stage_id == 0
        last_stage = stage_id == stage_count - 1
        read_buffer = _persistent_read_buffer(stage_id, buffer_names)
        write_buffer = _persistent_write_buffer(stage_id, buffer_names)

        simd_iters = (layout.butterfly_count + simd_lanes - 1) // simd_lanes
        batches: list[SIMDBatchPlan] = []

        for simd_it in range(simd_iters):
            first_bfly = simd_it * simd_lanes
            valid_lanes = min(simd_lanes, layout.butterfly_count - first_bfly)
            input_batch_base = simd_it * layout.input_batch_width

            loads = tuple(
                _make_load(
                    operand=operand,
                    first_stage=first_stage,
                    read_buffer=read_buffer,
                    base_offset=input_batch_base + operand * layout.input_stride,
                    valid_lanes=valid_lanes,
                    simd_lanes=simd_lanes,
                    force_scratchpad=True,
                )
                for operand in range(layout.radix)
            )

            outputs: list[OutputPlan] = []
            for output in range(layout.radix):
                outputs.append(
                    OutputPlan(
                        output=output,
                        twiddle=_make_twiddle(
                            inverse=inverse,
                            simd_lanes=simd_lanes,
                            valid_lanes=valid_lanes,
                            first_bfly=first_bfly,
                            output=output,
                            modulus=layout.twiddle_modulus,
                            stride=layout.twiddle_stride,
                            lane_divisor=layout.twiddle_lane_divisor,
                        ),
                        scale=inverse_scale if last_stage else None,
                        large_twiddle=False,
                        store=_make_store(
                            last_stage=last_stage,
                            write_buffer=write_buffer,
                            layout=layout,
                            simd_it=simd_it,
                            output=output,
                            valid_lanes=valid_lanes,
                            simd_lanes=simd_lanes,
                            force_scratchpad=True,
                        ),
                    )
                )

            batches.append(
                SIMDBatchPlan(
                    batch_id=simd_it,
                    valid_lanes=valid_lanes,
                    loads=loads,
                    outputs=tuple(outputs),
                )
            )

        stages.append(
            FFTStagePlan(
                stage_id=stage_id,
                radix=layout.radix,
                inverse=inverse,
                simd_iteration_count=simd_iters,
                batches=tuple(batches),
            )
        )

    return tuple(stages)


def _partition_vector_scalar(
    batches: tuple[SIMDBatchPlan, ...],
    *,
    workers_per_group: int,
    simd_lanes: int,
    scalar_worker_mode: Literal["adaptive", "reserved"],
) -> tuple[tuple[tuple[SIMDBatchPlan, ...], ...], tuple[SIMDBatchPlan, ...]]:
    """One stage's own `(persistent_vector_batches, persistent_scalar_
    batches)` -- see docs/persistent_leaf_design.md's "Stage work
    partition"/"Persistent batch partition" sections.

    Despite the parameter's name (kept as `workers_per_group` since that
    is this function's one and only caller's own historical bucket count),
    this is really just "how many buckets to partition into" -- the caller
    now passes `workers_per_fft` (the LOGICAL worker count), which equals
    `workers_per_group` (physical) whenever no worker-wave virtualization
    is in play, and is a larger multiple of it otherwise. Nothing in this
    function's own logic depends on which one it is.

    Full batches (`valid_lanes == simd_lanes`) go round-robin to vector
    workers; the (at most one, since only a stage's own last SIMD
    iteration can ever be partial -- `layouts_for_radices` guarantees
    this) partial batch goes entirely to the scalar worker, always
    worker `workers_per_group - 1`.

    `scalar_worker_mode="reserved"`: worker `workers_per_group - 1` is
    always scalar-reserved, tail or not (idle, not vector-repurposed, on
    a no-tail stage). `"adaptive"`: every worker is a vector worker
    unless this stage actually has a tail batch.
    """
    full_batches = tuple(b for b in batches if b.valid_lanes == simd_lanes)
    tail_batches = tuple(b for b in batches if b.valid_lanes < simd_lanes)
    if len(tail_batches) > 1:
        raise ValueError(
            "a stage may have at most one tail batch (only the last SIMD "
            "iteration can be partial)"
        )
    has_tail = bool(tail_batches)

    reserve_scalar = scalar_worker_mode == "reserved" or has_tail
    vector_worker_count = workers_per_group - 1 if reserve_scalar else workers_per_group
    if vector_worker_count < 1:
        raise ValueError("workers_per_group too small to leave any vector worker")

    vector_buckets: list[list[SIMDBatchPlan]] = [[] for _ in range(workers_per_group)]
    for i, batch in enumerate(full_batches):
        vector_buckets[i % vector_worker_count].append(batch)
    vector_batches = tuple(tuple(bucket) for bucket in vector_buckets)

    scalar_batches = tail_batches if reserve_scalar else ()
    return vector_batches, scalar_batches


def _make_persistent_host_plan(
    *, length: int, num_logical_blocks: int, launch_uthreads: int, simd_lanes: int,
    inverse: bool,
) -> HostPlan:
    """Not really the same contract as `fft_plan_core._make_host_plan`
    (that one describes one uthread == one whole logical FFT; here
    `launch_uthreads` is fixed regardless of `num_logical_blocks`, and
    many logical blocks share the same physical launch across rounds) --
    kept as a `HostPlan` anyway so `FFTCodegenPlan.host` stays a uniform
    field every consumer can read the same way. `total_elems`/`pool_elems`
    describe the *launch's* own DRAM footprint and pool size (one round's
    worth, `launch_uthreads` wide) -- codegen.fft_persistent_codegen.py
    is responsible for actually looping `ceil(num_logical_blocks /
    software_group_count)` rounds; this field does not encode round count.
    """
    return HostPlan(
        total_elems=length * num_logical_blocks,
        pool_elems=simd_lanes * launch_uthreads,
        length=length,
        total_uthreads=launch_uthreads,
        inverse=inverse,
        tolerance=1.0e-3,
    )


def make_persistent_leaf_plan(
    length: int,
    radices: tuple[int, ...],
    *,
    num_logical_blocks: int,
    inverse: bool = False,
    simd_lanes: int = 8,
    scalar_worker_mode: Literal["adaptive", "reserved"] = "adaptive",
    kernel_name: str = "PersistentFFT",
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
    inverse_scale: float | None | _Default = _DEFAULT,
    workers_per_fft: int | None = None,
    lowering_mode: Literal["wave", "fused", "physical"] = "physical",
    copy_mode: Literal["scalar", "vectorized_contiguous"] = "scalar",
) -> FFTCodegenPlan:
    """A length-`length` leaf FFT (single fused kernel, `radices` its own
    Cooley-Tukey/Stockham stage sequence -- same contract as `_build_plan`/
    `layouts_for_radices`), executed by the persistent-software-workgroup
    model: `target.num_ndp_units` software groups (one per physical NDP
    unit), each `target.interleave_chunk_uthreads` workers wide, process
    `num_logical_blocks` logical FFT blocks across `ceil(num_logical_blocks
    / target.num_ndp_units)` rounds inside one fixed-width host launch.

    Only `stripes_per_group == 1` is implemented; every other shape this
    function might otherwise compute raises `NotImplementedError` rather
    than silently generating a mapping that doesn't match the target (see
    docs/persistent_leaf_design.md's "Target-mapping invariant checks").

    `spad_capacity_bytes` is not a parameter here (unlike `_build_plan`):
    the byte-capacity check below is unconditional, since a persistent
    leaf always needs the full `16 * length` bytes of scratchpad (see
    `required_scratchpad_bytes` below) regardless of caller-supplied
    launch width -- there is no "smaller launch, less capacity needed"
    tradeoff the way there is for the ordinary per-uthread model.

    `inverse_scale`: the same `_DEFAULT` sentinel `_build_plan`/
    `make_cooperative_leaf_plan` use -- omitted, it's `1/length` when
    `inverse` else `None`; pass `None` explicitly to suppress that for a
    persistent leaf used as a non-final kernel in a recursive chain (the
    overall 1/N belongs on whichever one kernel in the chain is actually
    last, never automatically on every `inverse=True` kernel -- see
    `_build_plan`'s own docstring for the same rule). Needed for
    `fft_plan_recursive.py`'s own leaf builder to thread persistent leaves
    through a split exactly like every other leaf kind already does.

    `workers_per_fft`: `None` (the default) means "equal to `target.
    interleave_chunk_uthreads`" -- the plain, pre-existing one-wave case,
    byte-identical codegen to every plan built before this parameter
    existed. Any other positive value is accepted (as of 2026-09-13 --
    see `worker_waves`'s own docstring for the full generalization from
    "exact multiples only" to "any positive count"): `workers_per_fft`
    LOGICAL workers cooperate on each logical FFT block via `worker_waves(
    workers_per_fft, workers_per_group)` sequential passes of the same
    physical workers, the LAST of which is ragged whenever `workers_per_
    fft` doesn't evenly divide `workers_per_group` -- see
    `PersistentWorkgroupPlan.workers_per_fft`'s own docstring for why a
    ragged wave's extra physical lanes are safe dummy lanes (no FFT
    arithmetic, no scratchpad address, still reach every barrier every
    other lane in the group does, since they run the exact same compiled
    function). This target's own `interleave_chunk_uthreads=8` periodic
    address interleaving is what makes 8-wide temporal multiplexing valid
    at all (see docs/gpu_baseline_hardware_mapping_audit.md) -- but that
    fact never constrained WHICH `workers_per_fft` values are legal, only
    HOW they execute; a `workers_per_fft` that is neither a divisor nor a
    multiple of 8 was rejected by an earlier, overly conservative version
    of this function, not by any real architectural limit (see docs/
    ragged_worker_wave_generalization.md for the investigation that
    established this).

    `lowering_mode`: stored verbatim on the returned plan's own
    `PersistentWorkgroupPlan.lowering_mode` (metadata only -- see that
    field's own docstring for what it is and is not authoritative over).
    Default `"physical"` matches every plan built before this parameter
    existed.
    """
    if num_logical_blocks < 1:
        raise ValueError("num_logical_blocks must be >= 1")
    if not radices:
        raise ValueError("radices must be non-empty")

    workers_per_group = target.interleave_chunk_uthreads
    software_group_count = target.num_ndp_units
    stripes_per_group = 1

    if workers_per_fft is None:
        workers_per_fft = workers_per_group
    if workers_per_fft < 1:
        raise ValueError(f"workers_per_fft={workers_per_fft} must be >= 1")

    if target.mapping_stride_bytes % target.uthread_bytes != 0:
        raise NotImplementedError(
            f"target.mapping_stride_bytes={target.mapping_stride_bytes} is not a "
            f"multiple of target.uthread_bytes={target.uthread_bytes} -- the "
            f"persistent leaf's group<->physical-unit mapping assumes it is"
        )
    derived_interleave_chunk = target.mapping_stride_bytes // target.uthread_bytes
    if target.interleave_chunk_uthreads != derived_interleave_chunk:
        raise NotImplementedError(
            f"target.interleave_chunk_uthreads={target.interleave_chunk_uthreads} "
            f"!= mapping_stride_bytes // uthread_bytes ({derived_interleave_chunk}) "
            f"-- target profile is internally inconsistent"
        )
    # These two are unreachable today (workers_per_group/software_group_count
    # are always derived directly from the target two lines up, never an
    # independent caller input) -- kept as documentation of the invariant
    # docs/persistent_leaf_design.md's own "Target-mapping invariant checks"
    # section requires, and as a guard that stays correct if a future
    # version ever makes either value caller-overridable instead of
    # target-derived.
    if workers_per_group != target.interleave_chunk_uthreads:
        raise NotImplementedError(
            f"workers_per_group ({workers_per_group}) must equal "
            f"target.interleave_chunk_uthreads ({target.interleave_chunk_uthreads})"
        )
    if software_group_count != target.num_ndp_units:
        raise NotImplementedError(
            f"software_group_count ({software_group_count}) must equal "
            f"target.num_ndp_units ({target.num_ndp_units})"
        )
    if stripes_per_group != 1:
        raise NotImplementedError("only stripes_per_group == 1 is implemented")

    required_scratchpad_bytes = persistent_leaf_scratchpad_bytes(length)
    if required_scratchpad_bytes > target.spad_capacity_bytes:
        max_length = target.spad_capacity_bytes // persistent_leaf_scratchpad_bytes(1)
        raise ValueError(
            f"length={length} needs {required_scratchpad_bytes} bytes of scratchpad "
            f"(16*length, two banks of split-complex FP32), but "
            f"target.spad_capacity_bytes={target.spad_capacity_bytes} allows at most "
            f"{max_length} -- this is only the byte-capacity upper bound, not a "
            f"guarantee length={max_length} itself has a supported radix "
            f"factorization/stage layout"
        )

    product = 1
    for r in radices:
        product *= r
    if product != length:
        raise ValueError(
            f"product of radices ({product}) must equal length ({length})"
        )

    # Hard feasibility constraint, not a performance penalty -- see
    # target.max_kernel_register's own docstring for the real-hardware
    # crash this guards against. codegen.fft_persistent_codegen.
    # generate_persistent_fft_kernel re-checks this too (defense in
    # depth), but rejecting here means a caller building candidate plans
    # (e.g. a future search/ranking layer) never even constructs codegen
    # for an infeasible one.
    registered_kernels = 2 + len(radices)
    if registered_kernels > target.max_kernel_register:
        raise ValueError(
            f"radices={radices} needs {registered_kernels} distinct kernel "
            f"functions (preload + {len(radices)} stages + writeback), but "
            f"target.max_kernel_register={target.max_kernel_register} caps "
            f"one task's total registered kernels regardless of round count "
            f"-- see target.max_kernel_register's own docstring"
        )

    layouts = layouts_for_radices(length, radices, simd_lanes)
    buffer_names: tuple[str, str] = ("buf_a", "buf_b")
    if inverse_scale is _DEFAULT:
        inverse_scale = (1.0 / length) if inverse else None

    stages = _lower_persistent_stages(
        length=length,
        inverse=inverse,
        simd_lanes=simd_lanes,
        layouts=layouts,
        buffer_names=buffer_names,
        inverse_scale=inverse_scale,
    )
    stages = tuple(
        replace(
            stage,
            **_partition_stage_fields(
                stage,
                # The number of buckets a stage's batches are partitioned
                # into is the LOGICAL worker count (workers_per_fft), not
                # the physical one -- see `_partition_vector_scalar`'s own
                # `workers_per_group` parameter, which is really just
                # "bucket count" and is reused here unchanged. Equals
                # `workers_per_group` (today's only case) whenever
                # `workers_per_fft` was not overridden.
                workers_per_group=workers_per_fft,
                simd_lanes=simd_lanes,
                scalar_worker_mode=scalar_worker_mode,
            ),
        )
        for stage in stages
    )

    launch_uthreads = software_group_count * workers_per_group
    scratchpad_stride = 2 * length
    scratchpad_buffers = tuple(
        ScratchpadBufferPlan(name=name, elements=scratchpad_stride)
        for name in buffer_names
    )

    logical_block_stride = length

    return FFTCodegenPlan(
        length=length,
        inverse=inverse,
        total_uthreads=launch_uthreads,
        max_uthread=launch_uthreads,
        simd_lanes=simd_lanes,
        kernel_name=kernel_name,
        input_mapping=AddressMapping.contiguous(length),
        output_mapping=AddressMapping.contiguous(length),
        large_twiddle=None,
        scratchpad_uthread_stride=scratchpad_stride,
        scratchpad_buffers=scratchpad_buffers,
        stages=stages,
        host=_make_persistent_host_plan(
            length=length,
            num_logical_blocks=num_logical_blocks,
            launch_uthreads=launch_uthreads,
            simd_lanes=simd_lanes,
            inverse=inverse,
        ),
        persistent=PersistentWorkgroupPlan(
            stripes_per_group=stripes_per_group,
            workers_per_stripe=workers_per_group,
            workers_per_group=workers_per_group,
            workers_per_fft=workers_per_fft,
            software_group_count=software_group_count,
            logical_block_stride=logical_block_stride,
            scalar_worker_mode=scalar_worker_mode,
            lowering_mode=lowering_mode,
            copy_mode=copy_mode,
        ),
    )


def _partition_stage_fields(
    stage: FFTStagePlan,
    *,
    workers_per_group: int,
    simd_lanes: int,
    scalar_worker_mode: Literal["adaptive", "reserved"],
) -> dict[str, object]:
    vector_batches, scalar_batches = _partition_vector_scalar(
        stage.batches,
        workers_per_group=workers_per_group,
        simd_lanes=simd_lanes,
        scalar_worker_mode=scalar_worker_mode,
    )
    return {
        "persistent_vector_batches": vector_batches,
        "persistent_scalar_batches": scalar_batches,
    }


def worker_waves(workers_per_fft: int, workers_per_group: int) -> int:
    """`ceil(workers_per_fft / workers_per_group)` -- how many logical
    workers ANY ONE physical lane owns, at most, for a cooperation width
    `workers_per_fft` wider than the physical group. `workers_per_fft`
    need not be a multiple of `workers_per_group`: this is a genuine
    ceiling division (see `PersistentWorkgroupPlan.workers_per_fft`'s own
    docstring for the 2026-09-13 generalization from "exact multiples
    only" to "any positive count").

    REVISED 2026-09-14 (worker-wave FUSION -- see docs/
    worker_wave_fusion.md): this value is METADATA/DIAGNOSTICS ONLY now --
    "how many logical workers, at most, does one physical lane visit for
    this stage" -- NOT a runtime dispatch driver any more. Before this
    revision, `codegen.fft_persistent_codegen.emit_persistent_kernel_
    struct` called `device_main` `worker_waves(...)` times per stage, each
    a separate `launch_parallel[stage_N]()`; that repeated-launch
    structure is gone (see `emit_stage_phase`'s own docstring) -- a
    physical lane now visits all of its own logical workers (`worker_id`,
    `worker_id + workers_per_group`, ...) via a single runtime `while`
    loop INSIDE one stage invocation, so callers needing "how many stage
    invocations does this plan use" should no longer multiply by this
    function's own result (it is always 1 now, regardless of
    `workers_per_fft`) -- this function still answers a real, useful
    question (loop trip count / worst-case per-lane logical-worker count)
    for cost-model or reporting purposes, just not "launch_parallel call
    count" any more.

    A `workers_per_fft` that does NOT evenly divide `workers_per_group`
    (e.g. 6 or 10 against `workers_per_group=8`) leaves the LAST loop
    iteration ragged: only `workers_per_fft - (waves-1)*workers_per_group`
    of that iteration's `workers_per_group` physical lanes correspond to a
    real logical worker (`0 <= logical_worker_id < workers_per_fft`); the
    remaining physical lanes compute a `logical_worker_id >= workers_per_
    fft` that matches no dispatch branch in `codegen.fft_persistent_
    codegen._emit_worker_dispatch` (see that function's own docstring) --
    the `while` loop simply exits for them without ever entering that
    final iteration's body, touching no FFT arithmetic and no scratchpad
    address at all. `workers_per_fft <= workers_per_group` (e.g. 3, 5, 6,
    7) is the special case `waves == 1`: no loop is even emitted (see
    `emit_stage_phase`), byte-identical dispatch shape to before
    `workers_per_fft` existed, only the branch count differs.

    This ceiling division is a mixed-radix bijection by construction:
    `(iteration, physical_lane)` for `iteration in range(waves)`,
    `physical_lane in range(workers_per_group)` maps to `logical_worker_id
    = iteration * workers_per_group + physical_lane`, which ranges over
    `0 .. waves*workers_per_group - 1` exactly once each -- restricting to
    `logical_worker_id < workers_per_fft` therefore covers `{0, ...,
    workers_per_fft - 1}` exactly once each, with no gap and no duplicate,
    for ANY positive `workers_per_fft`, not just an exact multiple. See
    `verification.verify_fft_persistent.check_logical_worker_coverage` for
    this fact exercised as an explicit test.
    """
    if workers_per_fft < 1:
        raise ValueError(f"workers_per_fft={workers_per_fft} must be >= 1")
    if workers_per_group < 1:
        raise ValueError(f"workers_per_group={workers_per_group} must be >= 1")
    return -(-workers_per_fft // workers_per_group)


def flatten_logical_workers_to_physical_lanes(
    vector_batches: tuple[tuple[SIMDBatchPlan, ...], ...],
    scalar_batches: tuple[SIMDBatchPlan, ...],
    *,
    workers_per_fft: int,
    workers_per_group: int,
) -> tuple[tuple[SIMDBatchPlan, ...], ...]:
    """THE shared logical-worker -> physical-lane mapping: collapse
    `workers_per_fft` LOGICAL workers' worth of already-partitioned batches
    (`vector_batches`/`scalar_batches` -- exactly `stage.persistent_vector_
    batches`/`stage.persistent_scalar_batches`, built by
    `_partition_vector_scalar`) down to exactly `workers_per_group`
    PHYSICAL buckets -- physical lane `p`'s own bucket is the
    concatenation, in ascending logical-worker order, of every logical
    worker `j`'s own batches where `j % workers_per_group == p`.

    Moved here (2026-09-22) from `codegen.fft_persistent_codegen.
    _flatten_to_physical_lanes`, which now calls this function instead of
    computing its own copy (see that function's own docstring) -- so
    `codegen.fft_persistent_codegen`'s default "physical" (Mode C)
    execution-lowering strategy and `planning.search.fft_cost_model`'s own
    per-stage cost metrics read the identical mapping, instead of the cost
    model separately reading `persistent_vector_batches` at its raw
    LOGICAL (`workers_per_fft`-wide) granularity -- the exact "cost model
    silently assumes workers_per_fft-way physical concurrency, when the
    default codegen lowering only ever gives M2NDP `workers_per_group`
    (8) physical lanes" mismatch this refactor closes (see docs/
    logical_vs_physical_cost_model.md). `workers_per_fft <= workers_per_
    group` is the identity case: each physical lane gets AT MOST one
    logical worker's own batches (lane `j` gets logical worker `j`'s
    batches; lanes `>= workers_per_fft` are empty) -- byte-identical to
    reading `vector_batches` directly, which is exactly why this change is
    a no-op for every plan this project's own ordinary planner/search path
    builds (workers_per_fft is never set above `workers_per_group` there
    -- see `make_persistent_leaf_plan`'s own `workers_per_fft` docstring;
    only a GPU-baseline-derived plan, `gpu_baseline.common._map_worker_
    wave_kernel`, ever sets a wider one).

    The scalar/tail batch (owned by logical worker `workers_per_fft - 1`
    alone -- see `_partition_vector_scalar`'s own docstring) lands on
    whichever physical lane that logical worker maps to (`(workers_per_
    fft - 1) % workers_per_group`), appended AFTER that lane's own vector
    batches -- matching `codegen.fft_persistent_codegen._worker_body`'s
    "tail batch owned by the last logical worker, rendered like every
    other worker's batches" contract exactly.
    """
    if len(vector_batches) != workers_per_fft:
        raise ValueError(
            f"len(vector_batches)={len(vector_batches)} must equal "
            f"workers_per_fft={workers_per_fft}"
        )
    physical: list[list[SIMDBatchPlan]] = [[] for _ in range(workers_per_group)]
    for logical_worker_id in range(workers_per_fft):
        lane = logical_worker_id % workers_per_group
        physical[lane].extend(vector_batches[logical_worker_id])
    if scalar_batches:
        tail_lane = (workers_per_fft - 1) % workers_per_group
        physical[tail_lane].extend(scalar_batches)
    return tuple(tuple(bucket) for bucket in physical)


@dataclass(frozen=True)
class PhysicalLaneWorkload:
    """The physical-lane-flattened shape of one persistent stage's own
    workload -- what `planning.search.fft_cost_model.compute_stage_
    metrics` now measures a persistent stage's execution cost against
    (see `physical_lane_workload` below), and what a plan/report dump
    (section 16 of the task this was built from) shows per stage."""

    physical_lane_batches: tuple[tuple[SIMDBatchPlan, ...], ...]
    num_logical_workers: int
    num_physical_lanes: int
    max_batches_per_lane: int
    min_batches_per_lane: int
    avg_batches_per_lane: float
    active_physical_lanes: int


def physical_lane_workload(
    vector_batches: tuple[tuple[SIMDBatchPlan, ...], ...],
    scalar_batches: tuple[SIMDBatchPlan, ...],
    *,
    workers_per_fft: int,
    workers_per_group: int,
) -> PhysicalLaneWorkload:
    """`flatten_logical_workers_to_physical_lanes` plus the summary stats a
    cost model or report dump actually wants -- batch COUNTS here (`len`
    of each lane's own bucket), not a cost-unit-weighted quantity (see
    `fft_cost_model.compute_stage_metrics`'s own `chunks_per_batch`
    weighting for that; this function stays a plain, reusable "how many
    batches per physical lane" primitive that any caller -- cost model,
    diagnostics, a future report -- can weight however it needs, rather
    than baking in one particular cost unit here)."""
    physical_batches = flatten_logical_workers_to_physical_lanes(
        vector_batches, scalar_batches,
        workers_per_fft=workers_per_fft, workers_per_group=workers_per_group,
    )
    counts = [len(bucket) for bucket in physical_batches]
    active = sum(1 for c in counts if c > 0)
    total = sum(counts)
    return PhysicalLaneWorkload(
        physical_lane_batches=physical_batches,
        num_logical_workers=workers_per_fft,
        num_physical_lanes=workers_per_group,
        max_batches_per_lane=max(counts, default=0),
        min_batches_per_lane=min(counts, default=0),
        avg_batches_per_lane=(total / workers_per_group) if workers_per_group else 0.0,
        active_physical_lanes=active,
    )


def num_rounds(num_logical_blocks: int, software_group_count: int) -> int:
    return -(-num_logical_blocks // software_group_count)


def round_active_groups(
    round_index: int, num_logical_blocks: int, software_group_count: int
) -> int:
    round_base = round_index * software_group_count
    return max(0, min(software_group_count, num_logical_blocks - round_base))


# =============================================================================
# P2.4: persistent tail-round hybrid (see docs/priority2_execution_
# strategies.md). Every persistent leaf's own round count is `ceil(
# num_logical_blocks / software_group_count)` -- the LAST round is
# under-utilized whenever `num_logical_blocks` doesn't divide evenly by
# `software_group_count` (e.g. `num_ndp_units=32`, `num_logical_blocks=33`:
# round 0 uses all 32 groups, round 1 uses exactly 1). This section adds
# SELECTABLE alternatives for that tail portion -- reusing `make_
# persistent_leaf_plan` (full rounds), `fft_plan_core._build_plan` (non-
# cooperative tail), and `fft_plan_cooperative.make_cooperative_leaf_plan`
# (cooperative tail) OUTRIGHT, never a new execution mechanism -- per this
# task's own "reuse the existing execution implementations" instruction.
# =============================================================================

TailStrategy = Literal["all_persistent", "noncoop_tail", "cooperative_tail"]


@dataclass(frozen=True)
class PersistentTailHybridPlan:
    """The result of `make_persistent_tail_hybrid_plan`: `full_blocks`
    logical blocks lowered via the ordinary persistent leaf
    (`persistent_plan`), plus (unless `tail_strategy == "all_persistent"`
    or there is no genuine tail at all) `tail_blocks` MORE logical blocks
    lowered via a SEPARATE, independent kernel (`tail_plan`) using a
    different execution strategy. `tail_plan is None` exactly when
    `tail_blocks == 0` -- there is nothing for a second kernel to do, so
    none is built; `codegen.fft_persistent_codegen.
    generate_persistent_tail_hybrid_kernel` degenerates to the plain
    `generate_persistent_fft_kernel` call in that case, byte-identical to
    before this mechanism existed.

    `persistent_plan`/`tail_plan` are two COMPLETELY INDEPENDENT kernels
    (separate `NDPTask` structs, separate DRAM buffers, separate launches)
    -- there is no data dependency between "blocks 0..full_blocks-1" and
    "blocks full_blocks..num_logical_blocks-1", so no chaining/ordering
    concern exists beyond both being launched once each from the same
    host `main()` (see that codegen function's own docstring for why this
    avoids needing any pointer-offset arithmetic between them, a Mojo
    idiom this project's own codegen has never used or verified against
    the real toolchain elsewhere).
    """

    full_blocks: int
    tail_blocks: int
    tail_strategy: TailStrategy
    # `None` in the one edge case where `full_blocks == 0` (fewer than
    # `target.num_ndp_units` logical blocks total, under a non-"all_
    # persistent" strategy): there is no persistent portion at all, every
    # replica goes through `tail_plan` alone -- see `codegen.fft_
    # persistent_codegen.generate_persistent_tail_hybrid_kernel`'s own
    # docstring for how that degenerates (renders `tail_plan` as a plain
    # standalone kernel, no persistent struct/launch at all).
    persistent_plan: FFTCodegenPlan | None
    tail_plan: FFTCodegenPlan | None


def make_persistent_tail_hybrid_plan(
    length: int,
    radices: tuple[int, ...],
    *,
    num_logical_blocks: int,
    inverse: bool = False,
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
    tail_strategy: TailStrategy = "all_persistent",
    tail_cooperative_workers: int | None = None,
    kernel_name: str = "PersistentFFT",
    tail_kernel_name: str = "TailFFT",
    inverse_scale: float | None | _Default = _DEFAULT,
    workers_per_fft: int | None = None,
    lowering_mode: Literal["wave", "fused", "physical"] = "physical",
    copy_mode: Literal["scalar", "vectorized_contiguous"] = "scalar",
) -> PersistentTailHybridPlan:
    """Split `num_logical_blocks` into `full_blocks = (num_logical_blocks
    // target.num_ndp_units) * target.num_ndp_units` (an exact multiple of
    the physical unit count -- every persistent round in this portion uses
    ALL `target.num_ndp_units` software groups) plus `tail_blocks =
    num_logical_blocks - full_blocks` (`0` when it already divided evenly).

    `tail_strategy="all_persistent"` (the default -- byte-identical to
    every plan built before this function existed): `full_blocks` is
    forced back to `num_logical_blocks` and `tail_plan` stays `None` --
    the entire logical-block count goes through one ordinary persistent
    leaf, tail round under-utilization and all, exactly as `make_
    persistent_leaf_plan(length, radices, num_logical_blocks=num_logical_
    blocks, ...)` alone already produces.

    `tail_strategy="noncoop_tail"`: `tail_blocks` replicas (`0` is handled
    the same as `"all_persistent"` -- no genuine tail to special-case)
    instead go through `fft_plan_core._build_plan` -- one physical
    microthread per replica, the plain, always-available fallback
    strategy every leaf in this project can use.

    `tail_strategy="cooperative_tail"`: `tail_blocks` replicas go through
    `fft_plan_cooperative.make_cooperative_leaf_plan` instead, at
    `tail_cooperative_workers` (default: `choose_workers_per_fft`'s own
    heuristic answer for this `length`/`radices`, the same default every
    other cooperative leaf in this project uses when not given an
    explicit override).

    Every leaf-shape decision (radix sequence, stage layout, twiddle
    placement) is identical across all three strategies -- ONLY the
    EXECUTION mechanism for the tail differs, per this task's own "GPU
    logical plan must remain fixed" instruction (there is no GPU baseline
    involved here at all, but the same discipline applies: `length`/
    `radices` are never altered by this function, only how the tail's own
    physical launch is organized).
    """
    num_ndp_units = target.num_ndp_units
    full_blocks = (num_logical_blocks // num_ndp_units) * num_ndp_units
    tail_blocks = num_logical_blocks - full_blocks

    if tail_strategy == "all_persistent" or tail_blocks == 0:
        persistent_plan = make_persistent_leaf_plan(
            length, radices, num_logical_blocks=num_logical_blocks, inverse=inverse,
            target=target, kernel_name=kernel_name, inverse_scale=inverse_scale,
            workers_per_fft=workers_per_fft, lowering_mode=lowering_mode, copy_mode=copy_mode,
        )
        return PersistentTailHybridPlan(
            full_blocks=num_logical_blocks, tail_blocks=0, tail_strategy="all_persistent",
            persistent_plan=persistent_plan, tail_plan=None,
        )

    # `full_blocks == 0` (fewer than `num_ndp_units` logical blocks total):
    # no persistent portion at all -- every replica goes through the tail
    # strategy alone (see `PersistentTailHybridPlan.persistent_plan`'s own
    # docstring for why this is `None`, not an empty/degenerate persistent
    # plan `make_persistent_leaf_plan` itself would reject).
    persistent_plan = (
        make_persistent_leaf_plan(
            length, radices, num_logical_blocks=full_blocks, inverse=inverse,
            target=target, kernel_name=kernel_name, inverse_scale=inverse_scale,
            workers_per_fft=workers_per_fft, lowering_mode=lowering_mode, copy_mode=copy_mode,
        )
        if full_blocks > 0 else None
    )

    if tail_strategy == "noncoop_tail":
        tail_plan = _build_plan(
            length=length, inverse=inverse, total_uthreads=tail_blocks, simd_lanes=8,
            use_pingpong=pingpong_needed(len(radices)),
            layouts=layouts_for_radices(length, radices, 8),
            kernel_name=tail_kernel_name, inverse_scale=inverse_scale,
        )
    elif tail_strategy == "cooperative_tail":
        workers = (
            tail_cooperative_workers if tail_cooperative_workers is not None
            else choose_workers_per_fft(length, radices)
        )
        tail_plan = make_cooperative_leaf_plan(
            length, radices, workers_per_fft=workers, total_ffts=tail_blocks, inverse=inverse,
            kernel_name=tail_kernel_name, inverse_scale=inverse_scale,
        )
    else:
        raise ValueError(f"unknown tail_strategy {tail_strategy!r}")

    return PersistentTailHybridPlan(
        full_blocks=full_blocks, tail_blocks=tail_blocks, tail_strategy=tail_strategy,
        persistent_plan=persistent_plan, tail_plan=tail_plan,
    )
