from __future__ import annotations

"""Persistent-software-workgroup FFT leaf: a fixed number of software
groups (`workers_per_group` workers each, one group per physical NDP
unit) that each own one physical scratchpad region and process many
logical FFT blocks sequentially, across rounds, inside one host-level
launch -- instead of today's pattern (every other planner in this
package) where physical launch width scales with logical demand.

Full design: docs/persistent_leaf_design.md. Standalone and opt-in: this
module is never imported by make_fft_kernel.py, fft_plan_recursive.py,
fft_plan_search.py, or fft_cost_model.py, and nothing here changes their
behavior. Two entry points a caller (codegen/fft_persistent_codegen.py,
or a standalone script) uses:

* `make_persistent_leaf_plan(...)` -- the planner, this module.
* `generate_persistent_fft_kernel(...)` -- the codegen, a sibling module.

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

from dataclasses import replace
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
    _make_load,
    _make_store,
    _make_twiddle,
    layouts_for_radices,
)
from planning.core.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile


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

    required_scratchpad_bytes = 16 * length
    if required_scratchpad_bytes > target.spad_capacity_bytes:
        max_length = target.spad_capacity_bytes // 16
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


def num_rounds(num_logical_blocks: int, software_group_count: int) -> int:
    return -(-num_logical_blocks // software_group_count)


def round_active_groups(
    round_index: int, num_logical_blocks: int, software_group_count: int
) -> int:
    round_base = round_index * software_group_count
    return max(0, min(software_group_count, num_logical_blocks - round_base))
