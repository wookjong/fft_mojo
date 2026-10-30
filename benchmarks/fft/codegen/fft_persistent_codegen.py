from __future__ import annotations

"""Pure code emission for a persistent-software-workgroup FFT leaf (see
planning/fft_plan_persistent.py and docs/persistent_leaf_design.md). No FFT
planning happens here -- reuses fft_codegen.py's own `_emit_load`/
`_emit_twiddle`/`_emit_output`/`_emit_store`/`_emit_batch`/
`_emit_stage_batches` outright for stage bodies (every persistent stage's
`LoadPlan`/`StorePlan` already resolves to `source`/`destination ==
"scratchpad"` via `force_scratchpad`, so those functions' existing
"scratchpad" branches apply completely unchanged -- nothing here re-derives
an address, a radix, or a twiddle value).

What is genuinely new here, not reused from anywhere else:

* preload/writeback bulk DRAM<->scratchpad copy (no FFT math at all),
* the `gid -> worker_id -> software_group_id -> logical_block` dispatch
  prelude every phase function shares,
* per-worker `if worker_id == k: ...` dispatch inside a stage phase
  (mirrors fft_cooperative_codegen._emit_cooperative_stage's own
  runtime-dispatch idiom, but keyed on `global_uthread_id() %
  workers_per_group`, never `local_uthread_id()` -- see
  PersistentWorkgroupPlan's own docstring for why),
* the round-unrolled `device_main` (one host `.launch()`, many
  `launch_parallel` phases -- see docs/persistent_leaf_design.md's
  "Exactly one host .launch()" section),
* launch-pool allocation aligned to the target mapping stride.

Round and wave indices live in worker-private scratchpad cells, one cell per
physical worker on each unit. Every phase reads only its worker's own cell;
writeback advances its own round cell, and a wave phase advances its own wave
cell. The next launch_parallel starts after the preceding phase finishes.
There is no same-phase shared-counter read/write dependency, regardless of
worker issue order. A task registers one function per phase and reuses those
functions across rounds, keeping registration independent of the batch size.

"""

from typing import Literal
from codegen.common import emit_aligned_pool, emit_execution_record, emit_fft_launches
from planning.execution.launch_plan import resolve_fft_launch
from dataclasses import replace
from planning.core.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile

from codegen.common import (
    Emitter,
    emit_prelude as _emit_prelude,
    emit_reference_check as _emit_reference_check,
    emit_array_dump as _emit_array_dump,
    spad as _spad,
    emit_workspace_buffers,
    emit_loop_twiddle_table,
)
from codegen.fft_codegen import (
    _emit_stage_batches,
    _emit_loop_stage,
    _stage_compute_lanes,
    emit_kernel as _emit_kernel,
    generate_fft_kernel as _generate_fft_kernel,
)
from codegen.fft_cooperative_codegen import generate_cooperative_fft_kernel as _generate_cooperative_fft_kernel
from planning.core.fft_plan_core import FFTCodegenPlan, FFTStagePlan, SIMDBatchPlan
from planning.core.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile
from codegen.lowering import try_build_loop_stage, LOOP_MIN_FULL_BATCHES
from planning.execution.fft_plan_persistent import (
    PersistentTailHybridPlan,
    flatten_logical_workers_to_physical_lanes,
    num_rounds,
    worker_waves as _worker_waves,
    persistent_tracker_scratchpad_bytes,
)

# REVISED 2026-09-14 (physical-lane strip-mining -- see docs/
# physical_lane_strip_mining.md): three execution-lowering strategies for
# the SAME plan (same radix decomposition, same workers_per_fft, same
# _partition_vector_scalar batch/worker assignment -- nothing about the
# GPU-facing plan changes across modes, only how it is realized on
# M2NDP's fixed 8-physical-lane substrate), kept side by side so the real
# toolchain can compare them apples-to-apples rather than trusting
# source-level reasoning alone:
#
#   "wave"     -- Mode A, the ORIGINAL mechanism: workers_per_fft LOGICAL
#                 workers realized as ceil(workers_per_fft/workers_per_
#                 group) separate `launch_parallel[stage_N]()` calls from
#                 device_main, wave_index threaded via a scratchpad
#                 `wave_tracker` counter between calls. Proven correct,
#                 proven to pay a real per-wave barrier/launch cost.
#   "fused"    -- Mode B (2026-09-14 worker-wave fusion): ONE
#                 `launch_parallel[stage_N]()` call; each physical lane
#                 visits its own logical workers via a runtime `while
#                 logical_worker_id < workers_per_fft:` loop wrapped
#                 around the exact same workers_per_fft-way `if/elif`
#                 dispatch tree Mode A already built. Removed the
#                 barriers; real-hardware validation then found it can
#                 newly spill (an extra live range crossing the loop
#                 back-edge -- confirmed by disassembly: the loop needs
#                 4 more callee-saved integer registers, s4-s7, than
#                 Mode A/C ever need in the same function, plus 16 more
#                 bytes of local spill-slot space -- see docs/
#                 physical_lane_strip_mining.md's register-pressure
#                 section). CORRECTION 2026-09-14: VkFFT (9,8,3) at
#                 N=216, workers_per_fft=36 crashing the M2NDP-Detour
#                 simulator (`unmapped opcode: CSRRS` near a spilled
#                 stack-frame prologue) was ORIGINALLY (mis)attributed
#                 to this mode alone -- re-verified through the
#                 production `generate_recursive_fft_kernels` path (the
#                 original standalone-codegen probe used to find it used
#                 a different host-wrapper shape) and confirmed the
#                 identical crash (same stage, same 16-byte DRAM-spill
#                 frame, same CSRRS panic) occurs under EVERY mode --
#                 "wave" and "physical" included. The crash is a M2NDP-
#                 Detour simulator/toolchain gap (no CSRRS decode
#                 support for whatever the compiler emits right after
#                 any spilled prologue), not a consequence of this
#                 mode's own dispatch shape -- see docs/
#                 physical_lane_strip_mining.md's crash-analysis section.
#   "physical" -- Mode C (this revision): flattens the workers_per_fft
#                 logical-worker batch assignment down to `workers_per_
#                 group` (8) PHYSICAL buckets at CODEGEN TIME (a physical
#                 lane p's own batches are simply the concatenation, in
#                 ascending order, of every logical worker j's own
#                 already-partitioned batches where `j % workers_per_
#                 group == p` -- see `_flatten_to_physical_lanes`), then
#                 emits a plain `workers_per_group`-way `if/elif`
#                 dispatch, exactly the SAME shape `_emit_worker_dispatch`
#                 already renders for the `workers_per_fft <= workers_per_
#                 group` case -- no runtime logical_worker_id, no loop, no
#                 W-sized branch tree, one `launch_parallel` call. Total
#                 emitted batch code is IDENTICAL to Mode B (same batches,
#                 same math, same addresses) -- only the CONTROL FLOW
#                 wrapping them changes: a compile-time-fixed 8-way
#                 dispatch instead of a runtime loop over a W-way one.
#
# Default is "physical" (Mode C) once real-toolchain validation confirmed
# it matches or beats Mode B's cycles with none of Mode B's new spilling
# (the VkFFT crash itself is mode-independent -- see above) -- see docs/
# physical_lane_strip_mining.md's own validation section. "wave"/"fused"
# remain fully implemented and
# selectable (never deleted -- explicit opt-in via `mode=`) for future
# comparison, exactly mirroring how `workers_per_fft` itself is kept as
# GPU-planner metadata even though M2NDP's own physical execution width
# is always 8.
ExecutionMode = Literal["wave", "fused", "physical"]

# P2.3 (persistent preload/writeback copy strategy -- see docs/
# priority2_execution_strategies.md). "scalar" (the default, unchanged):
# `emit_bulk_copy_phase`'s own original per-lane strided one-element-at-a-
# time copy (`i = worker_id`, `i += workers_per_group`). "vectorized_
# contiguous": each physical lane instead owns one CONTIGUOUS block of
# `ceil(length / workers_per_group)` elements, copied `vector_width`-wide
# at a time (real load/store idiom `codegen.fft_codegen._emit_load`/
# `_emit_store` already use elsewhere in this project, `.load[width=V]`/
# inferred-width `.store`, never a new one), with a scalar tail for
# `length`s that don't divide evenly. See `emit_bulk_copy_phase`'s own
# docstring for why changing the lane<->index mapping this way is
# provably correctness-preserving.
CopyMode = Literal["scalar", "vectorized_contiguous"]


def validate_copy_width(copy_mode: CopyMode, vector_width: int) -> None:
    if copy_mode == "vectorized_contiguous" and (
        not isinstance(vector_width, int) or isinstance(vector_width, bool)
        or vector_width < 1
    ):
        raise ValueError("vector_width must be a positive integer for vectorized_contiguous copy")


def resolve_persistent_options(
    plan: FFTCodegenPlan, *, mode: ExecutionMode | None = None,
    copy_mode: CopyMode | None = None,
) -> tuple[ExecutionMode, CopyMode]:
    """Omitted options follow this leaf's plan; explicit options override it."""
    if plan.persistent is None:
        raise ValueError("persistent execution options require a persistent plan")
    resolved_mode = plan.persistent.lowering_mode if mode is None else mode
    resolved_copy = plan.persistent.copy_mode if copy_mode is None else copy_mode
    if resolved_mode not in ("wave", "fused", "physical"):
        raise ValueError(f"unknown mode {resolved_mode!r}")
    if resolved_copy not in ("scalar", "vectorized_contiguous"):
        raise ValueError(f"unknown copy_mode {resolved_copy!r}")
    return resolved_mode, resolved_copy


def _worker_body(
    stage: FFTStagePlan, worker_id: int, workers_per_fft: int, vector_compute_lanes: int | None,
) -> tuple[tuple, int | None]:
    """`(batches, compute_lanes)` for one LOGICAL worker (`0 ..
    workers_per_fft - 1`) on one stage -- the last logical worker owns
    this stage's own tail (partial) batch, if it has one, exclusively;
    every worker (the tail-owning one included) renders at
    `vector_compute_lanes` -- the same register-pressure-narrowed width
    `_stage_compute_lanes` already resolved for this stage (see
    `emit_stage_phase`'s own docstring for why skipping that narrowing
    entirely is not optional: a real crash + wrong answer, not just a
    spill warning, was confirmed on N=64's (4,4,4) middle stage before
    it was wired in). `workers_per_fft == workers_per_group` (today's
    only case before worker-wave virtualization existed) makes "logical"
    and "physical" worker id the same thing, unchanged.

    The tail-owning worker does **not** get a hardcoded `compute_lanes=1`
    ("genuine scalar arithmetic") the way the original design doc
    specified. That was tried first and is itself a confirmed, real bug:
    a radix-5 or radix-7 tail batch rendered at literal width=1 spills
    (272/432-byte frames, N=105 radices=(3,5,7)) regardless of function
    structure -- confirmed by isolating the tail-owning worker into its
    own standalone function (ruling out "shared function with other
    workers" as the cause) and by disassembly (byte-identical frame with
    or without that isolation). Root cause: at width=1, a radix-R
    butterfly's own `R` complex operands each become a separate scalar
    register with no way to pack more than one lane per register --
    genuinely more live scalar registers than this target provides for
    R>=5, independent of surrounding code. Rendering the SAME tail batch
    at `vector_compute_lanes` instead (identical to every other worker)
    uses this codebase's already-safe, already-proven `scalar_pack`/
    masked-lane mechanism (`fft_plan_core._make_load`'s own `valid_lanes
    < simd_lanes` branch, used everywhere else in this project for a
    partial batch) to render the same tail batch as a real vector
    register with some lanes masked, not literal per-lane scalars --
    packing multiple operands' worth of state per register the same way
    a full batch already does. Confirmed real-hardware: N=105
    radices=(3,5,7) is spill-free and correct with this fix, previously
    confirmed spilling at every compute_lanes tried (4, 2, 1, all forced
    fully scalar for the tail).
    """
    assert stage.persistent_vector_batches is not None
    assert stage.persistent_scalar_batches is not None
    if worker_id == workers_per_fft - 1 and stage.persistent_scalar_batches:
        return stage.persistent_scalar_batches, vector_compute_lanes
    return stage.persistent_vector_batches[worker_id], vector_compute_lanes


def _emit_persistent_batches(e: Emitter, *, plan: FFTCodegenPlan, stage: FFTStagePlan,
                             batches: tuple[SIMDBatchPlan, ...], compute_lanes: int | None,
                             twiddle_table: list[tuple[float, float]] | None) -> None:
    if plan.persistent.loop_stages:
        assert twiddle_table is not None
        # Workers within a stage write disjoint locations in another bank;
        # sorting changes only their local visit order, never ownership.
        loop = try_build_loop_stage(tuple(sorted(batches,key=lambda b:b.batch_id)),
            simd_lanes=plan.simd_lanes,min_full_batches=LOOP_MIN_FULL_BATCHES,
            twiddle_table=twiddle_table)
        if loop is not None:
            _emit_loop_stage(e,plan=plan,stage=stage,loop_plan=loop,compute_lanes=compute_lanes,
                             compact_residues=True)
            if loop.tail_batch is not None:
                _emit_stage_batches(e,plan=plan,stage=stage,batches=(loop.tail_batch,),compute_lanes=compute_lanes)
            return
    _emit_stage_batches(e,plan=plan,stage=stage,batches=batches,compute_lanes=compute_lanes)


def _emit_worker_dispatch(
    e: Emitter, *, plan: FFTCodegenPlan, stage: FFTStagePlan, workers_per_fft: int,
    vector_compute_lanes: int | None, dispatch_var: str = "worker_id",
    twiddle_table: list[tuple[float, float]] | None = None,
) -> None:
    """`if {dispatch_var} == 0: ... elif {dispatch_var} == 1: ... else: ...`
    -- one branch per LOGICAL worker (`0 .. workers_per_fft - 1`), each
    rendering that worker's own already-partitioned batches via the
    *exact same* `_emit_stage_batches` fft_codegen.py uses for the plain
    (non-persistent) path. Built in a sub-Emitter and re-indented by one
    level (the same idiom fft_transpose_codegen.py's own tail-branch
    emission uses) since `_emit_stage_batches`/`_emit_batch` hardcode an
    8-space base indent, one level shallower than this call site (inside
    an `if {dispatch_var} == k:` body, not directly inside a `@staticmethod
    def ...():`).

    `dispatch_var`: `"worker_id"` (the default, physical) when this stage
    has no worker-wave virtualization (`workers_per_fft ==
    workers_per_group`) -- unchanged from before this parameter existed.
    `emit_stage_phase` passes `"logical_worker_id"` instead once
    `worker_waves > 1`, so each branch is keyed on the LOGICAL worker id a
    physical worker computes for its current wave, not its own fixed
    physical id -- see `_emit_wave_prelude`.

    RAGGED WAVES (`workers_per_fft` not a multiple of `workers_per_group`,
    e.g. 6 or 10 against a physical width of 8 -- see `fft_plan_
    persistent.worker_waves`'s own docstring): this loop only ever emits
    branches for `dispatch_var == 0 .. workers_per_fft - 1`, and there is
    NO trailing `else`. A physical lane whose computed `dispatch_var`
    value is `>= workers_per_fft` -- which only happens on the single wave
    when `workers_per_fft <= workers_per_group`, or on the LAST wave
    otherwise -- matches none of these branches and falls straight through
    to this function's own end with no code executed at all: no load, no
    twiddle, no store, no scratchpad address computed. This is not new
    machinery added for the ragged case -- it is the exact same idiom this
    function already used for a worker whose own bucket happens to be
    empty (`if not batches: continue` above skips emitting that worker's
    branch too, for the SAME reason: nothing to do this call), now also
    covering a worker that has no logical identity at all this wave.
    """
    emitted_any = False
    for worker_id in range(workers_per_fft):
        batches, compute_lanes = _worker_body(
            stage, worker_id, workers_per_fft, vector_compute_lanes
        )
        if not batches:
            continue
        keyword = "if" if not emitted_any else "elif"
        emitted_any = True
        e.add(f"        {keyword} {dispatch_var} == {worker_id}:")
        sub = Emitter()
        _emit_persistent_batches(
            sub, plan=plan, stage=stage, batches=batches, compute_lanes=compute_lanes,
            twiddle_table=twiddle_table,
        )
        for line in sub.lines:
            e.add("    " + line if line else "")


def _flatten_to_physical_lanes(
    stage: FFTStagePlan, *, workers_per_fft: int, workers_per_group: int,
) -> tuple[tuple[SIMDBatchPlan, ...], ...]:
    """Mode C's own core transformation: collapse `workers_per_fft` LOGICAL
    workers' worth of already-partitioned batches (`stage.persistent_
    vector_batches`, `stage.persistent_scalar_batches` -- built by
    `_partition_vector_scalar`, unchanged by this function) down to
    exactly `workers_per_group` (8) PHYSICAL buckets, at CODEGEN TIME, so
    the emitted dispatch never needs a runtime `logical_worker_id` at all.

    Physical lane `p`'s own bucket is the concatenation, in ascending
    logical-worker order, of every logical worker `j`'s own batches where
    `j % workers_per_group == p` -- i.e. exactly the set of logical
    workers Mode B's `while logical_worker_id < workers_per_fft:
    logical_worker_id += workers_per_group` loop would have visited on
    physical lane `p`, just computed once, in Python, instead of by a
    runtime loop. The scalar/tail batch (owned by logical worker
    `workers_per_fft - 1` alone, see `_partition_vector_scalar`'s own
    docstring) lands on whichever physical lane that logical worker maps
    to (`(workers_per_fft - 1) % workers_per_group`) and is appended AFTER
    that lane's own vector batches, matching `_worker_body`'s existing
    "tail batch owned by the last logical worker, rendered like every
    other worker's batches" contract exactly.

    Concatenating multiple batches into one physical lane's own branch is
    not new capability invented here: `_emit_stage_batches` (called on the
    result, exactly as `_emit_worker_dispatch` already calls it on a
    single logical worker's own batches) already renders an arbitrary-
    length batch tuple correctly -- this is the exact same code path a
    persistent leaf with more SIMD batches than `workers_per_group`
    already exercises today whenever `workers_per_fft <=
    workers_per_group` (the untouched, pre-existing case). Safe to
    concatenate in any order (chosen here: ascending logical-worker order,
    for a deterministic, easy-to-audit mapping) because every batch is
    already fully independent within one stage -- see `emit_stage_phase`'s
    own docstring for why (disjoint scratchpad offsets, no same-stage
    producer/consumer relationship between logical workers at all).

    REVISED 2026-09-22: the actual flattening (the loop over `logical_
    worker_id` below) moved to `planning.execution.fft_plan_persistent.
    flatten_logical_workers_to_physical_lanes` -- a pure, planning-layer
    function `planning.search.fft_cost_model` now calls too, so a
    persistent stage's own cost estimate is measured against the exact
    same physical-lane mapping this codegen renders, instead of a second,
    independently-computed copy silently drifting out of sync with it (see
    docs/logical_vs_physical_cost_model.md). This function is now a thin,
    codegen-local wrapper (assert the plan-shape invariants codegen itself
    depends on, unpack `stage.persistent_vector_batches`/`_scalar_batches`,
    delegate) -- byte-identical output to before this refactor for every
    existing caller.
    """
    assert stage.persistent_vector_batches is not None
    assert stage.persistent_scalar_batches is not None
    assert len(stage.persistent_vector_batches) == workers_per_fft
    return flatten_logical_workers_to_physical_lanes(
        stage.persistent_vector_batches, stage.persistent_scalar_batches,
        workers_per_fft=workers_per_fft, workers_per_group=workers_per_group,
    )


def _emit_physical_lane_dispatch(
    e: Emitter, *, plan: FFTCodegenPlan, stage: FFTStagePlan, workers_per_fft: int,
    workers_per_group: int, vector_compute_lanes: int | None,
    twiddle_table: list[tuple[float, float]] | None = None,
) -> None:
    """Mode C's own dispatch emission: `if worker_id == 0: ... elif
    worker_id == 1: ... ` over exactly `workers_per_group` (8) PHYSICAL
    branches -- never more, regardless of `workers_per_fft` -- each
    rendering that lane's own FLATTENED batch list (see
    `_flatten_to_physical_lanes`) via the exact same `_emit_stage_batches`
    every other dispatch shape in this module already uses. No runtime
    `logical_worker_id`, no loop, no W-sized branch tree: this is
    textually the SAME shape `_emit_worker_dispatch` already renders for
    `workers_per_fft <= workers_per_group` (today's untouched case) --
    reusing that exact idiom, just fed pre-flattened physical-lane
    batches instead of raw per-logical-worker ones.
    """
    physical_batches = _flatten_to_physical_lanes(
        stage, workers_per_fft=workers_per_fft, workers_per_group=workers_per_group,
    )
    emitted_any = False
    for worker_id in range(workers_per_group):
        batches = physical_batches[worker_id]
        if not batches:
            continue
        keyword = "if" if not emitted_any else "elif"
        emitted_any = True
        e.add(f"        {keyword} worker_id == {worker_id}:")
        sub = Emitter()
        _emit_persistent_batches(
            sub, plan=plan, stage=stage, batches=batches, compute_lanes=vector_compute_lanes,
            twiddle_table=twiddle_table,
        )
        for line in sub.lines:
            e.add("    " + line if line else "")


def _wave_tracker_name(plan: FFTCodegenPlan) -> str:
    return _spad(plan.kernel_name, "wave_tracker")


def _emit_wave_prelude(e: Emitter, *, plan: FFTCodegenPlan, workers_per_group: int) -> None:
    """Read this physical worker's private wave counter."""
    tracker = _wave_tracker_name(plan)
    e.add(f"        var wave_index = Int({tracker}.load[DType.float32, 1](worker_id)[0])")
    e.add(f"        var logical_worker_id = wave_index * {workers_per_group} + worker_id")


def _emit_wave_bump(e: Emitter, *, plan: FFTCodegenPlan, worker_waves: int) -> None:
    """Advance the caller's counter, including lanes idle in a ragged wave.

    Each worker owns its cell. The phase completion synchronizes these
    counters for the next invocation without same-phase shared-cell races.
    """
    tracker = _wave_tracker_name(plan)
    e.add(f"        var next_wave = (wave_index + 1) % {worker_waves}")
    e.add(f"        {tracker}.store(worker_id, Float32(next_wave))")


def _round_tracker_name(plan: FFTCodegenPlan) -> str:
    return _spad(plan.kernel_name, "round_tracker")


def emit_tracker_initialization(e: Emitter, *, plan: FFTCodegenPlan, mode: ExecutionMode) -> None:
    """Controller initializes every unit before dispatch, without a kernel slot."""
    pw = plan.persistent
    trackers = ["round_tracker"]
    if mode == "wave" and _worker_waves(pw.workers_per_fft, pw.workers_per_group) > 1:
        trackers.append("wave_tracker")
    e.add(f"        for tracker_group in range({pw.software_group_count}):")
    for tracker in trackers:
        e.add(f"            var {tracker}_ptr = spad_addr({plan.kernel_name}.{tracker}, tracker_group)")
    e.add(f"            for tracker_worker in range({pw.workers_per_group}):")
    for tracker in trackers:
        e.add(f"                {tracker}_ptr.store(tracker_worker, Float32(0))")


def _emit_dispatch_prelude(
    e: Emitter, *, plan: FFTCodegenPlan, workers_per_group: int,
    software_group_count: int, num_logical_blocks: int, bump_round: bool,
) -> None:
    """Derive the software group and worker, then read this worker's
    private round cell. Writeback advances the same private cell; phase
    completion orders the next round after every worker has finished.
    """
    tracker = _round_tracker_name(plan)
    e.add("        var gid = global_uthread_id()")
    e.add(f"        var software_group_id = gid // {workers_per_group}")
    e.add(f"        var worker_id = gid % {workers_per_group}")
    e.add(f"        var round_index = Int({tracker}.load[DType.float32, 1](worker_id)[0])")
    e.add(f"        var round_base = round_index * {software_group_count}")
    e.add(f"        var active_groups = {software_group_count}")
    e.add(f"        var remaining_blocks = {num_logical_blocks} - round_base")
    e.add("        if remaining_blocks < active_groups:")
    e.add("            active_groups = remaining_blocks")
    e.add("        if software_group_id >= active_groups:")
    e.add("            return")


def emit_stage_phase(
    *,
    plan: FFTCodegenPlan,
    stage: FFTStagePlan,
    software_group_count: int,
    num_logical_blocks: int,
    workers_per_group: int,
    workers_per_fft: int | None = None,
    compute_lanes: int | None = 4,
    narrow_middle_stages: bool = True,
    mode: ExecutionMode = "physical",
    twiddle_table: list[tuple[float, float]] | None = None,
) -> tuple[str, list[str]]:
    """Builds this phase's own fresh `Emitter` and returns `(name, lines)`
    -- the same "translate exactly what would be emitted, from a small
    self-contained snippet" shape verify_fft_harness._translate_stage
    already uses, so verification/verify_fft_persistent.py can translate
    and exec these lines directly without re-deriving anything. One
    function total per stage (not one per round) -- see this module's own
    top docstring for why.

    `compute_lanes`/`narrow_middle_stages`: `stage.compute_lanes` wins
    when the planner already set it (see `FFTStagePlan.compute_lanes`'s
    own docstring), applied to every worker -- including the last (tail-
    owning) worker, which no longer gets a hardcoded `compute_lanes=1`
    (see `_worker_body`'s own docstring for why forcing that was itself
    the bug behind a real, confirmed radix-5/radix-7 spill, root-caused
    and fixed 2026-08-30). Only when `stage.compute_lanes` is `None` does
    this fall back to resolving through the *exact same* `_stage_compute_
    lanes` fft_codegen.py's plain/cooperative paths use. Defaults (`4`/
    `True`) match make_fft_kernel.py's own shipped defaults, not "no
    narrowing" -- skipping this entirely was tried first and produced a
    real crash (`vs2r.v: Unsupported Instruction`) plus a wrong answer on
    N=64's own (4,4,4) middle stage, confirmed on real hardware; see
    docs/persistent_leaf_design.md's own "Register-pressure discipline"
    section, which requires reusing this mechanism rather than treating a
    persistent-leaf spill as merely a performance caveat.

    `workers_per_fft`: `None` (the default) means "equal to
    `workers_per_group`" -- the plain one-wave case, byte-identical
    codegen to every call site that predates this parameter. Any other
    positive value is accepted, including one that is neither a divisor
    nor a multiple of `workers_per_group` (a "ragged" cooperation width,
    e.g. 6 or 10 against a physical width of 8) or an exact multiple
    greater than it (e.g. 36) -- see `fft_plan_persistent.worker_waves`'s
    own docstring for the `ceil(workers_per_fft / workers_per_group)`
    formula (`worker_waves`, kept as a pure metadata/diagnostics helper --
    see its own docstring for why it is NOT used to control runtime
    dispatch here any more).

    `mode`: which of the three execution-lowering strategies documented at
    this module's own top (`ExecutionMode`) to render this stage with --
    `"wave"` (Mode A, original, one `launch_parallel` per wave -- this
    function only renders the STAGE FUNCTION BODY the same way regardless
    of mode; `emit_persistent_kernel_struct` is what actually decides how
    many times `device_main` calls it), `"fused"` (Mode B, the 2026-09-14
    worker-wave-fusion revision: one call, a runtime `while
    logical_worker_id < workers_per_fft:` loop around the *exact same*
    `if/elif` dispatch chain `_emit_worker_dispatch` built for the
    original `workers_per_fft <= workers_per_group` case), or
    `"physical"` (Mode C, the default: one call, `workers_per_fft`
    logical workers flattened to `workers_per_group` physical buckets at
    CODEGEN TIME via `_flatten_to_physical_lanes`, dispatched with a
    plain `workers_per_group`-way `if/elif`, no runtime logical id, no
    loop, no W-sized branch tree). `workers_per_fft <= workers_per_group`
    collapses ALL THREE modes to the exact same one-call, no-loop,
    `workers_per_group`-way dispatch (today's original, untouched shape)
    -- this parameter, and every difference between modes, is only
    observable once `workers_per_fft > workers_per_group`.
    """
    is_first = stage.stage_id == 0
    is_last = stage.stage_id == len(plan.stages) - 1
    prev_radix = plan.stages[stage.stage_id - 1].radix if not is_first else None
    if stage.compute_lanes is not None:
        vector_compute_lanes = stage.compute_lanes
    else:
        vector_compute_lanes = _stage_compute_lanes(
            compute_lanes=compute_lanes, is_first=is_first, is_last=is_last, radix=stage.radix,
            narrow_middle_stages=narrow_middle_stages, prev_radix=prev_radix,
        )

    if workers_per_fft is None:
        workers_per_fft = workers_per_group
    waves = _worker_waves(workers_per_fft, workers_per_group)
    if twiddle_table is None and plan.persistent.loop_stages:
        twiddle_table = []

    e = Emitter()
    name = f"stage_{stage.stage_id}"
    e.add("    @staticmethod")
    e.add(f"    def {name}():")
    e.add(f"        ref p = {plan.kernel_name}.params[]")
    e.add(f"        comptime RADIX = {stage.radix}")
    _emit_dispatch_prelude(
        e, plan=plan, workers_per_group=workers_per_group,
        software_group_count=software_group_count, num_logical_blocks=num_logical_blocks,
        bump_round=False,
    )
    if plan.persistent.storage_mode == "global":
        e.add("        var logical_block = round_base + software_group_id")
        e.add(f"        var in_batch_base = logical_block * {plan.length}")
        e.add("        var out_batch_base = in_batch_base")
        e.add(f"        var spad_base = software_group_id * {2 * plan.length}")
    else:
        e.add("        var spad_base = 0")
    e.add()
    if waves <= 1:
        # All three modes collapse to this one shape when there is only
        # one wave -- no mode-specific machinery ever emitted.
        _emit_worker_dispatch(
            e, plan=plan, stage=stage, workers_per_fft=workers_per_fft,
            vector_compute_lanes=vector_compute_lanes, dispatch_var="worker_id",
            twiddle_table=twiddle_table,
        )
    elif mode == "physical":
        # Mode C: flattened at codegen time, no runtime logical id at all.
        _emit_physical_lane_dispatch(
            e, plan=plan, stage=stage, workers_per_fft=workers_per_fft,
            workers_per_group=workers_per_group, vector_compute_lanes=vector_compute_lanes,
            twiddle_table=twiddle_table,
        )
    elif mode == "fused":
        # Mode B: one physical lane visits several logical workers in a
        # plain runtime loop, no cross-call scratchpad state, no repeated
        # launch_parallel.
        e.add("        var logical_worker_id = worker_id")
        e.add(f"        while logical_worker_id < {workers_per_fft}:")
        sub = Emitter()
        _emit_worker_dispatch(
            sub, plan=plan, stage=stage, workers_per_fft=workers_per_fft,
            vector_compute_lanes=vector_compute_lanes, dispatch_var="logical_worker_id",
            twiddle_table=twiddle_table,
        )
        for line in sub.lines:
            e.add("    " + line if line else "")
        e.add(f"            logical_worker_id += {workers_per_group}")
    elif mode == "wave":
        # Mode A: this call handles exactly ONE wave, wave_index read from
        # the cross-call scratchpad wave_tracker (bumped at the end, once
        # this call's own dispatched body is done) -- device_main (see
        # emit_persistent_kernel_struct) is what actually calls this
        # function `waves` times in a row for this to mean anything.
        _emit_wave_prelude(e, plan=plan, workers_per_group=workers_per_group)
        e.add()
        _emit_worker_dispatch(
            e, plan=plan, stage=stage, workers_per_fft=workers_per_fft,
            vector_compute_lanes=vector_compute_lanes, dispatch_var="logical_worker_id",
            twiddle_table=twiddle_table,
        )
        e.add()
        _emit_wave_bump(e, plan=plan, worker_waves=waves)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    e.add()
    e.add()
    return name, e.lines


def emit_bulk_copy_phase(
    *,
    plan: FFTCodegenPlan,
    name: str,
    software_group_count: int,
    num_logical_blocks: int,
    workers_per_group: int,
    to_scratchpad: bool,
    buffer_name: str,
    bump_round: bool,
    copy_mode: CopyMode = "scalar",
    vector_width: int = 4,
) -> tuple[str, list[str]]:
    """Preload (`to_scratchpad=True`) or writeback (`False`) between DRAM
    (`block_base = logical_block * length`, natural order -- no FFT
    permutation on either side) and that group's own scratchpad bank.

    `copy_mode="scalar"` (the default, byte-identical to every plan
    rendered before this parameter existed): every worker in an active
    software group copies a disjoint, STRIDED slice (`worker_id, worker_
    id + workers_per_group, ...`), one element at a time (`.load[width=
    1]`/`.store`). Deliberately fully scalar -- correctness-first per
    docs/persistent_leaf_design.md's own priority order.

    `copy_mode="vectorized_contiguous"` (P2.3 -- see docs/
    priority2_execution_strategies.md): each worker instead owns one
    CONTIGUOUS block of `ceil(length / workers_per_group)` elements
    (`chunk_start = worker_id * chunk`, clamped to `length`), copied
    `vector_width` elements at a time via the SAME `.load[width=V]`/
    inferred-width `.store` idiom `codegen.fft_codegen._emit_load`/
    `_emit_store` already use for FFT-stage operands elsewhere in this
    project (never a new one), with a scalar (`width=1`) tail loop for
    whatever doesn't divide evenly by `vector_width`. `chunk` is a
    PYTHON-level (compile-time) constant -- `workers_per_group` and
    `plan.length` are both already known at codegen time -- so no new
    runtime division is introduced; only `worker_id * chunk` (one
    multiply) is a runtime expression, exactly as cheap as the scalar
    mode's own `i = worker_id` seed.

    CORRECTNESS: changing WHICH physical lane copies WHICH element index
    is provably safe regardless of `copy_mode` -- every stage that later
    reads this scratchpad bank addresses it by the element's own ABSOLUTE
    index (`_make_load`'s `force_scratchpad` branch, `base_offset`
    already an absolute position within this kernel -- see fft_plan_core.
    AddressMapping's own docstring), never by which lane originally wrote
    it. Both modes are also full BIJECTIONS on `[0, length)` by
    construction (scalar: `{worker_id + k*workers_per_group : k >= 0,
    result < length}` partitions cleanly by residue mod `workers_per_
    group`; vectorized_contiguous: `[worker_id*chunk, min(length, (worker_
    id+1)*chunk))` partitions cleanly by consecutive non-overlapping
    ranges) -- every index visited by exactly one lane, in either mode.
    Ping-pong buffer selection, twiddle placement, and preload-before-
    compute/writeback-after-compute ordering (`device_main`'s own
    `launch_parallel` sequence) are completely unaffected by this
    parameter -- it only changes the ADDRESS PATTERN and INSTRUCTION
    WIDTH of one bulk copy, nothing about the FFT math around it.

    `bump_round`: writeback advances this worker's private round cell after
    its copy. Other workers read separate cells, and phase completion orders
    the next preload after every worker has completed its writeback.
    """
    validate_copy_width(copy_mode, vector_width)
    e = Emitter()
    e.add("    @staticmethod")
    e.add(f"    def {name}():")
    e.add(f"        ref p = {plan.kernel_name}.params[]")
    _emit_dispatch_prelude(
        e, plan=plan, workers_per_group=workers_per_group,
        software_group_count=software_group_count, num_logical_blocks=num_logical_blocks,
        bump_round=bump_round,
    )
    if plan.persistent.storage_mode == "global":
        # Stages access input/output directly. The only extra phase advances
        # private counters, including inactive logical workers, after every
        # final-stage worker wave has completed.
        if not bump_round:
            raise ValueError("global persistent execution has no preload phase")
        tracker = _round_tracker_name(plan)
        e.add(f"        {tracker}.store(worker_id, Float32(round_index) + Float32(1))")
        e.add()
        e.add()
        return name, e.lines
    e.add("        var logical_block = round_base + software_group_id")
    e.add(f"        var block_base = logical_block * {plan.length}")
    buf = _spad(plan.kernel_name, buffer_name)

    if copy_mode == "scalar":
        e.add("        var i = worker_id")
        e.add(f"        while i < {plan.length}:")
        if to_scratchpad:
            e.add("            var vr = p.input_real_base.load[width=1](block_base + i)[0]")
            e.add("            var vi = p.input_imag_base.load[width=1](block_base + i)[0]")
            e.add(f"            {buf}.store(i, vr)")
            e.add(f"            {buf}.store({plan.length} + i, vi)")
        else:
            e.add(f"            var vr = {buf}.load[DType.float32, 1](i)[0]")
            e.add(f"            var vi = {buf}.load[DType.float32, 1]({plan.length} + i)[0]")
            e.add("            p.output_real_base.store(block_base + i, vr)")
            e.add("            p.output_imag_base.store(block_base + i, vi)")
        e.add(f"            i += {workers_per_group}")
    elif copy_mode == "vectorized_contiguous":
        chunk = -(-plan.length // workers_per_group)  # Python-level ceil division
        e.add(f"        var chunk_start = worker_id * {chunk}")
        e.add(f"        var chunk_end = chunk_start + {chunk}")
        e.add(f"        if chunk_end > {plan.length}:")
        e.add(f"            chunk_end = {plan.length}")
        e.add("        var i = chunk_start")
        e.add(f"        while i + {vector_width} <= chunk_end:")
        if to_scratchpad:
            e.add(f"            var vr = p.input_real_base.load[width={vector_width}](block_base + i)")
            e.add(f"            var vi = p.input_imag_base.load[width={vector_width}](block_base + i)")
            e.add(f"            {buf}.store(i, vr)")
            e.add(f"            {buf}.store({plan.length} + i, vi)")
        else:
            e.add(f"            var vr = {buf}.load[DType.float32, {vector_width}](i)")
            e.add(f"            var vi = {buf}.load[DType.float32, {vector_width}]({plan.length} + i)")
            e.add("            p.output_real_base.store(block_base + i, vr)")
            e.add("            p.output_imag_base.store(block_base + i, vi)")
        e.add(f"            i += {vector_width}")
        e.add("        while i < chunk_end:")
        if to_scratchpad:
            e.add("            var svr = p.input_real_base.load[width=1](block_base + i)[0]")
            e.add("            var svi = p.input_imag_base.load[width=1](block_base + i)[0]")
            e.add(f"            {buf}.store(i, svr)")
            e.add(f"            {buf}.store({plan.length} + i, svi)")
        else:
            e.add(f"            var svr = {buf}.load[DType.float32, 1](i)[0]")
            e.add(f"            var svi = {buf}.load[DType.float32, 1]({plan.length} + i)[0]")
            e.add("            p.output_real_base.store(block_base + i, svr)")
            e.add("            p.output_imag_base.store(block_base + i, svi)")
        e.add("            i += 1")
    else:
        raise ValueError(f"unknown copy_mode {copy_mode!r}")

    if bump_round:
        tracker = _round_tracker_name(plan)
        e.add(f"        {tracker}.store(worker_id, Float32(round_index) + Float32(1))")
    e.add()
    e.add()
    return name, e.lines


def _emit_params_struct(e: Emitter, *, plan: FFTCodegenPlan) -> None:
    e.add("@fieldwise_init")
    e.add(f"struct {plan.kernel_name}Params(Movable):")
    e.add("    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    if plan.persistent.loop_stages:
        e.add("    var loop_twiddle_real_base: UnsafePointer[Float32, MutAnyOrigin]")
        e.add("    var loop_twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    for buffer in plan.workspace_buffers:
        e.add(f"    var {buffer.name}: UnsafePointer[Float32, MutAnyOrigin]")
    e.add()
    e.add()


def emit_persistent_kernel_struct(
    e: Emitter, *, plan: FFTCodegenPlan, num_logical_blocks: int,
    compute_lanes: int | None = 4, narrow_middle_stages: bool = True,
    mode: ExecutionMode | None = None,
    copy_mode: CopyMode | None = None, vector_width: int = 4,
    max_kernel_register: int = DEFAULT_TARGET_PROFILE.max_kernel_register,
    spad_capacity_bytes: int | None = None,
) -> list[tuple[float, float]]:
    """Append one `NDPTask` struct for a persistent-software-workgroup FFT
    leaf to `e`: `Params`, `buf_a`/`buf_b`/`round_tracker` scratchpad,
    exactly `2 + len(plan.stages)` distinct kernel functions (`preload`,
    `stage_0`, ..., `writeback`; see this module's own top docstring for
    why round-specific functions don't work), and a `device_main` calling
    that same small function set once per round in the right order.

    Factored out of `generate_persistent_fft_kernel` so a caller building
    a *shared* multi-kernel host `main()` (`fft_transpose_codegen.
    generate_recursive_fft_kernels`, once a leaf in a recursive split
    opts into persistent execution) can emit this one leaf's own struct
    inline, the same way it already calls `_emit_kernel`/
    `_emit_physical_transpose_kernel` for the other two kernel shapes it
    knows how to render -- without also getting a second, standalone
    `main()` this function no longer emits. `generate_persistent_fft_
    kernel` (below) is now a thin wrapper: this struct plus its own
    self-contained host `main()`, unchanged in output for every existing
    caller.

    `plan` must come from `planning.execution.fft_plan_persistent.
    make_persistent_leaf_plan` (i.e. `plan.persistent is not None`);
    `num_logical_blocks` must match what that call was given (checked
    below, not re-derived, since `plan.host.total_elems` only encodes
    `length * num_logical_blocks` jointly).

    `mode`/`copy_mode`: omitted (`None`) uses this leaf's planned settings.
    Explicit values override independently without mutating the plan.

    `compute_lanes`/`narrow_middle_stages`: forwarded to `emit_stage_
    phase` -- see its own docstring for why the defaults (`4`/`True`,
    make_fft_kernel.py's own shipped defaults) are load-bearing here, not
    just a performance knob.
    """
    if plan.persistent is None:
        raise ValueError("emit_persistent_kernel_struct requires a persistent plan")
    if plan.host.total_elems != plan.length * num_logical_blocks:
        raise ValueError(
            f"num_logical_blocks={num_logical_blocks} is inconsistent with "
            f"plan.host.total_elems={plan.host.total_elems} for length={plan.length}"
        )

    mode, copy_mode = resolve_persistent_options(plan, mode=mode, copy_mode=copy_mode)
    validate_copy_width(copy_mode, vector_width)
    pw = plan.persistent
    workers_per_group = pw.workers_per_group
    workers_per_fft = pw.workers_per_fft
    waves = _worker_waves(workers_per_fft, workers_per_group)
    software_group_count = pw.software_group_count
    rounds = num_rounds(num_logical_blocks, software_group_count)
    uses_spad = pw.storage_mode == "scratchpad"
    extra_phases = 2 if uses_spad else 1
    tracker_bytes = persistent_tracker_scratchpad_bytes(workers_per_group, mode=mode,
                                                       workers_per_fft=workers_per_fft)
    scratchpad_bytes = sum(b.elements * 4 for b in plan.scratchpad_buffers) + tracker_bytes
    if spad_capacity_bytes is not None and scratchpad_bytes > spad_capacity_bytes:
        raise ValueError(f"persistent execution needs {scratchpad_bytes} scratchpad bytes including private trackers; "
                         f"capacity is {spad_capacity_bytes}")
    emit_execution_record(e, plan=plan, persistent_mode=mode, copy_mode=copy_mode,
                          vector_width=vector_width if copy_mode == "vectorized_contiguous" else 1,
                          compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
                          scratchpad_bytes=scratchpad_bytes, tracker_bytes=tracker_bytes,
                          host_launches=1, internal_rounds=rounds,
                          storage_mode=pw.storage_mode,
                          loop_stages=pw.loop_stages,
                          workspace_bytes=sum(b.elements * 4 for b in plan.workspace_buffers),
                          dram_bytes=plan.estimated_dram_bytes,
                          registered_functions=extra_phases + len(plan.stages),
                          phase_launches=rounds * (extra_phases + len(plan.stages) * (waves if mode == "wave" else 1)))
    from planning.execution.kernel_registration import validate_kernel_registration
    validate_kernel_registration(plan, max_kernel_register=max_kernel_register)

    buffer_names = tuple(b.name for b in plan.scratchpad_buffers) if uses_spad else ("buf_a", "buf_b")
    assert buffer_names == ("buf_a", "buf_b"), buffer_names
    final_buffer = buffer_names[len(plan.stages) % 2]

    _emit_params_struct(e, plan=plan)

    e.add(f"struct {plan.kernel_name}(NDPTask):")
    e.add(f"    comptime Params = {plan.kernel_name}Params")
    e.add()
    for buffer in plan.scratchpad_buffers:
        e.add(
            f'    comptime {buffer.name} = scratchpad[{buffer.elements}, Float32, '
            f'name="{plan.kernel_name.lower()}_{buffer.name}"]()'
        )
    # Scratchpad is reused by other tasks; initialize private counters on
    # every task entry rather than relying on simulator startup contents.
    e.add(
        f'    comptime round_tracker = scratchpad[{workers_per_group}, Float32, '
        f'name="{plan.kernel_name.lower()}_round_tracker"]()'
    )
    # `wave_tracker` only exists for Mode A ("wave") with more than one
    # wave -- Modes B ("fused") and C ("physical") resolve every logical
    # worker a physical lane owns inside ONE stage invocation (a runtime
    # loop for B, a compile-time-flattened static dispatch for C), so
    # neither needs any cross-call state persisted in scratchpad.
    if mode == "wave" and waves > 1:
        e.add(
            f'    comptime wave_tracker = scratchpad[{workers_per_group}, Float32, '
            f'name="{plan.kernel_name.lower()}_wave_tracker"]()'
        )
    e.add()

    if uses_spad:
        preload_name, lines = emit_bulk_copy_phase(
            plan=plan, name="preload", software_group_count=software_group_count,
            num_logical_blocks=num_logical_blocks, workers_per_group=workers_per_group,
            to_scratchpad=True, buffer_name=buffer_names[0], bump_round=False,
            copy_mode=copy_mode, vector_width=vector_width,
        )
        e.lines.extend(lines)

    stage_names: list[str] = []
    twiddle_table: list[tuple[float, float]] = []
    for stage in plan.stages:
        name, lines = emit_stage_phase(
            plan=plan, stage=stage, software_group_count=software_group_count,
            num_logical_blocks=num_logical_blocks, workers_per_group=workers_per_group,
            workers_per_fft=workers_per_fft,
            compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
            mode=mode,
            twiddle_table=twiddle_table,
        )
        e.lines.extend(lines)
        stage_names.append(name)

    writeback_name, lines = emit_bulk_copy_phase(
        plan=plan, name="writeback", software_group_count=software_group_count,
        num_logical_blocks=num_logical_blocks, workers_per_group=workers_per_group,
        to_scratchpad=False, buffer_name=final_buffer, bump_round=True,
        copy_mode=copy_mode, vector_width=vector_width,
    )
    e.lines.extend(lines)

    # Mode A ("wave") is the only mode where device_main itself must call
    # a stage's own phase function more than once per round (`waves`
    # separate launch_parallel calls, each one wave) -- Modes B and C
    # resolve every logical worker inside a single call, so both call
    # each stage exactly once per round regardless of workers_per_fft.
    stage_repeat = waves if (mode == "wave" and waves > 1) else 1

    e.add("    @staticmethod")
    e.add("    def device_main():")
    emit_tracker_initialization(e, plan=plan, mode=mode)
    for _ in range(rounds):
        if uses_spad:
            e.add(f"        launch_parallel[{plan.kernel_name}.{preload_name}]()")
        for stage_name in stage_names:
            for _ in range(stage_repeat):
                e.add(f"        launch_parallel[{plan.kernel_name}.{stage_name}]()")
        e.add(f"        launch_parallel[{plan.kernel_name}.{writeback_name}]()")
    e.add()
    e.add()
    return twiddle_table


def generate_persistent_fft_kernel(
    plan: FFTCodegenPlan, *, num_logical_blocks: int,
    compute_lanes: int | None = 4, narrow_middle_stages: bool = True,
    mode: ExecutionMode | None = None,
    copy_mode: CopyMode | None = None, vector_width: int = 4,
    target: TargetProfile = DEFAULT_TARGET_PROFILE, reference_check: bool = True,
) -> str:
    """Render a full persistent-software-workgroup FFT: `emit_persistent_
    kernel_struct`'s one `NDPTask` struct, plus this function's own
    self-contained host `main()` (one `.launch()`, buffer alloc/fill,
    reference check) -- the standalone single-file entry point every
    existing caller of this function still gets unchanged. See
    `emit_persistent_kernel_struct`'s own docstring for the struct-only
    half of this, now shared with `fft_transpose_codegen.
    generate_recursive_fft_kernels`.

    `mode`: omitted (`None`) follows `plan.persistent.lowering_mode`; an
    explicit value overrides it. Forwarded to `emit_persistent_kernel_struct`/
    `emit_stage_phase` -- see this module's own top-of-file `ExecutionMode` docstring
    for what `"wave"`/`"fused"`/`"physical"` each render.

    `copy_mode`/`vector_width` (P2.3): forwarded to `emit_persistent_
    kernel_struct`/`emit_bulk_copy_phase` -- see that function's own
    docstring for `"scalar"` vs. `"vectorized_contiguous"`. Omitted
    `copy_mode` follows the plan independently of the execution-mode override.
    """
    e = Emitter()
    _emit_prelude(e)
    e.add(f"comptime N = {plan.length}")
    e.add()
    loop_table = emit_persistent_kernel_struct(
        e, plan=plan, num_logical_blocks=num_logical_blocks,
        compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
        mode=mode, copy_mode=copy_mode, vector_width=vector_width,
        max_kernel_register=target.max_kernel_register,
        spad_capacity_bytes=target.spad_capacity_bytes,
    )

    host = plan.host
    e.add("def main() raises:")
    e.add(f"    if {plan.kernel_name}.emit_ir_if_asked():")
    e.add("        return")
    e.add()
    e.add(f"    var total_elems = {host.total_elems}")
    e.add("    var input_real = cxl_alloc[Float32](total_elems)")
    e.add("    var input_imag = cxl_alloc[Float32](total_elems)")
    e.add("    var output_real = cxl_alloc[Float32](total_elems)")
    e.add("    var output_imag = cxl_alloc[Float32](total_elems)")
    e.add("    var ref_real = cxl_alloc[Float32](total_elems)")
    e.add("    var ref_imag = cxl_alloc[Float32](total_elems)")
    e.add()
    e.add("    seed(0)")
    e.add("    for i in range(total_elems):")
    e.add("        input_real[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        input_imag[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        output_real[i] = Float32(0)")
    e.add("        output_imag[i] = Float32(0)")
    e.add("        ref_real[i] = Float32(0)")
    e.add("        ref_imag[i] = Float32(0)")
    e.add()

    # Individual cxl_alloc buffers only promise 64-byte alignment. Align
    # the launch pool to this target's interleave chunk before mapping IDs.
    e.add(f"    var pool_elems = {host.pool_elems}")
    emit_aligned_pool(e, name="uthread_pool", elems="pool_elems",
                      alignment_bytes=target.mapping_stride_bytes)
    e.add()

    loop_args = _emit_persistent_loop_table(e,plan=plan,table=loop_table)
    workspace_args = emit_workspace_buffers(e, plan=plan)
    params_args = ["input_real", "input_imag", "output_real", "output_imag", *loop_args, *workspace_args]
    e.add(f"    var rc = {plan.kernel_name}.launch(")
    e.add("        PooledRange.over(uthread_pool, pool_elems),")
    e.add(
        f"        {plan.kernel_name}Params({', '.join(params_args)}),"
    )
    e.add("    )")
    e.add()
    e.add("    if rc != 0:")
    e.add('        print("[host] persistent FFT failed, exit", rc)')
    e.add("        return")
    e.add()

    if reference_check:
        _emit_reference_check(
            e,
            n=plan.length,
            batch_count=num_logical_blocks,
            inverse=plan.inverse,
            input_real="input_real",
            input_imag="input_imag",
            output_real="output_real",
            output_imag="output_imag",
            ref_real="ref_real",
            ref_imag="ref_imag",
            tolerance=host.tolerance,
            label="persistent FFT",
        )
    else:
        _emit_array_dump(e, n=host.total_elems, real_name="input_real", imag_name="input_imag", marker="INPUT")
        _emit_array_dump(e, n=host.total_elems, real_name="output_real", imag_name="output_imag", marker="OUTPUT")


    return e.text()


def _emit_persistent_loop_table(e: Emitter, *, plan: FFTCodegenPlan,
                                table: list[tuple[float,float]], prefix: str = "") -> list[str]:
    if not plan.persistent.loop_stages:
        return []
    real, imag = f"{prefix}loop_twiddle_real", f"{prefix}loop_twiddle_imag"
    return emit_loop_twiddle_table(e,table=table,real_name=real,imag_name=imag)


def _emit_persistent_buffers_and_launch(
    e: Emitter, *, plan: FFTCodegenPlan, num_logical_blocks: int, prefix: str, reference_check: bool = True,
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
    loop_table: list[tuple[float,float]] | None = None,
) -> None:
    """The exact buffer-alloc/pool-alignment/launch/reference-check block
    `generate_persistent_fft_kernel`'s own host `main()` already emits,
    factored out so `generate_persistent_tail_hybrid_kernel` (P2.4) can
    render it into a SHARED host `main()` alongside a second, independent
    kernel's own block -- `prefix` (e.g. `"p_"`) namespaces every host
    variable so the two blocks never collide in that shared scope. Emits
    byte-identical Mojo (modulo the variable-name prefix) to the pre-
    existing single-kernel host body."""
    host = plan.host
    e.add(f"    var {prefix}total_elems = {host.total_elems}")
    e.add(f"    var {prefix}input_real = cxl_alloc[Float32]({prefix}total_elems)")
    e.add(f"    var {prefix}input_imag = cxl_alloc[Float32]({prefix}total_elems)")
    e.add(f"    var {prefix}output_real = cxl_alloc[Float32]({prefix}total_elems)")
    e.add(f"    var {prefix}output_imag = cxl_alloc[Float32]({prefix}total_elems)")
    e.add(f"    var {prefix}ref_real = cxl_alloc[Float32]({prefix}total_elems)")
    e.add(f"    var {prefix}ref_imag = cxl_alloc[Float32]({prefix}total_elems)")
    e.add()
    e.add(f"    for i in range({prefix}total_elems):")
    e.add(f"        {prefix}input_real[i] = Float32(random_float64(-1.0, 1.0))")
    e.add(f"        {prefix}input_imag[i] = Float32(random_float64(-1.0, 1.0))")
    e.add(f"        {prefix}output_real[i] = Float32(0)")
    e.add(f"        {prefix}output_imag[i] = Float32(0)")
    e.add(f"        {prefix}ref_real[i] = Float32(0)")
    e.add(f"        {prefix}ref_imag[i] = Float32(0)")
    e.add()

    e.add(f"    var {prefix}pool_elems = {host.pool_elems}")
    emit_aligned_pool(e, name=f"{prefix}uthread_pool", elems=f"{prefix}pool_elems",
                      alignment_bytes=target.mapping_stride_bytes)
    e.add()

    workspace_args = emit_workspace_buffers(e, plan=plan, prefix=prefix)
    params_args = [f"{prefix}{name}" for name in ("input_real", "input_imag", "output_real", "output_imag")]
    params_args.extend(_emit_persistent_loop_table(e,plan=plan,table=loop_table or [],prefix=prefix))
    params_args.extend(workspace_args)
    e.add(f"    var {prefix}rc = {plan.kernel_name}.launch(")
    e.add(f"        PooledRange.over({prefix}uthread_pool, {prefix}pool_elems),")
    e.add(
        f"        {plan.kernel_name}Params({', '.join(params_args)}),"
    )
    e.add("    )")
    e.add()
    e.add(f"    if {prefix}rc != 0:")
    e.add(f'        print("[host] persistent FFT failed, exit", {prefix}rc)')
    e.add("        return")
    e.add()

    if reference_check:
        _emit_reference_check(
            e, n=plan.length, batch_count=num_logical_blocks, inverse=plan.inverse,
            input_real=f"{prefix}input_real", input_imag=f"{prefix}input_imag",
            output_real=f"{prefix}output_real", output_imag=f"{prefix}output_imag",
            ref_real=f"{prefix}ref_real", ref_imag=f"{prefix}ref_imag",
            tolerance=host.tolerance, label="persistent-tail-hybrid: full rounds (persistent)",
            var_prefix=prefix,
        )


def _emit_plain_tail_buffers_and_launch(
    e: Emitter, *, plan: FFTCodegenPlan, batch_count: int, prefix: str, reference_check: bool = True, target: TargetProfile,
) -> None:
    """Host body for a `tail_plan` kernel (P2.4): mirrors `codegen.
    fft_codegen.generate_fft_kernel`'s own host block (non-cooperative
    tail -- no pool-alignment requirement at all, `plan.cooperation is
    None`) or `codegen.fft_cooperative_codegen.generate_cooperative_fft_
    kernel`'s own (cooperative tail -- the SAME interleave-chunk pool
    alignment fix that module's own docstring documents, required
    whenever `workers_per_fft > 1`), chosen by `plan.cooperation is not
    None`, `prefix`-namespaced into the shared host `main()` alongside
    `_emit_persistent_buffers_and_launch`'s own block. `batch_count`: the
    number of LOGICAL replicas this tail kernel covers (`plan.total_
    uthreads` for non-cooperative, already-physical `plan.total_uthreads
    // plan.cooperation.workers_per_fft` for cooperative -- see
    `CooperationPlan`'s own docstring for why those differ)."""
    host = plan.host
    e.add(f"    var {prefix}total_elems = {host.total_elems}")
    e.add(f"    var {prefix}input_real = cxl_alloc[Float32]({prefix}total_elems)")
    e.add(f"    var {prefix}input_imag = cxl_alloc[Float32]({prefix}total_elems)")
    e.add(f"    var {prefix}output_real = cxl_alloc[Float32]({prefix}total_elems)")
    e.add(f"    var {prefix}output_imag = cxl_alloc[Float32]({prefix}total_elems)")
    e.add(f"    var {prefix}ref_real = cxl_alloc[Float32]({prefix}total_elems)")
    e.add(f"    var {prefix}ref_imag = cxl_alloc[Float32]({prefix}total_elems)")
    e.add()
    e.add(f"    for i in range({prefix}total_elems):")
    e.add(f"        {prefix}input_real[i] = Float32(random_float64(-1.0, 1.0))")
    e.add(f"        {prefix}input_imag[i] = Float32(random_float64(-1.0, 1.0))")
    e.add(f"        {prefix}output_real[i] = Float32(0)")
    e.add(f"        {prefix}output_imag[i] = Float32(0)")
    e.add(f"        {prefix}ref_real[i] = Float32(0)")
    e.add(f"        {prefix}ref_imag[i] = Float32(0)")
    e.add()
    e.add(f"    var {prefix}pool_elems = {max(r.pool_elems for r in plan.launch_mapping.rounds())}")
    emit_aligned_pool(e, name=f"{prefix}uthread_pool", elems=f"{prefix}pool_elems",
                      alignment_bytes=plan.launch_mapping.alignment_bytes)
    emit_fft_launches(e, plan=plan, pool=f"{prefix}uthread_pool",
                     args=[f"{prefix}input_real", f"{prefix}input_imag",
                           f"{prefix}output_real", f"{prefix}output_imag"],
                     rc_prefix=f"{prefix}rc", label="tail FFT")

    label = "persistent-tail-hybrid: tail (cooperative)" if plan.cooperation is not None else \
        "persistent-tail-hybrid: tail (non-cooperative)"
    if reference_check:
        _emit_reference_check(
            e, n=plan.length, batch_count=batch_count, inverse=plan.inverse,
            input_real=f"{prefix}input_real", input_imag=f"{prefix}input_imag",
            output_real=f"{prefix}output_real", output_imag=f"{prefix}output_imag",
            ref_real=f"{prefix}ref_real", ref_imag=f"{prefix}ref_imag",
            tolerance=host.tolerance, label=label,
            var_prefix=prefix,
        )


def generate_persistent_tail_hybrid_kernel(
    hybrid: PersistentTailHybridPlan, *,
    compute_lanes: int | None = 4, narrow_middle_stages: bool = True,
    mode: ExecutionMode | None = None, copy_mode: CopyMode | None = None, vector_width: int = 4,
    tail_compute_lanes: int | None = 4, tail_narrow_middle_stages: bool = True,
    target: TargetProfile = DEFAULT_TARGET_PROFILE, reference_check: bool = True,
    spread_across_units: bool = True,
) -> str:
    """P2.4: render `hybrid` (`planning.execution.fft_plan_persistent.
    make_persistent_tail_hybrid_plan`'s own output) as ONE standalone
    Mojo file. `hybrid.tail_strategy == "all_persistent"` (`hybrid.
    tail_plan is None`) degenerates to a PLAIN call to `generate_
    persistent_fft_kernel` -- byte-identical to every plan rendered
    before this function existed, for the default strategy.

    Otherwise: TWO completely independent `NDPTask` structs (`hybrid.
    persistent_plan` via `emit_persistent_kernel_struct`, `hybrid.
    tail_plan` via `codegen.fft_codegen.emit_kernel` -- the SAME struct-
    only emitter `fft_transpose_codegen.generate_recursive_fft_kernels`
    already reuses for kernel-chaining, confirmed to already handle a
    cooperative `plan.cooperation is not None` shape too, since `fft_
    cooperative_codegen.generate_cooperative_fft_kernel` itself calls the
    exact same `_emit_params_struct`/`_emit_task_struct` pair `emit_
    kernel` wraps -- see that module's own `generate_cooperative_fft_
    kernel` body), TWO separate DRAM buffer sets (`p_*`/`t_*`-prefixed,
    see `_emit_persistent_buffers_and_launch`/`_emit_plain_tail_buffers_
    and_launch`), TWO separate `.launch()` calls, and TWO separate
    reference checks -- one shared host `main()`. No pointer-offset
    arithmetic between the two buffer sets is used anywhere (an
    unverified Mojo idiom this project's own codegen has never used
    elsewhere) -- the two kernels' own logical block ranges never need to
    share one contiguous DRAM array at all, since neither reads the
    other's output.
    """
    if hybrid.tail_plan is None:
        assert hybrid.persistent_plan is not None
        return generate_persistent_fft_kernel(
            hybrid.persistent_plan, num_logical_blocks=hybrid.full_blocks,
            compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
            mode=mode, copy_mode=copy_mode, vector_width=vector_width,
            target=target, reference_check=reference_check,
        )

    if hybrid.persistent_plan is None and reference_check:
        # `full_blocks == 0` (fewer than target.num_ndp_units logical
        # blocks total): no persistent portion at all -- render `tail_
        # plan` as an ordinary standalone kernel via whichever existing
        # generator matches its own shape, exactly as if the caller had
        # asked for that strategy alone (never a persistent struct/launch
        # for zero replicas).
        if hybrid.tail_plan.cooperation is not None:
            return _generate_cooperative_fft_kernel(
                hybrid.tail_plan, compute_lanes=tail_compute_lanes, target=target,
                spread_across_units=spread_across_units,
            )
        return _generate_fft_kernel(
            hybrid.tail_plan, compute_lanes=tail_compute_lanes,
            narrow_middle_stages=tail_narrow_middle_stages,
            target=target, spread_across_units=spread_across_units,
        )

    hybrid = replace(hybrid, tail_plan=replace(
        hybrid.tail_plan, launch_mapping=resolve_fft_launch(
            hybrid.tail_plan, target=target, spread_across_units=spread_across_units)))
    e = Emitter()
    _emit_prelude(e)
    active_plan = hybrid.persistent_plan or hybrid.tail_plan
    e.add(f"comptime N = {active_plan.length}")
    e.add()
    if hybrid.persistent_plan is not None:
        persistent_loop_table = emit_persistent_kernel_struct(
            e, plan=hybrid.persistent_plan, num_logical_blocks=hybrid.full_blocks,
            compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
            mode=mode, copy_mode=copy_mode, vector_width=vector_width,
            max_kernel_register=target.max_kernel_register,
            spad_capacity_bytes=target.spad_capacity_bytes,
        )
    _emit_kernel(
        e, plan=hybrid.tail_plan, compute_lanes=tail_compute_lanes,
        narrow_middle_stages=tail_narrow_middle_stages,
        max_kernel_register=target.max_kernel_register,
    )
    emit_execution_record(e, plan=hybrid.tail_plan, compute_lanes=tail_compute_lanes,
                          narrow_middle_stages=tail_narrow_middle_stages, loop_stages=False)

    if hybrid.tail_plan.cooperation is not None:
        tail_batch_count = hybrid.tail_plan.total_uthreads // hybrid.tail_plan.cooperation.workers_per_fft
    else:
        tail_batch_count = hybrid.tail_plan.total_uthreads
    assert tail_batch_count == hybrid.tail_blocks, (
        f"tail_plan's own logical replica count ({tail_batch_count}) != hybrid.tail_blocks "
        f"({hybrid.tail_blocks}) -- inconsistent PersistentTailHybridPlan"
    )

    e.add("def main() raises:")
    if hybrid.persistent_plan is not None:
        e.add(f"    if {hybrid.persistent_plan.kernel_name}.emit_ir_if_asked():")
        e.add("        return")
    e.add(f"    if {hybrid.tail_plan.kernel_name}.emit_ir_if_asked():")
    e.add("        return")
    e.add()
    e.add("    seed(0)")
    e.add()

    if hybrid.persistent_plan is not None:
        _emit_persistent_buffers_and_launch(
            e, plan=hybrid.persistent_plan, num_logical_blocks=hybrid.full_blocks, prefix="p_", reference_check=reference_check,
            target=target,
            loop_table=persistent_loop_table,
        )
    e.add()
    _emit_plain_tail_buffers_and_launch(
        e, plan=hybrid.tail_plan, batch_count=hybrid.tail_blocks, prefix="t_", target=target, reference_check=reference_check,
    )

    if not reference_check:
        prefixes = (["p_"] if hybrid.persistent_plan is not None else []) + ["t_"]
        for kind, marker in (("input", "INPUT"), ("output", "OUTPUT")):
            e.add(f'    print("{marker}_BEGIN")')
            for prefix in prefixes:
                e.add(f"    for i in range({prefix}total_elems):")
                e.add(f"        print({prefix}{kind}_real[i], {prefix}{kind}_imag[i])")
            e.add(f'    print("{marker}_END")')
            e.add()
    return e.text()
