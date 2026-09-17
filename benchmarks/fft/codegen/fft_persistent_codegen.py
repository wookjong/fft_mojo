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
* 256B-aligned launch-pool host allocation.

## ROUND STATE IS RUNTIME, NOT A COMPILE-TIME CONSTANT PER ROUND

The design doc's original text says `ROUND_BASE`/`ACTIVE_GROUPS` are
"compile-time constants baked into each emitted round-specific phase
function" -- i.e. one `preload_r0`/`preload_r1`/.../`stage_0_r0`/
`stage_0_r1`/... function per (phase, round) pair. **This does not work**:
the M2NDP-Detour simulator caps how many *distinct* kernel functions one
`NDPTask` may register at all, ever, for the lifetime of one host
`.launch()` -- `max_kernel_register=8` in
third_party/m2ndp-detour/config/performance/M2NDP/m2ndp.config, enforced
by `UThreadGenerator::can_register()` (third_party/m2ndp-detour/src/
uthread_generator.cc) and asserted in `NdpUnit::register_ndp_kernel`
(ndp_unit.cc:143). A task's own kernels are registered once, at
`.launch()` time, and only unregistered when the *whole task* finishes
(`M2NDP::unregister_task`, m2ndp.cc) -- never per `launch_parallel` call.
Confirmed the hard way on real hardware: N=64 (3 stages) at 2 rounds
needs `(2 + 3) * 2 = 10` distinct functions under the per-round-baked
scheme, and the 9th registration attempt aborts the whole simulator
(`Assertion 'can_register()' failed`), a toolchain-level crash, not a
graceful error.

**The fix, confirmed working on real hardware** (see
`/tmp/.../round_tracker_experiment.py` in this session's own scratchpad,
not kept in-repo): exactly one function per *phase* (`preload`,
`stage_0`, ..., `stage_{k-1}`, `writeback` -- `2 + stage_count` distinct
kernels total, independent of round count, comfortably under the cap for
every N this design supports at all, since `stage_count` is small by
construction -- see `make_persistent_leaf_plan`'s own scratchpad-capacity
bound). Round state lives in a 1-element **scratchpad** counter
(`round_tracker`, one independent instance per physical NDP unit, zero-
initialized) that only `writeback` increments -- after every phase in
that round has already read it, so within one round every phase sees the
same `round_index`. `device_main` simply calls the *same* small function
set `rounds` times in sequence; there is no way, and no need, for
`device_main` itself to touch `Kernel.params[]` (a real attempt to do
that failed with `"the M2NDP scratchpad is a kernel's, and this function
is not launched"` -- `params[]` access appears to require the accessing
function to itself be a `launch_parallel` target, which `device_main`
is not). Confirmed on real hardware that repeated `launch_parallel[Kernel.
same_fn]()` calls to the *same* target still fully barrier between calls
(round N+1's read never races round N's write) -- this is what makes the
whole scheme race-free.
"""

from typing import Literal

from codegen.common import (
    Emitter,
    emit_prelude as _emit_prelude,
    emit_reference_check as _emit_reference_check,
    spad as _spad,
)
from codegen.fft_codegen import _emit_stage_batches, _stage_compute_lanes
from planning.core.fft_plan_core import FFTCodegenPlan, FFTStagePlan, SIMDBatchPlan
from planning.execution.fft_plan_persistent import num_rounds, worker_waves as _worker_waves

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


def _emit_worker_dispatch(
    e: Emitter, *, plan: FFTCodegenPlan, stage: FFTStagePlan, workers_per_fft: int,
    vector_compute_lanes: int | None, dispatch_var: str = "worker_id",
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
        _emit_stage_batches(
            sub, plan=plan, stage=stage, batches=batches, compute_lanes=compute_lanes
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
    """
    assert stage.persistent_vector_batches is not None
    assert stage.persistent_scalar_batches is not None
    assert len(stage.persistent_vector_batches) == workers_per_fft
    physical: list[list[SIMDBatchPlan]] = [[] for _ in range(workers_per_group)]
    for logical_worker_id in range(workers_per_fft):
        lane = logical_worker_id % workers_per_group
        physical[lane].extend(stage.persistent_vector_batches[logical_worker_id])
    if stage.persistent_scalar_batches:
        tail_lane = (workers_per_fft - 1) % workers_per_group
        physical[tail_lane].extend(stage.persistent_scalar_batches)
    return tuple(tuple(bucket) for bucket in physical)


def _emit_physical_lane_dispatch(
    e: Emitter, *, plan: FFTCodegenPlan, stage: FFTStagePlan, workers_per_fft: int,
    workers_per_group: int, vector_compute_lanes: int | None,
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
        _emit_stage_batches(
            sub, plan=plan, stage=stage, batches=batches, compute_lanes=vector_compute_lanes
        )
        for line in sub.lines:
            e.add("    " + line if line else "")


def _wave_tracker_name(plan: FFTCodegenPlan) -> str:
    return _spad(plan.kernel_name, "wave_tracker")


def _emit_wave_prelude(e: Emitter, *, plan: FFTCodegenPlan, workers_per_group: int) -> None:
    """Mode A ("wave") only -- reads this physical unit's own `wave_
    tracker` (mirrors `round_tracker`: one independent zero-initialized
    scratchpad cell per physical unit) and derives `logical_worker_id`
    for THIS call. Race-free the same way `round_tracker` is: every
    worker of every active group reads `wave_tracker` here before any of
    them reaches `_emit_wave_bump` below, so every worker in one
    `launch_parallel[stage_N]()` call sees the SAME `wave_index`."""
    tracker = _wave_tracker_name(plan)
    e.add(f"        var wave_index = Int({tracker}.load[DType.float32, 1](0)[0])")
    e.add(f"        var logical_worker_id = wave_index * {workers_per_group} + worker_id")


def _emit_wave_bump(e: Emitter, *, plan: FFTCodegenPlan, worker_waves: int) -> None:
    """Mode A ("wave") only -- advance this physical unit's own `wave_
    tracker` by one, modulo `worker_waves`, gated to exactly one physical
    worker per active software group (`worker_id == 0`), the same
    discipline `writeback`'s own `round_tracker` bump uses."""
    tracker = _wave_tracker_name(plan)
    e.add("        if worker_id == 0:")
    e.add(f"            var next_wave = (wave_index + 1) % {worker_waves}")
    e.add(f"            {tracker}.store(0, Float32(next_wave))")


def _round_tracker_name(plan: FFTCodegenPlan) -> str:
    return _spad(plan.kernel_name, "round_tracker")


def _emit_dispatch_prelude(
    e: Emitter, *, plan: FFTCodegenPlan, workers_per_group: int,
    software_group_count: int, num_logical_blocks: int, bump_round: bool,
) -> None:
    """Every phase function's shared entry sequence: derive `gid`/
    `software_group_id`/`worker_id` (unchanged), then read this round's
    own `round_base`/`active_groups` from the per-unit scratchpad
    `round_tracker` instead of a compile-time-baked literal -- see this
    module's own top-of-file docstring for why. `bump_round=True`
    (writeback only) advances the tracker for the *next* round, gated the
    same way every other worker-disjoint write in this design is (exactly
    one worker per active group, after that group's own real work for
    this round is done -- appended by the caller, not here; this only
    emits the read side, common to every phase).
    """
    tracker = _round_tracker_name(plan)
    e.add("        var gid = global_uthread_id()")
    e.add(f"        var software_group_id = gid // {workers_per_group}")
    e.add(f"        var worker_id = gid % {workers_per_group}")
    e.add(f"        var round_index = Int({tracker}.load[DType.float32, 1](0)[0])")
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
    e.add("        var spad_base = 0")
    e.add()
    if waves <= 1:
        # All three modes collapse to this one shape when there is only
        # one wave -- no mode-specific machinery ever emitted.
        _emit_worker_dispatch(
            e, plan=plan, stage=stage, workers_per_fft=workers_per_fft,
            vector_compute_lanes=vector_compute_lanes, dispatch_var="worker_id",
        )
    elif mode == "physical":
        # Mode C: flattened at codegen time, no runtime logical id at all.
        _emit_physical_lane_dispatch(
            e, plan=plan, stage=stage, workers_per_fft=workers_per_fft,
            workers_per_group=workers_per_group, vector_compute_lanes=vector_compute_lanes,
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
) -> tuple[str, list[str]]:
    """Preload (`to_scratchpad=True`) or writeback (`False`): every worker
    in an active software group copies a disjoint, strided slice
    (`worker_id, worker_id + workers_per_group, ...`) of this round's own
    logical block between DRAM (`block_base = logical_block * length`,
    natural order -- no FFT permutation on either side) and that group's
    own scratchpad bank. Deliberately fully scalar (one element at a
    time, `.load[width=1]`/`.store`) rather than vectorized: correctness-
    first per docs/persistent_leaf_design.md's own priority order --
    vectorizing this loop is a real follow-up performance opportunity,
    not attempted here.

    `bump_round`: writeback only -- after this group's own real copy work
    is done, worker 0 advances the *this physical unit's own* round
    tracker by one, so the next call to `preload`/`stage_*`/`writeback`
    (device_main's next round in sequence) sees `round_index + 1`. Every
    other phase in the *same* round already read the old value before
    this runs (writeback is always last), so this is race-free -- see
    this module's own top docstring.
    """
    e = Emitter()
    e.add("    @staticmethod")
    e.add(f"    def {name}():")
    e.add(f"        ref p = {plan.kernel_name}.params[]")
    _emit_dispatch_prelude(
        e, plan=plan, workers_per_group=workers_per_group,
        software_group_count=software_group_count, num_logical_blocks=num_logical_blocks,
        bump_round=bump_round,
    )
    e.add("        var logical_block = round_base + software_group_id")
    e.add(f"        var block_base = logical_block * {plan.length}")
    e.add("        var i = worker_id")
    e.add(f"        while i < {plan.length}:")
    buf = _spad(plan.kernel_name, buffer_name)
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
    if bump_round:
        tracker = _round_tracker_name(plan)
        e.add("        if worker_id == 0:")
        e.add(f"            {tracker}.store(0, Float32(round_index) + Float32(1))")
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
    e.add()
    e.add()


def emit_persistent_kernel_struct(
    e: Emitter, *, plan: FFTCodegenPlan, num_logical_blocks: int,
    compute_lanes: int | None = 4, narrow_middle_stages: bool = True,
    mode: ExecutionMode = "physical",
) -> None:
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

    pw = plan.persistent
    workers_per_group = pw.workers_per_group
    workers_per_fft = pw.workers_per_fft
    waves = _worker_waves(workers_per_fft, workers_per_group)
    software_group_count = pw.software_group_count
    rounds = num_rounds(num_logical_blocks, software_group_count)
    # Defense in depth: `make_persistent_leaf_plan` already rejects this
    # (target-aware, via `target.max_kernel_register`) before a plan ever
    # reaches codegen; this hardcodes that field's own default (8) as a
    # backstop for a plan constructed some other way, since this function
    # takes no `target` of its own to check against.
    num_kernels = 2 + len(plan.stages)
    if num_kernels > 8:
        raise ValueError(
            f"this plan needs {num_kernels} distinct kernel functions "
            f"(preload + {len(plan.stages)} stages + writeback), but the M2NDP-"
            f"Detour simulator's max_kernel_register=8 caps one task's total "
            f"registered kernels regardless of round count -- see this module's "
            f"own top-of-file docstring"
        )

    buffer_names = tuple(b.name for b in plan.scratchpad_buffers)
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
    # One instance per physical NDP unit (like buf_a/buf_b), zero-
    # initialized -- confirmed real-hardware (a from-scratch experiment,
    # not kept in this repo): every unit's own first read before any
    # write returns 0, matching every other scratchpad buffer's own
    # zero-init behavior in this codebase.
    e.add(
        f'    comptime round_tracker = scratchpad[1, Float32, '
        f'name="{plan.kernel_name.lower()}_round_tracker"]()'
    )
    # `wave_tracker` only exists for Mode A ("wave") with more than one
    # wave -- Modes B ("fused") and C ("physical") resolve every logical
    # worker a physical lane owns inside ONE stage invocation (a runtime
    # loop for B, a compile-time-flattened static dispatch for C), so
    # neither needs any cross-call state persisted in scratchpad.
    if mode == "wave" and waves > 1:
        e.add(
            f'    comptime wave_tracker = scratchpad[1, Float32, '
            f'name="{plan.kernel_name.lower()}_wave_tracker"]()'
        )
    e.add()

    preload_name, lines = emit_bulk_copy_phase(
        plan=plan, name="preload", software_group_count=software_group_count,
        num_logical_blocks=num_logical_blocks, workers_per_group=workers_per_group,
        to_scratchpad=True, buffer_name=buffer_names[0], bump_round=False,
    )
    e.lines.extend(lines)

    stage_names: list[str] = []
    for stage in plan.stages:
        name, lines = emit_stage_phase(
            plan=plan, stage=stage, software_group_count=software_group_count,
            num_logical_blocks=num_logical_blocks, workers_per_group=workers_per_group,
            workers_per_fft=workers_per_fft,
            compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
            mode=mode,
        )
        e.lines.extend(lines)
        stage_names.append(name)

    writeback_name, lines = emit_bulk_copy_phase(
        plan=plan, name="writeback", software_group_count=software_group_count,
        num_logical_blocks=num_logical_blocks, workers_per_group=workers_per_group,
        to_scratchpad=False, buffer_name=final_buffer, bump_round=True,
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
    for _ in range(rounds):
        e.add(f"        launch_parallel[{plan.kernel_name}.{preload_name}]()")
        for stage_name in stage_names:
            for _ in range(stage_repeat):
                e.add(f"        launch_parallel[{plan.kernel_name}.{stage_name}]()")
        e.add(f"        launch_parallel[{plan.kernel_name}.{writeback_name}]()")
    e.add()
    e.add()


def generate_persistent_fft_kernel(
    plan: FFTCodegenPlan, *, num_logical_blocks: int,
    compute_lanes: int | None = 4, narrow_middle_stages: bool = True,
    mode: ExecutionMode = "physical",
) -> str:
    """Render a full persistent-software-workgroup FFT: `emit_persistent_
    kernel_struct`'s one `NDPTask` struct, plus this function's own
    self-contained host `main()` (one `.launch()`, buffer alloc/fill,
    reference check) -- the standalone single-file entry point every
    existing caller of this function still gets unchanged. See
    `emit_persistent_kernel_struct`'s own docstring for the struct-only
    half of this, now shared with `fft_transpose_codegen.
    generate_recursive_fft_kernels`.

    `mode`: forwarded to `emit_persistent_kernel_struct`/`emit_stage_
    phase` -- see this module's own top-of-file `ExecutionMode` docstring
    for what `"wave"`/`"fused"`/`"physical"` each render.
    """
    e = Emitter()
    _emit_prelude(e)
    e.add(f"comptime N = {plan.length}")
    e.add()
    emit_persistent_kernel_struct(
        e, plan=plan, num_logical_blocks=num_logical_blocks,
        compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
        mode=mode,
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

    # Pool.alloc (src/m2ndp_host.mojo) only aligns individual allocations
    # to 64 bytes even though the pool's own base is 256B-aligned -- a
    # given cxl_alloc address is therefore not guaranteed 256B-aligned.
    # Over-allocate by 64 extra Float32 elements (256 bytes) and advance
    # to the next 256B boundary by hand, the same ceil-to-multiple formula
    # Pool.alloc's own _POOL_ALIGN rounding uses -- see
    # docs/persistent_leaf_design.md's "uthread pool alignment" section.
    e.add(f"    var pool_elems = {host.pool_elems}")
    e.add("    var raw_pool = cxl_alloc[Float32](pool_elems + 64)")
    e.add("    var raw_addr = Int(raw_pool)")
    e.add("    var aligned_addr = (raw_addr + 255) // 256 * 256")
    e.add("    if aligned_addr % 256 != 0:")
    e.add('        print("[host] persistent pool alignment assertion failed:", aligned_addr)')
    e.add("        return")
    e.add(
        "    var uthread_pool = UnsafePointer[Float32, MutAnyOrigin]"
        "(unsafe_from_address=aligned_addr)"
    )
    e.add()

    e.add(f"    var rc = {plan.kernel_name}.launch(")
    e.add("        PooledRange.over(uthread_pool, pool_elems),")
    e.add(
        f"        {plan.kernel_name}Params(input_real, input_imag, output_real, output_imag),"
    )
    e.add("    )")
    e.add()
    e.add("    if rc != 0:")
    e.add('        print("[host] persistent FFT failed, exit", rc)')
    e.add("        return")
    e.add()

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

    return e.text()
