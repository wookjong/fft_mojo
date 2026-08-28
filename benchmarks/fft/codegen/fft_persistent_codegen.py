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

from codegen.common import (
    Emitter,
    emit_prelude as _emit_prelude,
    emit_reference_check as _emit_reference_check,
    spad as _spad,
)
from codegen.fft_codegen import _emit_stage_batches, _stage_compute_lanes
from planning.fft_plan_core import FFTCodegenPlan, FFTStagePlan
from planning.fft_plan_persistent import num_rounds


def _worker_body(
    stage: FFTStagePlan, worker_id: int, workers_per_group: int, vector_compute_lanes: int | None,
) -> tuple[tuple, int | None]:
    """`(batches, compute_lanes)` for one worker on one stage -- the last
    worker is the scalar worker (rendered at `compute_lanes=1`, genuine
    scalar arithmetic, per docs/persistent_leaf_design.md's "Scalar-tail
    arithmetic") whenever this stage actually assigned it any tail
    batches; every other worker (including the last, on a stage with no
    tail at all) is a plain vector worker at `vector_compute_lanes` --
    already resolved by the caller via `_stage_compute_lanes` (this
    stage's own register-pressure-narrowed width, the same mechanism
    fft_codegen.py's plain/cooperative paths use -- see
    `emit_stage_phase`'s own docstring for why skipping this is not
    optional: a real crash + wrong answer, not just a spill warning, was
    confirmed on N=64's (4,4,4) middle stage before this was wired in).
    """
    assert stage.persistent_vector_batches is not None
    assert stage.persistent_scalar_batches is not None
    if worker_id == workers_per_group - 1 and stage.persistent_scalar_batches:
        return stage.persistent_scalar_batches, 1
    return stage.persistent_vector_batches[worker_id], vector_compute_lanes


def _emit_worker_dispatch(
    e: Emitter, *, plan: FFTCodegenPlan, stage: FFTStagePlan, workers_per_group: int,
    vector_compute_lanes: int | None,
) -> None:
    """`if worker_id == 0: ... elif worker_id == 1: ... else: ...` -- one
    branch per worker, each rendering that worker's own already-partitioned
    batches via the *exact same* `_emit_stage_batches` fft_codegen.py uses
    for the plain (non-persistent) path. Built in a sub-Emitter and
    re-indented by one level (the same idiom fft_transpose_codegen.py's
    own tail-branch emission uses) since `_emit_stage_batches`/`_emit_batch`
    hardcode an 8-space base indent, one level shallower than this call
    site (inside an `if worker_id == k:` body, not directly inside a
    `@staticmethod def ...():`).
    """
    for worker_id in range(workers_per_group):
        batches, compute_lanes = _worker_body(
            stage, worker_id, workers_per_group, vector_compute_lanes
        )
        keyword = "if" if worker_id == 0 else "elif"
        e.add(f"        {keyword} worker_id == {worker_id}:")
        sub = Emitter()
        if batches:
            _emit_stage_batches(
                sub, plan=plan, stage=stage, batches=batches, compute_lanes=compute_lanes
            )
        else:
            # `_emit_batch` always emits at a hardcoded 8-space base indent
            # (assumes direct function-body placement); match that here so
            # the uniform +4 re-indent below lands "pass" at the same depth
            # as a sibling branch's real batch code, not 8 spaces shallower.
            sub.add("        pass")
        for line in sub.lines:
            e.add("    " + line if line else "")


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
    compute_lanes: int | None = 4,
    narrow_middle_stages: bool = True,
) -> tuple[str, list[str]]:
    """Builds this phase's own fresh `Emitter` and returns `(name, lines)`
    -- the same "translate exactly what would be emitted, from a small
    self-contained snippet" shape verify_fft_harness._translate_stage
    already uses, so verification/verify_fft_persistent.py can translate
    and exec these lines directly without re-deriving anything. One
    function total per stage (not one per round) -- see this module's own
    top docstring for why.

    `compute_lanes`/`narrow_middle_stages`: resolved through the *exact
    same* `_stage_compute_lanes` fft_codegen.py's plain/cooperative paths
    use, applied to every vector worker (the scalar worker stays forced
    at `compute_lanes=1` regardless -- already narrower than anything
    this could produce). Defaults (`4`/`True`) match make_fft_kernel.py's
    own shipped defaults, not "no narrowing" -- skipping this was tried
    first and produced a real crash (`vs2r.v: Unsupported Instruction`)
    plus a wrong answer on N=64's own (4,4,4) middle stage, confirmed on
    real hardware; see docs/persistent_leaf_design.md's own "Register-
    pressure discipline" section, which requires reusing this mechanism
    rather than treating a persistent-leaf spill as merely a performance
    caveat.
    """
    is_first = stage.stage_id == 0
    is_last = stage.stage_id == len(plan.stages) - 1
    prev_radix = plan.stages[stage.stage_id - 1].radix if not is_first else None
    vector_compute_lanes = _stage_compute_lanes(
        compute_lanes=compute_lanes, is_first=is_first, is_last=is_last, radix=stage.radix,
        narrow_middle_stages=narrow_middle_stages, prev_radix=prev_radix,
    )

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
    _emit_worker_dispatch(
        e, plan=plan, stage=stage, workers_per_group=workers_per_group,
        vector_compute_lanes=vector_compute_lanes,
    )
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


def generate_persistent_fft_kernel(
    plan: FFTCodegenPlan, *, num_logical_blocks: int,
    compute_lanes: int | None = 4, narrow_middle_stages: bool = True,
) -> str:
    """Render a full persistent-software-workgroup FFT: one `NDPTask`
    struct, one host-level `.launch()`, exactly `2 + len(plan.stages)`
    distinct kernel functions (`preload`, `stage_0`, ..., `writeback`;
    see this module's own top docstring for why round-specific functions
    don't work), `device_main` calling that same small function set once
    per round in the right order. `plan` must come from
    `planning.fft_plan_persistent.make_persistent_leaf_plan` (i.e.
    `plan.persistent is not None`); `num_logical_blocks` must match what
    that call was given (checked below, not re-derived, since
    `plan.host.total_elems` only encodes `length * num_logical_blocks`
    jointly).

    `compute_lanes`/`narrow_middle_stages`: forwarded to `emit_stage_
    phase` -- see its own docstring for why the defaults (`4`/`True`,
    make_fft_kernel.py's own shipped defaults) are load-bearing here, not
    just a performance knob.
    """
    if plan.persistent is None:
        raise ValueError("generate_persistent_fft_kernel requires a persistent plan")
    if plan.host.total_elems != plan.length * num_logical_blocks:
        raise ValueError(
            f"num_logical_blocks={num_logical_blocks} is inconsistent with "
            f"plan.host.total_elems={plan.host.total_elems} for length={plan.length}"
        )

    pw = plan.persistent
    workers_per_group = pw.workers_per_group
    software_group_count = pw.software_group_count
    rounds = num_rounds(num_logical_blocks, software_group_count)
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

    e = Emitter()
    _emit_prelude(e)
    e.add(f"comptime N = {plan.length}")
    e.add()
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
    e.add()

    phase_names: list[str] = []

    name, lines = emit_bulk_copy_phase(
        plan=plan, name="preload", software_group_count=software_group_count,
        num_logical_blocks=num_logical_blocks, workers_per_group=workers_per_group,
        to_scratchpad=True, buffer_name=buffer_names[0], bump_round=False,
    )
    e.lines.extend(lines)
    phase_names.append(name)

    for stage in plan.stages:
        name, lines = emit_stage_phase(
            plan=plan, stage=stage, software_group_count=software_group_count,
            num_logical_blocks=num_logical_blocks, workers_per_group=workers_per_group,
            compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
        )
        e.lines.extend(lines)
        phase_names.append(name)

    name, lines = emit_bulk_copy_phase(
        plan=plan, name="writeback", software_group_count=software_group_count,
        num_logical_blocks=num_logical_blocks, workers_per_group=workers_per_group,
        to_scratchpad=False, buffer_name=final_buffer, bump_round=True,
    )
    e.lines.extend(lines)
    phase_names.append(name)

    e.add("    @staticmethod")
    e.add("    def device_main():")
    for _ in range(rounds):
        for phase in phase_names:
            e.add(f"        launch_parallel[{plan.kernel_name}.{phase}]()")
    e.add()
    e.add()

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
