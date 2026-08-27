from __future__ import annotations

"""Code emission specific to cooperative-worker leaves (`workers_per_fft`
microthreads sharing one sub-FFT's scratchpad -- see `CooperationPlan` in
fft_plan_core.py and the builder/design rationale in planning/fft_plan_
cooperative.py) -- the counterpart to fft_codegen.py's own default
one-uthread-per-sub-FFT rendering. Split into its own module, mirroring the
same split already made at the planning layer (fft_plan_core.py vs.
fft_plan_cooperative.py), so "how does a cooperative leaf render" has one
place to look that isn't interleaved with the (much larger) default path.

Per-batch rendering itself (load/twiddle/butterfly/store text for one
SIMDBatchPlan) is NOT duplicated here -- `_emit_cooperative_stage` below
reuses fft_codegen._emit_stage_batches unchanged (see its own docstring):
a batch's own offsets never depended on which uthread runs it, only
*how many* uthreads share the surrounding stage's own dispatch, which is
exactly this module's own scope. This module therefore imports the shared
per-batch/task-struct emitters back from fft_codegen.py rather than
duplicating them; fft_codegen.py's own `_emit_stage` dispatch imports
`_emit_cooperative_stage` from here with a function-local import (not a
module-level one) specifically to avoid a module-level import cycle
between the two files.
"""

from codegen.common import Emitter, emit_prelude as _emit_prelude, emit_reference_check as _emit_reference_check
from codegen.fft_codegen import _emit_params_struct, _emit_stage_batches, _emit_task_struct
from planning.fft_plan_core import AddressMapping, AddressMappingKind, FFTCodegenPlan, FFTStagePlan


def _cooperative_mapping_base_expr(mapping: AddressMapping, kernel_length: int) -> str:
    """DRAM address base for a cooperative leaf's own input/output mapping.

    Always CONTIGUOUS -- a cooperative leaf's own AddressMapping never comes
    out any other way (PEELED/CROSSED/STRIDED belong to a wrapping multi-
    kernel/balanced/recursive plan's own boundary, never a leaf itself; see
    fft_plan_cooperative.py's module docstring) -- keyed by `logical_fft_id`
    (`global_uthread_id() // WORKERS_PER_FFT` -- see _emit_cooperative_stage
    for why this, not a group_id()-built numbering, is the sound choice).
    """
    assert mapping.kind == AddressMappingKind.CONTIGUOUS, (
        f"a cooperative leaf's own DRAM mapping must be CONTIGUOUS, got {mapping.kind}"
    )
    expr = f"logical_fft_id * {mapping.row_stride}"
    if mapping.base:
        expr += f" + {mapping.base}"
    return expr


def _emit_cooperative_stage(
    e: Emitter,
    *,
    plan: FFTCodegenPlan,
    stage: FFTStagePlan,
    is_first: bool,
    is_last: bool,
    compute_lanes: int | None,
) -> None:
    """Cooperative counterpart of the plain per-uthread stage body below:
    `WORKERS_PER_FFT` microthreads (consecutive `local_uthread_id()`s) share
    one `fft_slot`'s scratchpad, each executing only the batches
    `stage.worker_batches` assigned it (see CooperationPlan / fft_plan_
    cooperative.py) -- everything else (per-batch load/twiddle/butterfly/
    store rendering) reuses `_emit_stage_batches` unchanged, since a batch's
    own offsets never depended on which uthread runs it.

    Two different groupings of the same `WORKERS_PER_FFT`-sized worker set,
    for two different reasons, from two different primitives:

    * `fft_slot = local_uthread_id() // WORKERS_PER_FFT` (scratchpad -- see
      `spad_base` below): safe by definition, not by an assumption about
      interleaving -- `local_uthread_id()` *is* "this microthread's index on
      its own physical unit" (see src/m2ndp.mojo), so two microthreads
      sharing an `fft_slot` share a unit, and thus its scratchpad,
      unconditionally.

    * `logical_fft_id = global_uthread_id() // WORKERS_PER_FFT` (DRAM -- see
      _cooperative_mapping_base_expr): an earlier version of this function
      built this from `group_id() + num_groups() * fft_slot` instead --
      wrong, and concretely so: traced against this project's own real
      config (`m2ndp_interleave_size=256`, `packet_size=32`, so the address
      decoder hands out 8 consecutive global ids to one unit before rotating
      to the next -- docs/SIMULATION.md's "divided by the interleave
      stride"), `WORKERS_PER_FFT=4` puts *two* fft_slots (0 and 1) on the
      *same* unit, whose own `group_id() + num_groups()*fft_slot` gives
      `group_id()` and `group_id()+32` -- the second one lands nowhere near
      the real logical id (1) that a second sub-FFT on that same unit should
      get, so it either collided with another unit's id or fell outside
      `[0, total_ffts)` and silently dropped real work. `global_uthread_id()`
      has no such failure mode: the hardware hands it out dense over the
      *entire* launch regardless of how it interleaves units, so
      `global_uthread_id() // WORKERS_PER_FFT` covers `[0, total_ffts)`
      exactly once each, for any interleaving at all -- this is reading a
      primitive directly, not reconstructing one from others (contrast
      docs/STATUS.md's "IDs are primitives, not derived", which is about the
      latter). It also removes any need for `total_ffts` to relate to
      `num_groups()` a particular way.

      This does rely on one real precondition tied to the *scratchpad*
      grouping above: workers cooperating on one sub-FFT must physically
      share a unit, which holds only if `WORKERS_PER_FFT` divides the
      hardware's own interleave chunk (`m2ndp_interleave_size / packet_size`
      -- 8 in the config above) so that `local_uthread_id()`'s and
      `global_uthread_id()`'s own groupings of `WORKERS_PER_FFT` partition
      the same physical microthreads into the same groups (confirmed for
      `WORKERS_PER_FFT=4` against that config by direct trace, as above).
      A caller picking a `WORKERS_PER_FFT` that does not divide the real
      chunk size is not something this module can detect at plan time (the
      chunk size is a runtime config, not a Mojo-visible constant) -- flagged
      as a known restriction, not silently handled.
    """
    coop = plan.cooperation
    assert coop is not None
    e.add(f"        comptime WORKERS_PER_FFT = {coop.workers_per_fft}")
    e.add("        var local_id = local_uthread_id()")
    e.add("        var fft_slot = local_id // WORKERS_PER_FFT")
    e.add("        var worker_id = local_id % WORKERS_PER_FFT")
    e.add(f"        if fft_slot >= {coop.fft_slots_per_group}:")
    e.add("            return")
    if plan.scratchpad_buffers:
        e.add(f"        var spad_base = fft_slot * {plan.scratchpad_uthread_stride}")
    e.add("        var logical_fft_id = global_uthread_id() // WORKERS_PER_FFT")
    if is_first:
        e.add(
            f"        var in_batch_base = "
            f"{_cooperative_mapping_base_expr(plan.input_mapping, plan.length)}"
        )
    if is_last:
        e.add(
            f"        var out_batch_base = "
            f"{_cooperative_mapping_base_expr(plan.output_mapping, plan.length)}"
        )
    e.add()

    assert stage.worker_batches is not None
    for worker_id, batches in enumerate(stage.worker_batches):
        if not batches:
            continue
        sub = Emitter()
        _emit_stage_batches(sub, plan=plan, stage=stage, batches=batches, compute_lanes=compute_lanes)
        e.add(f"        if worker_id == {worker_id}:")
        for line in sub.lines:
            e.add("    " + line if line else "")
    e.add()


def generate_cooperative_fft_kernel(
    plan: FFTCodegenPlan, *, compute_lanes: int | None = None, in_place: bool = False
) -> str:
    """Render a standalone cooperative leaf (`plan.cooperation is not None` --
    see fft_plan_cooperative.make_cooperative_leaf_plan): one NDPTask, one
    launch, `workers_per_fft` microthreads sharing each sub-FFT's scratchpad
    instead of `generate_fft_kernel`'s one-uthread-per-sub-FFT. This function
    performs no FFT planning.

    Almost identical to `generate_fft_kernel`; the one real difference is
    `batch_count` for the host-side reference check and DRAM buffer layout,
    which must be the number of *logical* sub-FFTs (`total_ffts`), not
    `plan.total_uthreads` (now the *physical* microthread count -- see
    CooperationPlan's own docstring) -- `plan.host.total_elems`/`pool_elems`
    were already sized correctly for this split by `make_cooperative_leaf_plan`
    itself, so only the reference-check's own `batch_count` needs the same
    `// workers_per_fft` recovery here.

    `in_place`: pass the same buffer as both `input_*_base` and
    `output_*_base` in this kernel's own Params. Race-free by construction,
    not by anything special this flag renders: stage 0 only ever reads DRAM
    (never writes it) and only the last stage ever writes DRAM (never reads
    it -- see fft_plan_core._make_load/_make_store's own first_stage/
    last_stage source/destination split) and `launch_parallel` is a full
    synchronization barrier between stages, so by the time the last stage's
    `launch_parallel` call is even reached, stage 0 has already finished
    reading every element of the original input, across every group -- the
    last stage overwriting that same DRAM can never race a read that hasn't
    happened yet. `in_place` therefore only changes which pointer the host
    passes twice; every kernel-side load/store offset is identical either way.

    The host keeps its own untouched copy of the random input (`orig_*`) to
    compute the reference DFT from: once `in_place` aliases input and output,
    `input_real`/`input_imag` no longer hold the original signal by the time
    the reference check runs (the kernel already overwrote them), so the
    check must read the preserved copy instead -- this is a host-side
    bookkeeping fix for verifying an in-place run, not anything about the
    in-place claim above.
    """
    assert plan.cooperation is not None
    total_ffts = plan.total_uthreads // plan.cooperation.workers_per_fft

    e = Emitter()
    _emit_prelude(e)
    e.add(f"comptime N = {plan.length}")
    e.add(f"comptime MAX_UTHREAD_{plan.kernel_name} = {plan.max_uthread}")
    e.add()
    _emit_params_struct(e, plan=plan)
    _emit_task_struct(e, plan=plan, compute_lanes=compute_lanes)

    host = plan.host
    e.add("def main() raises:")
    e.add(f"    if {plan.kernel_name}.emit_ir_if_asked():")
    e.add("        return")
    e.add()
    e.add(f"    var total_elems = {host.total_elems}")
    e.add("    var input_real = cxl_alloc[Float32](total_elems)")
    e.add("    var input_imag = cxl_alloc[Float32](total_elems)")
    if in_place:
        e.add("    var output_real = input_real")
        e.add("    var output_imag = input_imag")
        # The kernel overwrites input_real/input_imag in place, so the
        # reference DFT below must read the signal from here instead.
        e.add("    var orig_real = cxl_alloc[Float32](total_elems)")
        e.add("    var orig_imag = cxl_alloc[Float32](total_elems)")
    else:
        e.add("    var output_real = cxl_alloc[Float32](total_elems)")
        e.add("    var output_imag = cxl_alloc[Float32](total_elems)")
    e.add("    var ref_real = cxl_alloc[Float32](total_elems)")
    e.add("    var ref_imag = cxl_alloc[Float32](total_elems)")
    e.add()
    e.add(f"    var pool_elems = {host.pool_elems}")
    e.add("    var uthread_pool = cxl_alloc[Float32](pool_elems)")
    e.add()

    e.add("    seed(0)")
    e.add("    for i in range(total_elems):")
    e.add("        input_real[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        input_imag[i] = Float32(random_float64(-1.0, 1.0))")
    if in_place:
        e.add("        orig_real[i] = input_real[i]")
        e.add("        orig_imag[i] = input_imag[i]")
    else:
        e.add("        output_real[i] = Float32(0)")
        e.add("        output_imag[i] = Float32(0)")
    e.add("        ref_real[i] = Float32(0)")
    e.add("        ref_imag[i] = Float32(0)")
    e.add()

    e.add(f"    var rc = {plan.kernel_name}.launch(")
    e.add("        PooledRange.over(uthread_pool, pool_elems),")
    e.add(
        f"        {plan.kernel_name}Params(input_real, input_imag, output_real, output_imag),"
    )
    e.add("    )")
    e.add()
    e.add("    if rc != 0:")
    e.add('        print("[host] FFT failed, exit", rc)')
    e.add("        return")
    e.add()

    _emit_reference_check(
        e,
        n=plan.length,
        batch_count=total_ffts,
        inverse=plan.inverse,
        input_real="orig_real" if in_place else "input_real",
        input_imag="orig_imag" if in_place else "input_imag",
        output_real="output_real",
        output_imag="output_imag",
        ref_real="ref_real",
        ref_imag="ref_imag",
        tolerance=host.tolerance,
        label="cooperative FFT",
    )

    return e.text()
