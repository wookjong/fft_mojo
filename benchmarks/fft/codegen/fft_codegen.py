from __future__ import annotations

"""Pure code emission from a fully lowered FFTCodegenPlan / DecomposedFFTPlan.

No FFT planning is performed here.  In particular this module does not:

* choose radices or stage order,
* calculate SIMD iteration counts or tail lanes,
* choose DRAM/scratchpad sources,
* choose ping-pong banks,
* choose whether an FFT is single-kernel or decomposed into N0*N1,
* choose a kernel's DRAM address mapping (contiguous vs. strided) or decide
  when a load/store falls back from vector to scalar-per-lane,
* calculate twiddle exponents/constants (small or large),
* calculate load/store indices,
* choose store layouts,
* build host reference FFT values.

All of those decisions are already materialized in fft_plangen.FFTCodegenPlan
/ fft_plangen.DecomposedFFTPlan. A decomposed FFT is two ordinary
FFTCodegenPlan kernels (see fft_plangen's module docstring); this module
renders each exactly as it would a single-kernel plan, and additionally
threads the large-twiddle table and the two kernels' DRAM hand-off through
one combined main() -- there is no separate "transpose kernel" abstraction
to emit.
"""

from codegen.fft_butterflies import emit_butterfly
from planning.fft_plan_core import (
    AddressMapping,
    AddressMappingKind,
    FFTCodegenPlan,
    FFTStagePlan,
    LargeTwiddlePlan,
    LoadPlan,
    MultiKernelFFTPlan,
    OutputPlan,
    SIMDBatchPlan,
    StorePlan,
    TwiddlePlan,
)
from planning.fft_plan_simple import DecomposedFFTPlan


class Emitter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _f32(value: float) -> str:
    """Format a planner-provided numeric constant as Mojo Float32 syntax."""
    if abs(value) < 1.0e-12:
        value = 0.0
    elif abs(value - 1.0) < 1.0e-12:
        value = 1.0
    elif abs(value + 1.0) < 1.0e-12:
        value = -1.0
    return f"Float32({value:.9g})"


def _spad(kernel_name: str, name: str) -> str:
    return f"{kernel_name}.{name}"


def _mapping_base_expr(mapping: AddressMapping, kernel_length: int) -> str:
    """`global_uthread_id() * row_stride [+ base]` -- the one runtime
    multiply a CONTIGUOUS/STRIDED AddressMapping ever costs. `elem*elem_stride`
    is folded into each load/store's own offset at plan time (see
    fft_plangen._make_load / _make_store), so that's the whole of what
    codegen computes at runtime for those two kinds: no per-access division
    or modulo.

    SPLIT was the original scheme for a middle kernel's output in an M>=3
    multi-kernel chain (see AddressMappingKind.SPLIT): `global_uthread_id()`
    mixes an already-transformed digit run (weight < a) with a
    not-yet-transformed remainder (weight >= a), and the kernel's own new
    digit needs to land *between* them -- one `%` and one `//`. No planner
    in this module constructs a SPLIT mapping any more (make_multi_kernel_plan
    uses PEELED instead, below, for every M>=3 chain); the kind and this
    description are kept only because PEELED's own derivation is explained
    by contrast to it.

    PEELED (see AddressMappingKind.PEELED) is SPLIT's replacement: the
    not-yet-transformed remainder itself splits further, into the *next*
    kernel's own digit (pulled to the innermost slot) and everything after
    it -- two `%`/`//` pairs instead of SPLIT's one.

    CROSSED (see AddressMappingKind.CROSSED, make_balanced_plan) splits
    `global_uthread_id()` into a batch index (`% batch_count`) and this
    side's own already-transformed digits (`// batch_count`), placing the
    batch index outermost instead of innermost -- the opposite of SPLIT/
    PEELED, since this is a transpose to the *other* side's benefit, not
    a within-side digit rotation.
    """
    if mapping.kind == AddressMappingKind.PEELED:
        a = mapping.peel_a
        k_next = mapping.peel_k_next
        tail = mapping.peel_tail_size
        return (
            f"(((global_uthread_id() // {a}) % {tail}) * {a * kernel_length * k_next}) + "
            f"((global_uthread_id() % {a}) * {k_next}) + "
            f"((global_uthread_id() // {a}) // {tail})"
        )
    if mapping.kind == AddressMappingKind.CROSSED:
        batch_count = mapping.peel_a
        side_length = mapping.row_stride
        return (
            f"(global_uthread_id() % {batch_count}) * {side_length} + "
            f"(global_uthread_id() // {batch_count})"
        )
    expr = f"global_uthread_id() * {mapping.row_stride}"
    if mapping.base:
        expr += f" + {mapping.base}"
    return expr


def _emit_load(e: Emitter, *, plan: FFTCodegenPlan, load: LoadPlan, width: int) -> None:
    """`width` is this (possibly chunked -- see `_chunk_batch`) load's own
    SIMD width, not necessarily `plan.simd_lanes`: the plan's `simd_lanes`
    is the hardware launch granule (ties to `PooledRange`/`VECTOR_WIDTH`,
    see `_chunk_batch`'s own docstring) and must not drive how wide a
    vector instruction the kernel body actually emits.
    """
    j = load.operand

    if load.mode == "vector":
        assert load.base_offset is not None
        if load.source == "input":
            e.add(
                f"        var rr{j} = p.input_real_base.load[width={width}]("
                f"in_batch_base + {load.base_offset})"
            )
            e.add(
                f"        var ii{j} = p.input_imag_base.load[width={width}]("
                f"in_batch_base + {load.base_offset})"
            )
        else:
            assert load.buffer_name is not None
            buf = _spad(plan.kernel_name, load.buffer_name)
            e.add(
                f"        var rr{j} = {buf}.load[DType.float32, {width}]("
                f"spad_base + {load.base_offset})"
            )
            e.add(
                f"        var ii{j} = {buf}.load[DType.float32, {width}]("
                f"spad_base + {plan.length} + {load.base_offset})"
            )
        return

    rr_lanes: list[str] = []
    ii_lanes: list[str] = []
    for lane, offset in enumerate(load.packed_lane_offsets):
        if offset is None:
            rr_lanes.append("Float32(0)")
            ii_lanes.append("Float32(0)")
            continue

        rr_scalar = f"rr{j}_lane{lane}"
        ii_scalar = f"ii{j}_lane{lane}"
        if load.source == "input":
            e.add(
                f"        var {rr_scalar} = p.input_real_base.load[width=1]("
                f"in_batch_base + {offset})"
            )
            e.add(
                f"        var {ii_scalar} = p.input_imag_base.load[width=1]("
                f"in_batch_base + {offset})"
            )
        else:
            assert load.buffer_name is not None
            buf = _spad(plan.kernel_name, load.buffer_name)
            e.add(
                f"        var {rr_scalar} = {buf}.load[DType.float32, 1]("
                f"spad_base + {offset})"
            )
            e.add(
                f"        var {ii_scalar} = {buf}.load[DType.float32, 1]("
                f"spad_base + {plan.length} + {offset})"
            )
        rr_lanes.append(f"{rr_scalar}[0]")
        ii_lanes.append(f"{ii_scalar}[0]")
    e.add(
        f"        var rr{j} = SIMD[DType.float32, {width}](" + ", ".join(rr_lanes) + ")"
    )
    e.add(
        f"        var ii{j} = SIMD[DType.float32, {width}](" + ", ".join(ii_lanes) + ")"
    )


def _emit_twiddle(
    e: Emitter, *, output: int, twiddle: TwiddlePlan, width: int
) -> None:
    e.add(
        f"        var twr{output} = SIMD[DType.float32, {width}]("
        + ", ".join(_f32(v) for v in twiddle.real)
        + ")"
    )
    e.add(
        f"        var twi{output} = SIMD[DType.float32, {width}]("
        + ", ".join(_f32(v) for v in twiddle.imag)
        + ")"
    )
    e.add(
        f"        var tr{output} = or{output} * twr{output} - "
        f"oi{output} * twi{output}"
    )
    e.add(
        f"        var ti{output} = or{output} * twi{output} + "
        f"oi{output} * twr{output}"
    )
    e.add(f"        or{output} = tr{output}")
    e.add(f"        oi{output} = ti{output}")


def _emit_large_twiddle(
    e: Emitter, *, output: int, store: StorePlan, width: int
) -> None:
    """Runtime cross-block twiddle, fused into this output's own store path
    (see fft_plangen.LargeTwiddlePlan): multiply by a value fetched from the
    large-twiddle DRAM table at exactly the address this output is about to
    store to (the table shares this kernel's output_mapping layout), rather
    than by a compile-time SIMD constant -- the exponent depends on
    global_uthread_id(), a runtime value, so it cannot be one.
    """
    k = output

    if store.mode == "vector":
        assert store.base_offset is not None
        e.add(
            f"        var ltwr{k} = p.large_twiddle_real_base.load[width={width}]("
            f"out_batch_base + {store.base_offset})"
        )
        e.add(
            f"        var ltwi{k} = p.large_twiddle_imag_base.load[width={width}]("
            f"out_batch_base + {store.base_offset})"
        )
    else:
        rr_lanes: list[str] = []
        ii_lanes: list[str] = []
        for lane, offset in enumerate(store.lane_offsets):
            rr_scalar = f"ltwr{k}_lane{lane}"
            ii_scalar = f"ltwi{k}_lane{lane}"
            e.add(
                f"        var {rr_scalar} = p.large_twiddle_real_base."
                f"load[width=1](out_batch_base + {offset})"
            )
            e.add(
                f"        var {ii_scalar} = p.large_twiddle_imag_base."
                f"load[width=1](out_batch_base + {offset})"
            )
            rr_lanes.append(f"{rr_scalar}[0]")
            ii_lanes.append(f"{ii_scalar}[0]")
        # Lanes past what this store actually writes never reach DRAM, so
        # they are padded with the neutral rotation (1, 0) rather than
        # fetched -- same convention fft_plangen._make_twiddle uses for a
        # partial SIMD batch's compile-time twiddle.
        pad = width - len(rr_lanes)
        rr_lanes += ["Float32(1)"] * pad
        ii_lanes += ["Float32(0)"] * pad
        e.add(
            f"        var ltwr{k} = SIMD[DType.float32, {width}](" + ", ".join(rr_lanes) + ")"
        )
        e.add(
            f"        var ltwi{k} = SIMD[DType.float32, {width}](" + ", ".join(ii_lanes) + ")"
        )

    e.add(f"        var ltr{k} = or{k} * ltwr{k} - oi{k} * ltwi{k}")
    e.add(f"        var lti{k} = or{k} * ltwi{k} + oi{k} * ltwr{k}")
    e.add(f"        or{k} = ltr{k}")
    e.add(f"        oi{k} = lti{k}")


def _emit_store(
    e: Emitter, *, plan: FFTCodegenPlan, output: int, store: StorePlan
) -> None:
    if store.mode == "vector":
        assert store.base_offset is not None
        if store.destination == "output":
            e.add(
                f"        p.output_real_base.store(out_batch_base + {store.base_offset}, "
                f"or{output})"
            )
            e.add(
                f"        p.output_imag_base.store(out_batch_base + {store.base_offset}, "
                f"oi{output})"
            )
        else:
            assert store.buffer_name is not None
            buf = _spad(plan.kernel_name, store.buffer_name)
            e.add(
                f"        {buf}.store(spad_base + {store.base_offset}, or{output})"
            )
            e.add(
                f"        {buf}.store(spad_base + {plan.length} + {store.base_offset}, oi{output})"
            )
        return

    for lane, offset in enumerate(store.lane_offsets):
        if store.destination == "output":
            e.add(
                f"        p.output_real_base.store(out_batch_base + {offset}, "
                f"or{output}[{lane}])"
            )
            e.add(
                f"        p.output_imag_base.store(out_batch_base + {offset}, "
                f"oi{output}[{lane}])"
            )
        else:
            assert store.buffer_name is not None
            buf = _spad(plan.kernel_name, store.buffer_name)
            e.add(
                f"        {buf}.store(spad_base + {offset}, or{output}[{lane}])"
            )
            e.add(
                f"        {buf}.store(spad_base + {plan.length} + {offset}, oi{output}[{lane}])"
            )


def _emit_output(
    e: Emitter, *, plan: FFTCodegenPlan, output_plan: OutputPlan, width: int
) -> None:
    k = output_plan.output
    if output_plan.twiddle is not None:
        _emit_twiddle(e, output=k, twiddle=output_plan.twiddle, width=width)

    if output_plan.scale is not None:
        scale = _f32(output_plan.scale)
        e.add(f"        or{k} *= {scale}")
        e.add(f"        oi{k} *= {scale}")

    if output_plan.large_twiddle:
        _emit_large_twiddle(e, output=k, store=output_plan.store, width=width)

    _emit_store(e, plan=plan, output=k, store=output_plan.store)
    e.add()


def _chunk_load(load: LoadPlan, offset: int, width: int) -> LoadPlan:
    if load.mode == "vector":
        assert load.base_offset is not None
        return LoadPlan(
            operand=load.operand,
            source=load.source,
            buffer_name=load.buffer_name,
            mode="vector",
            base_offset=load.base_offset + offset,
        )
    return LoadPlan(
        operand=load.operand,
        source=load.source,
        buffer_name=load.buffer_name,
        mode="scalar_pack",
        packed_lane_offsets=load.packed_lane_offsets[offset : offset + width],
    )


def _chunk_twiddle(twiddle: TwiddlePlan | None, offset: int, width: int) -> TwiddlePlan | None:
    if twiddle is None:
        return None
    return TwiddlePlan(
        real=twiddle.real[offset : offset + width],
        imag=twiddle.imag[offset : offset + width],
    )


def _chunk_store(store: StorePlan, offset: int, width: int) -> StorePlan:
    if store.mode == "vector":
        assert store.base_offset is not None
        return StorePlan(
            destination=store.destination,
            buffer_name=store.buffer_name,
            mode="vector",
            base_offset=store.base_offset + offset,
        )
    # `lane_offsets` is exactly `valid_lanes` long, which may be shorter
    # than `plan.simd_lanes` (a tail batch) -- slicing past its end (a
    # chunk entirely beyond valid_lanes) yields the empty tuple, which is
    # exactly right: nothing in that chunk is ever written.
    return StorePlan(
        destination=store.destination,
        buffer_name=store.buffer_name,
        mode="scalar_lanes",
        lane_offsets=store.lane_offsets[offset : offset + width],
    )


def _chunk_batch(
    batch: SIMDBatchPlan, *, simd_lanes: int, compute_lanes: int
) -> list[tuple[int, SIMDBatchPlan]]:
    """Split one hardware-width (`simd_lanes`) SIMDBatchPlan into one or
    more smaller `compute_lanes`-wide sub-batches for code emission.

    `simd_lanes` is the plan's launch granule -- it sizes `PooledRange`
    (see `src/m2ndp.mojo`'s `VECTOR_WIDTH`/`PooledRange.over`) and must
    stay exactly what the planner decided; nothing here touches offsets,
    uthread counts, or `plan.simd_lanes` itself. `compute_lanes` only
    controls how wide a vector *instruction* the kernel body emits to
    process that one already-decided batch -- e.g. one 8-wide batch becomes
    two 4-wide chunks instead of one 8-wide (LMUL=2 on this target's
    128-bit VLEN) operation. A smaller compute width avoids the RVV
    register-spill/insertelement patterns this target's simulator cannot
    run (see docs/STATUS.md); it costs nothing else since `emit_butterfly`
    itself is already width-agnostic (see fft_butterflies.py) and each
    chunk becomes its own block scope exactly like today's multi-batch case.

    Returns `[(width, chunk), ...]`; `width == simd_lanes` and a single
    element when `compute_lanes >= simd_lanes` (today's behavior, byte
    for byte).
    """
    if compute_lanes >= simd_lanes:
        return [(simd_lanes, batch)]

    chunks: list[tuple[int, SIMDBatchPlan]] = []
    n_chunks = (simd_lanes + compute_lanes - 1) // compute_lanes
    for c in range(n_chunks):
        offset = c * compute_lanes
        width = min(compute_lanes, simd_lanes - offset)
        loads = tuple(_chunk_load(load, offset, width) for load in batch.loads)
        outputs = tuple(
            OutputPlan(
                output=output_plan.output,
                twiddle=_chunk_twiddle(output_plan.twiddle, offset, width),
                scale=output_plan.scale,
                large_twiddle=output_plan.large_twiddle,
                store=_chunk_store(output_plan.store, offset, width),
            )
            for output_plan in batch.outputs
        )
        valid_lanes = max(0, min(width, batch.valid_lanes - offset))
        chunks.append(
            (
                width,
                SIMDBatchPlan(
                    batch_id=batch.batch_id,
                    valid_lanes=valid_lanes,
                    loads=loads,
                    outputs=outputs,
                ),
            )
        )
    return chunks


def _emit_batch(
    e: Emitter,
    *,
    plan: FFTCodegenPlan,
    stage: FFTStagePlan,
    batch: SIMDBatchPlan,
    width: int,
) -> None:
    e.add(
        f"        # ===== stage {stage.stage_id}, SIMD batch {batch.batch_id} "
        f"(valid lanes: {batch.valid_lanes}/{width}) ====="
    )

    for load in batch.loads:
        _emit_load(e, plan=plan, load=load, width=width)
    e.add()

    outputs_by_k = {output_plan.output: output_plan for output_plan in batch.outputs}

    def on_output(k: int) -> None:
        _emit_output(e, plan=plan, output_plan=outputs_by_k[k], width=width)

    emit_butterfly(
        e,
        indent="        ",
        radix=stage.radix,
        inverse=stage.inverse,
        on_output=on_output,
    )
    e.add()


def _emit_stage(
    e: Emitter, *, plan: FFTCodegenPlan, stage: FFTStagePlan, compute_lanes: int | None = None
) -> None:
    is_first = stage.stage_id == 0
    is_last = stage.stage_id == len(plan.stages) - 1

    e.add("    @staticmethod")
    e.add(f"    def stage_{stage.stage_id}():")
    e.add(f"        ref p = {plan.kernel_name}.params[]")
    e.add(f"        comptime RADIX = {stage.radix}")
    e.add(f"        comptime SIMD_ITERS = {stage.simd_iteration_count}")
    e.add()
    e.add("        var local_id = local_uthread_id()")
    e.add(f"        if local_id >= MAX_UTHREAD_{plan.kernel_name}:")
    e.add("            return")
    # A single-stage kernel (this kernel's own length == its one radix, as
    # every decomposed sub-FFT kernel is) never reads or writes its own
    # scratchpad -- first_stage and last_stage are the same stage, so every
    # load/store is DRAM-side. Only declare spad_base where something uses it.
    if plan.scratchpad_buffers:
        e.add(f"        var spad_base = local_id * {plan.scratchpad_uthread_stride}")
    # Each kernel's own AddressMapping settles this in one runtime multiply
    # (plus, only where the mapping isn't the origin, one add) -- except
    # SPLIT, a middle kernel's output in an M>=3 chain, which costs one %
    # and one // (see _mapping_base_expr). Declared only on the stage that
    # actually reads/writes DRAM through it.
    if is_first:
        e.add(
            f"        var in_batch_base = "
            f"{_mapping_base_expr(plan.input_mapping, plan.length)}"
        )
    if is_last:
        e.add(
            f"        var out_batch_base = "
            f"{_mapping_base_expr(plan.output_mapping, plan.length)}"
        )
    e.add()

    # Each batch's rr{k}/ii{k}/or{k}/oi{k} (and friends) are local to that
    # batch's own butterfly, not threads carried across batches -- but
    # _emit_batch always names them the same way regardless of batch_id, so
    # a stage with more than one SIMD batch (or, now, more than one
    # compute-width chunk within a batch -- see _chunk_batch) needs each
    # piece in its own block scope or the next one's `var rr0` redefines
    # the previous.
    pieces: list[tuple[int, SIMDBatchPlan]] = []
    for batch in stage.batches:
        pieces.extend(
            _chunk_batch(
                batch,
                simd_lanes=plan.simd_lanes,
                compute_lanes=compute_lanes if compute_lanes is not None else plan.simd_lanes,
            )
        )

    for width, piece in pieces:
        if len(pieces) > 1:
            sub = Emitter()
            _emit_batch(sub, plan=plan, stage=stage, batch=piece, width=width)
            e.add(f"        if True:  # batch {piece.batch_id} scope")
            for line in sub.lines:
                e.add("    " + line if line else "")
        else:
            _emit_batch(e, plan=plan, stage=stage, batch=piece, width=width)


def _emit_params_struct(e: Emitter, *, plan: FFTCodegenPlan) -> None:
    e.add("@fieldwise_init")
    e.add(f"struct {plan.kernel_name}Params(Movable):")
    e.add("    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    if plan.large_twiddle is not None:
        e.add("    var large_twiddle_real_base: UnsafePointer[Float32, MutAnyOrigin]")
        e.add("    var large_twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add()
    e.add()


def _emit_task_struct(
    e: Emitter, *, plan: FFTCodegenPlan, compute_lanes: int | None = None
) -> None:
    """The NDPTask struct: its scratchpad buffers (each uthread's own
    region, sized by scratchpad_uthread_stride -- see module docstring),
    its per-stage kernels, and device_main. Identical in shape whether this
    plan is a whole single-kernel FFT or one half of a decomposed one.

    `compute_lanes`: see `_chunk_batch` -- the SIMD width the emitted
    arithmetic instructions use, independent of `plan.simd_lanes` (the
    launch granule). `None` (the default) keeps every stage emitted at
    `plan.simd_lanes`, unchanged from before this parameter existed.
    """
    e.add(f"struct {plan.kernel_name}(NDPTask):")
    e.add(f"    comptime Params = {plan.kernel_name}Params")
    e.add()

    for buffer in plan.scratchpad_buffers:
        e.add(
            f'    comptime {buffer.name} = scratchpad[{buffer.elements}, Float32, '
            f'name="{plan.kernel_name.lower()}_{buffer.name}"]()'
        )
    if plan.scratchpad_buffers:
        e.add()

    for stage in plan.stages:
        _emit_stage(e, plan=plan, stage=stage, compute_lanes=compute_lanes)

    e.add("    @staticmethod")
    e.add("    def device_main():")
    for stage in plan.stages:
        e.add(f"        launch_parallel[{plan.kernel_name}.stage_{stage.stage_id}]()")
    e.add()
    e.add()


def _emit_kernel(e: Emitter, *, plan: FFTCodegenPlan, compute_lanes: int | None = None) -> None:
    e.add(f"comptime MAX_UTHREAD_{plan.kernel_name} = {plan.max_uthread}")
    e.add()
    _emit_params_struct(e, plan=plan)
    _emit_task_struct(e, plan=plan, compute_lanes=compute_lanes)


def _emit_prelude(e: Emitter) -> None:
    e.add("from std.sys import size_of")
    e.add("from std.random import random_float64, seed")
    e.add("from std.math import cos as host_cos, sin as host_sin")
    e.add()
    e.add(
        "from m2ndp import VECTOR_WIDTH, NDPTask, PooledRange, "
        "global_uthread_id, local_uthread_id, launch_parallel, scratchpad"
    )
    e.add("from m2ndp_host import cxl_alloc")
    e.add()
    e.add("comptime W = VECTOR_WIDTH // size_of[Float32]()")


def _emit_reference_check(
    e: Emitter,
    *,
    n: int,
    batch_count: int,
    inverse: bool,
    input_real: str,
    input_imag: str,
    output_real: str,
    output_imag: str,
    ref_real: str,
    ref_imag: str,
    tolerance: float,
    label: str,
) -> None:
    """A direct O(N^2) DFT, computed at host runtime against whatever input
    was just randomly generated -- independent of whichever radix
    decomposition or kernel split actually ran, single-kernel or
    decomposed alike (see fft_plangen.HostPlan / DecomposedHostPlan).
    Accumulated in Float64 so this check doesn't share the kernel's own
    fp32 rounding, rounding to Float32 only once, at the very end.

    `batch_count` independent length-`n` FFTs share one buffer, back to
    back (`batch*n + n_`) -- a single-kernel plan's own total_uthreads;
    always 1 for a decomposed plan (see DecomposedFFTPlan).
    """
    e.add("    var pi = Float64(3.141592653589793)")
    e.add(f"    var sign = Float64({1.0 if inverse else -1.0})")
    e.add(f"    for batch in range({batch_count}):")
    e.add(f"        var batch_base = batch * {n}")
    e.add(f"        for k in range({n}):")
    e.add("            var acc_r = Float64(0)")
    e.add("            var acc_i = Float64(0)")
    e.add(f"            for n_ in range({n}):")
    e.add(
        "                var angle = sign * 2.0 * pi * Float64(n_) * Float64(k) / "
        f"Float64({n})"
    )
    e.add("                var c = host_cos(angle)")
    e.add("                var s = host_sin(angle)")
    e.add(f"                var xr = Float64({input_real}[batch_base + n_])")
    e.add(f"                var xi = Float64({input_imag}[batch_base + n_])")
    e.add("                acc_r += xr * c - xi * s")
    e.add("                acc_i += xr * s + xi * c")
    if inverse:
        e.add(f"            acc_r /= Float64({n})")
        e.add(f"            acc_i /= Float64({n})")
    e.add(f"            {ref_real}[batch_base + k] = Float32(acc_r)")
    e.add(f"            {ref_imag}[batch_base + k] = Float32(acc_i)")
    e.add()

    e.add(f"    var tol = {_f32(tolerance)}")
    e.add(f"    for i in range({n * batch_count}):")
    e.add(f"        var err_r = {output_real}[i] - {ref_real}[i]")
    e.add(f"        var err_i = {output_imag}[i] - {ref_imag}[i]")
    e.add("        if err_r < Float32(0):")
    e.add("            err_r = -err_r")
    e.add("        if err_i < Float32(0):")
    e.add("            err_i = -err_i")
    e.add("        if err_r > tol or err_i > tol:")
    e.add(f'            print("[host] {label} mismatch at", i)')
    e.add(f'            print("  expected:", {ref_real}[i], {ref_imag}[i])')
    e.add(f'            print("  actual:  ", {output_real}[i], {output_imag}[i])')
    e.add('            print("  error:   ", err_r, err_i)')
    e.add("            return")
    e.add()
    e.add(f'    print("[host] {label} verification passed")')


def generate_fft_kernel(plan: FFTCodegenPlan, *, compute_lanes: int | None = None) -> str:
    """Render a single-kernel plan: the whole FFT in one NDPTask, one
    launch. This function performs no FFT planning. `compute_lanes`: see
    `_chunk_batch`; `None` (the default) keeps today's output unchanged.
    """
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
    e.add("    var output_real = cxl_alloc[Float32](total_elems)")
    e.add("    var output_imag = cxl_alloc[Float32](total_elems)")
    e.add("    var ref_real = cxl_alloc[Float32](total_elems)")
    e.add("    var ref_imag = cxl_alloc[Float32](total_elems)")
    e.add()
    e.add(f"    var pool_elems = {host.pool_elems}")
    e.add("    var uthread_pool = cxl_alloc[Float32](pool_elems)")
    e.add()

    # Fresh random input every run (std.random, seeded like every other
    # benchmark's main()), not a signal fixed once at generation time.
    e.add("    seed(0)")
    e.add("    for i in range(total_elems):")
    e.add("        input_real[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        input_imag[i] = Float32(random_float64(-1.0, 1.0))")
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
        batch_count=plan.total_uthreads,
        inverse=plan.inverse,
        input_real="input_real",
        input_imag="input_imag",
        output_real="output_real",
        output_imag="output_imag",
        ref_real="ref_real",
        ref_imag="ref_imag",
        tolerance=host.tolerance,
        label="FFT",
    )

    return e.text()


def generate_decomposed_fft_kernels(
    plan: DecomposedFFTPlan, *, compute_lanes: int | None = None
) -> str:
    """Render a decomposed (N = N0*N1) plan: two NDPTask structs, chained
    through DRAM the way two_tasks.mojo chains Scale/AddB -- launched one
    after the other from one host main(), never through a shared
    scratchpad. This function performs no FFT planning: which kernel owns
    which factor, every AddressMapping, and the large-twiddle table shape
    are already decided in `plan`. `compute_lanes`: see `_chunk_batch`;
    `None` (the default) keeps today's output unchanged.
    """
    e = Emitter()
    _emit_prelude(e)
    e.add(f"comptime N0 = {plan.n0}")
    e.add(f"comptime N1 = {plan.n1}")
    e.add(f"comptime N = {plan.n}")
    e.add()

    _emit_kernel(e, plan=plan.kernel0, compute_lanes=compute_lanes)
    _emit_kernel(e, plan=plan.kernel1, compute_lanes=compute_lanes)

    host = plan.host
    k0 = plan.kernel0
    k1 = plan.kernel1

    e.add("def main() raises:")
    e.add(f"    if {k0.kernel_name}.emit_ir_if_asked():")
    e.add("        return")
    e.add()
    e.add(f"    var n = {host.n}")
    e.add("    var input_real = cxl_alloc[Float32](n)")
    e.add("    var input_imag = cxl_alloc[Float32](n)")
    e.add("    var mid_real = cxl_alloc[Float32](n)")
    e.add("    var mid_imag = cxl_alloc[Float32](n)")
    e.add("    var output_real = cxl_alloc[Float32](n)")
    e.add("    var output_imag = cxl_alloc[Float32](n)")
    e.add("    var ref_real = cxl_alloc[Float32](n)")
    e.add("    var ref_imag = cxl_alloc[Float32](n)")
    e.add()

    assert plan.kernel0.large_twiddle is not None
    lt = plan.kernel0.large_twiddle
    e.add(f"    var large_twiddle_real = cxl_alloc[Float32](n)")
    e.add(f"    var large_twiddle_imag = cxl_alloc[Float32](n)")
    e.add("    var lt_pi = Float64(3.141592653589793)")
    e.add(f"    var lt_sign = Float64({1.0 if lt.inverse else -1.0})")
    e.add("    var r = 0")
    e.add(f"    while r < {lt.row_count}:")
    e.add("        var c1 = 0")
    e.add(f"        while c1 < {lt.output_count}:")
    e.add(
        "            var angle = lt_sign * 2.0 * lt_pi * Float64(r) * Float64(c1) / "
        f"Float64({lt.full_length})"
    )
    e.add(f"            large_twiddle_real[r * {lt.output_count} + c1] = Float32(host_cos(angle))")
    e.add(f"            large_twiddle_imag[r * {lt.output_count} + c1] = Float32(host_sin(angle))")
    e.add("            c1 += 1")
    e.add("        r += 1")
    e.add()

    e.add(f"    var pool0_elems = {k0.simd_lanes * k0.total_uthreads}")
    e.add(f"    var pool1_elems = {k1.simd_lanes * k1.total_uthreads}")
    e.add("    var pool0 = cxl_alloc[Float32](pool0_elems)")
    e.add("    var pool1 = cxl_alloc[Float32](pool1_elems)")
    e.add()

    e.add("    seed(0)")
    e.add("    for i in range(n):")
    e.add("        input_real[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        input_imag[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        mid_real[i] = Float32(0)")
    e.add("        mid_imag[i] = Float32(0)")
    e.add("        output_real[i] = Float32(0)")
    e.add("        output_imag[i] = Float32(0)")
    e.add("        ref_real[i] = Float32(0)")
    e.add("        ref_imag[i] = Float32(0)")
    e.add()

    e.add(f"    var rc0 = {k0.kernel_name}.launch(")
    e.add("        PooledRange.over(pool0, pool0_elems),")
    e.add(
        f"        {k0.kernel_name}Params(input_real, input_imag, mid_real, mid_imag, "
        "large_twiddle_real, large_twiddle_imag),"
    )
    e.add("    )")
    e.add("    if rc0 != 0:")
    e.add('        print("[host] FFT kernel0 failed, exit", rc0)')
    e.add("        return")
    e.add()

    e.add(f"    var rc1 = {k1.kernel_name}.launch(")
    e.add("        PooledRange.over(pool1, pool1_elems),")
    e.add(
        f"        {k1.kernel_name}Params(mid_real, mid_imag, output_real, output_imag),"
    )
    e.add("    )")
    e.add("    if rc1 != 0:")
    e.add('        print("[host] FFT kernel1 failed, exit", rc1)')
    e.add("        return")
    e.add()

    _emit_reference_check(
        e,
        n=plan.n,
        batch_count=1,
        inverse=plan.inverse,
        input_real="input_real",
        input_imag="input_imag",
        output_real="output_real",
        output_imag="output_imag",
        ref_real="ref_real",
        ref_imag="ref_imag",
        tolerance=host.tolerance,
        label="decomposed FFT",
    )

    return e.text()


def _emit_large_twiddle_table_precompute(
    e: Emitter, *, lt: LargeTwiddlePlan, real_name: str, imag_name: str, suffix: str
) -> None:
    """Host precompute for one non-last kernel's large-twiddle table (see
    fft_plangen.LargeTwiddlePlan / generate_multi_kernel_fft_kernels).
    `real_name`/`imag_name` are each `lt.full_length` Float32 elements,
    filled at every address the on-device fetch will ever read.

    Every local variable is suffixed: this runs once per non-last kernel
    in the same main(), and an earlier version of the single-kernel host
    check hit exactly this collision (two `var pi = ...`s in one scope --
    see the fft_plangen.make_decomposed_plan inverse_scale/lt_pi fix) for
    the same reason.
    """
    sign = 1.0 if lt.inverse else -1.0
    pi, lsign = f"lt_pi_{suffix}", f"lt_sign_{suffix}"
    r, c1 = f"lt_r_{suffix}", f"lt_c1_{suffix}"
    angle, val_r, val_i = f"lt_angle_{suffix}", f"lt_val_r_{suffix}", f"lt_val_i_{suffix}"

    e.add(f"    var {pi} = Float64(3.141592653589793)")
    e.add(f"    var {lsign} = Float64({sign})")
    e.add(f"    var {r} = 0")
    e.add(f"    while {r} < {lt.row_count}:")
    e.add(f"        var {c1} = 0")
    e.add(f"        while {c1} < {lt.output_count}:")
    e.add(
        f"            var {angle} = {lsign} * 2.0 * {pi} * Float64({r}) * "
        f"Float64({c1}) * Float64({lt.a}) / Float64({lt.full_length})"
    )
    e.add(f"            var {val_r} = Float32(host_cos({angle}))")
    e.add(f"            var {val_i} = Float32(host_sin({angle}))")
    if lt.k_next:
        # PEELED-addressed (see AddressMappingKind.PEELED): fill every
        # address the fetch can land on, for every already-transformed
        # out_a in [0, a) -- mirrors the kernel's own write formula
        # exactly (fft_codegen._mapping_base_expr's PEELED branch), split
        # here into d_next/rest since this loop already has `r` (=
        # remaining) directly rather than a packed address to invert.
        out_a, addr = f"lt_out_a_{suffix}", f"lt_addr_{suffix}"
        d_next, rest = f"lt_dnext_{suffix}", f"lt_rest_{suffix}"
        e.add(f"            var {d_next} = {r} // {lt.tail_size}")
        e.add(f"            var {rest} = {r} % {lt.tail_size}")
        e.add(f"            var {out_a} = 0")
        e.add(f"            while {out_a} < {lt.a}:")
        e.add(
            f"                var {addr} = {rest} * {lt.a * lt.output_count * lt.k_next} + "
            f"({out_a} + {c1} * {lt.a}) * {lt.k_next} + {d_next}"
        )
        e.add(f"                {real_name}[{addr}] = {val_r}")
        e.add(f"                {imag_name}[{addr}] = {val_i}")
        e.add(f"                {out_a} += 1")
    elif lt.a == 1:
        # Dense: every address visited exactly once.
        e.add(f"            {real_name}[{r} * {lt.output_count} + {c1}] = {val_r}")
        e.add(f"            {imag_name}[{r} * {lt.output_count} + {c1}] = {val_i}")
    else:
        # a-fold redundant: the SPLIT-addressed fetch reads the same value
        # for every out_a in [0, a) -- see AddressMappingKind.SPLIT.
        out_a, addr = f"lt_out_a_{suffix}", f"lt_addr_{suffix}"
        e.add(f"            var {out_a} = 0")
        e.add(f"            while {out_a} < {lt.a}:")
        e.add(
            f"                var {addr} = {out_a} + {c1} * {lt.a} + "
            f"{r} * {lt.a * lt.output_count}"
        )
        e.add(f"                {real_name}[{addr}] = {val_r}")
        e.add(f"                {imag_name}[{addr}] = {val_i}")
        e.add(f"                {out_a} += 1")
    e.add(f"            {c1} += 1")
    e.add(f"        {r} += 1")
    e.add()


def generate_multi_kernel_fft_kernels(
    plan: MultiKernelFFTPlan, *, compute_lanes: int | None = None
) -> str:
    """Render an M-kernel chained plan (see fft_plangen.make_multi_kernel_plan):
    M NDPTask structs, chained through DRAM one launch after another from
    one host main(), each non-last kernel's own large-twiddle table
    precomputed alongside it. Generalizes generate_decomposed_fft_kernels
    (exactly M=2, one bare radix per kernel) to any M>=1 and to each
    kernel's own layouts_for_radices multi-stage structure. This function
    performs no FFT planning: every AddressMapping and LargeTwiddlePlan is
    already decided in `plan`. `compute_lanes`: see `_chunk_batch`; `None`
    (the default) keeps today's output unchanged.
    """
    e = Emitter()
    _emit_prelude(e)
    e.add(f"comptime N = {plan.n}")
    e.add()

    for kernel in plan.kernels:
        _emit_kernel(e, plan=kernel, compute_lanes=compute_lanes)

    host = plan.host
    m = len(plan.kernels)
    k0 = plan.kernels[0]

    def buf_name(idx: int, part: str) -> str:
        # idx: -1 is the original input, m-1 is the final output, anything
        # in between is the intermediate buffer that many kernels wrote.
        if idx == -1:
            return f"input_{part}"
        if idx == m - 1:
            return f"output_{part}"
        return f"mid{idx}_{part}"

    e.add("def main() raises:")
    e.add(f"    if {k0.kernel_name}.emit_ir_if_asked():")
    e.add("        return")
    e.add()
    e.add(f"    var n = {host.n}")
    e.add("    var input_real = cxl_alloc[Float32](n)")
    e.add("    var input_imag = cxl_alloc[Float32](n)")
    for i in range(m - 1):
        e.add(f"    var mid{i}_real = cxl_alloc[Float32](n)")
        e.add(f"    var mid{i}_imag = cxl_alloc[Float32](n)")
    e.add("    var output_real = cxl_alloc[Float32](n)")
    e.add("    var output_imag = cxl_alloc[Float32](n)")
    e.add("    var ref_real = cxl_alloc[Float32](n)")
    e.add("    var ref_imag = cxl_alloc[Float32](n)")
    e.add()

    for i, kernel in enumerate(plan.kernels[:-1]):
        assert kernel.large_twiddle is not None
        e.add(f"    var large_twiddle{i}_real = cxl_alloc[Float32](n)")
        e.add(f"    var large_twiddle{i}_imag = cxl_alloc[Float32](n)")
        _emit_large_twiddle_table_precompute(
            e,
            lt=kernel.large_twiddle,
            real_name=f"large_twiddle{i}_real",
            imag_name=f"large_twiddle{i}_imag",
            suffix=str(i),
        )

    for i, kernel in enumerate(plan.kernels):
        e.add(f"    var pool{i}_elems = {kernel.simd_lanes * kernel.total_uthreads}")
        e.add(f"    var pool{i} = cxl_alloc[Float32](pool{i}_elems)")
    e.add()

    e.add("    seed(0)")
    e.add("    for i in range(n):")
    e.add("        input_real[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        input_imag[i] = Float32(random_float64(-1.0, 1.0))")
    for i in range(m - 1):
        e.add(f"        mid{i}_real[i] = Float32(0)")
        e.add(f"        mid{i}_imag[i] = Float32(0)")
    e.add("        output_real[i] = Float32(0)")
    e.add("        output_imag[i] = Float32(0)")
    e.add("        ref_real[i] = Float32(0)")
    e.add("        ref_imag[i] = Float32(0)")
    e.add()

    for i, kernel in enumerate(plan.kernels):
        is_last = i == m - 1
        in_r, in_i = buf_name(i - 1, "real"), buf_name(i - 1, "imag")
        out_r, out_i = buf_name(i, "real"), buf_name(i, "imag")
        e.add(f"    var rc{i} = {kernel.kernel_name}.launch(")
        e.add(f"        PooledRange.over(pool{i}, pool{i}_elems),")
        if is_last:
            e.add(
                f"        {kernel.kernel_name}Params({in_r}, {in_i}, {out_r}, {out_i}),"
            )
        else:
            e.add(
                f"        {kernel.kernel_name}Params({in_r}, {in_i}, {out_r}, {out_i}, "
                f"large_twiddle{i}_real, large_twiddle{i}_imag),"
            )
        e.add("    )")
        e.add(f"    if rc{i} != 0:")
        e.add(f'        print("[host] FFT kernel{i} failed, exit", rc{i})')
        e.add("        return")
        e.add()

    _emit_reference_check(
        e,
        n=plan.n,
        batch_count=1,
        inverse=plan.inverse,
        input_real="input_real",
        input_imag="input_imag",
        output_real="output_real",
        output_imag="output_imag",
        ref_real="ref_real",
        ref_imag="ref_imag",
        tolerance=host.tolerance,
        label="multi-kernel FFT",
    )

    return e.text()
