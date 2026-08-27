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

All of those decisions are already materialized in fft_plan_core.FFTCodegenPlan
/ fft_plan_simple.DecomposedFFTPlan. A decomposed FFT is two ordinary
FFTCodegenPlan kernels (see fft_plan_simple's module docstring); this module
renders each exactly as it would a single-kernel plan, and additionally
threads the large-twiddle table and the two kernels' DRAM hand-off through
one combined main() -- there is no separate "transpose kernel" abstraction
to emit.

Two *rendering-shape* choices this module used to decide inline -- whether a
stage's per-batch code loops or unrolls, and how a batch splits into
narrower compute_lanes-wide chunks -- now live in codegen/lowering.py
(imported here as `_try_build_loop_stage`/`_chunk_batch`) instead: still not
FFT planning (no address, radix, or twiddle-value decision), but not pure
text emission either, so they get their own module rather than blurring
this one's own "no planning" contract.
"""

from dataclasses import dataclass

from codegen.common import (
    Emitter,
    emit_array_dump as _emit_array_dump,
    emit_large_twiddle_table_precompute as _emit_large_twiddle_table_precompute,
    emit_prelude as _emit_prelude,
    emit_reference_check as _emit_reference_check,
    f32 as _f32,
    spad as _spad,
)
from codegen.fft_butterflies import emit_butterfly
from codegen.lowering import (
    LOOP_MIN_FULL_BATCHES as _LOOP_MIN_FULL_BATCHES,
    LoopLoad as _LoopLoad,
    LoopStagePlan as _LoopStagePlan,
    LoopStoreInfo as _LoopStoreInfo,
    chunk_batch as _chunk_batch,
    try_build_loop_stage as _try_build_loop_stage,
)
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


def _mapping_base_expr(mapping: AddressMapping, kernel_length: int) -> str:
    """`global_uthread_id() * row_stride [+ base]` -- the one runtime
    multiply a CONTIGUOUS/STRIDED AddressMapping ever costs. `elem*elem_stride`
    is folded into each load/store's own offset at plan time (see
    fft_plan_core._make_load / _make_store), so that's the whole of what
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
    (see fft_plan_core.LargeTwiddlePlan): multiply by a value fetched from the
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
        # fetched -- same convention fft_plan_core._make_twiddle uses for a
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


def _emit_load_loop(
    e: Emitter, *, plan: FFTCodegenPlan, load: _LoopLoad, offset_expr: str, width: int
) -> None:
    j = load.operand
    if load.source == "input":
        e.add(
            f"        var rr{j} = p.input_real_base.load[width={width}]("
            f"in_batch_base + {offset_expr})"
        )
        e.add(
            f"        var ii{j} = p.input_imag_base.load[width={width}]("
            f"in_batch_base + {offset_expr})"
        )
    else:
        assert load.buffer_name is not None
        buf = _spad(plan.kernel_name, load.buffer_name)
        e.add(
            f"        var rr{j} = {buf}.load[DType.float32, {width}]("
            f"spad_base + {offset_expr})"
        )
        e.add(
            f"        var ii{j} = {buf}.load[DType.float32, {width}]("
            f"spad_base + {plan.length} + {offset_expr})"
        )


def _emit_twiddle_loop(
    e: Emitter, *, output: int, table_offset_expr: str, width: int
) -> None:
    k = output
    e.add(
        f"        var twr{k} = p.loop_twiddle_real_base.load[width={width}]("
        f"{table_offset_expr})"
    )
    e.add(
        f"        var twi{k} = p.loop_twiddle_imag_base.load[width={width}]("
        f"{table_offset_expr})"
    )
    e.add(f"        var tr{k} = or{k} * twr{k} - oi{k} * twi{k}")
    e.add(f"        var ti{k} = or{k} * twi{k} + oi{k} * twr{k}")
    e.add(f"        or{k} = tr{k}")
    e.add(f"        oi{k} = ti{k}")


def _emit_store_loop(
    e: Emitter, *, plan: FFTCodegenPlan, output: int, store: _LoopStoreInfo,
    residue: int, chunk_offset: int, width: int,
) -> None:
    k = output
    if store.mode == "vector":
        assert store.bases is not None and store.stride is not None
        base = store.bases[residue] + chunk_offset
        offset_expr = f"({base} + outer_it * {store.stride})"
        if store.destination == "output":
            e.add(f"        p.output_real_base.store(out_batch_base + {offset_expr}, or{k})")
            e.add(f"        p.output_imag_base.store(out_batch_base + {offset_expr}, oi{k})")
        else:
            assert store.buffer_name is not None
            buf = _spad(plan.kernel_name, store.buffer_name)
            e.add(f"        {buf}.store(spad_base + {offset_expr}, or{k})")
            e.add(f"        {buf}.store(spad_base + {plan.length} + {offset_expr}, oi{k})")
        return

    assert store.lane_bases is not None and store.lane_stride is not None
    # Store vectorization, loop-stage case: mirrors _chunk_store's own
    # equivalent-instruction-selection exactly, just against this
    # renderer's own per-lane representation (_LoopStagePlan.lane_bases/
    # lane_stride, filled in _try_build_loop_stage from the same exact
    # StorePlan.lane_offsets fft_plan_core._make_store already computed --
    # nothing here recomputes an address). This width-wide slice of lanes
    # can share one vector store exactly when both hold: the lanes' own
    # bases are contiguous (stride 1, same fact _chunk_store checks) *and*
    # every one of them advances by the same amount per outer-loop
    # iteration (`lane_stride` uniform across the slice) -- only then does
    # a single `+ outer_it * stride` term stay correct for every lane the
    # vector store would cover at once. Not a new layout decision: every
    # value compared below was already fixed by the planner.
    lane_range = range(chunk_offset, chunk_offset + width)
    bases = [store.lane_bases[lane][residue] for lane in lane_range]
    strides = [store.lane_stride[lane] for lane in lane_range]
    if width > 0 and all(s == strides[0] for s in strides) and all(
        bases[i] == bases[0] + i for i in range(1, width)
    ):
        offset_expr = f"({bases[0]} + outer_it * {strides[0]})"
        if store.destination == "output":
            e.add(f"        p.output_real_base.store(out_batch_base + {offset_expr}, or{k})")
            e.add(f"        p.output_imag_base.store(out_batch_base + {offset_expr}, oi{k})")
        else:
            assert store.buffer_name is not None
            buf = _spad(plan.kernel_name, store.buffer_name)
            e.add(f"        {buf}.store(spad_base + {offset_expr}, or{k})")
            e.add(f"        {buf}.store(spad_base + {plan.length} + {offset_expr}, oi{k})")
        return

    for lane in range(width):
        abs_lane = chunk_offset + lane
        base = store.lane_bases[abs_lane][residue]
        offset_expr = f"({base} + outer_it * {store.lane_stride[abs_lane]})"
        if store.destination == "output":
            e.add(f"        p.output_real_base.store(out_batch_base + {offset_expr}, or{k}[{lane}])")
            e.add(f"        p.output_imag_base.store(out_batch_base + {offset_expr}, oi{k}[{lane}])")
        else:
            assert store.buffer_name is not None
            buf = _spad(plan.kernel_name, store.buffer_name)
            e.add(f"        {buf}.store(spad_base + {offset_expr}, or{k}[{lane}])")
            e.add(f"        {buf}.store(spad_base + {plan.length} + {offset_expr}, oi{k}[{lane}])")


def _emit_loop_chunk(
    e: Emitter, *, plan: FFTCodegenPlan, stage: FFTStagePlan, loop_plan: _LoopStagePlan,
    residue: int, chunk_offset: int, width: int,
) -> None:
    e.add(
        f"        # ===== stage {stage.stage_id}, loop residue {residue} chunk offset "
        f"{chunk_offset} (width {width}) ====="
    )
    for load in loop_plan.loads:
        base = load.bases[residue] + chunk_offset
        offset_expr = f"({base} + outer_it * {load.stride})"
        _emit_load_loop(e, plan=plan, load=load, offset_expr=offset_expr, width=width)
    e.add()

    stores_by_k = {s.output: s for s in loop_plan.stores}
    simd_lanes = plan.simd_lanes

    def on_output(k: int) -> None:
        tw = loop_plan.twiddles.get(k)
        if tw is not None:
            row_offset = tw.table_offset + residue * simd_lanes + chunk_offset
            table_offset_expr = f"({row_offset} + outer_it * {loop_plan.period * simd_lanes})"
            _emit_twiddle_loop(e, output=k, table_offset_expr=table_offset_expr, width=width)

        scale = loop_plan.scales.get(k)
        if scale is not None:
            e.add(f"        or{k} *= {_f32(scale)}")
            e.add(f"        oi{k} *= {_f32(scale)}")

        _emit_store_loop(
            e, plan=plan, output=k, store=stores_by_k[k],
            residue=residue, chunk_offset=chunk_offset, width=width,
        )
        e.add()

    emit_butterfly(e, indent="        ", radix=stage.radix, inverse=stage.inverse, on_output=on_output)
    e.add()


def _emit_loop_stage(
    e: Emitter, *, plan: FFTCodegenPlan, stage: FFTStagePlan, loop_plan: _LoopStagePlan,
    compute_lanes: int | None,
) -> None:
    simd_lanes = plan.simd_lanes
    cl = compute_lanes if compute_lanes is not None else simd_lanes
    n_chunks = (simd_lanes + cl - 1) // cl
    period = loop_plan.period

    body = Emitter()
    for r in range(period):
        for c in range(n_chunks):
            chunk_offset = c * cl
            width = min(cl, simd_lanes - chunk_offset)
            if period > 1 or n_chunks > 1:
                sub = Emitter()
                _emit_loop_chunk(
                    sub, plan=plan, stage=stage, loop_plan=loop_plan,
                    residue=r, chunk_offset=chunk_offset, width=width,
                )
                body.add(f"        if True:  # residue {r} chunk {c}")
                for line in sub.lines:
                    body.add("    " + line if line else "")
            else:
                _emit_loop_chunk(
                    body, plan=plan, stage=stage, loop_plan=loop_plan,
                    residue=r, chunk_offset=chunk_offset, width=width,
                )

    e.add("        var outer_it = 0")
    e.add(f"        while outer_it < {loop_plan.outer_iters}:")
    for line in body.lines:
        e.add("    " + line if line else "")
    e.add("            outer_it += 1")
    e.add()


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
    e: Emitter,
    *,
    plan: FFTCodegenPlan,
    stage: FFTStagePlan,
    compute_lanes: int | None = None,
    loop_stages: bool = False,
    min_loop_batches: int = _LOOP_MIN_FULL_BATCHES,
    twiddle_table: list[tuple[float, float]] | None = None,
) -> bool:
    """Returns True if this stage also emitted a separate `stage_{id}_tail`
    static method the caller (`_emit_task_struct`) needs its own
    `launch_parallel` call for -- see the loop_stages tail_batch branch
    below for why that tail is its own device function instead of more
    code appended after the loop inside `stage_{id}` itself."""
    is_first = stage.stage_id == 0
    is_last = stage.stage_id == len(plan.stages) - 1

    def emit_header(name: str) -> None:
        e.add("    @staticmethod")
        e.add(f"    def {name}():")
        e.add(f"        ref p = {plan.kernel_name}.params[]")
        e.add(f"        comptime RADIX = {stage.radix}")
        e.add(f"        comptime SIMD_ITERS = {stage.simd_iteration_count}")
        e.add()

    emit_header(f"stage_{stage.stage_id}")

    if plan.cooperation is not None:
        # loop_stages (the runtime-loop renderer) reasons about `stage.batches`
        # as a whole, not per-worker -- combining it with cooperation is not
        # supported yet (see fft_plan_cooperative.py's own docstring on what
        # this first cut defers); fail loudly rather than silently render
        # every worker's copy of the full batch set.
        if loop_stages:
            raise NotImplementedError(
                "loop_stages is not yet supported for a cooperative FFTCodegenPlan"
            )
        # Local import, not a module-level one: fft_cooperative_codegen.py
        # itself imports _emit_stage_batches/_emit_params_struct/
        # _emit_task_struct back from this module (see its own docstring),
        # so a module-level import here would be circular. This is the one
        # call site that needs to cross that boundary.
        from codegen.fft_cooperative_codegen import _emit_cooperative_stage

        _emit_cooperative_stage(
            e, plan=plan, stage=stage, is_first=is_first, is_last=is_last,
            compute_lanes=compute_lanes,
        )
        return False

    def emit_prelude() -> None:
        e.add("        var local_id = local_uthread_id()")
        e.add(f"        if local_id >= MAX_UTHREAD_{plan.kernel_name}:")
        e.add("            return")
        # A single-stage kernel (this kernel's own length == its one radix,
        # as every decomposed sub-FFT kernel is) never reads or writes its
        # own scratchpad -- first_stage and last_stage are the same stage,
        # so every load/store is DRAM-side. Only declare spad_base where
        # something uses it.
        if plan.scratchpad_buffers:
            e.add(f"        var spad_base = local_id * {plan.scratchpad_uthread_stride}")
        # Each kernel's own AddressMapping settles this in one runtime
        # multiply (plus, only where the mapping isn't the origin, one add)
        # -- except SPLIT, a middle kernel's output in an M>=3 chain, which
        # costs one % and one // (see _mapping_base_expr). Declared only on
        # the stage that actually reads/writes DRAM through it.
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

    emit_prelude()

    if loop_stages:
        assert twiddle_table is not None
        loop_plan = _try_build_loop_stage(
            stage, simd_lanes=plan.simd_lanes, min_full_batches=min_loop_batches,
            twiddle_table=twiddle_table,
        )
        if loop_plan is not None:
            _emit_loop_stage(e, plan=plan, stage=stage, loop_plan=loop_plan, compute_lanes=compute_lanes)
            if loop_plan.tail_batch is not None:
                # A separate @staticmethod (its own launch_parallel call,
                # see _emit_task_struct's device_main), not more code
                # appended here after the loop: confirmed by direct
                # A/B measurement (N=960's FFTRecNear0 stage_0) that
                # appending the tail inline -- even after trimming it down
                # to just the one real chunk of work, dead padding chunks
                # already dropped -- left an unchanged 128-byte register
                # spill; moving it to its own function is the next thing
                # to try, on the theory that this target's register
                # allocator scopes per compiled function, not per block,
                # so a tail sharing stage_{id}()'s own function forces one
                # spill budget across both regardless of the tail's own
                # size.
                emit_header(f"stage_{stage.stage_id}_tail")
                emit_prelude()
                _emit_stage_batches(e, plan=plan, stage=stage, batches=(loop_plan.tail_batch,), compute_lanes=compute_lanes)
                return True
            return False

    # Each batch's rr{k}/ii{k}/or{k}/oi{k} (and friends) are local to that
    # batch's own butterfly, not threads carried across batches -- but
    # _emit_batch always names them the same way regardless of batch_id, so
    # a stage with more than one SIMD batch (or, now, more than one
    # compute-width chunk within a batch -- see _chunk_batch) needs each
    # piece in its own block scope or the next one's `var rr0` redefines
    # the previous.
    _emit_stage_batches(e, plan=plan, stage=stage, batches=stage.batches, compute_lanes=compute_lanes)
    return False


def _emit_stage_batches(
    e: Emitter, *, plan: FFTCodegenPlan, stage: FFTStagePlan,
    batches: tuple[SIMDBatchPlan, ...], compute_lanes: int | None,
) -> None:
    pieces: list[tuple[int, SIMDBatchPlan]] = []
    for batch in batches:
        pieces.extend(
            _chunk_batch(
                batch,
                simd_lanes=plan.simd_lanes,
                compute_lanes=compute_lanes if compute_lanes is not None else plan.simd_lanes,
            )
        )
    # A tail SIMD batch chunked narrower than its own valid_lanes can leave
    # a piece with nothing valid in it at all (e.g. an 8-wide batch with 4
    # valid lanes, chunked into two 4-wide pieces: the second is entirely
    # padding) -- _chunk_store already gives such a piece an empty
    # lane_offsets (see its own "nothing in that chunk is ever written"
    # comment), so it computes a full butterfly on garbage/zero SIMD
    # constants and stores none of it: pure dead code, and register
    # pressure from live values nothing ever reads. Confirmed a real
    # contributor, not just theoretical -- N=960's FFTRecNear0 stage_0
    # kept spilling after _chunk_load's own vector-load fix alone (128-byte
    # frame, unchanged) until this filter dropped the piece entirely.
    pieces = [(width, piece) for width, piece in pieces if piece.valid_lanes > 0]

    for width, piece in pieces:
        if len(pieces) > 1:
            sub = Emitter()
            _emit_batch(sub, plan=plan, stage=stage, batch=piece, width=width)
            e.add(f"        if True:  # batch {piece.batch_id} scope")
            for line in sub.lines:
                e.add("    " + line if line else "")
        else:
            _emit_batch(e, plan=plan, stage=stage, batch=piece, width=width)


def _emit_params_struct(
    e: Emitter, *, plan: FFTCodegenPlan, add_loop_twiddle: bool = False
) -> None:
    e.add("@fieldwise_init")
    e.add(f"struct {plan.kernel_name}Params(Movable):")
    e.add("    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    if plan.large_twiddle is not None:
        e.add("    var large_twiddle_real_base: UnsafePointer[Float32, MutAnyOrigin]")
        e.add("    var large_twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    if add_loop_twiddle:
        # Shared per-kernel table pooling every looped stage's twiddle rows
        # (see the runtime-loop stage rendering note above _try_build_loop_stage)
        # -- always declared when the caller opts this kernel into loop_stages,
        # even if no stage in it actually ends up looping, so the field doesn't
        # depend on that per-stage outcome (see emit_kernel).
        e.add("    var loop_twiddle_real_base: UnsafePointer[Float32, MutAnyOrigin]")
        e.add("    var loop_twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add()
    e.add()


def _emit_task_struct(
    e: Emitter,
    *,
    plan: FFTCodegenPlan,
    compute_lanes: int | None = None,
    loop_stages: bool = False,
    min_loop_batches: int = _LOOP_MIN_FULL_BATCHES,
) -> list[tuple[float, float]]:
    """The NDPTask struct: its scratchpad buffers (each uthread's own
    region, sized by scratchpad_uthread_stride -- see module docstring),
    its per-stage kernels, and device_main. Identical in shape whether this
    plan is a whole single-kernel FFT or one half of a decomposed one.

    `compute_lanes`: see `_chunk_batch` -- the SIMD width the emitted
    arithmetic instructions use, independent of `plan.simd_lanes` (the
    launch granule). `None` (the default) keeps every stage emitted at
    `plan.simd_lanes`, unchanged from before this parameter existed.

    `loop_stages`: see the runtime-loop stage rendering note above
    _try_build_loop_stage. `False` (the default) keeps every stage's
    per-batch code fully unrolled, unchanged from before this parameter
    existed. Returns the flat (real, imag) twiddle table every looped stage
    of this kernel pooled its constants into -- empty when `loop_stages` is
    `False` or no stage in this kernel qualified to loop. The caller (see
    fft_transpose_codegen.generate_recursive_fft_kernels) owns turning this
    into an actual DRAM buffer and passing it through this kernel's launch
    Params, since that's a host/main()-level decision this module doesn't
    make on its own.
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

    twiddle_table: list[tuple[float, float]] = []
    tail_stage_ids: set[int] = set()
    for stage in plan.stages:
        has_tail = _emit_stage(
            e, plan=plan, stage=stage, compute_lanes=compute_lanes,
            loop_stages=loop_stages, min_loop_batches=min_loop_batches,
            twiddle_table=twiddle_table,
        )
        if has_tail:
            tail_stage_ids.add(stage.stage_id)

    e.add("    @staticmethod")
    e.add("    def device_main():")
    for stage in plan.stages:
        e.add(f"        launch_parallel[{plan.kernel_name}.stage_{stage.stage_id}]()")
        if stage.stage_id in tail_stage_ids:
            e.add(f"        launch_parallel[{plan.kernel_name}.stage_{stage.stage_id}_tail]()")
    e.add()
    e.add()
    return twiddle_table


def emit_kernel(
    e: Emitter,
    *,
    plan: FFTCodegenPlan,
    compute_lanes: int | None = None,
    loop_stages: bool = False,
    min_loop_batches: int = _LOOP_MIN_FULL_BATCHES,
) -> list[tuple[float, float]]:
    e.add(f"comptime MAX_UTHREAD_{plan.kernel_name} = {plan.max_uthread}")
    e.add()
    _emit_params_struct(e, plan=plan, add_loop_twiddle=loop_stages)
    return _emit_task_struct(
        e, plan=plan, compute_lanes=compute_lanes,
        loop_stages=loop_stages, min_loop_batches=min_loop_batches,
    )


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

    emit_kernel(e, plan=plan.kernel0, compute_lanes=compute_lanes)
    emit_kernel(e, plan=plan.kernel1, compute_lanes=compute_lanes)

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


def generate_multi_kernel_fft_kernels(
    plan: MultiKernelFFTPlan, *, compute_lanes: int | None = None
) -> str:
    """Render an M-kernel chained plan (see fft_plan_multikernel.make_multi_kernel_plan):
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
        emit_kernel(e, plan=kernel, compute_lanes=compute_lanes)

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
