from __future__ import annotations

"""Pure code emission from a fully lowered FFTCodegenPlan.

No FFT planning is performed here.  In particular this module does not:

* choose radices or stage order,
* calculate SIMD iteration counts or tail lanes,
* choose DRAM/scratchpad sources,
* choose ping-pong banks,
* calculate twiddle exponents/constants,
* calculate load/store indices,
* choose store layouts,
* build host reference FFT values.

All of those decisions are already materialized in fft_plangen.FFTCodegenPlan.
"""

from fft_butterflies import emit_butterfly
from fft_plangen import (
    FFTCodegenPlan,
    FFTStagePlan,
    LoadPlan,
    OutputPlan,
    SIMDBatchPlan,
    StorePlan,
    TwiddlePlan,
)


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


def _spad(name: str) -> str:
    return f"FFTFP32.{name}"


def _emit_load(e: Emitter, *, plan: FFTCodegenPlan, load: LoadPlan) -> None:
    j = load.operand

    if load.mode == "vector":
        assert load.base_offset is not None
        if load.source == "input":
            e.add(
                f"        var rr{j} = p.input_real_base.load[width=W]("
                f"batch_base + {load.base_offset})"
            )
            e.add(
                f"        var ii{j} = p.input_imag_base.load[width=W]("
                f"batch_base + {load.base_offset})"
            )
        else:
            assert load.buffer_name is not None
            buf = _spad(load.buffer_name)
            e.add(
                f"        var rr{j} = {buf}.load[DType.float32, W]("
                f"spad_base + {load.base_offset})"
            )
            e.add(
                f"        var ii{j} = {buf}.load[DType.float32, W]("
                f"spad_base + N + {load.base_offset})"
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
                f"batch_base + {offset})"
            )
            e.add(
                f"        var {ii_scalar} = p.input_imag_base.load[width=1]("
                f"batch_base + {offset})"
            )
        else:
            assert load.buffer_name is not None
            buf = _spad(load.buffer_name)
            e.add(
                f"        var {rr_scalar} = {buf}.load[DType.float32, 1]("
                f"spad_base + {offset})"
            )
            e.add(
                f"        var {ii_scalar} = {buf}.load[DType.float32, 1]("
                f"spad_base + N + {offset})"
            )
        rr_lanes.append(f"{rr_scalar}[0]")
        ii_lanes.append(f"{ii_scalar}[0]")
    e.add(
        f"        var rr{j} = SIMD[DType.float32, W](" + ", ".join(rr_lanes) + ")"
    )
    e.add(
        f"        var ii{j} = SIMD[DType.float32, W](" + ", ".join(ii_lanes) + ")"
    )


def _emit_twiddle(e: Emitter, *, output: int, twiddle: TwiddlePlan) -> None:
    e.add(
        f"        var twr{output} = SIMD[DType.float32, W]("
        + ", ".join(_f32(v) for v in twiddle.real)
        + ")"
    )
    e.add(
        f"        var twi{output} = SIMD[DType.float32, W]("
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


def _emit_store(e: Emitter, *, output: int, store: StorePlan) -> None:
    if store.mode == "vector":
        assert store.base_offset is not None
        if store.destination == "output":
            e.add(
                f"        p.output_real_base.store(batch_base + {store.base_offset}, "
                f"or{output})"
            )
            e.add(
                f"        p.output_imag_base.store(batch_base + {store.base_offset}, "
                f"oi{output})"
            )
        else:
            assert store.buffer_name is not None
            buf = _spad(store.buffer_name)
            e.add(
                f"        {buf}.store(spad_base + {store.base_offset}, or{output})"
            )
            e.add(
                f"        {buf}.store(spad_base + N + {store.base_offset}, oi{output})"
            )
        return

    for lane, offset in enumerate(store.lane_offsets):
        if store.destination == "output":
            e.add(
                f"        p.output_real_base.store(batch_base + {offset}, "
                f"or{output}[{lane}])"
            )
            e.add(
                f"        p.output_imag_base.store(batch_base + {offset}, "
                f"oi{output}[{lane}])"
            )
        else:
            assert store.buffer_name is not None
            buf = _spad(store.buffer_name)
            e.add(
                f"        {buf}.store(spad_base + {offset}, or{output}[{lane}])"
            )
            e.add(
                f"        {buf}.store(spad_base + N + {offset}, oi{output}[{lane}])"
            )


def _emit_output(e: Emitter, *, output_plan: OutputPlan) -> None:
    k = output_plan.output
    if output_plan.twiddle is not None:
        _emit_twiddle(e, output=k, twiddle=output_plan.twiddle)

    if output_plan.scale is not None:
        scale = _f32(output_plan.scale)
        e.add(f"        or{k} *= {scale}")
        e.add(f"        oi{k} *= {scale}")

    _emit_store(e, output=k, store=output_plan.store)
    e.add()


def _emit_batch(
    e: Emitter,
    *,
    plan: FFTCodegenPlan,
    stage: FFTStagePlan,
    batch: SIMDBatchPlan,
) -> None:
    e.add(
        f"        # ===== stage {stage.stage_id}, SIMD batch {batch.batch_id} "
        f"(valid lanes: {batch.valid_lanes}/{plan.simd_lanes}) ====="
    )

    for load in batch.loads:
        _emit_load(e, plan=plan, load=load)
    e.add()

    emit_butterfly(
        e,
        indent="        ",
        radix=stage.radix,
        inverse=stage.inverse,
    )
    e.add()

    for output_plan in batch.outputs:
        _emit_output(e, output_plan=output_plan)


def _emit_stage(e: Emitter, *, plan: FFTCodegenPlan, stage: FFTStagePlan) -> None:
    e.add("    @staticmethod")
    e.add(f"    def stage_{stage.stage_id}():")
    e.add("        ref p = FFTFP32.params[]")
    e.add(f"        comptime RADIX = {stage.radix}")
    e.add(f"        comptime SIMD_ITERS = {stage.simd_iteration_count}")
    e.add()
    e.add("        var local_id = local_uthread_id()")
    e.add("        if local_id >= MAX_UTHREAD:")
    e.add("            return")
    e.add(f"        var spad_base = local_id * {plan.scratchpad_uthread_stride}")
    e.add(f"        var batch_base = global_uthread_id() * {plan.batch_stride}")
    e.add()

    for batch in stage.batches:
        _emit_batch(e, plan=plan, stage=stage, batch=batch)


def generate_fft_kernel(plan: FFTCodegenPlan) -> str:
    """Render the supplied plan.  This function performs no FFT planning."""

    e = Emitter()

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
    e.add(f"comptime N = {plan.length}")
    e.add(f"comptime MAX_UTHREAD = {plan.max_uthread}")
    e.add()

    e.add("@fieldwise_init")
    e.add("struct FFTFP32Params(Movable):")
    e.add("    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add()
    e.add()

    e.add("struct FFTFP32(NDPTask):")
    e.add("    comptime Params = FFTFP32Params")
    e.add()

    for buffer in plan.scratchpad_buffers:
        e.add(
            f'    comptime {buffer.name} = scratchpad[{buffer.elements}, Float32, '
            f'name="fft_{buffer.name}"]()'
        )
    if plan.scratchpad_buffers:
        e.add()

    for stage in plan.stages:
        _emit_stage(e, plan=plan, stage=stage)

    e.add("    @staticmethod")
    e.add("    def device_main():")
    for stage in plan.stages:
        e.add(f"        launch_parallel[FFTFP32.stage_{stage.stage_id}]()")
    e.add()
    e.add()

    host = plan.host
    e.add("def main() raises:")
    e.add("    if FFTFP32.emit_ir_if_asked():")
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
    # benchmark's main() -- see e.g. vector_add.mojo/vector_exp.mojo), not a
    # signal fixed once at generation time.
    e.add("    seed(0)")
    e.add("    for i in range(total_elems):")
    e.add("        input_real[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        input_imag[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        output_real[i] = Float32(0)")
    e.add("        output_imag[i] = Float32(0)")
    e.add("        ref_real[i] = Float32(0)")
    e.add("        ref_imag[i] = Float32(0)")
    e.add()

    e.add("    var rc = FFTFP32.launch(")
    e.add("        PooledRange.over(uthread_pool, pool_elems),")
    e.add(
        "        FFTFP32Params(input_real, input_imag, output_real, output_imag),"
    )
    e.add("    )")
    e.add()
    e.add("    if rc != 0:")
    e.add('        print("[host] FFT failed, exit", rc)')
    e.add("        return")
    e.add()

    # Independent reference: a direct O(N^2) DFT, not the radix decomposition
    # the kernel runs, computed here at host runtime against whatever input
    # was just randomly generated above -- accumulated in Float64 so this
    # check doesn't share the kernel's own fp32 rounding, and only rounds to
    # Float32 once, at the very end.
    e.add("    var pi = Float64(3.141592653589793)")
    e.add(f"    var sign = Float64({1.0 if plan.inverse else -1.0})")
    e.add("    for batch in range(MAX_UTHREAD):")
    e.add("        var batch_base = batch * N")
    e.add("        for k in range(N):")
    e.add("            var acc_r = Float64(0)")
    e.add("            var acc_i = Float64(0)")
    e.add("            for n in range(N):")
    e.add(
        "                var angle = sign * 2.0 * pi * Float64(n) * Float64(k) / Float64(N)"
    )
    e.add("                var c = host_cos(angle)")
    e.add("                var s = host_sin(angle)")
    e.add("                var xr = Float64(input_real[batch_base + n])")
    e.add("                var xi = Float64(input_imag[batch_base + n])")
    e.add("                acc_r += xr * c - xi * s")
    e.add("                acc_i += xr * s + xi * c")
    if plan.inverse:
        e.add("            acc_r /= Float64(N)")
        e.add("            acc_i /= Float64(N)")
    e.add("            ref_real[batch_base + k] = Float32(acc_r)")
    e.add("            ref_imag[batch_base + k] = Float32(acc_i)")
    e.add()

    e.add(f"    var tol = {_f32(host.tolerance)}")
    e.add("    for i in range(total_elems):")
    e.add("        var err_r = output_real[i] - ref_real[i]")
    e.add("        var err_i = output_imag[i] - ref_imag[i]")
    e.add("        if err_r < Float32(0):")
    e.add("            err_r = -err_r")
    e.add("        if err_i < Float32(0):")
    e.add("            err_i = -err_i")
    e.add("        if err_r > tol or err_i > tol:")
    e.add('            print("[host] FFT mismatch at", i)')
    e.add('            print("  expected:", ref_real[i], ref_imag[i])')
    e.add('            print("  actual:  ", output_real[i], output_imag[i])')
    e.add('            print("  error:   ", err_r, err_i)')
    e.add("            return")
    e.add()
    e.add('    print("[host] FFT verification passed")')

    return e.text()