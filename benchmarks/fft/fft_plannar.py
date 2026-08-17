from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from pathlib import Path


@dataclass(frozen=True)
class FFTStage:
    radix: int

    # Number of independent radix-R butterflies in this stage.
    butterfly_count: int
    # butterfly_count / SIMD lanes, rounded up by the planner.
    simd_iteration_count: int

    # For one SIMD batch, radix operand j begins at
    #   batch_base + j * input_stride
    input_batch_width: int
    input_stride: int

    # For a normal vector store, radix output k begins at
    #   batch_base + k * output_stride
    output_batch_width: int
    output_stride: int

    # Optional Cooley-Tukey twiddle applied after the local radix FFT.
    # For SIMD lane b and radix output k:
    #   W_twiddle_modulus^(b * k * twiddle_stride)
    # Set to None when this stage does not need an extra twiddle.
    twiddle_modulus: int | None = None
    twiddle_stride: int = 1

    # If True, store vector outputs into scratchpad transposed:
    #   vector output k, lane b -> scratchpad[b, k]
    # This is useful when the next radix stage must FFT across the current
    # SIMD-lane dimension.
    transpose_output: bool = False


@dataclass(frozen=True)
class FFTSpec:
    length: int
    inverse: bool
    uthread_granularity: int
    max_uthread: int
    simd_lanes: int = 8   # M2NDP 256-bit vector / FP32 = 8 lanes
    # precision: str = "fp32"
    # layout: str = "planar"


@dataclass(frozen=True)
class FFTCodegenPlan:
    spec: FFTSpec
    stages: tuple[FFTStage, ...]


# Keep the original misspelled name usable while migrating code.
FFTCodegnePlan = FFTCodegenPlan


class Emitter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _f32(value: float) -> str:
    """Emit stable Float32 literals into the generated Mojo source."""
    if abs(value) < 1.0e-12:
        value = 0.0
    elif abs(value - 1.0) < 1.0e-12:
        value = 1.0
    elif abs(value + 1.0) < 1.0e-12:
        value = -1.0
    return f"Float32({value:.9g})"


def validate_plan(plan: FFTCodegenPlan) -> None:
    if plan.spec.length <= 0:
        raise ValueError("FFT length must be positive")
    if not plan.stages:
        raise ValueError("at least one FFT stage is required")
    if plan.spec.max_uthread <= 0:
        raise ValueError("max_uthread must be positive")
    if plan.spec.simd_lanes <= 0:
        raise ValueError("simd_lanes must be positive")

    radix_product = 1
    for sid, stage in enumerate(plan.stages):
        if stage.radix not in (2, 4, 8):
            raise ValueError(f"stage {sid}: only radix 2/4/8 is supported here")
        if stage.butterfly_count <= 0:
            raise ValueError(f"stage {sid}: butterfly_count must be positive")
        if stage.simd_iteration_count <= 0:
            raise ValueError(f"stage {sid}: simd_iteration_count must be positive")
        if stage.input_stride <= 0 or stage.output_stride <= 0:
            raise ValueError(f"stage {sid}: stride must be positive")
        if stage.twiddle_modulus is not None and stage.twiddle_modulus <= 0:
            raise ValueError(f"stage {sid}: twiddle_modulus must be positive")
        radix_product *= stage.radix

    # For this first fused-FFT generator, each stage is one factor of N.
    if radix_product != plan.spec.length:
        raise ValueError(
            f"product of radices ({radix_product}) must equal FFT length "
            f"({plan.spec.length})"
        )


def emit_generic_radix_dft(
    e: Emitter,
    *,
    indent: str,
    radix: int,
    inverse: bool,
) -> None:
    """Emit a correctness-first radix-R FFT on SIMD vectors.

    rr0..rr(R-1), ii0..ii(R-1) are SIMD vectors.  Therefore each SIMD lane
    computes one independent radix-R FFT.  This is intentionally expanded as
    a small DFT so the code-generation path is easy to inspect.  After the
    pipeline works, this function is the place to substitute hand-optimized
    radix-2/4/8 butterflies.
    """
    sign = 1.0 if inverse else -1.0

    for k in range(radix):
        e.add(f"{indent}# radix-{radix} output {k}")
        e.add(f"{indent}var or{k} = SIMD[DType.float32, W](0)")
        e.add(f"{indent}var oi{k} = SIMD[DType.float32, W](0)")

        for j in range(radix):
            angle = sign * 2.0 * pi * j * k / radix
            wr = cos(angle)
            wi = sin(angle)

            if abs(wi) < 1.0e-12 and abs(wr - 1.0) < 1.0e-12:
                e.add(f"{indent}or{k} += rr{j}")
                e.add(f"{indent}oi{k} += ii{j}")
            elif abs(wi) < 1.0e-12 and abs(wr + 1.0) < 1.0e-12:
                e.add(f"{indent}or{k} -= rr{j}")
                e.add(f"{indent}oi{k} -= ii{j}")
            else:
                e.add(
                    f"{indent}or{k} += rr{j} * {_f32(wr)} - ii{j} * {_f32(wi)}"
                )
                e.add(
                    f"{indent}oi{k} += rr{j} * {_f32(wi)} + ii{j} * {_f32(wr)}"
                )
        e.add()


def emit_twiddle_vectors(
    e: Emitter,
    *,
    indent: str,
    plan: FFTCodegenPlan,
    stage: FFTStage,
    simd_it: int,
) -> None:
    """Apply stage-specific twiddles to or0..or(R-1), oi0..oi(R-1)."""
    if stage.twiddle_modulus is None:
        return

    sign = 1.0 if plan.spec.inverse else -1.0
    lanes = plan.spec.simd_lanes
    first_bfly = simd_it * lanes

    e.add(f"{indent}# Cooley-Tukey twiddle multiplication")
    for k in range(stage.radix):
        wr_vals: list[str] = []
        wi_vals: list[str] = []
        for lane in range(lanes):
            bfly = first_bfly + lane
            exponent = bfly * k * stage.twiddle_stride
            angle = sign * 2.0 * pi * exponent / stage.twiddle_modulus
            wr_vals.append(_f32(cos(angle)))
            wi_vals.append(_f32(sin(angle)))

        e.add(
            f"{indent}var twr{k} = SIMD[DType.float32, W](" + ", ".join(wr_vals) + ")"
        )
        e.add(
            f"{indent}var twi{k} = SIMD[DType.float32, W](" + ", ".join(wi_vals) + ")"
        )
        e.add(f"{indent}var tr{k} = or{k} * twr{k} - oi{k} * twi{k}")
        e.add(f"{indent}var ti{k} = or{k} * twi{k} + oi{k} * twr{k}")
        e.add(f"{indent}or{k} = tr{k}")
        e.add(f"{indent}oi{k} = ti{k}")
    e.add()


def emit_stage(e: Emitter, plan: FFTCodegenPlan, stage_id: int) -> None:
    stage = plan.stages[stage_id]
    first = stage_id == 0
    last = stage_id == len(plan.stages) - 1

    e.add("    @staticmethod")
    e.add(f"    def stage_{stage_id}():")
    e.add("        ref p = FFTFP32.params[]")
    e.add(f"        comptime RADIX = {stage.radix}")
    e.add(f"        comptime SIMD_ITERS = {stage.simd_iteration_count}")
    e.add(f"        comptime INPUT_STRIDE = {stage.input_stride}")
    e.add(f"        comptime OUTPUT_STRIDE = {stage.output_stride}")
    e.add()

    # This first implementation deliberately maps the whole fused FFT to one
    # logical µthread.  SIMD still computes multiple small FFTs in parallel.
    e.add("        var uid = global_uthread_id()")
    e.add("        if uid != 0:")
    e.add("            return")
    e.add()

    for simd_it in range(stage.simd_iteration_count):
        input_batch_base = simd_it * stage.input_batch_width
        output_batch_base = simd_it * stage.output_batch_width

        e.add(f"        # ===== stage {stage_id}, SIMD batch {simd_it} =====")

        # Natural-order DRAM does NOT force a scalar strided load here.
        # Example N=64, radix=8, W=8:
        # rr0 = x[0:8], rr1 = x[8:16], ... rr7 = x[56:64].
        # Each SIMD lane is one of the 8 independent strided radix-8 FFTs.
        for j in range(stage.radix):
            idx = input_batch_base + j * stage.input_stride
            if first:
                e.add(
                    f"        var rr{j} = p.input_real_base.load[width=W]({idx})"
                )
                e.add(
                    f"        var ii{j} = p.input_imag_base.load[width=W]({idx})"
                )
            else:
                # `FFTFP32.buf` is a Scratchpad, not a raw pointer: its
                # `load` takes the element dtype and width by name (`dt`,
                # `w`), unlike `UnsafePointer.load`'s `width=`.
                e.add(
                    f"        var rr{j} = FFTFP32.buf.load[DType.float32, W]({idx})"
                )
                e.add(
                    f"        var ii{j} = FFTFP32.buf.load[DType.float32, W](N + {idx})"
                )
        e.add()

        emit_generic_radix_dft(
            e,
            indent="        ",
            radix=stage.radix,
            inverse=plan.spec.inverse,
        )

        emit_twiddle_vectors(
            e,
            indent="        ",
            plan=plan,
            stage=stage,
            simd_it=simd_it,
        )

        # Inverse normalization is fused only into the final write.
        if last and plan.spec.inverse:
            scale = _f32(1.0 / plan.spec.length)
            for k in range(stage.radix):
                e.add(f"        or{k} *= {scale}")
                e.add(f"        oi{k} *= {scale}")
            e.add()

        if last:
            # Final layout for the example plan is natural order:
            # output k is one W-wide contiguous vector.
            for k in range(stage.radix):
                idx = output_batch_base + k * stage.output_stride
                e.add(f"        p.output_real_base.store({idx}, or{k})")
                e.add(f"        p.output_imag_base.store({idx}, oi{k})")
            e.add()
        elif stage.transpose_output:
            # Transpose while writing to scratchpad.  No extra DRAM pass.
            # output k / lane b -> scratchpad[b][k]
            lanes = plan.spec.simd_lanes
            e.add("        # transpose into scratchpad for the next FFT stage")
            for lane in range(lanes):
                global_bfly = simd_it * lanes + lane
                for k in range(stage.radix):
                    idx = global_bfly * stage.output_batch_width + k
                    e.add(f"        FFTFP32.buf.store({idx}, or{k}[{lane}])")
                    e.add(f"        FFTFP32.buf.store(N + {idx}, oi{k}[{lane}])")
            e.add()
        else:
            for k in range(stage.radix):
                idx = output_batch_base + k * stage.output_stride
                e.add(f"        FFTFP32.buf.store({idx}, or{k})")
                e.add(f"        FFTFP32.buf.store(N + {idx}, oi{k})")
            e.add()


def generate_fft_kernel(plan: FFTCodegenPlan) -> str:
    validate_plan(plan)
    e = Emitter()

    # Based on the VectorAdd/Arachne NDPTask structure.
    e.add("from std.sys import size_of")
    e.add()
    e.add(
        "from m2ndp import VECTOR_WIDTH, NDPTask, PooledRange, "
        "global_uthread_id, launch_parallel, scratchpad"
    )
    e.add("from m2ndp_host import cxl_alloc")
    e.add()
    e.add("comptime W = VECTOR_WIDTH // size_of[Float32]()")
    e.add(f"comptime N = {plan.spec.length}")
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
    e.add(
        f'    comptime buf = scratchpad[2 * {plan.spec.length}, Float32, name="fft_buffer"]()'
    )
    e.add()

    for stage_id in range(len(plan.stages)):
        emit_stage(e, plan, stage_id)

    # Arachne uses device_main() to express inter-kernel control flow.
    e.add("    @staticmethod")
    e.add("    def device_main():")
    for stage_id in range(len(plan.stages)):
        e.add(f"        launch_parallel[FFTFP32.stage_{stage_id}]()")
    e.add()
    e.add()

    # Minimal host launcher matching vector_add.mojo.
    e.add("def main() raises:")
    e.add("    if FFTFP32.emit_ir_if_asked():")
    e.add("        return")
    e.add()
    e.add("    var input_real = cxl_alloc[Float32](N)")
    e.add("    var input_imag = cxl_alloc[Float32](N)")
    e.add("    var output_real = cxl_alloc[Float32](N)")
    e.add("    var output_imag = cxl_alloc[Float32](N)")
    e.add()
    e.add("    # TODO: host-side input initialization / reference check")
    e.add(f"    var pool_elems = W * {plan.spec.max_uthread}")
    e.add("    var rc = FFTFP32.launch(")
    e.add("        PooledRange.over(input_real, pool_elems),")
    e.add("        FFTFP32Params(input_real, input_imag, output_real, output_imag),")
    e.add("    )")
    e.add("    if rc != 0:")
    e.add('        print("[host] FFT failed, exit", rc)')
    e.add("        return")
    e.add('    print("[host] FFT finished")')

    return e.text()


if __name__ == "__main__":
    # Complete 64-point FFT as 8 x 8 Cooley-Tukey.
    #
    # Stage 0:
    #   - DRAM remains natural order.
    #   - rr0=x[0:8], rr1=x[8:16], ... are contiguous vector loads.
    #   - SIMD lane b therefore computes FFT of x[b], x[b+8], ..., x[b+56].
    #   - Apply W64^(b*k2), then transpose to scratchpad.
    #
    # Stage 1:
    #   - Scratchpad rows are contiguous across k2.
    #   - Another radix-8 FFT produces natural-order output vectors.
    spec = FFTSpec(
        length=64,
        inverse=False,
        uthread_granularity=64,
        max_uthread=1,
        simd_lanes=8,
    )

    stage0 = FFTStage(
        radix=8,
        butterfly_count=8,
        simd_iteration_count=1,
        input_batch_width=8,
        input_stride=8,
        output_batch_width=8,
        output_stride=8,
        twiddle_modulus=64,
        twiddle_stride=1,
        transpose_output=True,
    )

    stage1 = FFTStage(
        radix=8,
        butterfly_count=8,
        simd_iteration_count=1,
        input_batch_width=8,
        input_stride=8,
        output_batch_width=8,
        output_stride=8,
        twiddle_modulus=None,
        transpose_output=False,
    )

    plan = FFTCodegenPlan(
        spec=spec,
        stages=(stage0, stage1),
    )

    source = generate_fft_kernel(plan)
    output_path = Path(__file__).resolve().parent / "fft_fp32_generated.mojo"
    output_path.write_text(source, encoding="utf-8")

    print(f"generated: {output_path}")