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
    #   W_twiddle_modulus^((b // twiddle_lane_divisor) * k * twiddle_stride)
    # Set twiddle_modulus to None when this stage does not need an extra
    # twiddle. `twiddle_lane_divisor` only matters when a stage's SIMD lane
    # packs more than one logical index together (see `store_layout` below);
    # its default of 1 means "the whole lane index is the twiddle index",
    # i.e. no grouping.
    twiddle_modulus: int | None = None
    twiddle_stride: int = 1
    twiddle_lane_divisor: int = 1

    # How this stage's radix outputs land when they are not the final DRAM
    # write (i.e. this is not the last stage):
    #   "linear"          -- one contiguous W-wide vector store per output k,
    #                        at batch_base + k * output_stride. The default;
    #                        used when the next stage reads the same layout
    #                        this stage's own SIMD lanes are already in.
    #   "stage0_b_k2"     -- store output k, lane b at scratchpad[b, k], i.e.
    #                        index = global_bfly * output_batch_width + k.
    #                        Transposes so the next stage can FFT across the
    #                        current SIMD-lane dimension.
    #   "stage1_a_k1_k2"  -- for a middle stage whose lane packs two indices
    #                        together via `twiddle_lane_divisor` (call them
    #                        a = lane // twiddle_lane_divisor and
    #                        k2 = lane % twiddle_lane_divisor): store output
    #                        k1, lane (a, k2) at
    #                        index = a * (radix * twiddle_lane_divisor)
    #                              + k1 * twiddle_lane_divisor + k2,
    #                        keeping k2 contiguous under each (a, k1) so a
    #                        later stage can read a whole SIMD vector's worth
    #                        of k2 at once.
    store_layout: str = "linear"


@dataclass(frozen=True)
class FFTSpec:
    length: int
    inverse: bool
    uthread_granularity: int
    max_uthread: int
    simd_lanes: int = 8   # M2NDP 256-bit vector / FP32 = 8 lanes
    # precision: str = "fp32"
    # layout: str = "planar"

    # A stage's outputs are stored the moment each is computed (see
    # `emit_radix_outputs`), which reuses the very scratchpad region its
    # inputs were just read from. That's safe *within* one SIMD batch --
    # every input for the batch is loaded, into registers, before the first
    # store -- but not proven safe across batches or stages in general: a
    # later SIMD batch (in this stage or the next one) can still need to
    # read scratchpad data that an earlier batch's store already overwrote,
    # depending on how a plan's strides/widths line up. Whether that can
    # happen is a property of the specific plan, not something this
    # generator checks -- so it's the plan author's call, not a default:
    #   False (default) -- one scratchpad region, half the footprint.
    #                       Only safe when the plan guarantees no in-stage
    #                       or cross-stage read outlives an overwrite.
    #   True             -- two regions, ping-ponged one stage boundary at a
    #                       time (`_scratchpad_buf`): a stage's write
    #                       destination is always the region its own read
    #                       source did *not* come from, so a same-stage or
    #                       next-stage read can never observe a store that
    #                       should not have happened yet. Costs 2x the
    #                       scratchpad footprint.
    use_pingpong: bool = False


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
        if stage.twiddle_lane_divisor <= 0:
            raise ValueError(f"stage {sid}: twiddle_lane_divisor must be positive")
        if plan.spec.simd_lanes % stage.twiddle_lane_divisor != 0:
            raise ValueError(
                f"stage {sid}: twiddle_lane_divisor must divide simd_lanes"
            )
        if stage.store_layout not in ("linear", "stage0_b_k2", "stage1_a_k1_k2"):
            raise ValueError(f"stage {sid}: unknown store_layout {stage.store_layout!r}")
        radix_product *= stage.radix

    # For this first fused-FFT generator, each stage is one factor of N.
    if radix_product != plan.spec.length:
        raise ValueError(
            f"product of radices ({radix_product}) must equal FFT length "
            f"({plan.spec.length})"
        )


def _scratchpad_buf(plan: FFTCodegenPlan, stage_id: int) -> str:
    """Which scratchpad region stage `stage_id` writes to (the next stage
    that reads from it asks for the same `stage_id`, via `stage_id - 1`, to
    get the matching name).

    See `FFTSpec.use_pingpong` for the tradeoff this switches between.
    """
    if not plan.spec.use_pingpong:
        return "FFTFP32.buf"
    return "FFTFP32.buf_a" if stage_id % 2 == 0 else "FFTFP32.buf_b"


def emit_radix_outputs(
    e: Emitter,
    *,
    indent: str,
    plan: FFTCodegenPlan,
    stage: FFTStage,
    stage_id: int,
    simd_it: int,
    first: bool,
    last: bool,
    output_batch_base: int,
) -> None:
    """Emit one radix-R DFT, computing and storing outputs one at a time.

    rr0..rr(R-1), ii0..ii(R-1) (SIMD vectors, one independent radix-R FFT per
    lane) must all be loaded before any output can be computed -- each output
    is a sum over every input, so those R*2 registers are unavoidably live
    for the whole loop below. But every output does *not* need every other
    output alive: `or_k`/`oi_k` are twiddled, scaled and stored immediately
    after they're computed, halving the peak vector-register count a stage
    needs versus computing all R outputs before storing any of them.

    Storing this early is safe with respect to `rr`/`ii` themselves --
    they're register-resident values from the moment they're loaded,
    decoupled from the memory they came from, and every load for this batch
    happens before the first store here. It is *not* proven safe with
    respect to some other, later SIMD batch (in this stage or the next one)
    that might still need to read this data before it's overwritten -- see
    `FFTSpec.use_pingpong`, which is what `write_buf` below switches on.

    This is intentionally expanded as a small DFT so the code-generation path
    is easy to inspect. After the pipeline works, this function is the place
    to substitute hand-optimized radix-2/4/8 butterflies.
    """
    radix = stage.radix
    inverse = plan.spec.inverse
    sign = 1.0 if inverse else -1.0
    lanes = plan.spec.simd_lanes
    first_bfly = simd_it * lanes
    write_buf = _scratchpad_buf(plan, stage_id)

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

        if stage.twiddle_modulus is not None:
            wr_vals: list[str] = []
            wi_vals: list[str] = []
            for lane in range(lanes):
                bfly = first_bfly + lane
                twiddle_index = bfly // stage.twiddle_lane_divisor
                exponent = twiddle_index * k * stage.twiddle_stride
                angle = sign * 2.0 * pi * exponent / stage.twiddle_modulus
                wr_vals.append(_f32(cos(angle)))
                wi_vals.append(_f32(sin(angle)))

            e.add(
                f"{indent}var twr{k} = SIMD[DType.float32, W]("
                + ", ".join(wr_vals)
                + ")"
            )
            e.add(
                f"{indent}var twi{k} = SIMD[DType.float32, W]("
                + ", ".join(wi_vals)
                + ")"
            )
            e.add(f"{indent}var tr{k} = or{k} * twr{k} - oi{k} * twi{k}")
            e.add(f"{indent}var ti{k} = or{k} * twi{k} + oi{k} * twr{k}")
            e.add(f"{indent}or{k} = tr{k}")
            e.add(f"{indent}oi{k} = ti{k}")

        # Inverse normalization is fused only into the final write.
        if last and inverse:
            scale = _f32(1.0 / plan.spec.length)
            e.add(f"{indent}or{k} *= {scale}")
            e.add(f"{indent}oi{k} *= {scale}")

        if last:
            # Final layout for the example plan is natural order: output k
            # is one W-wide contiguous vector.
            idx = output_batch_base + k * stage.output_stride
            e.add(f"{indent}p.output_real_base.store(batch_base + {idx}, or{k})")
            e.add(f"{indent}p.output_imag_base.store(batch_base + {idx}, oi{k})")
        elif stage.store_layout == "stage0_b_k2":
            # Transpose while writing to scratchpad. No extra DRAM pass.
            # output k, lane b -> scratchpad[b][k].
            for lane in range(lanes):
                global_bfly = simd_it * lanes + lane
                idx = global_bfly * stage.output_batch_width + k
                e.add(f"{indent}{write_buf}.store(spad_base + {idx}, or{k}[{lane}])")
                e.add(
                    f"{indent}{write_buf}.store(spad_base + N + {idx}, oi{k}[{lane}])"
                )
        elif stage.store_layout == "stage1_a_k1_k2":
            # This stage's SIMD lane packs two logical indices together (see
            # `FFTStage.store_layout`): a = lane // twiddle_lane_divisor and
            # k2 = lane % twiddle_lane_divisor. Store output k1=k, lane
            # (a, k2) so k2 stays contiguous under each (a, k1) -- what the
            # next stage's plain vector load expects to find.
            divisor = stage.twiddle_lane_divisor
            group_stride = radix * divisor
            for lane in range(lanes):
                global_bfly = simd_it * lanes + lane
                a = global_bfly // divisor
                k2 = global_bfly % divisor
                idx = a * group_stride + k * divisor + k2
                e.add(f"{indent}{write_buf}.store(spad_base + {idx}, or{k}[{lane}])")
                e.add(
                    f"{indent}{write_buf}.store(spad_base + N + {idx}, oi{k}[{lane}])"
                )
        else:
            idx = output_batch_base + k * stage.output_stride
            e.add(f"{indent}{write_buf}.store(spad_base + {idx}, or{k})")
            e.add(f"{indent}{write_buf}.store(spad_base + N + {idx}, oi{k})")
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

    # One µthread computes one whole length-N FFT (SIMD still does RADIX of
    # it in parallel within a stage). `local_id` is dense per unit, so it
    # picks this µthread's private slice of the per-unit scratchpad --
    # MAX_UTHREAD slices were reserved for exactly this. `batch` is dense
    # across the whole launch, so it picks this µthread's slice of the
    # batched DRAM input/output instead: two different units can both hand
    # out local id 0, but never the same global id.
    e.add("        var local_id = local_uthread_id()")
    e.add("        if local_id >= MAX_UTHREAD:")
    e.add("            return")
    e.add("        var spad_base = local_id * (2 * N)")
    e.add("        var batch_base = global_uthread_id() * N")
    e.add()

    read_buf = _scratchpad_buf(plan, stage_id - 1)

    for simd_it in range(stage.simd_iteration_count):
        input_batch_base = simd_it * stage.input_batch_width
        output_batch_base = simd_it * stage.output_batch_width

        e.add(f"        # ===== stage {stage_id}, SIMD batch {simd_it} =====")

        # Natural-order DRAM does NOT force a scalar strided load here.
        # Example N=64, radix=8, W=8:
        # rr0 = x[0:8], rr1 = x[8:16], ... rr7 = x[56:64].
        # Each SIMD lane is one of the 8 independent strided radix-8 FFTs.
        # Every output below needs every one of these, so -- unlike the
        # outputs -- there is no way to shrink how many stay live at once.
        for j in range(stage.radix):
            idx = input_batch_base + j * stage.input_stride
            if first:
                e.add(
                    f"        var rr{j} = p.input_real_base.load[width=W](batch_base + {idx})"
                )
                e.add(
                    f"        var ii{j} = p.input_imag_base.load[width=W](batch_base + {idx})"
                )
            else:
                # `read_buf` is a Scratchpad, not a raw pointer: its `load`
                # takes the element dtype and width by name (`dt`, `w`),
                # unlike `UnsafePointer.load`'s `width=`.
                e.add(
                    f"        var rr{j} = {read_buf}.load[DType.float32, W](spad_base + {idx})"
                )
                e.add(
                    f"        var ii{j} = {read_buf}.load[DType.float32, W](spad_base + N + {idx})"
                )
        e.add()

        emit_radix_outputs(
            e,
            indent="        ",
            plan=plan,
            stage=stage,
            stage_id=stage_id,
            simd_it=simd_it,
            first=first,
            last=last,
            output_batch_base=output_batch_base,
        )


def generate_fft_kernel(plan: FFTCodegenPlan) -> str:
    validate_plan(plan)
    e = Emitter()

    # Based on the VectorAdd/Arachne NDPTask structure.
    e.add("from std.sys import size_of")
    e.add()
    e.add(
        "from m2ndp import VECTOR_WIDTH, NDPTask, PooledRange, "
        "global_uthread_id, local_uthread_id, launch_parallel, scratchpad"
    )
    e.add("from m2ndp_host import cxl_alloc")
    e.add()
    e.add("comptime W = VECTOR_WIDTH // size_of[Float32]()")
    e.add(f"comptime N = {plan.spec.length}")
    e.add(f"comptime MAX_UTHREAD = {plan.spec.max_uthread}")
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
    # Every region below is one `2 * N`-element slice per µthread the unit
    # can hold, so each keeps its own scratchpad slot at
    # `local_uthread_id() * (2 * N)` -- see `emit_stage`'s `spad_base`.
    # See `FFTSpec.use_pingpong` for what the choice of one region vs two
    # trades off.
    if plan.spec.use_pingpong:
        e.add(
            '    comptime buf_a = scratchpad[2 * N * MAX_UTHREAD, Float32, name="fft_buffer_a"]()'
        )
        e.add(
            '    comptime buf_b = scratchpad[2 * N * MAX_UTHREAD, Float32, name="fft_buffer_b"]()'
        )
    else:
        e.add(
            '    comptime buf = scratchpad[2 * N * MAX_UTHREAD, Float32, name="fft_buffer"]()'
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

    # One global µthread computes one complete length-N FFT.  Therefore the
    # host-side DRAM buffers must contain one N-element slice per spawned
    # µthread.  MAX_UTHREAD is also the batch count in this minimal launcher.
    e.add("    var total_elems = N * MAX_UTHREAD")
    e.add("    var input_real = cxl_alloc[Float32](total_elems)")
    e.add("    var input_imag = cxl_alloc[Float32](total_elems)")
    e.add("    var output_real = cxl_alloc[Float32](total_elems)")
    e.add("    var output_imag = cxl_alloc[Float32](total_elems)")
    e.add("    var ref_real = cxl_alloc[Float32](total_elems)")
    e.add("    var ref_imag = cxl_alloc[Float32](total_elems)")
    e.add()

    # The µthread pool is only used to create MAX_UTHREAD logical µthreads.
    # It is deliberately separate from the FFT data arrays: global_uthread_id()
    # selects the actual N-element batch through batch_base in emit_stage().
    e.add("    var pool_elems = W * MAX_UTHREAD")
    e.add("    var uthread_pool = cxl_alloc[Float32](pool_elems)")
    e.add()

    # ------------------------------------------------------------
    # Test input / reference
    # ------------------------------------------------------------
    e.add("    for i in range(total_elems):")
    e.add("        input_real[i] = Float32(0)")
    e.add("        input_imag[i] = Float32(0)")
    e.add("        output_real[i] = Float32(0)")
    e.add("        output_imag[i] = Float32(0)")
    e.add("        ref_real[i] = Float32(0)")
    e.add("        ref_imag[i] = Float32(0)")
    e.add()

    # Give each batch a different impulse location so a broken batch_base or
    # overlapping output slices cannot accidentally pass verification.
    impulse_amp = float(plan.spec.length) if plan.spec.inverse else 1.0
    sign = 1.0 if plan.spec.inverse else -1.0

    for batch in range(plan.spec.max_uthread):
        impulse_index = 0 if plan.spec.length == 1 else (batch + 1) % plan.spec.length
        batch_base = batch * plan.spec.length
        e.add(f"    # batch {batch}")
        e.add(
            f"    input_real[{batch_base + impulse_index}] = {_f32(impulse_amp)}"
        )

        for k in range(plan.spec.length):
            angle = (
                sign
                * 2.0
                * pi
                * impulse_index
                * k
                / plan.spec.length
            )
            e.add(
                f"    ref_real[{batch_base + k}] = {_f32(cos(angle))}"
            )
            e.add(
                f"    ref_imag[{batch_base + k}] = {_f32(sin(angle))}"
            )
        e.add()

    # ------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------
    e.add("    var rc = FFTFP32.launch(")
    e.add("        PooledRange.over(uthread_pool, pool_elems),")
    e.add(
        "        FFTFP32Params("
        "input_real, input_imag, output_real, output_imag"
        "),"
    )
    e.add("    )")
    e.add()

    e.add("    if rc != 0:")
    e.add('        print("[host] FFT failed, exit", rc)')
    e.add("        return")
    e.add()

    # ------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------
    e.add("    var tol = Float32(1.0e-3)")
    e.add("    for i in range(total_elems):")
    e.add("        var err_r = output_real[i] - ref_real[i]")
    e.add("        var err_i = output_imag[i] - ref_imag[i]")
    e.add()

    # abs()를 import할 필요 없도록 직접 absolute value 계산
    e.add("        if err_r < Float32(0):")
    e.add("            err_r = -err_r")
    e.add("        if err_i < Float32(0):")
    e.add("            err_i = -err_i")
    e.add()

    e.add("        if err_r > tol or err_i > tol:")
    e.add('            print("[host] FFT mismatch at", i)')
    e.add(
        '            print("  expected:", '
        "ref_real[i], ref_imag[i])"
    )
    e.add(
        '            print("  actual:  ", '
        "output_real[i], output_imag[i])"
    )
    e.add('            print("  error:   ", err_r, err_i)')
    e.add("            return")
    e.add()

    e.add('    print("[host] FFT verification passed")')

    return e.text()


def make_444_plan() -> FFTCodegenPlan:
    spec = FFTSpec(
        length=64,
        inverse=False,
        uthread_granularity=64,
        max_uthread=1,
        simd_lanes=8,
    )

    # n = b + 16*n2
    #
    # Stage 0:
    #   FFT over n2.
    #   B[b,k2] = FFT4(x[b + 16*n2]) * W64^(b*k2)
    #   Store PING[b,k2] at b*4+k2.
    stage0 = FFTStage(
        radix=4,
        butterfly_count=16,
        simd_iteration_count=2,
        input_batch_width=8,
        input_stride=16,
        output_batch_width=4,
        output_stride=4,
        twiddle_modulus=64,
        twiddle_stride=1,
        twiddle_lane_divisor=1,
        store_layout="stage0_b_k2",
    )

    # b = a + 4*c
    #
    # Stage 1:
    #   FFT over c.
    #   C[a,k1,k2] = FFT4(B[a+4*c,k2]) * W16^(a*k1)
    #
    # SIMD iteration 0 reads:
    #   [0:8], [16:24], [32:40], [48:56]
    # then writes logical PONG addresses [0:32].
    #
    # If PONG is replaced with PING, that [0:32] write destroys
    # addresses [8:16] and [24:32] that iteration 1 still needs.
    stage1 = FFTStage(
        radix=4,
        butterfly_count=16,
        simd_iteration_count=2,
        input_batch_width=8,
        input_stride=16,
        output_batch_width=8,
        output_stride=16,
        twiddle_modulus=16,
        twiddle_stride=1,
        twiddle_lane_divisor=4,
        store_layout="stage1_a_k1_k2",
    )

    # Stage 2:
    #   FFT over a.
    #   q = k1*4+k2 is contiguous.
    #   output index = q + 16*k0 = k2 + 4*k1 + 16*k0 (natural order).
    stage2 = FFTStage(
        radix=4,
        butterfly_count=16,
        simd_iteration_count=2,
        input_batch_width=8,
        input_stride=16,
        output_batch_width=8,
        output_stride=16,
        twiddle_modulus=None,
        store_layout="linear",
    )

    return FFTCodegenPlan(spec=spec, stages=(stage0, stage1, stage2))


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
        store_layout="stage0_b_k2",
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
    )

    plan = FFTCodegenPlan(
        spec=spec,
        stages=(stage0, stage1),
    )

    source = generate_fft_kernel(plan)
    output_path = Path(__file__).resolve().parent / "fft_fp32_generated.mojo"
    output_path.write_text(source, encoding="utf-8")

    print(f"generated: {output_path}")