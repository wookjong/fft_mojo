from __future__ import annotations

"""Codegen for the standalone tiled-transpose + fused large-twiddle
boundary kernel (see fft_plangen.FFTTransposePlan / make_balanced_transpose_
plan). Separate from fft_codegen.py's FFT-stage codegen on purpose: an
FFTTransposePlan is not an FFTStagePlan chain (no radix butterflies, no
ping-pong, one stage, a different DRAM access shape) -- fusing its logic
into fft_codegen.py's stage emitter would mix two genuinely different
responsibilities. This module performs no planning decisions of its own;
every shape (ki_near, ki_far, digit_multiplier_near, divisor_far, tile
count) is already decided in the FFTTransposePlan it's given.

Math (see FFTTransposePlan's own docstring for the full derivation): one
microthread owns one whole (ki_near x ki_far) tile, identified by
`tile_id = global_uthread_id()`, decomposed as `prefix = tile_id %
digit_multiplier_near`, `rest = tile_id // digit_multiplier_near`.  For
`elem_far` in [0, ki_far): the near side's own row `row_near = n_b*prefix +
elem_far*divisor_far + rest` holds `ki_near` contiguous elements (this
side's own plain output -- see AddressMapping/_build_batched_side's
`plain_boundary`); each is multiplied by the matching twiddle value (a
table addressed identically to the near side's own layout, so the fetch is
exactly as contiguous as the data fetch) and written into one scratchpad
tile buffer at the *transposed* offset `elem_near*ki_far + elem_far` (a
scalar-per-lane store -- the same idiom fft_codegen._make_store's
intermediate-stage branch already uses for an ordinary Stockham
permutation, reused here for this tile's own local transpose). Once every
row has been read, `ki_near` contiguous vector reads (one per `elem_near`,
width `ki_far`) come back out of the scratchpad and go straight to the far
side's own row `row_far = prefix + digit_multiplier_near*elem_near +
n_a*rest`, `ki_far` contiguous elements per row -- that side's own plain
input. digit_multiplier_near/divisor_far are chosen (in fft_plangen.py) so
this tiling always divides n_a/n_b exactly: there is never a ragged/tail
tile, so no masked-tail code path is needed here.

Known simplification (see the design writeup's "remaining limitations"):
every vector load/store here uses `width=ki_near` or `width=ki_far`
directly, not the kernel's own hardware SIMD lane count -- correct (still
elem_stride=1, never a scalar gather) but not necessarily hardware-vector-
width-aligned when ki_near/ki_far don't happen to match VECTOR_WIDTH.
"""

from fft_codegen import (
    Emitter,
    _emit_kernel,
    _emit_large_twiddle_table_precompute,
    _emit_prelude,
    _emit_reference_check,
)
from fft_plangen import BalancedTransposeFFTPlan, FFTTransposePlan


def _spad(kernel_name: str, name: str) -> str:
    return f"{kernel_name}.{name}"


def _emit_transpose_params_struct(e: Emitter, *, plan: FFTTransposePlan) -> None:
    e.add("@fieldwise_init")
    e.add(f"struct {plan.kernel_name}Params(Movable):")
    e.add("    var near_real_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var near_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var far_real_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var far_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var twiddle_real_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add()
    e.add()


def _emit_transpose_stage(e: Emitter, *, plan: FFTTransposePlan) -> None:
    ki_near = plan.ki_near
    ki_far = plan.ki_far
    dm = plan.digit_multiplier_near
    n_a = plan.n_a
    n_b = plan.n_b
    tile_elems = ki_near * ki_far

    e.add("    @staticmethod")
    e.add("    def stage_0():")
    e.add(f"        ref p = {plan.kernel_name}.params[]")
    e.add("        var local_id = local_uthread_id()")
    e.add(f"        if local_id >= MAX_UTHREAD_{plan.kernel_name}:")
    e.add("            return")
    e.add(f"        var spad_base = local_id * {plan.scratchpad_elements}")
    e.add()
    e.add("        var tile_id = global_uthread_id()")
    e.add(f"        var prefix = tile_id % {dm}")
    e.add(f"        var rest = tile_id // {dm}")
    e.add()

    # Load ki_far rows of ki_near contiguous elements each, twiddle-multiply,
    # scalar-store into the scratchpad tile at the transposed offset.
    for ef in range(ki_far):
        row_var = f"row_near{ef}"
        e.add(
            f"        var {row_var} = {n_b} * prefix + {ef} * {plan.divisor_far} + rest"
        )
        base = f"{row_var} * {ki_near}"
        e.add(
            f"        var rr{ef} = p.near_real_base.load[width={ki_near}]({base})"
        )
        e.add(
            f"        var ii{ef} = p.near_imag_base.load[width={ki_near}]({base})"
        )
        e.add(
            f"        var twr{ef} = p.twiddle_real_base.load[width={ki_near}]({base})"
        )
        e.add(
            f"        var twi{ef} = p.twiddle_imag_base.load[width={ki_near}]({base})"
        )
        e.add(f"        var tr{ef} = rr{ef} * twr{ef} - ii{ef} * twi{ef}")
        e.add(f"        var ti{ef} = rr{ef} * twi{ef} + ii{ef} * twr{ef}")
        tile_buf = _spad(plan.kernel_name, "tile_buf")
        for en in range(ki_near):
            spad_off = f"spad_base + {en * ki_far + ef}"
            spad_off_i = f"spad_base + {tile_elems + en * ki_far + ef}"
            e.add(f"        {tile_buf}.store(({spad_off}), tr{ef}[{en}])")
            e.add(f"        {tile_buf}.store(({spad_off_i}), ti{ef}[{en}])")
        e.add()

    # Read ki_near rows of ki_far contiguous elements back out of the
    # scratchpad tile, store into the far side's own plain layout.
    for en in range(ki_near):
        row_far = f"(prefix + {dm} * {en} + {n_a} * rest)"
        out_base = f"{row_far} * {ki_far}"
        spad_row = f"spad_base + {en * ki_far}"
        spad_row_i = f"spad_base + {tile_elems + en * ki_far}"
        tile_buf = _spad(plan.kernel_name, "tile_buf")
        e.add(f"        # far row elem_near={en}: row_far = {row_far}")
        e.add(
            f"        var or{en} = {tile_buf}.load[DType.float32, {ki_far}]({spad_row})"
        )
        e.add(
            f"        var oi{en} = {tile_buf}.load[DType.float32, {ki_far}]({spad_row_i})"
        )
        e.add(f"        p.far_real_base.store(({out_base}), or{en})")
        e.add(f"        p.far_imag_base.store(({out_base}), oi{en})")
        e.add()


def _emit_transpose_task_struct(e: Emitter, *, plan: FFTTransposePlan) -> None:
    e.add(f"struct {plan.kernel_name}(NDPTask):")
    e.add(f"    comptime Params = {plan.kernel_name}Params")
    e.add()
    e.add(
        f'    comptime tile_buf = scratchpad[{plan.scratchpad_elements * plan.max_uthread}, '
        f'Float32, name="{plan.kernel_name.lower()}_tile"]()'
    )
    e.add()

    _emit_transpose_stage(e, plan=plan)

    e.add("    @staticmethod")
    e.add("    def device_main():")
    e.add(f"        launch_parallel[{plan.kernel_name}.stage_0]()")
    e.add()
    e.add()


def _emit_transpose_kernel(e: Emitter, *, plan: FFTTransposePlan) -> None:
    e.add(f"comptime MAX_UTHREAD_{plan.kernel_name} = {plan.max_uthread}")
    e.add()
    _emit_transpose_params_struct(e, plan=plan)
    _emit_transpose_task_struct(e, plan=plan)


def _emit_transpose_twiddle_table_precompute(
    e: Emitter, *, plan: FFTTransposePlan, real_name: str, imag_name: str
) -> None:
    """Host precompute for the transpose's own twiddle table -- addressed
    identically to the near side's plain output layout (row_near*ki_near +
    elem_near), NOT AddressMapping.crossed's own dense (row*n_a+elem)
    layout, so the on-device fetch stays exactly as contiguous as the data
    fetch. Same underlying angle as LargeTwiddlePlan/CROSSED's own math
    (batch_id_within_near * combined_near_digit / n) -- not duplicated
    math, just a different address iteration to match this kernel's own
    DRAM access shape.
    """
    sign = 1.0 if plan.inverse else -1.0
    near_total_uthreads = plan.n // plan.ki_near
    e.add(f"    var tw_pi = Float64(3.141592653589793)")
    e.add(f"    var tw_sign = Float64({sign})")
    e.add(f"    var tw_row = 0")
    e.add(f"    while tw_row < {near_total_uthreads}:")
    e.add(f"        var tw_batch = tw_row % {plan.n_b}")
    e.add(f"        var tw_prefix_so_far = tw_row // {plan.n_b}")
    e.add(f"        var tw_elem = 0")
    e.add(f"        while tw_elem < {plan.ki_near}:")
    e.add(
        f"            var tw_combined = tw_prefix_so_far + "
        f"{plan.digit_multiplier_near} * tw_elem"
    )
    e.add(
        f"            var tw_angle = tw_sign * 2.0 * tw_pi * Float64(tw_batch) * "
        f"Float64(tw_combined) / Float64({plan.n})"
    )
    e.add(f"            var tw_addr = tw_row * {plan.ki_near} + tw_elem")
    e.add(f"            {real_name}[tw_addr] = Float32(host_cos(tw_angle))")
    e.add(f"            {imag_name}[tw_addr] = Float32(host_sin(tw_angle))")
    e.add(f"            tw_elem += 1")
    e.add(f"        tw_row += 1")
    e.add()


def generate_balanced_transpose_fft_kernels(plan: BalancedTransposeFFTPlan) -> str:
    """Render a full make_balanced_transpose_plan: near side kernels, the
    standalone transpose kernel, far side kernels -- chained through DRAM
    from one host main(), exactly the same "M kernels chained through DRAM"
    shape fft_codegen.generate_multi_kernel_fft_kernels already uses,
    generalized to one heterogeneous (FFTCodegenPlan | FFTTransposePlan)
    sequence. This function performs no FFT or transpose planning -- every
    AddressMapping/FFTTransposePlan field is already decided in `plan`.
    """
    e = Emitter()
    _emit_prelude(e)
    e.add(f"comptime N = {plan.n}")
    e.add()

    for kernel in plan.kernels_near:
        _emit_kernel(e, plan=kernel)
    _emit_transpose_kernel(e, plan=plan.transpose)
    for kernel in plan.kernels_far:
        _emit_kernel(e, plan=kernel)

    host = plan.host
    near = plan.kernels_near
    far = plan.kernels_far
    tr = plan.transpose

    def buf_name(idx: int, part: str) -> str:
        # idx: -1 original input, len(near)+len(far) final output, anything
        # else an intermediate buffer.
        total = len(near) + len(far)
        if idx == -1:
            return f"input_{part}"
        if idx == total:
            return f"output_{part}"
        return f"mid{idx}_{part}"

    e.add("def main() raises:")
    e.add(f"    if {near[0].kernel_name}.emit_ir_if_asked():")
    e.add("        return")
    e.add()
    e.add(f"    var n = {host.n}")
    e.add("    var input_real = cxl_alloc[Float32](n)")
    e.add("    var input_imag = cxl_alloc[Float32](n)")
    total_stages = len(near) + 1 + len(far)  # +1 for the transpose kernel
    for i in range(total_stages - 1):
        e.add(f"    var mid{i}_real = cxl_alloc[Float32](n)")
        e.add(f"    var mid{i}_imag = cxl_alloc[Float32](n)")
    e.add("    var output_real = cxl_alloc[Float32](n)")
    e.add("    var output_imag = cxl_alloc[Float32](n)")
    e.add("    var ref_real = cxl_alloc[Float32](n)")
    e.add("    var ref_imag = cxl_alloc[Float32](n)")
    e.add()

    for i, kernel in enumerate(near[:-1]):
        assert kernel.large_twiddle is not None
        e.add(f"    var large_twiddle{i}_real = cxl_alloc[Float32](n)")
        e.add(f"    var large_twiddle{i}_imag = cxl_alloc[Float32](n)")
        _emit_large_twiddle_table_precompute(
            e, lt=kernel.large_twiddle, real_name=f"large_twiddle{i}_real",
            imag_name=f"large_twiddle{i}_imag", suffix=str(i),
        )
    for j, kernel in enumerate(far[:-1]):
        idx = len(near) + 1 + j
        assert kernel.large_twiddle is not None
        e.add(f"    var large_twiddle{idx}_real = cxl_alloc[Float32](n)")
        e.add(f"    var large_twiddle{idx}_imag = cxl_alloc[Float32](n)")
        _emit_large_twiddle_table_precompute(
            e, lt=kernel.large_twiddle, real_name=f"large_twiddle{idx}_real",
            imag_name=f"large_twiddle{idx}_imag", suffix=str(idx),
        )

    e.add(f"    var transpose_twiddle_real = cxl_alloc[Float32](n)")
    e.add(f"    var transpose_twiddle_imag = cxl_alloc[Float32](n)")
    _emit_transpose_twiddle_table_precompute(
        e, plan=tr, real_name="transpose_twiddle_real", imag_name="transpose_twiddle_imag",
    )

    all_kernels = list(near) + [None] + list(far)  # None marks the transpose slot
    for i, kernel in enumerate(all_kernels):
        if kernel is None:
            e.add(f"    var poolT_elems = {tr.simd_lanes * tr.total_uthreads}")
            e.add(f"    var poolT = cxl_alloc[Float32](poolT_elems)")
        else:
            e.add(f"    var pool{i}_elems = {kernel.simd_lanes * kernel.total_uthreads}")
            e.add(f"    var pool{i} = cxl_alloc[Float32](pool{i}_elems)")
    e.add()

    e.add("    seed(0)")
    e.add("    for i in range(n):")
    e.add("        input_real[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        input_imag[i] = Float32(random_float64(-1.0, 1.0))")
    for i in range(total_stages - 1):
        e.add(f"        mid{i}_real[i] = Float32(0)")
        e.add(f"        mid{i}_imag[i] = Float32(0)")
    e.add("        output_real[i] = Float32(0)")
    e.add("        output_imag[i] = Float32(0)")
    e.add("        ref_real[i] = Float32(0)")
    e.add("        ref_imag[i] = Float32(0)")
    e.add()

    for i, kernel in enumerate(all_kernels):
        is_last = i == len(all_kernels) - 1
        in_r, in_i = buf_name(i - 1, "real"), buf_name(i - 1, "imag")
        out_r, out_i = buf_name(i, "real"), buf_name(i, "imag")
        if kernel is None:
            e.add(f"    var rcT = {tr.kernel_name}.launch(")
            e.add(f"        PooledRange.over(poolT, poolT_elems),")
            e.add(
                f"        {tr.kernel_name}Params({in_r}, {in_i}, {out_r}, {out_i}, "
                f"transpose_twiddle_real, transpose_twiddle_imag),"
            )
            e.add("    )")
            e.add(f"    if rcT != 0:")
            e.add(f'        print("[host] transpose kernel failed, exit", rcT)')
            e.add("        return")
            e.add()
            continue

        e.add(f"    var rc{i} = {kernel.kernel_name}.launch(")
        e.add(f"        PooledRange.over(pool{i}, pool{i}_elems),")
        if is_last:
            e.add(
                f"        {kernel.kernel_name}Params({in_r}, {in_i}, {out_r}, {out_i}),"
            )
        elif kernel.large_twiddle is not None:
            e.add(
                f"        {kernel.kernel_name}Params({in_r}, {in_i}, {out_r}, {out_i}, "
                f"large_twiddle{i}_real, large_twiddle{i}_imag),"
            )
        else:
            e.add(
                f"        {kernel.kernel_name}Params({in_r}, {in_i}, {out_r}, {out_i}),"
            )
        e.add("    )")
        e.add(f"    if rc{i} != 0:")
        e.add(f'        print("[host] FFT kernel{i} failed, exit", rc{i})')
        e.add("        return")
        e.add()

    _emit_reference_check(
        e, n=plan.n, batch_count=1, inverse=plan.inverse,
        input_real="input_real", input_imag="input_imag",
        output_real="output_real", output_imag="output_imag",
        ref_real="ref_real", ref_imag="ref_imag",
        tolerance=host.tolerance, label="balanced transpose FFT",
    )
    return e.text()
