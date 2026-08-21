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
    _f32,
)
from fft_plangen import (
    BalancedTransposeFFTPlan,
    FFTLeafPlan,
    FFTNode,
    FFTRecursiveNodePlan,
    FFTTransposePlan,
    PhysicalTransposePlan,
    RecursiveFFTPlan,
)


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


# --------------------------------------------------- generic tiled transpose
#
# Generalizes the ki_near x ki_far tile above to a physical tile size
# (tile_rows x tile_cols) chosen completely independently of any FFT
# chunk/radix length -- see fft_plangen.PhysicalTransposePlan and the
# module design writeup for this session. Used by PRE/MIDDLE/POST
# transposes in a make_recursive_transpose_plan tree; the ki_near x
# ki_far-tiled emitter above is untouched (regression baseline for
# make_balanced_transpose_plan).


def _emit_physical_transpose_params_struct(e: Emitter, *, plan: PhysicalTransposePlan) -> None:
    e.add("@fieldwise_init")
    e.add(f"struct {plan.kernel_name}Params(Movable):")
    e.add("    var src_real_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var src_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var dst_real_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var dst_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    if plan.twiddle_modulus is not None:
        e.add("    var twiddle_real_base: UnsafePointer[Float32, MutAnyOrigin]")
        e.add("    var twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add()
    e.add()


def _emit_tile_transfer(
    e: Emitter, *, plan: PhysicalTransposePlan, valid_rows: int, valid_cols: int
) -> None:
    """One valid_rows x valid_cols tile's worth of: contiguous vector
    loads from the source (+ matching twiddle rows, same addressing, if
    this is a MIDDLE transpose), elementwise twiddle, scalar-per-lane
    scratchpad store at the transposed offset, contiguous vector read-back,
    contiguous vector stores into the destination. Uses already-bound
    runtime vars `q`, `t_r`, `t_c`, `spad_base`. valid_rows/valid_cols are
    plan-time constants (the caller picks which constants for the fast
    path vs. a tail branch -- see _emit_physical_transpose_stage), so
    every load/store width here is a fixed integer, never a runtime value.
    """
    rows, cols = plan.rows, plan.cols
    tile_rows, tile_cols = plan.tile_rows, plan.tile_cols
    tile_elems = tile_rows * tile_cols
    indent = "        "

    for i in range(valid_rows):
        e.add(f"{indent}var src_row_{i} = t_r * {tile_rows} + {i}")
        e.add(
            f"{indent}var src_base_{i} = q * {rows * cols} + src_row_{i} * {cols} + "
            f"t_c * {tile_cols}"
        )
        e.add(f"{indent}var rr{i} = p.src_real_base.load[width={valid_cols}](src_base_{i})")
        e.add(f"{indent}var ii{i} = p.src_imag_base.load[width={valid_cols}](src_base_{i})")
        if plan.twiddle_modulus is not None:
            e.add(
                f"{indent}var tw_base_{i} = src_row_{i} * {cols} + t_c * {tile_cols}"
            )
            e.add(
                f"{indent}var twr{i} = p.twiddle_real_base.load[width={valid_cols}](tw_base_{i})"
            )
            e.add(
                f"{indent}var twi{i} = p.twiddle_imag_base.load[width={valid_cols}](tw_base_{i})"
            )
            e.add(f"{indent}var tr{i} = rr{i} * twr{i} - ii{i} * twi{i}")
            e.add(f"{indent}var ti{i} = rr{i} * twi{i} + ii{i} * twr{i}")
            rr_name, ii_name = f"tr{i}", f"ti{i}"
        else:
            rr_name, ii_name = f"rr{i}", f"ii{i}"
        tile_buf = _spad(plan.kernel_name, "tile_buf")
        for j in range(valid_cols):
            spad_off = f"spad_base + {j * tile_rows + i}"
            spad_off_i = f"spad_base + {tile_elems + j * tile_rows + i}"
            e.add(f"{indent}{tile_buf}.store({spad_off}, {rr_name}[{j}])")
            e.add(f"{indent}{tile_buf}.store({spad_off_i}, {ii_name}[{j}])")
        e.add()

    scale = None
    if plan.apply_inverse_scale:
        scale = 1.0 / (rows * cols)

    tile_buf = _spad(plan.kernel_name, "tile_buf")
    for j in range(valid_cols):
        e.add(f"{indent}var dst_row_{j} = t_c * {tile_cols} + {j}")
        e.add(
            f"{indent}var dst_base_{j} = q * {rows * cols} + dst_row_{j} * {rows} + "
            f"t_r * {tile_rows}"
        )
        spad_row = f"spad_base + {j * tile_rows}"
        spad_row_i = f"spad_base + {tile_elems + j * tile_rows}"
        e.add(f"{indent}var or{j} = {tile_buf}.load[DType.float32, {valid_rows}]({spad_row})")
        e.add(f"{indent}var oi{j} = {tile_buf}.load[DType.float32, {valid_rows}]({spad_row_i})")
        if scale is not None:
            e.add(f"{indent}or{j} = or{j} * {_f32(scale)}")
            e.add(f"{indent}oi{j} = oi{j} * {_f32(scale)}")
        e.add(f"{indent}p.dst_real_base.store(dst_base_{j}, or{j})")
        e.add(f"{indent}p.dst_imag_base.store(dst_base_{j}, oi{j})")
    e.add()


def _emit_physical_transpose_stage(e: Emitter, *, plan: PhysicalTransposePlan) -> None:
    rows, cols, tile_rows, tile_cols = plan.rows, plan.cols, plan.tile_rows, plan.tile_cols
    grid_rows, grid_cols = plan.grid_rows, plan.grid_cols
    has_row_tail = rows % tile_rows != 0
    has_col_tail = cols % tile_cols != 0
    tail_valid_rows = rows - (grid_rows - 1) * tile_rows
    tail_valid_cols = cols - (grid_cols - 1) * tile_cols

    e.add("    @staticmethod")
    e.add("    def stage_0():")
    e.add(f"        ref p = {plan.kernel_name}.params[]")
    e.add("        var local_id = local_uthread_id()")
    e.add(f"        if local_id >= MAX_UTHREAD_{plan.kernel_name}:")
    e.add("            return")
    e.add(f"        var spad_base = local_id * {plan.scratchpad_elements}")
    e.add()
    e.add("        var tile_id = global_uthread_id()")
    e.add(f"        var tiles_per_replica = {grid_rows * grid_cols}")
    e.add("        var q = tile_id // tiles_per_replica")
    e.add("        var local_tile = tile_id % tiles_per_replica")
    e.add(f"        var t_r = local_tile // {grid_cols}")
    e.add(f"        var t_c = local_tile % {grid_cols}")
    e.add()

    if not has_row_tail and not has_col_tail:
        _emit_tile_transfer(e, plan=plan, valid_rows=tile_rows, valid_cols=tile_cols)
        return

    row_tail_cond = f"t_r == {grid_rows - 1}"
    col_tail_cond = f"t_c == {grid_cols - 1}"
    if has_row_tail and has_col_tail:
        e.add(f"        if {row_tail_cond} and {col_tail_cond}:")
        sub = Emitter()
        _emit_tile_transfer(sub, plan=plan, valid_rows=tail_valid_rows, valid_cols=tail_valid_cols)
        for line in sub.lines:
            e.add("    " + line if line else "")
        e.add(f"        elif {row_tail_cond}:")
        sub = Emitter()
        _emit_tile_transfer(sub, plan=plan, valid_rows=tail_valid_rows, valid_cols=tile_cols)
        for line in sub.lines:
            e.add("    " + line if line else "")
        e.add(f"        elif {col_tail_cond}:")
        sub = Emitter()
        _emit_tile_transfer(sub, plan=plan, valid_rows=tile_rows, valid_cols=tail_valid_cols)
        for line in sub.lines:
            e.add("    " + line if line else "")
        e.add("        else:")
        sub = Emitter()
        _emit_tile_transfer(sub, plan=plan, valid_rows=tile_rows, valid_cols=tile_cols)
        for line in sub.lines:
            e.add("    " + line if line else "")
    elif has_row_tail:
        e.add(f"        if {row_tail_cond}:")
        sub = Emitter()
        _emit_tile_transfer(sub, plan=plan, valid_rows=tail_valid_rows, valid_cols=tile_cols)
        for line in sub.lines:
            e.add("    " + line if line else "")
        e.add("        else:")
        sub = Emitter()
        _emit_tile_transfer(sub, plan=plan, valid_rows=tile_rows, valid_cols=tile_cols)
        for line in sub.lines:
            e.add("    " + line if line else "")
    else:  # has_col_tail only
        e.add(f"        if {col_tail_cond}:")
        sub = Emitter()
        _emit_tile_transfer(sub, plan=plan, valid_rows=tile_rows, valid_cols=tail_valid_cols)
        for line in sub.lines:
            e.add("    " + line if line else "")
        e.add("        else:")
        sub = Emitter()
        _emit_tile_transfer(sub, plan=plan, valid_rows=tile_rows, valid_cols=tile_cols)
        for line in sub.lines:
            e.add("    " + line if line else "")


def _emit_physical_transpose_task_struct(e: Emitter, *, plan: PhysicalTransposePlan) -> None:
    e.add(f"struct {plan.kernel_name}(NDPTask):")
    e.add(f"    comptime Params = {plan.kernel_name}Params")
    e.add()
    e.add(
        f'    comptime tile_buf = scratchpad[{plan.scratchpad_elements * plan.max_uthread}, '
        f'Float32, name="{plan.kernel_name.lower()}_tile"]()'
    )
    e.add()
    _emit_physical_transpose_stage(e, plan=plan)
    e.add("    @staticmethod")
    e.add("    def device_main():")
    e.add(f"        launch_parallel[{plan.kernel_name}.stage_0]()")
    e.add()
    e.add()


def _emit_physical_transpose_kernel(e: Emitter, *, plan: PhysicalTransposePlan) -> None:
    e.add(f"comptime MAX_UTHREAD_{plan.kernel_name} = {plan.max_uthread}")
    e.add()
    _emit_physical_transpose_params_struct(e, plan=plan)
    _emit_physical_transpose_task_struct(e, plan=plan)


def _emit_physical_transpose_twiddle_table_precompute(
    e: Emitter, *, plan: PhysicalTransposePlan, real_name: str, imag_name: str
) -> None:
    """Host precompute for a MIDDLE transpose's own dense W_M twiddle
    table -- table[r*cols+c] = W_M^(r*c), size rows*cols == M (this node's
    own current modulus, never the top-level N -- see FFTRecursiveNodePlan),
    no per-replica duplication (see PhysicalTransposePlan's own docstring)."""
    assert plan.twiddle_modulus is not None
    sign = 1.0 if plan.inverse else -1.0
    e.add(f"    var tw_pi = Float64(3.141592653589793)")
    e.add(f"    var tw_sign = Float64({sign})")
    e.add(f"    var tw_r = 0")
    e.add(f"    while tw_r < {plan.rows}:")
    e.add(f"        var tw_c = 0")
    e.add(f"        while tw_c < {plan.cols}:")
    e.add(
        f"            var tw_angle = tw_sign * 2.0 * tw_pi * Float64(tw_r) * "
        f"Float64(tw_c) / Float64({plan.twiddle_modulus})"
    )
    e.add(f"            var tw_addr = tw_r * {plan.cols} + tw_c")
    e.add(f"            {real_name}[tw_addr] = Float32(host_cos(tw_angle))")
    e.add(f"            {imag_name}[tw_addr] = Float32(host_sin(tw_angle))")
    e.add(f"            tw_c += 1")
    e.add(f"        tw_r += 1")
    e.add()


def flatten_recursive_node(node: FFTNode) -> list:
    """Flat, ordered stage list for a recursive FFTNode: leaf -> [kernel];
    node -> [pre, near_fft.kernel, middle] + flatten(far_child) + [post].
    Each element is either an FFTCodegenPlan (an ordinary FFT kernel) or a
    PhysicalTransposePlan (a standalone transpose kernel) -- codegen
    dispatches on type, never re-decides anything (see fft_plangen.py's
    own "planner decides everything" discipline)."""
    if isinstance(node, FFTLeafPlan):
        return [node.kernel]
    assert isinstance(node, FFTRecursiveNodePlan)
    return (
        [node.pre_transpose, node.near_fft.kernel, node.middle_transpose]
        + flatten_recursive_node(node.far_child)
        + [node.post_transpose]
    )


def generate_recursive_fft_kernels(plan: RecursiveFFTPlan) -> str:
    """Render a full make_recursive_transpose_plan tree as a flat, ordered
    Mojo-ish kernel sequence chained through DRAM from one host main().
    Uses two alternating "work" DRAM buffers (see the module design
    writeup's own section on this) rather than one fresh N-sized buffer
    per stage, since a deep recursion can have many stages. This function
    performs no FFT/transpose planning -- every FFTCodegenPlan/
    PhysicalTransposePlan field is already decided in `plan`.
    """
    stages = flatten_recursive_node(plan.root)
    e = Emitter()
    _emit_prelude(e)
    e.add(f"comptime N = {plan.n}")
    e.add()

    twiddle_names: dict[int, tuple[str, str]] = {}
    for i, stage in enumerate(stages):
        if isinstance(stage, PhysicalTransposePlan):
            _emit_physical_transpose_kernel(e, plan=stage)
        else:
            _emit_kernel(e, plan=stage)

    host = plan.host
    m = len(stages)

    def buf_name(idx: int, part: str) -> str:
        if idx == -1:
            return f"input_{part}"
        if idx == m:
            return f"output_{part}"
        # alternate between two DRAM work buffers
        return f"work{idx % 2}_{part}"

    e.add("def main() raises:")
    first_kernel_name = stages[0].kernel_name
    e.add(f"    if {first_kernel_name}.emit_ir_if_asked():")
    e.add("        return")
    e.add()
    e.add(f"    var n = {host.n}")
    e.add("    var input_real = cxl_alloc[Float32](n)")
    e.add("    var input_imag = cxl_alloc[Float32](n)")
    e.add("    var work0_real = cxl_alloc[Float32](n)")
    e.add("    var work0_imag = cxl_alloc[Float32](n)")
    e.add("    var work1_real = cxl_alloc[Float32](n)")
    e.add("    var work1_imag = cxl_alloc[Float32](n)")
    e.add("    var output_real = cxl_alloc[Float32](n)")
    e.add("    var output_imag = cxl_alloc[Float32](n)")
    e.add("    var ref_real = cxl_alloc[Float32](n)")
    e.add("    var ref_imag = cxl_alloc[Float32](n)")
    e.add()

    for i, stage in enumerate(stages):
        if isinstance(stage, PhysicalTransposePlan) and stage.twiddle_modulus is not None:
            rn, imn = f"twiddle{i}_real", f"twiddle{i}_imag"
            twiddle_names[i] = (rn, imn)
            table_size = stage.rows * stage.cols
            e.add(f"    var {rn} = cxl_alloc[Float32]({table_size})")
            e.add(f"    var {imn} = cxl_alloc[Float32]({table_size})")
            _emit_physical_transpose_twiddle_table_precompute(e, plan=stage, real_name=rn, imag_name=imn)

    for i, stage in enumerate(stages):
        e.add(f"    var pool{i}_elems = {stage.simd_lanes * stage.total_uthreads}")
        e.add(f"    var pool{i} = cxl_alloc[Float32](pool{i}_elems)")
    e.add()

    e.add("    seed(0)")
    e.add("    for i in range(n):")
    e.add("        input_real[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        input_imag[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        work0_real[i] = Float32(0)")
    e.add("        work0_imag[i] = Float32(0)")
    e.add("        work1_real[i] = Float32(0)")
    e.add("        work1_imag[i] = Float32(0)")
    e.add("        output_real[i] = Float32(0)")
    e.add("        output_imag[i] = Float32(0)")
    e.add("        ref_real[i] = Float32(0)")
    e.add("        ref_imag[i] = Float32(0)")
    e.add()

    for i, stage in enumerate(stages):
        in_r, in_i = buf_name(i - 1, "real"), buf_name(i - 1, "imag")
        out_r, out_i = buf_name(i, "real"), buf_name(i, "imag")
        e.add(f"    var rc{i} = {stage.kernel_name}.launch(")
        e.add(f"        PooledRange.over(pool{i}, pool{i}_elems),")
        if isinstance(stage, PhysicalTransposePlan) and stage.twiddle_modulus is not None:
            rn, imn = twiddle_names[i]
            e.add(f"        {stage.kernel_name}Params({in_r}, {in_i}, {out_r}, {out_i}, {rn}, {imn}),")
        elif isinstance(stage, PhysicalTransposePlan):
            e.add(f"        {stage.kernel_name}Params({in_r}, {in_i}, {out_r}, {out_i}),")
        else:
            # every FFTCodegenPlan a recursive node builds (leaf or
            # near_fft) is always large_twiddle=None -- that cross-block
            # math lives entirely in the standalone MIDDLE transpose kernel.
            assert stage.large_twiddle is None
            e.add(f"        {stage.kernel_name}Params({in_r}, {in_i}, {out_r}, {out_i}),")
        e.add("    )")
        e.add(f"    if rc{i} != 0:")
        e.add(f'        print("[host] recursive FFT stage {i} ({stage.kernel_name}) failed, exit", rc{i})')
        e.add("        return")
        e.add()

    _emit_reference_check(
        e, n=plan.n, batch_count=1, inverse=plan.inverse,
        input_real="input_real", input_imag="input_imag",
        output_real="output_real", output_imag="output_imag",
        ref_real="ref_real", ref_imag="ref_imag",
        tolerance=host.tolerance, label="recursive tiled-transpose FFT",
    )
    return e.text()
