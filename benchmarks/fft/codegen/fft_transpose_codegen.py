from __future__ import annotations

"""Codegen for the standalone tiled-transpose + fused large-twiddle
boundary kernel (see fft_plan_balanced.FFTTransposePlan / make_balanced_transpose_
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
input. digit_multiplier_near/divisor_far are chosen (in fft_plan_balanced.py) so
this tiling always divides n_a/n_b exactly: there is never a ragged/tail
tile, so no masked-tail code path is needed here.

Known simplification (see the design writeup's "remaining limitations"):
every vector load/store here uses `width=ki_near` or `width=ki_far`
directly, not the kernel's own hardware SIMD lane count -- correct (still
elem_stride=1, never a scalar gather) but not necessarily hardware-vector-
width-aligned when ki_near/ki_far don't happen to match VECTOR_WIDTH.
"""

from codegen.common import (
    Emitter,
    emit_array_dump as _emit_array_dump,
    emit_large_twiddle_table_precompute as _emit_large_twiddle_table_precompute,
    emit_prelude as _emit_prelude,
    emit_reference_check as _emit_reference_check,
    f32 as _f32,
    spad as _spad,
)
from codegen.fft_codegen import emit_kernel as _emit_kernel
from codegen.fft_persistent_codegen import emit_persistent_kernel_struct
from planning.strategies.fft_plan_balanced import BalancedTransposeFFTPlan, FFTTransposePlan
from planning.core.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile
from planning.strategies.fft_plan_recursive import (
    PhysicalTransposePlan,
    RecursiveFFTPlan,
    flatten_recursive_node,
)


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


def _emit_tiled_kernel_task_struct(
    e: Emitter, *, kernel_name: str, scratchpad_elements: int, max_uthread: int, emit_stage
) -> None:
    """Shared struct shape for both transpose plan types: one scratchpad
    tile buffer, one stage, one device_main -- the two callers
    (_emit_transpose_task_struct / _emit_physical_transpose_task_struct)
    differ only in which stage-emitter they pass."""
    e.add(f"struct {kernel_name}(NDPTask):")
    e.add(f"    comptime Params = {kernel_name}Params")
    e.add()
    e.add(
        f'    comptime tile_buf = scratchpad[{scratchpad_elements * max_uthread}, '
        f'Float32, name="{kernel_name.lower()}_tile"]()'
    )
    e.add()
    emit_stage(e)
    e.add("    @staticmethod")
    e.add("    def device_main():")
    e.add(f"        launch_parallel[{kernel_name}.stage_0]()")
    e.add()
    e.add()


def _emit_transpose_task_struct(e: Emitter, *, plan: FFTTransposePlan) -> None:
    _emit_tiled_kernel_task_struct(
        e, kernel_name=plan.kernel_name, scratchpad_elements=plan.scratchpad_elements,
        max_uthread=plan.max_uthread,
        emit_stage=lambda e: _emit_transpose_stage(e, plan=plan),
    )


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
# chunk/radix length -- see fft_plan_recursive.PhysicalTransposePlan and the
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
    if plan.needs_q_table:
        e.add("    var q_table: UnsafePointer[Int, MutAnyOrigin]")
    if plan.needs_tr_table:
        e.add("    var t_r_table: UnsafePointer[Int, MutAnyOrigin]")
    if plan.needs_round_split:
        e.add("    var round_offset: Int")
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
    if plan.needs_round_split:
        e.add("        var tile_id = global_uthread_id() + p.round_offset")
    else:
        e.add("        var tile_id = global_uthread_id()")
    tiles_per_replica = grid_rows * grid_cols
    if plan.replica_count == 1:
        # Nothing to divide -- tile_id is already this replica's own local
        # tile index (see PhysicalTransposePlan.needs_q_table).
        e.add("        var q = 0")
        e.add("        var local_tile = tile_id")
    elif not plan.needs_q_table:
        e.add(f"        var tiles_per_replica = {tiles_per_replica}")
        e.add("        var q = tile_id // tiles_per_replica")
        e.add("        var local_tile = tile_id % tiles_per_replica")
    else:
        # A plain `//`/`%` by this non-power-of-2 constant is what used to
        # be here; see PhysicalTransposePlan.needs_q_table for why that's
        # unsafe on this target. `local_tile` comes back out via one MUL +
        # SUB (both implemented) instead of a second table lookup.
        e.add(f"        var q = p.q_table[tile_id]")
        e.add(f"        var local_tile = tile_id - q * {tiles_per_replica}")

    if grid_cols == 1:
        e.add("        var t_r = local_tile")
        e.add("        var t_c = 0")
    elif not plan.needs_tr_table:
        e.add(f"        var t_r = local_tile // {grid_cols}")
        e.add(f"        var t_c = local_tile % {grid_cols}")
    else:
        e.add(f"        var t_r = p.t_r_table[local_tile]")
        e.add(f"        var t_c = local_tile - t_r * {grid_cols}")
    e.add()

    if not has_row_tail and not has_col_tail:
        _emit_tile_transfer(e, plan=plan, valid_rows=tile_rows, valid_cols=tile_cols)
        return

    # One branch per (row-tail?, col-tail?) combination that can actually
    # occur for this plan, most-specific first, ending in the fast (no
    # tail) path -- each branch is a runtime condition (None for the
    # trailing `else`) paired with the plan-time-constant (valid_rows,
    # valid_cols) that branch should use, per fft_transpose_codegen.py's
    # own indented-body idiom.
    row_tail_cond = f"t_r == {grid_rows - 1}"
    col_tail_cond = f"t_c == {grid_cols - 1}"
    branches: list[tuple[str | None, int, int]] = []
    if has_row_tail and has_col_tail:
        branches.append((f"{row_tail_cond} and {col_tail_cond}", tail_valid_rows, tail_valid_cols))
    if has_row_tail:
        branches.append((row_tail_cond, tail_valid_rows, tile_cols))
    if has_col_tail:
        branches.append((col_tail_cond, tile_rows, tail_valid_cols))
    branches.append((None, tile_rows, tile_cols))

    for i, (cond, valid_rows, valid_cols) in enumerate(branches):
        if cond is None:
            e.add("        else:")
        else:
            e.add(f"        {'if' if i == 0 else 'elif'} {cond}:")
        sub = Emitter()
        _emit_tile_transfer(sub, plan=plan, valid_rows=valid_rows, valid_cols=valid_cols)
        for line in sub.lines:
            e.add("    " + line if line else "")


def _emit_physical_transpose_task_struct(e: Emitter, *, plan: PhysicalTransposePlan) -> None:
    _emit_tiled_kernel_task_struct(
        e, kernel_name=plan.kernel_name, scratchpad_elements=plan.scratchpad_elements,
        max_uthread=plan.max_uthread,
        emit_stage=lambda e: _emit_physical_transpose_stage(e, plan=plan),
    )


def _emit_physical_transpose_kernel(e: Emitter, *, plan: PhysicalTransposePlan) -> None:
    e.add(f"comptime MAX_UTHREAD_{plan.kernel_name} = {plan.max_uthread}")
    e.add()
    _emit_physical_transpose_params_struct(e, plan=plan)
    _emit_physical_transpose_task_struct(e, plan=plan)


def _emit_physical_transpose_twiddle_table_precompute(
    e: Emitter, *, plan: PhysicalTransposePlan, real_name: str, imag_name: str, index: int
) -> None:
    """Host precompute for a MIDDLE transpose's own dense W_M twiddle
    table -- table[r*cols+c] = W_M^(r*c), size rows*cols == M (this node's
    own current modulus, never the top-level N -- see FFTRecursiveNodePlan),
    no per-replica duplication (see PhysicalTransposePlan's own docstring).

    `index`: this call's own position in the caller's stage list (unique
    per call site, same value the caller already uses for real_name/
    imag_name) -- suffixed onto every local var this emits, since they all
    live in the shared `def main():` scope alongside every other stage's
    own precompute. A 2+-level recursion has more than one MIDDLE transpose
    with a twiddle table (one per split level), so unsuffixed names here
    ("tw_pi" etc, unlike real_name/imag_name which were already unique)
    redefined on the second call -- unreachable by any single-level-
    recursion N, which is every case this had been checked against before
    (Python verification never re-executes this host-level text at all;
    only the real Mojo compiler catches a redefinition -- confirmed
    reproducing on N=262144, whose split needs two recursion levels).
    """
    assert plan.twiddle_modulus is not None
    sign = 1.0 if plan.inverse else -1.0
    pi, sg, r, c, ang, addr = (
        f"tw_pi{index}", f"tw_sign{index}", f"tw_r{index}", f"tw_c{index}",
        f"tw_angle{index}", f"tw_addr{index}",
    )
    e.add(f"    var {pi} = Float64(3.141592653589793)")
    e.add(f"    var {sg} = Float64({sign})")
    e.add(f"    var {r} = 0")
    e.add(f"    while {r} < {plan.rows}:")
    e.add(f"        var {c} = 0")
    e.add(f"        while {c} < {plan.cols}:")
    e.add(
        f"            var {ang} = {sg} * 2.0 * {pi} * Float64({r}) * "
        f"Float64({c}) / Float64({plan.twiddle_modulus})"
    )
    e.add(f"            var {addr} = {r} * {plan.cols} + {c}")
    e.add(f"            {real_name}[{addr}] = Float32(host_cos({ang}))")
    e.add(f"            {imag_name}[{addr}] = Float32(host_sin({ang}))")
    e.add(f"            {c} += 1")
    e.add(f"        {r} += 1")
    e.add()


def _safe_round_size(
    max_uthread: int, *, num_ndp_units: int, interleave_chunk_uthreads: int
) -> int:
    """How many microthreads one host-side launch round may safely request
    (in place of today's flat `max_uthread` cap) when a caller wants a
    round to actually spread across more than one physical NDP unit,
    without ever giving any single unit more microthreads than its own
    scratchpad (`max_uthread`) was sized for.

    Which physical unit a microthread lands on is address interleaving,
    not launch size: confirmed 2026-08-28 by reading the real simulator
    source (`M2NDPConfig::get_matched_unit_id`, third_party/m2ndp-detour/
    src/m2ndp_config.h) -- `unit(addr) = (addr // 256) % num_ndp_units`.
    One mapped microthread is `interleave_chunk_uthreads` apart in this
    same unit-of-256-bytes terms (`interleave_chunk_uthreads = 256 //
    32 = 8`, see TargetProfile's own comment) -- so consecutive
    microthreads land on the same unit in blocks of `interleave_chunk_
    uthreads`, rotating through all `num_ndp_units` units every
    `interleave_chunk_uthreads * num_ndp_units` (256) microthreads, then
    wrapping back to unit 0 for microthread 256, unit 1 for 264, etc.

    A round of size `R <= safe_round_size` never gives one physical unit
    more than `max_uthread` microthreads: the worst case for any single
    unit is `interleave_chunk_uthreads * ceil(R / (num_ndp_units *
    interleave_chunk_uthreads))`, which is `<= max_uthread` exactly when
    `R <= (max_uthread // interleave_chunk_uthreads) * interleave_chunk_
    uthreads * num_ndp_units` -- the formula below. Holds for a launch's
    own tail round too (not just a "full" round), since the same bound
    applies to any `R` up to this cap, not only exact multiples of
    anything.

    Below `interleave_chunk_uthreads`, a round can span at most 2
    adjacent units (never a whole extra unit's worth) -- already safe
    today under the plain `max_uthread` cap, so this returns `max_uthread`
    unchanged in that case rather than a spurious 0 from the floor
    division.

    NOTE: this formula's own correctness rests on two claims this
    project has NOT yet independently verified end to end (see
    docs/DEV-plans or the FFT multi-unit-parallelism plan this session
    produced): (1) that this count-based bound is what actually matters
    -- i.e. that a physical unit's own `local_uthread_id()` values for
    its assigned microthreads are dense, `{0, 1, ..., count-1}`, with no
    gaps a plain count bound wouldn't catch; and (2) the simulator's own
    per-launch overhead doesn't dominate at every scale this generator
    can produce, which is a real open question, not assumed answered.
    Both need a real build+run experiment before this helper is trusted
    for anything beyond the pure address-interleaving arithmetic it
    actually proves.
    """
    if max_uthread < interleave_chunk_uthreads:
        return max_uthread
    return (max_uthread // interleave_chunk_uthreads) * interleave_chunk_uthreads * num_ndp_units


def _stage_round_size(
    stage: PhysicalTransposePlan, *, spread_across_units: bool, target: TargetProfile
) -> int:
    """The number of microthreads one host-side launch round for `stage`
    may safely request in a single `.launch()` call -- `stage.max_uthread`
    (today's exact behavior) unless `spread_across_units` is on AND
    `stage` is eligible: a `PhysicalTransposePlan` (no cooperative-worker
    concept at all), or an `FFTCodegenPlan` with no cooperative workers
    (`stage.cooperation is None`) -- a cooperative leaf has its own,
    separate, not-yet-verified placement story (see fft_cooperative_
    codegen.py and the C-track of this repo's own multi-NDP-unit plan)
    and is never touched here.

    Raises if `spread_across_units` is on for a stage whose own
    `simd_lanes` isn't 8: `_safe_round_size`'s formula (and the address-
    interleaving math it's built on) assumes one microthread is exactly
    32 bytes (`simd_lanes * size_of[Float32]() == 32`) -- true at every
    call site this project ships today (`simd_lanes` defaults to 8
    everywhere), but `--simd-lanes` is a CLI override this function has
    no other way to guard against silently mis-trusting.
    """
    if not spread_across_units:
        return stage.max_uthread
    if not isinstance(stage, PhysicalTransposePlan) and (
        stage.cooperation is not None or stage.persistent is not None
    ):
        # A persistent stage's own max_uthread (== launch_uthreads, a
        # target-fixed width -- see make_persistent_leaf_plan) is not a
        # per-physical-unit scratchpad cap this formula's own widening
        # story applies to at all (there is no "spread this launch's
        # microthreads across more units" question for a persistent leaf
        # -- its own preload/stage/writeback phases already dispatch
        # every physical unit's own software group internally); a
        # cooperative leaf has its own, separate, not-yet-verified
        # placement story either way (see the comment this branch already
        # carried for cooperation).
        return stage.max_uthread
    if stage.simd_lanes * 4 != 32:
        raise ValueError(
            f"spread_across_units requires simd_lanes=8 (one microthread must be "
            f"exactly 32 bytes, the address-interleaving granule _safe_round_size "
            f"relies on) -- got simd_lanes={stage.simd_lanes} for {stage.kernel_name}"
        )
    return _safe_round_size(
        stage.max_uthread, num_ndp_units=target.num_ndp_units,
        interleave_chunk_uthreads=target.interleave_chunk_uthreads,
    )


def generate_recursive_fft_kernels(
    plan: RecursiveFFTPlan, *, compute_lanes: int | None = None,
    narrow_middle_stages: bool = False, loop_stages: bool = True,
    reference_check: bool = True, target: TargetProfile = DEFAULT_TARGET_PROFILE,
    spread_across_units: bool = False, debug_print_pool_alignment: bool = False,
    persistent_mode: str = "physical",
) -> str:
    """Render a full make_recursive_transpose_plan tree as a flat, ordered
    Mojo-ish kernel sequence chained through DRAM from one host main().
    Uses two alternating "work" DRAM buffers (see the module design
    writeup's own section on this) rather than one fresh N-sized buffer
    per stage, since a deep recursion can have many stages. This function
    performs no FFT/transpose planning -- every FFTCodegenPlan/
    PhysicalTransposePlan field is already decided in `plan`.

    `compute_lanes`: the SIMD width the *arithmetic* in each leaf/near-FFT
    kernel's stages actually emits, independent of that kernel's own
    `simd_lanes` (the launch granule tied to `PooledRange`/`VECTOR_WIDTH`
    -- see fft_codegen._chunk_batch). `None` (the default) emits every
    stage at its own `simd_lanes`, unchanged from before this parameter
    existed. Pass something smaller (e.g. 4 on this target's 128-bit VLEN,
    vs. the default simd_lanes=8) to keep RVV at LMUL=1 and avoid the
    register-spill/insertelement instructions the M2NDP simulator's ISA
    subset does not implement (see docs/STATUS.md). Only the ordinary FFT
    stage kernels (_emit_kernel) take this; the standalone tiled-transpose
    kernels (_emit_physical_transpose_kernel) have their own separate
    width story (ki_near/ki_far, see this module's own docstring) that
    this does not touch.

    `narrow_middle_stages`: see `fft_codegen._stage_compute_lanes` -- two
    independent reasons to halve `compute_lanes` (floor 1) live there, and
    only one is gated by this flag. `False` (the default here; `True` in
    make_fft_kernel.py) leaves a stage that is neither its own kernel's
    first nor last at the caller's own `compute_lanes` -- the shape a real
    N=630 register-pressure failure was isolated to this session (a
    `csrr ..., vlenb` dynamic spill-slot read the M2NDP-Detour simulator's
    `ReadCsr` silently answers `0` for, corrupting the stack) even at
    `compute_lanes=4`, this target's documented "safe" default; narrower
    `compute_lanes` alone (2 or 1) also happened to avoid it, but only by
    accident of which spill *shape* the compiler picked, not because the
    underlying spill was gone.

    The *other* reason -- `stage.radix in fft_codegen._ALWAYS_NARROW_
    RADICES` (the same `vlenb` gap, reached by a big radix's own operand
    count instead of stage position: N=11/13/17 standalone all confirmed
    MISMATCH at compute_lanes=4, no scratchpad/middle-stage involvement at
    all) -- is unconditional, regardless of this flag's own value: this
    project has no configuration where that radix's own compute_lanes=4
    rendering has ever been confirmed to run clean, so there is no "old
    shape" for a caller to opt back into for it the way there is for the
    middle-stage liability.

    `reference_check`: `True` (the default) keeps today's fully self-
    contained host check -- a direct O(N^2) DFT computed right here in the
    generated Mojo, no external dependency, matching every other
    benchmark's own main() (see _emit_reference_check). That check is
    O(N^2) work on top of whatever the FFT itself costs, which stops being
    a rounding error once N is large (N=65536 is ~4.3e9 scalar trig
    evaluations on the host, alone dwarfing a benchmark run meant to
    measure the *device* kernels) -- `False` skips it and instead dumps the
    random input and the device's own output (see _emit_array_dump) between
    `INPUT_BEGIN`/`INPUT_END` and `OUTPUT_BEGIN`/`OUTPUT_END` markers, for a
    Python-side harness to parse and check against `numpy.fft` (O(N log N),
    and in optimized C) instead. This changes nothing about the device
    kernels or their own correctness -- only how large-N runs verify the
    result without host-side O(N^2) dominating the wall-clock time.

    `spread_across_units`: `False` (the default) keeps every host-side
    launch round exactly `stage.max_uthread` microthreads wide -- today's
    unchanged behavior, byte-identical output. `True` widens eligible
    rounds (see `_stage_round_size`'s own docstring for exactly which
    stages qualify) up to `_safe_round_size`'s own cap, so a round can
    genuinely spread across more than one physical NDP unit instead of
    every launch this generator has ever produced landing on unit 0
    alone (confirmed 2026-08-28: `_cap_max_uthread` sizes `max_uthread`
    for one unit's own scratchpad, and nothing before this flag existed
    ever asked for more than that many microthreads in one `.launch()`
    call). `target`: only consulted for `num_ndp_units`/
    `interleave_chunk_uthreads` when this flag is on.

    `debug_print_pool_alignment`: `False` (the default) unchanged output.
    `True` adds one `print()` per stage's own `pool{i}` right after it is
    allocated, dumping the real runtime address mod 64/128/256/2048 --
    ad hoc instrumentation for `planning/fft_unit_utilization.py`'s own
    Phase 1 runtime-alignment audit (docs/active_ndp_units_cost_task.md),
    same "probe kernel, not a planning/codegen decision" spirit as the
    standalone `id_dump.mojo`. Never changes which pool a stage actually
    gets (added strictly after the existing alloc/align logic, whichever
    branch that already is) -- observation only, no simulator source
    touched, no alignment behavior touched.
    """
    stages = flatten_recursive_node(plan.root)
    e = Emitter()
    _emit_prelude(e)
    e.add(f"comptime N = {plan.n}")
    e.add()

    # Section banners are purely cosmetic (never parsed back by
    # verify_fft_recursive.py -- that re-executes each stage's own
    # _emit_physical_transpose_stage/_emit_stage text directly, never this
    # function's overall output) -- here only so the generated file's own
    # shape (every kernel implementation, then the one host driver that
    # chains them) is visible at a glance. Each stage's kernel_name prefix
    # (Pre/Near/Mid/Leaf/Post -- see fft_plan_recursive.py's own naming)
    # already encodes its role in the PRE -> FFT(B) -> MIDDLE -> recurse ->
    # POST sequence.
    e.add("# " + "=" * 76)
    e.add("# KERNEL IMPLEMENTATIONS")
    e.add("# " + "=" * 76)
    e.add(f"# {len(stages)} stage(s), chained through DRAM in this order:")
    for i, stage in enumerate(stages):
        kind = "tiled transpose" if isinstance(stage, PhysicalTransposePlan) else "FFT"
        e.add(f"#   {i}: {stage.kernel_name} ({kind})")
    e.add()

    twiddle_names: dict[int, tuple[str, str]] = {}
    # Only FFTCodegenPlan stages (leaf/near_fft) go through _emit_kernel's
    # loop_stages path; each entry here is that stage's own pooled twiddle
    # table (see fft_codegen._emit_task_struct), empty when loop_stages is
    # False or nothing in that particular kernel qualified to loop.
    #
    # A cooperative stage (stage.cooperation is not None -- see
    # fft_plan_cooperative.py) loops too, per worker (see fft_cooperative_
    # codegen._emit_cooperative_stage's own docstring) -- confirmed
    # necessary rather than optional by a real N=1024 register-pressure
    # failure (2026-08-27) from rendering every worker's own batches fully
    # unrolled into one shared function. Still decided per stage (not a
    # single blanket toggle for the whole tree), so a recursive tree mixing
    # cooperative and plain leaves renders each correctly either way.
    loop_twiddle_tables: dict[int, list[tuple[float, float]]] = {}
    # This stage's own resolved loop_stages (see the comment above) -- the
    # launch site below needs this per-stage answer too, to know whether
    # *this* kernel's Params actually declared the loop-twiddle fields
    # (`add_loop_twiddle` inside _emit_kernel got the same value), not the
    # caller's blanket `loop_stages` argument.
    stage_loops: dict[int, bool] = {}
    for i, stage in enumerate(stages):
        e.add(f"# ---- stage {i}: {stage.kernel_name} ----")
        if isinstance(stage, PhysicalTransposePlan):
            _emit_physical_transpose_kernel(e, plan=stage)
        elif stage.persistent is not None:
            # A persistent leaf renders its own preload/stage_N/writeback
            # struct + round-unrolled device_main (emit_persistent_kernel_
            # struct, shared with the standalone generate_persistent_fft_
            # kernel entry point) -- not _emit_kernel's plain/cooperative
            # per-stage NDPTask shape at all, and it never loops (its own
            # emit_stage_phase already renders every worker's batches
            # fully unrolled -- see that module's own top docstring for
            # why round-specific *functions* don't work here, a different
            # liability than loop_stages targets elsewhere), so this
            # stage contributes nothing to loop_twiddle_tables/stage_loops.
            assert stage.host.total_elems % stage.length == 0, (
                f"stage {i} ({stage.kernel_name}): host.total_elems="
                f"{stage.host.total_elems} not a multiple of length={stage.length}"
            )
            num_logical_blocks = stage.host.total_elems // stage.length
            emit_persistent_kernel_struct(
                e, plan=stage, num_logical_blocks=num_logical_blocks,
                compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
                mode=persistent_mode,
            )
        else:
            stage_loop_stages = loop_stages
            stage_loops[i] = stage_loop_stages
            loop_twiddle_tables[i] = _emit_kernel(
                e, plan=stage, compute_lanes=compute_lanes,
                narrow_middle_stages=narrow_middle_stages, loop_stages=stage_loop_stages,
            )

    e.add("# " + "=" * 76)
    e.add("# HOST MAIN -- allocates DRAM buffers/twiddle tables, launches every")
    e.add("# kernel above in order, checks the result against an independent")
    e.add("# reference DFT computed at runtime.")
    e.add("# " + "=" * 76)
    e.add()

    host = plan.host
    m = len(stages)

    def buf_name(idx: int, part: str) -> str:
        if idx == -1:
            return f"input_{part}"
        if idx == m - 1:
            return f"output_{part}"
        # alternate between two DRAM work buffers
        return f"work{idx % 2}_{part}"

    e.add("def main() raises:")
    first_kernel_name = stages[0].kernel_name
    e.add(f"    if {first_kernel_name}.emit_ir_if_asked():")
    e.add("        return")
    e.add()
    # plan.batch independent length-host.n transforms sit back to back in
    # one flat buffer (see make_recursive_transpose_plan's own `batch`
    # docstring) -- every per-stage kernel below is already sized for this
    # (total_uthreads scales with root.r == plan.batch throughout the whole
    # tree), so only this host driver's own buffer sizing/fill/reference-
    # check needs to know about it explicitly.
    e.add(f"    var total_elems = {host.n * plan.batch}")
    e.add("    var input_real = cxl_alloc[Float32](total_elems)")
    e.add("    var input_imag = cxl_alloc[Float32](total_elems)")
    e.add("    var work0_real = cxl_alloc[Float32](total_elems)")
    e.add("    var work0_imag = cxl_alloc[Float32](total_elems)")
    e.add("    var work1_real = cxl_alloc[Float32](total_elems)")
    e.add("    var work1_imag = cxl_alloc[Float32](total_elems)")
    e.add("    var output_real = cxl_alloc[Float32](total_elems)")
    e.add("    var output_imag = cxl_alloc[Float32](total_elems)")
    e.add("    var ref_real = cxl_alloc[Float32](total_elems)")
    e.add("    var ref_imag = cxl_alloc[Float32](total_elems)")
    e.add()

    for i, stage in enumerate(stages):
        if isinstance(stage, PhysicalTransposePlan) and stage.twiddle_modulus is not None:
            rn, imn = f"twiddle{i}_real", f"twiddle{i}_imag"
            twiddle_names[i] = (rn, imn)
            table_size = stage.rows * stage.cols
            e.add(f"    var {rn} = cxl_alloc[Float32]({table_size})")
            e.add(f"    var {imn} = cxl_alloc[Float32]({table_size})")
            _emit_physical_transpose_twiddle_table_precompute(e, plan=stage, real_name=rn, imag_name=imn, index=i)

    # `tile_id // {tiles_per_replica,grid_cols}` tables (see
    # PhysicalTransposePlan.needs_q_table/needs_tr_table and
    # _emit_physical_transpose_stage): plain Python integer
    # division, at codegen time, into a small DRAM table -- avoids the
    # runtime `mulhsu` a non-power-of-2 constant division would otherwise
    # lower to, an opcode M2NDP-Detour's decoder does not implement.
    tile_coord_names: dict[int, tuple[str | None, str | None]] = {}
    for i, stage in enumerate(stages):
        if not isinstance(stage, PhysicalTransposePlan):
            continue
        q_name = tr_name = None
        if stage.needs_q_table:
            tiles_per_replica = stage.grid_rows * stage.grid_cols
            q_name = f"qtab{i}"
            e.add(f"    var {q_name} = cxl_alloc[Int]({stage.total_uthreads})")
            for tid in range(stage.total_uthreads):
                e.add(f"    {q_name}[{tid}] = {tid // tiles_per_replica}")
            e.add()
        if stage.needs_tr_table:
            tiles_per_replica = stage.grid_rows * stage.grid_cols
            tr_name = f"trtab{i}"
            e.add(f"    var {tr_name} = cxl_alloc[Int]({tiles_per_replica})")
            for lt in range(tiles_per_replica):
                e.add(f"    {tr_name}[{lt}] = {lt // stage.grid_cols}")
            e.add()
        tile_coord_names[i] = (q_name, tr_name)

    # loop_stages's own per-kernel pooled twiddle table (see
    # _try_build_loop_stage): a flat DRAM array of exact already-plan-
    # computed constants, filled by literal assignment (these are not a
    # runtime formula fftcodegen deliberately doesn't re-derive -- see the
    # runtime-loop stage rendering note in fft_codegen.py) and read back at
    # runtime by `simd_it` inside each looped stage. `_emit_kernel` always
    # declares the Params field whenever loop_stages was requested for that
    # kernel (see _emit_params_struct's add_loop_twiddle), even when this
    # table ends up empty, so the alloc below always runs alongside it --
    # `max(1, ...)` keeps a zero-length table a valid (unused) allocation.
    loop_twiddle_names: dict[int, tuple[str, str]] = {}
    for i, table in loop_twiddle_tables.items():
        rn, imn = f"looptw{i}_real", f"looptw{i}_imag"
        loop_twiddle_names[i] = (rn, imn)
        table_size = max(1, len(table))
        e.add(f"    var {rn} = cxl_alloc[Float32]({table_size})")
        e.add(f"    var {imn} = cxl_alloc[Float32]({table_size})")
        for idx, (vr, vi) in enumerate(table):
            e.add(f"    {rn}[{idx}] = {_f32(vr)}")
            e.add(f"    {imn}[{idx}] = {_f32(vi)}")
        e.add()

    # A stage's pool only ever needs to be as wide as its own biggest
    # round -- `max_uthread` microthreads by default (this stage's own
    # scratchpad cap), or `_stage_round_size`'s wider cap when
    # `spread_across_units` is on and this stage is eligible -- even when
    # `total_uthreads` is bigger and this stage below launches in several
    # rounds, since the pool is reused across rounds (sequential launches,
    # nothing live across them -- see the round loop below). Must track
    # whatever round size is actually used per stage below, or a launch
    # ends up narrower than the round it claims to cover (see
    # docs/SIMULATION.md's own "narrower than one stride spreads
    # unevenly" note -- that's about correctness, not just a wasted
    # allocation).
    round_sizes = {
        i: _stage_round_size(stage, spread_across_units=spread_across_units, target=target)
        for i, stage in enumerate(stages)
    }
    for i, stage in enumerate(stages):
        e.add(f"    var pool{i}_elems = {stage.simd_lanes * round_sizes[i]}")
        needs_chunk_alignment = not isinstance(stage, PhysicalTransposePlan) and (
            stage.cooperation is not None or stage.persistent is not None
        )
        if needs_chunk_alignment:
            # A cooperative stage's fft_slot/worker_id grouping
            # (_emit_cooperative_stage's own docstring), or a persistent
            # stage's own software_group_id (== global_uthread_id() //
            # workers_per_group, sharing that group's scratchpad the same
            # way -- see PersistentWorkgroupPlan's own docstring), is only
            # sound when local_uthread_id()'s/global_uthread_id()'s own
            # WORKERS_PER_FFT- or workers_per_group-sized groupings
            # partition the same physical microthreads -- which holds only
            # if *this launch's own pool* starts exactly on a hardware
            # interleave-chunk boundary (interleave_chunk_uthreads *
            # uthread_bytes = 256B on this target), since the M2NDP
            # address decoder places a microthread's physical unit from
            # its absolute DRAM address, not from an index relative to
            # the launch. `Pool.alloc` (src/m2ndp_host.mojo) only aligns
            # individual allocations to 64B, so a raw `cxl_alloc` address
            # is not guaranteed 256B-aligned -- confirmed the real, root
            # cause of a workers_per_fft=8 wrong answer previously
            # misdiagnosed as below-the-planning-layer (an unaligned pool
            # put half an 8-worker group on one physical unit and half on
            # the next, each writing into a *different* private
            # scratchpad the other half never sees). `fft_persistent_
            # codegen.py`'s own standalone host main already carries this
            # exact fix (see docs/persistent_leaf_design.md's own "uthread
            # pool alignment" section) for exactly this same reason; this
            # mirrors it here too rather than centralizing into `Pool.
            # alloc` itself, which every non-cooperative, non-persistent
            # benchmark in the repo also uses and does not need the
            # stricter alignment for.
            chunk_bytes = target.interleave_chunk_uthreads * target.uthread_bytes
            pad_elems = chunk_bytes // 4
            e.add(f"    var pool{i}_raw = cxl_alloc[Float32](pool{i}_elems + {pad_elems})")
            e.add(f"    var pool{i}_raw_addr = Int(pool{i}_raw)")
            e.add(
                f"    var pool{i}_addr = (pool{i}_raw_addr + {chunk_bytes - 1}) "
                f"// {chunk_bytes} * {chunk_bytes}"
            )
            e.add(f"    if pool{i}_addr % {chunk_bytes} != 0:")
            e.add(
                f'        print("[host] stage {i} ({stage.kernel_name}) pool '
                f'alignment assertion failed:", pool{i}_addr)'
            )
            e.add("        return")
            e.add(
                f"    var pool{i} = UnsafePointer[Float32, MutAnyOrigin]"
                f"(unsafe_from_address=pool{i}_addr)"
            )
        else:
            e.add(f"    var pool{i} = cxl_alloc[Float32](pool{i}_elems)")
        if debug_print_pool_alignment:
            kind = "transpose" if isinstance(stage, PhysicalTransposePlan) else (
                "persistent" if stage.persistent is not None
                else "cooperative" if stage.cooperation is not None
                else "non_cooperative"
            )
            e.add(f"    var pool{i}_check_addr = Int(pool{i})")
            e.add(
                f'    print("[pool_align] stage", {i}, "kind", "{kind}", '
                f'"addr", pool{i}_check_addr, "mod64", pool{i}_check_addr % 64, '
                f'"mod128", pool{i}_check_addr % 128, "mod256", pool{i}_check_addr % 256, '
                f'"mod2048", pool{i}_check_addr % 2048)'
            )
    e.add()

    e.add("    seed(0)")
    e.add("    for i in range(total_elems):")
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

    if not reference_check:
        _emit_array_dump(e, n=plan.n * plan.batch, real_name="input_real", imag_name="input_imag", marker="INPUT")

    for i, stage in enumerate(stages):
        in_r, in_i = buf_name(i - 1, "real"), buf_name(i - 1, "imag")
        out_r, out_i = buf_name(i, "real"), buf_name(i, "imag")

        if isinstance(stage, PhysicalTransposePlan):
            # See PhysicalTransposePlan.needs_round_split: a tile's own address is a function of
            # tile_id alone (via q/t_r/t_c), so splitting rounds only needs
            # tile_id itself shifted by a per-round `round_offset` -- no
            # DRAM pointer offsetting, unlike the FFTCodegenPlan branch
            # below (see there for why that one's different).
            fixed_args = [rn for rn in twiddle_names.get(i, ()) if rn is not None]
            fixed_args += [n for n in tile_coord_names.get(i, ()) if n is not None]

            round_size = round_sizes[i]
            rounds = -(-stage.total_uthreads // round_size)
            # `stage.needs_round_split` (whether the `round_offset` Params
            # field exists at all) is a plan-time decision against the old
            # max_uthread-only cap -- with spread_across_units on, `rounds`
            # can collapse to 1 here even when needs_round_split is True.
            # Harmless: an always-0 round_offset just gets declared and
            # passed once, never read past that.
            for r in range(rounds):
                round_count = min(round_size, stage.total_uthreads - r * round_size)
                suffix = f"{i}" if rounds == 1 else f"{i}_{r}"
                args = f"{in_r}, {in_i}, {out_r}, {out_i}"
                if fixed_args:
                    args += ", " + ", ".join(fixed_args)
                if stage.needs_round_split:
                    args += f", {r * round_size}"

                e.add(f"    var rc{suffix} = {stage.kernel_name}.launch(")
                e.add(f"        PooledRange.over(pool{i}, {stage.simd_lanes * round_count}),")
                e.add(f"        {stage.kernel_name}Params({args}),")
                e.add("    )")
                e.add(f"    if rc{suffix} != 0:")
                round_note = "" if rounds == 1 else f" round {r}/{rounds}"
                e.add(
                    f'        print("[host] recursive FFT stage {i} ({stage.kernel_name}'
                    f'{round_note}) failed, exit", rc{suffix})'
                )
                e.add("        return")
                e.add()
            continue

        # every FFTCodegenPlan a recursive node builds (leaf or near_fft) is
        # always large_twiddle=None -- that cross-block math lives entirely
        # in the standalone MIDDLE transpose kernel -- and always
        # AddressMapping.contiguous(row_stride=stage.length) on both sides
        # (see _build_recursive_node), so `global_uthread_id() * stage.length`
        # is the whole address story: round `r`'s microthreads
        # [r*max_uthread, r*max_uthread+round_count) read/write exactly the
        # contiguous `round_count*stage.length`-element slice starting
        # `r*max_uthread*stage.length` into this stage's own in/out buffers.
        # Offsetting the pointers `Params` gets by that amount and launching
        # over a `round_count`-sized (<= max_uthread) slice of the pool
        # covers it -- no different, in the DRAM buffers' own terms, from
        # this stage simply having been `rounds` separate, smaller stages.
        #
        # Launching the *whole* total_uthreads in one go over a scratchpad
        # sized for only max_uthread of them (local_uthread_id() cycles
        # 0..max_uthread-1 per physical unit, so anything that doesn't fit
        # on one unit needs another launch) used to hang the simulator
        # rather than error -- confirmed by hand with total_uthreads=4,
        # max_uthread=1, *without* accounting for address-interleaving at
        # all. `_stage_round_size` above (spread_across_units) is not that
        # naive removal of the cap -- it's a real, proven-safe wider bound
        # (see _safe_round_size's own docstring): no physical unit is ever
        # asked for more than max_uthread microthreads even when a round
        # spans several of them, so this doesn't reintroduce that hang.
        assert stage.large_twiddle is None
        round_size = round_sizes[i]
        rounds = -(-stage.total_uthreads // round_size)
        # elem_off advances by this stage's own *logical* replicas per
        # round, not physical microthreads once workers share one
        # replica's scratchpad -- see FFTCodegenPlan.replicas_per_round's
        # own docstring. That method's own value is plan-time-fixed
        # against the old max_uthread cap, so it's only reused as-is for
        # a cooperative stage (never widened by spread_across_units, see
        # _stage_round_size); for a non-cooperative stage, one physical
        # microthread is already one replica (replicas_per_round() ==
        # max_uthread whenever cooperation is None -- see that method's
        # own docstring), so round_size itself is the right value once a
        # round can be wider than max_uthread.
        replicas_per_round = round_size if stage.cooperation is None else stage.replicas_per_round()
        for r in range(rounds):
            round_count = min(round_size, stage.total_uthreads - r * round_size)
            elem_off = r * replicas_per_round * stage.length
            round_in_r = in_r if elem_off == 0 else f"({in_r} + {elem_off})"
            round_in_i = in_i if elem_off == 0 else f"({in_i} + {elem_off})"
            round_out_r = out_r if elem_off == 0 else f"({out_r} + {elem_off})"
            round_out_i = out_i if elem_off == 0 else f"({out_i} + {elem_off})"
            suffix = f"{i}" if rounds == 1 else f"{i}_{r}"

            e.add(f"    var rc{suffix} = {stage.kernel_name}.launch(")
            e.add(f"        PooledRange.over(pool{i}, {stage.simd_lanes * round_count}),")
            if stage_loops.get(i, False):
                ltrn, ltimn = loop_twiddle_names[i]
                e.add(
                    f"        {stage.kernel_name}Params({round_in_r}, {round_in_i}, "
                    f"{round_out_r}, {round_out_i}, {ltrn}, {ltimn}),"
                )
            else:
                e.add(
                    f"        {stage.kernel_name}Params({round_in_r}, {round_in_i}, "
                    f"{round_out_r}, {round_out_i}),"
                )
            e.add("    )")
            e.add(f"    if rc{suffix} != 0:")
            round_note = "" if rounds == 1 else f" round {r}/{rounds}"
            e.add(
                f'        print("[host] recursive FFT stage {i} ({stage.kernel_name}'
                f'{round_note}) failed, exit", rc{suffix})'
            )
            e.add("        return")
            e.add()

    if reference_check:
        _emit_reference_check(
            e, n=plan.n, batch_count=plan.batch, inverse=plan.inverse,
            input_real="input_real", input_imag="input_imag",
            output_real="output_real", output_imag="output_imag",
            ref_real="ref_real", ref_imag="ref_imag",
            tolerance=host.tolerance, label="recursive tiled-transpose FFT",
        )
    else:
        _emit_array_dump(e, n=plan.n * plan.batch, real_name="output_real", imag_name="output_imag", marker="OUTPUT")
    return e.text()
