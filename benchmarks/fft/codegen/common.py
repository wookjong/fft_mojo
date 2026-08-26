from __future__ import annotations

"""Emission-only helpers with no FFT-specific logic of their own -- shared
verbatim by every codegen/*.py module that renders Mojo source (fft_butterflies.py,
fft_codegen.py, fft_transpose_codegen.py) instead of each keeping (or, before
this module existed, in `_f32`'s case, silently duplicating) its own copy.
Nothing here decides an address, a radix, or a twiddle value -- see each
module's own docstring for what those modules *do* decide.
"""

from planning.fft_plan_core import LargeTwiddlePlan


class Emitter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def f32(value: float) -> str:
    """Format a planner-provided numeric constant as Mojo Float32 syntax."""
    if abs(value) < 1.0e-12:
        value = 0.0
    elif abs(value - 1.0) < 1.0e-12:
        value = 1.0
    elif abs(value + 1.0) < 1.0e-12:
        value = -1.0
    return f"Float32({value:.9g})"


def spad(kernel_name: str, name: str) -> str:
    return f"{kernel_name}.{name}"


def emit_prelude(e: Emitter) -> None:
    e.add("from std.sys import size_of")
    e.add("from std.random import random_float64, seed")
    e.add("from std.math import cos as host_cos, sin as host_sin")
    e.add()
    e.add(
        "from m2ndp import VECTOR_WIDTH, NDPTask, PooledRange, "
        "global_uthread_id, local_uthread_id, group_id, num_groups, "
        "launch_parallel, scratchpad"
    )
    e.add("from m2ndp_host import cxl_alloc")
    e.add()
    e.add("comptime W = VECTOR_WIDTH // size_of[Float32]()")


def emit_reference_check(
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
    decomposed alike (see fft_plan_core.MultiKernelHostPlan / fft_plan_simple.
    DecomposedFFTPlan's own host plan). Accumulated in Float64 so this check
    doesn't share the kernel's own fp32 rounding, rounding to Float32 only
    once, at the very end.

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

    e.add(f"    var tol = {f32(tolerance)}")
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


def emit_array_dump(
    e: Emitter, *, n: int, real_name: str, imag_name: str, marker: str
) -> None:
    """Prints `real_name`/`imag_name` between `{marker}_BEGIN`/`{marker}_END`
    lines, one `real imag` pair per element -- an opt-in alternative to
    `emit_reference_check`'s own O(N^2) direct-DFT host verification, for
    an N too large for that to finish in reasonable wall-clock time (see
    fft_transpose_codegen.generate_recursive_fft_kernels' own
    `reference_check` parameter). Numpy's FFT (O(N log N), highly
    optimized C) does the actual correctness check on the Python side
    instead; this only has to get the exact values the device kernel(s)
    actually read/wrote out to where a Python harness can read them back --
    no arithmetic happens here.
    """
    e.add(f'    print("{marker}_BEGIN")')
    e.add(f"    for i in range({n}):")
    e.add(f"        print({real_name}[i], {imag_name}[i])")
    e.add(f'    print("{marker}_END")')
    e.add()


def emit_large_twiddle_table_precompute(
    e: Emitter, *, lt: LargeTwiddlePlan, real_name: str, imag_name: str, suffix: str
) -> None:
    """Host precompute for one non-last kernel's large-twiddle table (see
    fft_plan_core.LargeTwiddlePlan / fft_codegen.generate_multi_kernel_fft_kernels).
    `real_name`/`imag_name` are each `lt.full_length` Float32 elements,
    filled at every address the on-device fetch will ever read.

    Every local variable is suffixed: this runs once per non-last kernel
    in the same main(), and an earlier version of the single-kernel host
    check hit exactly this collision (two `var pi = ...`s in one scope --
    see fft_plan_simple.make_decomposed_plan's own inverse_scale/lt_pi fix)
    for the same reason.
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
