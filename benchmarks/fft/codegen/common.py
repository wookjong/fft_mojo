from __future__ import annotations

"""Shared source-emission helpers, including resolved FFT launch rendering.
Radices, layouts, twiddle values and launch ranges are supplied by plans.
"""

from planning.core.fft_plan_core import LargeTwiddlePlan
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskKernelBinding:
    """A phase's owner/ABI and physical dispatch, independent of FFT math."""
    task_name: str
    params_field: str
    function_name: str
    active_uthreads: int
    local_slot_expr: str
    arena_name: str = "arena"
    window_batch: int = 0
    total_batch: int = 0
    window_n: int = 0

    def emit_window_prelude(self, e):
        if self.window_batch:
            e.add(f'        var batch_base = Int({self.task_name}.batch_base.load[DType.int32, 1](0)[0])')
            if self.total_batch % self.window_batch:
                e.add(f'        var live_batch = min({self.window_batch}, {self.total_batch} - batch_base)')
            e.add(f'        var window_base = batch_base * {self.window_n}')

    @property
    def params_expr(self):
        return f"{self.task_name}.params[].{self.params_field}"

    @property
    def arena_expr(self):
        return f"{self.task_name}.{self.arena_name}"


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


def emit_aligned_pool(e: Emitter, *, name: str, elems: str, alignment_bytes: int) -> None:
    """Allocate a launch pool aligned to the target's address-mapping chunk."""
    if not isinstance(alignment_bytes, int) or alignment_bytes < 4 or alignment_bytes % 4:
        raise ValueError("pool alignment must be a positive multiple of four bytes")
    e.add(f"    var {name}_raw = cxl_alloc[Float32]({elems} + {alignment_bytes // 4})")
    e.add(f"    var {name}_raw_addr = Int({name}_raw)")
    e.add(f"    var {name}_addr = ({name}_raw_addr + {alignment_bytes - 1}) // {alignment_bytes} * {alignment_bytes}")
    e.add(f"    if {name}_addr % {alignment_bytes} != 0:")
    e.add('        print("[host] pool alignment assertion failed")')
    e.add("        return")
    e.add(f"    var {name} = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address={name}_addr)")


def emit_fft_identity(e: Emitter, *, plan) -> bool:
    """Resolve padded physical IDs and host rounds into one logical FFT ID."""
    launch = plan.launch_mapping
    if launch is None or not launch.uses_bounds_params:
        return False
    e.add(f"        var launch_fft_id = {launch.local_fft_expr()}")
    e.add("        if launch_fft_id >= p.launch_fft_count:")
    e.add("            return")
    e.add("        var logical_fft_id = p.launch_fft_base + launch_fft_id")
    return True


def emit_fft_launches(e: Emitter, *, plan, pool: str, args: list[str], rc_prefix: str = "rc",
                      label: str = "FFT") -> None:
    """Host rounds use the same descriptor as device identity and pool sizing."""
    args = [*args, *emit_workspace_buffers(e, plan=plan, prefix=f"{rc_prefix}_")]
    launch = plan.launch_mapping
    if launch is None:
        if plan.persistent is None and plan.distributed is None:
            raise ValueError("FFT launch descriptor has not been resolved")
        from planning.execution.launch_plan import FFTLaunchRound
        rounds = (FFTLaunchRound(0, plan.host.total_elems // plan.length,
                                plan.total_uthreads, plan.host.pool_elems),)
        bounds, multiple = False, False
    else:
        rounds = launch.rounds()
        bounds = launch.uses_bounds_params
        multiple = launch.total_ffts > launch.round_fft_capacity
    for index, round_ in enumerate(rounds):
        params = list(args)
        if bounds:
            params.extend((str(round_.fft_base), str(round_.fft_count)))
        rc = f"{rc_prefix}_{index}" if multiple else rc_prefix
        e.add(f"    var {rc} = {plan.kernel_name}.launch(")
        e.add(f"        PooledRange.over({pool}, {round_.pool_elems}),")
        e.add(f"        {plan.kernel_name}Params({', '.join(params)}),")
        e.add("    )")
        e.add(f"    if {rc} != 0:")
        e.add(f'        print("[host] {label} failed, exit", {rc})')
        e.add("        return")
        e.add()


def emit_workspace_buffers(e: Emitter, *, plan, prefix: str = "") -> list[str]:
    """Allocate the plan's CXL staging buffers once before launching its task.

    A leaf-resident buffer (plan.leaf_storage is not None -- see
    LeafResidentSpadPlan) needs no CXL allocation or Params argument at all:
    it lives in the owning task's own shared arena instead (see
    fft_codegen._emit_params_struct, which drops the matching Params field).
    """
    if plan.leaf_storage is not None:
        return []
    names = []
    for buffer in plan.workspace_buffers:
        name = f"{prefix}{buffer.name}"
        e.add(f"    var {name} = cxl_alloc[Float32]({buffer.elements})")
        names.append(name)
    return names


def emit_loop_twiddle_table(e: Emitter, *, table: list[tuple[float,float]],
                           real_name: str, imag_name: str) -> list[str]:
    """Emit exact planned constants, compressing repeated adjacent entries."""
    e.add(f"    var {real_name} = cxl_alloc[Float32]({max(1,len(table))})")
    e.add(f"    var {imag_name} = cxl_alloc[Float32]({max(1,len(table))})")
    if len(table) >= 1024:
        # Encode exact FP32 bits so a large table is one literal and a host
        # decode loop, instead of tens of thousands of LLVM store nodes.
        # No trigonometry or FFT decisions are reconstructed here.
        import struct
        encoded = ''.join(f'{struct.unpack("<I",struct.pack("<f",v))[0]:08x}' for pair in table for v in pair)
        literal = f"{real_name}_hex"
        e.add(f'    var {literal} = String("{encoded}")')
        e.add(f"    for tw_idx in range({len(table)}):")
        for component,name in enumerate((real_name,imag_name)):
            e.add("        if True:")
            e.add("            var tw_bits = UInt32(0)")
            e.add("            for tw_nibble in range(8):")
            e.add(f"                var tw_code = Int({literal}.unsafe_ptr()[tw_idx * 16 + {component*8} + tw_nibble])")
            e.add("                var tw_digit = tw_code & 15")
            e.add("                if tw_code >= 97:")
            e.add("                    tw_digit += 9")
            e.add("                tw_bits = (tw_bits << 4) | UInt32(tw_digit)")
            e.add(f"            {name}.bitcast[UInt32]()[tw_idx] = tw_bits")
        return [real_name,imag_name]
    first = 0
    while first < len(table):
        end = first + 1
        while end < len(table) and table[end] == table[first]:
            end += 1
        r,i = table[first]
        if end - first >= 3:
            e.add(f"    for tw_idx in range({first}, {end}):")
            e.add(f"        {real_name}[tw_idx] = {f32(r)}")
            e.add(f"        {imag_name}[tw_idx] = {f32(i)}")
        else:
            for index in range(first,end):
                e.add(f"    {real_name}[{index}] = {f32(r)}")
                e.add(f"    {imag_name}[{index}] = {f32(i)}")
        first = end
    return [real_name,imag_name]


def emit_execution_record(e: Emitter, *, plan, **options) -> None:
    """Machine-readable resolved options, without changing the planned settings."""
    from dataclasses import asdict
    import json
    from planning.execution.fft_plan_lanes import resolve_stage_compute_lanes
    launch = plan.launch_mapping
    lanes = []
    for index, stage in enumerate(plan.stages):
        width = stage.compute_lanes
        if width is None:
            width = resolve_stage_compute_lanes(
                compute_lanes=options.get("compute_lanes"), is_first=index == 0,
                is_last=index == len(plan.stages) - 1, radix=stage.radix,
                prev_radix=plan.stages[index - 1].radix if index else None,
                narrow_middle_stages=options.get("narrow_middle_stages", False))
        lanes.append(width if width is not None else plan.simd_lanes)
    record = dict(kernel=plan.kernel_name, launch=asdict(launch) if launch else None,
                  arithmetic_lanes=lanes, **options)
    e.add("# FFT_EXECUTION " + json.dumps(record, sort_keys=True))


def emit_prelude(e: Emitter) -> None:
    e.add("from std.sys import size_of")
    e.add("from std.random import random_float64, seed")
    e.add("from std.math import cos as host_cos, sin as host_sin")
    e.add()
    e.add(
        "from m2ndp import VECTOR_WIDTH, NDPTask, PooledRange, "
        "global_uthread_id, local_uthread_id, group_id, num_groups, "
        "launch_parallel, scratchpad, spad_addr"
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
    var_prefix: str = "",
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

    `var_prefix`: namespaces this call's own `pi`/`sign`/`tol` locals (the
    only three declared directly in the caller's function-body scope,
    rather than inside a nested `for`/`while` block) -- default `""`
    keeps every existing single-kernel caller byte-identical. A caller
    that renders TWO independent kernels' verification blocks into one
    shared `main()` (`fft_persistent_codegen.generate_persistent_tail_
    hybrid_kernel`'s own `_emit_persistent_buffers_and_launch`/`_emit_
    plain_tail_buffers_and_launch`) MUST pass two different prefixes --
    confirmed as a real toolchain compile failure ("invalid redefinition
    of 'pi'") whenever both blocks are genuinely non-empty (`full_blocks
    > 0` and `tail_blocks > 0`, e.g. N_blocks=33/63/65 at num_ndp_units=
    32) before this parameter existed; see `emit_large_twiddle_table_
    precompute`'s own docstring for the same collision class, already
    fixed there the same way.
    """
    pi, sign, tol = f"{var_prefix}pi", f"{var_prefix}sign", f"{var_prefix}tol"
    e.add(f"    var {pi} = Float64(3.141592653589793)")
    e.add(f"    var {sign} = Float64({1.0 if inverse else -1.0})")
    e.add(f"    for batch in range({batch_count}):")
    e.add(f"        var batch_base = batch * {n}")
    e.add(f"        for k in range({n}):")
    e.add("            var acc_r = Float64(0)")
    e.add("            var acc_i = Float64(0)")
    e.add(f"            for n_ in range({n}):")
    e.add(
        f"                var angle = {sign} * 2.0 * {pi} * Float64(n_) * Float64(k) / "
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

    e.add(f"    var {tol} = {f32(tolerance)}")
    e.add(f"    for i in range({n * batch_count}):")
    e.add(f"        var err_r = {output_real}[i] - {ref_real}[i]")
    e.add(f"        var err_i = {output_imag}[i] - {ref_imag}[i]")
    e.add("        if err_r < Float32(0):")
    e.add("            err_r = -err_r")
    e.add("        if err_i < Float32(0):")
    e.add("            err_i = -err_i")
    e.add(f"        if err_r > {tol} or err_i > {tol}:")
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
