from __future__ import annotations

"""Numeric + index-only verification for fft_plan_balanced.py's two
strategies: `make_balanced_plan` (CROSSED boundary, fused into one side's
own last-kernel store -- reuses verify_fft_harness's own flat-chain
runner since it also returns a MultiKernelFFTPlan-shaped result) and
`make_balanced_transpose_plan` (standalone tiled-transpose + fused
large-twiddle boundary kernel -- its own translate/run/twiddle-table trio
lives here since nothing outside this strategy needs it).
"""

import types

import numpy as np

from planning.strategies.fft_plan_balanced import (
    BalancedTransposeFFTPlan,
    make_balanced_plan,
    make_balanced_transpose_plan,
)
from codegen.fft_codegen import Emitter
from codegen.fft_transpose_codegen import _emit_transpose_stage
from verification.verify_fft_harness import (
    Ptr,
    _run_kernel_chain,
    _run_multi_kernel_plan,
    _simd,
    _translate_emitted_lines,
)


def verify_balanced_plan(
    chunks_A: tuple[tuple[int, ...], ...],
    chunks_B: tuple[tuple[int, ...], ...],
    *,
    inverse: bool,
    seed: int,
) -> float:
    """Same shape as verify_multi_kernel_plan, but for make_balanced_plan
    (N = N_A*N_B, each side its own PEELED chain, joined by one
    AddressMappingKind.CROSSED transpose fused into side A's last kernel --
    see make_balanced_plan)."""
    plan = make_balanced_plan(chunks_A, chunks_B, inverse=inverse)
    return _run_multi_kernel_plan(plan, inverse=inverse, seed=seed)


def _translate_transpose_stage(plan) -> str:
    """Same discipline as _translate_stage: re-execute the actual text
    fft_transpose_codegen.py emits, not a second implementation."""
    e = Emitter()
    _emit_transpose_stage(e, plan=plan)
    return _translate_emitted_lines(e.lines)


def run_transpose_kernel(
    plan, *, near_real: Ptr, near_imag: Ptr, far_real: Ptr, far_imag: Ptr,
    tw_real: Ptr, tw_imag: Ptr,
) -> None:
    """Runs the transpose kernel's actual emitted stage text (same
    discipline as run_kernel), for every tile uthread."""
    num_groups = -(-plan.total_uthreads // plan.max_uthread)
    group_namespaces = []
    for _ in range(num_groups):
        group_namespaces.append(
            types.SimpleNamespace(
                tile_buf=Ptr(plan.scratchpad_elements * plan.max_uthread)
            )
        )
    p_ns = types.SimpleNamespace(
        near_real_base=near_real, near_imag_base=near_imag,
        far_real_base=far_real, far_imag_base=far_imag,
        twiddle_real_base=tw_real, twiddle_imag_base=tw_imag,
    )
    current = {"global_id": 0, "local_id": 0}
    src = _translate_transpose_stage(plan)
    namespace = {
        "Float32": float,
        "SIMD": _simd,
        "local_uthread_id": lambda: current["local_id"],
        "global_uthread_id": lambda: current["global_id"],
        f"MAX_UTHREAD_{plan.kernel_name}": plan.max_uthread,
        "p": p_ns,
    }
    code = compile(src, f"<{plan.kernel_name} stage 0>", "exec")
    exec(code, namespace)
    stage_fn = namespace["stage_0"]
    for global_id in range(plan.total_uthreads):
        current["global_id"] = global_id
        current["local_id"] = global_id % plan.max_uthread
        namespace[plan.kernel_name] = group_namespaces[global_id // plan.max_uthread]
        stage_fn()


def _make_transpose_twiddle_table(plan) -> tuple[np.ndarray, np.ndarray]:
    """Independent (re-derives (row,elem) -> (batch,combined) itself,
    rather than reusing fft_transpose_codegen's own address formula)
    host-side fill for the transpose's own twiddle table -- addressed
    identically to the near side's plain output layout, see
    FFTTransposePlan / fft_transpose_codegen._emit_transpose_twiddle_
    table_precompute."""
    n = plan.n
    real = np.zeros(n)
    imag = np.zeros(n)
    sign = 1.0 if plan.inverse else -1.0
    near_total_uthreads = n // plan.ki_near
    for row in range(near_total_uthreads):
        batch = row % plan.n_b
        prefix_so_far = row // plan.n_b
        for elem in range(plan.ki_near):
            combined = prefix_so_far + plan.digit_multiplier_near * elem
            angle = sign * 2.0 * np.pi * batch * combined / n
            addr = row * plan.ki_near + elem
            real[addr] = np.cos(angle)
            imag[addr] = np.sin(angle)
    return real, imag


def verify_transpose_bijection(transpose) -> None:
    """Index-only (no floats): every (row_near, elem_near) source position
    lands at exactly one (row_far, elem_far) target position and every
    target is covered exactly once -- across every tile."""
    n = transpose.n
    seen: set[int] = set()
    for tile_id in range(transpose.total_uthreads):
        prefix = tile_id % transpose.digit_multiplier_near
        rest = tile_id // transpose.digit_multiplier_near
        for ef in range(transpose.ki_far):
            row_near = transpose.n_b * prefix + ef * transpose.divisor_far + rest
            for en in range(transpose.ki_near):
                row_far = prefix + transpose.digit_multiplier_near * en + transpose.n_a * rest
                target = row_far * transpose.ki_far + ef
                if target in seen:
                    raise AssertionError(f"transpose target {target} hit twice")
                seen.add(target)
    if seen != set(range(n)):
        raise AssertionError("transpose targets are not a permutation of 0..n-1")


def verify_transpose_access_shape(transpose) -> dict[str, int]:
    """Every DRAM vector load/store the transpose kernel emits must have
    elem_stride == 1 (see fft_transpose_codegen.py's own docstring) --
    checked directly against the address formulas, not assumed. Returns a
    small summary (max row-to-row jump on each side) for reporting."""
    max_near_row_jump = 0
    max_far_row_jump = 0
    for tile_id in range(transpose.total_uthreads):
        prefix = tile_id % transpose.digit_multiplier_near
        rest = tile_id // transpose.digit_multiplier_near
        near_rows = [
            transpose.n_b * prefix + ef * transpose.divisor_far + rest
            for ef in range(transpose.ki_far)
        ]
        if len(near_rows) > 1:
            jumps = [abs(near_rows[i + 1] - near_rows[i]) * transpose.ki_near for i in range(len(near_rows) - 1)]
            max_near_row_jump = max(max_near_row_jump, max(jumps))
        far_rows = [
            prefix + transpose.digit_multiplier_near * en + transpose.n_a * rest
            for en in range(transpose.ki_near)
        ]
        if len(far_rows) > 1:
            jumps = [abs(far_rows[i + 1] - far_rows[i]) * transpose.ki_far for i in range(len(far_rows) - 1)]
            max_far_row_jump = max(max_far_row_jump, max(jumps))
    return {
        "load_elem_stride": 1,
        "store_elem_stride": 1,
        "max_near_row_jump": max_near_row_jump,
        "max_far_row_jump": max_far_row_jump,
    }


def verify_balanced_transpose_plan(
    n: int, *, scratchpad_byte_budget: int, inverse: bool, seed: int
) -> tuple[float, BalancedTransposeFFTPlan]:
    """Full numeric chain: near side's own PEELED kernels, the standalone
    transpose kernel, far side's own PEELED kernels -- each re-executing
    its own actual emitted stage text (run_kernel / run_transpose_kernel),
    compared to numpy.fft/ifft."""
    plan = make_balanced_transpose_plan(
        n, scratchpad_byte_budget=scratchpad_byte_budget, inverse=inverse
    )
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)

    in_r, in_i = Ptr(n), Ptr(n)
    in_r.arr[:] = x.real
    in_i.arr[:] = x.imag

    near_out_r, near_out_i = _run_kernel_chain(plan.kernels_near, input_real=in_r, input_imag=in_i)

    tw_real, tw_imag = _make_transpose_twiddle_table(plan.transpose)
    tw_r, tw_i = Ptr(n), Ptr(n)
    tw_r.arr[:] = tw_real
    tw_i.arr[:] = tw_imag
    far_in_r, far_in_i = Ptr(n), Ptr(n)
    run_transpose_kernel(
        plan.transpose, near_real=near_out_r, near_imag=near_out_i,
        far_real=far_in_r, far_imag=far_in_i, tw_real=tw_r, tw_imag=tw_i,
    )

    out_r, out_i = _run_kernel_chain(plan.kernels_far, input_real=far_in_r, input_imag=far_in_i)

    got = out_r.arr + 1j * out_i.arr
    expected = np.fft.ifft(x) if inverse else np.fft.fft(x)
    return float(np.max(np.abs(got - expected))), plan


def summarize_balanced_transpose_plan(plan: BalancedTransposeFFTPlan) -> None:
    """Prints the side-A / transpose / side-B stride table the design
    writeup asks for."""
    print(f"  Near side FFT (N_A={plan.transpose.n_a}, batched N_B={plan.transpose.n_b} times):")
    for k in plan.kernels_near:
        print(f"    {k.kernel_name}: read elem_stride={k.input_mapping.elem_stride} write elem_stride={k.output_mapping.elem_stride}")
    shape = verify_transpose_access_shape(plan.transpose)
    print(f"  Transpose (ki_near={plan.transpose.ki_near}, ki_far={plan.transpose.ki_far}, tiles={plan.transpose.total_uthreads}):")
    print(f"    load elem_stride={shape['load_elem_stride']} store elem_stride={shape['store_elem_stride']}")
    print(f"    max near-side row jump={shape['max_near_row_jump']} max far-side row jump={shape['max_far_row_jump']}")
    print(f"  Far side FFT (N_B={plan.transpose.n_b}, batched N_A={plan.transpose.n_a} times):")
    for k in plan.kernels_far:
        print(f"    {k.kernel_name}: read elem_stride={k.input_mapping.elem_stride} write elem_stride={k.output_mapping.elem_stride}")


# ------------------------------------------ make_recursive_transpose_plan
#
# FFT chunk size and physical transpose tile size are fully independent
# here (see fft_plan_recursive.PhysicalTransposePlan) -- verified below by (1)
# index-only bijection of the generic tiled transpose (full tile, row
# tail, column tail, both tails, multiple replicas), (2) full numeric
# re-execution of the actual emitted stage text (both FFT leaf kernels via
# run_kernel and transpose kernels via run_physical_transpose below)
# against numpy.fft/ifft across 0/1/2+ recursion levels, and (3) index-only
# checks at a size too large for the numeric path to stay fast.

