from __future__ import annotations

"""Numeric + index-only verification for fft_plan_multikernel.py's
make_multi_kernel_plan: layout bijection (does every kernel boundary's
AddressMapping visit each of its own DRAM addresses exactly once, pure
index arithmetic, no floats), boundary consistency (does the large-twiddle
table fill agree with the store address formula it's fetched through --
these two silently drifted apart once already in this codebase's history,
see verify_boundary_consistency's own docstring), and full numeric
re-execution vs. numpy.
"""

from codegen.fft_codegen import _mapping_base_expr
from planning.core.fft_plan_core import MultiKernelFFTPlan
from planning.strategies.fft_plan_multikernel import make_multi_kernel_plan
from verification.verify_fft_harness import _run_multi_kernel_plan


def verify_layout_bijection(
    chunks: tuple[tuple[int, ...], ...], *, plan: MultiKernelFFTPlan | None = None
) -> None:
    """Index-only check (no FFT math, no floating point): every kernel's
    input_mapping/output_mapping visits each of that kernel's own DRAM
    addresses exactly once across all its uthreads' logical (row, elem)
    pairs -- i.e. it's a real permutation, not just a plausible-looking
    formula. Raises AssertionError on any collision or gap.

    Reuses fft_codegen._mapping_base_expr -- the exact expression codegen
    emits for the row-dependent part of an address -- instead of
    re-deriving the row*row_stride / SPLIT %-// arithmetic a second time
    here; only the already-documented `+ elem*elem_stride` (AddressMapping's
    own definition) is added on top.

    `plan`: pass an already-built MultiKernelFFTPlan (e.g. when the caller
    is about to also run verify_boundary_consistency on the same chunks)
    to skip rebuilding it here.
    """
    if plan is None:
        plan = make_multi_kernel_plan(chunks)
    n = plan.n
    for kernel in plan.kernels:
        for mapping, side in (
            (kernel.input_mapping, "input"),
            (kernel.output_mapping, "output"),
        ):
            expr = _mapping_base_expr(mapping, kernel.length)
            addrs: set[int] = set()
            for row in range(kernel.total_uthreads):
                base = eval(expr, {"global_uthread_id": lambda: row})  # noqa: B023
                for elem in range(kernel.length):
                    addr = base + elem * mapping.elem_stride
                    if addr in addrs:
                        raise AssertionError(
                            f"{kernel.kernel_name} {side} mapping: address "
                            f"{addr} visited more than once (chunks={chunks})"
                        )
                    addrs.add(addr)
            if addrs != set(range(n)):
                raise AssertionError(
                    f"{kernel.kernel_name} {side} mapping: addresses are not "
                    f"a permutation of 0..{n - 1} (chunks={chunks})"
                )


def verify_boundary_consistency(
    chunks: tuple[tuple[int, ...], ...], *, plan: MultiKernelFFTPlan | None = None
) -> None:
    """Each non-last kernel's large-twiddle table is filled by an
    *independent* second implementation of the address formula (host-side
    _make_large_twiddle_table, inverting addr -> (row, output) with plain
    numpy) from the one the kernel's own store uses to compute that same
    address forward (row, output) -> addr (fft_codegen._mapping_base_expr,
    the same formula every load/store in this kernel actually emits).
    This is exactly the pairing that broke once already this session (the
    fetch and the fill silently used two different address formulas after
    output_mapping changed from SPLIT to PEELED, and every FFT numeric
    check still ran -- it just produced garbage) -- so check it directly,
    for every (row, output) pair a kernel's own stage visits, rather than
    relying on a floating-point FFT mismatch to notice a mismatch here.

    `plan`: see verify_layout_bijection's own `plan` parameter.
    """
    if plan is None:
        plan = make_multi_kernel_plan(chunks)
    for kernel in plan.kernels:
        lt = kernel.large_twiddle
        if lt is None:
            continue
        expr = _mapping_base_expr(kernel.output_mapping, kernel.length)
        for row in range(kernel.total_uthreads):
            base = eval(expr, {"global_uthread_id": lambda: row})  # noqa: B023
            for output in range(kernel.length):
                addr = base + output * kernel.output_mapping.elem_stride
                # Independently invert addr -> (b, digit) the same way
                # _make_large_twiddle_table does, and check it recovers
                # exactly the (row, output) that produced this address.
                if lt.k_next:
                    d_next = addr % lt.k_next
                    combined_ao = (addr // lt.k_next) % (lt.a * lt.output_count)
                    rest = addr // (lt.k_next * lt.a * lt.output_count)
                    digit = combined_ao // lt.a
                    b = d_next * lt.tail_size + rest
                else:
                    a_ki = lt.a * lt.output_count
                    b = addr // a_ki
                    digit = (addr % a_ki) // lt.a
                expected_b = row // lt.a
                if b != expected_b or digit != output:
                    raise AssertionError(
                        f"{kernel.kernel_name}: twiddle table address {addr} "
                        f"(from row={row}, output={output}) inverts to "
                        f"(b={b}, digit={digit}), expected "
                        f"(b={expected_b}, digit={output}) (chunks={chunks})"
                    )



def verify_multi_kernel_plan(
    chunks: tuple[tuple[int, ...], ...],
    *,
    inverse: bool,
    seed: int,
    spad_capacity_bytes: int | None = None,
) -> float:
    """Stage 3/4's builder: like verify_decomposed_plan, but chunks[i] can
    itself be a multi-stage radix sequence (each kernel absorbing more than
    one stage via scratchpad ping-pong), not just a single bare radix, and
    M can be any length -- chains an arbitrary number of kernels through
    DRAM, one large-twiddle table per non-last kernel (see
    make_multi_kernel_plan for the general M-kernel formulas).

    `spad_capacity_bytes`, when given, forces `run_kernel` to actually
    exercise more than one NDP-unit group for a kernel whose total launch
    exceeds what one unit's scratchpad holds (see FFTCodegenPlan's
    max_uthread/total_uthreads split) rather than the always-one-group
    case every other test here happens to stay within.
    """
    plan = make_multi_kernel_plan(
        chunks, inverse=inverse, spad_capacity_bytes=spad_capacity_bytes
    )
    return _run_multi_kernel_plan(plan, inverse=inverse, seed=seed)

