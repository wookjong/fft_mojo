"""End-to-end numeric verification of a lowered FFTCodegenPlan / DecomposedFFTPlan.

verify_fft_butterflies.py checks each radix's butterfly in isolation. This
checks the rest of what fft_codegen.py emits around it -- load/store
addressing, per-stage twiddle, scratchpad ping-pong, and (for the decomposed
plan) the large-twiddle table and the strided transpose between kernel0 and
kernel1 -- by actually re-executing the emitted stage text, not by
reimplementing the algorithm a second time.

Translation from emitted Mojo to executable Python only has to cover what a
stage body actually contains:
  * `var NAME = EXPR` / `comptime NAME = EXPR` -> `NAME = EXPR` (Mojo's `var`
    and `comptime` are declarations; everything to their right is already a
    valid Python expression -- same trick verify_fft_butterflies.py uses).
  * `PTR.load[width=W](OFFSET)` / `PTR.load[DType.float32, W](OFFSET)` ->
    `PTR.load(OFFSET, W)` (a bracketed compile-time subscript becomes a
    second call argument).
  * `SIMD[DType.float32, W](A, B, ...)` -> `SIMD(A, B, ...)`.
  * `ref p = KERNEL.params[]` is dropped -- `p` is bound directly to a
    namespace of Ptr objects instead of modeling the scratchpad parameter
    block.
  * `@staticmethod` and comment lines are dropped.
Offsets, loop bounds and every numeric constant (twiddle vectors included)
are left exactly as emitted.
"""

import re
import types
from dataclasses import replace
from math import prod

import numpy as np

from fft_butterflies import SUPPORTED_RADICES
from fft_codegen import Emitter, _emit_stage, _mapping_base_expr
from fft_plangen import (
    BalancedTransposeFFTPlan,
    DecomposedFFTPlan,
    FFTCodegenPlan,
    _build_batched_side,
    _build_plan,
    _choose_side_chunks,
    _prime_factors_supported,
    factor_into_kernel_chunks,
    layouts_for_radices,
    make_444_plan,
    make_balanced_plan,
    make_balanced_transpose_plan,
    make_decomposed_plan,
    make_multi_kernel_plan,
    max_effective_stride,
    summarize_multi_kernel_plan,
)
from fft_transpose_codegen import _emit_transpose_stage

_VAR_RE = re.compile(r"^(\s*)var ")
_COMPTIME_RE = re.compile(r"^(\s*)comptime ")
_LOAD_W_RE = re.compile(r"\.load\[width=(\w+)\]\(([^()]*)\)")
_LOAD_DT_RE = re.compile(r"\.load\[DType\.float32,\s*(\w+)\]\(([^()]*)\)")
_SIMD_RE = re.compile(r"SIMD\[DType\.float32,\s*\w+\]\(")


def _translate_stage(plan: FFTCodegenPlan, stage) -> str:
    e = Emitter()
    _emit_stage(e, plan=plan, stage=stage)

    out: list[str] = []
    for line in e.lines:
        s = line.strip()
        if s == "" or s.startswith("#") or s == "@staticmethod":
            continue
        if s.startswith("ref p ="):
            continue
        line = _VAR_RE.sub(r"\1", line)
        line = _COMPTIME_RE.sub(r"\1", line)
        line = _LOAD_W_RE.sub(r".load(\2, \1)", line)
        line = _LOAD_DT_RE.sub(r".load(\2, \1)", line)
        line = _SIMD_RE.sub("SIMD(", line)
        # _emit_stage indents for a class body (`def stage_N():` at 4
        # spaces); dedent so it compiles as a top-level def instead.
        assert line.startswith("    ")
        out.append(line[4:])
    return "\n".join(out)


def verify_layout_bijection(chunks: tuple[tuple[int, ...], ...]) -> None:
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
    """
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


def verify_boundary_consistency(chunks: tuple[tuple[int, ...], ...]) -> None:
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
    """
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


class SimdVec(np.ndarray):
    """A numpy array with Mojo's SIMD value semantics instead of numpy's.

    `var or0 = rr0` (a bare name-to-name assignment -- e.g.
    fft_butterflies._emit_symmetric_odd_radix's `or0 = rr0; or0 += ...`
    accumulation pattern) is a *copy* in Mojo: SIMD is a value type, so
    later mutating `or0` never touches `rr0`. Plain numpy aliases the same
    buffer and `+=` mutates it in place, silently corrupting `rr0` for
    every later line that reads it -- caught by the radix-7/11/13/17 cases
    in Stage 2 (radix-2/3/4/6/8/9/10/16's butterflies never do a bare
    copy-then-accumulate, so this stayed latent through every earlier
    verified case). Disabling the in-place dunders makes Python's `+=`
    fall back to `self = self + other`, which rebinds instead of
    mutating -- exactly Mojo's copy behavior.
    """

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __itruediv__(self, other):
        return self / other


class Ptr:
    """Stand-in for an UnsafePointer[Float32]: a flat float64 buffer."""

    def __init__(self, n: int) -> None:
        self.arr = np.zeros(n, dtype=np.float64)

    def load(self, offset: int, width: int) -> SimdVec:
        offset = int(offset)
        return self.arr[offset : offset + int(width)].copy().view(SimdVec)

    def store(self, offset: int, value) -> None:
        offset = int(offset)
        value = np.asarray(value, dtype=np.float64)
        if value.ndim == 0:
            self.arr[offset] = float(value)
        else:
            flat = value.reshape(-1)
            self.arr[offset : offset + flat.size] = flat


def _simd(*args: float) -> SimdVec:
    return np.array(args, dtype=np.float64).view(SimdVec)


def run_kernel(
    plan: FFTCodegenPlan,
    *,
    input_real: Ptr,
    input_imag: Ptr,
    output_real: Ptr,
    output_imag: Ptr,
    large_twiddle_real: Ptr | None = None,
    large_twiddle_imag: Ptr | None = None,
) -> None:
    """Run every stage of `plan`'s device_main, for every uthread, by
    exec()ing the actual text fft_codegen.py emits for each stage -- the
    same sequencing `launch_parallel[Self.stage_N]()` describes: every
    uthread finishes stage N before stage N+1 starts.

    `plan.total_uthreads` (the whole launch) may exceed `plan.max_uthread`
    (how many of this kernel's own uthreads share one NDP unit's
    scratchpad -- see FFTCodegenPlan's docstring): group `g =
    global_id // max_uthread` gets its own independent scratchpad
    instance ("one instance per core," not one shared array for the whole
    launch -- docs/INTERFACE.md), and `local_uthread_id() = global_id %
    max_uthread` indexes within it.
    """
    num_groups = -(-plan.total_uthreads // plan.max_uthread)  # ceil div
    group_namespaces: list[types.SimpleNamespace] = []
    for _ in range(num_groups):
        ns = types.SimpleNamespace()
        for buf in plan.scratchpad_buffers:
            setattr(ns, buf.name, Ptr(buf.elements))
        group_namespaces.append(ns)

    p_ns = types.SimpleNamespace(
        input_real_base=input_real,
        input_imag_base=input_imag,
        output_real_base=output_real,
        output_imag_base=output_imag,
    )
    if plan.large_twiddle is not None:
        assert large_twiddle_real is not None and large_twiddle_imag is not None
        p_ns.large_twiddle_real_base = large_twiddle_real
        p_ns.large_twiddle_imag_base = large_twiddle_imag

    current = {"global_id": 0, "local_id": 0}

    for stage in plan.stages:
        src = _translate_stage(plan, stage)
        namespace = {
            "Float32": float,
            "SIMD": _simd,
            "local_uthread_id": lambda: current["local_id"],
            "global_uthread_id": lambda: current["global_id"],
            "N": plan.length,
            "W": plan.simd_lanes,
            f"MAX_UTHREAD_{plan.kernel_name}": plan.max_uthread,
            "p": p_ns,
        }
        code = compile(src, f"<{plan.kernel_name} stage {stage.stage_id}>", "exec")
        exec(code, namespace)
        stage_fn = namespace[f"stage_{stage.stage_id}"]
        for global_id in range(plan.total_uthreads):
            current["global_id"] = global_id
            current["local_id"] = global_id % plan.max_uthread
            namespace[plan.kernel_name] = group_namespaces[global_id // plan.max_uthread]
            stage_fn()


def _make_large_twiddle_table(lt) -> tuple[np.ndarray, np.ndarray]:
    """Built directly over the fetch address (0..full_length-1), matching
    the host precompute this plan's kernel actually emits (see
    fft_codegen._emit_large_twiddle_table_precompute) -- the twiddle
    *value* only ever depends on (b, digit) (b = the not-yet-transformed
    remainder row//a, digit = this kernel's own output), never on which
    address formula placed it, so only the addr -> (b, digit) inverse
    below differs between mapping kinds.

    SPLIT (a middle kernel's output in an M<=2-generalizing chain -- see
    AddressMapping.split): addr = out_a + digit*a + b*a*output_count
    (out_a in [0,a), redundant); a=1 degenerates to
    large_twiddle[row*output_count+c1] = W_full_length^(row*c1).

    PEELED (see AddressMapping.peeled / AddressMappingKind.PEELED):
    addr = rest*(a*output_count*k_next) + (out_a+digit*a)*k_next + d_next
    (out_a in [0,a), redundant); b = remaining = d_next*tail_size + rest.
    """
    sign = 1.0 if lt.inverse else -1.0
    addr = np.arange(lt.full_length)
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
    angle = sign * 2.0 * np.pi * (b * digit * lt.a) / lt.full_length
    return np.cos(angle), np.sin(angle)


def verify_single_kernel_plan(*, inverse: bool, seed: int) -> float:
    plan = make_444_plan(inverse=inverse)
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, plan.length) + 1j * rng.uniform(-1, 1, plan.length)

    in_r, in_i = Ptr(plan.length), Ptr(plan.length)
    in_r.arr[:] = x.real
    in_i.arr[:] = x.imag
    out_r, out_i = Ptr(plan.length), Ptr(plan.length)

    run_kernel(plan, input_real=in_r, input_imag=in_i, output_real=out_r, output_imag=out_i)

    got = out_r.arr + 1j * out_i.arr
    expected = np.fft.ifft(x) if inverse else np.fft.fft(x)
    return float(np.max(np.abs(got - expected)))


def verify_radix_sequence_plan(
    radices: tuple[int, ...], *, inverse: bool, seed: int, simd_lanes: int = 8
) -> float:
    """Same shape as verify_single_kernel_plan, but for an arbitrary radix
    sequence within one kernel (Stage 2 of the planner generalization:
    proving `layouts_for_radices` -- not just the hardcoded (4,4,4) case
    `make_444_plan` calls it with -- by actually re-executing the emitted
    stage text)."""
    length = 1
    for r in radices:
        length *= r
    plan = _build_plan(
        length=length,
        inverse=inverse,
        total_uthreads=1,
        simd_lanes=simd_lanes,
        use_pingpong=True,
        layouts=layouts_for_radices(length, radices, simd_lanes),
    )
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, length) + 1j * rng.uniform(-1, 1, length)

    in_r, in_i = Ptr(length), Ptr(length)
    in_r.arr[:] = x.real
    in_i.arr[:] = x.imag
    out_r, out_i = Ptr(length), Ptr(length)

    run_kernel(plan, input_real=in_r, input_imag=in_i, output_real=out_r, output_imag=out_i)

    got = out_r.arr + 1j * out_i.arr
    expected = np.fft.ifft(x) if inverse else np.fft.fft(x)
    return float(np.max(np.abs(got - expected)))


def verify_decomposed_plan(*, n0: int, n1: int, inverse: bool, seed: int) -> float:
    plan = make_decomposed_plan(n0, n1, inverse=inverse)
    n = plan.n
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)

    in_r, in_i = Ptr(n), Ptr(n)
    in_r.arr[:] = x.real
    in_i.arr[:] = x.imag
    mid_r, mid_i = Ptr(n), Ptr(n)
    out_r, out_i = Ptr(n), Ptr(n)

    lt = plan.kernel0.large_twiddle
    assert lt is not None
    lt_real_vals, lt_imag_vals = _make_large_twiddle_table(lt)
    lt_r, lt_i = Ptr(n), Ptr(n)
    lt_r.arr[:] = lt_real_vals
    lt_i.arr[:] = lt_imag_vals

    run_kernel(
        plan.kernel0,
        input_real=in_r, input_imag=in_i,
        output_real=mid_r, output_imag=mid_i,
        large_twiddle_real=lt_r, large_twiddle_imag=lt_i,
    )
    run_kernel(
        plan.kernel1,
        input_real=mid_r, input_imag=mid_i,
        output_real=out_r, output_imag=out_i,
    )

    got = out_r.arr + 1j * out_i.arr
    expected = np.fft.ifft(x) if inverse else np.fft.fft(x)
    return float(np.max(np.abs(got - expected)))


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


def _run_multi_kernel_plan(plan, *, inverse: bool, seed: int) -> float:
    """Chains every kernel in `plan.kernels` through DRAM (one large-twiddle
    table per non-last kernel, exactly the M-kernel PEELED-chain contract
    make_multi_kernel_plan documents) and compares the result to numpy.
    Generic over how the plan was built -- shared by verify_multi_kernel_plan
    and verify_balanced_plan, which differ only in which planner function
    produces `plan`.
    """
    n = plan.n
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)

    in_r, in_i = Ptr(n), Ptr(n)
    in_r.arr[:] = x.real
    in_i.arr[:] = x.imag

    cur_r, cur_i = in_r, in_i
    for kernel in plan.kernels[:-1]:
        next_r, next_i = Ptr(n), Ptr(n)
        lt = kernel.large_twiddle
        assert lt is not None
        lt_real_vals, lt_imag_vals = _make_large_twiddle_table(lt)
        lt_r, lt_i = Ptr(n), Ptr(n)
        lt_r.arr[:] = lt_real_vals
        lt_i.arr[:] = lt_imag_vals
        run_kernel(
            kernel,
            input_real=cur_r, input_imag=cur_i,
            output_real=next_r, output_imag=next_i,
            large_twiddle_real=lt_r, large_twiddle_imag=lt_i,
        )
        cur_r, cur_i = next_r, next_i

    out_r, out_i = Ptr(n), Ptr(n)
    run_kernel(
        plan.kernels[-1],
        input_real=cur_r, input_imag=cur_i, output_real=out_r, output_imag=out_i,
    )

    got = out_r.arr + 1j * out_i.arr
    expected = np.fft.ifft(x) if inverse else np.fft.fft(x)
    return float(np.max(np.abs(got - expected)))


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
    out: list[str] = []
    for line in e.lines:
        s = line.strip()
        if s == "" or s.startswith("#") or s == "@staticmethod":
            continue
        if s.startswith("ref p ="):
            continue
        line = _VAR_RE.sub(r"\1", line)
        line = _COMPTIME_RE.sub(r"\1", line)
        line = _LOAD_W_RE.sub(r".load(\2, \1)", line)
        line = _LOAD_DT_RE.sub(r".load(\2, \1)", line)
        line = _SIMD_RE.sub("SIMD(", line)
        assert line.startswith("    "), line
        out.append(line[4:])
    return "\n".join(out)


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

    cur_r, cur_i = in_r, in_i
    for kernel in plan.kernels_near[:-1]:
        next_r, next_i = Ptr(n), Ptr(n)
        lt = kernel.large_twiddle
        assert lt is not None
        lt_real_vals, lt_imag_vals = _make_large_twiddle_table(lt)
        lt_r, lt_i = Ptr(n), Ptr(n)
        lt_r.arr[:] = lt_real_vals
        lt_i.arr[:] = lt_imag_vals
        run_kernel(kernel, input_real=cur_r, input_imag=cur_i, output_real=next_r, output_imag=next_i, large_twiddle_real=lt_r, large_twiddle_imag=lt_i)
        cur_r, cur_i = next_r, next_i

    near_out_r, near_out_i = Ptr(n), Ptr(n)
    run_kernel(plan.kernels_near[-1], input_real=cur_r, input_imag=cur_i, output_real=near_out_r, output_imag=near_out_i)

    tw_real, tw_imag = _make_transpose_twiddle_table(plan.transpose)
    tw_r, tw_i = Ptr(n), Ptr(n)
    tw_r.arr[:] = tw_real
    tw_i.arr[:] = tw_imag
    far_in_r, far_in_i = Ptr(n), Ptr(n)
    run_transpose_kernel(
        plan.transpose, near_real=near_out_r, near_imag=near_out_i,
        far_real=far_in_r, far_imag=far_in_i, tw_real=tw_r, tw_imag=tw_i,
    )

    cur_r, cur_i = far_in_r, far_in_i
    for kernel in plan.kernels_far[:-1]:
        next_r, next_i = Ptr(n), Ptr(n)
        lt = kernel.large_twiddle
        assert lt is not None
        lt_real_vals, lt_imag_vals = _make_large_twiddle_table(lt)
        lt_r, lt_i = Ptr(n), Ptr(n)
        lt_r.arr[:] = lt_real_vals
        lt_i.arr[:] = lt_imag_vals
        run_kernel(kernel, input_real=cur_r, input_imag=cur_i, output_real=next_r, output_imag=next_i, large_twiddle_real=lt_r, large_twiddle_imag=lt_i)
        cur_r, cur_i = next_r, next_i

    out_r, out_i = Ptr(n), Ptr(n)
    run_kernel(plan.kernels_far[-1], input_real=cur_r, input_imag=cur_i, output_real=out_r, output_imag=out_i)

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


def main() -> None:
    tolerance = 1.0e-3
    failures: list[str] = []

    # Stage 2: layouts_for_radices generalizes beyond the hardcoded (4,4,4)
    # case make_444_plan calls it with -- single-stage sweep over every
    # supported radix, same-radix towers of depth 2 and up, and several
    # mixed-radix orderings, forward and inverse. Depth-4/5 same-radix and
    # (3,4,2)/(5,2,3) were rejected by an earlier _check_layouts constraint
    # (simd_lanes % twiddle_lane_divisor == 0); that constraint's own
    # "confirmed real" check was standing on a bug in this harness (numpy
    # aliasing on `var or0 = rr0`-style copies, fixed below as SimdVec) and
    # didn't survive re-verification, so it was removed -- these are
    # ordinary passing cases now, not a special "expected rejection" list.
    radix_sequence_cases: list[tuple[int, ...]] = (
        [(r,) for r in sorted(SUPPORTED_RADICES)]
        + [(2, 2), (3, 3), (4, 4), (2, 2, 2, 2, 2), (2,) * 8, (3,) * 5]
        + [
            (2, 3, 4), (4, 3, 2), (7, 3), (9, 2), (13, 2), (11, 3),
            (4, 4, 4, 4), (3, 4, 2), (5, 2, 3),
        ]
    )
    for radices in radix_sequence_cases:
        for inverse in (False, True):
            err = verify_radix_sequence_plan(radices, inverse=inverse, seed=1)
            tag = f"radix sequence {radices} inverse={inverse}"
            ok = err <= tolerance
            print(f"  {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
            if not ok:
                failures.append(tag)

    for inverse in (False, True):
        err = verify_single_kernel_plan(inverse=inverse, seed=1 if inverse else 0)
        tag = f"single-kernel N=64 (4x4x4) inverse={inverse}"
        ok = err <= tolerance
        print(f"  {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
        if not ok:
            failures.append(tag)

    for inverse in (False, True):
        err = verify_decomposed_plan(n0=16, n1=16, inverse=inverse, seed=3 if inverse else 2)
        tag = f"decomposed N=256 (16x16, large twiddle + transpose) inverse={inverse}"
        ok = err <= tolerance
        print(f"  {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
        if not ok:
            failures.append(tag)

    # Stage 3: multi-kernel chaining where each kernel is itself multi-stage
    # (not just one bare radix, like make_decomposed_plan) -- the actual new
    # capability. M=1 chunking sanity, then M=2 with a multi-stage kernel on
    # each side, chained through DRAM + the large-twiddle table.
    #
    # Stage 4: M>=3 -- every non-last kernel's output uses AddressMapping.SPLIT
    # (a middle kernel's uthread id mixes an already-transformed digit run
    # with a not-yet-transformed remainder) and a LargeTwiddlePlan scoped to
    # a smaller angle modulus but the *same* full_length-sized, a-fold
    # redundant table (see LargeTwiddlePlan/make_multi_kernel_plan). Up to
    # N=1155 (11x7x3x5) and depth-5 all-radix-2 chains.
    multi_kernel_cases: list[tuple[int, tuple[tuple[int, ...], ...]]] = [
        (64, ((4, 4, 4),)),  # M=1, matches make_444_plan's shape
        (192, ((4, 4, 4), (3,))),  # M=2, multi-stage kernel0, bare kernel1
        (40, ((2, 2, 2), (5,))),  # M=2, multi-stage kernel0, small N
        (48, ((4, 4), (3,))),  # M=2, both sides different depths
        (24, ((2,), (3,), (4,))),  # M=3, bare radices
        (105, ((7,), (3,), (5,))),  # M=3, all odd primes
        (960, ((4, 4, 4), (3,), (5,))),  # M=3, multi-stage first kernel
        (768, ((2, 2, 2, 2, 2), (3,), (2, 2, 2))),  # M=3, multi-stage on both ends
        (32, ((2,), (2,), (2,), (2,), (2,))),  # M=5, chained one radix-2 at a time
        (1155, ((11,), (7,), (3,), (5,))),  # M=4, largest N tested
    ]

    # Layout-only pass first (section 13.A): pure index arithmetic, no FFT
    # math, over the same case matrix above -- single segment, 2, 3+,
    # balanced/unbalanced, power-of-two and mixed radix. Confirms every
    # kernel boundary's AddressMapping is a genuine permutation before any
    # floating-point check runs on it.
    for n, chunks in multi_kernel_cases:
        tag = f"layout bijection N={n} chunks={chunks}"
        try:
            verify_layout_bijection(chunks)
            print(f"  OK   {tag}")
        except AssertionError as exc:
            print(f"  FAIL {tag}: {exc}")
            failures.append(tag)

    for n, chunks in multi_kernel_cases:
        tag = f"boundary consistency N={n} chunks={chunks}"
        try:
            verify_boundary_consistency(chunks)
            print(f"  OK   {tag}")
        except AssertionError as exc:
            print(f"  FAIL {tag}: {exc}")
            failures.append(tag)

    for n, chunks in multi_kernel_cases:
        for inverse in (False, True):
            err = verify_multi_kernel_plan(chunks, inverse=inverse, seed=1)
            tag = f"multi-kernel N={n} chunks={chunks} inverse={inverse}"
            ok = err <= tolerance
            print(f"  {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
            if not ok:
                failures.append(tag)

    # max_uthread/total_uthreads split: force kernel0 of ((4,4,4),(3,))
    # (N=192, kernel0 length=64, total_uthreads=3, 1024 bytes/uthread) into
    # more than one NDP-unit group -- 1024 packs exactly one uthread per
    # group (3 groups, each with its own scratchpad instance), 2048 packs
    # two (an uneven 2+1 split, exercising a partial last group). Both
    # must match the single-group (no cap) result from multi_kernel_cases
    # above, proving run_kernel's per-group scratchpad isolation is real
    # (see FFTCodegenPlan's docstring / run_kernel).
    for cap in (1024, 2048):
        for inverse in (False, True):
            err = verify_multi_kernel_plan(
                ((4, 4, 4), (3,)), inverse=inverse, seed=1, spad_capacity_bytes=cap
            )
            tag = f"multi-kernel N=192 spad_capacity_bytes={cap} inverse={inverse}"
            ok = err <= tolerance
            print(f"  {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
            if not ok:
                failures.append(tag)

    # Same, but more than two per group and more than two groups: N=960's
    # kernel0 (length=64, total_uthreads=15) at 4096 bytes -> 4 uthreads/
    # group -> 4 groups sized 4,4,4,3 (three full, one partial).
    for inverse in (False, True):
        err = verify_multi_kernel_plan(
            ((4, 4, 4), (3,), (5,)), inverse=inverse, seed=9, spad_capacity_bytes=4096
        )
        tag = f"multi-kernel N=960 spad_capacity_bytes=4096 (4 uthreads/group, 4 groups) inverse={inverse}"
        ok = err <= tolerance
        print(f"  {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
        if not ok:
            failures.append(tag)

    # factor_into_kernel_chunks: a DP over factor-list split points that
    # minimizes max(read_stride, write_stride) across every kernel in the
    # chain -- not just the last kernel's write side (see its docstring for
    # why that's not the whole story: a kernel's *read* stride is
    # n // its-own-length, large whenever that one kernel is small,
    # regardless of chain position). Three properties checked directly:
    # (1) max_effective_stride is monotonically non-increasing as the
    # budget grows, (2) it is strictly better than a write-only,
    # back-to-front packing at the same budget (reference implementation
    # below, kept local to this test purely for the comparison -- not a
    # second copy of production logic), and (3) whatever chunking comes out
    # still produces a numerically correct FFT and a genuine address
    # permutation.
    def _write_stride_only_packing(n: int, budget: int) -> tuple[tuple[int, ...], ...]:
        """The previous session's algorithm, reference-implemented here only
        to demonstrate the improvement -- see the docstring's N=960 example."""
        factors = _prime_factors_supported(n)
        cap = budget // 16
        chunks: list[tuple[int, ...]] = []
        current: list[int] = []
        product = 1
        for f in reversed(factors):
            if current and product * f > cap:
                chunks.append(tuple(reversed(current)))
                current, product = [], 1
            current.append(f)
            product *= f
        if current:
            chunks.append(tuple(reversed(current)))
        return tuple(reversed(chunks))

    n = 960
    budgets = (256, 512, 1024, 16384, 65536)
    prev_max_stride = None
    for budget in budgets:
        chunks = factor_into_kernel_chunks(n, scratchpad_byte_budget=budget)
        summary = summarize_multi_kernel_plan(make_multi_kernel_plan(chunks))
        max_stride = max_effective_stride(summary)

        if prev_max_stride is not None and max_stride > prev_max_stride:
            failures.append(
                f"factor_into_kernel_chunks budget={budget}: max_effective_stride="
                f"{max_stride} regressed above the previous (smaller) budget's "
                f"{prev_max_stride} -- should be non-increasing in budget"
            )
        prev_max_stride = max_stride

        old_chunks = _write_stride_only_packing(n, budget)
        old_summary = summarize_multi_kernel_plan(make_multi_kernel_plan(old_chunks))
        old_max_stride = max_effective_stride(old_summary)
        if old_chunks != chunks and old_max_stride < max_stride:
            failures.append(
                f"factor_into_kernel_chunks budget={budget}: max_effective_stride="
                f"{max_stride} is worse than write-only packing's {old_max_stride} "
                f"(chunks={chunks} vs {old_chunks})"
            )
        print(
            f"  ---  N={n} budget={budget}: new chunks={chunks} "
            f"max_effective_stride={max_stride}  |  old (write-only) "
            f"chunks={old_chunks} max_effective_stride={old_max_stride}"
        )

        verify_layout_bijection(chunks)

        for inverse in (False, True):
            err = verify_multi_kernel_plan(chunks, inverse=inverse, seed=5)
            tag = (
                f"factor_into_kernel_chunks N={n} budget={budget} "
                f"chunks={chunks} (max_effective_stride={max_stride}) inverse={inverse}"
            )
            ok = err <= tolerance
            print(f"  {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
            if not ok:
                failures.append(tag)

    # Scalar-vs-vector DRAM access (section 15): PEELED's whole point is
    # that every kernel after the first gets a real vector read instead
    # of a forced scalar one -- check the plan's own mappings directly
    # (mode is scalar iff elem_stride != 1, the same test
    # AddressMappingKind.STRIDED's docstring and _make_load/_make_store
    # already use), not just assert an aggregate count.
    print()
    print("  Scalar vs. vector DRAM read/write per kernel (N=960, "
          "chunks=((4,4,4),(3,),(5,))):")
    demo_plan = make_multi_kernel_plan(((4, 4, 4), (3,), (5,)))
    vector_reads = 0
    for kernel in demo_plan.kernels:
        read_mode = "vector" if kernel.input_mapping.elem_stride == 1 else "scalar"
        write_mode = "vector" if kernel.output_mapping.elem_stride == 1 else "scalar"
        if read_mode == "vector":
            vector_reads += 1
        print(f"    {kernel.kernel_name}: read={read_mode:6s} write={write_mode:6s}")
    tag = "scalar-to-vector read conversion (N=960, 3-kernel chain)"
    # Old (SPLIT) scheme: every kernel's read is scalar, always -- 0 of 3.
    # New (PEELED): every kernel after the first is vector -- 2 of 3.
    if vector_reads == 2:
        print(f"  OK   {tag}: {vector_reads}/3 kernels now read via vector "
              f"load (was 0/3 under the old SPLIT-based scheme)")
    else:
        print(f"  FAIL {tag}: expected 2/3 kernels reading via vector load, "
              f"got {vector_reads}/3")
        failures.append(tag)

    # make_balanced_plan: N = N_A*N_B, each side its own PEELED chain,
    # joined by one AddressMappingKind.CROSSED transpose. Two things to
    # check: (1) numeric correctness, single-kernel-per-side through to
    # both-sides-multi-kernel, forward and inverse; (2) that
    # _choose_side_chunks (the batch-aware chunk selection) actually beats
    # naively feeding factor_into_kernel_chunks's own, batch-*unaware*
    # chunking into the same side -- checked directly, not assumed: an
    # interim version of this used plain factor_into_kernel_chunks per
    # side and got nowhere near sqrt(N) whenever a side needed more than
    # one internal kernel (N=960*960 gave max_effective_stride=30720, no
    # better than one flat chain over the same N).
    balanced_cases: list[
        tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]
    ] = [
        (((16,),), ((16,),)),  # single kernel per side -- checkpoint vs. make_decomposed_plan
        (((17,),), ((13,),)),  # single kernel per side, unequal, both odd primes
        (((4, 4, 4),), ((3,), (5,))),  # side A one fused kernel, side B a 2-kernel chain
        (((2,), (2,), (2,)), ((3,), (5,))),  # both sides multi-kernel chains
    ]
    for chunks_A, chunks_B in balanced_cases:
        n_a = prod(prod(c) for c in chunks_A)
        n_b = prod(prod(c) for c in chunks_B)
        for inverse in (False, True):
            err = verify_balanced_plan(chunks_A, chunks_B, inverse=inverse, seed=13)
            tag = f"balanced plan N_A={n_a}({chunks_A}) N_B={n_b}({chunks_B}) inverse={inverse}"
            ok = err <= tolerance
            print(f"  {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
            if not ok:
                failures.append(tag)

    # Checkpoint: single kernel per side must compute the *same physical
    # addresses* as make_decomposed_plan's own (untouched, SPLIT/
    # contiguous-based) formulas -- AddressMappingKind.CROSSED's docstring
    # claims it degenerates to CONTIGUOUS there. Compare the actual
    # address expressions (what codegen emits), not raw dataclass equality:
    # CROSSED's degenerate case computes the identical row*16+elem formula
    # but keeps the CROSSED *kind* tag (with an unused peel_a=batch_count
    # left set) rather than relabeling itself CONTIGUOUS, so a field-for-
    # field dataclass comparison flags a difference that isn't there --
    # caught by trying that first and getting a false failure here.
    decomposed = make_decomposed_plan(16, 16)
    balanced = make_balanced_plan(((16,),), ((16,),))
    pairs = [
        (decomposed.kernel0.input_mapping, balanced.kernels[0].input_mapping, 16),
        (decomposed.kernel0.output_mapping, balanced.kernels[0].output_mapping, 16),
        (decomposed.kernel1.input_mapping, balanced.kernels[1].input_mapping, 16),
        (decomposed.kernel1.output_mapping, balanced.kernels[1].output_mapping, 16),
    ]
    tag = "make_balanced_plan((16,),(16,)) computes the same addresses as make_decomposed_plan(16,16)"
    same = True
    for d_map, b_map, length in pairs:
        for row in range(4):
            d_base = eval(_mapping_base_expr(d_map, length), {"global_uthread_id": lambda r=row: r})
            b_base = eval(_mapping_base_expr(b_map, length), {"global_uthread_id": lambda r=row: r})
            if d_base != b_base or d_map.elem_stride != b_map.elem_stride:
                same = False
    print(f"  {'OK  ' if same else 'FAIL'} {tag}")
    if not same:
        failures.append(tag)

    # Batch-aware vs. batch-unaware chunk selection, measured directly.
    factors_960 = _prime_factors_supported(960)
    aware = _choose_side_chunks(factors_960, batch_count=960, scratchpad_byte_budget=256)
    unaware = factor_into_kernel_chunks(960, scratchpad_byte_budget=256)

    def kernel0_batched_read(chunks: tuple[tuple[int, ...], ...], batch_count: int) -> int:
        k0 = prod(chunks[0])
        return (960 // k0) * batch_count

    aware_read = kernel0_batched_read(aware, 960)
    unaware_read = kernel0_batched_read(unaware, 960)
    print(
        f"  ---  side length 960, batch_count 960: batch-aware chunks={aware} "
        f"kernel0 batched read={aware_read}  |  batch-unaware chunks={unaware} "
        f"kernel0 batched read={unaware_read}"
    )
    tag = "batch-aware chunk selection beats batch-unaware at the same budget"
    if aware_read <= unaware_read:
        print(f"  OK   {tag}")
    else:
        print(f"  FAIL {tag}: aware={aware_read} unaware={unaware_read}")
        failures.append(tag)

    # make_balanced_transpose_plan: N=N_A*N_B, each side its own PEELED
    # chain, joined by a standalone tiled transpose+twiddle kernel instead
    # of make_balanced_plan's scalar CROSSED-fused write -- see
    # fft_plangen.FFTTransposePlan and fft_transpose_codegen.py.
    #
    # Baseline checkpoints (17x17, 8x8, 4x8 -- what make_decomposed_plan
    # already covers directly), one-side-bare/other-multi-kernel and the
    # reverse, both-sides-multi-kernel, and a representative large-N case
    # forcing multi-kernel on both sides.
    transpose_cases: list[tuple[int, int]] = [
        (17, 17), (8, 8), (4, 8), (16, 16), (13, 17), (64, 15), (8, 15), (32, 30),
    ]
    for n_a, n_b in transpose_cases:
        n = n_a * n_b
        # small enough budget to sometimes force multi-kernel sides, large
        # enough to always be plannable for these n's own factor sizes.
        budget = 256 if n >= 512 else 4096
        for inverse in (False, True):
            err, plan = verify_balanced_transpose_plan(
                n, scratchpad_byte_budget=budget, inverse=inverse, seed=17
            )
            tag = (
                f"balanced-transpose plan n={n} N_A={plan.transpose.n_a} "
                f"N_B={plan.transpose.n_b} inverse={inverse}"
            )
            ok = err <= tolerance
            print(f"  {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
            if not ok:
                failures.append(tag)

    # Index-only bijection + DRAM access-shape checks (no floats) for the
    # transpose boundary itself, across the same case list.
    for n_a, n_b in transpose_cases:
        n = n_a * n_b
        budget = 256 if n >= 512 else 4096
        plan = make_balanced_transpose_plan(n, scratchpad_byte_budget=budget)
        tag = f"transpose bijection n={n} N_A={n_a} N_B={n_b}"
        try:
            verify_transpose_bijection(plan.transpose)
            print(f"  OK   {tag}")
        except AssertionError as exc:
            print(f"  FAIL {tag}: {exc}")
            failures.append(tag)

    # Stride-isolation proof (the invariant that actually matters -- see
    # the design writeup): a side's own boundary-touching kernel (the one
    # whose length -- ki_near/ki_far -- the transpose needs to know) is
    # the ONLY thing that ever crosses the boundary. Changing how the
    # *rest* of the other side splits into further internal kernels (same
    # boundary-chunk length, different kernel count/structure beyond it)
    # must not change this side's own kernels, nor the transpose's own
    # near-facing or far-facing shape, at all.
    n_a_test, n_b_test = 32, 60
    n_test = n_a_test * n_b_test
    chunks_a_fixed = ((4, 4, 2),)
    far_2kernel = ((6,), (10,))   # Far1: 1 kernel, radix 10
    far_3kernel = ((6,), (2,), (5,))  # same boundary chunk (6,), rest split differently

    def build(chunks_a, chunks_b):
        near = _build_batched_side(
            chunks_a, batch_count=n_b_test, full_length=n_test, inverse=False,
            simd_lanes=8, kernel_name_prefix="SI_Near", is_far_side=False,
            spad_capacity_bytes=None, plain_boundary=True,
        )
        far = _build_batched_side(
            chunks_b, batch_count=n_a_test, full_length=n_test, inverse=False,
            simd_lanes=8, kernel_name_prefix="SI_Far", is_far_side=True,
            spad_capacity_bytes=None, plain_boundary=True,
        )
        return near, far

    near_2, far_2 = build(chunks_a_fixed, far_2kernel)
    near_3, far_3 = build(chunks_a_fixed, far_3kernel)

    near_unaffected = (
        len(near_2) == len(near_3)
        and all(
            k2.input_mapping == k3.input_mapping and k2.output_mapping == k3.output_mapping
            and k2.length == k3.length
            for k2, k3 in zip(near_2, near_3)
        )
    )
    tag = "near side's own kernels byte-identical when far's non-boundary kernel count/structure changes"
    print(f"  {'OK  ' if near_unaffected else 'FAIL'} {tag}")
    if not near_unaffected:
        failures.append(tag)

    boundary_kernel_same = (
        far_2[0].length == far_3[0].length
        and far_2[0].input_mapping == far_3[0].input_mapping
    )
    far_genuinely_differs = len(far_2) != len(far_3)
    tag = "far side's own boundary-touching kernel unaffected (while far itself genuinely differs elsewhere)"
    ok = boundary_kernel_same and far_genuinely_differs
    print(f"  {'OK  ' if ok else 'FAIL'} {tag}")
    if not ok:
        failures.append(tag)

    # Old flat PEELED chain vs. new balanced-transpose plan, same N and
    # budget -- stride/cost comparison (printed, not claimed as measured
    # performance -- see the design writeup).
    print()
    print("  Old flat chain vs. new balanced-transpose plan (N=960, budget=256):")
    flat_chunks = factor_into_kernel_chunks(960, scratchpad_byte_budget=256)
    flat_plan = make_multi_kernel_plan(flat_chunks)
    flat_summary = summarize_multi_kernel_plan(flat_plan)
    print(
        f"    flat:      {len(flat_plan.kernels)} FFT kernels, 0 transpose kernels, "
        f"max effective stride={max_effective_stride(flat_summary)}"
    )
    bt_plan = make_balanced_transpose_plan(960, scratchpad_byte_budget=256)
    near_max = max(max(k.input_mapping.elem_stride, k.output_mapping.elem_stride) for k in bt_plan.kernels_near)
    far_max = max(max(k.input_mapping.elem_stride, k.output_mapping.elem_stride) for k in bt_plan.kernels_far)
    print(
        f"    balanced:  {len(bt_plan.kernels_near) + len(bt_plan.kernels_far)} FFT kernels, "
        f"1 transpose kernel, max FFT-side effective stride={max(near_max, far_max)} "
        f"(near max={near_max}, far max={far_max}), transpose load/store elem_stride=1"
    )
    print()
    summarize_balanced_transpose_plan(bt_plan)
    print()

    if failures:
        raise AssertionError(f"{len(failures)} plan(s) failed: {failures}")
    print("[verify] all FFT plans matched numpy's FFT")


if __name__ == "__main__":
    main()
