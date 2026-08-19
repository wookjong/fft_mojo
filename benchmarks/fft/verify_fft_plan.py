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

import numpy as np

from fft_butterflies import SUPPORTED_RADICES
from fft_codegen import Emitter, _emit_stage
from fft_plangen import (
    DecomposedFFTPlan,
    FFTCodegenPlan,
    _build_plan,
    layouts_for_radices,
    make_444_plan,
    make_decomposed_plan,
)

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
    """
    kernel_ns = types.SimpleNamespace()
    for buf in plan.scratchpad_buffers:
        setattr(kernel_ns, buf.name, Ptr(buf.elements))

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

    current_uid = {"id": 0}

    for stage in plan.stages:
        src = _translate_stage(plan, stage)
        namespace = {
            "Float32": float,
            "SIMD": _simd,
            "local_uthread_id": lambda: current_uid["id"],
            "global_uthread_id": lambda: current_uid["id"],
            "N": plan.length,
            "W": plan.simd_lanes,
            f"MAX_UTHREAD_{plan.kernel_name}": plan.max_uthread,
            plan.kernel_name: kernel_ns,
            "p": p_ns,
        }
        code = compile(src, f"<{plan.kernel_name} stage {stage.stage_id}>", "exec")
        exec(code, namespace)
        stage_fn = namespace[f"stage_{stage.stage_id}"]
        for uid in range(plan.max_uthread):
            current_uid["id"] = uid
            stage_fn()


def _make_large_twiddle_table(lt) -> tuple[np.ndarray, np.ndarray]:
    """Matches generate_decomposed_fft_kernels' host precompute exactly:
    large_twiddle[row*output_count + c1] = W_full_length^(row*c1)."""
    sign = 1.0 if lt.inverse else -1.0
    row = np.arange(lt.row_count)
    col = np.arange(lt.output_count)
    angle = sign * 2.0 * np.pi * np.outer(row, col) / lt.full_length
    return np.cos(angle).reshape(-1), np.sin(angle).reshape(-1)


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
        max_uthread=1,
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


# Non-last stages that are *expected* to be rejected: _check_layouts
# requires simd_lanes % twiddle_lane_divisor == 0 where twiddle_lane_divisor
# is P_s, the product of radices before that stage (see layouts_for_radices).
# Confirmed real (not overly conservative) by bypassing the check once and
# watching the numeric result go from ~1e-9 to ~O(1) wrong -- see the plan
# doc's Stage 2 notes. Every entry here must raise ValueError, not run.
_EXPECTED_INVALID_RADIX_SEQUENCES: tuple[tuple[int, ...], ...] = (
    (4, 4, 4, 4),  # P_2 = 16, doesn't divide simd_lanes=8
    (3, 4, 2),  # P_1 = 3
    (5, 2, 3),  # P_1 = 5
)


def main() -> None:
    tolerance = 1.0e-3
    failures: list[str] = []

    # Stage 2: layouts_for_radices generalizes beyond the hardcoded (4,4,4)
    # case make_444_plan calls it with -- single-stage sweep over every
    # supported radix, same-radix towers of depth 2 and (validly-ordered)
    # depth 5, and several mixed-radix orderings, forward and inverse.
    radix_sequence_cases: list[tuple[int, ...]] = (
        [(r,) for r in sorted(SUPPORTED_RADICES)]
        + [(2, 2), (3, 3), (4, 4), (2, 2, 2, 2, 2)]
        + [(2, 3, 4), (4, 3, 2), (7, 3), (9, 2), (13, 2), (11, 3)]
    )
    for radices in radix_sequence_cases:
        for inverse in (False, True):
            err = verify_radix_sequence_plan(radices, inverse=inverse, seed=1)
            tag = f"radix sequence {radices} inverse={inverse}"
            ok = err <= tolerance
            print(f"  {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
            if not ok:
                failures.append(tag)

    for radices in _EXPECTED_INVALID_RADIX_SEQUENCES:
        for inverse in (False, True):
            tag = f"radix sequence {radices} inverse={inverse} (expected rejection)"
            try:
                verify_radix_sequence_plan(radices, inverse=inverse, seed=1)
            except ValueError:
                print(f"  OK   {tag}: correctly rejected")
            else:
                print(f"  FAIL {tag}: should have been rejected but ran")
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

    if failures:
        raise AssertionError(f"{len(failures)} plan(s) failed: {failures}")
    print("[verify] all FFT plans matched numpy's FFT")


if __name__ == "__main__":
    main()
