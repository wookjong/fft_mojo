from __future__ import annotations

"""Numeric verification for fft_plan_simple.py's make_decomposed_plan, plus
the arbitrary-radix-sequence sweep that proves layouts_for_radices/
_build_plan themselves (fft_plan_core.py) generalize to any supported radix
sequence within one kernel -- see verify_fft_harness.py for the shared
re-execute-the-actual-emitted-text discipline every function here relies
on.

make_444_plan (fft_plan_simple.py) has no numeric check of its own here:
the specific N=64/(4,4,4)/simd_lanes=8 case it used to be checked with is
just one point on the general sweep below (included in the case list as
plain `(4, 4, 4)`, same as any other radix sequence) -- a dedicated
verify_single_kernel_plan duplicated that coverage and was removed.
make_444_plan itself stays: verify_fft_butterflies.py still calls it
directly, at simd_lanes=4, for an unrelated, non-redundant reason (an LLVM
register-spill regression case, confirmed against the real toolchain --
see that file's own comment).
"""

import numpy as np

from planning.fft_plan_core import _build_plan, layouts_for_radices, pingpong_needed
from planning.fft_plan_simple import make_decomposed_plan
from verification.verify_fft_harness import Ptr, _make_large_twiddle_table, run_kernel


def verify_radix_sequence_plan(
    radices: tuple[int, ...], *, inverse: bool, seed: int, simd_lanes: int = 8
) -> float:
    """Proves `layouts_for_radices` -- not just one hardcoded radix
    sequence -- by actually re-executing the emitted stage text, for
    whatever `radices` the caller sweeps over."""
    length = 1
    for r in radices:
        length *= r
    plan = _build_plan(
        length=length,
        inverse=inverse,
        total_uthreads=1,
        simd_lanes=simd_lanes,
        use_pingpong=pingpong_needed(len(radices)),
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

