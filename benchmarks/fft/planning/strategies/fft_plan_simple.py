from __future__ import annotations

"""The two oldest, simplest FFT planning strategies -- kept as the
from-first-principles baseline every other strategy in this package is
checked against (see fft_plan_core.py for the shared FFTCodegenPlan
lowering machinery both of these build on).

`make_444_plan`: a single fused N=64 (radix 4x4x4) kernel, no DRAM
handoff at all.

`make_decomposed_plan`: N=N0*N1, exactly two kernels chained through
DRAM, each a single bare radix -- the original two-kernel decomposition
every later multi-kernel/balanced strategy generalizes.
"""

from dataclasses import dataclass

from planning.core.fft_plan_core import (
    AddressMapping,
    FFTCodegenPlan,
    LargeTwiddlePlan,
    _build_plan,
    layouts_for_radices,
)


def make_444_plan(
    *, inverse: bool = False, total_uthreads: int = 1, simd_lanes: int = 8
) -> FFTCodegenPlan:
    """Create the fully lowered N=64, radix-4 x radix-4 x radix-4 plan."""

    return _build_plan(
        length=64,
        inverse=inverse,
        total_uthreads=total_uthreads,
        simd_lanes=simd_lanes,
        use_pingpong=True,
        layouts=layouts_for_radices(64, (4, 4, 4), simd_lanes=simd_lanes),
    )


# --------------------------------------------------------------- decomposition
#
# N too large for one uthread's scratchpad: N = N0*N1, run as two kernels
# chained through DRAM. See the module docstring's index-mapping walkthrough
# for the derivation; the short version is:
#
#   kernel0 (N0 uthreads, row=n0): strided load x[n1*N0+n0], forward N1-point
#     FFT over n1, multiply by W_N^(n0*k1), contiguous store mid[n0*N1+k1]
#   kernel1 (N1 uthreads, row=k1): strided load mid[n0*N1+k1] (the actual
#     transpose read of kernel0's contiguous-by-n0 store), forward N0-point
#     FFT over n0, strided store out[k0*N1+k1]
#
# Both kernels are ordinary single-stage FFTCodegenPlans (radix == their own
# length, `layouts_for_radices(radix, (radix,), simd_lanes)` already covers
# this); only their AddressMapping and (kernel0's) LargeTwiddlePlan differ
# from a single-kernel plan's defaults.


@dataclass(frozen=True)
class DecomposedHostPlan:
    """Host-side shape for the two-kernel main(). Same philosophy as
    HostPlan: fresh random input and an independent O(N^2) DFT reference
    (over the *full* N, not the N0/N1 decomposition either kernel runs) are
    generated/computed at Mojo host runtime -- nothing numeric is baked in
    here, only sizes."""

    n: int
    n0: int
    n1: int
    inverse: bool
    tolerance: float


@dataclass(frozen=True)
class DecomposedFFTPlan:
    """N = N0*N1, run as two kernels chained through DRAM -- never through a
    shared scratchpad, since nothing survives a kernel launch boundary but
    what was written to DRAM (see module docstring). Exactly one independent
    length-N FFT (no cross-FFT batching yet: `AddressMapping` is one runtime
    multiply, and batching would need a second one to fold the batch index
    in without falling back to runtime division -- see make_decomposed_plan).
    """

    n: int
    n0: int
    n1: int
    inverse: bool
    kernel0: FFTCodegenPlan
    kernel1: FFTCodegenPlan
    host: DecomposedHostPlan


def make_decomposed_plan(
    n0: int, n1: int, *, inverse: bool = False, simd_lanes: int = 8
) -> DecomposedFFTPlan:
    """N = n0*n1, planned as kernel0 (N0 uthreads, N1-point sub-FFTs, large
    twiddle fused into its output store) then kernel1 (N1 uthreads, N0-point
    sub-FFTs), chained through DRAM. Every decomposition decision -- which
    factor each kernel owns, each kernel's DRAM AddressMapping, the large
    twiddle's table shape, where the final 1/N inverse scale lands -- is
    made here; fft_codegen.py only renders what this plan already decided.

    One independent length-N FFT per call (see DecomposedFFTPlan); pass N0
    and N1 in SUPPORTED_RADICES (each becomes one kernel's single-stage
    radix -- see layouts_for_radices).
    """
    n = n0 * n1

    kernel0 = _build_plan(
        length=n1,
        inverse=inverse,
        total_uthreads=n0,
        simd_lanes=simd_lanes,
        use_pingpong=False,
        layouts=layouts_for_radices(n1, (n1,), simd_lanes),
        kernel_name="FFTFP32Kernel0",
        # x[n1*N0 + n0]: this uthread's row is n0 (row_stride=1), its N1
        # elements are spaced N0 apart in the original contiguous input.
        input_mapping=AddressMapping.strided(row_stride=1, elem_stride=n0),
        # mid[n0*N1 + k1]: contiguous per row -- this uthread's own N1
        # outputs land next to each other. kernel1 (row=k1) reads this same
        # buffer *transposed*: fixed k1, n0 stepping by N1 -- see kernel1's
        # input_mapping below, which is strided, not contiguous.
        output_mapping=AddressMapping.contiguous(row_stride=n1),
        large_twiddle=LargeTwiddlePlan(
            full_length=n, row_count=n0, output_count=n1, inverse=inverse
        ),
        # The intra-kernel butterfly must NOT apply 1/N here: this is not
        # the final kernel of the decomposition. The overall inverse scale
        # (1/n, not 1/n1) lands once, at kernel1.
        inverse_scale=None,
    )

    kernel1 = _build_plan(
        length=n0,
        inverse=inverse,
        total_uthreads=n1,
        simd_lanes=simd_lanes,
        use_pingpong=False,
        layouts=layouts_for_radices(n0, (n0,), simd_lanes),
        kernel_name="FFTFP32Kernel1",
        # mid[n0*N1 + k1]: kernel0 wrote this contiguously by *its* row n0
        # (output_mapping above). Read back by k1 instead, each of this
        # uthread's N0 elements (n0 = 0..N0-1) sits N1 apart in that same
        # layout -- this is the actual transpose read, and it is strided,
        # not contiguous (an earlier version of this function wrongly
        # assumed mid[k1*N0+n0], which is a different set of elements
        # entirely except where n0 happens to equal k1).
        input_mapping=AddressMapping.strided(row_stride=1, elem_stride=n1),
        # out[k0*N1 + k1]: k1 (this row) is the *fast* digit of the true
        # output index, so scattering across k0 (what this kernel produces)
        # is inherently strided by N1 -- see module docstring; no kernel
        # split avoids this for the kernel that owns k1.
        output_mapping=AddressMapping.strided(row_stride=1, elem_stride=n1),
        large_twiddle=None,
        inverse_scale=(1.0 / n) if inverse else None,
    )

    return DecomposedFFTPlan(
        n=n,
        n0=n0,
        n1=n1,
        inverse=inverse,
        kernel0=kernel0,
        kernel1=kernel1,
        host=DecomposedHostPlan(
            n=n, n0=n0, n1=n1, inverse=inverse, tolerance=1.0e-3
        ),
    )
