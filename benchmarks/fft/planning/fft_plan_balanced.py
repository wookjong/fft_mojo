from __future__ import annotations

"""N = N_A*N_B, each side its own possibly-multi-kernel PEELED chain
(reusing fft_plan_multikernel.py's own machinery, generalized via a
`batch_count` seed -- see `_build_batched_side`), joined at the boundary
by one of two schemes:

* `AddressMappingKind.CROSSED` (fft_plan_core.py), fused into one side's
  own last-kernel store -- `make_balanced_plan` and `make_balanced_transpose_
  plan`'s own internal PEELED chains both use this; only the boundary
  itself differs between them.
* A standalone tiled transpose + fused large-twiddle kernel (see
  `FFTTransposePlan`, rendered by fft_transpose_codegen.py) -- used only
  by `make_balanced_transpose_plan`, whose whole point is keeping both
  DRAM-facing ends of the boundary unit-stride even when a side needs
  more than one internal kernel (CROSSED's own fused store cannot do
  that: its own `elem_stride` grows with a multi-kernel side's own
  `digit_multiplier`).

`make_balanced_plan` is the older, simpler entry point (CROSSED boundary
only) -- kept as-is, a regression baseline and numeric reference.
`make_balanced_transpose_plan` is additive.
"""

from dataclasses import dataclass
from math import prod

from planning.fft_plan_core import (
    AddressMapping,
    FFTCodegenPlan,
    LargeTwiddlePlan,
    MultiKernelFFTPlan,
    MultiKernelHostPlan,
    _build_plan,
    _cap_max_uthread,
    _prime_factors_supported,
    layouts_for_radices,
    pingpong_needed,
)
from planning.fft_plan_multikernel import _choose_side_chunks


def _build_batched_side(
    chunks: tuple[tuple[int, ...], ...],
    *,
    batch_count: int,
    full_length: int,
    inverse: bool,
    simd_lanes: int,
    kernel_name_prefix: str,
    is_far_side: bool,
    spad_capacity_bytes: int | None,
    plain_boundary: bool = False,
) -> tuple[FFTCodegenPlan, ...]:
    """One side (N_A or N_B) of a make_balanced_plan split, as its own
    possibly-multi-kernel PEELED chain -- see AddressMappingKind.CROSSED.

    `batch_count`: size of the *other* side (this side's own chain runs
    `batch_count` independent copies, one per the other side's index).
    `is_far_side`: True for the side whose last kernel is the *overall*
    plan's last kernel (plain, unchanged `strided(elem_stride=a)` output,
    the inverse scale, no large_twiddle -- the near side never reaches
    this, its own last kernel is CROSSED instead, chained on through DRAM
    into the far side's own first kernel).

    Every internal (non-last) kernel here is exactly what
    make_multi_kernel_plan already builds for a single-chain FFT, with
    two differences threaded through by the caller rather than derived
    fresh: `a` starts at `batch_count` (not 1) instead of a real prior
    kernel's length, and every LargeTwiddlePlan/PEELED `full_length` is
    the *overall* N, not this side's own length -- both required for the
    cross-side twiddle to land correctly (verified: full_length must be
    the overall N, not this side's own length, or the numbers come out
    wrong despite the addressing alone still being a valid permutation --
    see the plan doc for the N_B=3 example that caught this).

    `plain_boundary` (default False -- every existing caller is byte-for-
    byte unaffected): for `make_balanced_transpose_plan` only. When True,
    the boundary this side touches is realized by a standalone tiled
    transpose kernel (see fft_transpose_codegen.py) instead of being fused
    into this side's own store/load -- so the near side's (`is_far_side=
    False`) terminal kernel writes its own plain `contiguous(row_stride=
    side_length)` (no CROSSED, no large_twiddle: the twiddle moves to the
    transpose kernel), and the far side's (`is_far_side=True`) first
    kernel reads its own plain `contiguous(row_stride=length)` (no strided
    transpose read). Everything else -- internal PEELED chain, ping-pong,
    the far side's own terminal inverse-scale -- is unchanged.
    """
    lengths = [prod(chunk) for chunk in chunks]
    side_length = prod(lengths)
    m = len(chunks)

    kernels: list[FFTCodegenPlan] = []
    a = batch_count
    for i, ki in enumerate(lengths):
        is_last = i == m - 1
        total_uthreads = (side_length // ki) * batch_count

        if i == 0:
            if plain_boundary and is_far_side:
                # Reads the transpose kernel's own plain output layout
                # (contiguous(row_stride=length)), not a transpose read --
                # the transpose kernel already did that.
                input_mapping = AddressMapping.contiguous(row_stride=ki)
            else:
                elem_stride0 = (side_length // ki) * batch_count
                input_mapping = AddressMapping.strided(row_stride=1, elem_stride=elem_stride0)
        else:
            input_mapping = AddressMapping.contiguous(row_stride=ki)

        if is_last:
            if is_far_side:
                output_mapping = AddressMapping.strided(row_stride=1, elem_stride=a)
                large_twiddle = None
                inverse_scale = (1.0 / full_length) if inverse else None
            elif plain_boundary:
                # Own plain layout for the transpose kernel to read -- row
                # is this kernel's own raw uthread id, elem its own just-
                # computed ki-sized digit (row_stride=ki, NOT side_length:
                # for a multi-kernel near side this is genuinely different,
                # verified by direct index simulation against CROSSED's
                # own formula before this was written -- see
                # FFTTransposePlan's docstring). No CROSSED, no fused
                # twiddle (the transpose kernel does both, tile-wise --
                # see fft_transpose_codegen.py).
                output_mapping = AddressMapping.contiguous(row_stride=ki)
                large_twiddle = None
                inverse_scale = None
            else:
                digit_multiplier = a // batch_count
                output_mapping = AddressMapping.crossed(
                    batch_count, side_length, digit_multiplier
                )
                large_twiddle = LargeTwiddlePlan(
                    full_length=full_length,
                    row_count=batch_count,
                    output_count=side_length,
                    inverse=inverse,
                )
                inverse_scale = None
        else:
            k_next = lengths[i + 1]
            tail_size = prod(lengths[i + 2 :]) if i + 2 < m else 1
            output_mapping = AddressMapping.peeled(a, k_next, tail_size)
            large_twiddle = LargeTwiddlePlan(
                full_length=full_length,
                row_count=total_uthreads // a,
                output_count=ki,
                inverse=inverse,
                a=a,
                k_next=k_next,
                tail_size=tail_size,
            )
            inverse_scale = None

        kernels.append(
            _build_plan(
                length=ki,
                inverse=inverse,
                total_uthreads=total_uthreads,
                simd_lanes=simd_lanes,
                use_pingpong=pingpong_needed(len(chunks[i])),
                layouts=layouts_for_radices(ki, chunks[i], simd_lanes),
                kernel_name=f"{kernel_name_prefix}{i}",
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                large_twiddle=large_twiddle,
                inverse_scale=inverse_scale,
                spad_capacity_bytes=spad_capacity_bytes,
            )
        )
        a *= ki

    return tuple(kernels)


@dataclass(frozen=True)
class FFTTransposePlan:
    """Standalone tiled-transpose + fused large-twiddle boundary between a
    make_balanced_transpose_plan near side and far side -- see
    make_balanced_transpose_plan's own docstring for the full derivation.
    Terminates the near side's physical layout and creates the far side's
    physical layout from scratch; the fused twiddle operates on logical
    coordinates recovered at this boundary, so neither side needs to know
    the other's internal chunk/kernel decomposition or stride.

    `ki_near`/`ki_far`: the near side's own terminal kernel's length and
    the far side's own first kernel's length -- these two axes are what
    actually get swapped (transposed); everything else is a spectator that
    only multiplies the tile count.

    `digit_multiplier_near` (= n_a // ki_near) and `divisor_far` (= n_b //
    ki_far): both 1 when that side is single-kernel (bare radix), matching
    the simplest case exactly. `tile_uthreads = digit_multiplier_near *
    divisor_far` independent (ki_near x ki_far) tiles partition the whole
    N elements exactly once (verified by direct index simulation against
    AddressMapping.crossed's own formula before this was written).

    One microthread owns one whole (ki_near x ki_far) tile: `ki_far`
    contiguous vector reads of `ki_near` elements each from the near
    side's own plain layout (+ matching twiddle-table rows, addressed
    identically), a local transpose through one scratchpad buffer, then
    `ki_near` contiguous vector writes of `ki_far` elements each into the
    far side's own plain layout. No further sub-tiling by a smaller
    tile_size is done in this first version -- see fft_transpose_codegen.py
    and the design writeup's "remaining limitations" for the SIMD-width-
    alignment follow-up this defers.
    """

    n: int
    n_a: int
    n_b: int
    ki_near: int
    digit_multiplier_near: int
    ki_far: int
    divisor_far: int
    inverse: bool
    kernel_name: str
    simd_lanes: int
    total_uthreads: int
    max_uthread: int
    scratchpad_elements: int


def _choose_balanced_split(
    n: int, *, scratchpad_byte_budget: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Pick a contiguous split of n's supported prime factors into
    (factors_A, factors_B) minimizing max(prod(factors_A), prod(factors_B))
    subject to both sides being plannable under scratchpad_byte_budget (via
    _choose_side_chunks) -- see make_balanced_transpose_plan.

    Only contiguous splits of the factor list are searched (matches how
    factor_into_kernel_chunks's own chunks are always contiguous runs of
    this same list) -- kept as its own function so a future subset-
    partition search can replace just this one, without touching the
    feasibility check or the caller.
    """
    factors = _prime_factors_supported(n)
    num_factors = len(factors)
    if num_factors < 2:
        raise ValueError(
            f"n={n} has fewer than 2 supported prime factors; cannot split"
        )

    best: tuple[int, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]] | None = None
    for k in range(1, num_factors):
        factors_a = factors[:k]
        factors_b = factors[k:]
        n_a = prod(factors_a)
        n_b = prod(factors_b)
        try:
            chunks_a = _choose_side_chunks(
                factors_a, batch_count=n_b, scratchpad_byte_budget=scratchpad_byte_budget
            )
            chunks_b = _choose_side_chunks(
                factors_b, batch_count=n_a, scratchpad_byte_budget=scratchpad_byte_budget
            )
        except ValueError:
            continue
        cost = max(n_a, n_b)
        if best is None or cost < best[0]:
            best = (cost, chunks_a, chunks_b)

    if best is None:
        raise ValueError(
            f"scratchpad_byte_budget={scratchpad_byte_budget} is too small "
            f"to fit n={n} into any balanced two-side split"
        )
    return best[1], best[2]


@dataclass(frozen=True)
class BalancedTransposeFFTPlan:
    """N = N_A*N_B, near side (own PEELED chain, batched N_B times) then a
    standalone tiled transpose+twiddle boundary then far side (own PEELED
    chain, batched N_A times, ending in the overall output) -- see
    make_balanced_transpose_plan. Flat, ordered, heterogeneous sequence
    for codegen: `kernels_near + (transpose,) + kernels_far`.
    """

    n: int
    inverse: bool
    kernels_near: tuple[FFTCodegenPlan, ...]
    transpose: FFTTransposePlan
    kernels_far: tuple[FFTCodegenPlan, ...]
    host: MultiKernelHostPlan


def make_balanced_transpose_plan(
    n: int,
    *,
    scratchpad_byte_budget: int,
    simd_lanes: int = 8,
    inverse: bool = False,
    spad_capacity_bytes: int | None = None,
) -> BalancedTransposeFFTPlan:
    """N = N_A*N_B (N_A ~ N_B ~ sqrt(N) where the factorization allows,
    minimizing max(N_A,N_B) otherwise -- see _choose_balanced_split), each
    side its own possibly-multi-kernel PEELED chain bounded by that side's
    own size (see _build_batched_side, reused unmodified for everything
    except the boundary itself), joined by one standalone tiled transpose
    kernel instead of make_balanced_plan's scalar CROSSED-fused write.

    The near side's own terminal kernel and the far side's own first
    kernel both use `_build_batched_side(..., plain_boundary=True)`: each
    writes/reads its own plain contiguous layout, with no fused twiddle
    and no strided/scalar transpose access on either end -- see that
    parameter's own docstring. The transpose kernel (FFTTransposePlan)
    does the permutation and the twiddle, tile-wise, so both DRAM-facing
    ends of the boundary stay unit-stride, and neither side's plan needs
    to know the other side's internal chunk/kernel decomposition.
    """
    chunks_a, chunks_b = _choose_balanced_split(n, scratchpad_byte_budget=scratchpad_byte_budget)
    n_a = prod(prod(chunk) for chunk in chunks_a)
    n_b = prod(prod(chunk) for chunk in chunks_b)
    host = MultiKernelHostPlan(n=n, inverse=inverse, tolerance=1.0e-3)

    kernels_near = _build_batched_side(
        chunks_a, batch_count=n_b, full_length=n, inverse=inverse,
        simd_lanes=simd_lanes, kernel_name_prefix="FFTFP32Near",
        is_far_side=False, spad_capacity_bytes=spad_capacity_bytes,
        plain_boundary=True,
    )
    kernels_far = _build_batched_side(
        chunks_b, batch_count=n_a, full_length=n, inverse=inverse,
        simd_lanes=simd_lanes, kernel_name_prefix="FFTFP32Far",
        is_far_side=True, spad_capacity_bytes=spad_capacity_bytes,
        plain_boundary=True,
    )

    ki_near = kernels_near[-1].length
    digit_multiplier_near = n_a // ki_near
    ki_far = kernels_far[0].length
    divisor_far = n_b // ki_far
    tile_uthreads = digit_multiplier_near * divisor_far

    scratchpad_elements = 2 * ki_near * ki_far  # real+imag, one tile buffer
    max_uthread = _cap_max_uthread(
        tile_uthreads, scratchpad_elements * 4, spad_capacity_bytes,
        context="a single transpose tile",
    )

    transpose = FFTTransposePlan(
        n=n, n_a=n_a, n_b=n_b, ki_near=ki_near,
        digit_multiplier_near=digit_multiplier_near, ki_far=ki_far,
        divisor_far=divisor_far, inverse=inverse,
        kernel_name="FFTFP32Transpose", simd_lanes=simd_lanes,
        total_uthreads=tile_uthreads, max_uthread=max_uthread,
        scratchpad_elements=scratchpad_elements,
    )

    return BalancedTransposeFFTPlan(
        n=n, inverse=inverse, kernels_near=kernels_near, transpose=transpose,
        kernels_far=kernels_far, host=host,
    )


def make_balanced_plan(
    chunks_A: tuple[tuple[int, ...], ...],
    chunks_B: tuple[tuple[int, ...], ...],
    *,
    inverse: bool = False,
    simd_lanes: int = 8,
    spad_capacity_bytes: int | None = None,
) -> MultiKernelFFTPlan:
    """N = N_A * N_B (N_A = prod(chunks_A), N_B = prod(chunks_B)), each
    side its own possibly-multi-kernel PEELED chain, joined by one
    AddressMappingKind.CROSSED transpose fused into side A's own last
    kernel's store (never a standalone kernel) -- generalizes
    make_decomposed_plan from "each side is exactly one bare radix" to
    "each side is any chunk sequence make_multi_kernel_plan could build
    on its own", while keeping make_decomposed_plan itself untouched as
    the from-first-principles baseline this checks against (the M=1
    case on both sides is exactly make_decomposed_plan(N_A, N_B), field
    for field where the addressing coincides -- see AddressMappingKind
    .CROSSED's own docstring for why it degenerates to CONTIGUOUS there).

    The value of this over a single flat make_multi_kernel_plan chain
    over N's *whole* factor list: max effective DRAM stride bounded by
    max(N_A, N_B) rather than by N -- choosing N_A ~ N_B ~ sqrt(N) reaches
    the four-step FFT's classic O(sqrt(N)) bound, regardless of how deep
    either side's own internal chain has to go to fit the scratchpad
    budget (each side's own chain is bounded by that side's own size,
    never by N -- verified directly, not assumed: N=2^20 split into two
    balanced ~1024-element halves, each itself a deep 10-kernel all-
    radix-2 chain, gives max stride 1024 = sqrt(N); the same N run as one
    flat 20-kernel chain gives ~524288 ~ N/2). Choosing chunks_A/chunks_B
    to actually be balanced is the caller's job here (see the plan doc's
    "balanced split search", not yet implemented) -- this function only
    builds whatever split it's given.
    """
    n_a = prod(prod(chunk) for chunk in chunks_A)
    n_b = prod(prod(chunk) for chunk in chunks_B)
    n = n_a * n_b
    host = MultiKernelHostPlan(n=n, inverse=inverse, tolerance=1.0e-3)

    kernels_a = _build_batched_side(
        chunks_A,
        batch_count=n_b,
        full_length=n,
        inverse=inverse,
        simd_lanes=simd_lanes,
        kernel_name_prefix="FFTFP32KernelA",
        is_far_side=False,
        spad_capacity_bytes=spad_capacity_bytes,
    )
    kernels_b = _build_batched_side(
        chunks_B,
        batch_count=n_a,
        full_length=n,
        inverse=inverse,
        simd_lanes=simd_lanes,
        kernel_name_prefix="FFTFP32KernelB",
        is_far_side=True,
        spad_capacity_bytes=spad_capacity_bytes,
    )

    return MultiKernelFFTPlan(
        n=n, inverse=inverse, kernels=kernels_a + kernels_b, host=host
    )
