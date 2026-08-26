from __future__ import annotations

"""Generalizes DecomposedFFTPlan (fft_plan_simple.py -- exactly 2 kernels,
each one bare radix) to a chain of M>=1 kernels, each itself a
layouts_for_radices multi-stage FFT via scratchpad ping-pong -- absorbing
as many radix stages as fit in one uthread's scratchpad budget per kernel,
minimizing DRAM handoffs instead of the two extremes make_444_plan/
make_decomposed_plan cover. M=1 and M=2 are exact generalizations of
those two, verified equivalent (see verify_fft_plan.py). Non-last kernels
use AddressMappingKind.PEELED (fft_plan_core.py) so every kernel after the
first gets a vector (not scalar) DRAM read.

`_choose_side_chunks` here is also reused by fft_plan_balanced.py: it is
factor_into_kernel_chunks's own chunk-selection DP, generalized for one
side of a balanced N=N_A*N_B split, where an outer `batch_count` (the
*other* side's size) multiplies this side's own kernel0 read cost and
every internal kernel's own write-seed `a`.
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
    _prime_factors_supported,
    layouts_for_radices,
    pingpong_needed,
)


def factor_into_kernel_chunks(
    n: int, *, scratchpad_byte_budget: int
) -> tuple[tuple[int, ...], ...]:
    """Factor n into a sequence of per-kernel radix chunks -- chunk i
    becomes kernel i's radix sequence for layouts_for_radices, processed
    in order and chained through DRAM by make_multi_kernel_plan.

    Built greedily under one cap: `16 * chunk_product <= scratchpad_byte_budget`
    (the `scratchpad_uthread_stride = 2*length` times 2 ping-pong buffers
    times 4 bytes/float convention `_build_plan` already uses). No default
    is offered: the real M2NDP per-uthread scratchpad size isn't
    established in this codebase, so callers pick a value.

    An earlier version of this also capped chunk depth/ordering to satisfy
    what looked like a real constraint in `_check_layouts` (every non-last
    *stage*'s cumulative radix product dividing simd_lanes) -- verified at
    the time by bypassing the check and watching results go to ~O(1)
    wrong. That verification was itself standing on a bug in the
    verification harness (numpy aliasing on `var or0 = rr0`-style copies --
    see verify_fft_plan.py's SimdVec), and re-run after fixing it, every
    previously-failing case (mixed radix, depth-8 same-radix towers, etc.)
    passes cleanly. The constraint was removed from `_check_layouts`
    accordingly, and chunk order/composition here is unconstrained beyond
    the scratchpad budget.

    Chosen to minimize the largest *effective DRAM cost* any kernel in the
    chain pays. Every kernel in an AddressMappingKind.PEELED chain (see
    make_multi_kernel_plan) pays one of two costs, and a chunking choice
    affects each differently: kernel0's *read* costs `n // (its own
    length)` -- unavoidable, nothing precedes it to fuse a layout reset
    into -- and every *non-last* kernel's *write* costs `a * (the *next*
    kernel's own length)` (`a` = the product of every earlier kernel's
    length; the last kernel's write is the exception, unchanged at plain
    `a`, since there's no next kernel to optimize for). Every kernel after
    the first reads at cost 1 (contiguous, PEELED's whole point), so only
    these two costs -- kernel0's read, and each non-last kernel's write --
    ever matter.

    A chunk's write cost depending on the *next* chunk's length (not just
    its own start) breaks naive optimal substructure: a DP keyed only on
    "best cost for factors[i:]" can't supply what a chunk ending at `i`
    needs (the length of *its own* immediately-following chunk) without
    also depending on how that suffix chooses to split, and that suffix's
    own optimum doesn't necessarily supply the length this chunk wants
    (checked directly, not assumed: an earlier, simpler version of this
    keyed only on (start) and produced a *worse* result at a *larger*
    budget on this same n=960 sweep -- impossible for a correct DP, since
    a larger budget can only add valid choices, never remove one).

    Fixed by keying state on the pair (chunk start, chunk end) instead:
    state `(start, j)` means "chunk [start:j) is chosen, but its write
    cost isn't finalized yet" -- exactly true until a *following* chunk
    [j:j2) is also chosen, at which point [start:j)'s write cost
    (`prefix_product[start] * length([j:j2))`) becomes computable and
    folds into the running max carried forward as the new state (j, j2).
    `O(F^2)` states, `O(F)` transitions each -- `O(F^3)`, still cheap for
    the factor count `F` (at most ~20 for any n this codebase addresses)
    -- and exact for this cost function, not a heuristic: every reachable
    (start, j) keeps only its minimum cost, so no choice that could affect
    the final answer is dropped. Reproduces the plan doc's hand-derived
    N=960 numbers (max cost 120 at scratchpad_byte_budget=256) and is
    monotonically non-increasing in budget, checked directly.
    """
    if scratchpad_byte_budget <= 0:
        raise ValueError("scratchpad_byte_budget must be positive")

    factors = _prime_factors_supported(n)
    cap = scratchpad_byte_budget // 16
    num_factors = len(factors)

    prefix_product = [1] * (num_factors + 1)
    for i, f in enumerate(factors):
        prefix_product[i + 1] = prefix_product[i] * f

    # Under AddressMappingKind.PEELED (see make_multi_kernel_plan), a
    # non-first kernel's read is always 1 (contiguous) -- only kernel0
    # pays n // its own length. A non-last kernel's write is
    # prefix_product[start] * (the *next* kernel's own length), not just
    # prefix_product[start] -- so a chunk [start:j]'s write cost isn't
    # knowable until the chunk *after* it is also chosen. That rules out
    # a single-value-per-position DP (the greedy choice that's locally
    # best for factors[j:] on its own doesn't necessarily supply the
    # `next kernel's own length` that minimizes chunk [start:j]'s write
    # cost -- optimal substructure genuinely fails for that formulation,
    # confirmed by hitting it directly: an earlier version keyed only on
    # a chunk's own (start, j) and it produced a *worse* result at a
    # *larger* budget for this same n=960 sweep, which is impossible for
    # a correct DP since a larger budget only adds valid choices).
    #
    # Fixed by keying state on the *pair* (start, j): "chunk [start:j] has
    # been chosen but its write cost isn't finalized yet -- that happens
    # the moment a following chunk [j:j2] is also chosen, which is also
    # exactly when [start:j]'s write cost becomes computable
    # (prefix_product[start] * (j2's chunk length)) and gets folded into
    # the running max carried forward as state (j, j2)." O(F^2) states,
    # O(F) transitions each = O(F^3) -- still cheap for F<=~20 -- and this
    # one is exact, not a heuristic: (start, j) is reached via `best[...]
    # = (cost, predecessor_start)`, always keeping the minimum cost seen
    # for that exact state, so every choice that could affect the final
    # answer is considered.
    #
    # Verified: this reproduces the plan doc's hand-derived N=960 numbers
    # (max cost 120 at scratchpad_byte_budget=256) and is monotonically
    # non-increasing in budget, checked directly across the same sweep
    # the removed version broke.
    best: dict[tuple[int, int], tuple[int, int | None]] = {}
    for j in range(1, num_factors + 1):
        chunk_length = prefix_product[j]  # prefix_product[0] == 1
        if chunk_length > cap:
            break
        best[(0, j)] = (n // chunk_length, None)

    for j in range(1, num_factors + 1):
        for start in range(j):
            state = best.get((start, j))
            if state is None:
                continue
            running_cost, _ = state
            for j2 in range(j + 1, num_factors + 1):
                next_length = prefix_product[j2] // prefix_product[j]
                if next_length > cap:
                    break
                write_cost = prefix_product[start] * next_length
                candidate = max(running_cost, write_cost)
                key = (j, j2)
                existing = best.get(key)
                if existing is None or candidate < existing[0]:
                    best[key] = (candidate, start)

    final: tuple[int, int] | None = None  # (total cost, last chunk's start)
    for start in range(num_factors):
        state = best.get((start, num_factors))
        if state is None:
            continue
        running_cost, _ = state
        total = max(running_cost, prefix_product[start])  # last kernel: write=a, no next-length multiplier
        if final is None or total < final[0]:
            final = (total, start)
    if final is None:
        raise ValueError(
            f"scratchpad_byte_budget={scratchpad_byte_budget} is too small "
            f"to fit n={n} into any valid chunk sequence"
        )

    boundaries = [num_factors, final[1]]
    j, start = num_factors, final[1]
    while start != 0:
        _, pred = best[(start, j)]
        assert pred is not None
        boundaries.append(pred)
        j, start = start, pred
    boundaries.reverse()

    return tuple(
        tuple(factors[boundaries[k] : boundaries[k + 1]])
        for k in range(len(boundaries) - 1)
    )


def _choose_side_chunks(
    factors: list[int], *, batch_count: int, scratchpad_byte_budget: int
) -> tuple[tuple[int, ...], ...]:
    """factor_into_kernel_chunks's own DP, generalized for one side of a
    make_balanced_plan split: `batch_count` (the *other* side's size)
    multiplies this side's kernel0 read cost and every internal (non-
    last) kernel's write-seed `a`, exactly the way AddressMappingKind
    .CROSSED's own docstring describes -- but *not* the terminal (CROSSED)
    kernel's own write cost, which is `digit_multiplier` alone (this
    side's own accumulated product, with the batch factor already divided
    back out by construction -- see AddressMapping.crossed).

    Necessary, not optional: naively feeding a plain
    factor_into_kernel_chunks(side_length, budget) result into
    make_balanced_plan's `_build_batched_side` measurably fails to reach
    anywhere near sqrt(N) whenever this side needs more than one internal
    kernel -- checked directly, not assumed (N=960 split into two
    factor_into_kernel_chunks(960, budget=4096)-chosen sides gave
    max_effective_stride=480, no better than one flat 8-kernel chain over
    the same N; the batch-aware version below gets both sides down near
    sqrt(N) instead). The reason: kernel0's read is `(side_length //
    its_own_length) * batch_count`, and `factor_into_kernel_chunks` alone
    has no idea `batch_count` is about to multiply whatever it picks.
    """
    if scratchpad_byte_budget <= 0:
        raise ValueError("scratchpad_byte_budget must be positive")
    if batch_count <= 0:
        raise ValueError("batch_count must be positive")

    cap = scratchpad_byte_budget // 16
    num_factors = len(factors)
    prefix_product = [1] * (num_factors + 1)
    for i, f in enumerate(factors):
        prefix_product[i + 1] = prefix_product[i] * f
    side_length = prefix_product[num_factors]

    best: dict[tuple[int, int], tuple[int, int | None]] = {}
    for j in range(1, num_factors + 1):
        chunk_length = prefix_product[j]
        if chunk_length > cap:
            break
        read_cost = (side_length // chunk_length) * batch_count
        best[(0, j)] = (read_cost, None)

    for j in range(1, num_factors + 1):
        for start in range(j):
            state = best.get((start, j))
            if state is None:
                continue
            running_cost, _ = state
            for j2 in range(j + 1, num_factors + 1):
                next_length = prefix_product[j2] // prefix_product[j]
                if next_length > cap:
                    break
                write_cost = (batch_count * prefix_product[start]) * next_length
                candidate = max(running_cost, write_cost)
                key = (j, j2)
                existing = best.get(key)
                if existing is None or candidate < existing[0]:
                    best[key] = (candidate, start)

    final: tuple[int, int] | None = None
    for start in range(num_factors):
        state = best.get((start, num_factors))
        if state is None:
            continue
        running_cost, _ = state
        # Terminal (CROSSED) kernel: write = digit_multiplier = this
        # side's own accumulated product, no batch_count multiplier.
        total = max(running_cost, prefix_product[start])
        if final is None or total < final[0]:
            final = (total, start)
    if final is None:
        raise ValueError(
            f"scratchpad_byte_budget={scratchpad_byte_budget} is too small "
            f"to fit this side into any valid chunk sequence"
        )

    boundaries = [num_factors, final[1]]
    j, start = num_factors, final[1]
    while start != 0:
        _, pred = best[(start, j)]
        assert pred is not None
        boundaries.append(pred)
        j, start = start, pred
    boundaries.reverse()

    return tuple(
        tuple(factors[boundaries[k] : boundaries[k + 1]])
        for k in range(len(boundaries) - 1)
    )


@dataclass(frozen=True)
class MultiKernelHostPlan:
    n: int
    inverse: bool
    tolerance: float


@dataclass(frozen=True)
class MultiKernelFFTPlan:
    """N run as a chain of M>=1 kernels (see factor_into_kernel_chunks /
    make_multi_kernel_plan), each itself a layouts_for_radices multi-stage
    FFT. M=1 is exactly a single-kernel plan. M=2 no longer matches
    make_decomposed_plan's own addressing field-for-field (that function
    is untouched, still SPLIT/contiguous-based); this one's non-last
    kernels use AddressMappingKind.PEELED instead, chosen so every kernel
    after the first gets a vector (not scalar) DRAM read -- both are
    independently numpy-verified correct, they just lay the intermediate
    array out differently. See AddressMappingKind.PEELED.
    """

    n: int
    inverse: bool
    kernels: tuple[FFTCodegenPlan, ...]
    host: MultiKernelHostPlan


def make_multi_kernel_plan(
    chunks: tuple[tuple[int, ...], ...],
    *,
    inverse: bool = False,
    simd_lanes: int = 8,
    spad_capacity_bytes: int | None = None,
) -> MultiKernelFFTPlan:
    """Build the FFTCodegenPlan chain for an already-decided chunk
    sequence (see factor_into_kernel_chunks): kernel i processes chunks[i]
    -- its own layouts_for_radices multi-stage FFT -- in order, chained
    through DRAM.

    The *first* kernel's DRAM input read is `strided(row_stride=1,
    elem_stride=n//K_0)` -- unavoidably O(n), since it reads the external
    input and nothing precedes it to fuse a layout reset into. Every
    *later* kernel's read is `strided(row_stride=K_i, elem_stride=1)`: a
    real vector load, courtesy of the previous kernel's PEELED write (see
    AddressMappingKind.PEELED) rather than the uniform `n//K_i` every
    kernel used to pay regardless of position. Every non-last kernel's
    output is `AddressMapping.peeled(a, k_next, tail_size)` (`a` = the
    product of the kernel lengths processed before it, same as SPLIT used;
    `k_next`/`tail_size` describe the *next* kernel's own digit and
    everything after it) plus a LargeTwiddlePlan scoped the same way SPLIT's
    was. The last kernel's output is unchanged from before:
    `strided(elem_stride=a)`, and only it carries the 1/n inverse scale --
    verified (by direct index simulation, not just plan-time inspection)
    to still land in the same numpy-correct natural order as the old
    SPLIT-based scheme despite reading from a PEELED predecessor.

    None of this was carried over from the M=2 case by analogy: it's an
    independent numpy simulation of the general M-kernel decomposition
    (cascaded per-kernel local DFT, a cross-kernel twiddle scoped to the
    *remaining* problem size, and a re-split store threading the new
    digit between the already- and not-yet-transformed parts of the
    uthread id) that was verified against numpy's fft/ifft for up to 6
    chained kernels and mixed radices before being written here -- see
    the plan doc's Stage 4 write-up for the derivation and where the
    first version of this went wrong.
    """
    if not chunks:
        raise ValueError("at least one kernel chunk is required")

    lengths = [prod(chunk) for chunk in chunks]
    n = prod(lengths)
    m = len(chunks)
    host = MultiKernelHostPlan(n=n, inverse=inverse, tolerance=1.0e-3)

    if m == 1:
        kernel = _build_plan(
            length=n,
            inverse=inverse,
            total_uthreads=1,
            simd_lanes=simd_lanes,
            use_pingpong=pingpong_needed(len(chunks[0])),
            layouts=layouts_for_radices(n, chunks[0], simd_lanes),
            spad_capacity_bytes=spad_capacity_bytes,
        )
        return MultiKernelFFTPlan(n=n, inverse=inverse, kernels=(kernel,), host=host)

    kernels: list[FFTCodegenPlan] = []
    a = 1  # product of the kernel lengths processed before the current one
    for i, ki in enumerate(lengths):
        is_last = i == m - 1
        total_uthreads = n // ki

        if i == 0:
            # First kernel: reads the external input, nothing precedes it
            # to fuse a layout reset into -- unchanged, still O(n) stride.
            input_mapping = AddressMapping.strided(row_stride=1, elem_stride=total_uthreads)
        else:
            # Every later kernel: contiguous, courtesy of the previous
            # kernel's PEELED output below -- see AddressMappingKind.PEELED.
            input_mapping = AddressMapping.contiguous(row_stride=ki)

        if is_last:
            # Unchanged: verified (by direct simulation, not just by
            # inspection) that the last kernel's own write formula still
            # lands in numpy-correct natural order even though its *read*
            # now comes from a PEELED predecessor instead of the old
            # uniform strided(elem_stride=n//K) -- see the plan doc.
            output_mapping = AddressMapping.strided(row_stride=1, elem_stride=a)
            large_twiddle = None
            inverse_scale = (1.0 / n) if inverse else None
        else:
            k_next = lengths[i + 1]
            tail_size = prod(lengths[i + 2 :]) if i + 2 < m else 1
            output_mapping = AddressMapping.peeled(a, k_next, tail_size)
            large_twiddle = LargeTwiddlePlan(
                full_length=n,
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
                kernel_name=f"FFTFP32Kernel{i}",
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                large_twiddle=large_twiddle,
                inverse_scale=inverse_scale,
                spad_capacity_bytes=spad_capacity_bytes,
            )
        )
        a *= ki

    return MultiKernelFFTPlan(n=n, inverse=inverse, kernels=tuple(kernels), host=host)
