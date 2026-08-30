"""End-to-end numeric verification of every FFT planning strategy in this
package -- the top-level test runner. Each strategy's own verify/summarize
functions live in its own verify_fft_*.py module (mirroring fft_plan_*.py's
own split); this file just imports all of them and runs the full sweep.

verify_fft_butterflies.py checks each radix's butterfly in isolation. This
checks the rest of what fft_codegen.py/fft_transpose_codegen.py emit around
it -- load/store addressing, per-stage twiddle, scratchpad ping-pong, large-
twiddle tables, and every transpose boundary -- by actually re-executing the
emitted stage text (see verify_fft_harness.py), not by reimplementing the
algorithm a second time.
"""

import sys
from math import prod
from pathlib import Path

# benchmarks/fft/ (this file's grandparent) holds the role directories
# (planning/, codegen/, verification/) as importable packages, plus
# make_fft_kernel.py/radix_spec.py themselves at the top level -- add it to
# sys.path so this script runs directly (`python3 verification/verify_fft_plan.py`) without
# needing `python3 -m`. Everything this file imports afterward inherits
# the same sys.path (it's process-global), so only entry-point scripts
# (this one and verify_fft_butterflies.py) need this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codegen.fft_codegen import _mapping_base_expr
from codegen.lowering import _chunk_store
from radix_spec import SUPPORTED_RADICES
from planning.fft_plan_core import (
    StorePlan,
    _make_store,
    _prime_factors_supported,
    _StageLayout,
    coalesce_radices,
    max_effective_stride,
    summarize_multi_kernel_plan,
)
from planning.fft_plan_simple import make_decomposed_plan
from planning.fft_plan_multikernel import _choose_side_chunks, factor_into_kernel_chunks, make_multi_kernel_plan
from planning.fft_plan_balanced import _build_batched_side, make_balanced_plan, make_balanced_transpose_plan
from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from planning.fft_plan_recursive import _build_physical_transpose, make_recursive_transpose_plan
from verification.verify_fft_simple import verify_decomposed_plan, verify_radix_sequence_plan
from verification.verify_fft_multikernel import verify_boundary_consistency, verify_layout_bijection, verify_multi_kernel_plan
from verification.verify_fft_balanced import (
    summarize_balanced_transpose_plan,
    verify_balanced_plan,
    verify_balanced_transpose_plan,
    verify_transpose_bijection,
)
from verification.verify_fft_recursive import (
    summarize_recursive_plan,
    verify_physical_transpose_shape,
    verify_recursive_plan,
    verify_recursive_tree_index_only,
)
from verification.verify_fft_cooperative import verify_cooperative_leaf


def main() -> None:
    tolerance = 1.0e-3
    failures: list[str] = []

    # Stage 2: layouts_for_radices generalizes beyond one hardcoded radix
    # sequence -- single-stage sweep over every supported radix, same-radix
    # towers of depth 2 and up, and several mixed-radix orderings, forward
    # and inverse. Depth-4/5 same-radix and (3,4,2)/(5,2,3) were rejected by
    # an earlier _check_layouts constraint (simd_lanes % twiddle_lane_divisor
    # == 0); that constraint's own "confirmed real" check was standing on a
    # bug in this harness (numpy aliasing on `var or0 = rr0`-style copies,
    # fixed below as SimdVec) and didn't survive re-verification, so it was
    # removed -- these are ordinary passing cases now, not a special
    # "expected rejection" list. `(4, 4, 4)` covers what a dedicated
    # verify_single_kernel_plan(make_444_plan(...)) used to check on its
    # own (removed as redundant -- see verify_fft_simple.py's own docstring).
    radix_sequence_cases: list[tuple[int, ...]] = (
        [(r,) for r in sorted(SUPPORTED_RADICES)]
        + [(2, 2), (3, 3), (4, 4), (4, 4, 4), (2, 2, 2, 2, 2), (2,) * 8, (3,) * 5]
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
    # Stage 4: M>=3 -- every non-last kernel's output uses AddressMappingKind
    # .PEELED (a middle kernel's uthread id mixes an already-transformed
    # digit run with a not-yet-transformed remainder) and a LargeTwiddlePlan scoped to
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
        # Both checks below only read this plan (never mutate it), so one
        # make_multi_kernel_plan build serves both instead of each check
        # rebuilding its own copy.
        shared_plan = make_multi_kernel_plan(chunks)

        tag = f"layout bijection N={n} chunks={chunks}"
        try:
            verify_layout_bijection(chunks, plan=shared_plan)
            print(f"  OK   {tag}")
        except AssertionError as exc:
            print(f"  FAIL {tag}: {exc}")
            failures.append(tag)

        tag = f"boundary consistency N={n} chunks={chunks}"
        try:
            verify_boundary_consistency(chunks, plan=shared_plan)
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
    # fft_plan_balanced.FFTTransposePlan and fft_transpose_codegen.py.
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

    # make_recursive_transpose_plan: FFT chunk size and physical transpose
    # tile size are fully independent (see fft_plan_recursive.PhysicalTransposePlan
    # / FFTRecursiveNodePlan) -- generalizes make_balanced_transpose_plan's
    # single 2-way split to a full six-step-FFT-style recursion, additive,
    # make_multi_kernel_plan/make_balanced_plan/make_balanced_transpose_plan
    # untouched.
    print("  Physical tile index tests (square/rectangular, tails, multiple replicas):")
    tile_cases = [
        ("square full tile", 8, 8, 1, 4, 4),
        ("rectangular full tile", 12, 20, 1, 3, 5),
        ("row tail", 10, 8, 1, 4, 4),
        ("column tail", 8, 10, 1, 4, 4),
        ("both tails", 11, 13, 1, 4, 5),
        ("multiple replicas", 6, 6, 5, 4, 4),
        ("degenerate 1xN", 1, 17, 3, 4, 4),
    ]
    for label, rows, cols, reps, tr, tc in tile_cases:
        pt = _build_physical_transpose(
            rows=rows, cols=cols, replica_count=reps, tile_rows=tr, tile_cols=tc,
            twiddle_modulus=None, inverse=False, kernel_name=f"PTTest_{label.replace(' ', '_')}",
            simd_lanes=8, spad_capacity_bytes=None, apply_inverse_scale=False,
        )
        tag = f"{label} ({rows}x{cols} reps={reps} tile={tr}x{tc})"
        try:
            shape = verify_physical_transpose_shape(pt)
            ok = shape["load_elem_stride"] == 1 and shape["store_elem_stride"] == 1
            print(f"    {'OK  ' if ok else 'FAIL'} {tag}: {shape}")
            if not ok:
                failures.append(tag)
        except AssertionError as exc:
            print(f"    FAIL {tag}: {exc}")
            failures.append(tag)

    print()
    print("  Recursive numeric tests (real emitted code vs. numpy.fft/ifft):")
    recursive_cases: list[tuple[int, int]] = [
        (30, 30 * 16),    # single leaf, no split
        (30, 6 * 16),     # one split level
        (60, 12 * 16),
        (64, 16 * 16),
        (120, 24 * 16),
        (960, 32 * 16),   # one split level, both sides multi-radix leaves
        (210, 21 * 16),   # 2*3*5*7 -- forces a 2-level recursion
        (256, 4096),      # single leaf, coalesced to (4,4,4,4) -- see store vectorization below
    ]
    for n, budget in recursive_cases:
        for inverse in (False, True):
            err, plan = verify_recursive_plan(n, scratchpad_byte_budget=budget, inverse=inverse, seed=23)
            ok = err <= tolerance
            tag = f"recursive plan n={n} budget={budget} inverse={inverse}"
            print(f"    {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
            if not ok:
                failures.append(tag)

    print()
    print("  loop_stages runtime-loop rendering (real emitted code vs. numpy.fft/ifft):")
    # verify_recursive_plan defaults to loop_stages=True (matching
    # generate_recursive_fft_kernels's own default -- see
    # verify_fft_recursive.run_recursive_plan's docstring), so every case
    # above already numerically exercises the looped-stage rendering
    # (_try_build_loop_stage/_emit_loop_stage), not just the fully-unrolled
    # one. These cases specifically straddle _LOOP_MIN_FULL_BATCHES (a
    # stage needs >= this many full SIMD batches to loop at all -- see
    # fft_codegen.py's module note above _LOOP_MIN_FULL_BATCHES): N=1024 is
    # a power of 2, so a stage's full-batch count only ever lands on a
    # power of 2 (N/radix/simd_lanes) -- it can never equal
    # _LOOP_MIN_FULL_BATCHES==3 exactly, only step from 2 (just below, so
    # _try_build_loop_stage must still fall back to the unrolled tail
    # rendering under loop_stages=True) to 4 (just above, so it loops).
    # budget=4096 is the specific N=1024 case _try_build_loop_stage's own
    # docstring calls out by name (FFTRecNear0 stages 4/5/6 need period
    # 2/4/8, doubling in step with the stage's own cumulative radix
    # product) -- kept as an explicit case since it's the one this repo's
    # own comments already reason about. The last case pins the opposite
    # default (loop_stages=False) so the fully-unrolled path some other
    # caller could still request from this same strategy stays covered
    # too, not just the True default every case above already exercises.
    loop_stage_cases: list[tuple[str, int, int, bool]] = [
        ("N=1024 below _LOOP_MIN_FULL_BATCHES (full batches=2, falls back to unroll)", 1024, 512, True),
        ("N=1024 above _LOOP_MIN_FULL_BATCHES (full batches=4, loops)", 1024, 1024, True),
        ("N=1024 period-doubling (FFTRecNear0 stages 4/5/6, periods 2/4/8)", 1024, 256 * 16, True),
        ("N=960 loop_stages explicitly disabled", 960, 32 * 16, False),
    ]
    for label, n, budget, loop_stages in loop_stage_cases:
        for inverse in (False, True):
            err, plan = verify_recursive_plan(
                n, scratchpad_byte_budget=budget, inverse=inverse, seed=31, loop_stages=loop_stages,
            )
            ok = err <= tolerance
            tag = f"{label} (n={n} budget={budget} loop_stages={loop_stages} inverse={inverse})"
            print(f"    {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
            if not ok:
                failures.append(tag)

    print()
    print("  round-split launches and non-power-of-2 tile-count lookup tables (real emitted code vs. numpy.fft/ifft):")
    # Both needs_round_split (generate_recursive_fft_kernels' own launch
    # split once max_uthread < total_uthreads -- see
    # verify_recursive_plan's own docstring for exactly what this harness
    # can and cannot check about it) and needs_q_table (the
    # replica_count > 1 branch of the mulhsu-avoidance tables -- the
    # sibling needs_tr_table branch is already exercised by the n=60/210
    # cases above, both of which have grid_cols=3/5) had zero regression
    # coverage before this: nothing above ever passes
    # max_concurrent_scratchpad_bytes, and no existing case's tiling
    # happens to produce a replica_count > 1 node. n=98 tile=3x2 was found
    # by sweeping random (n, tile_rows, tile_cols) triples for one whose
    # PRE/POST transpose lands at replica_count=7, grid=3x1 (tiles_per_
    # replica=3, non-power-of-2) -- not a specially-constructed N, just the
    # first hit.
    round_split_cases: list[tuple[str, int, int, int | None, int | None, int | None]] = [
        ("N=960 max_concurrent_scratchpad_bytes forces every kernel to round-split", 960, 512, None, None, 3840),
        ("N=98 tile=3x2 exercises needs_q_table (replica_count=7, grid=3x1)", 98, 196, 3, 2, None),
    ]
    for label, n, budget, tile_rows, tile_cols, cap in round_split_cases:
        for inverse in (False, True):
            err, plan = verify_recursive_plan(
                n, scratchpad_byte_budget=budget, inverse=inverse, seed=53,
                tile_rows=tile_rows, tile_cols=tile_cols, max_concurrent_scratchpad_bytes=cap,
            )
            ok = err <= tolerance
            tag = f"{label} (n={n} budget={budget} inverse={inverse})"
            print(f"    {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
            if not ok:
                failures.append(tag)

    print()
    print("  forced_worker_sequence (per-leaf cooperative workers, real emitted code vs. numpy.fft/ifft):")
    # forced_worker_sequence (fft_plan_recursive._build_recursive_node) lets
    # each leaf pick its own cooperative_workers instead of one uniform
    # choice for the whole tree -- the per-leaf mixing an earlier session
    # discussed but never implemented; fft_plan_search.
    # generate_per_leaf_worker_candidates now builds these for the search.
    # N=960 has exactly 2 leaves (FFTRecNear0 M=30, FFTRecLeaf1 M=32, see
    # this file's own "Debug/summary output (N=960...)" case below) -- both
    # single-leaf-cooperative combinations (only the near leaf, only the
    # far leaf) plus both-cooperative are covered, so this exercises a
    # cooperative leaf sitting next to a plain (one-uthread-per-sub-FFT)
    # one in the very same tree, not just cooperative-vs-not across two
    # entirely separate plans.
    worker_seq_cases: list[tuple[str, int, int, tuple[int | str | None, ...]]] = [
        ("N=960 only the near leaf cooperative", 960, 512, (2, None)),
        ("N=960 only the far leaf cooperative", 960, 512, (None, 2)),
        ("N=960 both leaves cooperative (different worker counts)", 960, 512, (2, 4)),
        ("N=960 near leaf auto, far leaf plain", 960, 512, ("auto", None)),
    ]
    for label, n, budget, seq in worker_seq_cases:
        for inverse in (False, True):
            err, plan = verify_recursive_plan(
                n, scratchpad_byte_budget=budget, inverse=inverse, seed=97,
                forced_worker_sequence=seq,
            )
            ok = err <= tolerance
            tag = f"{label} (n={n} budget={budget} worker_sequence={seq} inverse={inverse})"
            print(f"    {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
            if not ok:
                failures.append(tag)

    print()
    print("  cooperative loop_stages (per-worker runtime loop, real emitted code vs. numpy.fft/ifft):")
    # A cooperative stage looping per-worker (fft_cooperative_codegen.
    # _emit_cooperative_stage's own loop_stages support) is new as of
    # 2026-08-27, added after rendering every worker's own batches fully
    # unrolled into one shared function produced a real N=1024 register-
    # pressure failure on the actual M2NDP-Detour toolchain (7576-line
    # stage_1(), workers_per_fft=8, scratchpad_byte_budget=16384 -- no
    # split at all, so this exact case) -- Python-level correctness alone
    # never could have caught the original bug (it's a real-hardware-only
    # register allocator failure, see docs/STATUS.md), but this closes the
    # coverage gap for the *rendering itself* (the loop/tail selection,
    # residue math, per-worker twiddle-table offsets) all the same, and
    # pins the exact regression case for whoever re-runs run_fft_test.sh
    # against it. N=1024 budget=16384 is a single 5-stage (4,4,4,4,4)
    # cooperative leaf, no split/transpose at all -- every one of its 3
    # middle stages (1,2,3) is exactly what needed to loop to fix the real
    # spill.
    coop_loop_cases: list[tuple[str, int, int, int | str | None]] = [
        ("N=1024 single cooperative leaf, workers=auto (the real regression case)", 1024, 16384, "auto"),
        ("N=1024 single cooperative leaf, workers=4", 1024, 16384, 4),
        ("N=1024 single cooperative leaf, workers=2", 1024, 16384, 2),
    ]
    for label, n, budget, workers in coop_loop_cases:
        for inverse in (False, True):
            err, plan = verify_recursive_plan(
                n, scratchpad_byte_budget=budget, inverse=inverse, seed=101,
                cooperative_workers=workers, compute_lanes=4, narrow_middle_stages=True,
            )
            ok = err <= tolerance
            tag = f"{label} (n={n} budget={budget} workers={workers} inverse={inverse})"
            print(f"    {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
            if not ok:
                failures.append(tag)

    print()
    print("  standalone cooperative leaves (verify_fft_cooperative.verify_cooperative_leaf, real emitted code vs. numpy.fft/ifft):")
    # verify_fft_cooperative.py was never imported by this file (or any
    # other entry point) before this -- confirmed by grep, 2026-08-27: its
    # own run_cooperative_kernel/verify_cooperative_leaf had zero automated
    # coverage of any kind, despite fft_cooperative_codegen.py being a real,
    # separate rendering path this whole suite otherwise exercises
    # thoroughly for the plain (non-cooperative) case. Also found and fixed
    # in the same pass: run_cooperative_kernel carried its own parallel
    # reimplementation of run_kernel's group/local_id exec loop that had
    # silently fallen behind run_kernel's own compute_lanes/narrow_middle_
    # stages/loop_stages support -- every call here now defaults to the
    # *real* make_fft_kernel.py shape (compute_lanes=4, narrow_middle_
    # stages=True, loop_stages=True) instead of the stale full-width/fully-
    # unrolled one, see verify_cooperative_leaf's own docstring.
    #
    # total_ffts here is deliberately > 1 (workers cooperating on *each* of
    # several logical sub-FFTs sharing one launch, not just one) -- the
    # shape make_recursive_transpose_plan's own near_fft leaves actually
    # produce (r*a independent transforms) and the coop_loop_cases above
    # only exercise indirectly (root r=1 there). N=1024/workers=8 is the
    # same real N=1024 register-pressure regression as coop_loop_cases'
    # own first case, built directly through make_cooperative_leaf_plan
    # instead of the recursive tree -- narrower, more direct coverage of
    # exactly the fix in fft_cooperative_codegen.py itself.
    coop_leaf_cases: list[tuple[str, int, tuple[int, ...], int, int]] = [
        ("N=1024 radix (4,4,4,4,4), workers=8, single FFT (the real regression case)", 1024, (4, 4, 4, 4, 4), 8, 1),
        ("N=1024 radix (4,4,4,4,4), workers=8, total_ffts=3", 1024, (4, 4, 4, 4, 4), 8, 3),
        ("N=105 radix (3,5,7), workers=2, total_ffts=6", 105, (3, 5, 7), 2, 6),
        ("N=32 radix (4,4,2), workers=4, total_ffts=5", 32, (4, 4, 2), 4, 5),
    ]
    for label, length, radices, workers_per_fft, total_ffts in coop_leaf_cases:
        for inverse in (False, True):
            err = verify_cooperative_leaf(
                length, radices, workers_per_fft=workers_per_fft, total_ffts=total_ffts,
                inverse=inverse, seed=113,
            )
            ok = err <= tolerance
            tag = (
                f"{label} (length={length} radices={radices} workers_per_fft={workers_per_fft} "
                f"total_ffts={total_ffts} inverse={inverse})"
            )
            print(f"    {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
            if not ok:
                failures.append(tag)

    print()
    print("  compute_lanes-narrowed rendering (real emitted code vs. numpy.fft/ifft):")
    # compute_lanes controls only how wide a vector *instruction* each
    # stage's arithmetic emits (codegen.lowering._chunk_batch) --
    # completely separate from simd_lanes (the launch granule). Every
    # verify_recursive_plan case above leaves it at the default `None`
    # ("render at simd_lanes"), never the narrower width make_fft_kernel.py
    # actually renders by default on this target (min(simd_lanes,
    # target.lmul1_float32_lanes), 4 here -- see that module's own
    # docstring) -- so the whole chunking code path (codegen/lowering.py's
    # chunk_batch/_chunk_load/_chunk_store) was completely unexercised by
    # this suite, regardless of how many times it passed.
    #
    # N=630's default split (A=6/B=105, near leaf radices (3,5,7)) is a
    # confirmed real-hardware regression: `make_fft_kernel.py 630` at
    # compute_lanes=4 (this target's documented "safe" default) fails its
    # own host reference check on the real M2NDP-Detour simulator, while
    # compute_lanes=2/1 both pass -- isolated via a from-scratch harness
    # calling _emit_stage directly (this suite couldn't see it before,
    # since it never threaded compute_lanes through at all). These cases
    # confirm what that isolation already found: the *source semantics*
    # this harness can re-execute are correct at every width tested here
    # too -- if this ever fails, the bug has moved into this layer; while
    # it keeps passing, the still-unresolved N=630 mismatch is confirmed
    # to live below Python (LLVM backend codegen or the simulator's own
    # execution of spill code), not something this suite can catch.
    #
    # N=960's FFTRecNear0.stage_0 is the confirmed real tail-batch case
    # (4 of 8 lanes valid at compute_lanes=4 -- see lowering.py's own
    # comment on _chunk_load) that motivated the vectorized-tail-load fix;
    # included here so its own narrower-chunk rendering has direct
    # numeric coverage, not just the real-hardware spill/cycle-count check.
    compute_lanes_cases: list[tuple[str, int, int, int | None]] = [
        ("N=630 default split, compute_lanes=4 (target's documented default)", 630, 4096, 4),
        ("N=630 default split, compute_lanes=2", 630, 4096, 2),
        ("N=630 default split, compute_lanes=1", 630, 4096, 1),
        ("N=960 tail-batch leaf (FFTRecNear0.stage_0), compute_lanes=4", 960, 32 * 16, 4),
        ("N=960 tail-batch leaf, compute_lanes=2", 960, 32 * 16, 2),
        ("N=960 tail-batch leaf, compute_lanes=1", 960, 32 * 16, 1),
    ]
    for label, n, budget, compute_lanes in compute_lanes_cases:
        for inverse in (False, True):
            err, plan = verify_recursive_plan(
                n, scratchpad_byte_budget=budget, inverse=inverse, seed=71,
                compute_lanes=compute_lanes,
            )
            ok = err <= tolerance
            tag = f"{label} (n={n} budget={budget} compute_lanes={compute_lanes} inverse={inverse})"
            print(f"    {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
            if not ok:
                failures.append(tag)

    print()
    print("  narrow_middle_stages rendering (real emitted code vs. numpy.fft/ifft):")
    # narrow_middle_stages (make_fft_kernel.py's own new default) halves
    # compute_lanes (floor 1) for a stage that is neither its own kernel's
    # first nor last -- see codegen.fft_codegen._stage_compute_lanes's own
    # docstring for why: that's the one shape both the pre-existing N=54
    # radix-9-after-6 spill and this session's real N=630 register-
    # pressure isolation (FFTRecNear0's radix-5 stage_1) share. This only
    # changes *which width* an already-verified code path (_chunk_batch)
    # renders a middle stage at -- every compute_lanes value it can
    # produce is already covered by the sweep above -- so these cases
    # exist to confirm the *selection* itself (is_first/is_last, the
    # halving arithmetic) never picks a width the rest of this suite
    # hasn't already proven correct, at every compute_lanes value that
    # matters in practice (including 1, where floor-1 halving is already
    # a no-op, and the current make_fft_kernel.py default of 4).
    narrow_cases: list[tuple[str, int, int, int | None]] = [
        ("N=630 default split, compute_lanes=4 (make_fft_kernel.py's own default)", 630, 4096, 4),
        ("N=630 default split, compute_lanes=2", 630, 4096, 2),
        ("N=630 default split, compute_lanes=1", 630, 4096, 1),
        ("N=960 tail-batch leaf, compute_lanes=4", 960, 32 * 16, 4),
        ("N=960 tail-batch leaf, compute_lanes=2", 960, 32 * 16, 2),
        ("N=210 2-level recursion, compute_lanes=4", 210, 336, 4),
        ("N=105 single leaf radices (3,5,7), compute_lanes=4", 105, 9999999, 4),
    ]
    for label, n, budget, compute_lanes in narrow_cases:
        for inverse in (False, True):
            err, plan = verify_recursive_plan(
                n, scratchpad_byte_budget=budget, inverse=inverse, seed=83,
                compute_lanes=compute_lanes, narrow_middle_stages=True,
            )
            ok = err <= tolerance
            tag = f"{label} (n={n} budget={budget} compute_lanes={compute_lanes} narrow_middle_stages=True inverse={inverse})"
            print(f"    {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
            if not ok:
                failures.append(tag)

    print()
    print("  Store vectorization (_make_store, direct unit checks):")

    def _layout(*, radix: int, twiddle_lane_divisor: int) -> _StageLayout:
        return _StageLayout(
            radix=radix,
            butterfly_count=64,
            input_batch_width=8,
            input_stride=8,
            output_batch_width=8,
            output_stride=1,
            twiddle_modulus=64,
            twiddle_stride=1,
            twiddle_lane_divisor=twiddle_lane_divisor,
        )

    store_unit_cases: list[tuple[str, StorePlan, str]] = [
        (
            "full-width, p_s >= simd_lanes: contiguous -> vector",
            _make_store(
                last_stage=False, write_buffer="buf", layout=_layout(radix=2, twiddle_lane_divisor=8),
                simd_it=1, output=0, valid_lanes=8, simd_lanes=8,
            ),
            "vector",
        ),
        (
            "full-width, p_s < simd_lanes: not contiguous -> scalar_lanes",
            _make_store(
                last_stage=False, write_buffer="buf", layout=_layout(radix=2, twiddle_lane_divisor=1),
                simd_it=1, output=0, valid_lanes=8, simd_lanes=8,
            ),
            "scalar_lanes",
        ),
        (
            "tail batch, p_s >= simd_lanes (would-be-contiguous): still scalar_lanes",
            _make_store(
                last_stage=False, write_buffer="buf", layout=_layout(radix=2, twiddle_lane_divisor=8),
                simd_it=1, output=0, valid_lanes=5, simd_lanes=8,
            ),
            "scalar_lanes",
        ),
    ]
    for label, store, expected_mode in store_unit_cases:
        ok = store.mode == expected_mode and store.destination == "scratchpad" and store.buffer_name == "buf"
        if store.mode == "vector":
            ok = ok and store.base_offset is not None
        else:
            ok = ok and len(store.lane_offsets) == (5 if "tail" in label else 8)
        tag = f"_make_store: {label}"
        print(f"    {'OK  ' if ok else 'FAIL'} {tag}: mode={store.mode}")
        if not ok:
            failures.append(tag)

    print()
    print("  Store vectorization (_chunk_store, compute_lanes sub-slice promotion, direct unit checks):")
    chunk_store_cases: list[tuple[str, StorePlan, int, int, str, int | None]] = [
        (
            "contiguous first chunk of a non-contiguous full batch -> vector",
            StorePlan(destination="scratchpad", buffer_name="buf", mode="scalar_lanes", lane_offsets=(0, 1, 2, 3, 16, 17, 18, 19)),
            0, 4, "vector", 0,
        ),
        (
            "contiguous second chunk of a non-contiguous full batch -> vector",
            StorePlan(destination="scratchpad", buffer_name="buf", mode="scalar_lanes", lane_offsets=(0, 1, 2, 3, 16, 17, 18, 19)),
            4, 4, "vector", 16,
        ),
        (
            "non-contiguous chunk -> stays scalar_lanes",
            StorePlan(destination="scratchpad", buffer_name="buf", mode="scalar_lanes", lane_offsets=(0, 1, 4, 5, 8, 9, 12, 13)),
            0, 4, "scalar_lanes", None,
        ),
        (
            "chunk reaching past a tail batch's valid_lanes -> stays scalar_lanes (never promoted)",
            StorePlan(destination="scratchpad", buffer_name="buf", mode="scalar_lanes", lane_offsets=(0, 1, 2, 3, 4)),
            4, 4, "scalar_lanes", None,
        ),
        (
            "already-vector store just carries its offset forward",
            StorePlan(destination="scratchpad", buffer_name="buf", mode="vector", base_offset=8),
            4, 4, "vector", 12,
        ),
    ]
    for label, store, offset, width, expected_mode, expected_base in chunk_store_cases:
        chunked = _chunk_store(store, offset, width)
        ok = chunked.mode == expected_mode and chunked.base_offset == expected_base
        tag = f"_chunk_store: {label}"
        print(f"    {'OK  ' if ok else 'FAIL'} {tag}: mode={chunked.mode} base_offset={chunked.base_offset}")
        if not ok:
            failures.append(tag)

    print()
    print("  Radix coalescing (coalesce_radices, direct):")
    # Default policy (allowed=None -> radix-4 pairs only): confirmed via
    # the real Mojo -> llc -> M2NDP-Detour toolchain to be the largest
    # *always-safe* automatic choice -- 6/9/10 (the only other two-prime
    # products from {2,3,5,7,11,13,17} that land back in SUPPORTED_RADICES;
    # reaching 8/16 needs a triple merge this function never attempts) are
    # NOT merged by default because they aren't safe in general: N=54=(6,9)
    # spills its own radix-9 stage (reads 9 complex operands out of
    # scratchpad, a non-first stage) to a `vs1r.v` the simulator doesn't
    # implement, and N=160/320=(4,4,10) spill their radix-10 stage the same
    # way -- both an all-zero-output mismatch on real hardware, not just a
    # slowdown. radix-6/9 *alone* (N=6, N=9 -- trivially the kernel's only
    # stage) are clean, which is exactly why this needed the real toolchain
    # to catch, not just the Python-level numeric harness (see
    # verify_recursive_plan's own docstring on what that harness can and
    # cannot check about register-pressure/ISA-support failures).
    default_coalesce_cases: list[tuple[int, tuple[int, ...]]] = [
        (2, (2,)),
        (4, (4,)),
        (8, (4, 2)),
        (16, (4, 4)),
        (64, (4, 4, 4)),
        (256, (4, 4, 4, 4)),
        (6, (2, 3)),
        (9, (3, 3)),
        (10, (2, 5)),
        (54, (2, 3, 3, 3)),
        (160, (4, 4, 2, 5)),
        (320, (4, 4, 4, 5)),
    ]
    for n, expected in default_coalesce_cases:
        factors = _prime_factors_supported(n)
        got = coalesce_radices(factors)
        ok = (
            got == expected
            and prod(got) == n
            and all(r in SUPPORTED_RADICES for r in got)
        )
        tag = f"coalesce_radices(_prime_factors_supported({n})) == {expected}"
        print(f"    {'OK  ' if ok else 'FAIL'} {tag}: got {got}")
        if not ok:
            failures.append(tag)

    # The wider (allowed=SUPPORTED_RADICES) policy: available, opt-in, and
    # numerically correct on its own terms -- N=6/9/10 above already prove
    # (2,3)->6/(3,3)->9/(2,5)->10 are each individually valid supported
    # radices -- it's specifically the *default*'s job to stay off of it
    # until 6/9/10 are confirmed spill-free in the general (non-sole-stage)
    # case, not this function's.
    wide_coalesce_cases: list[tuple[int, tuple[int, ...]]] = [
        (6, (6,)), (9, (9,)), (10, (10,)),
    ]
    for n, expected in wide_coalesce_cases:
        got = coalesce_radices(_prime_factors_supported(n), allowed=SUPPORTED_RADICES)
        ok = got == expected and prod(got) == n
        tag = f"coalesce_radices(..., allowed=SUPPORTED_RADICES) for n={n} == {expected}"
        print(f"    {'OK  ' if ok else 'FAIL'} {tag}: got {got}")
        if not ok:
            failures.append(tag)

    # General invariants across every N reachable from supported primes up
    # to 300: product matches, every radix is supported, and -- the one
    # that actually proves "no reordering, decomposition semantics
    # preserved" -- expanding each output radix back into its own prime
    # factorization and concatenating reproduces the original factor list
    # exactly, in order.
    invariant_failures: list[str] = []
    for n in range(2, 301):
        try:
            factors = _prime_factors_supported(n)
        except ValueError:
            continue
        got = coalesce_radices(factors)
        if prod(got) != n:
            invariant_failures.append(f"n={n}: product(coalesce_radices(...))={prod(got)} != {n}")
            continue
        if not all(r in SUPPORTED_RADICES for r in got):
            invariant_failures.append(f"n={n}: {got} has a radix outside SUPPORTED_RADICES")
            continue
        expanded = tuple(f for r in got for f in _prime_factors_supported(r))
        if expanded != tuple(factors):
            invariant_failures.append(
                f"n={n}: re-expanding {got} gives {expanded}, not the original {factors} -- reordered!"
            )
    tag = "coalesce_radices invariants (product/support/order) for n=2..300"
    print(f"    {'OK  ' if not invariant_failures else 'FAIL'} {tag}")
    if invariant_failures:
        for msg in invariant_failures[:5]:
            print(f"      {msg}")
        failures.append(tag)

    print()
    print("  Store vectorization + radix coalescing together (N=256, radix-4 chain, real emitted code -- per-stage store mode):")
    # Frozen against the actual generated Mojo (not just the planner's own
    # mode -- fft_transpose_codegen.generate_recursive_fft_kernels defaults
    # to loop_stages=True, whose own store-vectorization lives in
    # fft_codegen._emit_store_loop, a separate renderer from the non-loop
    # _chunk_store path _make_store's promotion alone would suggest -- see
    # that function's own comment): stage 0 (p_s=1) can never be
    # contiguous, so it stays fully scalar; stage 1 (p_s=4) is not
    # contiguous at the full simd_lanes=8 width but *is* at compute_lanes=4
    # (a chunk lines up exactly on one p_s block), so _emit_store_loop's
    # own chunk-uniform-stride check promotes it; stage 2/3 (p_s=16/64,
    # already >= simd_lanes) are vector from _make_store directly. Radix-4
    # chain (not radix-2 x8) because coalesce_radices is now wired into
    # _build_recursive_node -- this test only makes sense with both
    # optimizations active together, which is exactly what it's checking.
    # compute_lanes=4 matches make_fft_kernel.py's own default (this
    # target's LMUL=1 width -- see its module docstring); generate_recursive
    # _fft_kernels' own default (None, unchunked at simd_lanes=8) is a
    # different configuration nothing in production actually uses.
    n256_plan = make_recursive_transpose_plan(256, scratchpad_byte_budget=4096)
    n256_source = generate_recursive_fft_kernels(n256_plan, compute_lanes=4)
    n256_expected_stores = {0: (0, 64), 1: (16, 0), 2: (32, 0), 3: (16, 0)}  # stage -> (vector, scalar)
    lines = n256_source.splitlines()
    kernel_start = next(i for i, l in enumerate(lines) if l.startswith("struct FFTRecLeaf0(NDPTask):"))
    for stage_id, (expected_vector, expected_scalar) in n256_expected_stores.items():
        start = next(
            i for i in range(kernel_start, len(lines)) if lines[i].strip() == f"def stage_{stage_id}():"
        )
        end = next(
            (i for i in range(start + 1, len(lines)) if lines[i].strip() == "@staticmethod"), len(lines)
        )
        block = lines[start:end]
        store_lines = [l for l in block if ".store(" in l]
        scalar_count = sum(1 for l in store_lines if l.rstrip().endswith("])"))
        vector_count = len(store_lines) - scalar_count
        ok = vector_count == expected_vector and scalar_count == expected_scalar
        tag = f"N=256 stage_{stage_id}: {expected_vector} vector / {expected_scalar} scalar store lines"
        print(f"    {'OK  ' if ok else 'FAIL'} {tag}: got {vector_count} vector / {scalar_count} scalar")
        if not ok:
            failures.append(tag)

    # deep (3-level) recursion with a tile shape that forces both row and
    # column tail branches, forward + inverse.
    n_deep = 2 * 3 * 5 * 7 * 11
    for inverse in (False, True):
        err, plan = verify_recursive_plan(
            n_deep, scratchpad_byte_budget=11 * 16, inverse=inverse, seed=29,
            tile_rows=3, tile_cols=4,
        )
        ok = err <= tolerance
        tag = f"deep recursive plan n={n_deep} tile=3x4 inverse={inverse}"
        print(f"    {'OK  ' if ok else 'FAIL'} {tag}: max error {err:.3e}")
        if not ok:
            failures.append(tag)

    print()
    print("  Index-only checks for a large N (numeric path too slow to be worth running):")
    for n_large, budget_large in [(2 ** 16, 256 * 16)]:
        plan = make_recursive_transpose_plan(n_large, scratchpad_byte_budget=budget_large)
        tag = f"index-only bijection + access-shape n={n_large}"
        try:
            verify_recursive_tree_index_only(plan.root)
            print(f"    OK   {tag}")
        except AssertionError as exc:
            print(f"    FAIL {tag}: {exc}")
            failures.append(tag)

    print()
    print("  Debug/summary output (N=960, one split level):")
    # Both plans below were already fully numerically verified in the
    # recursive_cases loop above -- build them directly here rather than
    # re-running verify_recursive_plan's own (expensive) full numeric
    # chain a second time just to get a plan object to summarize.
    summarize_recursive_plan(make_recursive_transpose_plan(960, scratchpad_byte_budget=32 * 16))
    print()
    print("  Debug/summary output (N=210, 2-level recursion):")
    summarize_recursive_plan(make_recursive_transpose_plan(210, scratchpad_byte_budget=21 * 16))
    print()

    if failures:
        raise AssertionError(f"{len(failures)} plan(s) failed: {failures}")
    print("[verify] all FFT plans matched numpy's FFT")

    from verification.verify_fft_search import main as verify_fft_search_main
    print()
    verify_fft_search_main()

    from verification.verify_fft_execution_cost import main as verify_fft_execution_cost_main
    print()
    verify_fft_execution_cost_main()


if __name__ == "__main__":
    main()
