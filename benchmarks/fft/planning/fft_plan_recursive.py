from __future__ import annotations

"""Generalizes make_balanced_transpose_plan's (fft_plan_balanced.py) single
2-way split to a full recursive (six-step-FFT-style) decomposition: FFT
chunk size and physical transpose tile size are two completely independent
choices (see PhysicalTransposePlan's own docstring). Additive: does not
touch make_multi_kernel_plan, make_balanced_plan, make_balanced_transpose_
plan, or any AddressMapping kind -- those remain the flat/2-way regression
baseline and numeric reference.

One recursive node FFTNode(M, R) owns exactly the contract "R independent
M-point transforms, natural contiguous order in (addr = q*M + n) both on
entry and on exit" -- see FFTLeafPlan/FFTRecursiveNodePlan. A node is
either a leaf (M fits one fused multi-radix kernel) or splits M = A*B and
emits PRE transpose -> B-point FFT (batched R*A times) -> MIDDLE transpose
with the current node's own W_M twiddle -> recursive FFTNode(A, R*B) ->
POST transpose. Every junction address was independently re-derived and
cross-checked (not just transcribed) against this contract before being
implemented, then verified again by direct index/numeric simulation
(bijection of the generic tiled transpose; multi-level recursion vs.
numpy.fft) before any codegen was written -- see verify_fft_plan.py.
"""

from dataclasses import dataclass
from typing import Union

from planning.fft_plan_core import (
    AddressMapping,
    FFTCodegenPlan,
    MultiKernelHostPlan,
    _build_plan,
    _cap_max_uthread,
    _prime_factors_supported,
    coalesce_radices,
    layouts_for_radices,
    pingpong_needed,
)
from planning.fft_plan_cooperative import choose_workers_per_fft, make_cooperative_leaf_plan


@dataclass(frozen=True)
class PhysicalTransposePlan:
    """One standalone tiled-transpose kernel: `replica_count` independent
    `rows x cols` matrices, each transposed to `cols x rows`, tile-wise.
    Completely independent of any FFT chunk/radix length -- `rows`/`cols`
    here are PRE/POST's (B,A) or MIDDLE's (A,B), never `ki_near`/`ki_far`.

    `twiddle_modulus`: None for PRE/POST (plain transpose). For MIDDLE,
    the *current recursive node's own* M (never the top-level N) -- see
    AddressMappingKind-adjacent docstrings elsewhere in this module for why
    a per-level modulus, not a global one, is what makes 3+-factor
    decomposition correct. The twiddle table is dense, `rows*cols` (=M)
    elements, addressed identically to this kernel's own source matrix
    (`table[r*cols+c]`) -- no per-replica duplication (see fft_transpose_
    codegen.py's precompute).

    `apply_inverse_scale`: True only for the *root* node's own POST
    transpose (or the root leaf, if the whole FFT fits in one kernel) when
    `inverse` -- the overall 1/N lands exactly once, fused into the final
    store, not per-node.

    One microthread owns one whole `tile_rows x tile_cols` physical tile
    (independent of `simd_lanes` up to this plan's own choice -- default is
    `min(simd_lanes, rows/cols)`, see make_recursive_transpose_plan). Every
    DRAM vector load/store is unit-stride (contiguous `tile_cols`-wide rows
    on the source side, `tile_rows`-wide rows on the destination side);
    row-to-row jumps are allowed and expected. Tail tiles (rows/cols not a
    multiple of tile_rows/tile_cols) are masked, never out-of-bounds -- see
    fft_transpose_codegen.py.
    """

    kernel_name: str
    rows: int
    cols: int
    replica_count: int
    tile_rows: int
    tile_cols: int
    grid_rows: int
    grid_cols: int
    total_uthreads: int
    max_uthread: int
    scratchpad_elements: int  # per-uthread: 2 * tile_rows * tile_cols
    twiddle_modulus: int | None
    inverse: bool
    simd_lanes: int
    apply_inverse_scale: bool = False


@dataclass(frozen=True)
class FFTLeafPlan:
    """FFTNode(m, r) that fits one fused multi-radix kernel outright (no
    transpose boundary at all) -- `kernel` is an ordinary FFTCodegenPlan
    with `length=m`, `total_uthreads=r`, both DRAM mappings plain
    `contiguous(row_stride=m)`, built via the existing layouts_for_radices/
    _build_plan exactly as every other single-kernel FFT in this module.
    """

    m: int
    r: int
    kernel: FFTCodegenPlan


@dataclass(frozen=True)
class FFTRecursiveNodePlan:
    """FFTNode(m, r) with m = a*b, too large for one fused kernel: PRE
    transpose -> near_fft (b-point, batched r*a times) -> middle_transpose
    (W_m twiddle fused) -> far_child (FFTNode(a, r*b)) -> post_transpose.
    See the module-level design writeup for the full derivation of why
    this exact shape sequence (b x a -> a x b -> [twiddle] -> b x a ->
    a x b) reproduces FFTNode's own natural-order contract at every
    junction, cross-checked address by address before implementation.
    """

    m: int
    r: int
    a: int
    b: int
    pre_transpose: PhysicalTransposePlan
    near_fft: FFTLeafPlan
    middle_transpose: PhysicalTransposePlan
    far_child: "FFTNode"
    post_transpose: PhysicalTransposePlan


FFTNode = Union[FFTLeafPlan, FFTRecursiveNodePlan]


@dataclass(frozen=True)
class RecursiveFFTPlan:
    n: int
    inverse: bool
    root: FFTNode
    host: MultiKernelHostPlan


def _choose_recursive_split(
    m: int, *, scratchpad_byte_budget: int
) -> int | None:
    """Returns b (the near_fft's own length) if m needs splitting, or None
    if m already fits one fused leaf kernel outright. b is chosen as the
    *largest* suffix-factor-product of m's own supported prime
    factorization that still fits scratchpad_byte_budget as a single
    leaf -- larger b means fewer recursion levels, fewer transpose kernels,
    and a leaf that fuses as many radix stages as it can (see the module
    design writeup's own reasoning for this heuristic; candidate
    generation is kept in this one function so a benchmark-driven cost
    model can replace just this later, same discipline
    factor_into_kernel_chunks's own docstring already established).
    """
    if scratchpad_byte_budget <= 0:
        raise ValueError("scratchpad_byte_budget must be positive")
    cap = scratchpad_byte_budget // 16
    if m <= cap:
        return None

    factors = _prime_factors_supported(m)
    suffix = [1] * (len(factors) + 1)
    for i in range(len(factors) - 1, -1, -1):
        suffix[i] = suffix[i + 1] * factors[i]

    for i in range(1, len(factors) + 1):
        if suffix[i] <= cap:
            if suffix[i] <= 1:
                raise ValueError(
                    f"scratchpad_byte_budget={scratchpad_byte_budget} is too "
                    f"small to make progress on m={m} (factors={factors}: "
                    f"not even the smallest one fits)"
                )
            return suffix[i]
    raise ValueError(
        f"scratchpad_byte_budget={scratchpad_byte_budget} is too small "
        f"for m={m} (factors={factors})"
    )


def _build_leaf_kernel(
    *,
    length: int,
    radices: tuple[int, ...],
    total_uthreads: int,
    simd_lanes: int,
    inverse: bool,
    inverse_scale: float | None,
    kernel_name: str,
    spad_capacity_bytes: int | None,
    max_concurrent_scratchpad_bytes: int | None,
    cooperative_workers: int | str | None,
) -> FFTCodegenPlan:
    """One leaf/near_fft kernel -- `_build_plan` (today's one-uthread-per-
    sub-FFT leaf) or `make_cooperative_leaf_plan` (see fft_plan_cooperative.py),
    picked by `cooperative_workers`:

    * `None` (the default): always `_build_plan`, byte-for-byte the plan
      this function returned before cooperative leaves existed.
    * `"auto"`: `choose_workers_per_fft` decides this leaf's own worker
      count from its own shape (length/radices/simd_lanes) alone.
    * a positive int: an upper bound `choose_workers_per_fft` still rounds
      down to a divisor of the hardware's own interleave chunk that fits
      this leaf's own busiest-stage batch count (see that function's own
      docstring) -- never a raw override, since a value it can't actually
      use safely would silently do nothing useful (or, worse, break the
      local/global grouping alignment `_emit_cooperative_stage` depends on).

    Either way, `choose_workers_per_fft` can decide `workers=1` is this
    leaf's own best answer (e.g. a leaf too small to have more than one
    SIMD batch in its busiest stage) -- `_build_plan` is used then too,
    identical output to `cooperative_workers=None`, so a caller opting in
    never pays for cooperation where it cannot help.
    """
    workers = 1
    if cooperative_workers is not None:
        max_workers = None if cooperative_workers == "auto" else cooperative_workers
        workers = choose_workers_per_fft(length, radices, simd_lanes=simd_lanes, max_workers=max_workers)

    if workers > 1:
        return make_cooperative_leaf_plan(
            length=length,
            radices=radices,
            workers_per_fft=workers,
            total_ffts=total_uthreads,
            inverse=inverse,
            simd_lanes=simd_lanes,
            kernel_name=kernel_name,
            inverse_scale=inverse_scale,
            spad_capacity_bytes=spad_capacity_bytes,
            max_concurrent_scratchpad_bytes=max_concurrent_scratchpad_bytes,
        )

    return _build_plan(
        length=length,
        inverse=inverse,
        total_uthreads=total_uthreads,
        simd_lanes=simd_lanes,
        use_pingpong=pingpong_needed(len(radices)),
        layouts=layouts_for_radices(length, radices, simd_lanes),
        kernel_name=kernel_name,
        input_mapping=AddressMapping.contiguous(row_stride=length),
        output_mapping=AddressMapping.contiguous(row_stride=length),
        large_twiddle=None,
        inverse_scale=inverse_scale,
        spad_capacity_bytes=spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=max_concurrent_scratchpad_bytes,
    )


def _build_physical_transpose(
    *,
    rows: int,
    cols: int,
    replica_count: int,
    tile_rows: int,
    tile_cols: int,
    twiddle_modulus: int | None,
    inverse: bool,
    kernel_name: str,
    simd_lanes: int,
    spad_capacity_bytes: int | None,
    apply_inverse_scale: bool,
    max_concurrent_scratchpad_bytes: int | None = None,
) -> PhysicalTransposePlan:
    grid_rows = -(-rows // tile_rows)
    grid_cols = -(-cols // tile_cols)
    total_uthreads = replica_count * grid_rows * grid_cols
    scratchpad_elements = 2 * tile_rows * tile_cols

    max_uthread = _cap_max_uthread(
        total_uthreads, scratchpad_elements * 4, spad_capacity_bytes,
        context=f"a single {tile_rows}x{tile_cols} transpose tile",
        max_concurrent_scratchpad_bytes=max_concurrent_scratchpad_bytes,
    )

    return PhysicalTransposePlan(
        kernel_name=kernel_name,
        rows=rows,
        cols=cols,
        replica_count=replica_count,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        total_uthreads=total_uthreads,
        max_uthread=max_uthread,
        scratchpad_elements=scratchpad_elements,
        twiddle_modulus=twiddle_modulus,
        inverse=inverse,
        simd_lanes=simd_lanes,
        apply_inverse_scale=apply_inverse_scale,
    )


def _build_recursive_node(
    m: int,
    r: int,
    *,
    scratchpad_byte_budget: int,
    simd_lanes: int,
    inverse: bool,
    spad_capacity_bytes: int | None,
    tile_rows: int | None,
    tile_cols: int | None,
    is_root: bool,
    node_id: list[int],
    max_concurrent_scratchpad_bytes: int | None = None,
    cooperative_workers: int | str | None = None,
) -> FFTNode:
    idx = node_id[0]
    node_id[0] += 1

    split_b = _choose_recursive_split(m, scratchpad_byte_budget=scratchpad_byte_budget)
    if split_b is None:
        radices = coalesce_radices(_prime_factors_supported(m))
        inverse_scale = (1.0 / m) if (inverse and is_root) else None
        kernel = _build_leaf_kernel(
            length=m,
            radices=radices,
            total_uthreads=r,
            simd_lanes=simd_lanes,
            inverse=inverse,
            inverse_scale=inverse_scale,
            kernel_name=f"FFTRecLeaf{idx}",
            spad_capacity_bytes=spad_capacity_bytes,
            max_concurrent_scratchpad_bytes=max_concurrent_scratchpad_bytes,
            cooperative_workers=cooperative_workers,
        )
        return FFTLeafPlan(m=m, r=r, kernel=kernel)

    b = split_b
    a = m // b
    tr = tile_rows if tile_rows is not None else min(simd_lanes, b, a)
    tc = tile_cols if tile_cols is not None else min(simd_lanes, a, b)
    tr = max(tr, 1)
    tc = max(tc, 1)

    pre = _build_physical_transpose(
        rows=b, cols=a, replica_count=r, tile_rows=tr, tile_cols=tc,
        twiddle_modulus=None, inverse=inverse, kernel_name=f"FFTRecPre{idx}",
        simd_lanes=simd_lanes, spad_capacity_bytes=spad_capacity_bytes,
        apply_inverse_scale=False, max_concurrent_scratchpad_bytes=max_concurrent_scratchpad_bytes,
    )

    near_radices = coalesce_radices(_prime_factors_supported(b))
    near_kernel = _build_leaf_kernel(
        length=b,
        radices=near_radices,
        total_uthreads=r * a,
        simd_lanes=simd_lanes,
        inverse=inverse,
        inverse_scale=None,
        kernel_name=f"FFTRecNear{idx}",
        spad_capacity_bytes=spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=max_concurrent_scratchpad_bytes,
        cooperative_workers=cooperative_workers,
    )
    near_fft = FFTLeafPlan(m=b, r=r * a, kernel=near_kernel)

    middle = _build_physical_transpose(
        rows=a, cols=b, replica_count=r, tile_rows=tr, tile_cols=tc,
        twiddle_modulus=m, inverse=inverse, kernel_name=f"FFTRecMid{idx}",
        simd_lanes=simd_lanes, spad_capacity_bytes=spad_capacity_bytes,
        apply_inverse_scale=False, max_concurrent_scratchpad_bytes=max_concurrent_scratchpad_bytes,
    )

    far_child = _build_recursive_node(
        a, r * b, scratchpad_byte_budget=scratchpad_byte_budget,
        simd_lanes=simd_lanes, inverse=inverse,
        spad_capacity_bytes=spad_capacity_bytes, tile_rows=tile_rows,
        tile_cols=tile_cols, is_root=False, node_id=node_id,
        max_concurrent_scratchpad_bytes=max_concurrent_scratchpad_bytes,
        cooperative_workers=cooperative_workers,
    )

    post = _build_physical_transpose(
        rows=b, cols=a, replica_count=r, tile_rows=tr, tile_cols=tc,
        twiddle_modulus=None, inverse=inverse, kernel_name=f"FFTRecPost{idx}",
        simd_lanes=simd_lanes, spad_capacity_bytes=spad_capacity_bytes,
        apply_inverse_scale=(inverse and is_root), max_concurrent_scratchpad_bytes=max_concurrent_scratchpad_bytes,
    )

    return FFTRecursiveNodePlan(
        m=m, r=r, a=a, b=b, pre_transpose=pre, near_fft=near_fft,
        middle_transpose=middle, far_child=far_child, post_transpose=post,
    )


def make_recursive_transpose_plan(
    n: int,
    *,
    scratchpad_byte_budget: int,
    simd_lanes: int = 8,
    tile_rows: int | None = None,
    tile_cols: int | None = None,
    inverse: bool = False,
    spad_capacity_bytes: int | None = None,
    max_concurrent_scratchpad_bytes: int | None = None,
    cooperative_workers: int | str | None = None,
) -> RecursiveFFTPlan:
    """N decomposed recursively (six-step-FFT style): each node either
    fuses into one multi-radix leaf kernel (see FFTLeafPlan) or splits
    M=A*B and emits PRE transpose -> B-point FFT -> MIDDLE transpose
    (W_M twiddle) -> recursive FFTNode(A, R*B) -> POST transpose (see
    FFTRecursiveNodePlan). Physical transpose tile size (tile_rows/
    tile_cols, default min(simd_lanes, the matrix's own two dimensions))
    is chosen completely independently of any FFT chunk length -- see the
    module design writeup. Additive: make_multi_kernel_plan/
    make_balanced_plan/make_balanced_transpose_plan are untouched.

    `max_concurrent_scratchpad_bytes`: see `_cap_max_uthread` -- caps `max_uthread`
    by microthread count, independent of `spad_capacity_bytes`'s byte cap.
    `None` (the default) applies none, unchanged from before this
    parameter existed.

    `cooperative_workers`: `None` (the default) keeps every leaf/near_fft
    exactly what `_build_plan` alone produced before cooperative leaves
    existed -- see `_build_leaf_kernel`'s own docstring for `"auto"` and a
    fixed-int cap. `PhysicalTransposePlan` (PRE/MIDDLE/POST) is never
    affected -- this only changes FFTLeafPlan's own internal execution
    granularity, per-node, exactly as fft_plan_cooperative.py's module
    docstring scopes it.
    """
    node_id = [0]
    root = _build_recursive_node(
        n, 1, scratchpad_byte_budget=scratchpad_byte_budget,
        simd_lanes=simd_lanes, inverse=inverse,
        spad_capacity_bytes=spad_capacity_bytes, tile_rows=tile_rows,
        tile_cols=tile_cols, is_root=True, node_id=node_id,
        max_concurrent_scratchpad_bytes=max_concurrent_scratchpad_bytes,
        cooperative_workers=cooperative_workers,
    )
    host = MultiKernelHostPlan(n=n, inverse=inverse, tolerance=1.0e-3)
    return RecursiveFFTPlan(n=n, inverse=inverse, root=root, host=host)
