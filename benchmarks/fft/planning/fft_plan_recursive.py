from __future__ import annotations

import math

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
from planning.fft_plan_persistent import make_persistent_leaf_plan
from planning.target_profile import DEFAULT_TARGET_PROFILE


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

    `needs_q_table`/`needs_tr_table`: whether `tile_id`'s own decomposition
    (`_emit_physical_transpose_stage`'s `q`/`t_r`) needs a plan-time-built
    lookup table instead of a plain runtime `//`/`%` -- true exactly when
    that division is both non-trivial (more than one replica/grid column;
    otherwise the quotient is always 0) and by a non-power-of-2 constant.
    LLVM's usual move for a non-power-of-2 constant divisor -- multiply by
    its reciprocal instead of a real divide -- emits `mulhsu`, an opcode
    M2NDP-Detour's decoder does not implement (confirmed: N=960's recursion
    produces a non-power-of-2 tiles_per_replica and panics without this).
    `needs_round_split`: whether this stage's own total_uthreads exceeds
    what max_uthread's scratchpad capacity fits in one `.launch()`, so
    fft_transpose_codegen.py's host loop needs more than one round. All
    three are pure functions of this plan's own already-decided fields
    (never a new decision -- see this dataclass's other fields), computed
    once here rather than re-derived from codegen at render time.
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
    needs_q_table: bool
    needs_tr_table: bool
    needs_round_split: bool
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
    # How many independent length-n transforms this plan's own root node
    # covers -- root.r itself already carries this (FFTNode(M, R)'s own
    # contract: "R independent M-point transforms", threaded multiplicatively
    # through every recursion level, so every leaf/transpose kernel this
    # tree renders is already sized for it with zero further plan changes --
    # see make_recursive_transpose_plan's own `batch` parameter). Kept as
    # its own explicit field anyway rather than making every caller re-derive
    # it from root.r (which isn't even always a plain int -- root can be a
    # leaf with no further nesting): host-side buffer sizing/reference-check
    # (generate_recursive_fft_kernels) needs this number directly.
    batch: int = 1


def flatten_recursive_node(node: FFTNode) -> list:
    """Flat, ordered stage list for a recursive FFTNode: leaf -> [kernel];
    node -> [pre, near_fft.kernel, middle] + flatten(far_child) + [post].
    Each element is either an FFTCodegenPlan (an ordinary FFT kernel) or a
    PhysicalTransposePlan (a standalone transpose kernel) -- a pure tree
    walk over already-decided plan data, no new decisions, so it lives in
    the planning layer alongside FFTNode/RecursiveFFTPlan themselves rather
    than in codegen (fft_transpose_codegen.generate_recursive_fft_kernels
    and planning/fft_cost_model.py's own metrics both need this same flat
    view -- one source, not a codegen-owned helper cost estimation would
    otherwise have to import backwards for)."""
    if isinstance(node, FFTLeafPlan):
        return [node.kernel]
    assert isinstance(node, FFTRecursiveNodePlan)
    return (
        [node.pre_transpose, node.near_fft.kernel, node.middle_transpose]
        + flatten_recursive_node(node.far_child)
        + [node.post_transpose]
    )


def _leaf_scratchpad_bytes(m: int) -> int:
    """Bytes a single leaf-uthread of length m actually needs for its own
    intermediate scratchpad buffer(s): 0 for a one-stage fused kernel (no
    intermediate at all), 8*m for two stages (pingpong_needed is False --
    one shared buffer), 16*m for three or more (both ping-pong banks).
    Mirrors _build_plan's own sizing exactly -- scratchpad_stride = 2*length,
    bytes_per_uthread = len(buffer_names) * scratchpad_stride * 4, from
    _scratchpad_buffer_names/pingpong_needed in fft_plan_core.py -- so a
    candidate leaf is judged by what it will actually cost, not a flat
    always-ping-pong assumption.
    """
    stage_count = len(coalesce_radices(_prime_factors_supported(m)))
    if stage_count <= 1:
        return 0
    buffers = 2 if pingpong_needed(stage_count) else 1
    return buffers * 2 * m * 4


def _recursive_split_candidates(
    m: int, *, scratchpad_byte_budget: int
) -> list[int]:
    """Every legal near_fft length b for a split of m -- every suffix-
    factor-product of m's own supported prime factorization whose actual
    scratchpad footprint (_leaf_scratchpad_bytes) fits scratchpad_byte_
    budget as a single leaf, largest first. `_choose_recursive_split`
    (below) is the single-answer caller every existing recursive-node
    builder still uses (candidates[0], today's exact "largest that fits"
    heuristic, unchanged); `planning/fft_plan_search.py`'s
    generate_split_candidates is the multi-answer caller that wants the
    whole list instead of just the first -- both go through this one
    function so there is exactly one source of legal-split logic, per
    factor_into_kernel_chunks's own docstring established discipline.

    Raises if not even the smallest candidate fits scratchpad_byte_budget
    (mirrors _choose_recursive_split's own prior error, since a caller with
    an empty list otherwise has to invent this diagnosis on its own).
    """
    if scratchpad_byte_budget <= 0:
        raise ValueError("scratchpad_byte_budget must be positive")

    factors = _prime_factors_supported(m)
    suffix = [1] * (len(factors) + 1)
    for i in range(len(factors) - 1, -1, -1):
        suffix[i] = suffix[i + 1] * factors[i]

    candidates = [
        suffix[i]
        for i in range(1, len(factors) + 1)
        if suffix[i] > 1 and _leaf_scratchpad_bytes(suffix[i]) <= scratchpad_byte_budget
    ]
    if not candidates:
        raise ValueError(
            f"scratchpad_byte_budget={scratchpad_byte_budget} is too "
            f"small to make progress on m={m} (factors={factors}: "
            f"not even the smallest one fits)"
        )
    return candidates


def _choose_recursive_split(
    m: int, *, scratchpad_byte_budget: int
) -> int | None:
    """Returns b (the near_fft's own length) if m needs splitting, or None
    if m already fits one fused leaf kernel outright. b is chosen as the
    *largest* legal candidate from _recursive_split_candidates -- larger b
    means fewer recursion levels, fewer transpose kernels, and a leaf that
    fuses as many radix stages as it can (see the module design writeup's
    own reasoning for this heuristic; this one heuristic pick is kept
    separate from candidate generation itself so a benchmark-driven search
    can consider the other candidates too, see fft_plan_search.py).
    """
    if scratchpad_byte_budget <= 0:
        raise ValueError("scratchpad_byte_budget must be positive")
    if _leaf_scratchpad_bytes(m) <= scratchpad_byte_budget:
        return None
    return _recursive_split_candidates(m, scratchpad_byte_budget=scratchpad_byte_budget)[0]


def _enumerate_leaf_segmentations(
    m: int, *, scratchpad_byte_budget: int, max_segmentations: int = 200
) -> list[tuple[int, ...]]:
    """Every full way to cut m's own supported-prime factorization
    (`_prime_factors_supported`, ascending order) into a chain of
    consecutive leaf-sized segments -- one entry per recursion level,
    near_fft-first, exactly the order `_build_recursive_node`'s own
    near_fft-then-recurse-into-far_child walk consumes them in (see
    `forced_split_sequence` there). The *final* base leaf (whatever
    remains once nothing more needs splitting) is never itself an entry --
    an exhausted tuple already means "stop, build the current remainder as
    one leaf", exactly `forced_split_sequence`'s own empty-tuple case --
    so a tuple's own entries never multiply back to `m`, only to
    `m / (the final base leaf's own length)`.

    `_choose_recursive_split` only ever considers ONE level at a time,
    greedily: "largest legal near_fft for *this* m", independent of how
    that choice shapes every level below it. This instead enumerates every
    JOINT choice across all levels at once, so a cost-model-driven search
    (see fft_plan_search.py) can score whole trees against each other --
    including a tree that splits further even where a level's own
    remainder already fits one leaf outright, since this project's own
    benchmark_fft_candidates.sh runs have shown that smaller/more-numerous
    pieces sometimes run faster despite "fewer, bigger leaves" being the
    existing heuristic's own reasoning (see fft_cost_model.CostWeights.
    transpose_tile_count's own comment).

    Because only the far_child ever recurses (near_fft is always resolved
    as an immediate leaf, never split again -- see FFTRecursiveNodePlan's
    own docstring), the whole tree is fully described by which *suffix* of
    the remaining factor list becomes each level's near_fft, in order --
    equivalently, a set of cut points over the factor index range
    `[0, len(factors))`. Enumerated by walking that range from the high
    end down; `segment_fits(j, i)` (does factors[j:i]'s own product fit
    one leaf) is the only legality check, mirroring
    `_recursive_split_candidates`'s own `_leaf_scratchpad_bytes` test.

    `max_segmentations`: a hard cap on how many full segmentations this
    returns (stops enumerating once reached, not a random sample) -- purely
    a combinatorial-explosion guard (see fft_plan_search.py's module
    docstring on the same concern for its own one-axis-at-a-time sweeps):
    a factor list with many small entries (e.g. N=2**20) has a branching
    choice at nearly every index, and this walk is otherwise exponential
    in the factor count. Every N this project has actually benchmarked
    (a handful to ~10 factors) stays far below this cap unhit.
    """
    factors = _prime_factors_supported(m)
    num_factors = len(factors)

    # suffix_product[i] = product of factors[i:] -- a segment factors[j:i]
    # (j < i) has product suffix_product[j] // suffix_product[i].
    suffix_product = [1] * (num_factors + 1)
    for i in range(num_factors - 1, -1, -1):
        suffix_product[i] = suffix_product[i + 1] * factors[i]

    fits_cache: dict[int, bool] = {}

    def segment_fits(j: int, i: int) -> bool:
        product = suffix_product[j] // suffix_product[i]
        cached = fits_cache.get(product)
        if cached is None:
            cached = _leaf_scratchpad_bytes(product) <= scratchpad_byte_budget
            fits_cache[product] = cached
        return cached

    results: list[tuple[int, ...]] = []

    def walk(i: int, acc: tuple[int, ...]) -> None:
        if len(results) >= max_segmentations:
            return
        if segment_fits(0, i):
            # The whole remaining prefix fits as one final (unsplit) leaf
            # -- _choose_recursive_split's own base case.
            results.append(acc)
            if len(results) >= max_segmentations:
                return
        # j must stay >= 1: near_fft = factors[j:i] can never consume the
        # *entire* remaining prefix (j == 0) as a "split" -- that would
        # leave far_child a length-1 FFT, which make_fft_kernel already
        # rejects outright (n < 2). j == 0 is exactly the "stop" case above.
        #
        # Largest segment (smallest j) first, matching _choose_recursive_
        # split's own "largest legal near_fft" preference: with
        # max_segmentations capping a possibly-huge space, this ordering
        # front-loads results near today's existing heuristic (shallow
        # trees) before the deep, many-tiny-leaves tail, rather than the
        # reverse -- smallest-segment-first would fill the whole cap with
        # maximally-deep recursions (near_fft = a single factor, every
        # level) before this function ever returns anything resembling
        # today's default.
        for j in range(1, i):
            if segment_fits(j, i):
                walk(j, acc + (suffix_product[j] // suffix_product[i],))

    walk(num_factors, ())
    return results


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
    interleave_chunk_uthreads: int,
    persistent_leaf: bool = False,
) -> FFTCodegenPlan:
    """One leaf/near_fft kernel -- `_build_plan` (today's one-uthread-per-
    sub-FFT leaf), `make_cooperative_leaf_plan` (see
    fft_plan_cooperative.py), or `make_persistent_leaf_plan` (see
    fft_plan_persistent.py), picked by `cooperative_workers`/
    `persistent_leaf`:

    * `persistent_leaf=True`: `make_persistent_leaf_plan`, this leaf's own
      local FFT (`length`, `radices`) executed by the persistent-software-
      workgroup model -- `total_uthreads` here is this leaf's own replica
      count (`r`/`r*a`, exactly what `total_uthreads` already means for
      every other leaf kind: see FFTNode(M, R)'s own "R independent
      M-point transforms" contract in this module's top docstring), passed
      straight through as `num_logical_blocks` -- persistent absorbs
      however many replicas that is into its own internal round loop
      (`preload -> stages -> writeback`, repeated `num_rounds(...)` times
      inside one launch), never needing this function's own caller to
      split it into multiple outer launches the way a plain/cooperative
      leaf's own round-split machinery does (see fft_transpose_codegen.
      generate_recursive_fft_kernels' own per-stage round loop, which a
      persistent stage always resolves to exactly one round of, by
      construction of `plan.total_uthreads == plan.max_uthread ==
      launch_uthreads`, a target-fixed width independent of replica
      count). Mutually exclusive with `cooperative_workers` (asserted by
      `_build_recursive_node`, this function's only caller) -- persistent
      and cooperative are two different execution strategies for the same
      leaf-lowering slot, never combined (PersistentWorkgroupPlan's own
      docstring).

    * `None`/`cooperative_workers` (the default when `persistent_leaf` is
      `False`): always `_build_plan`, byte-for-byte the plan this function
      returned before cooperative leaves existed.
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
    if persistent_leaf:
        return make_persistent_leaf_plan(
            length=length,
            radices=radices,
            num_logical_blocks=total_uthreads,
            inverse=inverse,
            simd_lanes=simd_lanes,
            kernel_name=kernel_name,
            inverse_scale=inverse_scale,
        )

    workers = 1
    if cooperative_workers is not None:
        max_workers = None if cooperative_workers == "auto" else cooperative_workers
        workers = choose_workers_per_fft(
            length, radices, simd_lanes=simd_lanes, max_workers=max_workers,
            interleave_chunk_uthreads=interleave_chunk_uthreads,
        )

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


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _default_tile_size(rows: int, cols: int, simd_lanes: int) -> int:
    """The square tile size `_build_recursive_node` uses for a PRE/MIDDLE/
    POST transpose when the caller leaves `tile_rows`/`tile_cols` at their
    default `None` -- i.e. every plain `make_recursive_transpose_plan(n,
    ...)` call with no explicit tile override, which is most of them (a
    bare `make_fft_kernel.py N` included).

    Before this existed, that default was just `min(simd_lanes, rows,
    cols)`, with no awareness of fft_plan_search.generate_tile_candidates'
    own real-hardware finding: only a tile size that evenly divides both
    `rows` and `cols` (a divisor of `gcd(rows, cols)`) is confirmed free
    of the row/col tail branch 2026-08-27's N=630 disassembly traced an
    LLVM tail-branch spill to (see that function's own comment for the
    full real-hardware data). N=630's own PRE transpose (rows=105, cols=6,
    gcd=3) is the concrete case this fixes: the naive default landed on
    tile=6 -- one of the two sizes (4 and 6) that specific investigation
    confirmed spills on real hardware -- purely by accident, since nothing
    about picking `min(simd_lanes, rows, cols)` ever consulted which sizes
    were safe. Confirmed by direct probe (planning.spill_probe.
    probe_spill_free) that this same plain, no-flags N=630 plan spilled at
    FFTRecPre0/FFTRecMid0/FFTRecPost0's own stage_0() before this fix.

    Picks the *largest* divisor of `gcd(rows, cols)` that still fits under
    the old `min(simd_lanes, rows, cols)` cap -- not always 1 (structurally
    always safe, but real hardware also shows small tiles are not always
    fastest, see generate_tile_candidates' own tile=1x1-losing-at-scale
    data) and not always the biggest available divisor (could exceed the
    SIMD/register-width budget the old cap was already protecting). When
    `gcd(rows, cols) == 1` (no divisor above 1 exists, e.g. two coprime
    dimensions), this falls back to `1` -- every transpose stage still
    renders correctly at tile=1x1, just with more, smaller tiles than the
    old default picked; there is no *safe* larger option to fall back to
    instead in that shape.
    """
    cap = max(1, min(simd_lanes, rows, cols))
    g = math.gcd(rows, cols)
    best = 1
    for d in range(1, g + 1):
        if g % d == 0 and d <= cap:
            best = d
    return best


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
    tiles_per_replica = grid_rows * grid_cols

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
        needs_q_table=replica_count > 1 and not _is_pow2(tiles_per_replica),
        needs_tr_table=grid_cols > 1 and not _is_pow2(grid_cols),
        needs_round_split=max_uthread < total_uthreads,
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
    interleave_chunk_uthreads: int = DEFAULT_TARGET_PROFILE.interleave_chunk_uthreads,
    allowed_radix_composites: frozenset[int] | None = None,
    forced_split_near_length: int | None = None,
    forced_split_sequence: tuple[int, ...] | None = None,
    forced_worker_sequence: tuple[int | str | None, ...] | None = None,
    persistent_leaf: bool = False,
) -> FFTNode:
    """`allowed_radix_composites`: `None` (the default) keeps every leaf's
    radix tier exactly `coalesce_radices`'s own default
    (`_DEFAULT_COALESCE_ALLOWED`, radix-4-only, the one confirmed safe on
    real hardware) -- passed explicitly, applies at *every* recursion
    level (a target-capability statement, not a one-shot override), see
    fft_plan_search.py's radix-tier candidates.

    `forced_split_near_length`: `None` (the default) keeps today's exact
    `_choose_recursive_split` heuristic (largest legal candidate). A given
    value skips that heuristic *once*, at this call only -- the recursive
    `far_child` call below always passes `forced_split_near_length=None`,
    so only the outermost (root) split is ever forced; every level below
    still picks its own best split normally. Must be a member of
    `_recursive_split_candidates(m, scratchpad_byte_budget=...)` (asserted
    below) -- this function never invents a split value a real budget
    wouldn't also allow.

    `forced_split_sequence`: like `forced_split_near_length`, but forces
    *every* level at once instead of only the root's: this call consumes
    its own first entry, then passes the rest on to `far_child` below
    (unlike `forced_split_near_length`, which the `far_child` call never
    receives). An empty tuple forces "stop, no further split" here --
    only legal when `m` already fits one leaf outright (asserted below),
    since there would otherwise be nothing left to force a split into.
    Built by `_enumerate_leaf_segmentations` (one entry per level, in this
    same near_fft-first build order) for `fft_plan_search.py`'s joint
    whole-tree candidates -- mutually exclusive with
    `forced_split_near_length` (the caller picks one or the other, never
    both, since one forces a single level and the other forces all of
    them).

    `forced_worker_sequence`: like `forced_split_sequence`, but for each
    *leaf's own execution strategy* instead of each level's own split --
    one entry per leaf this tree actually builds, near_fft-first, in the
    exact same visitation order `forced_split_sequence` already walks:
    this call consumes its own first entry for whichever leaf it builds
    (the near_fft leaf if `m` splits, the terminal leaf if it doesn't),
    then passes the rest to `far_child` below if it split. Independent of
    whether the split itself came from the heuristic, `forced_split_
    near_length`, or `forced_split_sequence` -- it only tracks how many
    leaves get built, not how the tree got that shape, so a caller can
    vary per-leaf strategy against *any* split choice.

    Each entry is either a `cooperative_workers`-shaped value (`None` =
    non-cooperative, `"auto"` or a positive int = cooperative -- see
    `_build_leaf_kernel`'s own docstring) or the literal string
    `"persistent"`, meaning *this one leaf* is built by `make_persistent_
    leaf_plan` instead, independent of what neighboring entries say --
    the actual per-leaf *mixed execution strategy* mechanism (near=
    persistent, far=non-coop, or any other combination): reuses this
    existing one-entry-per-leaf sequence rather than a parallel one,
    since a leaf's execution strategy and its cooperative worker count
    were never independent choices to begin with (exactly one of
    {non-cooperative, cooperative(workers), persistent} applies per
    leaf).

    Mutually exclusive with `cooperative_workers` and `persistent_leaf`
    (the caller picks a single uniform choice for every leaf via those
    two, or a per-leaf sequence via this one, never a mix -- asserted
    below). A terminal leaf (no further split) requires the sequence be
    down to exactly its own one entry, since nothing would consume the
    rest; built by `fft_plan_search.generate_per_leaf_worker_candidates`
    for one-leaf-at-a-time search, mirroring `_enumerate_leaf_
    segmentations`'s own role for splits.

    `persistent_leaf`: `False` (the default) keeps every leaf/near_fft
    kernel this tree builds exactly as `cooperative_workers` alone would
    -- `True` builds every one of them as a persistent-software-workgroup
    leaf instead (see `_build_leaf_kernel`'s own docstring). A single
    *uniform* choice for the whole tree, same discipline `cooperative_
    workers` follows before `forced_worker_sequence` existed -- for a
    *mixed* per-leaf choice (some persistent, some not), use `forced_
    worker_sequence`'s own `"persistent"` entries instead (see above).
    Mutually exclusive with both `cooperative_workers` and `forced_
    worker_sequence` (asserted below) when *this* uniform flag is `True`
    -- persistent and cooperative are two different execution strategies
    for the same leaf-lowering slot, never combined on one leaf
    (PersistentWorkgroupPlan's own docstring); a *uniform* `True` here and
    a *per-leaf* sequence both claiming to decide the same leaf's strategy
    is exactly the ambiguity this exclusion rules out, not a claim that
    the two features are unrelated.
    """
    idx = node_id[0]
    node_id[0] += 1

    if persistent_leaf:
        assert cooperative_workers is None, (
            "persistent_leaf and cooperative_workers are mutually exclusive -- "
            "persistent and cooperative are different execution strategies for "
            "the same leaf, never combined on one leaf"
        )
        assert forced_worker_sequence is None, (
            "persistent_leaf and forced_worker_sequence are mutually exclusive -- "
            "a per-leaf cooperative worker-count sequence has nothing to say "
            "about a leaf that isn't cooperative at all"
        )

    if forced_split_sequence is not None:
        assert forced_split_near_length is None, (
            "forced_split_sequence and forced_split_near_length are mutually "
            "exclusive -- the former already forces every level, including this one"
        )
        if forced_split_sequence:
            split_b = forced_split_sequence[0]
            legal = _recursive_split_candidates(m, scratchpad_byte_budget=scratchpad_byte_budget)
            if split_b not in legal:
                raise ValueError(
                    f"forced_split_sequence's next entry {split_b} is not a "
                    f"legal split of m={m} under scratchpad_byte_budget="
                    f"{scratchpad_byte_budget} (legal candidates: {legal})"
                )
        else:
            assert _leaf_scratchpad_bytes(m) <= scratchpad_byte_budget, (
                f"forced_split_sequence ran out with m={m} still too large for "
                f"one leaf under scratchpad_byte_budget={scratchpad_byte_budget} "
                f"-- _enumerate_leaf_segmentations should never produce this"
            )
            split_b = None
    elif forced_split_near_length is not None:
        legal = _recursive_split_candidates(m, scratchpad_byte_budget=scratchpad_byte_budget)
        if forced_split_near_length not in legal:
            raise ValueError(
                f"forced_split_near_length={forced_split_near_length} is not a "
                f"legal split of m={m} under scratchpad_byte_budget="
                f"{scratchpad_byte_budget} (legal candidates: {legal})"
            )
        split_b = forced_split_near_length
    else:
        split_b = _choose_recursive_split(m, scratchpad_byte_budget=scratchpad_byte_budget)

    if forced_worker_sequence is not None:
        assert cooperative_workers is None, (
            "forced_worker_sequence and cooperative_workers are mutually "
            "exclusive -- the former already picks every leaf's own choice, "
            "including this one"
        )
        if not forced_worker_sequence:
            raise ValueError(
                "forced_worker_sequence ran out of entries before every leaf "
                "was built -- it needs exactly one entry per leaf this tree "
                "actually builds, near_fft-first"
            )
        entry = forced_worker_sequence[0]
        rest_worker_sequence: tuple[int | str | None, ...] | None = forced_worker_sequence[1:]
        # "persistent" is a distinct sentinel from "auto"/an int (which
        # both mean "cooperative, this many workers") -- this one leaf
        # gets built by make_persistent_leaf_plan instead, independent of
        # what neighboring entries in the same sequence say. This is the
        # actual per-leaf *mixed execution strategy* mechanism Phase 4
        # needs (see docs/persistent_recursive_split.md's own "not done
        # yet" list): reusing forced_worker_sequence's existing one-entry-
        # per-leaf, near_fft-first walk rather than inventing a parallel
        # sequence parameter, since a leaf's execution strategy and its
        # cooperative worker count were never independent choices to begin
        # with -- exactly one of {non-cooperative, cooperative(workers),
        # persistent} applies per leaf.
        this_leaf_persistent = entry == "persistent"
        this_leaf_workers = None if this_leaf_persistent else entry
    else:
        this_leaf_workers = cooperative_workers
        this_leaf_persistent = persistent_leaf
        rest_worker_sequence = None

    if split_b is None:
        if forced_worker_sequence is not None and rest_worker_sequence:
            raise ValueError(
                f"forced_worker_sequence has {len(rest_worker_sequence)} "
                f"entries left after this terminal leaf (m={m}) -- it must "
                f"have exactly one entry per leaf, and this is the last one"
            )
        radices = coalesce_radices(_prime_factors_supported(m), allowed=allowed_radix_composites)
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
            cooperative_workers=this_leaf_workers,
            interleave_chunk_uthreads=interleave_chunk_uthreads,
            persistent_leaf=this_leaf_persistent,
        )
        return FFTLeafPlan(m=m, r=r, kernel=kernel)

    b = split_b
    a = m // b
    tr = tile_rows if tile_rows is not None else _default_tile_size(b, a, simd_lanes)
    tc = tile_cols if tile_cols is not None else _default_tile_size(a, b, simd_lanes)
    tr = max(tr, 1)
    tc = max(tc, 1)

    pre = _build_physical_transpose(
        rows=b, cols=a, replica_count=r, tile_rows=tr, tile_cols=tc,
        twiddle_modulus=None, inverse=inverse, kernel_name=f"FFTRecPre{idx}",
        simd_lanes=simd_lanes, spad_capacity_bytes=spad_capacity_bytes,
        apply_inverse_scale=False, max_concurrent_scratchpad_bytes=max_concurrent_scratchpad_bytes,
    )

    near_radices = coalesce_radices(_prime_factors_supported(b), allowed=allowed_radix_composites)
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
        cooperative_workers=this_leaf_workers,
        interleave_chunk_uthreads=interleave_chunk_uthreads,
        persistent_leaf=this_leaf_persistent,
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
        interleave_chunk_uthreads=interleave_chunk_uthreads,
        persistent_leaf=persistent_leaf,
        allowed_radix_composites=allowed_radix_composites,
        # forced_split_near_length intentionally NOT propagated -- only the
        # outermost (root) split is ever forced, see this function's own
        # docstring. forced_split_sequence, by contrast, IS propagated
        # (minus the entry this call already consumed) -- it forces every
        # level, not just the root. forced_worker_sequence follows the same
        # "propagate the rest" rule as forced_split_sequence, independent of
        # it (see this function's own docstring) -- cooperative_workers
        # itself is passed through unchanged (still None whenever
        # forced_worker_sequence is in use, per the assertion above).
        forced_split_sequence=forced_split_sequence[1:] if forced_split_sequence is not None else None,
        forced_worker_sequence=rest_worker_sequence,
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
    batch: int = 1,
    spad_capacity_bytes: int | None = None,
    max_concurrent_scratchpad_bytes: int | None = None,
    cooperative_workers: int | str | None = None,
    interleave_chunk_uthreads: int = DEFAULT_TARGET_PROFILE.interleave_chunk_uthreads,
    allowed_radix_composites: frozenset[int] | None = None,
    forced_split_near_length: int | None = None,
    forced_split_sequence: tuple[int, ...] | None = None,
    forced_worker_sequence: tuple[int | str | None, ...] | None = None,
    persistent_leaf: bool = False,
) -> RecursiveFFTPlan:
    """N decomposed recursively (six-step-FFT style): each node either
    fuses into one multi-radix leaf kernel (see FFTLeafPlan) or splits
    M=A*B and emits PRE transpose -> B-point FFT -> MIDDLE transpose
    (W_M twiddle) -> recursive FFTNode(A, R*B) -> POST transpose (see
    FFTRecursiveNodePlan). Physical transpose tile size (tile_rows/
    tile_cols, default `_default_tile_size` -- see its own docstring: the
    largest divisor of `gcd(rows, cols)` that still fits under
    `min(simd_lanes, rows, cols)`, so the default itself is always one of
    the tail-branch-free sizes real hardware has confirmed safe, not just
    whatever `min(simd_lanes, rows, cols)` happens to land on) is chosen
    completely independently of any FFT chunk length -- see the module
    design writeup. Additive: make_multi_kernel_plan/
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

    `allowed_radix_composites`/`forced_split_near_length`/
    `forced_split_sequence`/`forced_worker_sequence`: all `None` by
    default (today's exact single-heuristic behavior, unchanged) -- see
    `_build_recursive_node`'s own docstring. Exist so `fft_plan_search.py`
    can build a specific candidate plan instead of only "the one plan this
    budget/heuristic combination implies". `forced_worker_sequence` is
    mutually exclusive with `cooperative_workers` (asserted in
    `_build_recursive_node`) -- pass one or the other, never both.

    `persistent_leaf`: `False` (the default) -- see `_build_recursive_
    node`'s own docstring. Mutually exclusive with `cooperative_workers`
    and `forced_worker_sequence` (asserted in `_build_recursive_node`).

    `batch`: how many independent length-`n` transforms to run in one
    launch, `1` by default (today's exact prior behavior, a single
    transform). Passed straight through as the root node's own `r`
    (FFTNode(M, R)'s "R independent M-point transforms" contract already
    threads multiplicatively through every recursion level -- see
    FFTRecursiveNodePlan's own docstring's "b x a -> a x b -> ... -> a x b"
    derivation -- so every leaf/transpose kernel this tree renders is
    already correctly sized for any batch with zero further plan changes).
    The `batch` independent transforms sit back to back in one flat
    `batch*n`-element buffer (`transform_index*n + element_index`), the
    same layout `codegen.common.emit_reference_check`'s own `batch_count`
    already assumes for a single-kernel plan's `total_uthreads`.
    """
    node_id = [0]
    root = _build_recursive_node(
        n, batch, scratchpad_byte_budget=scratchpad_byte_budget,
        simd_lanes=simd_lanes, inverse=inverse,
        spad_capacity_bytes=spad_capacity_bytes, tile_rows=tile_rows,
        tile_cols=tile_cols, is_root=True, node_id=node_id,
        max_concurrent_scratchpad_bytes=max_concurrent_scratchpad_bytes,
        cooperative_workers=cooperative_workers,
        interleave_chunk_uthreads=interleave_chunk_uthreads,
        allowed_radix_composites=allowed_radix_composites,
        forced_split_near_length=forced_split_near_length,
        forced_split_sequence=forced_split_sequence,
        forced_worker_sequence=forced_worker_sequence,
        persistent_leaf=persistent_leaf,
    )
    host = MultiKernelHostPlan(n=n, inverse=inverse, tolerance=1.0e-3)
    return RecursiveFFTPlan(n=n, inverse=inverse, root=root, host=host, batch=batch)
