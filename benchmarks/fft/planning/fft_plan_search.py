from __future__ import annotations

"""Explicit, enumerable candidate search over the recursive planner's own
decision axes: split (A*B), radix tier, cooperative worker count, and
transpose tile size. Every candidate resolves to an ordinary
RecursiveFFTPlan -- exactly what generate_recursive_fft_kernels already
renders -- built via make_recursive_transpose_plan's own additive,
default-off parameters (forced_split_near_length, allowed_radix_composites,
cooperative_workers, tile_rows/tile_cols). Codegen never sees a candidate
or a choice, only the plan a candidate already resolved to.

Deliberately NOT a full cross product across all four axes (avoid
combinatorial explosion): generate_candidates builds one baseline (today's
exact single-heuristic plan) and then varies one axis at a time around it
-- a split sweep, then a worker sweep, a radix-tier sweep, and a tile
sweep, each holding the other axes at the baseline's own choice. This is a
staged/"star" search, not a joint one over the split axis alone --
generate_joint_split_candidates (below) is that joint search: every level
of the recursive split chosen together instead of one-at-a-time, via
_enumerate_leaf_segmentations, still scored by the same cost model as
everything else here. A true joint search across *all four* axes at once
remains future work.
"""

from dataclasses import dataclass, replace

from planning.fft_cost_model import PlanMetrics, estimate_cost, estimate_metrics
from planning.fft_plan_cooperative import worker_candidates_per_fft
from planning.fft_plan_recursive import (
    FFTLeafPlan,
    FFTNode,
    FFTRecursiveNodePlan,
    RecursiveFFTPlan,
    _enumerate_leaf_segmentations,
    _leaf_scratchpad_bytes,
    _recursive_split_candidates,
    make_recursive_transpose_plan,
)
from planning.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile

# Composite radices confirmed clean *as a leaf's own first stage only* (see
# fft_cost_model._NON_FIRST_STAGE_RISKY_RADICES's own comment for the N=54
# spill this is based on) -- offered as a candidate tier only when the
# target says it can tolerate the risk, and always scored non-zero risk by
# the cost model regardless. `10` is never offered: fft_plan_core.
# _prime_factors_supported's own comment documents it as unconditionally
# unsafe (N=160/320), not merely risky.
_WIDE_RADIX_TIER = frozenset({4, 6, 9})


@dataclass(frozen=True)
class PlanChoices:
    """What search knobs produced this candidate -- for format_plan_summary
    and debugging, never consumed by codegen. `None` in any field means
    "the planner's own default", exactly as `make_recursive_transpose_plan`
    already treats `None` for the matching parameter."""

    split_near_length: int | None
    radix_tier_name: str
    workers_per_fft: int | None
    tile: tuple[int, int] | None
    # Only set for generate_joint_split_candidates's own entries (every
    # other generator leaves this None): the full per-level near_fft chain
    # forced via make_recursive_transpose_plan's forced_split_sequence --
    # see _enumerate_leaf_segmentations' own docstring for why this can't
    # be summarized as a single split_near_length the way the one-level
    # split sweep's own candidates can.
    split_sequence: tuple[int, ...] | None = None


@dataclass(frozen=True)
class FFTPlanCandidate:
    n: int
    inverse: bool
    plan: RecursiveFFTPlan
    choices: PlanChoices
    metrics: PlanMetrics


def generate_split_candidates(n: int, *, scratchpad_byte_budget: int) -> list[int | None]:
    """Every legal root-level split choice for N: `None` (single leaf, no
    split) first if N itself fits the budget as one leaf, then every
    `_recursive_split_candidates` result (largest first, matching
    `_choose_recursive_split`'s own default pick as this list's first
    split entry)."""
    options: list[int | None] = []
    if _leaf_scratchpad_bytes(n) <= scratchpad_byte_budget:
        options.append(None)
    options.extend(_recursive_split_candidates(n, scratchpad_byte_budget=scratchpad_byte_budget))
    return options


def generate_radix_tiers(target: TargetProfile) -> list[tuple[str, frozenset[int] | None]]:
    """`("default", None)` always first (today's exact
    `_DEFAULT_COALESCE_ALLOWED` behavior) -- plus `("wide", _WIDE_RADIX_TIER)`
    only when `target.supports_vector_spill`, since a wider tier is exactly
    the kind of register-pressure risk that flag describes. Even offered,
    the wide tier is not claimed universally safe -- see fft_cost_model's
    own radix_risk_score, which still penalizes it."""
    tiers: list[tuple[str, frozenset[int] | None]] = [("default", None)]
    if target.supports_vector_spill:
        tiers.append(("wide", _WIDE_RADIX_TIER))
    return tiers


def generate_worker_candidates(
    length: int, radices: tuple[int, ...], *, target: TargetProfile, simd_lanes: int = 8,
) -> list[int]:
    """Every legal cooperative worker count for a leaf of this shape,
    ascending -- thin wrapper over `worker_candidates_per_fft` (see its own
    docstring for the legality rule), just naming this module's own axis
    consistently with the other three generators."""
    return worker_candidates_per_fft(
        length, radices, simd_lanes=simd_lanes,
        interleave_chunk_uthreads=target.interleave_chunk_uthreads,
    )


def generate_tile_candidates(rows: int, cols: int, *, simd_lanes: int) -> list[tuple[int, int]]:
    """A handful of legal (tile_rows, tile_cols) pairs for one PRE/MIDDLE/
    POST transpose of this matrix shape: the planner's own default (square,
    `min(simd_lanes, rows, cols)` -- see `_build_recursive_node`), the same
    bound applied *independently* per dimension instead of jointly (only
    different from the default when rows != cols), a doubled variant where
    that stays inside the matrix, and every halving of the default square
    size down to 1x1 -- not just one halving step. Real M2NDP runs (this
    project's own benchmark_fft_candidates.sh at N=1024) show tile=(2,2)
    beating the default (4,4)/(8,4), and 1x1 beating THAT again (1252
    cycles vs. 1516) -- one halving step alone stopped short of the actual
    optimum here, so this keeps going to the floor instead of guessing
    where to stop. Still small (at most log2(default_dim) extra entries)
    per section 10's own "avoid combinatorial explosion" instruction."""
    default_dim = max(1, min(simd_lanes, rows, cols))
    candidates = {(default_dim, default_dim)}
    candidates.add((max(1, min(simd_lanes, rows)), max(1, min(simd_lanes, cols))))
    dim = default_dim
    while dim > 1:
        dim = max(1, dim // 2)
        candidates.add((dim, dim))
    doubled = min(default_dim * 2, rows, cols)
    if doubled > default_dim:
        candidates.add((doubled, doubled))
    return sorted(candidates)


def _root_split_length(plan: RecursiveFFTPlan) -> int | None:
    root = plan.root
    return root.b if isinstance(root, FFTRecursiveNodePlan) else None


def _search_leaf_shape(plan: RecursiveFFTPlan) -> tuple[int, tuple[int, ...]]:
    """The (length, radices) of the leaf whose worker/tile axis this module
    sweeps -- the near_fft leaf if the root splits, else the root's own
    single leaf. Matches this module's own "vary the outermost level's
    choices" scope (see the module docstring)."""
    root = plan.root
    leaf = root.near_fft if isinstance(root, FFTRecursiveNodePlan) else root
    assert isinstance(leaf, FFTLeafPlan)
    return leaf.m, tuple(s.radix for s in leaf.kernel.stages)


def generate_candidates(
    n: int,
    *,
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
    inverse: bool = False,
    scratchpad_byte_budget: int = 4096,
    simd_lanes: int = 8,
    max_candidates: int = 40,
    max_joint_split_candidates: int = 16,
) -> list[FFTPlanCandidate]:
    """Baseline (today's exact default plan) plus one axis varied at a time
    around it -- see the module docstring for why this is a staged sweep,
    not a full cross product. Every candidate is a real, fully-built
    RecursiveFFTPlan (byte-for-byte what `generate_recursive_fft_kernels`
    already knows how to render), so nothing downstream needs to change to
    consume one.
    """

    def build(
        *, forced_split_near_length: int | None, allowed_radix_composites,
        cooperative_workers, tile_rows, tile_cols,
    ) -> RecursiveFFTPlan:
        return make_recursive_transpose_plan(
            n,
            scratchpad_byte_budget=scratchpad_byte_budget,
            simd_lanes=simd_lanes,
            inverse=inverse,
            spad_capacity_bytes=target.spad_capacity_bytes,
            max_concurrent_scratchpad_bytes=target.max_concurrent_scratchpad_bytes,
            interleave_chunk_uthreads=target.interleave_chunk_uthreads,
            forced_split_near_length=forced_split_near_length,
            allowed_radix_composites=allowed_radix_composites,
            cooperative_workers=cooperative_workers,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
        )

    candidates: list[FFTPlanCandidate] = []

    def add(plan: RecursiveFFTPlan, choices: PlanChoices) -> None:
        metrics = estimate_metrics(plan, target)
        metrics = replace(metrics, estimated_cost=estimate_cost(metrics))
        candidates.append(FFTPlanCandidate(n=n, inverse=inverse, plan=plan, choices=choices, metrics=metrics))

    # 1. baseline -- today's exact default heuristic, every axis default.
    baseline_plan = build(
        forced_split_near_length=None, allowed_radix_composites=None,
        cooperative_workers=None, tile_rows=None, tile_cols=None,
    )
    baseline_split = _root_split_length(baseline_plan)
    add(baseline_plan, PlanChoices(
        split_near_length=baseline_split, radix_tier_name="default",
        workers_per_fft=None, tile=None,
    ))

    # 2. split sweep.
    for b in generate_split_candidates(n, scratchpad_byte_budget=scratchpad_byte_budget):
        if b == baseline_split:
            continue
        plan = build(
            forced_split_near_length=b, allowed_radix_composites=None,
            cooperative_workers=None, tile_rows=None, tile_cols=None,
        )
        add(plan, PlanChoices(split_near_length=b, radix_tier_name="default", workers_per_fft=None, tile=None))

    # 3. worker sweep, split held at baseline's own choice.
    leaf_length, leaf_radices = _search_leaf_shape(baseline_plan)
    for w in generate_worker_candidates(leaf_length, leaf_radices, target=target, simd_lanes=simd_lanes):
        if w <= 1:
            continue  # workers=1 is byte-for-byte the baseline (cooperation off)
        plan = build(
            forced_split_near_length=baseline_split, allowed_radix_composites=None,
            cooperative_workers=w, tile_rows=None, tile_cols=None,
        )
        add(plan, PlanChoices(split_near_length=baseline_split, radix_tier_name="default", workers_per_fft=w, tile=None))

    # 4. radix-tier sweep, split held at baseline's own choice.
    for tier_name, allowed in generate_radix_tiers(target):
        if allowed is None:
            continue  # "default" tier is the baseline itself
        plan = build(
            forced_split_near_length=baseline_split, allowed_radix_composites=allowed,
            cooperative_workers=None, tile_rows=None, tile_cols=None,
        )
        add(plan, PlanChoices(split_near_length=baseline_split, radix_tier_name=tier_name, workers_per_fft=None, tile=None))

    # 5. tile sweep, only meaningful when the baseline actually splits.
    if baseline_split is not None:
        far_length = n // baseline_split
        default_tile_dim = max(1, min(simd_lanes, baseline_split, far_length))
        for tile in generate_tile_candidates(baseline_split, far_length, simd_lanes=simd_lanes):
            if tile == (default_tile_dim, default_tile_dim):
                continue
            plan = build(
                forced_split_near_length=baseline_split, allowed_radix_composites=None,
                cooperative_workers=None, tile_rows=tile[0], tile_cols=tile[1],
            )
            add(plan, PlanChoices(split_near_length=baseline_split, radix_tier_name="default", workers_per_fft=None, tile=tile))

    # 6. joint split-sequence sweep -- every level of the recursive split
    # chosen together (_enumerate_leaf_segmentations), not just the root
    # (step 2 only ever varies the outermost split, leaving every deeper
    # level at whatever _choose_recursive_split's own greedy pick is).
    # _enumerate_leaf_segmentations already orders its own results
    # largest-segment-first, so its own first entries resemble the
    # baseline most closely; skip that one exact duplicate, same as the
    # split sweep above.
    segmentations = _enumerate_leaf_segmentations(n, scratchpad_byte_budget=scratchpad_byte_budget)
    for seq in segmentations[:max_joint_split_candidates]:
        if seq and seq[0] == baseline_split and len(seq) == 1:
            continue  # byte-for-byte the baseline (single-level split, same b)
        plan = make_recursive_transpose_plan(
            n,
            scratchpad_byte_budget=scratchpad_byte_budget,
            simd_lanes=simd_lanes,
            inverse=inverse,
            spad_capacity_bytes=target.spad_capacity_bytes,
            max_concurrent_scratchpad_bytes=target.max_concurrent_scratchpad_bytes,
            interleave_chunk_uthreads=target.interleave_chunk_uthreads,
            forced_split_sequence=seq,
        )
        add(plan, PlanChoices(
            split_near_length=None, radix_tier_name="default", workers_per_fft=None,
            tile=None, split_sequence=seq,
        ))

    return candidates[:max_candidates]


def rank_candidates(candidates: list[FFTPlanCandidate]) -> list[FFTPlanCandidate]:
    return sorted(candidates, key=lambda c: c.metrics.estimated_cost)


def _describe_node(node: FFTNode, depth: int, lines: list[str]) -> None:
    pad = "  " * depth
    if isinstance(node, FFTLeafPlan):
        radices = tuple(s.radix for s in node.kernel.stages)
        coop = node.kernel.cooperation
        workers = f" workers={coop.workers_per_fft}" if coop is not None else ""
        lines.append(f"{pad}Leaf M={node.m} R={node.r} radices={radices}{workers}")
        return
    assert isinstance(node, FFTRecursiveNodePlan)
    lines.append(f"{pad}Node M={node.m} R={node.r}  split: A={node.a} B={node.b}")
    pt = node.pre_transpose
    lines.append(f"{pad}  PRE    matrix={pt.rows}x{pt.cols} tile={pt.tile_rows}x{pt.tile_cols} tiles={pt.total_uthreads}")
    _describe_node(node.near_fft, depth + 1, lines)
    mt = node.middle_transpose
    lines.append(f"{pad}  MIDDLE matrix={mt.rows}x{mt.cols} twiddle_modulus={mt.twiddle_modulus} tile={mt.tile_rows}x{mt.tile_cols}")
    _describe_node(node.far_child, depth + 1, lines)
    pot = node.post_transpose
    lines.append(f"{pad}  POST   matrix={pot.rows}x{pot.cols} tile={pot.tile_rows}x{pot.tile_cols}")


def format_plan_summary(candidate: FFTPlanCandidate, *, index: int | None = None) -> str:
    """Human-readable tree + choices + metrics dump -- everything a reader
    needs to understand *why* this candidate looks the way it does, not
    just its final cost number (section 13's own instruction: never an
    opaque cost without showing the components)."""
    lines: list[str] = []
    header = f"FFT N={candidate.n}{' (inverse)' if candidate.inverse else ''}"
    if index is not None:
        header = f"[{index}] {header}"
    lines.append(header)
    c = candidate.choices
    lines.append(
        f"  choices: split_near_length={c.split_near_length} "
        f"radix_tier={c.radix_tier_name} workers_per_fft={c.workers_per_fft} tile={c.tile}"
        + (f" split_sequence={c.split_sequence}" if c.split_sequence is not None else "")
    )
    lines.append("")
    _describe_node(candidate.plan.root, 1, lines)
    lines.append("")
    m = candidate.metrics
    lines.append("  estimated metrics:")
    lines.append(f"    recursion_depth          = {m.recursion_depth}")
    lines.append(f"    leaf_kernel_count        = {m.leaf_kernel_count}")
    lines.append(f"    transpose_kernel_count   = {m.transpose_kernel_count}")
    lines.append(f"    total_leaf_stage_count   = {m.total_leaf_stage_count}")
    lines.append(f"    total_transpose_tiles    = {m.total_transpose_tiles}")
    lines.append(f"    estimated_dram_bytes     = {m.estimated_dram_bytes}")
    lines.append(f"    max_scratchpad_bytes     = {m.max_scratchpad_bytes}")
    lines.append(f"    worst_worker_utilization = {m.worst_worker_utilization:.3f}")
    lines.append(f"    radix_risk_score         = {m.radix_risk_score}")
    lines.append(f"    estimated_cost           = {m.estimated_cost:.1f}")
    return "\n".join(lines)
