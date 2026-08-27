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
_enumerate_leaf_segmentations lets step 6 (below) vary every level of the
recursive split together instead of one-at-a-time, still scored by the
same cost model as everything else here.

Step 7 goes one further, for the non-cooperative case specifically
(cooperative_workers held at None throughout -- see generate_worker_
candidates/step 3 for the one place worker count is still swept): split
sequence, radix tier, and tile are evaluated *jointly* -- every
(split_sequence, radix_tier, tile) triple becomes its own candidate,
instead of step 6's split-only variation (radix/tile pinned to the
baseline's default) and step 4/5's radix-only/tile-only variation (split
pinned to the baseline's own pick). This is what actually lets a plan like
"non-baseline split AND non-default radix together" get built and scored
at all -- previously such a combination could only be reached by chance if
it happened to also be *each* axis's independently-best choice from the
same baseline, not because it was jointly evaluated. Cooperative worker
count is deliberately NOT folded into this same joint loop: unlike split/radix/
tile, whose legal choices don't depend on cooperation, worker legality
depends on the leaf shape a given split already committed to (see
worker_candidates_per_fft), so a true 4-way joint search (split_sequence x
radix_tier x tile x per-leaf workers) is left as future work rather than
bolted on here as a fourth nested loop.

Step 8 covers a *different* worker axis than step 3's, though: step 3
picks one uniform cooperative_workers for the whole tree; step 8
(generate_per_leaf_worker_candidates, via forced_worker_sequence) varies
one leaf's own worker count at a time, every other leaf left
uncooperative -- an earlier session's own "mix cooperative and plain
leaves within the same plan, chosen per-leaf" discussion, now wired up.
Split held at the baseline's own choice, same discipline as step 3 (and
not crossed with step 7's own joint search, for the same worker-legality-
depends-on-split reason above).
"""

from dataclasses import dataclass, replace

from planning.fft_cost_model import PlanMetrics, estimate_cost, estimate_metrics
from planning.fft_plan_cooperative import worker_candidates_per_fft
from planning.fft_plan_core import FFTCodegenPlan
from planning.fft_plan_recursive import (
    FFTLeafPlan,
    FFTNode,
    FFTRecursiveNodePlan,
    RecursiveFFTPlan,
    _enumerate_leaf_segmentations,
    _leaf_scratchpad_bytes,
    _recursive_split_candidates,
    flatten_recursive_node,
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
    # Only set for generate_candidates' own joint-split-sequence entries
    # (steps 6 and 7 below; every other step leaves this None): the full
    # per-level near_fft chain forced via make_recursive_transpose_plan's
    # forced_split_sequence -- see _enumerate_leaf_segmentations' own
    # docstring for why this can't be summarized as a single
    # split_near_length the way the one-level split sweep's own candidates
    # (step 2) can.
    split_sequence: tuple[int, ...] | None = None
    # Only set for generate_candidates' own step 8 (per-leaf worker sweep):
    # one entry per leaf the baseline split actually builds, near_fft-
    # first, forced via make_recursive_transpose_plan's own
    # forced_worker_sequence -- see that function's and
    # generate_per_leaf_worker_candidates' own docstrings for why a single
    # workers_per_fft can't represent "this one leaf, not the others."
    worker_sequence: tuple[int | str | None, ...] | None = None


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


def _leaf_kernels_in_order(plan: RecursiveFFTPlan) -> list[FFTCodegenPlan]:
    """Every leaf's own FFTCodegenPlan, near_fft-first -- the exact order
    `forced_worker_sequence`'s own entries line up against (see
    fft_plan_recursive._build_recursive_node's own docstring):
    flatten_recursive_node already walks PRE -> near_fft.kernel -> MIDDLE
    -> far_child -> POST at every level, so filtering its own flat list
    down to just the FFTCodegenPlan entries (never PhysicalTransposePlan)
    visits leaves in that same order for free -- nothing here re-derives
    the tree structure itself."""
    return [s for s in flatten_recursive_node(plan.root) if isinstance(s, FFTCodegenPlan)]


def generate_per_leaf_worker_candidates(
    plan: RecursiveFFTPlan, *, target: TargetProfile, simd_lanes: int = 8,
) -> list[tuple[int, tuple[int | str | None, ...]]]:
    """One `(leaf_index, forced_worker_sequence)` pair per (leaf, legal
    worker count > 1) combination for `plan`'s own tree shape -- varying
    exactly *one* leaf's own cooperative_workers at a time, every other
    leaf left uncooperative (`None`), the same "one axis at a time" star
    search this whole module already uses for every other axis (see the
    module docstring), applied to the per-leaf worker axis
    `forced_worker_sequence` exists for.

    Discussed but not implemented in an earlier session: mixing
    cooperative and plain (one-uthread-per-sub-FFT) leaves *within the
    same plan*, chosen per-leaf instead of one global `cooperative_workers`
    for the whole tree. This is that -- `plan` fixes which split/tile/
    radix-tier choice every resulting candidate shares (pass the baseline
    plan to sweep this axis the way `generate_candidates`' own step 3
    sweeps the single global-worker axis against the baseline split), so
    call this once per split/segmentation candidate you also want to
    cross with it, same as `generate_worker_candidates`' own caller does
    today.

    `leaf_index` is returned alongside each sequence purely for a
    caller's own labeling/debugging (e.g. `PlanChoices` doesn't need it,
    since the sequence itself already shows which slot is non-`None`) --
    never consumed by planning or codegen.
    """
    kernels = _leaf_kernels_in_order(plan)
    num_leaves = len(kernels)
    results: list[tuple[int, tuple[int | str | None, ...]]] = []
    for i, kernel in enumerate(kernels):
        radices = tuple(s.radix for s in kernel.stages)
        for w in generate_worker_candidates(kernel.length, radices, target=target, simd_lanes=simd_lanes):
            if w <= 1:
                continue  # workers=1 is byte-for-byte this leaf's own uncooperative baseline
            seq = tuple(w if j == i else None for j in range(num_leaves))
            results.append((i, seq))
    return results


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


def _select_top_candidates(
    candidates: list[FFTPlanCandidate], max_candidates: int
) -> list[FFTPlanCandidate]:
    """Keep the best `max_candidates` by estimated_cost, not by which sweep
    step generated them or how early in generation order they landed --
    `candidates[:max_candidates]` (this function's own predecessor) would
    silently starve whichever step runs last once the total exceeds the
    cap, no matter how good its own candidates score, purely because it
    happened to be appended after everything else. `candidates[0]` (step
    1's baseline, today's exact pre-search default plan) is always kept
    regardless of its own score: every existing caller can already build
    it with zero search machinery, so it must never be the one candidate a
    cap silently drops.

    A no-op (returns `candidates` unchanged, same order) whenever the list
    already fits under `max_candidates` -- true for every N this project
    has generated candidates for before step 7 (the joint split/radix/tile
    sweep) existed, so this is not a behavior change for any pre-existing
    caller until a sweep actually produces enough candidates to exceed the
    cap.
    """
    if len(candidates) <= max_candidates:
        return candidates
    baseline, rest = candidates[0], candidates[1:]
    rest_ranked = sorted(rest, key=lambda c: c.metrics.estimated_cost)
    return [baseline] + rest_ranked[: max_candidates - 1]


def generate_candidates(
    n: int,
    *,
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
    inverse: bool = False,
    scratchpad_byte_budget: int = 4096,
    simd_lanes: int = 8,
    batch: int = 1,
    max_candidates: int = 40,
    max_joint_split_candidates: int = 16,
    max_joint_combined_candidates: int = 128,
) -> list[FFTPlanCandidate]:
    """Baseline (today's exact default plan) plus one axis varied at a time
    around it -- see the module docstring for why this is a staged sweep,
    not a full cross product. Every candidate is a real, fully-built
    RecursiveFFTPlan (byte-for-byte what `generate_recursive_fft_kernels`
    already knows how to render), so nothing downstream needs to change to
    consume one.

    `batch`: passed straight through to every candidate's own
    make_recursive_transpose_plan call, unchanged across the whole sweep --
    it multiplies every leaf/transpose kernel's own total_uthreads
    uniformly (see that function's own `batch` docstring) and never
    affects which splits/tiers/workers/tiles are legal or how they rank
    against each other, so it is not itself a search axis here.

    `max_joint_combined_candidates`: caps step 7's own (split_sequence x
    radix_tier x tile) triples -- a separate knob from
    `max_joint_split_candidates` (which only caps how many split_sequence
    entries even enter that cross product), purely a compute-time safety
    valve (each triple is one full plan build + estimate_metrics call, not
    a real toolchain run). Final fairness across axes/steps is not this
    cap's job -- see the module-level top-K selection below, which prunes
    the whole combined list by estimated_cost rather than by which step or
    generation-order position a candidate came from.
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
            batch=batch,
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
            batch=batch,
            spad_capacity_bytes=target.spad_capacity_bytes,
            max_concurrent_scratchpad_bytes=target.max_concurrent_scratchpad_bytes,
            interleave_chunk_uthreads=target.interleave_chunk_uthreads,
            forced_split_sequence=seq,
        )
        add(plan, PlanChoices(
            split_near_length=None, radix_tier_name="default", workers_per_fft=None,
            tile=None, split_sequence=seq,
        ))

    # 7. joint split_sequence x radix_tier x tile sweep, non-cooperative only
    # (cooperative_workers held at None throughout -- worker count is its
    # own axis, step 3 above, and legal worker counts depend on the leaf
    # shape a split already committed to, unlike radix_tier/tile, so it is
    # deliberately not a fourth nested loop here -- see the module
    # docstring). Unlike step 6, which only varies split_sequence (radix/
    # tile pinned to the baseline's own default), this builds every
    # (seq, tier, tile) triple as one candidate -- the actual fix for the
    # "kernel split and radix split decided as two separate greedy passes"
    # gap this module's own docstring describes.
    #
    # radix_tier is a whole-tree global knob (generate_radix_tiers doesn't
    # depend on shape at all, unlike tile), so no per-sequence rebuild is
    # needed to enumerate it. tile candidates DO depend on shape, but only
    # on the root level's own (near, far) matrix -- exactly the same shape
    # step 5's tile sweep already derives from baseline_split/far_length,
    # here computed straight from seq[0] (no provisional plan build
    # needed: PhysicalTransposePlan's rows/cols at the root are fully
    # determined by seq[0] and n alone, before anything is actually built).
    # seq == () (N already fits one leaf, no split at all -- a legal
    # _enumerate_leaf_segmentations entry) has no transpose stage to tile,
    # so only the radix axis applies there; tile stays the single
    # planner-default entry (None).
    radix_tiers = generate_radix_tiers(target)
    # Per-seq (tier, allowed, tile) lists, built separately per split_sequence
    # so the cap below (if it ever binds) can round-robin *across* seqs
    # instead of draining seq[0]'s own full tier x tile cross product before
    # seq[1] ever gets a single entry in -- exactly the "don't just chop off
    # generation-order tail, one axis pays the whole price" bias this
    # module's own docstring warns against (split_sequence is the axis this
    # step exists to stop starving).
    per_seq_options: list[list[tuple[str, frozenset[int] | None, tuple[int, int] | None]]] = []
    for seq in segmentations[:max_joint_split_candidates]:
        if seq:
            root_near = seq[0]
            root_far = n // root_near
            tile_options: list[tuple[int, int] | None] = [
                *generate_tile_candidates(root_near, root_far, simd_lanes=simd_lanes), None,
            ]
        else:
            tile_options = [None]

        options = [
            (tier_name, allowed, tile)
            for tier_name, allowed in radix_tiers
            for tile in tile_options
            if not (tier_name == "default" and tile is None)  # step 6's own candidate already
        ]
        per_seq_options.append(options)

    seqs = segmentations[:max_joint_split_candidates]
    joint_triples: list[tuple[tuple[int, ...], str, frozenset[int] | None, tuple[int, int] | None]] = []
    for round_idx in range(max(len(opts) for opts in per_seq_options) if per_seq_options else 0):
        for seq, options in zip(seqs, per_seq_options):
            if round_idx < len(options):
                tier_name, allowed, tile = options[round_idx]
                joint_triples.append((seq, tier_name, allowed, tile))

    for seq, tier_name, allowed, tile in joint_triples[:max_joint_combined_candidates]:
        plan = make_recursive_transpose_plan(
            n,
            scratchpad_byte_budget=scratchpad_byte_budget,
            simd_lanes=simd_lanes,
            inverse=inverse,
            batch=batch,
            spad_capacity_bytes=target.spad_capacity_bytes,
            max_concurrent_scratchpad_bytes=target.max_concurrent_scratchpad_bytes,
            interleave_chunk_uthreads=target.interleave_chunk_uthreads,
            forced_split_sequence=seq,
            allowed_radix_composites=allowed,
            cooperative_workers=None,
            tile_rows=tile[0] if tile is not None else None,
            tile_cols=tile[1] if tile is not None else None,
        )
        add(plan, PlanChoices(
            split_near_length=None, radix_tier_name=tier_name, workers_per_fft=None,
            tile=tile, split_sequence=seq,
        ))

    # 8. per-leaf worker sweep, split held at baseline's own choice (same
    # "hold split fixed" discipline step 3's global-worker sweep already
    # uses) -- one candidate per (leaf, legal worker count) combination,
    # every *other* leaf left uncooperative. See generate_per_leaf_worker_
    # candidates' own docstring: this is the per-leaf mixing an earlier
    # session discussed but never implemented, now that forced_worker_
    # sequence gives it a hook. Deliberately not crossed with step 7's own
    # (split_sequence x radix_tier x tile) joint search -- worker legality
    # already depends on the split a leaf ends up with (see
    # worker_candidates_per_fft), so adding a fourth nested axis there
    # remains the future work the module docstring calls out; this stays
    # a fifth *independent* star-search arm off the baseline instead.
    for leaf_index, seq in generate_per_leaf_worker_candidates(baseline_plan, target=target, simd_lanes=simd_lanes):
        plan = make_recursive_transpose_plan(
            n,
            scratchpad_byte_budget=scratchpad_byte_budget,
            simd_lanes=simd_lanes,
            inverse=inverse,
            batch=batch,
            spad_capacity_bytes=target.spad_capacity_bytes,
            max_concurrent_scratchpad_bytes=target.max_concurrent_scratchpad_bytes,
            interleave_chunk_uthreads=target.interleave_chunk_uthreads,
            forced_split_near_length=baseline_split,
            forced_worker_sequence=seq,
        )
        add(plan, PlanChoices(
            split_near_length=baseline_split, radix_tier_name="default", workers_per_fft=None,
            tile=None, worker_sequence=seq,
        ))

    return _select_top_candidates(candidates, max_candidates)


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
        + (f" worker_sequence={c.worker_sequence}" if c.worker_sequence is not None else "")
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
