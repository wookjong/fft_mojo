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

import itertools
import math
from dataclasses import dataclass, replace

from planning.fft_cost_model import PlanMetrics, estimate_cost, estimate_metrics
from planning.fft_plan_cooperative import worker_candidates_per_fft
from planning.fft_plan_core import (
    _DEFAULT,
    FFTCodegenPlan,
    MultiKernelHostPlan,
    _Default,
    _prime_factors_supported,
    coalesce_radices,
)
from planning.fft_plan_lanes import apply_all_scalar_lanes_to_plan, apply_compute_lanes_to_plan
from planning.fft_plan_persistent import make_persistent_leaf_plan
from planning.fft_plan_recursive import (
    FFTLeafPlan,
    FFTNode,
    FFTRecursiveNodePlan,
    PhysicalTransposePlan,
    RecursiveFFTPlan,
    _enumerate_leaf_segmentations,
    _leaf_scratchpad_bytes,
    _recursive_split_candidates,
    flatten_recursive_node,
    make_recursive_transpose_plan,
)
from planning.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile

# Composite radices confirmed clean *as a leaf's own first stage only* (see
# fft_cost_model._RISKY_RADIX_PAIRS's own comment for the N=54=(6,9)
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
    # Only set for generate_candidates' own step 10 (persistent-leaf sweep):
    # "persistent" marks a candidate built via generate_persistent_leaf_
    # candidates (planning.fft_plan_persistent.make_persistent_leaf_plan),
    # a wholly separate execution model from the cooperative-worker family
    # every other field above describes -- see that function's own
    # docstring for why `workers_per_fft`/`worker_sequence` stay None on
    # these candidates (persistent's own worker count is a target-derived
    # constant, `target.interleave_chunk_uthreads`, never a search choice).
    # `None` (every other step): not a persistent candidate, unchanged.
    execution_strategy: str | None = None
    # Only set for generate_candidates' own lane-variant sweep
    # (generate_lane_variant_candidates): "unnarrowed" or "all_scalar" --
    # see that function's own docstring. `None` (every other step,
    # including the plain baseline): every stage's own `compute_lanes`
    # stays unset (`None`) on the plan itself, and codegen resolves it
    # live from whatever flat compute_lanes/narrow_middle_stages the
    # eventual renderer passes -- see planning.fft_plan_lanes' own module
    # docstring for why that is still today's exact default behavior.
    lane_variant: str | None = None


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

    Also offers `"persistent"` as one more per-leaf option (Phase 4's
    per-leaf mixed execution strategy, gated by `_leaf_persistent_
    feasible`) alongside each leaf's own legal cooperative worker counts
    -- a third value `forced_worker_sequence`'s own entries already
    support, not a new axis this function needs separate plumbing for.

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
        if _leaf_persistent_feasible(kernel.length, target=target):
            seq = tuple("persistent" if j == i else None for j in range(num_leaves))
            results.append((i, seq))
    return results


def generate_leaf_worker_sequences(
    plan: RecursiveFFTPlan, *, target: TargetProfile, simd_lanes: int = 8,
    max_sequences: int = 32,
) -> list[tuple[int | str | None, ...]]:
    """Every legal *joint* per-leaf worker-count assignment for `plan`'s own
    leaves (near_fft-first, the same order `forced_worker_sequence`'s own
    entries line up against) -- the full Cartesian product across each
    leaf's own legal candidates (`generate_worker_candidates`), not
    `generate_per_leaf_worker_candidates`'s one-leaf-at-a-time variation
    (every *other* leaf pinned uncooperative there). This is what actually
    lets e.g. `(1, 4)` and `(4, 1)` both exist as distinct sequences for a
    2-leaf tree -- neither is reachable from the single-axis sweep, since
    both leaves are non-baseline at once in each.

    `None` marks "this leaf uncooperative" (`worker_candidates_per_fft`'s
    own `1` collapsed to `None`, the same convention
    `generate_per_leaf_worker_candidates` already uses -- `forced_worker_
    sequence[i]=None` and `=1` build byte-identical leaves, so keeping
    both around would only inflate the product for no benefit). The
    all-`None` sequence (every leaf uncooperative -- today's exact
    baseline) is always first.

    `max_sequences`: caps how many sequences this returns -- a tree with
    several multi-candidate leaves is the product of each leaf's own
    candidate count, which can grow past what's worth building+scoring
    (see this module's own docstring on avoiding combinatorial
    explosion elsewhere). `itertools.product` is consumed lazily, so
    hitting the cap never pays for the untaken tail of the product.

    Each leaf's own option list also includes `"persistent"` when
    `_leaf_persistent_feasible` says so (Phase 4's per-leaf mixed
    execution strategy) -- so the Cartesian product this function already
    builds naturally covers every representative mix the design's own
    "minimal mixed candidates" list asks for (e.g. a 2-leaf tree gets
    `(persistent, persistent)`, `(persistent, None)`, `(None, persistent)`,
    and `(persistent, w)`/`(w, persistent)` for each legal cooperative `w`
    -- all for free from the same product, not a separate generator).
    """
    kernels = _leaf_kernels_in_order(plan)
    if not kernels:
        return [()]

    per_leaf_options: list[list[int | str | None]] = []
    for kernel in kernels:
        radices = tuple(s.radix for s in kernel.stages)
        cands = generate_worker_candidates(kernel.length, radices, target=target, simd_lanes=simd_lanes)
        options: list[int | str | None] = [None] + [w for w in cands if w > 1]
        if _leaf_persistent_feasible(kernel.length, target=target):
            options.append("persistent")
        per_leaf_options.append(options)

    baseline_seq: tuple[int | str | None, ...] = tuple(None for _ in kernels)
    sequences: list[tuple[int | str | None, ...]] = [baseline_seq]
    seen = {baseline_seq}
    for combo in itertools.product(*per_leaf_options):
        if len(sequences) >= max_sequences:
            break
        if combo in seen:
            continue
        seen.add(combo)
        sequences.append(combo)
    return sequences


def _leaf_lengths(plan: RecursiveFFTPlan) -> list[int]:
    """Each leaf's own FFT length (`m`), near_fft-first -- the one piece of
    shape information `generate_radix_execution_joint_candidates` needs
    per leaf that does *not* depend on which radix tier ends up choosing
    that leaf's own radix sequence (leaf lengths/positions come from the
    split, held fixed for this whole joint search -- see that function's
    own docstring), so they can be read once from any already-built plan
    sharing the same split, `default`-tier or not."""
    return [k.length for k in _leaf_kernels_in_order(plan)]


def generate_radix_execution_joint_candidates(
    n: int,
    *,
    target: TargetProfile,
    inverse: bool,
    scratchpad_byte_budget: int,
    simd_lanes: int,
    batch: int,
    baseline_plan: RecursiveFFTPlan,
    baseline_split: int | None,
    max_worker_sequences: int = 32,
    max_joint_candidates: int = 96,
) -> list[tuple[RecursiveFFTPlan, "PlanChoices"]]:
    """`(radix_tier, per-leaf worker sequence)` pairs evaluated *jointly*,
    for `baseline_plan`'s own tree shape (split held fixed at
    `baseline_split` -- see `make_recursive_transpose_plan`'s own
    `forced_split_near_length` docstring for why this pins only the root
    level, and why every deeper level still reproduces the *same* split
    `baseline_plan` already has: `_choose_recursive_split` is a pure
    function of `(m, scratchpad_byte_budget)` alone, independent of radix
    tier or worker choice, so nothing here needs to force those levels
    explicitly to hold them fixed).

    This is the axis fft_plan_search.py's own module docstring names as
    future work: step 7 already joins split_sequence x radix_tier x tile
    but deliberately excludes cooperative workers (worker legality depends
    on the split a leaf ends up with); step 3/4/8 each vary radix tier or
    worker count independently around the baseline, never together. This
    function is what actually builds `(radix configuration, execution
    configuration)` as one candidate, per this project's own design
    request -- crossing `generate_radix_tiers` with
    `generate_leaf_worker_sequences`, not sweeping either alone.

    Each leaf's own *realized* radix sequence (not the tier's own label)
    is computed per tier directly from `coalesce_radices`/
    `_prime_factors_supported` -- no throwaway plan build needed just to
    discover it -- and used, together with the worker sequence, as this
    function's own dedup key: two tiers that happen to coalesce a given N
    identically (e.g. a leaf with no radix-4-mergeable pairs at all) would
    otherwise build byte-identical plans under different tier labels.

    Every `(plan, choices)` pair returned already passed the real
    `make_recursive_transpose_plan` build -- a combination that isn't
    actually legal (e.g. a worker count `_build_recursive_node`'s own
    assertions reject for this specific tier's own leaf shape) raises
    there and is caught and skipped here, never scored.

    `max_worker_sequences`/`max_joint_candidates`: bound the two stages of
    this search the same way step 7's own `max_joint_split_candidates`/
    `max_joint_combined_candidates` bound theirs -- a pure compute-time
    safety valve (each candidate is one plan build + estimate_metrics
    call, not a real toolchain run).
    """
    worker_sequences = generate_leaf_worker_sequences(
        baseline_plan, target=target, simd_lanes=simd_lanes, max_sequences=max_worker_sequences,
    )
    radix_tiers = generate_radix_tiers(target)
    leaf_lengths = _leaf_lengths(baseline_plan)

    results: list[tuple[RecursiveFFTPlan, PlanChoices]] = []
    seen_keys: set[tuple[tuple[tuple[int, ...], ...], tuple[int | str | None, ...]]] = set()

    for tier_name, allowed in radix_tiers:
        realized_radices = tuple(
            coalesce_radices(_prime_factors_supported(m), allowed=allowed) for m in leaf_lengths
        )
        for seq in worker_sequences:
            if len(results) >= max_joint_candidates:
                return results
            key = (realized_radices, seq)
            if key in seen_keys:
                continue  # same actual (per-leaf radix, worker) shape as an earlier tier label
            seen_keys.add(key)
            if tier_name == "default" and seq == tuple(None for _ in leaf_lengths):
                continue  # byte-for-byte the baseline candidate (step 1) already covers this
            try:
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
                    allowed_radix_composites=allowed,
                    forced_worker_sequence=seq,
                )
            except (ValueError, AssertionError, NotImplementedError):
                # Illegal for this specific (tier, worker sequence) combination
                # (e.g. a worker count this tier's own coalesced leaf shape
                # can no longer support, or a "persistent" entry -- see
                # generate_leaf_worker_sequences' own docstring -- whose
                # coalesced radix sequence make_persistent_leaf_plan itself
                # rejects, e.g. too many distinct kernel functions for
                # target.max_kernel_register) -- reject before scoring,
                # never a candidate the caller sees, per this module's own
                # "obviously bad candidates are pruned before cost ranking"
                # discipline.
                continue
            choices = PlanChoices(
                split_near_length=baseline_split, radix_tier_name=tier_name,
                workers_per_fft=None, tile=None, worker_sequence=seq,
            )
            results.append((plan, choices))

    return results


def _leaf_persistent_feasible(length: int, *, target: TargetProfile) -> bool:
    """Whether *one leaf* of this length can be built as a persistent-
    software-workgroup kernel at all -- the per-leaf legality
    `generate_leaf_worker_sequences`/`generate_per_leaf_worker_candidates`
    need to offer `"persistent"` as one leaf's own execution-strategy
    option inside a split tree (Phase 4's per-leaf mixed strategy), as
    opposed to `_persistent_leaf_feasible` below, which answers a
    different question -- whether the *entire* N is worth persisting
    *unsplit* -- and is not reused here.

    Only the real, physical capacity check applies at leaf granularity:
    `16 * length <= target.spad_capacity_bytes` (mirrors
    `make_persistent_leaf_plan`'s own unconditional check, so a caller
    can skip the attempt instead of catching the `ValueError` it would
    otherwise raise). `_persistent_leaf_feasible`'s own *second*
    condition (`_leaf_scratchpad_bytes(n) <= scratchpad_byte_budget`,
    evidence-gated against the whole N) does not apply here and must
    not be reused: that condition existed specifically because an
    *unsplit* persistent leaf activates only one of `target.
    num_ndp_units` physical NDP units when `num_logical_blocks` is small
    (typically `batch`, often 1) -- a real, measured, ~50x-slower
    failure mode for exactly the N that condition excludes. A persistent
    leaf reached *through a split* has no such problem: its own
    `num_logical_blocks` is that leaf's own replica count from the split
    (`r`/`r*a`, typically dozens, not 1), fanning real rounds out across
    physical units the same way a cooperative or plain leaf's own launch
    already does -- confirmed on real hardware at parity with a plain
    split (docs/persistent_recursive_split.md's own N=960/1024 numbers,
    1587-vs-1594 and 7688-vs-7688 cycles). Remaining infeasibility this
    cheap check cannot predict (radix-specific `max_kernel_register`
    limits, target-mapping invariants) is caught the same way every
    other per-tier/per-sequence combination in this module already is:
    the actual `make_recursive_transpose_plan` build is wrapped in
    `try`/`except (ValueError, AssertionError, NotImplementedError)` by
    this function's own callers, never scored if it raises.
    """
    return 16 * length <= target.spad_capacity_bytes


def _persistent_leaf_feasible(
    n: int, *, target: TargetProfile, scratchpad_byte_budget: int,
) -> bool:
    """Whether `make_persistent_leaf_plan(n, ...)` can build *and is worth
    offering* as a single *unsplit* leaf covering the whole N -- see
    `_leaf_persistent_feasible` above for the different, per-leaf-inside-
    a-split question `generate_leaf_worker_sequences` needs instead.

    Two independent conditions, both required:

    1. `16 * n <= target.spad_capacity_bytes` -- mirrors `make_persistent_
       leaf_plan`'s own unconditional capacity check (see its own
       docstring for why persistent always needs the full `16 * length`
       bytes regardless of launch width) so a caller can skip the attempt
       instead of catching the `ValueError` it would otherwise raise.

    2. `_leaf_scratchpad_bytes(n) <= scratchpad_byte_budget` -- the
       *cooperative* family's own single-fused-leaf capacity rule
       (`generate_split_candidates`'s own check for offering `None`/no-
       split at all). Required here too, on real-measurement grounds, not
       merely because persistent has no split shape: every representative
       N this project measured (2026-08-30, 5 N where this holds vs. 2
       where it doesn't) that satisfies it had persistent beat the plain
       non-cooperative baseline by 28-46%; every N where it fails
       (N=960, N=1024 -- both need `make_recursive_transpose_plan`'s own
       PRE/MIDDLE/POST-transpose recursion under the cooperative family's
       real budget) had persistent lose by roughly 50x, because a single
       persistent leaf activates only one of `target.num_ndp_units`
       physical NDP units (`num_logical_blocks=batch`, usually 1, active)
       where the split/transpose structure fans real DRAM-tile work out
       across all of them -- a dimension `fft_cost_model.py` does not
       model at all (see docs/execution_cost_model_validation.md's own
       "Persistent as a search axis: representative sweep" section), so
       this condition cannot be left to cost-based ranking to catch. Using
       persistent's own `16 * n <= spad_capacity_bytes` alone (condition 1)
       would offer -- and, worse, cost-rank *first* -- a candidate
       confirmed catastrophically slower than the plan a caller already
       has, at exactly the N where that plan needs to split at all.
    """
    return (
        16 * n <= target.spad_capacity_bytes
        and _leaf_scratchpad_bytes(n) <= scratchpad_byte_budget
    )


def _wrap_persistent_leaf_as_recursive_plan(
    n: int, radices: tuple[int, ...], *, num_logical_blocks: int, inverse: bool,
    target: TargetProfile, batch: int,
) -> RecursiveFFTPlan:
    """`make_persistent_leaf_plan`'s own `FFTCodegenPlan` return, wrapped as
    a single-leaf `RecursiveFFTPlan` -- reusing `FFTLeafPlan`/
    `RecursiveFFTPlan` exactly as any other un-split candidate this module
    builds does, so every existing generic reader (`flatten_recursive_node`,
    `fft_cost_model.compute_stage_metrics`/`estimate_metrics`, `_plan_
    signature`) works on a persistent candidate with no special-casing --
    only `codegen`/`spill_probe`'s own render/probe entry points need to
    branch on `kernel.persistent is not None` (see `generate_recursive_fft_
    kernels` vs. `codegen.fft_persistent_codegen.generate_persistent_fft_
    kernel`, and `planning.spill_probe.probe_spill_free` vs. `probe_
    persistent_kernel_spill_free`), since those are the two places this
    project's own codegen/toolchain path genuinely differs by execution
    model, not the plan-level bookkeeping this wrapper covers.

    `r=num_logical_blocks` (not `batch`): `FFTNode(m, r)`'s own contract is
    "R independent M-point transforms" -- for a persistent leaf that's
    exactly `num_logical_blocks`, its own count of independent logical FFT
    blocks one launch covers (see `PersistentWorkgroupPlan`'s own
    docstring), the persistent-model analogue of what `batch` means for
    every other candidate this module builds.
    """
    kernel = make_persistent_leaf_plan(
        n, radices, num_logical_blocks=num_logical_blocks, inverse=inverse,
        target=target,
    )
    leaf = FFTLeafPlan(m=n, r=num_logical_blocks, kernel=kernel)
    host = MultiKernelHostPlan(n=n, inverse=inverse, tolerance=1.0e-3)
    return RecursiveFFTPlan(n=n, inverse=inverse, root=leaf, host=host, batch=batch)


def generate_persistent_leaf_candidates(
    n: int, *, target: TargetProfile, inverse: bool, batch: int,
    scratchpad_byte_budget: int = 4096,
) -> list[tuple[RecursiveFFTPlan, "PlanChoices"]]:
    """One persistent-leaf candidate per radix tier (`generate_radix_tiers`,
    the same tier set the cooperative-worker joint search offers), each at
    `num_logical_blocks=batch` -- the natural persistent-model analogue of
    "one candidate per (radix, worker) pair" the cooperative axis builds,
    except persistent's own "how many workers" is a target-derived
    constant (`target.interleave_chunk_uthreads`), not a search choice, so
    only radix varies here.

    `[]` (not an exception) when `n` can't build a persistent leaf at all,
    or measured evidence says it isn't worth offering (`_persistent_leaf_
    feasible`'s own docstring -- an N large enough to need `make_recursive
    _transpose_plan`'s own recursion under `scratchpad_byte_budget` loses
    to it by ~50x, not merely "doesn't apply"), or this specific radix
    tier's own coalescing raises for this N (the same "illegal combination
    is pruned before scoring, never surfaced as an error" discipline
    `generate_radix_execution_joint_candidates` already uses) -- so a
    caller (`generate_candidates`) can call this unconditionally for every
    N without its own feasibility check first.
    """
    if not _persistent_leaf_feasible(n, target=target, scratchpad_byte_budget=scratchpad_byte_budget):
        return []
    results: list[tuple[RecursiveFFTPlan, PlanChoices]] = []
    seen_radices: set[tuple[int, ...]] = set()
    for tier_name, allowed in generate_radix_tiers(target):
        radices = coalesce_radices(_prime_factors_supported(n), allowed=allowed)
        if radices in seen_radices:
            continue  # same tier coalescing already tried under a different label
        seen_radices.add(radices)
        try:
            plan = _wrap_persistent_leaf_as_recursive_plan(
                n, radices, num_logical_blocks=batch, inverse=inverse,
                target=target, batch=batch,
            )
        except (ValueError, NotImplementedError, AssertionError):
            continue
        choices = PlanChoices(
            split_near_length=None, radix_tier_name=tier_name, workers_per_fft=None,
            tile=None, execution_strategy="persistent",
        )
        results.append((plan, choices))
    return results


def generate_lane_variant_candidates(
    plan: RecursiveFFTPlan, choices: "PlanChoices", *, compute_lanes: int | None,
) -> list[tuple[RecursiveFFTPlan, "PlanChoices"]]:
    """Two bounded, tree-wide `stage.compute_lanes` variants of `plan` --
    `"unnarrowed"` (every FFT leaf's own middle stages rendered at full
    `compute_lanes`, no halving) and `"all_scalar"` (every stage of every
    leaf floored to `compute_lanes=1`) -- via `planning.fft_plan_lanes.
    apply_compute_lanes_to_plan`/`apply_all_scalar_lanes_to_plan`. The
    third, "baseline narrow_middle_stages=True" shape
    `generate_compute_lane_candidates` (fft_plan_lanes.py) would also
    offer is deliberately *not* added again here: every existing
    candidate this module already builds has `stage.compute_lanes` unset
    (`None`) on every stage, and codegen's own live-fallback default
    (`make_fft_kernel.py`'s shipped `compute_lanes=4, narrow_middle_
    stages=True`) already renders that exact same code at emit time -- a
    plan-level "baseline" candidate here would be cost-scored twice under
    two different `_plan_signature`s for what a real toolchain build
    renders identically, not a genuinely new candidate.

    Deliberately not crossed with every other axis (radix tier, worker
    sequence, split) -- this module's own "one axis at a time, star
    search around a fixed point" discipline (see the module docstring):
    `plan`/`choices` is whatever the caller already picked (typically the
    baseline, the same "hold everything else fixed" pattern step 3's
    worker sweep and step 8's per-leaf worker sweep both already use for
    their own axis), and this adds only the `compute_lanes` axis around
    it. `[]` when `compute_lanes` is `None` (nothing to narrow or floor --
    matches `resolve_stage_compute_lanes`'s own `None`-passthrough rule).
    """
    if compute_lanes is None:
        return []
    unnarrowed = apply_compute_lanes_to_plan(plan, compute_lanes=compute_lanes, narrow_middle_stages=False)
    all_scalar = apply_all_scalar_lanes_to_plan(plan)
    return [
        (unnarrowed, replace(choices, lane_variant="unnarrowed")),
        (all_scalar, replace(choices, lane_variant="all_scalar")),
    ]


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
    per section 10's own "avoid combinatorial explosion" instruction.

    Also includes every divisor of `gcd(rows, cols)` as a square tile --
    these, and *only* these, divide both `rows` and `cols` exactly, so
    `_emit_physical_transpose_stage`'s own `has_row_tail`/`has_col_tail`
    are both `False` and it renders a single fast-path branch with no
    tail case at all. Confirmed real-hardware significant, not just
    tidier code (2026-08-27, N=630's own PRE transpose, rows=105 cols=6,
    gcd=3): direct disassembly showed a tail branch is where LLVM's own
    runtime value-specialization lives (`bnez`/`bne` against exact
    constants) -- tile=4x4/6x6 (each with at least one tail) spilled,
    while tile=1x1/3x3 (gcd(105,6)=3's own divisors, no tail at all) were
    completely clean; tile=2x2/5x5 also had a tail yet didn't spill, so a
    tail alone doesn't guarantee trouble -- only *no* tail guarantees
    safety. Not guaranteed fastest on real cycles (a tail tile can still
    win), but gives planning.spill_probe-driven search at least one
    candidate per shape it never has to actually build+run to trust is
    spill-free at the tile level specifically.
    """
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
    gcd = math.gcd(rows, cols)
    for d in range(1, gcd + 1):
        if gcd % d == 0:
            candidates.add((d, d))
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


def _plan_signature(plan: RecursiveFFTPlan) -> tuple:
    """A canonical signature of everything a `RecursiveFFTPlan`'s own
    already-built structure actually determines about the rendered kernel
    -- radix sequence + cooperative worker count + per-stage compute_lanes
    per leaf, tile size per transpose stage -- used to dedup candidates
    that two different
    `generate_candidates` steps built via different routes (e.g. step 8's
    own single-leaf worker sweep and step 9's own joint radix x worker
    sweep can both produce "leaf 0 at workers=2, every other leaf
    uncooperative, default radix tier" for the exact same N) but that
    resolve to the byte-for-byte identical plan. Deliberately reads the
    plan's own already-built structure rather than trusting `PlanChoices`
    (generation metadata, not guaranteed a perfectly faithful summary --
    see `generate_radix_execution_joint_candidates`'s own dedup, which
    already made the same "realized structure, not the label" choice for
    its own narrower, single-step case)."""
    sig: list[tuple] = []
    for stage in flatten_recursive_node(plan.root):
        if isinstance(stage, PhysicalTransposePlan):
            sig.append(("T", stage.rows, stage.cols, stage.tile_rows, stage.tile_cols))
        else:
            radices = tuple(s.radix for s in stage.stages)
            lanes = tuple(s.compute_lanes for s in stage.stages)
            workers = stage.cooperation.workers_per_fft if stage.cooperation is not None else None
            # Persistent leaves never set `cooperation` (a separate, non-
            # overlapping execution model -- see `FFTCodegenPlan.persistent`'s
            # own docstring), so without this a persistent candidate and a
            # plain non-cooperative candidate of the same radix sequence
            # would collide on an identical ("L", length, radices, None)
            # signature despite rendering completely different code.
            # `total_uthreads`/`max_uthread` also fold in `num_logical_
            # blocks` (persistent's own launch-width driver), which
            # `PersistentWorkgroupPlan` alone does not encode.
            persistent = (
                (stage.persistent, stage.total_uthreads, stage.max_uthread)
                if stage.persistent is not None else None
            )
            sig.append(("L", stage.length, radices, workers, persistent, lanes))
    return tuple(sig)


def _dedup_candidates(candidates: list[FFTPlanCandidate]) -> list[FFTPlanCandidate]:
    """Drop every candidate whose own `_plan_signature` already appeared
    earlier in `candidates` -- first-occurrence order preserved, so
    `candidates[0]` (the baseline, `_select_top_candidates`'s own "always
    kept regardless of score" entry) survives untouched even if some
    later step happens to rebuild the identical plan under a different
    label."""
    seen: set[tuple] = set()
    deduped: list[FFTPlanCandidate] = []
    for c in candidates:
        sig = _plan_signature(c.plan)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(c)
    return deduped


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
    max_joint_worker_sequences: int = 32,
    max_joint_radix_execution_candidates: int = 96,
    compute_lanes: int | None | _Default = _DEFAULT,
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

    `max_joint_worker_sequences`/`max_joint_radix_execution_candidates`:
    step 9's own two caps -- see `generate_radix_execution_joint_candidates`'s
    own docstring (`max_worker_sequences`/`max_joint_candidates` there,
    same meaning, same "compute-time safety valve, not a fairness
    mechanism" discipline as the step 7 caps above).

    `compute_lanes`: `_DEFAULT` (the sentinel, not `None` -- `None` is
    itself a real, distinct meaning, "no lane restriction," matching
    `resolve_stage_compute_lanes`'s own convention) resolves to
    `min(simd_lanes, target.lmul1_float32_lanes)`, the exact value
    `make_fft_kernel.py` ships as its own default -- so step 11's lane-
    variant sweep (below) narrows/floors around the same width a real
    build would actually use unless a caller overrides it. Feeds only
    step 11; every other step's own candidates are unaffected (their own
    `stage.compute_lanes` stays unset, as it always has -- see
    `generate_lane_variant_candidates`'s own docstring for why the
    baseline shape isn't duplicated as a plan-level candidate here).
    """
    if compute_lanes is _DEFAULT:
        compute_lanes = min(simd_lanes, target.lmul1_float32_lanes)

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

    # 9. radix tier x per-leaf worker sequence, evaluated JOINTLY -- split
    # held at baseline's own choice, same discipline steps 3/8 already use.
    # This is the axis the module docstring (and steps 3/4/7/8's own
    # comments) all name as future work: radix composition and execution
    # strategy have never been searched together before this step existed
    # -- step 3/4 each vary one of the two alone around the baseline, and
    # step 7's own joint search explicitly excludes cooperative workers.
    # See generate_radix_execution_joint_candidates' own docstring for the
    # full reasoning; this just crosses it into the same candidate pool
    # everything else here already feeds `_select_top_candidates`.
    for plan, choices in generate_radix_execution_joint_candidates(
        n, target=target, inverse=inverse, scratchpad_byte_budget=scratchpad_byte_budget,
        simd_lanes=simd_lanes, batch=batch, baseline_plan=baseline_plan,
        baseline_split=baseline_split, max_worker_sequences=max_joint_worker_sequences,
        max_joint_candidates=max_joint_radix_execution_candidates,
    ):
        add(plan, choices)

    # 10. persistent-software-workgroup execution -- a wholly separate
    # execution model from every candidate above (all of which are either
    # plain or cooperative-worker leaves; see PersistentWorkgroupPlan's own
    # docstring), only offered when `_persistent_leaf_feasible` says so --
    # both a real capacity check and, since 2026-08-30's representative
    # sweep, an evidence-based gate matching `scratchpad_byte_budget`
    # (see that function's own docstring for the full reasoning and
    # docs/execution_cost_model_validation.md's own "Persistent as a
    # search axis: representative sweep" section for the numbers): every
    # N that would ALSO fit as a single fused leaf under the cooperative
    # family's own budget had persistent beat non-cooperative by 28-46%
    # (N=30,64,105,144,256); every N that needs `make_recursive_transpose_
    # plan`'s own recursion instead had persistent lose by ~50x (N=960,
    # 1024) -- a single persistent leaf only ever activates one of
    # `target.num_ndp_units` physical NDP units, where the split/transpose
    # structure fans real work out across all of them, a dimension the
    # cost model does not represent at all. This gate exists because that
    # failure is NOT caught by cost-based ranking on its own: for N=960,
    # the wrongly-cheap persistent candidate (its single-leaf `estimated_
    # dram_bytes` looks far smaller than the split structure's own,
    # multi-transpose-kernel total) ranked #1 by estimated_cost among 89
    # real candidates despite being confirmed ~50x slower on real
    # hardware -- confirmed before this gate existed, the reason it was
    # added the same session rather than left as a reported risk.
    for plan, choices in generate_persistent_leaf_candidates(
        n, target=target, inverse=inverse, batch=batch,
        scratchpad_byte_budget=scratchpad_byte_budget,
    ):
        add(plan, choices)

    # 11. compute_lanes variant sweep, split held at baseline's own choice
    # (same "hold everything else fixed, vary one axis" discipline every
    # other star-search arm above already uses) -- "unnarrowed" and
    # "all_scalar" tree-wide variants of the baseline plan itself (see
    # generate_lane_variant_candidates' own docstring for why the third,
    # "baseline narrow_middle_stages=True" shape isn't duplicated here).
    # This is the axis Phase 5 (planning.fft_plan_lanes.py) built the
    # mechanism for but never wired into search -- until this step,
    # compute_lanes was reachable only as a flat, whole-tree render-time
    # argument to codegen, invisible to cost-based ranking or the spill
    # probe's own candidate list.
    for plan, choices in generate_lane_variant_candidates(
        baseline_plan,
        PlanChoices(split_near_length=baseline_split, radix_tier_name="default", workers_per_fft=None, tile=None),
        compute_lanes=compute_lanes,
    ):
        add(plan, choices)

    candidates = _dedup_candidates(candidates)
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
    lines.append(f"    total_worker_stage_batches = {m.total_worker_stage_batches}")
    lines.append(f"    radix_risk_score         = {m.radix_risk_score}")
    lines.append(f"    estimated_cost           = {m.estimated_cost:.1f}")
    return "\n".join(lines)
