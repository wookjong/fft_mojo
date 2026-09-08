from __future__ import annotations

"""compute_lanes as a planning decision, not a codegen one (Phase 5 of
docs/cooperative_worker8_pool_alignment_fix.md's own phase list).

Before this module existed, `codegen.fft_codegen._stage_compute_lanes`
computed each stage's own render width live, at emit time, from the
caller's flat `compute_lanes`/`narrow_middle_stages` plus that one
stage's own `radix`/`prev_radix`/position -- a real, confirmed-necessary
register-pressure safety heuristic (see this module's own constants'
comments for the real-hardware evidence), but a *planning* decision
(which stage renders how wide) made inside codegen, which the project's
own architecture wants codegen never to do (`FFTCodegenPlan` should be
able to answer "what does this leaf render as" before codegen ever runs).

This module is the single owner of that heuristic now:
`resolve_stage_compute_lanes` is the exact same three-reason logic
`_stage_compute_lanes` used to own (moved, not reimplemented -- codegen
imports it back for its own now-fallback-only use, see that function's
own updated docstring), and `apply_compute_lanes`/`generate_compute_lane_
candidates` are the new planner-side entry points that actually populate
`FFTStagePlan.compute_lanes` on a built plan, so a caller building a plan
can decide every stage's own width before codegen ever sees it.
"""

from dataclasses import replace

from planning.core.fft_plan_core import FFTCodegenPlan
from planning.strategies.fft_plan_recursive import FFTLeafPlan, FFTNode, FFTRecursiveNodePlan, RecursiveFFTPlan

# See codegen.fft_codegen's own former copy of this comment (moved here
# unchanged) for the real-hardware evidence: N=11/13/17 standalone, single-
# stage, MISMATCH at compute_lanes=4 *and* 2 with the identical `csrr ...,
# vlenb` dynamic-spill-slot shape confirmed elsewhere in this project (the
# M2NDP-Detour `ReadCsr` gap) -- only compute_lanes=1 (fully scalar) passes,
# so these radices floor outright rather than merely halving. `10` was
# already known always-risky; radix 5/7 are deliberately NOT here (their
# own risk is confirmed only in the middle-stage shape `narrow_middle_
# stages` already covers, where a halving is enough).
_ALWAYS_NARROW_RADICES = frozenset({10, 11, 13, 17})

# N=54=(6,9): a 2-stage leaf whose *second* stage (radix 9, reading its
# operands out of scratchpad) is technically "last", not "middle" -- see
# codegen.fft_codegen's own former copy of this comment for why this pair
# is keyed on the *adjacent* (prev_radix, radix), not radix 9 alone: a
# direct (4, 9) chain ran clean, so whatever makes (6, 9) risky is specific
# to that sequence. Only the one real-hardware-confirmed pair is listed.
_RISKY_RADIX_PAIRS = frozenset({(6, 9)})


def resolve_stage_compute_lanes(
    *, compute_lanes: int | None, is_first: bool, is_last: bool, radix: int,
    narrow_middle_stages: bool, prev_radix: int | None = None,
) -> int | None:
    """Three independent, real-hardware-confirmed reasons to narrow the
    caller's own already-decided `compute_lanes`, each with its own
    confirmed-sufficient reduction -- more than one can apply at once
    (e.g. a radix-11 middle stage), in which case the floor-to-1 wins
    since it's the strongest. Moved verbatim from `codegen.fft_codegen.
    _stage_compute_lanes` (same three reasons, same evidence, same
    behavior for every existing caller) -- see that function's own
    former docstring for the full real-hardware evidence behind each:

    1. `narrow_middle_stages` and this is a *middle* stage (neither first
       nor last) -- halving (floor 1) confirmed sufficient.
    2. `radix in _ALWAYS_NARROW_RADICES` -- unconditional, floors outright
       (a halving is not enough for these radices).
    3. `(prev_radix, radix) in _RISKY_RADIX_PAIRS` -- unconditional,
       floors outright.

    `None` (no explicit compute_lanes -- "render at plan.simd_lanes") is
    left alone regardless of any reason: there is no already-decided
    width to narrow.
    """
    if compute_lanes is None:
        return compute_lanes
    if radix in _ALWAYS_NARROW_RADICES:
        return 1
    if prev_radix is not None and (prev_radix, radix) in _RISKY_RADIX_PAIRS:
        return 1
    is_middle = not is_first and not is_last
    if narrow_middle_stages and is_middle:
        return max(1, compute_lanes // 2)
    return compute_lanes


def _stage_lanes_for_plan(
    plan: FFTCodegenPlan, *, compute_lanes: int | None, narrow_middle_stages: bool,
) -> list[int | None]:
    lanes: list[int | None] = []
    for stage in plan.stages:
        is_first = stage.stage_id == 0
        is_last = stage.stage_id == len(plan.stages) - 1
        prev_radix = plan.stages[stage.stage_id - 1].radix if not is_first else None
        lanes.append(
            resolve_stage_compute_lanes(
                compute_lanes=compute_lanes, is_first=is_first, is_last=is_last,
                radix=stage.radix, narrow_middle_stages=narrow_middle_stages,
                prev_radix=prev_radix,
            )
        )
    return lanes


def apply_compute_lanes(
    plan: FFTCodegenPlan, *, compute_lanes: int | None, narrow_middle_stages: bool = False,
) -> FFTCodegenPlan:
    """Return a plan identical to `plan` except every stage's own
    `compute_lanes` field is populated by `resolve_stage_compute_lanes`
    -- the "baseline" candidate `generate_compute_lane_candidates` (below)
    always includes, and the one a caller reaches for directly when it
    just wants today's exact heuristic as an explicit per-stage plan
    instead of a codegen-time computation. A plan built this way needs no
    further `compute_lanes`/`narrow_middle_stages` arguments passed to
    codegen at all -- see `codegen.fft_codegen._emit_stage`'s own updated
    docstring for how it prefers `stage.compute_lanes` once set.
    """
    new_stages = tuple(
        replace(stage, compute_lanes=lanes)
        for stage, lanes in zip(
            plan.stages,
            _stage_lanes_for_plan(
                plan, compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
            ),
        )
    )
    return replace(plan, stages=new_stages)


def generate_compute_lane_candidates(
    plan: FFTCodegenPlan, *, compute_lanes: int | None,
) -> list[FFTCodegenPlan]:
    """A small, bounded set of per-stage compute_lanes candidates for
    `plan` -- deliberately not `{1, 2, 4}^stage_count` (the plan's own
    explicit "lane candidate explosion 방지" rule: real measurement found
    narrower is not always safer, e.g. persistent leaves preferring wide
    lanes, so a monotonic-fallback sweep would both explode and miss the
    real optimum). Every candidate here is plan-equivalent (same radix,
    same execution strategy) and differs *only* in `stage.compute_lanes`
    -- `fft_plan_search.py`'s own `_plan_signature` already keys on this
    field, so two of these never collapse into "the same candidate" by
    accident.

    Candidates, in order:

    A. baseline -- `apply_compute_lanes` with `narrow_middle_stages=True`,
       today's exact shipped default (`make_fft_kernel.py`'s own).
    B. unnarrowed -- `narrow_middle_stages=False`: the flat-width shape,
       for comparison against A (a caller wanting to confirm A's own
       narrowing is actually buying something for this specific plan,
       not paid unconditionally).
    C. all-scalar -- every stage floored to `compute_lanes=1`, the one
       width real-hardware evidence never found *wrong*, only sometimes
       slower (`_ALWAYS_NARROW_RADICES`/`_RISKY_RADIX_PAIRS` both floor
       to exactly this) -- a safety-first floor candidate distinct from
       "narrow the middle only."

    `compute_lanes=None` collapses every candidate to the same "render at
    plan.simd_lanes" plan (matches `resolve_stage_compute_lanes`'s own
    `None`-passthrough rule) -- returned as a single one-element list
    rather than three identical plans, so a caller sweeping candidates
    doesn't pay for or report duplicate work.
    """
    if compute_lanes is None:
        return [apply_compute_lanes(plan, compute_lanes=None, narrow_middle_stages=False)]

    baseline = apply_compute_lanes(plan, compute_lanes=compute_lanes, narrow_middle_stages=True)
    unnarrowed = apply_compute_lanes(plan, compute_lanes=compute_lanes, narrow_middle_stages=False)
    all_scalar = replace(
        plan,
        stages=tuple(replace(stage, compute_lanes=1) for stage in plan.stages),
    )
    candidates = [baseline, unnarrowed, all_scalar]

    # Dedup: distinct only by their own per-stage compute_lanes tuple --
    # e.g. a plan with no middle stage at all makes A and B identical.
    seen: set[tuple[int | None, ...]] = set()
    deduped: list[FFTCodegenPlan] = []
    for candidate in candidates:
        key = tuple(stage.compute_lanes for stage in candidate.stages)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _apply_to_node(
    node: FFTNode, *, compute_lanes: int | None, narrow_middle_stages: bool, all_scalar: bool,
) -> FFTNode:
    """Every leaf `apply_compute_lanes`/`generate_compute_lane_candidates`
    already knows how to handle one `FFTCodegenPlan` at a time -- a
    `RecursiveFFTPlan`'s own leaves sit nested inside `FFTLeafPlan`/
    `FFTRecursiveNodePlan` (a frozen-dataclass tree, not a flat list), so
    setting every leaf's own `compute_lanes` tree-wide needs this
    recursive rewrite instead of a single `apply_compute_lanes` call.
    `PhysicalTransposePlan` (PRE/MIDDLE/POST) stages are untouched --
    `compute_lanes` is an FFT-arithmetic-stage concept only (see
    `fft_transpose_codegen.generate_recursive_fft_kernels`'s own
    docstring: transpose kernels have their own separate width story).
    """
    if isinstance(node, FFTLeafPlan):
        if all_scalar:
            new_kernel = replace(
                node.kernel,
                stages=tuple(replace(s, compute_lanes=1) for s in node.kernel.stages),
            )
        else:
            new_kernel = apply_compute_lanes(
                node.kernel, compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
            )
        return replace(node, kernel=new_kernel)
    assert isinstance(node, FFTRecursiveNodePlan)
    return replace(
        node,
        near_fft=_apply_to_node(
            node.near_fft, compute_lanes=compute_lanes,
            narrow_middle_stages=narrow_middle_stages, all_scalar=all_scalar,
        ),
        far_child=_apply_to_node(
            node.far_child, compute_lanes=compute_lanes,
            narrow_middle_stages=narrow_middle_stages, all_scalar=all_scalar,
        ),
    )


def apply_compute_lanes_to_plan(
    plan: RecursiveFFTPlan, *, compute_lanes: int | None, narrow_middle_stages: bool = False,
) -> RecursiveFFTPlan:
    """Tree-wide `apply_compute_lanes`: every FFT leaf in `plan`'s own
    split tree gets its own `compute_lanes` populated by `resolve_stage_
    compute_lanes`, the same rule every leaf would already resolve to
    individually -- one flat `compute_lanes`/`narrow_middle_stages`
    applied uniformly across the whole tree, not a per-leaf choice (see
    `generate_lane_variant_candidates` in `fft_plan_search.py` for the
    bounded per-plan candidate set built from this)."""
    return replace(
        plan,
        root=_apply_to_node(
            plan.root, compute_lanes=compute_lanes,
            narrow_middle_stages=narrow_middle_stages, all_scalar=False,
        ),
    )


def apply_all_scalar_lanes_to_plan(plan: RecursiveFFTPlan) -> RecursiveFFTPlan:
    """Tree-wide floor: every stage of every FFT leaf in `plan`'s own
    split tree rendered at `compute_lanes=1` -- the one width real-
    hardware evidence never found *wrong*, only sometimes slower (see
    `generate_compute_lane_candidates`'s own "all-scalar" candidate,
    applied here tree-wide instead of to one leaf)."""
    return replace(
        plan,
        root=_apply_to_node(
            plan.root, compute_lanes=None, narrow_middle_stages=False, all_scalar=True,
        ),
    )
