from __future__ import annotations

"""Diagnostic-only estimate of how many of `target.num_ndp_units` physical
NDP units a plan's own launches concurrently activate -- NOT a cost term.
Nothing here is imported by `fft_cost_model.estimate_cost`/`CostWeights`,
and no `PlanMetrics` field reads it: same "exposed for offline execution-
model comparison, not consumed by estimate_cost" status
`StageExecutionMetrics`/`compute_stage_metrics` already have (see that
module's own docstring), deliberately kept that way here per this
project's own explicit scoping for this phase -- see
docs/active_ndp_units_cost_task.md's "diagnostic first" instruction. Folding
this into a cost weight is later, separate work, gated on the production-
formula data that doc names (real remeasurement of execution strategies at
a fixed real split, once persistent recursive split -- already implemented,
docs/persistent_recursive_split.md -- has more than the 2 existing N=960/
1024 data points), not on anything this module does.

## Re-derived from the real simulator source, not assumed

`codegen.fft_transpose_codegen._safe_round_size`'s own docstring already
flags that its formula rests on two claims "this project has NOT yet
independently verified end to end." This module re-derives the underlying
address-interleaving mechanism directly from the real M2NDP-Detour C++
source instead of trusting that formula (or `TargetProfile`'s own
comments about it) at face value:

* `M2NDPConfig::get_matched_unit_id` (third_party/m2ndp-detour/src/
  m2ndp_config.h:91-92): `unit(addr) = (addr / m_stride_size) %
  m_num_ndp_units`.
* `m_stride_size = 256` (m2ndp_config.h:350) -- confirms
  `TargetProfile.mapping_stride_bytes`.
* `M2NDPConfig::get_uthread_size` (m2ndp_config.cc:112-121): for a launch
  spanning byte range `[base, base+size)`, walks `count` from 0 to `size`
  in `PACKET_SIZE` steps and counts how many packets land on a given
  `ndp_id` via `((base+count) / m_stride_size) % m_num_ndp_units`.
* `PACKET_SIZE = MEM_ACCESS_SIZE = 32` (third_party/m2ndp-detour/src/
  common_defs.h:12,43) -- confirms `TargetProfile.uthread_bytes`.

`distinct_active_units` below is a closed-form reimplementation of that
exact `get_uthread_size` loop (one packet == one microthread's own
32-byte address span, contiguous from a launch's own base address) --
verified byte-for-byte against a brute-force reimplementation of the
literal loop across 200,000 random (base offset, launch size, stride,
unit count) combinations, including wraparound (a launch spanning more
than one full `interleave_chunk_uthreads * num_ndp_units`-microthread
period) -- see `verification/verify_fft_unit_utilization.py`.

## The one thing this derivation cannot know at plan time: the real base address

`get_matched_unit_id` keys off the launch pool's real, absolute DRAM
address -- decided by the host allocator at run time, not visible to this
planning-time module. Two real, previously-confirmed-on-hardware facts
bound how much this matters:

* [[fft-cooperative-worker8-pool-alignment]] / `docs/
  cooperative_worker8_pool_alignment_fix.md`: a cooperative leaf's launch
  pool is deliberately over-allocated and rounded up by hand to a full
  256-byte (`interleave_chunk_uthreads * uthread_bytes`) boundary --
  i.e. `base_offset_uthreads == 0` exactly, not merely likely.
* `docs/persistent_recursive_split.md`'s own point 5: the identical 256B
  alignment fix applies to a persistent stage too (`stage.cooperation is
  not None or stage.persistent is not None` gates it in
  `fft_transpose_codegen.py`) -- same exact-zero-offset guarantee.

Neither guarantee is in place for a plain non-cooperative leaf or a
`PhysicalTransposePlan` (PRE/MIDDLE/POST) launch -- both still go through
the default `Pool.alloc`, confirmed only 64-byte-aligned
(`cooperative_worker8_pool_alignment_fix.md`'s own root-cause section).
64 bytes is 2 microthreads' worth (`uthread_bytes=32`), so for those two
families `base_offset_uthreads` is only known to be even, one of
`{0, 2, 4, 6}` (mod `interleave_chunk_uthreads=8`) -- genuinely unknown at
plan time, not assumed 0. This module reports both ends of that range
(`active_units_best`/`active_units_worst`) for exactly those two families
rather than picking one and hiding the uncertainty; cooperative and
persistent get a single exact number (`alignment_exact=True`).
"""

from dataclasses import dataclass

from planning.fft_plan_core import FFTCodegenPlan
from planning.fft_plan_persistent import num_rounds, round_active_groups
from planning.fft_plan_recursive import (
    FFTLeafPlan,
    FFTNode,
    FFTRecursiveNodePlan,
    PhysicalTransposePlan,
    RecursiveFFTPlan,
)
from planning.target_profile import TargetProfile

# The maximum possible base-offset skew for a pool guaranteed only 64-byte
# (2-microthread) aligned, in units of `interleave_chunk_uthreads` -- see
# this module's own docstring. `range(0, interleave_chunk_uthreads, 2)`
# would be the real member enumeration; only the max (worst case for
# chunks-spanned) and 0 (best case) actually change the result of
# `distinct_active_units`, so callers need just those two, not the full set.
_MAX_64B_ALIGNED_OFFSET_STEP = 2


def distinct_active_units(
    logical_uthreads: int,
    *,
    base_offset_uthreads: int,
    interleave_chunk_uthreads: int,
    num_ndp_units: int,
) -> int:
    """How many distinct physical NDP units a contiguous launch of
    `logical_uthreads` microthreads activates, starting `base_offset_
    uthreads` microthreads into an `interleave_chunk_uthreads`-wide
    address-interleave chunk -- closed form for `M2NDPConfig::
    get_uthread_size`'s own per-packet loop (see module docstring),
    verified against a brute-force reimplementation of that literal loop.

    Only `base_offset_uthreads % interleave_chunk_uthreads` matters (which
    *specific* unit a launch starts on is a pure rotation that never
    changes how many distinct units get touched); the result is exact for
    any launch size, including one spanning multiple full
    `interleave_chunk_uthreads * num_ndp_units`-microthread wraparound
    periods (capped at `num_ndp_units` -- once every unit has been
    touched, a bigger launch cannot activate more).
    """
    if logical_uthreads <= 0:
        return 0
    offset_in_chunk = base_offset_uthreads % interleave_chunk_uthreads
    chunks_spanned = -(-(offset_in_chunk + logical_uthreads) // interleave_chunk_uthreads)
    return min(num_ndp_units, chunks_spanned)


@dataclass(frozen=True)
class UnitUtilizationEstimate:
    """One launch's own (logical_work_count, active-unit-count) pair --
    diagnostic only, see module docstring. `family` is one of
    `"non_cooperative"`, `"cooperative"`, `"persistent"`,
    `"transpose_pre"`, `"transpose_middle"`, `"transpose_post"`."""

    family: str
    # None for a transpose stage; one entry per leaf KERNEL, not per
    # `FFTStagePlan` within it -- every stage of one leaf shares the same
    # launch_uthreads/round shape, so a per-stage breakdown would be
    # redundant here (unlike `fft_cost_model.StageExecutionMetrics`, which
    # genuinely varies per stage).
    leaf_index: int | None
    # The physical launch width this estimate is over: `total_uthreads`
    # for every family except persistent, where it's `num_logical_blocks`
    # (persistent's own `total_uthreads` is a target-fixed launch width
    # independent of replica count -- see this module's own docstring and
    # docs/persistent_recursive_split.md's point 4 -- so counting launch_
    # uthreads there would silently always report the same number
    # regardless of how much real work this leaf does).
    logical_work_count: int
    active_units_best: int
    active_units_worst: int
    # True (cooperative, persistent): base_offset_uthreads is a confirmed
    # exact 0, best == worst. False (non_cooperative, every transpose
    # family): only 64B/2-microthread alignment is guaranteed, so best/
    # worst is a genuine range, not a rounding artifact.
    alignment_exact: bool


def _family_active_units(
    logical_uthreads: int, *, alignment_exact: bool, target: TargetProfile,
) -> tuple[int, int]:
    if alignment_exact:
        units = distinct_active_units(
            logical_uthreads, base_offset_uthreads=0,
            interleave_chunk_uthreads=target.interleave_chunk_uthreads,
            num_ndp_units=target.num_ndp_units,
        )
        return units, units
    best = distinct_active_units(
        logical_uthreads, base_offset_uthreads=0,
        interleave_chunk_uthreads=target.interleave_chunk_uthreads,
        num_ndp_units=target.num_ndp_units,
    )
    worst = distinct_active_units(
        logical_uthreads,
        base_offset_uthreads=target.interleave_chunk_uthreads - _MAX_64B_ALIGNED_OFFSET_STEP,
        interleave_chunk_uthreads=target.interleave_chunk_uthreads,
        num_ndp_units=target.num_ndp_units,
    )
    return best, worst


def _persistent_leaf_estimate(
    codegen_plan: FFTCodegenPlan, replicas: int, *, leaf_index: int, target: TargetProfile,
) -> UnitUtilizationEstimate:
    """`round_active_groups`/`num_rounds` (fft_plan_persistent.py) are
    reused unchanged, not re-derived: a persistent leaf's own software-
    group-to-physical-unit mapping is 1:1 by construction, not something
    this module's own address-interleave formula needs to re-derive.
    `PersistentWorkgroupPlan.workers_per_group` is asserted equal to
    `target.interleave_chunk_uthreads` at build time
    (fft_plan_persistent.make_persistent_leaf_plan) specifically so that
    software group `g`'s own `workers_per_group` consecutive microthreads
    coincide exactly with the `interleave_chunk_uthreads` consecutive
    microthreads the address decoder already places on one physical unit
    together -- given the confirmed-exact 256B pool alignment (see module
    docstring), `software_group_id() == physical_unit_id` directly, no
    separate address-interleave arithmetic needed. Worst case reported is
    the plan's own tail round (`round_active_groups` at the last round
    index), which can be far smaller than `num_ndp_units` when `replicas`
    isn't a multiple of it -- best case is a full round
    (`min(replicas, num_ndp_units)`)."""
    persistent = codegen_plan.persistent
    assert persistent is not None
    rounds = num_rounds(replicas, persistent.software_group_count)
    full_round = round_active_groups(0, replicas, persistent.software_group_count)
    tail_round = round_active_groups(rounds - 1, replicas, persistent.software_group_count)
    return UnitUtilizationEstimate(
        family="persistent",
        leaf_index=leaf_index,
        logical_work_count=replicas,
        active_units_best=full_round,
        active_units_worst=min(full_round, tail_round) if rounds > 0 else 0,
        alignment_exact=True,
    )


def _emit_transpose_estimate(
    stage: PhysicalTransposePlan, *, family: str, target: TargetProfile,
) -> UnitUtilizationEstimate:
    best, worst = _family_active_units(
        stage.total_uthreads, alignment_exact=False, target=target,
    )
    return UnitUtilizationEstimate(
        family=family, leaf_index=None,
        logical_work_count=stage.total_uthreads,
        active_units_best=best, active_units_worst=worst,
        alignment_exact=False,
    )


def _emit_leaf_estimate(
    leaf: FFTLeafPlan, *, leaf_index: int, target: TargetProfile,
) -> UnitUtilizationEstimate:
    codegen_plan = leaf.kernel
    if codegen_plan.persistent is not None:
        return _persistent_leaf_estimate(
            codegen_plan, leaf.r, leaf_index=leaf_index, target=target,
        )
    alignment_exact = codegen_plan.cooperation is not None
    family = "cooperative" if alignment_exact else "non_cooperative"
    best, worst = _family_active_units(
        codegen_plan.total_uthreads, alignment_exact=alignment_exact, target=target,
    )
    return UnitUtilizationEstimate(
        family=family, leaf_index=leaf_index,
        logical_work_count=codegen_plan.total_uthreads,
        active_units_best=best, active_units_worst=worst,
        alignment_exact=alignment_exact,
    )


def _walk_node(
    node: FFTNode, *, target: TargetProfile, results: list[UnitUtilizationEstimate],
    next_leaf_index: list[int],
) -> None:
    """Mirrors `flatten_recursive_node`'s own recursion (PRE -> near_fft
    -> MIDDLE -> far_child -> POST) exactly, but -- unlike that function,
    which deliberately returns an untagged flat list, see its own
    docstring -- tags each `PhysicalTransposePlan` with which of PRE/
    MIDDLE/POST it structurally is at this recursion level, which nothing
    in the plan's own data carries (position in the recursion IS the
    label). `next_leaf_index` is a one-element mutable box (not a plain
    int) so this recursive walk can share one counter across calls the
    same way a loop variable would in `fft_cost_model.compute_stage_
    metrics`'s own flat-list walk."""
    if isinstance(node, FFTLeafPlan):
        results.append(
            _emit_leaf_estimate(node, leaf_index=next_leaf_index[0], target=target)
        )
        next_leaf_index[0] += 1
        return
    assert isinstance(node, FFTRecursiveNodePlan)
    results.append(
        _emit_transpose_estimate(node.pre_transpose, family="transpose_pre", target=target)
    )
    _walk_node(node.near_fft, target=target, results=results, next_leaf_index=next_leaf_index)
    results.append(
        _emit_transpose_estimate(node.middle_transpose, family="transpose_middle", target=target)
    )
    _walk_node(node.far_child, target=target, results=results, next_leaf_index=next_leaf_index)
    results.append(
        _emit_transpose_estimate(node.post_transpose, family="transpose_post", target=target)
    )


def compute_unit_utilization(
    plan: RecursiveFFTPlan, target: TargetProfile,
) -> list[UnitUtilizationEstimate]:
    """Every leaf kernel and transpose stage in `plan`, each as one
    `UnitUtilizationEstimate` -- diagnostic only, see module docstring.
    Ordering and `leaf_index` convention match `fft_cost_model.compute_
    stage_metrics`'s own (execution order, leaf kernels counted only)."""
    results: list[UnitUtilizationEstimate] = []
    _walk_node(plan.root, target=target, results=results, next_leaf_index=[0])
    return results
