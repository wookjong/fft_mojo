from __future__ import annotations

"""Regression coverage for the mechanism-aware execution-cost correction
integrated into `planning/fft_cost_model.py` (2026-09-01):
`CostWeights.persistent_stage_batch_multiplier` (1.746) and
`persistent_extra_round_multiplier` (11.0), plus the two new `PlanMetrics`
fields feeding them (`persistent_worker_stage_batches`,
`persistent_extra_rounds`). Full derivation and real-hardware validation:
docs/active_ndp_units_cost_task.md's "Mechanism-Aware Correction" and
"Production Integration" sections; raw datasets: `docs/
cost_model_revalidation.csv`, `docs/cost_model_saturation.csv`.

Root cause this closes out: a prior pass found real hardware directly
FALSIFIES "just increase stage_work" -- persistent has the systematically
LOWEST `total_worker_stage_batches` at every N tested, so scaling one
positive weight up only makes an already-wrong pick look more attractive,
never less. The fix instead gives persistent's own batches a different,
data-derived per-batch rate and adds an explicit extra-round term for
persistent leaves whose own replica count exceeds one round's worth of
physical NDP units -- non-cooperative/cooperative leaves are completely
unaffected (both new metrics are exactly 0 for them, always).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.search.fft_cost_model import (
    DEFAULT_COST_WEIGHTS,
    estimate_cost,
    estimate_metrics,
)
from planning.strategies.fft_plan_recursive import make_recursive_transpose_plan
from planning.core.target_profile import DEFAULT_TARGET_PROFILE


def _plan(n: int, **kwargs):
    return make_recursive_transpose_plan(
        n, scratchpad_byte_budget=4096, simd_lanes=8,
        spad_capacity_bytes=DEFAULT_TARGET_PROFILE.spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=DEFAULT_TARGET_PROFILE.max_concurrent_scratchpad_bytes,
        **kwargs,
    )


def check_non_persistent_plans_byte_identical_to_pre_correction_formula() -> None:
    """A plan with no persistent leaf at all must compute an IDENTICAL
    estimated_cost to before this correction existed -- `persistent_
    worker_stage_batches`/`persistent_extra_rounds` are exactly 0 there,
    so `_execution_cost`'s own new weighted-sum formula collapses back to
    `stage_work * total_worker_stage_batches` exactly. Checked across
    non-cooperative and cooperative(workers=2/4/8) at a real split N."""
    n = 960
    for kwargs in ({}, {"cooperative_workers": 2}, {"cooperative_workers": 4}, {"cooperative_workers": 8}):
        plan = _plan(n, **kwargs)
        metrics = estimate_metrics(plan, DEFAULT_TARGET_PROFILE)
        assert metrics.persistent_worker_stage_batches == 0, (
            f"expected 0 persistent batches for a non-persistent plan, got "
            f"{metrics.persistent_worker_stage_batches} ({kwargs})"
        )
        assert metrics.persistent_extra_rounds == 0
        cost = estimate_cost(metrics, DEFAULT_COST_WEIGHTS)
        expected = (
            metrics.estimated_dram_bytes * DEFAULT_COST_WEIGHTS.memory_traffic
            + DEFAULT_COST_WEIGHTS.transpose_passes * metrics.transpose_kernel_count
            + DEFAULT_COST_WEIGHTS.recursion_depth_penalty * metrics.recursion_depth
            + DEFAULT_COST_WEIGHTS.transpose_tile_count * metrics.total_transpose_tiles
            + DEFAULT_COST_WEIGHTS.transpose_tail_risk_penalty * metrics.transpose_tail_tile_count
            + DEFAULT_COST_WEIGHTS.stage_work * metrics.total_leaf_stage_count
            + DEFAULT_COST_WEIGHTS.radix_risk_penalty * metrics.radix_risk_score
            + DEFAULT_COST_WEIGHTS.stage_work * metrics.total_worker_stage_batches
        )
        assert abs(cost - expected) < 1e-6, (
            f"non-persistent plan ({kwargs}) cost changed shape: {cost} != {expected}"
        )
    print(f"    OK   N={n}: non-persistent plans (non-coop, coop=2/4/8) compute byte-identical "
          f"estimated_cost to the pre-correction formula (persistent terms are exactly 0)")


def check_n216_regression_fixed() -> None:
    """The primary single-round persistent regression this correction was
    built for: N=216, real best is `cooperative_workers=4` (22316 cycles,
    docs/compute_lanes_spill_avoidance.md), but the pre-correction model
    always picked persistent (9 batches < cooperative's 15, regardless of
    `stage_work`'s own value -- a `stage_work` reweight alone cannot fix
    this, confirmed separately). `persistent_extra_rounds == 0` here (a
    single N=216 leaf's own replica count is 1, far under one round's
    worth of NDP units) -- success depends entirely on `persistent_
    stage_batch_multiplier`, not the round term."""
    n = 216
    noncoop = _plan(n)
    coop4 = _plan(n, cooperative_workers=4)
    persistent = _plan(n, persistent_leaf=True)

    m_persistent = estimate_metrics(persistent, DEFAULT_TARGET_PROFILE)
    assert m_persistent.persistent_extra_rounds == 0, (
        "expected N=216's persistent leaf to fit in a single round -- "
        "this test is specifically about the batch multiplier, not the round term"
    )
    assert m_persistent.persistent_worker_stage_batches > 0

    cost_noncoop = estimate_cost(estimate_metrics(noncoop, DEFAULT_TARGET_PROFILE), DEFAULT_COST_WEIGHTS)
    cost_coop4 = estimate_cost(estimate_metrics(coop4, DEFAULT_TARGET_PROFILE), DEFAULT_COST_WEIGHTS)
    cost_persistent = estimate_cost(m_persistent, DEFAULT_COST_WEIGHTS)

    assert cost_coop4 < cost_persistent, (
        f"expected cooperative_workers=4 to rank cheapest at N={n} (real best, 22316 cycles vs. "
        f"persistent's 23941), got cost_coop4={cost_coop4} >= cost_persistent={cost_persistent}"
    )
    assert cost_coop4 < cost_noncoop
    print(f"    OK   N={n}: cooperative_workers=4 now correctly ranks cheapest "
          f"(cost={cost_coop4:.2f} < persistent={cost_persistent:.2f} < noncoop={cost_noncoop:.2f}) "
          f"-- matches real hardware (22316 < 23941 < 43771 cycles)")


def check_n960_mixed_leaf_not_broken() -> None:
    """The mixed-leaf regression fixture (near=persistent, far=non-
    cooperative, real 31689 cycles, this project's own fastest N=960
    candidate among the 6 compared in docs/
    active_ndp_units_cost_task.md) must stay fully reachable and ranked
    well ahead of every UNIFORM persistent-only candidate -- the required
    invariant this whole correction was checked against before
    integration ("persistent is not killed by a blanket penalty where
    it's actually needed"). Not required to beat every candidate outright
    (a real, understood, un-chased 6.1% residual remains against
    cooperative w=8 -- see docs/active_ndp_units_cost_task.md) -- only
    required to stay far ahead of uniform persistent and non-cooperative,
    which a broken/overcorrected weight could plausibly destroy."""
    n = 960
    mixed = _plan(n, forced_worker_sequence=("persistent", None))
    uniform_persistent = _plan(n, persistent_leaf=True)
    noncoop = _plan(n)

    cost_mixed = estimate_cost(estimate_metrics(mixed, DEFAULT_TARGET_PROFILE), DEFAULT_COST_WEIGHTS)
    cost_uniform_persistent = estimate_cost(
        estimate_metrics(uniform_persistent, DEFAULT_TARGET_PROFILE), DEFAULT_COST_WEIGHTS
    )
    cost_noncoop = estimate_cost(estimate_metrics(noncoop, DEFAULT_TARGET_PROFILE), DEFAULT_COST_WEIGHTS)

    assert cost_mixed < cost_uniform_persistent, (
        f"mixed candidate (real 31689 cycles) must rank cheaper than uniform persistent "
        f"(real 39378 cycles): got mixed={cost_mixed} >= uniform_persistent={cost_uniform_persistent}"
    )
    assert cost_mixed < cost_noncoop, (
        f"mixed candidate must rank cheaper than non-cooperative baseline (real 48320 cycles): "
        f"got mixed={cost_mixed} >= noncoop={cost_noncoop}"
    )
    print(f"    OK   N={n}: mixed-leaf candidate (near=persistent/far=non-coop) stays ranked "
          f"well ahead of uniform persistent ({cost_mixed:.2f} < {cost_uniform_persistent:.2f}) "
          f"and non-cooperative ({cost_mixed:.2f} < {cost_noncoop:.2f}) -- not broken by this correction")


def check_persistent_extra_rounds_only_from_persistent_leaves() -> None:
    """A leaf with a large replica count that is NOT persistent must
    never contribute to `persistent_extra_rounds` -- only a genuinely
    persistent leaf whose own replica count exceeds `target.num_ndp_units`
    does. N=960's own far_child has 240 replicas (far more than 32) --
    confirmed 0 rounds when non-cooperative, and exactly `ceil(240/32) -
    1 == 7` when persistent."""
    n = 960
    noncoop = _plan(n)
    persistent = _plan(n, persistent_leaf=True)

    m_noncoop = estimate_metrics(noncoop, DEFAULT_TARGET_PROFILE)
    m_persistent = estimate_metrics(persistent, DEFAULT_TARGET_PROFILE)

    assert m_noncoop.persistent_extra_rounds == 0, (
        "a non-cooperative leaf, however many replicas it has, must never "
        "contribute to persistent_extra_rounds"
    )
    assert m_persistent.persistent_extra_rounds == 7, (
        f"expected far_child's own 240 replicas over 32 NDP units to need "
        f"ceil(240/32)-1 == 7 extra rounds, got {m_persistent.persistent_extra_rounds}"
    )
    print(f"    OK   N={n}: persistent_extra_rounds is 0 for a non-cooperative leaf regardless of "
          f"replica count, and exactly 7 for persistent's own far_child (240 replicas / 32 units)")


def check_coefficients_frozen() -> None:
    """Phase 1 freeze (docs/active_ndp_units_cost_task.md's "Cost model
    freeze" task, 2026-09-01): `persistent_stage_batch_multiplier`/
    `persistent_extra_round_multiplier` must not be retuned until a
    holdout validation on unseen N either confirms they generalize or
    finds a systematic failure that structurally requires a different
    value -- see that doc's own Phase 8 ("persistent model은 당분간
    freeze"). This test exists purely so an accidental or well-intentioned
    "let's nudge this a little" edit fails loudly instead of silently
    invalidating the holdout comparison's own frozen baseline."""
    assert DEFAULT_COST_WEIGHTS.persistent_stage_batch_multiplier == 1.746, (
        "persistent_stage_batch_multiplier changed -- frozen pending holdout validation, "
        "see docs/active_ndp_units_cost_task.md's Phase 1/8"
    )
    assert DEFAULT_COST_WEIGHTS.persistent_extra_round_multiplier == 11.0, (
        "persistent_extra_round_multiplier changed -- frozen pending holdout validation, "
        "see docs/active_ndp_units_cost_task.md's Phase 1/8"
    )
    print("    OK   persistent_stage_batch_multiplier=1.746 and persistent_extra_round_multiplier=11.0 "
          "unchanged (frozen pending holdout validation)")


def check_deterministic() -> None:
    """Same plan, same weights -> byte-for-byte identical estimated_cost
    across repeated calls -- the new terms introduce no hidden state."""
    n = 960
    plan = _plan(n, persistent_leaf=True)
    costs = [estimate_cost(estimate_metrics(plan, DEFAULT_TARGET_PROFILE)) for _ in range(3)]
    assert len(set(costs)) == 1, f"non-deterministic cost across repeated calls: {costs}"
    print(f"    OK   N={n}: persistent plan's own estimated_cost is deterministic "
          f"across 3 repeated calls (cost={costs[0]:.3f})")


def main() -> None:
    print("  fft_cost_model.py: mechanism-aware execution-cost correction (2026-09-01):")
    check_non_persistent_plans_byte_identical_to_pre_correction_formula()
    check_n216_regression_fixed()
    check_n960_mixed_leaf_not_broken()
    check_persistent_extra_rounds_only_from_persistent_leaves()
    check_coefficients_frozen()
    check_deterministic()
    print("[verify] mechanism-aware cost correction: all checks passed")


if __name__ == "__main__":
    main()
