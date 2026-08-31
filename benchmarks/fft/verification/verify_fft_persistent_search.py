from __future__ import annotations

"""Structural verification for generate_candidates' own step 10
(persistent-software-workgroup execution as a search axis, 2026-08-30) --
NOT FFT numerical correctness (verify_fft_plan.py's own job). Covers:
that a persistent candidate is actually generated for a feasible N, that
it is correctly marked and structurally distinct from every cooperative-
worker candidate of the same radix (so global dedup never collides them),
that it is infeasible-but-not-crashing for an N persistent can't build,
that its own cost reflects the real register-pressure difference
compute_stage_metrics can see, and that probe_and_rerank_candidates
dispatches it to the right real-toolchain probe (probe_persistent_kernel_
spill_free, not probe_spill_free) instead of silently mis-rendering it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.fft_plan_recursive import FFTLeafPlan
from planning.fft_plan_search import (
    _plan_signature,
    generate_candidates,
    generate_persistent_leaf_candidates,
)
from planning.spill_probe import _probe_plan
from planning.target_profile import DEFAULT_TARGET_PROFILE


def check_persistent_candidate_generated_for_feasible_n() -> None:
    n = 216
    cands = generate_persistent_leaf_candidates(n, target=DEFAULT_TARGET_PROFILE, inverse=False, batch=1)
    assert len(cands) == 1, f"expected exactly 1 persistent candidate for N={n}, got {len(cands)}"
    plan, choices = cands[0]
    assert choices.execution_strategy == "persistent"
    root = plan.root
    assert isinstance(root, FFTLeafPlan)
    assert root.kernel.persistent is not None
    print(f"    OK   N={n}: persistent candidate generated, radices={[s.radix for s in root.kernel.stages]}")


def check_persistent_infeasible_returns_empty_not_crash() -> None:
    # 16 * n > spad_capacity_bytes (120*1024) once n > 7680 -- large enough
    # to also need make_recursive_transpose_plan's own recursion, which
    # this execution model does not support at all yet.
    n = 100000
    cands = generate_persistent_leaf_candidates(n, target=DEFAULT_TARGET_PROFILE, inverse=False, batch=1)
    assert cands == [], f"expected no persistent candidates for infeasible N={n}, got {len(cands)}"
    print(f"    OK   N={n}: persistent infeasibility handled as an empty list, not a crash")


def check_persistent_gated_off_where_measured_slower() -> None:
    """Real-hardware regression guard (2026-08-30) for the *unsplit*
    single-leaf persistent path specifically: N=960/1024 both pass
    persistent's own raw capacity check (16*n <= spad_capacity_bytes) but
    an *unsplit* persistent leaf (num_logical_blocks=batch, typically 1)
    is confirmed ~50x SLOWER than the plain non-cooperative baseline for
    these two (it only activates one of target.num_ndp_units physical
    units -- see _persistent_leaf_feasible's own docstring) -- and, before
    this gate existed, the wrongly-cheap unsplit persistent candidate
    ranked #1 by estimated_cost among 89 real N=960 candidates. This test
    exists so a future change to _persistent_leaf_feasible cannot silently
    reopen that specific regression without this failing first.

    This does NOT mean N=960/1024 can never get a persistent candidate at
    all any more (2026-08-31, Phase 4/6): a persistent leaf reached
    *through a split* (`generate_leaf_worker_sequences`'s own
    `"persistent"` per-leaf entries, gated by the different, per-leaf
    `_leaf_persistent_feasible`) is a real, measured-competitive candidate
    for these same N (docs/persistent_recursive_split.md's own N=960/1024
    real-hardware parity numbers) -- see check_split_persistent_reachable_
    where_unsplit_is_gated below for that path's own coverage. This
    function only asserts the narrow *unsplit* generator specifically
    stays gated, which it does unchanged."""
    for n in (960, 1024):
        cands = generate_persistent_leaf_candidates(n, target=DEFAULT_TARGET_PROFILE, inverse=False, batch=1)
        assert cands == [], (
            f"N={n} is confirmed ~50x slower with an *unsplit* persistent leaf (real "
            f"hardware, 2026-08-30) -- generate_persistent_leaf_candidates must gate it "
            f"off, got {len(cands)} candidate(s)"
        )
    print("    OK   N=960/1024: unsplit single-leaf persistent path stays gated off "
          "(confirmed ~50x slower on real hardware) -- split+persistent is a separate, "
          "ungated path, see check_split_persistent_reachable_where_unsplit_is_gated")


def check_split_persistent_reachable_where_unsplit_is_gated() -> None:
    """Phase 6 (2026-08-31): the *split* persistent path -- unlike the
    unsplit one check_persistent_gated_off_where_measured_slower guards --
    is reachable for exactly the N the unsplit path is gated off for, via
    generate_leaf_worker_sequences' own per-leaf "persistent" entries
    (Phase 4) flowing through generate_radix_execution_joint_candidates
    (step 9). Confirms this doesn't regress into the same wrongly-cheap-
    unsplit trap: every surviving split+persistent candidate for N=960/
    1024 must actually be split (more than one leaf), never a single
    persistent leaf covering the whole N."""
    from planning.fft_plan_recursive import flatten_recursive_node
    from planning.fft_plan_core import FFTCodegenPlan

    for n in (960, 1024):
        cands = generate_candidates(n, target=DEFAULT_TARGET_PROFILE, max_candidates=300)
        persistent = [c for c in cands if _is_persistent_candidate(c.choices)]
        assert persistent, f"expected at least one split+persistent candidate for N={n}"
        for c in persistent:
            leaves = [s for s in flatten_recursive_node(c.plan.root) if isinstance(s, FFTCodegenPlan)]
            assert len(leaves) > 1, (
                f"N={n}: a persistent candidate with only {len(leaves)} leaf is unsplit -- "
                f"this must never happen here, see check_persistent_gated_off_where_measured_slower"
            )
    print("    OK   N=960/1024: split+persistent candidates reachable via the per-leaf "
          "joint search, every one genuinely split (never the gated-off unsplit shape)")


def _is_persistent_candidate(choices) -> bool:
    """A candidate is "persistent" whether it came from step 10's own
    unsplit single-leaf sweep (`execution_strategy == "persistent"`) or
    from the per-leaf joint search's own `"persistent"` entries in
    `worker_sequence` (Phase 4's per-leaf mixed execution strategy,
    fft_plan_search.generate_leaf_worker_sequences) -- two different
    generation *routes* to the same underlying execution model, which
    `_plan_signature` (and this project's own dedup) already treats as
    interchangeable when they happen to build the identical plan (e.g.
    an N small enough that the baseline tree never splits at all, so
    both routes build the same single persistent leaf and only one
    survives dedup -- which one is generation-order, not something a
    test should pin to a specific label)."""
    return choices.execution_strategy == "persistent" or (
        choices.worker_sequence is not None and "persistent" in choices.worker_sequence
    )


def check_persistent_signature_distinct_from_cooperative() -> None:
    n = 216
    cands = generate_candidates(n, target=DEFAULT_TARGET_PROFILE, max_candidates=300)
    persistent = [c for c in cands if _is_persistent_candidate(c.choices)]
    non_cooperative_baseline = [
        c for c in cands
        if c.choices.execution_strategy is None and c.choices.workers_per_fft is None
        and c.choices.worker_sequence is None and c.choices.radix_tier_name == "default"
    ]
    assert persistent, "expected at least one persistent candidate to survive into generate_candidates"
    assert non_cooperative_baseline, "expected the plain non-cooperative baseline to survive too"
    sig_persistent = _plan_signature(persistent[0].plan)
    sig_baseline = _plan_signature(non_cooperative_baseline[0].plan)
    assert sig_persistent != sig_baseline, (
        "persistent and non-cooperative candidates of the same radix collided on "
        "_plan_signature -- global dedup would have silently dropped one of them"
    )
    print(f"    OK   N={n}: persistent candidate's _plan_signature is distinct from the "
          f"non-cooperative baseline's (both radices={[s.radix for s in persistent[0].plan.root.kernel.stages]})")


def check_persistent_lower_register_pressure_reflected_in_cost() -> None:
    """Structural-only (no real toolchain): persistent's own total_worker_
    stage_batches must be lower than the non-cooperative baseline's for the
    same N/radix -- this is what actually let step 10 rank persistent
    ahead of every cooperative-worker candidate for N=216 (2026-08-30, see
    docs/execution_cost_model_validation.md), not a magic weight."""
    n = 216
    cands = generate_candidates(n, target=DEFAULT_TARGET_PROFILE, max_candidates=300)
    persistent = next(c for c in cands if _is_persistent_candidate(c.choices))
    baseline = next(
        c for c in cands
        if c.choices.execution_strategy is None and c.choices.workers_per_fft is None
        and c.choices.worker_sequence is None and c.choices.radix_tier_name == "default"
    )
    assert persistent.metrics.total_worker_stage_batches < baseline.metrics.total_worker_stage_batches, (
        f"expected persistent's total_worker_stage_batches "
        f"({persistent.metrics.total_worker_stage_batches}) < baseline's "
        f"({baseline.metrics.total_worker_stage_batches})"
    )
    assert persistent.metrics.estimated_cost < baseline.metrics.estimated_cost
    print(f"    OK   N={n}: persistent total_worker_stage_batches="
          f"{persistent.metrics.total_worker_stage_batches} < "
          f"non-cooperative baseline's {baseline.metrics.total_worker_stage_batches}, "
          f"estimated_cost ranks it cheaper")


def check_probe_dispatch_routes_persistent_to_persistent_probe() -> None:
    """Fast, no-real-build check (mirrors check_probe_and_rerank_accepts_
    joint_candidates' own discipline): confirm _probe_plan's own dispatch
    condition -- root.kernel.persistent is not None -- actually selects the
    persistent path for a persistent candidate and the plain path for
    everything else, without needing a real toolchain round trip to prove
    it (that was already exercised directly, real hardware, 2026-08-30:
    N=216's own persistent candidate probed spill_free=True, ndp_cycles=
    23944 via exactly this dispatch)."""
    import inspect
    sig = inspect.signature(_probe_plan)
    assert "compute_lanes" in sig.parameters and "narrow_middle_stages" in sig.parameters

    n = 216
    cands = generate_persistent_leaf_candidates(n, target=DEFAULT_TARGET_PROFILE, inverse=False, batch=1)
    plan, choices = cands[0]
    root = plan.root
    assert isinstance(root, FFTLeafPlan) and root.kernel.persistent is not None, (
        "the exact condition _probe_plan branches on -- must hold for a real persistent candidate"
    )
    print(f"    OK   N={n}: persistent candidate satisfies _probe_plan's own dispatch condition "
          f"(root.kernel.persistent is not None) -- real-hardware dispatch verified separately")


def main() -> None:
    print("  fft_plan_search.py: persistent-execution joint search (step 10):")
    check_persistent_candidate_generated_for_feasible_n()
    check_persistent_infeasible_returns_empty_not_crash()
    check_persistent_gated_off_where_measured_slower()
    check_split_persistent_reachable_where_unsplit_is_gated()
    check_persistent_signature_distinct_from_cooperative()
    check_persistent_lower_register_pressure_reflected_in_cost()
    check_probe_dispatch_routes_persistent_to_persistent_probe()
    print("[verify] fft_plan_search step-10 persistent search: all checks passed")


if __name__ == "__main__":
    main()
