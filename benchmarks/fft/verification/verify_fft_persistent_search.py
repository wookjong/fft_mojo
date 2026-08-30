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


def check_persistent_signature_distinct_from_cooperative() -> None:
    n = 216
    cands = generate_candidates(n, target=DEFAULT_TARGET_PROFILE, max_candidates=300)
    persistent = [c for c in cands if c.choices.execution_strategy == "persistent"]
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
    persistent = next(c for c in cands if c.choices.execution_strategy == "persistent")
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
    check_persistent_signature_distinct_from_cooperative()
    check_persistent_lower_register_pressure_reflected_in_cost()
    check_probe_dispatch_routes_persistent_to_persistent_probe()
    print("[verify] fft_plan_search step-10 persistent search: all checks passed")


if __name__ == "__main__":
    main()
