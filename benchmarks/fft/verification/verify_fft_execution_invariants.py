from __future__ import annotations

"""Execution-strategy invariants, fixed by test -- Phase 2 of the
cooperative/persistent/non-cooperative unification work (see
docs/cooperative_worker8_pool_alignment_fix.md for the Phase 1 bug this
follows from).

Why this file exists, distinct from verify_fft_cooperative.py/
verify_fft_persistent.py: those modules check that a plan's *numeric*
output matches numpy's FFT (via the real emitted-code harness in
verify_fft_harness.py) -- necessary, but proven *not sufficient* by the
workers_per_fft=8 pool-alignment bug, which the Python numeric harness
passed cleanly on throughout (it never models real DRAM addresses, so it
cannot see a bug that lives entirely in what address a launch's own pool
started at). This module adds the *structural* checks the plan.md phase-2
spec asks for -- properties that must hold by construction, independent
of any one numeric run, plus a genuine same-input cross-strategy
equivalence check (non-cooperative vs. cooperative vs. persistent) that
did not exist anywhere in this suite before.
"""

import numpy as np

from planning.fft_plan_core import FFTCodegenPlan, _build_plan, layouts_for_radices, pingpong_needed
from planning.fft_plan_cooperative import make_cooperative_leaf_plan
from planning.fft_plan_persistent import make_persistent_leaf_plan
from planning.fft_plan_recursive import flatten_recursive_node, make_recursive_transpose_plan
from planning.target_profile import DEFAULT_TARGET_PROFILE
from verification.verify_fft_cooperative import run_cooperative_kernel
from verification.verify_fft_harness import Ptr, run_kernel
from verification.verify_fft_persistent import run_persistent_kernel
from verification.verify_fft_recursive import run_recursive_plan


def check_noncooperative_invariants(plan: FFTCodegenPlan) -> None:
    """A plain (non-cooperative, non-persistent) leaf: exactly one worker
    per FFT slot, no cooperative worker-partition state, no persistent
    round state. `plan.cooperation`/`plan.persistent` being `None` *is*
    "one worker, no shared state" here -- both execution models attach
    their own state to a leaf only when opted into (CooperationPlan/
    PersistentWorkgroupPlan's own docstrings), so their absence is the
    whole invariant, not something to derive further from `worker_batches`
    (which is `None` on every stage of a plain plan for the same reason).
    """
    assert plan.cooperation is None, "non-cooperative plan carries CooperationPlan state"
    assert plan.persistent is None, "non-cooperative plan carries PersistentWorkgroupPlan state"
    for stage in plan.stages:
        assert stage.worker_batches is None, (
            f"non-cooperative stage {stage.stage_id} carries worker_batches"
        )


def check_cooperative_invariants(plan: FFTCodegenPlan, *, workers_per_fft: int) -> None:
    """A cooperative leaf: exactly `workers_per_fft` workers, every stage's
    own batches partitioned across them with exactly-once ownership (no
    batch missing, none duplicated) -- `_partition_batches`'s own
    round-robin contract (fft_plan_cooperative.py), checked here as a
    property of the *plan*, not re-derived from a numeric run. Also
    confirms cooperative and persistent state are mutually exclusive on
    one plan (PersistentWorkgroupPlan's own docstring: "never combined
    with cooperation on the same plan").
    """
    assert plan.cooperation is not None, "cooperative plan missing CooperationPlan state"
    assert plan.persistent is None, "cooperative plan also carries PersistentWorkgroupPlan state"
    assert plan.cooperation.workers_per_fft == workers_per_fft, (
        f"plan.cooperation.workers_per_fft={plan.cooperation.workers_per_fft} != {workers_per_fft}"
    )
    for stage in plan.stages:
        assert stage.worker_batches is not None, (
            f"cooperative stage {stage.stage_id} missing worker_batches"
        )
        assert len(stage.worker_batches) == workers_per_fft, (
            f"stage {stage.stage_id}: {len(stage.worker_batches)} worker buckets, "
            f"expected {workers_per_fft}"
        )
        expected_ids = {b.batch_id for b in stage.batches}
        seen_ids: list[int] = []
        for worker_id, bucket in enumerate(stage.worker_batches):
            for b in bucket:
                seen_ids.append(b.batch_id)
        assert len(seen_ids) == len(set(seen_ids)), (
            f"stage {stage.stage_id}: a batch_id is owned by more than one worker "
            f"(duplicates in {sorted(seen_ids)})"
        )
        assert set(seen_ids) == expected_ids, (
            f"stage {stage.stage_id}: worker buckets cover {sorted(set(seen_ids))}, "
            f"expected exactly {sorted(expected_ids)} (missing "
            f"{sorted(expected_ids - set(seen_ids))}, extra {sorted(set(seen_ids) - expected_ids)})"
        )


def check_persistent_invariants(plan: FFTCodegenPlan) -> None:
    """A persistent leaf: no cooperative state (mutually exclusive, see
    above), workers_per_group pinned to this target's own hardware
    interleave chunk (PersistentWorkgroupPlan's own docstring -- the 8
    consecutive global_uthread_id()s one physical unit's own interleave
    chunk hands out), and the current implementation's own single-stripe
    restriction.
    """
    assert plan.persistent is not None, "persistent plan missing PersistentWorkgroupPlan state"
    assert plan.cooperation is None, "persistent plan also carries CooperationPlan state"
    assert plan.persistent.workers_per_group == DEFAULT_TARGET_PROFILE.interleave_chunk_uthreads, (
        f"workers_per_group={plan.persistent.workers_per_group} != "
        f"interleave_chunk_uthreads={DEFAULT_TARGET_PROFILE.interleave_chunk_uthreads}"
    )
    assert plan.persistent.stripes_per_group == 1, (
        f"stripes_per_group={plan.persistent.stripes_per_group} != 1 (only supported value)"
    )


def verify_cross_strategy_equivalence(
    length: int, radices: tuple[int, ...], *,
    cooperative_workers: int, total_ffts: int, inverse: bool = False, seed: int = 0,
) -> dict[str, float]:
    """The same random input, through non-cooperative, cooperative, and
    persistent leaves built from the *same* (length, radices, inverse) --
    every strategy must agree with numpy's FFT *and* with each other.
    Three strategies each independently matching numpy is necessary but
    not sufficient for "these are the same computation" (they could share
    a bug that still happens to look right against numpy on the cases
    tried) -- comparing them directly against each other, on one shared
    input, is the stronger claim the plan's own "cross-strategy
    equivalence" phase asks for, and nothing in this suite checked it
    before this file.

    Returns `{"noncoop_vs_numpy": ..., "coop_vs_numpy": ...,
    "persistent_vs_numpy": ..., "noncoop_vs_coop": ...,
    "noncoop_vs_persistent": ..., "coop_vs_persistent": ...}` (max abs
    error per pair) -- caller decides the tolerance.
    """
    rng = np.random.default_rng(seed)
    total = length * total_ffts
    in_r = rng.uniform(-1, 1, total)
    in_i = rng.uniform(-1, 1, total)
    x = (in_r + 1j * in_i).reshape(total_ffts, length)
    ref = np.fft.ifft(x, axis=1) if inverse else np.fft.fft(x, axis=1)
    ref_flat = ref.reshape(-1)

    def _fresh_input() -> tuple[Ptr, Ptr]:
        p_r, p_i = Ptr(total), Ptr(total)
        p_r.arr[:] = in_r
        p_i.arr[:] = in_i
        return p_r, p_i

    # non-cooperative
    noncoop_plan = _build_plan(
        length=length, inverse=inverse, total_uthreads=total_ffts, simd_lanes=8,
        use_pingpong=pingpong_needed(len(radices)),
        layouts=layouts_for_radices(length, radices, 8),
        kernel_name="XEquivNoncoop",
    )
    check_noncooperative_invariants(noncoop_plan)
    nc_in_r, nc_in_i = _fresh_input()
    nc_out_r, nc_out_i = Ptr(total), Ptr(total)
    run_kernel(
        noncoop_plan, input_real=nc_in_r, input_imag=nc_in_i,
        output_real=nc_out_r, output_imag=nc_out_i,
    )
    noncoop_out = nc_out_r.arr + 1j * nc_out_i.arr

    # cooperative
    coop_plan = make_cooperative_leaf_plan(
        length=length, radices=radices, workers_per_fft=cooperative_workers,
        total_ffts=total_ffts, inverse=inverse, kernel_name="XEquivCoop",
    )
    check_cooperative_invariants(coop_plan, workers_per_fft=cooperative_workers)
    c_in_r, c_in_i = _fresh_input()
    c_out_r, c_out_i = Ptr(total), Ptr(total)
    run_cooperative_kernel(
        coop_plan, input_real=c_in_r, input_imag=c_in_i,
        output_real=c_out_r, output_imag=c_out_i,
    )
    coop_out = c_out_r.arr + 1j * c_out_i.arr

    # persistent
    persistent_plan = make_persistent_leaf_plan(
        length, radices, num_logical_blocks=total_ffts, inverse=inverse,
    )
    check_persistent_invariants(persistent_plan)
    p_in_r, p_in_i = _fresh_input()
    p_out_r, p_out_i = Ptr(total), Ptr(total)
    run_persistent_kernel(
        persistent_plan, num_logical_blocks=total_ffts,
        input_real=p_in_r, input_imag=p_in_i,
        output_real=p_out_r, output_imag=p_out_i,
    )
    persistent_out = p_out_r.arr + 1j * p_out_i.arr

    def _err(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b)))

    return {
        "noncoop_vs_numpy": _err(noncoop_out, ref_flat),
        "coop_vs_numpy": _err(coop_out, ref_flat),
        "persistent_vs_numpy": _err(persistent_out, ref_flat),
        "noncoop_vs_coop": _err(noncoop_out, coop_out),
        "noncoop_vs_persistent": _err(noncoop_out, persistent_out),
        "coop_vs_persistent": _err(coop_out, persistent_out),
    }


def verify_persistent_recursive_split(
    n: int, *, scratchpad_byte_budget: int, inverse: bool = False, seed: int = 0,
) -> dict[str, object]:
    """Phase 3 (persistent recursive-split support): a split plan built
    with `persistent_leaf=True` must use *exactly* the same split/radix
    decomposition as the same call without it (persistent is a leaf-
    lowering choice, never a data-layout algorithm -- see
    docs/cooperative_worker8_pool_alignment_fix.md's own phase list), and
    must match numpy's FFT numerically. Returns `{"same_split_shape":
    bool, "max_error": float}`; caller decides the tolerance.

    Doesn't touch real hardware (see that module's own docstring on why a
    Python numeric pass is necessary but not sufficient for the class of
    bug Phase 1 found) -- real-hardware confirmation for this exact check
    (N=960/1024, forward+inverse) is in docs/
    cooperative_worker8_pool_alignment_fix.md's own Phase 3 section,
    reproducible via run_fft_test.sh/make_fft_kernel.py's own
    `persistent_leaf`-equivalent path.
    """
    plan_persistent = make_recursive_transpose_plan(
        n, scratchpad_byte_budget=scratchpad_byte_budget, inverse=inverse, persistent_leaf=True,
    )
    plan_plain = make_recursive_transpose_plan(
        n, scratchpad_byte_budget=scratchpad_byte_budget, inverse=inverse,
    )

    leaves_persistent = [
        s for s in flatten_recursive_node(plan_persistent.root) if isinstance(s, FFTCodegenPlan)
    ]
    leaves_plain = [
        s for s in flatten_recursive_node(plan_plain.root) if isinstance(s, FFTCodegenPlan)
    ]
    same_split_shape = [(leaf.length, len(leaf.stages)) for leaf in leaves_persistent] == [
        (leaf.length, len(leaf.stages)) for leaf in leaves_plain
    ]
    for leaf in leaves_persistent:
        check_persistent_invariants(leaf)
    for leaf in leaves_plain:
        check_noncooperative_invariants(leaf)

    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)
    got = run_recursive_plan(plan_persistent, x, compute_lanes=4, narrow_middle_stages=True)
    ref = np.fft.ifft(x) if inverse else np.fft.fft(x)
    return {"same_split_shape": same_split_shape, "max_error": float(np.max(np.abs(got - ref)))}


def main() -> None:
    tolerance = 1e-6
    failures: list[str] = []

    print("  Structural invariants: non-cooperative/cooperative/persistent plan shape:")
    struct_cases: list[tuple[int, tuple[int, ...], int]] = [
        (64, (4, 4, 4), 2),
        (64, (4, 4, 4), 4),
        (105, (3, 5, 7), 2),
        (128, (4, 4, 4, 2), 8),
    ]
    for length, radices, workers in struct_cases:
        tag = f"length={length} radices={radices}"
        try:
            noncoop = _build_plan(
                length=length, inverse=False, total_uthreads=1, simd_lanes=8,
                use_pingpong=pingpong_needed(len(radices)),
                layouts=layouts_for_radices(length, radices, 8),
                kernel_name="StructNoncoop",
            )
            check_noncooperative_invariants(noncoop)

            coop = make_cooperative_leaf_plan(
                length=length, radices=radices, workers_per_fft=workers,
                total_ffts=1, kernel_name="StructCoop",
            )
            check_cooperative_invariants(coop, workers_per_fft=workers)
            print(f"    OK   {tag} workers={workers}: non-cooperative + cooperative invariants hold")
        except AssertionError as exc:
            print(f"    FAIL {tag} workers={workers}: {exc}")
            failures.append(f"{tag} workers={workers}")

    persistent_struct_cases: list[tuple[int, tuple[int, ...]]] = [
        (64, (4, 4, 4)),
        (105, (3, 5, 7)),
    ]
    for length, radices in persistent_struct_cases:
        tag = f"persistent length={length} radices={radices}"
        try:
            plan = make_persistent_leaf_plan(length, radices, num_logical_blocks=1)
            check_persistent_invariants(plan)
            print(f"    OK   {tag}: persistent invariants hold")
        except AssertionError as exc:
            print(f"    FAIL {tag}: {exc}")
            failures.append(tag)

    print()
    print("  Regression guard: workers_per_fft=8 offered by default (Phase 1 fix):")
    from planning.fft_plan_cooperative import choose_workers_per_fft, worker_candidates_per_fft

    candidates = worker_candidates_per_fft(128, (4, 4, 4, 2))
    tag = "worker_candidates_per_fft(128, (4,4,4,2)) includes 8 by default"
    if 8 in candidates:
        print(f"    OK   {tag}: {candidates}")
    else:
        print(f"    FAIL {tag}: {candidates}")
        failures.append(tag)

    chosen = choose_workers_per_fft(128, (4, 4, 4, 2))
    tag = "choose_workers_per_fft(128, (4,4,4,2)) picks 8 by default"
    if chosen == 8:
        print(f"    OK   {tag}: {chosen}")
    else:
        print(f"    FAIL {tag}: {chosen}")
        failures.append(tag)

    print()
    print("  Regression guard: cooperative-leaf host main() emits the 256B pool-alignment guard:")
    from codegen.fft_cooperative_codegen import generate_cooperative_fft_kernel

    guard_plan = make_cooperative_leaf_plan(
        length=128, radices=(4, 4, 4, 2), workers_per_fft=8, total_ffts=1,
        kernel_name="GuardCheck",
    )
    text = generate_cooperative_fft_kernel(guard_plan, compute_lanes=2)
    tag = "generate_cooperative_fft_kernel emits pool alignment guard"
    if "pool_addr % 256" in text and "pool_raw" in text:
        print(f"    OK   {tag}")
    else:
        print(f"    FAIL {tag}: alignment guard text not found in emitted kernel")
        failures.append(tag)

    print()
    print("  Cross-strategy numeric equivalence (same input, non-coop vs. cooperative vs. persistent):")
    equiv_cases: list[tuple[int, tuple[int, ...], int, int]] = [
        (64, (4, 4, 4), 2, 3),
        (105, (3, 5, 7), 2, 4),
    ]
    for length, radices, workers, total_ffts in equiv_cases:
        for inverse in (False, True):
            tag = (
                f"length={length} radices={radices} cooperative_workers={workers} "
                f"total_ffts={total_ffts} inverse={inverse}"
            )
            errs = verify_cross_strategy_equivalence(
                length, radices, cooperative_workers=workers, total_ffts=total_ffts,
                inverse=inverse, seed=71,
            )
            worst = max(errs.values())
            ok = worst <= tolerance
            print(f"    {'OK  ' if ok else 'FAIL'} {tag}: worst pairwise/numpy error {worst:.3e}")
            if not ok:
                print(f"         detail: {errs}")
                failures.append(tag)

    print()
    print("  Persistent recursive-split support (Phase 3): same split/radix as non-persistent, numpy-correct:")
    split_cases: list[tuple[int, int]] = [(960, 32 * 16), (1024, 32 * 16)]
    for n, budget in split_cases:
        for inverse in (False, True):
            tag = f"N={n} budget={budget} inverse={inverse}"
            result = verify_persistent_recursive_split(
                n, scratchpad_byte_budget=budget, inverse=inverse, seed=17,
            )
            ok = result["same_split_shape"] and result["max_error"] <= tolerance
            print(f"    {'OK  ' if ok else 'FAIL'} {tag}: {result}")
            if not ok:
                failures.append(tag)

    if failures:
        raise AssertionError(f"{len(failures)} execution-strategy invariant check(s) failed: {failures}")
    print("[verify] execution-strategy invariants: all checks passed")


if __name__ == "__main__":
    main()
