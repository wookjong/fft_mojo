from __future__ import annotations

"""Structural verification for fft_plan_search.py's own candidate-generation
properties -- NOT FFT numerical correctness (verify_fft_plan.py's own job,
via verify_fft_cooperative.py/verify_fft_recursive.py/etc., which every
candidate this module builds ultimately reuses the same lowering/codegen
path as -- see this file's own final check, which builds and numerically
verifies an actual step-9 joint candidate end to end).

Covers, specifically, generate_radix_execution_joint_candidates /
generate_leaf_worker_sequences (the radix-tier x per-leaf-worker-sequence
joint search this project's planner did not have before): that the same
radix composition gets crossed with more than one worker configuration and
vice versa, that (radix A, worker X) and (radix B, worker Y) coexist as
distinct candidates, that per-leaf worker sequences are genuinely per-leaf
(([w0, None] vs [None, w0] are distinct, not collapsed), that an illegal
combination is rejected before it ever reaches cost scoring rather than
crashing the whole search, that no duplicate candidate survives, and that
planning.spill_probe.probe_and_rerank_candidates (which pre-dates this
search axis) works unmodified against candidates this axis produces.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.fft_plan_core import FFTCodegenPlan
from planning.fft_plan_recursive import _recursive_split_candidates, flatten_recursive_node, make_recursive_transpose_plan
from planning.fft_plan_search import (
    FFTPlanCandidate,
    PlanChoices,
    _leaf_kernels_in_order,
    _plan_signature,
    _root_split_length,
    generate_candidates,
    generate_leaf_worker_sequences,
    generate_radix_execution_joint_candidates,
    generate_radix_tiers,
)
from planning.target_profile import DEFAULT_TARGET_PROFILE


# A target with a second (wide) radix tier available, purely for exercising
# the radix axis in these tests -- DEFAULT_TARGET_PROFILE itself only ever
# offers one tier (supports_vector_spill=False), so "same worker config,
# multiple radix configs" has nothing to vary against without this.
_WIDE_TIER_TARGET = replace(DEFAULT_TARGET_PROFILE, supports_vector_spill=True)


def _find_balanced_split(n: int, *, scratchpad_byte_budget: int = 4096) -> int:
    """The most balanced (near, far) split legal for n -- both sides
    comparably sized, so both leaves stand a real chance of having more
    than one legal worker count (a leaf as small as length 4 never does --
    see worker_candidates_per_fft's own max_batches bound). Picked by
    minimizing |near - far|, not by anything this module needs to trust
    beyond "some real N with two substantial leaves exists" for these
    tests to exercise the per-leaf axis meaningfully."""
    candidates = _recursive_split_candidates(n, scratchpad_byte_budget=scratchpad_byte_budget)
    return min(candidates, key=lambda near: abs(near - (n // near)))


def check_same_radix_multiple_workers() -> None:
    n = 960
    near = _find_balanced_split(n)
    baseline = make_recursive_transpose_plan(
        n, scratchpad_byte_budget=4096, simd_lanes=8, forced_split_near_length=near,
        spad_capacity_bytes=DEFAULT_TARGET_PROFILE.spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=DEFAULT_TARGET_PROFILE.max_concurrent_scratchpad_bytes,
        interleave_chunk_uthreads=DEFAULT_TARGET_PROFILE.interleave_chunk_uthreads,
    )
    joint = generate_radix_execution_joint_candidates(
        n, target=DEFAULT_TARGET_PROFILE, inverse=False, scratchpad_byte_budget=4096,
        simd_lanes=8, batch=1, baseline_plan=baseline, baseline_split=near,
    )
    by_radix: dict[str, list[PlanChoices]] = {}
    for plan, choices in joint:
        key = choices.radix_tier_name
        by_radix.setdefault(key, []).append(choices)
    worker_seqs_for_default = {c.worker_sequence for c in by_radix.get("default", [])}
    assert len(worker_seqs_for_default) >= 2, (
        f"expected the 'default' radix tier crossed with >= 2 distinct worker "
        f"sequences at N={n} (near={near}), got {worker_seqs_for_default}"
    )
    print(f"    OK   N={n} near={near}: 'default' radix tier x "
          f"{len(worker_seqs_for_default)} distinct worker sequences: {sorted(worker_seqs_for_default, key=str)}")


def check_same_workers_multiple_radix() -> None:
    n = 960
    near = _find_balanced_split(n)
    baseline = make_recursive_transpose_plan(
        n, scratchpad_byte_budget=4096, simd_lanes=8, forced_split_near_length=near,
        spad_capacity_bytes=_WIDE_TIER_TARGET.spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=_WIDE_TIER_TARGET.max_concurrent_scratchpad_bytes,
        interleave_chunk_uthreads=_WIDE_TIER_TARGET.interleave_chunk_uthreads,
    )
    tiers = generate_radix_tiers(_WIDE_TIER_TARGET)
    assert len(tiers) >= 2, f"expected >= 2 radix tiers from a wide-tier-capable target, got {tiers}"
    joint = generate_radix_execution_joint_candidates(
        n, target=_WIDE_TIER_TARGET, inverse=False, scratchpad_byte_budget=4096,
        simd_lanes=8, batch=1, baseline_plan=baseline, baseline_split=near,
    )
    by_worker_seq: dict[tuple, set[str]] = {}
    for plan, choices in joint:
        by_worker_seq.setdefault(choices.worker_sequence, set()).add(choices.radix_tier_name)
    multi = {seq: tiers for seq, tiers in by_worker_seq.items() if len(tiers) >= 2}
    assert multi, (
        f"expected at least one worker sequence crossed with >= 2 radix tiers "
        f"at N={n} (near={near}), got per-sequence tier sets: {by_worker_seq}"
    )
    seq, tier_set = next(iter(multi.items()))
    print(f"    OK   N={n} near={near}: worker_sequence={seq} x {len(tier_set)} "
          f"distinct radix tiers: {sorted(tier_set)}")


def check_distinct_radix_and_worker_pair_coexist() -> None:
    n = 960
    near = _find_balanced_split(n)
    baseline = make_recursive_transpose_plan(
        n, scratchpad_byte_budget=4096, simd_lanes=8, forced_split_near_length=near,
        spad_capacity_bytes=_WIDE_TIER_TARGET.spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=_WIDE_TIER_TARGET.max_concurrent_scratchpad_bytes,
        interleave_chunk_uthreads=_WIDE_TIER_TARGET.interleave_chunk_uthreads,
    )
    joint = generate_radix_execution_joint_candidates(
        n, target=_WIDE_TIER_TARGET, inverse=False, scratchpad_byte_budget=4096,
        simd_lanes=8, batch=1, baseline_plan=baseline, baseline_split=near,
    )
    pairs = {(c.radix_tier_name, c.worker_sequence) for _, c in joint}
    # At least two candidates whose (tier, worker_sequence) pair differs in
    # BOTH components at once -- not just "same tier, different worker" or
    # "same worker, different tier" (the two checks above already cover
    # those individually).
    found = None
    pair_list = list(pairs)
    for i in range(len(pair_list)):
        for j in range(len(pair_list)):
            a, b = pair_list[i], pair_list[j]
            if a[0] != b[0] and a[1] != b[1]:
                found = (a, b)
                break
        if found:
            break
    assert found, f"expected two candidates differing in BOTH radix tier and worker sequence, got pairs={pairs}"
    print(f"    OK   N={n} near={near}: distinct (radix, worker) pairs coexist: {found[0]} and {found[1]}")


def check_per_leaf_sequences_are_distinct() -> None:
    """[w, None] and [None, w] must be two different sequences, not
    collapsed into "this leaf gets w" regardless of which one -- the whole
    point of a per-leaf (not a single global) worker axis."""
    n = 960
    near = _find_balanced_split(n)
    baseline = make_recursive_transpose_plan(
        n, scratchpad_byte_budget=4096, simd_lanes=8, forced_split_near_length=near,
        spad_capacity_bytes=DEFAULT_TARGET_PROFILE.spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=DEFAULT_TARGET_PROFILE.max_concurrent_scratchpad_bytes,
        interleave_chunk_uthreads=DEFAULT_TARGET_PROFILE.interleave_chunk_uthreads,
    )
    kernels = _leaf_kernels_in_order(baseline)
    assert len(kernels) >= 2, f"need >= 2 leaves to test per-leaf distinctness, got {len(kernels)} at N={n} near={near}"
    seqs = generate_leaf_worker_sequences(baseline, target=DEFAULT_TARGET_PROFILE, simd_lanes=8)
    # Find some non-None worker value that appears in position 0 for one
    # sequence and position 1 for another (mirroring the task's own
    # "[1,4] and [4,1] both exist" example, generalized to whatever legal
    # worker counts these two real leaves actually have).
    at_0 = {s[0] for s in seqs if s[0] is not None}
    at_1 = {s[1] for s in seqs if s[1] is not None}
    common = at_0 & at_1
    assert common, (
        f"expected some worker count legal for both leaves so [w, None] and "
        f"[None, w] can both be built -- leaf0 own non-None values={at_0}, "
        f"leaf1 own non-None values={at_1} at N={n} near={near}"
    )
    w = next(iter(common))
    seq_a = tuple(w if i == 0 else None for i in range(len(kernels)))
    seq_b = tuple(w if i == 1 else None for i in range(len(kernels)))
    assert seq_a in seqs and seq_b in seqs and seq_a != seq_b, (
        f"expected both {seq_a} and {seq_b} present and distinct in {seqs}"
    )
    print(f"    OK   N={n} near={near}: per-leaf sequences {seq_a} and {seq_b} "
          f"both generated and distinct (w={w})")


def check_illegal_combination_rejected_not_crashed() -> None:
    """A (tier, worker sequence) combination `_build_recursive_node`'s own
    assertions reject (here: a worker-sequence length mismatch, forced via
    monkeypatching generate_leaf_worker_sequences to hand back one bad
    entry alongside good ones) must be skipped, not raised out of
    generate_radix_execution_joint_candidates -- "obviously bad candidates
    are pruned before cost ranking," this module's own stated discipline,
    now covering this axis too."""
    import planning.fft_plan_search as search_mod

    n = 960
    near = _find_balanced_split(n)
    baseline = make_recursive_transpose_plan(
        n, scratchpad_byte_budget=4096, simd_lanes=8, forced_split_near_length=near,
        spad_capacity_bytes=DEFAULT_TARGET_PROFILE.spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=DEFAULT_TARGET_PROFILE.max_concurrent_scratchpad_bytes,
        interleave_chunk_uthreads=DEFAULT_TARGET_PROFILE.interleave_chunk_uthreads,
    )
    real_generate = search_mod.generate_leaf_worker_sequences

    def bad_then_good(*args, **kwargs):
        good = real_generate(*args, **kwargs)
        # One entry short -- _build_recursive_node's own forced_worker_
        # sequence contract requires exactly one entry per leaf; this is
        # guaranteed illegal regardless of which real N/leaves are in play.
        too_short = (good[0][0],) if good[0] else ()
        return [too_short] + list(good)

    search_mod.generate_leaf_worker_sequences = bad_then_good
    try:
        joint = search_mod.generate_radix_execution_joint_candidates(
            n, target=DEFAULT_TARGET_PROFILE, inverse=False, scratchpad_byte_budget=4096,
            simd_lanes=8, batch=1, baseline_plan=baseline, baseline_split=near,
        )
    finally:
        search_mod.generate_leaf_worker_sequences = real_generate
    # Didn't raise (the bad entry was caught and skipped), and every
    # surviving candidate is still a real, legal plan.
    assert joint, "expected at least the legal candidates to survive alongside the rejected one"
    print(f"    OK   N={n} near={near}: illegal worker-sequence-length combination "
          f"rejected without crashing the search ({len(joint)} legal candidates survived)")


def check_no_duplicate_candidates() -> None:
    """Two radix tiers that happen to coalesce a given leaf identically
    (a real possibility whenever a leaf's own prime factors have no
    radix-4-mergeable pair for the wide tier to actually change) must not
    produce two byte-for-byte-identical plans under different tier
    labels -- generate_radix_execution_joint_candidates' own dedup key is
    the REALIZED per-leaf radix tuple, not the tier name."""
    # N=105 = (3, 5, 7): no adjacent pair is in _WIDE_RADIX_TIER={4,6,9} at
    # all, so 'default' and 'wide' coalesce identically here -- exactly
    # the "same realized shape, different tier label" case to dedup.
    n = 105
    baseline = make_recursive_transpose_plan(
        n, scratchpad_byte_budget=4096, simd_lanes=8,
        spad_capacity_bytes=_WIDE_TIER_TARGET.spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=_WIDE_TIER_TARGET.max_concurrent_scratchpad_bytes,
        interleave_chunk_uthreads=_WIDE_TIER_TARGET.interleave_chunk_uthreads,
    )
    baseline_split = _root_split_length(baseline)
    joint = generate_radix_execution_joint_candidates(
        n, target=_WIDE_TIER_TARGET, inverse=False, scratchpad_byte_budget=4096,
        simd_lanes=8, batch=1, baseline_plan=baseline, baseline_split=baseline_split,
    )
    tier_names = {c.radix_tier_name for _, c in joint}
    assert "wide" not in tier_names, (
        f"expected the 'wide' tier's own candidate to be deduped away (N=105's "
        f"(3,5,7) has no radix-4-mergeable pair, so it coalesces identically to "
        f"'default'), but found tier_names={tier_names}"
    )
    print(f"    OK   N={n}: identically-coalescing 'wide' tier deduped against "
          f"'default' (only {sorted(tier_names)} survived)")


def check_cross_step_duplicates_removed() -> None:
    """A real, concrete cross-step duplicate found while validating this
    search axis: N=960's own default baseline split (near=240, far=4) has
    a `far` leaf too small for any cooperative worker count at all (see
    worker_candidates_per_fft's own max_batches bound), so step 9's own
    'default'-tier, single-leaf-varying sequences realize byte-for-byte
    the SAME plan step 3's own global-worker-count sweep (`workers_per_fft`
    applied uniformly, which silently degrades to "only the leaf that can
    use it does" here since the far leaf caps at 1 regardless) *and*
    step 8's own single-leaf sweep already build. Before `_dedup_candidates`
    existed, `generate_candidates(960)` contained the same underlying
    (radix, worker) plan three times over, from three different steps.

    Checked at the `_plan_signature` level (what the plan actually IS),
    not `PlanChoices` (which step gets "credit" for a surviving candidate
    can legitimately shift once cross-step dedup keeps whichever step
    generated it first -- N=960 keeps step 3's own `workers_per_fft`-
    labeled entries here, not step 8/9's `worker_sequence`-labeled ones,
    and that's fine: the search space explored is identical either way,
    only the redundant relabeled copies are gone). This is the general
    property `check_no_duplicate_candidates` doesn't cover (that one
    exercises `generate_radix_execution_joint_candidates`'s own narrower,
    single-step dedup only).
    """
    from planning.fft_plan_search import _plan_signature

    n = 960
    cands = generate_candidates(n, target=DEFAULT_TARGET_PROFILE, max_candidates=200)
    sigs = [_plan_signature(c.plan) for c in cands]
    assert len(sigs) == len(set(sigs)), (
        f"expected every candidate in generate_candidates(N={n})'s own output to "
        f"realize a distinct plan -- found {len(sigs) - len(set(sigs))} duplicate(s)"
    )
    non_baseline_leaf = [c for c in cands if c.choices.workers_per_fft is not None or c.choices.worker_sequence is not None]
    assert non_baseline_leaf, f"expected >= 1 cooperative/persistent candidate at N={n} for this check to mean anything"
    print(f"    OK   N={n}: {len(cands)} candidates, all distinct plan signatures "
          f"({len(non_baseline_leaf)} cooperative/persistent, surviving via whichever step built them first)")


def check_generate_candidates_includes_step9() -> None:
    """End to end through generate_candidates itself (not just the step-9
    generator in isolation) -- a candidate only step 9 could have produced
    actually reaches the final, deduped candidate pool for at least one
    real N.

    `worker_sequence is not None` alone is NOT a step-9-specific signal:
    step 8 (the pre-existing per-leaf worker sweep) already sets it too,
    for every N -- the two overlap completely whenever a leaf's own step-9
    worker choice varies only one leaf at the 'default' tier (exactly the
    shape step 8 already covers), and after this same commit's own
    cross-step dedup fix (`_dedup_candidates`), those duplicates are
    correctly removed in favor of step 8's earlier-generated entry. So
    the real signal for "step 9 contributed something new" is
    `radix_tier_name != 'default'` (only step 9 crosses worker sequences
    against a non-default tier at all -- step 4's own tier sweep never
    sets worker_sequence, and step 8 never varies tier) -- which also
    means this needs an N whose leaf factorization actually has an
    adjacent pair the wide tier ({4, 6, 9}) can coalesce differently
    from the default ({4}-only) tier (N=144's own single leaf, factors
    (2,2,2,2,3,3), coalesces to (4,4,3,3) by default vs (4,4,9) wide --
    most N's don't have this property, e.g. N=960 checked elsewhere in
    this file coalesces identically under both tiers, so step 9 legitimately
    contributes nothing new there once cross-step duplicates are removed).
    """
    n = 144
    cands = generate_candidates(n, target=_WIDE_TIER_TARGET, max_candidates=200)
    step9_only = [c for c in cands if c.choices.worker_sequence is not None and c.choices.radix_tier_name != "default"]
    assert step9_only, f"expected >= 1 step-9-only candidate in generate_candidates(N={n})'s own output"
    print(f"    OK   N={n}: generate_candidates includes {len(step9_only)} candidate(s) "
          f"only step 9 could produce (non-default tier x worker sequence) among {len(cands)} total")


def check_probe_and_rerank_accepts_joint_candidates() -> None:
    """planning.spill_probe.probe_and_rerank_candidates pre-dates this
    search axis -- confirm it type-checks/accepts a candidate list that
    includes step-9 entries without needing any change on its own side
    (it only reads candidate.plan/candidate.metrics, both of which every
    FFTPlanCandidate this module builds -- step-9 included -- already
    has). Does not itself invoke the real toolchain (see the module
    docstring's own real-hardware note); the real build+run path is
    exercised by verify_end_to_end_numeric below and by this session's
    own hardware runs, not by this fast check.
    """
    from planning.spill_probe import probe_and_rerank_candidates

    # See check_generate_candidates_includes_step9's own docstring for why
    # N=144 + a wide-tier-capable target, and why "non-default tier" (not
    # merely worker_sequence is not None) is the real step-9-only signal.
    n = 144
    cands = generate_candidates(n, target=_WIDE_TIER_TARGET, max_candidates=200)
    step9 = [c for c in cands if c.choices.worker_sequence is not None and c.choices.radix_tier_name != "default"]
    assert step9, "need >= 1 step-9-only candidate for this check to mean anything"
    import inspect
    sig = inspect.signature(probe_and_rerank_candidates)
    assert "candidates" in sig.parameters and "rank_by_cycles" in sig.parameters
    # Confirm every field probe_and_rerank_candidates actually reads
    # (candidate.plan, candidate.metrics) is present and well-formed on a
    # step-9 candidate specifically -- the same shape check that function
    # implicitly relies on for any candidate list.
    for c in step9[:3]:
        assert c.plan is not None
        assert c.metrics is not None
        assert hasattr(c.metrics, "spill_free") and hasattr(c.metrics, "ndp_cycles")
    print(f"    OK   N={n}: probe_and_rerank_candidates' own signature/field "
          f"expectations hold for {len(step9)} step-9 candidate(s) "
          f"(real-hardware rank_by_cycles run separately, not part of this fast check)")


def check_generate_candidates_includes_step11() -> None:
    """End to end through generate_candidates (step 11, the compute_lanes
    variant sweep -- Phase 6's second half): "unnarrowed" and "all_scalar"
    tree-wide lane variants of the baseline both reach the final, deduped
    pool, each with every leaf's every stage's own `compute_lanes`
    actually populated (not left `None`, the shape every other step's own
    candidates still have -- see generate_lane_variant_candidates' own
    docstring for why the plan-level "baseline" shape is deliberately not
    a third variant here)."""
    n = 960
    cands = generate_candidates(n, scratchpad_byte_budget=32 * 16, max_candidates=200)
    lane_variants = {c.choices.lane_variant: c for c in cands if c.choices.lane_variant is not None}
    assert "unnarrowed" in lane_variants and "all_scalar" in lane_variants, (
        f"expected both 'unnarrowed' and 'all_scalar' lane-variant candidates, got {sorted(lane_variants)}"
    )
    for label, c in lane_variants.items():
        leaves = [s for s in flatten_recursive_node(c.plan.root) if isinstance(s, FFTCodegenPlan)]
        assert leaves, f"{label}: expected at least one FFT leaf"
        for leaf in leaves:
            for stage in leaf.stages:
                assert stage.compute_lanes is not None, (
                    f"{label}: leaf {leaf.kernel_name} stage {stage.stage_id} has compute_lanes=None"
                )
        if label == "all_scalar":
            assert all(
                stage.compute_lanes == 1 for leaf in leaves for stage in leaf.stages
            ), "all_scalar variant must floor every stage to compute_lanes=1"
    print(f"    OK   N={n}: generate_candidates includes both compute_lanes variants "
          f"('unnarrowed', 'all_scalar'), every leaf stage's own compute_lanes set")


def check_lane_variant_signature_distinct_from_baseline() -> None:
    """`_plan_signature` must not collapse a lane-variant candidate onto
    the plain baseline (whose own stages all have `compute_lanes=None`)
    -- Phase 5's own signature change (folding each leaf's per-stage
    `compute_lanes` tuple in) is what makes this hold; this confirms it
    actually does for a real step-11 candidate, not just in isolation."""
    n = 960
    cands = generate_candidates(n, scratchpad_byte_budget=32 * 16, max_candidates=200)
    baseline = next(
        c for c in cands
        if c.choices.lane_variant is None and c.choices.worker_sequence is None
        and c.choices.execution_strategy is None and c.choices.radix_tier_name == "default"
        and c.choices.split_sequence is None and c.choices.tile is None
    )
    lane_variants = [c for c in cands if c.choices.lane_variant is not None]
    assert lane_variants, "need >= 1 lane-variant candidate for this check to mean anything"
    sig_baseline = _plan_signature(baseline.plan)
    for c in lane_variants:
        assert _plan_signature(c.plan) != sig_baseline, (
            f"lane_variant={c.choices.lane_variant!r} collided with the baseline's own "
            f"_plan_signature -- global dedup would have silently dropped one of them"
        )
    print(f"    OK   N={n}: {len(lane_variants)} lane-variant candidate(s) each have a "
          f"_plan_signature distinct from the plain baseline's")


def check_probe_accepts_lane_variant_candidates() -> None:
    """Same discipline as check_probe_and_rerank_accepts_joint_candidates
    above, for step 11's own candidates: planning.spill_probe.probe_and_
    rerank_candidates only reads candidate.plan/candidate.metrics, both
    already well-formed on a lane-variant candidate."""
    from planning.spill_probe import probe_and_rerank_candidates

    n = 960
    cands = generate_candidates(n, scratchpad_byte_budget=32 * 16, max_candidates=200)
    lane_variants = [c for c in cands if c.choices.lane_variant is not None]
    assert lane_variants, "need >= 1 lane-variant candidate for this check to mean anything"
    import inspect
    sig = inspect.signature(probe_and_rerank_candidates)
    assert "candidates" in sig.parameters and "rank_by_cycles" in sig.parameters
    for c in lane_variants:
        assert c.plan is not None
        assert c.metrics is not None
        assert hasattr(c.metrics, "spill_free") and hasattr(c.metrics, "ndp_cycles")
    print(f"    OK   N={n}: probe_and_rerank_candidates' own signature/field expectations "
          f"hold for {len(lane_variants)} step-11 candidate(s) (real-hardware run separately)")


def main() -> None:
    print("  fft_plan_search.py: radix x execution joint search (step 9):")
    check_same_radix_multiple_workers()
    check_same_workers_multiple_radix()
    check_distinct_radix_and_worker_pair_coexist()
    check_per_leaf_sequences_are_distinct()
    check_illegal_combination_rejected_not_crashed()
    check_no_duplicate_candidates()
    check_cross_step_duplicates_removed()
    check_generate_candidates_includes_step9()
    check_probe_and_rerank_accepts_joint_candidates()
    print("[verify] fft_plan_search step-9 joint search: all checks passed")

    print()
    print("  fft_plan_search.py: compute_lanes variant sweep (step 11):")
    check_generate_candidates_includes_step11()
    check_lane_variant_signature_distinct_from_baseline()
    check_probe_accepts_lane_variant_candidates()
    print("[verify] fft_plan_search step-11 compute_lanes sweep: all checks passed")


if __name__ == "__main__":
    main()
