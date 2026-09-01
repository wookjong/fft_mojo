from __future__ import annotations

"""Verification for `planning/fft_unit_utilization.py` -- a diagnostic-only
active-NDP-unit-count estimate, NOT a cost term (see that module's own
docstring and docs/active_ndp_units_cost_task.md). These checks establish
two separate things:

1. `distinct_active_units`'s closed form is byte-for-byte identical to a
   brute-force reimplementation of the real simulator's own per-packet
   loop (`M2NDPConfig::get_uthread_size`, third_party/m2ndp-detour/src/
   m2ndp_config.cc:112-121) -- the actual re-derivation-from-source this
   phase was scoped to do, not a trust-the-docstring assumption.
2. The module reproduces the known real-hardware persistent-vs-split
   pathology (`docs/persistent_representative_sweep.md`'s N=960/1024
   unsplit-persistent numbers -- originally reported ~50x slower,
   corrected 2026-08-31 to ~2.26x once the split baseline's own cycle
   count was fixed, see that doc's own "CORRECTION" section) as a SANITY
   CHECK ONLY -- this data is explicitly NOT used to fit any formula or
   weight here (see this project's own instruction: unsplit N=960/1024 is
   for confirming the diagnostic reproduces a known pathology, not for
   calibration).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.fft_plan_persistent import make_persistent_leaf_plan
from planning.fft_plan_recursive import make_recursive_transpose_plan
from planning.fft_plan_search import _root_split_length
from planning.fft_unit_utilization import compute_unit_utilization, distinct_active_units
from planning.target_profile import DEFAULT_TARGET_PROFILE


def _brute_force_units(base_packet_index: int, count: int, *, chunk: int, num_units: int) -> int:
    """Literal reimplementation of `M2NDPConfig::get_uthread_size`'s own
    per-packet loop (m2ndp_config.cc:112-121), one packet per iteration
    -- deliberately not the closed form, so this is an independent check,
    not a restatement of `distinct_active_units` itself."""
    units = set()
    for i in range(count):
        packet_index = base_packet_index + i
        units.add((packet_index // chunk) % num_units)
    return len(units)


def check_distinct_active_units_matches_brute_force() -> None:
    """200,000 random (base offset, launch size, chunk, unit count)
    combinations, closed form vs. literal per-packet loop -- including
    launches large enough to wrap around the full interleave period more
    than once."""
    import random

    rng = random.Random(0)
    trials = 200_000
    mismatches = []
    for _ in range(trials):
        chunk = rng.choice([1, 2, 4, 8])
        num_units = rng.choice([1, 2, 4, 8, 32])
        base_packet_index = rng.randint(0, 5000)
        count = rng.randint(0, 2000)
        brute = _brute_force_units(base_packet_index, count, chunk=chunk, num_units=num_units)
        closed = distinct_active_units(
            count, base_offset_uthreads=base_packet_index,
            interleave_chunk_uthreads=chunk, num_ndp_units=num_units,
        )
        if brute != closed:
            mismatches.append((base_packet_index, count, chunk, num_units, brute, closed))
    assert not mismatches, f"closed form disagreed with brute force: {mismatches[:5]}"
    print(f"    OK   distinct_active_units matches a brute-force reimplementation of "
          f"M2NDPConfig::get_uthread_size's own per-packet loop across {trials} random "
          f"(base, size, chunk, num_units) combinations, including wraparound")


def check_distinct_active_units_edge_cases() -> None:
    """Hand-checked edge cases: zero launch, exactly-one-chunk launch,
    exactly-one-full-period launch, and the sub-chunk 2-unit-span case
    `_safe_round_size`'s own docstring calls out (a launch narrower than
    one interleave chunk can still span 2 adjacent units if it starts
    away from a chunk boundary)."""
    assert distinct_active_units(0, base_offset_uthreads=0, interleave_chunk_uthreads=8, num_ndp_units=32) == 0
    assert distinct_active_units(8, base_offset_uthreads=0, interleave_chunk_uthreads=8, num_ndp_units=32) == 1
    assert distinct_active_units(256, base_offset_uthreads=0, interleave_chunk_uthreads=8, num_ndp_units=32) == 32
    assert distinct_active_units(512, base_offset_uthreads=0, interleave_chunk_uthreads=8, num_ndp_units=32) == 32
    # 3 uthreads starting 6 into an 8-wide chunk: spans [6,7] (unit A) and [0,1] of
    # the next chunk (unit A+1) -- 2 distinct units from a launch narrower than one chunk.
    assert distinct_active_units(3, base_offset_uthreads=6, interleave_chunk_uthreads=8, num_ndp_units=32) == 2
    print("    OK   distinct_active_units: zero/one-chunk/one-period/sub-chunk-span edge cases")


def check_cooperative_persistent_report_exact_alignment() -> None:
    """Cooperative and persistent leaves must report `alignment_exact=True`
    (best == worst, a confirmed-exact 0 base offset -- see module
    docstring's citation of the two real pool-alignment fixes) while a
    plain non-cooperative leaf and every transpose stage must report
    `alignment_exact=False` with a genuine best/worst range whenever their
    own launch width isn't already chunk-aligned."""
    from planning.fft_plan_search import generate_radix_execution_joint_candidates

    n = 960
    baseline = make_recursive_transpose_plan(
        n, scratchpad_byte_budget=4096, simd_lanes=8,
        spad_capacity_bytes=DEFAULT_TARGET_PROFILE.spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=DEFAULT_TARGET_PROFILE.max_concurrent_scratchpad_bytes,
        interleave_chunk_uthreads=DEFAULT_TARGET_PROFILE.interleave_chunk_uthreads,
    )
    split = _root_split_length(baseline)
    joint = generate_radix_execution_joint_candidates(
        n, target=DEFAULT_TARGET_PROFILE, inverse=False, scratchpad_byte_budget=4096,
        simd_lanes=8, batch=1, baseline_plan=baseline, baseline_split=split,
    )
    cooperative_plan = None
    for plan, choices in joint:
        if choices.worker_sequence and any(isinstance(w, int) for w in choices.worker_sequence):
            cooperative_plan = plan
            break
    assert cooperative_plan is not None, "expected at least one cooperative candidate for N=960"

    estimates = compute_unit_utilization(cooperative_plan, DEFAULT_TARGET_PROFILE)
    coop_estimates = [e for e in estimates if e.family == "cooperative"]
    assert coop_estimates, f"expected a cooperative leaf estimate, got families {[e.family for e in estimates]}"
    for e in coop_estimates:
        assert e.alignment_exact is True
        assert e.active_units_best == e.active_units_worst

    transpose_estimates = [e for e in estimates if e.family.startswith("transpose_")]
    assert transpose_estimates, "expected at least one transpose stage for N=960's own split"
    for e in transpose_estimates:
        assert e.alignment_exact is False

    print(f"    OK   N={n}: cooperative leaf(s) report alignment_exact=True "
          f"(best==worst), all {len(transpose_estimates)} transpose stage(s) report "
          f"alignment_exact=False (genuine best/worst range)")


def check_unsplit_persistent_pathology_reproduced_as_sanity_check_only() -> None:
    """NOT formula fitting -- see this module's own docstring and
    docs/active_ndp_units_cost_task.md's explicit instruction that the old
    unsplit N=960/1024 comparison is a pathology-reproduction sanity check
    only. `docs/persistent_representative_sweep.md` originally reported
    this exact unsplit persistent leaf at ~50x slower than the split/
    non-persistent baseline -- corrected 2026-08-31 (see that doc's own
    "CORRECTION" section) to ~2.26x once the split baseline's own real
    cycle count is measured correctly (the "~50x" figure divided the
    correct unsplit number, 109074, by the split baseline's own
    ~23x-too-small buggy reading, 2114, instead of its real 48320). Still
    slower, just not catastrophically so. This check confirms the
    diagnostic shows why in active-unit terms (1 of 32 units, vs. the
    split baseline's transpose stages spreading across many), not that
    either the old 50x or the corrected 2.26x figure derives from this
    module's numbers -- neither does, and must not."""
    n = 960
    unsplit_persistent = make_persistent_leaf_plan(
        n, radices=(4, 4, 4, 3, 5), num_logical_blocks=1,
        simd_lanes=8, inverse=False, target=DEFAULT_TARGET_PROFILE,
    )
    from planning.fft_unit_utilization import _persistent_leaf_estimate

    estimate = _persistent_leaf_estimate(
        unsplit_persistent, 1, leaf_index=0, target=DEFAULT_TARGET_PROFILE,
    )
    assert estimate.active_units_best == 1, (
        f"expected the old unsplit persistent shape (num_logical_blocks=1) to activate "
        f"exactly 1 of {DEFAULT_TARGET_PROFILE.num_ndp_units} units, got {estimate.active_units_best}"
    )

    split_baseline = make_recursive_transpose_plan(n, scratchpad_byte_budget=4096, simd_lanes=8)
    split_estimates = compute_unit_utilization(split_baseline, DEFAULT_TARGET_PROFILE)
    transpose_units = [e.active_units_best for e in split_estimates if e.family.startswith("transpose_")]
    assert transpose_units and max(transpose_units) > estimate.active_units_best, (
        f"expected the split baseline's own transpose stages to activate more units than "
        f"the unsplit persistent leaf's 1: transpose_units={transpose_units}"
    )
    print(f"    OK   N={n}: unsplit persistent leaf (num_logical_blocks=1) activates "
          f"exactly 1/{DEFAULT_TARGET_PROFILE.num_ndp_units} units (real ~2.26x-slower, "
          f"corrected from the originally-reported ~50x, pathology from docs/"
          f"persistent_representative_sweep.md reproduced in active-unit terms -- "
          f"SANITY CHECK ONLY, not used for any formula/weight); split baseline's own "
          f"transpose stages reach up to {max(transpose_units)} units")


def check_split_persistent_consistent_with_corrected_real_gap() -> None:
    """`docs/persistent_recursive_split.md`'s own real-hardware table
    originally claimed persistent and non-persistent land at near parity
    at a fixed real split -- that specific claim was corrected 2026-08-31
    (see that doc's own "CORRECTION" section and docs/
    active_ndp_units_cost_task.md's Phase 1.5 writeup): the cycle numbers
    it cited were measured with a since-fixed bug that only ever captured
    the LAST kernel struct's own duration, not the true end-to-end total.
    The CORRECTED real totals are NOT parity -- persistent is genuinely
    ~18-19% faster (N=960: 39378 vs. 48320; N=1024: 39730 vs. 49061) --
    but the underlying *reason* this module's diagnostic gives is still
    consistent with that corrected direction: the far_child leaf (the big
    one, most of each plan's own total work) shows comparably-high
    active-unit counts for both strategies at this split (persistent
    slightly ahead), a world apart from the ~30x-gap the unsplit pathology
    check above reproduces. This is a consistency check against corrected
    real-hardware data, NOT a new calibration -- see this module's own
    docstring and docs/active_ndp_units_cost_task.md: the production cost
    formula still needs more such real remeasurement before it's
    decided."""
    for n in (960, 1024):
        persistent_plan = make_recursive_transpose_plan(
            n, scratchpad_byte_budget=4096, simd_lanes=8, persistent_leaf=True,
        )
        non_persistent_plan = make_recursive_transpose_plan(
            n, scratchpad_byte_budget=4096, simd_lanes=8,
        )
        persistent_far = [
            e for e in compute_unit_utilization(persistent_plan, DEFAULT_TARGET_PROFILE)
            if e.family == "persistent"
        ][-1]
        non_persistent_far = [
            e for e in compute_unit_utilization(non_persistent_plan, DEFAULT_TARGET_PROFILE)
            if e.family == "non_cooperative"
        ][-1]
        assert persistent_far.active_units_best >= DEFAULT_TARGET_PROFILE.num_ndp_units - 1, (
            f"N={n}: expected the far_child persistent leaf to nearly saturate all "
            f"{DEFAULT_TARGET_PROFILE.num_ndp_units} units, got {persistent_far.active_units_best}"
        )
        assert non_persistent_far.active_units_best >= DEFAULT_TARGET_PROFILE.num_ndp_units - 2, (
            f"N={n}: expected the far_child non-cooperative leaf to also nearly saturate "
            f"all {DEFAULT_TARGET_PROFILE.num_ndp_units} units (consistent with the "
            f"corrected ~18-19%-faster-not-parity real gap), got {non_persistent_far.active_units_best}"
        )
        print(f"    OK   N={n}: far_child active units -- persistent="
              f"{persistent_far.active_units_best}, non_cooperative="
              f"{non_persistent_far.active_units_best} of {DEFAULT_TARGET_PROFILE.num_ndp_units} "
              f"(both near-saturated -- consistent with the corrected real gap being a modest "
              f"~18-19%, not the 32x-apart unsplit-pathology shape; NOT parity, see docs/"
              f"persistent_recursive_split.md's own correction)")


def main() -> None:
    print("  fft_unit_utilization.py: diagnostic active-NDP-unit-count estimate:")
    check_distinct_active_units_matches_brute_force()
    check_distinct_active_units_edge_cases()
    check_cooperative_persistent_report_exact_alignment()
    check_unsplit_persistent_pathology_reproduced_as_sanity_check_only()
    check_split_persistent_consistent_with_corrected_real_gap()
    print("[verify] active-NDP-unit-count diagnostic: all checks passed")


if __name__ == "__main__":
    main()
