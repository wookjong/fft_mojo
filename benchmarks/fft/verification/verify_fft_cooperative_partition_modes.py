from __future__ import annotations

"""Verification for the cooperative batch-partition alternatives (P2.2 --
see docs/priority2_execution_strategies.md and `fft_plan_cooperative.
partition_batches`'s own docstring).

Covers both halves of the task's own requirement:

1. Structural correctness of all three selectable modes (`round_robin`,
   `contiguous`, `balanced_contiguous`) over synthetic batch lists at
   several (batch_count, worker_count) combinations, INCLUDING ones where
   `batch_count % worker_count != 0` (the case where `contiguous` and
   `balanced_contiguous` genuinely differ from each other, not just from
   `round_robin`) -- union(assignments) == all batches, intersections ==
   empty, deterministic result.
2. Real end-to-end FFT numeric correctness (via `verify_fft_cooperative.
   run_cooperative_kernel`, the actual emitted stage text) for all three
   modes on the SAME plan shape/input, confirming they compute the
   IDENTICAL FFT (only the batch-to-worker assignment differs, never the
   math).
3. `generate_partition_mode_candidates`'s own dedup: exactly 1 candidate
   survives when every stage's own batch count is too small for the
   modes to differ; exactly 2 or 3 when they genuinely do.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from planning.core.fft_plan_core import SIMDBatchPlan
from planning.execution.fft_plan_cooperative import (
    _ALL_PARTITION_MODES,
    generate_partition_mode_candidates,
    make_cooperative_leaf_plan,
    partition_batches,
)
from verification.verify_fft_cooperative import run_cooperative_kernel
from verification.verify_fft_harness import Ptr


def _dummy_batches(n: int) -> tuple[SIMDBatchPlan, ...]:
    return tuple(SIMDBatchPlan(batch_id=i, valid_lanes=8, loads=(), outputs=()) for i in range(n))


def check_structural_correctness() -> None:
    cases = [(1, 4), (4, 4), (5, 4), (16, 4), (17, 4), (17, 8), (3, 8), (100, 8), (105, 8)]
    for n, workers in cases:
        batches = _dummy_batches(n)
        for mode in _ALL_PARTITION_MODES:
            result = partition_batches(batches, workers, mode=mode)
            assert len(result) == workers, f"mode={mode} n={n} workers={workers}: wrong bucket count"
            all_ids = [b.batch_id for bucket in result for b in bucket]
            assert sorted(all_ids) == list(range(n)), (
                f"mode={mode} n={n} workers={workers}: union(assignments) != all batches "
                f"(got {sorted(all_ids)})"
            )
            assert len(all_ids) == len(set(all_ids)), (
                f"mode={mode} n={n} workers={workers}: duplicate batch assignment"
            )
            # determinism
            result2 = partition_batches(batches, workers, mode=mode)
            assert result == result2, f"mode={mode} n={n} workers={workers}: non-deterministic"
    print(f"    OK   all 3 partition modes: union==all batches, no duplicates, deterministic, "
          f"across {len(cases)} (batch_count, worker_count) combinations")


def check_contiguous_and_balanced_contiguous_genuinely_differ() -> None:
    """n=17, workers=4: contiguous uses a fixed chunk=ceil(17/4)=5, giving
    sizes [5,5,5,2] (front-loaded, ragged tail); balanced_contiguous
    spreads the single remainder batch onto the FIRST worker only, giving
    sizes [5,4,4,4] -- genuinely different bucket-size SHAPES, not just a
    different internal ordering of the same sizes."""
    batches = _dummy_batches(17)
    contiguous = partition_batches(batches, 4, mode="contiguous")
    balanced = partition_batches(batches, 4, mode="balanced_contiguous")
    contiguous_sizes = [len(b) for b in contiguous]
    balanced_sizes = [len(b) for b in balanced]
    assert contiguous_sizes == [5, 5, 5, 2], f"unexpected contiguous sizes: {contiguous_sizes}"
    assert balanced_sizes == [5, 4, 4, 4], f"unexpected balanced_contiguous sizes: {balanced_sizes}"
    assert contiguous_sizes != balanced_sizes
    print(f"    OK   n=17 workers=4: contiguous sizes={contiguous_sizes} (ragged tail) != "
          f"balanced_contiguous sizes={balanced_sizes} (remainder spread across the front) -- "
          f"genuinely distinct partitions, not just distinct orderings")


def check_round_robin_still_matches_pre_p22_behavior() -> None:
    """`partition_batches(..., mode="round_robin")` must be byte-identical
    to the pre-P2.2 `_partition_batches` implementation -- worker `w`
    owns `{w, w+workers, w+2*workers, ...}`."""
    batches = _dummy_batches(17)
    result = partition_batches(batches, 4, mode="round_robin")
    for w in range(4):
        expected_ids = list(range(w, 17, 4))
        got_ids = [b.batch_id for b in result[w]]
        assert got_ids == expected_ids, f"worker {w}: {got_ids} != {expected_ids}"
    print("    OK   partition_batches(mode='round_robin') matches the pre-P2.2 strided formula exactly")


def check_end_to_end_fft_correctness_all_modes() -> None:
    length, radices, workers_per_fft, total_ffts = 2048, (4, 4, 4, 4, 4, 2), 4, 3
    total = length * total_ffts
    rng = np.random.default_rng(7)
    xr = rng.uniform(-1, 1, total)
    xi = rng.uniform(-1, 1, total)
    expected = np.concatenate([
        np.fft.fft(xr[b * length:(b + 1) * length] + 1j * xi[b * length:(b + 1) * length])
        for b in range(total_ffts)
    ])
    outputs = {}
    for mode in _ALL_PARTITION_MODES:
        plan = make_cooperative_leaf_plan(
            length, radices, workers_per_fft=workers_per_fft, total_ffts=total_ffts, partition_mode=mode,
        )
        assert plan.cooperation is not None and plan.cooperation.partition_mode == mode
        in_r, in_i = Ptr(total), Ptr(total)
        for i in range(total):
            in_r.arr[i] = xr[i]
            in_i.arr[i] = xi[i]
        out_r, out_i = Ptr(total), Ptr(total)
        run_cooperative_kernel(plan, input_real=in_r, input_imag=in_i, output_real=out_r, output_imag=out_i)
        got = out_r.arr + 1j * out_i.arr
        err = float(np.max(np.abs(got - expected)))
        assert err < 1e-3, f"mode={mode}: max_err={err}"
        outputs[mode] = got
    # All three modes must compute the exact same FFT on the same input --
    # bit-for-bit, since only the batch-to-worker assignment differs.
    for mode in _ALL_PARTITION_MODES[1:]:
        diff = float(np.max(np.abs(outputs[mode] - outputs[_ALL_PARTITION_MODES[0]])))
        assert diff == 0.0, f"mode={mode} vs {_ALL_PARTITION_MODES[0]}: diff={diff} (expected exact 0)"
    print(f"    OK   N=2048 radices=(4,4,4,4,4,2): all 3 partition modes compute bit-for-bit "
          f"identical, numpy-correct FFT output (max_err < 1e-3 vs numpy.fft)")


def check_candidate_dedup() -> None:
    small = generate_partition_mode_candidates(64, (4, 4, 4), workers_per_fft=4, total_ffts=8)
    assert len(small) == 1, (
        f"expected all 3 modes to degenerate identically for this small case, got {len(small)} "
        f"distinct candidates"
    )
    large = generate_partition_mode_candidates(2048, (4, 4, 4, 4, 4, 2), workers_per_fft=4, total_ffts=3)
    assert len(large) == 2, (
        f"expected round_robin distinct from {{contiguous, balanced_contiguous}} (which coincide "
        f"when batch_count % workers_per_fft == 0) for this case, got {len(large)}"
    )
    print(f"    OK   generate_partition_mode_candidates dedups correctly: 1 candidate for a small "
          f"leaf (all modes degenerate identically), 2 for a larger one (round_robin distinct, "
          f"contiguous==balanced_contiguous when evenly divisible)")


def main() -> None:
    print("  Cooperative batch partition alternatives (P2.2):")
    check_structural_correctness()
    check_contiguous_and_balanced_contiguous_genuinely_differ()
    check_round_robin_still_matches_pre_p22_behavior()
    check_end_to_end_fft_correctness_all_modes()
    check_candidate_dedup()
    print("[verify] cooperative batch partition alternatives: all checks passed")


if __name__ == "__main__":
    main()
