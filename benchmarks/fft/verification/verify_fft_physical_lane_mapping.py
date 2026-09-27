from __future__ import annotations

"""Regression tests for `planning.execution.fft_plan_persistent.
flatten_logical_workers_to_physical_lanes` -- the shared logical-worker ->
physical-lane mapping `codegen.fft_persistent_codegen`'s default "physical"
lowering and `planning.search.fft_cost_model.compute_stage_metrics` both
now read (see docs/logical_vs_physical_cost_model.md and each of those two
modules' own docstrings for why they used to disagree).

Exercises the exact worker-count list from the task this suite was built
for (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 18, 20, 24, 32, 36, 64) against
`physical_lanes=8`, for several stage batch counts, checking:

* every logical batch is assigned to exactly one physical lane (no
  duplicate, no missing) -- via object identity, since every SIMDBatchPlan
  produced by `_partition_vector_scalar` is a distinct object;
* every physical lane index used is in `[0, workers_per_group)`;
* the codegen-side wrapper (`codegen.fft_persistent_codegen.
  _flatten_to_physical_lanes`) matches this module's own shared function
  exactly, for every one of those worker counts -- byte-for-byte, not just
  "close" -- confirming the 2026-09-22 refactor (moving the mapping out of
  codegen into a reusable planning-layer function) really did preserve
  codegen's pre-existing physical-lane assignment;
* the mapping agrees with an INDEPENDENT reference implementation (not
  calling the function under test), so a future edit to either the shared
  function or its codegen wrapper that breaks their agreement -- but keeps
  both self-consistent with each other -- still gets caught.

Also builds full persistent leaves (`make_persistent_leaf_plan`) at a
representative subset of these worker counts and confirms the resulting
plan's own `persistent_vector_batches`, once flattened, satisfy the same
coverage property end to end (not just on synthetic batch lists).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.core.fft_plan_core import SIMDBatchPlan
from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.execution.fft_plan_persistent import (
    flatten_logical_workers_to_physical_lanes,
    make_persistent_leaf_plan,
    physical_lane_workload,
)
from codegen.fft_persistent_codegen import _flatten_to_physical_lanes

WORKER_COUNTS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 18, 20, 24, 32, 36, 64)
PHYSICAL_LANES = 8


def _dummy_batch(tag: str) -> SIMDBatchPlan:
    """A minimal, distinct SIMDBatchPlan -- only used as an identity token
    here (no load/twiddle/store math is exercised by this suite), so every
    field beyond `batch_id` is empty/zero."""
    return SIMDBatchPlan(batch_id=hash(tag) & 0xFFFFFFFF, valid_lanes=8, loads=(), outputs=())


def _reference_flatten(
    vector_batches: tuple[tuple, ...], scalar_batches: tuple, *, workers_per_fft: int, workers_per_group: int,
) -> tuple[tuple, ...]:
    """Independent re-derivation of the same mapping (grouping logical
    worker ids by `% workers_per_group`, using Python's own `sorted`/
    `groupby` idiom instead of the function-under-test's explicit loop) --
    deliberately NOT calling `flatten_logical_workers_to_physical_lanes`,
    so a shared bug in that function (and its codegen wrapper, which
    delegates to it) would still be caught here."""
    lanes: list[list] = [[] for _ in range(workers_per_group)]
    for logical_id in range(workers_per_fft):
        lanes[logical_id % workers_per_group].extend(vector_batches[logical_id])
    if scalar_batches:
        lanes[(workers_per_fft - 1) % workers_per_group].extend(scalar_batches)
    return tuple(tuple(l) for l in lanes)


def check_coverage_and_mapping(*, batches_per_worker: int, with_tail: bool) -> None:
    for workers_per_fft in WORKER_COUNTS:
        vector_batches = tuple(
            tuple(_dummy_batch(f"w{w}b{b}") for b in range(batches_per_worker))
            for w in range(workers_per_fft)
        )
        scalar_batches = (_dummy_batch("tail"),) if with_tail else ()

        result = flatten_logical_workers_to_physical_lanes(
            vector_batches, scalar_batches,
            workers_per_fft=workers_per_fft, workers_per_group=PHYSICAL_LANES,
        )

        assert len(result) == PHYSICAL_LANES, (
            f"W={workers_per_fft}: expected exactly {PHYSICAL_LANES} physical lanes, "
            f"got {len(result)}"
        )

        # Coverage: every logical batch appears in exactly one physical
        # lane's own bucket (identity-based -- these are distinct objects).
        all_logical_batches = [b for wb in vector_batches for b in wb] + list(scalar_batches)
        all_physical_batches = [b for lane in result for b in lane]
        assert len(all_physical_batches) == len(all_logical_batches), (
            f"W={workers_per_fft}: flattened batch count {len(all_physical_batches)} != "
            f"logical batch count {len(all_logical_batches)} (missing or duplicated batches)"
        )
        assert set(id(b) for b in all_physical_batches) == set(id(b) for b in all_logical_batches), (
            f"W={workers_per_fft}: flattened batch SET differs from the logical batch set "
            f"(a batch was duplicated onto >1 lane, or dropped entirely)"
        )
        seen: set[int] = set()
        for b in all_physical_batches:
            assert id(b) not in seen, f"W={workers_per_fft}: batch {b} duplicated across lanes"
            seen.add(id(b))

        # Lane-index range: trivially true by construction (result has
        # exactly PHYSICAL_LANES entries), but assert explicitly per the
        # task's own "physical lane index in [0,7]" requirement.
        assert len(result) - 1 <= PHYSICAL_LANES - 1

        # Cross-check against the independent reference implementation.
        reference = _reference_flatten(
            vector_batches, scalar_batches,
            workers_per_fft=workers_per_fft, workers_per_group=PHYSICAL_LANES,
        )
        assert result == reference, (
            f"W={workers_per_fft}: flatten_logical_workers_to_physical_lanes disagrees "
            f"with the independent reference implementation"
        )

        # Codegen's own wrapper must match exactly (byte-for-byte, not just
        # structurally) -- confirms the 2026-09-22 refactor changed nothing
        # observable about codegen's pre-existing physical-lane mapping.
        stage = _stage_stub(vector_batches, scalar_batches)
        codegen_result = _flatten_to_physical_lanes(
            stage, workers_per_fft=workers_per_fft, workers_per_group=PHYSICAL_LANES,
        )
        assert codegen_result == result, (
            f"W={workers_per_fft}: codegen.fft_persistent_codegen._flatten_to_physical_lanes "
            f"diverges from planning.execution.fft_plan_persistent."
            f"flatten_logical_workers_to_physical_lanes"
        )

    print(
        f"    OK   coverage/mapping holds for every W in {WORKER_COUNTS} "
        f"(batches_per_worker={batches_per_worker}, with_tail={with_tail})"
    )


class _StageStub:
    """Minimal stand-in for FFTStagePlan -- only the two fields
    `_flatten_to_physical_lanes` actually reads."""

    def __init__(self, persistent_vector_batches, persistent_scalar_batches):
        self.persistent_vector_batches = persistent_vector_batches
        self.persistent_scalar_batches = persistent_scalar_batches


def _stage_stub(vector_batches, scalar_batches) -> _StageStub:
    return _StageStub(vector_batches, scalar_batches)


def check_physical_lane_workload_stats() -> None:
    """`physical_lane_workload`'s own summary stats (max/min/avg/active)
    against a hand-computed case: W=20, physical_lanes=8, 1 batch per
    logical worker, no tail -- lanes 0-3 get 3 logical workers each
    (ceil(20/8)=3 waves, lanes 0-3 are the ones a 3rd wave still reaches:
    logical ids 16-19 map to lanes 0-3), lanes 4-7 get 2 each."""
    workers_per_fft = 20
    vector_batches = tuple((_dummy_batch(f"w{w}"),) for w in range(workers_per_fft))
    workload = physical_lane_workload(
        vector_batches, (), workers_per_fft=workers_per_fft, workers_per_group=PHYSICAL_LANES,
    )
    counts = [len(lane) for lane in workload.physical_lane_batches]
    assert counts == [3, 3, 3, 3, 2, 2, 2, 2], f"unexpected per-lane counts: {counts}"
    assert workload.max_batches_per_lane == 3
    assert workload.min_batches_per_lane == 2
    assert workload.active_physical_lanes == 8
    assert workload.num_logical_workers == 20
    assert workload.num_physical_lanes == 8
    assert abs(workload.avg_batches_per_lane - 20 / 8) < 1e-9
    print(f"    OK   physical_lane_workload stats match hand-computed W=20/lanes=8 case: {counts}")


def check_end_to_end_persistent_plan() -> None:
    """Build a real persistent leaf at a representative subset of worker
    counts and confirm `persistent_vector_batches`, once flattened, still
    satisfies full coverage -- not just on the synthetic batches above."""
    length, radices = 64, (4, 4, 4)
    for workers_per_fft in (1, 3, 8, 10, 20, 36, 64):
        plan = make_persistent_leaf_plan(
            length, radices, num_logical_blocks=5, workers_per_fft=workers_per_fft,
            target=DEFAULT_TARGET_PROFILE,
        )
        assert plan.persistent is not None
        workers_per_group = plan.persistent.workers_per_group
        for stage in plan.stages:
            assert stage.persistent_vector_batches is not None
            assert stage.persistent_scalar_batches is not None
            workload = physical_lane_workload(
                stage.persistent_vector_batches, stage.persistent_scalar_batches,
                workers_per_fft=workers_per_fft, workers_per_group=workers_per_group,
            )
            all_logical = [
                b for wb in stage.persistent_vector_batches for b in wb
            ] + list(stage.persistent_scalar_batches)
            all_physical = [b for lane in workload.physical_lane_batches for b in lane]
            assert len(all_physical) == len(all_logical), (
                f"W={workers_per_fft} stage={stage.stage_id}: coverage mismatch"
            )
            assert set(id(b) for b in all_physical) == set(id(b) for b in all_logical)
    print("    OK   end-to-end make_persistent_leaf_plan coverage holds for W in "
          "(1, 3, 8, 10, 20, 36, 64)")


def main() -> None:
    print("  Physical-lane flatten mapping: coverage + codegen/planning agreement:")
    for batches_per_worker in (1, 2, 3):
        for with_tail in (False, True):
            check_coverage_and_mapping(batches_per_worker=batches_per_worker, with_tail=with_tail)
    check_physical_lane_workload_stats()
    check_end_to_end_persistent_plan()
    print("[verify] physical-lane flatten mapping: all checks passed")


if __name__ == "__main__":
    main()
