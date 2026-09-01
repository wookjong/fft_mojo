from __future__ import annotations

"""Phase C: synthetic NDP-unit-saturation sweep (docs/
active_ndp_units_cost_task.md). No naturally-occurring FFT candidate in
this project has enough replicas to push active units well past
`target.num_ndp_units` (32) -- see that doc's Phase 5 "saturation point"
gap. This holds one small leaf's own arithmetic work fixed (N=64,
radix (4,4,4)) and varies ONLY `batch` (replica count), independently for
non-cooperative / cooperative(workers=4) / persistent, so active-unit
count moves from far below 32 to far above it while everything else about
the kernel body stays identical.

Usage: python3 revalidate_saturation.py [--out-prefix docs/cost_model_saturation]
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from planning.fft_cost_model import DEFAULT_COST_WEIGHTS, estimate_metrics
from planning.fft_plan_persistent import num_rounds
from planning.fft_plan_recursive import make_recursive_transpose_plan
from planning.fft_unit_utilization import compute_unit_utilization
from planning.target_profile import DEFAULT_TARGET_PROFILE
from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from revalidate_cost_model import _build_and_run, _has_spill
from planning.spill_probe import _parse_ndp_cycles

N = 64  # fixed leaf: radix (4,4,4), small enough to keep each run fast
REPLICAS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
STRATEGIES = {
    "noncoop": {},
    "coop4": {"cooperative_workers": 4},
    "coop8": {"cooperative_workers": 8},  # kept, flagged separately -- known anomaly
    "persistent": {"persistent_leaf": True},
}


@dataclass
class SaturationResult:
    replicas: int
    strategy: str
    active_units: int
    total_worker_stage_batches: int
    persistent_waves: int | None
    measured_cycles: int | None
    cycles_per_replica: float | None
    spill_free: bool | None
    build_ok: bool
    run_ok: bool


def run_one(replicas: int, strategy: str, kwargs: dict) -> SaturationResult:
    plan = make_recursive_transpose_plan(
        N, scratchpad_byte_budget=4096, simd_lanes=8, batch=replicas,
        spad_capacity_bytes=DEFAULT_TARGET_PROFILE.spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=DEFAULT_TARGET_PROFILE.max_concurrent_scratchpad_bytes,
        **kwargs,
    )
    metrics = estimate_metrics(plan, DEFAULT_TARGET_PROFILE)
    estimates = compute_unit_utilization(plan, DEFAULT_TARGET_PROFILE)
    active_units = estimates[0].active_units_best

    waves = None
    if strategy == "persistent":
        waves = num_rounds(replicas, DEFAULT_TARGET_PROFILE.num_ndp_units)

    source = generate_recursive_fft_kernels(
        plan, target=DEFAULT_TARGET_PROFILE, reference_check=False,
        compute_lanes=4, narrow_middle_stages=True,
    )
    build_ok, run_ok, log = _build_and_run(source, timeout=180)
    cycles = _parse_ndp_cycles(log) if run_ok else None
    spill_free = (not _has_spill(log)) if run_ok else None

    return SaturationResult(
        replicas=replicas, strategy=strategy, active_units=active_units,
        total_worker_stage_batches=metrics.total_worker_stage_batches,
        persistent_waves=waves, measured_cycles=cycles,
        cycles_per_replica=(cycles / replicas) if cycles else None,
        spill_free=spill_free, build_ok=build_ok, run_ok=run_ok,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-prefix", default="docs/cost_model_saturation")
    parser.add_argument(
        "--only", default=None,
        help="comma-separated replicas:strategy pairs to (re)run, e.g. '128:noncoop,256:coop4' "
             "-- merges into the existing --out-prefix .json rather than overwriting the whole sweep",
    )
    args = parser.parse_args()

    existing_by_key: dict[tuple, dict] = {}
    if args.only:
        try:
            existing_by_key = {
                (r["replicas"], r["strategy"]): r
                for r in json.loads(Path(args.out_prefix + ".json").read_text())
            }
        except FileNotFoundError:
            pass
        wanted = set()
        for pair in args.only.split(","):
            rep, strat = pair.split(":")
            wanted.add((int(rep), strat))
        work_items = [(rep, strat, STRATEGIES[strat]) for rep, strat in sorted(wanted)]
    else:
        work_items = [
            (replicas, strategy, kwargs)
            for replicas in REPLICAS for strategy, kwargs in STRATEGIES.items()
        ]

    results: list[SaturationResult] = []
    total = len(work_items)
    for i, (replicas, strategy, kwargs) in enumerate(work_items, start=1):
        print(f"[{i}/{total}] replicas={replicas} strategy={strategy} ...", flush=True)
        r = run_one(replicas, strategy, kwargs)
        results.append(r)
        print(f"    -> active_units={r.active_units} batches={r.total_worker_stage_batches} "
              f"waves={r.persistent_waves} cycles={r.measured_cycles} "
              f"cycles/replica={r.cycles_per_replica}", flush=True)
        existing_by_key[(r.replicas, r.strategy)] = asdict(r)

    rows = list(existing_by_key.values()) if args.only else [asdict(r) for r in results]
    rows.sort(key=lambda r: (r["replicas"], r["strategy"]))
    Path(args.out_prefix + ".json").write_text(json.dumps(rows, indent=2))
    with open(args.out_prefix + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out_prefix}.csv/.json ({len(rows)} rows)")


if __name__ == "__main__":
    main()
