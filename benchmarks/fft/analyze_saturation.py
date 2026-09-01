"""Analysis for revalidate_saturation.py's own dataset (docs/
cost_model_saturation.csv) -- Phase C saturation-point check from
docs/active_ndp_units_cost_task.md.

Usage: python3 analyze_saturation.py [docs/cost_model_saturation.csv]
"""
import csv
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "docs/cost_model_saturation.csv"
rows = list(csv.DictReader(open(path)))
for r in rows:
    r["replicas"] = int(r["replicas"])
    r["active_units"] = int(r["active_units"])
    r["total_worker_stage_batches"] = int(r["total_worker_stage_batches"])
    r["measured_cycles"] = int(float(r["measured_cycles"])) if r["measured_cycles"] not in ("", "None") else None
    r["cycles_per_replica"] = float(r["cycles_per_replica"]) if r["cycles_per_replica"] not in ("", "None") else None

by_strategy = defaultdict(list)
for r in rows:
    by_strategy[r["strategy"]].append(r)

for strategy, group in by_strategy.items():
    group.sort(key=lambda r: r["replicas"])
    print(f"\n=== {strategy} ===")
    print(f"{'replicas':>9} {'active_units':>13} {'batches':>8} {'waves':>6} {'cycles':>8} {'cycles/replica':>15}")
    for r in group:
        cpr = f"{r['cycles_per_replica']:>15.2f}" if r["cycles_per_replica"] is not None else f"{'n/a':>15}"
        cycles_str = r["measured_cycles"] if r["measured_cycles"] is not None else "n/a"
        print(f"{r['replicas']:>9} {r['active_units']:>13} {r['total_worker_stage_batches']:>8} "
              f"{r['persistent_waves'] or '':>6} {cycles_str!s:>8} {cpr}")

    # crude saturation check: cycles_per_replica should DROP while active_units < 32,
    # then flatten/rise once active_units saturates at 32.
    below = [r for r in group if r["active_units"] < 32 and r["cycles_per_replica"]]
    at_or_above = [r for r in group if r["active_units"] >= 32 and r["cycles_per_replica"]]
    if below and at_or_above:
        print(f"  mean cycles/replica below saturation (active<32): "
              f"{sum(r['cycles_per_replica'] for r in below)/len(below):.2f}")
        print(f"  mean cycles/replica at/above saturation (active>=32): "
              f"{sum(r['cycles_per_replica'] for r in at_or_above)/len(at_or_above):.2f}")
