"""Analysis for revalidate_cost_model.py's own dataset (docs/
cost_model_revalidation.csv) -- Phase B (matched-pair cycles-per-batch
regression) and Phase E (ranking regression: regret, top-k hit,
Spearman) from docs/active_ndp_units_cost_task.md.

Usage: python3 analyze_revalidation.py [docs/cost_model_revalidation.csv]
"""
import csv
import sys
from collections import defaultdict
from statistics import mean, median, pstdev

path = sys.argv[1] if len(sys.argv) > 1 else "docs/cost_model_revalidation.csv"
rows = list(csv.DictReader(open(path)))
for r in rows:
    for k in ("estimated_cost", "memory_cost", "execution_cost", "regret"):
        if r[k] not in ("", "None"):
            r[k] = float(r[k])
    for k in ("total_worker_stage_batches", "n", "model_rank", "measured_rank"):
        if r[k] not in ("", "None"):
            r[k] = int(float(r[k]))
    if r["measured_cycles"] not in ("", "None"):
        r["measured_cycles"] = int(float(r["measured_cycles"]))
    else:
        r["measured_cycles"] = None

by_n = defaultdict(list)
for r in rows:
    by_n[r["n"]].append(r)

print("=" * 100)
print("PHASE B: matched-pair cycles-per-batch coefficient")
print("=" * 100)

pairs = []
for n, group in sorted(by_n.items()):
    baseline = next((r for r in group if r["label"] == "noncoop"), None)
    if baseline is None or baseline["measured_cycles"] is None:
        continue
    for r in group:
        if r["label"] in ("noncoop", "mixed_persistent_near_noncoop_far"):
            continue
        if r["measured_cycles"] is None:
            continue
        d_cycles = r["measured_cycles"] - baseline["measured_cycles"]
        d_batches = r["total_worker_stage_batches"] - baseline["total_worker_stage_batches"]
        if d_batches == 0:
            continue
        coef = d_cycles / d_batches
        pairs.append((n, r["label"], d_cycles, d_batches, coef))
        flag = "  <-- worker=8 anomaly, exclude from central estimate" if "coop8" in r["label"] else ""
        print(f"N={n:5} {r['label']:10} vs noncoop: d_cycles={d_cycles:7} d_batches={d_batches:5} "
              f"cycles_per_batch={coef:9.2f}{flag}")

clean = [c for _, label, _, _, c in pairs if "coop8" not in label]
if clean:
    print(f"\ncycles_per_batch (excluding worker=8): "
          f"median={median(clean):.2f} mean={mean(clean):.2f} "
          f"stddev={pstdev(clean):.2f} min={min(clean):.2f} max={max(clean):.2f} n={len(clean)}")

# simple linear regression: measured_cycle_delta ~= alpha * batch_delta + intercept
if len(clean) >= 2:
    xs = [d_batches for n, label, d_cycles, d_batches, c in pairs if "coop8" not in label]
    ys = [d_cycles for n, label, d_cycles, d_batches, c in pairs if "coop8" not in label]
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    alpha = sxy / sxx if sxx else float("nan")
    intercept = my - alpha * mx
    print(f"linear regression: measured_cycle_delta ~= {alpha:.2f} * batch_delta + {intercept:.2f}")
    print(f"(current CostWeights.stage_work = 0.1 -- compare against alpha above)")

print()
print("=" * 100)
print("PHASE E: ranking regression per N")
print("=" * 100)


def spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    rx = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: xs[i]))}
    ry = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: ys[i]))}
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n**2 - 1))


worst_regret = 0.0
worst_n = None
for n, group in sorted(by_n.items()):
    measured = [r for r in group if r["measured_cycles"] is not None]
    if len(measured) < 2:
        continue
    measured.sort(key=lambda r: r["measured_rank"])
    top1 = next(r for r in measured if r["model_rank"] == 1)
    best = next(r for r in measured if r["measured_rank"] == 1)
    top1_regret = top1["measured_cycles"] / best["measured_cycles"] - 1
    top3_labels = {r["label"] for r in measured if r["model_rank"] <= 3}
    best_in_top3 = best["label"] in top3_labels
    xs = [r["estimated_cost"] for r in measured]
    ys = [r["measured_cycles"] for r in measured]
    rho = spearman(xs, ys)
    if top1_regret > worst_regret:
        worst_regret = top1_regret
        worst_n = n
    print(f"N={n:5} (n_candidates={len(measured)}): model's #1 pick regret={top1_regret:.1%} "
          f"best_in_model_top3={best_in_top3} spearman={rho:.3f} "
          f"model_pick={top1['label']} real_best={best['label']}")

print(f"\nworst-case top-1 regret across all N: {worst_regret:.1%} (N={worst_n})")
