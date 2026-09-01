"""Production-vs-analysis equivalence check for the mechanism-aware
correction integrated into planning/fft_cost_model.py (persistent_stage_
batch_multiplier=1.746, persistent_extra_round_multiplier=11.0). Confirms
the ACTUAL production `estimate_cost` reproduces the same ranking
`analyze_mechanism_correction.py`'s own Model 2 already validated against
docs/cost_model_revalidation.csv -- not just similar, the same picks.

Usage: python3 verify_production_mechanism_correction.py
"""
import csv
from collections import defaultdict
from statistics import mean, median

from planning.fft_cost_model import DEFAULT_COST_WEIGHTS, estimate_cost, estimate_metrics
from planning.fft_plan_recursive import make_recursive_transpose_plan
from planning.target_profile import DEFAULT_TARGET_PROFILE as T

rows = list(csv.DictReader(open("docs/cost_model_revalidation.csv")))
for r in rows:
    r["n"] = int(r["n"])
    r["plan_kwargs"] = eval(r["plan_kwargs"])  # noqa: S307 -- our own repr() output
    r["measured_cycles"] = int(float(r["measured_cycles"])) if r["measured_cycles"] not in ("", "None") else None

for r in rows:
    plan = make_recursive_transpose_plan(
        r["n"], scratchpad_byte_budget=4096, simd_lanes=8,
        spad_capacity_bytes=T.spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=T.max_concurrent_scratchpad_bytes,
        **r["plan_kwargs"],
    )
    metrics = estimate_metrics(plan, T)
    r["production_cost"] = estimate_cost(metrics, DEFAULT_COST_WEIGHTS)
    r["old_model0_cost"] = float(r["estimated_cost"])


def spearman(xs, ys):
    n = len(xs)
    rx = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: xs[i]))}
    ry = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: ys[i]))}
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n**2 - 1)) if n > 1 else float("nan")


by_n = defaultdict(list)
for r in rows:
    by_n[r["n"]].append(r)

# Independently-known validated Model 2 picks (from analyze_mechanism_
# correction.py's own last run) -- the production formula must reproduce
# these exactly, not just "similarly."
EXPECTED_MODEL2_PICKS = {
    144: "persistent", 216: "coop4", 512: "coop4", 630: "coop4",
    960: "coop8", 1024: "coop4", 2048: "coop4",
}

print("=" * 100)
print("Production estimate_cost vs. old Model 0, per N")
print("=" * 100)

all_match = True
old_regrets, new_regrets = [], []
for n, group in sorted(by_n.items()):
    measured = [r for r in group if r["measured_cycles"] is not None]
    if len(measured) < 2:
        continue
    old_pick = min(measured, key=lambda r: r["old_model0_cost"])
    new_pick = min(measured, key=lambda r: r["production_cost"])
    real_best = min(measured, key=lambda r: r["measured_cycles"])
    old_regret = old_pick["measured_cycles"] / real_best["measured_cycles"] - 1
    new_regret = new_pick["measured_cycles"] / real_best["measured_cycles"] - 1
    old_regrets.append(old_regret)
    new_regrets.append(new_regret)
    xs = [r["production_cost"] for r in measured]
    ys = [r["measured_cycles"] for r in measured]
    rho = spearman(xs, ys)
    match = new_pick["label"] == EXPECTED_MODEL2_PICKS[n]
    all_match &= match
    print(f"N={n:5}: old_pick={old_pick['label']:35} new_pick={new_pick['label']:35} "
          f"real_best={real_best['label']:35}")
    print(f"        old_regret={old_regret:6.1%} new_regret={new_regret:6.1%} spearman={rho:.3f} "
          f"matches_validated_Model2={match}")

print()
print(f"old Model 0:  mean_regret={mean(old_regrets):.1%} median={median(old_regrets):.1%} "
      f"worst={max(old_regrets):.1%}")
print(f"new production: mean_regret={mean(new_regrets):.1%} median={median(new_regrets):.1%} "
      f"worst={max(new_regrets):.1%}")
print()
if all_match:
    print("EQUIVALENCE CONFIRMED: production estimate_cost reproduces every validated Model 2 pick exactly.")
else:
    print("MISMATCH: production does not reproduce the validated Model 2 picks -- see above. STOP, do not retune.")
    raise SystemExit(1)
