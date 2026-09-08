"""Mechanism-aware cost-model correction analysis (docs/
active_ndp_units_cost_task.md's "Mechanism-Aware Correction" task,
2026-09-01). Reuses the existing revalidate_cost_model.py/
revalidate_saturation.py datasets -- no new hardware runs. Tests, in
strict Occam's-razor order, whether the smallest possible correction
(Model 1: a persistent-round term) explains the ranking failures Model 0
(current production cost model) has, and only escalates to Model 2
(strategy-specific batch coefficient) if Model 1 is insufficient.

Usage: python3 analyze_mechanism_correction.py
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.execution.fft_plan_persistent import num_rounds
from planning.strategies.fft_plan_recursive import make_recursive_transpose_plan, flatten_recursive_node, FFTLeafPlan, FFTRecursiveNodePlan
from planning.core.target_profile import DEFAULT_TARGET_PROFILE as T

REVALIDATION = "docs/cost_model_revalidation.csv"
SATURATION = "docs/cost_model_saturation.csv"


def leaf_replicas(node) -> list[tuple[int, bool]]:
    """[(replicas, is_persistent), ...] for every leaf in this tree --
    mirrors fft_unit_utilization's own tree walk (r lives on FFTLeafPlan,
    not the flattened FFTCodegenPlan list)."""
    if isinstance(node, FFTLeafPlan):
        return [(node.r, node.kernel.persistent is not None)]
    assert isinstance(node, FFTRecursiveNodePlan)
    return [(node.near_fft.r, node.near_fft.kernel.persistent is not None)] + leaf_replicas(node.far_child)


def persistent_round_increment(spec_n: int, plan_kwargs: dict) -> tuple[int, bool]:
    """(sum(max(0, num_rounds(r, 32) - 1)) over every PERSISTENT leaf,
    any_persistent_leaf_present) -- both 0/False for a candidate with no
    persistent leaf at all; increment stays 0 even for a persistent
    candidate whose every leaf fits in a single round (r <= 32)."""
    plan = make_recursive_transpose_plan(
        spec_n, scratchpad_byte_budget=4096, simd_lanes=8,
        spad_capacity_bytes=T.spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=T.max_concurrent_scratchpad_bytes,
        **plan_kwargs,
    )
    total = 0
    any_persistent = False
    for replicas, is_persistent in leaf_replicas(plan.root):
        if not is_persistent:
            continue
        any_persistent = True
        rounds = num_rounds(replicas, T.num_ndp_units)
        total += max(0, rounds - 1)
    return total, any_persistent


# ---------------------------------------------------------------------
# Phase 3: does persistent_rounds correlate with real cycles better than
# total_worker_stage_batches, using the saturation dataset (the only
# dataset with real round_increment > 0 points for persistent)?
# ---------------------------------------------------------------------

def pearson(xs, ys):
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")


def spearman(xs, ys):
    n = len(xs)
    rx = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: xs[i]))}
    ry = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: ys[i]))}
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n**2 - 1)) if n > 1 else float("nan")


print("=" * 100)
print("PHASE 3: persistent_rounds vs total_worker_stage_batches -- correlation with real cycles")
print("=" * 100)

sat_rows = list(csv.DictReader(open(SATURATION)))
persistent_rows = [r for r in sat_rows if r["strategy"] == "persistent" and r["measured_cycles"] not in ("", "None")]
cycles = [int(float(r["measured_cycles"])) for r in persistent_rows]
batches = [int(float(r["total_worker_stage_batches"])) for r in persistent_rows]
waves = [int(float(r["persistent_waves"])) for r in persistent_rows]

print(f"persistent saturation points: {len(persistent_rows)}")
print(f"  correlation(cycles, total_worker_stage_batches): Pearson={pearson(cycles, batches):.3f} Spearman={spearman(cycles, batches):.3f}")
print(f"  correlation(cycles, persistent_waves):            Pearson={pearson(cycles, waves):.3f} Spearman={spearman(cycles, waves):.3f}")
print("  (total_worker_stage_batches is CONSTANT (3) across this whole sweep by construction --")
print("   it cannot correlate with anything since it never varies; waves is what actually moves)")

# Delta-cycles-per-extra-round, directly from the saturation curve (waves 1->2->4->8)
by_wave = {w: c for w, c in zip(waves, cycles)}
wave_points = sorted(set(waves))
print("\n  Δcycles per Δwave (persistent saturation curve, replicas 32/64/128/256):")
deltas = []
for i in range(1, len(wave_points)):
    w0, w1 = wave_points[i - 1], wave_points[i]
    if w1 - w0 == 0:
        continue
    d = (by_wave[w1] - by_wave[w0]) / (w1 - w0)
    deltas.append(d)
    print(f"    waves {w0}->{w1}: {d:.1f} cycles/extra-wave")
print(f"  median={median(deltas):.1f} mean={mean(deltas):.1f} stddev={pstdev(deltas):.1f} "
      f"min={min(deltas):.1f} max={max(deltas):.1f}")
ROUND_CYCLES = median(deltas)

print("\n" + "=" * 100)
print("PHASE B (cross-check): matched-pair persistent-only cycles-per-batch (single-round cases)")
print("=" * 100)
reval_rows = list(csv.DictReader(open(REVALIDATION)))
for r in reval_rows:
    r["total_worker_stage_batches"] = int(float(r["total_worker_stage_batches"]))
    r["measured_cycles"] = int(float(r["measured_cycles"])) if r["measured_cycles"] not in ("", "None") else None
    r["n"] = int(r["n"])
    r["plan_kwargs"] = eval(r["plan_kwargs"])  # noqa: S307 -- our own repr() output, trusted

by_n = defaultdict(list)
for r in reval_rows:
    by_n[r["n"]].append(r)

persistent_batch_coefs = []
for n, group in sorted(by_n.items()):
    base = next((r for r in group if r["label"] == "noncoop"), None)
    pers = next((r for r in group if r["label"] == "persistent"), None)
    if not base or not pers or base["measured_cycles"] is None or pers["measured_cycles"] is None:
        continue
    d_cycles = pers["measured_cycles"] - base["measured_cycles"]
    d_batches = pers["total_worker_stage_batches"] - base["total_worker_stage_batches"]
    if d_batches:
        coef = d_cycles / d_batches
        persistent_batch_coefs.append(coef)
        print(f"  N={n:5}: cycles_per_batch={coef:.1f}")
PERSISTENT_BATCH_COEF = median(persistent_batch_coefs)
print(f"  median persistent cycles_per_batch = {PERSISTENT_BATCH_COEF:.1f}")

ROUND_IN_BATCH_UNITS = ROUND_CYCLES / PERSISTENT_BATCH_COEF
print(f"\nround_cost_in_batch_equivalent_units = {ROUND_CYCLES:.1f} / {PERSISTENT_BATCH_COEF:.1f} "
      f"= {ROUND_IN_BATCH_UNITS:.2f}")
print("(derived from real data -- NOT a chosen constant: one extra persistent round costs "
      f"as much real cycle time as ~{ROUND_IN_BATCH_UNITS:.1f} extra total_worker_stage_batches "
      "would, at persistent's own already-measured per-batch rate)")


# ---------------------------------------------------------------------
# Phase 4/7: Model 0 vs Model 1 (persistent_rounds correction) vs Model 2
# (strategy-specific coefficient) on the full 24-candidate dataset.
# ---------------------------------------------------------------------

print("\n" + "=" * 100)
print("PHASE 4/7: Model 0 vs Model 1 (persistent_rounds) vs Model 2 (strategy-specific coefficient)")
print("=" * 100)

STAGE_WORK = 0.1  # unchanged, per explicit instruction -- never reweighted again

# Model 2's own strategy-specific coefficients, derived in Phase B of the
# PREVIOUS pass (median, excluding N=630 outlier and worker=8 anomaly):
#   persistent ~343 cycles/batch, cooperative ~620 cycles/batch.
# Expressed as a MULTIPLIER on stage_work (0.1) so Model 0's own
# non-cooperative/cooperative behavior is exactly preserved and only
# persistent's own rate changes: cooperative already implicitly uses
# stage_work=0.1 today (never separately measured as "wrong" -- Phase B's
# whole finding was specifically that PERSISTENT's rate differs).
COOPERATIVE_REFERENCE_COEF = 620.0  # this pass's own matched-pair median for cooperative (Phase B, prior pass)
# Direction check (important -- easy to get backwards): persistent's own
# measured cycles_per_batch (355.2) is LOWER than cooperative's (620.0),
# meaning persistent's own batch count UNDER-represents its real relative
# cost compared to cooperative's -- the same batch-count *reduction* buys
# LESS real speedup for persistent than for cooperative. To make
# persistent's own batches carry a comparably honest weight, they need to
# be scaled UP (multiplier > 1), not down -- the reciprocal of the naive
# ratio, not the ratio itself.
PERSISTENT_MULTIPLIER = COOPERATIVE_REFERENCE_COEF / PERSISTENT_BATCH_COEF
print(f"Model 2's own persistent-specific stage_work multiplier: "
      f"{COOPERATIVE_REFERENCE_COEF:.1f}/{PERSISTENT_BATCH_COEF:.1f} = {PERSISTENT_MULTIPLIER:.3f} "
      f"(i.e. persistent's own batches need {PERSISTENT_MULTIPLIER:.2f}x stage_work per batch to "
      f"carry the same real-cycle weight cooperative's own batches already do)")

for r in reval_rows:
    r["round_increment"], is_persistent_leaf_present = persistent_round_increment(r["n"], r["plan_kwargs"])
    r["model0_cost"] = float(r["estimated_cost"])
    r["model1_cost"] = r["model0_cost"] + STAGE_WORK * ROUND_IN_BATCH_UNITS * r["round_increment"]
    # Model 2: current cost, but with persistent's own batches re-rated at
    # PERSISTENT_MULTIPLIER instead of 1.0x stage_work -- undo the
    # existing 1.0x contribution for persistent leaves and re-add at the
    # measured rate, ALSO including Model 1's own round term (a strategy-
    # specific coefficient does not remove the separate need for a round
    # count -- rounds are a structural launch-count fact, the coefficient
    # is a per-batch rate; both apply).
    if is_persistent_leaf_present:
        r["model2_cost"] = (
            r["model0_cost"]
            - STAGE_WORK * r["total_worker_stage_batches"]
            + STAGE_WORK * PERSISTENT_MULTIPLIER * r["total_worker_stage_batches"]
            + STAGE_WORK * ROUND_IN_BATCH_UNITS * r["round_increment"]
        )
    else:
        r["model2_cost"] = r["model0_cost"]


def evaluate(model_key: str, label: str):
    print(f"\n--- {label} ---")
    worst_regret = 0.0
    worst_n = None
    regrets = []
    for n, group in sorted(by_n.items()):
        measured = [r for r in group if r["measured_cycles"] is not None]
        if len(measured) < 2:
            continue
        model_pick = min(measured, key=lambda r: r[model_key])
        real_best = min(measured, key=lambda r: r["measured_cycles"])
        regret = model_pick["measured_cycles"] / real_best["measured_cycles"] - 1
        regrets.append(regret)
        sorted_by_model = sorted(measured, key=lambda r: r[model_key])
        top3_labels = {r["label"] for r in sorted_by_model[:3]}
        top5_labels = {r["label"] for r in sorted_by_model[:5]}
        xs = [r[model_key] for r in measured]
        ys = [r["measured_cycles"] for r in measured]
        rho = spearman(xs, ys) if len(measured) > 1 else float("nan")
        flag = "  <<<< " if n in (216, 960) else ""
        print(f"{flag}N={n:5}: model_pick={model_pick['label']:35} real_best={real_best['label']:35} "
              f"regret={regret:6.1%} top3_hit={real_best['label'] in top3_labels} "
              f"top5_hit={real_best['label'] in top5_labels} spearman={rho:.3f}")
        if regret > worst_regret:
            worst_regret, worst_n = regret, n
    print(f"  mean_regret={mean(regrets):.1%} median_regret={median(regrets):.1%} "
          f"worst_regret={worst_regret:.1%} (N={worst_n})")


evaluate("model0_cost", "Model 0: current production cost model (unchanged)")
evaluate("model1_cost", "Model 1: Model 0 + persistent_rounds correction")
evaluate("model2_cost", "Model 2: Model 1 + persistent-specific stage coefficient")
