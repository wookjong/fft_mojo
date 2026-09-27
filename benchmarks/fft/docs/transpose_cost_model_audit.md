# Transpose cost-model coefficient audit (2026-09-22)

## Audit result

`fft_cost_model.py`'s two transpose-tile-related weights
(`CostWeights.transpose_tile_count = -2.0`, `.tile_oversaturation_penalty
= 10.0`) were ALREADY marked "FLAGGED SUSPECT, NOT YET RE-VERIFIED" in
their own comments before this task — both cite real M2NDP cycle numbers
(e.g. "N=1024: 192 tiles -> 2111 cycles") that this project's own
`_parse_ndp_cycles` fix (2026-08-31) later found were very likely
measured with the retired `tail -1`-on-Gantt-log convention: that
convention only ever captured the LAST kernel struct's own duration, not
the true end-to-end total. The exact N=1024 figure both weights cite is
confirmed wrong: the corrected total for that shape is 49061 cycles, not
2111. No re-sweep against the corrected parsing pipeline has been done
since.

`transpose_tail_risk_penalty` (20.0) is a SEPARATE, independently
real-hardware-confirmed signal (a tail-tile shape correlating with a
register-pressure liability, not this same cycle-count-fitting exercise)
and is NOT in scope for this audit.

## What this task does NOT do

Per the task's own section 14 instruction: no synthetic coefficient was
fabricated to replace the suspect pair, and neither weight's own
documented "suspect" status was silently upgraded to "confirmed." The
real Mojo → llc → M2NDP-Detour toolchain is not present in the
environment this work was done in (`<repo>/toolchain/bin/mojo(.real)`
does not exist; `third_party/m2ndp-detour` is present as unbuilt C++
source only) — confirmed directly, not assumed, before deciding not to
attempt a real measurement.

## What this task adds

* `CostWeights.enable_unverified_transpose_tile_term` (new field, default
  `True` — byte-identical planner behavior/regression results to every
  commit before this flag existed). `_memory_cost` now gates BOTH
  `transpose_tile_count` and `tile_oversaturation_penalty` behind this one
  flag (they're the same suspect-coefficient pair, always toggled
  together). Set `False` to rank candidates without trusting this pair
  while a real re-sweep is in progress — does not touch
  `transpose_tail_risk_penalty` or any other term.
* `revalidation/measure_transpose_tile_sweep.py`: generates the
  measurement half of a re-calibration sweep — fixed N/split/radix/
  execution-strategy/compute_lanes, varying only `tile_rows`/`tile_cols`
  (`None` i.e. today's shipped default, `1x1`, `2x2`, `4x4`), for N in
  `(630, 960, 1024)` (the same two N the suspect weights' own comments
  cite, plus one already used by `revalidation/revalidate_cost_model.py`
  for cross-script consistency). Records `tile_count`, `tail_tile_count`,
  `active_ndp_units` (via `planning.diagnostics.fft_unit_utilization.
  compute_unit_utilization`), `bytes_read`/`bytes_written`, and — critical
  to not repeating the original mistake — BOTH a per-kernel cycle count
  and a true end-to-end total in the SAME row (`_parse_ndp_cycles_by_
  task`, a local, independent re-derivation of `spill_probe._parse_ndp_
  cycles`'s own regex logic that returns the full per-task breakdown
  instead of only the final sum). Without `--execute` it only dry-runs
  the plan-building/codegen-generation half (confirms every candidate
  actually builds) and prints the exact command to measure for real.
* `revalidation/analyze_transpose_tile_sweep.py`: reads that CSV, splits
  into a calibration/validation set (deterministic, seeded, stratified by
  N), and reports each candidate metric's (`tile_count`, `tail_tile_
  count`, `active_ndp_units`, `bytes_read`) single-variable Pearson
  correlation and linear fit, evaluated on BOTH the calibration set it
  was fit on and the held-out validation set — the goal being "which
  metric explains cycle variation," not "what coefficient best fits every
  point" (the second is exactly how the original suspect coefficient was
  produced, from a handful of points fit with no held-out check).
  Deliberately refuses to run on a CSV with no measured rows (a dry-run
  export) rather than silently analyzing nothing.

Both scripts were exercised in this environment (plan-building dry run
for the measurement script; a clearly-labeled SELF-TEST/fabricated CSV,
never checked into the repo, for the analysis script's own arithmetic) —
neither run constitutes or was used to derive a real coefficient.

## To actually re-calibrate (not run as part of this change)

```
python3 revalidation/measure_transpose_tile_sweep.py --execute \
    --n 630 960 1024 --out revalidation/transpose_tile_sweep.csv
python3 revalidation/analyze_transpose_tile_sweep.py \
    revalidation/transpose_tile_sweep.csv
```

requires a machine with the real Mojo/M2NDP-Detour toolchain built (see
this repo's own `scripts/build.sh`/`scripts/setup.sh`/`scripts/env.sh`).
Once real measurements exist, whether to update `transpose_tile_count`/
`tile_oversaturation_penalty` (and to what magnitude), keep `enable_
unverified_transpose_tile_term=True` regardless, or flip its default to
`False`, is a decision for whoever reviews that real data — deliberately
left undecided here per the task's own "무작정 planner 결과를 대규모로
바꾸지 마라" instruction.

## P1.3 update (2026-09-26): legacy vs. Priority-2 experimental mode

`fft_cost_model.DEFAULT_COST_WEIGHTS` (production, unchanged) is the
LEGACY mode: `enable_unverified_transpose_tile_term=True`, exactly as
above -- every existing regression/candidate ranking keeps computing
byte-identically.

`fft_cost_model.PRIORITY2_COST_WEIGHTS` (new) is the PRIORITY-2
EXPERIMENTAL mode: `replace(DEFAULT_COST_WEIGHTS, enable_unverified_
transpose_tile_term=False)`. Every Priority-2 execution-mechanism A/B
comparison (`verification/priority2_ab_harness.py`, see docs/
priority2_execution_strategies.md) uses THIS constant for its own
`estimate_cost` calls, so a structural-cost comparison between two
M2NDP EXECUTION candidates for the identical GPU logical plan is never
contaminated by the still-unverified transpose-tile coefficient pair --
matters in principle for any future Priority-2 candidate that spans a
transpose boundary (none of the four Priority-2 mechanisms implemented
so far do; P2.1-P2.4 are all single-leaf mechanisms, so in practice
`DEFAULT_COST_WEIGHTS` and `PRIORITY2_COST_WEIGHTS` compute identically
for every Priority-2 candidate built in this task -- the distinction is
there for when a future comparison DOES cross a transpose boundary).

This does NOT claim the transpose cost model is now calibrated, and does
NOT change the production default (`DEFAULT_COST_WEIGHTS` still has
`enable_unverified_transpose_tile_term=True`) -- see the "What this task
does NOT do" section above, unchanged.
