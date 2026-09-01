# Execution cost model rework: validation (2026-08-30)

Follow-up to `planner_joint_search_validation_task.md`'s own measurement-driven
validation (14 N, 55 real measured candidates): that pass found `_execution_cost`
(via `worst_worker_utilization = min(active_workers/workers_per_fft across
cooperative stages)`) could not distinguish `workers_per_fft=2` from `=4` on
N=216's default radix tier even though real cycles differ by ~35% (30205 vs.
22316) -- every worker slot has >=1 batch in both configs, so `min(...)` is
1.0 either way. This doc records the follow-up: exposing stage-level metrics,
comparing 4 aggregation models against the same 55-candidate dataset, and the
model actually adopted.

## Phase 1/2: stage-level metrics and the reproduced information loss

`planning/fft_cost_model.py` gained `StageExecutionMetrics`/
`compute_stage_metrics(plan)` -- one entry per leaf stage, read straight off
`FFTStagePlan`/`SIMDBatchPlan` (no new constant): `simd_iteration_count`
(total SIMD-width batches for one logical FFT this stage), `max_batches_per_
worker` (`_partition_batches`'s own round-robin busiest-worker batch count --
`simd_iteration_count` itself when not cooperative), `worker_utilization`
(the old metric, kept for direct comparison), `effective_parallelism`
(`simd_iteration_count / max_batches_per_worker`).

N=216, default tier, worker=2 vs. worker=4 (`compute_stage_metrics` output):

| stage | radix | simd_iters | W=2 max/worker | W=2 util | W=2 eff.par | W=4 max/worker | W=4 util | W=4 eff.par |
|---|---|---|---|---|---|---|---|---|
| 0 | 4 | 7 | 4 | 1.0 | 1.75 | 2 | 1.0 | 3.5 |
| 1 | 2 | 14 | 7 | 1.0 | 2.0 | 4 | 1.0 | 3.5 |
| 2 | 3 | 9 | 5 | 1.0 | 1.8 | 3 | 1.0 | 3.0 |
| 3 | 3 | 9 | 5 | 1.0 | 1.8 | 3 | 1.0 | 3.0 |
| 4 | 3 | 9 | 5 | 1.0 | 1.8 | 3 | 1.0 | 3.0 |

`worker_utilization` (the production metric until this change) is **1.0 for
every stage at both worker counts** -- proven, not assumed, why `min()`-based
`worst_worker_utilization` scores W=2 and W=4 identically. `max_batches_per_
worker` sums to 26 (W=2) vs. 15 (W=4) -- real cycles 30205 vs. 22316, same
direction, comparable magnitude of improvement.

N=512/1024 (lopsided multi-leaf, near-leaf big + far-leaf `butterfly_count=1`):
the near-leaf's own `max_batches_per_worker` sum drops from 16 (W=2) to 8
(W=4) -- genuinely and correctly detected. The original claim here --
**"measured total cycles do not move at all" (1187/1195/1195 for N=512;
2111/2111/2111 for N=1024)** -- is **WRONG, corrected 2026-08-31**: those
numbers were extracted via the `tail -1`-on-Gantt-log convention that
`planning/spill_probe.py`'s `_parse_ndp_cycles` fix retired -- it only
ever captured the LAST kernel struct's (POST transpose, identical
regardless of the near-leaf's own worker count) own standalone duration,
not the real end-to-end total. Corrected totals (same split/radix/tile,
`benchmark_fft_candidates.sh` now sums per-struct correctly): N=512
non-cooperative=45814, worker=2=28753 (spill), worker=4=30926 (spill);
N=1024 non-cooperative=49061, worker=2=32054 (spill), worker=4=34226
(spill) -- a real ~30-37% drop, not "no movement." The near-leaf's own
internal speedup was ALWAYS on the critical path; the memory-cost
argument below was explaining an artifact of the measurement bug, not a
real DRAM-bound property of these plans -- see `docs/
active_ndp_units_cost_task.md`'s Phase 1.5/2 writeup for the full root
cause and `docs/persistent_recursive_split.md`'s own correction section
for the parallel fix to that doc's "parity" claim. Left the paragraph
below unedited for the historical record of what this doc originally
argued from the wrong number; do not trust its "DRAM-bound, not
leaf-compute-bound" conclusion.

## Phase 4: leaf/plan-level aggregation semantics

`flatten_recursive_node`'s own docstring/ordering (PRE -> near_fft -> MIDDLE
-> far_child -> POST) and `FFTCodegenPlan`'s own docstring ("chained through
DRAM ... never through a shared scratchpad, since nothing is guaranteed still
resident once a kernel launch returns") both describe kernels as separate,
sequential launches -- not overlapping. The *existing* `_memory_cost` term
already assumes this (`estimated_dram_bytes = len(stages) * plan.n * ...`,
an unconditional sum across every stage in the flattened list) -- summing
stage/leaf execution time the same way is not a new assumption, it is the
same one the memory-cost model already relies on.

## FLAGGED SUSPECT 2026-08-31, HIGHEST PRIORITY, NOT YET RE-VERIFIED

This section's 55-real-candidate dataset is the empirical basis for
**adopting Model D (`total_worker_stage_batches`) as the actual
production `_execution_cost`** -- the single most consequential
measurement in this whole doc. At least one of its own 14 N is confirmed
split (`N=960/split=30`, named explicitly in the "Policy-legal-only"
paragraph below), and any split N's real cycle measurement almost
certainly used the `tail -1`-on-Gantt-log convention `planning.spill_
probe._parse_ndp_cycles`'s 2026-08-31 fix retired (see this doc's own two
corrections above and `docs/active_ndp_units_cost_task.md`'s Phase 1.5
writeup for the confirmed root cause and magnitude: real totals for a
split N run ~19-23x higher than what `tail -1` reported, and -- more
importantly for a *ranking* comparison -- the old number was strategy-
blind for the leaf, always just the last kernel struct's own duration).
No reproducer script for this exact 55-candidate sweep survives in the
repo to rerun directly (checked: no matching Python source under
`benchmarks/fft/`), so this cannot be cheaply re-verified the way the
other corrections in this doc were. **Recommended before trusting this
comparison for anything new**: re-run a fresh version of this sweep
(`benchmark_fft_candidates.sh`, now fixed) across a representative set of
N including several confirmed-split ones, and check whether Model D still
wins -- not done this pass. The qualitative argument for D over A/B/C
(an unnormalized sum sees what a `[0,1]` utilization fraction structurally
cannot) does not depend on the exact numbers here and likely still holds,
but the specific accuracy percentages (78.3%, 71.4%, 57.6%, etc.) should
not be quoted as calibrated until this is redone.

## Phase 3/6: model comparison (55 real measured candidates, rebuilt +
re-scored with each model)

Four candidate aggregation formulas, all reading only `compute_stage_metrics`
output (no fitted constant):

- **A (current/rejected)**: `idle_worker_penalty * (1 - min(worker_utilization))`
- **B (rejected)**: `idle_worker_penalty * (1 - mean(worker_utilization))`
- **C (rejected)**: `idle_worker_penalty * (1 - work-weighted mean(worker_utilization))`
- **D (adopted)**: `stage_work * sum(max_batches_per_worker across every stage)`
  -- reuses `CostWeights.stage_work` (already-existing per-stage rate), not a
  new constant.

| model | mean Spearman (n cases) | top-1 exact | top-3 oracle | top-5 oracle | mean top-1 regret | worker-pair acc. | radix-pair acc. | joint-pair acc. |
|---|---|---|---|---|---|---|---|---|
| A current | -0.150 (8) | 4/14 | 12/14 | 13/14 | 0.376 | 65.2% (30/46) | 64.3% (9/14) | 48.5% (16/33) |
| B simple avg | -0.214 (8) | 4/14 | 12/14 | 13/14 | 0.376 | 63.0% (29/46) | 64.3% (9/14) | 42.4% (14/33) |
| C work-weighted | -0.214 (8) | 4/14 | 12/14 | 13/14 | 0.376 | 63.0% (29/46) | 64.3% (9/14) | 42.4% (14/33) |
| **D stage-time (adopted)** | **+0.460 (13)** | **10/14** | 13/14 | **14/14** | **0.013** | **78.3% (36/46)** | **71.4% (10/14)** | **57.6% (19/33)** |

B/C (utilization-fraction averaging) do not beat A -- both are *worse* on
every metric. This matches the theoretical argument in Phase 3's own task
spec: any `[0,1]`-normalized-by-`workers_per_fft` utilization fraction
structurally cannot see the *absolute* benefit of more workers (a stage at
"87.5% of its own max" looks the same whether that max is 2x or 4x), so
averaging the same blind metric differently was never going to fix it. Only
D, which sums the *unnormalized* `max_batches_per_worker`, captures the
quantity that actually shrinks when `workers_per_fft` goes up.

Policy-legal-only (spill-free candidates, N/A where <2 exist -- 9 of 14 N
had 0-1 spill-free candidates and cannot support a ranking comparison at
all, consistent with [[fft-spill-hard-filter]]'s own caution about trusting
a spill-included ranking): of the 4 N with >=2
spill-free candidates, A and D tie on 3 (N=30, N=144, N=960/split=30) and D
strictly wins on N=256 (A picks the non-cooperative candidate at 38836
cycles, 73.7% worse than the spill-free-and-faster worker=2 option at 22362;
D picks correctly, 0% regret).

## Phase 8: ResourceCost diagnosis only (no change made)

Out of scope for this pass per the task's own instruction. Not touched:
`_resource_cost`'s `radix_risk_penalty`/`spill_penalty` terms and their
values are unchanged (see `verify_fft_execution_cost.check_spill_policy_
unaffected`). Left for a dedicated follow-up.

## Adopted change

`planning/fft_cost_model.py`:
- Added `StageExecutionMetrics` + `compute_stage_metrics(plan)`.
- Added `PlanMetrics.total_worker_stage_batches` (kept `worst_worker_
  utilization` as a diagnostic-only field, no longer read by `_execution_
  cost`).
- `_execution_cost` now returns `weights.stage_work * metrics.total_worker_
  stage_batches`.
- Removed `CostWeights.idle_worker_penalty` (nothing reads it anymore).

`planning/fft_plan_search.py`: `format_plan_summary` prints `total_worker_
stage_batches` alongside the existing (still-kept) `worst_worker_utilization`
line.

`verification/verify_fft_execution_cost.py` (new, wired into `verify_fft_
plan.py`'s own `main()`): 8 checks -- stage metrics match plan structure,
effective_parallelism is monotonic in worker count on a real regression
case, a big leaf's own signal survives a tiny leaf sharing the same plan,
determinism, non-cooperative baseline behavior preserved exactly, candidate
*count* is proven independent of which cost function ranks them (an
adversarial monkeypatched cost function), and the spill/resource-cost policy
is untouched.

Full existing suite (`python3 -m verification.verify_fft_plan`) passes
unchanged after this edit.

## Follow-up: persistent execution as a search axis (2026-08-30)

The compute_lanes spill-avoidance investigation (see docs/
compute_lanes_spill_avoidance.md) surfaced a third execution model this
project had already implemented (`planning.fft_plan_persistent`,
`docs/persistent_leaf_design.md`) but never wired into `generate_
candidates` at all -- every candidate up to this point was either plain
or cooperative-worker. Real measurement (N=216, single fused leaf,
radices=(4,2,3,3,3), batch=1, apples-to-apples with every cooperative/
non-cooperative candidate already in the search): persistent is
spill-free and correct at the shipped default `compute_lanes=4` (23944
cycles) where the *same* radix's cooperative worker=2/4 candidates both
spill outright at that width -- and it beats every other confirmed-safe
candidate for this N, including the best rescued cooperative one (28784
cycles) and the plain baseline (43771 cycles).

**Added, step 10 of `generate_candidates` (`planning/fft_plan_search.py`)**:
- `generate_persistent_leaf_candidates(n, target, inverse, batch)`: one
  candidate per radix tier (`generate_radix_tiers`, same tiers the
  cooperative joint search offers), `num_logical_blocks=batch` -- `[]`
  (not an exception) when `16*n > target.spad_capacity_bytes` or this N
  needs `make_recursive_transpose_plan`'s own recursion (persistent only
  builds a single un-split leaf today, see `docs/persistent_leaf_design.md`).
- `_wrap_persistent_leaf_as_recursive_plan`: wraps `make_persistent_leaf_
  plan`'s own `FFTCodegenPlan` return as an ordinary single-leaf
  `RecursiveFFTPlan` (`FFTLeafPlan(kernel=...)` + a synthesized
  `MultiKernelHostPlan`) -- every generic reader (`flatten_recursive_node`,
  `compute_stage_metrics`/`estimate_metrics`, `_plan_signature`) works on
  it with zero special-casing; only the two places this project's own
  codegen/toolchain genuinely differs by execution model (rendering,
  probing) need to branch.
- `PlanChoices.execution_strategy: str | None`: `"persistent"` marks these
  candidates (`None` otherwise, unchanged).
- `_plan_signature` now folds `stage.persistent` (plus `total_uthreads`/
  `max_uthread`, since `num_logical_blocks` isn't otherwise encoded) into
  its own per-leaf signature tuple -- without this, a persistent candidate
  and the plain non-cooperative baseline of the *same* radix would
  collide (both show `cooperation=None`) and global dedup would silently
  drop one.

**`planning/fft_cost_model.py`**: `compute_stage_metrics` gained a third
branch reading `stage.persistent_vector_batches`/`persistent_scalar_
batches` (a persistent stage never sets `worker_batches` -- a separate,
non-overlapping partition, see `FFTStagePlan`'s own docstring -- so
without this every persistent stage was silently misread as single-
worker-serial, wildly overstating its own `total_worker_stage_batches`).
Verified directly against the same N=216 case: `total_worker_stage_
batches` = 9 (persistent) vs. 15 (cooperative worker=4) vs. 48 (plain
baseline) -- same relative order as the real measured cycles, with zero
new weight (the existing `stage_work` rate already does the job once the
batch count is read correctly).

**`planning/spill_probe.py`**: new `_probe_plan` dispatcher, used by
`probe_and_rerank_candidates` -- routes a candidate to `probe_persistent_
kernel_spill_free` when `plan.root.kernel.persistent is not None`, `probe_
spill_free` otherwise. Necessary, not cosmetic: `probe_spill_free`'s own
`generate_recursive_fft_kernels` has no idea `FFTCodegenPlan.persistent`
exists and would render a persistent leaf as though it were an ordinary
one (reading `stage.batches`/`worker_batches`, both irrelevant to
`persistent_vector_batches`/`persistent_scalar_batches`) -- wrong kernel
body, not just a wrong cost. Also resolves `compute_lanes=None` the same
way `probe_spill_free` itself does before dispatching (`probe_persistent_
kernel_spill_free`'s own default only applies when the argument is
omitted, not when `None` is passed through explicitly).

**Verified end to end on the real toolchain** (not just structurally):
`probe_and_rerank_candidates(rank_candidates(generate_candidates(216))[:4],
top_k=2, rank_by_cycles=True)` correctly probes all four top-ranked
candidates, confirms the persistent one spill-free at 23944 cycles
(ranked first), excludes the two spilling cooperative candidates per
existing [[fft-spill-hard-filter]] policy, and keeps the plain baseline
(43771 cycles) as the second safe option -- no code path change was
needed in that function beyond the dispatcher itself.

`verification/verify_fft_persistent_search.py` (new, wired into `verify_
fft_plan.py`'s own `main()`): 5 checks -- candidate generated for a
feasible N, infeasible N returns `[]` without crashing, signature is
distinct from the same-radix non-cooperative baseline, cost reflects the
real register-pressure difference, and the probe-dispatch condition holds
(the real-hardware round trip itself was run manually, not re-run inside
this fast check -- see this section's own paragraph above). Full existing
suite passes unchanged.

Not done this pass (deliberately, matching the scope this was raised
under): persistent is not yet crossed with worker/tile/split the way
radix x worker already is (only one `num_logical_blocks` value --
`batch` -- and one radix per tier is offered per N, no independent sweep
of either); no cost-model reweighting was needed or added, since the
existing `total_worker_stage_batches`/`stage_work` machinery already
ranked persistent correctly once it could see persistent's own batch
structure at all.
