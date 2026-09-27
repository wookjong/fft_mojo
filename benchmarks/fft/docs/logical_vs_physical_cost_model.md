# Logical workers vs. physical lanes: cost-model correction (2026-09-22)

## Problem

M2NDP's real physical concurrency for a persistent-software-workgroup FFT
leaf is fixed at `target.interleave_chunk_uthreads` (8) physical lanes per
NDP unit, regardless of how wide a GPU planner's own `workers_per_fft`
choice is. `PersistentWorkgroupPlan` has always kept these two concepts as
separate fields (`workers_per_fft`, the LOGICAL cooperation width a GPU
planner chose, vs. `workers_per_group`, the PHYSICAL lane count) — but
`planning.search.fft_cost_model.compute_stage_metrics` read a persistent
stage's own `max_batches_per_worker` straight off `stage.
persistent_vector_batches`, a partition across `workers_per_fft` LOGICAL
buckets, not the `workers_per_group` (8) PHYSICAL buckets `codegen.
fft_persistent_codegen`'s default "physical" lowering (`Mode C`,
`_flatten_to_physical_lanes`) actually renders.

For `workers_per_fft <= workers_per_group` (every plan the ordinary
planner/search path builds — `make_persistent_leaf_plan`'s own default
never exceeds 8 there) this was a no-op: each logical bucket already maps
1:1 onto a physical lane. It only mattered for a GPU-baseline-derived
persistent plan (`gpu_baseline.common._map_worker_wave_kernel`), where a
real GPU planner's own `threads_per_transform`/`workers_per_fft` choice
routinely exceeds 8. There, the old metric kept shrinking as
`workers_per_fft` grew — 64, 128, ... — implying unbounded speedup past a
hardware wall that doesn't move.

## Fix

* `planning.execution.fft_plan_persistent.flatten_logical_workers_to_
  physical_lanes` / `physical_lane_workload`: the flatten logic moved out
  of `codegen.fft_persistent_codegen._flatten_to_physical_lanes` into a
  reusable, planning-layer pure function. Codegen's own function is now a
  thin wrapper delegating to it — byte-identical output, confirmed by
  `verification/verify_fft_persistent.py`'s existing wave/fused/physical
  cross-mode equivalence sweep and the new `verify_fft_physical_lane_
  mapping.py`.
* `PersistentWorkgroupPlan.lowering_mode` (new field, default `"physical"`
  — matches codegen's own default): metadata only, lets a cost-model
  reader know which of `wave`/`fused`/`physical` a leaf is INTENDED to
  render as. `logical_workers_per_fft`/`physical_workers_per_group`/
  `logical_worker_groups`/`stage_launches` properties expose the same
  vocabulary the task's own section 1 asked for, without renaming the
  existing `workers_per_fft`/`workers_per_group` fields.
* `fft_cost_model.StageExecutionMetrics` gained `physical_lanes`,
  `physical_max_batches_per_lane`, `persistent_lowering_mode`.
  `PlanMetrics` gained `physical_total_worker_stage_batches`/
  `physical_persistent_worker_stage_batches` — computed via the shared
  flatten function for persistent stages, identical to the old logical
  fields for every non-persistent stage and every persistent stage with
  `workers_per_fft <= workers_per_group`. The OLD fields (`total_worker_
  stage_batches`, `persistent_worker_stage_batches`, `StageExecutionMetrics.
  max_batches_per_worker`) are UNCHANGED — no rename, no removed field.
* `_execution_cost` now sums the physical fields instead of the logical
  ones — the one behavior change, and only observable for a persistent
  plan whose `workers_per_fft` exceeds 8.

## Verification

`verification/verify_fft_physical_lane_mapping.py`: coverage (no
duplicate/missing batch) and codegen/planning agreement for
`workers_per_fft` in `(1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 18, 20, 24, 32,
36, 64)`.

`verification/verify_fft_physical_lane_cost_model.py`: same physical
workload → identical cost across clean multiples of 8; no `workers_per_fft`
ever beats the balanced W=8 baseline; a ragged `workers_per_fft` (9) costs
more than the balanced neighbor (8), confirming lane imbalance moves cost
up, not down.

Every pre-existing test in the repo (full `verify_fft_plan.py` aggregator,
`verify_fft_search.py`, `verify_fft_mechanism_aware_cost.py`, `verify_fft_
execution_cost.py`, `verify_fft_persistent.py`, `verify_fft_persistent_
search.py`) passed unchanged after this correction, confirming the "no-op
for every existing candidate" claim above.
