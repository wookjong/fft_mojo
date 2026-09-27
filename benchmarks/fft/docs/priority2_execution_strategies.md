# Priority 2: M2NDP execution strategy improvements (2026-09-26)

All four mechanisms below are SELECTABLE ALTERNATIVES layered onto the
EXISTING execution implementations -- no GPU logical decision (radix
choice, split, `workers_per_fft`, scheme) is ever changed by any of them,
and every existing default/behavior is preserved byte-for-byte unless a
caller explicitly opts into a new mode. Real M2NDP-Detour toolchain
cycle measurements are UNMEASURED throughout (confirmed absent in this
environment -- see docs/transpose_cost_model_audit.md's own toolchain-
availability check); every correctness claim below is proven via the
existing Mojo-to-Python translation harness (`verify_fft_harness.py` and
its per-strategy wrappers), re-executing the ACTUAL emitted code, never a
re-implementation.

## P2.1 -- stage-specific active physical workers

**Finding**: codegen ALREADY avoids unnecessary work for idle physical
lanes. Direct code inspection (not assumption) confirmed both `codegen.
fft_cooperative_codegen._emit_cooperative_stage` and `codegen.fft_
persistent_codegen._emit_worker_dispatch`/`_emit_physical_lane_dispatch`
skip emitting a dispatch branch for any lane whose own batch bucket is
empty (`if not batches: continue`) -- an idle lane touches zero FFT
arithmetic and zero scratchpad addresses for that stage, and this predates
this task entirely. There was therefore no codegen change to make here.

**What was added**: `planning.diagnostics.stage_active_workers` -- a pure
reporting layer (`StageActiveWorkerReport`, `compute_stage_active_worker_
reports`) exposing exactly the requested fields (`active_physical_
workers`, `idle_physical_workers`, `physical_batches_per_lane`, `max_
batches_per_lane`, `avg_batches_per_active_lane`, `imbalance_ratio`) per
stage, for cooperative, persistent, and non-cooperative leaves alike --
built entirely from data `fft_plan_persistent.physical_lane_workload`
(Step 1) and `stage.worker_batches` already compute, never a new formula.

**Files**: `planning/diagnostics/stage_active_workers.py` (new).
**Tests**: `verification/verify_stage_active_workers.py` -- physical
batch counts `<8`, `=8`, `>8`, and not-divisible-by-8, both strategies,
all passing.

## P2.2 -- cooperative batch partition

**Previous behavior**: `fft_plan_cooperative._partition_batches`
implemented ONLY round-robin (`worker w owns {w, w+W, w+2W, ...}`).

**New selectable behavior**: `fft_plan_cooperative.partition_batches(...,
mode=...)` dispatches to `"round_robin"` (default, byte-identical),
`"contiguous"` (fixed chunk `ceil(n/W)`, ragged tail on the last worker),
or `"balanced_contiguous"` (remainder spread one-per-worker across the
FIRST `n % W` workers, contiguous, max size difference of 1 between any
two workers). `CooperationPlan.partition_mode` records which mode built a
given plan (metadata, mirroring `PersistentWorkgroupPlan.lowering_mode`'s
own contract). `generate_partition_mode_candidates` builds the small,
bounded (<=3) candidate set with signature-based dedup (two modes that
produce a physically identical assignment collapse to one candidate).

**Files**: `planning/execution/fft_plan_cooperative.py`, `planning/core/
fft_plan_core.py` (new `CooperationPlan.partition_mode` field).
**Tests**: `verification/verify_fft_cooperative_partition_modes.py` --
structural correctness (union/no-dup/determinism) across 9 (batch_count,
worker_count) pairs including several not divisible by the worker count;
a concrete case (n=17, workers=4) where `contiguous` (`[5,5,5,2]`) and
`balanced_contiguous` (`[5,4,4,4]`) genuinely differ; end-to-end FFT
numeric correctness (all 3 modes bit-for-bit identical output) at
N=2048; dedup count (1 for a small leaf, 2 for a larger one).

## P2.3 -- persistent preload/writeback

**Previous behavior** (confirmed by reading the actual code, not
assumed): `emit_bulk_copy_phase`'s preload/writeback loop is fully
scalar, one element at a time (`.load[width=1]`), each physical lane `p`
striding through `{p, p+8, p+16, ...}` -- exactly the pattern the task
described, verified directly.

**New selectable behavior**: `copy_mode="vectorized_contiguous"` (new,
alongside the unchanged `"scalar"` default) -- each physical lane instead
owns one CONTIGUOUS block `[p*chunk, min(length, (p+1)*chunk))` with
`chunk = ceil(length / workers_per_group)` (a Python/codegen-time
constant), copied `vector_width` (default `4`, this project's own
`target.lmul1_float32_lanes`) elements at a time via the SAME `.load
[width=V]`/inferred-width `.store` idiom already used for FFT-stage
operands elsewhere in this codebase (never a new one), with a scalar
tail loop for the remainder. `PersistentWorkgroupPlan.copy_mode` records
the choice (metadata).

**Correctness**: proven structurally (both modes are full bijections over
`[0, length)`, checked directly in Python) AND numerically (bit-for-bit
identical FFT output between `scalar` and `vectorized_contiguous` on the
same input, at 6 cases including lengths not divisible by `workers_per_
group` or by `vector_width`, and combined with persistent worker-wave
virtualization).

**Files**: `codegen/fft_persistent_codegen.py` (`emit_bulk_copy_phase`,
`emit_persistent_kernel_struct`, `generate_persistent_fft_kernel`),
`planning/core/fft_plan_core.py` (new field), `planning/execution/
fft_plan_persistent.py` (threaded param), `verification/verify_fft_
persistent.py` (`run_persistent_kernel` gained the same param).
**Tests**: `verification/verify_fft_persistent_copy_modes.py`.

## P2.4 -- persistent tail-round hybrid

**Previous behavior**: `num_rounds = ceil(num_logical_blocks /
num_ndp_units)`; the last round is under-utilized whenever `num_logical_
blocks` doesn't divide evenly (e.g. 33 blocks / 32 units: round 1 uses
exactly 1 of 32 groups).

**New selectable behavior**: `fft_plan_persistent.make_persistent_tail_
hybrid_plan(..., tail_strategy=...)` splits `num_logical_blocks` into
`full_blocks` (an exact multiple of `num_ndp_units`) plus `tail_blocks`
(the remainder), building:

* `"all_persistent"` (default, unchanged): the whole count through one
  ordinary persistent leaf -- `PersistentTailHybridPlan.persistent_plan`
  is byte-identical to calling `make_persistent_leaf_plan` directly.
* `"noncoop_tail"`: `full_blocks` via the persistent leaf, `tail_blocks`
  via `fft_plan_core._build_plan` (one physical microthread per replica).
* `"cooperative_tail"`: `tail_blocks` via `fft_plan_cooperative.make_
  cooperative_leaf_plan` instead.

`codegen.fft_persistent_codegen.generate_persistent_tail_hybrid_kernel`
renders TWO fully independent `NDPTask` structs (reusing `emit_
persistent_kernel_struct` and `fft_codegen.emit_kernel` OUTRIGHT -- no new
struct-emission logic), TWO separate DRAM buffer sets, and TWO separate
`.launch()` calls from ONE shared host `main()`. Deliberately NO pointer-
offset arithmetic between the two buffer sets (an unverified Mojo idiom
this project's codegen has never used elsewhere) -- the two kernels'
logical block ranges never need to share one contiguous array, since
neither reads the other's output.

**Correctness**: proven at every task-required boundary (1, 31, 32, 33,
40, 63, 64, 65) -- `full_blocks + tail_blocks == num_logical_blocks`
always; each replica covered exactly once; both sub-plans independently
numerically correct via the existing per-strategy harnesses. The HOST-
LEVEL Mojo text (buffer alloc + two launches in one `main()`) is checked
structurally (well-formed, contains both kernel names/launches) but not
compiled/run -- UNMEASURED, no real toolchain available.

**Files**: `planning/execution/fft_plan_persistent.py` (`TailStrategy`,
`PersistentTailHybridPlan`, `make_persistent_tail_hybrid_plan`), `codegen/
fft_persistent_codegen.py` (`generate_persistent_tail_hybrid_kernel` +
two host-body helpers).
**Tests**: `verification/verify_fft_persistent_tail_hybrid.py`.

## Cost model

Per this task's own "do not broadly refit the cost model" instruction,
NO existing cost-model term was changed for P2. `PRIORITY2_COST_WEIGHTS`
(`fft_cost_model.py`) is the one addition -- `DEFAULT_COST_WEIGHTS` with
`enable_unverified_transpose_tile_term=False`, so any future Priority-2
A/B cost comparison doesn't get contaminated by the still-unverified
transpose-tile coefficient pair (see docs/transpose_cost_model_audit.md).
No new structural metric from this section (active_physical_workers,
physical_batches_per_lane, etc.) has been wired INTO the cost model's own
ranking formula -- they are diagnostic/reporting fields only, exactly the
"say so explicitly" the task's own COST MODEL section asks for rather
than inventing a fitted penalty for any of them.

## A/B experiment harness

`verification/priority2_ab_harness.py` builds, for a FIXED GPU logical
plan shape (length/radices/workers_per_fft held constant across every
experiment), the baseline candidate (Experiment 0) plus one candidate per
mechanism (Experiments 1-4), and reports each candidate's own structural
metrics side by side -- `estimated_cost` (LEGACY weights, for continuity)
and `PRIORITY2_COST_WEIGHTS`-based cost, spill status (`None` --
UNMEASURED, no toolchain), and `ndp_cycles` (`None` -- UNMEASURED). See
that script's own module docstring and the final report's own "D.
Controlled A/B results" section for why every cycle column reads
UNMEASURED rather than a fabricated number, and for the honest structural
(not timing) comparison it CAN make today.
