# GPU logical execution vs. M2NDP physical execution: an explicit boundary (2026-09-23, "Step 2")

## Why

Step 1 (docs/logical_vs_physical_cost_model.md) fixed the COST MODEL's own
confusion between a GPU planner's chosen `workers_per_fft` (LOGICAL) and
M2NDP's real physical lane count (PHYSICAL). This step adds the missing
DESCRIPTIVE layer: a reader (or report) previously had no single place to
see, side by side, "what did the original GPU library intend to execute"
and "how is M2NDP actually executing the same algorithm."

## Design

`planning/gpu_baseline/common.py` gained two reconstruction functions and
one report function, all built AFTER a `BaselineResult` already exists (a
RECONSTRUCTION over the already-built plan tree, not a new field threaded
through all four baseline modules' own recursive builders):

* `GPULogicalKernelPlan` / `reconstruct_gpu_logical_plan(result)` — one
  entry per kernel (FFT or transpose), in execution order, describing the
  GPU planner's own choice in GPU vocabulary (`gpu_workgroup_size`,
  `gpu_threads_per_transform`, `gpu_transforms_per_workgroup`, `gpu_
  shared_memory_bytes`, `gpu_transpose_role`, `gpu_fused_operations`,
  `gpu_global_read_bytes`/`write_bytes`).
* `M2NDPPhysicalKernelExecution` / `reconstruct_m2ndp_physical_plan(result,
  target=...)` — the M2NDP-physical counterpart, in the same order, using
  `"non_cooperative"` | `"cooperative"` | `"persistent"` | `"transpose"`
  as its `strategy`.
* `format_gpu_m2ndp_execution_report(result, target=...)` — renders both,
  per kernel, in two clearly separated sections; never merges a
  corresponding pair of fields into one value.

This is provably faithful for every field that survives onto the plan
tree, because `map_cooperative_kernel`/`_map_worker_wave_kernel` (the one
shared GPU→M2NDP mapping point every baseline funnels through) never
clamp or round a GPU planner's own `workers_per_fft` choice —
`CooperationPlan.workers_per_fft`/`PersistentWorkgroupPlan.workers_per_
fft` on the built plan ARE, verbatim, the GPU planner's own logical
choice. `gpu_scheme`/`pass_id` are read from each baseline's own root
`GPUKernelConfig.extra` (populated by that baseline's own top-level
`plan`/`plan_m2ndp`); `gpu_transpose_role` is derived from this project's
own pre-existing `..Pre{idx}`/`..Mid{idx}`/`..Post{idx}` kernel-naming
convention (every baseline already follows it).

## Known reconstruction limitation (found while writing tests)

`gpu_transforms_per_workgroup` is `None` for any FFT kernel lowered via
M2NDP's persistent/worker-wave-virtualization path — NOT a formatting
omission, a genuine data-loss point: `_map_worker_wave_kernel` discards
the GPU planner's own `fft_slots_wanted`/`num_transforms` choice entirely
before the M2NDP plan is even built (that function's own docstring:
"fft_slots_per_group is architecturally forced to 1 in this regime... a
GPU planner's own fft_slots_wanted > 1 is honored by running those
transforms one after another," i.e. more ROUNDS, never recorded as a
distinct number anywhere on the plan). An earlier version of this
reconstruction reported `1` here — WRONG, because `1` is M2NDP's own
forced PHYSICAL value, not the GPU's LOGICAL intent, exactly the
conflation section 7 of the task this was built from forbids. Caught by
`verify_gpu_execution_semantics.py`'s own cross-checks against each
baseline's raw planning function before this doc was written — see that
test's own docstrings for the concrete before/after numbers (clFFT
N=64: `get_radices` says `num_transforms=4`, but the plan-tree
reconstruction can only ever report `None`, correctly, since the real
answer isn't there to find).

## Verification

* `verify_gpu_execution_semantics.py` — every `GPULogicalKernelPlan` field
  that IS supposed to be recoverable is cross-checked against each
  baseline's OWN raw planning function (`clfft.get_radices`, `rocfft_
  default.SBRR_TABLE`, `vkfft.axisblock_for_leaf`, `rocfft.tune`'s own
  winning `KernelConfig`) — independent verification, not self-consistency
  against the plan tree it was built from.
* `verify_gpu_m2ndp_execution_mapping.py` — semantic invariants: GPU
  logical worker count survives M2NDP lowering unchanged; `workers_per_
  fft > 8` always produces `physical_workers_per_group == 8` (never more);
  every logical batch is executed exactly once; `vkfft.plan()` (frozen)
  and `vkfft.plan_m2ndp()` (adapted) can make different scheme decisions
  from the identical underlying algorithm.
* `verify_gpu_execution_report.py` — report structure (both sections
  present, one `KERNEL` block per reconstructed kernel, GPU section always
  precedes its own M2NDP section, corresponding fields never collapsed
  into one line) across representative cases: clFFT single-pass and
  large-1D, rocFFT tuned, rocFFT-default compiled-table, VkFFT `W<=8` and
  `W>8`, VkFFT M2NDP-adapted N=8192.

All three pass, alongside the full pre-existing regression suite
(unchanged).

## Remaining mismatches (not fixed in this step — descriptive audit only)

* **clFFT-m2ndp's own large-1D split threshold still uses the GPU 8-byte-
  per-complex-element convention, not M2NDP's real 16-byte persistent-leaf
  requirement.** `clfft.get_max_1d_length_m2ndp(target)` calls `get_max_1d_
  length(lds_bytes=target.spad_capacity_bytes)`, which still divides by
  `CLFFT_ELEM_BYTES=8` — the SAME class of bug Step 1 found and fixed for
  `vkfft.plan_m2ndp` (docs/vkfft_m2ndp_scratchpad_resource_model.md), not
  yet propagated to clFFT's own M2NDP-adapted sibling. A length whose
  large-1D split clFFT-m2ndp judges single-kernel-feasible by the 8-byte
  convention could still turn out `RESOURCE_INFEASIBLE` once mapped,
  exactly the "late mismatch" pattern Step 1 closed for VkFFT specifically.
  NOT fixed here per this step's own "do not optimize yet" scope (section
  12) — left as a concrete Priority-2 candidate.
* **rocFFT-default's own equivalent thresholds were not audited to the
  same depth** in this pass (its scheme selection is table/heuristic-
  driven, `map1DLengthSingle`, rather than a single LDS-byte-budget
  formula the way clFFT/VkFFT have) — whether the same class of mismatch
  applies to `CS_L1D_TRTRT`'s own fallback-chain thresholds is an open
  question, not yet answered either way.
* **`gpu_transforms_per_workgroup` is unrecoverable for any persistent-
  path leaf** (see above) — a real information-loss point in the current
  plan representation, not merely a reporting gap. Recovering it would
  require threading the GPU's own `fft_slots_wanted` through `_map_
  worker_wave_kernel` into `PersistentWorkgroupPlan` as new metadata (the
  invasive 4-file-threading alternative this step's design section
  explicitly chose not to do — see this doc's own "Design" section).
* **half_lds (rocFFT)'s effect on shared-memory footprint is not modeled**
  anywhere in this project's own scratchpad-byte accounting
  (`leaf_scratchpad_bytes`/`persistent_leaf_scratchpad_bytes` always
  assume the full two-real+imag-array footprint) — `gpu_shared_memory_
  bytes` on a `GPULogicalKernelPlan` reconstructed from a `half_lds=True`
  rocFFT config therefore over-reports what the real GPU kernel would
  actually use, though the M2NDP-side scratchpad number this project's
  own codegen actually allocates is unaffected (M2NDP has no half_lds
  equivalent to begin with).
* **clFFT's own SBCC and rocFFT-default's own `CS_L1D_CC`** (both real,
  correctly-DECIDED fused block-compute schemes) remain `UNSUPPORTED_
  CURRENT_CODEGEN` — this project's codegen has no fused tile-transpose-
  plus-FFT kernel shape at all, a pre-existing, honestly-reported scope
  gap unrelated to this step's own reconstruction work.
* **Rader's algorithm / Bluestein's algorithm** (VkFFT, and any residual
  prime factor in clFFT/rocFFT beyond their own direct-radix vocabularies)
  remain entirely unimplemented (`UNSUPPORTED_GPU_ALGORITHM`) — a real
  scope limit of this whole repository's codegen, not something this
  step's execution-boundary work touches.
