# GPU-planner-on-M2NDP: hardware-parameter mapping

## Purpose

The frozen `gpu-baseline-v1` reference (`gpu-clfft`, `gpu-rocfft-default`,
`gpu-vkfft`, `gpu-rocfft-tuned`) answers "what plan would a representative
real GPU's own library choose, and what happens if we force that exact GPU
choice onto M2NDP?" — every hardware-resource *input* that library's
planning algorithm consults is a fixed, representative-GPU constant
(clFFT's 32KiB LDS, rocFFT-default's 120 gfx908 CUs, VkFFT's 32-thread
warp), documented and held IDENTICAL across every length/batch that
baseline ever plans (see each module's own "FIXED IMPLEMENTATION
PARAMETERS" docstring section).

This document covers the **M2NDP-adapted** baselines added alongside it
(`gpu-clfft-m2ndp`, `gpu-rocfft-default-m2ndp`, `gpu-vkfft-m2ndp`), which
answer a different question: "what happens if we port the GPU library's
*planning algorithm itself* onto M2NDP, giving it M2NDP's own resource
characteristics instead of a representative GPU's?" These are the PRIMARY
baseline for GPU-vs-M2NDP-native comparison going forward — the frozen
`gpu-baseline-v1` names are kept, unmodified, as a reference/debugging
baseline (see each `*_m2ndp` function's own docstring for the exact,
minimal diff against its frozen sibling).

**The one rule every row below obeys**: the GPU library's own planning
CONTROL FLOW, TABLE, and DECOMPOSITION ALGORITHM never changes. Only
hardware-resource *inputs* to that unchanged algorithm are replaced, and
only where a real M2NDP quantity answers the *same question* the GPU
constant did. Where no such quantity exists, the GPU constant is kept
(classified `FIXED_ALGORITHM_PARAMETER` below) — never silently invented.

## Classification legend

- **EXACT_M2NDP_EQUIVALENT** — the M2NDP quantity answers literally the
  same physical question, at the same level of abstraction.
- **SEMANTIC_ADAPTER** — the M2NDP quantity plays the same *role* in an
  unchanged formula, but the underlying execution models differ enough
  (GPU workgroup vs. M2NDP NDP-unit/uthread) that claiming exact identity
  would overclaim.
- **FIXED_ALGORITHM_PARAMETER** — not a hardware query at all (a
  compiled table, a precision-derived constant, or a value M2NDP's own
  execution model makes moot) — kept unchanged, with justification.
- **NO_EQUIVALENT** — M2NDP has no analogous concept whatsoever; the
  value is kept only because the algorithm needs *some* value to run,
  never because it is meaningful on this target.

## Mapping table

| Library | Upstream hardware input | Old (source-faithful) value | M2NDP value | Semantic mapping | Classification | Affects |
|---|---|---|---|---|---|---|
| clFFT | `CLFFT_LDS_BYTES` (`envelope.limit_LocalMemSize`) | 32768 (32 KiB, typical real-GPU LDS) | `target.spad_capacity_bytes` = 122880 | Both answer "how many bytes of on-chip fast memory does one execution group get to hold an entire transform resident in" — feeds the SAME `GetMax1DLengthStockham = floor_po2(bytes / elem_size)` formula, unchanged, just fed a different budget. Kept consistent with the SAME quantity `map_cooperative_kernel`'s own feasibility check already uses for this baseline (not `max_concurrent_scratchpad_bytes`, which is a caller-facing contention throttle, a different concept). | SEMANTIC_ADAPTER | `plan()`'s single-kernel-vs-large-1D split decision (`is_1d_possible`'s own `large1d_threshold`) |
| clFFT | `CLFFT_MAX_WGS` (`CL_DEVICE_MAX_WORK_GROUP_SIZE`) | 256 | **unchanged** (256) | Gates which rows of the hand-tuned `SPEC_TABLE` specialization table are even consulted, and bounds `DetermineSizes`'s own fallback sizing. M2NDP's own cooperative-worker model has NO analogous hard ceiling on `workers_per_fft` at all — worker-wave virtualization already represents ANY positive cooperation width as `ceil(W/8)` sequential waves of 8 physical lanes (see `fft_plan_persistent.py`'s own docstring), so there is no smaller M2NDP number that would mean "this target's own workgroup-size limit" rather than simply crippling which specialization rows apply. Lowering this value would not adapt the algorithm to M2NDP's hardware — M2NDP's hardware imposes no such limit at all. | FIXED_ALGORITHM_PARAMETER | Which `SPEC_TABLE`/`DetermineSizes` branch `get_radices` takes |
| clFFT | `CLFFT_BLOCK_COMPUTE_GATE_SINGLE` (262144, `= 262144/PrecisionWidth`) | 262144 | unchanged | A precision-derived constant baked into clFFT's own SBCC eligibility table size, not a hardware query at all. | FIXED_ALGORITHM_PARAMETER | SBCC eligibility gate (`is_block_compute_length`) |
| rocFFT-default | `ROCFFT_DEFAULT_MULTIPROCESSOR_COUNT` (gfx908 CU count) | 120 | `target.num_ndp_units` = 32 | Both answer "how many independent physical compute units exist to keep busy" — feeds the SAME `total_batch // transforms_per_block >= multiprocessor_count` occupancy formula in `Decide1DScheme`, unchanged. Only fires at high batch (`total_batch // transforms_per_block` must reach the threshold) — at this project's usual `batch=1` the branch is never taken regardless of which count is used; the mapping's effect is batch-size-dependent, confirmed directly (`decide_scheme(4704, batch=40)`: 120-threshold picks `CS_L1D_CC`, 32-threshold picks `CS_KERNEL_STOCKHAM`). | SEMANTIC_ADAPTER | The single-kernel-vs-multi-kernel occupancy branch inside `Decide1DScheme`, batch>~32 only |
| rocFFT-default | `apply_solution`'s gfx908 solution-map file | fixed, real shipped file | unchanged | Not a hardware-resource input at all — a compiled tuning-database file for one specific real GPU arch, and proven (module docstring, exhaustive scan) to match zero configurations in this baseline's own FP32/out-of-place domain either way. Nothing to adapt. | FIXED_ALGORITHM_PARAMETER | No observable effect in this baseline's domain |
| VkFFT | `VKFFT_WARP_SIZE` (`Structs.h`'s "threads per warp/wavefront") | 32 | `target.interleave_chunk_uthreads` = 8 | Both answer "how many execution lanes run in the target's own natural lockstep/interleave granularity" — feeds the SAME `AxisBlockSplitter.h` `aimThreads`/`warpSize` seed formula (`axisblock_batch_single_pass`), unchanged. Confirmed to change the raw seed directly (exhaustive check, `threads_per_transform=1..299`); the final plan can still converge to the same answer for a specific length once `_postprocess_axis_upload0`'s own downstream reshaping runs — that is real algorithm behavior, not evidence the mapping is inert. | SEMANTIC_ADAPTER | `axisblock_batch_single_pass`'s own occupancy-estimate seed (num_passes==1 leaves only) |
| VkFFT | `VKFFT_AIM_THREADS` ("aim at this many threads per block. Default 128") | 128 | unchanged | VkFFT's own hand-tuned cross-GPU occupancy target, not a hardware capability query — the algorithm picks this as a design constant, it does not ask the device for it. | FIXED_ALGORITHM_PARAMETER | Same occupancy-estimate seed, alongside `warp_size` |
| VkFFT | `VKFFT_VENDOR_IS_NVIDIA` (register-count halving-loop vendor branch) | `True` | unchanged | M2NDP has no vendor-ID concept at all, and this branch selects among ALGORITHMIC register-counting formula variants, never asks "how much hardware exists." Kept exactly as the source-faithful baseline's own justification (a real, representative choice) — there is no M2NDP-specific reason to prefer a different branch, and inventing one would not be adapting to a real M2NDP resource. | FIXED_ALGORITHM_PARAMETER | `registers_per_thread_base_table`'s own vendor halving loop |
| VkFFT | `fixMaxRaderPrimeMult=89` / `fixMinRaderPrimeFFT=17` (Rader-vs-Bluestein residual classification constants) | 89 / 17 | unchanged | Vendor/precision-tied algorithm tuning constants from VkFFT's own real source, not hardware queries; Rader/Bluestein are out of this task's scope regardless (section 9 of the task this doc was built from). | FIXED_ALGORITHM_PARAMETER | Residual-length classification only, unreached in this baseline's own supported domain |
| VkFFT | `target.spad_capacity_bytes` (LDS-equivalent for `max_sequence_length_shared_memory*`/`choose_num_passes`/`split_*`) | already M2NDP-derived | unchanged (already correct) | Pre-existing target adaptation, confirmed consistent across every one of these functions during this audit — no fix needed. | EXACT_M2NDP_EQUIVALENT (pre-existing) | 1/2/3-pass decision, axis-split factor choice |

## Why `batch=1` mostly hides the rocFFT-default mapping

`ROCFFT_DEFAULT_MULTIPROCESSOR_COUNT` only gates a `total_batch //
transforms_per_block >= multiprocessor_count` comparison — at `batch=1`
(this project's own default, isolating plan quality from replica count)
the left side is always `<= 1`, so neither 120 nor 32 as the right side
changes the outcome. This mapping's real effect only shows up in a
batch-swept experiment, not the standard single-batch 27-N sweep. Recorded
here rather than silently omitted, per this task's own "no equivalent /
batch-size-dependent" honesty requirement.

## Verification

`verification/verify_gpu_baseline_m2ndp_adapted.py` — for each mapped
parameter, confirms (a) the M2NDP baseline's decisions differ from the
source-faithful baseline's ONLY in the documented direction/branch, never
elsewhere, and (b) every reachable `plan_m2ndp` result is numerically
correct against `numpy.fft`. `verification/verify_gpu_baseline_source_
fidelity.py`/`verify_gpu_baseline.py`/`verify_gpu_baseline_golden.py` (the
pre-existing suites covering the frozen `plan()` entry points) all still
pass unmodified — confirming this work never touched the frozen baseline's
own behavior.
