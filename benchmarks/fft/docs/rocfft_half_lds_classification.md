# rocFFT `half_lds` classification (P1.2, 2026-09-26)

## What `half_lds` means in the source-faithful GPU planner

`half_lds` is a field on rocFFT's own real `KernelConfig` (`function_map_
key.h`). When set, `DeriveMaxTPB` (`derive_max_tpb` in this repo) halves
`bytes_per_batch` (`length * BYTES_PER_ELEM`) before computing how many
transforms-per-block (`tpb`) fit in the GPU's own LDS budget
(`LDS_BYTE_LIMIT=32768`) -- a real GPU LDS-usage optimization that lets
more transforms cooperate in one workgroup for the same LDS budget. It is
swept as a genuine search axis in `_supported_kernel_configs` (`for
half_lds in (True, False)`) and can change which `KernelConfig` wins the
real measured-time tournament (`rocfft.tune`) -- it is not dead metadata.

## Classification: `NO_EQUIVALENT`

M2NDP has no corresponding memory optimization. M2NDP's own scratchpad
model (`gpu_baseline.common.leaf_scratchpad_bytes`, and `fft_plan_
persistent.persistent_leaf_scratchpad_bytes` for the persistent path) has
exactly ONE scratchpad-halving mechanism of its own -- `pingpong_needed`
(1 buffer for a <=2-stage leaf, 2 ping-pong buffers otherwise) -- keyed on
STAGE COUNT, a completely different axis from rocFFT's own PER-BATCH LDS
halving. Neither function takes a `half_lds`-shaped parameter, and
nothing in this repository's codegen implements a mechanism that would
let it. This is therefore not `EXACT_EQUIVALENT` (no matching mechanism),
not `M2NDP_ADAPTATION` (nothing was adapted -- there is nothing to adapt
to), and not `UNSUPPORTED` (it blocks nothing -- a `half_lds=True` config
still maps and runs correctly on M2NDP, it just doesn't get the LDS
saving there that it would on a real GPU).

## What changed

* `rocfft.gpu_logical_lds_bytes(length, *, half_lds)` -- the `DeriveMaxTPB`
  byte formula, factored out of its two inline call sites (`derive_max_
  tpb`, `conservative_max_tpb`, both now call it -- confirmed byte-
  identical behavior by regression test) into a named, reusable function.
* `rocfft._map_config` now stores `gpu_logical_lds_bytes` (the GPU's own
  byte accounting under the winning config's own `half_lds` choice) and
  `m2ndp_half_lds_status="NO_EQUIVALENT"` into `GPUKernelConfig.extra`,
  alongside the pre-existing `half_lds` key.
* `gpu_baseline.common.GPULogicalKernelPlan` gained three new fields:
  `gpu_memory_optimization` (`"half_lds"` or `None`), `gpu_logical_
  shared_memory_bytes` (the GPU's own byte accounting, read back from
  `extra` -- single-sourced, never recomputed), and `m2ndp_memory_
  optimization_status` (`"NO_EQUIVALENT"` or `None`). All three default
  `None` -- backward compatible with every existing reader.
* `format_gpu_m2ndp_execution_report` shows all three, distinctly, right
  after the GPU section's own `gpu_global_write_bytes` line, whenever
  `gpu_memory_optimization is not None` -- e.g. (N=24, a real winning
  config with `half_lds=True`):

  ```
  gpu_memory_optimization:       half_lds
  gpu_logical_shared_memory_bytes: 96
  m2ndp_memory_optimization_status: NO_EQUIVALENT
  ...
  -- M2NDP PHYSICAL EXECUTION --
  ...
  scratchpad_bytes:              384
  ```

  `96` (GPU, half_lds-halved) and `384` (M2NDP, real, unaffected by
  `half_lds`) are shown as two separately-labeled numbers, never merged.

## Resource feasibility

M2NDP resource feasibility (`map_cooperative_kernel`'s own `leaf_
scratchpad_bytes` check, and `_map_worker_wave_kernel`'s own `persistent_
leaf_scratchpad_bytes` check) was ALREADY, and remains, completely
independent of `half_lds` -- neither function has ever taken a `half_lds`
parameter, so there was no feasibility-side bug to fix here (unlike
P1.1's clFFT case). This audit's own job was to make the DIVERGENCE
between the GPU's own (optimistic, half_lds-aware) byte accounting and
M2NDP's real (half_lds-unaware) byte accounting VISIBLE, not to change
which number resource feasibility uses -- it already used, and continues
to use, the real M2NDP one (`scratchpad_bytes` in the report, always the
"Y" the task's own example calls for).

## Verification

`verification/verify_gpu_baseline_rocfft_half_lds.py` -- see that file's
own module docstring for the five checks (formula unit test, refactor
non-regression, M2NDP scratchpad byte-count independence from `half_lds`,
report shows both numbers distinctly for a real `half_lds=True` winning
config at N=24, and a non-rocFFT kernel reports no stale optimization
tag). All pass, alongside the full pre-existing regression suite.
