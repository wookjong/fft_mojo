# VkFFT M2NDP-adapted scratchpad resource model (2026-09-22)

## Problem

`vkfft.py`'s own shared-memory-sizing formulas (`max_sequence_length_
shared_memory` and its `_strided`/`_pow2` siblings) are a faithful port
of real VkFFT's `usedSharedMemory / complexSize` convention, with
`complexSize` fixed at `VKFFT_COMPLEX_SIZE_BYTES = 8` (one real GPU
shared-memory buffer, single precision). That is the CORRECT resource
input for the frozen, source-faithful baseline (`vkfft.plan()`), since a
real GPU keeps using its own real shared-memory-reuse scheme regardless
of what M2NDP does.

It is the WRONG resource input for `vkfft.plan_m2ndp()` (the M2NDP-adapted
baseline): once that baseline's own chosen `workers_per_fft` doesn't
divide `target.interleave_chunk_uthreads` (the common case for a real
GPU planner's wide `threads_per_transform` choice), the leaf that
configuration maps onto is actually lowered via `gpu_baseline.common.
_map_worker_wave_kernel` → `planning.execution.fft_plan_persistent.
make_persistent_leaf_plan`, which needs `16 * length` bytes
UNCONDITIONALLY (two full ping-pong banks, independent of stage count) —
double the 8-byte-per-element assumption the pass-count/axis-split
decision was made against.

Concretely, at N=8192 against this project's own
`DEFAULT_TARGET_PROFILE.spad_capacity_bytes = 122880`:

* 8-byte convention: `122880 // 8 = 15360 >= 8192` → looks one-pass
  feasible.
* Real M2NDP persistent-leaf requirement: `16 * 8192 = 131072 > 122880` →
  actually infeasible.

Before this fix, `vkfft.plan_m2ndp(8192)` picked the one-pass shape (same
as the frozen baseline) and only discovered the mismatch late, inside
`map_cooperative_kernel`/`_map_worker_wave_kernel`'s own resource check —
an avoidable, expensive "plan, then discover it never fit" round trip.

## Fix

* `planning.execution.fft_plan_persistent.persistent_leaf_scratchpad_
  bytes(length) -> int`: `make_persistent_leaf_plan`'s own inline
  `16 * length` formula, factored out as the ONE source of truth for "how
  many scratchpad bytes does a persistent leaf of this length need."
  `make_persistent_leaf_plan` itself now calls this instead of repeating
  the literal.
* `gpu_baseline.common.m2ndp_resource_complex_bytes() -> int`: returns
  `persistent_leaf_scratchpad_bytes(1) = 16` — bytes per complex element,
  single-sourced from the same formula, not an independently-typed
  literal.
* `vkfft.py`'s sizing chain (`max_sequence_length_shared_memory*`,
  `choose_num_passes`, `split_pow2_*`/`split_non_pow2_*`, `_grouped_
  batch_seed`, `_postprocess_axis_upload0`, `axisblock_batch_multipass_
  first/_later`, `axisblock_for_leaf`, `_leaf_result`, `_recursive_
  result`, `plan`) now all accept an optional `complex_size_bytes`
  parameter, defaulting to `VKFFT_COMPLEX_SIZE_BYTES` (8) everywhere —
  byte-identical behavior for every existing caller, `vkfft.plan()`
  included. `vkfft.plan_m2ndp()` is the ONE caller that overrides it, to
  `m2ndp_resource_complex_bytes()` (16).
* The VkFFT ALGORITHM itself (pass-selection control flow, axis-split
  search, axis-block batching formulas) is completely unchanged — only
  this one resource number differs between `plan()` and `plan_m2ndp()`,
  per the task's own "hardware resource input만 교체, algorithm logic는
  유지" instruction (section 9).

## Verification

`verification/verify_gpu_baseline_vkfft_m2ndp_scratchpad.py`:

* N=8192: 8-byte convention → `num_passes=1` (looks feasible); real M2NDP
  requirement (131072 bytes) exceeds capacity (122880 bytes); 16-byte
  corrected convention → `num_passes=2` (correctly avoids the one-pass
  attempt).
* Frozen `vkfft.plan(8192)` unchanged: still picks one-pass, still
  refuses `RESOURCE_INFEASIBLE` once mapped onto M2NDP (the late mismatch
  is *reachable* from the frozen baseline on purpose — a real GPU
  baseline should not reason about M2NDP's own resource limits).
* `vkfft.plan_m2ndp(8192)` now returns `BaselineStatus.OK` with a
  genuinely correct 2-pass plan (verified against `numpy.fft`,
  `max_err=2.07e-07`).
* `vkfft.plan()`'s own default (`complex_size_bytes` omitted) is
  byte-identical to explicitly passing `complex_size_bytes=8` at several
  representative N — confirms adding the parameter did not change the
  frozen baseline's own default behavior.

The pre-existing `verify_gpu_baseline_source_fidelity.py`,
`verify_gpu_baseline.py`, and `verify_gpu_baseline_m2ndp_adapted.py`
(which exercises `plan_m2ndp` for `clfft`/`rocfft_default`/`vkfft` at
many other N, forward+inverse) all passed unchanged after this fix.
