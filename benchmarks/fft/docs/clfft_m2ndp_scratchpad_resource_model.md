# clFFT M2NDP-adapted scratchpad resource model (P1.1, 2026-09-26)

## Root cause

`clfft.get_max_1d_length_m2ndp(target)` -- the single-kernel-vs-large-1D
split DECISION threshold for the M2NDP-adapted baseline (`clfft.
plan_m2ndp`/`plan_large1d_m2ndp`) -- called `get_max_1d_length(lds_bytes=
target.spad_capacity_bytes)`, which divided by `CLFFT_ELEM_BYTES=8` (real
clFFT's own single-buffer LDS convention). Same class of bug already
found and fixed for VkFFT (docs/vkfft_m2ndp_scratchpad_resource_model.md):
the M2NDP leaf this threshold selects is, for almost every length, lowered
via the persistent/worker-wave-virtualization path (`gpu_baseline.common.
_map_worker_wave_kernel` -> `make_persistent_leaf_plan`), which needs 16
bytes per element UNCONDITIONALLY -- double the 8-byte assumption the
split decision was made against.

Concretely, at N=8192 against `DEFAULT_TARGET_PROFILE.spad_capacity_bytes
=122880` (the SAME headline N the VkFFT fix used, deliberately, for direct
comparability):

* 8-byte convention: `floor_po2(122880 // 8) = 8192` -> `is_1d_possible
  (8192, 8192)` is True -> looks single-kernel-feasible.
* Real M2NDP requirement: `16 * 8192 = 131072 > 122880` -> actually
  infeasible.

Before this fix, `clfft.plan_m2ndp(8192)` picked the single-kernel shape
and only discovered the mismatch late, inside `map_cooperative_kernel`/
`_map_worker_wave_kernel`'s own resource check (`RESOURCE_INFEASIBLE`) --
confirmed directly (not assumed) before fixing it.

## Fix

* `clfft.get_max_1d_length` gained an `elem_bytes: int = CLFFT_ELEM_BYTES`
  parameter -- default preserves every existing caller's exact behavior
  (the frozen `plan()` never passes it).
* `clfft.get_max_1d_length_m2ndp` now passes `elem_bytes=gpu_baseline.
  common.m2ndp_resource_complex_bytes()` (16) -- the SAME resource-adapter
  function `vkfft.plan_m2ndp` already uses, single-sourced from
  `fft_plan_persistent.persistent_leaf_scratchpad_bytes`.
* The clFFT ALGORITHM itself (`is_1d_possible`, `choose_large1d_split`,
  the BitScanF bit-balancing formula, the 490-entry non-po2 table) is
  completely unchanged -- only this one resource number differs between
  `plan()` (8, frozen) and `plan_m2ndp()`/`plan_large1d_m2ndp()` (16,
  M2NDP-adapted).

## Verification

`verification/verify_gpu_baseline_clfft_m2ndp_scratchpad.py`:

* N=8192: 8-byte convention -> threshold=8192 (looks feasible); real
  M2NDP requirement (131072 bytes) exceeds capacity (122880 bytes);
  16-byte-corrected convention -> threshold=4096 (correctly routes into
  the large-1D split up front).
* Frozen `clfft.plan(8192)` unchanged: its own threshold still uses
  `CLFFT_LDS_BYTES`/`CLFFT_ELEM_BYTES=8` unconditionally, independent of
  `target`.
* `clfft.plan_m2ndp(8192)` now returns `BaselineStatus.OK` with a
  genuinely correct large-1D-decomposed plan (verified against
  `numpy.fft`, `max_err=1.78e-07`).
* `clfft.get_max_1d_length()`'s own default (`elem_bytes` omitted) is
  byte-identical to explicitly passing `elem_bytes=CLFFT_ELEM_BYTES=8` at
  several representative lengths -- confirms adding the parameter did not
  change the frozen baseline's default behavior.

Pre-existing `verify_gpu_baseline.py`, `verify_gpu_baseline_source_
fidelity.py`, and `verify_gpu_baseline_m2ndp_adapted.py` all passed
unchanged after this fix.
