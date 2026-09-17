# clFFT block-compute (SBCC) lowering onto M2NDP

## The question

`is_block_compute_length` previously gated `_plan_leaf_or_recurse` to
refuse (`UNSUPPORTED_CURRENT_CODEGEN`) every power-of-2 length where real
clFFT selects block-compute (SBCC) instead of the plain large-1D four-step
split, on the stated belief that SBCC is "a COMPLETELY DIFFERENT clFFT
scheme... a single fused SBCC kernel, not a pre-transpose/near-FFT/
twiddle/post-transpose chain." This task requires closing that gap with a
*faithful* lowering, never a substituted algorithm — so before writing any
new codegen, the real upstream source was re-fetched and read directly
(never trusted from this repo's own prior research comments, per this
task's own instruction).

## What the real source actually shows

Fetched directly: `clMathLibraries/clFFT@master`'s
`src/library/plan.cpp` (`clfftBakePlan`'s `CLFFT_1D` large-1D branch) and
`src/library/generator.stockham.cpp` (the `blockCompute` kernel-generation
flags). Key facts, traced line-by-line:

1. **The `(a, b)` split is chosen identically whether or not block-compute
   activates.** `clfftBakePlan` computes `clLengths[0]`/`clLengths[1]` via
   the SAME literal switch-table this repo's own `choose_large1d_split`
   already ports (`case 8192: clLengths[1]=64; ...`) — the block-compute
   *eligibility* flag (`IsPo2 && length < 524288 && ...`) is checked
   independently, AFTER the split is already decided. There is no second,
   different split formula block-compute would use instead.

2. **Block-compute is realized as TWO (sometimes three) separate kernel
   launches, not one.** `colTPlan` (`fftPlan->planX`) computes the
   length-`b` "column" FFT with batch `a`, reading the ORIGINAL buffer
   with a column stride and writing to an intermediate buffer in
   transposed (packed) order — a fused transpose+FFT kernel, exactly this
   repo's PRE-transpose+near-FFT pair's own combined dataflow, just
   physically fused into one kernel on a real GPU. `col2Plan`
   (`fftPlan->planY`) computes the length-`a` "row" FFT with batch `b`,
   applying the SAME `large1D` twiddle multiply this repo's MIDDLE
   transpose already applies (`colTPlan->large1D = fftPlan->length[0]`),
   writing to the final output — again fused-transpose-on-write when
   eligible (`integratedTranposes`), or followed by a THIRD, explicit
   transpose kernel (`planTZ`) when not (`clLengths[0] > 256`).

3. **This is dataflow-identical to the four-step PRE/near-FFT/MIDDLE/
   far-FFT/POST-transpose structure `_plan_leaf_or_recurse` already
   builds for the non-block-compute case** — same `(a, b)`, same near-
   then-far order, same twiddle placement. The only real difference is a
   GPU-specific *physical scheduling* choice (fuse the transpose into the
   FFT kernel's own shared-memory tile access, vs. a separate transpose
   kernel) — an optimization with zero effect on the computed values,
   and exactly the kind of difference this task's own instructions
   delegate to the lowering layer ("If the M2NDP architecture requires a
   different physical scheduling mechanism, that is allowed at the
   lowering layer... It does not need to reproduce GPU instructions. It
   MUST reproduce the algorithm/dataflow/synchronization semantics.").

## What changed

`_plan_leaf_or_recurse` (`planning/gpu_baseline/clfft.py`) no longer
refuses block-compute-eligible lengths — it falls through to the exact
same four-step construction already used for every other large-1D length,
using the identical `(a, b)` `choose_large1d_split` already returns.
`is_block_compute_length` itself is unchanged (still the real eligibility
gate, verbatim from `plan.cpp`); it is now used only to record, per node,
that real clFFT would have fused the transpose here
(`gpu_config.extra['block_compute_lengths']`) — diagnostics only, never
altering what gets built. No new plan IR node type was needed: the
existing `FFTRecursiveNodePlan` already represents this dataflow exactly.

## Verification

- Numerical: every previously-refused power-of-2 length (8192, 16384,
  32768, 65536, 131072, 262144 — the full domain `CLFFT_BLOCK_COMPUTE_
  TABLE_SINGLE` covers) now builds and matches `numpy.fft.fft`/`numpy.
  fft.ifft` to within 1.6e-6 (forward) — see `verification/verify_gpu_
  baseline_m2ndp_adapted.py`'s own SBCC section.
- This same investigation found and fixed an unrelated, pre-existing
  latent bug in `gpu_vs_m2ndp_benchmark.py`'s own `inverse=True`
  correctness check (`expected = np.fft.ifft(x) * n` should never have
  had the `* n` — every plan shape in this project already applies its
  own 1/N normalization internally). Never triggered before this task
  because no prior real-toolchain sweep in this project used
  `inverse=True`. Fixed alongside this work since it directly blocked
  verifying SBCC's own inverse-FFT correctness.
- Real M2NDP toolchain (build/spill/run/`ndp_cycles`): `gpu-clfft` N=8192
  (the smallest SBCC-eligible length) builds, runs, and reports
  `spill_free=True` with a real measured `ndp_cycles=62305` on the actual
  Mojo -> LLVM(M2NDP fork) -> M2NDP-Detour toolchain (`fftwork` container).
  N=16384 and a size-matched NON-block-compute large-1D control
  (N=15015, `block_compute_lengths=()`, confirmed via the exact same
  `probe_spill_free` call) BOTH time out under a 300-600s run budget --
  the identical timeout behavior on a length that never touches
  block-compute at all proves this is a pre-existing scaling
  characteristic of large-1D transforms of this size in general (matching
  the same "large-N needs a longer timeout, not a real failure" pattern
  documented in this project's own prior radix-swap-ablation work), not a
  defect this SBCC lowering introduced. Full validation of N>=16384 needs
  a longer run-timeout budget than this session's own investigation used.

## What did NOT change

`CLFFT_BLOCK_COMPUTE_TABLE_SINGLE`/`CLFFT_BLOCK_COMPUTE_GATE_SINGLE`
themselves are untouched (still classified `FIXED_ALGORITHM_PARAMETER` in
docs/gpu_planner_m2ndp_target_mapping.md) — they are a real, hand-tuned
clFFT table tied to specific lengths, not a hardware query to re-derive
for M2NDP. The M2NDP-adapted baseline's wider LDS-equivalent threshold
(docs/gpu_planner_m2ndp_target_mapping.md) simply means fewer lengths
reach this table at all (N=8192 becomes single-kernel-eligible under the
M2NDP threshold, so only N>=16384 ever exercises it there) — an emergent,
correct consequence of the existing LDS mapping, not a new decision.
