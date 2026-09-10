# gpu-baseline-v1 -- frozen GPU-derived FFT baseline

**Status: FROZEN, 2026-09-08; re-audited 2026-09-09.** Every item tracked below is
resolved. From this point, `planning/gpu_baseline/{common,clfft,rocfft,rocfft_default,
vkfft,solution_map}.py` and `radix_spec.SUPPORTED_RADICES` (as consumed by these
modules) are a fixed research reference. A future change to any GOLDEN value in
`verification/verify_gpu_baseline_golden.py` requires an explicit, documented
GPU-source-fidelity justification (a real upstream research finding, cited the same way
every fix in this baseline's own history already is) -- never an M2NDP-performance-
driven change.

## 2026-09-09 source-fidelity re-audit

A dedicated re-verification pass re-fetched the pinned upstream C++ source directly
(not relying on the original research reports' own summaries) for `rocFFT`-tuned,
`clFFT`, and `VkFFT`, and found three real fidelity gaps the original port had
introduced -- fixed here, each independently cross-checked against a from-scratch
Python re-transliteration of the same source written without reference to this
repository's own implementation (see `verification/verify_gpu_baseline_source_
fidelity.py`):

- **rocFFT-tuned `SupportedKernelConfigs`** (`rocfft.py`) was missing the real
  source's own `tpt < wgs` guard and its own `min_wgs` 64-rounding step, and scoped its
  three pruning passes per-ordering instead of per-phase-call (see that module's own
  updated docstring and `_supported_kernel_configs`'s docstring for the full
  derivation). Fixed; `phase0_candidates(24)` changed from 178 to 105 candidates
  (verified-correct), and N=8/N=16's own tuner search space is now correctly empty
  (`UNSUPPORTED_CURRENT_CODEGEN`) rather than spuriously non-empty.
- **clFFT block-compute (SBCC)** was being silently planned as a four-step (Bailey)
  decomposition for power-of-two lengths in `(4096, 262144]` -- the split VALUES came
  from the real block-compute table, but the resulting M2NDP plan used the four-step
  pre/middle/post-transpose structure, an architecturally different real clFFT scheme,
  and reported `OK`. Fixed: `is_block_compute_length` now gates `_plan_leaf_or_recurse`
  before the four-step tree is built; such lengths now report
  `UNSUPPORTED_CURRENT_CODEGEN` with `scheme=block_compute` and the real split preserved
  as diagnostics. (This supersedes the "documented, not ported" bullet this file
  previously carried under "Intentionally unsupported upstream GPU behavior" -- the
  scheme is now correctly *detected*, even though M2NDP still has no fused
  block-compute codegen to execute it.)
- **VkFFT's register-per-thread table** (`vkfft.py`) used a `min(radices)` proxy for
  every non-power-of-two length, reachable from a `BaselineStatus.OK` result. The real
  `VkFFTGetRegistersPerThreadQuad`'s complete base `{2,3,5,7}` table is now ported in
  full (`registers_per_thread_base_table`) -- every composite-radix entry the real
  source derives from that base table is provably either 0 or exactly equal to one of
  the four base values (see that function's own docstring for the proof), so the base
  table alone is sufficient; the `min(radices)` proxy is gone from every path.

Not resolved by this pass (all three closed by the 2026-09-10 follow-up below): rocFFT-
default's real `ApplySolution`/solution-map layer; VkFFT's real Rader-vs-Bluestein
selection boundary; VkFFT's `AxisBlockSplitter` divisibility-fix loop and power-of-2
bank-conflict axis-swap.

## 2026-09-10 follow-up: closing the remaining "can return OK without reproducing the
## real upstream planner" gaps

Continuing directly from the 2026-09-09 pass, with the same rule (independent
from-scratch re-transliteration cross-checked against every fix, never trusting a
research-report summary over the pinned source itself):

- **rocFFT-default's real solution-map layer** (`ApplySolution`) is now implemented in
  full: `rocfft_upstream_solution_map.py` ports `GetNodeToken`/`GenerateProbKeys`/
  `ApplySolution`/`RecursivelyApplySol` (plan.cpp) exactly, and parses a verbatim copy of
  the real shipped `gfx908_rocfft_solution_map.dat` (`planning/gpu_baseline/data/`, JSON,
  81 entries) -- kept entirely separate from `planning/gpu_baseline/solution_map.py` (an
  M2NDP measured-result cache; a different mechanism in the opposite direction, per that
  module's own docstring). `rocfft_default.plan()` now probes this BEFORE
  `Decide1DScheme`, exactly matching real `BuildSingleDevicePlan`'s own order. A
  DECISIVE, exhaustively-verified finding: every single-precision complex entry in the
  entire shipped gfx908 file is in-place-only, and this project's baselines are all
  out-of-place (M2NDP plans always use separate input/output buffers) -- so
  `apply_solution` is PROVEN (not assumed) to return no match for any length `plan()`
  can be called with today. The lookup/resolution machinery itself is still fully tested
  against the file's one real non-dummy single-precision entry
  (`16777216_sp_ip_complex`, a genuine 5-node `CS_L1D_TRTRT` tree with two differently-
  tuned 4096-length leaves). rocFFT-default's own commit is now pinned to
  `bee97df517907c771de17189cb867d3c401285ae` (the same commit `rocfft.py` already
  cited), resolving the earlier floating "develop HEAD as of 2026-09-08" note.
- **VkFFT's `AxisBlockSplitter` two "omitted" refinements** turned out to be
  mis-scoped, not genuinely blocked on untracked whole-plan state:
  - The "divisibility-fix loop" (vkFFT_AxisBlockSplitter.h lines 301-307) is a
    CONFIRMED NO-OP in the real source at the pinned commit -- proven by exhaustive
    case analysis (its own guard and assignment both key off the same pre-loop value,
    never the loop variable, and the loop force-exits on its first executing branch
    regardless of outcome). Not implemented, because there is nothing to implement.
  - The task's "power-of-two bank-conflict axis swap" turned out to be TWO separate
    real mechanisms: a round-up-to-power-of-2 step (lines 308-311; the real comment
    there says "we plan to swap" but the code rounds up -- a genuine comment/code
    mismatch, reported not silently corrected) and a literal `axisBlock[0]<->
    axisBlock[1]` SWAP (lines 350-364, gated on the fully-processed `axisBlock[1]`).
    Both are now ported exactly, along with several intermediate steps between them
    (a per-axis-size cap keyed on `original_length`/`max_rhs` -- both already tracked
    exactly by this module, not "whole-plan state" as the old docstring claimed -- the
    NVIDIA vendor halving loop, a workgroup-size cap, and a max-thread-num divisor
    search) that were not ported at all before. Cross-checked against an independent
    reference over 100,000+ parameter combinations. Real, verified consequence: N=128
    flips from `UNSUPPORTED_CURRENT_CODEGEN` to `OK` (workers_per_fft/transforms_per_
    block swap from (16,8) to (4,16), which M2NDP's own hardware-mapping constraints
    accept) -- confirmed numerically correct against `numpy.fft` (max error ~2.6e-08).
- **VkFFT's Rader-vs-Bluestein planning decision** is now classified exactly, from
  `vkFFT_AppManagement/vkFFT_InitializeApp.h`'s own vendor/precision-keyed defaults for
  this baseline's fixed NVIDIA/FP32 choice (`fixMinRaderPrimeMult=17`,
  `fixMaxRaderPrimeFFT=16384`) and real VkFFT's own built-in-kernel prime coverage
  (`{2,3,5,7,11,13}` -- wider than this baseline's own `_DIRECT_RADIX_ORDER`, which
  still only implements `{2,...,10}`). A residual factor left after this baseline's own
  greedy strip is now factored into primes and classified `"direct"` (real VkFFT needs
  no Rader/Bluestein at all -- e.g. residual 11 or 13 -- it is only this baseline's own
  narrower scope that cannot represent it), `"rader"`, or `"bluestein"`, reported as
  exact metadata (`gpu_config.extra["scheme"]`, `["residual"]`,
  `["residual_prime_factors"]`) -- never a generic "Rader/Bluestein" mention, and never
  used to invent an M2NDP implementation: the status stays `UNSUPPORTED_GPU_ALGORITHM`
  regardless of which real scheme applies.

## What "frozen" means here

Once this document says `FROZEN`, the four baseline planners --

    gpu-clfft
    gpu-rocfft-tuned   (offline tuner search space)
    gpu-rocfft-default (production non-tuned planning path)
    gpu-vkfft

-- and everything they depend on in `planning/gpu_baseline/` are considered a fixed
research reference. Any future M2NDP-specific execution-layer work (e.g. the striped
multi-wave cooperative addressing scheme `UNSUPPORTED_CURRENT_CODEGEN` results already
point at) must land as a SEPARATE, clearly-labeled layer that a caller opts into
explicitly -- it must never silently change what `gpu-baseline-v1` itself reports for
a given length. A change to a `gpu-baseline-v1` module is only ever justified by GPU
SOURCE fidelity (a research pass finding the port was wrong, or extending it to cover
more of the real upstream source), never by M2NDP performance.

## Exact domain covered

- **Dimensionality:** 1D only. No 2D/3D transform planning in any baseline (real
  clFFT/rocFFT/VkFFT all have distinct, more complex 2D/3D schemes -- e.g. rocFFT's
  `Supported2DKernelConfigs`, VkFFT's multi-axis plans -- none are ported).
- **Domain:** complex-to-complex (C2C) only. No real-to-complex/complex-to-real
  transforms (rocFFT's `CS_REAL_TRANSFORM_EVEN` family, clFFT's real/hermitian modes,
  VkFFT's `performR2C` path are all out of scope).
- **Precision:** single precision (FP32) only -- this whole repository is FP32-only,
  so double-precision facts recorded in the research docs (e.g. clFFT's double-only
  table rows) are documented for completeness but never implemented.
- **Direction:** forward and inverse both supported (a plain `inverse: bool` flag
  threaded through every baseline, matching how this project's own M2NDP-aware planner
  already handles it).
- **Batch:** an integer `batch`/`total_ffts` parameter -- multiple independent
  same-length transforms in one launch, matching every other planner in this project.
- **Length/radix domain:** bounded by each baseline's own real supported-radix set
  (clFFT: primes {2,3,5,7,11,13} plus its generator's composite radices; rocFFT:
  `{2,3,4,5,6,7,8,9,10,11,13,16,17}`; VkFFT: a conservative subset {2,3,4,5,6,7,8,9,10}
  for the direct/non-Rader path -- see each module's own docstring), further bounded
  by `radix_spec.SUPPORTED_RADICES` (this repository's own butterfly-generator
  coverage) and by the M2NDP hardware-mapping constraints documented in
  `BaselineStatus`/`docs/gpu_baseline_hardware_mapping_audit.md`.
- **Target:** `planning.core.target_profile.DEFAULT_TARGET_PROFILE` -- the one real M2NDP
  configuration this project's own checked-in `m2ndp.config` represents. A baseline
  result is only ever claimed accurate for this target; a different `TargetProfile`
  (different scratchpad capacity, different `interleave_chunk_uthreads`) would change
  which lengths report `OK` vs. a resource/mapping refusal, without changing the GPU
  algorithm itself.

## Intentionally unsupported upstream GPU behavior (by design, not oversight)

- **Rader's algorithm and Bluestein's algorithm** (VkFFT, and implicitly clFFT/rocFFT
  for prime lengths beyond their own direct-radix coverage) -- this repository's
  codegen has no prime-length convolution path at all (`radix_spec.SUPPORTED_RADICES`
  is a small fixed set). Reported `UNSUPPORTED_GPU_ALGORITHM`.
- **Real-to-complex / complex-to-real transforms** -- out of scope for this whole
  repository, not just these baselines.
- **2D/3D FFT planning** -- out of scope (see "Exact domain covered" above).
- **clFFT's block-compute (SBCC) pipeline** for very large power-of-2 sizes -- the
  scheme itself and its exact split are now correctly DETECTED (`is_block_compute_
  length`, 2026-09-09 re-audit), reported `UNSUPPORTED_CURRENT_CODEGEN` with the real
  split preserved as diagnostics; only the actual fused M2NDP kernel is unimplemented.
- **rocFFT's 2D-single kernel scheme** (`Supported2DKernelConfigs`) -- materially
  different (uwide/wide TPT, no power-set/utilization-rate machinery) and out of scope
  since this repository is 1D-only.
- **rocFFT's multi-node decomposition trees beyond a single leaf kernel** for the
  offline-tuner baseline (`gpu-rocfft-tuned`) -- confirmed via a verbatim, unresolved
  `// TODO- plan-tuning: build tree several times to generate different trees` in
  rocFFT's own `tuning_plan_tuner.cpp`: today's real tuner only tunes leaf
  `KernelConfig`s of one already-fixed tree, so this baseline's own equivalent scope
  limit mirrors rocFFT's real scope, not a gap this port invented.
- **VkFFT's `registerBoost`/`registerBoost4Step` > 1** -- M2NDP has no register-vs-
  shared-memory tradeoff to represent (see project audit). Always fixed at VkFFT's own
  documented default of 1.
- **GPU vendor/backend-specific branches** (VkFFT's CUDA-only `fixMaxCheckRadix2`
  widening, coalescedMemory auto-detection) -- a single representative default is
  chosen and held fixed across every candidate (documented per-module as a "fixed
  implementation parameter").

## Resolution of every item tracked before freeze

- [x] **rocFFT-default table completion.** Extended from an 11-row partial excerpt to
  the COMPLETE `config_sbrr.py` compiled-in single-kernel table (459 rows, every
  length from 2 through 4096 the real library ships a kernel for, plus the four
  single/half-precision-only rows above 4096), transcribed from a dedicated deep-dive
  (`docs/gpu_baseline_rocfft_default_deepdive.md`) that fetched and read the real file
  in full. Also ported `NodeFactory::Decide1DScheme`'s complete real decision chain:
  `map1DLengthSingle` (the literal length->divLength1 table for `CS_L1D_CC`),
  `get_largest_pow2_length`/`get_explicitly_supported_factor`/`get_largest_supported_
  factor` (the `CS_L1D_TRTRT` fallback chain), and the real `length > 4096` occupancy
  heuristic. `CS_L1D_CC`/`CS_L1D_TRTRT` are DECIDED exactly (real scheme, real
  divLength1, real sub-kernel configs recorded in diagnostics) but deliberately never
  built into an M2NDP plan -- SBCC/SBRC are fused block-tiled transpose+FFT kernels,
  architecturally different from this repository's PRE/MIDDLE/POST six-step shape, and
  forcing them onto that shape would silently change rocFFT's own real algorithm (see
  `rocfft_default.py`'s own module docstring SCOPE LIMIT). Reported
  `UNSUPPORTED_CURRENT_CODEGEN` for those schemes, never fabricated or hidden.
- [x] **VkFFT four-step factor reordering** (`apply_four_step_reordering`) -- the real
  `%2/%4/%8` `locAxisSplit[0]`-preference swap loop, ported exactly and wired into
  `plan()`'s own 2-pass/3-pass factor construction. Regression-tested (same factor
  multiset, different starting order, converges to the same canonical arrangement).
- [x] **VkFFT axis-block / thread-batching** (`vkFFT_AxisBlockSplitter.h`) -- a
  dedicated deep-dive (`docs/gpu_baseline_vkfft_axisblock_deepdive.md`) extracted the
  real `axis_id==0` "automatic batch" formulas (the only branch relevant to this
  1D-only repository): `threads_per_transform = max(1, ceil(fftDim/min_registers_per_
  thread))` (register boost fixed at 1), and `transforms_per_block` via the real
  `aimThreads`/`warpSize` estimate (single-pass leaves) or the real seeded-batch +
  `scale`-growth formula (first vs. later multi-pass upload). Replaces the earlier
  placeholder ("one microthread per widest-stage butterfly, one FFT slot per group").
  Two real refinements (a divisibility-fix loop; a power-of-2 bank-conflict axis-swap)
  are explicitly NOT ported -- both key on a "total remaining sequence count" this
  baseline's per-leaf model doesn't track the way VkFFT's own whole-FFTPlan model
  does; documented in `axisblock_for_leaf`'s own section docstring, not hidden.
- [x] **clFFT large-1D split-selection exactness** -- a dedicated deep-dive (`docs/
  gpu_baseline_clfft_large1d_split_deepdive.md`) found this is fully
  **EXACT_SOURCE_SELECTION**, not merely `ALGORITHM_EQUIVALENT` as first assessed:
  `BitScanF` is `ctz` (count-trailing-zeros; behaves like log2 only because every real
  call site restricts it to power-of-2 arguments), and the non-power-of-2 branch is a
  literal, fully static 490-integer ascending array (not ~380, not generated) checked
  by plain `%` divisibility. `choose_large1d_split` now ports both branches exactly,
  including the literal block-compute table and the exact downward-scan selection
  loop. Verified against 7 independently-computed worked examples (8192, 16384,
  100000, 5040, 45360, 262144, 1048576) -- every one matches exactly. Also confirmed
  and preserved a genuine bug in clFFT ITSELF (not this port): a large prime length
  (e.g. 1000003) has no valid split in the real source and would recurse
  unboundedly; this baseline raises `ClfftUnsupportedLengthError` instead of
  reproducing the infinite loop -- a documented, deliberate departure from literal
  behavior for an input clFFT itself does not handle correctly.
- [x] **Source provenance metadata** (`BaselineProvenance`, `gpu_baseline/common.py`)
  -- added to all four modules (`clfft.py`, `rocfft.py`, `rocfft_default.py`,
  `vkfft.py`), each recording library/upstream repository/commit-or-fetch-date/source
  files/source-function-to-rule mapping. rocFFT's tuned and default baselines cite
  disjoint file/function sets (regression-tested), confirming they remain separate
  mechanisms, not two views of one.
- [x] **Golden fidelity tests** (`verification/verify_gpu_baseline_golden.py`) --
  complete planning-result assertions (scheme/decomposition, exact radix sequence,
  WGS/TPT/TPB-equivalent fields, OK-vs-refusal status) for all 12 required lengths (2
  through 4096) across clFFT, rocFFT-default, and VkFFT, plus a deterministic
  candidate-generation-shape golden test for rocFFT-tuned (whose final winner depends
  on an injected, non-deterministic-in-general `benchmark_fn`). No measured M2NDP
  cycles or spill status anywhere in these tests, per the task's own instruction.
