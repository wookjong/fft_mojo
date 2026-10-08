# VkFFT multi-pass scheduler correction — 2026-10-07

Resolves finding 2 in `gpu_baseline_review_2026-10-07.md` for both
`vkfft.plan()` and `vkfft.plan_m2ndp()`.

## Source and domain

Verified against DTolm/VkFFT commit
`066a17c17068c0f11c9298d848c2976c71fad1c1`:

- [Scheduler.h](https://github.com/DTolm/VkFFT/blob/066a17c17068c0f11c9298d848c2976c71fad1c1/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h): used shared memory, pass estimate, two/three-pass split, and reordering.
- [Plan_FFT.h](https://github.com/DTolm/VkFFT/blob/066a17c17068c0f11c9298d848c2976c71fad1c1/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h): `stageStartSize = product(axisSplit[:axis_upload_id])`.
- [RunApp.h](https://github.com/DTolm/VkFFT/blob/066a17c17068c0f11c9298d848c2976c71fad1c1/vkFFT/vkFFT/vkFFT_AppManagement/vkFFT_RunApp.h): forward and inverse C2C upload execution order.

The baseline retains its existing domain: one-dimensional C2C,
`reorderFourStep=true`, no convolution/Bluestein, register boost 1,
bandwidth boost disabled, representative NVIDIA vendor. Alternative
vendor branching is exercised in the regression tests. Support for
non-default reorder/Bluestein configurations is not added by this change.
Earlier source extractions did not record a SHA; the newly pinned revision
establishes provenance for this correction, not for their historical snapshot.

## Implementation

- Scheduler power-of-two axes now use `sharedMemorySizePow2`, derived as
  the largest power of two no greater than the target scratchpad budget.
  Non-power-of-two axes retain the raw budget. AxisBlock keeps its separate
  raw/power-of-two bounds as before.
- The reordered two-pass branch uses the strided limit and caps its
  power-of-eight preference at 512. Both factors in divisor search obey
  the strided limit.
- NVIDIA inputs above 262144 use divisor search even for power-of-two
  lengths. The dedicated three-pass power-of-two branch uses the source's
  128-byte first-stage coalescing rule; it no longer invokes the two-pass
  splitter twice.
- Three-pass divisor search preserves upstream factor order and checks
  the third factor too. Failed two-pass search promotes to three passes.
- Before lowering, factors must be positive, divide exactly, multiply
  back to N, fit each upload's bound, and remain powers of two for a
  power-of-two input.
- Upload IDs decrease from `numPasses-1` to 0 for both forward and inverse.
  Upload 0 now receives its AxisBlock branch on the last executed FFT.
  `stageStartSize` uses only lower upload-index factors; it no longer
  includes higher upload factors at the middle stage of a three-pass plan.
- Original factor order, execution IDs, and per-leaf settings are stored
  in `gpu_config.extra`. This metadata addition supports this correction;
  it does not resolve the separate execution-report reconstruction finding.

## Corrected examples

Default scratchpad budget 122880 bytes, power-of-two scheduler budget
65536 bytes, coalesced memory 32 bytes, register boost 1:

| N | Previous split | Corrected split |
| --- | --- | --- |
| 65536 | `(1024,64)` | `(512,128)` |
| 131072 | `(2048,64)` | `(512,256)` |
| 262144 | `(4096,64)` | `(512,512)` |
| 16777216, 8-byte elements | `(15360,64,17)` | `(256,256,256)` |
| 16777216, 16-byte elements | `(7680,64,34)` | `(256,256,256)` |

The N=2^24 plans now build successfully on the default M2NDP profile and
contain only power-of-two leaf FFTs. Other sizes can still legitimately
fail the subsequent M2NDP resource check; scheduling feasibility and
physical mapping feasibility remain distinct.

## Verification and reproduction

Run from `benchmarks/fft`:

```
python -m verification.verify_gpu_baseline_vkfft_multipass
```

Seven regression tests cover the reported splits, vendor threshold,
two-to-three-pass fallback, invalid splits refused before lowering,
upload numbering and stageStartSize, batches 1 and 3, code generation,
and numerical forward/inverse output against NumPy. Numerical cases are
N=16384 (source planner), N=8192 and 10000 (adapted planner), plus a real
three-upload N=8192 plan under a smaller scratchpad profile.

`verification/fixtures/vkfft_multipass_upstream.json` contains 122 decisions
captured from the original C++ split/reordering blocks. These are host
scheduler outputs, not measurements on a GPU. The capture covers three
memory budgets, 8/16-byte elements, NVIDIA/non-NVIDIA branches, power-of-two
and mixed/prime lengths, and unsupported sizes. All match the Python port.
The capture generator imports no baseline planner and checks the input
source against its pinned SHA-256 before compiling it:

```
python -m verification.generate_vkfft_multipass_fixture --scheduler /path/to/vkFFT_Scheduler.h
```

A host C++ compiler (`g++` by default) is required only for regeneration;
the regression suite consumes the committed JSON fixture directly.
Existing baseline, golden, scratchpad, source-fidelity, and execution-semantics
checks also pass. Numerical tests execute the Python stage interpreter;
compiled Mojo/M2NDP output validation and performance comparisons belong
to the later report findings and were not run for this correction.
