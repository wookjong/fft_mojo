# VkFFT `vkFFT_AxisBlockSplitter.h` Deep Dive

Target file: `vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h`
Branch: `master`. Fetched in full (471 lines) via `curl` from
`https://raw.githubusercontent.com/DTolm/VkFFT/master/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h`.

The whole file is exactly one function:

```c
static inline VkFFTResult VkFFTSplitAxisBlock(VkFFTApplication* app, VkFFTPlan* FFTPlan, VkFFTAxis* axis, pfUINT axis_id, pfUINT axis_upload_id, pfUINT allowedSharedMemory, pfUINT allowedSharedMemoryPow2) {
```

It is called exactly once in the whole repo, from `vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h:452`:

```c
VkFFTSplitAxisBlock(app, FFTPlan, axis, axis_id, axis_upload_id, allowedSharedMemory, allowedSharedMemoryPow2);
```

with `allowedSharedMemory`/`allowedSharedMemoryPow2` computed just above it (`vkFFT_Plan_FFT.h:114-125`):

```c
pfUINT allowedSharedMemory = app->configuration.sharedMemorySize;
pfUINT allowedSharedMemoryPow2 = app->configuration.sharedMemorySizePow2;

if (axis->specializationConstants.useRaderMult) {
	allowedSharedMemory -= (axis->specializationConstants.useRaderMult - 1) * axis->specializationConstants.complexSize;
	allowedSharedMemoryPow2 -= (axis->specializationConstants.useRaderMult - 1) * axis->specializationConstants.complexSize;
}
```

**Structural note (important):** the function has two entirely disjoint algorithms selected by a single runtime check at line 41:

```c
if (app->configuration.groupedBatch[axis_id])
{
    ... lines 41-193 ...
    return VKFFT_SUCCESS;
}
```

- If the user has explicitly forced a batch count for this dimension (`configuration.groupedBatch[axis_id] != 0`), the function runs the **"forced batch" branch** (lines 41-193) and returns immediately.
- Otherwise (the default — `groupedBatch[]` defaults to 0, per its doc comment "try to force this many FFTs to be performed by one threadblock for each dimension" at `vkFFT_Structs.h:205`), execution falls through past line 193 to the **"automatic batch" branch** (lines 207-469), which is what virtually all real VkFFT plans execute.

Both branches independently recompute an initial `axis->groupedBatch` from `maxSequenceLengthSharedMemory`/`maxSingleSizeStrided` at the very top of the function (lines 27-39, shared code, executed before the `if` split), then each branch runs its own set of `axisBlock[0..3]` derivations. The two branches are structurally near-duplicates of each other (same formulas, same variable names), but the "automatic" branch (default) has substantially more logic (aimThreads/warpSize-based batch estimation, a divisibility-fix loop, a power-of-2 rounding step for the batch, and stronger bank-conflict/shared-memory rebalancing at the end of the function). All numeric detail below is annotated by branch.

---

## 1. File fetched in full

Confirmed: 471 lines, license header + single header-guarded function `VkFFTSplitAxisBlock`. Full content reproduced (in relevant excerpts) below; nothing was skipped.

---

## 2 & 3. Threads-per-transform, transforms-per-block, and the hardware cap — exact code

### 2.0 Shared top-of-function setup (runs unconditionally, lines 27-39)

```c
pfUINT maxBatchCoalesced = app->configuration.coalescedMemory / axis->specializationConstants.complexSize;
axis->groupedBatch = maxBatchCoalesced;

pfUINT maxSequenceLengthSharedMemory = allowedSharedMemory / axis->specializationConstants.complexSize;
pfUINT maxSequenceLengthSharedMemoryPow2 = allowedSharedMemoryPow2 / axis->specializationConstants.complexSize;
pfUINT maxSingleSizeStrided = (app->configuration.coalescedMemory > axis->specializationConstants.complexSize) ? allowedSharedMemory / (app->configuration.coalescedMemory) : allowedSharedMemory / axis->specializationConstants.complexSize;
pfUINT maxSingleSizeStridedPow2 = (app->configuration.coalescedMemory > axis->specializationConstants.complexSize) ? allowedSharedMemoryPow2 / (app->configuration.coalescedMemory) : allowedSharedMemoryPow2 / axis->specializationConstants.complexSize;
if (((FFTPlan->numAxisUploads[axis_id] == 1) && (axis_id == 0)) || ((axis_id == 0) && (!axis->specializationConstants.reorderFourStep) && (axis_upload_id == 0))) {
	axis->groupedBatch = (maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim.data.i > axis->groupedBatch) ? maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim.data.i : axis->groupedBatch;
}
else {
	axis->groupedBatch = (maxSingleSizeStrided / axis->specializationConstants.fftDim.data.i > 1) ? maxSingleSizeStrided / axis->specializationConstants.fftDim.data.i * axis->groupedBatch : axis->groupedBatch;
}
```

So `axis->groupedBatch` (the batch-count field, VkFFT's own name for "transforms per block") is *seeded* here from either:
- **non-strided, single-upload/first-upload-without-4-step-reorder case:** `maxSequenceLengthSharedMemory / fftDim` (as many whole transforms of length `fftDim` as fit in the *raw* shared-memory budget, in units of `complexSize`), floor-clamped to be at least `maxBatchCoalesced`.
- **all other (strided) cases:** `maxSingleSizeStrided / fftDim`, multiplied back onto `maxBatchCoalesced` — i.e. the batch is quantized in units of `maxBatchCoalesced = coalescedMemory / complexSize` transforms.

`maxBatchCoalesced` (coalesced-memory-derived transform count) is the recurring "quantum" the whole file snaps `groupedBatch` to (see also the final rounding at line 247: `axis->groupedBatch = (axis->groupedBatch / maxBatchCoalesced) * maxBatchCoalesced;`).

### 2.1 "Threads per transform" — VkFFT's actual name is `axisBlock[0]` (axis 0) or `axisBlock[1]` (axis ≥ 1)

VkFFT does **not** use a "threads-per-transform"/TPT vocabulary word anywhere in this file. The concept is expressed purely as one dimension of the 2-D/3-D compute-shader workgroup size stored in `axis->axisBlock[4]`. Which axisBlock index holds "threads that cooperate on one transform" vs "how many independent transforms share the block" **flips** depending on `axis_id`:

- **`axis_id == 0` (non-strided/first axis):** `axisBlock[0]` = threads per transform, `axisBlock[1]` = transforms batched per block (this can later be **swapped**, see §2.4).
- **`axis_id >= 1` (strided axes):** `axisBlock[1]` = threads per transform, `axisBlock[0]` = transforms batched per block.

The core "threads assigned to one transform of this length" formula is **identical in shape** everywhere it appears (forced branch lines 48/100/153; automatic branch lines 266/369/422):

```c
axis->axisBlock[0] = (((pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread)) / axis->specializationConstants.registerBoost > 1)
    ? ((pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread)) / axis->specializationConstants.registerBoost
    : 1;
```

i.e. symbolically:

```
threads_per_transform = max(1, ceil(fftDim / min_registers_per_thread) // registerBoost)
```

(integer division `//` after the `ceil`, since both operands are `pfUINT`). `fftDim` is `axis->specializationConstants.fftDim.data.i` — the **already-decided** per-upload sub-length (i.e. VkFFT's own name for what the prompt calls the "locAxisSplit factor"; see §4). `min_registers_per_thread` and `registerBoost` are consumed, not recomputed, from `vkFFT_Scheduler.h` (§4).

### 2.2 "Transforms batched per block" in the default (automatic) branch, axis_id==0, first upload (lines 292-299)

```c
if (axis->specializationConstants.reorderFourStep && (FFTPlan->numAxisUploads[axis_id] > 1))
	axis->axisBlock[1] = axis->groupedBatch;
else {
	//axis->axisBlock[1] = (axis->axisBlock[0] < app->configuration.warpSize) ? app->configuration.warpSize / axis->axisBlock[0] : 1;
	pfUINT estimate_batch = (((axis->axisBlock[0] / app->configuration.warpSize) == 1) && ((axis->axisBlock[0] / (double)app->configuration.warpSize) < 1.5)) ? app->configuration.aimThreads / app->configuration.warpSize : app->configuration.aimThreads / axis->axisBlock[0];
	if (estimate_batch == 0) estimate_batch = 1;
	axis->axisBlock[1] = ((axis->axisBlock[0] < app->configuration.aimThreads) && ((axis->axisBlock[0] < app->configuration.warpSize) || (axis->specializationConstants.useRader))) ? estimate_batch : 1;
}
```

Symbolically (non-4-step-reorder-multi-upload case, the common path):

```
if 1.0*axisBlock[0] < warpSize*1.5 rounds to exactly 1 warp (axisBlock[0]//warpSize == 1 and axisBlock[0]/warpSize < 1.5):
    estimate_batch = aimThreads // warpSize
else:
    estimate_batch = aimThreads // axisBlock[0]
estimate_batch = max(estimate_batch, 1)

axisBlock[1] = estimate_batch   if axisBlock[0] < aimThreads AND (axisBlock[0] < warpSize OR useRader)
             = 1                otherwise
```

This is VkFFT's real "how many transforms per block" heuristic for the biggest/most-common case: it aims for `aimThreads` total threads per block (`app->configuration.aimThreads`, doc comment "aim at this many threads per block. Default 128", `vkFFT_Structs.h:198`), batching more small transforms together when a single transform's thread count (`axisBlock[0]`) is small relative to a warp (`app->configuration.warpSize`, `vkFFT_Structs.h:294`).

Then a **divisibility-fix loop** (lines 301-307) grows `axisBlock[1]` upward (never past `2*currentAxisBlock1`) so that it evenly divides the number of independent sequences to batch, as long as it still fits shared memory:

```c
pfUINT currentAxisBlock1 = axis->axisBlock[1];
for (pfUINT i = currentAxisBlock1; i < 2 * currentAxisBlock1; i++) {
	if (((FFTPlan->numAxisUploads[0] > 1) && (!(((FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->specializationConstants.fftDim.data.i) % axis->axisBlock[1]) == 0))) || ((FFTPlan->numAxisUploads[0] == 1) && (!(((FFTPlan->actualFFTSizePerAxis[axis_id][1] / r2cmult) % axis->axisBlock[1]) == 0)))) {
		if (i * axis->specializationConstants.fftDim.data.i * axis->specializationConstants.complexSize <= allowedSharedMemory) axis->axisBlock[1] = i;
		i = 2 * currentAxisBlock1;
	}
}
```

Then a **power-of-2 rounding step** for the batch count purely to help the later bank-conflict axis-swap (line 308-311):

```c
if (((axis->specializationConstants.fftDim.data.i % 2 == 0) || (axis->axisBlock[0] < app->configuration.numSharedBanks / 4)) && (!(((!axis->specializationConstants.reorderFourStep) || (axis->specializationConstants.useBluesteinFFT)) && (FFTPlan->numAxisUploads[0] > 1))) && (axis->axisBlock[1] > 1) && (axis->axisBlock[1] * axis->specializationConstants.fftDim.data.i < maxSequenceLengthSharedMemoryPow2) && (!((app->configuration.performZeropadding[0] || app->configuration.performZeropadding[1] || app->configuration.performZeropadding[2])))) {
	//we plan to swap - this reduces bank conflicts
	axis->axisBlock[1] = (pfUINT)pow(2, (pfUINT)pfceil(log2((double)axis->axisBlock[1])));
}
```

Followed by several more clamps (R2C-merge shared-memory overflow check disabling `mergeSequencesR2C`, batch-vs-actual-sequence-count clamps, an **NVIDIA-only halving loop** at 330-335, `maxComputeWorkGroupSize[1]` clamp, and a final "shrink until it fits `maxThreadNum`" search loop at 338-347) before the register-boost-aware shared-memory cap:

```c
while ((axis->axisBlock[1] * (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.registerBoost)) > maxSequenceLengthSharedMemory) axis->axisBlock[1] /= 2;
axis->groupedBatch = axis->axisBlock[1];
```

This is the direct answer to prompt item 5 ("LDS-budget cap"): the batch count is halved (integer `/=2`, i.e. repeated `//2`) until

```
axisBlock[1] * (fftDim / registerBoost) <= maxSequenceLengthSharedMemory
```

where `maxSequenceLengthSharedMemory = allowedSharedMemory / complexSize` (line 30) is literally `sharedMemorySize / complexSize` (minus the Rader adjustment from `vkFFT_Plan_FFT.h:118-119`). Note **`fftDim` here is divided by `registerBoost`**, not by anything derived from `axisBlock[0]` — i.e. the LDS occupied per *thread group* for the transform data is modeled as `(fftDim/registerBoost) * complexSize` bytes per batched sequence (register-boosted elements are assumed to live in registers, not shared memory), exactly matching how the earlier-covered `registerBoost` scaling of `maxSingleSizeNonStrided/Strided` in `vkFFT_Scheduler.h` is defined.

### 2.3 Non-first-upload / axis_id==0 case (automatic branch, lines 369-417) — an `aimThreads`-driven "scale" instead

```c
axis->axisBlock[1] = ((pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread) / axis->specializationConstants.registerBoost > 1) ? (pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread) / axis->specializationConstants.registerBoost : 1;
...
pfUINT scale = app->configuration.aimThreads / axis->axisBlock[1] / axis->groupedBatch;
if ((scale > 1) && ((axis->specializationConstants.fftDim.data.i * axis->groupedBatch * scale <= maxSequenceLengthSharedMemory))) axis->groupedBatch *= scale;

axis->axisBlock[0] = ((pfUINT)axis->specializationConstants.stageStartSize.data.i > axis->groupedBatch) ? axis->groupedBatch : axis->specializationConstants.stageStartSize.data.i;
if (app->configuration.vendorID == 0x10DE) {
	while ((axis->axisBlock[1] * axis->axisBlock[0] >= 2 * app->configuration.aimThreads) && (axis->axisBlock[0] > maxBatchCoalesced)) {
		axis->axisBlock[0] /= 2;
		if (axis->axisBlock[0] < maxBatchCoalesced) axis->axisBlock[0] = maxBatchCoalesced;
	}
}
if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
if (axis->axisBlock[0] * axis->axisBlock[1] > maxThreadNum) {
	for (pfUINT i = 1; i <= axis->axisBlock[0]; i++) {
		if ((axis->axisBlock[0] / i) * axis->axisBlock[1] <= maxThreadNum)
		{
			axis->axisBlock[0] /= i;
			i = axis->axisBlock[0] + 1;
		}
	}
}
axis->axisBlock[2] = 1;
axis->axisBlock[3] = axis->specializationConstants.fftDim.data.i;
axis->groupedBatch = axis->axisBlock[0];
```

Here `scale = aimThreads // (axisBlock[1] * groupedBatch)` is used to *grow* the already-seeded `groupedBatch` (from §2.0) when it and the per-transform thread count together are still under `aimThreads`, gated by fitting in `maxSequenceLengthSharedMemory`. Then `axisBlock[0]` (the batch dimension here, since `axis_id==0` but not the first upload) is capped by `stageStartSize` (a Scheduler-derived stride, see §4) and then goes through the same NVIDIA-only halving loop and a final linear search (`for i in 1..axisBlock[0]`) to find the largest divisor of `axisBlock[0]` that brings `axisBlock[0]*axisBlock[1]` under `maxThreadNum`.

### 2.4 Strided axes, `axis_id >= 1` (automatic branch, lines 420-468 — this is the block quoted fully for item "does the logic differ for STRIDED axes")

```c
if (axis_id >= 1) {

	axis->axisBlock[1] = ((pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread) / axis->specializationConstants.registerBoost > 1) ? ((pfUINT)pfceil(axis->specializationConstants.fftDim.data.i / (double)axis->specializationConstants.min_registers_per_thread)) / axis->specializationConstants.registerBoost : 1;
	if (axis->specializationConstants.useRaderMult) {
		/* ... Rader thread-count search, identical pattern to §2.5 ... */
	}
	if (axis->specializationConstants.useRaderFFT) {
		if (axis->axisBlock[1] < axis->specializationConstants.minRaderFFTThreadNum) axis->axisBlock[1] = axis->specializationConstants.minRaderFFTThreadNum;
	}

	axis->axisBlock[0] = (FFTPlan->actualFFTSizePerAxis[axis_id][0] > axis->groupedBatch) ? axis->groupedBatch : FFTPlan->actualFFTSizePerAxis[axis_id][0];
	if (app->configuration.vendorID == 0x10DE) {
		while ((axis->axisBlock[1] * axis->axisBlock[0] >= 2 * app->configuration.aimThreads) && (axis->axisBlock[0] > maxBatchCoalesced)) {
			axis->axisBlock[0] /= 2;
			if (axis->axisBlock[0] < maxBatchCoalesced) axis->axisBlock[0] = maxBatchCoalesced;
		}
	}
	if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
	if (axis->axisBlock[0] * axis->axisBlock[1] > maxThreadNum) {
		for (pfUINT i = 1; i <= axis->axisBlock[0]; i++) {
			if ((axis->axisBlock[0] / i) * axis->axisBlock[1] <= maxThreadNum)
			{
				axis->axisBlock[0] /= i;
				i = axis->axisBlock[0] + 1;
			}
		}
	}
	axis->axisBlock[2] = 1;
	axis->axisBlock[3] = axis->specializationConstants.fftDim.data.i;
	axis->groupedBatch = axis->axisBlock[0];
}
```

Differences vs the non-strided (`axis_id==0`) path:
- `axisBlock[1]` (threads-per-transform) is computed with the *same* `ceil(fftDim/min_registers_per_thread)/registerBoost` formula, but there is **no** `aimThreads`/`warpSize`-based batch estimation, no divisibility-fix loop, and no power-of-2 rounding of the batch.
- The batch dimension `axisBlock[0]` is simply `min(actualFFTSizePerAxis[axis_id][0], groupedBatch)` — capped by the actual number of remaining sequences to transform in this axis, not by a heuristic estimate.
- The **NVIDIA vendor-ID halving loop** (`vendorID == 0x10DE`) is present here too, applied to `axisBlock[0]` instead of `axisBlock[1]`.
- There is **no register-boost-based shared-memory `while` cap** on `axisBlock[1]` in this branch (unlike axis_id==0/upload==0's line 348) — the shared-memory fit for strided axes is instead enforced earlier/later via `maxSingleSizeStrided`/the "half bandwidth technique" block (§2.6), not inside this per-axis section.
- No axis-swap / bank-conflict pow2-rounding logic at all for strided axes (that logic is specific to the `axis_id==0` sections).

### 2.5 Rader-prime special case (identical pattern repeated 5 times: forced-branch axis0-upload0 lines 49-68, forced-branch axis0-other-upload lines 101-119, forced-branch axis≥1 lines 154-172, automatic-branch axis0-upload0 lines 267-286, automatic-branch axis0-other-upload lines 370-388, automatic-branch axis≥1 lines 423-441)

Quoted once (automatic branch, axis_id==0, upload==0, lines 267-286):

```c
if (axis->specializationConstants.useRaderMult) {
	pfUINT locMaxBatchCoalesced = ((axis_id == 0) && (((axis_upload_id == 0) && ((!app->configuration.reorderFourStep) || (app->useBluesteinFFT[axis_id]))) || (axis->specializationConstants.numAxisUploads == 1))) ? 1 : maxBatchCoalesced;
	pfUINT final_rader_thread_count = 0;
	for (pfUINT i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
		if (axis->specializationConstants.raderContainer[i].type == 1) {
			pfUINT temp_rader = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / (double)((axis->specializationConstants.rader_min_registers / 2) * 2)) / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2));
			pfUINT active_rader = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader);
			if (active_rader > 1) {
				if ((((double)active_rader - (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)temp_rader) >= 0.5) && ((((pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)(active_rader - 1)) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2)) * locMaxBatchCoalesced) <= app->configuration.maxThreadsNum)) active_rader--;
			}
			pfUINT local_estimate_rader_threadnum = (pfUINT)pfceil((axis->specializationConstants.fftDim.data.i / axis->specializationConstants.raderContainer[i].prime) / (double)active_rader) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);

			pfUINT temp_rader_thread_count = ((pfUINT)pfceil(axis->axisBlock[0] / (double)((axis->specializationConstants.raderContainer[i].prime + 1) / 2))) * ((axis->specializationConstants.raderContainer[i].prime + 1) / 2);
			if (temp_rader_thread_count < local_estimate_rader_threadnum) temp_rader_thread_count = local_estimate_rader_threadnum;
			if (temp_rader_thread_count > final_rader_thread_count) final_rader_thread_count = temp_rader_thread_count;
		}
	}
	axis->axisBlock[0] = final_rader_thread_count;
	if (axis->axisBlock[0] * axis->groupedBatch > maxThreadNum) axis->groupedBatch = locMaxBatchCoalesced;
}
if (axis->specializationConstants.useRaderFFT) {
	if (axis->axisBlock[0] < axis->specializationConstants.minRaderFFTThreadNum) axis->axisBlock[0] = axis->specializationConstants.minRaderFFTThreadNum;
}
```

This **overrides** the `min_registers_per_thread`/`registerBoost`-derived thread count computed in §2.1 whenever any Rader-prime factor uses direct-multiplication (`raderContainer[i].type == 1`): it re-derives a thread count per Rader prime (rounding each prime's "active" thread count up to a multiple of `(prime+1)/2`), and takes the max across all such primes as the final `axisBlock[0]`/`axisBlock[1]`. Separately, `useRaderFFT` (a different Rader strategy — FFT-based rather than direct multiplication) instead just enforces a **floor**: `axisBlock[...] = max(axisBlock[...], minRaderFFTThreadNum)`, where `minRaderFFTThreadNum` was pre-computed in `vkFFT_Scheduler.h` via `VkFFTGetRaderFFTThreadsNum`.

### 2.6 Post-loop, pre-final-recompute rebalancing (automatic branch only, lines 210-260) — the LDS/coalesced-quantization pass

This code runs **once**, immediately after the initial `groupedBatch` seed (§2.0) and *before* the big axis-specific `axisBlock[]` recompute (§2.1-2.4) — i.e. it adjusts `axis->groupedBatch` a second time, which then feeds as an input into `axisBlock[0]`'s formulas in §2.3/§2.4 (via `axis->groupedBatch` used at lines 396, 446, 74/127/177/179 in the forced branch too).

```c
if (app->configuration.vendorID == 0x10DE) {
	if (FFTPlan->numAxisUploads[axis_id] == 2) {
		if ((axis_upload_id > 0) || (axis->specializationConstants.fftDim.data.i <= 512)) {
			if ((pfUINT)(axis->specializationConstants.fftDim.data.i * (64 / axis->specializationConstants.complexSize)) <= maxSequenceLengthSharedMemory) {
				axis->groupedBatch = 64 / axis->specializationConstants.complexSize;
				maxBatchCoalesced = 64 / axis->specializationConstants.complexSize;
			}
			if ((pfUINT)(axis->specializationConstants.fftDim.data.i * (128 / axis->specializationConstants.complexSize)) <= maxSequenceLengthSharedMemory) {
				axis->groupedBatch = 128 / axis->specializationConstants.complexSize;
				maxBatchCoalesced = 128 / axis->specializationConstants.complexSize;
			}
		}
	}
	if (FFTPlan->numAxisUploads[axis_id] == 3) {
		if ((pfUINT)(axis->specializationConstants.fftDim.data.i * (64 / axis->specializationConstants.complexSize)) <= maxSequenceLengthSharedMemory) {
			axis->groupedBatch = 64 / axis->specializationConstants.complexSize;
			maxBatchCoalesced = 64 / axis->specializationConstants.complexSize;
		}
		if ((pfUINT)(axis->specializationConstants.fftDim.data.i * (128 / axis->specializationConstants.complexSize)) <= maxSequenceLengthSharedMemory) {
			axis->groupedBatch = 128 / axis->specializationConstants.complexSize;
			maxBatchCoalesced = 128 / axis->specializationConstants.complexSize;
		}
	}
}
else {
	if ((FFTPlan->numAxisUploads[axis_id] == 2) && (axis_upload_id == 0) && (axis->specializationConstants.fftDim.data.i * maxBatchCoalesced <= maxSequenceLengthSharedMemory)) {
		axis->groupedBatch = (pfUINT)pfceil(axis->groupedBatch / 2.0);
	}
	if ((FFTPlan->numAxisUploads[axis_id] == 3) && (axis_upload_id == 0) && ((pfUINT)axis->specializationConstants.fftDim.data.i < maxSequenceLengthSharedMemory / (2 * axis->specializationConstants.complexSize))) {
		axis->groupedBatch = (pfUINT)pfceil(axis->groupedBatch / 2.0);
	}
}
if (axis->groupedBatch < maxBatchCoalesced) axis->groupedBatch = maxBatchCoalesced;
axis->groupedBatch = (axis->groupedBatch / maxBatchCoalesced) * maxBatchCoalesced;
//half bandiwdth technique
if (!((axis_id == 0) && (FFTPlan->numAxisUploads[axis_id] == 1)) && !((axis_id == 0) && (axis_upload_id == 0) && (!axis->specializationConstants.reorderFourStep)) && ((pfUINT)axis->specializationConstants.fftDim.data.i > maxSingleSizeStrided)) {
	axis->groupedBatch = maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim.data.i;
	if (axis->groupedBatch == 0) axis->groupedBatch = 1;
}

if ((app->configuration.halfThreads) && (axis->groupedBatch * axis->specializationConstants.fftDim.data.i * axis->specializationConstants.complexSize >= app->configuration.sharedMemorySize))
	axis->groupedBatch = (pfUINT)pfceil(axis->groupedBatch / 2.0);
if (axis->groupedBatch > app->configuration.warpSize) axis->groupedBatch = (axis->groupedBatch / app->configuration.warpSize) * app->configuration.warpSize;
if (axis->groupedBatch > 2 * maxBatchCoalesced) axis->groupedBatch = (axis->groupedBatch / (2 * maxBatchCoalesced)) * (2 * maxBatchCoalesced);
if (axis->groupedBatch > 4 * maxBatchCoalesced) axis->groupedBatch = (axis->groupedBatch / (4 * maxBatchCoalesced)) * (4 * maxBatchCoalesced);
```

This is the section that directly matches the prompt's hypothesized `min(some_thread_cap, sharedMemorySize/(length*complexSize))` shape: `axis->groupedBatch = maxSequenceLengthSharedMemory / fftDim` under the "half bandwidth technique" condition (fftDim bigger than `maxSingleSizeStrided`, i.e. it doesn't even fit the coalesced-quantized shared-memory budget) with a floor of 1. Otherwise the batch is snapped down to multiples of `warpSize`, then to multiples of `2*maxBatchCoalesced`, then `4*maxBatchCoalesced` — a series of power-of-2-like quantizations for coalescing/bank-conflict reasons, not a single clean formula.

### 3. Hardware max-thread-count constraint — exact quotes

`app->configuration.maxThreadsNum` (doc: "max number of threads from VkPhysicalDeviceLimits", `vkFFT_Structs.h:290`) is read into a local exactly once per branch and used as the universal cap:

Forced branch (line 43): `pfUINT maxThreadNum = app->configuration.maxThreadsNum;`
Automatic branch (line 261, with an abandoned/commented-out alternative formula directly above it):
```c
//pfUINT maxThreadNum = (axis_id) ? (maxSingleSizeStrided * app->configuration.coalescedMemory / axis->specializationConstants.complexSize) / (axis->specializationConstants.min_registers_per_thread * axis->specializationConstants.registerBoost) : maxSequenceLengthSharedMemory / (axis->specializationConstants.min_registers_per_thread * axis->specializationConstants.registerBoost);
//if (maxThreadNum > app->configuration.maxThreadsNum) maxThreadNum = app->configuration.maxThreadsNum;
pfUINT maxThreadNum = app->configuration.maxThreadsNum;
```
(The commented-out lines show VkFFT once considered deriving `maxThreadNum` from shared memory directly, but the shipped code just uses the hardware limit verbatim — worth flagging since it shows the *intended* design rationale even though it's dead code.)

`maxThreadNum` gates every `axisBlock[0]*axisBlock[1] > maxThreadNum` shrink loop (e.g. lines 75-77, 135-144, 183-185, 338-347, 404-413, 454-463) — always via a linear search for the largest integer divisor `i` of the *batch* dimension such that dividing by `i` brings the product at or under `maxThreadNum` (e.g. lines 338-347):

```c
if (axis->axisBlock[0] * axis->axisBlock[1] > maxThreadNum) {
	for (pfUINT i = 1; i <= axis->axisBlock[1]; i++) {
		if ((axis->axisBlock[1] / i) * axis->axisBlock[0] <= maxThreadNum)
		{
			axis->axisBlock[1] /= i;
			i = axis->axisBlock[1] + 1;
		}
	}
}
```

Additionally, `app->configuration.maxComputeWorkGroupSize[0]`/`[1]` (doc: "maxComputeWorkGroupCount from VkPhysicalDeviceLimits" — comment is a copy/paste of the `maxComputeWorkGroupCount` doc but the field is actually work-group **size**, `vkFFT_Structs.h:289`) independently clamps each axisBlock dimension (e.g. lines 72-73, 134, 182, 291, 336, 403, 453) *before* the `maxThreadNum` product check.

---

## 4. Upstream `vkFFT_Scheduler.h` values consumed, and their struct definitions

`VkFFTSplitAxisBlock` reads (never writes) these `axis->specializationConstants` fields, all of which are set by `vkFFT_Scheduler.h`'s per-upload loop before `VkFFTSplitAxisBlock` is ever called:

| Field read in AxisBlockSplitter | Set in Scheduler.h at | Assignment |
|---|---|---|
| `specializationConstants.fftDim.data.i` | `vkFFT_Scheduler.h:3208` | `axes[k].specializationConstants.fftDim.data.i = locAxisSplit[k];` — this **is** the already-decided per-upload axis-split factor the prompt refers to. |
| `specializationConstants.registerBoost` | `vkFFT_Scheduler.h:3268` | `axes[k].specializationConstants.registerBoost = registerBoost;` |
| `specializationConstants.min_registers_per_thread` | `vkFFT_Scheduler.h:3270` (and forced to 2 at `3273` if the computed `registers_per_thread` came back 0) | `axes[k].specializationConstants.min_registers_per_thread = min_registers_per_thread;` |
| `specializationConstants.minRaderFFTThreadNum` | `vkFFT_Scheduler.h:3266` | `res = VkFFTGetRaderFFTThreadsNum(...)` |
| `specializationConstants.rader_min_registers`, `.raderRegisters`, `.numRaderPrimes`, `.raderContainer[]`, `.useRaderMult`, `.useRaderFFT` | `vkFFT_Scheduler.h` (Rader classification block, lines ~3190-3266) | Rader-prime bookkeeping consumed as-is by §2.5. |
| `specializationConstants.stageStartSize.data.i` | `vkFFT_Plan_FFT.h:127-130` (not Scheduler.h itself, but likewise computed *before* the axis-block call, from `FFTPlan->axisSplit[axis_id][i]`) | `axis->specializationConstants.stageStartSize.data.i *= FFTPlan->axisSplit[axis_id][i];` for `i` in `0..axis_upload_id` — i.e. the product of the *coarser* upload's split sizes, used in §2.3 as `axisBlock[0]`'s pre-vendor-loop cap. |
| `FFTPlan->numAxisUploads[axis_id]`, `FFTPlan->actualFFTSizePerAxis[axis_id][...]`, `FFTPlan->axisSplit[axis_id][...]` | Scheduler.h (the pow2/non-pow2 axis-split search covered by the earlier pass) | Read throughout, e.g. lines 34, 177, 213/226/238/242, 303, 312, 329, 446. |

**`VkFFTSplitAxisBlock` itself does NOT re-derive a batch count from the register-boosted size independently** — it consumes `registerBoost` directly as a scalar divisor inside its own thread-count formula (`ceil(fftDim/min_registers_per_thread)/registerBoost`, §2.1) and inside its own shared-memory cap (`axisBlock[1] * (fftDim/registerBoost) > maxSequenceLengthSharedMemory`, §2.2). It does not call back into any Scheduler.h function, and it does not itself recompute `maxSingleSizeNonStrided`/`maxSingleSizeStrided` with register-boost scaling the way Scheduler.h's own axis-split search does (Scheduler.h's `maxSingleSizeStrided`/`maxSingleSizeNonStrided *= registerBoost` step, covered in the prior pass, is a **separate, earlier** computation used only to decide the split itself — AxisBlockSplitter.h recomputes its own **un-boosted** `maxSingleSizeStrided` locally at line 32-33, and only re-introduces `registerBoost` later as a plain division in the two spots above). This answers prompt item 2's `registerBoost` question directly: it feeds in as a **divisor of the thread/LDS formulas**, not as a multiplier of a re-derived max-size.

### Struct definitions for every field referenced above

`VkFFTAxis` (`vkFFT_Structs.h:1037-1116`, relevant fields only):
```c
typedef struct {
	char data[128];
	...
	pfUINT numBindings;
	pfUINT axisBlock[4];
	pfUINT groupedBatch;
	VkFFTSpecializationConstantsLayout specializationConstants;
	VkFFTPushConstantsLayout pushConstants;
	...
} VkFFTAxis;
```

`VkFFTPlan` (`vkFFT_Structs.h:1118-1130`):
```c
typedef struct {
	pfUINT actualFFTSizePerAxis[VKFFT_MAX_FFT_DIMENSIONS][VKFFT_MAX_FFT_DIMENSIONS];
	pfUINT numAxisUploads[VKFFT_MAX_FFT_DIMENSIONS];
	pfUINT axisSplit[VKFFT_MAX_FFT_DIMENSIONS][4];
	VkFFTAxis axes[VKFFT_MAX_FFT_DIMENSIONS][4];

	pfUINT bigSequenceEvenR2C;
	pfUINT actualPerformR2CPerAxis[VKFFT_MAX_FFT_DIMENSIONS];
	VkFFTAxis R2Cdecomposition;
	VkFFTAxis inverseBluesteinAxes[VKFFT_MAX_FFT_DIMENSIONS][4];
} VkFFTPlan;
```

`VkFFTSpecializationConstantsLayout` (`vkFFT_Structs.h:719-1014`, only the fields consumed by AxisBlockSplitter):
```c
typedef struct {
	...
	PfContainer fftDim;                    // :727 — per-upload sub-length, "locAxisSplit factor"
	int numAxisUploads;                    // :736
	int registers_per_thread;              // :737
	int min_registers_per_thread;          // :739
	...
	PfContainer stageStartSize;            // :778
	int reorderFourStep;                   // :787
	int complexSize;                       // :810
	...
	int registerBoost;                     // :821
	int warpSize;                          // :822
	int numSharedBanks;                    // :823
	...
	int axisSwapped;                       // :833
	int mergeSequencesR2C;                 // :835
	...
	int useRader;                          // :844
	int numRaderPrimes;                    // :845
	int minRaderFFTThreadNum;              // :846
	VkFFTRaderContainer* raderContainer;   // :847
	...
	int useRaderMult;                      // :851
	...
	int rader_min_registers;               // :865
	int useRaderFFT;                       // :867
	...
} VkFFTSpecializationConstantsLayout;
```

`VkFFTConfiguration` (`vkFFT_Structs.h`, relevant excerpts, exact line numbers cited inline):
```c
pfUINT coalescedMemory;   // :197 — bytes; 32 on Nvidia/AMD, 64 on Intel (scaled for half precision)
pfUINT aimThreads;        // :198 — "aim at this many threads per block. Default 128"
pfUINT numSharedBanks;    // :199 — "how many banks shared memory has. Default 32"
pfUINT groupedBatch[VKFFT_MAX_FFT_DIMENSIONS]; // :205 — "try to force this many FFTs to be performed by one threadblock for each dimension" (0 = auto/off, the default)
pfUINT registerBoost;         // :277
pfUINT registerBoostNonPow2;  // :278
pfUINT registerBoost4Step;    // :279
pfUINT maxComputeWorkGroupSize[VKFFT_MAX_FFT_DIMENSIONS]; // :289
pfUINT maxThreadsNum;         // :290 — "max number of threads from VkPhysicalDeviceLimits"
pfUINT sharedMemorySizeStatic;// :291
pfUINT sharedMemorySize;      // :292 — "available for allocation shared memory size, in bytes"
pfUINT sharedMemorySizePow2;  // :293
pfUINT warpSize;              // :294 — "number of threads per warp/wavefront"
pfUINT halfThreads;           // :295 — "Intel fix"
pfUINT reorderFourStep;       // :297 — default 1
pfUINT vendorID;              // :302 — "vendorID 0x10DE - NVIDIA, 0x8086 - Intel, 0x1002 - AMD, etc."
```

---

## 5. Worked examples

All three examples assume: `sharedMemorySize = 49152` bytes (48 KiB), `complexSize = 8` (single-precision complex), `coalescedMemory = 32` bytes → `maxBatchCoalesced = coalescedMemory/complexSize = 4`, `maxSequenceLengthSharedMemory = sharedMemorySize/complexSize = 6144`, `maxSingleSizeStrided = sharedMemorySize/coalescedMemory = 1536`, `aimThreads = 128` (default), `warpSize = 32`, `maxThreadsNum = 1024`, `maxComputeWorkGroupSize[0..1] = 1024`, `configuration.groupedBatch[axis_id] = 0` (default → automatic branch), no Rader, no zeropadding, `registerBoost = 1`.

### Example A — `axis_id = 0`, `axis_upload_id = 0`, single-upload axis, `fftDim = 1024`, `min_registers_per_thread = 8`

1. `axisBlock[0] = max(1, ceil(1024/8)/1) = 128` (§2.1).
2. Not Rader.
3. `128 <= maxThreadNum(1024)` and `<= maxComputeWorkGroupSize[0](1024)` — no clamp.
4. `numAxisUploads[0]==1`, so the `reorderFourStep && numAxisUploads>1` branch is false → else branch (§2.2): `axisBlock[0]/warpSize = 128/32 = 4 ≠ 1` → `estimate_batch = aimThreads/axisBlock[0] = 128/128 = 1`. Since `axisBlock[0](128) < aimThreads(128)` is **false**, `axisBlock[1] = 1` (not `estimate_batch`).
5. Divisibility loop: `currentAxisBlock1=1`, loop body only runs for `i=1`, no change since `axisBlock[1]` is already 1 (loop is effectively inert at batch=1).
6. Pow2-swap-preview condition requires `axisBlock[1] > 1` — false, skipped.
7. `while (axisBlock[1]*(fftDim/registerBoost) > maxSequenceLengthSharedMemory)`: `1*1024=1024 <= 6144` — no reduction. `groupedBatch = axisBlock[1] = 1`.
8. Bank-conflict swap condition again requires `axisBlock[1] > 1` — false, no swap.

**Result: `axisBlock = [128, 1, 1, 1024]`, `groupedBatch = 1`.** Interpretation: 128 threads cooperate on a single length-1024 transform (each thread owns 8 elements = `min_registers_per_thread`), and exactly 1 transform occupies the block — the transform alone already reaches `aimThreads`.

### Example B — same config, `fftDim = 64`, `min_registers_per_thread = 8`

1. `axisBlock[0] = max(1, ceil(64/8)/1) = 8`.
2. No clamps triggered (8 is well under all caps).
3. `axisBlock[0]/warpSize = 8/32 = 0 ≠ 1` → else: `estimate_batch = aimThreads/axisBlock[0] = 128/8 = 16`.
4. `axisBlock[0](8) < aimThreads(128)` true, and `axisBlock[0](8) < warpSize(32)` true → `axisBlock[1] = estimate_batch = 16`.
5. Divisibility loop: assume the actual per-axis sequence count is exactly divisible by 16 (data-dependent) — no change, `axisBlock[1]` stays 16.
6. Pow2-swap-preview: `fftDim%2==0` true, `axisBlock[1](16) > 1` true, `16*64=1024 < maxSequenceLengthSharedMemoryPow2(6144)` true, no zeropadding → **triggers**: `axisBlock[1] = 2^ceil(log2(16)) = 16` (already exact power of 2, no visible change here, but would round e.g. 12 up to 16).
7. Further clamps (R2C/merge, vendor NVIDIA loop `16*8=128 < 2*128=256` so loop doesn't fire, maxComputeWorkGroupSize, maxThreadNum check `8*16=128<=1024` fine) — no change.
8. `while` shared-memory cap: `16*(64/1)=1024 <= 6144` — no reduction. `groupedBatch = axisBlock[1] = 16`.
9. Bank-conflict swap: same condition as step 6, now checked against the non-pow2 `maxSequenceLengthSharedMemory` (6144, same result) — **triggers**: swap `axisBlock[0]` and `axisBlock[1]`: `axisBlock[0] = 16`, `axisBlock[1] = 8`, `axisSwapped = 1`.

**Result: `axisBlock = [16, 8, 1, 64]`, `groupedBatch = 16` (the `groupedBatch` scalar is captured *before* the swap and is not itself swapped), `axisSwapped = 1`.** Interpretation: 8 threads would cooperate per length-64 transform, batching 16 transforms per block — then the two dimensions are physically swapped in the launched workgroup (`[16,8,1]`) purely to reduce shared-memory bank conflicts, while the *logical* batch count recorded in `axis->groupedBatch` stays 16.

### Example C — strided axis, `axis_id = 1`, `fftDim = 256`, `min_registers_per_thread = 8`, `actualFFTSizePerAxis[1][0] = 1000` (plenty of columns to batch)

1. Top-of-function seed (§2.0): since `axis_id != 0`, the `else` branch applies: `maxSingleSizeStrided(1536)/fftDim(256) = 6 > 1` → `groupedBatch = 6 * maxBatchCoalesced(4) = 24`.
2. §2.6 rebalancing (non-NVIDIA path shown; assume `numAxisUploads[1] != 2,3` so neither halving branch fires): `groupedBatch(24) >= maxBatchCoalesced(4)` OK; `groupedBatch = (24/4)*4 = 24`; half-bandwidth-technique condition requires `fftDim(256) > maxSingleSizeStrided(1536)` — **false**, skipped; `warpSize` snap: `24 <= warpSize(32)` — skipped; `24 <= 2*4=8`? no, `24 > 8` → snap: `groupedBatch = (24/8)*8 = 24` (unchanged); `24 > 4*4=16` → snap: `groupedBatch = (24/16)*16 = 16`. So after §2.6, `groupedBatch = 16`.
3. §2.4 strided-axis block: `axisBlock[1] = max(1, ceil(256/8)/1) = 32`.
4. `axisBlock[0] = min(actualFFTSizePerAxis[1][0](1000), groupedBatch(16)) = 16`.
5. Non-NVIDIA: skip the vendor halving loop. `16 <= maxComputeWorkGroupSize[0](1024)` OK. `axisBlock[0]*axisBlock[1] = 16*32 = 512 <= maxThreadNum(1024)` — no shrink needed.
6. `groupedBatch = axisBlock[0] = 16`.

**Result (non-NVIDIA): `axisBlock = [16, 32, 1, 256]`, `groupedBatch = 16`.**
**Result if `vendorID == 0x10DE` instead (NVIDIA halving loop active at step 5):** `32*16=512 >= 2*128=256` true and `16 > maxBatchCoalesced(4)` true → `axisBlock[0] /= 2 = 8`; `32*8=256>=256` true, `8>4` true → `axisBlock[0]=4`; `32*4=128>=256` false → stop. **Result: `axisBlock = [4, 32, 1, 256]`, `groupedBatch = 4`.** This is a concrete, code-derived illustration of the vendor-specific branch materially changing the batch count (24→16→4 across the pipeline) versus the non-NVIDIA path (24→16, unchanged from there).

These three examples are hand-traced directly from the quoted C branches above; no running VkFFT instance was used, but every arithmetic step corresponds line-for-line to the quoted source, so a Python port can be checked against them.

---

## 6. Vendor/backend-specific branches — what a faithful port should pick

1. **`app->configuration.vendorID == 0x10DE` (NVIDIA) runtime branches** appear 6 times (lines 128-133, 212-236 with its own `else` for non-NVIDIA, 330-335, 397-402, 447-452, and the mirrored ones in the forced branch). This is a **runtime** device-ID check, not a compile-time backend macro — it applies identically regardless of `VKFFT_BACKEND` (CUDA/HIP/OpenCL/Level Zero/Vulkan/Metal), purely based on which physical GPU vendor is detected. A faithful port should **pick one behavior and state it explicitly**; since the earlier pass's own convention was "VKFFT_BACKEND==1 (CUDA)" for consistency, and CUDA plans overwhelmingly target NVIDIA hardware, **the NVIDIA branch (`vendorID==0x10DE` true) is the internally-consistent choice** to match that prior convention — i.e. always take the "if" side of every `vendorID==0x10DE` check, not the implicit/`else` non-NVIDIA side. (Example C above demonstrates the numeric divergence this causes.)

2. **`#if(VKFFT_BACKEND==0)` (Vulkan/GLSL) dead branches** at lines 82-89 and 351-358 are **fully commented out** (`/* ... #else*/ ... //#endif`), meaning the *shipped* code unconditionally executes what used to be the `#else` (non-Vulkan) path — the axis-swap-for-bank-conflicts logic runs the same way for every backend now, regardless of `fftDim` being a power of 2 or not. **No branch choice is actually needed here**: the file, as it exists on `master`, already has only one live code path (the commented-out `VKFFT_BACKEND==0` special case is inert). A port should simply implement the unconditional swap shown active in §2.2/§2.4's Example B, matching all backends including CUDA.

3. **`//#if(VKFFT_BACKEND!=2)` / `//#endif`** at lines 211 and 225 (with the comment "for some reason, hip doesn't get performance increase from having variable shared memory strides") are **also commented out** — i.e. this HIP-exclusion guard is not actually compiled in; the NVIDIA-branch code at lines 212-236 currently runs for HIP too if `vendorID` happened to read `0x10DE` (which it wouldn't on real HIP/AMD hardware, since vendor ID would be `0x1002`). This is dead/vestigial code, not an active backend fork — no decision needed for a port beyond noting it's inert.

4. There are **no `VKFFT_BACKEND==1/2/3/4/5` (CUDA/HIP/OpenCL/LevelZero/Metal) preprocessor branches anywhere else** in this file — everything else is portable C using only `pfUINT`/`pfceil`/`pow`/`log2` and the `VkFFTApplication`/`VkFFTAxis`/`VkFFTPlan` structs, which are backend-agnostic host-side types. So beyond items 1-3 above, this file requires **no backend-specific choice** — it is already effectively backend-generic.

---

## Files/URLs actually fetched

- `https://raw.githubusercontent.com/DTolm/VkFFT/master/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h` (target file, 471 lines, fetched via `curl`, read in full via the Read tool) — saved locally to `vkFFT_AxisBlockSplitter.h` in the scratchpad.
- `https://raw.githubusercontent.com/DTolm/VkFFT/master/vkFFT/vkFFT/vkFFT_Structs/vkFFT_Structs.h` (1193 lines; read sections 1-310, 680-870, 1000-1150 for struct/config field definitions) — saved locally as `vkFFT_Structs.h`.
- `https://raw.githubusercontent.com/DTolm/VkFFT/master/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h` (3300 lines; grepped for `VkFFTSplitAxisBlock`/`allowedSharedMemory` [not found — call site lives elsewhere] and for the `fftDim.data.i =` / `registerBoost =` / `min_registers_per_thread =` assignments at lines 3208, 3268, 3270, 3273; read lines 3190-3280 for context) — saved locally as `vkFFT_Scheduler.h`.
- `https://raw.githubusercontent.com/DTolm/VkFFT/master/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_RecursiveFFTGenerators.h` (1423 lines; grepped, ruled out as the caller) — saved locally as `vkFFT_RecursiveFFTGenerators.h`.
- `https://raw.githubusercontent.com/DTolm/VkFFT/master/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h` (795 lines; this is the actual call site at line 452; read lines 60-190 and 400-460 for `allowedSharedMemory` computation and the call context) — saved locally as `vkFFT_Plan_FFT.h`.
- `https://api.github.com/repos/DTolm/VkFFT/contents/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions` and `.../vkFFT_PlanManagement` and `.../vkFFT_PlanManagement/vkFFT_Plans` (GitHub contents API, unauthenticated, used only to enumerate directory listings to locate the caller of `VkFFTSplitAxisBlock`).
- (A first attempt used the `WebFetch` tool against the raw URL for the target file, but it returned an AI-generated summary rather than raw text — abandoned in favor of `curl` + local `Read`, which is what produced the verbatim quotes above.)

---

## Confidence levels / not independently verified

- **High confidence:** every quoted code block above is copy-pasted verbatim (via `curl` to `raw.githubusercontent.com` + the Read tool, with exact line numbers) from the live `master` branch as of 2026-09-08. Struct field definitions and their doc comments are likewise verbatim quotes from `vkFFT_Structs.h`.
- **High confidence:** the call site (`vkFFT_Plan_FFT.h:452`) and the `allowedSharedMemory`/`allowedSharedMemoryPow2` computation feeding into `VkFFTSplitAxisBlock` are directly verified by grep + read, not inferred.
- **High confidence:** the Scheduler.h origin of `fftDim.data.i`, `registerBoost`, `min_registers_per_thread`, `minRaderFFTThreadNum` is directly verified by grep + read of `vkFFT_Scheduler.h` lines 3190-3280.
- **Medium confidence:** the three worked numeric examples in §5 are hand-traced arithmetic through the quoted branches under a specific set of assumed config values (48 KiB shared memory, `aimThreads=128`, `warpSize=32`, etc. — these are the documented *defaults* per `vkFFT_Structs.h` comments, but I did not run VkFFT's actual device-query/default-fill code path, e.g. `vkFFT_AppManagement` initialization, to confirm these are exactly what a real GPU probe would produce). The `%`/`/`/`ceil` arithmetic itself was verified by manual re-derivation, not by executing C code.
- **Not verified / out of scope for this pass:** the exact default value and computation of `app->configuration.stageStartSize`-independent fields like `sharedMemorySizePow2`'s relationship to `sharedMemorySize` (i.e., is it always the largest power of 2 ≤ `sharedMemorySize`? Assumed yes from the field's doc comment but the assignment code wasn't located in this pass). Also not verified: the exact defaults-filling code for `configuration.groupedBatch[]`, `aimThreads`, `warpSize`, `numSharedBanks`, `coalescedMemory`, `vendorID` (these live in `VkFFTInitializeApp`/device-query files that were out of scope for this AxisBlockSplitter-focused pass — their doc-comment defaults were used as-is, not re-derived from code).
- **Not verified:** behavior of the `configuration.groupedBatch[axis_id] != 0` "forced" branch (lines 41-193) end-to-end with a real nonzero value, since it is the non-default path and no worked example was built for it beyond quoting its code (it is structurally a simplified subset of the automatic branch, missing the `aimThreads`/warpSize estimation, divisibility loop, and pow2-batch-rounding steps). One structural oddity was spotted but not chased further: line 179, `axis->axisBlock[0] = (axis->axisBlock[0] > app->configuration.groupedBatch[1]) ? app->configuration.groupedBatch[axis_id] : axis->axisBlock[0];` — compares against `groupedBatch[1]` (hardcoded index) but assigns from `groupedBatch[axis_id]`, which looks like it could be a copy-paste artifact in the upstream source; flagged here rather than silently "corrected" in the port.
