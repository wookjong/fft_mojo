# VkFFT Deterministic Planning/Scheduling — Source-Verified Research Report

Repository: `github.com/DTolm/VkFFT`, branch `master`.
**Important path correction**: the library root inside the repo is `vkFFT/vkFFT/...` (the checkout has a top-level `vkFFT/` directory, and the actual library sources live in a *second* `vkFFT/` subdirectory below it). All `#include` statements in the source itself confirm this, e.g. `#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"`. So the real raw URL for the scheduler is:

```
https://raw.githubusercontent.com/DTolm/VkFFT/master/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h
```

(single-`vkFFT/` paths, e.g. `vkFFT/vkFFT_PlanManagement/...`, 404.) All file paths below are given relative to the repo root exactly as they must be pasted after `master/`.

All line numbers refer to the exact file revision fetched during this research session (see "Sources Fetched" section for exact retrieval method/time; master is a moving branch, so line numbers may drift by the time of implementation — the function/variable names are the stable anchors).

---

## 1. Register scheduling — the "good/bad sequence" rule

**File**: `vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h`
**Functions**: `VkFFTGetRegistersPerThreadQuad(...)` (quad/double-double precision path, line 25) and `VkFFTGetRegistersPerThread(...)` (line 308, the normal single/double precision path — it early-returns into the Quad variant only if `quadDoubleDoublePrecision`/`quadDoubleDoublePrecisionDoubleMemory` is set, otherwise runs an almost-identical radix-register table inline).

Both functions build a per-radix register requirement table `registers_per_thread_per_radix[33]` (indexed by radix 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32), driven by a big nested-`if`/`switch` cascade on `loc_multipliers[2..7]` (how many factors of 2/3/5/7 the sequence has). This is a hardcoded lookup table of "how many complex values per thread does radix-R want", not a formula — e.g. (lines 34-40, `VkFFTGetRegistersPerThreadQuad`):

```c
if (loc_multipliers[2] > 0) {
  if (loc_multipliers[3] > 0) {
    if (loc_multipliers[5] > 0) {
      if (loc_multipliers[7] > 0) {
        registers_per_thread_per_radix[2] = 6;
        registers_per_thread_per_radix[3] = 6;
        registers_per_thread_per_radix[5] = 5;
        registers_per_thread_per_radix[7] = 7;
        ...
```

After the table is filled, derived composite radices are synthesized from it (lines 285-297):

```c
registers_per_thread_per_radix[32] = ((registers_per_thread_per_radix[2] % 32) == 0) ? registers_per_thread_per_radix[2] : 0;
registers_per_thread_per_radix[16] = ((registers_per_thread_per_radix[2] % 16) == 0) ? registers_per_thread_per_radix[2] : 0;
registers_per_thread_per_radix[8]  = ((registers_per_thread_per_radix[2] % 8)  == 0) ? registers_per_thread_per_radix[2] : 0;
registers_per_thread_per_radix[4]  = ((registers_per_thread_per_radix[2] % 4)  == 0) ? registers_per_thread_per_radix[2] : 0;
if ((registers_per_thread_per_radix[2] >= 12) && (registers_per_thread_per_radix[3] >= 12)) {
    registers_per_thread_per_radix[12] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[3]) ? registers_per_thread_per_radix[3] : registers_per_thread_per_radix[2];
    if ((registers_per_thread_per_radix[12] % 12) != 0) registers_per_thread_per_radix[12] = 0;
}
registers_per_thread_per_radix[6]  = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[3]) ? registers_per_thread_per_radix[3] : registers_per_thread_per_radix[2];
registers_per_thread_per_radix[9]  = ((registers_per_thread_per_radix[3] % 9) == 0) ? registers_per_thread_per_radix[3] : 0;
registers_per_thread_per_radix[10] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[5]) ? registers_per_thread_per_radix[5] : registers_per_thread_per_radix[2];
registers_per_thread_per_radix[14] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[7]) ? registers_per_thread_per_radix[7] : registers_per_thread_per_radix[2];
registers_per_thread_per_radix[15] = (registers_per_thread_per_radix[3] > registers_per_thread_per_radix[5]) ? registers_per_thread_per_radix[5] : registers_per_thread_per_radix[3];
```

Then `min_registers_per_thread` and `registers_per_thread` (the max) are computed by scanning the table (lines 299-302), and **this is the exact confirmed "good sequence" test** (lines 303-304, duplicated verbatim at lines 1661-1662 for the non-quad path):

```c
for (int i = 0; i < 33; i++) {
    if ((registers_per_thread_per_radix[i] != 0) && (registers_per_thread_per_radix[i] < min_registers_per_thread[0])) min_registers_per_thread[0] = registers_per_thread_per_radix[i];
    if ((registers_per_thread_per_radix[i] != 0) && (registers_per_thread_per_radix[i] > registers_per_thread[0])) registers_per_thread[0] = registers_per_thread_per_radix[i];
}
if ((registers_per_thread[0] > 16) || (registers_per_thread[0] >= 2 * min_registers_per_thread[0])) isGoodSequence[0] = 0;
else isGoodSequence[0] = 1;
```

**Confirmed**: the draft's claimed condition is exactly right, verbatim, with real names `registers_per_thread[0]` (the max register need across radices used) and `min_registers_per_thread[0]` (the min). It is indeed a **binary good/bad classification**, not a weighted/continuous penalty — `isGoodSequence` is a 0/1 out-parameter.

**What happens on "bad" — CORRECTION to the likely assumption**: `isGoodSequence` is **not** consulted as a general accept/reject gate for arbitrary user-requested FFT sizes, nor does it trigger "reordering" of an existing radix split. Its only two call sites in the whole scheduler (`VkFFTPlanAxis`-adjacent code, lines ~2454-2483 and ~2535-2566) are both inside the **Bluestein convolution-length search** (see section 7): once VkFFT has decided a length needs Bluestein (some leftover prime factor after Stockham/Rader factoring), it needs to pick a *padded* convolution length `>= 2N-1`. It does this with a `while (!FFTSizeSelected)` loop that:
1. optionally rounds the candidate up to a power of 2 if close enough (`(pow2 * 0.75) <= tempSequence`),
2. otherwise factors the candidate by 2..7,
3. if the candidate doesn't fully factor into {2,3,5,7} (`testSequence != 1`), increments `tempSequence` and retries,
4. if it does fully factor, calls `VkFFTGetRegistersPerThread(...)` to get `isGoodSequence`; **if bad, `tempSequence++` and the whole loop retries** (lines 2479-2483):
```c
res = VkFFTGetRegistersPerThread(app, (int)tempSequence, 0, max_rhs / tempSequence, axes->specializationConstants.useRader, multipliers, registers_per_thread_per_radix, &registers_per_thread, &min_registers_per_thread, &isGoodSequence);
if (res != VKFFT_SUCCESS) return res;
if (isGoodSequence) FFTSizeSelected = 1;
else tempSequence++;
```
So the real effect of "bad" is: **reject this candidate padded length and linearly search upward (`tempSequence++`) for the next composite-of-{2,3,5,7} length whose radix decomposition is register-balanced**, purely for the internal Bluestein zero-padding size — not for the user's originally requested/already-decomposed axis length. Confidence: **high**, directly read from source, two call sites checked exhaustively via grep.

For the *normal* (non-Bluestein) axis-length radix split, `registers_per_thread`/`min_registers_per_thread` (computed by the very same function) are still used pervasively downstream to size `registerBoost`, thread counts, etc. (see sections 3-4), but the boolean `isGoodSequence` flag itself is discarded there.

---

## 2. Stage/radix grouping — the power-of-two workload-balance scheduler

**File**: same, `vkFFT_Scheduler.h`, embedded inside the `else` branch reached when `loc_multipliers[2]>0` but `loc_multipliers[3]==loc_multipliers[5]==loc_multipliers[7]==0` (i.e. a pure power-of-2 sequence not already covered by a lookup-table special case) — lines 145-187 (Quad path) and the byte-identical lines 1194-1237 (non-quad path, inside `VkFFTGetRegistersPerThread`, itself inside a `switch` on `loc_multipliers[2]`).

Exact quoted code (lines 146-179):

```c
int max_loc_multipliers_pow2 = 0;
pfUINT active_threads_y = max_rhs / 64; //estimate workbalance across CU (assume we have 64 CU)
if (active_threads_y == 0) active_threads_y = 1;
int testMinStages = 10000000;
int maxRadixMinStages = 1;
int fixMaxCheckRadix2 = 3;

for (int i = 1; i <= fixMaxCheckRadix2; i++) {
    int numStages = (int)pfceil(log2(fft_length) / ((double)i));
    if (numStages < testMinStages) {
        testMinStages = numStages;
        maxRadixMinStages = i;
    }
}
for (int i = maxRadixMinStages; i >= 1; i--) {
    pfUINT active_threads_x = (active_threads_y * fft_length) / ((int)pow(2, i));
    if (active_threads_x >= 128) {
        max_loc_multipliers_pow2 = i;
        i = 1;
    }
}
if (max_loc_multipliers_pow2 < 3) max_loc_multipliers_pow2 = 3;

int final_loc_multipliers_pow2 = 1;
int num_stages_min = (int)log2(fft_length);
for (int i = 2; i <= max_loc_multipliers_pow2; i++) {
    int num_stages = (int)pfceil(((int)log2(fft_length)) / (double)i);
    if (num_stages < num_stages_min) {
        final_loc_multipliers_pow2 = i;
        num_stages_min = num_stages;
    }
}
registers_per_thread_per_radix[2] = (loc_multipliers[2] > final_loc_multipliers_pow2) ? (int)pow(2, final_loc_multipliers_pow2) : (int)pow(2, loc_multipliers[2]);
registers_per_thread_per_radix[2] = (loc_multipliers[2] < 3) ? (int)pow(2, loc_multipliers[2]) : registers_per_thread_per_radix[2];
```

**Confirmed exactly as the draft describes**, including the literal comment `//estimate workbalance across CU (assume we have 64 CU)`. Real variable names: `active_threads_y`, `active_threads_x`, `max_loc_multipliers_pow2` (the "grouping exponent" — i.e. use radix `2^max_loc_multipliers_pow2` as the largest single-kernel radix), `final_loc_multipliers_pow2` (the actually-chosen exponent minimizing `numStages`), `fixMaxCheckRadix2` (search bound, hardcoded to 3, i.e. only radices 2^1..2^3 = 2,4,8 are considered as the "min stages" seed).

Mechanism in words: `active_threads_y = max_rhs/64` estimates how many independent "columns"/batches of the transform can be spread across an assumed 64 compute units to keep them busy; `active_threads_x` then estimates threads-per-column if the largest radix grouping tried is `2^i`; the loop walks `i` **downward** from `maxRadixMinStages` looking for the first (i.e. largest) `i` that keeps `active_threads_x >= 128` (a hardcoded occupancy floor — need at least 128 active x-threads per column-batch). If none qualifies, it clamps to a floor of `max_loc_multipliers_pow2 = 3` (i.e. never use a grouping radix smaller than 2^3=8 for pow2 kernels). Finally, among candidate group sizes `2..max_loc_multipliers_pow2`, it picks the `final_loc_multipliers_pow2` that literally minimizes `ceil(log2(fft_length)/i)` — the number of Stockham stages — ties broken by the *first* `i` encountered in ascending order (since `<` not `<=` is used in the comparison).

**Backend-specific tweak** (non-quad path only, line 1200-1202, CUDA backend):
```c
#if(VKFFT_BACKEND==1)
fixMaxCheckRadix2 = (((fft_length >= 1024) || (fft_length == 256)) && (extraSharedMemoryForPow2) && (!useRader)) ? 5 : 3;
#endif
```
On CUDA, for lengths >=1024 or ==256, with extra shared memory available and no Rader in play, the search bound is widened to `fixMaxCheckRadix2 = 5` (i.e. radices up to 2^5=32 considered), otherwise 3. This is a deterministic, size/backend-conditioned constant — not empirical/benchmark tuning — but it is backend-specific (`VKFFT_BACKEND==1` is the CUDA compile-time macro).

Confidence: **high** — exact match to the draft's claims, verbatim comment confirmed.

---

## 3. Shared memory / registerBoost / registerBoost4Step

**Files**: `vkFFT_Structs.h` (config field docs) + `vkFFT_Scheduler.h` (`VkFFTPlanAxis`/scheduler body, no separate top-level function — it's inline in the big per-axis planning routine, roughly lines 2230-2900) + `vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h` (`VkFFTSplitAxisBlock`, consumes the results).

### Definitions (non-strided vs strided max sequence length)
```c
// Scheduler.h line 2240-2241
int usedSharedMemory = (((app->configuration.size[axis_id] & (app->configuration.size[axis_id] - 1)) == 0) && (!app->configuration.performDCT) && (!app->configuration.performDST))
    ? (int)app->configuration.sharedMemorySizePow2 : (int)app->configuration.sharedMemorySize;
int maxSequenceLengthSharedMemory = usedSharedMemory / complexSize;      // NON-STRIDED capacity
...
// line 2588
int maxSequenceLengthSharedMemoryStrided = (app->configuration.coalescedMemory > complexSize)
    ? usedSharedMemory / ((int)app->configuration.coalescedMemory)
    : usedSharedMemory / complexSize;                                    // STRIDED capacity
```
Confirmed: **non-strided** max single-kernel length = `usedSharedMemory / complexSize` (every shared-mem slot usable). **Strided** max single-kernel length = `usedSharedMemory / coalescedMemory` (shared memory budget divided by the coalescing granule, because a strided axis needs `coalescedMemory`-wide buffers per logical element to keep global-memory accesses coalesced) — strictly smaller in general. `sharedMemorySizePow2` (a config field, "power of 2 which is less or equal to sharedMemorySize, in bytes", `Structs.h` line 293) is used instead of the raw `sharedMemorySize` specifically for power-of-2-length transforms (so shared-memory bank layouts stay power-of-2-friendly).

### registerBoost
`Structs.h` line 277 (verbatim doc comment):
```c
pfUINT registerBoost; //specify if register file size is bigger than shared memory and can be used to extend it X times (on Nvidia 256KB register file can be used instead of 32KB of shared memory, set this constant to 4 to emulate 128KB of shared memory). Default 1
```
This **directly confirms the draft's claim**: registerBoost is deliberately about using *more* registers than the arithmetic minimum, to emulate a larger shared-memory budget and thereby avoid shared-memory (or global-memory, in the multi-upload case) round-trips. Computation (`Scheduler.h` line 2582-2586, run once initially, then possibly overridden — see below):
```c
int registerBoost = 1;
for (int i = 1; i <= app->configuration.registerBoost; i++) {
    if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (i * i) == 0)
        registerBoost = i;
}
```
i.e. `registerBoost` is the **largest `i <= configuration.registerBoost` such that `i*i` divides the FFT length** — register-boosting only kicks in when the length is divisible by a perfect square up to the configured cap, because the boost factor is applied as an extra "square" split of the axis (part folded into the per-thread register count, part into the coalesced/strided batch dimension). `maxSingleSizeNonStrided`/`maxSingleSizeStrided` are then scaled by this `registerBoost` (lines 2587, 2589, 2621-2622):
```c
maxSingleSizeNonStrided *= registerBoost;      // (conditionally, non-strided axis)
maxSingleSizeStrided = maxSequenceLengthSharedMemoryStrided * registerBoost;
```
Later (`VkFFTOptimizeRadixKernels`, `Scheduler.h` lines 2016-2036 for `registerBoost==2`, 2037+ for `registerBoost==4`) the radix multiplier counts (`loc_multipliers[2/4/8/16/32]`) are explicitly *rebalanced* to make sure enough radix-4 (or radix-2, for boost 2) stages exist to "absorb" the boost — i.e. registerBoost doesn't just relabel bookkeeping, it changes which radix decomposition is picked, confirming it truly forces more per-thread register usage. Example (`registerBoost==2`, line 2016-2020):
```c
if ((registerBoost == 2) && (loc_multipliers[2] == 0)) {
    if (loc_multipliers[4] > 0) {
        loc_multipliers[4]--;
        loc_multipliers[2] = 2;
    }
    ...
```
`registerBoostNonPow2` (`Structs.h` line 278, "specify if register overutilization should be used on non power of 2 sequences") gates whether boosting is allowed for non-pow2 lengths at all — checked at `Scheduler.h` line 2617:
```c
if (((canBoost == 0) || (((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] & (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - 1)) != 0) && (!app->configuration.registerBoostNonPow2))) && (registerBoost > 1)) {
    registerBoost = 1;
    numPasses++;
}
```
i.e. if the length is not a power of 2 (`N & (N-1) != 0`) and `registerBoostNonPow2` is off, any computed `registerBoost>1` is forced back to 1 (and `numPasses` is bumped by one to compensate for the lost capacity).

### registerBoost4Step
`Structs.h` line 279:
```c
pfUINT registerBoost4Step; //specify if register file overutilization should be used in big sequences (>2^14), same definition as registerBoost. Default 1
```
Used only once the axis needs `>1` pass (`temp>1` branch, `Scheduler.h` line 2594-2599):
```c
if (temp > 1) {//more passes than one
    for (int i = 1; i <= app->configuration.registerBoost4Step; i++) {
        if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (i * i) == 0) {
            registerBoost = i;
        }
    }
    if ((!app->configuration.performConvolution)) maxSingleSizeNonStrided = maxSequenceLengthSharedMemory * registerBoost;
    if ((!app->configuration.performConvolution)) maxSingleSizeStrided = maxSequenceLengthSharedMemoryStrided * registerBoost;
    ...
```
So `registerBoost4Step` is literally the same perfect-square-divisor search as `registerBoost`, but it is the register-boost cap specifically applied in the **multi-upload / four-step regime** (the doc comment's ">2^14" is descriptive framing in the comment, not an explicit numeric branch condition in this code — the real gating condition in code is `temp > 1`, i.e. "does the axis already need more than one shared-memory upload at boost=1"). Confidence: **high** on the code path itself; **medium** on whether ">2^14" appears as a literal numeric constant anywhere else in the codebase — grep of `Scheduler.h` found no literal `16384`/`2^14` comparison tied to this variable, so that threshold is descriptive/historical in the comment rather than an enforced branch condition in the current scheduler.

---

## 4. Number of passes (1 / 2 / 3 / more uploads) and four-step switching

**File**: `vkFFT_Scheduler.h`, same per-axis planning block, lines ~2590-2968.

Base pass count estimate (line 2593):
```c
temp = (axis_id == nonStridedAxisId) ? (pfUINT)pfceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)maxSingleSizeNonStrided)
                                      : (pfUINT)pfceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)maxSingleSizeStrided);
if (temp > 1) {//more passes than one
    ... (registerBoost4Step search shown above) ...
    temp = ((axis_id == nonStridedAxisId) && ((!app->configuration.reorderFourStep) || (app->useBluesteinFFT[axis_id])))
             ? FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxSingleSizeNonStrided
             : FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxSingleSizeStrided;
    if (app->configuration.reorderFourStep && (!app->useBluesteinFFT[axis_id]))
        numPasses = (int)pfceil(log2(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]) / log2(maxSingleSizeStrided));
    else
        numPasses += (int)pfceil(log2(temp) / log2(maxSingleSizeStrided));
}
```
So: if the axis fits in one upload's worth of shared memory (`temp<=1`), `numPasses=1`. Otherwise `numPasses` is computed as a **log-based number of passes needed to shrink the axis down to `maxSingleSizeStrided`-sized chunks per pass**, with a different formula depending on whether "four-step reordering" (`reorderFourStep`) is enabled and whether Bluestein is in play — the reorder-four-step case uses a direct `log_maxSingleSizeStrided(N)` (implying every pass, including the first, is capped by the *strided* limit), while the non-reorder / Bluestein case treats the first pass as capped by the (larger) non-strided or boosted limit and only the remaining `temp` factor by the strided limit.

Then `registerBoost` is finally locked in (line 2608-2620, shown in section 3) and forced hard overrides are applied via **two explicit user-configurable thresholds** (`Structs.h` lines 232-233, quoted verbatim):
```c
pfUINT swapTo2Stage4Step; //specify at which number to switch from 1 upload to 2 upload 4-step FFT, in case if making max sequence size lower than coalesced sequence helps to combat TLB misses. Default 0 - disabled.
pfUINT swapTo3Stage4Step; //specify at which number to switch from 2 upload to 3 upload 4-step FFT, in case if making max sequence size lower than coalesced sequence helps to combat TLB misses. Default 0 - disabled. Must be at least 65536
```
and applied (`Scheduler.h` line 2648-2650):
```c
if ((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] >= app->configuration.swapTo2Stage4Step) && (numPasses < 3)) numPasses = 2;//Force set to 2 stage 4 step algorithm
if ((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] >= app->configuration.swapTo3Stage4Step) && (app->configuration.swapTo3Stage4Step >= 65536)) numPasses = 3;//Force set to 3 stage 4 step algorithm
if (forceRaderTwoUpload && (numPasses == 1)) numPasses = 2;//Force set Rader cases that use more than 512 or maxNumThreads threads per one of Rader primes
```
These thresholds are **disabled by default (0)** and are explicit user knobs, not autotuned — but when set, they are hard, deterministic force-overrides of the computed `numPasses`. `forceRaderTwoUpload` is itself set deterministically (`Scheduler.h` lines 2342-2343) whenever a chosen Rader prime factor would need more than 512 threads or more than `configuration.maxThreadsNum` threads:
```c
if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / i > 512) forceRaderTwoUpload = 1;
if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / i > app->configuration.maxThreadsNum) forceRaderTwoUpload = 1;
```
Finally: `if (numPasses > 3) return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;` (line 2890-2893) — **VkFFT deterministically caps at 3 uploads/passes**; there is no ">3 passes" support path (a 4th-pass attempt inside the 3-pass non-pow2 split failure path just falls through to this same hard error, see section 5).

Confidence: **high**, all thresholds/paths read directly from source.

---

## 5. Axis splitting: pow2 vs non-pow2, unit-stride vs non-unit-stride

**File**: `vkFFT_Scheduler.h`, lines ~2652-2889 (still inside the same per-axis planning function, gated on `numPasses==2` then `numPasses==3`).

### Power-of-2, 2-pass (lines 2656-2708) — CONFIRMS power-of-8 preference
```c
if (isPowOf2 && (!((app->configuration.vendorID == 0x10DE) && (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] > 262144)))) {
    if ((axis_id == nonStridedAxisId) && ((!app->configuration.reorderFourStep) || (app->useBluesteinFFT[axis_id]))) {
        int maxPow8SharedMemory = (int)pow(8, ((int)log2(maxSequenceLengthSharedMemory)) / 3);
        //unit stride
        if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxPow8SharedMemory <= maxSingleSizeStrided) {
            locAxisSplit[0] = maxPow8SharedMemory;
        }
        else if (... / maxSequenceLengthSharedMemory <= maxSingleSizeStrided) {
            locAxisSplit[0] = maxSequenceLengthSharedMemory;
        }
        else { /* fall back to registerBoost-scaled sizes, possibly halving registerBoost repeatedly */ }
    }
    else { // strided axis
        int maxPow8Strided = (int)pow(8, ((int)log2(maxSingleSizeStrided)) / 3);
        if (maxPow8Strided > 512) maxPow8Strided = 512;
        //all FFTs are considered as non-unit stride
        ...
    }
    locAxisSplit[1] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / locAxisSplit[0];
    if (locAxisSplit[1] < 64) { /* rebalance so second-pass size >= 64 */ locAxisSplit[1] = 64; }
    if (locAxisSplit[1] > locAxisSplit[0]) { swap(locAxisSplit[0], locAxisSplit[1]); } // keep [0] the larger
}
```
**Confirmed exactly**: `maxPow8SharedMemory = 8^floor(log2(maxSequenceLengthSharedMemory)/3)` — this is literally "the largest power of 8 that fits in the shared-memory-limited max sequence length" (since `8 = 2^3`, `log2(x)/3` counts how many powers-of-8 fit). This is used as the **first-choice split size** for the non-strided/unit-stride first pass, only falling back to plain `maxSequenceLengthSharedMemory` or registerBoost-scaled sizes if the power-of-8 choice would leave too much work (`> maxSingleSizeStrided`) for the second pass. There is also a hardcoded ceiling `maxPow8Strided > 512 => 512` for the strided-axis variant, and a floor of `64` on the smaller split dimension (`if (locAxisSplit[1] < 64) ... locAxisSplit[1] = 64`) — apparently an occupancy/coalescing floor. There's also an Nvidia-specific carve-out: `vendorID == 0x10DE` (NVIDIA's PCI vendor ID) with length `>262144` skips the whole power-of-8 pow2 path and instead falls into the generic (non-pow2-style) sqrt-divisor search below — a deterministic, vendor-conditioned branch, not benchmark tuning.

### Power-of-2, 3-pass (lines 2751-2844) — same power-of-8 preference, applied twice
Same `maxPow8SharedMemory`/`maxPow8Strided` construction, chosen for the first split; **and** an explicit TLB/coalescing-driven variant for the strided branch (lines 2778-2789, comment quoted verbatim):
```c
//to account for TLB misses, it is best to coalesce the unit-strided stage to 128 bytes
...
int maxSingleSizeStrided128 = usedSharedMemory / (128);
int maxPow8_128 = (int)pow(8, ((int)log2(maxSingleSizeStrided128)) / 3);
//unit stride
if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxPow8_128 <= maxPow8Strided * maxSingleSizeStrided)
    locAxisSplit[0] = maxPow8_128;
```
**This confirms the draft's "128-byte" claim literally** — `usedSharedMemory / 128` is used, with an explicit comment naming "128 bytes" and "TLB misses" as the rationale, independent of the general `coalescedMemory` config field (which is typically 32 or 64 bytes per the `Structs.h` doc comment quoted in section 6). Confidence: **high**, exact numeral and exact comment text confirmed.

### Non-power-of-2, 2-pass (lines 2709-2749) — sqrt-based divisor search
```c
int sqrtSequence = (int)pfceil(sqrt(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]));
for (int i = 0; i < sqrtSequence; i++) {
    if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (sqrtSequence - i) == 0) {
        if ((sqrtSequence - i <= maxSingleSizeStrided) && (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrtSequence - i) <= maxSequenceLengthSharedMemory)) {
            locAxisSplit[0] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrtSequence - i);
            locAxisSplit[1] = sqrtSequence - i;
            i = sqrtSequence;
            successSplit = 1;
        }
    }
}
if (successSplit == 0) numPasses = 3; // fall through to 3-pass if no valid divisor pair found
```
**Confirmed**: an exact-divisor search starting at `ceil(sqrt(N))` and walking **downward** (`sqrtSequence - i` for increasing `i`) until it finds a divisor `d` of `N` such that `d <= maxSingleSizeStrided` and `N/d <= maxSequenceLengthSharedMemory` (resource-constrained), i.e. the closest-to-square divisor pair that also respects the shared-memory/coalescing budgets. If no such divisor exists anywhere from `sqrt(N)` down to 1, it doesn't error — it bumps to `numPasses = 3` and retries with a 3-way split.

### Non-power-of-2, 3-pass (lines 2845-2888) — nested sqrt/cube-root-style search
```c
int sqrt3Sequence = (int)pfceil(pow(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id], 1.0 / 3.0));
for (int i = 0; i < sqrt3Sequence; i++) {
    if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (sqrt3Sequence - i) == 0) {
        int sqrt2Sequence = (int)pfceil(sqrt(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrt3Sequence - i)));
        for (int j = 0; j < sqrt2Sequence; j++) {
            if ((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrt3Sequence - i)) % (sqrt2Sequence - j) == 0) {
                if ((sqrt3Sequence - i <= maxSingleSizeStrided) && (sqrt2Sequence - j <= maxSingleSizeStrided) && (.../ (sqrt3Sequence - i) / (sqrt2Sequence - j) <= maxSingleSizeStridedHalfBandwidth)) {
                    locAxisSplit[0] = .../ (sqrt3Sequence - i) / (sqrt2Sequence - j);
                    locAxisSplit[1] = sqrt3Sequence - i;
                    locAxisSplit[2] = sqrt2Sequence - j;
                    ... successSplit = 1;
                }
            }
        }
    }
}
if (successSplit == 0) numPasses = 4; // -> immediately hits the numPasses>3 hard error at line 2890
```
**Confirmed**: this is exactly the draft's "cube-root-style split" — first find a divisor near `N^(1/3)` (walked downward from `ceil(N^(1/3))`), then recursively find a divisor of the remaining quotient near its square root (same downward-walk pattern as the 2-pass case), giving a 3-way factor split as close to cubic as divisibility and the resource caps (`maxSingleSizeStrided`, `maxSingleSizeStridedHalfBandwidth`) allow. Failure here does not retry a 4th pass — it just sets `numPasses=4`, which the very next check (`if (numPasses>3) return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;`) turns into a hard error. **VkFFT's deterministic planner therefore rejects, rather than approximates, non-composite-friendly lengths that don't admit a good 3-way split** (such lengths are expected to have already been routed through Bluestein/Rader before reaching this point, or the user must reduce `coalescedMemory`/increase `sharedMemorySize` config).

There is also a non-strided (`axis_id == nonStridedAxisId`) variant of the 3-pass non-pow2 search (lines 2846-2865) that runs the *outer* search over `maxSequenceLengthSharedMemory - i` directly (not `N^(1/3)`) — i.e., for the non-strided axis it searches near the shared-memory capacity, not near the cube root, before doing the inner sqrt search on the remaining factor.

### Four-step reordering / factor-reordering for memory access
After `locAxisSplit[]` is finalized, if four-step reordering is enabled and Bluestein isn't in play, VkFFT explicitly **reorders which pass gets which split factor** to favor even, then %4, then %8 divisibility in `locAxisSplit[0]` (lines 2945-2966):
```c
if (((app->configuration.reorderFourStep) && (!app->useBluesteinFFT[axis_id]))) {
    for (int i = 0; i < numPasses; i++) {
        if ((locAxisSplit[0] % 2 != 0) && (locAxisSplit[i] % 2 == 0)) { swap(locAxisSplit[0], locAxisSplit[i]); }
    }
    for (int i = 0; i < numPasses; i++) {
        if ((locAxisSplit[0] % 4 != 0) && (locAxisSplit[i] % 4 == 0)) { swap(locAxisSplit[0], locAxisSplit[i]); }
    }
    for (int i = 0; i < numPasses; i++) {
        if ((locAxisSplit[0] % 8 != 0) && (locAxisSplit[i] % 8 == 0)) { swap(locAxisSplit[0], locAxisSplit[i]); }
    }
}
```
This is the real "factor-reordering rule for memory access" the draft alludes to: it greedily prefers the *first* pass (`locAxisSplit[0]`, which is transposed via the temp buffer under four-step) to be divisible by 8, then 4, then 2, presumably to keep transpose/global-memory strides friendlier to coalescing on that pass.

Confidence for all of section 5: **high** — every formula/threshold above is a direct quote or close paraphrase of the fetched source.

---

## 6. Memory-access heuristics: coalescing, 128-byte TLB rule, four-step, factor reordering

Consolidating findings already surfaced above, plus additional confirmation:

- **`coalescedMemory`** (`Structs.h` line 197, verbatim): `pfUINT coalescedMemory;//in bytes, for Nvidia and AMD is equal to 32, Intel is equal 64, scaled for half precision. Gonna work regardles, but if specified by user correctly, the performance will be higher.` — this is the primary, general-purpose coalescing granule (32B Nvidia/AMD, 64B Intel by convention), used throughout as the divisor for "strided" capacity (`maxSequenceLengthSharedMemoryStrided = usedSharedMemory / coalescedMemory`) and for thread-block batching (`maxBatchCoalesced = coalescedMemory / complexSize` in `vkFFT_AxisBlockSplitter.h` line 27).
- **The literal "128-byte" constant** is a *separate*, hardcoded special case used only in the 3-pass power-of-2 strided-axis split (`Scheduler.h` line 2785, quoted in section 5): `int maxSingleSizeStrided128 = usedSharedMemory / (128);` with the comment `//to account for TLB misses, it is best to coalesce the unit-strided stage to 128 bytes`. So **both numbers are real**: `coalescedMemory` (typically 32/64, user/vendor configurable) is the general per-vendor coalescing width, while `128` is a separate, hardcoded literal specifically invoked as a TLB-page/coalescing heuristic inside the 3-pass pow2 splitter — not derived from `coalescedMemory`.
- **Four-step reordering** (`reorderFourStep` config flag, `Structs.h` line 297: `// unshuffle Four step algorithm. Requires tempbuffer allocation (0 - off, 1 - on). Default 1.`) changes both the `numPasses` formula (section 4) and which axis-split factor is assigned to which pass (the %8/%4/%2 reordering loop, section 5) — its purpose per the doc comment is to "unshuffle" (i.e. do an explicit bit/digit-reversal-avoiding transpose via `tempBuffer`) between passes, which is the classic four-step FFT algorithm's global-memory transpose stage.
- **Occupancy floors** recur as hardcoded constants throughout: `active_threads_x >= 128` (section 2), `locAxisSplit[1] < 64 => forced to 64` (section 5, both 2-pass and 3-pass pow2 branches), and the Rader-specific `> 512` / `> maxThreadsNum` thread-count triggers for `forceRaderTwoUpload` (section 4). None of these are computed from a device query beyond the user-supplied `configuration.maxThreadsNum`/`coalescedMemory`/`sharedMemorySize(Pow2)` — they are fixed, deterministic literals in the algorithm.

Confidence: **high** on the literal `128` constant and its comment (directly quoted); **high** on `coalescedMemory`'s 32/64 convention (directly quoted from doc comment, but note this is a *documented convention for the user to set*, not a value VkFFT auto-detects/hardcodes per vendor in this file — no vendor-ID-keyed lookup table for `coalescedMemory` was found in the fetched files).

---

## 7. Rader's algorithm and Bluestein's algorithm (brief, boundary-documentation level)

**Trigger logic** (`vkFFT_Scheduler.h`, inside `VkFFTPlanAxis`, lines ~2288-2407): for each axis, VkFFT first strips out factors of {2,3,...} up to `configuration.fixMinRaderPrimeMult` (Stockham radix kernels) directly (line 2295-2301). It then attempts **Rader's algorithm** for remaining prime factors in the range `[fixMinRaderPrimeMult, fixMaxRaderPrimeMult)` / up to `fixMaxRaderPrimeFFT` (lines 2304-2383, including a "Sophie Germain safe prime" check at lines 2324-2332 that decides whether a prime factor should be handled via a **Rader-multiplication** path—`useRaderMult`, needs LUT—versus a **Rader-FFT** path—recursing into `VkFFTConstructRaderTree`, defined in the same file at line 1733). Only if, after both direct-radix and Rader factoring, a residual factor remains (`tempSequence != 1`, line 2406) does VkFFT fall back to **Bluestein's algorithm**:
```c
//initial Bluestein check
if (tempSequence != 1) {
    app->useBluesteinFFT[axis_id] = 1;
    ...
    app->configuration.registerBoost = 1;   // Bluestein disables registerBoost
    tempSequence = 2 * FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - 1;  // classic Bluestein padding >= 2N-1
    ... // search for a good composite/pow2 padded length (section 1's isGoodSequence loop)
```
So the **priority order is: direct radix kernels (2,3,5,7,...,fixMinRaderPrimeMult) → Rader (mult or FFT variant) for remaining prime factors up to configurable bounds → Bluestein for whatever is left** (typically only needed for one large "hard" prime factor, or when Rader's resource requirements are rejected).

**Implementing files**:
- Rader kernel code generation: `vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RaderKernels.h` (e.g. `appendFFTRaderStage(...)`, references `sc->currentRaderContainer->stageRadix`, `registers_per_thread_per_radix`, `registerBoost` — i.e. Rader sub-transforms go through the *same* register-scheduling machinery as section 1-3).
- Rader tree construction / planning: `VkFFTConstructRaderTree` and `VkFFTGetRaderFFTStages`, both in `vkFFT_Scheduler.h` (not a separate file, despite the RaderKernels.h/PlanManagement split suggested by directory names).
- Bluestein convolution codegen: `vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/PrePostProcessing/vkFFT_Bluestein.h` (`appendBluesteinMultiplication(...)` — the chirp-multiply pre/post-processing step around the padded convolution FFT).

**Boundary note for the port**: since the M2NDP port is explicitly not replicating Rader/Bluestein, the practical boundary is: any FFT length whose prime factorization (after stripping factors up to whatever `fixMinRaderPrimeMult`-equivalent bound the port chooses) leaves a residual prime factor larger than that bound should be treated as "outside deterministic-radix support" in the port, mirroring VkFFT's own `tempSequence != 1` trigger condition above.

Confidence: **high** on file locations and trigger condition (`tempSequence != 1` after radix+Rader factoring ⇒ Bluestein); **medium** on full internal Rader-tree recursion details (`VkFFTConstructRaderTree` internals were not fully traced line-by-line, consistent with the task's instruction that exhaustive fidelity is not required here).

---

## Sources Fetched (all successfully retrieved and read in full via raw.githubusercontent.com, master branch)

1. `vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h` (3300 lines) — primary source for sections 1-5, 7.
2. `vkFFT/vkFFT/vkFFT_Structs/vkFFT_Structs.h` (1193 lines) — config field documentation, sections 3, 4, 6.
3. `vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h` (471 lines) — section 3/6 (thread-block/coalesced-batch sizing consuming scheduler output).
4. `vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_ManageLUT.h` (1772 lines) — fetched, browsed for LUT/Rader-mult context, not separately quoted.
5. `vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_RecursiveFFTGenerators.h` (1423 lines) — fetched, checked (no direct hits for the specific claims; scheduling logic lives in Scheduler.h, not here despite the promising name).
6. `vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h` (795 lines) — fetched, used to confirm `VkFFTPlanAxis` call context and Bluestein/four-step flags flowing into per-axis specialization constants.
7. `vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RaderKernels.h` (2376 lines) — fetched, used for section 7 (Rader codegen confirmation).
8. `vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/PrePostProcessing/vkFFT_Bluestein.h` (315 lines) — fetched, used for section 7 (Bluestein codegen confirmation).

Retrieval method: `curl` to `raw.githubusercontent.com/DTolm/VkFFT/master/<path>` for each file above (verified HTTP 200 / correct C content by direct file read), then local `grep`/`read` for exact line-numbered quoting — this avoids any LLM-summarization drift in the quoted code blocks above (all quotes are copy-pasted from the actual downloaded file content, not paraphrased by an intermediate model).

## Draft Claims — Verification Status Summary

| # | Claim | Status | Confidence |
|---|---|---|---|
| 1 | `registers_per_thread > 16` / `>= 2*min_registers_per_thread` "good/bad" binary rule (not weighted) | **Confirmed verbatim** (real names `registers_per_thread[0]`, `min_registers_per_thread[0]`, `isGoodSequence[0]`) | High |
| 1 | What happens on "bad" (rejected/reordered/re-split?) | **Corrected**: only consulted inside the Bluestein padded-length search; "bad" ⇒ linear `tempSequence++` retry, not a general rejection/reorder of user-requested lengths | High |
| 2 | `active_threads_y = max_rhs/64` with "assume 64 CUs" comment; `active_threads_x` threshold; minimize-stages grouping | **Confirmed verbatim**, including exact comment text, plus an additional CUDA-only (`VKFFT_BACKEND==1`) size-conditioned widening of the search bound not mentioned in the draft | High |
| 3 | registerBoost/registerBoost4Step deliberately over-use registers vs. shared memory | **Confirmed verbatim** via `Structs.h` doc comments and the perfect-square-divisor + radix-rebalancing logic in `VkFFTOptimizeRadixKernels` | High |
| 3 | Non-strided vs strided max sequence size definitions | **Confirmed**: non-strided = `usedSharedMemory/complexSize`; strided = `usedSharedMemory/coalescedMemory` | High |
| 4 | 1/2/3-pass thresholds, four-step switch (`swapTo2Stage4Step`/`swapTo3Stage4Step`) | **Confirmed verbatim**, including the hard cap at 3 passes (`numPasses>3` ⇒ error) | High |
| 5 | Pow2: preference for power-of-8 subsequences | **Confirmed verbatim** (`maxPow8SharedMemory = pow(8, floor(log2(...)/3))`), used in both 2-pass and 3-pass pow2 splitters | High |
| 5 | Non-pow2: divisor search near sqrt / cube-root under resource constraints | **Confirmed verbatim**: `ceil(sqrt(N))` downward-walk for 2-pass; `ceil(N^(1/3))` downward-walk plus nested sqrt-walk for 3-pass | High |
| 6 | 128-byte TLB coalescing constant | **Confirmed verbatim**, exact literal `128` and exact comment "to account for TLB misses...coalesce...to 128 bytes", found only in the 3-pass pow2 strided splitter (not a global constant) | High |
| 6 | Four-step / factor-reordering rules for memory access | **Confirmed**: explicit %8/%4/%2 divisibility-preferring swap loop on `locAxisSplit[0]` when `reorderFourStep` is active | High |
| 7 | Rader's algorithm implemented | **Confirmed**: `VkFFTConstructRaderTree`/`VkFFTGetRaderFFTStages` in `vkFFT_Scheduler.h`, kernel codegen in `vkFFT_RaderKernels.h` | High (trigger logic); Medium (full internal tree-construction detail not exhaustively traced) |
| 7 | Bluestein's algorithm implemented | **Confirmed**: triggered when `tempSequence != 1` after radix+Rader factoring; codegen in `vkFFT_Bluestein.h` | High |

## Items Not Fully Verified / Out of Scope of This Pass

- The exact numeric contents of `VkFFTConstructRaderTree`'s internal generator/primitive-root search (how it picks between the "Rader-FFT" sub-container path vs. multiplication path beyond the Sophie-Germain-prime check shown) were read structurally but not exhaustively traced statement-by-statement — acceptable per the task's instruction that section 7 only needs high-level fidelity.
- `vkFFT_ManageLUT.h` and `vkFFT_RecursiveFFTGenerators.h` were fully downloaded and scanned but did not contain additional scheduling thresholds beyond what's in `vkFFT_Scheduler.h`; they are LUT-table-value generation and recursive-kernel-string-generation utilities consumed *after* planning decisions are made, not planning logic themselves — flagged here in case a future pass wants deeper LUT-precision-related determinism checks.
- The registerBoost4Step doc-comment's ">2^14" framing has no directly corresponding literal numeric branch in the scheduler (the real gate is `temp > 1`, i.e. "more than one pass already needed at boost=1") — treat ">2^14" as descriptive context in the source comment rather than an enforced threshold to replicate.
- `coalescedMemory`'s per-vendor values (32 Nvidia/AMD, 64 Intel) are documented as a *user-configuration convention* in the doc comment, not verified as an auto-detected/hardcoded vendor-ID lookup table anywhere in the fetched files — if the port needs auto-detection behavior, this specific claim needs a separate check of VkFFT's device-query/initialization code (not in the files fetched for this pass).
