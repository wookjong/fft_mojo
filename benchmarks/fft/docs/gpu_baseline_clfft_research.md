# clFFT Planning/Tuning Philosophy — Source-Verified Report

All facts below were verified against the **actual clFFT source** fetched directly from
`https://raw.githubusercontent.com/clMathLibraries/clFFT/master/src/library/<file>` on 2026-09-08.
Line numbers refer to the exact files as downloaded (listed in Section 7). Everything is a direct
read of the code — no paraphrasing of "how GPU FFTs usually work."

---

## 1. Specialization table (`KernelCoreSpecs` / `SpecRecord`)

**File:** `src/library/generator.stockham.cpp`, lines 284–354 (macro `RADIX_TABLE_COMMON` at 284–291;
class `KernelCoreSpecs<PR>` at 295–354).

Record shape (line 298–305):
```cpp
struct SpecRecord
{
    size_t length;
    size_t workGroupSize;
    size_t numTransforms;
    size_t numPasses;
    size_t radices[12]; // Setting upper limit of number of passes to 12
};
```

### RADIX_TABLE_COMMON — shared by both P_SINGLE and P_DOUBLE (lines 284–291)

| Length | WorkGroupSize | NumTransforms | NumPasses | Radices |
|---|---|---|---|---|
| 2048 | 256 | 1 | 4 | 8, 8, 8, 4 |
| 512  | 64  | 1 | 3 | 8, 8, 8 |
| 256  | 64  | 1 | 4 | 4, 4, 4, 4 |
| 64   | 64  | 4 | 3 | 4, 4, 4 |
| 32   | 64  | 16 | 2 | 8, 4 |
| 16   | 64  | 16 | 2 | 4, 4 |
| 4    | 64  | 32 | 2 | 2, 2 |
| 2    | 64  | 64 | 1 | 2 |

This exactly matches the draft's "common" block, verbatim.

### P_SINGLE-only additional entries (lines 322–325, inside the `case P_SINGLE:` block)

| Length | WGS | NT | NumPasses | Radices |
|---|---|---|---|---|
| 4096 | 256 | 1 | 4 | 8, 8, 8, 8 |
| 1024 | 128 | 1 | 4 | 8, 8, 4, 4 |
| 128  | 64  | 4 | 3 | 8, 4, 4 |
| 8    | 64  | 32 | 2 | 4, 2 |

### P_DOUBLE-only additional entries (lines 341–343, inside the `case P_DOUBLE:` block)

| Length | WGS | NT | NumPasses | Radices |
|---|---|---|---|---|
| 1024 | 128 | 1 | 4 | 8, 8, 4, 4 |
| 128  | 64  | 4 | 3 | 8, 8, 2 |
| 8    | 64  | 16 | 3 | 2, 2, 2 |

**IMPORTANT CORRECTION to the draft:** the draft calls 4096/1024/128/8 "single-precision-only
additions." This is only true for **4096** (genuinely absent from the double table — double
precision length 4096 falls through to `DetermineSizes`). **1024 is not single-only** — it is
duplicated verbatim (identical WGS/NT/radices) in the double-precision block too. **128 and 8 are
present in BOTH tables but with DIFFERENT radix decompositions** (128: single=[8,4,4] vs
double=[8,8,2]; 8: single WGS64/NT32/[4,2] vs double WGS64/NT16/[2,2,2]) — they are not "single
only," they simply differ per precision.

**No entries exist for 8192/16384/32768/etc.** in this table — the draft's speculation that there
"may be more entries... e.g. 32768, 16384, 8192" is **false for `KernelCoreSpecs`**. Those larger
power-of-2 sizes are instead handled by an entirely separate mechanism: the **block-compute (SBCC)
column-length switch statement** in `plan.cpp` (see Section 4) — a different table used only to
pick a transpose/column split for `blockCompute` mode, not a Stockham single-kernel radix plan.

**Gating condition (a nuance absent from the draft):** the table is only consulted when
`params.fft_MaxWorkGroupSize >= 256` (`generator.stockham.cpp` line 3023:
`if((params.fft_MaxWorkGroupSize >= 256) && (pRadices != NULL))`), and separately, at the
WGS/NumTransforms level, `FFTGeneratedStockhamAction::initParams()` only trusts the table's
`(wgs, nt)` when `this->plan->envelope.limit_WorkGroupSize >= 256` (line 4586:
`if((t_wgs != 0) && (t_nt != 0) && (this->plan->envelope.limit_WorkGroupSize >= 256))`); otherwise
it calls `DetermineSizes(...)` even for a length that IS in the table. So the specialization table
is not unconditionally authoritative — it's gated behind a device work-group-size capability check.

---

## 2. `DetermineSizes()` — the general heuristic for lengths not covered by the table

**File:** `generator.stockham.cpp`, lines 388–521. Full function read and verified.

### Setup (lines 399–417)
Only primes `{13, 11, 7, 5, 3, 2}` are "supported" for factoring (in that order, largest-first)
```cpp
size_t baseRadix[] = {13,11,7,5,3,2}; // list only supported primes
...
for(size_t r=0; r<baseRadixSize; r++) {
    size_t rad = baseRadix[r];
    size_t e = 1;
    while(!(l%rad)) { l /= rad; e *= rad; }
    primeFactorsExpanded[rad] = e;
}
assert(l == 1); // Makes sure the number is composed of only supported primes
```
`primeFactorsExpanded[p]` ends up holding `p^k` (the full power of `p` dividing `length`), not the
exponent `k` itself. Any length containing a prime factor outside {2,3,5,7,11,13} trips the assert
(i.e., is unsupported by this generator entirely).

### length == 1 special case (lines 392–397): `workGroupSize = 64; numTrans = 64;`

### Pure prime powers (lines 419–450) — verified exactly:
- **Pure power of 2** (`primeFactorsExpanded[2] == length`):
  - `length >= 1024` → `workGroupSize = min(256, MAX_WGS)`, `numTrans = 1`
  - `length == 512` → `workGroupSize = 64`, `numTrans = 1`
  - `length >= 16` (i.e. 16-256 excluding those above) → `workGroupSize = 64`, `numTrans = 256/length`
  - else (length < 16, i.e. 1/2/4/8, though 1/2/4 are already special-cased/table-covered) →
    `workGroupSize = 64`, `numTrans = 128/length`
- **Pure power of 3**: `workGroupSize = (MAX_WGS>=256) ? 243 : 27`; `numTrans = length>=3*wgs ? 1 : (3*wgs)/length`
- **Pure power of 5**: `workGroupSize = (MAX_WGS>=128) ? 125 : 25`; `numTrans = length>=5*wgs ? 1 : (5*wgs)/length`
- **Pure power of 7**: `workGroupSize = 49`; `numTrans = length>=7*wgs ? 1 : (7*wgs)/length`
- **Pure power of 11**: `workGroupSize = 121`; `numTrans = length>=11*wgs ? 1 : (11*wgs)/length`
- **Pure power of 13**: `workGroupSize = 169`; `numTrans = length>=13*wgs ? 1 : (13*wgs)/length`

(The draft did not ask about these in detail but they are the necessary complement to the mixed
case and are reported here verbatim since they gate entry into the mixed-prime `else` branch.)

### Mixed-prime branch (lines 451–520) — the draft's claimed table, checked one by one:

```cpp
size_t leastNumPerWI = 1;
size_t maxWorkGroupSize = MAX_WGS;

if (primeFactorsExpanded[2]*primeFactorsExpanded[3] == length) {
    if (length % 12 == 0) { leastNumPerWI = 12; maxWorkGroupSize = 128; }
    else                  { leastNumPerWI =  6; maxWorkGroupSize = 256; }
} else if (primeFactorsExpanded[2]*primeFactorsExpanded[5] == length) {
    if (length % 20 == 0) { leastNumPerWI = 20; maxWorkGroupSize = 64; }
    else                  { leastNumPerWI = 10; maxWorkGroupSize = 128; }
} else if (primeFactorsExpanded[2]*primeFactorsExpanded[7] == length) {
    leastNumPerWI = 14; maxWorkGroupSize = 64;
} else if (primeFactorsExpanded[3]*primeFactorsExpanded[5] == length) {
    leastNumPerWI = 15; maxWorkGroupSize = 128;
} else if (primeFactorsExpanded[3]*primeFactorsExpanded[7] == length) {
    leastNumPerWI = 21; maxWorkGroupSize = 128;
} else if (primeFactorsExpanded[5]*primeFactorsExpanded[7] == length) {
    leastNumPerWI = 35; maxWorkGroupSize = 64;
} else if (primeFactorsExpanded[2]*primeFactorsExpanded[3]*primeFactorsExpanded[5] == length) {
    leastNumPerWI = 30; maxWorkGroupSize = 64;
} else if (primeFactorsExpanded[2]*primeFactorsExpanded[3]*primeFactorsExpanded[7] == length) {
    leastNumPerWI = 42; maxWorkGroupSize = 60;
} else if (primeFactorsExpanded[2]*primeFactorsExpanded[5]*primeFactorsExpanded[7] == length) {
    leastNumPerWI = 70; maxWorkGroupSize = 36;
} else if (primeFactorsExpanded[3]*primeFactorsExpanded[5]*primeFactorsExpanded[7] == length) {
    leastNumPerWI = 105; maxWorkGroupSize = 24;
} else if (primeFactorsExpanded[2]*primeFactorsExpanded[11] == length) {
    leastNumPerWI = 22; maxWorkGroupSize = 128;
} else if (primeFactorsExpanded[2]*primeFactorsExpanded[13] == length) {
    leastNumPerWI = 26; maxWorkGroupSize = 128;
} else {
    leastNumPerWI = 210; maxWorkGroupSize = 12;
}

if (pr==P_DOUBLE) {
    //leastNumPerWI /= 2;      <-- COMMENTED OUT IN REAL SOURCE
    maxWorkGroupSize /= 2;
}

if (maxWorkGroupSize > MAX_WGS) maxWorkGroupSize = MAX_WGS;
assert (leastNumPerWI > 0 && length % leastNumPerWI == 0);

for (size_t lnpi = leastNumPerWI; lnpi <= length; lnpi += leastNumPerWI) {
    if (length % lnpi != 0) continue;
    if (length / lnpi <= MAX_WGS) { leastNumPerWI = lnpi; break; }
}

numTrans = maxWorkGroupSize / (length / leastNumPerWI);
numTrans = numTrans < 1 ? 1 : numTrans;
workGroupSize = numTrans * (length / leastNumPerWI);
```

**Verification verdict: every mixed-factor rule in the draft is CONFIRMED CORRECT**, including the
2·3, 2·5, 2·7, 3·5, 3·7, 5·7, 2·3·5, 2·3·7, 2·5·7, 3·5·7, 2·11, 2·13, and the `210/12` fallback for
anything not matching any of those patterns (i.e. any product of 3+ distinct primes not in the
explicit list above, e.g. 2·3·11, or any length using all of 2,3,5,7,11,13 together, etc., falls to
`leastNumPerWI=210, maxWorkGroupSize=12`).

**One correction/nuance the draft omitted:** for `P_DOUBLE`, only `maxWorkGroupSize` is halved —
the `leastNumPerWI /= 2` line exists in the source but is **commented out**, so `leastNumPerWI`
(and hence "elements per work-item") is identical between single and double precision for the
mixed-prime branch; only the resulting workgroup-size ceiling differs. A faithful port must not
halve `leastNumPerWI` for double precision.

### General algorithm (the actual "increase elements-per-work-item" loop), stated abstractly:

1. Start from a per-shape `leastNumPerWI` (baseline "elements per work item", e.g. 6 or 12 for a
   2×3 length) and a per-shape `maxWorkGroupSize` ceiling.
2. Halve `maxWorkGroupSize` for double precision (not `leastNumPerWI`).
3. Clamp `maxWorkGroupSize` to the device's `MAX_WGS`.
4. Walk candidate "elements per work item" values `lnpi = leastNumPerWI, 2*leastNumPerWI,
   3*leastNumPerWI, ...` up to `length`. For the first `lnpi` that (a) evenly divides `length` and
   (b) yields `length/lnpi <= MAX_WGS` (i.e., the implied per-transform workgroup size is not too
   large), lock that in as the real `leastNumPerWI`.
5. `numTrans = maxWorkGroupSize / (length/leastNumPerWI)`, floored to at least 1 — i.e., how many
   whole independent transforms of that per-transform-WGS can be packed into one physical
   workgroup up to the `maxWorkGroupSize` budget.
6. `workGroupSize = numTrans * (length/leastNumPerWI)` — the final physical OpenCL workgroup size.

This is a discrete search over multiples of the base "elements per work item," not a continuous
formula — it grows `lnpi` step-by-step until the per-transform workgroup size fits under `MAX_WGS`,
then derives `numTrans`/`workGroupSize` from whatever `lnpi` it landed on.

---

## 3. Radix selection

### 3a. Table-covered lengths
When `GetRadices()` finds the length in `specTable` **and** `params.fft_MaxWorkGroupSize >= 256`
(`generator.stockham.cpp` lines 3018–3052), the radix **sequence and order** are taken verbatim,
in array order, from the `SpecRecord.radices[]` array (e.g. length 2048 always factors as
8→8→8→4 in that exact pass order, never reordered).

### 3b. Fallback radix decomposition (no table entry, or table gated off) — lines 3053–3099
```cpp
size_t cRad[] = {13,11,10,8,7,6,5,4,3,2,1}; // Must be in descending order
...
while(true) {
    size_t rad;
    for(size_t r=0; r<cRadSize; r++) {
        rad = cRad[r];
        if((rad > cnPerWI) || (cnPerWI % rad)) continue;
        if(!(R % rad)) break;
    }
    L = LS * rad; R /= rad;
    radices.push_back(rad);
    passes.push_back(Pass<PR>(pid, length, rad, cnPerWI, L, LS, R, ...));
    pid++; LS *= rad;
    if(R == 1) break;
}
```
**Algorithm:** greedily, pass by pass, pick the **largest** radix from `{13,11,10,8,7,6,5,4,3,2,1}`
(checked in that descending order) such that (a) it does not exceed `cnPerWI` (elements handled per
work item) and evenly divides `cnPerWI`, and (b) it evenly divides the remaining factor `R` of the
length. Repeat until `R == 1`.

**Supported generator radices are therefore {1,2,3,4,5,6,7,8,10,11,13}** — note this is a superset
of the primes used by `DetermineSizes`'s factoring step (`{2,3,5,7,11,13}`): the codegen butterfly
layer additionally hand-supports composite radices **6** and **10** as first-class fused
butterflies (and implicitly 4, 8, 9-via-3·3 etc. through repeated factor pulls), plus a trivial
radix-1 pass. Any length whose prime factorization contains something other than 2,3,5,7,11,13 will
already have failed the `assert(l==1)` in `DetermineSizes`, so the fallback radix loop only ever
needs to decompose numbers built from those six primes (grouped opportunistically into 6s and 10s
when helpful).

---

## 4. Large-1D / multi-pass decomposition

### 4a. `GetMax1DLength` / device threshold
**File:** `generator.stockham.cpp` lines 4670–4690 (`FFTPlan::GetMax1DLengthStockham`):
```cpp
size_t LdsperElement = this->ElementSize();
size_t result = pEnvelope->limit_LocalMemSize / (1 * LdsperElement);
result = FloorPo2(result);
*longest = result;
```
`ElementSize()` (`plan.cpp` line 4836-4839) = `sizeof(std::complex<float>)`=8 bytes (single) or
`sizeof(std::complex<double>)`=16 bytes (double). Default `envelope.limit_LocalMemSize = 32768`
and `envelope.limit_WorkGroupSize = 256` (`plan.cpp` lines 4704–4705, `FFTPlan::SetEnvelope`),
subsequently clamped down to the real device's `CL_DEVICE_LOCAL_MEM_SIZE` /
`CL_DEVICE_MAX_WORK_GROUP_SIZE`. With the defaults this yields:
- **Single precision: Large1DThreshold = FloorPo2(32768/8) = 4096**
- **Double precision: Large1DThreshold = FloorPo2(32768/16) = 2048**

This matches, and explains, the separately hardcoded `threshold = 4096 (single) / 2048 (double)`
seen later in `plan.cpp` line 794-796 for the in-place split heuristic — both derive from the same
LDS-budget-per-element logic.

**So: clFFT switches from a single Stockham kernel to the multi-pass (transpose-based) pipeline
exactly when the requested 1D length exceeds this LDS-derived threshold** (4096 single / 2048
double by default, scaled by actual device LDS size) **or** when `Is1DPossible()` rejects the
length outright even below threshold because its radix mix can't be handled in one kernel.

### 4b. `Is1DPossible` (`plan.h` lines 608–625)
```cpp
static bool Is1DPossible(size_t length, size_t large1DThreshold) {
    if (length > large1DThreshold) return false;
    if ((length%7==0) && (length%5==0) && (length%3==0)) return false;
    // radix 11 & 2 is ok, anything else we cannot do in 1 kernel
    if ((length%11==0) && ((length%13==0)||(length%7==0)||(length%5==0)||(length%3==0))) return false;
    // radix 13 & 2 is ok, anything else we cannot do in 1 kernel
    if ((length%13==0) && ((length%11==0)||(length%7==0)||(length%5==0)||(length%3==0))) return false;
    return true;
}
```
Beyond the pure size threshold, a length is rejected from single-kernel treatment if it mixes
3·5·7, or mixes 11/13 with any of {3,5,7,11,13} (11 and 13 may only combine with pure powers of 2).

### 4c. Splitting N into row/column sizes (`plan.cpp`, `clfftBakePlan`, CLFFT_1D branch)

- If `IsPo2(length)` (lines 643–710): either uses a **block-compute (SBCC)** hardcoded
  column-length switch table (only when strides are unit, out-of-place-friendly, and
  `length <= 262144/PrecisionWidth`) — a **separate hardcoded table from `KernelCoreSpecs`**:

  Single precision (lines 654–665):
  ```
  8192->64, 16384->64, 32768->128, 65536->256, 131072->64, 262144->64, 524288->256, 1048576->256
  ```
  Double precision (lines 669–680):
  ```
  4096->64, 8192->64, 16384->64, 32768->128, 65536->64, 131072->64, 262144->128, 524288->256
  ```
  (Here the number is the **column length** `clLengths[1]`; the row length `clLengths[0]` =
  `length/clLengths[1]`.) — Otherwise it falls to bit-scan logic (`BitScanF`) balancing
  `log2(length)` across two roughly-equal power-of-2 factors, biased by `Large1DThreshold`.

- If **not** a power of 2 (lines 711–770): clFFT walks a **hardcoded ascending list of ~380
  "nicely-factorable" supported composite sizes** (2, 3, 4, ... up to 4096, all products of
  {2,3,5,7,11,13} that fit reasonable radix decompositions) and picks the **largest entry `d`** at
  or below `min(supported.max, Large1DThreshold)`'s "half-power-of-2" starting point such that
  `length % d == 0` **and** `Is1DPossible(d, Large1DThreshold)` — that becomes `clLengths[1]`
  (column size), with `clLengths[0] = length / clLengths[1]` the row size.

- Below both paths, there is also a distinct in-place-with-no-temp-buffer path
  (`split1D_for_inplace`, referenced at line 804) used only when
  `clfftGetRequestLibNoMemAlloc()` is set.

### 4d. The transpose→row-FFT→transpose→row-FFT→transpose pipeline ("4-step"/"6-step" with fused twiddle)

For the general (non-block-compute) large-1D path, `clfftBakePlan` builds and recursively bakes
**five internal sub-plans** (`plan.cpp` lines ~811–1250):

1. **`planTX`** (`CLFFT_2D` transpose): input → temp buffer, reshaping `length` as
   `clLengths[0] × clLengths[1]`.
2. **`planX`** (`CLFFT_1D`, size `clLengths[1]`, batched over `clLengths[0]`): row FFT, temp → temp
   (or → output). `row1Plan->large1D = 0;` (comment: *"twiddling is done in row2"* — meaning this
   first row pass explicitly does NOT apply the large-N twiddle factors).
3. **`planTY`** (`CLFFT_2D` transpose): the **twiddle-fused** transpose — `trans2Plan->large1D =
   fftPlan->length[0];` (line 1051). This nonzero `large1D` is what triggers
   `fft_3StepTwiddle = true` inside the transpose-kernel generator (see 4e below), so the middle
   transpose kernel multiplies each element by `exp(-2πi·k0·k1/N)` (the classic Cooley–Tukey /
   Bailey four-step twiddle) as part of the transpose, rather than in a separate pass.
4. **`planY`** (`CLFFT_1D`, size `clLengths[0]`, in-place): second row FFT.
5. **`planTZ`** (`CLFFT_2D` transpose, `transOutHorizontal = true`): temp → output, restoring the
   original element order.

Each sub-plan is created via `clfftCreateDefaultPlanInternal` and immediately baked via a recursive
`clfftBakePlan(...)` call (lines 978, 1022, 1085, 1134, and the corresponding call for `planTZ`),
confirming this is planned/composed once at bake time, not chosen dynamically at execution time.

### 4e. Where the twiddle multiply actually happens
**File:** `generator.transpose.gcn.cpp`.
- Line 1069-1073: `if (this->plan->large1D != 0) { ...; this->signature.fft_3StepTwiddle = true; ...}`
  — a transpose action turns on 3-step twiddling purely because the plan it was built from
  (`trans2Plan`) had `large1D` set to the *original, undecomposed* FFT length.
- Lines 393-401, 761-763: when `fft_3StepTwiddle` is set, the generator emits a
  `StockhamGenerator::TwiddleTableLarge` table of size `fft_N[0]*fft_N[1]` (i.e. the *full* original
  length) and calls `genTwiddleMath(...)` (defined at line 176) to inject the complex multiply
  directly into the transpose kernel body.

This confirms the draft's premise: clFFT's "3-step" large-1D handling is literally the four-step
Bailey algorithm (transpose / row-FFT / twiddle-fused-transpose / row-FFT / transpose-back), with
the twiddle multiplication fused into the *second* transpose kernel rather than issued as its own
kernel.

---

## 5. Bake semantics — heuristic/codegen, not benchmark-based autotuning

**Confirmed. No timing, benchmarking, or performance-comparison code exists anywhere in the bake
path.** A search for `benchmark`, `autotun`, `clock()`, `QueryPerformance`, `profiling` across
`plan.cpp`, `generator.stockham.cpp`, `transform.cpp`, and `action.transpose.cpp` returned **zero
matches**.

What `clfftBakePlan` (`plan.cpp` line 454 onward) actually does is entirely deterministic and
data/heuristic-driven:
1. Validate/normalize the plan (strip length-1 dims, check strides, check precision support).
2. Compute `Large1DThreshold` via `GetMax1DLength` (pure arithmetic off LDS size, Section 4a).
3. If needed, deterministically decompose the length into a transpose/row-FFT/row-FFT/transpose
   pipeline of sub-plans (Section 4), each of which is separately, recursively baked (deterministic
   recursion, not a search over alternatives).
4. For a directly-generatable Stockham kernel, `FFTGeneratedStockhamAction::initParams()`
   (`generator.stockham.cpp` line 4491) picks `(workGroupSize, numTransforms)` from the
   **hardcoded `KernelCoreSpecs` table** if present and the device's WGS limit allows it, else
   falls back to the **closed-form `DetermineSizes()` heuristic** (Section 2) — again, a single
   deterministic answer, no comparison of alternatives.
5. Radices are selected by direct table lookup (Section 3a) or a deterministic greedy
   largest-divisor loop (Section 3b) — again, exactly one candidate is ever generated.
6. `generateKernel()` (line 4692) stringizes the OpenCL C kernel source and calls `clBuildProgram`.
7. **`fft_binary_lookup.cpp`** provides an on-disk **compiled-binary cache** keyed by an MD5 hash of
   the generated source + compile options + device (`FFTBinaryLookup`, lines 82-320) — this is a
   *build-artifact* cache to avoid recompiling identical kernels, **not** a mechanism that ever
   builds/times multiple candidate kernels and picks the fastest. Only one kernel variant is ever
   generated or compiled per plan configuration.

**Conclusion: `clfftBakePlan` is planning-then-codegen-then-compile of a single, heuristically
predetermined kernel configuration. There is no runtime autotuning/benchmarking anywhere in the
baking process** — this fully confirms the draft's premise in Section 5 of the task.

---

## 6. Places the draft was WRONG or needed correction

1. **"single-precision-only additions: 4096, 1024, 128, 8"** — wrong for **1024** (present
   identically in the double table too) and misleading for **128** and **8** (present in *both*
   tables, but with different radix decompositions per precision, not absent from double). Only
   **4096** is genuinely single-only.
2. **Draft speculated the table might have more entries (e.g. 32768, 16384, 8192)** — confirmed
   **false**: `KernelCoreSpecs` tops out at 4096 (single) / 2048 (common table's max, both
   precisions). Those larger sizes are governed by a **separate, unrelated hardcoded table** — the
   block-compute (SBCC) column-length switch in `plan.cpp` (Section 4c) — which the draft did not
   mention and which uses different semantics (column length for a 2-factor split, not a WGS/NT/
   radix-sequence spec).
3. **Mixed-prime `DetermineSizes` rules**: all of the draft's worked values (12/128, 6/256, 20/64,
   10/128, 14/64, 15/128, 21/128, 35/64, 30/64, 42/60, 70/36, 105/24, 22/128, 26/128, 210/12) are
   **confirmed exactly correct** as `(leastNumPerWI, maxWorkGroupSize)` pairs — no corrections
   needed there.
4. **Undocumented-by-draft nuance**: for double precision, only `maxWorkGroupSize` is halved in
   `DetermineSizes`; the `leastNumPerWI /= 2` halving is present in the source **only as a commented
   -out line** and is NOT active. A port that halves both would diverge from real clFFT behavior.
5. **Undocumented-by-draft gating**: the specialization table is only used when the device's
   work-group-size capability is `>= 256` (checked in two places: `GetRadices` consultation gated
   by `params.fft_MaxWorkGroupSize >= 256`, and the `(wgs,nt)` result gated by
   `envelope.limit_WorkGroupSize >= 256`). Below that, even a table-covered length falls through
   entirely to `DetermineSizes()`.
6. Everything else requested (specialization table core values, general `DetermineSizes` loop
   logic, fallback radix decomposition existence/order, large-1D pipeline structure, bake-is-not-
   autotuning) is **confirmed accurate** against the real source.

---

## 7. Source files actually fetched and read (verbatim, full-file downloads via `curl` from
`https://raw.githubusercontent.com/clMathLibraries/clFFT/master/src/library/<name>`, master branch,
fetched 2026-09-08)

- `generator.stockham.cpp` (154,630 bytes) — `KernelCoreSpecs`/`SpecRecord` table, `DetermineSizes`,
  radix-selection loop (table + fallback), `GetMax1DLengthStockham`, `FFTGeneratedStockhamAction::
  initParams`/`getWorkSizes`/`generateKernel`.
- `generator.stockham.h` (71,684 bytes) — supporting declarations.
- `plan.cpp` (174,642 bytes) — `clfftBakePlan`, large-1D decomposition (`planTX/planX/planTY/planY/
  planTZ`), block-compute column-length tables, `FFTPlan::ElementSize`, `FFTPlan::SetEnvelope`,
  `FFTPlan::GetMax1DLength`.
- `plan.h` (20,470 bytes) — `Is1DPossible`, `FFTPlan` class declarations.
- `transform.cpp` (48,948 bytes) — execution-time use of `GetMax1DLength`/`Is1DPossible`.
- `action.transpose.cpp` (28,218 bytes) — transpose action wrapper.
- `generator.transpose.cpp` (172,439 bytes) — non-GCN transpose generator.
- `generator.transpose.gcn.cpp` (45,399 bytes) — GCN transpose generator, `fft_3StepTwiddle`/
  `genTwiddleMath` (the fused large-1D twiddle logic).
- `fft_binary_lookup.cpp` (757 lines) — on-disk compiled-binary MD5 cache (confirms no autotuning).

Additionally confirmed via GitHub API listing (`api.github.com/repos/clMathLibraries/clFFT/contents
/src/library`) that no `action.stockham.cpp` file exists in the repository (the requested path from
the task brief does not exist; the equivalent logic lives inside `generator.stockham.cpp`'s
`FFTGeneratedStockhamAction` methods instead).
