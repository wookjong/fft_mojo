# clFFT large-1D split selection: deep dive (BitScanF + non-po2 table)

Repository: `github.com/clMathLibraries/clFFT`, branch `master`.
Method: fetched raw files via `raw.githubusercontent.com/clMathLibraries/clFFT/master/src/library/<file>` and
`src/include/<file>`, read them in full/relevant ranges, and cross-checked line numbers by direct grep on the
downloaded copies (stored locally during this research pass). All line numbers below refer to the current
`master` HEAD at the time of research (2026-09-08).

Files fetched and inspected in full or by targeted range:
- `src/library/plan.cpp` (4840 lines total)
- `src/library/plan.h`
- `src/library/private.h`
- `src/library/generator.stockham.cpp`
- `src/library/generator.stockham.h`
- `src/library/fft_binary_lookup.cpp`
- `src/library/lifetime.cpp`
- `src/library/repo.cpp`, `repo.h` (checked, not load-bearing for this question)

---

## 1. `BitScanF` — full implementation and call sites

### 1.1 Definition

`src/library/private.h`, lines 322-336:

```cpp
static inline bool IsPo2 (size_t u) {
	return (u != 0) &&  (0 == (u & (u-1)));
}

template<typename T>
static inline T DivRoundingUp (T a, T b) {
	return (a + (b-1)) / b;
}

static inline size_t BitScanF (size_t n) {
	assert (n != 0);
	unsigned long tmp = 0;
	BSF (& tmp, n);
	return (size_t) tmp;
}
```

`BSF` itself (private.h, lines 58-92) is a thin platform wrapper around the x86 "bit scan forward" instruction /
its GCC equivalent — i.e. **index of the lowest set bit**, not a general log2:

```cpp
#if defined( _MSC_VER )
	#include <intrin.h>
	#if defined( _WIN64 )
		inline void BSF( unsigned long* index, size_t& mask )
		{
			_BitScanForward64( index, mask );
		}
	#else
		inline void BSF( unsigned long* index, size_t& mask )
		{
			_BitScanForward( index, mask );
		}
	#endif
#elif defined( __GNUC__ )
	inline void BSF( unsigned long * index, size_t & mask )
	{
		*index = __builtin_ctz( mask );
	}
#endif
```

So `BitScanF(n)` = **count of trailing zero bits of `n`** (`ctz`). Every call site in the large-1D split logic
only ever passes a power-of-2 value (`fftPlan->length[0]` guarded by an `IsPo2` check, or `Large1DThreshold`,
which is itself always a power of 2 — see §4 and the `BUG_CHECK(IsPo2(Large1DThreshold))` at plan.cpp:640).
For a power-of-2 argument, `ctz(n) == log2(n)` exactly, which is why the source comments read `BitScanF(n)` as
"this is log2(n)" — it is log2 only because the call sites restrict it to powers of two, not because the
function itself computes log2 in general.

There is no other definition of `BitScanF` anywhere in the repository (`src/include`, `src/library`, `src/tests`,
`src/client`, `src/stats` were all grep'd for the token — only the three call sites in `plan.cpp` and the one
definition in `private.h` exist).

### 1.2 Call sites and full surrounding logic — the po2 branch

`src/library/plan.cpp`, the relevant range is `clfftBakePlan`, lines 633-771 (verified directly; the "643-710"
range cited by the prior pass covers only the inner `if(IsPo2(...))` body, not the full branch). Quoting the
complete block, lines 633-710 (po2 case) plus 711-771 (non-po2 case, covered in §2):

```cpp
	case CLFFT_1D:
		{
			if ( !Is1DPossible(fftPlan->length[0], Large1DThreshold) )
			{
				size_t clLengths[] = { 1, 1, 0 };
				size_t in_1d, in_x, count;

				BUG_CHECK (IsPo2 (Large1DThreshold))


				if( IsPo2(fftPlan->length[0]) )
				{
					// Enable block compute under these conditions
					if( (fftPlan->inStride[0] == 1) && (fftPlan->outStride[0] == 1) && !rc
						&& (fftPlan->length[0] <= 262144/PrecisionWidth(fftPlan->precision)) && (fftPlan->length.size() <= 1)
						&& (!clfftGetRequestLibNoMemAlloc() || (fftPlan->placeness == CLFFT_OUTOFPLACE)) )
					{
						fftPlan->blockCompute = true;

						if(1 == PrecisionWidth(fftPlan->precision))
						{
							switch(fftPlan->length[0])
							{
							case 8192:		clLengths[1] = 64;	break;
							case 16384:		clLengths[1] = 64;	break;
							case 32768:		clLengths[1] = 128;	break;
							case 65536:		clLengths[1] = 256;	break;
							case 131072:	clLengths[1] = 64;	break;
							case 262144:	clLengths[1] = 64;	break;
							case 524288:	clLengths[1] = 256; break;
							case 1048576:	clLengths[1] = 256; break;
							default:		assert(false);
							}
						}
						else
						{
							switch(fftPlan->length[0])
							{
							case 4096:		clLengths[1] = 64;	break;
							case 8192:		clLengths[1] = 64;	break;
							case 16384:		clLengths[1] = 64;	break;
							case 32768:		clLengths[1] = 128;	break;
							case 65536:		clLengths[1] = 64;	break;
							case 131072:	clLengths[1] = 64;	break;
							case 262144:	clLengths[1] = 128;	break;
							case 524288:	clLengths[1] = 256; break;
							default:		assert(false);
							}
						}
					}
					else
					{
						if( clfftGetRequestLibNoMemAlloc() && !rc && (fftPlan->placeness == CLFFT_INPLACE) )
						{
							in_x = BitScanF(fftPlan->length[0]);
							in_x /= 2;
							clLengths[1] = (size_t)1 << in_x;
						}
						else if( fftPlan->length[0] > (Large1DThreshold * Large1DThreshold) )
						{
							clLengths[1] = fftPlan->length[0] / Large1DThreshold;
						}
						else
						{
							in_1d = BitScanF (Large1DThreshold);	// this is log2(LARGE1D_THRESHOLD)
							in_x  = BitScanF (fftPlan->length[0]);	// this is log2(length)
							BUG_CHECK (in_1d > 0)
							count = in_x/in_1d;
							if (count*in_1d < in_x)
							{
								count++;
								in_1d = in_x / count;
								if (in_1d * count < in_x) in_1d++;
							}
							clLengths[1] = (size_t)1 << in_1d;
						}
					}
				}
```
(`rc` = `(fftPlan->inputLayout == CLFFT_REAL) || (fftPlan->outputLayout == CLFFT_REAL)`, defined at plan.cpp:494.)

**Complete algorithm for the po2 case** (`length` = `fftPlan->length[0]`, `T` = `Large1DThreshold`, both
powers of 2, `length > T` since we're inside the `!Is1DPossible` branch and `Is1DPossible` already rejects
`length > T`):

1. If block-compute is eligible — contiguous unit strides in and out, not a real-transform, 1-D plan,
   `length <= 262144/PrecisionWidth(precision)` (262144 for single, 131072 for double), and not
   ("no-mem-alloc" mode AND in-place) — **and** `length` is one of the 8 literal sizes in a hardcoded
   per-precision switch table, use that literal table value for `clLengths[1]`. (Full tables quoted above;
   this matches the prior pass's extraction exactly — single: `{8192:64, 16384:64, 32768:128, 65536:256,
   131072:64, 262144:64, 524288:256, 1048576:256}`; double: `{4096:64, 8192:64, 16384:64, 32768:128,
   65536:64, 131072:64, 262144:128, 524288:256}`.) **If block-compute is eligible but the length is not one
   of the 8 listed cases, the code hits `default: assert(false);`** — undefined/unreachable in a correctly
   functioning build; in a release (`NDEBUG`) build `assert` is compiled out and `clLengths[1]` is left at
   whatever it held (unset → the initial value `1`), a latent bug, not a defined algorithm. In practice this
   can't currently happen because the switch's own case list plus the `<=262144/PrecisionWidth` gate are
   consistent for single precision (all 8 single-precision cases are ≤ 262144 — see caveat below) but the
   *double*-precision table lists no entry above 524288 while the gate allows up to 131072, so it's
   self-consistent there too. (One genuine inconsistency found: the single-precision table's own 524288 and
   1048576 cases are **unreachable** through this code path, since the gate requires `length <= 262144` for
   single precision — 524288 and 1048576 both fail that gate and fall to the `else` branch below instead.
   This looks like dead/vestigial code, not something to port.)
2. Otherwise (not block-compute-eligible, i.e. the normal case for `length > 262144` single / `> 131072`
   double, or multi-dimensional plans, or non-unit strides):
   a. If `CLFFT_REQUEST_LIB_NOMEMALLOC` env var is set **and** not a real transform **and** placeness is
      in-place: `clLengths[1] = 1 << (BitScanF(length)/2)`, i.e. `2^(floor(log2(length)/2))` — an exact
      half-split-in-bits, biased toward a smaller second factor when `log2(length)` is odd. (Gated behind an
      opt-in env var — see §4.)
   b. Else if `length > Large1DThreshold^2`: `clLengths[1] = length / Large1DThreshold` (so
      `clLengths[0] = Large1DThreshold` exactly — the row size is pinned to the threshold itself and the
      column size absorbs everything else).
   c. Else (the "balanced" case, `Large1DThreshold < length <= Large1DThreshold^2`): the **exact BitScanF
      bit-balancing algorithm**:
      ```
      in_1d = BitScanF(Large1DThreshold)   // = log2(Large1DThreshold), call it t
      in_x  = BitScanF(length)             // = log2(length), call it x
      count = x / in_1d                    // integer division
      if (count * in_1d < x) {
          count++
          in_1d = x / count                // integer division, re-derive bit-width per factor
          if (in_1d * count < x) in_1d++
      }
      clLengths[1] = 1 << in_1d
      ```
      In closed form: `count = ceil(x / t)` is the number of "large1D-sized" factors needed to cover all `x`
      bits; then `in_1d = ceil(x / count)` is re-derived (bits per factor, rounded up) so that
      `count` factors of `2^in_1d` each cover at least `2^x`. `clLengths[1] = 2^in_1d` and (line 809)
      `clLengths[0] = length / clLengths[1]`. This is not literally "balance across exactly two factors" —
      it is "pick `clLengths[1]` as `2^ceil(x / ceil(x/t))`", which happens to produce a near-square split
      for the direct two-way case in this range (since `x <= 2t` here, `count` is 1 or 2). Concretely, whenever
      `t < x <= 2t` (which is exactly the range this `else` branch is reached in, given `length <= T^2`
      implies `x <= 2t`): `count = ceil(x/t)` is 1 (only if `x<=t`, impossible here since `length>T`) or 2
      (`t < x <= 2t`), so `count=2`, `in_1d = ceil(x/2)`, and `clLengths[1] = 2^ceil(x/2)`,
      `clLengths[0] = 2^floor(x/2)` — i.e. **the two power-of-2 factors that are as close to each other in
      bit-width as possible, with the larger bit-count assigned to `clLengths[1]` when `x` is odd.** This
      confirms and makes exact the prior pass's paraphrase "balancing log2(length) across two roughly-equal
      power-of-2 factors."

`clLengths[0] = fftPlan->length[0] / clLengths[1]` is set once, uniformly, at plan.cpp:809, after either
branch above has set `clLengths[1]`.

---

## 2. The non-power-of-2 "nicely factorable sizes" mechanism

This is a **literal, fully static, hardcoded array of exactly 490 integers** (not ~380 as the prior pass
estimated) — confirmed by parsing the actual source text (regex-extracted all integers between the array's
braces and counted: `len == 490`, strictly ascending, first element `1`, last element `4096`). It is not
generated algorithmically at runtime; it is compiled straight into `plan.cpp` as source-code literals. There is
no separate data file — it lives inline in `clfftBakePlan`.

`src/library/plan.cpp`, lines 711-771 in full (the prior pass's own "711-770" citation is off by one line at
the end but otherwise correct):

```cpp
				else
				{
					// This array must be kept sorted in the ascending order

					size_t supported[] = {	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 24,
											25, 26, 27, 28, 30, 32, 33, 35, 36, 39, 40, 42, 44, 45, 48, 49, 50, 52, 54,
											55, 56, 60, 63, 64, 65, 66, 70, 72, 75, 77, 78, 80, 81, 84, 88, 90, 91, 96,
											98, 99, 100, 104, 105, 108, 110, 112, 117, 120, 121, 125, 126, 128, 130, 132,
											135, 140, 143, 144, 147, 150, 154, 156, 160, 162, 165, 168, 169, 175, 176,
											180, 182, 189, 192, 195, 196, 198, 200, 208, 210, 216, 220, 224, 225, 231,
											234, 240, 242, 243, 245, 250, 252, 256, 260, 264, 270, 273, 275, 280, 286,
											288, 294, 297, 300, 308, 312, 315, 320, 324, 325, 330, 336, 338, 343, 350,
											351, 352, 360, 363, 364, 375, 378, 384, 385, 390, 392, 396, 400, 405, 416,
											420, 429, 432, 440, 441, 448, 450, 455, 462, 468, 480, 484, 486, 490, 495,
											500, 504, 507, 512, 520, 525, 528, 539, 540, 546, 550, 560, 567, 572, 576,
											585, 588, 594, 600, 605, 616, 624, 625, 630, 637, 640, 648, 650, 660, 672,
											675, 676, 686, 693, 700, 702, 704, 715, 720, 726, 728, 729, 735, 750, 756,
											768, 770, 780, 784, 792, 800, 810, 819, 825, 832, 840, 845, 847, 858, 864,
											875, 880, 882, 891, 896, 900, 910, 924, 936, 945, 960, 968, 972, 975, 980,
											990, 1000, 1001, 1008, 1014, 1024, 1029, 1040, 1050, 1053, 1056, 1078, 1080,
											1089, 1092, 1100, 1120, 1125, 1134, 1144, 1152, 1155, 1170, 1176, 1183, 1188,
											1200, 1210, 1215, 1225, 1232, 1248, 1250, 1260, 1274, 1280, 1287, 1296, 1300,
											1320, 1323, 1331, 1344, 1350, 1352, 1365, 1372, 1375, 1386, 1400, 1404, 1408,
											1430, 1440, 1452, 1456, 1458, 1470, 1485, 1500, 1512, 1521, 1536, 1540, 1560,
											1568, 1573, 1575, 1584, 1600, 1617, 1620, 1625, 1638, 1650, 1664, 1680, 1690,
											1694, 1701, 1715, 1716, 1728, 1750, 1755, 1760, 1764, 1782, 1792, 1800, 1815,
											1820, 1848, 1859, 1872, 1875, 1890, 1911, 1920, 1925, 1936, 1944, 1950, 1960,
											1980, 2000, 2002, 2016, 2025, 2028, 2048, 2058, 2079, 2080, 2100, 2106, 2112,
											2145, 2156, 2160, 2178, 2184, 2187, 2197, 2200, 2205, 2240, 2250, 2268, 2275,
											2288, 2304, 2310, 2340, 2352, 2366, 2376, 2400, 2401, 2420, 2430, 2450, 2457,
											2464, 2475, 2496, 2500, 2520, 2535, 2541, 2548, 2560, 2574, 2592, 2600, 2625,
											2640, 2646, 2662, 2673, 2688, 2695, 2700, 2704, 2730, 2744, 2750, 2772, 2800,
											2808, 2816, 2835, 2860, 2880, 2904, 2912, 2916, 2925, 2940, 2970, 3000, 3003,
											3024, 3025, 3042, 3072, 3080, 3087, 3120, 3125, 3136, 3146, 3150, 3159, 3168,
											3185, 3200, 3234, 3240, 3250, 3267, 3276, 3300, 3328, 3360, 3375, 3380, 3388,
											3402, 3430, 3432, 3456, 3465, 3500, 3510, 3520, 3528, 3549, 3564, 3575, 3584,
											3600, 3630, 3640, 3645, 3675, 3696, 3718, 3744, 3750, 3773, 3780, 3822, 3840,
											3850, 3861, 3872, 3888, 3900, 3920, 3960, 3969, 3993, 4000, 4004, 4032, 4050,
											4056, 4095, 4096 };

					size_t lenSupported = sizeof(supported)/sizeof(supported[0]);
					size_t maxFactoredLength = (supported[lenSupported-1] < Large1DThreshold) ? supported[lenSupported-1] : Large1DThreshold;

					size_t halfPowerLength = (size_t)1 << ( (StockhamGenerator::CeilPo2(fftPlan->length[0]) + 1) / 2 );
					size_t factoredLengthStart =  (halfPowerLength < maxFactoredLength) ? halfPowerLength : maxFactoredLength;

					size_t indexStart = 0;
					while(supported[indexStart] < factoredLengthStart) indexStart++;

					for(size_t i = indexStart; i >= 1; i--)
					{
						if( fftPlan->length[0] % supported[i] == 0 )
						{
							if (Is1DPossible(supported[i], Large1DThreshold))
							{
								clLengths[1] = supported[i];
								break;
							}
						}
					}
				}
```

This directly disproves the previous pass's hedge about "generated on the fly from a factorability check" — it
is a **literal static array**, checked with plain `%` divisibility, not a primality/factorization-into-{2,3,5,7,11,13}
test. (Every entry in the array does in fact factor into {2,3,5,7,11,13} only — but the code never computes
that; it just tests `length % supported[i] == 0` against the fixed list.)

**Exact selection loop** (correcting the prior pass's guess of "largest d dividing length, d factors completely
into supported primes, and Is1DPossible(d, threshold)"):

1. `maxFactoredLength = min(supported[489] /* = 4096 */, Large1DThreshold)`.
2. `halfPowerLength = 2 ^ ceil((CeilPo2(length) + 1) / 2)` — i.e. roughly `2^(ceil(log2(length)/2))`, an
   upper-bound estimate of `sqrt(length)` rounded to the next power-of-2-ish granularity via the specific
   integer formula `(CeilPo2(length)+1)/2` (integer division, so this is `ceil((CeilPo2(length)+1)/2)`... more
   precisely it's `1 << ((CeilPo2(n)+1)/2)` with C++ integer truncation, not a ceiling of a real division —
   quoted verbatim above).
3. `factoredLengthStart = min(halfPowerLength, maxFactoredLength)` — the starting search point, an
   approximation of "the factor closest to sqrt(length), capped at the table's/threshold's max."
4. `indexStart` = index of the **first** array entry `>= factoredLengthStart` (linear scan upward from index 0).
5. **The scan then walks DOWNWARD** from `indexStart` to `1` inclusive (`for (i = indexStart; i >= 1; i--)`),
   and returns the **first** (i.e., largest-valued, since the array is ascending and we're decrementing the
   index) entry `supported[i]` such that:
   - `length % supported[i] == 0` (it evenly divides the original length), and
   - `Is1DPossible(supported[i], Large1DThreshold)` is true (the found divisor is itself directly Stockham-computable in one kernel pass, i.e. `supported[i] <= Large1DThreshold` and it isn't one of the disallowed radix-11/13-mixed-with-{3,5,7} combinations — see `Is1DPossible`, `src/library/plan.h` lines 608-624, quoted below).
   - **Break on first match** — the loop stops at the first `i` (walking down) that satisfies both conditions, so it is the **largest supported divisor of `length` that is ≤ the sqrt-ish starting point AND independently satisfies `Is1DPossible`.**
6. If no such `i` in `[1, indexStart]` satisfies both conditions, the loop finishes with no `break`, and
   `clLengths[1]` **silently keeps its initial value of `1`** (set at plan.cpp:637,
   `size_t clLengths[] = { 1, 1, 0 };`) — see §4/§5 for why this matters (the prime-length test case).

`Is1DPossible`, `src/library/plan.h`, lines 608-624:

```cpp
static bool Is1DPossible(size_t length, size_t large1DThreshold)
{
	if (length > large1DThreshold)
		return false;

	if ( (length%7 == 0) && (length%5 == 0) && (length%3 == 0) )
		return false;

	// radix 11 & 2 is ok, anything else we cannot do in 1 kernel
	if ( (length % 11 == 0) && ((length % 13 == 0) || (length % 7 == 0) || (length % 5 == 0) || (length % 3 == 0)) )
		return false;
	
	// radix 13 & 2 is ok, anything else we cannot do in 1 kernel
	if ( (length % 13 == 0) && ((length % 11 == 0) || (length % 7 == 0) || (length % 5 == 0) || (length % 3 == 0)) )
		return false;

	return true;
}
```

`CeilPo2`/`FloorPo2` (`src/library/generator.stockham.h`, lines 90-111):

```cpp
	inline size_t CeilPo2 (size_t n)
	{
		size_t v = 1, t = 0;
		while(v < n)
		{
			v <<= 1;
			t++;
		}
		return t;
	}

	inline size_t FloorPo2 (size_t n)
	{
		size_t tmp;
		while (0 != (tmp = n & (n-1)))
			n = tmp;
		return n;
	}
```

Immediately after the po2/non-po2 branches (plan.cpp lines 772-807), there is a large commented-out block of
special-cased lengths (10000, 100000, 10000000, 100000000, 1000000000, 3099363912, 39366, 78732, 354294 —
these are dead/disabled, wrapped in `/* ... */`) and then a live special path gated by
`clfftGetRequestLibNoMemAlloc() && placeness==CLFFT_INPLACE && inputLayout==outputLayout && length > threshold`
(threshold = 4096 single / 2048 double, plan.cpp:794-807) that overrides `clLengths[1]` using
`split1D_for_inplace` (plan.cpp lines 38-178, a radix-{2,3,5}-aware recursive splitter for in-place transpose
friendliness). This path is **off by default** (see §4).

---

## 3. Tie-breaking specifics

- **Po2 case**: no search/tie-break is needed — the algorithm is a closed-form deterministic computation
  (block-compute table lookup, or the `length/Threshold` formula, or the `BitScanF` bit-balance formula). There
  is exactly one output for a given `(length, Large1DThreshold, precision, strides, placeness, rc, dims)`
  tuple. Where it does balance two factors (the `else` sub-branch, §1.2c), the exact rule is: split
  `x = log2(length)` bits into two power-of-2 factors whose bit-widths are `floor(x/2)` and `ceil(x/2)`, with
  the **larger** bit-width (hence larger value) assigned to `clLengths[1]` when `x` is odd.

- **Non-po2 case**: the tie-break is exactly the loop in §2 — **not** "largest divisor overall," but "largest
  entry in the fixed 490-value ascending table that is `<=` a sqrt-derived starting index, divides `length`,
  and independently passes the same `Is1DPossible` single-kernel-feasibility check used for the top-level
  gate." Concretely this is a **downward linear scan from a computed starting index, first-match-wins**, i.e.
  effectively "largest qualifying supported[i] with `i <= indexStart`." It is not searching upward, and it does
  not consider `supported[0] == 1` as a fallback candidate explicitly (the loop bound is `i >= 1`), though since
  `clLengths[1]` is pre-initialized to `1` anyway this has no behavioral effect — a total-failure case degrades
  to `clLengths[1] = 1` regardless.

---

## 4. Explicit determination: recoverable vs. device-dependent vs. unrecoverable

**A. Fully recoverable as a pure function of `length` and a chosen `Large1DThreshold`:**
- The `IsPo2(length)` dispatch.
- The block-compute literal-table lookup (both precisions), *given* that block-compute eligibility holds
  (contiguous strides, complex-to-complex, 1-D, in the precision-scaled size window) — this table is pure data,
  no device dependency once you know precision and length.
- The `BitScanF` bit-balancing algorithm for the po2 "else" branch (§1.2 b/c) — 100% pure integer arithmetic on
  `length` and `Large1DThreshold`. **This is the part your background note flagged as unextracted; it is now
  fully specified above and is completely portable.**
- The 490-entry `supported[]` table and its exact downward-scan selection loop for the non-po2 branch (§2) —
  also 100% pure integer arithmetic and static data once `Large1DThreshold` is known. **This is also now fully
  specified and portable.**
- `Is1DPossible`, `CeilPo2`, `FloorPo2`, `PrecisionWidth`, `ElementSize` — all pure functions of their inputs,
  no device dependency.

**B. Depends on a runtime device-capability value that this logic treats as an opaque input, but which is
computed the SAME way every time given the same device (so it is recoverable *given* a real or assumed device
LDS size, just not derivable from `length` alone):**
- `Large1DThreshold` itself. It is **not** a compile-time constant "~4096" — it is computed once per plan by
  `FFTPlan::GetMax1DLengthStockham` (`src/library/generator.stockham.cpp`, lines 4670-4690):
  ```cpp
  clfftStatus FFTPlan::GetMax1DLengthStockham (size_t * longest) const
  {
  	const FFTEnvelope * pEnvelope = NULL;
  	OPENCL_V(this->GetEnvelope (& pEnvelope), _T("GetEnvelope failed"));
  	BUG_CHECK (NULL != pEnvelope);
  	ARG_CHECK (NULL != longest)
  	size_t LdsperElement = this->ElementSize();
  	size_t result = pEnvelope->limit_LocalMemSize / (1 * LdsperElement);
  	result = FloorPo2 (result);
  	*longest = result;
  	return CLFFT_SUCCESS;
  }
  ```
  `ElementSize()` (plan.cpp:4836-4839) is `sizeof(std::complex<double>)==16` for double, `sizeof(std::complex<float>)==8`
  for single — pure and precision-only. But `pEnvelope->limit_LocalMemSize` comes from `FFTPlan::SetEnvelope`
  (plan.cpp lines 4651-4744), which **defaults to 32768 bytes and then takes the minimum with the real
  `CL_DEVICE_LOCAL_MEM_SIZE` queried via `clGetDeviceInfo` across every device in the OpenCL context**
  (plan.cpp:4704, 4720-4722):
  ```cpp
  envelope.limit_LocalMemSize  = 32768;
  ...
  OPENCL_V( ::clGetDeviceInfo( devId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof( cl_ulong ), &memsize, NULL ), ... );
  envelope.limit_LocalMemSize = std::min<size_t> (envelope.limit_LocalMemSize, memsize);
  ```
  So **`Large1DThreshold = FloorPo2( min(32768, real_device_LDS_bytes) / ElementSize )`.** For any device with
  `CL_DEVICE_LOCAL_MEM_SIZE >= 32768` (essentially all discrete/integrated GPUs clFFT targets — AMD GCN parts
  are 32-64 KB, NVIDIA and Intel iGPUs are typically 32-164 KB), the `min` is 32768 and the threshold is a
  **fixed, device-independent** `FloorPo2(32768/8) = 4096` for single precision and `FloorPo2(32768/16) = 2048`
  for double precision — this is exactly the "~4096" figure your background note already had, now derived
  rather than assumed. **Only on a hypothetical/unusual device reporting `CL_DEVICE_LOCAL_MEM_SIZE < 32768`
  would the threshold differ**, and in that case it is still perfectly recoverable — just requires knowing (or
  assuming) that one device number. There is no other hidden device dependency: `limit_WorkGroupSize`,
  `limit_Dimensions`, `limit_Size[]` are also queried here but are **not** used anywhere in the split-selection
  code (§1, §2) — only `limit_LocalMemSize` feeds `Large1DThreshold`.
- Practical conclusion: **given a stated/assumed `Large1DThreshold` (4096 for single / 2048 for double is the
  correct value for essentially every real GPU), the entire split algorithm — po2 and non-po2 — is a 100% pure,
  byte-for-byte-reproducible function of `length`, precision, and the handful of plan flags (`placeness`,
  `inputLayout`/`outputLayout`, strides, `dim`).** No closed-source or opaque logic exists anywhere in this
  path; everything is visible in the public `master` source. The "device dependency" reduces to one integer
  (`CL_DEVICE_LOCAL_MEM_SIZE`) whose real-world value is overwhelmingly 32768-or-more, making 4096/2048 the
  correct value to hardcode for a port targeting typical hardware, with the caveat documented.

**C. Additional non-default runtime-flag dependencies** (not device queries, but environment/plan-state
switches that change which formula fires — all discoverable, all off by default):
- `CLFFT_REQUEST_LIB_NOMEMALLOC` environment variable (checked via `getenv`, `src/library/fft_binary_lookup.cpp`
  lines 65-79): **defaults to `false`** (`static bool request_nomemalloc(false);`, only flipped by
  `clfftInitRequestLibNoMemAlloc()` if the env var is set at all, called once from `lifetime.cpp:47`). When
  false (the normal/default case), the `in_x=BitScanF(length)/2` half-split branch (§1.2a), the
  `split1D_for_inplace` override (plan.cpp:794-807), and several transpose-strategy branches are all inert.
- `placeness` (in-place vs out-of-place) and `rc` (real-transform flag) gate the block-compute eligibility and
  the no-mem-alloc branches, but are plan-construction inputs the caller controls, not device state — fully
  known/specifiable, not "unrecoverable."

**D. Genuinely broken / not a well-defined algorithm for some inputs (not "unrecoverable from source" but
"the source itself does not handle this input correctly"):** See §5's discussion of `1000003`. When `length`
is not divisible by any of the 490 supported values in the search range (e.g., large primes, or composites
whose only divisors above ~sqrt(length)-ish are outside the ≤4096 table), the non-po2 loop falls through with
**no match**, `clLengths[1]` stays at its default-initialized `1`, and `clLengths[0] = length` — i.e., the
"split" is a no-op that reproduces the original unsplittable length. Tracing forward (plan.cpp:1091,
`clfftCreateDefaultPlanInternal(&fftPlan->planY, ..., CLFFT_1D, &clLengths[0])` followed by
`clfftBakePlan(fftPlan->planY, ...)` at plan.cpp:1134) shows this recurses into `clfftBakePlan` with the
**exact same length and the same computed threshold**, which will again fail `Is1DPossible`, again fail to find
a divisor, and recurse again — **unbounded recursion (stack overflow) for a real prime length above the
threshold**, not a graceful `CLFFT_NOTIMPLEMENTED`. There is no supported-radix/factorability pre-check
anywhere earlier in `clfftBakePlan` that would catch this (grepped for `NOTIMPLEMENTED` — the only ones in this
area are unrelated: `CLFFT_TRANSPOSED_NOTIMPLEMENTED` for a transpose+3D combination, and
`SP_MAX_LEN`/`DP_MAX_LEN` checks that only apply to real-input/real-output transforms). **This is a genuine gap
in clFFT itself, not a gap in this research** — the real library does not have a defined, safe answer for such
lengths; porting it faithfully means reproducing the same failure mode (or, more sensibly, adding your own
guard the original code lacks).

---

## 5. Worked examples

Assumptions used (stated explicitly per §4): complex-to-complex, out-of-place, unit strides, single
`CLFFT_1D` plan (`length.size()==1`), `CLFFT_REQUEST_LIB_NOMEMALLOC` unset (default), single precision unless
noted, and `Large1DThreshold = 4096` (the value that holds for any device with `CL_DEVICE_LOCAL_MEM_SIZE >=
32768`, which is essentially all real GPUs — see §4B for the derivation and the caveat).

All values below were computed by directly transcribing the extracted algorithm (§1, §2) into a short Python
check against the literal 490-entry table extracted from the live source (not re-derived by hand), so they
are exact reproductions of what the C++ would compute for these inputs/assumptions.

| length | Is1DPossible(length,4096)? | po2? | branch taken | clLengths[0] | clLengths[1] | check |
|---|---|---|---|---|---|---|
| 8192 | No (>4096) | Yes | block-compute table (single, case 8192) | 128 | 64 | 128×64=8192 |
| 16384 | No | Yes | block-compute table (single, case 16384) | 256 | 64 | 256×64=16384 |
| 100000 | No | No | supported[] downward scan; `indexStart=168` (`supported[168]=512`), first hit at `i=165` → `supported[165]=500` | 200 | 500 | 200×500=100000 |
| 1000003 | No | No (1000003 is prime) | supported[] downward scan from `indexStart=245` (`supported[245]=1024`) down to `i=1`: **no divisor found** (prime) | 1000003 | 1 (unchanged default) | **degenerate — see §4D; real clFFT recurses into `clfftBakePlan` on the same length/threshold and does not terminate normally** |
| 5040 | No | No | supported[] scan, `indexStart=72` (`supported[72]=128`), hit at `i=71` → `supported[71]=126` | 40 | 126 | 40×126=5040 |
| 45360 (=2⁴·3⁴·5·7) | No | No | supported[] scan, `indexStart=112` (`supported[112]=256`), hit at `i=111` → `supported[111]=252` | 180 | 252 | 180×252=45360 |
| 262144 | No | Yes | block-compute table (single, case 262144) | 4096 | 64 | 4096×64=262144 |
| 1048576 | No | Yes, but `>262144` so block-compute gate fails | `BitScanF` balance: `x=log2(1048576)=20`, `t=log2(4096)=12`, `count=ceil(20/12)=2`, `in_1d=ceil(20/2)=10` | 1024 | 1024 | 1024×1024=1048576 |

Citations for each row: block-compute rows → plan.cpp lines 646-666 (single-precision switch); non-po2 rows →
plan.cpp lines 711-770 + the `supported[]` array lines 715-749; `1048576` → plan.cpp lines 683, 691, 695-708
(the `else` chain — `length[0] > Threshold` at 646's gate check fails since `262144/PrecisionWidth(single)=262144
< 1048576`, so it falls past block-compute into the `BitScanF`-balance formula at 695-708 since
`1048576 <= 4096*4096=16777216`).

**Note on 1000003**: this is not a case where "the algorithm produces an answer we couldn't extract" — the
algorithm's own logic, faithfully read, produces **no valid split** for this input under the real clFFT source.
This is disclosed as a limitation of clFFT itself (§4D), not a gap in this research.

---

## 6. Confidence / not-independently-verified

**High confidence (read directly from live master source, byte-for-byte, at the cited lines):**
- `BitScanF` definition and semantics (ctz), private.h:322-336, 58-92.
- The full po2 branch (block-compute table, `BitScanF`-balance formula, `length/Threshold` formula), plan.cpp:633-710.
- The full non-po2 branch including the literal 490-entry `supported[]` array and the exact downward-scan
  selection loop, plan.cpp:711-771.
- `Is1DPossible`, plan.h:608-624.
- `CeilPo2`/`FloorPo2`, generator.stockham.h:90-111.
- `GetMax1DLengthStockham` and the `Large1DThreshold` derivation chain (`ElementSize`, `SetEnvelope`,
  `FFTEnvelope`), generator.stockham.cpp:4670-4690, plan.cpp:4651-4744, 4836-4839, plan.h:375-394.
- `clfftGetRequestLibNoMemAlloc` defaulting to false via `getenv`, fft_binary_lookup.cpp:65-79.
- The recursive `clfftBakePlan(fftPlan->planY, ...)` call chain that turns an unsplittable length into
  unbounded recursion, plan.cpp:1091, 1134 (and the parallel real/complex-layout variants at ~1382-1416,
  1544-1592, 1776-1827, 2115-2212).

**Medium confidence / worth an independent double-check if this is going into production:**
- I did not build or run clFFT itself against a real OpenCL device to confirm the `1000003` prime case
  actually crashes at runtime rather than being caught by some check elsewhere in the call graph I didn't trace
  (e.g. inside `clfftCreateDefaultPlanInternal`, `FFTRepo`, or a length-validation step in `clfftCreatePlan`
  before `clfftBakePlan` is ever reached). The **static-source** conclusion (no divisor found → default
  `clLengths[1]=1` → recursion into the same state) is solid; whether some earlier, unexamined layer of the
  public API rejects egregiously prime lengths before this point was not exhaustively verified across every
  file in `src/library` (e.g., `clfftCreatePlan` proper in `transform.cpp`/`plan.cpp` outside the `clfftBakePlan`
  function was not fully read line-by-line).
- The exact reachability analysis of the single-precision block-compute table's `524288`/`1048576` entries
  being "dead code" is based on the single 262144/PrecisionWidth gate found at plan.cpp:647 and the absence of
  any other `blockCompute = true` assignment path feeding this specific switch (verified by grepping all
  `blockCompute` assignments in plan.cpp — only one, at line 650, feeds this switch; the other four assignments
  at lines 1967, 2098, 2145, and their neighbors are for the row/column sub-plans of already-split 2-D
  problems, a different code path with its own separate logic not covered by this research question). This is
  presented as a well-supported observation, not an exhaustively proven claim about every possible call path
  into this switch.
- I assumed "typical GPU" for `Large1DThreshold=4096` (single)/`2048` (double). This is correct for the
  overwhelming majority of real devices but is explicitly a per-device value in the real library, as documented
  in §4B.
