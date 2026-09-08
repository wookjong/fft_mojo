# rocFFT Kernel-Tuning Search: Source-Verified Report

**Repository (current, live):** `github.com/ROCm/rocm-libraries`, monorepo path `projects/rocfft/`
(the old standalone `github.com/ROCm/rocFFT` repo is now marked `[DEPRECATED] Moved to ROCm/rocm-libraries repo`).

**Branch/commit snapshot used for this report:** `develop` @ commit `bee97df517907c771de17189cb867d3c401285ae` (fetched 2026-09-08 via `raw.githubusercontent.com/ROCm/rocm-libraries/develop/...` and the GitHub Contents API).

All file paths below are relative to `projects/rocfft/` inside that monorepo, e.g. `library/src/tuning_kernel_tuner.cpp`.

---

## 1. KernelConfig — the real, complete field set

Two related structs matter. `KernelConfig` is the tunable-parameter struct; `FMKey`/`FMKeyBase` is the cache/lookup key that *embeds* a `KernelConfig`.

**File:** `library/src/include/function_map_key.h`, `struct KernelConfig` (lines 34–~160).

```cpp
struct KernelConfig
{
    bool                use_3steps_large_twd  = false;
    bool                half_lds              = false;
    bool                direct_to_from_reg    = false;
    bool                intrinsic_buffer_inst = false;
    unsigned int        transforms_per_block  = 0;
    int                 workgroup_size        = 0;
    std::array<int, 2>  threads_per_transform = {0, 0};
    std::vector<size_t> factors               = {0};
    // above data is what we can tune
    //
    // the followings are other information of this kernel.
    // not tunable values, they come from the tuned problem nodes.
    EmbeddedType ebType = EmbeddedType::NONE;
    int direction = -1;
    int               static_dim = 0;
    PlacementCode     placement  = PC_UNSET;
    rocfft_array_type iAryType   = rocfft_array_type_complex_interleaved;
    rocfft_array_type oAryType   = rocfft_array_type_complex_interleaved;
};
```

Notes / corrections vs. the draft:
- `threads_per_transform` is `std::array<int,2>`, not a scalar — slot `[0]` is used for 1D/normal kernels (`[1]` left `0`); both slots are used for 2D-single kernels (`Supported2DKernelConfigs`, see §9) where `[0]`/`[1]` are TPT for dim0/dim1 respectively.
- `factors` is the **ordered** factor/radix sequence (order matters — it's literally the Stockham pass order), not just a multiset.
- The struct also carries non-tunable, context-derived fields (`ebType`, `direction`, `static_dim`, `placement`, `iAryType`, `oAryType`) needed later for AOT/RTC kernel generation but not searched over.
- The comparison / hashing (`operator==`, `operator<`, `std::hash` via `SimpleHash` on `FMKey`) is done over the *tunable* subset (`use_3steps_large_twd, half_lds, direct_to_from_reg, intrinsic_buffer_inst, transforms_per_block, workgroup_size, threads_per_transform, factors`) for `KernelConfig::operator==/<`, and over `(lengths, precision, scheme, sbrcTrans, kernel_config, gcn_arch_name)` for `FMKey`.

**`FMKeyBase`/`FMKey`** (same file, lines 288–403) — the actual cache key wrapping a `KernelConfig`:

```cpp
struct FMKeyBase {
    size_t lds_size_bytes = 0;        // NOT a key field (informational only)
    std::array<size_t, 3> lengths;
    rocfft_precision precision;
    ComputeScheme    scheme;
    std::string      gcn_arch_name;
};
struct FMKey : public FMKeyBase {
    SBRC_TRANSPOSE_TYPE sbrcTrans     = NONE;
    KernelConfig        kernel_config = KernelConfig::EmptyConfig();
};
```

There is also `PPFMKey` (partial-pass kernels, `kernel_config_1`/`kernel_config_2`, plus `transform_type`, `batch_low`, `batch_high`) used for the 3D partial-pass scheme — not part of the "core" search space but present in the same header.

**Confirmed:** the draft's list of dimensions (factors/radix order, threads_per_transform, transforms_per_block, workgroup_size, half_lds, direct_to_from_reg, intrinsic_buffer_inst, use_3steps_large_twd) is correct and complete for the *tunable* portion of `KernelConfig`. The one correction is that `threads_per_transform` is a 2-slot array, not scalar, to support 2D-single kernels.

---

## 2. Supported factor set

**File:** `library/src/tuning_kernel_tuner.cpp`, line 39:

```cpp
static const std::vector<size_t> supported_factors = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17};
```

Identical constant also appears independently in the standalone single-kernel tuner `library/src/rocfft_kernel_config_search.cpp` line 49 (`static const std::vector<unsigned int> supported_factors = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17};`).

**Verified exactly as the draft claims: {2,3,4,5,6,7,8,9,10,11,13,16,17}.** No 12, 14, 15, or other values are present in either copy of the constant.

---

## 3. `Factorize(length)` — real algorithm

**File:** `library/src/tuning_kernel_tuner.cpp`, function `Factorize` (lines 108–134):

```cpp
std::set<std::vector<size_t>> Factorize(size_t length)
{
    std::set<std::vector<size_t>> ret;
    for(auto factor : supported_factors)
    {
        if(length % factor == 0)
        {
            size_t remain = length / factor;
            if(remain == 1)
                ret.insert({factor});
            else
            {
                auto remain_factorization = Factorize(remain);
                for(auto& remain_factors : remain_factorization)
                {
                    std::vector<size_t> factors{factor};
                    std::copy(remain_factors.begin(), remain_factors.end(), std::back_inserter(factors));
                    std::sort(factors.begin(), factors.end());
                    ret.insert(factors);
                }
            }
        }
    }
    return ret;
}
```

Algorithm confirmed: recursive/exhaustive over the fixed `supported_factors` list; for every factor `f` that evenly divides `length`, recurse on `length/f`, prepend `f`, sort the resulting vector, and insert into a `std::set` (so factorizations are de-duplicated as **sorted multisets** — order is stripped at this stage; orderings/permutations are reintroduced later, in the permutation loop of `SupportedKernelConfigs` for phase 0, or `GetAllFactorizationsForPhase1` for phase 1). Lengths that have no factor in `supported_factors` (e.g. large primes) simply produce an empty factorization set — this is exactly the "Prime number" case flagged later in `rocfft_offline_tuner.cpp` (`"This fft problem hasn't been supported yet. (Prime number or 2D-Single)"`).

---

## 4. `GetMaxRadicesSize()`

**File:** `library/src/tuning_kernel_tuner.cpp`, lines 136–148:

```cpp
size_t GetMaxRadicesSize(const std::set<std::vector<size_t>>& all_factors_set)
{
    size_t min_size = TWIDDLES_MAX_RADICES + 1;
    for(auto factors : all_factors_set)
        min_size = std::min(factors.size(), min_size);
    // don't try kernels with too many radices
    return min_size + 2;
    // return min_size + 3;
}
```

`TWIDDLES_MAX_RADICES = 8` (`library/src/include/twiddles.h` line 30), so the initial `min_size` is seeded at 9 before being pulled down by the actual shortest factorization found for the length.

**Confirmed exactly:** `max_radix_count = (minimum factor-count among all factorizations of the length) + 2`. There is a commented-out alternative (`+3`) directly beneath it, left in as a historical/tuning knob, not active code — worth noting for anyone porting this, since it shows the margin (`+2` vs `+3`) was itself tuned empirically. This cap is applied in `SupportedKernelConfigs` only when `is_phase0` is true (`if(is_phase0 && (factorization.size() > max_radices_size)) continue;`, line 523) — i.e. only during Phase 0's own factorization enumeration, not to the phase-1 permutation set (which is derived from Phase-0 survivors and therefore already respects the cap indirectly). There's also a hard-coded special case: `if(length == 336) --max_radices_size;` (line 499–500) — an explicit, length-specific exception "to avoid 336 from expanding to 5 factors."

`Supported2DKernelConfigs` (§9) applies the analogous `max_radices_size_0`/`max_radices_size_1` caps per-dimension, unconditionally (not gated on phase), lines 365 and 377.

---

## 5. `SupportedThreadsPerTransform()`

**File:** `library/src/tuning_kernel_tuner.cpp`, lines 150–186 (`PowerSet` + `SupportedThreadsPerTransform`):

```cpp
std::set<std::vector<size_t>> PowerSet(std::vector<size_t>::const_iterator begin,
                                       std::vector<size_t>::const_iterator end)
{
    std::set<std::vector<size_t>> ret;
    if(std::distance(begin, end) == 1) { ret.insert({*begin}); ret.insert({}); }
    else
    {
        auto remain = PowerSet(begin + 1, end);
        for(auto r : remain) { ret.insert(r); r.push_back(*begin); ret.insert(r); }
    }
    return ret;
}

std::set<size_t> SupportedThreadsPerTransform(const std::vector<size_t>& factorization)
{
    std::set<size_t> tpts;
    auto tpt_candidates = PowerSet(factorization.begin(), factorization.end());
    for(auto tpt : tpt_candidates)
    {
        if(tpt.empty()) continue;
        tpts.insert(product(tpt.begin(), tpt.end()));
    }
    return tpts;
}
```

**Confirmed exactly as drafted:** it computes the full power set of the factorization vector (2^n subsets, n = number of factors), takes the product of each non-empty subset, and the resulting set of distinct products is the TPT candidate set. (Identical logic, `unsigned int` variant, exists in the standalone tool `rocfft_kernel_config_search.cpp` lines 84–120, called `power_set`/`supported_threads_per_transform`.)

---

## 6. `GetUtilizationRate()` and rejection rule

**File:** `library/src/tuning_kernel_tuner.cpp`, lines 188–204:

```cpp
std::vector<double> GetUtilizationRate(size_t length, const std::vector<size_t>& factors, size_t tpt)
{
    std::vector<double> ret; // [height_1, height_2,..., height_n, average]
    double util_rate = 0;
    for(auto width : factors)
    {
        double height = static_cast<double>(length) / width / tpt;
        ret.push_back(height);
        util_rate += height;
    }
    util_rate /= factors.size();
    ret.push_back(util_rate);
    return ret;
}
```

Rejection site, `SupportedKernelConfigs`, lines 533–543:

```cpp
for(auto tpt = tpts.begin(), last = tpts.end(); tpt != last;)
{
    std::vector<double> util_rates = GetUtilizationRate(length, factorization, *tpt);
    auto max_rates = std::max_element(util_rates.begin(), util_rates.end());
    auto avg_rate  = util_rates.back();
    // if average rate < 1.0 or > 8.0 , or any of heights > 8.0, then it's bad
    if(avg_rate < 1.0 || *max_rates > 8.0)
        tpts_with_bad_util_rate.insert(*tpt);
    ++tpt;
}
```

**Confirmed exactly:** `height_i = length / factor_i / tpt` for each factor in the factorization; `avg_rate` = mean of the heights. Rejection fires when `avg_rate < 1.0` **or** `max(all elements of the vector, heights+average) > 8.0`. Note the code technically takes `max_element` over the vector that *includes* the appended average — but since the arithmetic mean can never exceed the maximum of its own inputs, `*max_rates` is always equal to `max(height_i)` in practice, so this is behaviorally identical to the draft's `max(height_i) > 8.0`, just implemented as "max over heights-plus-average" rather than "max over heights alone." One nuance: the inline comment says "if average rate < 1.0 or **> 8.0**", but the executable code never checks `avg_rate > 8.0` as a separate condition — only `avg_rate < 1.0` is checked directly; the ">8.0" half of the comment is actually enforced via `*max_rates > 8.0` (and since `avg <= max`, this indirectly bounds `avg_rate` too, but it's not literally the same test the comment describes). This is a second, smaller comment/code mismatch beyond the one in §8, worth flagging with the same "trust the executable code" caveat.

These bad TPTs are removed from `configs` afterward only if `all_tpts.size() > tpts_with_bad_util_rate.size()` (lines 705–722) — i.e. the pruning is skipped if it would eliminate every candidate TPT for that node.

---

## 7. `DeriveMaxTPB` / `ConservativeMaxTPB`

**File:** `library/src/tuning_kernel_tuner.cpp`, lines 50–103.

```cpp
static const size_t LDS_BYTE_LIMIT    = 32 * 1024;
static const size_t BYTES_PER_FLOAT2  = sizeof(float) * 2;
static const size_t BYTES_PER_DOUBLE2 = sizeof(double) * 2;

size_t DeriveMaxTPB(size_t length, bool is_single, bool half_lds, bool use_ltwd_3steps,
                    size_t large1D, size_t tpt, size_t wgs_bound)
{
    size_t bytes_per_elem  = (is_single) ? BYTES_PER_FLOAT2 : BYTES_PER_DOUBLE2;
    size_t bytes_per_batch = length * bytes_per_elem;
    if(half_lds)
        bytes_per_batch /= 2;
    if(use_ltwd_3steps)
    {
        size_t ltwd_base, ltwd_steps;
        get_large_twd_base_steps(large1D, use_ltwd_3steps, ltwd_base, ltwd_steps);
        if(ltwd_base < 8)
            bytes_per_batch += ((1 << ltwd_base) * 3) * bytes_per_elem;
    }
    size_t tpb = LDS_BYTE_LIMIT / bytes_per_batch;
    while(tpt * tpb > wgs_bound)
        --tpb;
    return tpb;
}

size_t ConservativeMaxTPB(size_t length, bool is_single)
{
    size_t bytes_per_elem  = (is_single) ? BYTES_PER_FLOAT2 : BYTES_PER_DOUBLE2;
    size_t bytes_per_batch = length * bytes_per_elem;
    // theoretically half_lds doubles tpb, but empirically at the edge case
    // where tpb exactly fits LDS_BYTE_LIMIT (occ=2), doubling via half_lds
    // usually drops occupancy to 1 rather than keeping it at 2 — so a
    // conservative bound uses the non-half-lds value even for half_lds configs
    size_t conservative_max_tpb = LDS_BYTE_LIMIT / bytes_per_batch;
    if(length >= 1024)
        conservative_max_tpb += 1;
    return conservative_max_tpb;
}
```

Key facts:
- `LDS_BYTE_LIMIT = 32 KiB` (hard-coded target, independent of the actual device's LDS size — a `TODO` comment right above these constants literally says `// TODO- support half precision`, but not LDS-size-per-arch).
- `half_lds` halves the effective per-batch LDS footprint (`bytes_per_batch /= 2`), doubling the naive max TPB — this is the exact mechanism, confirming the draft's half-LDS handling claim.
- **Large-twiddle 3-step LDS accounting:** only when `use_ltwd_3steps` is true does it call `get_large_twd_base_steps(large1D, use_ltwd_3steps, ltwd_base, ltwd_steps)` (defined in `library/src/plan.cpp` lines 5214–5234: `base = use3steps ? clamp(4, 6, (CeilPo2(large1DLen)+2)/3) : 8`). Only if `ltwd_base < 8` (which only happens in 3-step mode, since 2-step/no-3-step mode always uses `base = 8`) does it add `(1 << ltwd_base) * 3` complex elements' worth of bytes to `bytes_per_batch` — i.e. the large-twiddle table is charged against the same LDS budget as the transform data.
- `tpb = LDS_BYTE_LIMIT / bytes_per_batch`, then the `TPT * TPB <= WGS` constraint is enforced by decrementing `tpb` in a `while` loop until `tpt * tpb <= wgs_bound`. This is the literal, verified mechanism for the "`TPT*TPB <= WGS`" constraint the draft describes.
- `ConservativeMaxTPB` computes a **cheaper, half-LDS-agnostic upper bound** (ignores `tpt`/`wgs_bound`/large-twiddle entirely) used purely to prune candidates early: any candidate whose derived `tpb > conservative_max_tpb` is rejected outright (`SupportedKernelConfigs` line 594: `if(tpb > conservative_tpb) { ...reject... }`), on the stated rationale that half-LDS's theoretical 2x TPB headroom rarely survives in practice without dropping occupancy from 2 to 1. There's also a length-dependent `+1` fudge factor for `length >= 1024`.

---

## 8. Other search-space reductions, including the "largest 33%" discrepancy

All in `library/src/tuning_kernel_tuner.cpp`, function `SupportedKernelConfigs`.

**a) Radix-count cap** (Phase 0 only) — §4 above.

**b) `tpt == length` and its associated bad-TPB removal** (lines 578–581, 675–700):

```cpp
// this tpt and tpb will be reject
if(tpt == length)
    tpbs_to_remove.insert(max_tpb);
...
// [reduce search space]
// if we have other options than tpt == length,
// then we can remove all configs with tpt==length
if(all_tpts.size() >= 2 && tpbs_to_remove.size() > 0)
{
    for(auto config = configs.begin(), last = configs.end(); config != last;)
    {
        if(config->threads_per_transform[0] == (int)length)
            config = configs.erase(config);
        else if(tpbs_to_remove.count((size_t)config->transforms_per_block) > 0)
            config = configs.erase(config);
        else
            ++config;
    }
    all_tpts.erase(length);
}
```
Confirmed: whenever `tpt == length` (i.e. the entire transform is done by one thread with no threading across the transform), the `max_tpb` that configuration derived is recorded as "bad," and — provided at least one other TPT choice exists — *both* every config with `tpt == length` *and* every config anywhere that happens to share that same `transforms_per_block` value gets removed, on the theory that a TPB value only reachable via the degenerate `tpt==length` case is itself suspect.

**c) Bad-utilization-rate TPT rejection** — §6 above.

**d) The "largest 33% TPT" comment vs. the actual code — CONFIRMED DISCREPANCY.**

Lines 724–754:
```cpp
// [reduce search space]
// we can remove the largest 33% tpt, since they are always in low perf.
if(all_tpts.size() > 0 && is_phase0)
{
    size_t num_tpts_to_remove = (all_tpts.size() - 1) / 2;
    if(num_tpts_to_remove > 0)
    {
        std::set<size_t>    tpts_to_remove;
        std::vector<size_t> tpts_vec(all_tpts.begin(), all_tpts.end());
        std::sort(tpts_vec.begin(), tpts_vec.end());
        // the largest #-num_tpts_to_remove tpts will be removed
        for(size_t i = 0; i < num_tpts_to_remove; ++i)
        {
            tpts_to_remove.insert(tpts_vec.back());
            tpts_vec.pop_back();
        }
        for(auto config = configs.begin(), last = configs.end(); config != last;)
        {
            if(tpts_to_remove.count((size_t)(config->threads_per_transform[0])) > 0)
                config = configs.erase(config);
            else
                ++config;
        }
    }
}
```

The comment says "remove the largest 33% tpt." The executable formula is `num_tpts_to_remove = (n - 1) / 2` where `n = all_tpts.size()` (integer division). This is **not** 33% — it approaches **50%** as `n` grows, and is only coincidentally ≈33% at `n = 3` (`(3-1)/2 = 1`, i.e. 1 of 3 removed):

| n (distinct TPTs) | num_tpts_to_remove = (n-1)/2 | fraction removed |
|---|---|---|
| 2 | 0 | 0% |
| 3 | 1 | 33% |
| 4 | 1 | 25% |
| 5 | 2 | 40% |
| 6 | 2 | 33% |
| 7 | 3 | 43% |
| 10 | 4 | 40% |
| large n | ~n/2 | →50% |

**Verdict: the draft's claim is correct — the comment ("largest 33%") is inconsistent with the implementation, which removes roughly the top half (asymptotically 50%, not 33%) of the sorted distinct-TPT set.** This report treats the code (`(n-1)/2`, largest values removed) as authoritative, since that is what actually executes; the comment appears to be stale/inaccurate documentation, not a bug being described accurately. This pruning is gated on `is_phase0` — it does **not** apply during Phase 1.

**e) Other pruning present in the same function** (all in `SupportedKernelConfigs`, all only loosely mentioned in the draft, listed here for completeness):
- `num_tpb_try = (tpt * max_tpb == wgs) ? 1 : 2;` (line 583) — tries at most `max_tpb` and `max_tpb+1` transforms-per-block, not an exhaustive TPB sweep.
- `final_wgs <= (wgs - 64)` and `final_wgs > max_wgs` are rejected (lines 588–591) — TPB search only accepted if resulting `final_wgs` lands within the current 64-wide `wgs` bucket being iterated.
- `if(length >= 64 && final_wgs < 64) continue;` (line 603) — no work-group smaller than 64 threads once length reaches 64.
- `if(IsPo2(length) && (length % final_wgs != 0)) continue;` (line 609) — power-of-two lengths require the work-group size to evenly divide the length.
- `direct_to_from_reg` loop is pinned to `{true /*, false*/}` only (line 619) — `false` is explicitly commented out: *"from current benchmark result, dir-reg mode always ranks high"* — i.e. the search space for this flag was manually collapsed to a single value based on prior empirical results, not searched at all in current code.
- `half_lds` is disallowed for `sbrc`/`sbcr` kernels (line 556: "only sbrr and sbcc support half-lds").
- `intrinsic_buffer_inst` requires `direct_to_from_reg` and is only allowed for `sbcc`/`sbcr` (lines 630–648).
- `min_wgs`/`max_wgs` are configurable via env vars `MIN_WGS`/`MAX_WGS` (default 64/512 for 1D; 768 for 2D in `Supported2DKernelConfigs`), rounded down to multiples of 64.

---

## 9. Two-phase tuning: Phase 0 → keep best 3 → Phase 1 permutations

**Phase count is literally 2**, driven by the offline tuner binary, not the enumeration file itself:

`library/src/rocfft_offline_tuner.cpp`, line 234: `static const int TUNING_PHASE = 2;` and the driving loop `for(int curr_phase = 0; curr_phase < TUNING_PHASE; ++curr_phase)`.

**Phase-0 → Phase-1 hand-off ("keep best 3"):** `library/src/tuning_helper.cpp`, function `TuningBenchmarker::PropagateBestFactorsToNextPhase()` (lines 277–306):

```cpp
// we will focus on the best 3 factors (at most) in the next phase tuning (permuting)
size_t num_target_factors = 3;
if(best_factors.size() > num_target_factors)
    best_factors.resize(num_target_factors);
```

The candidate list feeding this (`best_factors`) is built by walking `benchmark_infos_of_node`, which was **sorted by measured `milli_seconds`** earlier in `FindWinnerForCurrNode` (see §10), and de-duplicating by `factors_str` in first-seen (i.e. best-time) order — so this literally is "the best (up to) 3 distinct factorizations by measured time." **Confirmed exactly as drafted.**

**Phase-1 permutation strategy — cyclic shifts, not exhaustive enumeration, above a threshold of 6:**

`library/src/tuning_kernel_tuner.cpp`, `GetAllFactorizationsForPhase1` (lines 241–299):

```cpp
std::vector<std::vector<size_t>> permutations;
std::vector<size_t> permuting_factors = good_factors;
while(std::next_permutation(permuting_factors.begin(), permuting_factors.end()))
    permutations.push_back(permuting_factors);

if(permutations.size() > 6)
{
    permutations.clear();
    size_t              factor_len = good_factors.size();
    std::vector<size_t> reversed   = good_factors;
    std::reverse(reversed.begin(), reversed.end());

    good_factors.insert(good_factors.end(), good_factors.begin(), good_factors.end());
    reversed.insert(reversed.end(), reversed.begin(), reversed.end());
    for(size_t i = 0; i < factor_len; ++i)
    {
        std::vector<size_t> shifted_ori(good_factors.begin() + i, good_factors.begin() + i + factor_len);
        std::vector<size_t> shifted_rev(reversed.begin() + i, reversed.begin() + i + factor_len);
        permutations.push_back(shifted_ori);
        permutations.push_back(shifted_rev);
    }
}
```

**Confirmed exactly as drafted:** it first tries `std::next_permutation` to enumerate all permutations of the (already sorted, de-duplicated) factor multiset; if that count exceeds 6, it throws away the full permutation list and instead generates exactly `2 * factor_len` candidates: all `factor_len` cyclic shifts of the *original* sequence, and all `factor_len` cyclic shifts of the *reversed* sequence (via the classic "double the array, slide a window" trick). Note this means the number of factors actually run in Phase 1 is bounded by `2*factor_len` in the "too many permutations" case, deterministic and orderly (no randomness), not a subsample or heuristic search.

`SupportedKernelConfigs` gates this: `is_phase0 = target_factors_strs.empty()`. When `is_phase0` is true (Phase 0), `no_permutation = is_phase0 = true` and factorizations are used un-permuted (order as returned by `Factorize`, i.e. sorted ascending) — **confirmed: Phase 0 truly benchmarks only unpermuted factorizations.** When `is_phase0` is false (Phase 1), `no_permutation` is forced back to `true` (line 505) because permutation has already been performed manually via `GetAllFactorizationsForPhase1` — the `do { ... } while(no_permutation == false && std::next_permutation(...))` loop at the bottom of the function (line 671) becomes a single pass in Phase 1.

For 2D kernels, the analogous logic lives in `Supported2DKernelConfigs` (lines 301–457), which separately factorizes `len0`/`len1`, and in Phase 1 uses `GetAllFactorizationsForPhase1` independently per dimension (lines 352–353), plus explicitly re-includes the un-permuted Phase-0 factor sets so 2D cross-combinations that are "new" (a permuted dim0 with an original dim1, etc.) are still tried (lines 356–390).

---

## 10. Winner selection: real measured execution time, plus actual rejection/failure conditions

**Confirmed: winners are selected purely by measured wall/GPU time, never by a static cost formula.**

`library/src/tuning_helper.cpp`, `TuningBenchmarker::FindWinnerForCurrNode` (lines 227–275):
```cpp
std::sort(bench_infos_vec.begin(), bench_infos_vec.end(),
          [](BenchmarkInfo& a, BenchmarkInfo& b) { return a.milli_seconds < b.milli_seconds; });
auto& winner_of_this_phase = bench_infos_vec.front();
if(winner_of_this_phase.milli_seconds < curr_best_msec) { /* promote as new winner */ }
```
`milli_seconds` is populated by `UpdateCurrBenchResult(double ms, double gflops)` (line 213), called from the actual GPU-timing driver.

**The actual benchmarking driver:** `library/src/rocfft_offline_tuner.cpp`, function `offline_tune_problems` (lines 109–390). For each candidate kernel config, per node, per phase, it:
1. Frees and **recreates the real rocFFT plan** (`params.free(); params.create_plan();`) so the actual compiled/RTC kernel for that exact `KernelConfig` is used.
2. Executes it once as a warm-up (`params.execute(...)`), then executes it `ntrial` more times wrapped in `hipEventRecord`/`hipEventElapsedTime` (lines 291–312) — real HIP event-based GPU timing, not an estimate.
3. Takes the **median** of the `ntrial` times (lines 330–335) as `ms_median`, computes `gflops_median` from a closed-form operation count (`opscount`, based on `5*N*log2(N)*batch` for complex, `2.5x` for real transforms — lines 228–232), and records both.

**Real, actually-implemented rejection/failure conditions found in this driver (lines 269–289):**
```cpp
BenchmarkInfo info = offline_tuner->GetCurrBenchmarkInfo();
if(info.threads_per_trans[1] != 0)       // 2D-single kernel
{
    if(info.occupancy < 0)               // "unable to gen kernel"
    { ...skip, ms=max_double, gflops=0... }
}
else
{
    if(info.occupancy == 1 || info.occupancy < 0)  // occupancy 1 or "unable to gen"
    { ...skip, ms=max_double, gflops=0... }
}
```
So the two concrete, verified rejection paths are:
- **`occupancy < 0`** — an explicit sentinel meaning "the kernel could not be generated" (i.e. an effective compile/kernel-generation failure surfaces as `occupancy == -1`, computed upstream during kernel generation/RTC, and is skipped rather than crashing the whole tuning run).
- **`occupancy == 1`** (non-2D kernels only) — the tuner explicitly refuses to consider single-occupancy kernels as viable winners; this matches the "aim for occupancy-2" comment seen in the LDS-budgeting code (`rocfft_kernel_config_search.cpp` line 174: `specs.lds_byte_limit = device_prop.sharedMemPerBlock / 2; // aim for occupancy-2`).

**What is NOT verified/present as a per-candidate rejection in this file:** there is no numerical-correctness check (no comparison against a reference FFT) anywhere in `offline_tune_problems`. Hard failures — a HIP API call failing, or `params.create_plan()` failing — are surfaced via `HIP_V_THROW`/`LIB_V_THROW`, which **throw a `std::runtime_error` and abort the whole tuning run**, rather than being caught and treated as "this one candidate loses." So: correctness-mismatch rejection and per-candidate compile-failure catch-and-continue (beyond the occupancy<0 sentinel) are **not substantiated by this file** — if such logic exists, it is upstream, inside whatever computes `occupancy` (kernel-generation / RTC code, not inspected in this pass — see "Not Verified" section below).

---

## 11. `solution_map.h` — cache key and stored value

**File:** `library/src/include/solution_map.h`.

**Key — `ProblemKey`** (lines 54–79):
```cpp
struct ProblemKey { std::string arch; std::string probToken; };
```
`arch` is either the concrete GCN arch name (e.g. from `get_arch_name(deviceProp)`) or the literal fallback string `"any"` (see `GenerateProbKeys`, `library/src/plan.cpp` lines 6199–6215, which tries `{archName, "any"}` × `{full_token, min_token}` — full-match first, then progressively looser).

`probToken` is generated by `GetNodeToken` (`library/src/plan.cpp`, lines 6122–6197), which builds **two** token strings per node:
- **`min_token`**: scheme abbreviation + length(s) + precision (`sp_`/`dp_`/`half_`) + placement (`ip_`/`op_`) + `real_fwd`/`real_bwd` or `complex`/`complex_fwd`/`complex_bwd`. (For C2R, the solution is keyed on the **complex** output length, not the real input length.)
- **`full_token`**: everything in `min_token` **plus** `batch`, `inStride`/`outStride` (all dims), `iDist`/`oDist`, `iOffset`/`oOffset`.

So the draft's claim is confirmed and made precise: the solution map key is **not** simply "target/device + length + precision + direction" — it explicitly separates a loose match (`min_token`: length, precision, placement, real/complex+direction only — no batch/stride/dist/offset) from an exact match (`full_token`: adds batch, strides, distances, offsets), and lookup tries full-match first, then falls back to the min-match, across both the specific arch and a generic `"any"` arch.

**Value — `SolutionNode`** (lines 134–186):
```cpp
struct SolutionNode {
    std::string      arch_name;
    SolutionNodeType sol_node_type; // SOL_DUMMY / SOL_BUILTIN_KERNEL / SOL_KERNEL_ONLY / SOL_LEAF_NODE / SOL_INTERNAL_NODE
    ComputeScheme    using_scheme;
    FMKey            kernel_key;               // only meaningful for kernel-only/leaf nodes
    std::vector<SolutionPtr> solution_childnodes; // {child_token, child_option} pairs
};
```
This is a **tree-shaped** value: an `SOL_INTERNAL_NODE` points (via `solution_childnodes`, each a `{child_token, child_option}` pointing into another `ProblemKey`'s solution vector) to its decomposition children; an `SOL_LEAF_NODE`'s single child points to a `SOL_KERNEL_ONLY` node, which carries the actual `FMKey` (length/precision/scheme/arch + the tuned `KernelConfig`). So the stored "value" per problem token is effectively the whole plan/decomposition tree plus, at each leaf, the winning `KernelConfig`.

---

## 12. Plan-tree tuning scope: one tree today, TODO for several

**Confirmed: the TODO exists, verbatim, and the scope claim is accurate.**

`library/src/tuning_plan_tuner.cpp`, function `EnumerateTrees` (lines 102–139):
```cpp
void EnumerateTrees(ExecPlan& execPlan)
{
    ...
    // TODO- plan-tuning: build tree several times to generate different trees
    {
        execPlan.rootPlan->RecursiveBuildTree();
        ...
        execPlan.rootPlan->CollectLeaves(execPlan.execSeq, execPlan.fuseShims);
        ...
        if(TuningBenchmarker::GetSingleton().GetPacket()->tuning_phase == 0)
            SerializeTree(execPlan.rootPlan.get(), archName, root_min_token, root_full_token);
        EnumerateKernelConfigs(execPlan);
    }
}
```
The `{ ... }` block that immediately follows the TODO comment is executed exactly once per call — `RecursiveBuildTree()` builds a single decomposition using rocFFT's normal (non-tuning) planning heuristics, and everything downstream (`SerializeTree`, `EnumerateKernelConfigs`) operates on that one tree's leaves. There is no loop, no alternate-decomposition branch, and no code path elsewhere in `tuning_plan_tuner.cpp`/`tuning_kernel_tuner.cpp`/`tuning_helper.cpp` that builds a second, different tree for the same problem and compares the two. `SerializeTree` is called only once (guarded by `tuning_phase == 0`, so the tree structure itself is fixed after phase 0 and only kernel configs at its leaves are varied in phase 1).

**Confirmed scope statement:** today's tuning system builds **one** decomposition tree per problem (using rocFFT's regular, non-tuning tree-building logic) and tunes only the `KernelConfig` search space (§1–§9) at that tree's leaf (kernel) nodes across two phases. It does not explore alternative decompositions/plans — the `TODO` explicitly flags this as intended future work, unresolved as of the fetched commit.

---

## Files/URLs actually fetched and read in this research

All via `raw.githubusercontent.com/ROCm/rocm-libraries/develop/projects/rocfft/...` and `api.github.com/repos/ROCm/rocm-libraries/contents/...`, snapshot at commit `bee97df517907c771de17189cb867d3c401285ae`:

- `library/src/tuning_kernel_tuner.cpp` (861 lines) — core: `Factorize`, `GetMaxRadicesSize`, `PowerSet`, `SupportedThreadsPerTransform`, `GetUtilizationRate`, `DeriveMaxTPB`, `ConservativeMaxTPB`, `GetAllFactorizationsForPhase1`, `Supported2DKernelConfigs`, `SupportedKernelConfigs`, `EnumerateKernelConfigs`.
- `library/src/tuning_plan_tuner.cpp` (138 lines) — `SerializeTree`, `EnumerateTrees` (contains the plan-tree TODO).
- `library/src/tuning_helper.cpp` (506 lines) — `TuningBenchmarker` implementation: `FindWinnerForCurrNode`, `PropagateBestFactorsToNextPhase`, `ExportWinnerToSolutions`, `ExportCSV`, `MergingSolutionsMaps`.
- `library/src/rocfft_kernel_config_search.cpp` (644 lines) — standalone single-kernel brute-force/manual tuner (older/simpler sibling tool; independently confirms `supported_factors`, `Factorize`/`factorize`, `PowerSet`/`power_set`, `SupportedThreadsPerTransform`).
- `library/src/rocfft_offline_tuner.cpp` (534 lines) — the actual CLI driver that runs the two-phase loop (`TUNING_PHASE = 2`), does real GPU timing via `hipEvent*`, and implements the occupancy-based skip logic.
- `library/src/solution_map.cpp` (976 lines, fetched but only spot-checked; primary structural facts came from the header).
- `library/src/include/function_map_key.h` (637 lines) — `KernelConfig`, `FMKeyBase`, `FMKey`, `PPFMKey`, `GetKernelToken`, `get_alternative_FMKey`.
- `library/src/include/solution_map.h` (360 lines) — `ProblemKey`, `SolutionPtr`, `SolutionNode`, `solution_map` class API.
- `library/src/include/tuning_helper.h`, `tuning_kernel_tuner.h`, `tuning_plan_tuner.h` (fetched for declarations; not separately quoted).
- `library/src/plan.cpp` (fetched in full; used for `GetNodeToken` (lines 6122–6197), `GenerateProbKeys` (lines 6199+), and `get_large_twd_base_steps` (lines 5214–5234)).
- `library/src/node_factory.cpp`, `library/src/tree_node.cpp` (fetched, grepped, no additional load-bearing facts found beyond what's above).
- `library/src/include/twiddles.h` (fetched — `TWIDDLES_MAX_RADICES = 8`) and `library/src/twiddles.cpp` (fetched, no additional facts needed).
- Directory listings via GitHub Contents API for `library/src/` and `library/src/include/` (confirmed current file layout) and `projects/rocfft/clients/` (confirmed no separate "tuner" client directory — the offline tuner lives in `library/src/rocfft_offline_tuner.cpp`).
- `github.com/ROCm/rocm-libraries` and `github.com/ROCm/rocFFT` (WebSearch only, to confirm the repo move/current path).

---

## Claims from the draft that could NOT be independently verified (or were only partially verified)

1. **Per-candidate correctness/numerical-mismatch rejection.** The draft's item 10 asked about "numerical mismatch" as a rejection condition. I found no such check in `rocfft_offline_tuner.cpp` (the actual driver) — only the `occupancy < 0` / `occupancy == 1` skip conditions are real, verified per-candidate rejections; a HIP/library-call failure aborts the whole run rather than being caught per-candidate. **Confidence: high that no correctness check exists in this file; I did not trace how `occupancy` itself is computed (likely in RTC/kernel-generation code, e.g. `rtc_stockham_gen.cpp`/`rtc_kernel.cpp`, which I did not fetch in this pass), so I cannot rule out that a compile failure is what actually produces `occupancy == -1` under the hood.** This is a gap, not a contradiction of the draft.
2. **`SupportedThreadsPerTransform`/`GetUtilizationRate`/etc. for the 2D path** (`Supported2DKernelConfigs`) use a much simpler, different scheme (only "uwide"/"wide" TPT extremes per dimension — `len/min(factors)` and `len/max(factors)` — no power-set TPT enumeration, no utilization-rate filter). This is real and verified (lines 301–457 of `tuning_kernel_tuner.cpp`), but the draft's items 3–6 were phrased as if they apply universally; they in fact apply only to the 1D (`SupportedKernelConfigs`) path. Flagging so the porter doesn't assume 2D uses the same machinery.
3. **Exactly how `occupancy` and `granularity` (`num_blocks / numCUs`) are computed** — referenced in `BenchmarkInfo`/`GetCurrBenchmarkInfo` (`tuning_helper.cpp` lines 183–211) but the actual occupancy-calculation code path was not located/fetched in this pass (likely in RTC compile/launch-bounds code). Low confidence on exact formula; only its consumption (the skip-thresholds in §10) is verified.
4. **Whether `solution_map.cpp`'s on-disk text format** (used by `write_solution_map_data`/`read_solution_map_data`) has any additional versioning/migration quirks beyond the `SolutionMapConverter` class stub seen in the header (`remove_invalid_half_lds`, `remove_callback_nodes`) — the 976-line `.cpp` was fetched but not read in full; only the header's public surface was used for §11. Medium confidence on the header-level facts; low/no confidence on serialization-format edge cases.
5. I did not find or check for an equivalent "TODO several decomposition trees" comment having been *resolved* anywhere else in the codebase (e.g. changelog, PR history) — I only confirmed it is still present, unresolved, in the current `develop` snapshot. If the porter needs to know whether this is scheduled/in-progress work, that would require checking open PRs/issues on `ROCm/rocm-libraries`, which was out of scope here.

Everything else in sections 1–9 and 11–12 (the struct fields, the exact factor set, `Factorize`, `GetMaxRadicesSize`'s `+2` formula, `SupportedThreadsPerTransform`'s power-set method, `GetUtilizationRate`'s exact formula and `1.0`/`8.0` thresholds, `DeriveMaxTPB`/`ConservativeMaxTPB`, the tpt==length pruning, the "largest 33%"-comment-vs-`(n-1)/2`-code discrepancy, the phase-0/phase-1 split, the "best 3" propagation, the >6-permutations cyclic-shift fallback, real-time-based winner selection, and the solution-map key/value shapes) is **directly quoted from source fetched in this session** and should be treated as high-confidence / directly traceable.
