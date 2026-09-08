# rocFFT Production (Non-Tuned) Planning Path — Source-Verified Research Report

Repository: `github.com/ROCm/rocm-libraries`, branch `develop`, path `projects/rocfft/`.
All line numbers refer to the files as fetched on 2026-09-08 (see file list in §7).

---

## 1. Default tree-building path — exact functions/files

The decomposition-tree builder that runs on **every** `rocfft_plan_create` call (tuned or not)
is `NodeFactory::DecideNodeScheme` / `NodeFactory::Decide1DScheme` / `Decide2DScheme` /
`Decide3DScheme` / `DecideRealScheme`, all in
**`library/src/node_factory.cpp`** (declared in `library/src/include/node_factory.h`).

Call chain from the public API down to this function:

```
rocfft_plan_create (plan.cpp)
  -> rocfft_plan_t::BuildSingleDevicePlan<probe_solution_map=true>()   [plan.cpp:1686]
       -> execPlan.rootPlan = NodeFactory::CreateExplicitNode(rootPlanData, nullptr)   [plan.cpp:1701]
            -> NodeFactory::CreateExplicitNode()                       [node_factory.cpp:499]
                 -> determined_scheme = DecideNodeScheme(pool, nodeData, parent)   [node_factory.cpp:529]
                      -> Decide1DScheme / Decide2DScheme / Decide3DScheme / DecideRealScheme
       -> (if solution map probe succeeds) ApplySolution(execPlan)     [plan.cpp:1709]
       -> ProcessNode(execPlan)                                        [plan.cpp:1752, defined 6372]
            -> execPlan.rootPlan->RecursiveBuildTree(rootScheme)       [plan.cpp:6377]
                 -> TreeNode::RecursiveBuildTree() (plan.cpp:5383) just forwards to the
                    per-node-type virtual BuildTree_internal(child_scheme), which for internal
                    nodes (L1D_TRTRT/CRT/CC, 2D_RC, 3D_TRTRT, etc.) recursively calls
                    NodeFactory::CreateExplicitNode() again for each child length — this is the
                    recursive tree-building referenced by tuning_plan_tuner.cpp's EnumerateTrees.
```

`RecursiveBuildTree` itself (plan.cpp:5383-5398) is a two-line dispatcher: it takes an optional
`SchemeTree* solution_scheme` (non-null only when a solution-map entry was found) and calls the
node's own overridden `BuildTree_internal`. It carries **no decomposition heuristics of its
own** — all the "how do I cut this length up" logic lives in `NodeFactory::DecideNodeScheme` and
its per-dimension helpers.

### Real default heuristics found in `Decide1DScheme` (node_factory.cpp:628-819)

1. `SupportedLength()` (node_factory.cpp:359) — if the function pool has no entry and no
   factorization works at all, forces **Bluestein** (`CS_BLUESTEIN`).
2. **If the function pool already has a compiled-in single kernel for this exact length**
   (`pool.has_function(FMKey(length, precision))`, node_factory.cpp:637): use it directly as
   `CS_KERNEL_STOCKHAM` — UNLESS length > 4096, in which case there's a batch/occupancy
   heuristic (compare `totalBatch / kernel.transforms_per_block` against
   `multiProcessorCount`) to decide whether the single big kernel still gets enough workgroups
   to fill the GPU; if not, it falls through to a multi-kernel decomposition.
3. **If no single-kernel exists** for the length:
   - Power-of-two lengths ≤ 262144 use `CS_L1D_CC` (Stockham block-column-column, i.e. a
     transpose + column kernel), with the "divide-length" factor pulled from a static lookup
     table `map1DLengthSingle` / `map1DLengthDouble` (hardcoded maps, not derived from
     `Factorize`/tuning). Above that block threshold, or when the lookup table doesn't have an
     entry, it falls back to `CS_L1D_TRTRT` (transpose-row-transpose-row-transpose) using
     `pool.get_largest_pow2_length()`.
   - Non-power-of-two lengths use the same `map1DLengthSingle`/`map1DLengthDouble` static
     tables for `CS_L1D_CC`; on failure it falls back to `CS_L1D_TRTRT`, using
     `get_explicitly_supported_factor()` / `get_largest_supported_factor()`
     (node_factory.cpp:258-284) which only search the **already-compiled `function_pool`
     entries** (not a tuning search space) for a divisor length that itself has a kernel.
4. 2D/3D pick between `CS_KERNEL_2D_SINGLE` (fits in LDS), `CS_2D_RC`/`CS_3D_RC`,
   `CS_3D_BLOCK_RC`, `CS_3D_PP` (partial pass), or RTRT-style fallbacks, again gated by
   `pool.has_function(...)` checks against the compiled function pool — never by
   `SupportedKernelConfigs` or any tuning-search function.

**Confidence: High.** Directly read from node_factory.cpp; confirmed by call-graph grep showing
zero references to `Factorize`/`GetMaxRadicesSize`/`SupportedThreadsPerTransform`/
`GetUtilizationRate`/`DeriveMaxTPB`/`SupportedKernelConfigs` anywhere in this file (see §5).

---

## 2. Default `KernelConfig` selection for an untuned leaf — exact functions/files, formula

There is **no runtime default-KernelConfig-generator function** analogous to
`SupportedKernelConfigs`. Instead, the default `KernelConfig` for every leaf kernel is a
**pre-computed, compiled-in table entry**, looked up by key, not computed by any formula at
plan-creation time.

Mechanism, end to end:

- A leaf node's `GetKernelKey()` (base impl in `library/src/include/tree_node.h:862-869`) returns:
  ```cpp
  virtual FMKey GetKernelKey() const
  {
      if(specified_key)
          return *specified_key.get();
      return (dimension == 1) ? FMKey(length[0], precision, scheme)
                              : FMKey(length[0], length[1], precision, scheme);
  }
  ```
  When no solution-map entry supplied `specified_key`, this `FMKey` constructor defaults its
  embedded `kernel_config` field to `KernelConfig::EmptyConfig()`
  (`function_map_key.h:336,346,361` — `KernelConfig kernel_config = KernelConfig::EmptyConfig()`).
  For `LeafNode` specifically, `GetKernelKey()` is overridden (tree_node.cpp:103-109) to return
  `FMKey::EmptyFMKey()` if `externalKernel == false` (used for non-Stockham-family leaves like
  `TransposeNode`); for Stockham-family nodes (`Stockham1DNode`, `SBCCNode`, `SBRCNode`,
  `SBCRNode` — all of which set `externalKernel = true` in their constructors, e.g.
  `tree_node_1D.h:88`), it falls through to the base `TreeNode::GetKernelKey()` above, i.e. a
  **real, non-empty** `FMKey` with an **empty kernel_config**.
- `pool.get_kernel(key)` / `pool.has_function(key)` (function_pool.h:425-433, 377-382) call
  `get_actual_key(key, def_key_pool)` (function_pool.h:277-301), which looks the "simple key"
  (length/precision/scheme, empty config) up in `def_key_pool` — a side-table mapping
  "simple/empty-config key" → "full key with the real, hand-picked config" — and substitutes it.
  `def_key_pool` is populated at **AOT/compile time** by `insert_default_entry()`
  (function_pool.h:490-518), whose doc comment states plainly:

  > "Insert a key-kernel pair for AOT generator... That is, the default kernel-config we set in
  > the kernel-generator.py we save a pair as `<key-empty-config, key-actual-config>` that
  > allows us to use the empty-config key to get the default kernel"

- `insert_default_entry` is called from generated C++ functions `function_pool_init_0..N`
  (built by `kernel-generator.py`'s `generate_cpu_function_pool_pieces` /
  `generate_cpu_function_pool_main`, kernel-generator.py:310-474), which are in turn invoked
  from the `function_pool_data::function_pool_data()` constructor — i.e. **static
  data populated once at program/library load time**, entirely at build time, from Python
  config lists in `library/src/device/kernels/configs/config_sbrr.py`,
  `config_sbcc.py`, `config_sbcr.py`, `config_sbrc.py`, `config_2d_single.py`, `config_pp_3d.py`.

- The "formula" that turns each config-table row into transforms-per-block is exactly one
  line, in `kernel-generator.py:664-666` (`generate_kernel_functions`):
  ```python
  transforms_per_block = launcher.transforms_per_block
  workgroup_size = launcher.workgroup_size
  threads_per_transform = workgroup_size // transforms_per_block
  ```
  and conversely for the small-kernel (`sbrr`) table rows that specify
  `threads_per_transform` directly, `transforms_per_block` is derived as
  `workgroup_size / threads_per_transform` — a trivial division, not a search. (For the
  large/SBCC/SBCR tables, `workgroup_size` itself is derived by a simple formula
  `block_width * product(factors)/min(factors)`, kernel-generator.py:535-537/549-551 — again
  not a search over candidate widths.)

**There is no `MIN_WGS`, "final workgroup size must evenly divide length" or utilization-rate
filtering anywhere in this default path.** Those concepts exist only inside
`tuning_kernel_tuner.cpp`'s `SupportedKernelConfigs` (see §5).

**Confidence: High.**

---

## 3. Small-length (N=8/16) handling — exact mechanism, with source quotes

rocFFT does **not** have a separate hand-written "builtin kernel" code path or a distinct
kernel-generation mechanism for tiny lengths. N=8 and N=16 go through **exactly the same**
`CS_KERNEL_STOCKHAM` single-kernel mechanism as every other length that has a compiled-in
kernel — the only thing that's different is that their `KernelConfig` was chosen by a rocFFT
developer and hardcoded into a table, rather than searched for.

Source: `library/src/device/kernels/configs/config_sbrr.py` (the "small kernels" / Stockham
single-kernel config table, consumed by `kernel-generator.py:list_small_kernels()`,
kernel-generator.py:512-522):

```python
sbrr_kernels = [
    NS(length=   1, workgroup_size= 64, threads_per_transform=  1, factors=(1,), runtime_compile=True),
    NS(length=   2, workgroup_size= 64, threads_per_transform=  1, factors=(2,), runtime_compile=True),
    NS(length=   3, workgroup_size= 64, threads_per_transform=  1, factors=(3,), runtime_compile=True),
    NS(length=   4, workgroup_size=128, threads_per_transform=  1, factors=(4,), runtime_compile=True),
    NS(length=   5, workgroup_size=128, threads_per_transform=  1, factors=(5,), runtime_compile=True),
    NS(length=   6, workgroup_size=128, threads_per_transform=  1, factors=(6,), runtime_compile=True),
    NS(length=   7, workgroup_size= 64, threads_per_transform=  1, factors=(7,), runtime_compile=True),
    NS(length=   8, workgroup_size= 64, threads_per_transform=  4, factors=(4, 2), runtime_compile=True),
    NS(length=   9, workgroup_size= 64, threads_per_transform=  3, factors=(3, 3), runtime_compile=True),
    NS(length=  10, workgroup_size= 64, threads_per_transform=  1, factors=(10,), runtime_compile=True),
    ...
    NS(length=  16, workgroup_size= 64, threads_per_transform=  4, factors=(4, 4), runtime_compile=True),
    ...
```

So:
- **N=8**: `workgroup_size=64`, `threads_per_transform=4`, `factors=(4,2)` →
  `transforms_per_block = 64 // 4 = 16`.
- **N=16**: `workgroup_size=64`, `threads_per_transform=4`, `factors=(4,4)` →
  `transforms_per_block = 64 // 4 = 16`.

These are literal, hand-authored integers checked into the repository by rocFFT developers
(comment at the top of the file: "Note: Default half_lds is True and default
direct_to_from_reg is True as well"), not the output of any runtime or offline search
algorithm. `runtime_compile=True` on these entries means the kernel source is generated and
RTC-compiled the first time it's needed (same RTC machinery used for every other Stockham
kernel) — it does **not** mean the *parameters* (WGS/TPT/factors) are chosen dynamically; only
the kernel *source code* is JIT-compiled from these fixed parameters. (A handful of larger
"small" lengths, e.g. length=32/36/40/64/96, omit `runtime_compile=True`, meaning they are
compiled ahead-of-time into the shipped library binary instead of RTC'd at first use — but the
WGS/TPT/factors are equally fixed table values either way.)

`list_small_kernels()` tags every row with `scheme='CS_KERNEL_STOCKHAM'`, which is exactly the
scheme `NodeFactory::Decide1DScheme` selects directly (node_factory.cpp:637-664) whenever
`pool.has_function(FMKey(length, precision))` is true and length ≤ 4096 — true for both 8 and
16 by construction, since they are in this table.

**Confidence: High** for the table contents and the wgs/tpt/tpb values; **High** for the
claim that this is the sole mechanism (no other small-length-specific code path was found in
`node_factory.cpp`, `tree_node*.cpp`, or `plan.cpp` — the only length-based special-casing found
elsewhere is unrelated hardware/arch hacks for specific large lengths, e.g. `length[0] ==
43008` on gfx90a, node_factory.cpp:765, and `length[0] == 262144` on gfx906, node_factory.cpp:702).

---

## 4. Solution-map consultation order at plan-creation time

**Yes — the solution map is always probed first**, before any default/formula-based
decomposition, for an ordinary (non-tuning) `rocfft_plan_create` call.

`BuildSingleDevicePlan` is a template on `bool probe_solution_map`, **defaulted to `true`**:
```cpp
// library/src/include/plan.h:590
template <bool scope_solution_map = true>
std::unique_ptr<ExecPlan> BuildSingleDevicePlan(...);
```
All ordinary call sites (`plan.cpp:2160`, `2370`, `2446`) invoke it with no explicit template
argument, i.e. with probing **on**. Inside it (plan.cpp:1684-1789):

```cpp
execPlan.rootPlan = NodeFactory::CreateExplicitNode(rootPlanData, nullptr);   // default-scheme leaf/tree, CS_NONE resolved via DecideNodeScheme

if constexpr(probe_solution_map)
{
    if(TuningBenchmarker::GetSingleton().IsInitializingTuning() == false)
    {
        execPlan.rootScheme = ApplySolution(execPlan);      // <-- solution map lookup, ALWAYS attempted
        if(execPlan.rootScheme)
        {
            execPlan.rootPlan = NodeFactory::CreateExplicitNode(rootPlanData, nullptr, execPlan.rootScheme->curScheme);
        }
    }
}
...
ProcessNode(execPlan);   // -> RecursiveBuildTree(execPlan.rootScheme)   [nullptr if no solution found]
```

`ApplySolution` (plan.cpp:6353-6370) calls `GenerateProbKeys` to build a token
(length/precision/placement/strides/dist/batch/etc., `GetNodeToken`, plan.cpp:6122-6197) and
tries, in order: `(this GPU's arch name, full_token)`, `(this arch, min_token)`, `("any",
full_token)`, `("any", min_token)`. It looks these up via
`solution_map::get_solution_map().has_solution_node(...)` against data loaded from
**shipped `.dat` files**.

**Confirmed shipped files** (`library/solution_map/` directory, listed via GitHub API):
```
gfx908_rocfft_solution_map.dat
gfx90a_rocfft_solution_map.dat
gfx942_rocfft_solution_map.dat
```
Only these three specific archs (MI100/MI200/MI300-class) ship a pre-populated, checked-in
solution map. There is **no generic "any" file** and no file for other archs (e.g. consumer
RDNA GPUs, gfx900, gfx1030, gfx1100, gfx1201, CDNA1's siblings, etc.). `solution_map.cpp:36`
(`def_solution_map_path = "rocfft_solution_map.dat"`) plus `get_solution_map_path()`
(`solution_map.cpp:72-84`) constructs the filename as `"<arch>_rocfft_solution_map.dat"`, so on
any GPU that isn't one of those three, the on-disk file simply does not exist and the lookup
fails outright (empty/no solution map loaded for that arch/at all).

**On a miss** (`rootScheme == nullptr`), `ProcessNode` (plan.cpp:6372-6377) calls
`execPlan.rootPlan->RecursiveBuildTree(nullptr)`, and every internal `BuildTree_internal`
override then falls through to plain `NodeFactory::CreateExplicitNode(...)` calls with
`determined_scheme == CS_NONE`, which forces `DecideNodeScheme` — i.e., exactly the
formula/heuristic default path of §1/§2. There is also an exception-based second-chance
mechanism: if `ProcessNode` throws while trying to honor a *found* solution-map entry (e.g. it
doesn't actually fit the buffers), `BuildSingleDevicePlan` catches it and retries once with
`probe_solution_map` flipped to `false` (plan.cpp:1756-1770), guaranteeing the default path is
always reachable as a fallback even after a partial solution-map hit.

**Confidence: High** for the always-probe-first behavior and the template default; **High**
for which `.dat` files are actually shipped in the repo (directly listed via the GitHub
contents API); **Medium** on whether those three `.dat` files contain any entry at all for
N=8/N=16 — I did not fetch/parse the (likely large, binary/JSON-ish) `.dat` file contents, but
per §6 it would not matter even if they did, because `Decide1DScheme`'s function-pool check
happens as part of building the *default* tree used to probe the solution map's token in the
first place, and separately, small single-kernel lengths are exactly the case rocFFT's own
tuning system treats as trivial/not-worth-tuning (see §6).

---

## 5. Tuning-only vs. always-used decisions — explicit call-graph evidence

Grepped the entire non-tuning production surface (`node_factory.cpp`, `tree_node.cpp`,
`tree_node_1D/2D/3D/real/bluestein.cpp`, `plan.cpp`, `function_pool.h`, `solution_map.cpp`) for
the five ported functions:

```
$ grep -rn "SupportedKernelConfigs\|GetMaxRadicesSize\|SupportedThreadsPerTransform\|GetUtilizationRate\|DeriveMaxTPB" .
./library_src_tuning_kernel_tuner.cpp   <- only file containing ANY of these symbols
```

All five are **defined exclusively in `library/src/tuning_kernel_tuner.cpp`**:
```
tuning_kernel_tuner.cpp:50   size_t DeriveMaxTPB(...)
tuning_kernel_tuner.cpp:136  size_t GetMaxRadicesSize(...)
tuning_kernel_tuner.cpp:175  std::set<size_t> SupportedThreadsPerTransform(...)
tuning_kernel_tuner.cpp:189  ...GetUtilizationRate(...)
tuning_kernel_tuner.cpp:459  std::set<KernelConfig> SupportedKernelConfigs(...)
```
and `SupportedKernelConfigs` (the only public entry point of the group) is called from just one
place — `tuning_kernel_tuner.cpp:816`, itself gated behind
`TuningBenchmarker::GetSingleton().IsInitializingTuning()`/`IsProcessingTuning()` state that is
only ever set true by the tuning-benchmark machinery driven from `rocfft_offline_tuner.cpp`
(and, per `plan.cpp:1707/1741`, `EnumerateTrees`/tuning-mode branches in
`BuildSingleDevicePlan`). Also, `MIN_WGS`/`MAX_WGS` environment-variable reads
(`tuning_kernel_tuner.cpp:484-487`) exist **only** in this file — nowhere in the default path.

Build-system note (important nuance): `tuning_kernel_tuner.cpp` and `tuning_plan_tuner.cpp` are
in fact part of the **main `rocfft_source` list** compiled into the production `rocfft` shared
library itself (`library/src/CMakeLists.txt:244-269`) — they are *not* gated behind the
`ROCFFT_BUILD_OFFLINE_TUNER` CMake option the way `rocfft_offline_tuner.cpp`/
`rocfft_solmap_convert.cpp` are (CMakeLists.txt:391-404). So the tuning-search code **is present
as compiled symbols inside every rocFFT install**, but it is **dead code from the perspective
of the call graph an ordinary `rocfft_plan_create` traverses** — it is only reachable via the
`TuningBenchmarker` singleton's tuning-mode flags, which nothing in the ordinary plan-create
path ever sets to true. (`rocfft_kernel_config_search` is a wholly separate standalone
executable/tool with its own from-scratch `factorize()`/config-search implementation,
independent of `tuning_kernel_tuner.cpp`; it is not linked into `rocfft_plan_create` either.)

**Always-used, regardless of tuning status:**
- `NodeFactory::DecideNodeScheme`/`Decide1DScheme`/`Decide2DScheme`/`Decide3DScheme`
  (node_factory.cpp) — called unconditionally whenever `CreateExplicitNode` receives
  `determined_scheme == CS_NONE`, which is the normal state absent a solution-map hit.
- `function_pool::has_function`/`get_kernel`/`get_actual_key` (function_pool.h) — consulted by
  `Decide1DScheme` and by every leaf's `KernelCheck`/`HasKernel`/`GetKernel`.
- `ApplySolution`/`GenerateProbKeys`/`RecursivelyApplySol` (plan.cpp) — always attempted first
  (§4), succeed only for gfx908/gfx90a/gfx942 and only for whatever specific problem
  tokens exist in those three shipped `.dat` files.

**Tuning-CLI-only, never touched otherwise:**
- `Factorize`, `GetMaxRadicesSize`, `SupportedThreadsPerTransform`, `GetUtilizationRate`,
  `DeriveMaxTPB`, `SupportedKernelConfigs` — all in `tuning_kernel_tuner.cpp`, reachable only
  through `TuningBenchmarker` tuning-mode state driven by `rocfft_offline_tuner`.
- `EnumerateTrees` (`tuning_plan_tuner.cpp`) — called only from the
  `TuningBenchmarker::GetSingleton().IsInitializingTuning()` branch of `BuildSingleDevicePlan`
  (plan.cpp:1741-1747), which an ordinary plan-create call never enters.

**Confidence: High**, based on exhaustive grep across every production-path file fetched, plus
reading the actual gating conditionals in `plan.cpp` and the CMakeLists source-file lists.

---

## 6. Direct answer: do N=8/N=16 bypass the tuning search space entirely?

**Yes, confirmed.** N=8 and N=16 (like every length listed in `config_sbrr.py`,
`config_sbcc.py`, `config_sbcr.py`, `config_sbrc.py`) are served by a **compiled-in, hand-picked
default `KernelConfig`** that is inserted into the `function_pool` at library-build time via
`insert_default_entry()` (function_pool.h:490), fed from a static Python table
(`config_sbrr.py:37-38` for N=8/16 specifically), with `transforms_per_block` derived by the
one-line division `workgroup_size // threads_per_transform` (kernel-generator.py:666) — not by
`Factorize`/`GetMaxRadicesSize`/`SupportedThreadsPerTransform`/`GetUtilizationRate`/
`DeriveMaxTPB`/`SupportedKernelConfigs`, none of which are invoked anywhere on this path (§5).

At plan-creation time, `NodeFactory::Decide1DScheme` (node_factory.cpp:637) finds
`pool.has_function(FMKey(8, precision))` (resp. 16) already true — because the entry was baked
in at build time — and immediately returns `CS_KERNEL_STOCKHAM` as a single leaf kernel node,
without ever calling any workgroup-size/thread-per-transform/utilization-rate search logic, and
without any "MIN_WGS floor" or "final workgroup size must evenly divide the length" check
existing anywhere in this call path. Those concepts (`min_wgs`, defaulting to 64,
`tuning_kernel_tuner.cpp:486`) exist solely inside the **tuning search-space generator**
`SupportedKernelConfigs`, which is used only when the offline tuner is deliberately asked to
find *alternative/better* configs for a length (to possibly replace the default via an entry
written into a `rocfft_solution_map.dat`) — it is never invoked to *originate* the default
config that ships in the library, and never invoked at all on the ordinary user's
`rocfft_plan_create` path.

One nuance worth flagging for the M2NDP port: `SupportedKernelConfigs` itself does **not**
actually enforce a hard `min_wgs = 64` floor for short lengths — it explicitly *lowers*
`min_wgs` when the length itself is smaller (`tuning_kernel_tuner.cpp:490-491`:
`min_wgs = (length < min_wgs) ? length : min_wgs;` followed by rounding down to a multiple of
64, which drives `min_wgs` to 0 for any length < 64 that isn't itself a multiple of 64). So even
if the tuner's search space *were* somehow exercised for N=8, it would not necessarily produce
zero candidates purely from a hardcoded 64 floor the way the ported logic assumed — the
zero-candidate outcome your M2NDP port observed is a property of how the port's search-space
formula was translated, not of anything in rocFFT's real production behavior, since real
rocFFT never runs that search for these lengths at all.

**Confidence: High**, directly evidenced by the exact config-table row, the exact call
sequence (`Decide1DScheme` → `pool.has_function` → true → `CS_KERNEL_STOCKHAM`), and the
absence of any tuning-search call in that sequence.

---

## 7. Confidence summary and exact files/URLs fetched

| # | Finding | Confidence |
|---|---------|-----------|
| 1 | Default tree-building = `NodeFactory::DecideNodeScheme`/`Decide1DScheme` etc. in node_factory.cpp, driven by `TreeNode::RecursiveBuildTree` in plan.cpp | High |
| 2 | Default KernelConfig = compiled-in table via `insert_default_entry`/`def_key_pool`, no runtime formula | High |
| 3 | N=8/N=16 exact WGS/TPT/factors from config_sbrr.py, tpb = wgs // tpt | High |
| 4 | Solution map always probed first (`BuildSingleDevicePlan<true>` default); only 3 archs ship `.dat` files | High (probe order + shipped files); Medium (whether those 3 files contain N=8/16 entries — not opened) |
| 5 | Tuning-search functions (`Factorize`/`GetMaxRadicesSize`/`SupportedThreadsPerTransform`/`GetUtilizationRate`/`DeriveMaxTPB`/`SupportedKernelConfigs`) exist only in tuning_kernel_tuner.cpp, reachable only via `TuningBenchmarker` tuning-mode flags | High |
| 6 | N=8/N=16 fully bypass the tuning search space in production | High |

### Files fetched (raw.githubusercontent.com/ROCm/rocm-libraries/develop/projects/rocfft/...)

- `library/src/node_factory.cpp`, `library/src/include/node_factory.h`
- `library/src/tree_node.cpp`, `library/src/include/tree_node.h`
- `library/src/tree_node_1D.cpp`, `library/src/include/tree_node_1D.h`
- `library/src/tree_node_2D.cpp`, `library/src/include/tree_node_2D.h`
- `library/src/tree_node_3D.cpp`, `library/src/include/tree_node_3D.h`
- `library/src/tree_node_real.cpp`, `library/src/include/tree_node_real.h`
- `library/src/tree_node_bluestein.cpp`, `library/src/include/tree_node_bluestein.h`
- `library/src/plan.cpp`, `library/src/include/plan.h`
- `library/src/include/function_pool.h`
- `library/src/include/function_map_key.h`
- `library/src/solution_map.cpp`, `library/src/include/solution_map.h`
- `library/src/compute_scheme.cpp`, `library/src/include/compute_scheme.h`
- `library/src/tuning_plan_tuner.cpp`, `library/src/include/tuning_plan_tuner.h`
- `library/src/tuning_kernel_tuner.cpp`, `library/src/include/tuning_kernel_tuner.h`
- `library/src/rocfft_kernel_config_search.cpp`
- `library/src/CMakeLists.txt`
- `library/src/device/kernel-generator.py`
- `library/src/device/generator.py`
- `library/src/device/generator/stockham_gen.cpp`, `library/src/device/generator/stockham_gen.h`
- `library/src/device/kernels/configs/config_arch.py`
- `library/src/device/kernels/configs/config_sbcc.py`
- `library/src/device/kernels/configs/config_sbrr.py`

### Directory listings obtained via GitHub Contents API
(`api.github.com/repos/ROCm/rocm-libraries/contents/projects/rocfft/...?ref=develop`)
- `library/src`
- `library/src/include`
- `library/src/device`
- `library/src/device/generator`
- `library/src/device/kernels`
- `library/src/device/kernels/configs`
- `library` (revealed the `solution_map/` directory)
- `library/solution_map` (revealed the 3 shipped `.dat` files)

Not fetched (out of scope / not needed for these questions): the binary/text contents of
`gfx908_rocfft_solution_map.dat`, `gfx90a_rocfft_solution_map.dat`,
`gfx942_rocfft_solution_map.dat` themselves; `rtc_stockham_gen.cpp`/`.h` (kernel source-code
emission, not config selection); `assignment_policy.cpp` (buffer assignment, post-scheme);
`config_sbcr.py`/`config_sbrc.py`/`config_2d_single.py`/`config_pp_3d.py` (same mechanism as
`config_sbcc.py`, already established).
