# GPU-planner-on-M2NDP: hardware-parameter mapping

## Purpose

The frozen `gpu-baseline-v1` reference (`gpu-clfft`, `gpu-rocfft-default`,
`gpu-vkfft`, `gpu-rocfft-tuned`) answers "what plan would a representative
real GPU's own library choose, and what happens if we force that exact GPU
choice onto M2NDP?" — every hardware-resource *input* that library's
planning algorithm consults is a fixed, representative-GPU constant
(clFFT's 32KiB LDS, rocFFT-default's 120 gfx908 CUs, VkFFT's 32-thread
warp), documented and held IDENTICAL across every length/batch that
baseline ever plans (see each module's own "FIXED IMPLEMENTATION
PARAMETERS" docstring section).

This document covers the **M2NDP-adapted** baselines added alongside it
(`gpu-clfft-m2ndp`, `gpu-rocfft-default-m2ndp`, `gpu-vkfft-m2ndp`), which
answer a different question: "what happens if we port the GPU library's
*planning algorithm itself* onto M2NDP, giving it M2NDP's own resource
characteristics instead of a representative GPU's?" These are the PRIMARY
baseline for GPU-vs-M2NDP-native comparison going forward — the frozen
`gpu-baseline-v1` names are kept, unmodified, as a reference/debugging
baseline (see each `*_m2ndp` function's own docstring for the exact,
minimal diff against its frozen sibling).

**The one rule every row below obeys**: the GPU library's own planning
CONTROL FLOW, TABLE, and DECOMPOSITION ALGORITHM never changes. Only
hardware-resource *inputs* to that unchanged algorithm are replaced, and
only where a real M2NDP quantity answers the *same question* the GPU
constant did. Where no such quantity exists, the GPU constant is kept
(classified `FIXED_ALGORITHM_PARAMETER` below) — never silently invented.

## Classification legend

- **EXACT_M2NDP_EQUIVALENT** — the M2NDP quantity answers literally the
  same physical question, at the same level of abstraction.
- **SEMANTIC_ADAPTER** — the M2NDP quantity plays the same *role* in an
  unchanged formula, but the underlying execution models differ enough
  (GPU workgroup vs. M2NDP NDP-unit/uthread) that claiming exact identity
  would overclaim.
- **FIXED_ALGORITHM_PARAMETER** — not a hardware query at all (a
  compiled table, a precision-derived constant, or a value M2NDP's own
  execution model makes moot) — kept unchanged, with justification.
- **NO_EQUIVALENT** — M2NDP has no analogous concept whatsoever; the
  value is kept only because the algorithm needs *some* value to run,
  never because it is meaningful on this target.

## Mapping table

| Library | Upstream hardware input | Old (source-faithful) value | M2NDP value | Semantic mapping | Classification | Affects |
|---|---|---|---|---|---|---|
| clFFT | `CLFFT_LDS_BYTES` (`envelope.limit_LocalMemSize`) | 32768 (32 KiB, typical real-GPU LDS) | `target.spad_capacity_bytes` = 122880 | Both answer "how many bytes of on-chip fast memory does one execution group get to hold an entire transform resident in" — feeds the SAME `GetMax1DLengthStockham = floor_po2(bytes / elem_size)` formula, unchanged, just fed a different budget. Kept consistent with the SAME quantity `map_cooperative_kernel`'s own feasibility check already uses for this baseline (not `max_concurrent_scratchpad_bytes`, which is a caller-facing contention throttle, a different concept). | SEMANTIC_ADAPTER | `plan()`'s single-kernel-vs-large-1D split decision (`is_1d_possible`'s own `large1d_threshold`) |
| clFFT | `CLFFT_MAX_WGS` (`CL_DEVICE_MAX_WORK_GROUP_SIZE`) | 256 | **unchanged** (256) | Gates which rows of the hand-tuned `SPEC_TABLE` specialization table are even consulted, and bounds `DetermineSizes`'s own fallback sizing. M2NDP's own cooperative-worker model has NO analogous hard ceiling on `workers_per_fft` at all — worker-wave virtualization already represents ANY positive cooperation width as `ceil(W/8)` sequential waves of 8 physical lanes (see `fft_plan_persistent.py`'s own docstring), so there is no smaller M2NDP number that would mean "this target's own workgroup-size limit" rather than simply crippling which specialization rows apply. Lowering this value would not adapt the algorithm to M2NDP's hardware — M2NDP's hardware imposes no such limit at all. | FIXED_ALGORITHM_PARAMETER | Which `SPEC_TABLE`/`DetermineSizes` branch `get_radices` takes |
| clFFT | `CLFFT_BLOCK_COMPUTE_GATE_SINGLE` (262144, `= 262144/PrecisionWidth`) | 262144 | unchanged | A precision-derived constant baked into clFFT's own SBCC eligibility table size, not a hardware query at all. | FIXED_ALGORITHM_PARAMETER | SBCC eligibility gate (`is_block_compute_length`) |
| rocFFT-default | `ROCFFT_DEFAULT_MULTIPROCESSOR_COUNT` (gfx908 CU count) | 120 | `target.num_ndp_units` = 32 | Both answer "how many independent physical compute units exist to keep busy" — feeds the SAME `total_batch // transforms_per_block >= multiprocessor_count` occupancy formula in `Decide1DScheme`, unchanged. Only fires at high batch (`total_batch // transforms_per_block` must reach the threshold) — at this project's usual `batch=1` the branch is never taken regardless of which count is used; the mapping's effect is batch-size-dependent, confirmed directly (`decide_scheme(4704, batch=40)`: 120-threshold picks `CS_L1D_CC`, 32-threshold picks `CS_KERNEL_STOCKHAM`). | SEMANTIC_ADAPTER | The single-kernel-vs-multi-kernel occupancy branch inside `Decide1DScheme`, batch>~32 only |
| rocFFT-default | `apply_solution`'s gfx908 solution-map file | fixed, real shipped file | unchanged | Not a hardware-resource input at all — a compiled tuning-database file for one specific real GPU arch, and proven (module docstring, exhaustive scan) to match zero configurations in this baseline's own FP32/out-of-place domain either way. Nothing to adapt. | FIXED_ALGORITHM_PARAMETER | No observable effect in this baseline's domain |
| VkFFT | `VKFFT_WARP_SIZE` (`Structs.h`'s "threads per warp/wavefront") | 32 | **unchanged** (32) — REVISED, see below | `interleave_chunk_uthreads` is documented (`target_profile.py`'s own comment) as a pure DRAM address-interleaving stride ("which physical NDP unit does address X land on"), not a SIMT lockstep execution width -- M2NDP microthreads are each independently generated/retired hardware FGMT, never executing in the per-instruction lockstep a GPU warp implies. An EARLIER revision of this baseline mapped `warp_size -> target.interleave_chunk_uthreads` (8); a dedicated re-audit (prompted by external review) found this conflated two genuinely different hardware concepts and reverted it. No verified M2NDP quantity answers the same question. | **NO_EQUIVALENT** (corrected from an earlier, incorrect SEMANTIC_ADAPTER classification) | `axisblock_batch_single_pass`'s own occupancy-estimate seed (num_passes==1 leaves only) — now identical for both baselines |
| VkFFT | `choose_pow2_grouping_radix`'s assumed 64 compute units (`active_threads_y = max_rhs // 64`) | 64 | `target.num_ndp_units` = 32 | Both answer "how many independent physical compute units exist to keep busy" — the SAME mapping already applied to rocFFT-default's own `multiprocessor_count`, feeding the SAME kind of workload-balance estimate, unchanged formula shape. Found during the same re-audit that reverted `warp_size` above — this is the ACTUAL CU-count-shaped hardware input this module has, previously missed entirely. | SEMANTIC_ADAPTER | `choose_pow2_grouping_radix`'s own power-of-2 grouping-radix DECISION (feeds `pow2_radix_sequence`'s real radix choice) — **empirically found to produce zero observed decision differences** across an exhaustive N=2^4..2^27 x max_rhs=1..65536 sweep in this project's own domain: the surrounding `VKFFT_ACTIVE_THREADS_X_FLOOR=128`/`max_loc_multipliers_pow2` clamps and the second (stage-count-minimizing) selection pass appear to absorb this input's effect at every value pair tested. Implemented and wired correctly; recorded here as a real, verified null result, not silently omitted. |
| VkFFT | `VKFFT_COALESCED_MEMORY_BYTES` (memory-coalescing transaction width) | 32 | 32 (value unchanged; **not yet wired to `target.uthread_bytes`**) | `target.uthread_bytes` (one M2NDP microthread's own vector width, 32 bytes = 8 FP32 lanes x 4 bytes) is numerically identical to VkFFT's own NVIDIA/AMD coalescing-width constant, but these are NOT proven to be the same concept (GPU warp-level memory coalescing granularity vs. one M2NDP microthread's own SIMD width) -- a real, open question this re-audit surfaced but did not resolve. Threading a `coalesced_memory_bytes` parameter through this module's ~15 call sites for a value that stays 32 either way was judged not worth the regression risk without first resolving the semantic question. | SEMANTIC_ADAPTER candidate, **unresolved** | `max_sequence_length_shared_memory_strided`'s own divisor — currently sourced from the plain module constant for both baselines |
| VkFFT | Persistent-leaf ping-pong scratchpad overhead vs. `max_sequence_length_shared_memory`'s own `spad_capacity_bytes / 8`-bytes-per-element formula | not modeled | not yet modeled | **Identified gap, not yet fixed.** `make_persistent_leaf_plan`'s own unconditional feasibility check (`fft_plan_persistent.py`) requires `16 * length` bytes, not `8 * length` — every VkFFT-baseline case lowers through this persistent mechanism once `workers_per_fft` exceeds the physical interleave chunk (`map_cooperative_kernel`'s own dispatch), which is nearly always. Concretely: at N=8192, `max_sequence_length_shared_memory(target) = 15360` says "1 pass fits" (8192 < 15360), but the REAL persistent-leaf requirement is `8192*16=131072` bytes against a `122880`-byte budget -- infeasible. The EXISTING, separate feasibility check inside `map_cooperative_kernel`/`_map_worker_wave_kernel` already catches this and correctly returns `RESOURCE_INFEASIBLE` (confirmed: `plan_m2ndp(8192)` does exactly this) rather than silently building a broken plan, so this is a PLANNING-ACCURACY gap, not a correctness bug -- VkFFT's own `choose_num_passes` is more optimistic than M2NDP's real lowering, causing avoidable `RESOURCE_INFEASIBLE` refusals for lengths a more accurate byte-per-element budget (16, not 8, specifically for the persistent-lowering case) would have routed to 2 passes instead. Deferred rather than rushed: the "correct" `bytes_per_element` is lowering-mechanism-dependent (this repo's own architectural choice, not a VkFFT algorithm property), and the CORRECT fix point (feed `choose_num_passes` an already-adjusted budget vs. adjust ad hoc at the top level) needs its own dedicated pass. | SEMANTIC_ADAPTER (formula must stay VkFFT's own — only the byte-per-element multiplier is M2NDP-lowering-specific), **not yet implemented** | `choose_num_passes`'s own 1-vs-2-vs-3-pass decision; concretely demonstrated at N=8192 |
| VkFFT | `VKFFT_MAX_THREADS_NUM` / `VKFFT_MAX_COMPUTE_WORKGROUP_SIZE` (`VkPhysicalDeviceLimits`-derived, both 1024) | 1024 / 1024 | unchanged | Real hardware-capability inputs in VkFFT's own source (not algorithm constants) — but M2NDP's own worker-wave virtualization means there is no hard ceiling on cooperative worker count analogous to a GPU's hard workgroup-size limit (identical reasoning to clFFT's own `CLFFT_MAX_WGS`, above): M2NDP CAN represent any batch/threads-per-transform product via waves, so there is no smaller M2NDP number that would mean "this target's own hard thread ceiling" rather than simply crippling batch sizing for no M2NDP-architectural reason. Missed in the original version of this document (found via external review) — added now as an explicit, reasoned NO_EQUIVALENT rather than left undocumented. | NO_EQUIVALENT | `axisblock_for_leaf`'s own batch-clamping divisor searches |
| VkFFT | `VKFFT_NUM_SHARED_BANKS` (GPU shared-memory bank count, 32) | 32 | unchanged | Gates a real VkFFT axis-swap/batching heuristic (`_postprocess_axis_upload0`'s own `axis_block0 < VKFFT_NUM_SHARED_BANKS // 4` check) whose real purpose is avoiding GPU shared-memory BANK CONFLICTS -- a hardware behavior this project has NOT verified M2NDP's own scratchpad even has (no bank-conflict model appears anywhere in this repo's own M2NDP target/simulator-facing code reviewed so far). Missed in the original version of this document (found via external review). Kept unchanged rather than guessed in either direction (removing the heuristic entirely, or inventing an M2NDP bank count) pending a dedicated look at the M2NDP-Detour simulator's own scratchpad source for whether banking/conflict behavior exists at all. | **NO_EQUIVALENT, unresolved** — needs simulator-source verification before any change | `_postprocess_axis_upload0`'s own axis-swap-eligibility branch |
| VkFFT | `VKFFT_VENDOR_IS_NVIDIA` (register-count table branch AND the batch-halving occupancy loop) | `True` | unchanged | Two distinct uses, re-examined after external review: (1) the Rader/Bluestein boundary constants this flag also selects have PROVEN zero observable effect (Rader/Bluestein are out of this project's scope entirely, module docstring). (2) The "NVIDIA vendor halving loop" (`_postprocess_axis_upload0`'s own `batch //= 2` occupancy-avoidance loop) DOES have a real, observable effect on `batch` sizing -- this is a genuine occupancy-shaping heuristic whose INTENT (avoid over-subscribing one physical execution group beyond ~2x the aim-thread target) is architecture-general, not actually NVIDIA-specific magic, even though real VkFFT happens to gate it behind vendor detection. Disabling it for M2NDP without evidence it should run differently would be an unjustified guess in the OTHER direction (the task's own "do not invent" rule cuts both ways); kept unchanged pending dedicated verification of whether M2NDP's own occupancy characteristics warrant a different threshold. | FIXED_ALGORITHM_PARAMETER for (1); **unresolved** for (2) | (2): `_postprocess_axis_upload0`'s own batch-halving loop |
| VkFFT | `fixMaxRaderPrimeMult=89` / `fixMinRaderPrimeFFT=17` (Rader-vs-Bluestein residual classification constants) | 89 / 17 | unchanged | Vendor/precision-tied algorithm tuning constants from VkFFT's own real source, not hardware queries; Rader/Bluestein are out of this task's scope regardless (section 9 of the task this doc was built from). | FIXED_ALGORITHM_PARAMETER | Residual-length classification only, unreached in this baseline's own supported domain |
| VkFFT | `target.spad_capacity_bytes` (LDS-equivalent for `max_sequence_length_shared_memory*`/`choose_num_passes`/`split_*`) | already M2NDP-derived | unchanged (already correct) | Pre-existing target adaptation, confirmed consistent across every one of these functions during this audit — no fix needed regarding WHICH target field is used (see the separate ping-pong-overhead row above for the byte-per-element FORMULA gap this same field feeds). | SEMANTIC_ADAPTER (revised from an earlier, overclaimed EXACT_M2NDP_EQUIVALENT — a GPU's LDS and M2NDP's scratchpad are analogous, not identical, resources) | 1/2/3-pass decision, axis-split factor choice |

## Why `batch=1` mostly hides the rocFFT-default mapping

`ROCFFT_DEFAULT_MULTIPROCESSOR_COUNT` only gates a `total_batch //
transforms_per_block >= multiprocessor_count` comparison — at `batch=1`
(this project's own default, isolating plan quality from replica count)
the left side is always `<= 1`, so neither 120 nor 32 as the right side
changes the outcome. This mapping's real effect only shows up in a
batch-swept experiment, not the standard single-batch 27-N sweep. Recorded
here rather than silently omitted, per this task's own "no equivalent /
batch-size-dependent" honesty requirement.

## Related lowering documents

`docs/gpu_baseline_clfft_sbcc_lowering.md` and `docs/gpu_baseline_rocfft_
cc_trtrt_lowering.md` cover the EXECUTION-LOWERING gaps this task also
closed (clFFT block-compute, rocFFT-default CS_L1D_CC/TRTRT) — a
different axis from this document's own hardware-PARAMETER mapping; both
are part of the same overall M2NDP-adapted-baseline effort.

## Revision (external review)

An external review of this document's own first version found the VkFFT
section incomplete and one mapping (`warp_size -> interleave_chunk_
uthreads`) conceptually wrong — verified independently against this
project's own `target_profile.py` documentation and reverted here (the
`VKFFT_WARP_SIZE` row above). The review's other concrete findings
(`choose_pow2_grouping_radix`'s own hardcoded 64-CU assumption; `VKFFT_
MAX_THREADS_NUM`/`MAX_COMPUTE_WORKGROUP_SIZE`/`NUM_SHARED_BANKS`
previously missing from this table entirely; the persistent-leaf ping-
pong byte-per-element mismatch, demonstrated concretely at N=8192; the
NVIDIA-vendor batch-halving loop's real occupancy effect) are all now
reflected above, each independently re-verified against this module's own
source and this project's own `TargetProfile`/`fft_plan_persistent.py`
before being accepted or reasoned about further — never accepted on the
review's own say-so alone.

## Verification

`verification/verify_gpu_baseline_m2ndp_adapted.py` — for each mapped
parameter, confirms (a) the M2NDP baseline's decisions differ from the
source-faithful baseline's ONLY in the documented direction/branch, never
elsewhere, and (b) every reachable `plan_m2ndp` result is numerically
correct against `numpy.fft`. `verification/verify_gpu_baseline_source_
fidelity.py`/`verify_gpu_baseline.py`/`verify_gpu_baseline_golden.py` (the
pre-existing suites covering the frozen `plan()` entry points) all still
pass unmodified — confirming this work never touched the frozen baseline's
own behavior.
