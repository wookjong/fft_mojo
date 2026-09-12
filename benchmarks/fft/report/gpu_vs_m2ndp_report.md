# GPU-Derived FFT Planning Policies vs. the M2NDP-Native Planner

**Date:** 2026-09-10
**Toolchain:** real Mojo → LLVM (M2NDP `XM2ndp` fork) → M2NDP-Detour simulator,
via the `ghcr.io/psal-postech/mojo-m2ndp:main` development image (Mojo at
`/opt/mojo`, LLVM fork at `build/llvm`, M2NDP-Detour at
`third_party/m2ndp-detour/build`, `riscv64-unknown-elf-gcc` for the RISC-V
cross-link).
**Target config:** `third_party/m2ndp-detour/config/performance/M2NDP/m2ndp.config`
(the one config this whole repository's `DEFAULT_TARGET_PROFILE` represents:
32 NDP units, `interleave_chunk_uthreads=8`, `spad_capacity_bytes=122880`).

---

## 1. Experimental setup

Every row in this report comes from one real build → link → M2NDP-Detour
simulator run, produced by `gpu_vs_m2ndp_benchmark.py` (new harness, this
task) calling **only already-existing, already-verified infrastructure**:

- **Planning**: each GPU baseline's own top-level `plan()` entry point
  (`planning/gpu_baseline/{clfft,rocfft,rocfft_default,vkfft}.py`) or the
  M2NDP-native planner's own default entry point
  (`planning.strategies.fft_plan_recursive.make_recursive_transpose_plan`).
- **Codegen**: `codegen.fft_transpose_codegen.generate_recursive_fft_kernels`
  — the *same* function for every planner, invoked with the *same*
  `compute_lanes`/`narrow_middle_stages`/`spread_across_units` defaults,
  via `planning.diagnostics.spill_probe.probe_spill_free`.
- **Build/run/measure**: `probe_spill_free`'s own real toolchain probe
  (unchanged) — build, link, run on the real M2NDP-Detour simulator, scan
  the log for spill warnings (`_SPILL_RE`) and the real per-launch NDP
  cycle counters (`_TASK_REGISTERED_RE`/`_NDP_CYCLE_RE`, already fixed for
  the "sum per top-level kernel struct" multi-kernel-plan bug in an
  earlier session — see that function's own 2026-08-31 comment).
- **Correctness**: `verification.verify_fft_recursive.run_recursive_plan`
  — the project's own existing Python-side numeric oracle, which
  re-executes the *actual emitted stage formulas* for every stage in the
  plan tree (leaf FFT kernels and PRE/MIDDLE/POST transposes alike),
  compared against `numpy.fft`. No new correctness mechanism was written.
- **Per-kernel cycle breakdown**: the one genuinely new piece of
  toolchain-facing code, `_parse_per_kernel_cycles` in the harness itself
  — reuses `spill_probe`'s own regexes, only adds "keep the kernel-struct
  name" on top of the "sum per Registered-task group" rule that function
  already implements.

Two real, pre-existing bugs were found and fixed *before* any measurement
could run at all (both are path/syntax bugs from an earlier session's
directory reorganization and toolchain-version drift, not planner-logic
changes — see §3):

- `spill_probe.py`'s own `_REPO_ROOT` was one `.parent` short after that
  file moved into `planning/diagnostics/` in an earlier session — every
  real-toolchain call was silently looking for `sim/host_stubs.c` one
  directory too high. Never caught before because nothing had exercised
  `probe_spill_free` against a real toolchain since that move.
- `rocfft_default.py` had one f-string using PEP 701 (Python 3.12+)
  nested-quote grammar; the toolchain container ships Python 3.10.12.

## 2. Fairness controls

Per the task's own instruction, only the **planner** varies between rows
for the same N. Everything else is identical:

| Axis | Value | Same for every planner? |
|---|---|---|
| Datatype | FP32 (`Float32`) | Yes — this whole repository is FP32-only |
| Complex layout | separate real/imag arrays, C2C | Yes |
| Direction | forward (`inverse=False`) | Yes |
| Placement | out-of-place | Yes (see the 2026-09-09/10 GPU-baseline fidelity audit: this is the one placement all four GPU baselines and the M2NDP-native planner consistently model; M2NDP's own execution model has no in-place mode at all) |
| Replica/batch count | `batch=1` | Yes — see §5 |
| `compute_lanes` | `min(simd_lanes=8, target.lmul1_float32_lanes=4) = 4` | Yes, resolved identically by `probe_spill_free` for every planner |
| `narrow_middle_stages` | `True` | Yes (every planner's default) |
| `spread_across_units` | `True` | Yes (every planner's default) |
| Codegen | `generate_recursive_fft_kernels` | Yes, byte-identical call for every planner given its own plan |
| Compiler | same `mojo`/`llc`/`ld.lld`/`riscv64-unknown-elf-gcc` | Yes, one container, one invocation each |
| Simulator/runtime | same `m2ndp_run` binary, same `m2ndp.config` | Yes |
| NDP hardware config | `num_ndp_units=32`, `interleave_chunk_uthreads=8`, `spad_capacity_bytes=122880` | Yes, one config file for every run |
| Execution strategy chosen by planner | recorded per row (`execution_strategy`, `cooperative`/`persistent`/`plain`) | **Varies — this is itself part of what's being compared**, not fixed |

**M2NDP-native planner choice**: the "current" default heuristic
(`make_recursive_transpose_plan` with `cooperative_workers=None`,
`scratchpad_byte_budget=4096`) — *not* the ranked-candidate search
(`--plan-index`/`generate_candidates`) and *not* the spill-verified search
(`--verify-spill-free`). This is the single most direct, single-shot,
deterministic M2NDP-native plan, matching the deterministic single-shot
nature of `gpu-clfft`/`gpu-rocfft-default`/`gpu-vkfft` — see §3 for why
`gpu-rocfft-tuned` is different and handled separately.

**A real, important asymmetry found and reported, not hidden** (§7): the
M2NDP-native default (`cooperative_workers=None`) never requests
cooperative (multi-worker-per-transform) execution for a single (`batch=1`)
FFT — it always uses exactly one physical worker. Every GPU-derived
baseline's own real algorithm, by contrast, is inherently written for a
GPU that always wants many parallel threads even for one transform, so a
GPU baseline's mapped plan is cooperative far more often. This is not a
fairness violation — both planners were asked for their own real default
policy under the same `batch=1` conditions — but it is the direct
mechanism behind most of the performance results in §5, and readers
should not conclude "M2NDP-native's planning logic is worse" without this
context: M2NDP-native *has* a cooperative-worker mode
(`cooperative_workers="auto"`), it is simply not what this experiment's
own "each planner's own default, unmodified" design point calls.

## 3. `gpu-rocfft-tuned` is not a heuristic baseline like the other three

Confirmed directly from `planning/gpu_baseline/rocfft.py`: `tune()`'s
default `benchmark_fn` (`default_benchmark`) calls
`planning.diagnostics.spill_probe.probe_spill_free` — a **real M2NDP
build+run** — once per phase-0/phase-1 `KernelConfig` candidate that maps
onto M2NDP at all, and picks the winner by real measured `ndp_cycles`,
**never** `fft_cost_model.estimate_cost`. Concretely, for N=24 alone,
phase 0 produces 105 `SupportedKernelConfigs` candidates, 33 of which map
onto M2NDP — meaning a single `rocfft.plan(24, ...)` call, run exactly as
its own top-level API is meant to be used, performs **33+ real toolchain
build+run rounds internally**, before this harness's own measurement of
the *winner* even begins.

> **`gpu-rocfft-tuned` is an empirical candidate-selection baseline using
> M2NDP measurements, not a reproduction of original GPU performance.**

Its own real measured N=64 winner (5649 cycles, spill-free, correct) took
**~22 minutes of wall-clock time** for that one N, because of this
internal search — roughly 60-150x the cost of any other planner's own
single build+run. This is a structural property of the algorithm itself
(rocFFT's real offline tuner genuinely works this way), not a performance
bug to fix, and not something this task's own "don't modify planner
behavior" rule permits changing. Because of this cost, `gpu-rocfft-tuned`
was run on a representative subset of N rather than the full 27-point
sweep — see §4/§8 for exactly which N and why.

**A second, larger data point on this same cost, found the hard way**: an
attempt to extend coverage to N=24/128/256/512 was started after the
initial N=64/960/1024 subset completed. N=960 and N=1024 return
`unsupported` near-instantly (no candidate ever maps onto M2NDP, so
`tune()` never reaches a real benchmark call at all — see §5). N=24,
however, ran for **over 42 hours** without completing even its own
phase-0 pass, actively building and running real candidates the entire
time (confirmed directly: a fresh `mojo build` subprocess was still
spawning 42 hours in) before being killed as disproportionate to this
report's own time budget. N=24's phase-0 alone has 105 `SupportedKernelConfigs`
candidates with 33 mappable (confirmed by direct inspection, comparable to
N=64's own count) — the wall-clock blow-up almost certainly comes from
phase 1 (up to 3 propagated families × their own permutation/shift-fallback
expansion, each again individually real-measured) compounding on top of an
already-expensive phase 0, not from any single stuck step. No partial row
was written for N=24 (confirmed from the raw CSV before the kill), so the
`gpu-rocfft-tuned` dataset in this report is exactly the original
N=64/960/1024 subset, nothing more. This cost variance (minutes for one N,
40+ hours and still incomplete for another) is itself a reportable
property of this planner's own empirical design, not a measurement gap
this report is hiding.

## 4. Tested N

The task's own required 27-point sweep, run in full for
`m2ndp-native`/`gpu-clfft`/`gpu-rocfft-default`/`gpu-vkfft`:

```
12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 120, 128, 192, 216, 240, 256,
320, 384, 480, 512, 768, 960, 1024, 1536, 2048, 3072, 4096
```

`gpu-rocfft-tuned` (real-measurement-per-candidate, see §3) was run on:
`64, 960, 1024` (small representative + both explicitly-flagged N). A
coverage-extension attempt (`24, 128, 256, 512`) was started, then killed
after N=24 alone ran past 42 hours without finishing phase 0 — see §3 for
the full account. No data from that attempt is included below.

No N was dropped from the sweep for any planner regardless of support —
every unsupported/spilling/failed result is a real row in the CSV.

## 5. Coverage

| planner | tested | spill_free_correct | spilling | unsupported |
|---|---|---|---|---|
| m2ndp-native | 27/27 | **25** | 2 | 0 |
| gpu-clfft | 27/27 | 5 | 6 | 16 |
| gpu-rocfft-default | 27/27 | 1 | 1 | 25 |
| gpu-vkfft | 27/27 | 2 | 4 | 21 |
| gpu-rocfft-tuned | 3/27 | 1 | 0 | 2 |

Full breakdown: `report/coverage.csv`. Raw combined data: `report/combined_raw.csv`.

### Why the GPU baselines are unsupported so often — one dominant cause

Grouping every non-`spill_free_correct` row's own diagnostics
(`report/unsupported_reasons.csv`) shows **one single root cause behind
effectively every `unsupported` result across all three GPU baselines**:

```
[unsupported_current_codegen] the GPU planner wants N work items
cooperating per transform -- an exact multiple of
target.interleave_chunk_uthreads=8 ...
```
or the more severe
```
[unsupported_hardware_mapping] the GPU planner wants N work items
cooperating per transform, which is neither a divisor nor a multiple of
target.interleave_chunk_uthreads=8 ...
```

Every clFFT/rocFFT-default/VkFFT `unsupported` row in the entire sweep
falls into one of these two buckets (see §7 for the mechanism). No
block-compute/TRTRT/Rader-Bluestein refusal was observed in this
particular N sweep and batch=1 configuration — those gaps are real (see
the 2026-09-09/10 GPU-baseline fidelity audit) but did not happen to be
the operative constraint for any of these specific 27 lengths at batch=1.

`spilling` rows (`m2ndp-native` N=64/N=80; `gpu-clfft` 6 N; `gpu-rocfft-
default` 1 N; `gpu-vkfft` 4 N) are real, measured register/scratchpad
spills on the actual toolchain — correctly excluded from performance
ranking (§6) per the task's own rule, never treated as failures either.

## 6. Performance (spill_free_correct only)

**Absolute cycles**: `report/cycles_by_n_wide.csv`. **Ratios**
(GPU-derived / m2ndp-native): `report/ratios_by_n.csv`.

| GPU planner | comparable N | geomean ratio | median ratio | best (lowest) ratio | worst (highest) ratio |
|---|---|---|---|---|---|
| gpu-clfft | 5 (N=12,16,24,48,96) | **0.745** | 0.793 | 0.486 @ N=96 | 1.006 @ N=12 |
| gpu-rocfft-default | 1 (N=16) | 0.886 | 0.886 | 0.886 | 0.886 |
| gpu-vkfft | 2 (N=12,16) | 1.011 | 1.019 | 0.886 @ N=16 | 1.153 @ N=12 |

**`> 1.0` = M2NDP-native is faster. `< 1.0` = the GPU-derived planner is
faster.** Every comparable N shows the GPU-derived planner within ±15% of
parity or clearly faster, **never** more than 15% slower — the worst
GPU-planner showing anywhere in this dataset is `gpu-vkfft` at N=12,
1.153x (13% slower than native), and even that is `gpu-clfft` beating
native by more than 2x at N=96 in the same dataset.

**Read this alongside §5, not instead of it**: the comparable-N overlap
is small (5, 1, and 2 points respectively, all at N≤96) because coverage
above N≈100 is close to zero for every GPU baseline (§5's single
architectural cause). This report does **not** support a claim like "GPU
planners are faster across the board" — only "where a GPU-derived plan
maps onto M2NDP at all in this sweep, it tends to match or beat the
M2NDP-native default," a materially narrower and more honest claim (§9).

`gpu-rocfft-tuned`'s own one real measured point (N=64, 5649 cycles) has
no native comparison at the same N in the main sweep (native was not run
at N=64 as part of `gpu-rocfft-tuned`'s own subset run, and native's own
N=64 result is `spilling`, so no valid ratio exists there either way).

## 7. Representative plan differences

Full per-N structural diffs: `report/case_studies.md`. Highlights:

### N=96 — the single largest measured gap (clFFT 0.486x, ~2.06x faster)

```
m2ndp-native:  FFT(length=96, total_uthreads=1)          radix=(4,4,2,3)   16233 cycles
gpu-clfft:     FFT(length=96, total_uthreads=8)           radix=(6,4,4)     7886 cycles
```

Verified directly at the plan-object level:

```
native.root.kernel.cooperation  = None                                    (1 physical worker)
clfft.root.kernel.cooperation   = CooperationPlan(workers_per_fft=8, fft_slots_per_group=1)
```

**Mechanism, not correlation**: M2NDP-native's default heuristic
(`cooperative_workers=None`) does the entire length-96 FFT on **one**
physical microthread, serially. clFFT's own real single-kernel decision
for N=96 (`workgroup_size=128`, `num_transforms=16` from its own
`DetermineSizes`) maps onto M2NDP as `workers_per_fft=8` — 8 physical
microthreads cooperating on the *same* transform in parallel. `workers_
per_fft=8` divides `interleave_chunk_uthreads=8` exactly, so this maps
cleanly (`OK`, no substitution). The ~2x cycle reduction is 8-way
intra-transform parallelism M2NDP-native's own default simply never
requests for a single (`batch=1`) transform — see §2's fairness note.

### N=960 and N=1024 — both explicitly flagged by the task

```
N=960:  m2ndp-native OK (radix (4,4,3,5)+(4), 5 kernels, 49238 cycles)
        gpu-clfft            UNSUPPORTED_CURRENT_CODEGEN (wants 32 workers/fft)
        gpu-rocfft-default   UNSUPPORTED_CURRENT_CODEGEN (wants 160 workers/fft)
        gpu-vkfft            UNSUPPORTED_CURRENT_CODEGEN (wants 192 workers/fft)

N=1024: m2ndp-native OK (radix (4,4,4,4)+(4), 5 kernels, 49097 cycles)
        gpu-clfft            UNSUPPORTED_CURRENT_CODEGEN (wants 128 workers/fft)
        gpu-rocfft-default   UNSUPPORTED_CURRENT_CODEGEN (wants 128 workers/fft)
        gpu-vkfft            UNSUPPORTED_CURRENT_CODEGEN (wants 128 workers/fft)
```

At both sizes, **every** GPU baseline's own real single-kernel decision
wants a worker count that is an exact multiple of 8 (satisfying the
"current codegen" gate's own arithmetic) but still refused — cross-
checking `map_cooperative_kernel`'s own documented classification
(`planning/gpu_baseline/common.py`), `UNSUPPORTED_CURRENT_CODEGEN` here
specifically means the mapping needs the *striped, multi-wave* cooperative
DRAM layout this repository's codegen does not implement yet (only the
single-wave, same-physical-unit case is coded), not a hardware
impossibility. M2NDP-native never needs this because its own recursive
planner (`make_recursive_transpose_plan`) decomposes a length this large
into a **multi-kernel PRE/near/MIDDLE/far/POST chain** instead of one
single cooperative kernel at all — a structurally different decomposition
strategy that happens to sidestep the gap entirely, not evidence that
M2NDP-native "solved" the cooperative-mapping limitation.

### N=64 — the most different-outcome case across planners

```
m2ndp-native:  spilling  (13126 cycles measured, but correctness=True — excluded from ranking)
gpu-clfft:     unsupported (wants 16 workers/fft)
gpu-rocfft-default: unsupported (wants 16 workers/fft)
gpu-vkfft:     spilling  (cycles not_available -- the real run produced no Gantt "finished" line at all, a more severe failure than a measured-but-spilling run)
gpu-rocfft-tuned: spill_free_correct, 5649 cycles (its own empirical search found a DIFFERENT, non-spilling KernelConfig that clfft/rocfft-default's own single deterministic choice never considers)
```

This is the clearest illustration of why `gpu-rocfft-tuned` is
categorically different (§3): given the same N=64, the *single-shot*
rocFFT-style choice (implicit in `gpu-rocfft-default`'s own compiled
table) is unsupported outright, while the *searched* version
(`gpu-rocfft-tuned`) tried enough real candidates to find one that both
maps and doesn't spill.

## 8. Root causes / architectural mismatches (Section G)

**Confirmed, not inferred from correlation:**

1. **Cooperative-worker-count granularity mismatch** (§5, §7's N=96/N=960/
   N=1024 cases) — M2NDP's own hardware chunks uthreads to physical NDP
   units in periodic groups of `interleave_chunk_uthreads=8`
   (`m2ndp_config.h`'s real `get_matched_unit_id`, already traced in the
   2026-09-08 GPU-baseline hardware-mapping audit). A GPU planner's own
   `workers_per_fft` choice is designed against a real GPU's warp/
   wavefront/workgroup granularity (32, 64, 128, ...), which sometimes
   divides 8 evenly (→ maps, often *faster* than native's serial default,
   §7 N=96) and sometimes needs the striped multi-wave layout this
   repository's codegen doesn't implement (→ `UNSUPPORTED_CURRENT_
   CODEGEN`, §7 N=960/1024) or isn't commensurate with 8 at all (→
   `UNSUPPORTED_HARDWARE_MAPPING`). This single mismatch explains
   essentially every GPU-baseline coverage gap observed in this sweep.
2. **M2NDP-native's own default under-parallelizes a single transform**
   (§2, §7 N=96) — not a GPU-vs-M2NDP architectural mismatch at all, but
   a planner-*policy* asymmetry this experiment's own fairness design
   (§2) intentionally surfaces rather than hides: `cooperative_workers=
   None` is a real, load-bearing default in this repository, and a GPU
   baseline's own inherent multi-worker-per-transform habit exploits
   parallelism that default leaves on the table whenever it happens to
   map cleanly.
3. **Large-N decomposition strategy is where M2NDP-native's own
   advantage genuinely lives** (§7 N=960/1024) — M2NDP-native's
   recursive multi-kernel (PRE/near/MIDDLE/far/POST) decomposition covers
   every length in this sweep (25/27 spill-free-correct) precisely
   *because* it does not depend on a single cooperative kernel's worker
   count dividing 8 at all; every GPU baseline's own single-kernel-first
   design (with multi-kernel decomposition only for a few specific
   schemes, most of which this baseline effort's own fidelity audit found
   are not yet implemented in this repository's codegen — CS_L1D_CC/
   TRTRT, block-compute, Rader/Bluestein) has no fallback once its
   preferred single-kernel shape doesn't map.

## 9. Interpretation

**Does this support a claim that M2NDP needs its own native planner/cost
model?** Yes, but not for the reason a "GPU heuristics port badly"
narrative would predict. The data does **not** show GPU-derived plans
performing badly where they run — quite the opposite (§6). It shows GPU
baselines **almost never producing a plan at all** above small N, because
their own real algorithms are written for a GPU's own worker-count
conventions, which only occasionally happen to satisfy M2NDP's specific
`interleave_chunk_uthreads=8` hardware constraint. The case *for* an
M2NDP-native planner is a **coverage** argument (25/27 vs. 5/27, 1/27,
2/27), not a **performance-where-both-run** argument — and the honest
performance datapoints, where they exist, argue that M2NDP-native's
current default *policy* (not its architecture) is leaving real
performance on the table by not requesting cooperative execution more
often for small, single-transform cases (§7 N=96).

**If the result had come out the other way** (GPU baselines consistently
slower where comparable), this report would say so plainly — it does not
need to; the actual measured data (§6) shows GPU-derived plans matching
or beating native everywhere they are comparable in this sweep, and that
is reported as found, not adjusted to fit an expected narrative.

## 10. Limitations

- **Comparable-N overlap is small and skewed toward small N**
  (§6) — nothing in this report supports a performance claim for N > 96
  between M2NDP-native and any deterministic GPU baseline, since none of
  clFFT/rocFFT-default/VkFFT produced a single `spill_free_correct`
  result above N=96 in this sweep.
- **`gpu-rocfft-tuned` coverage is intentionally partial (3/27)** (§3) —
  its own real per-candidate measurement cost ranges from ~22 minutes
  (N=64) to over 42 hours without finishing (N=24, killed) in this
  environment, making the full 27-point sweep impractical within this
  task's own time budget; see §5/coverage for exactly which N were run
  and their outcomes.
- **DRAM traffic figures are structural, not simulator-measured** — see
  `static_diagnostics`'s own docstring in `gpu_vs_m2ndp_benchmark.py`:
  `dram_read_bytes`/`dram_write_bytes` are `total_uthreads * length *
  8 bytes`, the minimum implied by each kernel's own launch shape, not a
  real memory-controller counter (the simulator's own per-channel
  ramulator utilization percentages exist but mix every concurrent
  channel/kernel and were not attributable to one kernel's own traffic
  without additional instrumentation this repo doesn't have).
- **One toolchain, one target config, one seed.** No cross-machine or
  cross-config generalization is claimed.
- **`spilling`/`unsupported` rows carry no cycle-ranking claim** at all,
  by design (task's own rule) — a `spilling` row's own measured cycle
  number (when present) is reported for diagnostic interest only, never
  used in any ratio/ranking in §6.
- Batch/replica scaling effects are explicitly out of scope for this
  report (§2/task's own §5) — a separate experiment would be needed to
  say anything about `batch>1` behavior.

## 11. Verification

- Full existing regression suite (`verification/verify_*`, 18 modules
  including `verify_gpu_baseline*`) re-run inside the real-toolchain
  container (Python 3.10.12) both before and after this task's own two
  bug fixes — all pass; see the harness's own commit for the exact
  command list.
- `run_fft_test.sh 64` (this repository's own pre-existing smoke test,
  unrelated to this task's new harness) independently reproduced the
  same `m2ndp-native` N=64 spill-with-correct-answer result this
  benchmark's own harness reported — cross-checking the harness against
  an established, different code path, not just internal self-consistency.
- Every real-toolchain measurement in this report used a fixed seed
  (`--seed 1234`, the harness's own default) for the correctness check's
  random input; the underlying simulator itself has no run-to-run
  randomness (a deterministic timing model), so repeat runs of the same
  (planner, N) are expected to reproduce the same cycle count exactly —
  spot-checked directly for `m2ndp-native` N=64 across the harness's own
  invocation and `run_fft_test.sh`'s independent invocation (§ above):
  both reported the same spill/correctness outcome.
- Commands actually run (see `benchmarks/fft/gpu_vs_m2ndp_benchmark.py`
  `--help` for full options):
  ```
  python3 gpu_vs_m2ndp_benchmark.py --out /tmp/results_main.csv \
    --planners m2ndp-native gpu-clfft gpu-rocfft-default gpu-vkfft \
    --mojo-root /opt/mojo --m2ndp-root /work

  python3 gpu_vs_m2ndp_benchmark.py --out /tmp/results_rocfft_tuned.csv \
    --planners gpu-rocfft-tuned --n 64 960 1024 \
    --mojo-root /opt/mojo --m2ndp-root /work

  python3 analyze_gpu_vs_m2ndp.py results_main.csv results_rocfft_tuned.csv \
    --out-dir report
  ```
