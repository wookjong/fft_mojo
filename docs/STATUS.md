# Status and open work

Where the PoC stands and what is left to do. Everything below was measured
on the artifacts in `out/`; nothing is projected. Regenerate with
`./scripts/build.sh && ./scripts/verify.sh`.

The backend-facing contract is in [`INTERFACE.md`](INTERFACE.md); this file
does not repeat it.

Last updated: 2026-07-23, against Mojo `1.0.0b2.dev2026061203`.

---

## 1. What works

| | |
|---|---|
| RISC-V/RVV codegen | `target triple = "riscv64-unknown-unknown-elf"`, `vsetvli` / `vle32.v` / `vadd.vv` / `vse32.v` selected from ordinary Mojo `SIMD` |
| Custom target | `#kgen.target<...>` written by hand, bypassing `std.gpu`'s closed vendor detection |
| Scratchpad | `internal addrspace(3) global` — an untyped byte blob named `memory_blob_<hash>`, not the `name=` argument — shared across kernels when declared as a comptime struct member |
| Atomics | `atomicrmw add` / `fadd`, on ordinary memory and on `addrspace(3)`, at `monotonic` ordering |
| Indirect access | plain load → sext → GEP → load; no special construct needed |
| Predicate scan | `v.lt(x)` selects `vmslt.vx` |
| Multi-kernel modules | `mojo build --emit llvm` with `@export`; several kernels per file |

Benchmarks ported from
[M2NDP-public](https://github.com/PSAL-POSTECH/M2NDP-public) — 6 of ~23:

| Benchmark | Exercises |
|---|---|
| `memcpy` | vector load + store |
| `memset` | scalar splat to vector store |
| `vector_add` | RVV vectorization |
| `imdb_lt_int64` | predicate scan → bitmap |
| `spmv` | indirect access, atomic combine |
| `histogram` | scratchpad shared across INIT/BODY/FINAL phases |

### LLVM baseline

`./scripts/build-llvm.sh check`, against the pinned submodule (LLVM 23.1.0,
`release/23.x`, RISC-V only, assertions on): **2596/2596 RISC-V CodeGen lit
tests pass.**

The baseline before `FeatureVendorXM2ndp` was 2595/2595. Adding the feature
broke exactly one test — `features-info.ll`, which checks the full
`-mattr=help` listing — and the baseline is what made that immediately
attributable rather than a mystery. Updated, plus a new
`attributes-m2ndp.ll`, giving 2596.

The artifacts also round-trip through that build: `llc` accepts all six
`out/*.ll` and `llvm-mc -filetype=obj` assembles what it produces. So the
frontend and backend LLVM versions are compatible in practice, not only by
version number.

---

## 2. What the backend has to support

The contract lives in [`INTERFACE.md`](INTERFACE.md) — symbols, scratchpad
placement and addressing, synchronization, the two operations that have no
spelling at this level, and recovering the mapped address. In rough order of
how much is blocked on each:

1. **The four ID symbols** → register reads, marked `readnone` so they hoist
2. **Scratchpad placement** — addrspace(3) globals currently lower to
   `.comm` (ordinary `.bss`); the annotation does not survive to the object
   file
3. **Vector atomic** and **mask-to-bitmap** — no spelling exists; each needs
   its own intrinsic
4. **FP atomic add** — expands to an LR/SC retry loop today
5. **Recovering `ADDR`/`OFFSET`** from `base[id * W]`

The architecture questions that used to block this are settled and written
up in INTERFACE.md: the scratchpad base is the same on every core, one
launch group is resident on a core at a time, and the contents survive
kernel launches within a task. Together those mean the scratchpad keeps a
fixed address and needs one offset per global — AMDGPU's LDS model.

One question is still open, and only item 1 waits on it: **which four
registers carry the µthread IDs.** Registering the vendor feature and the
scratchpad work can both start without it.

## 3. Toolchain constraint

| Version | RISC-V backend | `stdlib_plugin` target field |
|---------|----------------|------------------------------|
| `1.0.0b3.dev2026072114` | no | yes |
| `1.0.0b3.dev2026061206` | yes | no |
| `1.0.0b2.dev2026061203` | yes | no |

No released nightly has both. RISC-V was dropped partway through the b3
series; `std/_plugin` — the sanctioned way to register a backend with the
stdlib, with `cuda`/`hip`/`metal` reference implementations — landed after.

That matters because the plugin path is where this should eventually live.
`scratchpad()`'s signature already matches the `stack_allocation_fn` hook,
so the body can move into an `std/_plugin/m2ndp/` overlay once a toolchain
ships both. Until then the library open-codes the MLIR operations directly.

Worth finding out whether the RISC-V removal was deliberate or a regression;
if it comes back in a plugin-capable build, the long-term path opens up.

---

## 4. Decisions worth not relitigating

Recorded because each was reached by measuring something that contradicted
an earlier assumption.

**No barrier.** µthreads are created and retired by hardware FGMT, so there
is no well-defined set to synchronize. An earlier version had a
`group_barrier()` transliterated from `__syncthreads()`. Atomics combine
within a kernel; kernel boundaries order phases, since `device_main`
launches synchronously.

**IDs are primitives, not derived.** `global_uthread_id()` was computed as
`group_id() * group_size() + uthread_id()`, a CUDA transliteration. The
hardware hands a µthread its identity in scalar registers at spawn.

**Kernels take ordinary parameters.** Surfacing the calling convention
(`kernel_arg(0)`, raw byte offsets) would bake it into every benchmark and
break the rule that `benchmarks/` survives the backend switchover unchanged.

**Not `stack_allocation`.** `std.memory.stack_allocation[N, T,
address_space=SHARED]()` looks like the obvious spelling for the scratchpad
and silently produces wrong code here: its promotion to an addrspace(3)
global is gated on `is_gpu()`, so on a RISC-V triple it becomes a plain
alloca on the stack — private per µthread, so nothing is shared. It compiles
clean. The library open-codes `pop.global_alloc` instead.

**Scratchpad declared at struct level.** Calling `scratchpad()` separately
in each function mints a fresh symbol per call site (`@buf`, `@buf_0`, …), so
phases that should share memory silently do not. A comptime struct member is
evaluated once and shared.

---

## 5. Open work

**Benchmarks** — 6 of ~23 ported. Next, roughly in order of what new ground
they cover:

- `gemv` — FP16, stages the input vector in the scratchpad; a pattern none
  of the current six exercises
- `softmax`, `layerNorm` — multi-kernel reductions, and the first use of
  transcendentals. The Mojo binary ships the full Sleef RVV math library
  (`Sleef_expdx_u10rvvm2` and friends, LMUL=2), but whether `exp`/`sqrt`
  actually route there on this target is **unverified**
- `sssp` — Arachne's own device-side control-flow example; would exercise
  `device_main` rather than a kernel body
- `pagerank`, `dlrm`, `kmeans` — further memory-bound coverage

**Unverified**

- Nothing has ever been *executed*. No simulator run, no correctness test.
  This PoC verifies the shape of the IR, not behaviour
- Sleef RVV transcendental linkage (above)
- µthread launch — out of scope here by design; belongs to the M²NDP runtime
  or Arachne

**Library**

- `atomic_add_lanes` is per-lane, not vector, atomicity. Documented, but it
  is a name that invites misreading
- No `min`/`max`/`cas` atomics yet; only `add`
