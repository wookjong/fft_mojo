# Status and open work

Where the PoC stands and what is left to do. Everything below was measured
on the artifacts in `out/`; nothing is projected. Regenerate with
`./scripts/build.sh && ./scripts/verify.sh`.

The backend-facing contract is in [`INTERFACE.md`](INTERFACE.md); this file
does not repeat it.

Last updated: 2026-07-24, against Mojo `1.0.0b2.dev2026061203`.

---

## 1. What works

| | |
|---|---|
| RISC-V/RVV codegen | `target triple = "riscv64-unknown-unknown-elf"`, `vsetvli` / `vle32.v` / `vadd.vv` / `vse32.v` selected from ordinary Mojo `SIMD` |
| Custom target | `#kgen.target<...>` written by hand, bypassing `std.gpu`'s closed vendor detection |
| Scratchpad | `internal addrspace(3) global` — an untyped byte blob named `memory_blob_<hash>`, not the `name=` argument — shared across kernels when declared as a comptime struct member |
| Atomics | `atomicrmw add` / `fadd`, on ordinary memory and on `addrspace(3)`, at `monotonic` ordering; `fadd` selects a real instruction with `+xm2ndp` rather than a cmpxchg loop |
| Indirect access | plain load → sext → GEP → load; no special construct needed |
| Predicate scan | `v.lt(x)` selects `vmslt.vx` |
| Multi-kernel modules | several kernels per file, held by one task struct; none exported — naming a kernel from `device_main` is what keeps it alive |
| Tasks | a struct conforming to `NDPTask` carries its kernels, its `device_main`, and the target, machine and packet size it runs with |
| Launching from the host | `Histogram.launch(PooledRange.over(xs), HistogramParams(xs, ys))` — compiled for the task's target, run under Spike, results downloaded into the caller's lists |

Benchmarks ported from
[M2NDP-public](https://github.com/PSAL-POSTECH/M2NDP-public) — **24 workloads,
covering 20 of its 23 directories**. Every one is a task and every one runs:
`./scripts/host-run.sh` compiles each for its target, runs it under Spike and
checks the answer against one the host computes itself.

| Benchmark | Exercises |
|---|---|
| `memcpy` | vector load + store |
| `memset` | scalar splat to vector store; a scalar kernel argument, and a range that is an output buffer |
| `vector_add` | RVV vectorization |
| `residual` | the same shape over fp32 — a transformer skip connection |
| `relu` | mask and merge (`vmfge.vf`, `vmerge.vvm`) |
| `vector_exp` | a transcendental with no vendor instruction behind it |
| `gelu` | that exponent inside a longer float expression |
| `narrow` / `wide` | fp16 conversion, one instruction each way |
| `imdb_lt_int64` | predicate scan → bitmap |
| `imdb_gteq_lt_int64` | two bounds, and'd |
| `imdb_gt_lt_fp32` | the same over floats, a bitmap byte per microthread |
| `imdb_two_col_and` / `imdb_three_col_and` | combining scan results |
| `kmeans_assign` | reduce to a minimum, then find its lane |
| `gemv_aggregation` | float vector atomic (`vfamoaddei32.v`) |
| `gemv` | fp16 weights, fp32 accumulation, atomic combine |
| `spmv` | indirect access; one row to a µthread |
| `dlrm_sls` | a data-dependent loop: gather a variable-length list of rows |
| `pagerank_inicsr` | gather/scatter over CSR |
| `sssp` | one Bellman-Ford pass |
| `histogram` | scratchpad shared across INIT/BODY/FINAL phases |
| `softmax` | three kernels in order; scalar float atomics (`famomax.w`, `famoadd.w`) |
| `layernorm` | both moments in one pass, then a rescale kernel |

Checked at 1, 4 and 8 cores and several interleavings; the answers agree,
which is what tests the per-core scratchpad claim. `.github/workflows/test.yml`
runs the whole set on every push and pull request, at two configurations.

Three of the upstream directories are not ported. `naive_bayes`'s kernel body
is a single vector load — the workload is unfinished upstream, so there is
nothing to port. `opt/fc` and `opt/attention` are 600 and 950 lines of
generated assembly apiece, and the rest of `opt` is covered: `activation` is
`relu`, `residual` is `residual`, `layernom` is `layernorm`.

Every benchmark takes its position from its own index in the range, which is
what the reference gives its kernels as well. How that index maps onto a core
is M2NDP-public's rule; `test/interleave.cases` is the table that pins it.

### LLVM baseline

`./scripts/build-llvm.sh check`, against the pinned submodule (LLVM 23.1.0,
`release/23.x`, RISC-V only, assertions on): **3210/3210 RISC-V lit tests
pass**, CodeGen and MC together. The 3210th is `xm2ndp-device-main.ll`, which
pins the split between controller-side code and kernels.

The CodeGen baseline before `FeatureVendorXM2ndp` was 2595/2595. Adding the
feature broke exactly one test — `features-info.ll`, which checks the full
`-mattr=help` listing — and the baseline is what made that immediately
attributable rather than a mystery.

Every failure seen at any point since has been a tool the suite needs and
the build did not produce, never a codegen difference: 31 the first time
(`llvm-objdump`, `llvm-readobj`, `llvm-readelf`, `llvm-dwarfdump`) and 7
when MC was added (`yaml2obj`, `llvm-otool`, `split-file`, `llvm-nm`).
`build-llvm.sh check` builds all of them now.

The artifacts also round-trip through that build: `llc` accepts every
`out/*.ll` and `llvm-mc -filetype=obj` assembles what it produces. So the
frontend and backend LLVM versions are compatible in practice, not only by
version number.

---

## 2. What the backend has to support

The contract lives in [`INTERFACE.md`](INTERFACE.md) — symbols, scratchpad
placement and addressing, synchronization, the two operations that have no
spelling at this level, and recovering the mapped address. In rough order of
how much is blocked on each:

1. **The four ID symbols — done.** They lower to reads of live-in registers,
   not to calls. Every kernel that used one lost its stack frame with the
   call; the loop-invariance problem went with it. The register assignment
   is provisional and lives in `RISCVM2ndpArgInfo.h`
2. **Scratchpad — done.** Globals are laid out by the compiler into one
   block in `.spad`, and each becomes a constant offset from the base
   pointer the hardware supplies. An access is a single instruction with no
   address materialization
3. **Vector atomic — done.** Neither layer could express an *indexed* one:
   LLVM's vector `atomicrmw` is contiguous, and RVV's indexed AMOs were
   dropped before 1.0, so there was nothing standard to lower to either.
   Now there is `llvm.riscv.m2ndp.*` and 52 instructions behind it, reached
   from Mojo through an external symbol. `histogram`'s body is one
   instruction where it was sixteen. See INTERFACE.md
4. **Mask-to-bitmap** — still needs its own intrinsic; untouched
5. **FP atomic add — done.** RISC-V has no floating-point AMO at all, so an
   `atomicrmw fadd` is a cmpxchg loop; `famoadd`/`famomin`/`famomax` at
   `.h`/`.w`/`.d` replace it with one instruction. No benchmark exercises it
   -- none has µthreads sharing a float — so its coverage is the lit suite's
6. **Recovering `ADDR`/`OFFSET`** from `base[id * W]`

Kernel arguments now come from the scratchpad rather than from registers,
and every kernel is call-free and frame-free: across the six benchmarks,
zero calls and zero stack frames. There are no callee-saved registers
either — nothing resumes after a kernel — so the whole register file is
free, and a frame appearing at all now warns, since it can only mean a
spill to DRAM. Calls are rejected outright, including the ones the compiler
emits on its own. The benchmarks no longer define `main` --
it was Mojo scaffolding for building an executable, and dropping it took the
`KGEN_CompilerRT_*` runtime calls out of the device modules with it.

The architecture questions that used to block this are settled and written
up in INTERFACE.md: the scratchpad base is the same on every core, one
launch group is resident on a core at a time, and the contents survive
kernel launches within a task. Together those mean the scratchpad keeps a
fixed address and needs one offset per global — AMDGPU's LDS model.

No question blocks the remaining work. Which registers carry which values
is still unsettled, but it stopped being a blocker once the assignment was
confined to one table: a provisional choice can be measured now and
corrected in one place later, the same bargain taken for the AMO
encodings.

## 3. Where the frontend sets the shape

What the launch interface can and cannot say, tried against Mojo
`1.0.0b2.dev2026061203`. Recorded so the same ground is not covered twice, and
so a toolchain that moves one of these is recognised when it arrives.

**Field reflection does not exist.** `__fields__`, `__field_names__`,
`fields_of[T]()` and `__type_of(T).__fields__` were all tried; none does. So a
parameter block cannot be walked with its field types in hand, and `launch`
reads it as a run of uniform records instead -- the field count being the
block's size divided by one field's.

That is also why the host end of a launch stays unchecked. `NDPTask` carries
the block as an associated type (`comptime Params: Movable`), so the
trait-declared `device_main` is typed against it and the kernels read it
through `Self.params()`: from `device_main` inward the parameters are one
declaration and the compiler checks the names and the types. What is left is
the caller's own ordering of `PooledRange` against the buffers it names.

`size_of[T.Params]()` is a compile-time value, and since the block is all
addresses that gives the field count. An arity check against it is possible
and is not implemented: it catches the least dangerous mistake and would read
as a guarantee it is not.

**A function reaches `external_call` only as a parameter.** As a runtime
argument it does not convert: a declared function's type carries its name, so
`def body() -> None` is not `def() -> None`, and no spelling of the type
bridges them. Bound to a compile-time parameter whose type is inferred it
stays the function it is, and `materialize` hands it on -- which is how
`launch_serial` and `launch_parallel` wrap the launch symbols.

Taking the kernel's *address* instead compiles and is worse than not wrapping.
`UnsafePointer(to=k)` is the address of a slot holding the function rather
than the function, so the launcher is handed one indirection too many; and the
IR stores the kernel address into that slot, which leaves the function's only
use a `store` rather than a launch. `isM2ndpKernel` then cannot see it
launched, and the kernel compiles with the ordinary ABI, silently. The store
does not optimise away either, since the slot's address escapes into the call.

**A variadic's length is not a compile-time value.** `comptime n =
len(args)` is rejected as a dynamic value, so anything derived from how many
arguments a call was given has to be checked at run time.

## 4. Toolchain constraint

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

## 5. Decisions worth not relitigating

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

## 6. Open work

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
