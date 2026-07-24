# CLAUDE.md

Operating notes for Claude Code when working in this repository.

## What this project is

A PoC for writing **M²NDP** workloads in Mojo and compiling them for a
RISC-V/RVV target to get LLVM IR and assembly.

M²NDP = a CXL memory-expander based **GPNDP** (General-Purpose NDP). It is
not a fixed-function accelerator but a **RISC-V Vector (RVV) based µthread
architecture**, using an SPMD µthread launch programming model.

The GPU analogy is useful but only up to a point, and the early version of
this library leaned on it too hard. M²NDP identifies a µthread by the
address it is mapped to, not by a `(block, thread)` pair, and its scratchpad
is scoped to an **NDP core** rather than a threadblock (Table 1 of the
M²NDP paper). Accordingly the library exposes Arachne's two IDs —
`GlobalUThreadID()` / `LocalUThreadID()` — and derives anything else.

**There is no M²NDP compiler backend yet.** So M²NDP-specific operations
are expressed as external symbols, letting backend work and
workload/library work proceed in parallel. The symbol set fixed in this
repo is the interface contract between the two.

## Repository layout

```
src/m2ndp.mojo        the model: kernels' primitives, NDPTask, launching a task
src/m2ndp_host.mojo   host-side machinery a launch runs on (files, processes, tools)
benchmarks/           ports of M2NDP-public/examples/benchmarks
  memcpy.mojo         vector load + store, nothing else
  memset.mojo         scalar splat to a vector store
  vector_add.mojo     confirms RVV vectorization
  imdb_lt_int64.mojo  predicate scan -> bitmap (vmslt.vx)
  spmv.mojo           CSR SpMV — indirect access + atomic combine
  histogram.mojo      scratchpad shared across INIT/BODY/FINAL phases
config/machine.conf   the NDP hardware a run is modelled on
sim/                  the device-side launcher, and the Spike extension
scripts/
  setup.sh            install the Mojo toolchain (merge 3 nightly wheels -> ./toolchain)
  env.sh              environment variables (source it)
  build.sh            each benchmark's device IR + assembly -> out/*.ll, out/*.s
  verify.sh           automated artifact checks
  build-llvm.sh       our LLVM (vendor extension) + lld
  build-spike.sh      the simulator and sim/ext/ as a loadable extension
  spike-smoke.sh      does the pipeline, and the extension, stand up
  host-run.sh         run a workload from its host program
docs/SIMULATION.md    running compiled workloads, and what that does not catch
docs/INTERFACE.md     the backend contract in detail
docs/EXAMPLES.md      annotated source -> LLVM IR -> assembly walkthrough
docs/STATUS.md        what works, what the backend must supply, open work
out/                  generated artifacts (not tracked by git)
toolchain/            installed Mojo (not tracked by git)
```

## Commands

```bash
./scripts/setup.sh              # install Mojo (once)
./scripts/build.sh              # every benchmark, llvm + asm
./scripts/build.sh spmv         # one benchmark only
EMISSION=asm ./scripts/build.sh # one emission only (llvm|asm)
./scripts/verify.sh             # check the artifacts

./scripts/build-llvm.sh         # our LLVM; `check` also runs the RISC-V lit suite
./scripts/build-spike.sh        # simulator + sim/ext/ extension library
./scripts/spike-smoke.sh        # pipeline and extension stand up
./scripts/host-run.sh           # run both workloads from their host programs
./scripts/host-run.sh histogram 4 8   # one, at a given cores/interleave
```

If Mojo is already available, skip setup and just point at it:

```bash
export MOJO_ROOT=/path/to/modular   # the directory holding bin/ and lib/
```

**After any change, validate with `./scripts/build.sh &&
./scripts/verify.sh` before moving on.** Do not commit with verify failing.

## Toolchain constraint (important)

The toolchain **must have a RISC-V backend registered**. Not every Mojo
nightly does. A build without it fails with:

```
error: no compiler backend is registered for target 'riscv64-unknown-unknown-elf';
this target is not supported by this build
```

Which nightlies work is tabulated in `docs/STATUS.md`; the repo targets
`1.0.0b2.dev2026061203`.

Do **not** test for this by grepping the binary for `riscv`: builds that
reject the target still contain plenty of such strings, and `strings` is not
always installed (an earlier version of this file claimed b3 had "zero riscv
references" on exactly that mistake). `setup.sh` compiles a probe kernel
instead. Because it defaults to the *latest* nightly, a fresh clone can end
up unable to build.

## How the vendor lock-in is bypassed

README's "How it works" covers this; the operationally relevant parts:

- `compile_info`'s `target` parameter is a `!kgen.target` MLIR attribute that
  can be written by hand, so `std.sys.info`'s closed vendor detection is
  simply not consulted. `m2ndp_target()` in `src/m2ndp.mojo` is that
  attribute, and `NDPTask.target` defaults to it. The format was derived from
  stdlib `std/gpu/host/info.mojo`'s `_get_a100_target()`.
- The same move applies to stdlib routines that branch on `is_gpu()`: where
  one takes a different path for GPUs, open-code the MLIR operation that
  path emits. `scratchpad()` does this with `pop.global_alloc`.
- `llvm_intrinsic[...]` only accepts intrinsics LLVM already knows —
  `could not find LLVM intrinsic: "llvm.m2ndp.uthread.id"` — so hypothetical
  ones go through `external_call` (from `std.ffi`, not `std.sys.ffi`).

## Interface contract

The whole benchmark set needs **4 symbols and 1 address space**. Details in
`docs/INTERFACE.md`; annotated codegen in `docs/EXAMPLES.md`; status and
open work in `docs/STATUS.md`.

| Symbol | Signature |
|--------|-----------|
| `__m2ndp_local_uthread_id` | `i32 ()` |
| `__m2ndp_global_uthread_id` | `i32 ()` |
| `__m2ndp_group_size` | `i32 ()` |
| `__m2ndp_group_id` | `i32 ()` |

The two IDs follow Arachne's `GlobalUThreadID()` / `LocalUThreadID()`.
"Group" = the µthreads sharing one scratchpad (those on one NDP core);
`group_id()` is which one. There is no barrier and no core ID.

`addrspace(3)` = the scratchpad, one instance per NDP core, obtained with
`scratchpad[count, T, name=...]()`.

**The switchover point once the backend lands is the single file
`src/m2ndp.mojo`.** Replace the `external_call` in each function body with
a real intrinsic.

## Known limitations / candidate next tasks

`docs/STATUS.md` is the maintained list: what works, what the backend has to
supply, and what is left to do. Keep it current rather than duplicating it
here. The invariants that constrain how this repo is changed:

1. **`benchmarks/` must survive the backend switchover unchanged.** Isolate
   everything M²NDP-specific in `src/m2ndp.mojo`. In particular the calling
   convention — args arriving through the scratchpad, a µthread receiving a
   mapped address rather than an index — stays out of benchmark code.
2. **µthread launch is out of scope.** This PoC covers kernel bodies only.
   Launch belongs to the M²NDP runtime or Arachne. Do not add a launch API.
3. **Do not reintroduce a barrier.** µthreads are created and retired by
   hardware FGMT, so there is no set to synchronize. An earlier version had
   one, transliterated from `__syncthreads()`.
4. **Do not "fix" the `pop.global_alloc` open-coding** by switching to
   `std.memory.stack_allocation[..., address_space=SHARED]()`. Its promotion
   is gated on `is_gpu()`, so on a RISC-V triple it silently becomes a
   per-µthread stack slot: it compiles clean and computes the wrong answer.
5. **Declare a scratchpad once, as a comptime struct member,** if more than
   one kernel touches it. Per-function calls mint a fresh symbol each.

The reasoning behind 3-5, and the measurements that produced it, is in
`docs/STATUS.md` §4.

## Working notes

- **Code in `benchmarks/` must stay unchanged across the backend
  switchover.** Isolate everything M²NDP-specific in `src/m2ndp.mojo`. In
  particular the M²NDP calling convention — args arriving through the
  scratchpad, a µthread receiving its mapped address rather than an index —
  stays out of benchmark code; kernels take ordinary parameters and index
  them, and recovering the hardware form is the compiler's job.
- Mojo 1.0 syntax: `fn` is gone — use `def`. The parameter name `out` is
  reserved (use `res` etc.).
- Two ways a workload gets compiled, and they are not interchangeable.
  `build.sh` builds the whole module with target flags, for `out/*.ll` and
  the artifact checks. `NDPTask.launch` compiles the task itself through
  `compile_info` at launch time, for the target the task declares. Only the
  second is how a workload actually runs.
- `compile_info` must be called at run time; folding it at comptime fails
  inside the stdlib with nothing pointing at the cause. And it must emit IR,
  not assembly: Mojo's own LLVM does not know the vendor extension, so its
  assembly is unfinished.
- `build.sh` and `host-run.sh` both copy `src/` and the workload into a temp
  directory before compiling (because of Mojo's module search path). They
  assume the workload imports `m2ndp`.
- `build.sh` writes to a temp file and only replaces `out/*` on success, so
  a failed build never destroys artifacts from a previous good run.
- All comments, documentation and commit messages in this repo are in
  English.
- Commit author: `YWHyuk <wonhyuk@postech.ac.kr>`.

## Background (when deeper context is needed)

This PoC came out of prior work reverse-engineering the Mojo compiler's
internals to analyze its lowering pipeline (`lit` → `kgen` → `pop` → LLVM).
That work established that `std.gpu` selects vendor-specific LLVM intrinsic
names at compile time through `comptime if is_nvidia_gpu() / elif
is_amd_gpu() / ...` branches, and that those detection functions are
closed-source. This repo's approach — a separate library plus a
hand-constructed target attribute — is the way around that constraint.
