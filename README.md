# mojo-m2ndp

A PoC for writing M²NDP (RISC-V Vector based µthread GPNDP) workloads in
Mojo, compiling them for a RISC-V/RVV target, and running them.

A workload is one file: its kernels, the `device_main` that launches them,
and the host code that feeds it and checks the answer -- single source, the
way a `.cu` is. Naming the task compiles it for the target the task declares
and runs it under Spike:

```mojo
_ = Histogram.launch(pool, PooledRange.over(samples, n),
                     HistogramParams(samples, hist))
```

**It started without a backend**, and the seam is still there: M²NDP
operations are expressed as external symbols, so compiler work and
workload work can proceed in parallel, and the symbol set is the contract
between them. There is now a backend behind that seam --
[an LLVM fork](https://github.com/PSAL-POSTECH/llvm-project-m2ndp) with an
`XM2ndp` vendor extension -- so the symbols lower to real instructions and a
kernel gets the calling convention the hardware wants. What has not changed
is that `benchmarks/` never had to know.

## Quick start

```bash
git clone <this-repo> && cd mojo-m2ndp
./scripts/setup.sh          # install the Mojo toolchain (./toolchain)
./scripts/build.sh          # each benchmark's device IR -> out/*.ll (+ .s with our llc)
./scripts/verify.sh         # check the artifacts
```

To actually run one, under Spike, with the workload's own host code:

```bash
./scripts/build-llvm.sh     # our LLVM, with the vendor extension
./scripts/build-spike.sh    # the simulator and the extension library
./scripts/host-run.sh       # every benchmark, checked against its own answer
```

Or take the development image, which carries both toolchains already built,
and skip the setup entirely:

```bash
docker run --rm -it ghcr.io/psal-postech/mojo-m2ndp:main
./scripts/build.sh && ./scripts/verify.sh
./scripts/host-run.sh       # run what came out — see docs/SIMULATION.md
```

Do not mount over `/work`: the toolchains live there and a mount hides them.
The LLVM source is not in the image either — 2.6 GB that nothing above reads,
since the compiler is already built. Rebuilding LLVM or running its lit suite
needs a checkout.

If Mojo is already installed, skip setup and point at it:

```bash
export MOJO_ROOT=/path/to/modular    # the directory holding bin/ and lib/
./scripts/build.sh
```

### Toolchain requirement

The toolchain must have a **RISC-V backend registered**. Not every Mojo
nightly does — a build without it fails with:

```
error: no compiler backend is registered for target 'riscv64-unknown-unknown-elf';
this target is not supported by this build
```

Pin a known-good build with `MOJO_VERSION=<version> ./scripts/setup.sh`;
the repo targets `1.0.0b2.dev2026061203`. Which nightlies work, and why no
current one is ideal, is in [`docs/STATUS.md`](docs/STATUS.md).

## What comes out

The benchmarks are ports of
[M2NDP-public](https://github.com/PSAL-POSTECH/M2NDP-public)'s hand-written
kernels, so each one can be read against the assembly it came from. Full
annotated walkthrough in [`docs/EXAMPLES.md`](docs/EXAMPLES.md).

### vector_add — RVV vectorization, and the kernel ABI

`out/vector_add.s`, the whole kernel:

```asm
ld        a1, 0(a0)                   # the task's parameters, from the
ld        a2, 8(a0)                   #   scratchpad -- a0 being the base
ld        a0, 16(a0)                  #   the hardware gave
sext.w    a5, a5                      # a5 IS global_uthread_id -- no call
slli      a5, a5, 5
vsetivli  zero, 8, e32, m2, ta, ma    # RVV
vle32.v   v8, (a1)
vadd.vv   v8, v8, v10
vse32.v   v8, (a0)
ret                                   # no frame: a kernel preserves nothing
```

Two halves are visible here. RVV comes from the Mojo compiler, reached purely
through a hand-written target attribute. Everything else is the extension: a
kernel that takes no arguments and reads the task's parameters out of the
scratchpad, the identity values as live-in registers rather than calls, and no
frame because nothing resumes after a kernel.

### spmv — indirect access, atomic combine

One group per row; its µthreads take a strided slice of the nonzeros.
Indirect access (`x[col_idx[k]]`) is just a dependent load chain — no
special construct needed:

```llvm
%28 = load i32, ptr %27, align 4                     ; col_idx[k]
%29 = sext i32 %28 to i64
%30 = getelementptr inbounds float, ptr %2, i64 %29  ; &x[col_idx[k]]
%32 = load float, ptr %30, align 4                   ; x[col_idx[k]]
%33 = fmul contract float %31, %32
%34 = fadd contract float %20, %33                   ; -> fmadd.s in asm
```

The partial sums are combined with an atomic, not a barrier:

```llvm
%40 = atomicrmw fadd ptr %39, float %38 monotonic, align 4
```

**M²NDP has no barrier.** µthreads are created and retired by hardware FGMT,
so there is no well-defined set to synchronize; atomics combine within a
kernel and kernel boundaries synchronize between them. RISC-V has no
floating-point AMO at all, so that `fadd` was a compare-exchange loop until
the extension gave it `famoadd.w`.

`__m2ndp_group_size` used to be re-called on every loop iteration -- an
opaque external call cannot be proven loop-invariant -- which was the real
cost of the external-symbol approach. It is a live-in register now, so the
loop reads it once. The three calls left in `out/spmv.s` are all
controller-side -- `device_main`, the launch inside it, and the range the
runtime sets first -- and none is in a kernel, where a call is rejected
outright.

### histogram — scratchpad across three kernels, and an indexed vector atomic

Three kernels share a per-core bin array. Declaring the scratchpad at struct
level is what keeps them on the same storage:

```mojo
struct Histogram(NDPTask):
    comptime bins = scratchpad[BINS, Int32, name="hist_bins"]()
```
```llvm
@memory_blob_de5f15ab6daf7941 = internal addrspace(3) global [1024 x i8] zeroinitializer

; init:  store i32 0, ptr addrspace(3) %9
; body:  %9  = atomicrmw add ptr addrspace(3) %8, i32 1 monotonic
; final: %13 = atomicrmw add ptr %10, i32 %12 monotonic
```

One global, all three kernels indexing off it. Calling `scratchpad()`
separately in each would mint a fresh symbol per call site and they would
silently use different memory.

The body tallies sixteen samples with one instruction, which is the point of
the extension's headline addition -- an *indexed* vector atomic, where every
lane has its own address:

```asm
m2ndp.vamoaddei32.v  v12, (a0), v8, v12
```

Neither layer could express that before. LLVM's vector `atomicrmw` is
contiguous, and RVV's indexed AMOs were dropped before 1.0, so there was
nothing standard to lower to. Sixteen scalar atomics were the alternative.

## Layout

```
src/m2ndp.mojo        the model: kernels' primitives, NDPTask, launching a task
src/m2ndp_host.mojo   host-side machinery a launch runs on (files, processes, tools)
benchmarks/           ports of M2NDP-public/examples/benchmarks -- each one
                      holds its kernels, its device_main and its host main
  memcpy.mojo         vector load + store, nothing else
  memset.mojo         scalar splat to a vector store
  vector_add.mojo     confirms RVV vectorization
  imdb_lt_int64.mojo  predicate scan -> bitmap (vmslt.vx)
  spmv.mojo           CSR SpMV — indirect access + atomic combine
  histogram.mojo      scratchpad shared across INIT/BODY/FINAL phases
config/machine.conf   the NDP hardware a run is modelled on
sim/                  the device-side launcher, and the Spike extension
scripts/
  setup.sh            install the Mojo toolchain
  env.sh              environment variables (source it)
  build.sh            each benchmark's device IR + assembly -> out/
  verify.sh           check the artifacts
  build-llvm.sh       our LLVM (vendor extension) + lld
  build-spike.sh      the simulator and sim/ext/ as a loadable extension
  spike-smoke.sh      does the pipeline, and the extension, stand up
  host-run.sh         run a workload and check its own answer
docs/SIMULATION.md    running compiled workloads, and what that does not catch
docs/INTERFACE.md     the backend contract in detail
docs/EXAMPLES.md      annotated source -> LLVM IR -> assembly walkthrough
docs/STATUS.md        what works, what the backend must supply, open work
third_party/          the LLVM fork and Spike, as submodules
out/                  generated artifacts (not tracked by git)
```

## Writing a benchmark

One file, three parts: what the host passes, the task, and the host code.

```mojo
@fieldwise_init
struct HistogramParams(Movable):
    var samples: UnsafePointer[Int32, MutAnyOrigin]   # plain pointers:
    var out_hist: UnsafePointer[Int32, MutAnyOrigin]  # the pool is shared

struct Histogram(NDPTask):
    comptime Params = HistogramParams             # what the host fills in
    comptime packet = UNROLL * size_of[Int32]()   # bytes one µthread takes
    # Declared once at struct level so every kernel shares one allocation.
    comptime bins = scratchpad[BINS, Int32, name="hist_bins"]()

    # A kernel takes no arguments: the task's parameters are in the
    # scratchpad, so it reads the fields it wants by name.
    @staticmethod
    def body():
        ...Histogram.params()[].samples...

    @staticmethod
    def device_main(params: UnsafePointer[HistogramParams, MutAnyOrigin]):
        launch_serial[Histogram.initialize]()
        launch_parallel[Histogram.body]()
        launch_serial[Histogram.finalize]()

def main() raises:
    if Histogram.emit_ir_if_asked():
        return
    var pool = Pool()                       # the memory the device shares
    var samples = pool.alloc[Int32](n)      # host writes straight into it
    var hist = pool.alloc[Int32](BINS)
    ...fill samples...
    _ = Histogram.launch(pool, PooledRange.over(samples, n),
                         HistogramParams(samples, hist))
    ...hist already holds the result; check it here...
```

Kernels take ordinary parameters and index them; recovering the hardware's
mapped-address form is the compiler's job. Nothing is exported: conforming to
`NDPTask` gives the task the one entry point the host launches it through,
and naming a kernel from `device_main` is what keeps it alive -- and what
tells the backend it is a kernel at all.

`device_main` decides the order. The kernel is a parameter of the launch
rather than an argument to it, which is what keeps the launch symbols inside
the library -- `external_call` takes a function only where it is named at the
call site, and a parameter is where it keeps that name.

What a benchmark never says is what hardware it runs on. That is
`config/machine.conf`, which the runtime reads; pointing
`M2NDP_MACHINE_CONFIG` at another description is how the same program is
shown to give the same answer on a different machine.

## The pool is shared, not copied

M²NDP is near-data processing: the data is already in the CXL pool and the
NDP cores are attached to that memory. Host and device see the same bytes.
There is no `cudaMemcpy` here because there is nothing to copy — which is why
a parameter is a plain pointer and no field says "in" or "out".

Two processes do not get that for free, so the pool is a file both sides map
at the same address:

```
host                                    spike
  mmap(MAP_FIXED, pool_base) ──┐   ┌── --device=m2ndp_pool,<file>,<base>,<size>
                               └───┴──  the same pages
```

`pool.alloc[Int32](n)` returns an address that is a device address too, so a
kernel loads exactly what the host stored. Nothing is uploaded before a run
and nothing is downloaded after it; results are in the caller's pool because
they were written there.

The device side is 40 lines in `sim/ext/`, registered through the same
`--extlib` the instruction extension already uses, so the Spike submodule
stays untouched. It is an `abstract_mem_t` rather than a plain MMIO device on
purpose: `sim_t::addr_to_mem` only takes the MMU's fast path for a memory, by
asking it for `contents()`. A device would cost a virtual call per access.

`pool_base` and `pool_bytes` are in `config/machine.conf` with the rest of the
machine — how much memory the device has is the hardware's business, not a
workload's.

## The symbol contract

The seam between workload and compiler is a set of names. Details in
[`docs/INTERFACE.md`](docs/INTERFACE.md).

**What a µthread is handed** — lowered to live-in registers, not calls:

| Symbol | Signature | Meaning |
|--------|-----------|---------|
| `__m2ndp_local_uthread_id` | `i32 ()` | index within the group; also the scratchpad slot |
| `__m2ndp_global_uthread_id` | `i32 ()` | index across all cores; identifies the mapped data |
| `__m2ndp_group_size` | `i32 ()` | µthreads sharing one scratchpad |
| `__m2ndp_group_id` | `i32 ()` | which group this µthread belongs to |
| `__m2ndp_task_params` | `ptr ()` | the task's parameter block, in this core's scratchpad |

**Launching** — implemented by the runtime, and load-bearing besides: the
backend decides a function is a kernel by seeing its address reach one of
these, so a kernel that is never launched is not one.

| Symbol | Meaning |
|--------|---------|
| `__m2ndp_rt_launch_task` | a task's entry point; the host calls it |
| `__m2ndp_set_task_range` | the range a task is mapped over, hence the µthread count |
| `__m2ndp_launch_parallel` | one µthread per packet of the range |
| `__m2ndp_launch_serial` | one µthread per core |

**Operations with no spelling at this level** — `__m2ndp_vamoadd_*` for the
indexed vector atomic, since neither `atomicrmw` nor RVV 1.0 can express it.
Mask-to-bitmap still has none and is open.

| Address space | Use |
|---------------|-----|
| `addrspace(3)` | scratchpad (group-shared memory) |

The extension supplies all of the above except mask-to-bitmap. Where a symbol
becomes an intrinsic, **only the function bodies in `src/m2ndp.mojo` change** —
benchmark code does not. See [`docs/STATUS.md`](docs/STATUS.md) for what is
left.

## How it works

Three things make this work.

**1. The compile target can be constructed by hand.**
`std.gpu`'s hardware detection (`is_nvidia_gpu()` and friends) is locked
inside the closed-source `std.sys.info`, so new hardware cannot be
registered in that scheme. But the target parameter the kernel-compilation
entry point takes is a `!kgen.target` MLIR attribute, and that can be
written directly — bypassing the detection scheme entirely.

The same trick covers stdlib routines that branch on the vendor check: where
a routine takes a different path for GPUs, the library open-codes the MLIR
operation that path emits. `scratchpad()` does exactly this with
`pop.global_alloc`.

(Newer stdlib releases add `std/_plugin`, a supported hook interface for
registering a backend, selected by a `stdlib_plugin` field on the target
attribute. No nightly yet ships it *and* a RISC-V backend — see the
toolchain table above — but it is where this should eventually move, and the
library's `scratchpad()` signature already matches the corresponding hook.)

```mojo
def m2ndp_target() -> __mlir_type.`!kgen.target`:
    return __mlir_attr[
        `#kgen.target<triple = "riscv64-unknown-elf", `,
        `arch = "generic-rv64", `,
        `features = "+m,+a,+f,+d,+v,+zvl128b,+xm2ndp", `,
        ...
    ]
```

`+v,+zvl128b` turns RVV on. `+xm2ndp` is our extension: the frontend's own
LLVM does not know it and says so on every build, but it copies the string
into the `target-features` attribute verbatim, so the marker survives into
the IR and our llc picks it up. A task can override the whole target, which
is the whole of what lowering the same workload elsewhere takes.

**2. M²NDP operations are expressed as external symbols.**
Mojo's `llvm_intrinsic[...]` only accepts intrinsics LLVM already knows;
unknown names are rejected during translation. So `external_call` instead.
It survives into the LLVM IR as `declare` + `call`, which makes the mapping
points explicit — and lets the workload half be written before the compiler
half exists, which is how this was built.

**3. The task compiles itself, for its own target.**
`compile_info` takes that same `!kgen.target`, so a host program asks a task
for its device code rather than a build script compiling the module:

```mojo
var ir = Histogram.device_ir()      # compile_info, for Histogram.target
```

Only what the entry point reaches comes back, which is why a single file can
hold both halves: the host `main` is not device code and is never compiled as
any. That is nvcc's two-pass model, arrived at from the other direction.

## Requirements

`python3` + `pip` for the Mojo toolchain, and that is all `build.sh` and
`verify.sh` need. Running a workload also wants the LLVM fork and Spike,
which `build-llvm.sh` and `build-spike.sh` build from the submodules; the
development image carries both already. `riscv64-unknown-elf-gcc` compiles
the device-side launcher.

Verified on Linux x86-64.
