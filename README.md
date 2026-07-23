# mojo-m2ndp

A PoC for writing M²NDP (RISC-V Vector based µthread GPNDP) workloads in
Mojo and compiling them for a RISC-V/RVV target to get LLVM IR and assembly.

**It works without a backend.** M²NDP-specific operations are expressed as
external symbols, so compiler-backend work and workload/library work can
proceed in parallel. The symbol set fixed here is the interface contract
between the two.

## Quick start

```bash
git clone <this-repo> && cd mojo-m2ndp
./scripts/setup.sh          # install the Mojo toolchain (./toolchain)
./scripts/build.sh          # benchmarks -> out/*.ll, out/*.s
./scripts/verify.sh         # check the artifacts
```

Or take the development image, which carries both toolchains already built,
and skip the setup entirely:

```bash
docker run --rm -it ghcr.io/psal-postech/mojo-m2ndp:main
./scripts/build.sh && ./scripts/verify.sh
./scripts/spike-smoke.sh    # run what came out — see docs/SIMULATION.md
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

### vector_add — RVV vectorization

`out/vector_add.s`:

```asm
.attribute 5, "rv64i2p1_..._v1p0_..._zve32f1p0_zve64d1p0_zvl128b1p0..."

call      __m2ndp_global_uthread_id   # M²NDP interface symbol
vsetvli   zero, a0, e32, m2, ta, ma   # RVV
vle32.v   v8, (a1)                    # vector load
vadd.vv   v8, v8, v10                 # vector add
vse32.v   v8, (a2)                    # vector store
```

The RVV backend inside the Mojo compiler is real and reachable purely
through a hand-written target attribute. No backend work needed.

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
kernel and kernel boundaries synchronize between them.

Note `__m2ndp_group_size` is re-called on **every** loop iteration: an
opaque external call cannot be proven loop-invariant. That is the real cost
of the external-symbol approach, and it disappears once the symbols become
intrinsics.

### histogram — scratchpad across three phases

Three phases of one kernel share a per-core bin array. Declaring the
scratchpad at struct level is what keeps them on the same storage:

```mojo
struct Histogram:
    comptime bins = scratchpad[BINS, Int32, name="hist_bins"]()
```
```llvm
@memory_blob_de5f15ab6daf7941 = internal addrspace(3) global [1024 x i8] zeroinitializer

; init:  store i32 0, ptr addrspace(3) %9
; body:  %9  = atomicrmw add ptr addrspace(3) %8, i32 1 monotonic
; final: %13 = atomicrmw add ptr %10, i32 %12 monotonic
```

One global, all three functions indexing off it. Calling `scratchpad()`
separately in each function would mint a fresh symbol per call site and the
phases would silently use different memory.

## Layout

```
src/m2ndp.mojo        M²NDP primitive library + compile target definition
benchmarks/           ports of M2NDP-public/examples/benchmarks
  memcpy.mojo         vector load + store, nothing else
  memset.mojo         scalar splat to a vector store
  vector_add.mojo     confirms RVV vectorization
  imdb_lt_int64.mojo  predicate scan -> bitmap (vmslt.vx)
  spmv.mojo           CSR SpMV — indirect access + atomic combine
  histogram.mojo      scratchpad shared across INIT/BODY/FINAL phases
scripts/
  setup.sh            install the Mojo toolchain
  env.sh              environment variables (source it)
  build.sh            compile benchmarks -> out/
  verify.sh           check the artifacts
docs/INTERFACE.md     the backend contract in detail
docs/EXAMPLES.md      annotated source -> LLVM IR -> assembly walkthrough
docs/STATUS.md        what works, what the backend must supply, open work
out/                  generated artifacts (not tracked by git)
```

## Writing a benchmark

```mojo
from m2ndp import global_uthread_id, atomic_add, scratchpad

comptime BINS = 256

struct Histogram:
    # Declared once at struct level so every phase shares one allocation.
    comptime bins = scratchpad[BINS, Int32, name="hist_bins"]()

@export                               # kernels are entry points, not called
def histogram_body(samples: UnsafePointer[Int32, MutAnyOrigin]):
    var bin = Int(samples[global_uthread_id()])
    _ = atomic_add(Histogram.bins + bin, Int32(1))
```

Kernels take ordinary parameters and index them; recovering the hardware's
mapped-address form is the compiler's job. `@export` is required — nothing
in the module calls a kernel, so it would otherwise be eliminated as dead
code.

## Backend interface contract

The entire benchmark set needs **4 symbols and 1 address space**. Details in
[`docs/INTERFACE.md`](docs/INTERFACE.md).

| Symbol | Signature | Meaning |
|--------|-----------|---------|
| `__m2ndp_local_uthread_id` | `i32 ()` | index within the group; also the scratchpad slot |
| `__m2ndp_global_uthread_id` | `i32 ()` | index across all cores; identifies the mapped data |
| `__m2ndp_group_size` | `i32 ()` | µthreads sharing one scratchpad |
| `__m2ndp_group_id` | `i32 ()` | which group this µthread belongs to |

| Address space | Use |
|---------------|-----|
| `addrspace(3)` | scratchpad (group-shared memory) |

When the backend is ready, replace **only the function bodies** in
`src/m2ndp.mojo` with real intrinsics. **Benchmark code does not change.**

Beyond these symbols the backend also has to decide how addrspace(3) globals
are placed and addressed, and two operations have no spelling at this level
at all (vector atomic, mask-to-bitmap). See
[`docs/INTERFACE.md`](docs/INTERFACE.md) for the contract and
[`docs/STATUS.md`](docs/STATUS.md) for what is left to do.

## How it works

Two things make this PoC possible.

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
        `features = "+m,+a,+f,+d,+v,+zvl128b", `,
        ...
    ]
```

`+v,+zvl128b` in `features` is what turns RVV on.

**2. M²NDP operations are expressed as external symbols.**
Mojo's `llvm_intrinsic[...]` only accepts intrinsics LLVM already knows;
unknown names are rejected during translation. So before the backend
exists, use `external_call`. It survives into the LLVM IR as `declare` +
`call`, which makes the backend mapping points explicit.

## Requirements

`python3` + `pip`. Verified on Linux x86-64.
