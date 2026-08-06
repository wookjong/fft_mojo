# mojo-m2ndp

A PoC for writing M²NDP (RISC-V Vector based µthread GPNDP) workloads in
Mojo, compiling them for a RISC-V/RVV target, and running them.

A workload is one file: its kernels, the `device_main` that launches them,
and the host code that feeds it and checks the answer -- single source, the
way a `.cu` is. Naming the task compiles it for the target the task declares
and runs it on the M²NDP-Detour timing simulator:

```mojo
_ = Histogram.launch(PooledRange.over(samples, n),
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

## Getting started

The shortest path from nothing to something running on the simulator is the
development image, which carries both toolchains already built:

```bash
docker run --rm -it ghcr.io/psal-postech/mojo-m2ndp:main
./scripts/host-run.sh hello
```

That compiles `benchmarks/hello.mojo` for the device, links it, runs it on
M²NDP-Detour, and — among a few hundred lines of the simulator's own logging —
prints:

```
Hello[2026-08-06 21:11:02.310] [info] NDP  0: average Data Cache Hit : 0, ...
, world!
2 + 2[2026-08-06 21:11:02.475] [info] NDP  0: average Data Cache Hit : 0, ...
 = 4
...
[host] hello ok
```

`Hello, world!` and `2 + 2 = 4` are the device talking. They arrive a byte at a
time, as the controller writes them to the UART, and the simulator logs to the
same terminal — so a log line lands wherever it happens to fall, including
mid-word. Nothing is wrong when that happens.

`[host] hello ok` is the host program, after the launch returned.

Do not mount over `/work`: the toolchains live there and a mount hides them.

### Hello, world

The whole workload — kernels, `device_main`, and the host `main` that launches
it — is one file, the way a `.cu` is:

```mojo
from m2ndp import NDPTask, DeviceConsole, PooledRange
from m2ndp_host import cxl_alloc


@fieldwise_init
struct HelloParams(Movable):
    var scratch: UnsafePointer[Int32, MutAnyOrigin]


struct Hello(NDPTask):
    comptime Params = HelloParams

    @staticmethod
    def device_main():
        var con = DeviceConsole()
        con.write("Hello, world!\n")
        con.write("2 + 2 = ", 2 + 2, "\n")


def main() raises:
    if Hello.emit_ir_if_asked():
        return
    var scratch = cxl_alloc[Int32](8)
    var rc = Hello.launch(PooledRange.over(scratch, 8), HelloParams(scratch))
    if rc != 0:
        print("[host] hello failed, exit", rc)
        return
    print("[host] hello ok")
```

`device_main` is the device side: it runs on the NDP controller and says which
kernels run in what order. This one launches none — it only prints, which is
what makes it the smallest workload there is.

`DeviceConsole` writes to the controller's UART and the runtime streams that to
the host stdout. `write` takes any `Writable`, so a value formats the way
`print` does on the host. Only `device_main` gets one: a kernel runs on the
cores, and they own no UART.

The rest is what every task needs. `Params` is declared once and both sides
read it — the pool is shared memory, so a parameter is a plain pointer and
nothing is transferred. `launch` takes the range of the pool the task runs
over, which is what settles how many microthreads there are (one per 32-byte
packet). Nothing here launches a kernel, so the range goes unused, but a range
there must be.

From here, [`benchmarks/vector_add.mojo`](benchmarks/vector_add.mojo) is the
same shape with one kernel and real data, and
[Writing a benchmark](#writing-a-benchmark) below walks through it.

### Building from a checkout

```bash
git clone <this-repo> && cd mojo-m2ndp
./scripts/setup.sh          # install the Mojo toolchain (./toolchain)
./scripts/build.sh          # each benchmark's device IR -> out/*.ll (+ .s with our llc)
./scripts/verify.sh         # check the artifacts
```

To run one rather than inspect it, you also need our LLVM:

```bash
./scripts/build-llvm.sh     # our LLVM, with the vendor extension
./scripts/host-run.sh       # every benchmark, checked against its own answer
./scripts/test.sh           # the tiers, cheapest first — see tests/
```

The image has no LLVM source in it — 2.6 GB that nothing above reads, since the
compiler is already built. Rebuilding LLVM or running its lit suite needs a
checkout.

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
slli      a5, a5, 6                   # x 64: one packet
vsetivli  zero, 16, e32, m4, ta, ma   # RVV, 16 int32 lanes = one packet
vle32.v   v8, (a1)
vadd.vv   v8, v8, v12
vse32.v   v8, (a0)
ret                                   # no frame: a kernel preserves nothing
```

Two halves are visible here. RVV comes from the Mojo compiler, reached purely
through a hand-written target attribute. Everything else is the extension: a
kernel that takes no arguments and reads the task's parameters out of the
scratchpad, the identity values as live-in registers rather than calls, and no
frame because nothing resumes after a kernel.

### spmv — indirect access

One row to a µthread: it walks the row itself and stores the answer once, so
nothing has to be combined. Indirect access (`x[col_idx[k]]`) is just a
dependent load chain — no special construct needed:

```llvm
%36 = load i32, ptr %35, align 4                     ; col_idx[k]
%37 = sext i32 %36 to i64
%38 = getelementptr inbounds float, ptr %3, i64 %37  ; &x[col_idx[k]]
%40 = load float, ptr %38, align 4                   ; x[col_idx[k]]
%41 = fmul contract float %39, %40
%42 = fadd contract float %16, %41                   ; -> fmadd.s in asm
```

Every call left in `out/spmv.s` is controller-side -- the launch and the range
the runtime sets before it -- and none is in a kernel, where a call is
rejected outright.

**M²NDP has no barrier.** µthreads are created and retired by hardware FGMT,
so there is no well-defined set to synchronize; where µthreads do share an
output, as histogram's do, they combine with an atomic, and kernel boundaries
synchronize between phases.

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

One global for the bins, all three kernels indexing off it. Calling
`scratchpad()` separately in each would mint a fresh symbol per call site and
they would silently use different memory. The task's parameter block is a
second such global, declared by the trait rather than the workload -- which is
why a kernel reads it at a constant offset and needs no argument.

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
benchmarks/           one file per workload, each holding its kernels, its
                      device_main and its host main. Mostly ports of
                      M2NDP-public/examples/benchmarks -- 24 of them -- plus
                      hello.mojo and the devmain_* controller cases.
                      docs/STATUS.md lists what each one exercises
tests/manifest.tsv    which cases each test tier draws from
sim/                  the device-side launcher (m2ndp_launcher.c) and host stubs
scripts/
  setup.sh            install the Mojo toolchain
  env.sh              environment variables (source it)
  build.sh            each benchmark's device IR + assembly -> out/
  verify.sh           check the artifacts
  test.sh             the test tiers, cheapest first (t0 toolchain .. t4 E2E)
  build-llvm.sh       our LLVM (vendor extension) + lld
  link-m2ndp.sh       link a task against the controller launcher
  host-run.sh         run a workload and check its own answer
  timing-run.sh       run a benchmark on M²NDP-Detour's controller
docs/SIMULATION.md    running compiled workloads, and what that does not catch
docs/TIMING.md        the M²NDP-Detour timing model and how a task runs on it
docs/INTERFACE.md     the backend contract in detail
docs/EXAMPLES.md      annotated source -> LLVM IR -> assembly walkthrough
docs/STATUS.md        what works, what the backend must supply, open work
third_party/          the LLVM fork and M²NDP-Detour, as submodules
out/                  generated artifacts (not tracked by git)
```

## Writing a benchmark

One file, three parts: what the host passes, the task, and the host code.

```mojo
comptime UNROLL = PACKET // size_of[Int32]()   # samples in one packet

@fieldwise_init
struct HistogramParams(Movable):
    var samples: UnsafePointer[Int32, MutAnyOrigin]   # plain pointers:
    var out_hist: UnsafePointer[Int32, MutAnyOrigin]  # the pool is shared

struct Histogram(NDPTask):
    comptime Params = HistogramParams             # what the host fills in
    # Declared once at struct level so every kernel shares one allocation.
    comptime bins = scratchpad[BINS, Int32, name="hist_bins"]()

    # A kernel takes no arguments: the parameters are a scratchpad global
    # like `bins`, so reading a field is a constant offset from the base.
    @staticmethod
    def body():
        ...Histogram.params[].samples...

    @staticmethod
    def device_main():
        launch_serial[Histogram.initialize]()
        launch_parallel[Histogram.body]()
        launch_serial[Histogram.finalize]()

def main() raises:
    if Histogram.emit_ir_if_asked():
        return
    var samples = cxl_alloc[Int32](n)       # the shared pool; host writes here
    var hist = cxl_alloc[Int32](BINS)
    ...fill samples...
    _ = Histogram.launch(PooledRange.over(samples, n),
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

What a benchmark never says is what hardware it runs on -- not the core count,
not the interleave, and not the packet its kernels index by, which is `PACKET`
and comes with the library. That is all the simulator config, which the runtime
reads; pointing `M2NDP_CONFIG` at another description is how the same program is
shown to give the same answer on a different machine.

## The pool is shared, not copied

M²NDP is near-data processing: the data is already in the CXL pool and the
NDP cores are attached to that memory. Host and device see the same bytes.
There is no `cudaMemcpy` here because there is nothing to copy — which is why
a parameter is a plain pointer and no field says "in" or "out".

Two processes do not get that for free, so the pool is a file both sides map
at the same address:

```
host                                    Detour
  mmap(MAP_FIXED, pool_base) ──┐   ┌── mmap(MAP_FIXED, pool_base)
                               └───┴──  the same pages
```

`cxl_alloc[Int32](n)` returns an address that is a device address too, so a
kernel loads exactly what the host stored. There is one pool per process --
the runtime creates it on the first `cxl_alloc` and `launch` reaches the same
one -- so a workload never names it. Nothing is uploaded before a run
and nothing is downloaded after it; results are in the caller's pool because
they were written there.

A launch is then five steps, and only the middle three are compilation:

1. **Fill.** The host writes into the mapping. `Pool` bumps a pointer per
   `alloc`; there is no free, a run being short.
2. **Compile.** `Task.device_ir()` asks `compile_info` for the entry point and
   what it reaches, for the target the task declares.
3. **Finish and link.** Our `llc` does the M²NDP lowering the frontend's LLVM
   cannot, and `ld.lld` puts the result beside the controller launcher.
4. **Run.** `m2ndp_run` attaches the pool the host mapped and runs the task on
   Detour's controller; the addresses that cross are pool addresses:

   ```
   m2ndp_run <config> task.elf <pool_file> <pool_base> <pool_bytes> \
             <range_base> <range_size> <params> <params_bytes>
   ```

   The controller runs `device_main`, whose launches ring a doorbell to run each
   kernel on the units. It reads no files and writes none: the parameter block is
   already in the pool, and the controller copies it into each unit's scratchpad
   so a kernel finds it at a constant offset.
5. **Read.** The host reads its own pointers. The process exits; the mapping
   does not need unwinding for the answer to be there.

Nothing else has to be told where the pool is: both sides map the same file at
the base fixed in `address_map.h`, so no simulator device or MMIO plumbing sits
between the kernel and the bytes.

`pool_bytes` is the simulator config's `memory_expander_size`; the base is fixed
in `address_map.h`. How much memory the device has is the hardware's business,
not a workload's.

## The symbol contract

The seam between workload and compiler is a set of names. Details in
[`docs/INTERFACE.md`](docs/INTERFACE.md).

**What a µthread is handed** — lowered to live-in registers, not calls:

| Symbol | Signature | Meaning |
|--------|-----------|---------|
| `__m2ndp_local_uthread_id` | `i32 ()` | index among the µthreads on this core |
| `__m2ndp_global_uthread_id` | `i32 ()` | index across all cores; identifies the mapped data |
| `__m2ndp_group_id` | `i32 ()` | which group this µthread belongs to |
| `__m2ndp_declare_params` | `void (ptr addrspace(3))` | which global the host fills; the backend exports its offset and drops the call |

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
`verify.sh` need. Running a workload also wants the LLVM fork and M²NDP-Detour:
`build-llvm.sh` builds the fork from its submodule and CMake builds Detour; the
development image carries both already. `riscv64-unknown-elf-gcc` compiles the
device-side launcher.

Verified on Linux x86-64.
