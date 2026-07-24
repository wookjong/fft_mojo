# Running M²NDP code

Nothing in this project had ever been executed. `verify.sh` checks the shape
of the IR and `build-llvm.sh check` checks that the backend agrees with
itself, but neither says whether an M²NDP instruction *does* the right thing.
This is the beginning of an answer.

Spike, the RISC-V ISA simulator, is the vehicle:

```bash
git submodule update --init --depth 1 third_party/riscv-isa-sim
apt-get install -y device-tree-compiler
./scripts/build-llvm.sh          # lld comes from here
./scripts/build-spike.sh check
```

---

## What stands up today

```
[smoke] checking
  RVV through the pipeline     OK
  indexed vector atomic        OK
```
```
[host-run] histogram, cores=1 interleave=1
[host] histogram ok
```

Two scripts, because they answer different questions. `spike-smoke.sh` asks
whether the instructions do what they say, from hand-written assembly.
`host-run.sh` asks whether a compiled workload runs, launched the way the
device launches one -- a host program in Mojo names a task, and the task is
compiled, run and checked from there.

The split is deliberate. The first check uses no M²NDP instruction at all — it
exists so the pipeline (assemble → link → load → execute → report) is known
good on its own, and a later failure means the extension rather than the
plumbing.

The second runs `m2ndp.vamoaddei32.v` by hand. Four lanes hit bins 1, 1, 3
and 0, so the array must come out `{1, 2, 0, 1}`; two lanes deliberately
collide, which is what a broken indexed atomic gets wrong.

`host-run.sh` is where the compiled workloads run: `histogram` exercises the
whole set at once — scratchpad globals at compiler-assigned offsets, an
indexed vector atomic, and three kernels sharing one scratchpad across
launches. The host program fills the inputs, launches the task, and folds its
own histogram to check against, so the result being checked does not come from
the same place as the result being produced.

The scratchpad-per-core property had been agreed and written down but never
tested. Running the same workload at one core and at four is what tests it —
`./scripts/host-run.sh histogram 4 8` against `... histogram 1 1` --
four cores at an interleave of eight, against one core.

## In CI

`.github/workflows/test.yml` runs all of this on every push and pull request:
the artifacts and their checks, the smoke test, and every workload at two
machine configurations. It runs on GitHub-hosted runners, pulling the image
`image.yml` publishes rather than building anything -- the toolchains are
already in it, and only building them needs the self-hosted runner. The
workspace is the checkout being tested, with `build` linked to the image's
tools, so what runs is the branch and not the copy baked into the image.

## Launching from the host

The device code is compiled Mojo; so is the program that launches it. A host
program names a task and gets it compiled, for the target the task declares,
run, and its results back:

```mojo
var hist = List[Int32](length=256, fill=0)
_ = Histogram.launch(PooledRange.over(samples),
                     HistogramParams(samples, hist))
# hist now holds the result
```

A launch takes what it is mapped over and what it is given, and nothing else.
`PooledRange` is the region of the pool the task covers -- divided by the
machine's packet, that is the microthread count -- and saying it as
`PooledRange.over(samples)` beats two bare integers whose meaning a reader has
to reconstruct.

There is deliberately no argument for the machine. How many cores exist and
how finely work is interleaved across them is the hardware's, so the runtime reads
`config/machine.conf` and configures itself; `M2NDP_MACHINE_CONFIG` points at
a different description. Host code that genuinely depends on the machine can
read the same file -- `Config.load().get("cores")` -- which exactly one
benchmark does, and only because of the limitation below.

Launching is a method on the task, not a free function taking one: it is the
other half of `device_main`, and a reader of a workload should not have to go
elsewhere to find how it is run. `NDPTask.launch` compiles the task through
`compile_info` for `Histogram.target`, finishes the lowering with our llc,
links against the device-side launcher, runs it under Spike, and downloads the
outputs into the caller's lists. The target and the packet size are the task's; the
machine is the config's. Varying the machine means varying the config, which is what
`scripts/host-run.sh` does to show an answer does not depend on the hardware.

`src/m2ndp_host.mojo` holds the machinery under it — files, processes, the
toolchain — and knows nothing about tasks. That is what keeps the dependency
one-way: the model reaches for the machinery, never the reverse.

A benchmark is **one file**, the way a `.cu` is: its kernels, its
`device_main`, and the host `main` that fills the inputs, launches and checks
the answer. Nothing splits them, because nothing has to — `compile_info`
compiles only what the entry point reaches, which is the same thing nvcc's
device pass does with a single source.

What that costs is a whole-module device build: `mojo build --target-triple
riscv64` on such a file fails, since it would compile the host `main` for the
device too. So `build.sh` asks the task instead —

```bash
./spmv --emit-ir          # NDPTask.device_ir(), through compile_info
```

— and gets exactly the IR a launch compiles, rather than an approximation of
it. `out/*.s` then comes from our llc, so for the first time the checked
assembly is the assembly that runs: identity values as register reads,
`m2ndp.vamoaddei32.v` as one instruction, and no frame on a kernel.

`launch` and `device_ir` are host-side methods on a device-side trait, which
sounds worse than it is: a device build reaches neither, so neither is
instantiated, and none of the host machinery has to exist on a core.

Five things about this are not obvious, and each cost a debugging session:

- **`compile_info` is a run-time call, not a comptime one.** Folding it at
  compile time fails inside the stdlib with nothing pointing at the cause.
- **It has to emit IR, not assembly.** Mojo's own LLVM has never heard of the
  vendor extension, so its assembly is unfinished — kernels with frames, calls
  where there should be register reads. Our llc is what finishes it.
- **An input field copies its bytes up front.** A field that only kept an
  address would not keep its list alive: nothing mentions the list after the
  expression that built the parameter block, so it can be freed before the
  upload reads it. See `Arg` in `src/m2ndp_host.mojo`.
- **Commands to `system()` are NUL-terminated by hand.** `String.unsafe_ptr()`
  promises no terminator, and the shell reads one command plus whatever
  followed it in memory otherwise — a syntax error on a well-formed line.
- **A defaulted `comptime` member needs a declared type to be overridable.**
  Written `comptime machine = Machine(1, 1)`, a conforming task cannot override
  it and the default is the only value it can ever have; written
  `comptime machine: Machine = Machine(1, 1)`, it can. The failure is a
  conformance error pointing at the task, not at the trait.

## The launcher

`sim/` implements the launcher half of the contract in C:

- **the scratchpad region is the launcher's to provide.** `.spad` only
  reserves a size; `__m2ndp_spad_size` says how much, and the base pointer is
  the region's own address, the compiler having laid the globals out from
  there. See `sim/launch.h`.
- **the identity values arrive in registers**, one per value. The assignment
  is provisional and lives in `RISCVM2ndpArgInfo.h`; `sim/launch.h` is the
  only other place that knows it, so settling the hardware ABI changes two
  files.
- **microthreads run one at a time**, sequentially. The contract has no
  barrier and combining happens through atomics, so that is a legal schedule
  — but it is one schedule out of many, and a race it does not happen to
  expose is a race this cannot find.

## Who decides what runs

Not the launcher. A task is a struct holding its kernels, the scratchpad they
share, and the `device_main` that says which of them runs in what order:

```mojo
struct Histogram(NDPTask):
    comptime bins = scratchpad[BINS, Int32, name="hist_bins"]()

    comptime Params = HistogramParams

    @staticmethod
    def initialize(): ...
    @staticmethod
    def body():                      # Histogram.params().samples.ptr
        ...
    @staticmethod
    def finalize(): ...

    @staticmethod
    def device_main():
        launch_serial[Histogram.initialize]()
        launch_parallel[Histogram.body]()
        launch_serial[Histogram.finalize]()
```

A kernel takes no arguments, so a launch names one and stops there. The
launcher copies the task's parameters into every core's scratchpad before
running a kernel on it, and the kernel reads them from its own with
`Histogram.params()` — a read of the base register the hardware supplies, with
each field a constant offset from it, the way a scratchpad global is.

That is what leaves nothing to pad and no positions to line up: one pair of
launch symbols serves every kernel of every task, so there is one signature to
agree on, and the empty one carries no order to get wrong. What a kernel reads
is named and typed by `Params` instead. The one cast from what the host filled
happens in the trait's `__m2ndp_rt_launch_task`, so no workload writes one.

The rule is enforced rather than agreed: the backend rejects a kernel that
declares an argument, since the frontend cannot state it. See
`xm2ndp-kernel-no-args.ll`.

The kernel is a *parameter* of the launch, not an argument to it. That is what
lets the launch symbols stay inside the library: `external_call` takes a
function only where it is named at the call site, and one passed as a runtime
argument does not convert, a declared function's type carrying its name. As a
parameter it keeps that identity, and what comes out is
`call void @__m2ndp_launch_serial(ptr @initialize)` — which is also what the
backend reads.

Conforming to `NDPTask` is the whole interface to the host. The trait carries
a default `__m2ndp_rt_launch_task`, so every task gets the entry point it is
launched through without a workload writing any launch glue — and that entry
point is the only symbol a task exports. `device_main` and the kernels stay
internal, which is what keeps one task per ELF from colliding with the next.

A launch has two halves, and only one of them is the machine's:

- **`__m2ndp_set_task_range(base, size)`**, in `sim/launcher.c`. A task runs
  over a memory range, and that range is what settles how many microthreads
  there are — one per packet. Nothing can be launched until this is known,
  which is why the runtime calls it before handing over.
- **`device_main`**, in the workload. Which kernels, in what order.

Underneath, `__m2ndp_launch_parallel` and `__m2ndp_launch_serial` are the
machine again. A `parallel` launch spreads one microthread per packet of the
range over the cores; a `serial` launch runs one microthread on each core,
which is what a kernel walking the scratchpad rather than the data needs —
alone on its core, so a strided walk from `local_uthread_id()` by
`group_size()` covers all of it. Both return only once every microthread has
retired, so a launch is synchronous and the order written is the order that
happens. Neither takes a size: how much work there is was settled when the
task was launched.

The topology — how many cores, how microthreads map to them, where the
scratchpads are — is not passed in. It is state the machine already holds
when a kernel is launched, so the model holds it the same way: file statics
in `launcher.c`, set before the task runs.

### What falls out of the frontend rather than the design

**Arguments arrive as one pointer**, the way CUDA's do. `@export` cannot be
applied to a parametric function, so the runtime entry point has one fixed
signature; a parameter per argument would then cap how many a task could take.
That entry point is where the block stops being untyped: it casts once, to the
task's `Params`, and everything below it works in that type.

**No kernel takes arguments**, because `external_call` allows one signature per
symbol name — a kernel using five buffers and one using none reach the same
entry. Putting the parameters where every kernel can already reach them is what
makes that one signature workable, and what makes a kernel's arguments names
and types instead of positions.

**A kernel named as a value becomes a closure copy**, and that copy is what
runs. The frontend names it after the function the value appeared in, so a
kernel launched from `device_main` is called something like
`Histogram::device_main(...)_closure_0` — a name that says the opposite of
what the function is.

Which is why the backend does not read names at all. A kernel is a function
whose address reaches one of the launch symbols:

```llvm
call void @__m2ndp_launch_serial(ptr @"Histogram::device_main(...)_closure_0", ...)
```

`isM2ndpKernel` looks for exactly that, in the function's own use list. Those
symbol names are this project's contract — the same ones `sim/launcher.c`
implements — so nothing outside the repository can change what the predicate
reads, which is not true of a mangling scheme. Everything not launched is
controller code, which is the right default: a function nothing spawns is not
a kernel in any useful sense.

Getting this backwards is not a diagnostic but a fault — a kernel would read
its arguments out of registers nobody filled in — so
`xm2ndp-device-main.ll` pins it with the two kinds deliberately misnamed:
a launched `device_main_lookalike` that must get the kernel ABI, and a
launching `kernel_closure_0` that must not.

## The extension is loadable, not a fork

This was the open question, and the answer is better than expected.

Spike's `extension_t` interface looked scalar-oriented — the bundled examples
are a RoCC accelerator and a cache-flush instruction. But `processor_t::VU`
is public and `vectorUnit_t::elt<T>()` gives element access, so an extension
can implement the whole set without patching the simulator.
`sim/ext/m2ndp_ext.cc` does exactly that.

So the submodule stays pristine and there is no rebase burden as Spike moves.

**Both flags are needed**, which is not obvious:

```bash
spike --extlib=build/spike/libm2ndp_ext.so --extension=m2ndp ...
```

`--extlib` loads the library; `--extension` turns it on. With only the first,
the instructions are still illegal and the failure looks like a decode bug.

## Things that cost time, recorded so they do not again

**`match` must be a subset of `mask`.** Spike tests `(insn & mask) == match`,
so a match bit outside the mask can never be satisfied and the instruction
silently traps as illegal. Derive it: `match = encoding & mask`.

**Bare metal starts with the vector unit off.** `mstatus.VS` and `mstatus.FS`
are zero out of reset, so the first vector or floating-point instruction
traps. Whatever the M²NDP runtime turns out to be, it has to set them.

**`.spad` must be assigned to no segment, and saying nothing is not the
same as saying none.** The section is allocatable, so it ends up somewhere:
GNU ld gives it a segment of its own at address 0, and lld folds it into the
neighbouring one and drags that down to 0. Either way a loader honours the
address and fails, because there is no memory at 0 and there was never meant
to be — the region is the launcher's to provide. `scripts/m2ndp.lds` says
`:NONE` outright. Leaving the assignment off works under GNU ld and not
under lld, which inherits the neighbouring segment.

Marking the section non-allocatable also avoids the segment, but turns it
into `PROGBITS` and puts the whole reservation in the file as zeros.

**The linker has to be lld.** A distribution binutils is older than this
LLVM and rejects the ISA string it emits — ours stops at `zmmul`. It only
shows up once objects from two toolchains are linked together, because
merging the RISC-V attributes is what triggers the check, and discarding the
section in the link script does not help: the merge happens first.

**Spike has to be told where memory is.** `scripts/m2ndp.lds` puts code at
`0x10000`; Spike's default region starts at `0x80000000`, so it needs
`-m0x10000:0x1ff0000`. The size is not arbitrary — the region has to stop
short of `0x2000000`, where Spike's CLINT lives, or startup fails with a
device overlap.

**Spike's release tags are stale** — v1.1.0 is from 2021. The submodule is
pinned to a recent `master` commit instead, the same way the LLVM submodule
follows `release/23.x` rather than a tag.

**`--varch` is gone.** VLEN now comes from the ISA string (`zvl128b`).

## Where the scratchpad lives

Nowhere, as far as the link is concerned, and that is the point.

An earlier version of this file had the simulator answer the question by
giving `.spad` an absolute address. The calling convention work answered it
differently and better: the compiler assigns every scratchpad variable a
constant offset from a base pointer the hardware supplies, `.spad` only
reserves the space, and `__m2ndp_spad_size` tells a launcher how far above
its region to put the base. See `scripts/m2ndp.lds` and
[`INTERFACE.md`](INTERFACE.md).

So the tests link with the task's own script rather than a simulator-specific
one. The layout being exercised should be the layout the compiler was built
against; a second script would only be a second thing to get out of step.

Nothing here sets the base register yet — the tests are hand-written assembly
that does not use compiler-assigned scratchpad. That is the next piece.

## What this cannot check

Worth stating plainly, because a passing simulation reads like more than it
is.

- **Races.** µthreads will be run one at a time. The contract has no barrier
  and combining happens through atomics, so a sequential schedule is a legal
  one — but it means atomicity bugs cannot surface.
- **Performance.** Spike is functional; there is no timing model.
- **Agreement with real M²NDP hardware.** The encodings and semantics are
  ours, provisional, and the simulator implements the same guesses the
  compiler does. This checks that the two halves agree with each other, not
  that either matches the architecture. That has to wait for a spec.

## What the instructions do here

The extension decides an instruction by (operation, index width) from the
encoding and takes the *data* width from `vtype` at run time — the same split
the indexed loads and stores use, and the reason 52 vector instructions need
only 52 entries rather than 52 × 4.

Floating-point arithmetic goes through softfloat rather than the host's
`float`, so NaN propagation, signed zero and the exception flags agree with
the rest of Spike instead of with whatever the host happens to do.

`aq` and `rl` are decoded but have no effect. Nothing here reorders anything,
so the four ordering variants of a scalar atomic are one entry.

## Next

- Run a compiled kernel rather than hand-written assembly. Needs the loader
  to place arguments and set up microthread identity, which depends on the
  calling convention still being settled.
- A per-benchmark harness: generate inputs, run, compare against a reference
  computed on the host.
- Disassembly. `get_disasms` returns nothing, so a trace prints raw bits for
  these; `llvm-objdump` is what they get read with, but a trace is easier to
  follow when it is all in one place.
