# Running M²NDP code

`verify.sh` checks the shape of the IR and `build-llvm.sh check` checks that the
backend agrees with itself, but neither says whether an M²NDP instruction *does*
the right thing. Running it does.

The vehicle is [M²NDP-Detour](https://github.com/PSAL-POSTECH/M2NDP-Detour), the
cycle-level M²NDP timing simulator. It both *executes* a task and *times* it: a
controller runs the compiled `device_main`, which rings a doorbell to launch each
kernel on the NDP units, and the run reports the cycles it took. Building it is a
plain CMake build (the development image carries it already):

```bash
./scripts/build-llvm.sh          # our LLVM + lld
cmake -S third_party/m2ndp-detour -B third_party/m2ndp-detour/build
cmake --build third_party/m2ndp-detour/build -j"$(nproc)" --target m2ndp_run
```

For the model itself -- the controller, the units, the memory hierarchy -- see
[`TIMING.md`](TIMING.md). This file is about *running* a compiled workload and
what a run does and does not catch.

---

## What stands up today

```
[host-run] histogram
[host] histogram ok
```

`host-run.sh` is where the compiled workloads run: a host program in Mojo names a
task, and the task is compiled, linked against the controller launcher, run on
Detour and checked -- all from there. `histogram` exercises the whole set at once:
scratchpad globals at compiler-assigned offsets, an indexed vector atomic, and
three kernels sharing one scratchpad across launches. The host fills the inputs,
launches the task, and folds its own histogram to check against, so the result
being checked does not come from the same place as the result being produced.

Detour also ships small C harnesses (`dev_launch`, `dev_launch_loop`,
`dev_launch_masked`) that stage a fixture and run it without the Mojo host -- the
fastest way to prove the controller path stands up.

## In CI

`.github/workflows/test.yml` runs this on every push and pull request: the
artifacts and their checks (`build.sh` + `verify.sh`), and `timing-run.sh` on
Detour's controller. The tests are on GitHub-hosted runners and build nothing;
only the image is built, on the self-hosted runner, because the toolchains in it
are 25 minutes and more disk than a hosted runner has free.

It calls `image.yml` rather than naming a tag, and runs in the image that call
returns, by digest. So a branch is tested against the image built from the branch
-- a change to the compiler is tested with that compiler. The workspace is the
checkout being tested, with `build` linked to the image's tools, so what runs is
the branch and not the copy baked into the image.

## Launching from the host

The device code is compiled Mojo; so is the program that launches it. A host
program names a task and gets it compiled, for the target the task declares, run,
and its results back:

```mojo
var hist = List[Int32](length=256, fill=0)
_ = Histogram.launch(PooledRange.over(samples),
                     HistogramParams(samples, hist))
# hist now holds the result
```

A launch takes what it is mapped over and what it is given, and nothing else.
`PooledRange` is the region of the pool the task covers -- divided by the machine's
packet, that is the microthread count -- and saying it as `PooledRange.over(samples)`
beats two bare integers whose meaning a reader has to reconstruct.

There is deliberately no argument for the machine. How many cores exist and how
finely work is interleaved across them is the hardware's, so the runtime reads the
simulator config and configures itself; `M2NDP_CONFIG` points at a different
description. Host code that genuinely depends on the machine can read the same
file -- `Config.load().get("num_ndp_units")`.

Launching is a method on the task, not a free function taking one: it is the other
half of `device_main`, and a reader of a workload should not have to go elsewhere
to find how it is run. `NDPTask.launch` compiles the task through `compile_info`
for `Histogram.target`, finishes the lowering with our llc, links against the
controller launcher, runs it on Detour over the shared pool, and reads the outputs
back out of that pool. The target and the packet size are the task's; the machine
is the config's.

`src/m2ndp_host.mojo` holds the machinery under it -- the pool, files, processes,
the toolchain -- and knows nothing about tasks. That is what keeps the dependency
one-way: the model reaches for the machinery, never the reverse.

A benchmark is **one file**, the way a `.cu` is: its kernels, its `device_main`,
and the host `main` that fills the inputs, launches and checks the answer. Nothing
splits them, because nothing has to -- `compile_info` compiles only what the entry
point reaches, the same thing nvcc's device pass does with a single source.

What that costs is a whole-module device build: `mojo build --target-triple
riscv64` on such a file fails, since it would compile the host `main` for the
device too. So `build.sh` asks the task instead --

```bash
./spmv --emit-ir          # NDPTask.device_ir(), through compile_info
```

-- and gets exactly the IR a launch compiles. `out/*.s` then comes from our llc,
so the checked assembly is the assembly that runs: identity values as register
reads, `m2ndp.vamoaddei32.v` as one instruction, and no frame on a kernel.

Four things about this are not obvious, and each cost a debugging session:

- **`compile_info` is a run-time call, not a comptime one.** Folding it at compile
  time fails inside the stdlib with nothing pointing at the cause.
- **It has to emit IR, not assembly.** Mojo's own LLVM has never heard of the
  vendor extension, so its assembly is unfinished -- kernels with frames, calls
  where there should be register reads. Our llc is what finishes it.
- **A parameter field copies its bytes up front.** A field that only kept an
  address would not keep its list alive: nothing mentions the list after the
  expression that built the parameter block, so it could be freed before the run
  reads it.
- **Paths handed to `open`/`system()` are NUL-terminated by hand.**
  `String.unsafe_ptr()` promises no terminator, so without one the call names the
  file after the path plus whatever bytes follow it in memory.

## The launcher

`sim/m2ndp_launcher.c` is the device half of the launch contract, linked in place
of the workload's own entry: a controller runs it, and it drives the kernels.

- **A launch is a doorbell.** `__m2ndp_launch_parallel`/`__m2ndp_launch_serial`
  write a launch command to the fixed device MMIO block (`m2ndp_launch_abi.h`) and
  ring the doorbell; the controller runs the launch and sets completion, which the
  launcher polls. So a launch returns only once its kernel is done -- the launch is
  synchronous and the order written is the order that happens.
- **`__m2ndp_rt_launch_task(base, size)`** hands the controller the task's data
  range. A task runs over that range, and the range is what settles how many
  microthreads there are -- one per packet -- so nothing can be launched until it
  is known.

Every fixed address -- the pool, the doorbell MMIO, the per-unit scratchpads, the
execution stacks -- lives in one place, `third_party/m2ndp-detour/src/address_map.h`.
The base of each region is fixed there; only the pool's size is a config knob, so
no two regions can be configured into overlap. The host maps the pool at that base
and the simulator attaches the same file at the same base, which is what makes a
pointer mean the same thing to each. Run with `M2NDP_DUMP_MAP=1` to print the map.

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

A kernel takes no arguments, so a launch names one and stops there. The controller
copies the task's parameters into every unit's scratchpad before running a kernel
on it, and the kernel reads them from its own with `Histogram.params()` -- a read
of the base register the hardware supplies, with each field a constant offset from
it, the way a scratchpad global is.

That is what leaves nothing to pad and no positions to line up: one pair of launch
symbols serves every kernel of every task, so there is one signature to agree on,
and the empty one carries no order to get wrong. What a kernel reads is named and
typed by `Params` instead. The one cast from what the host filled happens in the
trait's `__m2ndp_rt_launch_task`, so no workload writes one.

The rule is enforced rather than agreed: the backend rejects a kernel that declares
an argument, since the frontend cannot state it. See `xm2ndp-kernel-no-args.ll`.

The kernel is a *parameter* of the launch, not an argument to it. That is what lets
the launch symbols stay inside the library: `external_call` takes a function only
where it is named at the call site, and one passed as a runtime argument does not
convert, a declared function's type carrying its name. As a parameter it keeps that
identity, and what comes out is
`call void @__m2ndp_launch_serial(ptr @initialize)` -- which is also what the
backend reads.

Conforming to `NDPTask` is the whole interface to the host. The trait carries a
default `__m2ndp_rt_launch_task`, so every task gets the entry point it is launched
through without a workload writing any launch glue -- and that entry point is the
only symbol a task exports. `device_main` and the kernels stay internal, which is
what keeps one task per ELF from colliding with the next.

A `parallel` launch spreads one microthread per packet of the range over the units;
a `serial` launch runs one microthread on each unit, which is what a kernel walking
the scratchpad rather than the data needs -- alone on its unit, so the whole of it
is that microthread's to walk. Which unit a microthread lands on is the address it
was mapped to, divided by the interleave stride, modulo the unit count. A range that
starts mid-round, or ends mid-packet, or is narrower than one stride simply spreads
unevenly; the reference tolerates that and so does this.

### What falls out of the frontend rather than the design

**Arguments arrive as one pointer**, the way CUDA's do. `@export` cannot be applied
to a parametric function, so the runtime entry point has one fixed signature; a
parameter per argument would then cap how many a task could take. That entry point
is where the block stops being untyped: it casts once, to the task's `Params`.

**A kernel named as a value becomes a closure copy**, and that copy is what runs.
The frontend names it after the function the value appeared in, so a kernel launched
from `device_main` is called something like `Histogram::device_main(...)_closure_0`
-- a name that says the opposite of what the function is.

Which is why the backend does not read names at all. A kernel is a function whose
address reaches one of the launch symbols:

```llvm
call void @__m2ndp_launch_serial(ptr @"Histogram::device_main(...)_closure_0", ...)
```

`isM2ndpKernel` looks for exactly that, in the function's own use list. Those symbol
names are this project's contract -- the same ones the launcher implements -- so
nothing outside the repository can change what the predicate reads. Everything not
launched is controller code, which is the right default: a function nothing spawns
is not a kernel in any useful sense. `xm2ndp-device-main.ll` pins it with two kinds
deliberately misnamed: a launched `device_main_lookalike` that must get the kernel
ABI, and a launching `kernel_closure_0` that must not.

## Where the scratchpad lives

Nowhere, as far as the link is concerned, and that is the point. The compiler
assigns every scratchpad variable a constant offset from a base pointer the hardware
supplies; `.spad` only reserves the space, and `__m2ndp_spad_size` tells the
controller how far above the scratchpad to put the base. The controller seeds that
base register per microthread from the unit's scratchpad, which the address map
places at a fixed slot per unit. See `scripts/m2ndp.lds` and
[`INTERFACE.md`](INTERFACE.md).

So a task links with its own script rather than a simulator-specific one. The layout
being exercised should be the layout the compiler was built against; a second script
would only be a second thing to get out of step.

## What this catches, and what it does not

- **Correctness.** The instructions run for real, so a workload's answer is checked
  against one the host computes itself.
- **Timing.** Detour is cycle-level: a run reports the cycles a launch took, unlike
  a purely functional simulator.
- **Concurrency, within the model.** Microthreads occupy real slots with their own
  stacks and interleave on the units, so a schedule is not forced to be sequential.
  It is still one model's schedule, not every legal one, so a race it does not
  happen to expose is a race this cannot find.
- **Not agreement with real M²NDP hardware.** The encodings and semantics are ours,
  provisional, and the simulator implements the same guesses the compiler does. This
  checks that the two halves agree with each other, not that either matches the
  architecture. That has to wait for a spec.
