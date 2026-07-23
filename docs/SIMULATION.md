# Running M²NDP code

Nothing in this project had ever been executed. `verify.sh` checks the shape
of the IR and `build-llvm.sh check` checks that the backend agrees with
itself, but neither says whether an M²NDP instruction *does* the right thing.
This is the beginning of an answer.

Spike, the RISC-V ISA simulator, is the vehicle:

```bash
git submodule update --init --depth 1 third_party/riscv-isa-sim
apt-get install -y device-tree-compiler binutils-riscv64-unknown-elf
./scripts/build-spike.sh check
```

---

## What stands up today

```
[smoke] checking
  RVV through the pipeline     OK
  indexed vector atomic        OK
  all 64 instructions          OK
```

The split is deliberate. The first uses no M²NDP instruction at all — it
exists so the pipeline (assemble → link → load → execute → report) is known
good on its own, and a later failure means the extension rather than the
plumbing.

The second runs `m2ndp.vamoaddei32.v` by hand. Four lanes hit bins 1, 1, 3
and 0, so the array must come out `{1, 2, 0, 1}`; two lanes deliberately
collide, which is what a broken indexed atomic gets wrong.

The third is every instruction: 52 indexed vector atomics and 12 scalar
floating-point ones, 188 checks. `sim/gen-tests.py` emits it, computing the
expected results in Python — writing 64 of these by hand would be 64 chances
to work the answer out the same wrong way the simulator does. **The exit code
is the first test that disagreed**, so a failure names the instruction rather
than saying only that something is wrong.

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
