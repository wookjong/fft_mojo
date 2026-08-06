# Backend interface contract

This document defines the LLVM-level interface that `src/m2ndp.mojo`
generates. The backend team only has to handle these symbols and this
address space — they never need to read the workload code.

For what each of these looks like in context, see
[`EXAMPLES.md`](EXAMPLES.md).

## Symbols

Every symbol comes out as a C-ABI function taking no arguments.

| Symbol | LLVM signature | Returns | Meaning |
|--------|----------------|---------|---------|
| `__m2ndp_local_uthread_id` | `declare i32 @__m2ndp_local_uthread_id()` | `i32` | index among the µthreads on this core |
| `__m2ndp_global_uthread_id` | `declare i32 @__m2ndp_global_uthread_id()` | `i32` | index across all cores; identifies the mapped data |
| `__m2ndp_group_id` | `declare i32 @__m2ndp_group_id()` | `i32` | which group this µthread belongs to |
| `__m2ndp_num_groups` | `declare i32 @__m2ndp_num_groups()` | `i32` | how many groups the task spreads across |
| `__m2ndp_spad_capacity` | `declare i32 @__m2ndp_spad_capacity()` | `i32` | bytes of scratchpad on one unit |

The IDs are `i32` and get sign-extended to `i64` at every use site, since
the index type is 64-bit (`index_bit_width = 64`).

### Loop invariance — fixed

An opaque external call cannot be hoisted out of a loop, so a kernel reading
its own identity inside one paid for the read on every iteration. The
intrinsics these become are `IntrNoMem` and speculatable, so the read happens
once before the loop and the body just uses the value.

### Group index

The two µthread IDs follow Arachne's `GlobalUThreadID()` / `LocalUThreadID()`:
one across all cores, one within a core. Both come from the hardware — a
µthread is handed its identity in scalar registers when spawned.

"Group" means the set of µthreads sharing one scratchpad, i.e. those
resident on one NDP core. `local_uthread_id()` indexes into it and
`group_id()` says which one it is.

M2NDP-public gives a µthread three values and no more: the address it was
mapped to, that address less the range's base, and which core it is on. A
kernel there works out everything else from its offset. `local_uthread_id()`
is ours; the reference has no per-core index and no count of the µthreads
sharing a core.

### How the IDs arrive — live-in registers

The values are placed in registers when the microthread is spawned, so the
symbols lower to reads of those registers. They are **not reserved**:
each is copied into a virtual register at function entry, after which the
physical register goes back into the allocation pool. A kernel that never
reads a value produces no copy at all.

The copy is anchored to function entry rather than to the point of the
call, so a read from inside a loop still takes the value as it was on
entry. That is what makes it safe for the hardware to write the register
once and for anything to reuse it afterwards.

What this replaced is worth stating, since it was the single largest cost
in the generated code. A call in a leaf kernel needs a return address
saved, argument registers evacuated, callee-saved registers to evacuate
them into, and a frame to spill those. Measured across the benchmarks that
use identity values:

| | calls before | calls after | kernel frames before | after |
|---|---|---|---|---|
| `vector_add` | 1 | 0 | 1 | 0 |
| `spmv` | 1 | 0 | 1 | 0 |
| `histogram` | 5 | 0 | 3 | 0 |

**Which register carries which value is provisional.** The assignment lives
in `RISCVM2ndpArgInfo.h` and nowhere else, so settling the hardware ABI
changes one table. The descriptor there also allows a value to arrive in
memory, or in a bitfield of a register shared with another value — AMDGPU
needs all three forms for the same problem, and being ready for them costs
nothing.

## Scratchpad

Unlike the symbols above, this one is not a mechanical substitution. The
symbols say "replace this call with that instruction"; the scratchpad asks
the backend to make placement and addressing decisions.

### What the compiler emits

Each `scratchpad[count, T, name=...]()` in a kernel becomes one global in
address space 3. The compiler assigns storage, so no offsets appear in
source:

```mojo
comptime bins = scratchpad[BINS, Int32, name="hist_bins"]()   # BINS = 256
```
```llvm
@memory_blob_de5f15ab6daf7941 = internal addrspace(3) global [1024 x i8] zeroinitializer, align 4

%9 = getelementptr inbounds i32, ptr addrspace(3) @memory_blob_de5f15ab6daf7941, i64 %4
     store i32 0, ptr addrspace(3) %9, align 4
```

Two properties of that global the backend cannot assume away:

- **The `name=` argument does not reach the symbol.** It is spelled
  `memory_blob_<hash>`, so a scratchpad buffer cannot be identified by name
  in the IR or the object file. Anything that has to recognise these must
  key off the address space, not the symbol name.
- **The global is an untyped byte blob**, `[1024 x i8]`, not `[256 x i32]`.
  The element type appears only on the GEPs and the accesses.

`histogram` is the only one of the six benchmarks that allocates a
scratchpad. `spmv` writes its answer straight to ordinary memory and has none.

Address space 3 follows the GPU shared-memory convention. If M²NDP wants a
different number, change it in `src/m2ndp.mojo`; ordinary memory stays in
address space 0 as a plain `ptr`.

### Placement and layout

The scratchpad belongs to a task, so there is no fixed address to name. Each
global becomes a constant offset from a base pointer the hardware hands the
microthread at spawn:

```asm
sw zero, 16(a0)           # a0 = scratchpad base
```

No address materialization at all — the offset lands in the access itself.

```
base |[ globals, the task's parameters among them ]
```

The base is where a task's scratchpad starts and the globals are laid out from
it, so every offset is positive. There is nothing else in the region: a kernel
takes no arguments, so there is no argument area to keep clear of.

### Who decides the offsets, and why it is the compiler

`RISCVM2ndpLowerScratchpad` assigns them, ordered by name so the result does
not depend on the order globals happen to appear. The globals then collapse
into one opaque block in `.spad`, which reserves the space and gives the
section a size.

Leaving this to the linker was the obvious alternative and does not work.
Expressing one global's distance from another needs to know them all, and
only whole-module code does — a symbol difference is not a relocatable
expression, so the linker cannot be asked for it:

```
error: expected relocatable expression
    lw a1, %lo(bins - __spad_arg_base)(a0)
```

Deciding in the compiler also removes relocations entirely. The offsets are
plain constants.

**This holds because one task is one module.** If a task were ever linked
from several objects, no single compilation would see all the globals and
the layout would have to move back to the linker — at the cost of the
arguments-first ordering.

### Reaching a peer's scratchpad — the offset symbol

A kernel accesses its own scratchpad through `scratchpad_base`, the per-core
register above. `device_main` — and a kernel reaching another group — cannot: in
controller code that register holds something else, so
`riscv_m2ndp_scratchpad_base` is rejected outside a kernel. To let those form a
peer address instead, a task asks for a global's offset with a call the layout
pass resolves, mirroring `__m2ndp_declare_params`:

| | |
|---|---|
| `__m2ndp_scratchpad_offset` | `declare i64 @__m2ndp_scratchpad_offset(ptr addrspace(3))` — a scratchpad global's constant offset within `.spad`, no base |

`RISCVM2ndpLowerScratchpad` replaces each call with the constant offset and skips
its use of the global in the base-relative rewrite, so no `scratchpad_base` is
emitted and the kernel-only check leaves it alone. The mojo `spad_addr` primitive
adds this offset to the target unit's region base, giving an addrspace(0) absolute
address the memory path routes to that unit. See
[`DEV-spad-addr-peer.md`](DEV-spad-addr-peer.md).

### What the launcher needs

Two symbols it cannot know on its own:

| | |
|---|---|
| `__m2ndp_spad_size` | size of the global area, from the linker script; a core's region has to hold it |
| `__m2ndp_params_offset` | where the task's parameters sit, from the compiler |

The second is how a task's parameters get to a kernel. They are one of its
scratchpad globals, so the compiler picks the offset, and the launcher writes
the block at `base + __m2ndp_params_offset` before running a kernel there.
Which global that is cannot be read off a name -- every one is
`memory_blob_<hash>` -- so the task marks it:

```llvm
call void @__m2ndp_declare_params(ptr addrspace(3) @memory_blob_...)
```

`RISCVM2ndpLowerScratchpad` reads that, exports the offset, and deletes the
call.

`scripts/m2ndp.lds` also declares the scratchpad as a 128 KiB memory region,
so a task that asks for more fails at link time rather than overlapping
something at run time:

```
lld: error: section '.spad' will not fit in region 'spad': overflowed by 1024 bytes
```

### Decisions the backend has to make

**1. Placement — done.** addrspace(3) globals are laid out by the compiler
and reach `.spad`; see above.

**2. Addressing — done, and not the way this document first recorded it.**
It said the scratchpad base was the same on every core, so a single
link-time address would do and no base register was needed. That followed
from an earlier reading of the architecture. The scratchpad is per task and
its base arrives in a register, so accesses are base-relative after all —
which the note under point 3 had already flagged as the consequence if the
assumption moved.

**3. Instance scope — one instance per core, one launch group per core.**
The scratchpad belongs to the NDP core and is shared by every µthread on it
(Table 1 of the M²NDP paper). Concurrent programs on one core are out of
scope by assumption, so "per core" and "per launch group" cannot diverge.

**4. Lifetime — contents survive kernel launches within a task.**
`histogram` already relies on this. It splits INIT/BODY/FINAL into three
kernels over one shared scratchpad global, and would compute nothing if a
launch reset the buffer. The backend must not treat a kernel boundary as
the end of the buffer's live range.

## Kernel arguments

A kernel is launched, not called, and takes no arguments. What it works on is
the task's, and the launcher writes the task's parameter block into the
argument area of every core's scratchpad before running a kernel there — so a
kernel reads its buffers out of that block, each one a load at a small offset
from the base.

```asm
vector_add:
  ld a1, 0(a0)       # p.a         a0 = scratchpad base
  ld a2, 8(a0)       # p.b
  ld a0, 16(a0)      # p.c
  ...
  ret
```

One instruction each, and no address materialized — that is what the
arguments-first layout buys. The block is one pointer per buffer and nothing
else, so it is as wide as the buffers a task names; the launcher is told that
width rather than agreeing on one, and the length and direction of each buffer
stay on the host, where the launch reads them. The loads are marked invariant,
since a kernel's parameters do not change while it runs.

The block is one of the task's scratchpad globals, so its address is a constant
offset from the base like any other. A kernel that declares an argument is
rejected: nothing would have written it.

**Vectors keep the ordinary register assignment.** The argument area holds
what the launcher writes — scalars and pointers — and a vector is something
a kernel produces rather than something it is handed. A scalable vector
could not be placed there at all, its size not being known until run time.

### No calls, enforced

The ABI has no calls, and that is now checked rather than merely observed.
The ones worth catching are the ones nobody wrote — the compiler emits
`memcpy` for a copy too large to expand, `__atomic_*` for an operation the
hardware lacks, soft-float helpers for arithmetic it cannot do:

```
error: in function big_copy: M2NDP has no calls, but this requires 'memcpy'
error: in function vec_atomic: M2NDP has no calls, but this requires '__atomic_load'
```

The callee is named because for a compiler-emitted call it is the only thing
that explains why a kernel with no calls in it suddenly has one.

An error, unlike the spill warning: a spill is expensive, but calling a
function that is not on the device cannot work at all.

This is the check the `softmax` and `layerNorm` ports will run into first.
STATUS.md records that whether `exp`/`sqrt` route to the Sleef RVV library is
unverified — if they do, it is as a call, and this will say so.

### No callee-saved registers, and a warning when it spills

A kernel is launched, not called. Nothing resumes after it expecting its
registers intact, so there is nothing to preserve: the callee-saved set is
empty and the whole register file is available at no cost.

That is not a small saving. On a kernel with enough live values to reach
into the `s` registers:

| | standard ABI | M2NDP |
|---|---|---|
| frame | 112 bytes | none |
| save / reload pairs | 13 | 0 |
| instructions | 102 | 74 |

Those saves would also be DRAM accesses, since the stack lives there.

With nothing left to preserve, a frame can only mean the register allocator
ran out and started spilling — to DRAM, not to the scratchpad. A kernel can
fall off that cliff silently: it still compiles and still computes the right
answer, only slowly. So emitting a frame warns:

```
warning: M2NDP kernel spills to memory (176-byte frame); spills go to DRAM
```

A warning rather than an error, because spilling is expensive, not wrong.
None of the six benchmarks trip it.

### How a kernel is recognised

It is not: **every function in an M2NDP module is a kernel.** The ABI has no
calls, so there is nothing else a function could be, and the extension alone
decides the argument convention. No marker, no separate calling convention
ID, no list passed to the compiler.

What made this look untrue was `main` and its closures sharing the module.
Those are Mojo scaffolding for building an executable and have no place in a
device binary; the benchmarks no longer define `main`, and the modules now
contain only kernels. That also removed the `KGEN_CompilerRT_*` calls that
came with them.

The one way a non-kernel could appear is a helper the frontend did not
inline. That is already a violation — it would need a call — so the same
diagnostic that enforces the call-free ABI catches it.

### Why not `stack_allocation`

`std.memory.stack_allocation[N, T, address_space=SHARED]()` looks like the
obvious spelling and **silently produces wrong code on this target.** Its
promotion to an addrspace(3) global is gated on `is_gpu()`; on a RISC-V
triple it falls through to a plain `alloca`, the address space is dropped
during codegen, and the buffer ends up on the stack — private to each
µthread, so nothing is shared and cross-µthread reductions read garbage.
Observed: the frame grows by the buffer size (`addi sp, sp, -336`) and no
scratchpad symbol appears in the assembly at all.

`src/m2ndp.mojo` therefore open-codes the same `pop.global_alloc` operation
that `stack_allocation` uses on GPU targets, bypassing the vendor check. Its
signature deliberately matches `std._plugin`'s `stack_allocation_fn` hook so
the body can move into a plugin overlay once a toolchain ships both a RISC-V
backend and the plugin selector.

## What a module has to satisfy

A kernel and controller code are not alike, and every way of confusing them
compiles and runs — a kernel reads a register a spawn would have set, a
caller's values do not survive, the launcher jumps to an integer. The failure
is a wrong answer rather than a crash, which is why the rules are checked
rather than written down. `RISCVM2ndpVerify` holds them, and
`xm2ndp-module-rules.ll` shows each one being broken:

| Rule | Because |
|---|---|
| a kernel takes no arguments | its parameters are in the scratchpad; an argument would be read from somewhere nothing wrote |
| a kernel is launched, never called | it preserves nothing, and is entered without what a spawn sets |
| a launch is given a kernel | anything else reaches the launcher as an address to jump to |
| a launch symbol is called with the signature it was declared with | otherwise it is not a call to that symbol, and nothing it names is a kernel |
| the scratchpad is a kernel's | controller code has none, and the register a kernel finds its base in holds its own first argument |

One rule is not in the pass and cannot be: **a kernel may make no calls.** Most
of the calls worth catching do not exist in the IR — `memcpy` for a large copy,
`__atomic_*` for an operation the hardware lacks, the soft-float helpers — so
that check lives in `LowerCall`, where the backend first synthesises them.

The frontend states none of this. `external_call` checks nothing about the
function whose address it passes on, and a kernel is an ordinary static method
that anything may call.

**What the pass cannot see is a call the frontend already removed.** A small
kernel called directly from `device_main` is inlined before codegen, so the
call is gone; what catches it then is the scratchpad rule, since the inlined
body reads a base the controller does not have. A kernel that touches no
scratchpad and is inlined this way runs once on the controller instead of once
per microthread, and nothing says so.

## Synchronization

There is none to map: the library has no barrier. µthreads are created and
retired by hardware FGMT, so there is no well-defined set to synchronize.
Two mechanisms take its place.

**Atomics**, for combining within a kernel. These lower to LLVM `atomicrmw`
rather than to an M²NDP symbol, so the backend sees a standard instruction:

```llvm
%40 = atomicrmw fadd ptr %39, float %38 monotonic, align 4
%9  = atomicrmw add ptr addrspace(3) %8, i32 1 monotonic, align 4
```

`monotonic` is `Ordering.RELAXED` — accumulation does not need `seq_cst`,
and the weaker ordering leaves fewer fences to emit.

RISC-V's `+a` has integer AMOs but no floating-point atomic add, so
`atomicrmw fadd` expands to an LR/SC retry loop:

```asm
.LBB0_5:
	lr.w	a2, (s0)
	bne	a2, a1, .LBB0_7
	sc.w	a3, a0, (s0)
	bnez	a3, .LBB0_5
```

Contended rows in SpMV pay for that. If M²NDP has a native FP atomic add,
this is the pattern to match.

There is no vector atomic; see "Operations with no spelling at this level".

**Kernel boundaries**, for ordering between kernels. Launches from
`device_main` are synchronous, so anything that would need `__syncthreads()`
on a GPU is split into two kernels here.

## Operations with no spelling at this level

Two operations in the reference kernels have no usable spelling here.
Neither can be fixed with a library wrapper — each needs a dedicated
intrinsic, expressed as an `external_call` the way the ID symbols are.

"No usable spelling" rather than "rejected": the mask-to-bitmap case really
is missing, but the vector atomic is subtler than that and is the one worth
reading carefully.

**Vector atomic.** `histogram`'s reference kernel tallies 16 samples with a
single `vamoaddei32.v` (indexed vector atomic); ours emits 16 scalar
atomics. This is the core operation of that benchmark.

On the Mojo side `pop.atomic.rmw` rejects a vector operand:

```
error: 'pop.atomic.rmw' op operand #0 must be pointer to whose type is an
arithmetic dtype, but got '!kgen.pointer<...SIMD<f32, 4>>'
```

LLVM IR is the more interesting half, and the reason it blocks is not the
one you would guess. `atomicrmw` is *not* scalar-only — LLVM 23 takes a
fixed vector under an `elementwise` flag, and the RISC-V backend compiles
it:

```llvm
%old = atomicrmw elementwise add ptr %p, <4 x i32> %v monotonic, align 4
```

Two things make it the wrong tool anyway.

- **It is contiguous, not indexed.** One pointer and one vector value: every
  lane goes to the same buffer at consecutive offsets. `vamoaddei32.v`
  sends each lane to its own address, which is the whole point when the
  lanes are histogram bin indices. Nothing in LLVM IR expresses an atomic
  scatter — `llvm.masked.scatter` carries no atomicity and there is no
  atomic VP intrinsic.
- **What it lowers to is not an atomic instruction.** On RISC-V it becomes
  `__atomic_load` followed by a `__atomic_compare_exchange` retry loop over
  the whole 16-byte vector — correct, but two libcalls and a runtime
  dependency.

The hardware side is empty too: RVV's vector AMOs were the draft `Zvamo`
extension and were dropped before RVV 1.0 was ratified. There is no trace
of `Zvamo` or `vamo*` anywhere in LLVM 23, and no vendor extension supplies
them either — every ratified RISC-V atomic extension (`A`, `Zaamo`,
`Zalrsc`, `Zabha`, `Zacas`) is scalar.

So an indexed vector atomic has no standard spelling at either layer, and
defining one as a vendor intrinsic under `HasVendorXM2ndp` is the intended
path rather than a workaround.

#### What exists so far

The whole indexed-AMO set the RVV 0.10 draft defined -- swap, add, xor, and,
or, min, max, minu, maxu -- at index element widths 8, 16, 32 and 64:

```asm
m2ndp.vamoaddei32.v v8, (a0), v12, v8        # encoding: [0x2f,0x64,0xc5,0x06]
m2ndp.vamoaddei32.v v8, (a0), v12, v8, v0.t  # masked
```

`vs2` carries per-lane byte offsets, `rs1` the base, and `vd` is both the
operand and where the previous values come back. The index element width is
the only type information in the encoding; the data SEW comes from `vtype`,
exactly as it does for the indexed loads and stores.

Plus floating-point forms, which the draft never had: `vfamoadd`,
`vfamoswap`, `vfamomin`, `vfamomax`. Only the operations that mean anything
for floats, so no `xor`/`and`/`or` and no signed/unsigned split. These matter
because RISC-V has **no** floating-point atomic add anywhere, scalar or
vector -- so an `atomicrmw fadd` becomes an LR/SC retry loop without them.

52 instructions in total. `llvm.riscv.m2ndp.*` intrinsics select into them,
one intrinsic to one instruction with a `vsetvli` in front.

**The encoding and the operand shape are provisional.** They follow
`vamoaddei32.v` as RVV 0.10 defined it, that being the instruction the
reference kernels were written against; the draft was dropped before 1.0, so
nothing standard occupies those bits and nothing standard blesses them. The
floating-point `funct5` values are ours outright, picked from what the
integer operations leave free.

### Scalar floating-point atomics

RISC-V has no floating-point AMO at all — the A extension is integer-only —
so `atomicrmw fadd` becomes a cmpxchg loop. `famoadd`, `famoswap`, `famomin`
and `famomax` at `.h`, `.w` and `.d`, with the usual `.aq`/`.rl`/`.aqrl`
forms, and `atomicrmw` selects into them directly.

An accumulation into shared memory, one instruction against a retry loop:

```asm
.LBB0_7:                                  ; without +xm2ndp
    lr.w  a2, (s0)
    bne   a2, a1, .LBB0_8
    sc.w  a3, a0, (s0)
    bnez  a3, .LBB0_6
```
```asm
    m2ndp.famoadd.w fa5, fs0, (s0)        ; with it
```

`fsub` is deliberately not covered — it is not one of the four operations, so
it still expands. Neither is `xchg`: `ATOMIC_SWAP`'s node profile is
integer-only and an `atomicrmw xchg` on a float reaches the DAG bitcast to an
integer, so there is nothing floating-point left to match. `famoswap` exists
for the assembler and an intrinsic could reach it later.

The floating-point `funct5` values are shared between the vector and scalar
forms of an operation. They had to be chosen to be free in the scalar AMO
space as well: `0b00010` and `0b00011` would have been the obvious
neighbours of add and swap, but they are `lr` and `sc`.

### Reaching it from Mojo

The frontend cannot emit `llvm.riscv.m2ndp.*` -- Mojo's own LLVM has never
heard of it. So these follow the same contract as the µthread ID symbols: an
external call, rewritten by a backend pass.

```mojo
_ = atomic_add_indexed(Histogram.bins, chunk * 4, SIMD[DType.int32, 16](1))
```
```llvm
%8 = call <16 x i32> @__m2ndp_vamoadd_i32(ptr addrspace(3) @memory_blob_...,
                                          <16 x i32> %7, <16 x i32> splat (i32 1))
```

The symbol carries the element type (`_i32`, `_i64`, `_f32`, `_f64`) because
the frontend has no way to overload on vector type; the lane count is left to
the argument types.

`RISCVM2ndpLowerExternalOps` turns that into the intrinsic, widening the
fixed vectors into a scalable container and setting `vl` to the lane count.
Unlike scratchpad placement, it is gated on the **function's**
`target-features` rather than on `-mattr`: this happens inside a function, so
the per-function subtarget is available.

`histogram`'s body is now what the reference kernel is:

```asm
vle32.v v12, (a0)                          ; load 16 samples
vsll.vi v12, v12, 2                        ; sample -> byte offset
m2ndp.vamoaddei32.v v8, (a1), v12, v8      ; 16 bins, one instruction
```

Sixteen scalar `amoadd`s before.

**Mask register to bitmap.** `imdb_lt_int64`'s reference finishes with
`vmv.x.s` + `sb`, because an RVV mask register already holds one bit per
lane. `SIMD[bool, W]` has no conversion to an integer bitmask and `Int()`
only instantiates at width 1, so the lanes must be tested and OR'd back one
at a time. The compiler still produces the mask register, then spends ~15
and/or instructions rebuilding the bit pattern it already had.

## Recovering the mapped address

Benchmarks index ordinary parameters:

```llvm
%4 = call i32 @__m2ndp_global_uthread_id()
%6 = mul i64 %5, 8
%7 = getelementptr inbounds i32, ptr %0, i64 %6
```

The hardware already handed the µthread the address it was mapped to
(`ADDR`, in `a2`) and its byte offset within the range (`OFFSET`, in `a1`),
and kernel arguments arrive through the scratchpad. `id * W` scaled by the
element size is `OFFSET`, so `base + id*W` is `base + OFFSET` for every array,
and for the array the range was taken over it is `ADDR` outright. Rewriting
the index into that form is where the Arachne paper's static instruction
reduction comes from.

This is deliberately not surfaced in the source: the mapping is a calling
convention, and putting it in benchmark code would bake the convention into
every kernel and break the rule that `benchmarks/` survives the backend
switchover unchanged. Recovering the hardware form is the compiler's job.

`RISCVM2ndpMapAddress` does it, after `RISCVM2ndpLowerExternalOps` so the id
is already an intrinsic. It matches a GEP whose index is the µthread id
scaled to a **byte stride equal to the packet** — anything else is not the
mapping — and reads `OFFSET`/`ADDR` from two intrinsics,
`llvm.riscv.m2ndp.offset` and `llvm.riscv.m2ndp.addr`, that lower to the
live-in registers.

Two things bound it:

- **Only parallel kernels.** In a serial launch `a1` is a slot index rather
  than a byte offset, so an address folded into it would be wrong.
  `m2ndpLaunchKind` tells the two apart by the launch symbol.
- **The range parameter is named, not guessed.** For `base + id*W` to be
  `ADDR`, `base` has to be the pointer the range was taken over. Which
  parameter that is is a runtime fact — the host chooses it with
  `PooledRange.over` — so it cannot be read off the compiled-once IR. The
  device code is compiled at launch, though, where the host knows both the
  range's base and the parameter block: it scans the block for the pointer
  equal to that base and passes its byte offset to the backend. With none
  named, only the offset rewrite runs.

### The switch

`M2NDP_MAP_ADDRESS` selects how far a launch takes this, defaulting to the
full recovery:

```
M2NDP_MAP_ADDRESS=addr     mapped address where it fits (default)
M2NDP_MAP_ADDRESS=offset   base + mapped offset only, safe for any array
M2NDP_MAP_ADDRESS=off      leave indices as written
```

The host turns that into the backend's `-m2ndp-map-address`, and — except in
`off` — passes `-m2ndp-packet` from the simulator config and, in `addr`, the
`-m2ndp-range-param` byte offset it found. `off` is what the code compiles to
without the pass at all, so it is the baseline the other two are measured
against.

## Compile target

Current settings (`m2ndp_target()` in `src/m2ndp.mojo`):

```
triple          = "riscv64-unknown-elf"
arch            = "generic-rv64"
features        = "+m,+a,+f,+d,+v,+zvl128b,+xm2ndp"
data_layout     = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
index_bit_width = 64
simd_bit_width  = 128
```

Replace `arch`/`features`/`data_layout` once the real M²NDP architecture is
settled. `+v` in `features` enables RVV and `+zvl128b` sets the minimum
vector register length.

### `+xm2ndp`

`FeatureVendorXM2ndp` in our LLVM fork, following the `FeatureVendorXTHead*`
pattern in `RISCVFeatures.td`. It carries no instructions yet; it exists so
the name parses and later work has a predicate to gate on.

Two toolchains see this string and they do not agree, which is worth being
precise about:

- **Mojo's LLVM does not know it** and says so on every build — `'+xm2ndp'
  is not a recognized feature for this target (ignoring feature)`. Expected.
  It drops the feature from its own subtarget but copies the string into the
  `target-features` function attribute unchanged.
- **Our llc does know it.** It builds the per-function subtarget from that
  attribute, so the extension is live in codegen with no `-mattr` on the
  command line. `+xbogusfeat` in the same position warns; `+xm2ndp` does
  not — that difference is what shows the name is really being recognised.

`verify.sh` checks that all six modules carry the marker, so a Mojo upgrade
that stopped passing unknown features through would be caught rather than
silently producing plain RISC-V.

One thing it does *not* do: the `.attribute 5` ISA string in the assembly
still reads `rv64i2p1`, because that directive comes from the command-line
subtarget rather than from function attributes. That is pre-existing LLVM
behaviour and not specific to this extension — `+m`/`+a`/`+v` are missing
from it too. Pass `-mattr` to llc if the ISA string matters.

## Switching over once the backend exists

`src/m2ndp.mojo` is the single file where `external_call` becomes a real
intrinsic. For example:

```mojo
# now (no backend)
def local_uthread_id() -> Int:
    return Int(external_call["__m2ndp_local_uthread_id", Int32]())

# once the backend is ready
def local_uthread_id() -> Int:
    return Int(llvm_intrinsic["llvm.m2ndp.local.uthread.id", Int32]())
```

Benchmark code (`benchmarks/*.mojo`) does not change.

## Inspecting the artifacts

```bash
./scripts/build.sh
grep -h "declare.*__m2ndp" out/*.ll | sort -u   # symbol list
grep -c "addrspace(3)" out/histogram.ll          # scratchpad usage (the only one)
```
