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
| `__m2ndp_local_uthread_id` | `declare i32 @__m2ndp_local_uthread_id()` | `i32` | index within the group, `[0, group_size)`; also the scratchpad slot |
| `__m2ndp_global_uthread_id` | `declare i32 @__m2ndp_global_uthread_id()` | `i32` | index across all cores; identifies the mapped data |
| `__m2ndp_group_size` | `declare i32 @__m2ndp_group_size()` | `i32` | µthreads sharing one scratchpad |
| `__m2ndp_group_id` | `declare i32 @__m2ndp_group_id()` | `i32` | which group this µthread belongs to |

The IDs are `i32` and get sign-extended to `i64` at every use site, since
the index type is 64-bit (`index_bit_width = 64`).

### Loop invariance

These are opaque external calls today, so LLVM cannot hoist them out of
loops — `__m2ndp_group_size` is re-called on every iteration of the SpMV
accumulation loop. When the backend replaces them with intrinsics, mark
them `readnone`/`speculatable` (or the intrinsic equivalent) so that stops
happening. See EXAMPLES.md §3.3 for the concrete code.

### Group index

The two µthread IDs follow Arachne's `GlobalUThreadID()` / `LocalUThreadID()`:
one across all cores, one within a core. Both come from the hardware — a
µthread is handed its identity in scalar registers when spawned.

"Group" means the set of µthreads sharing one scratchpad, i.e. those
resident on one NDP core. `local_uthread_id()` indexes into it,
`group_size()` is its size, `group_id()` says which one it is.

### Open: which registers carry the IDs

Because the values arrive in registers rather than being computed, the four
symbols should lower to reads of reserved registers — `getReservedRegs` in
`RISCVRegisterInfo.cpp` — rather than to new instructions. **Which four
registers is still undecided**, and that blocks the lowering: there is
nothing to read from until the ABI names them.

Nothing else waits on this. Registering the vendor feature so `-mattr=
+xm2ndp` parses does not need it, and neither does the scratchpad work.

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

%8 = getelementptr inbounds i32, ptr addrspace(3) @memory_blob_de5f15ab6daf7941, i64 %7
%9 = atomicrmw add ptr addrspace(3) %8, i32 1 monotonic, align 4
```

Two properties of that global the backend cannot assume away:

- **The `name=` argument does not reach the symbol.** It is spelled
  `memory_blob_<hash>`, so a scratchpad buffer cannot be identified by name
  in the IR or the object file. Anything that has to recognise these must
  key off the address space, not the symbol name.
- **The global is an untyped byte blob**, `[1024 x i8]`, not `[256 x i32]`.
  The element type appears only on the GEPs and the accesses.

`histogram` is the only one of the six benchmarks that allocates a
scratchpad — it is where all 37 `addrspace(3)` references live. `spmv`
combines through atomics on ordinary memory and has none.

Address space 3 follows the GPU shared-memory convention. If M²NDP wants a
different number, change it in `src/m2ndp.mojo`; ordinary memory stays in
address space 0 as a plain `ptr`.

### Placement: `.spad`

With `+xm2ndp` on, an addrspace(3) global is emitted into a `.spad` section
rather than into `.bss`:

```asm
.section .spad,"aw",@nobits
```

`.spad` is `SHT_NOBITS` with `SHF_ALLOC | SHF_WRITE` — on-chip memory that
is uninitialized at load and occupies nothing in the object file, the same
shape as `.bss`.

Implemented as a `SelectSectionForGlobal` override in
`RISCVELFTargetObjectFile`, which is the hook LLVM provides for exactly
this. Returning a section other than `getBSSSection()` also steers
`AsmPrinter` off its BSS-local path — otherwise the global would become
`.local`/`.comm` before the section ever mattered.

**The gate is `-mattr`, not the function attribute.** Globals are emitted
outside any function, so there is no per-function subtarget to consult and
the module-level one decides. `llc` therefore needs `-mattr=+xm2ndp` on the
command line; the `+xm2ndp` that `m2ndp_target()` puts in `target-features`
is not enough on its own. This is ordinary LLVM behaviour — function
attributes are per-function overrides, the module target comes from the
command line — and not something specific to this extension.

Without the extension nothing changes: the address space keeps whatever it
meant before, and the global still lowers to a common symbol.

### What this looked like before

```asm
.type  memory_blob_de5f15ab6daf7941,@object
.local memory_blob_de5f15ab6daf7941
.comm  memory_blob_de5f15ab6daf7941,1024,4

auipc  s1, %pcrel_hi(memory_blob_de5f15ab6daf7941)
addi   s1, s1, %pcrel_lo(.Lpcrel_hi0)
```

`.comm` put the buffer in `.bss` — ordinary memory. The addrspace(3)
annotation survived only as far as LLVM IR; nothing in the object file said
"scratchpad". This is still what a build without `+xm2ndp` produces, and
what the addressing example further down shows.

### Decisions the backend has to make

**1. Placement — done.** addrspace(3) globals reach a `.spad` section; see
above. What is *not* done is assigning offsets within a per-core window, so
several scratchpad globals in one module still each get their own symbol
rather than being packed into one buffer. That is the AMDGPU LDS model
(`AMDGPULowerModuleLDSPass.cpp`) and it is the next piece of this.

The three terms below are the architecture decisions the placement work
depends on, and they are now settled.

**2. Addressing — one address, identical on every core.** The scratchpad
base does not vary per core. So the form already in the output stands: the
symbol keeps a single link-time address and accesses stay PC-relative, an
`auipc`/`addi` pair against `%pcrel_hi`/`%pcrel_lo`. No scratchpad base
register is needed, and no relocation work beyond point 1 — what has to
change is where that address lands, not how it is computed.

**3. Instance scope — one instance per core, one launch group per core.**
The scratchpad belongs to the NDP core and is shared by every µthread on it
(Table 1 of the M²NDP paper). Concurrent programs on one core are out of
scope by assumption, so "per core" and "per launch group" cannot diverge:
the backend assigns exactly one offset per addrspace(3) global within the
per-core window. That is AMDGPU's LDS model — see
`AMDGPULowerModuleLDSPass.cpp`, not NVPTX.

If that assumption is ever relaxed, point 2 falls with it. A base that
differs per launch group defeats a fixed address even when it is uniform
across cores, and accesses would have to become base-register-relative
after all.

**4. Lifetime — contents survive kernel launches within a task.**
`histogram` already relies on this. It splits INIT/BODY/FINAL into three
kernels over one shared scratchpad global, and would compute nothing if a
launch reset the buffer. The backend must not treat a kernel boundary as
the end of the buffer's live range.

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

**Kernel boundaries**, for ordering between phases. Launches from
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

Encoding first, ISel second — the assembler has to accept an instruction
before its selection can be checked.

```asm
m2ndp.vamoaddei32.v v8, (a0), v12, v8        # encoding: [0x2f,0x64,0xc5,0x06]
m2ndp.vamoaddei32.v v8, (a0), v12, v8, v0.t  # masked
```

`vs2` carries per-lane byte offsets, `rs1` the base, and `vd` is both the
addend and where the previous values come back. Assembler, disassembler and
the `llvm.riscv.m2ndp.vamoadd` intrinsic are in place; **nothing selects the
intrinsic into the instruction yet.** That needs the RVV pseudo machinery so
VL and VTYPE get set up, and is the next piece.

Both the operand shape and the encoding are **provisional**. They are taken
from `vamoaddei32.v` as RVV 0.10 defined it, which is the instruction the
reference kernels were written against. Since that draft was dropped, no
standard claims those bits — and nothing blesses them either.

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
(`ADDR`) and its byte offset within the pool (`OFFSET`), and kernel
arguments arrive through the scratchpad. Turning `base[id * W]` back into
that form is a backend optimization, and it is where the Arachne paper's
22.2% static instruction reduction comes from.

This is deliberately not surfaced in the source: the mapping is a calling
convention, and putting it in benchmark code would bake the convention into
every kernel and break the rule that `benchmarks/` survives the backend
switchover unchanged.

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
