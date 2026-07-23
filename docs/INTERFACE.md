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

### What is lost by the time it reaches the object file

This is the part that needs backend work. Today the global lowers to an
ordinary common symbol:

```asm
.type  memory_blob_de5f15ab6daf7941,@object
.local memory_blob_de5f15ab6daf7941
.comm  memory_blob_de5f15ab6daf7941,1024,4

auipc  s1, %pcrel_hi(memory_blob_de5f15ab6daf7941)
addi   s1, s1, %pcrel_lo(.Lpcrel_hi0)
```

`.comm` puts the buffer in `.bss` — ordinary memory. The addrspace(3)
annotation survives only as far as LLVM IR; nothing in the object file says
"scratchpad". Compare NVPTX, which emits `.shared .align 4 .b8 name[256]`.

### Decisions the backend has to make

1. **Placement.** Emit addrspace(3) globals into scratchpad memory rather
   than `.bss`.
2. **Addressing.** Accesses are currently PC-relative: the address is
   materialized with an `auipc`/`addi` pair against `%pcrel_hi`/`%pcrel_lo`
   (there is no `%hi`/`%lo` form in the output). PC-relative or not, the
   symbol resolves to one link-time address shared by every core. If the
   scratchpad base differs per core or per launch group, this cannot work
   and the accesses must become relative to a scratchpad base register.
3. **Instance scope.** One instance per NDP core, shared by every µthread on
   it (Table 1 of the M²NDP paper). Whether a buffer is also distinct per
   launch group is an architecture decision the IR does not express.
4. **Lifetime.** Whether the contents survive across kernel launches within a
   task. Nothing in the IR constrains this either way.

Points 2–4 cannot be expressed from the workload side; they are contract
terms that have to be agreed and then documented here.

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

Two operations in the reference kernels cannot be expressed from Mojo or
from LLVM IR. Both were confirmed by a verifier rejecting them, not
inferred, and neither can be fixed with a library wrapper — each needs a
dedicated intrinsic, expressed as an `external_call` the way the ID symbols
are.

**Vector atomic.** `histogram`'s reference kernel tallies 16 samples with a
single `vamoaddei32.v` (indexed vector atomic); ours emits 16 scalar
atomics. `pop.atomic.rmw` rejects a vector operand:

```
error: 'pop.atomic.rmw' op operand #0 must be pointer to whose type is an
arithmetic dtype, but got '!kgen.pointer<...SIMD<f32, 4>>'
```

and LLVM's `atomicrmw` likewise takes only scalars, so both layers agree.
This is the core operation of that benchmark.

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
features        = "+m,+a,+f,+d,+v,+zvl128b"
data_layout     = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
index_bit_width = 64
simd_bit_width  = 128
```

Replace `arch`/`features`/`data_layout` once the real M²NDP architecture is
settled. `+v` in `features` enables RVV and `+zvl128b` sets the minimum
vector register length.

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
