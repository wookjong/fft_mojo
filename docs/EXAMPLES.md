# Codegen walkthrough

What each benchmark compiles to. These are ports of
[M2NDP-public](https://github.com/PSAL-POSTECH/M2NDP-public)'s hand-written
kernels, so each can be read against the assembly it came from.

Regenerate with `./scripts/build.sh`. All IR below is verbatim from `out/`.

Contents:
1. [vector_add](#1-vector_add--indexing-and-rvv) — indexing, RVV
2. [spmv](#2-spmv--indirect-access-atomic-combine) — indirect access, atomic combine
3. [histogram](#3-histogram--scratchpad-across-three-phases) — scratchpad across three phases
4. [What the backend has to handle](#4-what-the-backend-has-to-handle)

---

## 1. vector_add — indexing and RVV

```mojo
comptime W = 8

struct VectorAdd(NDPTask):
    @staticmethod
    def body(a: ..., b: ..., c: ...):
        var i = global_uthread_id() * W
        c.store(i, a.load[width=W](i) + b.load[width=W](i))
```

```llvm
%4  = call i32 @__m2ndp_global_uthread_id()
%6  = mul i64 %5, 8
%7  = getelementptr inbounds i32, ptr %0, i64 %6
%8  = load <8 x i32>, ptr %7, align 4
%10 = load <8 x i32>, ptr %9, align 4
%11 = add <8 x i32> %8, %10
      store <8 x i32> %11, ptr %12, align 4
```

`SIMD[int32, 8]` becomes a native vector type and the RISC-V backend selects
RVV for it — `vsetvli` / `vle32.v` / `vadd.vv` / `vse32.v`, with the ISA
recorded in `.attribute 5, "rv64i..._v1p0_..._zvl128b1p0..."`. This is the
load-bearing result of the PoC: the RVV backend inside the Mojo compiler is
real and reachable purely through the target attribute.

### What is different from the reference kernel

The hand-written version does no index arithmetic at all:

```asm
add x1, x1, OFFSET       ; other arrays: their base + this µthread's offset
vle32.v v1, (ADDR)       ; this µthread's own chunk
```

A µthread is handed the address it was mapped to (`ADDR`) and its byte
offset within the pool (`OFFSET`); kernel arguments arrive through the
scratchpad. Our version instead takes ordinary parameters and computes
`base[id * W]`, which costs a `mul` and a GEP per array.

That is deliberate. The mapping is a calling convention, and putting it in
the source (`kernel_arg(0)`, raw byte offsets) would bake the convention
into every benchmark and break the rule that `benchmarks/` survives the
backend switchover unchanged. Recovering `ADDR`/`OFFSET` from
`base[global_uthread_id() * W]` is the compiler's job — it is also where the
paper's 22.2% static instruction reduction comes from.

---

## 2. spmv — indirect access, atomic combine

```mojo
var row = group_id()
var k = Int(row_ptr[row]) + local_uthread_id()
var acc = Float32(0)
while k < end:
    acc += values[k] * x[Int(col_idx[k])]
    k += group_size()
_ = atomic_add(y + row, acc)
```

One group per row; its µthreads take a strided slice of the nonzeros.

### 2.1 Indirect access needs no special construct

`x[col_idx[k]]` is a plain dependent load chain — load an index, extend it,
use it to index `x`:

```llvm
%28 = load i32, ptr %27, align 4                        ; col_idx[k]      <-- load 1
%29 = sext i32 %28 to i64
%30 = getelementptr inbounds float, ptr %2, i64 %29     ; &x[col_idx[k]]
%32 = load float, ptr %30, align 4                      ; x[col_idx[k]]   <-- load 2
%33 = fmul contract float %31, %32
%34 = fadd contract float %20, %33
```

If M²NDP wants gather semantics or a prefetch hint on the second load, that
is a backend pattern-match on this shape. The `fmul`/`fadd contract` pair
also fuses into a single `fmadd.s` in the assembly.

Note `__m2ndp_group_size` is called **inside** the loop: it is an opaque
external call, so LLVM cannot prove it loop-invariant and will not hoist it.
Marking the eventual intrinsics `readnone`/`speculatable` fixes this. It is
the concrete cost of the external-symbol approach.

### 2.2 Combining without a barrier

```llvm
%40 = atomicrmw fadd ptr %39, float %38 monotonic, align 4
```

M²NDP has no barrier — µthreads are created and retired by hardware FGMT, so
there is no well-defined set to synchronize. Where a GPU kernel would
`__syncthreads()` and tree-reduce through shared memory, µthreads here
combine with an atomic. `monotonic` is `Ordering.RELAXED`: accumulation does
not need the stdlib's default `seq_cst`, and a weaker ordering leaves the
backend fewer fences to emit.

Worth flagging for the backend: RISC-V's `+a` has integer AMOs but no
floating-point atomic add, so LLVM expands this into an LR/SC retry loop:

```asm
.LBB0_5:
	lr.w	a2, (s0)
	bne	a2, a1, .LBB0_7
	sc.w	a3, a0, (s0)
	bnez	a3, .LBB0_5
```

Contended rows will pay for that. If M²NDP has a native FP atomic add, this
is the pattern to match.

---

## 3. histogram — scratchpad across three phases

The reference kernel has three phases sharing one per-core bin array:

```asm
INITIALIZER:  vse32.v v1, (x4)                     ; zero the bins
KERNELBODY:   vamoaddei32.v x0, (x1), v{u}, v30    ; bins[sample] += 1
FINALIZER:    vamoaddei32.v x0, (x6), v1, v2       ; flush bins to the output
```

### 3.1 One scratchpad, three kernels

Declaring the scratchpad at struct level is what keeps the phases on the
same storage:

```mojo
struct Histogram:
    comptime bins = scratchpad[BINS, Int32, name="hist_bins"]()
```

```llvm
@memory_blob_de5f15ab6daf7941 = internal addrspace(3) global [1024 x i8] zeroinitializer, align 4
```

Exactly one addrspace(3) global, and all three functions index off it:

```llvm
; Histogram.initialize
%9  = getelementptr inbounds i32, ptr addrspace(3) @memory_blob_..., i64 %4
      store i32 0, ptr addrspace(3) %9, align 4

; Histogram.body
%8  = getelementptr inbounds i32, ptr addrspace(3) @memory_blob_..., i64 %7
%9  = atomicrmw add ptr addrspace(3) %8, i32 1 monotonic, align 4

; Histogram.finalize
%11 = getelementptr inbounds i32, ptr addrspace(3) @memory_blob_..., i64 %5
%12 = load i32, ptr addrspace(3) %11, align 4
%13 = atomicrmw add ptr %10, i32 %12 monotonic, align 4
```

A `comptime` member is evaluated once and the result shared, so there is one
allocation site. Calling `scratchpad()` separately inside each function
would instead mint a fresh symbol per call site
(`@buf`, `@buf_0`, `@buf_1` …) and the phases would silently use different
memory. That failure mode is quiet — it compiles and runs, it just computes
the wrong answer.

Two things the comptime path does *not* do, both checked: it does not leak
the comptime interpreter's heap address into the IR (contrast
`comptime g = stack_allocation[...]()`, which bakes in
`inttoptr (i64 2000000000)`), and comptime-time writes do not survive into
the initializer — the global stays `zeroinitializer`, so initialization has
to happen at runtime in the INITIALIZER phase, exactly as the reference does.

### 3.2 Phases are kernel boundaries

There is no barrier between the phases and none is needed: they are separate
kernel launches, and `device_main` launches synchronously. Kernel boundaries
are the synchronization mechanism M²NDP has.

### 3.3 The vector atomic does not survive

The reference tallies 16 samples with one `vamoaddei32.v` — an indexed
vector atomic. Ours becomes 16 scalar atomics:

```llvm
%9  = atomicrmw add ptr addrspace(3) %8, i32 1 monotonic, align 4
%12 = ... ; ×16
```

This is not a library limitation that a wrapper could fix. `pop.atomic.rmw`
rejects a vector operand outright —

```
error: 'pop.atomic.rmw' op operand #0 must be pointer to whose type is an
arithmetic dtype, but got '!kgen.pointer<...SIMD<f32, 4>>'
```

— and LLVM's `atomicrmw` likewise takes only scalars. Both layers agree, so
there is no spelling that reaches the instruction. Since this is the core
operation of the benchmark, closing the gap needs a dedicated M²NDP
intrinsic, expressed the way the indexing symbols are.

---

## 4. What the backend has to handle

| Construct | IR form | Backend action |
|---|---|---|
| `local_uthread_id()` | `call i32 @__m2ndp_local_uthread_id()` | → ID read (must be hoistable) |
| `global_uthread_id()` | `call i32 @__m2ndp_global_uthread_id()` | → ID read (must be hoistable) |
| `group_size()` | `call i32 @__m2ndp_group_size()` | → ID read (**must** be hoistable; see 2.1) |
| `group_id()` | `call i32 @__m2ndp_group_id()` | → ID read (must be hoistable) |
| `scratchpad[N, T, name=...]()` | `@<name> = internal addrspace(3) global` | → place in scratchpad memory, not `.bss`; decide absolute vs. per-group base register |
| `atomic_add()` | `atomicrmw ... monotonic` | → native atomic; note no FP AMO in `+a` (2.2) |
| indirect access | load → sext → GEP → load | nothing required; optionally pattern-match for gather/prefetch |
| SIMD | `<N x T>` ops | already handled — selects RVV |
| index arithmetic | `base[id * W]` | → recover `ADDR`/`OFFSET` (see 1) |
| *(missing)* vector atomic | — | needs its own intrinsic; unreachable from LLVM IR (3.3) |

The four ID symbols are the entire calling-convention surface. Everything
else is either a standard LLVM construct or an address-space annotation.
