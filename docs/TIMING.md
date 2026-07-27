# Timing simulation with M²NDP-Detour

A workload compiled here is a real RISC-V/RVV+`xm2ndp` binary. Cycle-level
timing comes from [M²NDP-Detour](https://github.com/PSAL-POSTECH/M2NDP-Detour),
the microarchitecture simulator behind the MICRO'24 model, wired in as the
`third_party/m2ndp-detour` submodule.

Detour keeps its own functional execution and microarchitecture models. What
changes is the instruction front end: instead of parsing a hand-written
`.traceg` dialect into `NdpInstruction` up front, Detour decodes the compiled
binary **on demand** with LLVM and runs on that decode directly.

## The idea

A real CPU simulator fetches, decodes, and issues one instruction at a time.
Detour instead pre-parses the whole kernel into an `NdpInstruction` array before
executing. The one thing that forces this up-front pass is register accounting:
`count_required_regs` scans every instruction to size each µthread's register
footprint (occupancy) before dispatch.

The compiler already knows that footprint -- its register allocator assigned the
registers, `LMUL` groups included. So it emits the footprint per kernel, the
up-front scan disappears, and Detour decodes on demand like any pipeline
simulator. That removes the pre-parse, the `NdpInstruction` translation, and the
static `vtype`/`LMUL` reconstruction in one move.

```
kernel ELF ──► fetch(PC) ──► decode one McInsn (LLVM, cached by PC) ──► issue ──► execute
                                                                          │
compiler-emitted per-kernel {x,f,v} footprint ──► occupancy sizing        ▼
dynamic CSR (Detour executes vset) ──► LMUL for dependencies        shared mmap pool
```

## Why on-demand decode works here

- **Dependencies are dynamic.** Detour is a simulator, not a static analyzer.
  The scoreboard is checked at issue, where the µthread's CSR already holds the
  real `vtype` (Detour executed the preceding `vsetvli`). `GetIssueCount` already
  reads `context.csr->vtype_vlmul` on demand, so `LMUL`-aware register grouping
  needs no static analysis.
- **Occupancy is static, and the compiler owns it.** The per-kernel register
  footprint is the same for every µthread and is fixed at compile time. The
  compiler emits it; nothing needs to scan instructions to recover it.
- **Control flow is resolved.** The disassembler gives branch targets as numeric
  offsets, so on-demand fetch follows them without a label table.

## Compiler: per-kernel register footprint

The mojo LLVM fork emits, per kernel function, the register footprint the
allocator produced: the number of scalar (`x`), floating-point (`f`), and
vector (`v`) registers used, with vector registers counted by their `LMUL` group
(a `VRM4` value is four physical vector registers).

- **Source:** after register allocation, the used physical registers per class,
  read from the function's register usage. `LMUL` comes from the vector register
  class, which the allocator already assigned -- no `vtype` tracking.
- **Emission:** a `.m2ndp.kinfo` section holding one record per kernel:
  `{ function symbol, nx, nf, nv }`. The function symbol (a relocation) keys the
  record; Detour matches it to the kernel it is running.

This is small -- a few numbers per kernel -- and authoritative, since it is the
allocation the binary was built against.

## Detour: on-demand decode

- **Fetch → decode.** At each fetch, decode one instruction from the bytes at the
  current PC with the `xm2ndp`-aware LLVM `MCDisassembler`, into a `McInsn`
  (opcode, operand roles and register classes from `MCInstrDesc`, byte-PC). A
  decode cache keyed by PC decodes each address once and reuses it.
- **Execute and time on `McInsn`.** The executor and scoreboard read `McInsn`
  directly: definitions and uses come from `MCInstrDesc` (`getNumDefs`, register
  class), so there is no operand-slot convention to reproduce. `LMUL`-expanded
  register groups come from the dynamic CSR at issue.
- **Occupancy** is sized from the compiler footprint (`.m2ndp.kinfo`), replacing
  `count_required_regs`.
- **Per-µthread register offset** stays: each µthread's architectural registers
  map to its own physical slice, exactly as today, now applied to `McInsn`
  register numbers.

## Kept and dropped

| Kept | Dropped |
|---|---|
| Detour's functional execution semantics | Pre-parse into an `NdpInstruction` array |
| Timing / cache / DRAM / interconnect models | `mc_to_ndp` mapping and operand-slot swaps |
| Per-µthread register offset (rename), occupancy | `operand_type_map`-based operand placement |
| `mc_decoder` (now a single-instruction decode) | `count_required_regs` and static `vtype` scan |

## Memory

Memory is the shared CXL pool, `mmap`ed by both host and Detour at the same base
-- the near-data model Spike uses, where a pointer means the same thing on both
sides. The host stages the pool before the run and checks the answer against it
after; there are no `_input.data`/`_output.data` files. Detour's `detour` branch
is growing a `PooledMemory` that is exactly this, adopted once complete.

## Phases

1. **Compiler footprint** (LLVM fork): compute per-kernel `{nx, nf, nv}` after
   register allocation, emit `.m2ndp.kinfo`. Verify the section on a benchmark.
2. **Decoder**: single-instruction on-demand decode + PC-keyed cache; read
   `.m2ndp.kinfo`.
3. **Detour fetch**: replace array-index fetch with on-demand decode.
4. **Detour execute/timing**: read `McInsn` (def/use + dynamic CSR `LMUL`); size
   occupancy from the footprint.
5. Remove the pre-parse, `mc_to_ndp`, and `count_required_regs`.
6. End-to-end per benchmark, gated on the golden answer.

## Scope

A real Detour rework -- fetch and the instruction interface move to `McInsn` --
plus a focused compiler change (footprint emission). It removes more than it
adds: no pre-parse, no operand mapping, no static `vtype` reconstruction, and
Detour fetches and decodes like a pipeline simulator. Editing the backend uses
the LLVM submodule as its own worktree with a private `llc` build.

## Correctness

Detour executes functionally, so a run must match the Mojo golden answer, and
that match validates the decode and the register/dependency handling. The
compiler footprint is cross-checked against a reference count.

## Superseded

The `mc-frontend` work that maps decoded instructions onto `NdpInstruction`
(`mc_to_ndp`, the opcode/operand tables) is kept only as the reference decode
path; the on-demand architecture above replaces it. `mc_decoder` -- LLVM decode
into `McInsn` -- carries forward.
