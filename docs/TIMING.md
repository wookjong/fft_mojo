# Timing simulation with M²NDP-Detour

A workload compiled here is a real RISC-V/RVV+`xm2ndp` binary. Cycle-level
timing comes from [M²NDP-Detour](https://github.com/PSAL-POSTECH/M2NDP-Detour),
the microarchitecture simulator behind the MICRO'24 model, wired in as the
`third_party/m2ndp-detour` submodule.

Detour upstream reads a hand-written assembly dialect (`.traceg`) plus text
memory-image and launch files. This integration replaces that whole text
front-end: a real compiled kernel drives the timing model directly. It is a fork
of Detour's front-end, not an added mode — the text path is removed.

Work starts from the current stable point of Detour's `detour` branch. A
shared-pool memory (`PooledMemory`) is landing there separately; until that
version is complete, memory uses Detour's existing map, and the pool is adopted
when it is ready.

## Scope: a machine that runs a launched kernel

The first target is deliberately narrow. Detour is a machine that **executes one
launched kernel**: given a compiled kernel body and a launch descriptor, it runs
the µthreads against the shared pool and reports cycles. It does not run
`device_main`, does not intercept launch intrinsics, and does not sequence
multiple kernels — those are deferred (see the end).

The launch descriptor — kernel entry, `base`, `size`, `packet`, `stride`,
parameter block — is supplied by the host, reusing the argument contract the
Spike path already passes.

## What is reused and what is replaced

Detour is already a combined functional+timing simulator: it executes each
instruction functionally at the timed issue point, against shared memory, and
its microarchitecture model (caches, TLB, DRAM via Ramulator, interconnect via
BookSim, in-order sub-core pipeline, register-limited µthread occupancy)
produces the cycles. That engine stays. Only the front-end — how instructions,
memory, and launches enter it — changes.

| Layer | Disposition |
|---|---|
| Text parser (`m2ndp_parser` `.traceg` path, `parse_kernel_launch`) | **removed** |
| String opcode tables (`rvv_material.h` `opcode_type_map`/`operand_type_map`) | **removed** (enums kept) |
| Special-token registers (`ADDR`/`OFFSET`/`NDPID`/`UTHREADID`…) | **removed** (ABI seeding) |
| `_input.data` / `_output.data` text memory images | **removed** (shared mmap pool) |
| Shared-pool / scratchpad memory | **from Detour** once `PooledMemory` lands; existing map until then |
| `NdpInstruction` / `NdpKernel`, `Execute*`, timing core | **kept** |
| µthread enumeration, rename, register-limited occupancy | **kept** |
| Cache / TLB / DRAM / interconnect models | **kept** |

## Why detour's own engine, not Spike co-simulation

Producer-consumer and other value/timing-dependent behavior is modelled
correctly only when functional execution happens *at the timed issue point* —
Detour's engine already does this (issue-order memory visibility). Feeding a
pre-extracted trace would lose it; delegating execution to Spike in lock-step
would reproduce it, but at the cost of rebuilding one simulator on top of the
other.

An opcode census of every benchmark's real compiled `.text` (102 distinct
opcodes, decoded with the LLVM `xm2ndp` disassembler, zero decode failures)
shows Detour's executor already covers all but a handful. Within a kernel body
the gaps are the `frm` rounding-mode CSR and `vrgather`; `AUIPC` appears only in
the launch entry, which this scope does not run. With the gap that small,
Detour's own engine is the target and Spike stays out of the timing loop.

## Pipeline

```
kernel ELF ──► MCDisassembler decode ──► NdpKernel ──► Detour engine
   │              (structured, no text)   (pre-decoded    (execute at issue
launch args         from build/llvm        array +          + microarch timing)
(host) ─────────────────────────────────   byte-PC)
                                              │
shared mmap pool + scratchpad ◄───────────────┘
(host stages, Detour maps at the same base)
```

## Decode front-end

Decoding is a one-time pass that fills the pre-decoded instruction array the
executor already indexes by PC — the same shape the text parser produced, from
structured decode instead of text. It is not decode-on-fetch; the executor's
"read at PC" is unchanged.

The decoder uses the LLVM `MCDisassembler` from this project's toolchain
(`build/llvm`, i.e. `third_party/llvm-project`), which knows the `xm2ndp`
extension. Setup is standard: `llvm-config` supplies flags; link
`mcdisassembler riscvdisassembler riscvdesc riscvinfo mc object support`. No LLVM
source change is needed for decoding.

Per kernel it emits:

1. `vector<NdpInstruction>` for the kernel function and the device helpers it
   calls, each annotated with its real byte-PC (needed by instruction-cache
   timing, and by `AUIPC` if the scope later grows).
2. `map<byte_addr → array index>` so branch/jump/call targets — which the
   disassembler gives as resolved offsets — resolve to array indices. The old
   label model (`.LOOP`/`.SKIP`) disappears: targets are numeric.
3. Symbol metadata read from the ELF, not reconstructed (see below).

The `MCInst → NdpInstruction` mapping is a table keyed on the LLVM opcode
(e.g. `RISCV::VFMACC_VV → {Opcode::VFMACC, OperandType::VV}`), roughly 102
entries, plus register-number and operand-order mapping. `x0` maps to the zero
register and is never renamed.

## Metadata comes from the compiler, not reconstruction

The ELF carries what the text parser used to rebuild. The decoder reads it:

| Fact | Source |
|---|---|
| Kernel entry point and size | symbol table (`…::body()` FUNC symbol) |
| Device helpers called by the kernel | symbol table + call targets |
| Scratchpad / params layout | `__m2ndp_spad_size`, `__m2ndp_spad_globals`, `__m2ndp_params_offset` |
| ISA features for the decoder | `.attribute` string |

### Register footprint

Register-limited occupancy — how many µthreads run concurrently before the
physical register file is exhausted — depends on each kernel's register
footprint (`x`/`f`/`v` counts). This is a timing input, not cosmetic.

The compiler is the authoritative source. Detour's rename confirms it: each
µthread's architectural registers map to a private physical slice with a static
one-to-one mapping (`// Believe in compiler`), reserving exactly the
architectural count — Detour does no dynamic renaming to break false
dependencies. Anti-dependency avoidance — spending extra registers so writes do
not stall on earlier readers — is the compiler's doing and is already in the
architectural allocation. More registers means fewer false dependencies but
lower occupancy; modelling that trade-off needs the exact count.

Detour's reconstruction of the count (in `uthread_generator`) rests on two
assumptions that hand-written kernels satisfy but real compiled code breaks:
linear `vtype`/`LMUL` tracking (assumes vector-config placement is independent
of control flow), and destination-oriented counting (misses read-only ABI
inputs). For simple kernels the reconstruction is correct, so this scope uses it
as-is; emitting the footprint from the mojo backend, and feeding
`NdpKernel::kernel_body_{x,f,v}regs` directly, is a later robustness step for
kernels with branchy `vtype` usage.

## Kernel launch setup

Given the launch descriptor, Detour sets up:

1. **µthread enumeration and mapping** — `count = ⌈size/packet⌉`; for µthread
   `u`: `addr = base + u·packet`, `offset = u·packet`, `global_id = u`,
   `core = addr/stride mod cores`. Detour's existing interleave and
   `get_uthread_size` are reused.
2. **Per-µthread ABI register seeding** — the launch writes the µthread's
   initial architectural registers per the calling convention (`sim/launch.h`):
   `a0` scratchpad base, `a1` offset, `a2` mapped addr, `a3` ndp id, `a4` local
   uthread id, `a5` global uthread id, `a6` group id, `t0`/`pc` kernel entry.
   This replaces the special-token registers: compiled code reads `a2`, so `a2`
   must hold the mapped address.
3. **Scratchpad** — a per-core `Scratchpad` is created; the parameter block is
   copied from the pool into scratchpad at `__m2ndp_params_offset`, where the
   kernel reads its arguments.
4. **Occupancy and rename** — automatic: per-µthread register demand partitions
   the physical register file and gates dispatch.

Detour's section model (`INITIALIZER`/`KERNELBODY`/…) is dropped; a kernel is a
single body.

## Memory

Memory is the shared CXL pool, `mmap`ed by both host and Detour at the same
fixed base (`MAP_SHARED`) — the near-data model Spike uses, where a pointer
means the same thing on both sides. The host stages the pool before the run and
checks the answer against it after; there are no `_input.data`/`_output.data`
files. Detour's memory backend is polymorphic (`Context::memory_map`), so the
pool is a map behind that interface, with a per-core scratchpad alongside.

Detour's `detour` branch is growing a `PooledMemory` that is exactly this — an
`mmap` at a fixed base whose `Load`/`Store` dereference the address directly.
This adopts it once that version is complete; until then a small pool-backed map
serves the same role.

## Execute-coverage gaps

Within a kernel body, Detour's executor covers the real opcode set except:

- **`frm` rounding-mode CSR** (`fsrmi`/`fsrm`) — appears in floating-point
  kernels; Detour models only `vxrm`. Add an `frm` field, honor it in softfloat,
  or default round-to-nearest and let the golden check confirm.
- **`vrgather`** — one kernel uses it; implement in `ExecuteVector` on the
  pattern of the adjacent `VCOMPRESS`.

`AUIPC` is out of scope (launch-entry only). Operand-type variants that fall
through to `UnimplementedError` are the remaining scan. These gaps were surveyed
on this base commit.

## Correctness

Detour executes functionally, so its result is checkable: a run must match the
Mojo golden answer, and that match validates the timing demand (correct
addresses and control flow). The fork does not consume `.traceg`, so upstream
trace parity is not a reference — the golden answer is the anchor.

## Milestones

- **M0** — submodule on a stable `detour` commit that builds, an `mc-frontend`
  working branch, and `llvm-config` linked into the build.
- **M1** — `mc_decoder`: kernel ELF + entry symbol → `NdpKernel` (kernel and
  helpers, byte-PC, `byte→index` map).
- **M2** — `MCInst → NdpInstruction` mapping (opcode/operand/register), `x0`.
- **M3** — launch feeding (host args → launch), ABI register seeding, the shared
  pool, scratchpad parameter copy: `vector_add` runs standalone and matches its
  golden answer.
- **M4** — remove the text path; fill the execute gaps (`frm`, `vrgather`);
  operand-type scan.
- **M5** — broaden to every benchmark; invoke Detour from the host launch path
  in place of Spike; produce timing.

## Deferred

- `device_main` orchestration executed on Detour, launch-intrinsic interception,
  and multi-kernel sequencing (which subsumes `AUIPC` and host-argument/HTIF
  handling).
- Compiler-emitted register footprint, for kernels whose `vtype` usage defeats
  the reconstructed count.
- Adopting Detour's `PooledMemory` once its complete version lands upstream, in
  place of the interim pool-backed map.

## Risks

- **Operand mapping** — `MCInst` operand order versus Detour's `src[]`/`dest`
  convention, per instruction format; the golden check catches errors.
- **No trace parity** — the golden answer is the only validation; a few kernels
  may warrant manual cross-checks.
