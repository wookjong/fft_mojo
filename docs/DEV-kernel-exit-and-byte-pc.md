# Dev setup — explicit kernel-exit instruction + byte-address branch redesign

A guide to spin up a **separate worktree** for two coupled changes that continue
the compiled-binary → timing-simulation work:

1. **Backend (LLVM fork):** an explicit **kernel-termination instruction** (`exit`,
   in the `xm2ndp` extension) — like a GPU's `s_endpgm` (AMD) / `EXIT` (NVIDIA).
   The backend lowers a **kernel function's** return terminators to it; helper
   functions keep `ret`, so kernels can call functions later.
2. **Detour:** replace the index-based control flow (`loop_map`, `csr.pc` as an
   index, `ret`-dropped-at-load) with a **real byte-address PC**: fetch by byte
   address, advance by instruction size, branch to byte targets, terminate on the
   `exit` instruction (which maps to Detour's existing `EXIT` opcode).

These are one set: the branch redesign needs a real terminator, and `exit` is it.

---

## Base state this builds on

Everything below is already done and pushed; the new worktree starts from here.

| Repo | Branch | Commit | What it has |
|---|---|---|---|
| mojo (`mojo-m2ndp`) | `timing-detour` | `95bf64b` | `docs/TIMING.md`, submodule bumps, Dockerfile deps |
| detour (`M2NDP-Detour`) | `mc-frontend` | `0363181` | Full compiled-binary front end + timing-sim run (below) |
| LLVM (`llvm-project-m2ndp`) | `m2ndp/kinfo` | `c0d643367c` | `.m2ndp.kinfo` footprint emission |

**What already works on `mc-frontend`** (the vertical slice this extends):
compiled `vector_add.o` → `McKernel` decode → `populate` → `Uop` → `load_ndp_kernel_from_elf`
→ `mc_launch_sequence` → `register_ndp_kernel_from_elf` → `launch_kernel_direct`
(Arachne spawn, no CXL) → per-µthread `seed_launch_inputs` (a0=ScratchpadBase,
a5=GlobalUThreadID) → fetch/decode/issue timing pipeline → `C = A + B`, run by
`perf_runner/dev_launch.cc`. `.traceg` is gone. See the memory notes
`timing-detour-*` and `detour-timing-facts` for the full state.

The current control flow is index-based and only handles **straight-line leaf
kernels**: branches use `loop_map` (empty for compiled kernels → would crash),
`csr.pc` is an instruction index, and the trailing `ret` is dropped at load in
`mc_kernel_load.cc`. That is what this work replaces.

---

## 1. Create the worktree

From the main checkout (`/work/mojo-m2ndp`), branch off `timing-detour`:

```bash
cd /work/mojo-m2ndp
git worktree add .claude/worktrees/kernel-exit -b kernel-exit timing-detour
cd .claude/worktrees/kernel-exit
```

The submodules must sit on the branches above, and the LLVM submodule must be its
own git worktree so `llc` can be rebuilt privately (see the `backend-branch-submodule-worktree`
memory). The existing `mapped-address` worktree is the working template — mirror it:

```bash
# Point each submodule at its work branch (both are pushed).
git -C third_party/m2ndp-detour fetch origin && \
  git -C third_party/m2ndp-detour checkout mc-frontend
git -C third_party/llvm-project fetch origin && \
  git -C third_party/llvm-project checkout m2ndp/kinfo
```

If a submodule is not yet a worktree here, add it as one from its module gitdir
(the pattern the current setup uses:
`gitdir: /work/mojo-m2ndp/.git/modules/third_party/llvm-project/worktrees/...`).
The simplest reliable path is to **copy the working `mapped-address` worktree's
submodule setup**, or re-run the project's `scripts/setup.sh` and then check out
the two branches.

---

## 2. Build

**LLVM `llc`** (needed for the `exit` instruction) — worktree-local ninja dir:

```bash
# Configure once (see scripts/build-llvm.sh); then incremental:
ninja -C build/llvm llc
```
`build/llvm` is a real ninja build over this tree's `third_party/llvm-project`.
RVV+xm2ndp feature string: `+m,+a,+f,+d,+v,+zvl128b,+zfh,+zvfh,+xm2ndp`.

**Detour** — conan 1.56 (libstdc++11 / ABI=1) + cmake, per `detour-build-setup`:

```bash
cd third_party/m2ndp-detour
export CC=gcc CXX=g++
# scripts/build_timing.sh does: conan install .. ; cmake -DPERFORMANCE_BUILD=1 ...
bash scripts/build_timing.sh     # -> build/lib/libNDPSim_lib.so + build/bin/dev_launch
```

---

## 3. Task A — backend `exit` instruction (LLVM fork)

**Goal:** emit an explicit terminator for kernel functions instead of `ret`.

- **Define the instruction** in the `xm2ndp` extension (alongside the other
  vendor ops; look for the `Xm2ndp` `.td` and `RISCVM2ndpArgInfo.h`). Give it an
  encoding and a mnemonic the disassembler prints. Naming: **`exit`** — it lines
  up with NVIDIA's `EXIT`/PTX `exit` and Detour's existing `EXIT` opcode; AMD's
  precedent is `s_endpgm`.
- **Lower kernel returns to it.** A function is a *kernel* when it is launched
  through `__m2ndp_launch_*` (`M2ndpLaunchPrefix` in `RISCVM2ndpArgInfo.h`) — the
  same test that decides the calling convention. In a late pass (or during
  lowering), replace **every return terminator in a kernel function's CFG** with
  `exit`. Controller/helper functions keep `ret`, so a kernel can `call` a helper
  and the helper's `ret` returns normally.
- **Verify:** `llc -mattr=+...,+xm2ndp -filetype=obj kernel.ll -o k.o`, then
  `llvm-objdump -d` shows the kernel body ending in `exit` (not `ret`); helper
  functions still end in `ret`.

## 4. Task B — on-demand byte-address PC in Detour (detailed design)

**Goal:** the µthread fetches, renames, and executes one instruction at a time by
byte-address PC, like a real pipeline — no pre-built section, no `loop_map`, no
static vtype scan. Chosen over the "keep the batch-rendered vector + a byte→index
map" alternative because that keeps the very things this removes (pre-parse, batch
Convert, a second coordinate system).

### Foundation that already exists — use it
Detour has **two rename modes** (`SubCore::allocate_uthread`):
- *Static (default):* `RenamePush` → `Convert(section)` renames the whole vector up
  front (static vtype scan). **This mode is removed.**
- *Dynamic* (`m_config->dynamic_register_renaming()`): `initialize_uthread` +
  per-instruction `RegisterUnit::rename(uop, context)` at **decode**
  (`SubCore::decode_instruction`), using `context.uthread->decode_ctx` (SEW/LMUL,
  updated as vsets decode). **B builds on this mode.**

The only thing the dynamic mode still does wrong for B: it *pre-builds*
`uthr->insts` and the queue fetches `uthr->insts.at(csr.pc)` by index. B replaces
that with on-demand fetch.

### The pipeline, on demand
```
fetch  (instruction_queue): arch = section->uop(csr.pc)   # McKernel::inst(pc)->populate, cached per PC
                            push COPY to decode queue      # rename mutates per µthread
                            icache addr = base_addr + csr.pc   # pc is bytes now (drop *WORD_SIZE)
                            advance: non-branch -> csr.pc += arch.mc->size ; branch/jump/exit -> block
decode (decode_instruction): rename(uop, context)          # arch->physical, decode_ctx vtype  [EXISTS]
issue  (issue_instruction):  uop.Execute(context)          # semantics unchanged
                             branch taken -> csr.pc = byte target ; exit -> µthread done
```

### What the µthread holds
- `McKernel* mc` for its section (from `NdpKernel::mc_kernels`) + `csr.pc` (byte).
- **Remove `UThread::insts`** and `NdpKernel::kernel_body_insts` (the pre-parse).
- Per-µthread physical register map, built incrementally by `rename()` across the
  run — already how the dynamic mode works.

### Caching (three layers)
- **McInsn** per PC — `McKernel::inst(pc)` (exists).
- **arch Uop** (populate result) per PC — add `McKernel::uop(pc)` = decode + populate,
  cached; static, shared by all µthreads of the kernel.
- **physical Uop** (rename result) — per µthread, recomputed at decode each time a PC
  is seen (so a loop re-renames each iteration).

### Byte-address control flow (`ndp_instruction.cc`)
- `csr.pc` = byte address; init = the section's entry (`McKernel::start()`).
- `ExecuteBranch`/`IsBranch`: produce/return a **byte target** taken from the
  decoded branch operand; **delete `loop_map`** and the `imm/4` in the `JALR` case
  (`JALR` becomes `pc = ReadGPR(ra) + imm`, for real indirect jumps / function
  return).
- **`exit`** decodes to a McInsn `populate` maps to Detour's `EXIT`
  (`rvv_material.h` `{"exit", EXIT}`); on execute set the µthread done. Remove the
  `pc > max_pc` termination.
- **Stop dropping `ret`** at load — with `exit` as the terminator, a `ret` is a real
  function return.

### Removals
`kernel_body_insts`, `UThread::insts`, `Convert` + the static mode, `loop_map`, the
`is_section_return` drop, `max_pc`-based termination.

### Files
`sub_core.cc` (allocate/fetch/decode/issue, mode), `instruction_queue.{h,cc}`
(fetch by pc → McKernel), `register_unit.{h,cc}` (drop Convert/static),
`ndp_instruction.cc` (`ExecuteBranch`/`IsBranch`, EXIT-done), `mc_kernel*.{h,cc}`
(add `McKernel::uop(pc)`, stop pre-populating, keep terminator), `common.h`
(`UThread`, `NdpKernel`), `uthread_generator.cc` (`create_uthread`, drop insts,
lazy `issue_count_list`).

### Details to verify before/while coding
1. **Branch operand — absolute or offset?** Decode one branch via `McKernel::inst`
   and inspect the operand; `populate` computes the byte target accordingly
   (`target = inst.addr + offset` if relative).
2. **Loop renaming.** Confirm re-decoding the same PC (loop body) reuses/frees
   physicals correctly rather than leaking one per iteration — check how
   `DestRename`/`LookUp` + `FreeRegs` behave across a back-edge. (M2NDP's per-µthread
   offset mapping should make this a fixed mapping, but verify.)
3. **decode_ctx vs live CSR.** `rename()` uses decode-time SEW/LMUL; issue-time CSR
   is the truly live value. In-order decode makes them agree for loop-invariant
   vtype; verify for a kernel that changes vtype inside a loop.
4. **Config.** Turn on `dynamic_register_renaming`; the `config/performance/M2NDP`
   config is the one `dev_launch` uses.

**Detour already has the `EXIT` opcode and the dynamic per-instruction rename**, so
B is mostly rewiring fetch to `McKernel` + byte-PC and deleting the static/pre-parse
paths, not writing a new executor.

### Ordered milestones
The static mode is **not dead yet** — `dev_launch` runs it today
(`config/performance/M2NDP/m2ndp.config` has `dynamic_register_renaming=0`, and the
default is `false`). So it is retired *as the last step of B*, not before, or
`dev_launch` breaks. Order:

1. **Byte-PC control flow** (`ExecuteBranch`/`IsBranch` byte targets, `exit`→done,
   drop `loop_map` + `imm/4`, stop dropping `ret`). Works in the current static mode
   first — vector_add golden still passes.
2. **On-demand fetch** — `McKernel::uop(pc)`; the *dynamic* path fetches from it
   instead of `uthr->insts`; byte-PC advance by size. Turn the config to
   `dynamic_register_renaming=1`; validate vector_add + a branch/loop workload.
3. **Retire the static mode** — delete `Convert`, `RenamePush`, the static branch in
   `allocate_uthread`, and the now-unused `dynamic_register_renaming` switch itself
   (dynamic becomes the only path). Remove `kernel_body_insts` / `UThread::insts`.
4. **Golden** across workloads.

Retiring static (step 3) is a required deliverable, not optional cleanup — once
dynamic on-demand is the path, the static/Convert machinery is pure complexity and
comes out.

---

## 5. Validate

- `dev_launch` a compiled kernel end to end (from `config/performance/M2NDP`):
  ```bash
  cd third_party/m2ndp-detour/config/performance/M2NDP
  ../../build/bin/dev_launch /path/to/vector_add.o ./m2ndp.config
  ```
  Expect `ALL PASSED` — now with `exit` as the terminator and byte-address PC.
- **Add a control-flow workload** (a kernel with a loop / branch — e.g. one of the
  benchmarks that has one) and confirm it runs; this is what the old `loop_map`
  path could not do.
- Later, a kernel that **calls a helper** exercises `ret` (function return) vs
  `exit` (kernel end) and the cross-function fetch (McKernel must then cover the
  whole `.text`, not one symbol's range).

---

## Pointers / gotchas

- Launch ABI: `RISCVM2ndpArgInfo::getProvisional()` (a0=ScratchpadBase … a5=GlobalUThreadID).
  Kernel launch carries **no arguments**; task params live in the scratchpad.
- `populate` maps LLVM opcodes to Detour opcodes by normalized name
  (`mc_kernel_loader.cc`); `exit` must resolve to `EXIT`.
- Detour build is ABI=1 to match the project LLVM; conan profile `libstdc++11`.
- The LLVM fork has ~390 pre-existing lit failures — don't chase them.
- Push the submodule branch before bumping the mojo gitlink; branch per task.
