# Cooperative single-slot spill: assembly-level root cause and fix (2026-09-04)

Answers the task asking why cooperative FFT leaves spill where persistent
ones don't, using real assembly (not just source-level comparison), and
requiring a 2x2 ablation to separate the two candidate causes before
touching production codegen. Analysis commit: `14a220a` (repo HEAD at the
start of this session; the fix below is on top of it, uncommitted as of
this writing).

## 1. Repro

`N=64`, `radices=(4,4,4)`, `workers_per_fft=2`, `compute_lanes=4`,
`narrow_middle_stages=True` (make_fft_kernel.py's own defaults), single
leaf (`num_logical_blocks=1` / `total_ffts=1`, so `fft_slots_per_group=1` --
see `planning.fft_plan_cooperative.make_cooperative_leaf_plan`, `base.
max_uthread` capped by `total_ffts` itself here):

```
./run_fft_test.sh 64 --cooperative-workers 2 --compute-lanes 4
```

Real hardware (M2NDP-Detour via the `ghcr.io/psal-postech/mojo-m2ndp:main`
image): PASS, but

```
warning: <unknown>:0:0: in function fft_fp32_N64_generated::FFTRecLeaf0::
stage_1() void (): M2NDP kernel spills to memory (16-byte frame)
```

matching `docs/persistent_vs_cooperative_findings.md`'s own item 4 exactly.
The equivalent persistent leaf (same N/radices/compute_lanes,
`make_persistent_leaf_plan`+`generate_persistent_fft_kernel`,
`num_logical_blocks=1`) is correct and **has no stack frame at all** in any
stage function.

## 2. Generated-code comparison

Both share the exact same per-batch load/twiddle/butterfly/store emitter
(`fft_codegen._emit_stage_batches`/`_emit_batch`) -- confirmed by reading
`fft_cooperative_codegen.py` and `fft_persistent_codegen.py`, not assumed.
The only structural differences are in each one's own prelude/dispatch:

| | Cooperative (`_emit_cooperative_prelude`) | Persistent (`emit_stage_phase`) |
|---|---|---|
| Scratchpad base | `var spad_base = fft_slot * scratchpad_uthread_stride` (runtime `//`/`%` of `local_uthread_id()`) | `var spad_base = 0` (literal, always) |
| Worker dispatch | `if worker_id == 0: ...` then a separate `if worker_id == 1: ...` (`fft_cooperative_codegen.py:220`, pre-fix) | `if worker_id == 0: ... elif worker_id == 1: ... ` (`fft_persistent_codegen._emit_worker_dispatch`) |

Persistent's `spad_base=0` is not a special case for this N -- it is
*always* 0, because persistent has no `fft_slot`/multi-tenant-scratchpad
concept at all (`workers_per_group` is architecturally fixed at
`target.interleave_chunk_uthreads`, one logical block fully occupies one
software group). Cooperative's `fft_slot` is architecturally real (more
than one sub-FFT's scratchpad can share a physical unit when
`workers_per_fft < interleave_chunk_uthreads`) -- except in this repro,
where `total_ffts=1` forces `fft_slots_per_group=1`, so `fft_slot` is
*provably* always 0 too, just not in a way the compiler can see from the
source alone.

## 3. Assembly comparison (real, `M2NDP_DUMP=all`, not reconstructed)

`src/m2ndp.mojo`'s own `Self.launch()` has a built-in exact-flags dump
(`M2NDP_DUMP=asm`/`all` env var, see `_dump`) -- used here instead of a
hand-invoked `llc`, so the flags (`-mattr=...`, `-m2ndp-map-address=...`,
`-m2ndp-range-param=...`) are exactly what the real launch used, not a
guess. Confirmed once by cross-checking a manual `llc -filetype=asm` run
against `--emit-ir`'s own IR: both reproduce the identical 16-byte-frame
warning, so the manual reconstruction is faithful too, but `M2NDP_DUMP` was
used for every number below.

**Cooperative `FFTRecLeaf0::stage_1()`** (737 asm lines): one scalar GPR
spill, confirmed scalar not vector (`sd`/`ld`, not `vs1r.v`/`fsd`):

```
sd  a1, 8(sp)     # 8-byte Folded Spill
...
ld  a1, 8(sp)     # 8-byte Folded Reload
```

Immediately after `spad_base` (`a5`) is computed (`slli a5, a5, 7`), the
compiled code materializes **~26 distinct address registers**
(`s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,ra,t0,t1,t2,t3,t6,a1..a7,...`), each
one `addi rX, a5, <per-batch offset>` -- all before the `worker_id`
dispatch branch, all simultaneously live. This is the actual register-
pressure driver: `spad_base` is a single *runtime* value shared by every
scratchpad address in the stage, so every `spad_base + offset` becomes an
independent "ready" node the scheduler can (and does) hoist early, forcing
~26 of them to coexist.

**Persistent `PersistentFFT::stage_1()`** (526 asm lines, same radix, same
batch structure): **no `addi sp, sp, -N` at all** -- zero frame. Address
computation is a tight, serialized `addr, use, addr, use, ...` pattern
reusing one register:

```
addi a1, a0, 576
vle32.v v8, (a1)
addi a1, a0, 832
vle32.v v9, (a1)
...
```

Same instruction *shape* per address (one `addi base, offset` before each
vector load -- RISC-V `vle32.v` has no immediate-offset form, so this step
exists either way), but because the base (`a0`) is the same value for
*every* address rather than a shared *runtime-computed* value threading a
division result through, nothing forces more than one address register
live at a time. This is the mechanism, confirmed by direct instruction-
level reading of both functions, not inferred from source alone.

## 4. 2x2 ablation (real hardware, same 4 build+run points)

Built by monkeypatching `codegen.fft_cooperative_codegen._emit_cooperative_
prelude`/`_emit_cooperative_stage` with faithful copies (verified
byte-identical to the real `generate_cooperative_fft_kernel` output before
introducing either toggle) that add two independent flags, then rendering
through the real `_emit_task_struct(..., narrow_middle_stages=True)` path
(not the standalone `generate_cooperative_fft_kernel`, which does not
forward `narrow_middle_stages` at all and would have rendered stage_1 at
an unrepresentative, unnarrowed width -- caught by first reproducing the
baseline this way and finding it did *not* spill, then re-deriving through
the actual production stage-width resolution). Fixed: `N=64`,
`radices=(4,4,4)`, `workers_per_fft=2`, `compute_lanes=4`,
`narrow_middle_stages=True`, `fft_slots_per_group=1` (required for the
constant-base variants to be semantically valid at all).

| Variant | `spad_base` | dispatch | frame | spill | correct | `ndp_cycles` |
|---|---|---|---|---|---|---:|
| A (baseline) | dynamic | independent `if` | 16B | **yes** | yes | 7969 |
| B | constant 0 | independent `if` | 0 | no | yes | 6033 |
| C | dynamic | `if`/`elif` | 48B | **yes (worse)** | yes | 7579 |
| D | constant 0 | `if`/`elif` | 0 | no | yes | **5702** |

Conclusion: **removed in B → dynamic `spad_base` is the primary, sufficient
cause.** Dispatch structure alone (C) does not fix it and makes the spill
*worse* (48B vs. 16B) -- `if`/`elif` on top of a still-dynamic `spad_base`
adds a real CFG merge the register allocator has to reconcile, without
removing the ~26-register cluster driving the spill in the first place.
D (both changes) is spill-free and the fastest of the four (5702 vs. A's
7969, a 28.5% cycle reduction) -- so the dispatch-keyword change is worth
keeping, but only *alongside* the constant-base fix that already made the
function spill-free, never on its own.

## 5. Fix applied

`codegen/fft_cooperative_codegen.py`, both gated on
`plan.cooperation.fft_slots_per_group == 1` (the one condition under which
`fft_slot` is provably always 0, matching persistent's own precondition
for the same specialization):

1. `_emit_cooperative_prelude`: emit `var spad_base = 0` (a Mojo compile-
   time literal, not a runtime expression) instead of
   `fft_slot * scratchpad_uthread_stride`.
2. `_emit_cooperative_stage`: worker dispatch (both the main body and the
   `stage_{id}_tail` body) uses `if`/`elif` instead of independent `if`s.

`fft_slots_per_group > 1` (a real physical group hosting more than one
cooperating sub-FFT's scratchpad) is untouched -- same dynamic `spad_base`
expression, same independent-`if` dispatch as before this fix, since the
ablation never validated either change for that case and variant C showed
the dispatch change alone is not safe to assume beneficial. Confirmed by
direct inspection of a genuine `fft_slots_per_group=4` case
(`gen_coop_repro.py 64 4,4,4 --workers 2 --total-ffts 4`) that both stay
byte-for-byte unchanged in the generated source.

No radix or N is hardcoded anywhere in the fix -- both branches key off
`plan.cooperation.fft_slots_per_group`, a plan-level field already computed
by the planner for every cooperative leaf.

## 6. Results

`N=64`, `radices=(4,4,4)`, `workers_per_fft=2`, `compute_lanes=4` (the
exact repro), through the real unmodified `run_fft_test.sh`:

| | before | after |
|---|---:|---:|
| spill | yes (16B frame, `stage_1`) | **no** |
| correctness | PASS | PASS |
| `ndp_cycles` | 7969 | **5702** (-28.5%) |

Bonus, unplanned confirmation: `N=216` `workers_per_fft=2` and `=4`
(`docs/compute_lanes_spill_avoidance.md`'s own "cooperative bookkeeping /
live state" rescued-by-narrowing case) are now **spill-free at the shipped
default `compute_lanes=4`**, no `--compute-lanes 2` narrowing needed at
all -- consistent with the same dynamic-`spad_base` mechanism, at a
different N/radix.

## 7. Regression

Real hardware, `run_fft_test.sh` unless noted:

- `N=64` `workers_per_fft` in `{2,4,8}`, `compute_lanes=4`, forward and
  inverse: all PASS, no spill warning.
- `N=216` `workers_per_fft` in `{2,4}`, `compute_lanes=4`: all PASS, no
  spill warning (previously needed `--compute-lanes 2`; see above).
- `N=128` `radices=(4,4,4,2)` `workers_per_fft=8` (the pool-alignment /
  `interleave_chunk_uthreads` correctness case from `docs/
  cooperative_worker8_pool_alignment_fix.md`): still PASS at
  `compute_lanes=4` (spill-free) and `compute_lanes=2` (spill warning) --
  **the `compute_lanes=2` spill was confirmed present before this fix
  too** (re-tested with the pre-fix module swapped back in, same result),
  so it is a pre-existing, unrelated register-pressure limit at this
  larger 4-stage/8-worker configuration, not a regression from this change
  and not something this fix claims to solve (see "remaining limitations"
  below).
- Genuine multi-slot cooperative (`fft_slots_per_group=4`,
  `gen_coop_repro.py 64 4,4,4 --workers 2 --total-ffts 4`): PASS,
  generated source confirmed unchanged (dynamic `spad_base`, independent
  `if`s) by direct inspection.
- Full Python-level suite, `python3 -m verification.verify_fft_plan`:
  195/195 checks pass, including every persistent/cooperative/non-
  cooperative cross-strategy numeric-equivalence and structural-invariant
  check -- persistent is untouched by this change (no persistent file was
  edited) and its own checks confirm that.

## 8. Remaining limitations (multi-slot, and beyond this fix's scope)

- `fft_slots_per_group > 1` cooperative leaves keep the original dynamic
  `spad_base`/independent-`if` shape and can still spill -- this fix does
  not attempt the worker-strided-loop alternative the task's own step 6
  describes as a fallback; that remains unimplemented and unmeasured.
- `N=128` `workers_per_fft=8` `compute_lanes=2` still spills, pre- and
  post-fix alike -- a larger-scale (4-stage, 8-worker) register-pressure
  case this targeted fix does not reach; not investigated further here.
- Narrower `compute_lanes` (2, 1) at the original `N=64`
  `workers_per_fft=2` repro still spill even after this fix -- only the
  shipped default (`compute_lanes=4`) was rescued; this matches the task's
  own instruction not to "solve" spills by narrowing `compute_lanes`, and
  the shipped default is the configuration that matters.

## 9. Commands run (for reproduction)

```
docker run -d --name fftwork ghcr.io/psal-postech/mojo-m2ndp:main sleep infinity
docker cp src fftwork:/work/ && docker cp benchmarks fftwork:/work/ \
    && docker cp sim fftwork:/work/ && docker cp scripts fftwork:/work/
# (strip CRLF from the Windows checkout, then:)
cd /work/benchmarks/fft
bash run_fft_test.sh 64 --cooperative-workers 2 --compute-lanes 4   # repro
M2NDP_DUMP=all ./host > full_dump.log 2>&1                          # real asm/IR dump
python3 -m verification.verify_fft_plan                             # 195/195
```
