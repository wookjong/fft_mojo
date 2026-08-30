# Persistent vs. cooperative FFT leaf: real-hardware comparison findings

Answers docs/persistent_vs_cooperative_comparison_task.md. All numbers below
are real M2NDP-Detour build+run results (`planning.spill_probe.probe_source_
spill_free`), not estimates. Forward transforms only; N=32/64/128, radices
chosen by this project's own existing planner (`make_recursive_transpose_plan`
for cooperative, the same radix sequence passed to `make_persistent_leaf_plan`
for persistent) so both strategies always run the *same* mathematical
decomposition. `--cooperative-workers 8` is a cap, not a forced value --
`choose_workers_per_fft` picks the actual `workers_per_fft` (2 for N=32/64,
8 for N=128); this was not overridden, since forcing 8 onto a leaf where
`choose_workers_per_fft` says 2 is useful would just idle 6 workers on
every stage for no benefit -- not a fair per-strategy config.

## 1. Why the original per-(phase x round) persistent design failed

See docs/persistent_leaf_design.md's own "POST-IMPLEMENTATION CORRECTION"
section and codegen/fft_persistent_codegen.py's top docstring. Summary:
`max_kernel_register=8` (real M2NDP-Detour config) caps how many *distinct*
kernel functions one task may ever register, for the whole life of one host
`.launch()`. Baking `ROUND_BASE`/`ACTIVE_GROUPS` into a separate function per
round blew past that for any real multi-round case (N=64 x 2 rounds needed
10) and aborted the whole simulator process, not a graceful error.

## 2. The `max_kernel_register=8` constraint

Now enforced at plan-build time: `make_persistent_leaf_plan` raises
`ValueError` if `2 + len(radices) > target.max_kernel_register` (a real
`TargetProfile` field now, `planning/target_profile.py`), before codegen is
even attempted. A structural test
(`verify_fft_persistent._check_registered_kernel_count_stable_across_rounds`)
confirms 2/8/32-round plans for the same N all register exactly the same 5
functions (`preload`, `stage_0`, `stage_1`, `stage_2`, `writeback`).

## 3. The round-counter fix (unchanged from the prior session)

Round state lives in a 1-element scratchpad counter, one instance per
physical NDP unit, incremented only by `writeback` after every other phase
in that round has already read it. Re-confirmed correct at 2 and 8 rounds on
real hardware this session too (see table below, N=64/N=32 multi-round rows).

## 4. Cooperative spill root cause (N=64, radices=(4,4,4), workers_per_fft=2)

`FFTRecLeaf0.stage_1()` -- the radix-4 *middle* stage -- spills (16-byte
frame) at the project's own default `compute_lanes=4`, `narrow_middle_
stages=True`. Same liability class as every other radix-4-middle
register-pressure case this project has hit (see [[fft-cost-model-session]]
memory's own N=64 note from the persistent-leaf work). `loop_stages`
(default on) does not engage here: `_try_build_loop_stage` needs a minimum
full-batch count to bother looping, and at `workers_per_fft=2` each worker's
own per-stage batch count is too small (`ceil((64//4)/8)=2` batches total,
split 1-each across 2 workers) to ever cross that threshold -- so this
mitigation is a structural no-op at this worker count, not a bug.

## 5. What fixed it

Nothing new -- `--compute-lanes 2` (an existing, already-exposed knob,
`probe_spill_free`'s own `compute_lanes` parameter / `make_fft_kernel.py`'s
own `--compute-lanes` flag). No new mechanism, no hand-written special case,
per the task's own instruction. Confirmed real-hardware spill-free and
correct at N=64, every block count tried (8/32/40); **N=64 blocks=1 could
not be made spill-free at compute_lanes=4, 2, or 1** -- the only
configuration in this sweep with no working cooperative candidate at all
(see table; not investigated further, out of scope for this comparison).

## 6. Fairness conditions used

Same N, same radices (both strategies get literally the same tuple), same
logical block / batch count, forward-only, same `TargetProfile`
(`DEFAULT_TARGET_PROFILE`). `workers_per_fft` (cooperative) vs.
`workers_per_group` (persistent, architecturally fixed at 8) were **not**
forced equal -- see header note. `compute_lanes` was **not** forced equal
either: each strategy used its own minimal sufficient value (persistent:
default 4, always sufficient in this sweep; cooperative: 4 when safe, else
2). This reflects what a real caller would actually ship for each strategy,
not an artificial handicap.

## 7. Comparison table

All rows: `inverse=False`, `simd_lanes=8`, `DEFAULT_TARGET_PROFILE`.
"Best" cooperative = narrowest safe `compute_lanes` tried, in order (4, 2, 1).

| N | Radices | Blocks | Strategy | workers | Registered fns | Spill | Correct | ndp_cycles |
|---|---|---:|---|---:|---:|---|---|---:|
| 32 | (4,4,2) | 1 | cooperative (cl=4) | 2 | n/a | free | yes | 6805 |
| 32 | (4,4,2) | 1 | persistent | 8 | 5 | free | yes | **6492** |
| 32 | (4,4,2) | 8 | cooperative (cl=4) | 2 | n/a | free | yes | 6869 |
| 32 | (4,4,2) | 8 | persistent | 8 | 5 | free | yes | **6551** |
| 32 | (4,4,2) | 32 | cooperative (cl=4) | 2 | n/a | free | yes | 6889 |
| 32 | (4,4,2) | 32 | persistent | 8 | 5 | free | yes | **6597** |
| 32 | (4,4,2) | 40 | cooperative (cl=4) | 2 | n/a | free | yes | **6909** |
| 32 | (4,4,2) | 40 | persistent (2 rounds) | 8 | 5 | free | yes | 9608 |
| 64 | (4,4,4) | 1 | cooperative | 2 | n/a | **none spill-free (cl 4/2/1 all spill)** | -- | -- |
| 64 | (4,4,4) | 1 | persistent | 8 | 5 | free | yes | **7676** |
| 64 | (4,4,4) | 8 | cooperative (cl=2) | 2 | n/a | free | yes | 10083 |
| 64 | (4,4,4) | 8 | persistent | 8 | 5 | free | yes | **7709** |
| 64 | (4,4,4) | 32 | cooperative (cl=2) | 2 | n/a | free | yes | 10092 |
| 64 | (4,4,4) | 32 | persistent | 8 | 5 | free | yes | **7747** |
| 64 | (4,4,4) | 40 | cooperative (cl=2) | 2 | n/a | free | yes | **10097** |
| 64 | (4,4,4) | 40 | persistent (2 rounds) | 8 | 5 | free | yes | 11903 |
| 128 | (4,4,4,2) | 8 | cooperative | 8 | n/a | free (cl 2/1) | **NO -- wrong answer, every compute_lanes** | -- |
| 128 | (4,4,4,2) | 8 | persistent | 8 | 6 | free | yes | 13312 |
| 128 | (4,4,4,2) | 40 | cooperative | 8 | n/a | free (cl 2/1) | **NO -- wrong answer, every compute_lanes** | -- |
| 128 | (4,4,4,2) | 40 | persistent (2 rounds) | 8 | 6 | free | yes | 20539 |

Bold = faster of the two, only where both sides are actually valid
(correct + spill-free).

## 8. Controlled same-radix comparison

Every row above already is one -- both strategies always ran the identical
radix sequence per N. No separate "controlled" vs. "best" split was needed
for N=32/64; no working cooperative baseline exists for N=128 at all (see
item 9).

## 9. Best spill-free comparison, where different from "controlled"

**UPDATE 2026-08-30 -- fixed, see below.** N=128 originally had no "best
cooperative": every candidate tried (cl=4/2/1) produced a wrong answer
(`mismatch at 32`), independent of spilling -- cl=2 and cl=1 were *both*
spill-free (no warning) *and* wrong, a real blind spot in the existing
spill-only safety net (spill-free is necessary evidence of safety here,
not sufficient). Root-caused far enough to fix, not fully to the
instruction level: confirmed real-hardware wrong at `workers_per_fft=8`
(the full `interleave_chunk_uthreads`) across every `compute_lanes`/batch
size tried, while the Python-level numeric harness (the *actual* emitted
stage text, re-executed) passes cleanly at the same configuration --
meaning the bug is below this codebase's own planning/codegen layer,
matching this project's own established pattern for this class of issue
(compare the N=630 `vlenb`-CSR and transpose-tile findings in
[[fft-cost-model-session]]). It also matches an *already-documented* case
in this same codebase: `fft_cooperative_codegen._emit_cooperative_stage`'s
own docstring records a 2026-08-27 N=1024/`workers_per_fft=8` wrong-answer
finding, believed fully fixed at the time by making `loop_stages` apply
per-worker -- that fix was necessary but **not sufficient**; this session
found the same worker-count/wrong-answer pattern persists independent of
`loop_stages`/function size.

**Fix landed** (`planning/fft_plan_cooperative.py`,
`worker_candidates_per_fft`/`choose_workers_per_fft` gain
`exclude_full_interleave_chunk: bool = True`): `workers_per_fft ==
interleave_chunk_uthreads` (8) is no longer offered as an automatic
candidate, by default -- both the `"auto"` path and an explicit
`--cooperative-workers 8` (which was *already* only ever an upper bound
for `choose_workers_per_fft`, never a raw override -- see
`fft_plan_recursive.py`'s own docstring on this) now fall back to the
next-largest safe divisor (4, on this target), confirmed real-hardware
correct at every `compute_lanes`/batch size tried for N=128. Persistent's
own N=128 numbers still stand as the only option confirmed *both*
spill-free *and* correct at that N in this sweep -- cooperative's
now-default `workers_per_fft=4` is correct but still spills (a separate,
pre-existing, purely register-pressure issue, mitigable the same way as
every other case in this doc: `--compute-lanes 2`).

## 10. Sensitivity

**Round-boundary crossing dominates persistent's own cost curve.** At both
N=32 and N=64, persistent is faster than cooperative for every block count
that fits in one round (<= 32, this target's `software_group_count`) and
*slower* the moment block count crosses into a second round (40 blocks):

- N=32: persistent 6492/6551/6597 (blocks 1/8/32, all faster than
  cooperative) -> 9608 at blocks=40 (crosses to 2 rounds, now slower than
  cooperative's 6909).
- N=64: persistent 7676/7709/7747 (blocks 1/8/32, all faster) -> 11903 at
  blocks=40 (2 rounds, now slower than cooperative's 10097).

This is a genuine **crossover as logical-block count changes** -- exactly
the signal the task asked to highlight. Cooperative's own cost is close to
flat across block counts at fixed N (6805->6909 for N=32, 10083->10097 for
N=64, a <2% spread) since its own launch-width/round model doesn't share
persistent's fixed 32-group ceiling. Concretely: persistent pays a full
extra `preload -> stage_0..k -> writeback` launch_parallel sequence for the
tail round (8 of 40 blocks, in this sweep) at close to the same fixed cost
as the first round's 32-block round -- a large fixed overhead for a small
tail, whereas cooperative's own round/launch structure (inherited from
`generate_recursive_fft_kernels`'s pre-existing, `spread_across_units`-aware
round-split logic) apparently amortizes better here.

**Leaf size (N)**: too few points to separate cleanly from the round-boundary
effect above, since N=32/64 show the same qualitative pattern and N=128
has no valid cooperative point to compare against. Persistent's own
ndp_cycles scale roughly with `N * rounds` as expected (N=32: ~6.5K one
round; N=64: ~7.7K one round; N=128: ~13.3K one round, ~20.5K two rounds).

**Worker utilization**: not independently varied in this sweep (cooperative's
`workers_per_fft` was always the planner's own natural choice, never forced)
-- a real follow-up would vary this deliberately.

## 11. Is there enough evidence for a selection rule yet?

Partial, not final (matches the task's own "do not build the final cost
model yet" instruction):

- **Prefer persistent when the full logical-block count fits in one round**
  (`num_logical_blocks <= target.num_ndp_units`) -- consistently faster in
  every tested case (N=32, N=64) and is the *only* correct+spill-free option
  at N=128 in this sweep.
- **Cooperative may win once block count exceeds one round's worth**, but
  only where it can be made both spill-free (may need a narrower
  `compute_lanes`, a real cost of more scalar instructions) *and* correct --
  and this sweep found a real, unresolved case (N=128, `workers_per_fft=8`)
  where cooperative is not correct at any `compute_lanes` tried, so
  "cooperative as fallback" cannot be assumed reliable without per-case
  verification.
- No rule here should be trusted without the same real spill+correctness
  probing this sweep used -- neither side's cycle count is meaningful
  without both checks passing first (see item 4/9 above for cases where
  skipping either would have produced a misleading number).
