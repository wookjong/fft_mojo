# Persistent recursive-split support (Phase 3, 2026-08-31)

## What this adds

Persistent execution (`planning/fft_plan_persistent.py`,
`codegen/fft_persistent_codegen.py`) was previously only ever used as a
standalone giant leaf covering an entire N in one fused kernel -- never
combined with the recursive PRE -> near leaf -> MIDDLE -> far leaf -> POST
split every other leaf-execution strategy already supports
(`planning/fft_plan_recursive.py`). This phase makes persistent a
**leaf-lowering choice**, not a data-layout algorithm: the exact same
split, radix decomposition, transpose tiles, and large-twiddle math the
recursive planner already produces, with only *how one leaf executes*
changed.

    make_recursive_transpose_plan(n, scratchpad_byte_budget=..., persistent_leaf=True)

produces the identical tree `persistent_leaf=False` would, except every
leaf/near_fft kernel is built by `make_persistent_leaf_plan` instead of
`_build_plan`. Mutually exclusive with `cooperative_workers` and
`forced_worker_sequence` (asserted in `_build_recursive_node`) -- one
leaf, one execution strategy, never combined (matches
`PersistentWorkgroupPlan`'s own pre-existing "never combined with
cooperation" rule).

## Why the old N=960/1024 "persistent is ~50x slower" result doesn't apply

That comparison (see [[fft-persistent-vs-cooperative]] /
`docs/persistent_vs_cooperative_findings.md`) was:

    split non-persistent   (PRE -> near leaf -> MIDDLE -> far leaf -> POST)
        vs.
    unsplit persistent     (one giant leaf covering all of N)

not an execution-strategy comparison at fixed data layout -- persistent
was doing more total work (unsplit == the whole transform in one leaf)
under a structurally different plan. That result is retired as evidence
about persistent's own intrinsic performance; see below for the real
comparison.

## What changed, concretely

1. **`make_persistent_leaf_plan` gained `inverse_scale`** (the same
   `_DEFAULT` sentinel `_build_plan`/`make_cooperative_leaf_plan` already
   use) -- needed so a persistent leaf used as a *non-final* kernel in a
   chain (a near_fft, or occasionally a far leaf that isn't last) can
   suppress the `1/N` scale the way every other leaf kind already does.
2. **`fft_plan_recursive.py`**: `persistent_leaf: bool = False` threaded
   through `make_recursive_transpose_plan` -> `_build_recursive_node` ->
   `_build_leaf_kernel`, alongside `cooperative_workers` (same threading
   discipline, uniform-for-the-whole-tree today -- per-leaf mixed
   strategies are Phase 4, not this one).
3. **`fft_persistent_codegen.py`**: `generate_persistent_fft_kernel` split
   into `emit_persistent_kernel_struct` (the `NDPTask` struct + Params +
   preload/stage_N/writeback + round-unrolled `device_main` -- reusable)
   and a thin standalone-`main()` wrapper around it. Confirmed the
   refactor is byte-for-byte output-identical for every existing caller
   (diffed before/after on a representative case).
4. **`fft_transpose_codegen.generate_recursive_fft_kernels`**: a new
   per-stage branch (`stage.persistent is not None`) calls
   `emit_persistent_kernel_struct` instead of `_emit_kernel` in the
   "KERNEL IMPLEMENTATIONS" loop. No other special-casing was needed in
   the launch-emission loop: a persistent stage's own `total_uthreads ==
   max_uthread == launch_uthreads` (a target-fixed width, independent of
   replica count -- persistent absorbs however many logical blocks there
   are into its own internal round loop) makes the *existing* round-count
   formula (`rounds = ceil(total_uthreads / round_size)`) resolve to
   exactly 1 by construction, and the existing plain-Params code path
   (`stage_loops.get(i, False)` false for persistent, since it never
   loops via that mechanism) already matches persistent's own minimal
   4-field Params struct.
5. **Two real bugs this integration would otherwise have hit**, found and
   fixed along the way, both applying the *same* underlying fact
   (physical-unit placement depends on the launch pool's absolute DRAM
   address, not an index relative to the launch -- see
   `cooperative_worker8_pool_alignment_fix.md`):
   - `_stage_round_size` exempted cooperative stages from
     `spread_across_units`'s own widening formula, but not persistent
     ones -- `spread_across_units=True` (make_fft_kernel.py's own
     default) would have silently mis-widened a persistent leaf's launch
     pool sizing. Now exempts both.
   - The 256B pool-alignment fix (Phase 1) was gated on `stage.cooperation
     is not None`; a persistent stage's own `software_group_id` (`==
     global_uthread_id() // workers_per_group`) needs the exact same
     alignment for the exact same reason (workers sharing one group's
     scratchpad must physically share a unit) -- `fft_persistent_
     codegen.py`'s own *standalone* host `main()` already had this fix
     (`docs/persistent_leaf_design.md`'s "uthread pool alignment"
     section); the shared recursive-pipeline host `main()` needed it too.
     Now gated on `stage.cooperation is not None or stage.persistent is
     not None`.
6. **`verification/verify_fft_recursive.run_recursive_plan`** gained a
   `stage.persistent is not None` branch dispatching to
   `run_persistent_kernel` (it cannot reuse `run_kernel` at all --
   different round/group model entirely, see that function's own
   docstring) -- without this, a persistent-split plan was numerically
   unverifiable through the recursive-tree harness at all.
7. **`verification/verify_fft_execution_invariants.
   verify_persistent_recursive_split`**: same split/radix shape as the
   non-persistent call, plus numpy correctness -- new permanent regression
   coverage, chained into `verify_fft_plan.py`'s main suite.

## Verification

**Python numeric harness** (`verify_fft_execution_invariants.py`, chained
into the main suite): N=960/1024, forward+inverse, persistent-split vs.
non-persistent-split produce the *same split/radix decomposition* (same
leaf lengths, same stage counts) and both match `numpy.fft` to the same
tolerance.

**Real M2NDP-Detour hardware**:

| N | plan | inverse | result | ndp cycles |
|---|---|---|---|---|
| 960 | split, persistent leaves | forward | PASS, no spill | 1587 |
| 960 | split, non-persistent (same split) | forward | PASS, no spill | 1594 |
| 1024 | split, persistent leaves | inverse | PASS, no spill | 7688 |
| 1024 | split, non-persistent (same split) | inverse | PASS, no spill | 7688 |

Persistent and non-persistent are at **parity** under the same split --
N=960 forward: 1587 vs. 1594 cycles (persistent marginally *faster*);
N=1024 inverse: 7688 vs. 7688 (identical) -- a complete reversal of the
old ~50x-slower giant-leaf number. This is the real, apples-to-apples
execution-strategy comparison the old result never was.

## What this does not yet do

- Per-leaf mixed execution strategy (near=persistent, far=non-coop, etc.)
  -- `persistent_leaf` is still a single uniform choice for the whole
  tree, same as `cooperative_workers` before `forced_worker_sequence`
  existed. Phase 4.
- A search/cost-model axis for persistent-vs-cooperative-vs-non-coop *per
  leaf* -- this phase only makes persistent *reachable* through the
  recursive split, not yet a candidate `fft_plan_search.py` generates or
  ranks.
- The `_leaf_scratchpad_bytes(n) <= scratchpad_byte_budget` gate
  `fft_plan_search.py`'s own persistent candidate step (step 10) uses is
  still sized against the *whole* N, a leftover from when persistent was
  giant-leaf-only -- revisiting it for per-leaf legality is separate,
  future work (Phase 4's own "persistent gate 재검토" item).
- No `num_logical_blocks` tuning axis (matches the plan's own explicit
  "not this time" list -- sensitivity was already found small).

No `third_party/m2ndp-detour` source was touched.
