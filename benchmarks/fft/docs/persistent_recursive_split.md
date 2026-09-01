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

## CORRECTION (2026-08-31): the "parity" claim above was measured wrong

The 1587-vs-1594 (N=960) and 7688-vs-7688 (N=1024) numbers in this doc's
own "Real M2NDP-Detour hardware" table were extracted with `grep
"Gantt info:.*finished NDP kernel" | grep -oP 'ndp cycle \K[0-9]+' | tail
-1` (`benchmark_fft_candidates.sh`'s and `planning/spill_probe.py`'s own
convention at the time) -- confirmed **wrong** for any multi-kernel
(split) plan: `src/m2ndp.mojo`'s `Self.launch()` spawns a fresh
`m2ndp_run` *subprocess* per top-level kernel struct (PRE, near leaf,
MIDDLE, far leaf, POST each their own process), and `M2NDPConfig::
m_ndp_cycle` restarts at 0 in each one -- confirmed directly in a real run
log ("Registered task ... ndp cycle 0" once per struct). `tail -1` only
ever captured the LAST struct's (POST transpose) own standalone duration,
which is identical whether the near/far leaves are persistent or plain --
explaining why this table found "parity" in the first place: it was
comparing the *same* POST-transpose number to itself, not the real
end-to-end pipeline. Full root cause: `planning/spill_probe.py`'s
`_parse_ndp_cycles` (fixed 2026-08-31) and
`docs/active_ndp_units_cost_task.md`'s own writeup.

**Corrected real totals** (sum of each kernel struct's own final cycle
value, same split/radix/tile/compute_lanes as this doc's original table,
`benchmark_fft_candidates.sh`/`spill_probe.py` now fixed to compute this
automatically):

| N | plan | inverse | corrected total ndp cycles |
|---|---|---|---:|
| 960 | split, persistent leaves | forward | 39378 |
| 960 | split, non-persistent (same split) | forward | 48320 |
| 1024 | split, persistent leaves | inverse | 39730 |
| 1024 | split, non-persistent (same split) | inverse | 49061 |

Persistent is genuinely **~18-19% faster** than non-persistent at this
split -- a real, meaningful effect, not the "parity" originally reported
(nor the old "~50x slower" pre-split-support number this doc's own
"why the old result doesn't apply" section already retired). Note this
new comparison used `probe_pool_alignment.py`'s own default `cooperative_
workers`-free / `persistent_leaf`-only construction, not necessarily
byte-identical host-generation code to the original table (e.g. inverse
vs forward differs per row above, matching the original rows) -- treat
these as a fresh, correctly-measured data point superseding the original
table's conclusion, not a byte-for-byte re-run of the exact same binaries.

**What this does NOT retire**: the persistent-leaf-inside-a-split
mechanism itself (Phase 3's own implementation) is unaffected -- this
correction is entirely about how cycles were *measured* after the plans
were already built and run correctly. Also see `docs/
active_ndp_units_cost_task.md`'s Phase 2 dataset: at the same splits,
**cooperative execution (workers=2/4/8) beats both persistent and
non-persistent**, e.g. N=1024 coop2=32054 vs. persistent=39730 vs.
non-persistent=49061 -- a comparison this doc never made at all (it only
ever compared persistent vs. non-persistent, never against cooperative at
this split).
