# Persistent joins the radix x execution-strategy joint search (Phase 6, 2026-08-31)

## What changed

`fft_plan_search.py` already had a "radix tier x per-leaf worker
sequence" joint search (`generate_radix_execution_joint_candidates`,
step 9 of `generate_candidates`), built on
`generate_leaf_worker_sequences`'s own full Cartesian product across each
leaf's legal cooperative worker counts. Persistent execution was
reachable only through a wholly separate path (`generate_persistent_leaf_
candidates`, step 10): one candidate per radix tier, always a single
*unsplit* leaf covering the entire N, gated off entirely whenever N
needed `make_recursive_transpose_plan`'s own recursion (`_persistent_
leaf_feasible`'s second condition) -- because that was measured ~50x
slower, back when persistent could only mean "one giant unsplit leaf."

Phase 3/4 (this session, earlier) made persistent a leaf-lowering choice
reachable *through* a split, including mixed per-leaf. This phase wires
that into the actual search: `generate_leaf_worker_sequences` and
`generate_per_leaf_worker_candidates` now offer `"persistent"` as one
more per-leaf option, alongside each leaf's own legal cooperative worker
counts, gated by a new `_leaf_persistent_feasible` (the real, physical
per-leaf capacity check -- `16 * length <= spad_capacity_bytes` -- not
the old N-wide, evidence-gated heuristic, which does not apply once
persistent runs at leaf granularity with a real replica count instead of
`num_logical_blocks=1`).

Since `generate_leaf_worker_sequences` already builds the full Cartesian
product across each leaf's own options, adding `"persistent"` as one
more option automatically produces exactly the plan's own "minimal mixed
candidates" list for free, from the same mechanism -- no separate
generator needed:

    2-leaf tree -> (None, persistent), (persistent, None),
                   (persistent, persistent),
                   (w, persistent), (persistent, w)  for each legal w

This flows straight through the *existing* joint search
(`generate_radix_execution_joint_candidates`), crossed with every radix
tier exactly as any other worker sequence already was -- `"persistent"`
entries needed no new code there beyond widening the `except` clause to
also catch `NotImplementedError` (`make_persistent_leaf_plan`'s own
target-mapping-invariant failure mode).

## The old unsplit gate's scope, clarified

`_persistent_leaf_feasible`/`generate_persistent_leaf_candidates` (step
10) still exist unchanged, and are still correctly gated off for
N=960/1024 -- that specific claim (an *unsplit* persistent leaf, one
physical NDP unit doing all the work, is ~50x slower) was never wrong,
just narrower in scope than its own docstring made it sound. Both
functions' docstrings now say so explicitly, and a new regression test
(`check_split_persistent_reachable_where_unsplit_is_gated`) confirms the
two paths coexist correctly: the unsplit generator stays empty for
N=960/1024, while the per-leaf joint search now reaches a genuinely
*split* persistent candidate for the same N.

## Verification

- `generate_leaf_worker_sequences(N=960's baseline plan, ...)`: 9 total
  sequences, 5 containing `"persistent"` -- exactly the representative
  mixed set (near-only, far-only, both, and two worker+persistent
  combinations).
- `generate_candidates(960, ...)`: 98 total candidates (up from 91
  before this phase), 5 involving a persistent leaf, all with real
  `estimated_cost` values competitive with the top-ranked non-persistent
  candidates (~75.6K vs. ~71.3K) -- not the old wrongly-cheap-then-
  catastrophically-slow shape the gate was built to prevent.
- A candidate pulled straight out of `generate_candidates(960, ...)`
  (`worker_sequence=("persistent", None)`, i.e. near leaf persistent, far
  leaf non-cooperative -- chosen by the search, not hand-built) rendered
  through the real `generate_recursive_fft_kernels` and **built and ran
  correctly on the real M2NDP-Detour simulator**.
- `verify_fft_persistent_search.py` (existing regression suite): two
  pre-existing tests (`check_persistent_signature_distinct_from_
  cooperative`, `check_persistent_lower_register_pressure_reflected_in_
  cost`) updated to recognize a persistent candidate via *either*
  generation route (`execution_strategy == "persistent"` from step 10,
  or `"persistent"` in `worker_sequence` from the per-leaf joint search)
  -- both routes can legitimately build the identical plan for a small
  enough N (no split at all), and dedup correctly keeps only one; a test
  pinned to one specific label was the bug, not the dedup. Full existing
  regression suite unaffected otherwise.

No `third_party/m2ndp-detour` source touched.

## What this does not yet do

- compute_lanes is not yet a joint search axis alongside radix/execution
  strategy -- still separate, later work within this same phase.
- `num_logical_blocks` remains a canonical/default value, not a search
  axis (per the plan's own explicit scoping -- measured sensitivity was
  already found small).
- The physical-parallelism dimension the old gate's own docstring named
  (`fft_cost_model.py` not representing active-NDP-unit count at all) is
  still not modeled explicitly -- Phase 7.
