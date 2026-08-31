# Per-leaf mixed execution strategy (Phase 4, 2026-08-31)

## What this adds

Before this, a whole recursive split tree picked one *uniform* leaf
execution strategy: `cooperative_workers=...` (every leaf cooperative) or
`persistent_leaf=True` (every leaf persistent, Phase 3) or neither (every
leaf non-cooperative). This phase lets each leaf the tree actually builds
pick its own strategy independently -- near=persistent/far=non-coop,
near=cooperative(2)/far=persistent, or any other mix -- by extending the
existing `forced_worker_sequence` mechanism rather than adding a parallel
one.

## Mechanism

`forced_worker_sequence` already walked one entry per leaf, near_fft-
first, deciding that leaf's own `cooperative_workers` value (`None` =
non-cooperative, `"auto"`/int = cooperative). This phase adds one more
possible entry value: the literal string `"persistent"`, meaning *this*
leaf is built by `make_persistent_leaf_plan` instead -- independent of
what neighboring entries in the same sequence say. A leaf's execution
strategy and its cooperative worker count were never independent choices
to begin with (exactly one of {non-cooperative, cooperative(workers),
persistent} applies per leaf), so reusing the one sequence keeps this
minimal rather than inventing a second parallel per-leaf parameter.

    make_recursive_transpose_plan(
        960, scratchpad_byte_budget=32*16,
        forced_worker_sequence=("persistent", None),  # near=persistent, far=non-coop
    )

`persistent_leaf` (the Phase 3 *uniform* toggle) and `cooperative_workers`
(the pre-existing uniform toggle) remain mutually exclusive with
`forced_worker_sequence` as a whole -- a caller picks either "one strategy
for the whole tree" or "a sequence, one entry per leaf", never a mix of
the two mechanisms. Within the sequence itself, any per-leaf mix of the
three strategies is legal.

No new candidate-generation/search-layer work is in this phase (Phase 6
per the plan's own phase list) -- this is the mechanism, not the search.

## Verification

**Python numeric harness** (`verify_fft_execution_invariants.
verify_mixed_leaf_strategy`, chained into the main suite): N=960, every
pairwise combination of {persistent, non-cooperative, cooperative(2),
cooperative(auto)} across the near/far split, forward+inverse -- all
match `numpy.fft`.

**Real M2NDP-Detour hardware**: two representative combinations, both
PASS with no mismatch:
- near=persistent, far=non-cooperative (`forced_worker_sequence=
  ("persistent", None)`)
- near=cooperative(2), far=persistent (`forced_worker_sequence=(2,
  "persistent")`)

Full existing regression suite unaffected. No `third_party/m2ndp-detour`
source touched.

## What this does not yet do

- No search/cost-model integration: `fft_plan_search.py` does not yet
  generate or rank mixed-strategy candidates. Phase 6.
- No bounded "representative candidate set" generator for mixed
  strategies (the plan's own Phase 4-2 wants a small, curated set --
  `(noncoop,noncoop)`, `(persistent,persistent)`, `(persistent,noncoop)`,
  `(noncoop,persistent)`, and cooperative combinations where legal --
  rather than the full Cartesian product); this phase proves the
  mechanism works and is correct, the bounded generator is search-layer
  work for Phase 6.
- The persistent-leaf scratchpad-legality gate (`fft_plan_search.py`'s
  own persistent candidate step) is still sized against the whole N, not
  reconsidered per-leaf (Phase 4-3 in the plan, still open).
