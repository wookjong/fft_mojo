# compute_lanes joins the search (Phase 6, second half, 2026-08-31)

## What changed

Phase 5 gave `FFTStagePlan` its own `compute_lanes` field and a planner-
side resolver (`planning/fft_plan_lanes.py`), but nothing generated
candidates that actually used it -- every plan `fft_plan_search.py` built
still had `stage.compute_lanes = None` on every stage, deferring to
codegen's live-fallback resolution at render time. This phase wires
compute_lanes into `generate_candidates` as a real search axis (step 11).

`fft_plan_lanes.py` gained a tree-wide rewrite pair, since a
`RecursiveFFTPlan`'s leaves sit nested inside frozen-dataclass
`FFTLeafPlan`/`FFTRecursiveNodePlan` nodes, not a flat list the way one
`FFTCodegenPlan`'s own stages are:

- `apply_compute_lanes_to_plan(plan, ...)` -- every FFT leaf in the tree
  gets `resolve_stage_compute_lanes` applied, `PhysicalTransposePlan`
  (PRE/MIDDLE/POST) stages untouched (compute_lanes is an FFT-arithmetic
  concept only).
- `apply_all_scalar_lanes_to_plan(plan)` -- every stage of every leaf
  floored to `compute_lanes=1`.

`fft_plan_search.generate_lane_variant_candidates(plan, choices, ...)`
uses these to build two bounded variants of a given plan: `"unnarrowed"`
(no middle-stage halving) and `"all_scalar"` (the floor real-hardware
evidence never found *wrong*, only sometimes slower). The third,
"baseline `narrow_middle_stages=True`" shape is deliberately *not* added
again as a plan-level candidate: every other candidate this module
already builds has `stage.compute_lanes` unset, and codegen's own live
fallback (`make_fft_kernel.py`'s shipped default) already renders that
exact code at emit time -- adding it here would score the same eventual
build twice under two different signatures.

`generate_candidates` gained a `compute_lanes` parameter (default
`min(simd_lanes, target.lmul1_float32_lanes)`, matching `make_fft_
kernel.py`'s own shipped default) and step 11: `generate_lane_variant_
candidates` applied to the baseline plan, held at the baseline split --
same "one axis at a time, star search around a fixed point" discipline
every other step in this module already uses (worker sweep, radix-tier
sweep, per-leaf worker sweep all hold everything else fixed the same
way).

## Why this is additive

Every existing caller of `generate_candidates` is unaffected: the new
`compute_lanes` parameter has a default, steps 1-10 are untouched, and
step 11 only ever adds two new candidates (not crossed with every other
axis) rather than changing anything already there.

## Verification

- `generate_candidates(960, ...)`: 98 -> 100 candidates, the 2 new ones
  each with every leaf's every stage's own `compute_lanes` genuinely
  populated (not `None`) -- confirmed `"all_scalar"` floors every stage
  to `1`.
- `_plan_signature` (Phase 5's own change) correctly keeps both lane
  variants distinct from the plain baseline -- confirmed directly, not
  just by construction.
- `probe_and_rerank_candidates` (pre-dates this axis) accepts step-11
  candidates with no changes needed on its own side.
- A candidate pulled straight out of `generate_candidates` (`lane_
  variant="all_scalar"`, search-chosen, not hand-built) rendered through
  the real `generate_recursive_fft_kernels` and **built and ran correctly
  on the real M2NDP-Detour simulator** (N=960).
- Full existing regression suite unaffected.

No `third_party/m2ndp-detour` source touched.

## What this does not yet do

- `estimate_cost` does not yet differentiate the two lane variants from
  each other or from the baseline (both scored identically in this
  session's own runs) -- the cost model does not model compute_lanes at
  all yet (Phase 7's own "7-1. compute_lanes cost" item: real emitted-
  chunk counts, not an arbitrary penalty). Ranking these candidates
  meaningfully today happens downstream, at the spill-probe/real-hardware
  measurement stage (a spill is a real, per-lane-width property the probe
  already sees), not via `estimated_cost`.
- Only two variants (`unnarrowed`, `all_scalar`) around the baseline
  split/radix -- not crossed with worker sequences, persistent leaves, or
  non-baseline radix tiers/splits. Extending this to per-leaf-independent
  lane choices (different leaves narrowed differently, matching the
  plan's own "worker x compute_lanes"/"persistent x compute_lanes" joint
  candidates) is further work within this axis, not done here.
