# compute_lanes as a spill-avoidance knob: real-hardware sweep (2026-08-30)

Follow-up to the ExecutionCost rework (`execution_cost_model_validation.md`)
and the exact-VLEN compiler investigation (dead end -- see that section's
own "compiler-side fixed-VLEN" notes): given the M2NDP-Detour simulator
itself is never touched, this pass asked whether `compute_lanes` narrowing
(already a codegen parameter, `generate_recursive_fft_kernels`'s own
`compute_lanes`) can generate genuinely simulator-safe (spill-free,
correct) candidates for cases the current `_stage_compute_lanes` heuristic
doesn't already handle.

24 real build+run points (8 representative candidates x compute_lanes in
{1, 2, 4}, `narrow_middle_stages=True` unless noted): `compute_lanes_
sweep.py`/`worker_lanes_sweep.py`/`persistent_lanes_sweep.py` in this
session's own scratchpad, raw JSON not checked in (real-hardware runs are
reproducible from the plan/config recorded here, not meant to be re-parsed
as a frozen dataset).

## Results

At the shipped default (`compute_lanes=4`, `narrow_middle_stages=True`),
5 representative candidates spill:

| candidate | root cause | lanes=2 | lanes=1 |
|---|---|---|---|
| N=105 single leaf, radix 5 genuine middle stage | operand count, stage position | still spills | still spills (+1 more stage) |
| N=630 default split, radix 5 genuine middle stage | same | still spills | still spills (+1 more stage) |
| N=64 single leaf, pure radix-4 chain | unknown -- no `_stage_compute_lanes` trigger fires at all | still spills, and SLOWER | still spills (+1 more stage) |
| N=216 cooperative worker=2 | cooperative bookkeeping / live state | **spill-free** | **spill-free** |
| N=216 cooperative worker=4 | same | **spill-free** | still spills (different stage) |

**Rescue rate: 2/5.** compute_lanes narrowing, down to fully scalar, never
rescues a radix-5-genuine-middle spill (matches `fft_cost_model.py`'s own
prior N=630/N=105 finding, now confirmed with two more real reruns) or
N=64's pure-radix-4 spill -- narrower is not even monotonically safer for
N=64 (lanes=1 spills an *additional* stage vs. lanes=4). It reliably
rescues the two worker-cooperative cases, which is the one category the
current `_stage_compute_lanes` heuristic has zero visibility into: that
function (`codegen/fft_codegen.py:655`) does not take a worker-count
parameter at all.

Both rescued candidates are correct (100% PASS) and, more importantly,
both **beat the previously-best-known spill-free alternative** for N=216
(the plain non-cooperative baseline, 43771 cycles) once rescued:

| rescued candidate | cycles | vs. 43771 baseline |
|---|---|---|
| worker=2, lanes=2 | 35572 | 18.7% faster |
| worker=2, lanes=1 | 37113 | 15.2% faster |
| worker=4, lanes=2 | 28784 | 34.2% faster |

(Comparing a rescued candidate's cycles against its *own* spilling
number at lanes=4 is not a meaningful ratio -- that spilling number was
never a usable option to begin with under [[fft-spill-hard-filter]], so
"how much slower than the unsafe version" answers nothing a search would
act on. The only decision-relevant comparison is against the best
alternative that was actually safe to pick before this rescue existed.)

## `_stage_compute_lanes` heuristic: what it does and doesn't catch

Three triggers, none aware of cooperative worker count:
`narrow_middle_stages`+middle-stage-position (halves), `radix in
_ALWAYS_NARROW_RADICES={10,11,13,17}` (floors to 1), `(prev_radix, radix)
in _RISKY_RADIX_PAIRS={(6,9)}` (floors to 1). For radix-5-genuine-middle,
the heuristic already applies its strongest available lever (halving) and
this sweep confirms -- again -- that it is insufficient at any width, a
known, documented gap (`fft_cost_model.py`'s own `_RISKY_AS_MIDDLE_
RADICES` comment), not a heuristic bug. For N=216 worker=2/4, the
heuristic has *no* mechanism that could ever fire: the radix sequence
(4,2,3,3,3) trips none of the three triggers, and worker count isn't an
input at all. This is the real, actionable gap this sweep found: **worker
count must become an input to the lane-width decision.**

## ExecutionCost and compute_lanes

`compute_lanes` is not reflected in `PlanMetrics`/`compute_stage_metrics`
at all -- confirmed by reading `layouts_for_radices` (the function that
decides `simd_iteration_count`/`batches` at the planning layer): it never
receives `compute_lanes`, which is a purely codegen-time rendering
parameter (`fft_codegen._chunk_batch`) applied *after* planning has
already fixed the batch structure. So "lanes down -> simd_iteration_count
up -> total_worker_stage_batches up -> execution_cost up" does not
happen today, not because it's already handled, but because there is no
channel for it to happen through. Promoting `compute_lanes` to a real
plan field (see below) would need to add that channel -- deliberately not
done this pass (no new cost weight, per this investigation's own scope).

## Persistent execution: the actual answer for N=216

While investigating this, a third, already-implemented execution model
(`planning.fft_plan_persistent`, `docs/persistent_leaf_design.md`) turned
out to dominate both cooperative rescues for the same N/radix outright,
with *no* narrowing needed:

| execution strategy | compute_lanes | spill_free | cycles |
|---|---|---|---|
| non-cooperative baseline | 4 (shipped default) | yes | 43771 |
| cooperative worker=4, rescued | 2 | yes | 28784 |
| **persistent**, num_logical_blocks=1 (apples-to-apples, batch=1) | 4 (shipped default) | yes | **23944** |

Persistent needed no compute_lanes intervention at all here -- it was
already spill-free at the shipped default width. This was folded into
`generate_candidates` as a new search axis (step 10) the same session --
see `execution_cost_model_validation.md`'s own "Follow-up: persistent
execution as a search axis" section for the implementation and real-
hardware verification.

## Recommendation

Given the evidence: **worker x compute_lanes is the axis actually worth
searching**, not compute_lanes alone as an independent knob (radix-5-
genuine-middle and pure-radix-4 cases showed compute_lanes cannot rescue
them regardless of width, so a blanket "compute_lanes helps spills"
generalization would be wrong). Two concrete next steps this data
supports, neither done this pass:

1. Extend `_stage_compute_lanes` (or its planner-level replacement) to
   take worker count as an input -- the single clearest false negative
   this sweep found.
2. Now that persistent is a real search-axis alternative for feasible N
   (step 10), re-evaluate how much of the "rescue cooperative spills via
   narrower lanes" problem is actually still worth solving vs. simply
   letting persistent win those candidates outright when it's available
   and faster anyway.
