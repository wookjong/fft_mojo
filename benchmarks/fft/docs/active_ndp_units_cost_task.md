# Scope: model active-NDP-unit count in the cost model (Phase 7)

**Status (2026-08-31, later session): diagnostic metric implemented;
Phase 1 (runtime alignment audit) complete; Phase 1.5 found and fixed a
much bigger, project-wide "measured ndp cycles" methodology bug affecting
every split-plan real-hardware comparison this project has ever made
(several historical docs corrected); Phase 2 (fair split-level
execution-strategy dataset) complete as a side effect of the Phase 1.5
fix. Still NOT wired into cost.** See "Phase 1 findings" and especially
"Phase 1.5" near the end of this doc -- read Phase 1.5 first, it's the
most consequential section.

Follow-up to [[fft-radix-execution-persistent-joint-search]]'s own closing
line and `docs/compute_lanes_joint_search.md`'s now-closed Phase 7-1
(compute_lanes chunk-count weighting, 2026-08-31, see that doc's own
"Update 2026-08-31" section). This is the *other* named gap, still open:
`fft_cost_model.py` has no representation at all of how many physical NDP
units a plan's own launches actually activate concurrently.

## The gap, in the project's own words

`planning/fft_plan_search.py`'s `_persistent_leaf_feasible` docstring:

> a single persistent leaf activates only one of `target.num_ndp_units`
> physical NDP units (`num_logical_blocks=batch`, usually 1, active)
> where the split/transpose structure fans real DRAM-tile work out across
> all of them -- a dimension `fft_cost_model.py` does not model at all
> ... so this condition cannot be left to cost-based ranking to catch.

`docs/persistent_representative_sweep.md`'s own cost-model section is
more direct still: for N=960/1024 the model doesn't just miss this, it is
**inverted** -- persistent's `estimated_dram_bytes` looks cheapest
because the model counts total bytes moved as if by one serial stream,
with "no channel for 'but N of those bytes move over N different
physical units at once.'" That doc's own verdict: *"No new weight can fix
this -- the quantity the model would need to weight does not exist in
`PlanMetrics` at all yet."*

Today this gap is papered over by a **structural gate**
(`_persistent_leaf_feasible`'s second condition,
`_leaf_scratchpad_bytes(n) <= scratchpad_byte_budget`), not a cost term --
found empirically to separate 5 real wins from 2 real ~50x losses in one
7-N sweep. That gate should **stay** regardless of what this work
produces ([[fft-spill-hard-filter]]'s same discipline: a soft cost signal
must never be trusted alone to keep a confirmed-catastrophic candidate
out -- see "Non-goals" below).

## Data that already exists -- reread before running anything new

`docs/persistent_representative_sweep.md` already has real M2NDP-Detour
cycle counts that are direct evidence for this exact hypothesis. Reread
it in full before touching code:

- The 7-N table (N=30/64/105/144/256/960/1024: non-cooperative /
  best-cooperative / persistent cycles, spill status, and the
  side-by-side `estimated_cost` comparison showing the model backwards at
  960/1024).
- The `num_logical_blocks` sensitivity table (N=216, N=64 across
  blocks={1,2,4,8}) -- already shows this axis is "essentially flat," a
  real constraint on how steep any new active-units term should be.
- `round_active_groups` (`planning/fft_plan_persistent.py:459`) already
  computes, per round, exactly "how many of `target.num_ndp_units`
  software groups are active" for a persistent plan -- reuse this, don't
  re-derive it.

**Likely NOT enough on its own**: every non-960/1024 row in that table is
a single fused leaf (no transpose stages), so it only exercises the
persistent-vs-serial-single-unit end of this axis. The split/transpose
case (PRE/MIDDLE/POST stages spreading `total_uthreads` across units) has
no comparable sweep varying total_uthreads specifically around the
`num_ndp_units` threshold -- `_TILE_PARALLELISM_SATURATION_UTHREADS`
(`fft_cost_model.py`'s existing 2048-uthread constant) is the closest
existing real data point, but it was measured for a different question
("does more tiles help past this size," 4 points, 2 N) and its own
comment already flags "nowhere near enough ... reread with more
benchmark_fft_candidates.sh data before trusting this far."

## The candidate formula -- unverified, flag this explicitly

`codegen/fft_transpose_codegen.py`'s `_safe_round_size` already encodes
the address-interleaving arithmetic this needs:

    active_units(total_uthreads) = min(
        num_ndp_units,
        ceil(total_uthreads / interleave_chunk_uthreads),
    )

(with the same sub-`interleave_chunk_uthreads` edge case
`_safe_round_size` calls out: below that width a round can only ever span
~2 adjacent units, not a clean ceil). This would need to be **hand-synced
into `fft_cost_model.py`**, not imported (this module is deliberately
codegen/toolchain-free -- same discipline `_stage_chunk_count`/
`_RISKY_RADIX_PAIRS` already follow for exactly this reason).

Do NOT treat this formula as settled. `_safe_round_size`'s own docstring
says outright that it rests on two claims "this project has NOT yet
independently verified end to end": (1) that a physical unit's own
assigned microthread count, not just density/gaps, is what bounds real
time, and (2) that per-launch overhead doesn't dominate at the scales
this generator produces. Both need checking (or at least the existing
persistent sweep needs re-reading with this specific question in mind)
before this shape is trusted as a cost term, not just as a compile-time
safety bound (its only currently-verified use).

## Proposed shape of the fix (for discussion, not yet decided)

1. A new `PlanMetrics` field, something like
   `min_active_ndp_unit_fraction: float` (0.0-1.0, the worst -- most
   underutilized -- stage/round in the plan: `active_units /
   num_ndp_units`, taking the min across every stage the same way
   `worst_worker_utilization` already does across leaves) or a per-stage
   breakdown feeding a new sum in `_execution_cost`, matching whichever
   shape survives the same "4 aggregation models against real
   measurements" comparison `docs/execution_cost_model_validation.md`
   already ran once for `total_worker_stage_batches` -- reuse that
   comparison methodology, don't skip straight to one model.
2. Compute it for all three shapes that need it: a persistent leaf
   (`round_active_groups`'s own logic, reused), a `PhysicalTransposePlan`
   stage (`active_units(stage.total_uthreads)`), and a plain/cooperative
   fused leaf (`active_units(codegen_plan.total_uthreads)` -- likely
   always 1 unit today, single-launch, which is exactly what makes N=256
   non-cooperative slow and is already correctly captured by
   `total_worker_stage_batches`; confirm this term doesn't double-count
   that).
3. A new `CostWeights` term, large enough that N=960/1024's persistent
   candidate stops ranking below its own non-cooperative sibling in that
   exact `docs/persistent_representative_sweep.md` table -- the same "did
   this fix the known-inverted case" bar `tile_oversaturation_penalty`
   was held to for its own crossover.
4. Validate against the *existing* 7-N sweep table first (free, no new
   hardware time) before considering new `benchmark_fft_candidates.sh`
   runs for the split/transpose gap called out above.

## Non-goals

- Do not remove or weaken `_persistent_leaf_feasible`'s existing
  structural gate as part of this work. Even a well-calibrated cost term
  is a *ranking* signal, not a hard-exclusion one -- same reasoning
  [[fft-spill-hard-filter]] already established for `radix_risk_score`
  vs. a confirmed `spill_free=False`. The gate stays regardless of how
  good the new term turns out to be; this is additive.
- Not attempting persistent's split-recursion gap (`docs/
  persistent_representative_sweep.md`'s own "Split support: investigated,
  not implemented" section) -- explicitly out of scope there already,
  unrelated to this cost-model gap, and already has its own stated
  prerequisite (find a real N whose natural split gives a near_fft
  replica count close to 32) before it's worth revisiting.
- Not a combinatorial new search axis -- like compute_lanes (Phase 7-1),
  this is a cost-model term over plans the search *already* generates,
  not a new `generate_candidates` step.

## Stopping condition

A good stopping point: the new term (a) reproduces the correct
non-cooperative-beats-persistent ranking on N=960/1024 using only the
already-existing sweep data, (b) leaves every one of the 5 small-N wins
in that same table ranked correctly (persistent still cheaper), and (c)
passes a `verify_fft_execution_cost.py`-style regression check pinned to
that table, the same discipline `check_compute_lanes_differentiated_by_
chunk_count` just established for Phase 7-1. Do not chase the
split/transpose sub-case's own missing real-hardware sweep as a blocker
for landing (a)-(c) -- flag it as a known-incomplete follow-up the same
way this doc flags it now, not a gate on shipping the persistent-leaf fix
that already has real data behind it.

## Update: diagnostic metric implemented (2026-08-31, later session)

User instruction for this pass, precisely: implement active-NDP-unit-count
as a **diagnostic metric only** (no `CostWeights`/`PlanMetrics` wiring
yet), independently **re-derive** the µthread-ID -> interleave-chunk ->
NDP-unit mapping from source instead of trusting `_safe_round_size`'s own
formula, compute (logical work count, concurrently-active-unit count) for
each of non-cooperative / cooperative / persistent / transpose-PRE /
transpose-MIDDLE / transpose-POST, treat the old unsplit N=960/1024
persistent numbers as a pathology-reproduction sanity check only (never
formula-fitting data), and defer the production cost formula until real
remeasurement exists at a fixed split across execution strategies.

**Delivered**: `planning/fft_unit_utilization.py` (new module,
`compute_unit_utilization(plan, target) -> list[UnitUtilizationEstimate]`)
and `verification/verify_fft_unit_utilization.py`. Explicitly diagnostic --
nothing in `fft_cost_model.py` imports it, no `PlanMetrics`/`CostWeights`
field reads it, same "exposed for offline comparison, not consumed by
estimate_cost" status `StageExecutionMetrics` already has.

**Re-derivation, not assumption** -- read directly from the real
M2NDP-Detour C++ source rather than trusting the Python-side comments or
`_safe_round_size`'s own (self-admittedly unverified) formula:

- `M2NDPConfig::get_matched_unit_id` (`third_party/m2ndp-detour/src/
  m2ndp_config.h:91-92`): `unit(addr) = (addr / m_stride_size) %
  m_num_ndp_units`.
- `m_stride_size = 256` (`m2ndp_config.h:350`) -- confirms
  `TargetProfile.mapping_stride_bytes` independently, not by re-reading
  the same Python comment that already claimed it.
- `M2NDPConfig::get_uthread_size` (`m2ndp_config.cc:112-121`): the real
  per-packet loop a launch's own unit-count actually comes from.
- `PACKET_SIZE = MEM_ACCESS_SIZE = 32` (`common_defs.h:12,43`) --
  confirms `TargetProfile.uthread_bytes`.

`fft_unit_utilization.distinct_active_units` is a closed-form
reimplementation of `get_uthread_size`'s own loop, verified against a
brute-force reimplementation of that literal loop across 200,000 random
(base offset, launch size, chunk, unit count) combinations including
wraparound (`verify_fft_unit_utilization.check_distinct_active_units_
matches_brute_force`) -- not merely asserted correct.

**A genuinely new finding surfaced by doing this properly**: the address-
interleaving formula only gives an exact answer when the launch pool's
real base address is known to be chunk-aligned. Two families (cooperative,
persistent) ARE confirmed exactly 256B-aligned by two real, already-fixed
bugs ([[fft-cooperative-worker8-pool-alignment]] and `docs/
persistent_recursive_split.md`'s own point 5) -- `alignment_exact=True`,
best==worst. The other two (plain non-cooperative, every `PhysicalTransposePlan`
PRE/MIDDLE/POST stage) only ever got the *default* `Pool.alloc`'s 64-byte
(2-microthread) alignment, never audited for this specific question before
-- `alignment_exact=False`, a genuine best/worst range reported honestly
rather than assuming 0 offset. This uncertainty was not previously
documented anywhere in this project; it is now, in
`fft_unit_utilization.py`'s own module docstring.

**Per-family logical work count and mechanism** (see the module's own
docstrings for the full reasoning):

- non-cooperative / cooperative: `codegen_plan.total_uthreads` (the real
  physical launch width already includes any cooperative worker-count
  scaling) fed through `distinct_active_units`.
- persistent: `FFTLeafPlan.r` (logical replica count -- `flatten_
  recursive_node` itself drops this, so a separate tree walk preserves it)
  fed through the *already-existing* `round_active_groups`/`num_rounds`
  (`fft_plan_persistent.py`), reused rather than re-derived: persistent's
  own `workers_per_group == target.interleave_chunk_uthreads` invariant
  (asserted at build time) means `software_group_id() == physical_unit_id`
  directly, once the confirmed-exact 256B alignment above holds -- no
  separate address-interleave arithmetic needed for this family.
- transpose PRE/MIDDLE/POST: `stage.total_uthreads`, tagged by structural
  recursion position (nothing in `PhysicalTransposePlan` itself carries a
  PRE/MIDDLE/POST label -- position in the recursion IS the label, so this
  needed its own tree walk rather than reusing `flatten_recursive_node`'s
  untagged flat list).

**Sanity checks run, both confirmed passing, neither used for fitting**:

1. *Old unsplit pathology, reproduced honestly.* A standalone
   `make_persistent_leaf_plan(960, num_logical_blocks=1, ...)` (the exact
   shape `docs/persistent_representative_sweep.md` measured at ~50x
   slower) reports `active_units=1` of 32 -- confirming the diagnostic
   correctly explains *why* in active-unit terms. This number is not fed
   into any weight or formula anywhere; it exists only to confirm the
   metric isn't nonsense on the one case this project already has an
   extreme, well-understood real-hardware answer for.
2. *Real split-level comparison, checked for consistency.* `docs/
   persistent_recursive_split.md`'s own real-hardware numbers -- at the
   time this check was written, reported as parity (1587 vs. 1594 cycles
   at N=960; 7688 vs. 7688 at N=1024) -- predicted the diagnostic should
   show comparably-high active-unit counts for both strategies' `far_child`
   (the leaf carrying most of the work) at this split, not the ~30x-apart
   shape the unsplit case shows. Confirmed: N=960 far_child -- persistent
   32/32, non-cooperative 30/32; N=1024 far_child -- both 32/32. **That
   "parity" premise was itself corrected 2026-08-31 -- see "Phase 1.5"
   below** -- the real relationship is persistent ~18-19% faster, not
   parity, but the active-unit consistency check's own conclusion is
   unaffected (both directions -- exact parity or a modest persistent
   edge -- are consistent with "both near-saturated," just not with the
   30x-apart unsplit shape). Interesting side finding, also not used for
   fitting: the *near_fft* leaf (only 4 replicas) shows persistent
   activating 4 units against non-cooperative's 1 -- a real structural
   edge for persistent at small leaf size that the corrected ~18-19%
   total-cycle gap shows is simply not where either plan's bottleneck sits
   at this N (the far_child and transpose stages dominate total cycles
   here).

**Why the production formula is still not decided, per explicit
instruction**: `docs/persistent_recursive_split.md` has exactly 2 real
split-level data points (N=960 forward, N=1024 inverse; persistent vs.
non-persistent only -- no cooperative in that comparison, no `num_logical_
blocks` variation beyond whatever the natural split produces). That is
consistency-check material, not a fitting set. Before any `CostWeights`
term is added: (1) extend that same real-split remeasurement across more
N and to include cooperative, not just persistent-vs-non-persistent; (2)
only then pick and calibrate an aggregation shape, the same "compare
against real measurements before choosing a model" discipline `docs/
execution_cost_model_validation.md` already used once for `total_worker_
stage_batches`. The `_persistent_leaf_feasible` structural gate stays
exactly as-is regardless -- unchanged by any of this work.

## Phase 1 findings: real runtime alignment audit (2026-08-31, real hardware)

User instruction: before touching alignment behavior at all, confirm
whether it's a real problem. Instrumented `generate_recursive_fft_kernels`
with an opt-in `debug_print_pool_alignment` flag (default `False`, no
behavior change for any existing caller) that prints each stage's own
`pool{i}` runtime address mod 64/128/256/2048 right after allocation --
observation only, no simulator source touched, no alignment behavior
changed. Driver: `benchmarks/fft/probe_pool_alignment.py` (builds a real
plan for one N/strategy combination with the flag on);
`benchmarks/fft/analyze_pool_probe.py` cross-checks the observed addresses
against `fft_unit_utilization`'s predictions. Real M2NDP-Detour
build+run, N in {512, 960, 1024} (960/1024 already the project's own
canonical split-heavy N; 512 added for a different near/far balance) x
strategy in {non-cooperative, cooperative(workers=4), persistent},
`compute_lanes=4`/`narrow_middle_stages=True` (the real shipped default --
the first probe attempt omitted these and every config panicked/spilled
uselessly; re-run with the correct defaults produced valid cycles).

**Q1: does the real allocator/runtime implicitly give >64B alignment?**
No. Every non-cooperative/transpose pool observed one of `mod256 in {0,
128, 192}` (i.e. `base_offset_uthreads in {0, 4, 6}`, all even -- the 64B/
2-microthread granularity `Pool.alloc` actually guarantees, no more, no
less). Cooperative/persistent pools were `mod256 == 0` in every single
observation (9/9 configs) -- the existing explicit alignment fix holds
perfectly on real hardware, exactly as designed.

**Q2: does alignment vary run-to-run?** No -- the earlier same-config
repeat-run check (2 reps x 9 configs, first probe pass) showed byte-for-
byte identical addresses every time. The allocator is fully deterministic
given the same generated kernel; alignment only changes across *different*
N/strategy combinations (different cumulative `cxl_alloc` sizes ahead of a
given pool), never across repeated runs of the same one.

**Q3: does the alignment difference change `distinct_active_units`'s own
prediction?** Yes, empirically, not just hypothetically: of the 15 real
non-cooperative/transpose `(stage, config)` observations, **6 (40%)**
produced a different active-unit count than the `base_offset_uthreads=0`
assumption would predict -- every single differing case landed exactly on
the already-computed `active_units_worst` bound, never off it. E.g. N=1024
stage 0 (PRE transpose, 64 real microthreads): offset-0 prediction is 8
active units, the real observed offset (4) gives 9 -- a 12.5% swing on
that one launch. This is not a rare edge case in this sample.

**Q4: does the active-unit difference show up in measured cycles?**
**CORRECTED 2026-08-31 -- the original answer below was itself built on
the buggy cycle measurement Phase 1.5 found and fixed; see that section.**
Original (now-retracted) reasoning, kept for the record: "N=960 and
N=1024 non-cooperative total cycles came out nearly identical (2114 vs.
2111) ... consistent with total plan cycle time being dominated by a few
sequential launches." Those two numbers (2114, 2111) were both just the
LAST kernel struct's (POST transpose, N-independent-ish at this scale)
own duration under the retired `tail -1` convention -- of course they
looked nearly identical, and of course that observation couldn't
distinguish anything, because it wasn't measuring the whole plan at all.
With the corrected totals (N=960 non-cooperative=48320, N=1024
non-cooperative=49061 -- see Phase 1.5), Q4 is still not cleanly isolated
by a controlled single-variable experiment (same plan, same
`total_uthreads`, deliberately different padding to force a different
`base_offset_uthreads`) -- that remains a real, not-yet-done next step --
but the Phase 2 dataset (Phase 1.5's own table) now shows LEAF EXECUTION
STRATEGY has a large, real effect on total cycles (cooperative ~30% faster
than non-cooperative, persistent ~18-19% faster) at a fixed split, which
is a different and much bigger signal than the sub-unit alignment
question Q4 originally asked about. Whether the specific alignment-driven
active-unit swings (8 vs. 9 of 32) matter on top of that is still open.

**Conclusion for now**: the honest-uncertainty design in `fft_unit_
utilization.py` (best/worst range for non-cooperative/transpose,
`alignment_exact=False`) was the right call, not overcaution -- the
worst-case bound is real and hit 40% of the time in this sample, not a
theoretical tail case. No alignment behavior was changed (per instruction
-- observe first). Whether to actually *fix* the 64B->256B gap for
non-cooperative/transpose (mirroring the existing cooperative/persistent
fix) is a separate decision, not attempted here.

## Phase 1.5 (2026-08-31): found and fixed a much bigger bug -- the "measured cycles" methodology itself

While building Phase 3's per-launch table, parsing the raw Gantt log for
N=960 non-cooperative in detail (not just grepping the last line) surfaced
a serious, previously-unknown bug in how this entire project has measured
"real ndp cycles" for any **split** (multi-kernel) plan -- not specific to
active-NDP-units work at all, but foundational to every real-hardware
comparison this project has ever made for an N requiring
`make_recursive_transpose_plan`'s actual recursion.

**The bug**: `src/m2ndp.mojo`'s `Self.launch()` spawns a **brand new
`m2ndp_run` subprocess for every top-level kernel struct** in a recursive
plan (PRE, near leaf, MIDDLE, far leaf, POST each get their own process).
`M2NDPConfig::m_ndp_cycle` (third_party/m2ndp-detour/src/m2ndp_config.h)
starts at 0 in each fresh process -- confirmed directly in a real run log:
`Host 0 Registered task .../task.elf id 0 at core cycle 0 ndp cycle 0`
appears once per distinct struct. Multiple `.launch()`-driven stages of
the SAME struct (e.g. one leaf's several `stage_N` launches) DO share one
process/clock and accumulate correctly -- only cross-struct boundaries
reset. `benchmark_fft_candidates.sh` and `planning/spill_probe.py`'s own
`_parse_ndp_cycles` (this project's two real places that extract "total
ndp cycles") both used `grep "Gantt info:.*finished NDP kernel" |
grep -oP 'ndp cycle \K[0-9]+' | tail -1` -- the single LAST match across
the whole log -- which, for a split plan, is only ever the LAST struct's
(typically a POST transpose) own standalone duration. Every earlier
struct's real cost (often the dominant one) was silently discarded.

**Magnitude, confirmed on a real N=630 non-cooperative run**: 5 structs,
own final cycle values 1872 (Pre0) / 26517 (Near0, cumulative across its
own 4 launches) / 2428 (Mid0) / 2562 (Leaf1) / 1867 (Post0). Old `tail -1`
reported **1867**. Correct total (sum of each struct's own final value):
**35246** -- ~19x higher. A single-kernel (unsplit) plan was never
affected: one struct, one process, `tail -1` and "sum of one group"
coincide there.

**Fix**: `planning/spill_probe._parse_ndp_cycles` and a new
`sum_ndp_cycles()` shell function in `benchmark_fft_candidates.sh` now
group Gantt lines by `Registered task` boundaries and sum each group's own
last value. Regression coverage: `verification/
verify_fft_ndp_cycle_parsing.py` (4 synthetic-log checks, including the
exact real N=630 values above). No simulator source touched -- this is a
log-parsing fix in this project's own Python/shell tooling.

**Re-measured this session's own Phase 1/2 data with the fix** (same
splits, `compute_lanes=4`/`narrow_middle_stages=True` shipped defaults):

| N | non-coop | coop2 | coop4 | coop8 | persistent |
|---|---:|---:|---:|---:|---:|
| 512 | 45814 | 28753* | 30926* | 31058* | 35868 |
| 630 | 35246* | 28679* | 24400* | 24400* | 27886 |
| 960 | 48320 | 34805* | 32311* | 33616* | 39378 |
| 1024 | 49061 | 32054* | 34226* | 34618* | 39730 |

(* = spill warning present; every persistent and non-cooperative row is
spill-free.) This **completely reverses** the Phase 2 conclusion this doc
originally drew from the buggy numbers ("cycles barely move across
strategies"): cooperative is genuinely **~28-37% faster** than
non-cooperative despite spilling, persistent is genuinely **~18-22%
faster** than non-cooperative and always spill-free, at every N tested.
The old "doesn't move" observation was an artifact of always measuring
the SAME strategy-invariant POST-transpose kernel alone.

**Historical docs corrected** (each now has its own "CORRECTION" section,
not silently edited in place):

- `docs/persistent_recursive_split.md` -- the "parity" claim (1587 vs.
  1594 at N=960; 7688 vs. 7688 at N=1024) is wrong; real relationship is
  persistent ~18-19% faster than non-persistent at the same split, and
  cooperative faster still (a comparison that doc never made at all).
- `docs/execution_cost_model_validation.md` -- the "measured total cycles
  do not move at all" claim for N=512/1024 lopsided plans is wrong (real
  data: ~30-37% movement); its own "DRAM-bound, not leaf-compute-bound"
  conclusion for those N does not survive. Separately, its own Phase 3/6
  section (the 55-real-candidate dataset that is the empirical basis for
  **adopting `total_worker_stage_batches` as the actual production
  `_execution_cost`**) is flagged HIGHEST PRIORITY / NOT YET RE-VERIFIED:
  at least one of its 14 N (`N=960/split=30`) is confirmed split, and no
  reproducer script survives in the repo to cheaply rerun it. This is the
  single biggest open item this correction pass surfaced but did not
  close.
- `docs/persistent_representative_sweep.md` -- the N=960/1024
  non-cooperative/cooperative rows (2114, 2107, 2111) are wrong (real:
  48320, ~34-38K depending on strategy, 49061); the unsplit-giant-leaf
  persistent values (109074, 110976) are confirmed UNAFFECTED (single
  task/process each, directly reconfirmed by rebuilding and rerunning the
  N=960 case: 109074, byte-for-byte, PASS). The "~50x slower" persistent-
  vs-split-baseline framing is corrected to **~2.26x slower** -- still a
  real loss, not a catastrophic one -- which reopens (does not resolve)
  the question of whether `_persistent_leaf_feasible`'s hard structural
  gate is still the right mechanism at that magnitude vs. letting cost-
  based ranking handle it.
- `planning/fft_cost_model.py` -- `CostWeights.transpose_tile_count`/
  `tile_oversaturation_penalty` and `_TILE_PARALLELISM_SATURATION_
  UTHREADS`'s own real-cycle citations (N=1024/960/630 tile sweeps) are
  flagged suspect (comment-only change, weights themselves untouched) --
  same bug, not yet re-verified; the qualitative direction may still
  hold (tile size is a transpose-only parameter and POST transpose's own
  duration does scale with it) but the exact magnitudes/thresholds do not
  have a confirmed-correct basis right now.

**What was NOT done this pass**: a full re-run of the 55-candidate
Phase-3/6 validation dataset (no surviving reproducer, would need
rebuilding from scratch); re-deriving `transpose_tile_count`/
`tile_oversaturation_penalty` from a corrected tile sweep; auditing every
remaining doc under `docs/` exhaustively (the ones checked -- `persistent_
vs_cooperative_findings.md`, `compute_lanes_spill_avoidance.md`'s N=216
table, `persistent_vs_cooperative_comparison_task.md`'s N=64 table -- use
only single-leaf N and are confirmed unaffected). Active-NDP-units Phase
2 (fair split-level execution-strategy dataset) is effectively
**complete and correct** as a side effect of this fix -- the table above
IS that dataset, just discovered via a detour through a bigger bug.

## Phase 3 (2026-08-31): per-launch utilization table

Built from the SAME run logs Phase 1.5's corrected-total sweep already
produced (`/tmp/phase2_runs/N{512,630,960,1024}_{noncoop,coop2,coop4,
coop8,persistent}/run.log`) -- no new hardware time needed, just parsing
each log's own `Registered task`-delimited groups instead of only the
group-sum total. Raw data: `docs/phase3_per_launch_data.csv`. One row per
launch (PRE / near leaf / MIDDLE / far leaf / POST), columns: N, strategy,
family, logical_work (uthreads, or replicas for persistent), active_best/
active_worst (from `fft_unit_utilization`), the last (=own, since within-
group cycles already accumulate) cycle value, and how many `.launch()`
calls shared that group.

**Headline finding, N=960 (representative of all 4 N -- same pattern
throughout)**:

| strategy | near_fft (4 replicas) cycles | far_child (240 replicas) cycles |
|---|---:|---:|
| non-cooperative | 39419 | 1394 |
| cooperative w=2 | 25942 | 1391 |
| cooperative w=4 | 23409 | 1389 |
| cooperative w=8 | 24749 | 1395 |
| persistent | 22831 | 9090 |

The **near_fft leaf, not the far_child, is the real bottleneck** at every
N tested here -- non-cooperative near_fft alone (39419) is larger than
this whole plan's PRE+MIDDLE+POST+far_child combined (2110+3283+2114+1394
= 8901). This was invisible under the old buggy measurement (which only
ever saw POST's own ~2114). Every leaf-execution-strategy win this
project has found for this shape comes from shrinking near_fft, not
far_child (already efficiently parallel at 30-31/32 units even
non-cooperative, since 240 replicas comfortably exceeds 32 units).

**Persistent's own real story, now visible per-launch**: wins big on
near_fft (22831, cheapest of all 5) but LOSES on far_child (9090, ~6.5x
worse than every other strategy's ~1390-1400) -- persistent's round-based
overhead actively hurts an already-parallel leaf instead of helping it.
Net total (39378) lands between cooperative (best) and non-cooperative
(worst) precisely because it wins where the bottleneck is and loses where
it wasn't one. This is a genuinely new, real finding no prior doc stated
(the old "parity" framing had no visibility into per-leaf cost at all).

## Phase 4 (2026-08-31): wave model -- holds for persistent, not for plain/cooperative

Checked whether `waves = ceil(logical_replicas / active_physical_units)`
(no fitted coefficient) explains the per-launch cycles above.

**Persistent (its own explicit round design)**: `waves = num_rounds
(replicas, 32)` is architecturally exact, not estimated --
`round_active_groups`/`num_rounds` already are this formula. near_fft
(4 replicas, 4 active units): `waves=ceil(4/4)=1` -- one round, matches
being the cheapest near_fft variant. far_child (240 replicas, 32 active
units): `waves=ceil(240/32)=8` -- eight rounds, and indeed 24 `.launch()`
calls appear in that group (persistent's own preload/stage_N/writeback
phase structure per round -- `n_launches_in_group=24` matches `8 rounds x
~3 phases`, consistent with the round design's own known launch-count
formula). The 8-round overhead is the direct, structural explanation for
far_child's 9090 -- not a mystery, a predictable consequence of a small
active-unit-count-relative-to-replicas ratio forcing multiple sequential
rounds.

**Plain/cooperative (no round mechanism)**: the naive wave formula does
NOT explain the data on its own. Cooperative worker=2's near_fft has the
SAME active-unit count as non-cooperative (both `distinct_active_units=1`,
since 4 and 8 total_uthreads both fit in one 8-wide interleave chunk) --
identical wave count under the naive formula -- yet real cycles differ by
34% (25942 vs. 39419). The missing piece is NOT active-unit count at all
here: cooperative's own `workers_per_fft` also divides the SIMD-batch work
*within* one physical unit, a dimension `fft_cost_model.
StageExecutionMetrics.max_batches_per_worker`/`total_worker_stage_batches`
already tracks (Phase 7-1's own subject) and `distinct_active_units` was
never meant to. **Conclusion**: "waves" is the right concept specifically
for persistent's own architecturally-distinct round mechanism; for plain/
cooperative launches, the equivalent already exists (`total_worker_stage_
batches`) and active-NDP-units is a genuinely separate, multiplicative
factor on top of it, not a replacement or an alternative formulation of
the same thing.

## Phase 5 (2026-08-31): physical parallelism vs. cycles, real data

1. **Same work, more active units => generally fewer cycles**: near_fft
   non-coop (1 unit, 39419) -> coop2 (1 unit, 25942, via within-unit
   worker split, not more units) -> coop4 (2 units, 23409). Direction
   holds through 2 units.
2. **1/2/4 scaling, NOT cleanly monotonic past that**: coop8 (4 units,
   24749) is WORSE than coop4 (2 units, 23409) -- confirms this project's
   own pre-existing, separately-documented `workers_per_fft=8`
   liability (the exact-one-interleave-chunk case with its own known
   correctness/performance quirks) rather than a new finding, but this is
   the first time it shows up as a *cycle regression*, not just a
   correctness risk.
3. **Saturation point**: not reachable within this dataset -- near_fft
   only has 4 replicas, so active units never exceeds 4 regardless of
   strategy; far_child already sits at 30-32/32 (near-saturated) at every
   strategy except persistent's round-based 32 (same peak, different
   overhead). A real saturation-point sweep needs a leaf with enough
   replicas to push active units well past 32 while varying strategy --
   not available in this 4-N sample, flagged as a real gap.
4. **Same active-unit count, different strategy**: coop8 (4 units,
   24749) vs. persistent (4 units, 22831) on the SAME near_fft leaf --
   persistent ~8% faster at an IDENTICAL active-unit count. Confirms
   active-unit count alone is not sufficient to predict cycles even
   holding it fixed; the *mechanism* (address-interleave-driven
   cooperative grouping vs. persistent's own software-group-is-physical-
   unit design) carries a real, separate cost difference.
5. **Remaining gap explained by existing terms?** Partially. Cooperative
   worker-count's within-unit effect is exactly `total_worker_stage_
   batches`'s own subject (already in `_execution_cost`, Phase 7-1
   already extended it for compute_lanes). The persistent-vs-cooperative-
   at-equal-active-units gap (point 4) is NOT yet explained by anything
   in `fft_cost_model.py` -- plausibly synchronization/round-transition
   overhead specific to each mechanism, not modeled anywhere today. Not
   resolved this pass.

## Phase 6 (2026-08-31): mixed leaf strategies, real hardware

N=960, forced per-leaf strategy via `forced_worker_sequence` (near_fft
first, far_child second), default split/radix/tile held fixed throughout:

| near_fft | far_child | real total cycles | spill |
|---|---|---:|---|
| non-coop | non-coop | 48320 | no |
| persistent | persistent | 39378 | no |
| **persistent** | **non-coop** | **31689** | **no** |
| non-coop | persistent | 55985 | no |
| cooperative w=4 | persistent | 39977 | YES |
| persistent | cooperative w=4 | 31689 | no |

**Confirms the Phase 3 prediction exactly**: applying persistent only
where the real bottleneck is (near_fft) while leaving the already-
efficient far_child alone gives **31689** -- better than every single
*uniform* (same-strategy-for-both-leaves) candidate this session measured
for this N, including pure cooperative (34805, Phase 1.5's own table) and
pure persistent (39378). Applying persistent to the WRONG leaf
(non-coop/persistent -- leaving the real bottleneck untouched while
paying persistent's round overhead on the already-fine far_child) is
worse than doing nothing (55985 vs. 48320) -- confirms this isn't "more
persistent is always better," it's specifically about matching the
mechanism to which leaf actually needs it. Once near_fft is persistent,
far_child's own choice (non-coop vs. cooperative w=4) doesn't matter
(31689 both ways) -- consistent with Phase 3's finding that far_child was
never compute-bound to begin with.

This result was cross-validated for free: `generate_candidates(960)`'s
own step-9 joint search already offers `worker_sequence=('persistent',
None)` as a real candidate (not something this phase invented) -- built
and run independently via `benchmark_fft_candidates.sh --max-candidates
12` the same session, it measured **31758** cycles, matching this
section's own hand-built **31689** to within 0.2% (real build/measurement
noise, not a discrepancy). Existing correctness coverage already confirms
this exact combination is numerically correct at N=960
(`verify_fft_execution_invariants.py`: `forced_worker_sequence=
('persistent', None)` N=960, max error 7.7e-08).

**Caveat**: this is the best mixed *execution-strategy* candidate at the
DEFAULT split/tile. The same `benchmark_fft_candidates.sh` run found an
even better candidate overall (19249 cycles, a plain non-cooperative plan
at a *different* `split_sequence`/`tile` choice, no execution-strategy
trick at all) -- confirming this project's own existing multi-axis joint
search architecture (split x radix x tile x worker x persistent) is
doing real, necessary work, and no single axis (execution strategy alone)
tells the whole story. Phase 6's own finding stays valid and useful
*within* the execution-strategy axis; it is not a claim that mixed
persistent is the global optimum for N=960 overall.

## Phase 7 (2026-08-31): why estimate_cost misses this today -- a smaller fix than originally scoped

Checked directly whether `fft_cost_model.py`'s EXISTING `total_worker_
stage_batches` mechanism (no new PlanMetrics field) already reflects the
real near_fft-is-the-bottleneck finding, before designing a new
aggregation model from scratch:

| N=960 config | `total_worker_stage_batches` | `_execution_cost` | `_memory_cost` | ratio (`_execution_cost` / total) |
|---|---:|---:|---:|---:|
| default (non-coop) | 33 | 3.3 | 76690.0 | 0.00004 |
| cooperative w=8 | 6 | 0.6 | 76690.0 | 0.00001 |
| persistent | 6 | 0.6 | 76690.0 | 0.00001 |

**`total_worker_stage_batches` already correctly detects the improvement
direction and a good chunk of its relative magnitude**: it drops 5.5x
(33 -> 6) switching from non-cooperative to either cooperative w=8 or
persistent -- consistent with near_fft's own real cycle drop (Phase 3:
39419 -> ~23-25K, roughly 1.6-1.7x, smaller than 5.5x because real cycles
also include fixed per-launch overhead this static batch-count metric
doesn't model, but the *direction* and *that leaf strategy matters a lot
here* are both already right). **The reason this never influences
`estimated_cost`'s own ranking is `CostWeights.stage_work=0.1`'s absolute
size, not a missing metric**: `_memory_cost` for this N is ~76690
regardless of leaf strategy (same DRAM bytes, same transpose tile shape),
so a 3.3 -> 0.6 swing in `_execution_cost` changes the total by <0.004%
-- invisible next to memory cost's own uncertainty, let alone its
magnitude. This exactly explains why plans 10/11 in the real
`benchmark_fft_candidates.sh` N=960 run (workers=8, persistent) scored
`estimated_cost=76691.1` each, statistically indistinguishable from the
plain baseline's ~76693.8.

**Revised understanding of what "the fix" actually is**: not necessarily
a brand-new `PlanMetrics` active-unit field feeding a brand-new
`CostWeights` term (Phase 7's original scoping assumed the underlying
metric didn't exist at all) -- `total_worker_stage_batches` already IS
close to the right shape of signal for the *leaf* dimension. What's
missing is (a) `stage_work`'s own weight being calibrated for
`_compute_cost`'s flat per-stage-count term, not sized to let
`_execution_cost` meaningfully compete with `_memory_cost` when leaf
execution strategy is the thing varying, and (b) `PhysicalTransposePlan`
stages (PRE/MIDDLE/POST) are entirely excluded from `compute_stage_
metrics` (`if isinstance(node, PhysicalTransposePlan): continue`), so
none of their own real cost (~7500 of N=960's ~48320 real total, roughly
constant across leaf strategies -- confirmed in Phase 3's table) is
modeled by `_execution_cost` either; `_memory_cost`'s own DRAM-byte
estimate is the only thing standing in for it today, uncalibrated against
Phase 3's own real per-transpose-stage cycle counts.

**A plan_time formula that matches Phase 1-6's own real data**
(conceptual, not yet a `CostWeights` change):

    plan_time ~= sum(transpose_stage_time for PRE/MIDDLE/POST)
               + sum(leaf_execution_time for near_fft, far_child, ...)

    transpose_stage_time(stage)  ~ f(stage.total_uthreads, active_units(stage))    -- NOT modeled by any per-stage term today
    leaf_execution_time(leaf)    ~ g(total_worker_stage_batches contribution) x (some per-strategy overhead correction)

Both `f` and `g` need real calibration against Phase 1-6's own dataset
(20+ real per-launch cycle values already collected, `docs/
phase3_per_launch_data.csv`) before being trusted as `CostWeights`
constants -- not done this pass; this is the concrete next step for
whoever picks this back up, not a finished formula.

## Phase 8 (2026-08-31): production integration -- conditions only PARTIALLY met, do not integrate yet

Checking this doc's own stated conditions against everything Phases 1-7
actually established:

- **active-unit prediction verified against real mapping**: YES
  (Phase 1, brute-force + real hardware).
- **alignment uncertainty understood**: YES (Phase 1: real, ~40% hit
  rate on the worst-case bound, not theoretical).
- **split-level strategy comparison data sufficient**: PARTIAL. 4 N x 5
  strategies + 5 mixed-leaf combinations (Phases 2, 6) is real and
  informative, but Phase 5's own saturation-point question (active units
  well past 32) has zero data points -- no leaf in this sample has enough
  replicas to need it.
- **active-unit/wave metric direction vs. cycle confirmed**: YES for
  persistent (`waves` is architecturally exact there) and directionally
  for plain/cooperative (Phase 5), but Phase 5 point 4 also found a real,
  unexplained gap even at matched active-unit counts (persistent vs.
  cooperative w=8) -- "confirmed" is doing some work here, not "fully
  explained."
- **N=960/1024 predicted in the correct direction**: YES, but not by a
  new active-unit cost term -- by the ALREADY-EXISTING `total_worker_
  stage_batches`, once its weight is reconsidered (Phase 7). This
  condition is met in spirit, not by the mechanism this doc originally
  expected to build.
- **small/single-leaf persistent's own real win not overly suppressed**:
  **NOT CHECKED THIS PASS.** Any reweighting of `stage_work` (Phase 7's
  own proposal) changes `_compute_cost` too (same weight, different
  term) -- whether that breaks the N=216-style small-N persistent win
  (`docs/compute_lanes_spill_avoidance.md`'s own 43771/28784/23944 table)
  is a real, unverified risk of the exact fix Phase 7 points toward, not
  a solved problem.

**Verdict: do not connect anything to `PlanMetrics`/`fft_cost_model.
CostWeights` yet.** Two concrete, scoped items remain before that's
safe: (1) a real weight/calibration pass for `stage_work` (or a sibling
weight) against Phase 1-6's own already-collected per-launch data,
specifically checking it does NOT regress N=216-scale single-leaf
persistent rankings; (2) closing the Phase 5 saturation-point gap (or
explicitly deciding it doesn't matter for this project's real N range).
Neither was attempted this pass -- this doc's own scope was diagnostics
and real-data-gathering, and per Phase 8's own standing instruction, no
new arbitrary weight gets added without that calibration step actually
happening first.

## Phase A-H (2026-09-01): reproducible harness, coefficient estimation, saturation sweep, production decision

Full re-run per explicit instruction: no arbitrary weight, no new model
architecture until the smallest hypothesis (reweight `stage_work` alone)
is tested against real data and either confirmed or falsified.

### Phase A: reproducible harness (new, permanent, in-repo)

- `revalidate_cost_model.py` + `revalidate_candidates.py` -- a
  deterministic, hand-specified candidate list (NOT a reconstruction of
  the lost 55-candidate/14-N dataset -- a new one, meant to be re-run
  whenever a weight changes). 24 real candidates across N in {144, 216,
  512, 630, 960, 1024, 2048}, covering small single-leaf, mixed-leaf,
  asymmetric split, radix-5-spill, and two large-N split cases. Output:
  `docs/cost_model_revalidation.csv`/`.json`.
- `revalidate_saturation.py` -- synthetic replica sweep (1 through 256)
  at a fixed small leaf (N=64), independently for non-cooperative /
  cooperative(workers=4 request) / cooperative(workers=8 request) /
  persistent. Output: `docs/cost_model_saturation.csv`/`.json`.
- `analyze_revalidation.py` / `analyze_saturation.py` -- the actual
  Phase B/C/E computations (cycles-per-batch regression, regret/top-k/
  Spearman, saturation tables) read straight from those CSVs, runnable by
  anyone without re-deriving the analysis by hand.
- Every single-leaf-N row cross-validated exactly against pre-existing
  historical figures already in this repo's own docs before being
  trusted (N=144: 26062/17580/14713 vs. `persistent_representative_
  sweep.md`'s 26062/17580/14551; N=216: 43771/22316 vs. `compute_lanes_
  spill_avoidance.md`'s identical figures; N=64 replica=1 in the
  saturation sweep: 13133/7770 vs. the same sweep doc's 13133/7770) --
  the harness is trustworthy, not just internally consistent.
- Found and fixed a real bug in the harness itself along the way: neither
  script originally passed `spad_capacity_bytes`/`max_concurrent_
  scratchpad_bytes` explicitly, so a high-replica leaf (128/256 at N=64)
  built an uncapped, too-wide launch that failed at LINK time
  (`.spad section will not fit in region 'spad': overflowed by 48
  bytes`) -- not a simulator crash, a real scratchpad-capacity modeling
  gap when a caller omits these two parameters. Fixed in both scripts
  (now pass `DEFAULT_TARGET_PROFILE`'s own values explicitly); the main
  24-candidate revalidation never actually hit this (none of its real
  candidates need that many concurrent uthreads) so it did not need
  rerunning.

### Phase B: matched-pair cycles-per-batch coefficient -- NOT a single number

| N | pair | Δcycles | Δbatches | cycles/batch |
|---|---|---:|---:|---:|
| 144 | coop4 vs noncoop | -8482 | -14 | 605.9 |
| 144 | persistent vs noncoop | -11349 | -18 | 630.5 |
| 216 | coop4 vs noncoop | -21455 | -33 | 650.2 |
| 216 | persistent vs noncoop | -19830 | -39 | 508.5 |
| 512 | coop4 vs noncoop | -14888 | -24 | 620.3 |
| 512 | persistent vs noncoop | -9946 | -28 | 355.2 |
| 630 | coop4 vs noncoop | -10846 | -6 | 1807.7 |
| 630 | persistent vs noncoop | -7360 | -7 | 1051.4 |
| 960 | coop2 vs noncoop | -13515 | -16 | 844.7 |
| 960 | coop4 vs noncoop | -16009 | -23 | 696.0 |
| 960 | coop8 vs noncoop | -14704 | -27 | 544.6 (worker=8 anomaly, excluded from estimate) |
| 960 | persistent vs noncoop | -8942 | -27 | 331.2 |
| 1024 | coop4 vs noncoop | -14835 | -24 | 618.1 |
| 1024 | persistent vs noncoop | -9331 | -28 | 333.3 |
| 2048 | coop4 vs noncoop | -14867 | -24 | 619.5 |
| 2048 | persistent vs noncoop | -8649 | -28 | 308.9 |

Overall (excluding worker=8): median=619.5, mean=665.4, stddev=362.2,
range 308.9-1807.7. Linear regression (intercept NOT forced to 0):
`measured_cycle_delta ~= 259.8 * batch_delta - 6816.2`.

**N=630 is a genuine, explained outlier (1051-1808), not noise to
discard**: this is the project's own known radix-5-genuine-middle-stage
case (`_RISKY_AS_MIDDLE_RADICES` in `fft_cost_model.py`) with a small
`Δbatches` denominator (-6/-7), so the ratio is both mechanically
noisier and reflects a real, different spill/register-pressure regime,
not the same phenomenon the other 14 points measure. Kept, not deleted,
per instruction -- but excluded from the central "typical" estimate the
same way worker=8 is.

**The decisive finding: persistent and cooperative do NOT share one
coefficient.** Persistent's own column (630.5, 508.5, 355.2, 331.2,
333.3, 308.9) is consistently LOWER than cooperative's (605.9, 650.2,
620.3, 696.0, 618.1, 619.5) at every matched N, and persistent's own
coefficient visibly DECREASES with N while cooperative's stays roughly
flat. A single scalar `stage_work` cannot correctly represent both
mechanisms from the same `total_worker_stage_batches` count.

### Phase C: synthetic saturation sweep (N=64 fixed leaf, replicas 1-256)

Full data: `docs/cost_model_saturation.csv`. Headline table (cycles per
replica, the throughput-normalized view):

| replicas | active_units | non-coop | cooperative (w=2 actual*) | persistent |
|---:|---:|---:|---:|---:|
| 1 | 1 | 13133.0 | 7770.0 | 7676.0 |
| 8 | 1-2 | 1643.6 | 1072.9 | 963.6 |
| 32 | 4-8 | 411.0 | 273.6 | **242.1** |
| 64 | 8-16 | 205.6 | **137.1** | 186.6 (waves=2 begins) |
| 128 | 16-32 | 205.6 | 137.1 | 154.3 (waves=4) |
| 256 | 32 | **205.6** | **137.1** | 138.2 (waves=8) |

(* `cooperative_workers=8` silently resolved to the SAME actual
`workers_per_fft=2` as `cooperative_workers=4` for this specific N=64/
radix-(4,4,4) leaf -- `choose_workers_per_fft` treats the request as a
ceiling and this leaf's own legality caps out at 2 regardless. Real,
already-known planner behavior, not a harness bug -- flagged rather than
silently treated as a genuine worker=4-vs-8 comparison; this sweep does
NOT independently test worker=8 at this N.)

**Non-cooperative and cooperative both hit a floor well BEFORE 32 active
units** -- cooperative's cycles/replica is already flat at 137.1 by
active_units=16 (replicas=64), non-cooperative's flat at 205.6 by
active_units=8 (replicas=32-64). This floor is a genuinely surprising
result worth flagging honestly: **not what "saturation at 32 physical
units" alone would predict** -- something else (plausibly fixed per-
launch/host-orchestration overhead dominating this leaf's own tiny real
compute at this scale) caps throughput earlier than physical-unit count
does, for these two strategies specifically. Not explained further this
pass.

**Persistent uniquely scales cleanly all the way to the real 32-unit
boundary** (963.6 -> 242.1, continuously improving through active_units
8/16/32, no early floor), then transitions to clean, linear round-based
growth once replicas exceed 32: absolute cycles 7747 (waves=1) -> 11940
(waves=2) -> 19755 (waves=4) -> 35376 (waves=8), i.e. **~4000-4200 extra
cycles per additional wave**, remarkably consistent (4193, then 3907.5,
then 3905.25 per wave across the three transitions) -- the cleanest
confirmation in this whole investigation that `waves = ceil(replicas /
active_units)` is architecturally exact for persistent, not just
directionally right.

**Revised Phase 4 conclusion (more general than originally stated)**:
the "extra round = extra linear cost" pattern is NOT unique to
persistent's own architecture after all -- non-cooperative and
cooperative BOTH show the identical signature once THEIR OWN launch
needs multiple internal rounds for a different reason (scratchpad-
capacity capping via `_cap_max_uthread`, not NDP-unit-count saturation):
non-coop's cycles exactly DOUBLE from replica 64->128 and 128->256
(13160 -> 26320 -> 52640, matching `max_uthread` capping to 64, forcing
2 then 4 rounds) with cycles/replica dead flat at 205.6 throughout;
cooperative shows the same doubling pattern. **"Waves" is a property of
"does THIS launch need multiple host-orchestrated rounds," which every
strategy can hit for its own reason (physical-unit saturation for
persistent specifically at replicas>32; scratchpad capacity for anyone at
a high enough replica/uthread count) -- not a persistent-only concept as
Phase 4 originally concluded.** `total_worker_stage_batches` already
correctly stays constant across replica count within one round (3 for
cooperative/persistent, 6 for non-coop in this data) precisely because
it's a per-round, per-worker quantity -- the round COUNT is what's
missing from today's cost model for every strategy alike, not a
persistent-specific gap.

### Phase D: stage_work weight candidates -- Option A directly falsified

Tested `stage_work` at its current value (0.1) and at two values derived
from Phase B's own regression (260, and the near-median coefficient 620)
against the full 24-candidate dataset's own ranking. **Ranking did not
change at ANY of the three weights.** Root cause, confirmed directly:
persistent has the LOWEST `total_worker_stage_batches` at every single N
tested (e.g. N=216: persistent=9 vs. cooperative=15; N=1024:
persistent=5 vs. cooperative=9) -- true even though persistent's REAL
measured cycles are HIGHER than cooperative's at 5 of 7 N in this
dataset (see Phase E below). **Scaling one positive scalar weight on a
quantity that is already systematically smallest for the wrong winner
can only make that wrong winner look MORE attractive, never less** --
there is no `stage_work` value, however large, that fixes this. This
directly and empirically falsifies the leading hypothesis this phase was
scoped to test first.

### Phase E: ranking regression (24-candidate dataset, all 7 N)

| N | candidates | model's #1 pick | real best | regret | Spearman | best in model's top-3? |
|---|---:|---|---|---:|---:|---|
| 144 | 3 | persistent | persistent | 0.0% | 1.000 | yes |
| 216 | 3 | persistent | **coop4** | 7.3% | 0.500 | yes |
| 512 | 3 | persistent | **coop4** | 16.0% | 0.500 | yes |
| 630 | 3 | persistent | **coop4** | 14.3% | 0.500 | yes |
| 960 | 6 | coop8 | **mixed (persistent-near/noncoop-far)** | 6.1% | 0.714 | yes |
| 1024 | 3 | persistent | **coop4** | 16.1% | 0.500 | yes |
| 2048 | 3 | persistent | **coop4** | 12.8% | 0.500 | yes |

Worst-case top-1 regret: **16.1% (N=1024)**. The real best is `coop4` in
5 of 7 N -- the model's own systematic bias toward persistent (Phase D)
means it currently OVER-favors persistent, the opposite of the risk this
task's own Phase F worried about (persistent being unfairly suppressed).
Real best is never in last place under the model's own ranking (always
top-3 of 3-6), so the model is not catastrophically wrong -- but it is
consistently, specifically wrong in the same direction for split-N cases.

### Phase F: N=216 regression check

| ranking source | order (best to worst) |
|---|---|
| current model (`stage_work=0.1`) | persistent (3458.0) < coop4 (3458.8*) < noncoop (3461.3) |
| reweighted model (`stage_work=620`, Phase D) | **identical order** -- persistent still ranks first |
| actual measured | **coop4 (22316)** < persistent (23941) < noncoop (43771) |

(*cost values this close only differ in the 4th significant figure --
`_memory_cost`, identical for all three since DRAM bytes/tile shape don't
change, dominates the total at ~3456; seebelow.) N=216 is NOT a case
where reweighting `stage_work` breaks anything NEW -- the model was
ALREADY picking persistent over the real winner (coop4) before any
reweighting, and stays exactly as wrong after, because (Phase D) no
scalar reweight changes this ranking at all. This resolves Phase 8's
original worry ("does a stage_work increase suppress N=216's persistent
win?") in an unexpected direction: there is no small-N persistent win
being protected here to begin with -- the REAL winner at N=216 is
cooperative, and the model already (wrongly) prefers persistent
regardless of `stage_work`'s value.

### Phase G: mixed-leaf regression fixture

`revalidate_candidates.py`'s own `mixed_persistent_near_noncoop_far`
entry (N=960, `forced_worker_sequence=("persistent", None)`) is now a
permanent row in the reproducible dataset -- 31689 cycles, confirmed
fastest of all 6 real N=960 candidates in this run, matching Phase 6's
own hand-built result (31689) and the search's own independently-found
candidate (31758) to within 0.2%. This fixture now lives in a
repository-tracked, re-runnable script (not just prose in this doc) --
the invariant it protects ("the planner can reach a genuinely faster
per-leaf-mixed candidate than any uniform strategy") is checked every
time `revalidate_cost_model.py` runs, not just asserted once.

### Phase H: production integration decision

**D. 데이터 부족 — 이 정확한 방향(단순 stage_work 재조정)으로는 production
변경 보류. 그러나 원인은 규명되었음, "더 조사 필요"가 아니라 "이 접근은
작동하지 않는다"는 것을 실측으로 확인.**

Checking the required conditions explicitly:

1. `stage_work` coefficient has real-measurement backing? **YES, but it
   backing shows two DIFFERENT coefficients (persistent ~300-630 falling
   with N; cooperative ~600-850 roughly flat), not one -- a single
   `stage_work` constant cannot represent both.**
2. N=216 regression-free? **N/A in the intended sense -- the model was
   never correct at N=216 to begin with, and Phase D showed reweighting
   cannot fix it either way. Nothing gets WORSE, but nothing gets FIXED.**
3. N=960/representative N ranking improved or unchanged? **UNCHANGED at
   every weight tried (Phase D) -- the specific mechanism this phase set
   out to test (reweight alone) has zero effect on ranking, confirmed
   directly, not inferred.**
4. Worst-case regret not meaningfully worsened? **Unchanged (Phase D) --
   neither better nor worse; today's 16.1% worst-case stands either way.**
5. Saturation results consistent with the physical-parallelism
   interpretation? **YES, and MORE informative than expected -- see
   Phase C's revised Phase 4 conclusion (waves is general, not
   persistent-only).**
6. Existing test suites pass? **YES** (full existing `verify_fft_*.py`
   suite reruns clean after every code change this pass -- no production
   `fft_cost_model.py`/`CostWeights` value was actually changed, only
   `revalidate_*.py`'s own harness code and this doc).

**Verdict: DO NOT integrate a `stage_work` reweight into production.**
It was the leading, smallest-change hypothesis and this phase's own
explicit job was to test it before considering anything larger -- it is
now falsified by direct experiment, not merely untested. Per this task's
own stated priority order (1. reweight existing constant, 2. small
formula correction, 3. new dimension only as last resort), the
falsification of (1) means the next real candidate is **(2): a small,
mechanism-aware formula correction** -- concretely, a persistent-specific
term (e.g. `stage_work_persistent` distinct from `stage_work_
cooperative`, or an explicit `waves`-multiplier applied to whichever
strategy's own launch actually needs multiple rounds, generalized per
Phase C's own finding to cover non-cooperative/cooperative too) --
**not attempted this pass**, since it is new-formula-shaped work this
task's own scope was to gate behind exactly this falsification result.

## Mechanism-Aware Correction (2026-09-01): Model 1/2, tested against the existing datasets, no new hardware runs

Per explicit instruction: `stage_work` reweighting is a closed experiment
(falsified above, not retried). This phase tests, strictly in order,
whether the smallest possible correction on top of the UNCHANGED
production formula explains the remaining ranking failures -- stopping
the moment a small model is sufficient. New script: `analyze_mechanism_
correction.py` (reads the existing `docs/cost_model_revalidation.csv`
and `docs/cost_model_saturation.csv`, no new toolchain runs needed).

### Phase 1: production formula, confirmed from code (not docs)

`estimate_cost = _memory_cost + _compute_cost + _resource_cost +
_execution_cost`, and `_execution_cost = CostWeights.stage_work *
PlanMetrics.total_worker_stage_batches` -- exactly as this doc's own
earlier sessions already established; reconfirmed directly against
`fft_cost_model.py` before touching anything, per instruction.

### Phase 3: persistent_rounds vs total_worker_stage_batches, real correlation

Using the saturation dataset's own persistent column (9 points,
replicas 1-256): `total_worker_stage_batches` is **literally constant
(3)** across the entire sweep -- it structurally cannot correlate with
anything, since nothing about it varies with replica count once a leaf's
own stage shape is fixed. `persistent_waves` (`ceil(replicas/32)`)
correlates with real cycles at **Pearson=1.000** across the same 9
points. The Δcycles-per-extra-wave transitions (waves 1->2: 4193.0,
2->4: 3907.5, 4->8: 3905.2) are remarkably consistent: median=3907.5,
stddev=135.1 (3.5% of the median) -- about as clean as a real-hardware
measurement gets in this project.

### Phase 4/D: Model 1 -- persistent_rounds correction, coefficient derived not chosen

`round_cost_in_batch_equivalent_units = median(Δcycles/Δwave) /
median(persistent's own cycles-per-batch, single-round matched pairs)
= 3907.5 / 355.2 = 11.0`. Model 1: `execution_cost += stage_work * 11.0
* sum(max(0, num_rounds(leaf.r, 32) - 1) for every persistent leaf)` --
`stage_work` itself untouched. Applied to the leaf-level replica count
(`FFTLeafPlan.r`), not the top-level `batch` parameter -- these differ
for a split plan (N=960's far_child has `r=240` even though the whole
transform is `batch=1`), which is why this term is a genuine no-op for
every single-leaf N in this dataset (144, 216: `r=1`, `rounds=1`,
increment=0 by construction) but a real, large correction for every
split N with a big-replica leaf.

### Phase D/G: Model 2 -- persistent-specific stage coefficient, direction double-checked

Persistent's own matched-pair cycles-per-batch (median 355.2, single-
round cases) is LOWER than cooperative's (620.0, this pass's own
reference) -- meaning persistent's batch count *under-represents* its
real relative cost, not over-represents it. **Getting the correction
direction right required care**: the naive ratio (355.2/620.0 = 0.573,
"scale persistent's batches down") makes the ranking WORSE, not better
(confirmed by testing it) -- the correct direction is the reciprocal
(620.0/355.2 = 1.746, "scale persistent's batches UP to the same
honesty cooperative's already have"), which is what Model 2 actually
uses. Formula: for any persistent leaf, replace its own `stage_work *
batches` contribution with `stage_work * 1.746 * batches`, then add
Model 1's own round term on top (rounds are a structural launch-count
fact independent of the per-batch rate correction).

### Phase 5/E: N=216 litmus test

| model | pick | regret | Spearman |
|---|---|---:|---:|
| Model 0 (current) | persistent | 7.3% | 0.500 |
| Model 1 (+rounds) | persistent (unchanged -- rounds=1 here, zero effect) | 7.3% | 0.500 |
| Model 2 (+coefficient) | **coop4 (correct)** | **0.0%** | **1.000** |

Confirms analytically and empirically that Model 1 alone cannot fix
N=216 (its own persistent leaf never exceeds one round at this N, so the
correction term is always exactly 0) -- N=216 specifically needed the
coefficient correction, not the round correction. No N=216-specific rule
was written; the same globally-derived `1.746` multiplier does this.

### Phase 6/E: N=960 mixed-leaf regression check

| model | pick | regret | Spearman |
|---|---|---:|---:|
| Model 0 | coop8 | 6.1% | 0.371 |
| Model 1 | coop8 (unchanged pick, but Spearman improves) | 6.1% | 0.771 |
| Model 2 | coop8 (still unchanged) | 6.1% | 0.714 |

**Neither correction breaks the mixed-leaf candidate** (its own real
36.9% margin over the plain-persistent candidate stays fully reachable
and correctly ranked well ahead of every uniform-persistent option in
all three models) -- the required invariant ("persistent is not killed
by a blanket penalty where it's actually needed") holds throughout. The
remaining 6.1% gap (coop8=33616 vs. mixed=31689, a real 5.7% difference)
is a genuine near-tie neither correction resolves -- not investigated
further this pass; not clearly an active-unit-saturation question per
se (Phase 9/Model 3 was not attempted -- see "why we stopped" below).

### Phase 7: full 24-candidate dataset, all three models

| model | mean regret | median regret | worst regret (N) |
|---|---:|---:|---:|
| Model 0 (current production) | 10.4% | 12.8% | 16.1% (N=1024) |
| Model 1 (+ persistent_rounds) | 1.9% | 0.0% | 7.3% (N=216) |
| **Model 2 (+ persistent coefficient)** | **0.9%** | **0.0%** | **6.1% (N=960)** |

Model 2 gets **6 of 7 N to exactly 0% regret** (144, 216, 512, 630, 1024,
2048); the sole remaining nonzero-regret case (N=960, 6.1%) is
unaffected in either direction by any correction tried.

### Phase 8 (worker=8 anomaly): excluded from fitting throughout, as instructed

Every coefficient in Model 1/2 was derived exclusively from non-coop-
vs-cooperative(w=4)/persistent matched pairs and the persistent-only
saturation curve -- `cooperative_workers=8` data was never used to fit
anything here (consistent with the prior pass's own finding that
worker=8 is a separate, already-known non-monotonic anomaly). Nothing
in Model 1/2 is conditioned on or corrects for worker=8 specifically;
its own already-anomalous candidates (N=960 coop8) are scored by the
SAME formula as everyone else and simply keep ranking exactly where
their own real cycles put them.

### Why Model 3 (effective-active-unit correction) was NOT attempted

Per the task's own explicit gating ("Model 2로 부족할 때만 진행"): Model 2
already achieves 6/7 exact and a worst-case regret of 6.1%, down from
16.1% -- a 62% reduction, achieved with two data-derived terms and zero
new arbitrary constants. The one remaining gap (N=960's own 6.1%, a
genuine near-tie) has no clear, already-demonstrated link to the earlier
saturation sweep's own "early floor at active_units=8-16" finding (that
finding was for a synthetic small leaf at N=64, not shown to generalize
to N=960's own far_child at 240 replicas / 30-31 active units, already
well past where the N=64 sweep's own floor appeared). Inventing an
`effective_active_units` correction to chase a 6.1% residual on a single
N, without a demonstrated mechanistic link, would be exactly the kind of
unjustified complexity this task explicitly forbids. Stopped here per
Occam's razor, as instructed.

### Final decision

**G: B -- persistent_rounds + strategy-specific coefficient (Model 2) is
needed, and appears SUFFICIENT for the dataset in hand.** Neither
Model 0 (current) nor Model 1 alone clears the N=216 condition; Model 2
clears every explicit condition this task listed except worst-case
regret being driven to exactly 0 (it isn't, at N=960, by design -- not
attempted further, per Occam's razor):

1. Mechanism-explainable? YES -- persistent's own preload/writeback
   round structure (Model 1) and its own genuinely different per-batch
   real-cycle rate (Model 2) both have direct, already-documented
   mechanistic grounding (`round_active_groups`/`num_rounds`, and the
   matched-pair regression), not fitted-then-rationalized.
2. N=216 improved? **YES -- 7.3% -> 0.0% regret, correct pick.**
3. N=960 mixed-leaf regression-free? **YES -- unaffected, still
   reachable, still correctly ranked far ahead of uniform strategies.**
4. Overall regret improved? **YES -- mean 10.4% -> 0.9%.**
5. Worst-case regret not worsened? **YES -- 16.1% -> 6.1%, improved.**
6. Saturation data consistent with the model's own interpretation?
   **YES -- Pearson=1.000 between waves and cycles is the direct basis
   for Model 1's own coefficient.**
7. Worker=8 anomaly excluded from fitting? **YES, throughout.**
8. Existing test suites pass? **YES** (rerun after this analysis;
   nothing in `fft_cost_model.py`/`CostWeights` was actually touched by
   this analysis pass -- see below).

**Production status: `safe to integrate` in the sense that every
explicit condition is met on the evidence in hand -- but NOT YET
INTEGRATED this pass.** This phase's own scope was to test whether a
minimal mechanism-aware correction explains the data (it does), not to
modify `fft_cost_model.py`/`CostWeights` itself; per the task's own
"do not immediately modify production code" instruction, the actual
`CostWeights`/`PlanMetrics` change (adding a `persistent_round_
increment`-style field and the two derived constants, 11.0 and 1.746) is
a deliberately separate, explicit next step for the user to authorize,
not something this analysis pass performs on its own. The two
coefficients (11.0 batch-equivalents per extra persistent round; 1.746x
multiplier on a persistent leaf's own batch contribution) are the
concrete, data-derived values a production change would use -- kept here
for whoever does that integration, not applied.

## Production Integration (2026-09-01)

Model 2's two validated constants are now wired into
`planning/fft_cost_model.py` production code -- the analysis/validation
scripts (`revalidate_cost_model.py`, `analyze_mechanism_correction.py`)
are unchanged and remain useful as the reproducible dataset generator/
re-analysis tool, not as the thing that computes ranking anymore.

### Files changed

| file | change | reason |
|---|---|---|
| `planning/fft_cost_model.py` | `StageExecutionMetrics` gained `is_persistent: bool`; `PlanMetrics` gained `persistent_worker_stage_batches: int` and `persistent_extra_rounds: int`; `CostWeights` gained `persistent_stage_batch_multiplier: float = 1.746` and `persistent_extra_round_multiplier: float = 11.0`; new module-level helpers `_leaf_kernels_with_replicas`/`_persistent_extra_rounds`; `estimate_metrics` populates the two new `PlanMetrics` fields; `_execution_cost` reweights persistent's own batch contribution and adds the round term | the actual Model 2 integration |
| `verification/verify_fft_mechanism_aware_cost.py` (new) | 5 regression checks: non-persistent plans byte-identical to the pre-correction formula, N=216 fixed, N=960 mixed-leaf not broken, `persistent_extra_rounds` correctly 0 for non-persistent/7 for persistent's own far_child, determinism | locks in this exact behavior |
| `verify_production_mechanism_correction.py` (new, repo root) | one-shot equivalence check: production `estimate_cost` vs. the already-validated `analyze_mechanism_correction.py` Model 2 picks, across the full 24-candidate dataset | proves integration correctness before trusting it for anything |

No other file was touched. No codegen, planner-decision (`fft_plan_search.py`, `fft_plan_recursive.py`, `fft_plan_persistent.py`), or simulator source was modified -- confirmed by `git status` scope after this change.

### Exact production formula (from the actual code, not pseudocode)

```python
# planning/fft_cost_model.py, _execution_cost:
nonpersistent_worker_stage_batches = (
    metrics.total_worker_stage_batches - metrics.persistent_worker_stage_batches
)
weighted_stage_work = (
    nonpersistent_worker_stage_batches
    + weights.persistent_stage_batch_multiplier * metrics.persistent_worker_stage_batches
    + weights.persistent_extra_round_multiplier * metrics.persistent_extra_rounds
)
return weights.stage_work * weighted_stage_work
```

`persistent_worker_stage_batches` (`estimate_metrics`): `sum(sm.max_batches_per_worker * sm.chunks_per_batch for sm in compute_stage_metrics(plan) if sm.is_persistent)` -- the exact same per-stage terms `total_worker_stage_batches` already summed, filtered to stages whose own `stage.persistent_vector_batches is not None`.

`persistent_extra_rounds` (`_persistent_extra_rounds`): for every leaf in the plan tree (walked via a new `_leaf_kernels_with_replicas`, since `flatten_recursive_node` itself drops each leaf's own `r` replica count), `sum(max(0, num_rounds(leaf.r, target.num_ndp_units) - 1) for persistent leaves)` -- `num_rounds` reused unchanged from `fft_plan_persistent.py`, not re-derived.

### Constant provenance

| constant | dataset | derivation | physical meaning |
|---|---:|---|---|
| `persistent_stage_batch_multiplier = 1.746` | `docs/cost_model_revalidation.csv` (matched pairs, N=144/216/512/630/960/1024/2048) | `620.0 / 355.2` -- cooperative's own median cycles-per-batch (620.0) divided by persistent's own (355.2), NOT the other way (the naive ratio makes ranking worse, confirmed by testing it) | a persistent leaf's own batches, even in a single round, carry real preload/writeback launch cost `total_worker_stage_batches` was never designed to see |
| `persistent_extra_round_multiplier = 11.0` | `docs/cost_model_saturation.csv` (36-point synthetic sweep, N=64 fixed leaf, replicas 1-256) | `3907.5 / 355.2` -- the median real cycles-per-extra-wave (from 3 independent wave-count transitions, stddev 3.5% of the median) expressed in the same batch-equivalent units the multiplier above uses | once a persistent leaf's own replica count exceeds one round's worth of physical NDP units, each extra round repeats the same fixed-width launch (preload/stage_N/writeback), a real, additional, and until now completely unmodeled cost |

### Production-vs-analysis equivalence

**Exact.** `verify_production_mechanism_correction.py` confirms production `estimate_cost` reproduces every one of the previously-validated Model 2 picks across all 7 N in the dataset (144, 216, 512, 630, 960, 1024, 2048) -- not merely similar rankings, the identical candidate label wins at every N. No discrepancy found; no retuning was needed.

### N=216 regression

| | pick | regret |
|---|---|---:|
| old production (pre-integration) | persistent | 7.3% |
| new production (Model 2 integrated) | **cooperative_workers=4 (correct)** | **0.0%** |
| real measured best | cooperative_workers=4 (22316 cycles) | - |

`persistent_extra_rounds == 0` for this candidate (confirmed by
`check_n216_regression_fixed`) -- the fix here comes entirely from
`persistent_stage_batch_multiplier`, exactly as predicted.

### N=960 regression

| | pick | regret |
|---|---|---:|
| old production | coop8 | 6.1% |
| new production | coop8 (unchanged) | 6.1% |
| real measured best | mixed (persistent-near/non-coop-far), 31689 cycles | - |

The mixed-leaf fixture is confirmed NOT broken by the new correction
(`check_n960_mixed_leaf_not_broken`): its own cost stays well below both
uniform persistent and non-cooperative, exactly the required invariant.
The residual 6.1% gap against `coop8` is unchanged by this integration,
as expected (Model 2 was never claimed to close it -- see the
"Mechanism-Aware Correction" section above for why Model 3 was not
pursued).

### Full dataset metrics, old production vs. new production

| | mean regret | median regret | worst-case regret |
|---|---:|---:|---:|
| old production (pre-integration) | 10.4% | 12.8% | 16.1% (N=1024) |
| **new production (Model 2 integrated)** | **0.9%** | **0.0%** | **6.1% (N=960)** |

(Per-N picks and Spearman correlation: see `verify_production_
mechanism_correction.py`'s own output, byte-identical to the "Phase 7"
table in the "Mechanism-Aware Correction" section above.)

### Test results

| command | result |
|---|---|
| `python3 verification/verify_fft_mechanism_aware_cost.py` | PASS (5/5 new checks) |
| `python3 verification/verify_fft_execution_cost.py` | PASS (unchanged) |
| `python3 verification/verify_fft_search.py` | PASS (unchanged) |
| `python3 verification/verify_fft_persistent_search.py` | PASS (unchanged) |
| `python3 verification/verify_fft_unit_utilization.py` | PASS (unchanged) |
| `python3 verification/verify_fft_ndp_cycle_parsing.py` | PASS (unchanged) |
| `PYTHONPATH=. python3 verification/verify_fft_execution_invariants.py` | PASS (unchanged -- numerical/plan/codegen invariants, confirms this change touched cost-ranking only) |
| `python3 verification/verify_fft_plan.py` | PASS (unchanged -- the project's own main numerical correctness suite) |
| `python3 verify_production_mechanism_correction.py` | PASS -- exact equivalence confirmed |

No test was weakened or deleted to make this change pass.

### Remaining limitations (not blockers)

- N=960's own residual ~6.1% regret (`coop8` vs. the real-best mixed
  candidate) is unchanged by this integration -- not attempted, per
  Occam's razor (see "Mechanism-Aware Correction" section's own "why
  Model 3 was not attempted").
- `workers_per_fft=8`'s own known non-monotonic anomaly is untouched --
  no new penalty, no special-case, no exclusion added for it.
- Non-cooperative/cooperative's own early-saturation behavior (measured
  in the saturation sweep, active_units=8-16 well before the physical
  32-unit ceiling) is not modeled by any term here -- interesting
  physical interpretation, not required for this correction, not added.

### Final production status

**Production integration successful.** Every condition this task and the
prior validation pass required is met: mechanism-explainable (both terms
have direct physical grounding, not fitted-then-rationalized), N=216
fixed, N=960 mixed-leaf fixture unbroken, overall and worst-case regret
both improved, saturation data directly supports the round term
(Pearson=1.000), the worker=8 anomaly was excluded from every coefficient
derivation and remains untouched, and the full existing test suite
(numerical, plan, codegen invariants, execution-cost regression,
persistent/cooperative search, alignment diagnostic, cycle-parsing fix)
passes unchanged.
