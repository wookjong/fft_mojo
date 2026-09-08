from __future__ import annotations

"""Approximate, transparent cost metrics for a already-built RecursiveFFTPlan
-- for ranking/debugging candidates from fft_plan_search.py, not a precise
M2NDP performance model (real behavior depends on DRAM access pattern,
uthread scheduling, register spills, and simulator scheduling details this
module does not attempt to model). Every term below is a documented
heuristic over the plan's own already-decided structure, not a fitted
formula -- see PlanMetrics' own field comments for what each one means and
why, and CostWeights for how they combine. Nothing here makes a planning
decision; it only measures one already-built plan.
"""

from dataclasses import dataclass

from planning.strategies.fft_plan_recursive import (
    FFTLeafPlan,
    FFTNode,
    FFTRecursiveNodePlan,
    PhysicalTransposePlan,
    RecursiveFFTPlan,
    flatten_recursive_node,
)
from planning.core.fft_plan_core import FFTCodegenPlan, FFTStagePlan
from planning.execution.fft_plan_persistent import num_rounds
from planning.core.target_profile import TargetProfile

# Composite/prime radices confirmed clean as a leaf's *first* stage
# (reading straight from DRAM) but not as a later stage (reading its
# operands out of scratchpad instead):
#
# * (6, 9) as an *adjacent stage pair*: fft_plan_core._prime_factors_
#   supported's own comment documents the concrete N=54=(6,9) spill this
#   was originally based on (radix-9 stage, immediately preceded by
#   radix-6). Not universal: a direct (4, 9) chain (N=36, forced via
#   allowed_radix_composites) built and ran clean, zero spill, and 6/9
#   standalone (N=6, N=9) are clean too -- so whatever makes this pairing
#   risky depends on the specific (6, 9) sequence, not "radix 6 or 9 in a
#   non-first position" alone. Scored here by adjacent-pair membership in
#   _RISKY_RADIX_PAIRS below, kept in sync by hand with codegen.fft_codegen.
#   _RISKY_RADIX_PAIRS (not imported -- this module is deliberately
#   toolchain/codegen-free, see this module's own docstring, and planning
#   code does not import from codegen) -- the same set codegen's own
#   _stage_compute_lanes floors to compute_lanes=1 for. This used to be a
#   standalone-radix set (_NON_FIRST_STAGE_RISKY_RADICES
#   = {6, 9}) that over-counted risk for the confirmed-safe (4, 9) shape;
#   fixed 2026-08-27 alongside codegen's own sequence-aware fix so this
#   score and the actual narrowing decision agree on what is risky.
#   Root cause confirmed 2026-08-27 (see codegen.fft_codegen.
#   _ALWAYS_NARROW_RADICES's own comment): the same M2NDP-Detour `ReadCsr`
#   vlenb gap as 10/11/13/17 below -- compute_lanes=1 (not codegen's own
#   narrow_middle_stages default, which only halves, and only for a
#   genuine middle stage; N=54=(6,9) is a 2-stage kernel, so its radix-9
#   stage is technically "last") confirmed clean by direct probe.
# * 11, 13, 17: confirmed by direct probe (each forced into a (4, r)
#   chain -- N=44, N=52, N=68 respectively, r as the second/scratchpad-
#   reading stage): all three spill *and* silently produce an all-zero
#   (wrong, not just slow) result -- and, unlike 6/9, confirmed risky even
#   completely alone (N=11/13/17 as a standalone single-stage kernel,
#   2026-08-27), so reclassified here from _NON_FIRST_STAGE_RISKY_RADICES
#   to _ALWAYS_RISKY_RADICES: the earlier classification undersold the
#   real risk, since it was only ever probed embedded in a (4, r) chain.
#   Fixed at the codegen level by `narrow_middle_stages`'s own unconditional
#   floor-to-1 for these radices (see codegen.fft_codegen.
#   _ALWAYS_NARROW_RADICES) -- this risk_score term stays anyway as a
#   cost-model-visible signal independent of whether that codegen fix is
#   in effect for a given render.
#
# `10` fails outright at *any* stage position (N=160/320, and reconfirmed
# by the same probe as (4, 10) = N=40 -- spills and mismatches even
# directly after a radix-4 first stage) -- see the same fft_plan_core
# comment. Same fix and same root cause as 11/13/17 above.
_RISKY_RADIX_PAIRS = frozenset({(6, 9)})
_ALWAYS_RISKY_RADICES = frozenset({10, 11, 13, 17})

# Radix 5 as a *genuine* middle stage (neither first nor last -- has a
# preceding stage AND a following one, e.g. the 5 in a 3-stage (3, 5, 7)
# leaf) spills on real hardware in a way `narrow_middle_stages` cannot
# fix, unlike every other middle-stage liability this module tracks.
# Confirmed 2026-08-28 by direct probe of N=105 (a single fused (3, 5, 7)
# leaf, isolated -- no transpose stage, no other kernel) at every
# compute_lanes this project ships or could plausibly ship:
#
#   compute_lanes=4, narrow_middle_stages=False (no narrowing at all):
#     stage_1 (radix 5) spills, 128-byte frame
#   compute_lanes=4, narrow_middle_stages=True (stage_1 renders at
#     narrowed width 2, this project's own real default combination):
#     stage_1 spills, 128-byte frame -- identical to the unnarrowed case
#   compute_lanes=2 (stage_1 narrows further to 1, the floor):
#     stage_1 spills, 1296-byte frame -- WORSE, not better
#   compute_lanes=1 (already at the floor, narrowing is a no-op):
#     stage_1 AND stage_2 (radix 7, the *last* stage) both spill,
#     1296- and 64-byte frames respectively
#   loop_stages=True at compute_lanes=4/narrow_middle_stages=True:
#     stage_1 still spills, same 128-byte frame -- rules out per-batch
#     unrolling/function size as the mechanism too
#
# So compute_lanes narrowing -- the fix for every *other* middle-stage
# liability this module and codegen.fft_codegen track -- does not apply
# here at any width, and the one lever that does exist (narrower) makes
# it worse past width 2. This also retroactively explains why N=630's own
# FFTRecNear0::stage_1() (the exact case narrow_middle_stages was
# originally built around) was previously reported "confirmed clean at
# the halved width": that claim was checking run_fft_test.sh's PASS/FAIL
# reference check, not the spill warning specifically -- and N=630's
# default-tile plan does print `[PASS] (spill warning!)`, i.e. it was
# never actually spill-free, just not (this time) numerically wrong,
# exactly the "it happened to pass its reference check" case [[fft-spill-
# hard-filter]] warns against trusting.
#
# No codegen-level fix exists yet (nothing here floors or gates
# compute_lanes for this -- there is no width that helps), so this is a
# *cost-model-only* signal for now: it steers estimated_cost away from
# radix-5-as-middle and, more importantly, means planning.diagnostics.spill_probe.
# probe_and_rerank_candidates' own hard exclusion actually has somewhere
# else to fall back to (e.g. N=105's own (5, 7)+(3,) or (7,)+(3, 5) split
# candidates, which put radix 5 first or last instead of in the middle --
# both confirmed clean positions) rather than exhausting every candidate
# and raising NoSpillFreeCandidateError. A plain default (no --verify-
# spill-free) build of an N whose only decomposition has radix 5 in a
# genuine middle position has no protection from this yet.
_RISKY_AS_MIDDLE_RADICES = frozenset({5})

# FLAGGED SUSPECT 2026-08-31, NOT YET RE-VERIFIED: N=1024 needs a real
# split (multiple kernel structs); every cycle number below was almost
# certainly measured with the `tail -1`-on-Gantt-log convention `planning.
# spill_probe._parse_ndp_cycles`'s 2026-08-31 fix retired (it only ever
# captured the LAST kernel struct's own duration, not the true end-to-end
# total -- see docs/active_ndp_units_cost_task.md's Phase 1.5 writeup).
# N=16384 may or may not need a split depending on scratchpad_byte_budget
# -- not independently checked here. Direction may still hold (tile size
# is a transpose-only parameter, and the always-last-measured POST
# transpose kernel's own duration does scale with it), but the exact
# threshold (2048) has not been rechecked against corrected totals --
# re-sweep with the now-fixed benchmark_fft_candidates.sh first.
#
# Real M2NDP runs (benchmark_fft_candidates.sh, --batch sweep at N=1024 plus
# one N=16384 point) of a transpose stage's own total_uthreads vs. whether a
# smaller tile (more, smaller tiles) still wins over a bigger one:
#
#   1024 uthreads (N=1024, batch=1): tile=1x1 wins (1252 vs. 1516 cycles)
#   2048 uthreads (N=1024, batch=2): tile=1x1 wins (1355 vs. 1471 cycles)
#   4096 uthreads (N=1024, batch=4): tile=1x1 LOSES (1786 vs. 1560 cycles)
#  16384 uthreads (N=16384, batch=1): tile=1x1 LOSES (2258 vs. 1787 cycles)
#
# So CostWeights.transpose_tile_count's own "more/smaller tiles is usually
# faster" (see its own comment) only holds up to some point -- past it, the
# same extra tiles apparently cost more in launch/scheduling overhead than
# they gain in parallelism. Only 4 points, all at one of two N -- nowhere
# near enough to fit *where* the crossover really sits, so this uses the
# last CONFIRMED-still-winning point (2048) as the threshold, the same
# "last confirmed-safe point, not a guessed midpoint" discipline target_
# profile.DEFAULT_TARGET_PROFILE's own max_concurrent_scratchpad_bytes
# comment already uses. Reread with more benchmark_fft_candidates.sh data
# before trusting this far from N=1024/16384.
_TILE_PARALLELISM_SATURATION_UTHREADS = 2048


@dataclass(frozen=True)
class PlanMetrics:
    recursion_depth: int
    leaf_kernel_count: int
    transpose_kernel_count: int
    total_leaf_stage_count: int
    total_transpose_tiles: int
    # The single busiest PRE/MIDDLE/POST transpose stage's own total_uthreads
    # (0 if this plan has no transpose stage at all, i.e. a single fused
    # leaf) -- see CostWeights.tile_oversaturation_penalty's own comment for
    # why this, not total_transpose_tiles (summed across stages), is the
    # quantity the "smaller tile stops helping" crossover was measured
    # against.
    max_transpose_stage_uthreads: int
    estimated_dram_bytes: int
    max_scratchpad_bytes: int
    # 1.0 = no cooperative leaf ever idles a worker. Kept as a diagnostic
    # field (format_plan_summary still prints it) but NO LONGER what
    # `_execution_cost` uses -- see `total_worker_stage_batches` below and
    # docs/execution_cost_model_validation.md for why: this can only see
    # whether a worker slot is *occupied*, never how many batches the
    # *busiest* one still has, so it is 1.0 (identical) for e.g.
    # workers_per_fft=2 and =4 on the exact same stage whenever the
    # round-robin split leaves no worker empty -- confirmed on real
    # hardware (N=216) to hide a genuine ~35% cycle difference between
    # those two worker counts.
    worst_worker_utilization: float
    # `sum(max_batches_per_worker * chunks_per_batch across every leaf
    # stage)` -- see `StageExecutionMetrics.max_batches_per_worker`/
    # `chunks_per_batch`'s own docstrings. The busiest worker's own serial
    # batch count, summed over every stage of every leaf kernel in this
    # plan (stages run sequentially within a kernel; kernels run
    # sequentially as separate launches -- see `flatten_recursive_node`'s
    # own execution-order docstring), each stage's batch count weighted by
    # how many real vector-width chunks codegen actually emits per batch
    # at that stage's own resolved compute_lanes -- an absolute proxy for
    # total serial SIMD-*instruction* time, not a [0,1] utilization
    # fraction and not a raw batch count blind to compute_lanes either
    # (see `chunks_per_batch`'s own docstring and docs/
    # compute_lanes_joint_search.md's "Phase 7-1" item this closes out).
    # This is what `_execution_cost` uses (chosen 2026-08-30 after
    # comparing 4 aggregation models against 55 real measured candidates
    # -- see docs/execution_cost_model_validation.md): unlike any
    # utilization-fraction model (worst-case, simple-average, or
    # work-weighted-average all tried and rejected), this is the one
    # quantity that actually shrinks when workers_per_fft goes up on an
    # already-evenly-split stage, which is the entire point; the
    # `chunks_per_batch` weighting (added 2026-08-31) is what makes it
    # also grow when compute_lanes narrows on an otherwise-identical
    # stage, which it could not see before.
    total_worker_stage_batches: int
    # The subset of `total_worker_stage_batches` contributed by persistent-
    # software-workgroup leaf stages specifically (`StageExecutionMetrics.
    # is_persistent`) -- `0` for a plan with no persistent leaf at all.
    # Kept as its own field rather than folded away so `total_worker_
    # stage_batches` stays exactly what it always was (every existing
    # reader of that field, diagnostic or otherwise, sees byte-identical
    # values); `_execution_cost` reads this one separately because
    # persistent's own batches carry a different real per-batch cost than
    # non-cooperative/cooperative's (see `CostWeights.persistent_stage_
    # batch_multiplier`'s own docstring and docs/
    # active_ndp_units_cost_task.md's "Mechanism-Aware Correction"
    # section for the real-hardware matched-pair measurements this is
    # based on).
    persistent_worker_stage_batches: int
    # `_persistent_extra_rounds`'s own return value -- how many EXTRA
    # host-orchestrated rounds (beyond the first) this plan's own
    # persistent leaves need, summed across leaves. `0` whenever no leaf
    # is persistent, or every persistent leaf's own replica count already
    # fits in one round (`r <= target.num_ndp_units`). This is real,
    # additional cost `total_worker_stage_batches` cannot see at all --
    # a persistent leaf's own launch width is architecturally fixed
    # (`target.num_ndp_units * interleave_chunk_uthreads` uthreads,
    # independent of replica count), so more replicas than that show up
    # as more ROUNDS of the same fixed-width launch (preload/stage_N/
    # writeback repeated), never as more SIMD batches within one round --
    # see `CostWeights.persistent_extra_round_multiplier`'s own docstring
    # for the real saturation-sweep measurement (Pearson=1.000 between
    # this quantity and measured cycles) this term is calibrated against.
    persistent_extra_rounds: int
    radix_risk_score: float          # 0.0 = every leaf's radix sequence is the confirmed-safe kind
    # How many PRE/MIDDLE/POST transpose stages have a tile that does NOT
    # divide `rows`/`cols` exactly (`rows % tile_rows != 0 or cols %
    # tile_cols != 0` -- see fft_plan_search.generate_tile_candidates' own
    # comment for the real-hardware finding this is based on). 0 means
    # every transpose stage's own tile evenly divides both dimensions --
    # confirmed real-hardware clean at every no-tail size tried (N=630's
    # PRE transpose, gcd(105,6)=3's own divisors 1 and 3). A stage WITH a
    # tail is not confirmed unsafe (tile=2x2/5x5 also had one and stayed
    # clean) -- only a weak risk signal, not a certainty, which is why
    # transpose_tail_risk_penalty (below) stays small relative to
    # radix_risk_penalty.
    transpose_tail_tile_count: int
    # `None` (the default, and the only value estimate_metrics itself ever
    # produces) means "not probed" -- unlike every other field above, this
    # one cannot be computed from the plan alone (see planning/spill_probe.
    # py's own module docstring: it needs a real Mojo -> llc -> M2NDP-
    # Detour build+run, tens of seconds to minutes, not something
    # estimate_metrics can afford to do for every candidate). A caller who
    # wants this signal calls spill_probe.probe_spill_free explicitly and
    # folds the result in via spill_probe.apply_spill_probe -- estimate_cost
    # then only applies spill_penalty once this is actually `True`/`False`,
    # never for the untouched `None` default, so every existing caller's
    # ranking is bit-for-bit unchanged unless it opts in. See CostWeights.
    # spill_penalty's own comment and [[fft-spill-hard-filter]] for why
    # this soft weighting is a diagnostic signal only, never the actual
    # mechanism that keeps a confirmed-spilling candidate out of a final
    # pick -- planning.diagnostics.spill_probe.probe_and_rerank_candidates' own hard
    # exclusion is.
    spill_free: bool | None = None
    # `None` (the default) means "not probed," the same discipline as
    # `spill_free` above and for the same reason -- a real measured cycle
    # count (planning.diagnostics.spill_probe.SpillProbeResult.ndp_cycles, the
    # simulator's own Gantt-log total, not an estimate) only exists once a
    # caller has actually built+run this plan. Folded in by spill_probe.
    # apply_spill_probe alongside spill_free, from the same probe -- no
    # separate probe call needed. Used by planning.diagnostics.spill_probe.
    # probe_and_rerank_candidates' own `rank_by_cycles` to pick the
    # fastest *measured* candidate among several already-confirmed-
    # spill-free ones, rather than trusting estimated_cost's static
    # ranking to have picked the fastest one first (that ranking is a
    # cheap pre-filter for which candidates are worth a real probe at
    # all, not a claim that its own order matches real hardware speed --
    # see this project's own N=16384 tile=(4,4) vs (2,2) mismatch).
    ndp_cycles: int | None = None
    estimated_cost: float = 0.0      # filled in by estimate_cost, 0.0 until then


@dataclass(frozen=True)
class CostWeights:
    """One documented per-unit weight per PlanMetrics term -- not a fitted
    model (see this module's own docstring). Tune here; candidate
    generation never needs to change when these do. Ranking matters most
    *within* one generate_candidates(n) call, where most of these terms
    move independently of each other (e.g. a worker-count sweep holds
    estimated_dram_bytes essentially constant while worst_worker_utilization
    varies) -- see each field's own comment for why its magnitude was
    picked relative to the others.
    """

    memory_traffic: float = 1.0
    # Each transpose kernel is a full extra launch + DRAM round trip beyond
    # what memory_traffic already counts per stage -- a per-kernel
    # fixed-overhead term on top of the raw byte count.
    transpose_passes: float = 50.0
    # Also the per-serial-batch rate `_execution_cost` charges against
    # `total_worker_stage_batches` (2026-08-30) -- see that function's
    # own docstring for why it reuses this rate rather than introducing
    # a new one (the old idle_worker_penalty weight this replaced is
    # gone: nothing else read it).
    stage_work: float = 0.1
    # Derived from reproducible matched-pair measurements (`analyze_
    # mechanism_correction.py`, real hardware, N in {144, 216, 512, 630,
    # 960, 1024, 2048}): persistent's own measured cycles-per-batch
    # (median 355.2 across single-round matched pairs) is LOWER than
    # cooperative's (620.0) -- meaning a persistent leaf's own `total_
    # worker_stage_batches` contribution structurally UNDER-represents
    # its real relative cost (the same batch-count reduction buys less
    # real speedup for persistent than for cooperative), not the other
    # way around. `1.746 = 620.0 / 355.2` -- the reciprocal of the naive
    # ratio, not the ratio itself (easy to get backwards: scaling
    # persistent's batches DOWN, the naive direction, makes an already-
    # wrong ranking worse, confirmed by testing it -- see docs/
    # active_ndp_units_cost_task.md's "Mechanism-Aware Correction"
    # section for the full derivation and the concrete N=216 regression
    # this fixes: real best is `cooperative_workers=4`, but the
    # unweighted model always picks `persistent` there since 9 batches
    # < 15 regardless of `stage_work`'s own value -- confirmed no
    # `stage_work` reweight alone can fix this, only a persistent-
    # specific rate can). Physical interpretation: `total_worker_stage_
    # batches` counts SIMD-iteration batches within a stage, but a
    # persistent stage's own preload/writeback round-management work
    # (see `PersistentWorkgroupPlan`) is real launch cost that metric was
    # never designed to see, even within a single round.
    persistent_stage_batch_multiplier: float = 1.746
    # Derived from the same session's own persistent saturation sweep
    # (`revalidate_saturation.py`, N=64 fixed leaf, replicas 1-256, real
    # hardware): once a persistent leaf's own replica count exceeds
    # `target.num_ndp_units`, extra ROUNDS (not extra batches -- a
    # persistent launch's own width is architecturally fixed, see
    # `PlanMetrics.persistent_extra_rounds`'s own docstring) appear, and
    # each extra round costs ~3900-4200 real cycles (median 3907.5,
    # stddev 3.5% of the median across 3 independent wave-count
    # transitions -- about as clean as a real-hardware measurement gets
    # in this project), while `measured_cycles` correlates with `waves`
    # at Pearson=1.000 across that same 9-point sweep (`total_worker_
    # stage_batches` itself is CONSTANT across the whole sweep by
    # construction, so it cannot see this cost at all). `11.0 = 3907.5 /
    # 355.2` -- that median per-round cycle cost expressed in the same
    # batch-equivalent units `persistent_stage_batch_multiplier` uses
    # (persistent's own measured cycles-per-batch), not a separately
    # chosen constant. Applied to `PlanMetrics.persistent_extra_rounds`
    # (`max(0, rounds - 1)`, summed per persistent leaf) -- never to the
    # first round, which is already covered by `persistent_stage_batch_
    # multiplier` above, and never to a non-cooperative/cooperative leaf
    # (neither has a round concept at all -- see docs/
    # active_ndp_units_cost_task.md's own "execution mechanisms that must
    # remain distinct" discussion).
    persistent_extra_round_multiplier: float = 11.0
    # A spill on real hardware is a correctness failure (the kernel panics
    # or silently zeroes its output), not a slowdown -- this must dominate
    # every other term whenever radix_risk_score is nonzero.
    radix_risk_penalty: float = 5000.0
    # NOT the authoritative spill policy -- see [[fft-spill-hard-filter]]
    # (a confirmed real spill must disqualify a candidate outright, never
    # just adjust a score) -- that's enforced by planning.diagnostics.spill_probe.
    # probe_and_rerank_candidates hard-excluding any candidate whose
    # PlanMetrics.spill_free a real probe confirmed False, before this
    # weight ever gets a chance to matter for it. This term only fires
    # once spill_free has actually been probed (see that field's own
    # docstring -- estimate_cost skips it entirely for the untouched
    # `None` default) and exists purely as a *diagnostic* signal for a
    # caller reading estimated_cost directly (format_plan_summary, ad hoc
    # debugging) without going through the hard-exclusion path -- e.g. so
    # a spilling candidate's own cost still visibly reflects that DRAM-
    # spill traffic is real cost on top of whatever else is wrong with it,
    # rather than looking identical to a spill-free sibling. Never rely on
    # this weight alone to keep a confirmed-spilling candidate out of a
    # final recommendation; only probe_and_rerank_candidates' own hard
    # exclusion does that reliably.
    spill_penalty: float = 2000.0
    # A weak, pre-probe nudge toward tile choices that structurally can't
    # hit the tail-branch liability transpose_tail_tile_count flags (see
    # that field's own comment) -- deliberately much smaller than
    # radix_risk_penalty/spill_penalty, since a tail tile is only a risk
    # signal, not a confirmed one (tile=2x2/5x5 in the real N=630 case
    # this is based on both had a tail and stayed clean). Sized just large
    # enough to break a near-tie in favor of the no-tail option (e.g. two
    # tile sizes with otherwise-similar transpose_tile_count/DRAM cost),
    # not to override a genuinely cheaper has-tail candidate outright --
    # a caller who wants the stronger guarantee should reach for
    # planning.diagnostics.spill_probe's real probe instead, same as radix_risk_score
    # vs. a confirmed spill_free=False.
    transpose_tail_risk_penalty: float = 20.0
    recursion_depth_penalty: float = 100.0
    # FLAGGED SUSPECT 2026-08-31, NOT YET RE-VERIFIED (see docs/
    # active_ndp_units_cost_task.md's Phase 1.5 writeup and planning.
    # spill_probe._parse_ndp_cycles's own fix comment): every N cited below
    # (1024, 960, 630) needs a real split, so every one of these cycle
    # numbers was almost certainly measured with the retired `tail -1`
    # Gantt-log convention -- confirmed for N=1024 specifically (the
    # "192 tiles -> 2111 cycles" figure below is byte-for-byte this
    # project's own old, wrong non-cooperative N=1024 reading; the
    # corrected total for that exact shape is 49061, not 2111). The
    # *direction* this weight encodes (more/smaller tiles often faster)
    # may well still hold -- tile choice is a PRE/MIDDLE/POST-transpose-only
    # parameter, and `tail -1` happens to land on a POST-transpose kernel,
    # whose own duration DOES scale with tile size unlike leaf strategy --
    # but the weight's own magnitude (`-2.0`, picked from these exact
    # numbers) has not been rechecked against corrected totals. Re-sweep
    # with `benchmark_fft_candidates.sh` (now fixed) before trusting this
    # weight's magnitude for anything beyond breaking a near-tie.
    #
    # Original (now-suspect) justification, kept for the record: real
    # M2NDP runs (this project's own benchmark_fft_candidates.sh sweeps at
    # N=1024/960/630) show MORE, SMALLER transpose tiles usually finishing
    # in *fewer* ndp cycles, not more -- e.g. N=1024: 192 tiles -> 2111
    # cycles vs. 48 tiles -> 7421-7871 cycles; N=630: 210 tiles -> 1867
    # cycles (the fastest of 6 real candidates) vs. the 54-tile baseline's
    # 4426. The opposite of the usual "more kernel launches = more
    # overhead" assumption, plausibly because a smaller tile fits this
    # target's own SIMD/register width more cleanly -- but NOT cleanly
    # monotonic (N=630's own 46-tile candidate ran *slower*, 5561 cycles,
    # than its 54-tile baseline), so this weight is deliberately small:
    # before it existed, every split/tile candidate for one N was an exact
    # estimated_cost tie (total_transpose_tiles was computed in PlanMetrics
    # but never read here), so ranking among them fell back to generation-
    # order luck. This only needs to break that exact tie in the right
    # direction, not carry serious absolute weight -- treat any single
    # ranking decision it flips as a hint to verify with benchmark_fft_
    # candidates.sh, not a settled answer.
    transpose_tile_count: float = -2.0
    # FLAGGED SUSPECT 2026-08-31, NOT YET RE-VERIFIED -- same issue as
    # transpose_tile_count immediately above: N=1024 needs a real split, so
    # the "1786 vs. 1560" real-cycle citation below almost certainly used
    # the retired `tail -1` convention too. Counteracts transpose_tile_
    # count once a candidate's own busiest transpose stage passes
    # _TILE_PARALLELISM_SATURATION_UTHREADS -- see that constant's own
    # comment for the 4 real data points this is based on. Sized so that
    # N=1024/batch=4's tile=1x1 candidate (4096 uthreads, 2048 over
    # threshold) actually ranks behind its own tile=2x2 sibling (real
    # cycles: 1786 vs. 1560) -- picked as the smallest multiple of 10
    # that does, not a fitted rate; recheck this weight if more benchmark_
    # fft_candidates.sh data at other N/batch combinations disagrees.
    tile_oversaturation_penalty: float = 10.0


DEFAULT_COST_WEIGHTS = CostWeights()


@dataclass(frozen=True)
class StageExecutionMetrics:
    """Per-computational-stage execution shape, one level more granular
    than `PlanMetrics.worst_worker_utilization` -- exposed for offline
    execution-model comparison (see docs/execution_cost_model_validation.md),
    not consumed by `estimate_cost`/`PlanMetrics` itself. Every field is
    read straight off the already-built plan (`FFTStagePlan`/
    `SIMDBatchPlan`) -- no new constant, no per-radix instruction-count
    guess.

    Confirmed real-hardware finding this exists to make visible (N=216,
    default radix tier): `worst_worker_utilization` is 1.0 for both
    `workers_per_fft=2` and `workers_per_fft=4` (every worker has >=1
    batch on every stage, so nothing here ever looks idle), yet measured
    cycles are 30205 vs. 22316 -- a real ~35% difference `worst_worker_
    utilization` cannot see, because it only asks "is every worker slot
    occupied," never "how many batches does the *busiest* worker still
    have to do." `max_batches_per_worker` (below) is that missing
    quantity: `_partition_batches` (fft_plan_cooperative.py) round-robins
    a stage's own `simd_iteration_count` batches across `workers_per_fft`
    workers, so the busiest worker's own batch count -- not whether every
    worker has *a* batch -- is what actually bounds this stage's parallel
    completion time.
    """

    leaf_index: int
    stage_id: int
    radix: int
    # Total butterflies this stage processes for one logical FFT replica --
    # `sum(batch.valid_lanes for batch in stage.batches)`, the exact
    # quantity `simd_iteration_count = ceil(butterfly_count / simd_lanes)`
    # was itself derived from (see layouts_for_radices), read back out
    # rather than re-derived from radix/length so this stays correct even
    # for a tail (`valid_lanes < simd_lanes`) batch.
    butterfly_count: int
    # `len(stage.batches)` -- how many SIMD-width iterations one implicit
    # (non-cooperative) worker would run serially for this stage.
    simd_iteration_count: int
    # `None` when this leaf is not cooperative (`stage.worker_batches is
    # None`) -- the whole stage is one implicit worker's serial work.
    workers_per_fft: int | None
    # Workers with >=1 batch assigned this stage; `1` when not
    # cooperative. This is the numerator `_leaf_worker_utilization`
    # (the existing, coarser metric) uses.
    active_workers: int
    # The busiest worker's own batch count this stage -- `simd_iteration_
    # count` itself when not cooperative (one implicit worker owns
    # everything); `max(len(wb) for wb in stage.worker_batches)`
    # otherwise. THE quantity worst_worker_utilization cannot see: this
    # is what actually bounds a cooperative stage's parallel time, not
    # whether every worker slot is merely non-empty.
    max_batches_per_worker: int
    # `active_workers / workers_per_fft` (1.0 when not cooperative) --
    # kept for direct comparison against today's production metric; see
    # this dataclass's own docstring for why it cannot distinguish
    # workers_per_fft=2 from =4 on an evenly-divisible stage.
    worker_utilization: float
    # `simd_iteration_count / max_batches_per_worker` -- how many workers'
    # worth of *real* speedup this stage actually realized (<= workers_
    # per_fft always; == workers_per_fft exactly when the round-robin
    # split divides evenly; == 1.0 when not cooperative).
    effective_parallelism: float
    # `ceil(simd_lanes / stage.compute_lanes)` -- the real number of vector-
    # width chunks `codegen.lowering.chunk_batch` emits per already-decided
    # simd_lanes-wide batch of this stage (kept in sync by hand with that
    # function's own `n_chunks` formula -- this module is deliberately
    # codegen-free, see module docstring). `1` whenever `stage.compute_lanes`
    # is `None` (unset -- codegen's own live fallback decides the real
    # render width from a flat caller argument this plan can't see, so this
    # counts it as un-narrowed rather than guessing, same discipline as
    # every other "None means not probed/not decided here" field in this
    # module) or `>= simd_lanes` (chunk_batch's own no-op threshold). A
    # narrower compute_lanes means MORE, narrower vector instructions doing
    # the exact same butterfly work -- e.g. compute_lanes=1 on an
    # simd_lanes=8 stage emits 8 scalar-width chunks per batch instead of
    # one 8-wide vector op. See `_stage_chunk_count`'s own comment for why
    # this, not an arbitrary penalty, is what makes `_execution_cost`
    # finally differentiate compute_lanes variants (docs/
    # compute_lanes_joint_search.md's own "Phase 7-1" item).
    chunks_per_batch: int
    # `stage.persistent_vector_batches is not None` -- whether this stage
    # belongs to a persistent-software-workgroup leaf, the same test
    # `compute_stage_metrics` itself already branches on. Added 2026-09-01
    # so `estimate_metrics` can split `total_worker_stage_batches` into its
    # persistent/non-persistent components without re-walking the plan a
    # second time -- see `PlanMetrics.persistent_worker_stage_batches` and
    # docs/active_ndp_units_cost_task.md's "Mechanism-Aware Correction"
    # section for why persistent's own batches need a different per-batch
    # rate than non-cooperative/cooperative's.
    is_persistent: bool


def _stage_chunk_count(stage: FFTStagePlan, simd_lanes: int) -> int:
    """`ceil(simd_lanes / stage.compute_lanes)` -- exactly `codegen.lowering.
    chunk_batch`'s own `n_chunks` formula (`(simd_lanes + compute_lanes - 1)
    // compute_lanes`, guarded by the same `compute_lanes >= simd_lanes` ->
    1-chunk no-op case), reimplemented here rather than imported since this
    module is deliberately codegen/toolchain-free (see module docstring) --
    kept in sync by hand, the same discipline `_RISKY_RADIX_PAIRS` above
    already uses for a codegen constant this module needs to know about.
    `stage.compute_lanes is None` (unset -- the majority of candidates this
    project's planner builds; see fft_plan_lanes.py's own module docstring)
    resolves to 1, not a guess at whatever flat compute_lanes codegen might
    apply at render time -- this plan-level module has no visibility into
    that caller-supplied value, so an unset stage costs identically to
    today's un-narrowed baseline, the same "None means not decided here"
    rule PlanMetrics.spill_free/ndp_cycles already use."""
    if stage.compute_lanes is None or stage.compute_lanes >= simd_lanes:
        return 1
    return (simd_lanes + stage.compute_lanes - 1) // stage.compute_lanes


def compute_stage_metrics(plan: RecursiveFFTPlan) -> list[StageExecutionMetrics]:
    """Every leaf kernel's own per-stage execution shape, flattened in
    plan execution order (`flatten_recursive_node`'s own ordering: PRE ->
    near_fft -> MIDDLE -> far_child -> POST) -- transpose stages carry no
    per-worker-cooperation concept in this codebase, so only `FFTCodegenPlan`
    (leaf kernel) entries contribute. `leaf_index` counts leaf kernels only
    (0, 1, 2, ... in execution order), not the mixed leaf+transpose flat
    index `flatten_recursive_node` itself returns.
    """
    result: list[StageExecutionMetrics] = []
    leaf_index = 0
    for node in flatten_recursive_node(plan.root):
        if isinstance(node, PhysicalTransposePlan):
            continue
        for stage in node.stages:
            if stage.persistent_vector_batches is not None:
                # Persistent-software-workgroup stage (see PersistentWorkgroupPlan /
                # fft_plan_persistent.py): `worker_batches` is never set here
                # (a separate, non-overlapping partition -- see FFTStagePlan's
                # own docstring), so this must be read from `persistent_
                # vector_batches`/`persistent_scalar_batches` instead, or
                # every persistent stage would be misread as single-worker-
                # serial (`worker_batches is None`) below, wildly overstating
                # its own serial batch count.
                vector_batches = stage.persistent_vector_batches
                scalar_batches = stage.persistent_scalar_batches or ()
                simd_iteration_count = (
                    sum(len(wb) for wb in vector_batches) + len(scalar_batches)
                )
                butterfly_count = sum(
                    b.valid_lanes for wb in vector_batches for b in wb
                ) + sum(b.valid_lanes for b in scalar_batches)
                workers_per_fft = len(vector_batches)
                active_workers = sum(1 for wb in vector_batches if wb) or (
                    1 if scalar_batches else 0
                )
                # The scalar tail is always owned by the last worker (see
                # PersistentWorkgroupPlan.scalar_worker_mode's own docstring)
                # -- its own per-worker total is vector batches PLUS the
                # scalar tail, everyone else's is vector batches alone.
                per_worker_totals = [len(wb) for wb in vector_batches]
                if per_worker_totals:
                    per_worker_totals[-1] += len(scalar_batches)
                elif scalar_batches:
                    per_worker_totals = [len(scalar_batches)]
                max_batches_per_worker = max(per_worker_totals, default=0)
                worker_utilization = (
                    active_workers / workers_per_fft if workers_per_fft else 1.0
                )
            elif stage.worker_batches is None:
                simd_iteration_count = len(stage.batches)
                butterfly_count = sum(b.valid_lanes for b in stage.batches)
                workers_per_fft = None
                active_workers = 1
                max_batches_per_worker = simd_iteration_count
                worker_utilization = 1.0
            else:
                simd_iteration_count = len(stage.batches)
                butterfly_count = sum(b.valid_lanes for b in stage.batches)
                workers_per_fft = len(stage.worker_batches)
                active_workers = sum(1 for wb in stage.worker_batches if wb)
                max_batches_per_worker = max(
                    (len(wb) for wb in stage.worker_batches), default=0
                )
                worker_utilization = (
                    active_workers / workers_per_fft if workers_per_fft else 1.0
                )
            effective_parallelism = (
                simd_iteration_count / max_batches_per_worker
                if max_batches_per_worker
                else 1.0
            )
            result.append(
                StageExecutionMetrics(
                    leaf_index=leaf_index,
                    stage_id=stage.stage_id,
                    radix=stage.radix,
                    butterfly_count=butterfly_count,
                    simd_iteration_count=simd_iteration_count,
                    workers_per_fft=workers_per_fft,
                    active_workers=active_workers,
                    max_batches_per_worker=max_batches_per_worker,
                    worker_utilization=worker_utilization,
                    effective_parallelism=effective_parallelism,
                    chunks_per_batch=_stage_chunk_count(stage, node.simd_lanes),
                    is_persistent=stage.persistent_vector_batches is not None,
                )
            )
        leaf_index += 1
    return result


def _tree_depth(node: FFTNode) -> int:
    if isinstance(node, FFTLeafPlan):
        return 0
    assert isinstance(node, FFTRecursiveNodePlan)
    return 1 + _tree_depth(node.far_child)


def _leaf_kernels_with_replicas(node: FFTNode) -> list[tuple[FFTCodegenPlan, int]]:
    """Every leaf `FFTCodegenPlan` in this tree paired with its own
    `FFTLeafPlan.r` (logical replica count) -- `flatten_recursive_node`
    itself deliberately drops `r` (its own docstring: "a pure tree walk
    over already-decided plan data," returning only `FFTCodegenPlan`/
    `PhysicalTransposePlan`), and a persistent leaf's own real round count
    depends on replica count, not on anything `StageExecutionMetrics`
    already carries (a persistent leaf's own `total_uthreads` is a
    target-fixed launch width, independent of replica count -- see
    `PersistentWorkgroupPlan`'s own architecture and docs/
    persistent_recursive_split.md's point 4) -- so this needs its own
    walk, the same one `planning.diagnostics.fft_unit_utilization` already uses for
    its own (diagnostic-only, not imported here -- see this module's own
    docstring on staying free of that module) purposes."""
    if isinstance(node, FFTLeafPlan):
        return [(node.kernel, node.r)]
    assert isinstance(node, FFTRecursiveNodePlan)
    return [(node.near_fft.kernel, node.near_fft.r)] + _leaf_kernels_with_replicas(
        node.far_child
    )


def _persistent_extra_rounds(plan: RecursiveFFTPlan, target: TargetProfile) -> int:
    """`sum(max(0, num_rounds(leaf.r, target.num_ndp_units) - 1))` over
    every persistent leaf in this plan -- 0 for a plan with no persistent
    leaf at all, and 0 even for a persistent leaf whose own replica count
    fits in a single round (`r <= target.num_ndp_units`). `num_rounds`
    (`fft_plan_persistent.py`) is reused unchanged, not re-derived: a
    persistent leaf's own round count is already exactly this formula by
    construction (`ceil(num_logical_blocks / software_group_count)`),
    confirmed against a real hardware saturation sweep to correlate with
    measured cycles at Pearson=1.000 (see docs/
    active_ndp_units_cost_task.md's "Mechanism-Aware Correction" section)
    -- the `- 1` matches that same sweep's own finding that the FIRST
    round costs the same as a persistent leaf's own already-counted
    `total_worker_stage_batches` contribution; only EXTRA rounds beyond
    the first are additional, unmodeled cost."""
    total = 0
    for codegen_plan, replicas in _leaf_kernels_with_replicas(plan.root):
        if codegen_plan.persistent is None:
            continue
        rounds = num_rounds(replicas, target.num_ndp_units)
        total += max(0, rounds - 1)
    return total


def _leaf_radix_risk(codegen_plan: FFTCodegenPlan) -> float:
    risk = 0.0
    prev_radix: int | None = None
    n_stages = len(codegen_plan.stages)
    for stage in codegen_plan.stages:
        is_middle = 0 < stage.stage_id < n_stages - 1
        if stage.radix in _ALWAYS_RISKY_RADICES:
            risk += 1.0
        elif prev_radix is not None and (prev_radix, stage.radix) in _RISKY_RADIX_PAIRS:
            risk += 1.0
        elif is_middle and stage.radix in _RISKY_AS_MIDDLE_RADICES:
            risk += 1.0
        prev_radix = stage.radix
    return risk


def _leaf_worker_utilization(codegen_plan: FFTCodegenPlan) -> float | None:
    """`None` if this leaf isn't cooperative (nothing to measure); else the
    worst (lowest) fraction of `workers_per_fft` actually busy across this
    leaf's own stages -- see FFTStagePlan.worker_batches' own docstring:
    `_partition_batches` always returns exactly `workers_per_fft` buckets,
    some possibly empty on a thin stage."""
    if codegen_plan.cooperation is None:
        return None
    worst = 1.0
    for stage in codegen_plan.stages:
        if not stage.worker_batches:
            continue
        active = sum(1 for wb in stage.worker_batches if wb)
        worst = min(worst, active / len(stage.worker_batches))
    return worst


def estimate_metrics(plan: RecursiveFFTPlan, target: TargetProfile) -> PlanMetrics:
    stages = flatten_recursive_node(plan.root)

    leaf_kernel_count = 0
    transpose_kernel_count = 0
    total_leaf_stage_count = 0
    total_transpose_tiles = 0
    max_transpose_stage_uthreads = 0
    max_scratchpad_bytes = 0
    radix_risk_score = 0.0
    transpose_tail_tile_count = 0
    utilizations: list[float] = []

    for stage in stages:
        if isinstance(stage, PhysicalTransposePlan):
            transpose_kernel_count += 1
            total_transpose_tiles += stage.total_uthreads
            max_transpose_stage_uthreads = max(max_transpose_stage_uthreads, stage.total_uthreads)
            max_scratchpad_bytes = max(max_scratchpad_bytes, stage.scratchpad_elements * 4)
            if stage.rows % stage.tile_rows != 0 or stage.cols % stage.tile_cols != 0:
                transpose_tail_tile_count += 1
        else:
            leaf_kernel_count += 1
            total_leaf_stage_count += len(stage.stages)
            radix_risk_score += _leaf_radix_risk(stage)
            leaf_bytes = sum(buf.elements for buf in stage.scratchpad_buffers) * 4
            max_scratchpad_bytes = max(max_scratchpad_bytes, leaf_bytes)
            util = _leaf_worker_utilization(stage)
            if util is not None:
                utilizations.append(util)

    # One full real+imag read + write per stage in the chain, times how many
    # independent batch transforms actually move through it -- see the
    # module design writeup's own "DRAM full-array passes" accounting
    # (fft_plan_recursive's own docstrings describe every stage as touching
    # the whole N-element buffer once each way) and make_recursive_transpose
    # _plan's own `batch` docstring (every stage's total_uthreads already
    # scales with it, so the DRAM traffic estimate must too or every batch
    # sweep would under-count it identically regardless of batch).
    estimated_dram_bytes = len(stages) * plan.n * plan.batch * 2 * 2 * 4
    stage_metrics = compute_stage_metrics(plan)
    total_worker_stage_batches = sum(
        sm.max_batches_per_worker * sm.chunks_per_batch for sm in stage_metrics
    )
    persistent_worker_stage_batches = sum(
        sm.max_batches_per_worker * sm.chunks_per_batch
        for sm in stage_metrics if sm.is_persistent
    )
    persistent_extra_rounds = _persistent_extra_rounds(plan, target)

    return PlanMetrics(
        recursion_depth=_tree_depth(plan.root),
        leaf_kernel_count=leaf_kernel_count,
        transpose_kernel_count=transpose_kernel_count,
        total_leaf_stage_count=total_leaf_stage_count,
        total_transpose_tiles=total_transpose_tiles,
        max_transpose_stage_uthreads=max_transpose_stage_uthreads,
        estimated_dram_bytes=estimated_dram_bytes,
        max_scratchpad_bytes=max_scratchpad_bytes,
        worst_worker_utilization=min(utilizations) if utilizations else 1.0,
        total_worker_stage_batches=total_worker_stage_batches,
        persistent_worker_stage_batches=persistent_worker_stage_batches,
        persistent_extra_rounds=persistent_extra_rounds,
        radix_risk_score=radix_risk_score,
        transpose_tail_tile_count=transpose_tail_tile_count,
    )


def _memory_cost(metrics: PlanMetrics, weights: CostWeights) -> float:
    """Data-movement terms: raw DRAM traffic, one fixed cost per transpose
    kernel boundary (a full extra launch + DRAM round trip), the transpose
    tile-count terms (more/fewer tiles changes the access pattern, not the
    byte count), and `recursion_depth_penalty` -- kept here, not in
    `_execution_cost`, since depth is a *kernel-partition* quantity (each
    extra recursion level is 3 more transpose kernels/DRAM boundaries, see
    `FFTRecursiveNodePlan`), the same "where do we cross a DRAM boundary"
    question Step 2 of this project's own planning-flow design answers,
    not a question about how one already-decided kernel gets *executed*."""
    oversaturation = max(
        0, metrics.max_transpose_stage_uthreads - _TILE_PARALLELISM_SATURATION_UTHREADS
    )
    return (
        weights.memory_traffic * metrics.estimated_dram_bytes
        + weights.transpose_passes * metrics.transpose_kernel_count
        + weights.recursion_depth_penalty * metrics.recursion_depth
        + weights.transpose_tile_count * metrics.total_transpose_tiles
        + weights.tile_oversaturation_penalty * oversaturation
        + weights.transpose_tail_risk_penalty * metrics.transpose_tail_tile_count
    )


def _compute_cost(metrics: PlanMetrics, weights: CostWeights) -> float:
    """Arithmetic-work terms: a flat per-stage cost (butterfly + twiddle
    instruction count is not modeled per-radix here -- see this module's
    own docstring on what it deliberately does not attempt -- so
    `total_leaf_stage_count` is the whole of this term today). Kept as its
    own function even with one term so a future per-radix instruction-count
    model has a single, obvious place to grow into without touching
    memory/resource/execution accounting."""
    return weights.stage_work * metrics.total_leaf_stage_count


def _resource_cost(metrics: PlanMetrics, weights: CostWeights) -> float:
    """Register-pressure/spill-risk terms: the static per-radix risk score
    (`radix_risk_score`, this module's own compile-time-known proxy for
    "will this spill" -- see the frozensets at the top of this module) and
    the real, measured spill penalty (only applied once a caller has
    actually probed `spill_free` -- see that field's own docstring; `None`
    means "not probed," never treated as either answer here). NOT the
    authoritative spill policy either way -- see [[fft-spill-hard-filter]]:
    `planning.diagnostics.spill_probe.probe_and_rerank_candidates` hard-excludes a
    confirmed-spilling candidate outright, before this term ever gets a
    chance to matter for it; this is a diagnostic weight for a caller
    reading `estimated_cost` directly."""
    spill_term = weights.spill_penalty if metrics.spill_free is False else 0.0
    return weights.radix_risk_penalty * metrics.radix_risk_score + spill_term


def _execution_cost(metrics: PlanMetrics, weights: CostWeights) -> float:
    """Execution-strategy terms: `total_worker_stage_batches` (the busiest
    worker's own serial SIMD-iteration count, summed over every stage --
    see that field's own docstring) is the one PlanMetrics field that
    depends on *how a kernel runs*, not on its own radix/memory shape.

    Reuses `weights.stage_work` (the same per-stage-work rate `_compute_
    cost` already charges per stage, applied here per serial batch
    instead) rather than a new constant -- chosen 2026-08-30 after
    comparing this against `worst_worker_utilization`-based models
    (worst-case/simple-average/work-weighted-average, all [0,1]
    utilization fractions) on 55 real measured candidates: every
    utilization-fraction model normalizes away the exact quantity that
    matters (workers_per_fft=2 and =4 can both look like "every worker
    100% busy" on the same evenly-split stage, since utilization only
    asks "is any worker idle," never "how many batches is the busiest
    one left with") -- see docs/execution_cost_model_validation.md for
    the full comparison (mean Spearman correlation: -0.15 worst-case,
    -0.21 both averaging variants, +0.46 this one; worker-pair prediction
    accuracy: 65% worst-case vs 78% this one). The natural place for a
    future real synchronization/barrier/exchange-overhead term to land
    once this project's cost model tracks one (see this module's own
    docstring: every existing term is a static function of the plan,
    nothing here models cooperative communication cost yet).

    Also (2026-08-31) the only place `compute_lanes` reaches the cost
    model at all: `total_worker_stage_batches` folds in each stage's own
    `chunks_per_batch` (see that field's own docstring) so a narrower
    compute_lanes -- more, smaller vector chunks doing the identical
    butterfly work -- actually costs more here, real emitted-chunk counts
    rather than an arbitrary penalty (docs/compute_lanes_joint_search.md's
    own "Phase 7-1" item, closed out by this change).

    Mechanism-aware correction (2026-09-01, docs/
    active_ndp_units_cost_task.md): a `stage_work` reweight alone was
    tried and rejected first (real hardware data directly falsified it --
    persistent has the systematically LOWEST `total_worker_stage_batches`
    at every N tested yet is not always the fastest, so scaling one
    positive weight up only entrenches the wrong pick further, never
    fixes it). `total_worker_stage_batches` is split into its persistent
    and non-persistent components (`PlanMetrics.persistent_worker_stage_
    batches`) and re-weighted separately: persistent's own batches at
    `persistent_stage_batch_multiplier` instead of `1.0`, plus a
    separate `persistent_extra_round_multiplier * persistent_extra_
    rounds` term for any persistent leaf whose own replica count needs
    more than one host-orchestrated round -- see both weights' own
    docstrings for the real-hardware derivation of each. A plan with no
    persistent leaf at all (`persistent_worker_stage_batches ==
    persistent_extra_rounds == 0`) computes byte-identically to before
    this change."""
    nonpersistent_worker_stage_batches = (
        metrics.total_worker_stage_batches - metrics.persistent_worker_stage_batches
    )
    weighted_stage_work = (
        nonpersistent_worker_stage_batches
        + weights.persistent_stage_batch_multiplier * metrics.persistent_worker_stage_batches
        + weights.persistent_extra_round_multiplier * metrics.persistent_extra_rounds
    )
    return weights.stage_work * weighted_stage_work


def estimate_cost(metrics: PlanMetrics, weights: CostWeights = DEFAULT_COST_WEIGHTS) -> float:
    """Total estimated cost -- the sum of four independently-documented
    sub-scores (`_memory_cost`/`_compute_cost`/`_resource_cost`/
    `_execution_cost`), split out (2026-08-30) so each can be read,
    reasoned about, and eventually calibrated against real measurements
    on its own, without the others -- see this module's own docstring for
    why a fitted, unified formula was never the goal here. Purely a
    regrouping of the exact same terms this function already summed
    before the split (same weights, same total for any given `metrics`) --
    not a ranking change."""
    return (
        _memory_cost(metrics, weights)
        + _compute_cost(metrics, weights)
        + _resource_cost(metrics, weights)
        + _execution_cost(metrics, weights)
    )
