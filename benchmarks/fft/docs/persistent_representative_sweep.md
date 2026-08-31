# Persistent execution: representative sweep and cost-model validation (2026-08-30)

Follow-up to `execution_cost_model_validation.md`'s own "persistent as a
search axis" section (step 10 added, single N=216 real data point). This
pass measured across 7 representative N, explained WHY persistent wins
where it wins (code-level, not guessed), found the cost model gets large N
backwards, and found + fixed a live footgun this created in `generate_
candidates` before it shipped further. Simulator source untouched
throughout (read-only, for the vlenb root-cause work covered elsewhere).

## Representative sweep: non-cooperative vs best-cooperative vs persistent

Same radix (default tier, `coalesce_radices` unforced), batch=1,
`compute_lanes=4` (shipped default) throughout. "best-cooperative" is the
single lowest-`estimated_cost` real cooperative candidate for that N (not
rescued via compute_lanes narrowing -- see `compute_lanes_spill_
avoidance.md` for that separate axis); its own spill status is reported
honestly, not cherry-picked safe.

| N | shape | non-cooperative | best-cooperative | persistent |
|---|---|---|---|---|
| 30 | tiny/single-leaf | 8763, safe | 6830, safe | **6306, safe** |
| 64 | small/radix-2-heavy | 13133, safe | 7770, SPILLS | **7676, safe** |
| 105 | small/odd-radix | 26732, SPILLS | 15305, SPILLS | **14466, safe** |
| 144 | medium/wide-tier-relevant | 26062, safe | 17580, SPILLS | **14551, safe** |
| 256 | medium/radix-2-heavy | 38836, safe | 25596, SPILLS | **23605, safe** |
| 960 | large, needs split | 2114, safe | 2107, SPILLS | 109074, safe |
| 1024 | large, needs split | 2111, safe | 2111, SPILLS | 110976, safe |

(Cycle counts; "safe"/"SPILLS" is `spill_free` from a real build+run.
Persistent's own correctness is always checked -- `generate_persistent_
fft_kernel` bakes the reference check in unconditionally -- and was PASS
in every row above, including both N=960/1024 "loss" cases.)

**Persistent vs. non-cooperative, classified at >5%:**
- **Win** (5/7): N=30 (-28.1%), N=64 (-41.6%), N=105 (-45.9%), N=144
  (-44.2%), N=256 (-39.2%). Geometric mean speedup: **~40% faster**.
- **Loss** (2/7): N=960 (+5060%, ~51.6x slower), N=1024 (+5162%, ~52.6x
  slower). Not marginal losses -- catastrophic.
- No near-tie (±5%) case in this sample.

**Best-cooperative is not usable as-is in 5/7 cases** (spills at the
shipped default compute_lanes) -- persistent's own real comparison set is
mostly "vs. non-cooperative," the only other candidate actually safe to
pick without a separate compute_lanes rescue.

## Why persistent wins for small/medium single-leaf N (N=216 traced in code)

Not guessed -- traced through real generated Mojo source and the
`compute_stage_metrics` numbers directly.

- **DRAM traffic: identical.** Both non-cooperative and persistent are a
  single fused leaf here (no split); both read the whole input from DRAM
  once, keep every intermediate stage in scratchpad (`buf_a`/`buf_b`
  ping-pong for non-cooperative, the persistent leaf's own scratchpad
  region), and write the whole output once. Non-cooperative's `stage_1`
  body reads from `FFTRecLeaf0.buf_a` (scratchpad), not DRAM -- confirmed
  directly in the generated source, not assumed from the design doc's own
  high-level diagram (which describes the split/multi-kernel case, not
  this one).
- **Arithmetic work: identical.** Same radix decomposition, same
  butterfly counts.
- **The actual difference: real parallelism.** Non-cooperative's own
  `comptime MAX_UTHREAD_FFTRecLeaf0 = 1` -- ONE physical microthread does
  the entire FFT serially, batch=1. Persistent's generated source shows
  `var worker_id = gid % 8` with explicit `if worker_id == 0: ... elif
  worker_id == 1: ...` dispatch in every stage -- confirmed via `compute_
  stage_metrics`: `workers_per_fft=8` on every stage. Persistent is
  running the equivalent of **cooperative worker=8** on this one block.
- **Why cooperative itself can't just use worker=8**: `workers_per_fft ==
  interleave_chunk_uthreads` (8) is explicitly excluded from the ordinary
  cooperative path's own candidate list (`fft_plan_cooperative.
  worker_candidates_per_fft`'s `exclude_full_interleave_chunk=True`
  default) -- a confirmed real-hardware wrong-answer bug, root cause never
  identified at the instruction level (see that function's own docstring).
  Persistent's different addressing (round-based, `software_group_id`/
  `round_base`, preload/writeback split from stage computation) evidently
  avoids whatever breaks the plain cooperative path at exactly 8 workers
  -- confirmed correct (PASS) at every N tested this pass, including the
  two that lose on cycles. This is itself a useful clue for anyone later
  root-causing the cooperative worker=8 bug: whatever breaks it is
  specific to that path's own addressing/sync mechanism, not an inherent
  hardware limit at 8-workers-per-group.
- Launch count is the same either way (7 `launch_parallel` calls for both,
  N=216) -- the win is not about fewer kernel launches.

## Why persistent loses badly for large N (N=960/1024)

Persistent has no split/recursion support (see "Split support" section
below) -- it always renders the WHOLE length-N transform as one fused
leaf. For N=960/1024, that means:

- Only **one of `target.num_ndp_units` (32) physical NDP units** is ever
  active (`num_logical_blocks=1` -> 1 active software group; the other 31
  immediately return via `if software_group_id >= active_groups: return`).
- The non-cooperative/cooperative structure, by contrast, is forced to
  `make_recursive_transpose_plan`'s own PRE/MIDDLE/POST-transpose split at
  this size, and those transpose stages alone launch dozens of physical
  microthreads spread across many/all 32 units concurrently (`PRE total_
  uthreads=60` for N=960) -- real cross-unit parallelism persistent's
  single-leaf shape cannot get any of.
- Net effect confirmed directly: even though N=960 requires objectively
  *more* total arithmetic work than N=256, its non-cooperative/cooperative
  cycle count (2114/2107) is **far lower** than N=256's (38836) --
  `ndp_cycles` measures when the *last* of many concurrent launches
  finishes, not total work, so heavy cross-unit parallelism dominates
  total-work differences by a wide margin. Persistent's single-leaf shape
  forgoes that parallelism entirely for N large enough to need it.

## Cost model: gets small N right in direction, badly wrong in magnitude, and gets large N backwards

| N | non-coop total cost | persistent total cost | model says | real result |
|---|---|---|---|---|
| 30 | 480.8 | 480.6 | persistent ~0.04% cheaper | persistent 28% faster |
| 64 | 1024.9 | 1024.6 | persistent ~0.03% cheaper | persistent 42% faster |
| 105 | 6681.3 | 6680.6 | persistent ~0.01% cheaper | persistent 46% faster |
| 144 | 2306.6 | 2304.8 | persistent ~0.08% cheaper | persistent 44% faster |
| 256 | 4099.6 | 4096.8 | persistent ~0.07% cheaper | persistent 39% faster |
| 960 | 76693.8 | 15362.5 | persistent **80% cheaper** | persistent **50x slower** |
| 1024 | 81789.8 | 16386.5 | persistent **80% cheaper** | persistent **50x slower** |

Direction-only pairwise accuracy: 5/7 (71%) -- looks passable until you
see WHY: for the 5 "correct" small-N cases, `memory_cost` (identical
DRAM-byte estimate for both, since both are un-split single leaves) so
totally dominates the ~0.5-2.2-unit `execution_cost` gap that the model's
own predicted margin (0.01-0.08%) is two to three orders of magnitude
smaller than the real one (28-46%) -- right direction, no real signal on
magnitude. For N=960/1024 the model isn't just imprecise, it is **inverted**:
persistent's `estimated_dram_bytes` (a single leaf's own stage count x N)
looks far smaller than the split structure's own (multiple transpose
kernels each moving the whole array), so `memory_cost` -- the dominant
term everywhere -- rewards exactly the shape that is 50x slower in
reality.

**Information genuinely missing from the model, not a weight problem**:
no representation at all of *how many physical NDP units are concurrently
active* (`software_group_count` vs. `active_groups`, or equivalent
tile-parallelism for the split/transpose case). `estimated_dram_bytes`
counts total bytes moved as if by one serial stream; it has no channel for
"but N of those bytes move over N different physical units at once." No
new weight can fix this -- the quantity the model would need to weight
does not exist in `PlanMetrics` at all yet. Not added this pass (per this
investigation's own explicit scope: no arbitrary weight, no cost-model
rewrite).

**Found and fixed the direct consequence, same session**: before a
mitigation existed, `generate_candidates(960)`'s own `rank_candidates`
put the persistent candidate at **#1 of 89** by `estimated_cost` (15362.5
vs. the next candidate's 71291.3 -- not a close call) -- meaning `probe_
and_rerank_candidates` with the default `rank_by_cycles=False` would very
likely have recommended a candidate confirmed ~50x slower than the
correct answer. Fixed via `_persistent_leaf_feasible` (see `fft_plan_
search.py`): persistent is now only offered when N would ALSO fit as a
single fused leaf under the *cooperative family's own* `scratchpad_byte_
budget` (not persistent's own much more permissive `16*n <=
spad_capacity_bytes`, which alone allows N up to 7680) -- a structural
feasibility gate, not a cost-weight tweak, chosen because it perfectly
separates this sweep's 5 wins from its 2 losses (every win has `_leaf_
scratchpad_bytes(n) <= 4096`; both losses don't). N=960/1024 now generate
zero persistent candidates; regression test added (`verify_fft_
persistent_search.check_persistent_gated_off_where_measured_slower`).

## num_logical_blocks sensitivity (bounded {1,2,4,8})

| N | blocks=1 | blocks=2 | blocks=4 | blocks=8 |
|---|---|---|---|---|
| 216 | 23944 | 23894 | 23874 | 22131 |
| 64 | 7676 | 7694 | 7706 | 7709 |

Essentially flat both directions -- N=216 improves a marginal 7.6% by
blocks=8 (still only 1 of 32 groups doing anything different per round;
the whole 1-32 range shares one round), N=64 is flat-to-slightly-worse.
Consistent with the earlier single-N=216 finding (1 vs. 32 blocks nearly
identical). **Conclusion: not worth promoting to a tuning axis** --
matches Phase 5's own stated fallback ("대부분 N에서 block count가 cycle에
거의 영향을 주지 않는다면... canonical/default 값을 선택"). `num_logical_
blocks=batch` (today's behavior, whatever the caller's own batch is) stays
the right default; no separate sweep needed.

## radix tier x persistent (N=144, default vs. wide)

default: 14551 cycles. wide (`coalesce_radices` allows {4,6,9}): 13526
cycles -- persistent still benefits from the wide radix tier (~7% faster),
same direction as every other execution strategy. Radix and execution
strategy are not fully independent, but the *effect* here (wide tier
helps) matches what the joint radix x worker search already found for
cooperative -- no new, persistent-specific radix behavior discovered.

## compute_lanes x persistent (N=1024, single fused leaf despite the gate -- tested directly to probe the mechanism, not as a search candidate)

| lanes | spill_free | cycles |
|---|---|---|
| 1 | **False** (confirmed spilling) | 164863 |
| 2 | True | 135783 |
| 4 (shipped default) | True | **110976** |

Persistent is NOT immune to spill in general -- large enough N (1024,
single fused leaf) spills at `compute_lanes=1`. Unlike the cooperative
rescue cases (`compute_lanes_spill_avoidance.md`), narrower is *worse*
here on every axis (both cycles and spill), matching the general finding
that persistent, when it does spill, behaves like everything else: full
width first, narrow only if forced.

## inverse=True correctness (N=216)

spill_free=True, PASS, 24161 cycles (forward was 23944-23944 depending on
block count) -- persistent's own inverse path is correct and comparably
fast, not a forward-only special case.

## Split support: investigated, not implemented

User asked directly whether persistent's split-recursion gap (the root
cause of the N=960/1024 loss) could be fixed rather than merely gated.
Investigated `make_persistent_leaf_plan`/`_build_recursive_node`
(`fft_plan_recursive.py`)'s own leaf-construction contract:

- Architecturally plausible in principle: `total_ffts`/`r` (a near_fft's
  own replica count) maps conceptually onto persistent's own `num_
  logical_blocks`, the same way `cooperative_workers` already lets a
  near_fft be built via `make_cooperative_leaf_plan` instead of `_build_
  plan` today.
- Two concrete blockers found: (1) `make_persistent_leaf_plan` always
  builds its own `AddressMapping.contiguous(length)` mapping -- slotting
  into a near_fft position needs a mapping matching the PRE-transpose's
  own intermediate-buffer layout instead, not implemented; (2) more
  importantly, persistent is architecturally fixed at exactly `target.
  num_ndp_units` (32) software groups regardless of caller-supplied
  width (three separate `NotImplementedError` guards enforce this) --
  for N=960's own real split (A=4, B=240), the near_fft's own replica
  count is only 4, meaning even a correctly-wired persistent near_fft
  would leave 28/32 groups idle every round, the *same* underutilization
  that made the top-level case lose by 50x. This is not confirmed to fix
  N=960/1024 even if built.
- User's own decision (asked directly, given the address-mapping risk
  and the uncertain payoff for the exact regression this session found):
  **hold off**. Not attempted this pass. If revisited, first find a real
  N whose natural split produces a near_fft with a replica count close to
  32 -- validate the underutilization hypothesis is even the right
  target before investing in the address-mapping work.

## Search-space growth

Persistent adds exactly 1 candidate per feasible N (one radix tier,
`DEFAULT_TARGET_PROFILE` doesn't expose the wide tier since `supports_
vector_spill=False`): N=30 14->? / N=64 72->73 / N=144 79->80 / N=216
89->90 / N=256 84->85, roughly 1-7% relative growth. N=960/1024 add zero
(gated). Not a combinatorial concern.

## Multi-leaf mixed execution strategy: not attempted

Not measured or implemented this pass (explicitly out of scope: "먼저
representative case 몇 개를 수동으로 만들어... 의미가 없다면 search space를
넓히지 않는다") -- and, given the split-support finding above, current
infrastructure cannot build a persistent leaf as one branch of a
multi-leaf recursive plan at all yet (same address-mapping/replica-count
blockers), so this would need the split-support work above as a
prerequisite regardless.

## ExecutionStrategy representation: design note only, no refactor

Current representation (`PlanChoices.workers_per_fft`/`worker_sequence`/
`execution_strategy`) already prevents the one invalid combination that
matters in practice -- `generate_persistent_leaf_candidates` never sets
`workers_per_fft`/`worker_sequence` on a persistent candidate, and no
code path combines `execution_strategy="persistent"` with a cooperative
worker count. A future `ExecutionStrategy = NonCooperative | Cooperative
(workers) | Persistent(num_logical_blocks)` sum type would make that
invariant structural instead of by-convention, and would be worth doing
if/when split support (above) adds a second axis that genuinely needs to
vary per-leaf -- not warranted yet for a single top-level choice per plan.

## Correctness coverage

Confirmed PASS (real reference check, unconditionally baked into
`generate_persistent_fft_kernel`) across: forward (6/7 representative N;
the two N=960/1024 loss cases were also confirmed PASS before being
gated off), inverse (N=216), multiple radices (mixed: N=30 (2,3,5)-family,
N=105 odd-radix, N=144/N=256 wide-tier-relevant, radix-2-heavy N=64/256),
`num_logical_blocks` in {1,2,4,8,32}, `compute_lanes` in {1,2,4}, default
and wide radix tier. Not tested: `batch>1` (multiple independent length-N
transforms in one launch, as opposed to `num_logical_blocks>1`'s "one
transform, many rounds" -- these are different knobs; `generate_
persistent_leaf_candidates` currently sets `num_logical_blocks=batch`,
conflating them, worth flagging for whoever next touches this: a caller
wanting `batch=4` independent transforms gets `num_logical_blocks=4`
today, which is a real but untested equivalence).
