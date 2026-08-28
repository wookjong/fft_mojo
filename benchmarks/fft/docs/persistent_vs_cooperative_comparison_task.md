# Continue Persistent vs Cooperative FFT Comparison — Spill-Free Apples-to-Apples

The persistent FFT implementation is now correct and committed.

Important: do NOT redesign or revert the current persistent round mechanism.

The previous "one function per (phase × round)" design was proven wrong on real M2NDP-Detour hardware because the runtime allows at most:

    max_kernel_register = 8

distinct registered kernel functions per task.

This is a hard architectural limit.

The corrected persistent design is:

    one registered function per phase
        preload
        stage_0
        stage_1
        ...
        writeback

with the current round number stored in scratchpad and advanced only by
writeback after all other phases in the round have consumed the current value.

This corrected design has already been validated on real hardware for
multiple rounds.

Treat:

    registered kernel functions > 8

as a HARD REJECT for any future candidate plan, not a performance penalty.


============================================================
CURRENT VERIFIED RESULT
============================================================

Real M2NDP-Detour result:

    N = 64
    logical_blocks = 40
    workers = 8

Persistent:
    correct
    spill-free
    11,903 ndp_cycles

Cooperative:
    correct
    8,763 ndp_cycles
    BUT spills in FFTRecLeaf0.stage_1

Therefore this result is NOT a fair performance comparison.

Do not conclude that cooperative is faster based on this data point.


============================================================
PRIMARY GOAL
============================================================

Continue only far enough to obtain a genuinely fair comparison where
the compared implementations are all:

    numerically correct
    spill-free
    same FFT length
    same radix sequence if possible
    same logical block count
    same precision
    same transform direction
    same target configuration

The main comparison should be:

    cooperative
    persistent

If practical, also include:

    non-cooperative

But cooperative vs persistent is the minimum required comparison.


============================================================
TASK 1 — FIND A SPILL-FREE COOPERATIVE CONFIGURATION
============================================================

Start from the existing N=64 / 40-block case.

Investigate why the cooperative plan spills.

Inspect:

    generated stage plan
    radix sequence
    register pressure
    loop-stage lowering
    narrow-middle-stage decisions
    risky radix handling
    spill_probe output

Try the smallest reasonable changes first.

Possible directions include:

    less aggressive radix coalescing
    smaller radix butterflies
    narrow_middle_stages
    loop-stage lowering
    a slightly smaller leaf size
    another already-supported radix sequence

Do NOT add arbitrary hand-written special cases just to make one benchmark pass.

Prefer a configuration that follows the existing planning mechanisms.

Every candidate must be run through:

    probe_spill_free

Confirmed spill:

    candidate reject


============================================================
TASK 2 — KEEP THE COMPARISON FAIR
============================================================

Once a spill-free cooperative candidate is found, build the persistent
candidate under equivalent conditions.

Prefer exact equality of:

    N
    radices
    logical_blocks
    forward/inverse
    input layout
    output layout

If the exact same radix sequence cannot be made spill-free for both strategies,
report that clearly.

In that case perform two comparisons:

A. Controlled comparison

    same mathematical kernel/radix sequence
    even if it is not individually optimal

B. Best spill-free comparison

    best valid spill-free cooperative plan
    best valid spill-free persistent plan

Do not mix these two interpretations.


============================================================
TASK 3 — COLLECT MORE THAN ONE DATA POINT
============================================================

Do not stop after one fair benchmark unless the environment makes further
runs prohibitively expensive.

Try to collect a small matrix such as:

    N = 32, 64, 128

and:

    logical_blocks = 1, 8, 32, 40

You do not need the full Cartesian product.

Prioritize cases that reveal different parallelism regimes:

    small block count
    approximately one block per NDP unit
    more blocks than NDP units

For the current 32-unit target, especially useful block counts are:

    1
    8
    32
    40

The goal is to see whether the winner changes with workload shape.


============================================================
TASK 4 — MEASURE THE RIGHT THINGS
============================================================

For every benchmark candidate record:

    FFT length
    radix sequence
    stage count
    logical block count
    execution strategy
    workers per FFT/group
    scratchpad bytes per unit
    registered kernel-function count
    spill/frame status
    ndp_cycles
    correctness error

Also record any important generated-code differences that plausibly explain
the result.

Do not report cycle numbers from incorrect or spilling candidates as if they
were valid performance results.


============================================================
TASK 5 — HARD CONSTRAINTS DISCOVERED FROM REAL HARDWARE
============================================================

Update the planner/cost-model constraints if not already done.

At minimum:

    registered_kernel_function_count <= 8

must be a hard feasibility constraint.

For the corrected persistent implementation, function count should depend on:

    preload
    number of FFT stages
    writeback

and NOT on round count.

Expected form:

    registered_functions = stage_count + 2

Round count should increase repeated launch_parallel executions, but must not
increase the number of distinct registered functions.

Add a structural test for this.

For example:

    2 rounds
    8 rounds
    32 rounds

should all emit/register the same distinct phase-function set.


============================================================
TASK 6 — DO NOT REGRESS THE ROUND-COUNTER FIX
============================================================

The current scratchpad round-counter mechanism was hardware-validated.

Do not replace it with per-round generated phase functions.

Verify that:

    preload reads current round
    every stage reads the same current round
    writeback reads the same current round
    writeback increments the round counter only after completing that round

Also preserve the harness behavior that freezes the round-counter/tracker value
for all workers within one simulated phase call.

The Python verification harness must model the hardware phase semantics, not
sequentially expose tracker updates within the same phase.


============================================================
TASK 7 — OPTIONAL NON-COOPERATIVE BASELINE
============================================================

If easy to obtain, include non-cooperative as a third baseline.

Use the same fairness rules:

    correct
    spill-free
    same workload

This would give:

    non-cooperative
    cooperative
    persistent

under the same workload shape.

Do not delay the cooperative-vs-persistent comparison substantially just to
include non-cooperative.


============================================================
STOPPING CONDITION
============================================================

A good stopping point is reached once we have at least:

    one controlled spill-free cooperative vs persistent comparison

preferably plus:

    2–4 additional spill-free data points across different
    N / logical-block regimes

At that point, stop expanding scope and summarize the trend.

Do NOT try to build the final execution-strategy cost model yet.

The immediate goal is empirical characterization.


============================================================
FINAL REPORT
============================================================

Report:

1. why the original per-(phase × round) design failed
2. the real max_kernel_register=8 constraint
3. the current round-counter fix
4. the cooperative spill root cause
5. what changes were needed to obtain a spill-free cooperative plan
6. the exact fairness conditions used
7. a comparison table

Suggested table:

| N | Blocks | Radices | Strategy | Spill | Scratchpad | Registered funcs | Cycles |
|---|---:|---|---|---|---:|---:|---:|

8. controlled same-radix comparison
9. best spill-free comparison, if different
10. whether performance appears sensitive to:
       FFT leaf size
       logical block count
       worker utilization
11. whether there is enough evidence yet to derive an execution-strategy
    selection rule

Do not claim a general winner from a single benchmark.

If the results show a crossover between cooperative and persistent as
logical-block count changes, highlight that clearly because it directly informs
the future execution-strategy selector.


============================================================
IMPLEMENTATION ORDER
============================================================

Proceed in this order:

    inspect cooperative spill
        ->
    find spill-free cooperative candidate
        ->
    match persistent workload
        ->
    verify both numerically
        ->
    verify both spill-free
        ->
    real hardware cycle comparison
        ->
    collect a few additional points
        ->
    summarize trend
        ->
    stop

Do not redesign the persistent architecture unless a new real-hardware
correctness failure requires it.
