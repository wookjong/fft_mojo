# Execution option search (step 6)

Native `generate_candidates` now explores the implemented cooperative partition,
persistent copy, and persistent batch-tail alternatives. The default heuristic
build remains unchanged; candidate selection (`--plan-index`) and spill/cycle
search use the expanded pool.

## Bounded candidate generation

Step 12 adds three independent search arms around already feasible candidates:

- Cooperative: round-robin, contiguous, balanced-contiguous batch assignments.
  One leaf changes at a time. Seeds with physically identical assignments for
  all modes are skipped; realized worker batch IDs participate in deduplication.
- Persistent: scalar or vectorized-contiguous preload/writeback for one leaf at
  a time. The plan carries the selected copy strategy and default codegen follows
  it, including mixed recursive trees.
- Tail: all-persistent remains the existing root candidate. Non-cooperative and
  cooperative hybrid tails are offered for unsplit persistent roots with at least
  one full round and a genuine remainder (`batch > num_ndp_units`, nonzero mod).
  Existing unsplit-persistent capacity/performance gates remain in place. No
  hybrid candidate bypasses the gate for large FFTs that should decompose.

The default seed cap is four per family, with at most 48 added variants before
normal feasibility checks and overall selection. Families are interleaved before
this cap. Neither new options nor old split/radix/worker/tile axes form a full
Cartesian product. `max_execution_option_candidates=0` or
`max_execution_option_seeds=0` disables the new arms. The existing overall cap
(default 40) and baseline retention still apply. Therefore not every option is
promised to survive every top-K pool.

`PlanChoices.cooperative_partitions`, `persistent_copies` and `tail_strategy`
report realized decisions; the plan itself is authoritative for deduplication,
metrics and code generation. Default choices are recorded too, so reports show
scalar versus vector copies and all-persistent versus actual hybrid tails.

## Hybrid execution and accounting

A root `FFTLeafPlan.tail_hybrid` holds the existing `PersistentTailHybridPlan`.
Flattening returns its actual full-round and tail task kernels; the original
whole-batch kernel is not counted as an additional task. Nested hybrid leaves
are rejected: recursive segment binding is not implemented by the existing
hybrid mechanism and is not implied by this search extension.

Metrics sum actual task element counts. Two disjoint batch tasks incur one
read/write of the combined batch, not two serial full-array passes. Persistent
extra-round counts use the full portion's replica count; registration limits
apply separately to each task, matching codegen's distinct function inventory.
Hybrid tails use the existing unlooped stage emitter.

Tail factories now inherit target scratchpad limits. Resource-capped tails use
multiple host launches with corresponding logical FFT pointer offsets, rather
than launching more microthreads than their scratchpad slots allow. Both forward
and inverse partitions use the same overall per-transform normalization.

Recursive rendering and generic spill probing dispatch hybrid roots through the
actual two-task generator. `reference_check=False` produces one combined INPUT
and OUTPUT marker pair, with full replicas followed by tail replicas. Thus the
existing NumPy verification flow can inspect the entire batch rather than only
one component. No-tail and tail-only generator paths retain their reference-check
behavior and also support dump mode.

## Cost and performance limits

Existing worker-load metrics score the actual cooperative partitions. Copy
strategies expose `persistent_copy_instructions`: the busiest physical lane's
load/store count for real and imaginary data, both directions and all rounds,
using the emitter's default vector width 4 and scalar cleanup for ragged chunks.
DRAM byte estimates remain equal for scalar and vector copies.

`CostWeights.persistent_copy_instruction` defaults to zero because no instruction
latency has been calibrated. An explicit weight can score that proxy, while
`probe_and_rerank_candidates(..., rank_by_cycles=True)` can compare actual measured
cycles. The static model does not prove a partition/copy/tail speedup and does
not newly model all host-launch or communication overheads.

## Validation

`python -m verification.verify_fft_execution_option_search` contains ten tests:
all three axes and physical deduplication, caps and baseline retention, batch
boundaries 1/31/32/33/40/63/64/65, actual hybrid metrics, generated source for all
representative candidates, forward/inverse option numerics, resource-capped tail
rounds, copy reporting/optional cost weights, two-task probe dispatch, high-level
plan-index selection, rejection of nested hybrids, and retention of the existing
large-FFT persistent gate. Several checks are grouped in individual test methods.

The numerical harness interprets emitted device bodies, not an FFT algorithm
reimplementation. Existing search, persistent options, launch support, execution
invariants, registration limits and GPU baseline suites also pass. Actual Mojo
compilation, M2NDP simulation and performance measurement remain unavailable in
this environment.

```sh
python make_fft_kernel.py 64 --batch 33 --dump-candidates
python make_fft_kernel.py 64 --batch 33 --plan-index K --no-reference-check
```

Choose K from that invocation's ranked dump; indexes depend on N, batch and
search settings. Use the existing cycle-probe workflow with sufficient top-K to
compare options on the real toolchain.
