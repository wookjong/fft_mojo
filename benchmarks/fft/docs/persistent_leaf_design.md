# Persistent Software-Workgroup FFT Leaf (standalone, opt-in)

## Context

`/root/fft_mojo/benchmarks/fft/` needs a new, standalone, opt-in FFT leaf execution model: a fixed number of **persistent software workgroups** (`workers_per_group=8` workers each, one workgroup per physical NDP unit) that each own one physical scratchpad region and process **many logical FFT blocks sequentially, across rounds, inside one host-level launch** — instead of today's pattern where physical launch width scales with logical demand.

Deliverable is exactly two new entry points:

- `make_persistent_leaf_plan(...)`
- `generate_persistent_fft_kernel(...)`

The new path must be validated numerically and on real M2NDP-Detour hardware, and kept fully separate from `make_fft_kernel.py` / the recursive planner / cost model. No wiring into automatic plan selection is part of this effort.

Priority order:

1. correctness
2. correct M2NDP mapping interpretation
3. numerical verification
4. spill-free generated code
5. performance

## Implementation status

**This design is now considered implementation-ready. Proceed with implementation immediately.**

Do not redesign the persistent FFT architecture unless the live repository directly contradicts one of the assumptions below. If that happens, report the discrepancy and make the smallest correctness-preserving adaptation.

One final structural-test correction must be applied while implementing:

- Do **not** count every generic occurrence of `.launch(` in the generated source when proving that there is one host-level persistent FFT launch.
- Count only the persistent FFT kernel's own host launch using its exact generated kernel class name, or another equally precise pattern.

For example:

```text
count("PersistentFFT.launch(") == 1
```

If the generated class has a different name, use that actual class name.

This correction affects only the launch-structure test. It does **not** change the execution architecture.

**Core invariant, unchanged by anything below: existing FFT mathematics are not touched.**

The persistent path changes only:

* where stage input comes from,
* where stage output goes,
* how many workers cooperate,
* how rounds reuse one NDP unit's scratchpad.

It must never change:

* radix decomposition,
* butterfly arithmetic,
* twiddle exponents,
* the Stockham permutation,
* first-stage indexing,
* final output order,
* inverse normalization.

```text
existing:    DRAM -> FFT stages -> DRAM

persistent:  DRAM
               -> preload
               -> scratchpad FFT stages
               -> writeback
               -> DRAM
```

This plan has been revised three times from an earlier draft by direct user review before implementation started. This version is the accepted architecture with review corrections applied throughout; it is not a redesign.

---

## Verified hardware facts

Confirmed by an Explore agent reading `third_party/m2ndp-detour/src/` directly, not just documentation, and cross-checked against the same-day prior-session findings already represented in `planning/target_profile.py` (`DEFAULT_TARGET_PROFILE`) and `codegen/fft_transpose_codegen.py`'s `_safe_round_size`.

### NDP-unit mapping

`M2NDPConfig::get_matched_unit_id`, `m2ndp_config.h:91-93`:

```text
unit(addr) = (addr // stride) % num_ndp_units
```

Current live values:

```text
stride = 256 bytes
UTHREAD_BYTES = 32 bytes
num_ndp_units = 32
```

`stride = 256` comes from the `m_stride_size` header default; the live config does not explicitly set `ndp_stride`, which should be stated precisely in the final report rather than described as a config-driven value.

Therefore:

```text
interleave_chunk_uthreads
    = 256 / 32
    = 8
```

This is a **derived ratio**, not a named hardware constant and not a residency cap.

The implementation must read the corresponding values from the target profile rather than re-hardcoding literal `8`, `32`, or `256`.

### Residency

Current configuration:

```text
4 sub-cores
× 16 uthread_slots
= 64 resident slots / NDP unit
```

`workers_per_group=8` is therefore much smaller than the available residency.

This is only a resource/performance fact.

**Correctness must not depend on all workers remaining resident across phases.**

### `launch_parallel` synchronization

The narrow property this design needs is:

> this execution model does not rely on a µthread register/context surviving across separate `launch_parallel` calls.

Each `launch_parallel` is a synchronous doorbell/completion handshake (`m2ndp_launch_abi.h`) and provides the phase boundary used by this design.

The implementation must not rely on a device-side barrier or wait loop.

Every:

```text
preload
FFT stage
writeback
```

phase is therefore a separate `launch_parallel`.

### `group_id()`

Confirmed:

```text
group_id() = physical NDP unit index
num_groups() = num_ndp_units
```

This comes from `register_unit.cc:85-105`, using `req->own_unit`.

`group_id()` is used only by the ID-mapping diagnostic as an independent observation of which hardware unit a µthread actually ran on.

It must not participate in persistent worker/group mapping.

### `local_uthread_id()`

`local_uthread_id()` density is **not** relied on.

Persistent worker/group identity is derived only from `global_uthread_id()`.

The diagnostic may record `local_uthread_id()` for observation, but no correctness path may depend on it.

### µthread pool alignment

`Pool.alloc` in `src/m2ndp_host.mojo:175-192` aligns individual allocations only to:

```text
_POOL_ALIGN = 64 bytes
```

Although the pool's own base is 256B aligned, a particular `cxl_alloc` address depends on previous allocations and is therefore not guaranteed to be 256B aligned.

The generated host `main()` must therefore:

1. over-allocate the µthread-pool buffer by at least 256 bytes,
2. advance the usable pointer to the next 256B boundary,
3. create `PooledRange` from that aligned pointer,
4. assert at runtime:

```text
pool_addr % 256 == 0
```

Misalignment must fail loudly rather than silently corrupting group-to-unit mapping.

---

## Launch-width derivation

SIMD arithmetic width and µthread spawn granularity are different concepts and must remain separate.

Use:

```text
launch_uthreads
    = software_group_count * workers_per_group

launch_bytes
    = launch_uthreads * UTHREAD_BYTES

launch_elements
    = launch_bytes / sizeof(Float32)
```

For the current target:

```text
launch_uthreads
    = 32 * 8
    = 256 uthreads

launch_bytes
    = 256 * 32
    = 8192 bytes

launch_elements
    = 8192 / 4
    = 2048 Float32 elements
```

`software_group_count` numerically equals `target.num_ndp_units`, but the implementation must obtain it from the target profile.

The fixed `launch_elements` is the `PooledRange` size for the single host-level launch.

It does not shrink for a tail round.

---

## Worker / software-group / logical-block identity

`software_group_id` must never be assumed equal to the physical NDP unit index.

The required property is:

* all workers belonging to one `software_group_id` map to the same physical NDP unit,
* the software groups form a one-to-one mapping onto the physical units,
* that mapping may be a permutation.

For a properly aligned launch pool:

```text
actual_unit
    = ((pool_base // stride) + software_group_id)
      % num_ndp_units
```

Every phase independently computes:

```text
gid = global_uthread_id()

worker_id
    = gid % workers_per_group

software_group_id
    = gid // workers_per_group

if software_group_id >= ACTIVE_GROUPS:
    return

logical_block
    = ROUND_BASE + software_group_id

block_base
    = logical_block * length
```

`ROUND_BASE` and `ACTIVE_GROUPS` are compile-time constants baked into each emitted round-specific phase function.

They are not runtime Params.

All 8 workers sharing one `software_group_id` therefore compute the exact same `block_base`.

Never use:

```text
global_uthread_id() * length
```

as the logical FFT block base.

That would incorrectly assign a different logical FFT to every worker.

Because:

```text
workers_per_group
    == interleave_chunk_uthreads
```

the 8 consecutive `gid`s belonging to one software group coincide with the 8 µthreads placed into one hardware interleave stripe.

---

## Scratchpad ownership and footprint

There is exactly one persistent software workgroup per physical NDP unit.

Scratchpad is physically replicated per unit and shared by all µthreads executing on that unit.

Therefore:

```text
spad_base = 0
```

for every worker.

Do not:

* allocate scratchpad per worker,
* multiply scratchpad size by `workers_per_group`,
* create cooperative-style `fft_slot` subregions.

For split-complex FP32:

```text
real bank = N * 4 bytes
imag bank = N * 4 bytes

one FFT bank = 8N bytes
```

Persistent execution always uses two banks:

```text
required scratchpad
    = 16N bytes / NDP unit
```

`make_persistent_leaf_plan` must reject a plan if:

```text
16 * length > target.spad_capacity_bytes
```

For the default 120KiB budget:

```text
120 * 1024 / 16 = 7680
```

Two separate limits must be reported:

### Byte-capacity upper bound

```text
N <= 7680
```

for the default profile.

### Actual planner-supported maximum

The largest FFT length at or below that byte limit for which the planner can actually produce a supported radix factorization and stage layout.

Do not claim `7680` itself is necessarily a supported FFT size.

---

## Buffer count — always two banks

Do not reuse `pingpong_needed()` blindly.

Persistent execution is:

```text
preload
    ->
stage 0
    ->
stage 1
    ->
...
    ->
stage k-1
    ->
writeback
```

Every FFT stage reads scratchpad and writes scratchpad.

For every plan with at least one FFT stage:

```text
preload:
    write buf_a

stage i:
    read  buf[i % 2]
    write buf[(i + 1) % 2]

writeback:
    read buf[stage_count % 2]
```

Always allocate:

```text
buf_a
buf_b
```

No one-bank or single-stage in-place optimization is part of this effort.

---

## Fixed lowering approach — `force_scratchpad`

The most important lowering rule is:

> preserve mathematical stage position exactly and change only the physical storage endpoint.

Existing `layouts_for_radices` already computes stage-position-dependent properties such as:

* `output_stride`,
* `twiddle_modulus`,
* `twiddle_lane_divisor`.

These remain unchanged.

The DRAM/scratchpad conflation exists in `_make_load` / `_make_store`.

Current behavior conceptually includes:

```text
first_stage=True
    -> input DRAM

last_stage=True
    -> output DRAM
```

However, `first_stage` / `last_stage` also determine mathematical FFT semantics.

Therefore persistent lowering must **not** fake first/last stages as middle stages.

### Required minimal change

Add:

```python
force_scratchpad: bool = False
```

to `_make_load` and `_make_store`.

Default behavior must remain exactly unchanged for every existing caller.

When `force_scratchpad=True`:

### `_make_load`

If `first_stage=True`:

* preserve the exact existing first-stage `base_offset`,
* preserve all mathematical indexing,
* change only:

```text
source:
    input -> scratchpad

buffer_name:
    -> read_buffer
```

### `_make_store`

If `last_stage=True`:

* preserve the existing natural-final-order store offset,
* preserve all final-stage semantics,
* change only:

```text
destination:
    output -> scratchpad

buffer_name:
    -> write_buffer
```

Do not alter:

* `base_offset`,
* permutation,
* twiddle metadata,
* stage layout,
* scale,
* any mathematical stage semantics.

---

## Explicit inverse-scale preservation

Inverse normalization is not inherited automatically.

The persistent path uses its own:

```text
_lower_persistent_stages()
```

rather than `_lower_stages()`.

Therefore `_lower_persistent_stages()` must explicitly reproduce the existing rule.

For every stage:

```python
first_stage = stage_id == 0
last_stage = stage_id == stage_count - 1

scale = inverse_scale if last_stage else None
```

with:

```python
inverse_scale = 1.0 / length if inverse else None
```

The final persistent FFT stage must therefore have:

```text
last_stage = True
scale      = same value as ordinary lowering
```

even though its destination is scratchpad.

Inverse scaling must not be moved to writeback unless preserving the existing rule is found impossible in the live implementation.

Required invariant:

```text
persistent final-stage scale
    ==
normal final-stage scale
```

---

## `_lower_persistent_stages`

The persistent path does not call `_lower_stages()` directly.

Implement a small persistent-specific loop in:

```text
planning/fft_plan_persistent.py
```

that structurally mirrors `_lower_stages()` while using:

```text
_make_load(... force_scratchpad=True)
_make_twiddle(...)
_make_store(... force_scratchpad=True)
```

and the fixed two-bank scheme.

`_lower_stages()` itself remains untouched.

Preload and writeback are separate bulk-copy phases and are not FFT stages.

They should not go through `_make_load` / `_make_store`.

---

## Module ownership

To avoid circular imports:

### `planning/fft_plan_core.py`

Contains data structures:

```text
PersistentWorkgroupPlan
FFTCodegenPlan
FFTStagePlan
```

and the additive persistent fields.

### `planning/fft_plan_persistent.py`

Contains persistent algorithms:

```text
make_persistent_leaf_plan(...)
_lower_persistent_stages(...)
_partition_vector_scalar(...)
persistent-specific helpers
```

`PersistentWorkgroupPlan` therefore follows the existing `CooperationPlan` pattern.

Suggested fields:

```text
stripes_per_group
workers_per_stripe
workers_per_group
software_group_count
logical_block_stride
scalar_worker_mode
```

---

## Target-mapping invariant checks

`make_persistent_leaf_plan()` must not simply read the target values and assume they are mutually compatible.

Explicitly validate:

```text
mapping_stride % target.uthread_bytes == 0

target.interleave_chunk_uthreads
    == mapping_stride // target.uthread_bytes

workers_per_group
    == target.interleave_chunk_uthreads

software_group_count
    == target.num_ndp_units

stripes_per_group
    == 1
```

The current runtime implementation only supports:

```text
stripes_per_group = 1
```

If an invariant fails, reject the plan clearly.

Do not silently generate a mapping that no longer matches the target.

If `TargetProfile` does not expose a required hardware parameter, add it to the profile rather than hardcoding the value in the persistent planner.

---

## Round loop

Let:

```text
B = num_logical_blocks
G = software_group_count
```

Then:

```text
rounds = ceil(B / G)
```

For round `r`:

```text
ROUND_BASE
    = r * G

ACTIVE_GROUPS
    = min(G, B - ROUND_BASE)
```

Example for 40 blocks and 32 groups:

```text
round 0:
    blocks 0..31
    ACTIVE_GROUPS = 32

round 1:
    blocks 32..39
    ACTIVE_GROUPS = 8
```

Groups 8..31 return immediately in round 1.

---

## Exactly one host `.launch()`

This is a hard requirement.

All rounds must be Python-unrolled inside one generated `device_main`.

Example:

```text
def device_main():

    launch_parallel[Kernel.preload_r0]()
    launch_parallel[Kernel.stage_0_r0]()
    launch_parallel[Kernel.stage_1_r0]()
    ...
    launch_parallel[Kernel.writeback_r0]()

    launch_parallel[Kernel.preload_r1]()
    launch_parallel[Kernel.stage_0_r1]()
    ...
    launch_parallel[Kernel.writeback_r1]()
```

Each:

```text
preload_rX
stage_Y_rX
writeback_rX
```

is an independently generated `@staticmethod`.

Do not reuse `fft_transpose_codegen.py`'s multi-host-launch round splitting literally.

### Important terminology

`launch_parallel` calls are device-side phase dispatches.

They are **not** separate host-level `Kernel.launch(...)` calls.

---

## Proving "one host `.launch()`"

Use two independent structural checks.

### 1. Host side

Verify exactly one launch of the generated persistent FFT kernel class.

Do **not** count all generic `.launch(` occurrences.

For example, if the emitted class is named `PersistentFFT`:

```text
count("PersistentFFT.launch(") == 1
```

If the actual generated class name differs, use that exact class name or another equally precise match.

Unrelated helper launches must not affect this test.

### 2. Device side

Verify that `device_main` contains all phases for all rounds in the expected order:

```text
preload_r0
stage_*_r0
writeback_r0

preload_r1
stage_*_r1
writeback_r1
...
```

The final report must show both:

* the single host launch,
* the generated `device_main`.

One is not evidence for the other.

---

## Preload / writeback phases

Preload and writeback are separate `launch_parallel` phases.

All `workers_per_group` workers participate.

The vector/scalar role split applies only to FFT arithmetic.

Every worker uses the same:

```text
logical_block
block_base
```

and operates on a disjoint portion of that logical FFT.

Reuse the existing tail-safe chunk load/store mechanisms where practical.

Preload:

```text
DRAM natural-order input
    ->
buf_a natural order
```

Writeback:

```text
final scratchpad bank
    ->
DRAM natural-order output
```

Neither phase should add FFT permutations.

---

## Stage work partition

Support two modes:

```text
scalar_worker_mode="adaptive"
scalar_worker_mode="reserved"
```

### Adaptive mode

For each stage:

```python
if any(batch.valid_lanes < simd_lanes for batch in stage.batches):
    vector_workers = workers_per_group - 1
    scalar_worker_id = workers_per_group - 1
else:
    vector_workers = workers_per_group
    scalar_worker_id = None
```

Exact role semantics:

#### Adaptive + no tail

```text
vector workers:
    workers 0..7

scalar worker:
    none
```

Some vector workers may legitimately receive no batch.

#### Adaptive + tail

```text
vector workers:
    workers 0..6

scalar worker:
    worker 7
```

The scalar worker owns every partial batch.

### Reserved mode

Always:

```text
vector workers:
    workers 0..6

scalar-reserved worker:
    worker 7
```

For a no-tail stage, worker 7 may simply have no arithmetic.

---

## Persistent batch partition

Implement:

```text
_partition_vector_scalar(...)
```

Full batches:

```text
valid_lanes == simd_lanes
```

are assigned round-robin to vector workers.

Partial batches:

```text
valid_lanes < simd_lanes
```

are assigned only to the scalar worker.

Add persistent-specific optional `FFTStagePlan` fields:

```text
persistent_vector_batches
persistent_scalar_batches
```

Do not overload the existing cooperative `worker_batches` contract.

---

## Scalar-tail arithmetic

The scalar worker must execute genuine scalar arithmetic.

Use the existing radix-specific butterfly generator with:

```text
compute_lanes = 1
```

Do not implement:

* scalar addressing with full-width SIMD arithmetic,
* zero-padded SIMD as a substitute,
* generic DFT fallback.

Use the existing radix-specific butterfly code.

---

## Register-pressure discipline

Reuse the existing mechanisms:

```text
narrow_middle_stages
loop_stages
_ALWAYS_NARROW_RADICES
_RISKY_RADIX_PAIRS
_RISKY_AS_MIDDLE_RADICES
planning/spill_probe.py
```

A confirmed spill or unexpected frame is a hard failure.

Do not simply report it as a performance caveat.

---

## Synchronization correctness

Correctness must use the simplest available argument.

### Within one phase

Workers operate on disjoint work:

* disjoint FFT batches,
* or disjoint preload/writeback ranges.

There is no same-phase cross-worker producer/consumer dependency.

### Between phases

The previous `launch_parallel` completes before the next begins.

### Between FFT stages

Ping-pong buffers ensure:

```text
previous stage writes bank B
        ↓ phase completion
next stage reads bank B
```

Therefore correctness depends on:

```text
launch_parallel completion ordering
```

and does not depend on:

* `local_uthread_id()` density,
* simultaneous worker residency,
* immediate same-phase store visibility,
* an undocumented barrier.

Explicitly state in the final report:

> The persistent FFT does not rely on intra-phase cross-worker scratchpad visibility.

---

## New vs. reused files

### New

#### `planning/fft_plan_persistent.py`

Contains:

```text
make_persistent_leaf_plan(...)
_lower_persistent_stages(...)
_partition_vector_scalar(...)
buffer naming helpers
persistent mapping helpers if actually needed
```

For unsupported:

```text
stripes_per_group != 1
```

raise `NotImplementedError`.

#### `codegen/fft_persistent_codegen.py`

Contains:

```text
phase-prelude emission
preload/writeback emission
per-stage vector/scalar dispatch
round-unrolled device_main
256B-aligned launch-pool host allocation
generate_persistent_fft_kernel(...)
```

#### `verification/verify_fft_persistent.py`

Runs the actual generated kernel text using the existing verification harness.

Do not reimplement FFT mathematics in the verifier.

### Additive edit only

#### `planning/fft_plan_core.py`

Add:

```text
PersistentWorkgroupPlan
FFTCodegenPlan.persistent
FFTStagePlan.persistent_vector_batches
FFTStagePlan.persistent_scalar_batches
force_scratchpad=False on _make_load
force_scratchpad=False on _make_store
```

Existing callers must preserve their exact previous behavior.

### Do not modify behavior of

```text
make_fft_kernel.py
fft_plan_recursive.py
fft_plan_search.py
fft_cost_model.py
_lower_stages()
third_party/m2ndp-detour/
```

Existing non-cooperative, cooperative, recursive-transpose, and search paths must continue passing their current tests.

---

## Testing

### Import test

Verify:

```text
fft_plan_core
fft_plan_persistent
```

can be imported without circular-import failure.

### Existing-path regression

After adding `force_scratchpad=False`, run the entire existing suite unchanged.

### Mathematical plan-equivalence test

For identical:

```text
length
radices
inverse
```

compare normal lowering against persistent lowering on mathematical properties.

Require equality of:

```text
radix
stage position
load base_offset
store base_offset
valid_lanes
twiddle metadata
twiddle exponent/modulus/divisor semantics
output stride
Stockham permutation
final output order
scale
all other mathematical stage-layout metadata
```

Do not require literal equality of:

```text
source
destination
buffer_name
```

because physical storage differs intentionally.

Normalize the producer/consumer buffer graph rather than comparing bank names.

Required invariant:

```text
FFT mathematics:
    identical

physical storage:
    intentionally different
```

### Inverse-scale structural test

For `inverse=True`:

```text
persistent final-stage scale
    ==
normal final-stage scale
```

Also verify the numerical inverse FFT result.

### Adaptive worker-role test

Check:

```text
adaptive + no tail:
    all workers vector-capable
    no scalar role

adaptive + tail:
    last worker scalar
    others vector

reserved:
    last worker always scalar-reserved
```

### Target-invariant rejection tests

Construct incompatible profiles and verify rejection for:

```text
workers_per_group != interleave_chunk_uthreads

mapping_stride % uthread_bytes != 0

software_group_count != num_ndp_units

stripes_per_group != 1
```

### No-tail numerical case

Verify:

```text
persistent_scalar_batches == ()
vector_worker_count == workers_per_group
all full batches assigned exactly once
no duplicate
no dropped batch
empty worker lists allowed
```

and compare result against NumPy FFT.

### Tail cases

Test:

```text
tail = 1
tail = simd_lanes - 1
```

Partial batches must belong only to the scalar worker.

### 3+ stage leaf

Verify:

```text
preload
stage 0
stage 1
stage 2+
writeback
```

end-to-end.

### Multi-round test

For:

```text
num_logical_blocks = 40
```

verify:

```text
round 0:
    blocks 0..31

round 1:
    blocks 32..39
```

Explicitly check:

```text
block 0
block 31
block 32
block 39
```

Groups 8..31 in round 1 must not touch logical-block DRAM data.

### Scratchpad reuse

Give the same software group very different round-0 and round-1 inputs.

Verify no stale scratchpad leakage.

### Forward / inverse

Run both.

### ID-mapping diagnostic

Kernel records:

```text
global_uthread_id
local_uthread_id
worker_id
software_group_id
group_id()
```

The kernel records observations only.

Host/test independently computes:

```text
expected_worker_id
    = gid % workers_per_group

expected_software_group_id
    = gid // workers_per_group

expected_unit
    = ((pool_base // mapping_stride)
       + software_group_id)
      % num_ndp_units
```

Then compare:

```text
recorded group_id()
    ==
expected_unit
```

### Spill/frame

Run:

```text
planning.spill_probe.probe_spill_free
```

against the actual generated persistent kernel.

Any confirmed spill is a failure.

### Launch-structure tests

Verify host-side:

```text
exact generated persistent kernel launch count == 1
```

using the actual generated kernel class name.

Do **not** count generic `.launch(` strings.

Verify device-side:

```text
all round phases appear in device_main
in the exact intended order
```

Also verify:

```text
launch_uthreads
    == software_group_count * workers_per_group
```

---

## Real M2NDP-Detour validation

Before performance claims, perform correctness-first runs on the real execution environment.

Minimum cases:

1. small no-tail
2. scalar-tail
3. 3+ stage
4. 40 logical blocks
5. forward
6. inverse

Do not proceed to performance conclusions if any of these fail:

```text
mathematical plan-equivalence
inverse-scale equivalence
adaptive worker-role tests
target mapping invariants
launch-structure checks
numerical correctness
```

---

## Performance comparison

After correctness and spill-free validation, compare:

```text
existing cooperative leaf
persistent adaptive-scalar
persistent reserved-scalar
```

If practical, also compare against the existing masked-SIMD tail handling.

Measure:

```text
ndp_cycles
generated source line count
compile time
spill/frame status
scratchpad bytes / unit
```

Across:

```text
num_logical_blocks =
    1
    8
    32
    40
    64
```

If practical, also generate:

```text
256
1024
```

blocks to characterize Python-unrolled code-size and compile-time scaling.

Report actual results.

Do not assume persistent, adaptive scalar, or reserved scalar is faster.

---

## Final report

The final report must include:

1. files changed,
2. live-source hardware/runtime facts with file/line references,
3. any original assumptions found wrong,
4. `software_group_count`,
5. `workers_per_group`,
6. `launch_uthreads`,
7. `launch_bytes`,
8. `PooledRange` element count,
9. exact `gid -> worker_id -> software_group_id -> logical_block` mapping,
10. expected vs observed physical-unit diagnostic,
11. pool-alignment implementation,
12. scratchpad byte-capacity upper bound,
13. actual largest planner-supported persistent leaf,
14. preload/stage/writeback buffer-flow table,
15. normal-vs-persistent mathematical plan-equivalence result,
16. explicit inverse-scale preservation result,
17. exact host persistent-kernel launch-count check,
18. full generated `device_main`,
19. numerical verification results,
20. 40-block round-mapping verification,
21. spill/frame results,
22. cooperative vs persistent performance,
23. adaptive vs reserved scalar performance,
24. generated-code-size / compile-time scaling,
25. remaining limitations.

The final report must separately show:

```text
HOST:
    exactly one persistent FFT Kernel.launch(...)

DEVICE:
    all round-specific launch_parallel phases
    inside device_main
```

Do not use one as proof of the other.

---

## Implementation order

The architecture is accepted and implementation may begin immediately.

Proceed in this order:

```text
quick live-source consistency check
    ->
module/data-structure ownership
    ->
force_scratchpad implementation
    ->
existing-suite regression
    ->
mathematical plan-equivalence test
    ->
persistent planner
    ->
persistent codegen
    ->
exact host-launch-count test
    ->
device_main round-order test
    ->
Python numerical verification
    ->
independent ID-mapping diagnostic
    ->
spill/frame verification
    ->
real M2NDP correctness
    ->
performance measurement
    ->
final report
```

Do not restart architectural design work unless the live source directly contradicts a required assumption.

Otherwise, **proceed directly with implementation and verification now.**
