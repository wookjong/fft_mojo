# M2NDP uthread -> physical NDP unit mapping: source-traced audit

Phase 1 of the 2026-09-08 baseline fidelity/mapping audit. Every claim below is traced
directly from the real M2NDP-Detour simulator source (`third_party/m2ndp-detour/src/`)
and the Mojo-side runtime primitives (`src/m2ndp.mojo`), not inferred from this
project's own FFT-planner comments (which is exactly what the audit was asked not to
do, and what an earlier pass through those comments alone would have gotten only
approximately right).

## 1. The real address decoder

`third_party/m2ndp-detour/src/m2ndp_config.h`:

```cpp
int get_matched_unit_id(uint64_t origin_addr) const {
  return origin_addr / m_stride_size % m_num_ndp_units;
}
...
int m_stride_size = 256;   // compiled-in default
```

`m_stride_size` IS parser-overridable (`third_party/m2ndp-detour/src/m2ndp_parser.cc`:
`else if (name == "ndp_stride") config->m_stride_size = atoi(value.c_str());`), but this
project's own checked-in `third_party/m2ndp-detour/config/performance/M2NDP/m2ndp.config`
does not set an `ndp_stride=` key at all -- it only sets `m2ndp_interleave_size=256` and
`channel_interleave_size=256`, both of which are SEPARATE config fields
(`m_m2ndp_interleave_size`, `m_channel_interleave_size`) from `m_stride_size`. So the
value actually used for `get_matched_unit_id` on this target is the simulator's own
compiled-in default of 256 bytes, which the project's own config happens never to
override.

`UTHREAD_SPAWN_UNIT` (`common_defs.h`) = 32 bytes = `PACKET_SIZE` = `MEM_ACCESS_SIZE`,
one microthread's address footprint. So:

```
interleave_chunk_uthreads = m_stride_size / UTHREAD_SPAWN_UNIT = 256 / 32 = 8
```

**8 consecutive uthreads route to the same physical NDP unit before the decoder rotates
to the next unit** -- this is a real, evidenced fact about the specific hardware
configuration this project's `target_profile.DEFAULT_TARGET_PROFILE` models (256B
interleave granularity appears independently in three places in the same config file:
the simulator's own `m_stride_size` default, `m2ndp_interleave_size`, and
`channel_interleave_size` -- consistent with a deliberate design point, not an
overlooked default), but it is NOT an immutable, un-overridable fact about "M2NDP" in
the abstract: the simulator's own config parser supports a different `ndp_stride`, and
a differently-configured M2NDP target (same ISA, same simulator) could have a larger
interleave chunk with zero change to the FFT planner or codegen.

## 2. How a global id becomes (physical unit, local_uthread_id())

`uthread_generator.cc::generate_requests`: each physical unit (`m_ndp_id`) walks the
**entire** launch address range `[base, base+size)` in `UTHREAD_SPAWN_UNIT` steps and
keeps only the addresses matching its own id (`check_addr_match`), assigning them a
dense, unit-local counter `ndp_req_idx` in increasing address order:

```cpp
for (addr_t addr = base; addr < base + size; addr += UTHREAD_SPAWN_UNIT) {
  if (!check_addr_match(addr)) continue;
  RequestInfo* req = new RequestInfo{ ..., .ndp_req_id = ndp_req_idx++, .own_unit = m_ndp_id, ... };
}
```

`register_unit.cc::seed_launch_inputs` is where these become the values a running
kernel actually reads:

```cpp
seed_m2ndp_inputs(key, /*local_id*/ req->ndp_req_id,
                       /*global_id*/ req->offset / PACKET_SIZE,
                       /*group_id*/ (uint64_t)req->own_unit);
```

So (assuming `base` is aligned to a `256`-byte boundary, which this project's own
cooperative codegen already arranges via its documented pool-rounding fix):

```
global_uthread_id(g)      = g                              (dense, 0..N-1 over the whole launch)
physical_unit(g)          = floor(g / 8) % num_ndp_units
local_uthread_id(g)       = floor(g / (8*num_ndp_units)) * 8 + (g % 8)
group_id()                = physical_unit(g), directly (own_unit, not reconstructed)
```

`local_uthread_id()`/`global_uthread_id()` (`src/m2ndp.mojo`) are themselves
`external_call["__m2ndp_local_uthread_id", ...]`/`...global...` stubs awaiting a real
LLVM intrinsic backend (`docs/INTERFACE.md`) -- the M2NDP-Detour simulator's own
register-seeding above is the actual, authoritative definition of what they return
during simulation today.

## 3. Concrete examples (num_ndp_units=32, base aligned)

| g | physical_unit(g) | local_uthread_id(g) |
|---|---|---|
| 0..7 | 0 | 0..7 |
| 8..15 | 1 | 0..7 |
| 16..23 | 2 | 0..7 |
| ... | ... | ... |
| 248..255 | 31 | 0..7 |
| 256..263 | **0** (wraps) | **8..15** |
| 512..519 | 0 | 16..23 |

**8 consecutive uthreads (g=0..7)**: all unit 0, local ids 0..7 -- a complete
cooperative group of `workers_per_fft=8` fits entirely on one physical unit, in one
"wave". This is exactly what `workers_per_fft <= interleave_chunk_uthreads` (today's
implemented case) already exploits.

**16 consecutive uthreads (g=0..15)**: g=0..7 -> unit 0 (local 0..7); g=8..15 -> unit 1
(local 0..7). A naive `workers_per_fft=16` cooperative group, addressed contiguously,
is split 8+8 across **two different physical units with two different, non-
communicating scratchpad instances** -- this is the real, proven mechanism behind
every clFFT/rocFFT/VkFFT `UNSUPPORTED_*_MAPPING` result for a workgroup wider than 8 in
this repository's port.

**64 consecutive uthreads (g=0..63)**: spans chunks 0..7, i.e. **8 different physical
units** (0 through 7), each owning one 8-wide slice with local ids 0..7.

**256 consecutive uthreads (g=0..255)**: spans chunks 0..31, i.e. **all 32 physical
units** on this target (chunk index 0..31 mod 32 = itself), each owning exactly one
8-wide slice.

## 4. Is a wider cooperative group architecturally possible at all?

Yes, for an exact multiple of `interleave_chunk_uthreads`, via a mechanism this
repository does not currently implement. The decoder's chunk-to-unit assignment is
periodic with period `interleave_chunk_uthreads * num_ndp_units` (256 uthreads on this
target): global id `g` and `g + 256` always land on the **same** physical unit, with
`local_uthread_id()` values exactly 8 apart (see the g=0..7 vs g=256..263 row above).

This project's own execution model already guarantees a **global** barrier between
stages (`launch_parallel[Self.stage_N]()`: every uthread of the whole launch finishes
stage N before stage N+1 starts -- confirmed by this repository's own existing
numeric-verification harness, `verification/verify_fft_harness.run_kernel`'s own
documented semantics, not a new assumption introduced by this audit). So workers 0..7
(the first "wave" on unit 0) and workers 8..15 (the SAME unit's second wave, 256 global
ids later) genuinely CAN cooperate correctly through unit 0's own persistent
scratchpad: both waves finish stage N, in whatever order the simulator schedules them,
before the barrier releases stage N+1, at which point either wave can safely read what
the other wrote.

**What this would require, that does not exist today:** a DRAM buffer layout where one
logical FFT slot's `workers_per_fft` (say 16) elements are NOT stored contiguously
(global ids 0..15), but STRIDED across the periodic-chunk boundary (global ids
`{0..7} ∪ {256..263}`) -- a fundamentally different `AddressMapping` kind from any of
CONTIGUOUS/STRIDED/SPLIT/PEELED/CROSSED already in `planning/fft_plan_core.py`, plus a
host-side scatter/gather (or a differently-shaped launch) to actually produce that
layout, plus new cooperative-codegen logic to consume it. None of this exists in this
repository's current planner or codegen.

## 5. Answering Phase 1's A/B/C question

**Neither a clean A nor a clean B nor a clean C -- the honest answer is a proven,
three-way split, now encoded directly in `BaselineStatus`:**

- `workers_per_fft` divides `interleave_chunk_uthreads` (i.e. `<= 8` and a divisor):
  **OK**, exactly what `fft_plan_cooperative.py`/`fft_cooperative_codegen.py` already
  implement correctly. Not a bug in those modules (ruling out a clean "B").
- `workers_per_fft` is an exact multiple of `interleave_chunk_uthreads` (16, 32, 64,
  ...): **architecturally reachable** (proven above via the periodic-chunk argument),
  but **no current codegen implements the striped layout needed** ->
  `UNSUPPORTED_CURRENT_CODEGEN`. This is real support for option **C** ("the current
  mapping is too literal") in a precise, bounded sense: the CURRENT convention
  (contiguous global ids = one cooperative group) is a choice, not the only one the
  hardware allows, but exploiting the alternative needs new implementation, not merely
  a different existing knob.
- `workers_per_fft` is neither a divisor nor a multiple (15, 5, 12, ...):
  **genuinely impossible** on this target's own periodic chunking, for any DRAM layout
  -- this is the real, evidenced core of option **A**.

`gpu_baseline/common.py::map_cooperative_kernel` implements exactly this three-way
test. Re-running the six-length comparison table after this reclassification moves
every clFFT length from 64 to 4096 (workers_per_fft always a clean multiple of 8, since
clFFT's own workgroup sizes and transform counts are both powers of two times small
factors) from the old blanket `UNSUPPORTED_MAPPING` to the more precise
`UNSUPPORTED_CURRENT_CODEGEN` -- a materially more optimistic, and more accurate,
finding: these are not proven-impossible on this hardware, they are gaps in what this
repository's planner/codegen has built so far.

## 6. What this audit does NOT do

Per the task's own Phase 7 ("do not optimize"): this audit does not implement the
striped multi-wave cooperative layout described in section 4. Doing so would be a
genuine, real M2NDP-specific codegen extension -- valuable future work, but out of
scope for a baseline-fidelity pass, and explicitly the kind of "close the gap this
audit merely reclassified" work that belongs in a clearly separate, later change once
this baseline itself is considered frozen.
