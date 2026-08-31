# Cooperative workers_per_fft=8: root cause and fix (2026-08-31)

## Symptom

`workers_per_fft == interleave_chunk_uthreads` (8, on this target) produced
a real-hardware wrong answer, first seen 2026-08-30: N=128
radices=(4,4,4,2), `mismatch at 32`, independent of `compute_lanes`, batch
size, and `loop_stages`. The Python-level numeric harness
(`verify_fft_cooperative.verify_cooperative_leaf`, which re-executes the
*actual emitted* stage text) passed cleanly at the same configuration.
That combination -- real hardware wrong, the emitted stage logic itself
provably correct -- was (reasonably, at the time) read as "below this
codebase's own planning/codegen layer," matching this project's own
vlenb-CSR/transpose-tile pattern of confirmed simulator-level defects, and
mitigated by excluding `workers_per_fft=8` from automatic candidate
generation (`fft_plan_cooperative.py`,
`exclude_full_interleave_chunk: bool = True`, commit `9de1d8d`).

That diagnosis was wrong. This is a real, fixable planning/codegen-layer
bug -- specifically a host-side allocation-alignment bug, invisible to the
Python harness because that harness never modeled real DRAM addresses.

## Root cause

`Pool.alloc` (`src/m2ndp_host.mojo:175-192`, this project's own CXL-pool
bump allocator every `cxl_alloc` call goes through) aligns each individual
allocation to only `_POOL_ALIGN = 64` bytes. The M2NDP address decoder,
however, assigns a microthread's physical NDP unit from its *absolute*
DRAM address: `(addr // 256) % num_ndp_units`, where 256 =
`interleave_chunk_uthreads * uthread_bytes` (8 * 32 on this target) --
confirmed directly against `M2NDPConfig::get_matched_unit_id`
(`target_profile.py`'s own `interleave_chunk_uthreads` field comment).

A cooperative leaf's correctness (`fft_cooperative_codegen.
_emit_cooperative_stage`) depends on `local_uthread_id()`'s and
`global_uthread_id()`'s own `WORKERS_PER_FFT`-sized groupings partitioning
the *same* physical microthreads -- which holds only if this launch's own
pool starts exactly on a 256B interleave-chunk boundary. If it does not,
every `group_id()` (physical-unit) boundary inside the launch is silently
shifted by a few microthreads relative to where the codegen's own
`fft_slot`/`worker_id` math assumes it is.

`workers_per_fft=8` exactly fills one interleave chunk -- zero slack to
absorb any shift. Any nonzero misalignment splits that one cooperative
group across two physical units, each with its own *private* scratchpad:
half the workers write a stage's output into unit A's scratchpad, the
other half read the same `fft_slot` from unit B's scratchpad, which never
received that write. This produces exactly the observed signature: a
fixed wrong output index, independent of `compute_lanes`/batch/
`loop_stages` (none of which touch pool allocation at all).

### Empirical confirmation

An identity-probe kernel (`benchmarks/fft/id_dump.mojo`, ad hoc, not part
of the FFT planner) launched 1040 microthreads and dumped
`global_uthread_id()`/`local_uthread_id()`/`group_id()` per microthread,
directly on the real M2NDP-Detour simulator. Result: `local_uthread_id()`
does **not** reset to 0 at every interleave-chunk boundary -- it is
`round * interleave_chunk_uthreads + (global_uthread_id() %
interleave_chunk_uthreads)`, where `round = global_uthread_id() //
(interleave_chunk_uthreads * num_ndp_units)` and `group_id() =
(global_uthread_id() // interleave_chunk_uthreads) % num_ndp_units` --
confirmed for 4+ full wraparound rounds and a ragged tail.

Directly comparing two generated kernels for the *same* N=128
radices=(4,4,4,2) workers_per_fft=8 case -- one built via the bare
standalone-leaf renderer (`generate_cooperative_fft_kernel`, passing) and
one via the real `make_fft_kernel.py` -> `generate_recursive_fft_kernels`
pipeline (failing, `mismatch at 32`) -- showed the generated *kernel
struct bodies were byte-for-byte identical* (only cosmetic naming
differed). The only structural difference was in host-side `main()`: the
pipeline path allocates several extra buffers before its own launch pool.
An instrumented build that printed `Int(pool) % 256` confirmed it directly:
`mod 256 = 0` for the passing case, `mod 256 = 128` for the failing one.

### Why workers_per_fft in {1,2,4} looked safe before

`Pool.alloc`'s 64-byte (2-uthread) granularity only ever produces an
*even* misalignment. `workers_per_fft=2` tolerates any even misalignment
unconditionally; `workers_per_fft=4` tolerates only some of them (offsets
of 0 or 4 uthreads, not 2 or 6). Every `workers_per_fft=4` case actually
tested before this fix happened to land on a safe offset -- this was a
latent risk for `workers_per_fft=4` too, not a `workers_per_fft=8`-only
bug, simply never hit by the specific buffer layouts exercised so far.
`workers_per_fft=8` has *no* safe nonzero offset, so it failed reliably.

## Fix

Mirrors a fix that already existed for a different leaf-execution
strategy: `fft_persistent_codegen.py`'s host `main()` emission already
over-allocates its own launch pool and rounds the address up to the next
256B boundary by hand (see `docs/persistent_leaf_design.md`'s "uthread
pool alignment" section) -- this exact class of bug had already forced
that fix once, for persistent leaves, and was simply never connected to
cooperative ones.

Applied the same pattern to the two cooperative-leaf-emitting paths:

- `codegen/fft_cooperative_codegen.py`'s `generate_cooperative_fft_kernel`
  (the standalone single-leaf renderer).
- `codegen/fft_transpose_codegen.py`'s `generate_recursive_fft_kernels`
  (the production default pipeline `make_fft_kernel.py` actually uses),
  gated per stage on `stage.cooperation is not None` -- non-cooperative
  stages are unaffected (they have no cross-unit grouping requirement to
  protect, and skipping them keeps this fix's blast radius scoped to the
  one execution strategy that actually needs it, rather than touching
  `Pool.alloc` itself, which every non-FFT benchmark in this repo also
  uses).

Both now: allocate `pool_elems + chunk_bytes/4` extra `Float32`s, compute
`(raw_addr + chunk_bytes - 1) // chunk_bytes * chunk_bytes`, assert the
result really is a multiple of `chunk_bytes` (fail loudly, not silently,
on the assertion), and build the launch `PooledRange` from that aligned
pointer instead of the raw `cxl_alloc` result.

`fft_plan_cooperative.py`'s `exclude_full_interleave_chunk` now defaults
to `False` -- `workers_per_fft=8` is offered again by
`worker_candidates_per_fft`/`choose_workers_per_fft`'s automatic
selection. The flag itself was kept (not removed) as an explicit opt-out
for comparison, but there is no longer a known-good reason to set it.

## Verification

- `verification/verify_fft_plan.py` (full existing regression suite):
  passes unchanged.
- Real M2NDP-Detour hardware, `benchmarks/fft/run_fft_test.sh`:
  - N=64/128/256, `--cooperative-workers 8` (forced): all PASS (spill
    warnings present at some, as before -- spill-free-ness is a separate,
    pre-existing, orthogonal question; not this fix's job).
  - N=30/105/216, `--cooperative-workers auto` (now free to reach 8):
    all still PASS.
  - The exact original repro (N=128 radices=(4,4,4,2), default and
    `--compute-lanes 2`/`1`, through the real unmodified
    `make_fft_kernel.py --cooperative-workers 8` CLI): PASS at every
    `compute_lanes` tried, where it previously failed at all of them.

## What this does *not* change

- No `third_party/m2ndp-detour` (simulator) source was touched.
- The separate, pre-existing N=1024 register-pressure/function-size
  cooperative bug (`fft_cooperative_codegen._emit_cooperative_stage`'s own
  docstring, fixed 2026-08-27 by making `loop_stages` apply per worker) is
  unrelated to this one and remains fixed as it was.
- Spilling at some `compute_lanes` for `workers_per_fft=8` is unrelated,
  pre-existing, and still governed by the project's existing spill-probe
  policy (a spilling candidate is still simulator-unsafe and must not be
  trusted or measured as a final answer regardless of this fix).
