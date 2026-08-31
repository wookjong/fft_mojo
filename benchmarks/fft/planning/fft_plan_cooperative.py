from __future__ import annotations

"""Cooperative-worker leaf FFTs: N microthreads share one sub-FFT instead of
one microthread owning it outright -- see `CooperationPlan` (fft_plan_core.py)
for the field-level contract this module fills in.

Why this is a thin wrapper, not a new lowering path: `_lower_stages`'s own
`_make_load`/`_make_twiddle`/`_make_store` already compute every load/twiddle/
store offset as an absolute position *within this one sub-FFT* (see
fft_plan_core.AddressMapping's own docstring: "elem*elem_stride ... resolves
to a plan-time constant per load/store") -- nothing in a `SIMDBatchPlan` is
uthread-relative. So "N workers share one sub-FFT" needs no new address
math at all: it only needs (1) a partition of each stage's already-resolved
batches across workers, (2) scratchpad sized per-FFT-slot instead of
per-uthread (already what `_build_plan`'s own `max_uthread` capacity math
computes, once "uthread" is read as "FFT slot" -- see `make_cooperative_leaf_plan`),
and (3) a DRAM address for the *slot*, not the physical microthread.

That third point is the one genuinely new decision, and it interacts with a
documented project constraint (docs/STATUS.md, "IDs are primitives, not
derived": `global_uthread_id()` must never be *reconstructed* from
`group_id()`/`local_uthread_id()`, since the real hardware address-interleave
stride that relates them is a runtime config the compiler has no closed form
for -- see docs/SIMULATION.md's "divided by the interleave stride, modulo the
unit count"). A cooperative leaf's own DRAM mapping is *always*
`AddressMapping.contiguous` (never PEELED/CROSSED/STRIDED -- those belong to
a wrapping multi-kernel/balanced/recursive plan, not a leaf), and
fft_codegen.py addresses it with `logical_fft_id = global_uthread_id() //
WORKERS_PER_FFT` -- reading the one primitive the hardware already hands out
dense over the *entire* launch, whatever the interleaving, not reconstructing
it from anything. (An earlier version of this built `logical_fft_id` from
`group_id() + num_groups() * fft_slot` instead -- concretely wrong, traced
against this project's own real config: see
fft_codegen._emit_cooperative_stage's own docstring for the failure and why
`global_uthread_id()` alone avoids it.) `worker_id`/`fft_slot`, by contrast,
*do* come from `local_uthread_id()` -- safe for a different reason: that
primitive *is*, by definition, "this microthread's index on its own physical
unit," so two microthreads sharing an `fft_slot` share a unit (and its
scratchpad) unconditionally, no interleaving knowledge required.
"""

import math
from dataclasses import replace

from planning.fft_plan_core import (
    _DEFAULT,
    CooperationPlan,
    FFTCodegenPlan,
    SIMDBatchPlan,
    _build_plan,
    _Default,
    layouts_for_radices,
    pingpong_needed,
)
from planning.target_profile import DEFAULT_TARGET_PROFILE


def worker_candidates_per_fft(
    length: int,
    radices: tuple[int, ...],
    *,
    simd_lanes: int = 8,
    max_workers: int | None = None,
    interleave_chunk_uthreads: int = DEFAULT_TARGET_PROFILE.interleave_chunk_uthreads,
    exclude_full_interleave_chunk: bool = False,
) -> list[int]:
    """Every legal cooperative worker count for a leaf of this shape,
    ascending -- every divisor of the hardware's own interleave chunk
    (`interleave_chunk_uthreads`) that does not exceed this leaf's own
    *busiest* stage's SIMD batch count (`max(ceil((length // radix) /
    simd_lanes) for radix in radices)`): a worker beyond that count is
    never active on *any* stage of this leaf -- pure launch overhead for
    zero benefit. A worker beyond a *thinner* stage's own (smaller) batch
    count but still within the busiest stage's is fine: it idles on the
    thin stage and works on the busy one (see CooperationPlan /
    `_partition_batches`'s own possibly-uneven, possibly-empty per-worker
    split) -- only the bound above describes a worker with literally
    nothing to do anywhere.

    `choose_workers_per_fft` (below) is the single-answer caller every
    existing leaf builder still uses (`candidates[-1]`, today's exact
    "largest that's ever useful" heuristic, unchanged); `planning/
    fft_plan_search.py`'s generate_worker_candidates is the multi-answer
    caller that wants the whole list instead -- both go through this one
    function, same discipline `_recursive_split_candidates` established
    for split candidates.

    `max_workers`: an optional caller-side cap (e.g. a fixed
    `cooperative_workers` a caller passed through
    `fft_plan_recursive.make_recursive_transpose_plan`) -- `None` (the
    default) applies none beyond the reasoning above.

    `exclude_full_interleave_chunk`: `False` (the default, since the real
    root cause below was fixed) -- history for why this flag exists at
    all: `workers_per_fft == interleave_chunk_uthreads` (8, on this
    project's own target) was confirmed real-hardware wrong 2026-08-30
    (N=128 radices=(4,4,4,2), mismatch at a fixed output index,
    independent of compute_lanes/batch size/loop_stages, Python-level
    numeric harness passing cleanly throughout) and defaulted to excluded
    while that looked like a below-the-planning-layer defect matching this
    project's vlenb-CSR/transpose-tile pattern. It was not: root-caused
    2026-08-31 to `Pool.alloc` (`src/m2ndp_host.mojo`) only aligning
    individual `cxl_alloc` allocations to 64B, not to this target's own
    256B hardware interleave chunk (`interleave_chunk_uthreads *
    uthread_bytes`) -- since the M2NDP address decoder places a
    microthread's physical unit from its *absolute* DRAM address (not an
    index relative to the launch), an unaligned launch pool silently
    shifts every `group_id()` boundary by a few microthreads. A
    `workers_per_fft` that exactly fills the interleave chunk (8 here) has
    zero slack to absorb that shift -- any nonzero misalignment splits the
    group across two physical units, each with its own private scratchpad,
    so half the cooperating workers write stage output the other half's
    reads never see. `workers_per_fft` values that are proper divisors of
    the chunk (1/2/4) merely *happened* to survive every case actually
    tested before this fix, for the same reason: the bump allocator's own
    64B/2-uthread granularity only ever produces an even misalignment, and
    a small enough worker count can tolerate some (never all) of those --
    this was a latent risk for them too, not a workers=8-only bug.
    Real fix (`fft_transpose_codegen.generate_recursive_fft_kernels` and
    `fft_cooperative_codegen.generate_cooperative_fft_kernel`, mirroring
    `fft_persistent_codegen.py`'s own pre-existing identical fix -- see
    docs/persistent_leaf_design.md's "uthread pool alignment" section,
    which this exact class of bug had already forced once, just never
    connected to cooperative leaves): over-allocate the launch pool and
    round its address up to the next 256B boundary by hand at host-main
    time. Confirmed real-hardware correct at workers_per_fft=8 for the
    same N=128 case, every compute_lanes tried (still spills at some --
    a caller wanting a confirmed spill-free plan still needs
    `--verify-spill-free`/`probe_spill_free`, an orthogonal question this
    flag never touched). Pass `True` only to reproduce the old excluded
    behavior for comparison -- there is no known-good reason to exclude
    this worker count from candidate generation any more.
    """
    if not radices:
        raise ValueError("radices must be non-empty")
    max_batches = max(-(-(length // radix) // simd_lanes) for radix in radices)
    cap = max_batches if max_workers is None else min(max_batches, max_workers)
    divisors = [
        d for d in range(1, interleave_chunk_uthreads + 1)
        if interleave_chunk_uthreads % d == 0
        and not (exclude_full_interleave_chunk and d == interleave_chunk_uthreads)
    ]
    return [d for d in divisors if d <= cap]


def choose_workers_per_fft(
    length: int,
    radices: tuple[int, ...],
    *,
    simd_lanes: int = 8,
    max_workers: int | None = None,
    interleave_chunk_uthreads: int = DEFAULT_TARGET_PROFILE.interleave_chunk_uthreads,
    exclude_full_interleave_chunk: bool = False,
) -> int:
    """A leaf's own useful cooperative worker count -- the *largest* legal
    candidate from `worker_candidates_per_fft` (see its own docstring for
    the legality rule, and for `exclude_full_interleave_chunk`'s own
    real-hardware-confirmed-wrong-answer rationale). A heuristic, not a
    cost model: it says how many workers can ever be useful for this
    leaf, not how many are *optimal* -- optimality needs real cycle
    measurements across leaf shapes, deliberately left as future work
    (see fft_plan_recursive._choose_recursive_split's own docstring for
    the same "candidate generation stays swappable, a real cost model
    comes later" discipline this follows; `planning/fft_plan_search.py`
    is where that search now lives).
    """
    candidates = worker_candidates_per_fft(
        length, radices, simd_lanes=simd_lanes, max_workers=max_workers,
        interleave_chunk_uthreads=interleave_chunk_uthreads,
        exclude_full_interleave_chunk=exclude_full_interleave_chunk,
    )
    return candidates[-1]


def default_cooperative_scratchpad_budget(n: int) -> int:
    """A candidate leaf-size `scratchpad_byte_budget` (see
    fft_plan_recursive._choose_recursive_split) to *try* when
    `cooperative_workers` is enabled -- `16 * ceil(sqrt(n))`, i.e. cap the
    leaf at roughly `sqrt(n)` elements, the balanced (four-step-FFT-style)
    split point.

    Read this as a documented starting guess to override with an explicit
    `scratchpad_byte_budget`, not something safe to default to blindly:
    measured end-to-end on the real M2NDP-Detour simulator at three N
    (forward FFT, `cooperative_workers="auto"`; cap = budget // 16 is each
    leaf's own element cap, so this function's own answer is the row where
    "cap/sqrt(N)" reads "1x"):

        N=1024  (sqrt=32): budget=128  (cap=8,   0.25x): 57717
                            budget=256  (cap=16,  0.5x):  39162
                            budget=512  (cap=32,  1x, this function's answer): 34837
                            budget=1024 (cap=64,  2x): 32487
                            budget=2048 (cap=128, 4x): 33337
                            budget=4096 (cap=256, 8x, = flat 4096 default): 27736 (lowest)
                            budget=8192 (cap=512, no more splitting help): 47494
                            budget=16384 (cap=1024, no split at all): FAILS -- register
                                spill (1888-byte frame in stage_1), confirmed by the same
                                kernel body passing cleanly through the Python numeric
                                harness (verify_fft_cooperative.verify_cooperative_leaf) --
                                a real-toolchain-only failure, not a logic bug in this module.

        N=4096  (sqrt=64): budget=256  (cap=16,  0.25x): 54529
                            budget=512  (cap=32,  0.5x):  45058
                            budget=1024 (cap=64,  1x, this function's answer): 36489 (lowest)
                            budget=2048 (cap=128, 2x): 38926
                            budget=4096 (cap=256, 4x, = flat 4096 default): 42527

        N=16384 (sqrt=128): budget=512  (cap=32,  0.25x): 144672
                             budget=1024 (cap=64,  0.5x): 120630 (lowest)
                             budget=2048 (cap=128, 1x, this function's answer): 129223
                             budget=4096 (cap=256, 2x, = flat 4096 default): 141699
                             budget=8192 (cap=512, 4x): 210087
                             budget=16384 (cap=1024, 8x): 350371

    No single row (`sqrt(n)`, a fixed absolute cap, or the flat `4096`
    default) wins at all three: this function's own `sqrt(n)` answer beats
    the flat `4096` default at N=4096 (14%) and N=16384 (9%), but *loses*
    to it at N=1024 (26% worse -- there the flat default happens to already
    be near that N's own true optimum, for reasons this investigation did
    not pin down). So `make_fft_kernel.make_fft_kernel` does **not** call
    this function to override its own default -- it keeps the flat `4096`
    unconditionally, cooperative or not, precisely because no formula found
    so far is safe to default to instead. This function stays exported as a
    documented, explicit-opt-in starting point for a caller tuning one
    specific N (pass its result as `scratchpad_byte_budget=` and compare
    against nearby values, the way the table above was built), not
    something either this module or make_fft_kernel.py applies on its own.
    A real answer needs a benchmark-driven search over candidate budgets
    per N, the way _choose_recursive_split's own docstring already flags
    as future work -- the mechanism behind exactly where each N's true
    optimum falls (a genuine latency/rounds/launch-count/register-pressure
    tradeoff, not a simple parallelism count or a clean closed form) isn't
    understood well enough yet to extrapolate safely.
    """
    root = math.isqrt(n)
    if root * root != n:
        root += 1
    return 16 * root


def _partition_batches(
    batches: tuple[SIMDBatchPlan, ...], workers_per_fft: int
) -> tuple[tuple[SIMDBatchPlan, ...], ...]:
    """Round-robin: worker `w` owns `batches[w], batches[w+workers_per_fft], ...`
    -- the simplest even split (see CooperationPlan's own module note on
    `active_workers`; a cost-weighted split, or a stage-by-stage narrower
    `active_workers` for a low-parallelism stage, is future work this
    function's caller can layer on without changing its own signature: it
    always returns exactly `workers_per_fft` buckets, some possibly empty
    when a stage has fewer batches than workers)."""
    buckets: list[list[SIMDBatchPlan]] = [[] for _ in range(workers_per_fft)]
    for i, batch in enumerate(batches):
        buckets[i % workers_per_fft].append(batch)
    return tuple(tuple(bucket) for bucket in buckets)


def make_cooperative_leaf_plan(
    length: int,
    radices: tuple[int, ...],
    *,
    workers_per_fft: int,
    total_ffts: int = 1,
    inverse: bool = False,
    simd_lanes: int = 8,
    kernel_name: str = "FFTFP32Coop",
    inverse_scale: float | None | _Default = _DEFAULT,
    spad_capacity_bytes: int | None = None,
    max_concurrent_scratchpad_bytes: int | None = None,
) -> FFTCodegenPlan:
    """A length-`length` leaf FFT (single fused kernel, `radices` its own
    Cooley-Tukey/Stockham stage sequence -- same contract as `_build_plan`/
    `layouts_for_radices`), executed by `workers_per_fft` cooperating
    microthreads per sub-FFT instead of one.

    `total_ffts`: how many independent length-`length` sub-FFTs this kernel's
    one launch covers (before multiplying by `workers_per_fft` for the
    physical launch size -- see below). `1` for a single standalone FFT;
    a recursive/multi-kernel plan's own replica count otherwise.

    `inverse_scale`: passed straight through to `_build_plan` unresolved --
    the same `_DEFAULT` sentinel (see `_build_plan`'s own docstring):
    omitted, it's `1/length` when `inverse` else `None`; pass `None`
    explicitly to suppress that (every non-final kernel in a chain wants
    this -- the overall 1/N belongs on whichever one kernel is actually
    last).

    Reuses `_build_plan` outright for every stage's load/twiddle/store
    lowering and for the scratchpad-capacity search (`_cap_max_uthread`,
    via `spad_capacity_bytes`/`max_concurrent_scratchpad_bytes`) -- called
    with `total_uthreads=total_ffts`, so what `_build_plan` computes as
    "how many uthreads fit per group" (`max_uthread`) and "bytes for one
    uthread's own ping-pong buffers" are already exactly "how many FFT
    *slots* fit per group" and "bytes for one FFT slot's own ping-pong
    buffers": nothing about that math changes when many workers, not one
    uthread, keep that slot busy. `use_pingpong=pingpong_needed(len(radices))`:
    only a 3+-stage leaf actually needs both banks -- see `pingpong_needed`'s
    own docstring for why 1-2 stages don't (this halves scratchpad for
    every leaf short enough to matter, e.g. every `near_fft`/leaf a
    recursive plan's own `scratchpad_byte_budget` was already sized around).

    After that, `base.max_uthread` (the per-group cap `_build_plan` computed,
    read here as `fft_slots_per_group`) is used to re-express
    `total_uthreads`/`max_uthread` on the *returned* plan in their other,
    equally-real meaning: the *physical* microthread counts fft_codegen.py's
    host-launch / round-split machinery already expects those two fields to
    hold (see FFTCodegenPlan's own docstring) -- `workers_per_fft` times as
    many as the FFT-slot counts `_build_plan` computed. Every other field
    `_build_plan` returned (scratchpad_buffers sized per slot already,
    input/output AddressMapping, host plan, ...) needs no change at all.
    """
    if workers_per_fft < 1:
        raise ValueError("workers_per_fft must be >= 1")

    base = _build_plan(
        length=length,
        inverse=inverse,
        total_uthreads=total_ffts,
        simd_lanes=simd_lanes,
        use_pingpong=pingpong_needed(len(radices)),
        layouts=layouts_for_radices(length, radices, simd_lanes),
        kernel_name=kernel_name,
        inverse_scale=inverse_scale,
        spad_capacity_bytes=spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=max_concurrent_scratchpad_bytes,
    )
    fft_slots_per_group = base.max_uthread

    new_stages = tuple(
        replace(
            stage,
            worker_batches=_partition_batches(stage.batches, workers_per_fft),
        )
        for stage in base.stages
    )

    # `base.host.pool_elems` (see fft_plan_core._make_host_plan) was sized by
    # `_build_plan` for `total_ffts` physical uthreads -- one per logical FFT,
    # today's assumption. This plan launches `workers_per_fft` times as many
    # physical microthreads (see `total_uthreads` below), so the pool a host
    # main's `PooledRange.over` spawns them from must grow by the same factor;
    # `total_elems` (the DRAM data itself: length * total_ffts) is unaffected.
    new_host = replace(base.host, pool_elems=base.host.pool_elems * workers_per_fft)

    return replace(
        base,
        stages=new_stages,
        total_uthreads=base.total_uthreads * workers_per_fft,
        max_uthread=fft_slots_per_group * workers_per_fft,
        host=new_host,
        cooperation=CooperationPlan(
            workers_per_fft=workers_per_fft,
            fft_slots_per_group=fft_slots_per_group,
        ),
    )
