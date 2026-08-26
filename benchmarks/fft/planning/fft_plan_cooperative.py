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


# The M2NDP address decoder hands consecutive microthreads to the same NDP
# unit in blocks of this size before rotating to the next unit (the real
# config's `m2ndp_interleave_size / packet_size` -- see
# make_fft_kernel.py's own `_SPAD_CAPACITY_BYTES` for why a constant
# duplicated from the C++ config, not read from it, is this repo's only
# option). `workers_per_fft` must divide this so that local_uthread_id()'s
# and global_uthread_id()'s own groupings of `workers_per_fft` partition the
# same physical microthreads into the same groups -- see
# fft_codegen._emit_cooperative_stage's own docstring for the concrete trace
# (against this exact config) that found this the hard way.
_INTERLEAVE_CHUNK_UTHREADS = 8


def choose_workers_per_fft(
    length: int,
    radices: tuple[int, ...],
    *,
    simd_lanes: int = 8,
    max_workers: int | None = None,
) -> int:
    """A leaf's own useful cooperative worker count.

    The largest divisor of the hardware's own interleave chunk
    (`_INTERLEAVE_CHUNK_UTHREADS`, see above) that does not exceed this
    leaf's own *busiest* stage's SIMD batch count (`max(ceil((length //
    radix) / simd_lanes) for radix in radices)`): a worker beyond that
    count is never active on *any* stage of this leaf -- pure launch
    overhead for zero benefit. A worker beyond a *thinner* stage's own
    (smaller) batch count but still within the busiest stage's is fine: it
    idles on the thin stage and works on the busy one (see
    CooperationPlan / `_partition_batches`'s own possibly-uneven,
    possibly-empty per-worker split) -- only the bound above describes a
    worker with literally nothing to do anywhere.

    `max_workers`: an optional caller-side cap (e.g. a fixed
    `cooperative_workers` a caller passed through
    `fft_plan_recursive.make_recursive_transpose_plan`) -- `None` (the
    default) applies none beyond the reasoning above.

    A heuristic, not a cost model: it says how many workers can ever be
    useful for this leaf, not how many are *optimal* -- optimality needs
    real cycle measurements across leaf shapes, deliberately left as
    future work (see fft_plan_recursive._choose_recursive_split's own
    docstring for the same "candidate generation stays swappable, a real
    cost model comes later" discipline this follows).
    """
    if not radices:
        raise ValueError("radices must be non-empty")
    max_batches = max(-(-(length // radix) // simd_lanes) for radix in radices)
    cap = max_batches if max_workers is None else min(max_batches, max_workers)
    divisors = [d for d in range(1, _INTERLEAVE_CHUNK_UTHREADS + 1) if _INTERLEAVE_CHUNK_UTHREADS % d == 0]
    return max(d for d in divisors if d <= cap)


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
