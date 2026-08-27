from __future__ import annotations

"""Decisions that shape *how* a fully-planned FFTCodegenPlan gets rendered
into Mojo source -- as distinct from fft_codegen.py, which only emits text
from whatever this module (or the plan itself) already decided. Everything
here is a pure function of an FFTCodegenPlan (or one of its stages/batches):
no FFT planning happens here either (radices, addresses, twiddle values,
ping-pong banks, ... -- all already fixed in the plan by the time anything
in this module runs), but *rendering-shape* choices that used to live
inline inside fft_codegen.py's own emit functions do:

* whether a stage's per-batch code renders as one Mojo `while` loop over the
  batch index, or fully unrolled (`try_build_loop_stage`'s period search --
  see its own docstring for why a stage may need this at all, and the
  module note further down for why the search is provably safe, never a
  source of a wrong runtime offset),
* how a hardware-width (`simd_lanes`) SIMDBatchPlan splits into narrower
  `compute_lanes`-wide chunks for emission (`chunk_batch`), and whether a
  chunk's own store, once windowed to that narrower width, is still
  contiguous enough to promote from scalar-per-lane to one vector store
  (`_chunk_store` -- the same equivalent-instruction-selection question
  fft_plan_core._make_store already asks at the full simd_lanes width, just
  re-asked at the narrower width the emitted arithmetic actually uses).
"""

from dataclasses import dataclass

from planning.fft_plan_core import (
    FFTStagePlan,
    LoadPlan,
    OutputPlan,
    SIMDBatchPlan,
    StorePlan,
    TwiddlePlan,
)


def _chunk_load(load: LoadPlan, offset: int, width: int) -> LoadPlan:
    if load.mode == "vector":
        assert load.base_offset is not None
        return LoadPlan(
            operand=load.operand,
            source=load.source,
            buffer_name=load.buffer_name,
            mode="vector",
            base_offset=load.base_offset + offset,
        )
    # `packed_lane_offsets` is exactly `simd_lanes` long, `None` marking a
    # planner-selected padding lane (see LoadPlan's own docstring; a tail
    # SIMD batch's own invalid lanes render this way -- fft_plan_core.
    # _make_load only ever promotes to "vector" at the *full* simd_lanes
    # width, so any partial batch starts here unconditionally). The exact
    # mirror of _chunk_store's own re-promotion just below: a batch that
    # can't vectorize at simd_lanes width may still have a compute_lanes-
    # wide *sub*-slice that's fully populated and contiguous (a tail
    # batch's own valid prefix, most commonly) -- same equivalent-
    # instruction-selection question _make_load already asks at the full
    # width, just re-asked at the narrower width the emitted arithmetic
    # actually uses. Confirmed a real, not just theoretical, case: N=960's
    # FFTRecNear0 stage_0 tail batch (4 valid of 8 lanes, compute_lanes=4)
    # rendered 4 separate scalar loads of contiguous offsets before this
    # fix, one vector load[width=4] after -- no change to *which* offsets
    # get read, only how many instructions read them.
    sliced = load.packed_lane_offsets[offset : offset + width]
    if len(sliced) == width and all(o is not None for o in sliced) and all(
        sliced[i] == sliced[0] + i for i in range(1, width)  # type: ignore[operator]
    ):
        return LoadPlan(
            operand=load.operand,
            source=load.source,
            buffer_name=load.buffer_name,
            mode="vector",
            base_offset=sliced[0],
        )
    return LoadPlan(
        operand=load.operand,
        source=load.source,
        buffer_name=load.buffer_name,
        mode="scalar_pack",
        packed_lane_offsets=sliced,
    )


def _chunk_twiddle(twiddle: TwiddlePlan | None, offset: int, width: int) -> TwiddlePlan | None:
    if twiddle is None:
        return None
    return TwiddlePlan(
        real=twiddle.real[offset : offset + width],
        imag=twiddle.imag[offset : offset + width],
    )


def _chunk_store(store: StorePlan, offset: int, width: int) -> StorePlan:
    if store.mode == "vector":
        assert store.base_offset is not None
        return StorePlan(
            destination=store.destination,
            buffer_name=store.buffer_name,
            mode="vector",
            base_offset=store.base_offset + offset,
        )
    # `lane_offsets` is exactly `valid_lanes` long, which may be shorter
    # than `plan.simd_lanes` (a tail batch) -- slicing past its end (a
    # chunk entirely beyond valid_lanes) yields the empty tuple, which is
    # exactly right: nothing in that chunk is ever written.
    sliced = store.lane_offsets[offset : offset + width]
    # Store vectorization, compute_lanes case: the planner already
    # promotes a full simd_lanes-wide store to "vector" whenever its own
    # offsets are contiguous (fft_plan_core._make_store) -- but a store
    # that stays "scalar_lanes" at simd_lanes width can still have a
    # contiguous compute_lanes-wide *sub*-slice (e.g. simd_lanes=8 offsets
    # [0,1,2,3,16,17,18,19]: not contiguous as a whole, but each
    # compute_lanes=4 chunk is). This is not a new layout decision --
    # `sliced` is exactly the same exact destination offsets the planner
    # already computed, just windowed to this one compute chunk -- so
    # asking whether *this* window happens to be contiguous is the same
    # equivalent-instruction-selection question _make_store itself asks
    # at the full simd_lanes width, only re-asked at compute_lanes
    # granularity because that's the width the emitted arithmetic (and
    # thus the store instruction sitting right after it) actually uses.
    # `len(sliced) == width` excludes a chunk that reaches past
    # valid_lanes (a tail) -- never promoted, so a partial chunk keeps
    # writing exactly the lanes it always did.
    if len(sliced) == width and all(
        sliced[i] == sliced[0] + i for i in range(1, width)
    ):
        return StorePlan(
            destination=store.destination,
            buffer_name=store.buffer_name,
            mode="vector",
            base_offset=sliced[0],
        )
    return StorePlan(
        destination=store.destination,
        buffer_name=store.buffer_name,
        mode="scalar_lanes",
        lane_offsets=sliced,
    )


def chunk_batch(
    batch: SIMDBatchPlan, *, simd_lanes: int, compute_lanes: int
) -> list[tuple[int, SIMDBatchPlan]]:
    """Split one hardware-width (`simd_lanes`) SIMDBatchPlan into one or
    more smaller `compute_lanes`-wide sub-batches for code emission.

    `simd_lanes` is the plan's launch granule -- it sizes `PooledRange`
    (see `src/m2ndp.mojo`'s `VECTOR_WIDTH`/`PooledRange.over`) and must
    stay exactly what the planner decided; nothing here touches offsets,
    uthread counts, or `plan.simd_lanes` itself. `compute_lanes` only
    controls how wide a vector *instruction* the kernel body emits to
    process that one already-decided batch -- e.g. one 8-wide batch becomes
    two 4-wide chunks instead of one 8-wide (LMUL=2 on this target's
    128-bit VLEN) operation. A smaller compute width avoids the RVV
    register-spill/insertelement patterns this target's simulator cannot
    run (see docs/STATUS.md); it costs nothing else since `emit_butterfly`
    itself is already width-agnostic (see fft_butterflies.py) and each
    chunk becomes its own block scope exactly like today's multi-batch case.

    Returns `[(width, chunk), ...]`; `width == simd_lanes` and a single
    element when `compute_lanes >= simd_lanes` (today's behavior, byte
    for byte).
    """
    if compute_lanes >= simd_lanes:
        return [(simd_lanes, batch)]

    chunks: list[tuple[int, SIMDBatchPlan]] = []
    n_chunks = (simd_lanes + compute_lanes - 1) // compute_lanes
    for c in range(n_chunks):
        offset = c * compute_lanes
        width = min(compute_lanes, simd_lanes - offset)
        loads = tuple(_chunk_load(load, offset, width) for load in batch.loads)
        outputs = tuple(
            OutputPlan(
                output=output_plan.output,
                twiddle=_chunk_twiddle(output_plan.twiddle, offset, width),
                scale=output_plan.scale,
                large_twiddle=output_plan.large_twiddle,
                store=_chunk_store(output_plan.store, offset, width),
            )
            for output_plan in batch.outputs
        )
        valid_lanes = max(0, min(width, batch.valid_lanes - offset))
        chunks.append(
            (
                width,
                SIMDBatchPlan(
                    batch_id=batch.batch_id,
                    valid_lanes=valid_lanes,
                    loads=loads,
                    outputs=outputs,
                ),
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Runtime-loop stage rendering.
#
# fft_codegen._emit_stage's default path renders one Mojo source block per
# SIMDBatchPlan -- the plan already lists one per `simd_lanes`-wide slice of
# a stage's butterflies (see fft_plan_core._lower_stages), so a stage with N
# large enough to need many such slices renders that many *copies* of the
# same load/twiddle/butterfly/store shape in the kernel body. That is fine
# for a handful of batches, but the compiled function eventually outgrows
# this target's RVV register file: the LLVM backend starts spilling whole
# vector registers with `vs1r.v`, an opcode M2NDP-Detour's decoder does not
# implement (see docs/STATUS.md / make_fft_kernel.py's own docstring on
# `compute_lanes`) -- the kernel then silently computes nothing for that
# microthread instead of crashing.
#
# `try_build_loop_stage` below decides whether the *same* per-batch plan
# data can instead render as one Mojo `while` loop over the batch index, so
# the compiled function's size stops growing with N. This works because a
# `SIMDBatchPlan`'s per-operand load offset and (for a "vector"-mode store)
# its store offset are already exactly `simd_it * <constant stride>` apart
# (see fft_plan_core._make_load/_make_store) -- `try_build_loop_stage`
# doesn't re-derive that; it *verifies* it directly from the plan's own
# already-computed offsets across every consecutive pair of full-width
# batches, and returns `None` (fall back to the unrolled path, unchanged)
# the moment any pair doesn't match. So this can never emit a wrong runtime
# offset -- it either proves the batch sequence is uniform and loops it, or
# gives up and unrolls exactly as before. Twiddle constants aren't offsets;
# a stage's per-batch TwiddlePlan values are exact already-computed floats,
# so those are pooled into one small per-kernel DRAM table (loaded at
# runtime by `simd_it`) rather than re-derived.
#
# Only whole (`valid_lanes == simd_lanes`) batches are looped; at most one
# trailing partial batch (`butterfly_count` not a multiple of `simd_lanes`)
# is left for the caller to render unlooped, exactly as before.

LOOP_MIN_FULL_BATCHES = 3

# Largest period try_build_loop_stage will accept -- bounds the search below
# and keeps a degenerate "period == every batch" (no smaller period found,
# so nothing would actually be shared across outer-loop iterations) from
# ever being accepted. See try_build_loop_stage's own docstring for what
# period *is* here.
LOOP_MAX_PERIOD = 64


def _arith_stride(seq: list[int]) -> int | None:
    """`seq[i+1] - seq[i]` if that difference is the same for every
    consecutive pair, else `None`. `0` (not `None`) for a length-<2 `seq`:
    there's nothing to contradict a stride of 0, and callers that need a
    real stride pass sequences with at least two elements."""
    if len(seq) < 2:
        return 0
    stride = seq[1] - seq[0]
    for i in range(1, len(seq) - 1):
        if seq[i + 1] - seq[i] != stride:
            return None
    return stride


def _periodic_stride(seq: list[int], period: int) -> tuple[tuple[int, ...], int] | None:
    """Split `seq` into `period` interleaved residue classes (residue `r`:
    `seq[r], seq[r+period], seq[r+2*period], ...`) and check each one is
    itself a plain arithmetic progression, *all* sharing one common
    difference (only the starting point may differ by residue) -- i.e.
    `seq[i] == bases[i % period] + (i // period) * stride`. Returns
    `(bases, stride)` if so, else `None`. Requires `period` to divide
    `len(seq)` evenly and every residue to have at least 2 rows (so there's
    something to actually contradict a stride) -- callers are expected to
    have already checked both.

    `period=1` is exactly `_arith_stride` (bases is a 1-tuple) -- this is
    the general form the loop-stage builder now needs; see the module note
    above `try_build_loop_stage`. A store's own per-batch destination
    offset is uniform *within* a residue only once a stage's cumulative
    twiddle-lane_divisor grows past `simd_lanes` (see that function's
    docstring): before that point period=1 already covers it.
    """
    n = len(seq)
    rows = n // period
    bases: list[int] = []
    stride: int | None = None
    for r in range(period):
        vals = [seq[r + row * period] for row in range(rows)]
        st = _arith_stride(vals)
        if st is None:
            return None
        if stride is None:
            stride = st
        elif st != stride:
            return None
        bases.append(vals[0])
    assert stride is not None
    return tuple(bases), stride


@dataclass
class LoopLoad:
    operand: int
    source: str
    buffer_name: str | None
    bases: tuple[int, ...]  # one per residue, length == loop_plan.period
    stride: int


@dataclass
class LoopStoreInfo:
    output: int
    destination: str
    buffer_name: str | None
    mode: str  # "vector" or "scalar_lanes"
    bases: tuple[int, ...] | None  # vector mode: one per residue
    stride: int | None
    lane_bases: tuple[tuple[int, ...], ...] | None  # scalar_lanes: [lane][residue]
    lane_stride: tuple[int, ...] | None  # [lane]


@dataclass
class LoopTwiddleInfo:
    output: int
    table_offset: int


@dataclass
class LoopStagePlan:
    period: int
    outer_iters: int
    loads: tuple[LoopLoad, ...]
    stores: tuple[LoopStoreInfo, ...]
    twiddles: dict[int, LoopTwiddleInfo]
    scales: dict[int, float | None]
    tail_batch: SIMDBatchPlan | None


def try_build_loop_stage(
    stage: FFTStagePlan,
    *,
    simd_lanes: int,
    min_full_batches: int,
    twiddle_table: list[tuple[float, float]],
) -> "LoopStagePlan | None":
    """Try to prove `stage.batches` is a uniform, loopable sequence (see the
    module note above) and, if so, at what period.

    A Stockham-autosort intermediate store's own per-batch destination
    offset (see fft_plan_core._make_store) is `n2*group_stride +
    output*p_s + b_s` where `p_s` is this stage's own cumulative radix
    product -- linear in the batch index (period=1) only while `p_s <=
    simd_lanes`; once a stage's `p_s` grows past `simd_lanes`, consecutive
    batches' destinations jump by a *larger* stride every `p_s/simd_lanes`
    batches, i.e. the sequence is linear *within* each of `p_s/simd_lanes`
    interleaved residues, not across all of them at once (confirmed against
    the plan's own already-computed offsets: N=1024's FFTRecNear0 stage 4/5/
    6 need period 2/4/8 respectively, doubling in step with `p_s` doubling
    each radix-2 stage -- see docs/STATUS.md and this function's own git
    history for the concrete offsets). Loads stay period=1 always (their
    own base_offset is `simd_it * input_batch_width`, no such `p_s` term).

    Search: try periods 1, 2, 3, ... up to `LOOP_MAX_PERIOD` (and dividing
    the batch count, with room for at least 2 outer-loop iterations); the
    first period at which *every* load and store offset sequence validates
    via `_periodic_stride` is accepted. Falls back to full unroll (`None`)
    if none does, or the batch count is too small to bother -- exactly as
    before this period generalization existed, just reached less often.

    Appends this stage's twiddle rows to the shared per-kernel
    `twiddle_table` (in place) only once every check has passed -- never
    leaves orphaned rows in it on a failed/aborted attempt.
    """
    full = [b for b in stage.batches if b.valid_lanes == simd_lanes]
    tail = [b for b in stage.batches if b.valid_lanes != simd_lanes]
    n_full = len(full)
    if n_full < min_full_batches or len(tail) > 1:
        return None

    n_operands = len(full[0].loads)
    if any(len(b.loads) != n_operands for b in full):
        return None
    load_seqs: list[list[int]] = []
    load_meta = []
    for j in range(n_operands):
        if any(b.loads[j].mode != "vector" for b in full):
            return None
        offs = [b.loads[j].base_offset for b in full]
        if any(o is None for o in offs):
            return None
        load_seqs.append(offs)
        first = full[0].loads[j]
        load_meta.append((j, first.source, first.buffer_name))

    n_outputs = len(full[0].outputs)
    if any(len(b.outputs) != n_outputs for b in full):
        return None

    # kind: ("vector", k, None, seq) or ("lane", k, lane, seq)
    store_seqs: list[tuple[str, int, int | None, list[int]]] = []
    scales: dict[int, float | None] = {}
    store_meta: dict[int, tuple[str, str, str | None]] = {}  # k -> (mode, destination, buffer_name)
    twiddle_needed: dict[int, bool] = {}

    for k in range(n_outputs):
        outs = [b.outputs[k] for b in full]
        if any(o.output != k for o in outs):
            return None
        if any(o.large_twiddle for o in outs):
            return None
        scale_vals = {o.scale for o in outs}
        if len(scale_vals) != 1:
            return None
        scales[k] = outs[0].scale

        store0 = outs[0].store
        if any(o.store.mode != store0.mode for o in outs):
            return None
        if any(o.store.destination != store0.destination for o in outs):
            return None
        if any(o.store.buffer_name != store0.buffer_name for o in outs):
            return None
        store_meta[k] = (store0.mode, store0.destination, store0.buffer_name)

        if store0.mode == "vector":
            offs = [o.store.base_offset for o in outs]
            if any(o is None for o in offs):
                return None
            store_seqs.append(("vector", k, None, offs))
        else:
            if any(len(o.store.lane_offsets) != simd_lanes for o in outs):
                return None
            for lane in range(simd_lanes):
                store_seqs.append(("lane", k, lane, [o.store.lane_offsets[lane] for o in outs]))

        twiddle_vals = [o.twiddle for o in outs]
        if any((t is None) != (twiddle_vals[0] is None) for t in twiddle_vals):
            return None
        twiddle_needed[k] = twiddle_vals[0] is not None
        if twiddle_vals[0] is not None:
            for t in twiddle_vals:
                assert t is not None
                if len(t.real) != simd_lanes or len(t.imag) != simd_lanes:
                    return None

    all_seqs = load_seqs + [s[3] for s in store_seqs]
    max_period = min(LOOP_MAX_PERIOD, n_full // 2)
    period = None
    for p in range(1, max_period + 1):
        if n_full % p != 0:
            continue
        if all(_periodic_stride(seq, p) is not None for seq in all_seqs):
            period = p
            break
    if period is None:
        return None
    outer_iters = n_full // period
    if outer_iters < 2:
        return None

    loop_loads = tuple(
        LoopLoad(
            operand=j, source=source, buffer_name=buffer_name,
            bases=(bp := _periodic_stride(seq, period))[0], stride=bp[1],
        )
        for (j, source, buffer_name), seq in zip(load_meta, load_seqs)
    )

    vector_data: dict[int, tuple[tuple[int, ...], int]] = {}
    lane_data: dict[int, dict[int, tuple[tuple[int, ...], int]]] = {}
    for kind, k, lane, seq in store_seqs:
        result = _periodic_stride(seq, period)
        assert result is not None
        if kind == "vector":
            vector_data[k] = result
        else:
            lane_data.setdefault(k, {})[lane] = result

    loop_stores: list[LoopStoreInfo] = []
    local_twiddle_rows: list[tuple[float, float]] = []
    twiddle_local_offset: dict[int, int] = {}
    for k in range(n_outputs):
        mode, destination, buffer_name = store_meta[k]
        if mode == "vector":
            bases, stride = vector_data[k]
            loop_stores.append(
                LoopStoreInfo(
                    output=k, destination=destination, buffer_name=buffer_name,
                    mode="vector", bases=bases, stride=stride, lane_bases=None, lane_stride=None,
                )
            )
        else:
            lb = tuple(lane_data[k][lane][0] for lane in range(simd_lanes))
            ls = tuple(lane_data[k][lane][1] for lane in range(simd_lanes))
            loop_stores.append(
                LoopStoreInfo(
                    output=k, destination=destination, buffer_name=buffer_name,
                    mode="scalar_lanes", bases=None, stride=None, lane_bases=lb, lane_stride=ls,
                )
            )

        if twiddle_needed[k]:
            # Batch-major (not residue-major): batch b's row lands at
            # table_offset + b*simd_lanes, so a fixed (residue, chunk, lane)
            # steps by period*simd_lanes per outer_it -- see
            # fft_codegen._emit_loop_chunk.
            twiddle_local_offset[k] = len(local_twiddle_rows)
            for b in full:
                t = b.outputs[k].twiddle
                assert t is not None
                for lane in range(simd_lanes):
                    local_twiddle_rows.append((t.real[lane], t.imag[lane]))

    base = len(twiddle_table)
    twiddle_table.extend(local_twiddle_rows)
    loop_twiddles = {
        k: LoopTwiddleInfo(output=k, table_offset=base + off)
        for k, off in twiddle_local_offset.items()
    }

    return LoopStagePlan(
        period=period,
        outer_iters=outer_iters,
        loads=loop_loads,
        stores=tuple(loop_stores),
        twiddles=loop_twiddles,
        scales=scales,
        tail_batch=tail[0] if tail else None,
    )
