from __future__ import annotations

"""Pure code emission from a fully lowered FFTCodegenPlan / DecomposedFFTPlan.

No FFT planning is performed here.  In particular this module does not:

* choose radices or stage order,
* calculate SIMD iteration counts or tail lanes,
* choose DRAM/scratchpad sources,
* choose ping-pong banks,
* choose whether an FFT is single-kernel or decomposed into N0*N1,
* choose a kernel's DRAM address mapping (contiguous vs. strided) or decide
  when a load/store falls back from vector to scalar-per-lane,
* calculate twiddle exponents/constants (small or large),
* calculate load/store indices,
* choose store layouts,
* build host reference FFT values.

All of those decisions are already materialized in fft_plangen.FFTCodegenPlan
/ fft_plangen.DecomposedFFTPlan. A decomposed FFT is two ordinary
FFTCodegenPlan kernels (see fft_plangen's module docstring); this module
renders each exactly as it would a single-kernel plan, and additionally
threads the large-twiddle table and the two kernels' DRAM hand-off through
one combined main() -- there is no separate "transpose kernel" abstraction
to emit.
"""

from dataclasses import dataclass

from codegen.fft_butterflies import emit_butterfly
from planning.fft_plan_core import (
    AddressMapping,
    AddressMappingKind,
    FFTCodegenPlan,
    FFTStagePlan,
    LargeTwiddlePlan,
    LoadPlan,
    MultiKernelFFTPlan,
    OutputPlan,
    SIMDBatchPlan,
    StorePlan,
    TwiddlePlan,
)
from planning.fft_plan_simple import DecomposedFFTPlan


class Emitter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _f32(value: float) -> str:
    """Format a planner-provided numeric constant as Mojo Float32 syntax."""
    if abs(value) < 1.0e-12:
        value = 0.0
    elif abs(value - 1.0) < 1.0e-12:
        value = 1.0
    elif abs(value + 1.0) < 1.0e-12:
        value = -1.0
    return f"Float32({value:.9g})"


def _spad(kernel_name: str, name: str) -> str:
    return f"{kernel_name}.{name}"


def _mapping_base_expr(mapping: AddressMapping, kernel_length: int) -> str:
    """`global_uthread_id() * row_stride [+ base]` -- the one runtime
    multiply a CONTIGUOUS/STRIDED AddressMapping ever costs. `elem*elem_stride`
    is folded into each load/store's own offset at plan time (see
    fft_plangen._make_load / _make_store), so that's the whole of what
    codegen computes at runtime for those two kinds: no per-access division
    or modulo.

    SPLIT was the original scheme for a middle kernel's output in an M>=3
    multi-kernel chain (see AddressMappingKind.SPLIT): `global_uthread_id()`
    mixes an already-transformed digit run (weight < a) with a
    not-yet-transformed remainder (weight >= a), and the kernel's own new
    digit needs to land *between* them -- one `%` and one `//`. No planner
    in this module constructs a SPLIT mapping any more (make_multi_kernel_plan
    uses PEELED instead, below, for every M>=3 chain); the kind and this
    description are kept only because PEELED's own derivation is explained
    by contrast to it.

    PEELED (see AddressMappingKind.PEELED) is SPLIT's replacement: the
    not-yet-transformed remainder itself splits further, into the *next*
    kernel's own digit (pulled to the innermost slot) and everything after
    it -- two `%`/`//` pairs instead of SPLIT's one.

    CROSSED (see AddressMappingKind.CROSSED, make_balanced_plan) splits
    `global_uthread_id()` into a batch index (`% batch_count`) and this
    side's own already-transformed digits (`// batch_count`), placing the
    batch index outermost instead of innermost -- the opposite of SPLIT/
    PEELED, since this is a transpose to the *other* side's benefit, not
    a within-side digit rotation.
    """
    if mapping.kind == AddressMappingKind.PEELED:
        a = mapping.peel_a
        k_next = mapping.peel_k_next
        tail = mapping.peel_tail_size
        return (
            f"(((global_uthread_id() // {a}) % {tail}) * {a * kernel_length * k_next}) + "
            f"((global_uthread_id() % {a}) * {k_next}) + "
            f"((global_uthread_id() // {a}) // {tail})"
        )
    if mapping.kind == AddressMappingKind.CROSSED:
        batch_count = mapping.peel_a
        side_length = mapping.row_stride
        return (
            f"(global_uthread_id() % {batch_count}) * {side_length} + "
            f"(global_uthread_id() // {batch_count})"
        )
    expr = f"global_uthread_id() * {mapping.row_stride}"
    if mapping.base:
        expr += f" + {mapping.base}"
    return expr


def _emit_load(e: Emitter, *, plan: FFTCodegenPlan, load: LoadPlan, width: int) -> None:
    """`width` is this (possibly chunked -- see `_chunk_batch`) load's own
    SIMD width, not necessarily `plan.simd_lanes`: the plan's `simd_lanes`
    is the hardware launch granule (ties to `PooledRange`/`VECTOR_WIDTH`,
    see `_chunk_batch`'s own docstring) and must not drive how wide a
    vector instruction the kernel body actually emits.
    """
    j = load.operand

    if load.mode == "vector":
        assert load.base_offset is not None
        if load.source == "input":
            e.add(
                f"        var rr{j} = p.input_real_base.load[width={width}]("
                f"in_batch_base + {load.base_offset})"
            )
            e.add(
                f"        var ii{j} = p.input_imag_base.load[width={width}]("
                f"in_batch_base + {load.base_offset})"
            )
        else:
            assert load.buffer_name is not None
            buf = _spad(plan.kernel_name, load.buffer_name)
            e.add(
                f"        var rr{j} = {buf}.load[DType.float32, {width}]("
                f"spad_base + {load.base_offset})"
            )
            e.add(
                f"        var ii{j} = {buf}.load[DType.float32, {width}]("
                f"spad_base + {plan.length} + {load.base_offset})"
            )
        return

    rr_lanes: list[str] = []
    ii_lanes: list[str] = []
    for lane, offset in enumerate(load.packed_lane_offsets):
        if offset is None:
            rr_lanes.append("Float32(0)")
            ii_lanes.append("Float32(0)")
            continue

        rr_scalar = f"rr{j}_lane{lane}"
        ii_scalar = f"ii{j}_lane{lane}"
        if load.source == "input":
            e.add(
                f"        var {rr_scalar} = p.input_real_base.load[width=1]("
                f"in_batch_base + {offset})"
            )
            e.add(
                f"        var {ii_scalar} = p.input_imag_base.load[width=1]("
                f"in_batch_base + {offset})"
            )
        else:
            assert load.buffer_name is not None
            buf = _spad(plan.kernel_name, load.buffer_name)
            e.add(
                f"        var {rr_scalar} = {buf}.load[DType.float32, 1]("
                f"spad_base + {offset})"
            )
            e.add(
                f"        var {ii_scalar} = {buf}.load[DType.float32, 1]("
                f"spad_base + {plan.length} + {offset})"
            )
        rr_lanes.append(f"{rr_scalar}[0]")
        ii_lanes.append(f"{ii_scalar}[0]")
    e.add(
        f"        var rr{j} = SIMD[DType.float32, {width}](" + ", ".join(rr_lanes) + ")"
    )
    e.add(
        f"        var ii{j} = SIMD[DType.float32, {width}](" + ", ".join(ii_lanes) + ")"
    )


def _emit_twiddle(
    e: Emitter, *, output: int, twiddle: TwiddlePlan, width: int
) -> None:
    e.add(
        f"        var twr{output} = SIMD[DType.float32, {width}]("
        + ", ".join(_f32(v) for v in twiddle.real)
        + ")"
    )
    e.add(
        f"        var twi{output} = SIMD[DType.float32, {width}]("
        + ", ".join(_f32(v) for v in twiddle.imag)
        + ")"
    )
    e.add(
        f"        var tr{output} = or{output} * twr{output} - "
        f"oi{output} * twi{output}"
    )
    e.add(
        f"        var ti{output} = or{output} * twi{output} + "
        f"oi{output} * twr{output}"
    )
    e.add(f"        or{output} = tr{output}")
    e.add(f"        oi{output} = ti{output}")


def _emit_large_twiddle(
    e: Emitter, *, output: int, store: StorePlan, width: int
) -> None:
    """Runtime cross-block twiddle, fused into this output's own store path
    (see fft_plangen.LargeTwiddlePlan): multiply by a value fetched from the
    large-twiddle DRAM table at exactly the address this output is about to
    store to (the table shares this kernel's output_mapping layout), rather
    than by a compile-time SIMD constant -- the exponent depends on
    global_uthread_id(), a runtime value, so it cannot be one.
    """
    k = output

    if store.mode == "vector":
        assert store.base_offset is not None
        e.add(
            f"        var ltwr{k} = p.large_twiddle_real_base.load[width={width}]("
            f"out_batch_base + {store.base_offset})"
        )
        e.add(
            f"        var ltwi{k} = p.large_twiddle_imag_base.load[width={width}]("
            f"out_batch_base + {store.base_offset})"
        )
    else:
        rr_lanes: list[str] = []
        ii_lanes: list[str] = []
        for lane, offset in enumerate(store.lane_offsets):
            rr_scalar = f"ltwr{k}_lane{lane}"
            ii_scalar = f"ltwi{k}_lane{lane}"
            e.add(
                f"        var {rr_scalar} = p.large_twiddle_real_base."
                f"load[width=1](out_batch_base + {offset})"
            )
            e.add(
                f"        var {ii_scalar} = p.large_twiddle_imag_base."
                f"load[width=1](out_batch_base + {offset})"
            )
            rr_lanes.append(f"{rr_scalar}[0]")
            ii_lanes.append(f"{ii_scalar}[0]")
        # Lanes past what this store actually writes never reach DRAM, so
        # they are padded with the neutral rotation (1, 0) rather than
        # fetched -- same convention fft_plangen._make_twiddle uses for a
        # partial SIMD batch's compile-time twiddle.
        pad = width - len(rr_lanes)
        rr_lanes += ["Float32(1)"] * pad
        ii_lanes += ["Float32(0)"] * pad
        e.add(
            f"        var ltwr{k} = SIMD[DType.float32, {width}](" + ", ".join(rr_lanes) + ")"
        )
        e.add(
            f"        var ltwi{k} = SIMD[DType.float32, {width}](" + ", ".join(ii_lanes) + ")"
        )

    e.add(f"        var ltr{k} = or{k} * ltwr{k} - oi{k} * ltwi{k}")
    e.add(f"        var lti{k} = or{k} * ltwi{k} + oi{k} * ltwr{k}")
    e.add(f"        or{k} = ltr{k}")
    e.add(f"        oi{k} = lti{k}")


def _emit_store(
    e: Emitter, *, plan: FFTCodegenPlan, output: int, store: StorePlan
) -> None:
    if store.mode == "vector":
        assert store.base_offset is not None
        if store.destination == "output":
            e.add(
                f"        p.output_real_base.store(out_batch_base + {store.base_offset}, "
                f"or{output})"
            )
            e.add(
                f"        p.output_imag_base.store(out_batch_base + {store.base_offset}, "
                f"oi{output})"
            )
        else:
            assert store.buffer_name is not None
            buf = _spad(plan.kernel_name, store.buffer_name)
            e.add(
                f"        {buf}.store(spad_base + {store.base_offset}, or{output})"
            )
            e.add(
                f"        {buf}.store(spad_base + {plan.length} + {store.base_offset}, oi{output})"
            )
        return

    for lane, offset in enumerate(store.lane_offsets):
        if store.destination == "output":
            e.add(
                f"        p.output_real_base.store(out_batch_base + {offset}, "
                f"or{output}[{lane}])"
            )
            e.add(
                f"        p.output_imag_base.store(out_batch_base + {offset}, "
                f"oi{output}[{lane}])"
            )
        else:
            assert store.buffer_name is not None
            buf = _spad(plan.kernel_name, store.buffer_name)
            e.add(
                f"        {buf}.store(spad_base + {offset}, or{output}[{lane}])"
            )
            e.add(
                f"        {buf}.store(spad_base + {plan.length} + {offset}, oi{output}[{lane}])"
            )


def _emit_output(
    e: Emitter, *, plan: FFTCodegenPlan, output_plan: OutputPlan, width: int
) -> None:
    k = output_plan.output
    if output_plan.twiddle is not None:
        _emit_twiddle(e, output=k, twiddle=output_plan.twiddle, width=width)

    if output_plan.scale is not None:
        scale = _f32(output_plan.scale)
        e.add(f"        or{k} *= {scale}")
        e.add(f"        oi{k} *= {scale}")

    if output_plan.large_twiddle:
        _emit_large_twiddle(e, output=k, store=output_plan.store, width=width)

    _emit_store(e, plan=plan, output=k, store=output_plan.store)
    e.add()


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
    return LoadPlan(
        operand=load.operand,
        source=load.source,
        buffer_name=load.buffer_name,
        mode="scalar_pack",
        packed_lane_offsets=load.packed_lane_offsets[offset : offset + width],
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


def _chunk_batch(
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
# `_emit_stage`'s default path (below) renders one Mojo source block per
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
# The functions below are an alternative renderer for the *same* per-batch
# plan data: one Mojo `while` loop over the batch index, instead of one
# unrolled block per batch, so the compiled function's size stops growing
# with N. This works because a `SIMDBatchPlan`'s per-operand load offset and
# (for a "vector"-mode store) its store offset are already exactly
# `simd_it * <constant stride>` apart (see fft_plan_core._make_load/
# _make_store) -- `_try_build_loop_stage` below doesn't re-derive that; it
# *verifies* it directly from the plan's own already-computed offsets across
# every consecutive pair of full-width batches, and returns `None` (falls
# back to the unrolled path, unchanged) the moment any pair doesn't match.
# So this can never emit a wrong runtime offset -- it either proves the
# batch sequence is uniform and loops it, or gives up and unrolls exactly as
# before. Twiddle constants aren't offsets; a stage's per-batch TwiddlePlan
# values are exact already-computed floats, so those are pooled into one
# small per-kernel DRAM table (loaded at runtime by `simd_it`) rather than
# re-derived.
#
# Only whole (`valid_lanes == simd_lanes`) batches are looped; at most one
# trailing partial batch (`butterfly_count` not a multiple of `simd_lanes`)
# is rendered exactly as today, unrolled, after the loop.

_LOOP_MIN_FULL_BATCHES = 3

# Largest period _try_build_loop_stage will accept -- bounds the search below
# and keeps a degenerate "period == every batch" (no smaller period found,
# so nothing would actually be shared across outer-loop iterations) from
# ever being accepted. See _try_build_loop_stage's own docstring for what
# period *is* here.
_LOOP_MAX_PERIOD = 64


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
    above `_try_build_loop_stage`. A store's own per-batch destination
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
class _LoopLoad:
    operand: int
    source: str
    buffer_name: str | None
    bases: tuple[int, ...]  # one per residue, length == loop_plan.period
    stride: int


@dataclass
class _LoopStoreInfo:
    output: int
    destination: str
    buffer_name: str | None
    mode: str  # "vector" or "scalar_lanes"
    bases: tuple[int, ...] | None  # vector mode: one per residue
    stride: int | None
    lane_bases: tuple[tuple[int, ...], ...] | None  # scalar_lanes: [lane][residue]
    lane_stride: tuple[int, ...] | None  # [lane]


@dataclass
class _LoopTwiddleInfo:
    output: int
    table_offset: int


@dataclass
class _LoopStagePlan:
    period: int
    outer_iters: int
    loads: tuple[_LoopLoad, ...]
    stores: tuple[_LoopStoreInfo, ...]
    twiddles: dict[int, _LoopTwiddleInfo]
    scales: dict[int, float | None]
    tail_batch: SIMDBatchPlan | None


def _try_build_loop_stage(
    stage: FFTStagePlan,
    *,
    simd_lanes: int,
    min_full_batches: int,
    twiddle_table: list[tuple[float, float]],
) -> "_LoopStagePlan | None":
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

    Search: try periods 1, 2, 3, ... up to `_LOOP_MAX_PERIOD` (and dividing
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
    max_period = min(_LOOP_MAX_PERIOD, n_full // 2)
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
        _LoopLoad(
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

    loop_stores: list[_LoopStoreInfo] = []
    local_twiddle_rows: list[tuple[float, float]] = []
    twiddle_local_offset: dict[int, int] = {}
    for k in range(n_outputs):
        mode, destination, buffer_name = store_meta[k]
        if mode == "vector":
            bases, stride = vector_data[k]
            loop_stores.append(
                _LoopStoreInfo(
                    output=k, destination=destination, buffer_name=buffer_name,
                    mode="vector", bases=bases, stride=stride, lane_bases=None, lane_stride=None,
                )
            )
        else:
            lb = tuple(lane_data[k][lane][0] for lane in range(simd_lanes))
            ls = tuple(lane_data[k][lane][1] for lane in range(simd_lanes))
            loop_stores.append(
                _LoopStoreInfo(
                    output=k, destination=destination, buffer_name=buffer_name,
                    mode="scalar_lanes", bases=None, stride=None, lane_bases=lb, lane_stride=ls,
                )
            )

        if twiddle_needed[k]:
            # Batch-major (not residue-major): batch b's row lands at
            # table_offset + b*simd_lanes, so a fixed (residue, chunk, lane)
            # steps by period*simd_lanes per outer_it -- see _emit_loop_chunk.
            twiddle_local_offset[k] = len(local_twiddle_rows)
            for b in full:
                t = b.outputs[k].twiddle
                assert t is not None
                for lane in range(simd_lanes):
                    local_twiddle_rows.append((t.real[lane], t.imag[lane]))

    base = len(twiddle_table)
    twiddle_table.extend(local_twiddle_rows)
    loop_twiddles = {
        k: _LoopTwiddleInfo(output=k, table_offset=base + off)
        for k, off in twiddle_local_offset.items()
    }

    return _LoopStagePlan(
        period=period,
        outer_iters=outer_iters,
        loads=loop_loads,
        stores=tuple(loop_stores),
        twiddles=loop_twiddles,
        scales=scales,
        tail_batch=tail[0] if tail else None,
    )


def _emit_load_loop(
    e: Emitter, *, plan: FFTCodegenPlan, load: _LoopLoad, offset_expr: str, width: int
) -> None:
    j = load.operand
    if load.source == "input":
        e.add(
            f"        var rr{j} = p.input_real_base.load[width={width}]("
            f"in_batch_base + {offset_expr})"
        )
        e.add(
            f"        var ii{j} = p.input_imag_base.load[width={width}]("
            f"in_batch_base + {offset_expr})"
        )
    else:
        assert load.buffer_name is not None
        buf = _spad(plan.kernel_name, load.buffer_name)
        e.add(
            f"        var rr{j} = {buf}.load[DType.float32, {width}]("
            f"spad_base + {offset_expr})"
        )
        e.add(
            f"        var ii{j} = {buf}.load[DType.float32, {width}]("
            f"spad_base + {plan.length} + {offset_expr})"
        )


def _emit_twiddle_loop(
    e: Emitter, *, output: int, table_offset_expr: str, width: int
) -> None:
    k = output
    e.add(
        f"        var twr{k} = p.loop_twiddle_real_base.load[width={width}]("
        f"{table_offset_expr})"
    )
    e.add(
        f"        var twi{k} = p.loop_twiddle_imag_base.load[width={width}]("
        f"{table_offset_expr})"
    )
    e.add(f"        var tr{k} = or{k} * twr{k} - oi{k} * twi{k}")
    e.add(f"        var ti{k} = or{k} * twi{k} + oi{k} * twr{k}")
    e.add(f"        or{k} = tr{k}")
    e.add(f"        oi{k} = ti{k}")


def _emit_store_loop(
    e: Emitter, *, plan: FFTCodegenPlan, output: int, store: _LoopStoreInfo,
    residue: int, chunk_offset: int, width: int,
) -> None:
    k = output
    if store.mode == "vector":
        assert store.bases is not None and store.stride is not None
        base = store.bases[residue] + chunk_offset
        offset_expr = f"({base} + outer_it * {store.stride})"
        if store.destination == "output":
            e.add(f"        p.output_real_base.store(out_batch_base + {offset_expr}, or{k})")
            e.add(f"        p.output_imag_base.store(out_batch_base + {offset_expr}, oi{k})")
        else:
            assert store.buffer_name is not None
            buf = _spad(plan.kernel_name, store.buffer_name)
            e.add(f"        {buf}.store(spad_base + {offset_expr}, or{k})")
            e.add(f"        {buf}.store(spad_base + {plan.length} + {offset_expr}, oi{k})")
        return

    assert store.lane_bases is not None and store.lane_stride is not None
    # Store vectorization, loop-stage case: mirrors _chunk_store's own
    # equivalent-instruction-selection exactly, just against this
    # renderer's own per-lane representation (_LoopStagePlan.lane_bases/
    # lane_stride, filled in _try_build_loop_stage from the same exact
    # StorePlan.lane_offsets fft_plan_core._make_store already computed --
    # nothing here recomputes an address). This width-wide slice of lanes
    # can share one vector store exactly when both hold: the lanes' own
    # bases are contiguous (stride 1, same fact _chunk_store checks) *and*
    # every one of them advances by the same amount per outer-loop
    # iteration (`lane_stride` uniform across the slice) -- only then does
    # a single `+ outer_it * stride` term stay correct for every lane the
    # vector store would cover at once. Not a new layout decision: every
    # value compared below was already fixed by the planner.
    lane_range = range(chunk_offset, chunk_offset + width)
    bases = [store.lane_bases[lane][residue] for lane in lane_range]
    strides = [store.lane_stride[lane] for lane in lane_range]
    if width > 0 and all(s == strides[0] for s in strides) and all(
        bases[i] == bases[0] + i for i in range(1, width)
    ):
        offset_expr = f"({bases[0]} + outer_it * {strides[0]})"
        if store.destination == "output":
            e.add(f"        p.output_real_base.store(out_batch_base + {offset_expr}, or{k})")
            e.add(f"        p.output_imag_base.store(out_batch_base + {offset_expr}, oi{k})")
        else:
            assert store.buffer_name is not None
            buf = _spad(plan.kernel_name, store.buffer_name)
            e.add(f"        {buf}.store(spad_base + {offset_expr}, or{k})")
            e.add(f"        {buf}.store(spad_base + {plan.length} + {offset_expr}, oi{k})")
        return

    for lane in range(width):
        abs_lane = chunk_offset + lane
        base = store.lane_bases[abs_lane][residue]
        offset_expr = f"({base} + outer_it * {store.lane_stride[abs_lane]})"
        if store.destination == "output":
            e.add(f"        p.output_real_base.store(out_batch_base + {offset_expr}, or{k}[{lane}])")
            e.add(f"        p.output_imag_base.store(out_batch_base + {offset_expr}, oi{k}[{lane}])")
        else:
            assert store.buffer_name is not None
            buf = _spad(plan.kernel_name, store.buffer_name)
            e.add(f"        {buf}.store(spad_base + {offset_expr}, or{k}[{lane}])")
            e.add(f"        {buf}.store(spad_base + {plan.length} + {offset_expr}, oi{k}[{lane}])")


def _emit_loop_chunk(
    e: Emitter, *, plan: FFTCodegenPlan, stage: FFTStagePlan, loop_plan: _LoopStagePlan,
    residue: int, chunk_offset: int, width: int,
) -> None:
    e.add(
        f"        # ===== stage {stage.stage_id}, loop residue {residue} chunk offset "
        f"{chunk_offset} (width {width}) ====="
    )
    for load in loop_plan.loads:
        base = load.bases[residue] + chunk_offset
        offset_expr = f"({base} + outer_it * {load.stride})"
        _emit_load_loop(e, plan=plan, load=load, offset_expr=offset_expr, width=width)
    e.add()

    stores_by_k = {s.output: s for s in loop_plan.stores}
    simd_lanes = plan.simd_lanes

    def on_output(k: int) -> None:
        tw = loop_plan.twiddles.get(k)
        if tw is not None:
            row_offset = tw.table_offset + residue * simd_lanes + chunk_offset
            table_offset_expr = f"({row_offset} + outer_it * {loop_plan.period * simd_lanes})"
            _emit_twiddle_loop(e, output=k, table_offset_expr=table_offset_expr, width=width)

        scale = loop_plan.scales.get(k)
        if scale is not None:
            e.add(f"        or{k} *= {_f32(scale)}")
            e.add(f"        oi{k} *= {_f32(scale)}")

        _emit_store_loop(
            e, plan=plan, output=k, store=stores_by_k[k],
            residue=residue, chunk_offset=chunk_offset, width=width,
        )
        e.add()

    emit_butterfly(e, indent="        ", radix=stage.radix, inverse=stage.inverse, on_output=on_output)
    e.add()


def _emit_loop_stage(
    e: Emitter, *, plan: FFTCodegenPlan, stage: FFTStagePlan, loop_plan: _LoopStagePlan,
    compute_lanes: int | None,
) -> None:
    simd_lanes = plan.simd_lanes
    cl = compute_lanes if compute_lanes is not None else simd_lanes
    n_chunks = (simd_lanes + cl - 1) // cl
    period = loop_plan.period

    body = Emitter()
    for r in range(period):
        for c in range(n_chunks):
            chunk_offset = c * cl
            width = min(cl, simd_lanes - chunk_offset)
            if period > 1 or n_chunks > 1:
                sub = Emitter()
                _emit_loop_chunk(
                    sub, plan=plan, stage=stage, loop_plan=loop_plan,
                    residue=r, chunk_offset=chunk_offset, width=width,
                )
                body.add(f"        if True:  # residue {r} chunk {c}")
                for line in sub.lines:
                    body.add("    " + line if line else "")
            else:
                _emit_loop_chunk(
                    body, plan=plan, stage=stage, loop_plan=loop_plan,
                    residue=r, chunk_offset=chunk_offset, width=width,
                )

    e.add("        var outer_it = 0")
    e.add(f"        while outer_it < {loop_plan.outer_iters}:")
    for line in body.lines:
        e.add("    " + line if line else "")
    e.add("            outer_it += 1")
    e.add()


def _emit_batch(
    e: Emitter,
    *,
    plan: FFTCodegenPlan,
    stage: FFTStagePlan,
    batch: SIMDBatchPlan,
    width: int,
) -> None:
    e.add(
        f"        # ===== stage {stage.stage_id}, SIMD batch {batch.batch_id} "
        f"(valid lanes: {batch.valid_lanes}/{width}) ====="
    )

    for load in batch.loads:
        _emit_load(e, plan=plan, load=load, width=width)
    e.add()

    outputs_by_k = {output_plan.output: output_plan for output_plan in batch.outputs}

    def on_output(k: int) -> None:
        _emit_output(e, plan=plan, output_plan=outputs_by_k[k], width=width)

    emit_butterfly(
        e,
        indent="        ",
        radix=stage.radix,
        inverse=stage.inverse,
        on_output=on_output,
    )
    e.add()


def _emit_stage(
    e: Emitter,
    *,
    plan: FFTCodegenPlan,
    stage: FFTStagePlan,
    compute_lanes: int | None = None,
    loop_stages: bool = False,
    min_loop_batches: int = _LOOP_MIN_FULL_BATCHES,
    twiddle_table: list[tuple[float, float]] | None = None,
) -> None:
    is_first = stage.stage_id == 0
    is_last = stage.stage_id == len(plan.stages) - 1

    e.add("    @staticmethod")
    e.add(f"    def stage_{stage.stage_id}():")
    e.add(f"        ref p = {plan.kernel_name}.params[]")
    e.add(f"        comptime RADIX = {stage.radix}")
    e.add(f"        comptime SIMD_ITERS = {stage.simd_iteration_count}")
    e.add()
    e.add("        var local_id = local_uthread_id()")
    e.add(f"        if local_id >= MAX_UTHREAD_{plan.kernel_name}:")
    e.add("            return")
    # A single-stage kernel (this kernel's own length == its one radix, as
    # every decomposed sub-FFT kernel is) never reads or writes its own
    # scratchpad -- first_stage and last_stage are the same stage, so every
    # load/store is DRAM-side. Only declare spad_base where something uses it.
    if plan.scratchpad_buffers:
        e.add(f"        var spad_base = local_id * {plan.scratchpad_uthread_stride}")
    # Each kernel's own AddressMapping settles this in one runtime multiply
    # (plus, only where the mapping isn't the origin, one add) -- except
    # SPLIT, a middle kernel's output in an M>=3 chain, which costs one %
    # and one // (see _mapping_base_expr). Declared only on the stage that
    # actually reads/writes DRAM through it.
    if is_first:
        e.add(
            f"        var in_batch_base = "
            f"{_mapping_base_expr(plan.input_mapping, plan.length)}"
        )
    if is_last:
        e.add(
            f"        var out_batch_base = "
            f"{_mapping_base_expr(plan.output_mapping, plan.length)}"
        )
    e.add()

    if loop_stages:
        assert twiddle_table is not None
        loop_plan = _try_build_loop_stage(
            stage, simd_lanes=plan.simd_lanes, min_full_batches=min_loop_batches,
            twiddle_table=twiddle_table,
        )
        if loop_plan is not None:
            _emit_loop_stage(e, plan=plan, stage=stage, loop_plan=loop_plan, compute_lanes=compute_lanes)
            if loop_plan.tail_batch is not None:
                _emit_stage_batches(e, plan=plan, stage=stage, batches=(loop_plan.tail_batch,), compute_lanes=compute_lanes)
            return

    # Each batch's rr{k}/ii{k}/or{k}/oi{k} (and friends) are local to that
    # batch's own butterfly, not threads carried across batches -- but
    # _emit_batch always names them the same way regardless of batch_id, so
    # a stage with more than one SIMD batch (or, now, more than one
    # compute-width chunk within a batch -- see _chunk_batch) needs each
    # piece in its own block scope or the next one's `var rr0` redefines
    # the previous.
    _emit_stage_batches(e, plan=plan, stage=stage, batches=stage.batches, compute_lanes=compute_lanes)


def _emit_stage_batches(
    e: Emitter, *, plan: FFTCodegenPlan, stage: FFTStagePlan,
    batches: tuple[SIMDBatchPlan, ...], compute_lanes: int | None,
) -> None:
    pieces: list[tuple[int, SIMDBatchPlan]] = []
    for batch in batches:
        pieces.extend(
            _chunk_batch(
                batch,
                simd_lanes=plan.simd_lanes,
                compute_lanes=compute_lanes if compute_lanes is not None else plan.simd_lanes,
            )
        )

    for width, piece in pieces:
        if len(pieces) > 1:
            sub = Emitter()
            _emit_batch(sub, plan=plan, stage=stage, batch=piece, width=width)
            e.add(f"        if True:  # batch {piece.batch_id} scope")
            for line in sub.lines:
                e.add("    " + line if line else "")
        else:
            _emit_batch(e, plan=plan, stage=stage, batch=piece, width=width)


def _emit_params_struct(
    e: Emitter, *, plan: FFTCodegenPlan, add_loop_twiddle: bool = False
) -> None:
    e.add("@fieldwise_init")
    e.add(f"struct {plan.kernel_name}Params(Movable):")
    e.add("    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add("    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    if plan.large_twiddle is not None:
        e.add("    var large_twiddle_real_base: UnsafePointer[Float32, MutAnyOrigin]")
        e.add("    var large_twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    if add_loop_twiddle:
        # Shared per-kernel table pooling every looped stage's twiddle rows
        # (see the runtime-loop stage rendering note above _try_build_loop_stage)
        # -- always declared when the caller opts this kernel into loop_stages,
        # even if no stage in it actually ends up looping, so the field doesn't
        # depend on that per-stage outcome (see _emit_kernel).
        e.add("    var loop_twiddle_real_base: UnsafePointer[Float32, MutAnyOrigin]")
        e.add("    var loop_twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]")
    e.add()
    e.add()


def _emit_task_struct(
    e: Emitter,
    *,
    plan: FFTCodegenPlan,
    compute_lanes: int | None = None,
    loop_stages: bool = False,
    min_loop_batches: int = _LOOP_MIN_FULL_BATCHES,
) -> list[tuple[float, float]]:
    """The NDPTask struct: its scratchpad buffers (each uthread's own
    region, sized by scratchpad_uthread_stride -- see module docstring),
    its per-stage kernels, and device_main. Identical in shape whether this
    plan is a whole single-kernel FFT or one half of a decomposed one.

    `compute_lanes`: see `_chunk_batch` -- the SIMD width the emitted
    arithmetic instructions use, independent of `plan.simd_lanes` (the
    launch granule). `None` (the default) keeps every stage emitted at
    `plan.simd_lanes`, unchanged from before this parameter existed.

    `loop_stages`: see the runtime-loop stage rendering note above
    _try_build_loop_stage. `False` (the default) keeps every stage's
    per-batch code fully unrolled, unchanged from before this parameter
    existed. Returns the flat (real, imag) twiddle table every looped stage
    of this kernel pooled its constants into -- empty when `loop_stages` is
    `False` or no stage in this kernel qualified to loop. The caller (see
    fft_transpose_codegen.generate_recursive_fft_kernels) owns turning this
    into an actual DRAM buffer and passing it through this kernel's launch
    Params, since that's a host/main()-level decision this module doesn't
    make on its own.
    """
    e.add(f"struct {plan.kernel_name}(NDPTask):")
    e.add(f"    comptime Params = {plan.kernel_name}Params")
    e.add()

    for buffer in plan.scratchpad_buffers:
        e.add(
            f'    comptime {buffer.name} = scratchpad[{buffer.elements}, Float32, '
            f'name="{plan.kernel_name.lower()}_{buffer.name}"]()'
        )
    if plan.scratchpad_buffers:
        e.add()

    twiddle_table: list[tuple[float, float]] = []
    for stage in plan.stages:
        _emit_stage(
            e, plan=plan, stage=stage, compute_lanes=compute_lanes,
            loop_stages=loop_stages, min_loop_batches=min_loop_batches,
            twiddle_table=twiddle_table,
        )

    e.add("    @staticmethod")
    e.add("    def device_main():")
    for stage in plan.stages:
        e.add(f"        launch_parallel[{plan.kernel_name}.stage_{stage.stage_id}]()")
    e.add()
    e.add()
    return twiddle_table


def _emit_kernel(
    e: Emitter,
    *,
    plan: FFTCodegenPlan,
    compute_lanes: int | None = None,
    loop_stages: bool = False,
    min_loop_batches: int = _LOOP_MIN_FULL_BATCHES,
) -> list[tuple[float, float]]:
    e.add(f"comptime MAX_UTHREAD_{plan.kernel_name} = {plan.max_uthread}")
    e.add()
    _emit_params_struct(e, plan=plan, add_loop_twiddle=loop_stages)
    return _emit_task_struct(
        e, plan=plan, compute_lanes=compute_lanes,
        loop_stages=loop_stages, min_loop_batches=min_loop_batches,
    )


def _emit_prelude(e: Emitter) -> None:
    e.add("from std.sys import size_of")
    e.add("from std.random import random_float64, seed")
    e.add("from std.math import cos as host_cos, sin as host_sin")
    e.add()
    e.add(
        "from m2ndp import VECTOR_WIDTH, NDPTask, PooledRange, "
        "global_uthread_id, local_uthread_id, launch_parallel, scratchpad"
    )
    e.add("from m2ndp_host import cxl_alloc")
    e.add()
    e.add("comptime W = VECTOR_WIDTH // size_of[Float32]()")


def _emit_reference_check(
    e: Emitter,
    *,
    n: int,
    batch_count: int,
    inverse: bool,
    input_real: str,
    input_imag: str,
    output_real: str,
    output_imag: str,
    ref_real: str,
    ref_imag: str,
    tolerance: float,
    label: str,
) -> None:
    """A direct O(N^2) DFT, computed at host runtime against whatever input
    was just randomly generated -- independent of whichever radix
    decomposition or kernel split actually ran, single-kernel or
    decomposed alike (see fft_plangen.HostPlan / DecomposedHostPlan).
    Accumulated in Float64 so this check doesn't share the kernel's own
    fp32 rounding, rounding to Float32 only once, at the very end.

    `batch_count` independent length-`n` FFTs share one buffer, back to
    back (`batch*n + n_`) -- a single-kernel plan's own total_uthreads;
    always 1 for a decomposed plan (see DecomposedFFTPlan).
    """
    e.add("    var pi = Float64(3.141592653589793)")
    e.add(f"    var sign = Float64({1.0 if inverse else -1.0})")
    e.add(f"    for batch in range({batch_count}):")
    e.add(f"        var batch_base = batch * {n}")
    e.add(f"        for k in range({n}):")
    e.add("            var acc_r = Float64(0)")
    e.add("            var acc_i = Float64(0)")
    e.add(f"            for n_ in range({n}):")
    e.add(
        "                var angle = sign * 2.0 * pi * Float64(n_) * Float64(k) / "
        f"Float64({n})"
    )
    e.add("                var c = host_cos(angle)")
    e.add("                var s = host_sin(angle)")
    e.add(f"                var xr = Float64({input_real}[batch_base + n_])")
    e.add(f"                var xi = Float64({input_imag}[batch_base + n_])")
    e.add("                acc_r += xr * c - xi * s")
    e.add("                acc_i += xr * s + xi * c")
    if inverse:
        e.add(f"            acc_r /= Float64({n})")
        e.add(f"            acc_i /= Float64({n})")
    e.add(f"            {ref_real}[batch_base + k] = Float32(acc_r)")
    e.add(f"            {ref_imag}[batch_base + k] = Float32(acc_i)")
    e.add()

    e.add(f"    var tol = {_f32(tolerance)}")
    e.add(f"    for i in range({n * batch_count}):")
    e.add(f"        var err_r = {output_real}[i] - {ref_real}[i]")
    e.add(f"        var err_i = {output_imag}[i] - {ref_imag}[i]")
    e.add("        if err_r < Float32(0):")
    e.add("            err_r = -err_r")
    e.add("        if err_i < Float32(0):")
    e.add("            err_i = -err_i")
    e.add("        if err_r > tol or err_i > tol:")
    e.add(f'            print("[host] {label} mismatch at", i)')
    e.add(f'            print("  expected:", {ref_real}[i], {ref_imag}[i])')
    e.add(f'            print("  actual:  ", {output_real}[i], {output_imag}[i])')
    e.add('            print("  error:   ", err_r, err_i)')
    e.add("            return")
    e.add()
    e.add(f'    print("[host] {label} verification passed")')


def generate_fft_kernel(plan: FFTCodegenPlan, *, compute_lanes: int | None = None) -> str:
    """Render a single-kernel plan: the whole FFT in one NDPTask, one
    launch. This function performs no FFT planning. `compute_lanes`: see
    `_chunk_batch`; `None` (the default) keeps today's output unchanged.
    """
    e = Emitter()
    _emit_prelude(e)
    e.add(f"comptime N = {plan.length}")
    e.add(f"comptime MAX_UTHREAD_{plan.kernel_name} = {plan.max_uthread}")
    e.add()
    _emit_params_struct(e, plan=plan)
    _emit_task_struct(e, plan=plan, compute_lanes=compute_lanes)

    host = plan.host
    e.add("def main() raises:")
    e.add(f"    if {plan.kernel_name}.emit_ir_if_asked():")
    e.add("        return")
    e.add()
    e.add(f"    var total_elems = {host.total_elems}")
    e.add("    var input_real = cxl_alloc[Float32](total_elems)")
    e.add("    var input_imag = cxl_alloc[Float32](total_elems)")
    e.add("    var output_real = cxl_alloc[Float32](total_elems)")
    e.add("    var output_imag = cxl_alloc[Float32](total_elems)")
    e.add("    var ref_real = cxl_alloc[Float32](total_elems)")
    e.add("    var ref_imag = cxl_alloc[Float32](total_elems)")
    e.add()
    e.add(f"    var pool_elems = {host.pool_elems}")
    e.add("    var uthread_pool = cxl_alloc[Float32](pool_elems)")
    e.add()

    # Fresh random input every run (std.random, seeded like every other
    # benchmark's main()), not a signal fixed once at generation time.
    e.add("    seed(0)")
    e.add("    for i in range(total_elems):")
    e.add("        input_real[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        input_imag[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        output_real[i] = Float32(0)")
    e.add("        output_imag[i] = Float32(0)")
    e.add("        ref_real[i] = Float32(0)")
    e.add("        ref_imag[i] = Float32(0)")
    e.add()

    e.add(f"    var rc = {plan.kernel_name}.launch(")
    e.add("        PooledRange.over(uthread_pool, pool_elems),")
    e.add(
        f"        {plan.kernel_name}Params(input_real, input_imag, output_real, output_imag),"
    )
    e.add("    )")
    e.add()
    e.add("    if rc != 0:")
    e.add('        print("[host] FFT failed, exit", rc)')
    e.add("        return")
    e.add()

    _emit_reference_check(
        e,
        n=plan.length,
        batch_count=plan.total_uthreads,
        inverse=plan.inverse,
        input_real="input_real",
        input_imag="input_imag",
        output_real="output_real",
        output_imag="output_imag",
        ref_real="ref_real",
        ref_imag="ref_imag",
        tolerance=host.tolerance,
        label="FFT",
    )

    return e.text()


def generate_decomposed_fft_kernels(
    plan: DecomposedFFTPlan, *, compute_lanes: int | None = None
) -> str:
    """Render a decomposed (N = N0*N1) plan: two NDPTask structs, chained
    through DRAM the way two_tasks.mojo chains Scale/AddB -- launched one
    after the other from one host main(), never through a shared
    scratchpad. This function performs no FFT planning: which kernel owns
    which factor, every AddressMapping, and the large-twiddle table shape
    are already decided in `plan`. `compute_lanes`: see `_chunk_batch`;
    `None` (the default) keeps today's output unchanged.
    """
    e = Emitter()
    _emit_prelude(e)
    e.add(f"comptime N0 = {plan.n0}")
    e.add(f"comptime N1 = {plan.n1}")
    e.add(f"comptime N = {plan.n}")
    e.add()

    _emit_kernel(e, plan=plan.kernel0, compute_lanes=compute_lanes)
    _emit_kernel(e, plan=plan.kernel1, compute_lanes=compute_lanes)

    host = plan.host
    k0 = plan.kernel0
    k1 = plan.kernel1

    e.add("def main() raises:")
    e.add(f"    if {k0.kernel_name}.emit_ir_if_asked():")
    e.add("        return")
    e.add()
    e.add(f"    var n = {host.n}")
    e.add("    var input_real = cxl_alloc[Float32](n)")
    e.add("    var input_imag = cxl_alloc[Float32](n)")
    e.add("    var mid_real = cxl_alloc[Float32](n)")
    e.add("    var mid_imag = cxl_alloc[Float32](n)")
    e.add("    var output_real = cxl_alloc[Float32](n)")
    e.add("    var output_imag = cxl_alloc[Float32](n)")
    e.add("    var ref_real = cxl_alloc[Float32](n)")
    e.add("    var ref_imag = cxl_alloc[Float32](n)")
    e.add()

    assert plan.kernel0.large_twiddle is not None
    lt = plan.kernel0.large_twiddle
    e.add(f"    var large_twiddle_real = cxl_alloc[Float32](n)")
    e.add(f"    var large_twiddle_imag = cxl_alloc[Float32](n)")
    e.add("    var lt_pi = Float64(3.141592653589793)")
    e.add(f"    var lt_sign = Float64({1.0 if lt.inverse else -1.0})")
    e.add("    var r = 0")
    e.add(f"    while r < {lt.row_count}:")
    e.add("        var c1 = 0")
    e.add(f"        while c1 < {lt.output_count}:")
    e.add(
        "            var angle = lt_sign * 2.0 * lt_pi * Float64(r) * Float64(c1) / "
        f"Float64({lt.full_length})"
    )
    e.add(f"            large_twiddle_real[r * {lt.output_count} + c1] = Float32(host_cos(angle))")
    e.add(f"            large_twiddle_imag[r * {lt.output_count} + c1] = Float32(host_sin(angle))")
    e.add("            c1 += 1")
    e.add("        r += 1")
    e.add()

    e.add(f"    var pool0_elems = {k0.simd_lanes * k0.total_uthreads}")
    e.add(f"    var pool1_elems = {k1.simd_lanes * k1.total_uthreads}")
    e.add("    var pool0 = cxl_alloc[Float32](pool0_elems)")
    e.add("    var pool1 = cxl_alloc[Float32](pool1_elems)")
    e.add()

    e.add("    seed(0)")
    e.add("    for i in range(n):")
    e.add("        input_real[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        input_imag[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        mid_real[i] = Float32(0)")
    e.add("        mid_imag[i] = Float32(0)")
    e.add("        output_real[i] = Float32(0)")
    e.add("        output_imag[i] = Float32(0)")
    e.add("        ref_real[i] = Float32(0)")
    e.add("        ref_imag[i] = Float32(0)")
    e.add()

    e.add(f"    var rc0 = {k0.kernel_name}.launch(")
    e.add("        PooledRange.over(pool0, pool0_elems),")
    e.add(
        f"        {k0.kernel_name}Params(input_real, input_imag, mid_real, mid_imag, "
        "large_twiddle_real, large_twiddle_imag),"
    )
    e.add("    )")
    e.add("    if rc0 != 0:")
    e.add('        print("[host] FFT kernel0 failed, exit", rc0)')
    e.add("        return")
    e.add()

    e.add(f"    var rc1 = {k1.kernel_name}.launch(")
    e.add("        PooledRange.over(pool1, pool1_elems),")
    e.add(
        f"        {k1.kernel_name}Params(mid_real, mid_imag, output_real, output_imag),"
    )
    e.add("    )")
    e.add("    if rc1 != 0:")
    e.add('        print("[host] FFT kernel1 failed, exit", rc1)')
    e.add("        return")
    e.add()

    _emit_reference_check(
        e,
        n=plan.n,
        batch_count=1,
        inverse=plan.inverse,
        input_real="input_real",
        input_imag="input_imag",
        output_real="output_real",
        output_imag="output_imag",
        ref_real="ref_real",
        ref_imag="ref_imag",
        tolerance=host.tolerance,
        label="decomposed FFT",
    )

    return e.text()


def _emit_large_twiddle_table_precompute(
    e: Emitter, *, lt: LargeTwiddlePlan, real_name: str, imag_name: str, suffix: str
) -> None:
    """Host precompute for one non-last kernel's large-twiddle table (see
    fft_plangen.LargeTwiddlePlan / generate_multi_kernel_fft_kernels).
    `real_name`/`imag_name` are each `lt.full_length` Float32 elements,
    filled at every address the on-device fetch will ever read.

    Every local variable is suffixed: this runs once per non-last kernel
    in the same main(), and an earlier version of the single-kernel host
    check hit exactly this collision (two `var pi = ...`s in one scope --
    see the fft_plangen.make_decomposed_plan inverse_scale/lt_pi fix) for
    the same reason.
    """
    sign = 1.0 if lt.inverse else -1.0
    pi, lsign = f"lt_pi_{suffix}", f"lt_sign_{suffix}"
    r, c1 = f"lt_r_{suffix}", f"lt_c1_{suffix}"
    angle, val_r, val_i = f"lt_angle_{suffix}", f"lt_val_r_{suffix}", f"lt_val_i_{suffix}"

    e.add(f"    var {pi} = Float64(3.141592653589793)")
    e.add(f"    var {lsign} = Float64({sign})")
    e.add(f"    var {r} = 0")
    e.add(f"    while {r} < {lt.row_count}:")
    e.add(f"        var {c1} = 0")
    e.add(f"        while {c1} < {lt.output_count}:")
    e.add(
        f"            var {angle} = {lsign} * 2.0 * {pi} * Float64({r}) * "
        f"Float64({c1}) * Float64({lt.a}) / Float64({lt.full_length})"
    )
    e.add(f"            var {val_r} = Float32(host_cos({angle}))")
    e.add(f"            var {val_i} = Float32(host_sin({angle}))")
    if lt.k_next:
        # PEELED-addressed (see AddressMappingKind.PEELED): fill every
        # address the fetch can land on, for every already-transformed
        # out_a in [0, a) -- mirrors the kernel's own write formula
        # exactly (fft_codegen._mapping_base_expr's PEELED branch), split
        # here into d_next/rest since this loop already has `r` (=
        # remaining) directly rather than a packed address to invert.
        out_a, addr = f"lt_out_a_{suffix}", f"lt_addr_{suffix}"
        d_next, rest = f"lt_dnext_{suffix}", f"lt_rest_{suffix}"
        e.add(f"            var {d_next} = {r} // {lt.tail_size}")
        e.add(f"            var {rest} = {r} % {lt.tail_size}")
        e.add(f"            var {out_a} = 0")
        e.add(f"            while {out_a} < {lt.a}:")
        e.add(
            f"                var {addr} = {rest} * {lt.a * lt.output_count * lt.k_next} + "
            f"({out_a} + {c1} * {lt.a}) * {lt.k_next} + {d_next}"
        )
        e.add(f"                {real_name}[{addr}] = {val_r}")
        e.add(f"                {imag_name}[{addr}] = {val_i}")
        e.add(f"                {out_a} += 1")
    elif lt.a == 1:
        # Dense: every address visited exactly once.
        e.add(f"            {real_name}[{r} * {lt.output_count} + {c1}] = {val_r}")
        e.add(f"            {imag_name}[{r} * {lt.output_count} + {c1}] = {val_i}")
    else:
        # a-fold redundant: the SPLIT-addressed fetch reads the same value
        # for every out_a in [0, a) -- see AddressMappingKind.SPLIT.
        out_a, addr = f"lt_out_a_{suffix}", f"lt_addr_{suffix}"
        e.add(f"            var {out_a} = 0")
        e.add(f"            while {out_a} < {lt.a}:")
        e.add(
            f"                var {addr} = {out_a} + {c1} * {lt.a} + "
            f"{r} * {lt.a * lt.output_count}"
        )
        e.add(f"                {real_name}[{addr}] = {val_r}")
        e.add(f"                {imag_name}[{addr}] = {val_i}")
        e.add(f"                {out_a} += 1")
    e.add(f"            {c1} += 1")
    e.add(f"        {r} += 1")
    e.add()


def generate_multi_kernel_fft_kernels(
    plan: MultiKernelFFTPlan, *, compute_lanes: int | None = None
) -> str:
    """Render an M-kernel chained plan (see fft_plangen.make_multi_kernel_plan):
    M NDPTask structs, chained through DRAM one launch after another from
    one host main(), each non-last kernel's own large-twiddle table
    precomputed alongside it. Generalizes generate_decomposed_fft_kernels
    (exactly M=2, one bare radix per kernel) to any M>=1 and to each
    kernel's own layouts_for_radices multi-stage structure. This function
    performs no FFT planning: every AddressMapping and LargeTwiddlePlan is
    already decided in `plan`. `compute_lanes`: see `_chunk_batch`; `None`
    (the default) keeps today's output unchanged.
    """
    e = Emitter()
    _emit_prelude(e)
    e.add(f"comptime N = {plan.n}")
    e.add()

    for kernel in plan.kernels:
        _emit_kernel(e, plan=kernel, compute_lanes=compute_lanes)

    host = plan.host
    m = len(plan.kernels)
    k0 = plan.kernels[0]

    def buf_name(idx: int, part: str) -> str:
        # idx: -1 is the original input, m-1 is the final output, anything
        # in between is the intermediate buffer that many kernels wrote.
        if idx == -1:
            return f"input_{part}"
        if idx == m - 1:
            return f"output_{part}"
        return f"mid{idx}_{part}"

    e.add("def main() raises:")
    e.add(f"    if {k0.kernel_name}.emit_ir_if_asked():")
    e.add("        return")
    e.add()
    e.add(f"    var n = {host.n}")
    e.add("    var input_real = cxl_alloc[Float32](n)")
    e.add("    var input_imag = cxl_alloc[Float32](n)")
    for i in range(m - 1):
        e.add(f"    var mid{i}_real = cxl_alloc[Float32](n)")
        e.add(f"    var mid{i}_imag = cxl_alloc[Float32](n)")
    e.add("    var output_real = cxl_alloc[Float32](n)")
    e.add("    var output_imag = cxl_alloc[Float32](n)")
    e.add("    var ref_real = cxl_alloc[Float32](n)")
    e.add("    var ref_imag = cxl_alloc[Float32](n)")
    e.add()

    for i, kernel in enumerate(plan.kernels[:-1]):
        assert kernel.large_twiddle is not None
        e.add(f"    var large_twiddle{i}_real = cxl_alloc[Float32](n)")
        e.add(f"    var large_twiddle{i}_imag = cxl_alloc[Float32](n)")
        _emit_large_twiddle_table_precompute(
            e,
            lt=kernel.large_twiddle,
            real_name=f"large_twiddle{i}_real",
            imag_name=f"large_twiddle{i}_imag",
            suffix=str(i),
        )

    for i, kernel in enumerate(plan.kernels):
        e.add(f"    var pool{i}_elems = {kernel.simd_lanes * kernel.total_uthreads}")
        e.add(f"    var pool{i} = cxl_alloc[Float32](pool{i}_elems)")
    e.add()

    e.add("    seed(0)")
    e.add("    for i in range(n):")
    e.add("        input_real[i] = Float32(random_float64(-1.0, 1.0))")
    e.add("        input_imag[i] = Float32(random_float64(-1.0, 1.0))")
    for i in range(m - 1):
        e.add(f"        mid{i}_real[i] = Float32(0)")
        e.add(f"        mid{i}_imag[i] = Float32(0)")
    e.add("        output_real[i] = Float32(0)")
    e.add("        output_imag[i] = Float32(0)")
    e.add("        ref_real[i] = Float32(0)")
    e.add("        ref_imag[i] = Float32(0)")
    e.add()

    for i, kernel in enumerate(plan.kernels):
        is_last = i == m - 1
        in_r, in_i = buf_name(i - 1, "real"), buf_name(i - 1, "imag")
        out_r, out_i = buf_name(i, "real"), buf_name(i, "imag")
        e.add(f"    var rc{i} = {kernel.kernel_name}.launch(")
        e.add(f"        PooledRange.over(pool{i}, pool{i}_elems),")
        if is_last:
            e.add(
                f"        {kernel.kernel_name}Params({in_r}, {in_i}, {out_r}, {out_i}),"
            )
        else:
            e.add(
                f"        {kernel.kernel_name}Params({in_r}, {in_i}, {out_r}, {out_i}, "
                f"large_twiddle{i}_real, large_twiddle{i}_imag),"
            )
        e.add("    )")
        e.add(f"    if rc{i} != 0:")
        e.add(f'        print("[host] FFT kernel{i} failed, exit", rc{i})')
        e.add("        return")
        e.add()

    _emit_reference_check(
        e,
        n=plan.n,
        batch_count=1,
        inverse=plan.inverse,
        input_real="input_real",
        input_imag="input_imag",
        output_real="output_real",
        output_imag="output_imag",
        ref_real="ref_real",
        ref_imag="ref_imag",
        tolerance=host.tolerance,
        label="multi-kernel FFT",
    )

    return e.text()
