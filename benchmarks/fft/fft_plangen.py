from __future__ import annotations

"""FFT planning for the M2NDP code generator.

The planner resolves every compile-time decision that affects generated code:

* supported radix selection
* stage ordering
* SIMD batch count and tail lanes
* DRAM vs. scratchpad input source
* scratchpad ping-pong bank selection
* exact input offsets
* exact stage-twiddle vectors
* inverse normalization
* exact output destination and offsets
* scratchpad allocation sizes

Host-side test input and reference values are deliberately *not* decided
here: `fft_codegen.py` emits Mojo that generates its own random input and
computes its own independent reference at runtime (see `HostPlan`), the same
way every other benchmark's `main()` does.

`fft_codegen.py` consumes the resulting plan as a read-only code-generation IR.
It must not reconstruct FFT layout, twiddle, masking, or ping-pong decisions.
"""

from dataclasses import dataclass
from enum import Enum
from math import cos, pi, prod, sin
from typing import Literal, Union

from fft_butterflies import SUPPORTED_RADICES


LoadSource = Literal["input", "scratchpad", "large_twiddle"]
LoadMode = Literal["vector", "scalar_pack"]
StoreDestination = Literal["output", "scratchpad"]
StoreMode = Literal["vector", "scalar_lanes"]


class AddressMappingKind(Enum):
    """How a kernel's DRAM-facing side (its first-stage load or its
    last-stage store) walks memory across the sub-FFT's own N elements.

    CONTIGUOUS: element index i sits at `row*row_stride + i` -- a unit-stride
    vector load/store, one instruction per SIMD batch.

    STRIDED: element index i sits at `row*row_stride + i*elem_stride` with
    `elem_stride != 1` -- there is no vector instruction for that, so codegen
    falls back to one scalar access per lane (the same fallback already used
    for a partial/tail SIMD batch).

    `row` is always `global_uthread_id()`: the logical sub-FFT id. This is
    address generation, not a physical shared-memory transpose -- see
    `AddressMapping`.

    SPLIT: element index i sits at `(row % a) + (row // a) * (a * kernel_length)
    + i * a` -- a middle kernel's output within an M>=3 multi-kernel chain
    (see make_multi_kernel_plan), whose `row` mixes an already-transformed
    digit run (weight < a) with a not-yet-transformed remainder (weight
    >= a). CONTIGUOUS is the a=1 special case of this (the modulo/div both
    vanish) -- kept as its own kind because it needs neither a division
    nor the kernel's own length, and is the only kind M<=2 chains ever use.

    PEELED: a non-last kernel's output, chosen so the *next* kernel's own
    input read becomes `strided(row_stride=next kernel's length,
    elem_stride=1)` -- a real vector load -- instead of today's
    `strided(row_stride=1, elem_stride=n // next kernel's length)`, which
    is always scalar-per-lane. `row` mixes three things: an
    already-transformed run (weight < a, same `a` as SPLIT), the next
    kernel's own digit `d_next` (weight `tail_size`, within the
    not-yet-transformed remainder `row // a`), and everything still after
    that (`rest`, weight >= tail_size). SPLIT keeps the whole
    not-yet-transformed remainder together at the outer (large-stride) end;
    PEELED instead pulls `d_next` out to the innermost slot and pushes
    `rest` outward, so the *next* kernel's read costs it nothing. Element
    index i (this kernel's own new digit) sits between: `(row %
    a)*k_next + i*(a*k_next) + ((row // a) // tail_size) + ((row // a) %
    tail_size)*(a*kernel_length*k_next)`. Derived and verified by direct
    index-permutation simulation (bijective, and every downstream read
    pulls a clean single-digit sweep) up to a 10-kernel chain before this
    touched fft_plangen.py -- same discipline SPLIT's own derivation used.
    This is a read-locality optimization, not a stride bound: the worst
    stride anywhere in the chain is unchanged (conserved, not reduced) --
    it moves from every kernel's read to this kernel's own write, whose
    `i*(a*k_next)` term grows exactly the way SPLIT's old `i*a` did. What
    it does change, provably: every kernel after the first gets a real
    vector read instead of a forced scalar one.

    CROSSED: the last kernel of one *side* of a make_balanced_plan split
    (N = N_A * N_B, each side possibly its own multi-kernel PEELED chain --
    see make_balanced_plan), writing so the *other* side can start its own
    chain fresh (bounded by that side's own size, not by N). `row` mixes a
    batch index (weight 1, range `batch_count` -- which of the other
    side's independent transforms this uthread belongs to) with this
    side's own already-transformed digits (weight `batch_count`, the same
    kind of `a`-scoped quantity SPLIT/PEELED already track, just seeded to
    `batch_count` instead of 1 for this whole side's chain). Element index
    i (this kernel's own new digit) sits at `(row % batch_count) *
    side_length + (row // batch_count) + i * digit_multiplier`: batch
    outermost (so the far side's own first kernel can read this side's
    `side_length` elements contiguously), this side's fully-combined
    digit (`row // batch_count` combined with `i`) innermost. Degenerates
    to plain `contiguous(row_stride=side_length)` when this side is a
    single kernel (`digit_multiplier=1`, `row` already just the batch
    index) -- exactly make_decomposed_plan's own kernel0 formula, checked
    directly as the base case before trusting the multi-kernel
    generalization. Verified by direct index simulation *and* full
    complex-arithmetic comparison against numpy.fft (both sides
    multi-kernel chains, forward and inverse) before this touched
    fft_plangen.py.
    """

    CONTIGUOUS = "contiguous"
    STRIDED = "strided"
    SPLIT = "split"
    PEELED = "peeled"
    CROSSED = "crossed"


@dataclass(frozen=True)
class AddressMapping:
    """addr(row, elem) = base + row*row_stride + elem*elem_stride.

    `row` is this sub-FFT's logical id (`global_uthread_id()`, a runtime
    value -- one multiply). `elem` is the natural in-FFT element index
    (0..length-1), which `_lower_stages` already resolves to a plan-time
    constant per load/store; multiplying it by `elem_stride` stays a
    plan-time constant too. Nothing here is decided by codegen: a plan
    carries one `AddressMapping` for its DRAM input and one for its DRAM
    output, and codegen only ever reads `row_stride`/`elem_stride`/`base`
    off of them.

    A single-kernel FFT's input and output are both
    `AddressMapping.contiguous(length)` -- the plan's `batch_stride` today.
    A decomposed FFT's kernels use `strided(...)` on whichever side isn't
    naturally contiguous, so that the *other* side keeps its vector
    load/store. See `make_decomposed_plan` for how N0*N1 picks these.
    """

    kind: AddressMappingKind
    row_stride: int
    elem_stride: int = 1
    base: int = 0

    # PEELED only -- see AddressMappingKind.PEELED. row_stride is unused
    # (and left 0) for this kind: PEELED's row-dependent term isn't a
    # single multiply, so _mapping_base_expr dispatches on these instead.
    peel_a: int = 0
    peel_k_next: int = 0
    peel_tail_size: int = 0

    @staticmethod
    def contiguous(row_stride: int, base: int = 0) -> "AddressMapping":
        return AddressMapping(AddressMappingKind.CONTIGUOUS, row_stride, 1, base)

    @staticmethod
    def strided(row_stride: int, elem_stride: int, base: int = 0) -> "AddressMapping":
        if elem_stride == 1:
            raise ValueError("a strided mapping needs elem_stride != 1")
        return AddressMapping(AddressMappingKind.STRIDED, row_stride, elem_stride, base)

    @staticmethod
    def split(a: int, base: int = 0) -> "AddressMapping":
        """`a` (this kind's `row_stride` and `elem_stride` both, by
        construction -- see AddressMappingKind.SPLIT) is the accumulated
        product of the kernels processed before this one in the chain.
        `a=1` is valid and degenerates to plain contiguous(kernel_length).
        """
        if a <= 0:
            raise ValueError("a split mapping needs a > 0")
        return AddressMapping(AddressMappingKind.SPLIT, a, a, base)

    @staticmethod
    def peeled(a: int, k_next: int, tail_size: int, base: int = 0) -> "AddressMapping":
        """`a`: accumulated product of kernels before this one (same as
        `split`'s `a`). `k_next`: the *next* kernel's own local length.
        `tail_size`: product of every kernel's length after next (1 if
        there is none -- i.e. next is the last kernel in the chain).
        `a=1` is valid (this kernel is first in the chain).
        """
        if a <= 0:
            raise ValueError("a peeled mapping needs a > 0")
        if k_next <= 0 or tail_size <= 0:
            raise ValueError("a peeled mapping needs k_next > 0 and tail_size > 0")
        return AddressMapping(
            AddressMappingKind.PEELED,
            row_stride=0,
            elem_stride=a * k_next,
            base=base,
            peel_a=a,
            peel_k_next=k_next,
            peel_tail_size=tail_size,
        )

    @staticmethod
    def crossed(
        batch_count: int, side_length: int, digit_multiplier: int, base: int = 0
    ) -> "AddressMapping":
        """`batch_count`: how many independent transforms on the *other*
        side this side's own output is interleaved across. `side_length`:
        this side's own total length (N_A or N_B). `digit_multiplier`: the
        product of this side's own chunk lengths *before* the current
        (last) one -- 1 if this side is a single kernel. Reuses `peel_a`
        for `batch_count` and `row_stride` for `side_length`; see
        AddressMappingKind.CROSSED.
        """
        if batch_count <= 0 or side_length <= 0 or digit_multiplier <= 0:
            raise ValueError(
                "a crossed mapping needs batch_count, side_length, "
                "digit_multiplier all > 0"
            )
        return AddressMapping(
            AddressMappingKind.CROSSED,
            row_stride=side_length,
            elem_stride=digit_multiplier,
            base=base,
            peel_a=batch_count,
        )


@dataclass(frozen=True)
class ScratchpadBufferPlan:
    name: str
    elements: int


@dataclass(frozen=True)
class LoadPlan:
    """One already-resolved complex SIMD input load."""

    operand: int
    source: LoadSource
    buffer_name: str | None
    mode: LoadMode

    # Used by vector mode.
    base_offset: int | None = None

    # Used by scalar_pack mode.  One entry for every SIMD lane.
    # None means that lane is planner-selected padding and must become zero.
    packed_lane_offsets: tuple[int | None, ...] = ()


@dataclass(frozen=True)
class TwiddlePlan:
    """Exact SIMD twiddle constants for one radix output."""

    real: tuple[float, ...]
    imag: tuple[float, ...]


@dataclass(frozen=True)
class StorePlan:
    """One already-resolved complex output store."""

    destination: StoreDestination
    buffer_name: str | None
    mode: StoreMode

    # Used by vector mode.
    base_offset: int | None = None

    # Used by scalar_lanes mode.  One exact destination offset per valid lane.
    lane_offsets: tuple[int, ...] = ()


@dataclass(frozen=True)
class LargeTwiddlePlan:
    """Cross-block twiddle for one factor of an N = N0*N1 decomposition,
    fused into a kernel's final-stage output path (never a separate stage
    or kernel -- see module docstring).

    W_N^(row*output), row = global_uthread_id() (this kernel's logical
    sub-FFT id, a *runtime* value) and output = this stage's plan-time-known
    output digit. That runtime dependency is exactly what makes this
    different from the ordinary per-stage `TwiddlePlan`: an exponent that
    depends on `global_uthread_id()` cannot be folded into a SIMD compile-time
    constant, so it cannot reuse that mechanism.

    twiddle source: a small DRAM table the host precomputes once at startup,
    with std.math cos/sin -- the same "computed at host runtime, not baked
    into the plan" approach `fft_codegen.py` already uses for the reference
    DFT. `full_length` (== the overall FFT's N, not a per-kernel value) is
    the table's own size, laid out to match this kernel's own
    `output_mapping` exactly -- `row_count*output_count == full_length //
    a` distinct values, but the table itself is `full_length` long and the
    fetch reuses the store's own already-computed address as-is (so a
    SPLIT output_mapping's address, which mixes an already-transformed `a`
    with this stage's `output`, reads a value that's the same across every
    `a` -- `a`-fold redundant, not a second independent address
    computation). `a=1` (the default, and every M<=2 chain's only case) is
    the M=2/kernel0 form: row_count*output_count == full_length exactly,
    no redundancy.

    A full twiddle table is the direct approach; it is what's needed here.
    rocFFT/clFFT-style digit-split reconstruction (W_N^u = T0[u0]*T1[u1]*...)
    would trade table size for extra multiplies on very large N -- worth
    adding if a table of size N stops being affordable, not before.

    `output_mapping` may be SPLIT or PEELED (see AddressMappingKind); the
    twiddle *value* (W_N^(row*output), i.e. row_count/output_count/a/
    inverse above) never depends on which one, since that's the same
    underlying math either way -- only the table's own physical layout
    does, since the fetch reuses the store's address as-is. `k_next` and
    `tail_size` (0/1 for SPLIT, matching AddressMapping.peeled's own
    parameters for PEELED) select which address formula fills the table.
    """

    full_length: int
    row_count: int
    output_count: int
    inverse: bool
    a: int = 1
    table_name: str = "large_twiddle"
    # 0 selects the legacy SPLIT table layout; a positive value selects
    # PEELED's (see AddressMapping.peeled -- same two parameters).
    k_next: int = 0
    tail_size: int = 1


@dataclass(frozen=True)
class OutputPlan:
    output: int
    twiddle: TwiddlePlan | None
    scale: float | None
    large_twiddle: bool
    store: StorePlan


@dataclass(frozen=True)
class SIMDBatchPlan:
    batch_id: int
    valid_lanes: int
    loads: tuple[LoadPlan, ...]
    outputs: tuple[OutputPlan, ...]


@dataclass(frozen=True)
class FFTStagePlan:
    stage_id: int
    radix: int
    inverse: bool
    simd_iteration_count: int
    batches: tuple[SIMDBatchPlan, ...]


@dataclass(frozen=True)
class HostPlan:
    """Parameters for the host-generated test: `fft_codegen.py` emits Mojo
    code that generates its own random input and its own reference at
    *runtime* (std.random + a direct O(N^2) DFT, independent of the radix
    decomposition the kernel runs) -- there are no baked-in numbers to plan
    here, only the shape of that runtime computation."""

    total_elems: int
    pool_elems: int
    length: int
    total_uthreads: int
    inverse: bool
    tolerance: float


@dataclass(frozen=True)
class FFTCodegenPlan:
    """Fully lowered plan consumed by fft_codegen.py.

    One `FFTCodegenPlan` is one *kernel*: one `NDPTask` struct, one launch,
    one `local_uthread_id()`-gated scratchpad region. A single-kernel FFT is
    exactly one of these. A decomposed FFT (see `DecomposedFFTPlan`) is two,
    chained through DRAM the way `two_tasks.mojo` chains `Scale`/`AddB` --
    never through a shared scratchpad, since nothing is guaranteed still
    resident once a kernel launch returns.

    `total_uthreads` and `max_uthread` are two different quantities that
    the field name `max_uthread` alone used to carry, ambiguously:

    * `total_uthreads` -- how many microthreads this kernel's *launch*
      covers in total (`PooledRange.over(pool, simd_lanes*total_uthreads)`),
      spread across however many NDP units the hardware maps them onto.
    * `max_uthread` -- how many of *one NDP unit's own* microthreads
      `local_uthread_id()` ever ranges over (docs/PRIMITIVES.md's
      `spad_capacity()`, "bytes of scratchpad on one unit", divided by
      what one uthread's own scratchpad footprint costs -- see
      `_build_plan`'s `spad_capacity_bytes`). This is what
      `ScratchpadBufferPlan.elements` and the `MAX_UTHREAD_<kernel>`
      device-side guard are sized against, since the scratchpad is one
      instance *per core*, shared by every uthread that lands on it --
      not one slot per uthread in the whole launch.

    `max_uthread <= total_uthreads` always; they're equal (today's
    behavior, unchanged) whenever `_build_plan` isn't given a
    `spad_capacity_bytes` to size against, or the kernel uses no
    scratchpad at all (a single-stage kernel -- see
    `_scratchpad_buffer_names`), since then there's nothing to divide
    across units.
    """

    length: int
    inverse: bool
    total_uthreads: int
    max_uthread: int
    simd_lanes: int
    kernel_name: str

    # DRAM-facing address mappings -- see `AddressMapping`. A single-kernel
    # FFT's are both `contiguous(length)`, which is today's `batch_stride`
    # generalized: row_stride==length, elem_stride==1 on both sides.
    input_mapping: AddressMapping
    output_mapping: AddressMapping
    large_twiddle: LargeTwiddlePlan | None

    scratchpad_uthread_stride: int

    scratchpad_buffers: tuple[ScratchpadBufferPlan, ...]
    stages: tuple[FFTStagePlan, ...]
    host: HostPlan


@dataclass(frozen=True)
class _StageLayout:
    """Planner-internal stage description.

    These fields are intentionally not exposed to fft_codegen.py.  The planner
    lowers them into explicit load/twiddle/store operations first.
    """

    radix: int
    butterfly_count: int
    input_batch_width: int
    input_stride: int
    output_batch_width: int
    output_stride: int
    twiddle_modulus: int | None = None
    twiddle_stride: int = 1
    twiddle_lane_divisor: int = 1


def _check_layouts(
    *,
    length: int,
    total_uthreads: int,
    simd_lanes: int,
    layouts: tuple[_StageLayout, ...],
) -> None:
    if length <= 0:
        raise ValueError("FFT length must be positive")
    if total_uthreads <= 0:
        raise ValueError("total_uthreads must be positive")
    if simd_lanes <= 0:
        raise ValueError("simd_lanes must be positive")
    if not layouts:
        raise ValueError("at least one FFT stage is required")

    product = 1
    for sid, stage in enumerate(layouts):
        if stage.radix not in SUPPORTED_RADICES:
            supported = ", ".join(str(r) for r in sorted(SUPPORTED_RADICES))
            raise ValueError(
                f"stage {sid}: radix-{stage.radix} is unsupported; "
                f"supported radices are {{{supported}}}"
            )
        if stage.butterfly_count <= 0:
            raise ValueError(f"stage {sid}: butterfly_count must be positive")
        if stage.input_batch_width <= 0 or stage.output_batch_width <= 0:
            raise ValueError(f"stage {sid}: batch width must be positive")
        if stage.input_stride <= 0 or stage.output_stride <= 0:
            raise ValueError(f"stage {sid}: stride must be positive")
        if stage.twiddle_modulus is not None and stage.twiddle_modulus <= 0:
            raise ValueError(f"stage {sid}: twiddle_modulus must be positive")
        if stage.twiddle_lane_divisor <= 0:
            raise ValueError(f"stage {sid}: twiddle_lane_divisor must be positive")
        product *= stage.radix

    if product != length:
        raise ValueError(
            f"product of stage radices ({product}) must equal FFT length ({length})"
        )


def _scratchpad_buffer_names(
    *,
    stage_count: int,
    use_pingpong: bool,
) -> tuple[str, ...]:
    # No intermediate result exists for a one-stage FFT.
    if stage_count <= 1:
        return ()
    if use_pingpong:
        return ("buf_a", "buf_b")
    return ("buf",)


def _write_buffer_for_stage(
    *,
    stage_id: int,
    stage_count: int,
    buffer_names: tuple[str, ...],
) -> str | None:
    if stage_id == stage_count - 1:
        return None
    if not buffer_names:
        raise ValueError("intermediate stage requires scratchpad storage")
    if len(buffer_names) == 1:
        return buffer_names[0]
    return buffer_names[stage_id % len(buffer_names)]


def _read_buffer_for_stage(
    *,
    stage_id: int,
    stage_count: int,
    buffer_names: tuple[str, ...],
) -> str | None:
    if stage_id == 0:
        return None
    return _write_buffer_for_stage(
        stage_id=stage_id - 1,
        stage_count=stage_count,
        buffer_names=buffer_names,
    )


def _make_load(
    *,
    operand: int,
    first_stage: bool,
    read_buffer: str | None,
    base_offset: int,
    valid_lanes: int,
    simd_lanes: int,
    dram_elem_stride: int = 1,
) -> LoadPlan:
    source: LoadSource = "input" if first_stage else "scratchpad"
    buffer_name = None if first_stage else read_buffer

    if source == "scratchpad" and buffer_name is None:
        raise ValueError("scratchpad load requires a resolved buffer name")

    # A non-unit DRAM element stride (this kernel's input_mapping is
    # STRIDED -- see AddressMapping) has no vector-load form: fall back to
    # one scalar load per lane, same fallback a partial/tail SIMD batch
    # already uses. Only the DRAM ("input") side can be strided this way;
    # scratchpad loads are always this kernel's own contiguous layout.
    force_scalar = source == "input" and dram_elem_stride != 1

    if valid_lanes == simd_lanes and not force_scalar:
        return LoadPlan(
            operand=operand,
            source=source,
            buffer_name=buffer_name,
            mode="vector",
            base_offset=base_offset,
        )

    stride = dram_elem_stride if source == "input" else 1
    return LoadPlan(
        operand=operand,
        source=source,
        buffer_name=buffer_name,
        mode="scalar_pack",
        packed_lane_offsets=tuple(
            ((base_offset + lane) * stride) if lane < valid_lanes else None
            for lane in range(simd_lanes)
        ),
    )


def _make_twiddle(
    *,
    inverse: bool,
    simd_lanes: int,
    valid_lanes: int,
    first_bfly: int,
    output: int,
    modulus: int | None,
    stride: int,
    lane_divisor: int,
) -> TwiddlePlan | None:
    if modulus is None or output == 0:
        return None

    sign = 1.0 if inverse else -1.0
    wr: list[float] = []
    wi: list[float] = []
    for lane in range(simd_lanes):
        if lane >= valid_lanes:
            wr.append(1.0)
            wi.append(0.0)
            continue

        bfly = first_bfly + lane
        twiddle_index = bfly // lane_divisor
        exponent = twiddle_index * output * stride
        angle = sign * 2.0 * pi * exponent / modulus
        wr.append(cos(angle))
        wi.append(sin(angle))

    return TwiddlePlan(real=tuple(wr), imag=tuple(wi))


def _make_store(
    *,
    last_stage: bool,
    write_buffer: str | None,
    layout: _StageLayout,
    simd_it: int,
    output: int,
    valid_lanes: int,
    simd_lanes: int,
    dram_elem_stride: int = 1,
) -> StorePlan:
    output_batch_base = simd_it * layout.output_batch_width

    if last_stage:
        base = (output_batch_base + output * layout.output_stride) * dram_elem_stride
        # Symmetric with _make_load: a non-unit DRAM element stride (this
        # kernel's output_mapping is STRIDED) has no vector-store form.
        if valid_lanes == simd_lanes and dram_elem_stride == 1:
            return StorePlan(
                destination="output",
                buffer_name=None,
                mode="vector",
                base_offset=base,
            )
        return StorePlan(
            destination="output",
            buffer_name=None,
            mode="scalar_lanes",
            lane_offsets=tuple(
                base + lane * dram_elem_stride for lane in range(valid_lanes)
            ),
        )

    if write_buffer is None:
        raise ValueError("intermediate stage requires a resolved write buffer")

    # One formula for every non-last (Stockham-autosort) stage. P_s --
    # this stage's twiddle_lane_divisor, the product of the radices before
    # it -- picks both the twiddle exponent (in _make_twiddle) and this
    # store permutation, so the next stage's read is always the same
    # "N_local/radix contiguous blocks" shape (see _make_load's
    # input_stride) regardless of which stage wrote it or what radix it
    # was. Was two hand-authored special cases (one per stage position);
    # collapsed after confirming both were this same formula evaluated at
    # P_s=1 and P_s=radix respectively.
    p_s = layout.twiddle_lane_divisor
    group_stride = layout.radix * p_s
    offsets = []
    for lane in range(valid_lanes):
        bfly = simd_it * simd_lanes + lane
        n2 = bfly // p_s
        b_s = bfly % p_s
        offsets.append(n2 * group_stride + output * p_s + b_s)
    return StorePlan(
        destination="scratchpad",
        buffer_name=write_buffer,
        mode="scalar_lanes",
        lane_offsets=tuple(offsets),
    )


def _lower_stages(
    *,
    length: int,
    inverse: bool,
    simd_lanes: int,
    layouts: tuple[_StageLayout, ...],
    buffer_names: tuple[str, ...],
    input_mapping: AddressMapping,
    output_mapping: AddressMapping,
    large_twiddle: LargeTwiddlePlan | None,
    inverse_scale: float | None,
) -> tuple[FFTStagePlan, ...]:
    stages: list[FFTStagePlan] = []
    stage_count = len(layouts)

    for stage_id, layout in enumerate(layouts):
        first_stage = stage_id == 0
        last_stage = stage_id == stage_count - 1
        read_buffer = _read_buffer_for_stage(
            stage_id=stage_id,
            stage_count=stage_count,
            buffer_names=buffer_names,
        )
        write_buffer = _write_buffer_for_stage(
            stage_id=stage_id,
            stage_count=stage_count,
            buffer_names=buffer_names,
        )

        simd_iters = (layout.butterfly_count + simd_lanes - 1) // simd_lanes
        batches: list[SIMDBatchPlan] = []

        for simd_it in range(simd_iters):
            first_bfly = simd_it * simd_lanes
            valid_lanes = min(simd_lanes, layout.butterfly_count - first_bfly)
            input_batch_base = simd_it * layout.input_batch_width

            loads = tuple(
                _make_load(
                    operand=operand,
                    first_stage=first_stage,
                    read_buffer=read_buffer,
                    base_offset=input_batch_base + operand * layout.input_stride,
                    valid_lanes=valid_lanes,
                    simd_lanes=simd_lanes,
                    dram_elem_stride=(
                        input_mapping.elem_stride if first_stage else 1
                    ),
                )
                for operand in range(layout.radix)
            )

            outputs: list[OutputPlan] = []
            for output in range(layout.radix):
                outputs.append(
                    OutputPlan(
                        output=output,
                        twiddle=_make_twiddle(
                            inverse=inverse,
                            simd_lanes=simd_lanes,
                            valid_lanes=valid_lanes,
                            first_bfly=first_bfly,
                            output=output,
                            modulus=layout.twiddle_modulus,
                            stride=layout.twiddle_stride,
                            lane_divisor=layout.twiddle_lane_divisor,
                        ),
                        scale=inverse_scale if last_stage else None,
                        large_twiddle=last_stage and large_twiddle is not None,
                        store=_make_store(
                            last_stage=last_stage,
                            write_buffer=write_buffer,
                            layout=layout,
                            simd_it=simd_it,
                            output=output,
                            valid_lanes=valid_lanes,
                            simd_lanes=simd_lanes,
                            dram_elem_stride=(
                                output_mapping.elem_stride if last_stage else 1
                            ),
                        ),
                    )
                )

            batches.append(
                SIMDBatchPlan(
                    batch_id=simd_it,
                    valid_lanes=valid_lanes,
                    loads=loads,
                    outputs=tuple(outputs),
                )
            )

        stages.append(
            FFTStagePlan(
                stage_id=stage_id,
                radix=layout.radix,
                inverse=inverse,
                simd_iteration_count=simd_iters,
                batches=tuple(batches),
            )
        )

    return tuple(stages)


def _make_host_plan(
    *,
    length: int,
    inverse: bool,
    total_uthreads: int,
    simd_lanes: int,
) -> HostPlan:
    """No numbers are baked in here: `fft_codegen.py` emits Mojo that
    generates its own random input (`std.random`, seeded, matching every
    other benchmark's `main()`) and its own reference (a direct O(N^2) DFT
    computed at host runtime, independent of the radix decomposition the
    kernel runs) each time the program is launched, rather than a fixed
    signal computed once by this planner and frozen into the source. See
    `fft_codegen.py`'s host-emission section for that computation."""
    return HostPlan(
        total_elems=length * total_uthreads,
        pool_elems=simd_lanes * total_uthreads,
        length=length,
        total_uthreads=total_uthreads,
        inverse=inverse,
        tolerance=1.0e-3,
    )


class _Default:
    """Sentinel distinguishing 'caller didn't say' from 'caller explicitly
    said None' -- see `_build_plan`'s `inverse_scale`."""


_DEFAULT = _Default()


def _build_plan(
    *,
    length: int,
    inverse: bool,
    total_uthreads: int,
    simd_lanes: int,
    use_pingpong: bool,
    layouts: tuple[_StageLayout, ...],
    kernel_name: str = "FFTFP32",
    input_mapping: AddressMapping | None = None,
    output_mapping: AddressMapping | None = None,
    large_twiddle: LargeTwiddlePlan | None = None,
    inverse_scale: float | None | _Default = _DEFAULT,
    spad_capacity_bytes: int | None = None,
) -> FFTCodegenPlan:
    """`spad_capacity_bytes` is one NDP unit's own scratchpad size (see
    `FFTCodegenPlan`'s docstring) -- optional and `None` by default, which
    keeps today's behavior exactly (`max_uthread == total_uthreads`, i.e.
    assume the whole launch could land on one unit, the always-safe but
    possibly oversized choice `elements = scratchpad_stride * total_uthreads`
    already made). Given a real budget, `max_uthread` is capped to however
    many of *this* kernel's own uthreads (each needing
    `len(buffer_names) * scratchpad_stride * 4` bytes for its ping-pong
    footprint) fit in it -- moot for a single-stage kernel, which uses no
    scratchpad at all (see `_scratchpad_buffer_names`).
    """
    _check_layouts(
        length=length,
        total_uthreads=total_uthreads,
        simd_lanes=simd_lanes,
        layouts=layouts,
    )

    # Default: today's single-kernel behavior -- one contiguous row per
    # sub-FFT, `row_stride == length`, unchanged from the old hardcoded
    # `batch_stride`.
    if input_mapping is None:
        input_mapping = AddressMapping.contiguous(length)
    if output_mapping is None:
        output_mapping = AddressMapping.contiguous(length)
    if inverse_scale is _DEFAULT:
        inverse_scale = (1.0 / length) if inverse else None

    buffer_names = _scratchpad_buffer_names(
        stage_count=len(layouts),
        use_pingpong=use_pingpong,
    )
    scratchpad_stride = 2 * length

    if not buffer_names or spad_capacity_bytes is None:
        max_uthread = total_uthreads
    else:
        bytes_per_uthread = len(buffer_names) * scratchpad_stride * 4
        if bytes_per_uthread > spad_capacity_bytes:
            raise ValueError(
                f"spad_capacity_bytes={spad_capacity_bytes} is too small to "
                f"fit even a single uthread of kernel {kernel_name!r} "
                f"(needs {bytes_per_uthread} bytes)"
            )
        max_uthread = min(total_uthreads, spad_capacity_bytes // bytes_per_uthread)

    scratchpad_buffers = tuple(
        ScratchpadBufferPlan(
            name=name,
            elements=scratchpad_stride * max_uthread,
        )
        for name in buffer_names
    )

    return FFTCodegenPlan(
        length=length,
        inverse=inverse,
        total_uthreads=total_uthreads,
        max_uthread=max_uthread,
        simd_lanes=simd_lanes,
        kernel_name=kernel_name,
        input_mapping=input_mapping,
        output_mapping=output_mapping,
        large_twiddle=large_twiddle,
        scratchpad_uthread_stride=scratchpad_stride,
        scratchpad_buffers=scratchpad_buffers,
        stages=_lower_stages(
            length=length,
            inverse=inverse,
            simd_lanes=simd_lanes,
            layouts=layouts,
            buffer_names=buffer_names,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            large_twiddle=large_twiddle,
            inverse_scale=inverse_scale,
        ),
        host=_make_host_plan(
            length=length,
            inverse=inverse,
            total_uthreads=total_uthreads,
            simd_lanes=simd_lanes,
        ),
    )


def layouts_for_radices(
    length: int, radices: tuple[int, ...], simd_lanes: int
) -> tuple[_StageLayout, ...]:
    """Every stage's _StageLayout for a length-`length` FFT run as one
    kernel's `radices` sequence (in Cooley-Tukey/Stockham order: `length`
    == product(radices)), derived from nothing but the radix sequence
    itself -- no per-stage hand authoring.

    `P_s` (this stage's `twiddle_lane_divisor`) is the product of the
    radices before it; both `_make_twiddle`'s exponent and `_make_store`'s
    intermediate-stage permutation key off it (see `_make_store`), which
    is what makes every non-last stage's read of its input always the same
    "length/radix contiguous blocks" shape regardless of which stage wrote
    it (`input_stride = length // radix` here, on every stage).

    `input_batch_width`/`output_batch_width` are pinned to `simd_lanes`:
    they only matter (`simd_it * width`) once a stage has more than one
    SIMD batch, and at that point the load/store math requires the batch
    stride to be exactly the lane count.
    """
    stage_count = len(radices)
    layouts: list[_StageLayout] = []
    p = 1
    for stage_id, radix in enumerate(radices):
        last_stage = stage_id == stage_count - 1
        layouts.append(
            _StageLayout(
                radix=radix,
                butterfly_count=length // radix,
                input_batch_width=simd_lanes,
                input_stride=length // radix,
                output_batch_width=simd_lanes,
                output_stride=p if last_stage else 1,
                twiddle_modulus=None if last_stage else length // p,
                twiddle_stride=1,
                # Unused (and not necessarily a divisor of simd_lanes) on
                # the last stage: there is no twiddle there (modulus=None
                # above short-circuits _make_twiddle), and _make_store's
                # last-stage branch doesn't read this field either.
                twiddle_lane_divisor=1 if last_stage else p,
            )
        )
        p *= radix
    return tuple(layouts)


def make_444_plan(
    *, inverse: bool = False, total_uthreads: int = 1, simd_lanes: int = 8
) -> FFTCodegenPlan:
    """Create the fully lowered N=64, radix-4 x radix-4 x radix-4 plan."""

    return _build_plan(
        length=64,
        inverse=inverse,
        total_uthreads=total_uthreads,
        simd_lanes=simd_lanes,
        use_pingpong=True,
        layouts=layouts_for_radices(64, (4, 4, 4), simd_lanes=simd_lanes),
    )


# --------------------------------------------------------------- decomposition
#
# N too large for one uthread's scratchpad: N = N0*N1, run as two kernels
# chained through DRAM. See the module docstring's index-mapping walkthrough
# for the derivation; the short version is:
#
#   kernel0 (N0 uthreads, row=n0): strided load x[n1*N0+n0], forward N1-point
#     FFT over n1, multiply by W_N^(n0*k1), contiguous store mid[n0*N1+k1]
#   kernel1 (N1 uthreads, row=k1): strided load mid[n0*N1+k1] (the actual
#     transpose read of kernel0's contiguous-by-n0 store), forward N0-point
#     FFT over n0, strided store out[k0*N1+k1]
#
# Both kernels are ordinary single-stage FFTCodegenPlans (radix == their own
# length, `layouts_for_radices(radix, (radix,), simd_lanes)` already covers
# this); only their AddressMapping and (kernel0's) LargeTwiddlePlan differ
# from a single-kernel plan's defaults.


@dataclass(frozen=True)
class DecomposedHostPlan:
    """Host-side shape for the two-kernel main(). Same philosophy as
    HostPlan: fresh random input and an independent O(N^2) DFT reference
    (over the *full* N, not the N0/N1 decomposition either kernel runs) are
    generated/computed at Mojo host runtime -- nothing numeric is baked in
    here, only sizes."""

    n: int
    n0: int
    n1: int
    inverse: bool
    tolerance: float


@dataclass(frozen=True)
class DecomposedFFTPlan:
    """N = N0*N1, run as two kernels chained through DRAM -- never through a
    shared scratchpad, since nothing survives a kernel launch boundary but
    what was written to DRAM (see module docstring). Exactly one independent
    length-N FFT (no cross-FFT batching yet: `AddressMapping` is one runtime
    multiply, and batching would need a second one to fold the batch index
    in without falling back to runtime division -- see make_decomposed_plan).
    """

    n: int
    n0: int
    n1: int
    inverse: bool
    kernel0: FFTCodegenPlan
    kernel1: FFTCodegenPlan
    host: DecomposedHostPlan


def make_decomposed_plan(
    n0: int, n1: int, *, inverse: bool = False, simd_lanes: int = 8
) -> DecomposedFFTPlan:
    """N = n0*n1, planned as kernel0 (N0 uthreads, N1-point sub-FFTs, large
    twiddle fused into its output store) then kernel1 (N1 uthreads, N0-point
    sub-FFTs), chained through DRAM. Every decomposition decision -- which
    factor each kernel owns, each kernel's DRAM AddressMapping, the large
    twiddle's table shape, where the final 1/N inverse scale lands -- is
    made here; fft_codegen.py only renders what this plan already decided.

    One independent length-N FFT per call (see DecomposedFFTPlan); pass N0
    and N1 in SUPPORTED_RADICES (each becomes one kernel's single-stage
    radix -- see layouts_for_radices).
    """
    n = n0 * n1

    kernel0 = _build_plan(
        length=n1,
        inverse=inverse,
        total_uthreads=n0,
        simd_lanes=simd_lanes,
        use_pingpong=False,
        layouts=layouts_for_radices(n1, (n1,), simd_lanes),
        kernel_name="FFTFP32Kernel0",
        # x[n1*N0 + n0]: this uthread's row is n0 (row_stride=1), its N1
        # elements are spaced N0 apart in the original contiguous input.
        input_mapping=AddressMapping.strided(row_stride=1, elem_stride=n0),
        # mid[n0*N1 + k1]: contiguous per row -- this uthread's own N1
        # outputs land next to each other. kernel1 (row=k1) reads this same
        # buffer *transposed*: fixed k1, n0 stepping by N1 -- see kernel1's
        # input_mapping below, which is strided, not contiguous.
        output_mapping=AddressMapping.contiguous(row_stride=n1),
        large_twiddle=LargeTwiddlePlan(
            full_length=n, row_count=n0, output_count=n1, inverse=inverse
        ),
        # The intra-kernel butterfly must NOT apply 1/N here: this is not
        # the final kernel of the decomposition. The overall inverse scale
        # (1/n, not 1/n1) lands once, at kernel1.
        inverse_scale=None,
    )

    kernel1 = _build_plan(
        length=n0,
        inverse=inverse,
        total_uthreads=n1,
        simd_lanes=simd_lanes,
        use_pingpong=False,
        layouts=layouts_for_radices(n0, (n0,), simd_lanes),
        kernel_name="FFTFP32Kernel1",
        # mid[n0*N1 + k1]: kernel0 wrote this contiguously by *its* row n0
        # (output_mapping above). Read back by k1 instead, each of this
        # uthread's N0 elements (n0 = 0..N0-1) sits N1 apart in that same
        # layout -- this is the actual transpose read, and it is strided,
        # not contiguous (an earlier version of this function wrongly
        # assumed mid[k1*N0+n0], which is a different set of elements
        # entirely except where n0 happens to equal k1).
        input_mapping=AddressMapping.strided(row_stride=1, elem_stride=n1),
        # out[k0*N1 + k1]: k1 (this row) is the *fast* digit of the true
        # output index, so scattering across k0 (what this kernel produces)
        # is inherently strided by N1 -- see module docstring; no kernel
        # split avoids this for the kernel that owns k1.
        output_mapping=AddressMapping.strided(row_stride=1, elem_stride=n1),
        large_twiddle=None,
        inverse_scale=(1.0 / n) if inverse else None,
    )

    return DecomposedFFTPlan(
        n=n,
        n0=n0,
        n1=n1,
        inverse=inverse,
        kernel0=kernel0,
        kernel1=kernel1,
        host=DecomposedHostPlan(
            n=n, n0=n0, n1=n1, inverse=inverse, tolerance=1.0e-3
        ),
    )


# ------------------------------------------------------- M-kernel chaining
#
# Generalizes DecomposedFFTPlan (exactly 2 kernels, each one bare radix)
# to a chain of M>=1 kernels, each itself a layouts_for_radices multi-stage
# FFT via scratchpad ping-pong -- absorbing as many radix stages as fit in
# one uthread's scratchpad budget per kernel, minimizing DRAM handoffs
# instead of the two extremes make_444_plan/make_decomposed_plan cover
# today. M=1 and M=2 are exact generalizations of those two, verified
# equivalent (see verify_fft_plan.py). M>=3 needs LargeTwiddlePlan's
# exponent generalized for a kernel whose own uthread id mixes an
# already-transformed digit with not-yet-transformed ones -- not yet
# derived/verified, see the plan doc's Stage 4.


def _prime_factors_supported(n: int) -> list[int]:
    """Factor n into primes SUPPORTED_RADICES covers. Composite-radix
    coalescing (e.g. four radix-2 stages -> one radix-16) is deferred --
    see plan Stage 6 -- so only the prime subset of SUPPORTED_RADICES is
    used here, even though composites like 16 are themselves valid radices
    elsewhere in this module.
    """
    primes = sorted(r for r in SUPPORTED_RADICES if all(r % d for d in range(2, r)))
    factors: list[int] = []
    remaining = n
    for p in primes:
        while remaining % p == 0:
            factors.append(p)
            remaining //= p
    if remaining != 1:
        raise ValueError(
            f"{n} has a prime factor outside the primes SUPPORTED_RADICES "
            f"covers ({primes}); cannot factor for kernel chunking"
        )
    return factors


def factor_into_kernel_chunks(
    n: int, *, scratchpad_byte_budget: int
) -> tuple[tuple[int, ...], ...]:
    """Factor n into a sequence of per-kernel radix chunks -- chunk i
    becomes kernel i's radix sequence for layouts_for_radices, processed
    in order and chained through DRAM by make_multi_kernel_plan.

    Built greedily under one cap: `16 * chunk_product <= scratchpad_byte_budget`
    (the `scratchpad_uthread_stride = 2*length` times 2 ping-pong buffers
    times 4 bytes/float convention `_build_plan` already uses). No default
    is offered: the real M2NDP per-uthread scratchpad size isn't
    established in this codebase, so callers pick a value.

    An earlier version of this also capped chunk depth/ordering to satisfy
    what looked like a real constraint in `_check_layouts` (every non-last
    *stage*'s cumulative radix product dividing simd_lanes) -- verified at
    the time by bypassing the check and watching results go to ~O(1)
    wrong. That verification was itself standing on a bug in the
    verification harness (numpy aliasing on `var or0 = rr0`-style copies --
    see verify_fft_plan.py's SimdVec), and re-run after fixing it, every
    previously-failing case (mixed radix, depth-8 same-radix towers, etc.)
    passes cleanly. The constraint was removed from `_check_layouts`
    accordingly, and chunk order/composition here is unconstrained beyond
    the scratchpad budget.

    Chosen to minimize the largest *effective DRAM cost* any kernel in the
    chain pays. Every kernel in an AddressMappingKind.PEELED chain (see
    make_multi_kernel_plan) pays one of two costs, and a chunking choice
    affects each differently: kernel0's *read* costs `n // (its own
    length)` -- unavoidable, nothing precedes it to fuse a layout reset
    into -- and every *non-last* kernel's *write* costs `a * (the *next*
    kernel's own length)` (`a` = the product of every earlier kernel's
    length; the last kernel's write is the exception, unchanged at plain
    `a`, since there's no next kernel to optimize for). Every kernel after
    the first reads at cost 1 (contiguous, PEELED's whole point), so only
    these two costs -- kernel0's read, and each non-last kernel's write --
    ever matter.

    A chunk's write cost depending on the *next* chunk's length (not just
    its own start) breaks naive optimal substructure: a DP keyed only on
    "best cost for factors[i:]" can't supply what a chunk ending at `i`
    needs (the length of *its own* immediately-following chunk) without
    also depending on how that suffix chooses to split, and that suffix's
    own optimum doesn't necessarily supply the length this chunk wants
    (checked directly, not assumed: an earlier, simpler version of this
    keyed only on (start) and produced a *worse* result at a *larger*
    budget on this same n=960 sweep -- impossible for a correct DP, since
    a larger budget can only add valid choices, never remove one).

    Fixed by keying state on the pair (chunk start, chunk end) instead:
    state `(start, j)` means "chunk [start:j) is chosen, but its write
    cost isn't finalized yet" -- exactly true until a *following* chunk
    [j:j2) is also chosen, at which point [start:j)'s write cost
    (`prefix_product[start] * length([j:j2))`) becomes computable and
    folds into the running max carried forward as the new state (j, j2).
    `O(F^2)` states, `O(F)` transitions each -- `O(F^3)`, still cheap for
    the factor count `F` (at most ~20 for any n this codebase addresses)
    -- and exact for this cost function, not a heuristic: every reachable
    (start, j) keeps only its minimum cost, so no choice that could affect
    the final answer is dropped. Reproduces the plan doc's hand-derived
    N=960 numbers (max cost 120 at scratchpad_byte_budget=256) and is
    monotonically non-increasing in budget, checked directly.
    """
    if scratchpad_byte_budget <= 0:
        raise ValueError("scratchpad_byte_budget must be positive")

    factors = _prime_factors_supported(n)
    cap = scratchpad_byte_budget // 16
    num_factors = len(factors)

    prefix_product = [1] * (num_factors + 1)
    for i, f in enumerate(factors):
        prefix_product[i + 1] = prefix_product[i] * f

    # Under AddressMappingKind.PEELED (see make_multi_kernel_plan), a
    # non-first kernel's read is always 1 (contiguous) -- only kernel0
    # pays n // its own length. A non-last kernel's write is
    # prefix_product[start] * (the *next* kernel's own length), not just
    # prefix_product[start] -- so a chunk [start:j]'s write cost isn't
    # knowable until the chunk *after* it is also chosen. That rules out
    # a single-value-per-position DP (the greedy choice that's locally
    # best for factors[j:] on its own doesn't necessarily supply the
    # `next kernel's own length` that minimizes chunk [start:j]'s write
    # cost -- optimal substructure genuinely fails for that formulation,
    # confirmed by hitting it directly: an earlier version keyed only on
    # a chunk's own (start, j) and it produced a *worse* result at a
    # *larger* budget for this same n=960 sweep, which is impossible for
    # a correct DP since a larger budget only adds valid choices).
    #
    # Fixed by keying state on the *pair* (start, j): "chunk [start:j] has
    # been chosen but its write cost isn't finalized yet -- that happens
    # the moment a following chunk [j:j2] is also chosen, which is also
    # exactly when [start:j]'s write cost becomes computable
    # (prefix_product[start] * (j2's chunk length)) and gets folded into
    # the running max carried forward as state (j, j2)." O(F^2) states,
    # O(F) transitions each = O(F^3) -- still cheap for F<=~20 -- and this
    # one is exact, not a heuristic: (start, j) is reached via `best[...]
    # = (cost, predecessor_start)`, always keeping the minimum cost seen
    # for that exact state, so every choice that could affect the final
    # answer is considered.
    #
    # Verified: this reproduces the plan doc's hand-derived N=960 numbers
    # (max cost 120 at scratchpad_byte_budget=256) and is monotonically
    # non-increasing in budget, checked directly across the same sweep
    # the removed version broke.
    best: dict[tuple[int, int], tuple[int, int | None]] = {}
    for j in range(1, num_factors + 1):
        chunk_length = prefix_product[j]  # prefix_product[0] == 1
        if chunk_length > cap:
            break
        best[(0, j)] = (n // chunk_length, None)

    for j in range(1, num_factors + 1):
        for start in range(j):
            state = best.get((start, j))
            if state is None:
                continue
            running_cost, _ = state
            for j2 in range(j + 1, num_factors + 1):
                next_length = prefix_product[j2] // prefix_product[j]
                if next_length > cap:
                    break
                write_cost = prefix_product[start] * next_length
                candidate = max(running_cost, write_cost)
                key = (j, j2)
                existing = best.get(key)
                if existing is None or candidate < existing[0]:
                    best[key] = (candidate, start)

    final: tuple[int, int] | None = None  # (total cost, last chunk's start)
    for start in range(num_factors):
        state = best.get((start, num_factors))
        if state is None:
            continue
        running_cost, _ = state
        total = max(running_cost, prefix_product[start])  # last kernel: write=a, no next-length multiplier
        if final is None or total < final[0]:
            final = (total, start)
    if final is None:
        raise ValueError(
            f"scratchpad_byte_budget={scratchpad_byte_budget} is too small "
            f"to fit n={n} into any valid chunk sequence"
        )

    boundaries = [num_factors, final[1]]
    j, start = num_factors, final[1]
    while start != 0:
        _, pred = best[(start, j)]
        assert pred is not None
        boundaries.append(pred)
        j, start = start, pred
    boundaries.reverse()

    return tuple(
        tuple(factors[boundaries[k] : boundaries[k + 1]])
        for k in range(len(boundaries) - 1)
    )


def _choose_side_chunks(
    factors: list[int], *, batch_count: int, scratchpad_byte_budget: int
) -> tuple[tuple[int, ...], ...]:
    """factor_into_kernel_chunks's own DP, generalized for one side of a
    make_balanced_plan split: `batch_count` (the *other* side's size)
    multiplies this side's kernel0 read cost and every internal (non-
    last) kernel's write-seed `a`, exactly the way AddressMappingKind
    .CROSSED's own docstring describes -- but *not* the terminal (CROSSED)
    kernel's own write cost, which is `digit_multiplier` alone (this
    side's own accumulated product, with the batch factor already divided
    back out by construction -- see AddressMapping.crossed).

    Necessary, not optional: naively feeding a plain
    factor_into_kernel_chunks(side_length, budget) result into
    make_balanced_plan's `_build_batched_side` measurably fails to reach
    anywhere near sqrt(N) whenever this side needs more than one internal
    kernel -- checked directly, not assumed (N=960 split into two
    factor_into_kernel_chunks(960, budget=4096)-chosen sides gave
    max_effective_stride=480, no better than one flat 8-kernel chain over
    the same N; the batch-aware version below gets both sides down near
    sqrt(N) instead). The reason: kernel0's read is `(side_length //
    its_own_length) * batch_count`, and `factor_into_kernel_chunks` alone
    has no idea `batch_count` is about to multiply whatever it picks.
    """
    if scratchpad_byte_budget <= 0:
        raise ValueError("scratchpad_byte_budget must be positive")
    if batch_count <= 0:
        raise ValueError("batch_count must be positive")

    cap = scratchpad_byte_budget // 16
    num_factors = len(factors)
    prefix_product = [1] * (num_factors + 1)
    for i, f in enumerate(factors):
        prefix_product[i + 1] = prefix_product[i] * f
    side_length = prefix_product[num_factors]

    best: dict[tuple[int, int], tuple[int, int | None]] = {}
    for j in range(1, num_factors + 1):
        chunk_length = prefix_product[j]
        if chunk_length > cap:
            break
        read_cost = (side_length // chunk_length) * batch_count
        best[(0, j)] = (read_cost, None)

    for j in range(1, num_factors + 1):
        for start in range(j):
            state = best.get((start, j))
            if state is None:
                continue
            running_cost, _ = state
            for j2 in range(j + 1, num_factors + 1):
                next_length = prefix_product[j2] // prefix_product[j]
                if next_length > cap:
                    break
                write_cost = (batch_count * prefix_product[start]) * next_length
                candidate = max(running_cost, write_cost)
                key = (j, j2)
                existing = best.get(key)
                if existing is None or candidate < existing[0]:
                    best[key] = (candidate, start)

    final: tuple[int, int] | None = None
    for start in range(num_factors):
        state = best.get((start, num_factors))
        if state is None:
            continue
        running_cost, _ = state
        # Terminal (CROSSED) kernel: write = digit_multiplier = this
        # side's own accumulated product, no batch_count multiplier.
        total = max(running_cost, prefix_product[start])
        if final is None or total < final[0]:
            final = (total, start)
    if final is None:
        raise ValueError(
            f"scratchpad_byte_budget={scratchpad_byte_budget} is too small "
            f"to fit this side into any valid chunk sequence"
        )

    boundaries = [num_factors, final[1]]
    j, start = num_factors, final[1]
    while start != 0:
        _, pred = best[(start, j)]
        assert pred is not None
        boundaries.append(pred)
        j, start = start, pred
    boundaries.reverse()

    return tuple(
        tuple(factors[boundaries[k] : boundaries[k + 1]])
        for k in range(len(boundaries) - 1)
    )


@dataclass(frozen=True)
class MultiKernelHostPlan:
    n: int
    inverse: bool
    tolerance: float


@dataclass(frozen=True)
class MultiKernelFFTPlan:
    """N run as a chain of M>=1 kernels (see factor_into_kernel_chunks /
    make_multi_kernel_plan), each itself a layouts_for_radices multi-stage
    FFT. M=1 is exactly a single-kernel plan. M=2 no longer matches
    make_decomposed_plan's own addressing field-for-field (that function
    is untouched, still SPLIT/contiguous-based); this one's non-last
    kernels use AddressMappingKind.PEELED instead, chosen so every kernel
    after the first gets a vector (not scalar) DRAM read -- both are
    independently numpy-verified correct, they just lay the intermediate
    array out differently. See AddressMappingKind.PEELED.
    """

    n: int
    inverse: bool
    kernels: tuple[FFTCodegenPlan, ...]
    host: MultiKernelHostPlan


def make_multi_kernel_plan(
    chunks: tuple[tuple[int, ...], ...],
    *,
    inverse: bool = False,
    simd_lanes: int = 8,
    spad_capacity_bytes: int | None = None,
) -> MultiKernelFFTPlan:
    """Build the FFTCodegenPlan chain for an already-decided chunk
    sequence (see factor_into_kernel_chunks): kernel i processes chunks[i]
    -- its own layouts_for_radices multi-stage FFT -- in order, chained
    through DRAM.

    The *first* kernel's DRAM input read is `strided(row_stride=1,
    elem_stride=n//K_0)` -- unavoidably O(n), since it reads the external
    input and nothing precedes it to fuse a layout reset into. Every
    *later* kernel's read is `strided(row_stride=K_i, elem_stride=1)`: a
    real vector load, courtesy of the previous kernel's PEELED write (see
    AddressMappingKind.PEELED) rather than the uniform `n//K_i` every
    kernel used to pay regardless of position. Every non-last kernel's
    output is `AddressMapping.peeled(a, k_next, tail_size)` (`a` = the
    product of the kernel lengths processed before it, same as SPLIT used;
    `k_next`/`tail_size` describe the *next* kernel's own digit and
    everything after it) plus a LargeTwiddlePlan scoped the same way SPLIT's
    was. The last kernel's output is unchanged from before:
    `strided(elem_stride=a)`, and only it carries the 1/n inverse scale --
    verified (by direct index simulation, not just plan-time inspection)
    to still land in the same numpy-correct natural order as the old
    SPLIT-based scheme despite reading from a PEELED predecessor.

    None of this was carried over from the M=2 case by analogy: it's an
    independent numpy simulation of the general M-kernel decomposition
    (cascaded per-kernel local DFT, a cross-kernel twiddle scoped to the
    *remaining* problem size, and a re-split store threading the new
    digit between the already- and not-yet-transformed parts of the
    uthread id) that was verified against numpy's fft/ifft for up to 6
    chained kernels and mixed radices before being written here -- see
    the plan doc's Stage 4 write-up for the derivation and where the
    first version of this went wrong.
    """
    if not chunks:
        raise ValueError("at least one kernel chunk is required")

    lengths = [prod(chunk) for chunk in chunks]
    n = prod(lengths)
    m = len(chunks)
    host = MultiKernelHostPlan(n=n, inverse=inverse, tolerance=1.0e-3)

    if m == 1:
        kernel = _build_plan(
            length=n,
            inverse=inverse,
            total_uthreads=1,
            simd_lanes=simd_lanes,
            use_pingpong=True,
            layouts=layouts_for_radices(n, chunks[0], simd_lanes),
            spad_capacity_bytes=spad_capacity_bytes,
        )
        return MultiKernelFFTPlan(n=n, inverse=inverse, kernels=(kernel,), host=host)

    kernels: list[FFTCodegenPlan] = []
    a = 1  # product of the kernel lengths processed before the current one
    for i, ki in enumerate(lengths):
        is_last = i == m - 1
        total_uthreads = n // ki

        if i == 0:
            # First kernel: reads the external input, nothing precedes it
            # to fuse a layout reset into -- unchanged, still O(n) stride.
            input_mapping = AddressMapping.strided(row_stride=1, elem_stride=total_uthreads)
        else:
            # Every later kernel: contiguous, courtesy of the previous
            # kernel's PEELED output below -- see AddressMappingKind.PEELED.
            input_mapping = AddressMapping.contiguous(row_stride=ki)

        if is_last:
            # Unchanged: verified (by direct simulation, not just by
            # inspection) that the last kernel's own write formula still
            # lands in numpy-correct natural order even though its *read*
            # now comes from a PEELED predecessor instead of the old
            # uniform strided(elem_stride=n//K) -- see the plan doc.
            output_mapping = AddressMapping.strided(row_stride=1, elem_stride=a)
            large_twiddle = None
            inverse_scale = (1.0 / n) if inverse else None
        else:
            k_next = lengths[i + 1]
            tail_size = prod(lengths[i + 2 :]) if i + 2 < m else 1
            output_mapping = AddressMapping.peeled(a, k_next, tail_size)
            large_twiddle = LargeTwiddlePlan(
                full_length=n,
                row_count=total_uthreads // a,
                output_count=ki,
                inverse=inverse,
                a=a,
                k_next=k_next,
                tail_size=tail_size,
            )
            inverse_scale = None

        kernels.append(
            _build_plan(
                length=ki,
                inverse=inverse,
                total_uthreads=total_uthreads,
                simd_lanes=simd_lanes,
                use_pingpong=True,
                layouts=layouts_for_radices(ki, chunks[i], simd_lanes),
                kernel_name=f"FFTFP32Kernel{i}",
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                large_twiddle=large_twiddle,
                inverse_scale=inverse_scale,
                spad_capacity_bytes=spad_capacity_bytes,
            )
        )
        a *= ki

    return MultiKernelFFTPlan(n=n, inverse=inverse, kernels=tuple(kernels), host=host)


@dataclass(frozen=True)
class KernelStrideSummary:
    """Everything about one kernel's DRAM-facing layout that a plan summary
    or a stride-bound check needs -- derived entirely from fields
    FFTCodegenPlan already carries (input_mapping/output_mapping.elem_stride,
    length, each stage's radix), not new decisions. See
    summarize_multi_kernel_plan.
    """

    kernel_name: str
    local_length: int
    radices: tuple[int, ...]
    read_stride: int
    write_stride: int
    fused_transition: str  # "NONE" (this kernel's DRAM side is contiguous)
    #                         or "STORE" (a layout transition is folded into
    #                         this kernel's own output mapping -- see
    #                         AddressMappingKind.SPLIT / make_multi_kernel_plan;
    #                         this codebase never uses a LOAD-side fusion or a
    #                         standalone transpose kernel, so those aren't
    #                         separate enum values here, only documented cases
    #                         that don't occur -- see the plan's design notes).


def summarize_multi_kernel_plan(
    plan: MultiKernelFFTPlan,
) -> tuple[KernelStrideSummary, ...]:
    summaries = []
    for kernel in plan.kernels:
        transition = "NONE" if kernel.output_mapping.elem_stride == 1 else "STORE"
        summaries.append(
            KernelStrideSummary(
                kernel_name=kernel.kernel_name,
                local_length=kernel.length,
                radices=tuple(stage.radix for stage in kernel.stages),
                read_stride=kernel.input_mapping.elem_stride,
                write_stride=kernel.output_mapping.elem_stride,
                fused_transition=transition,
            )
        )
    return tuple(summaries)


def max_effective_stride(summaries: tuple[KernelStrideSummary, ...]) -> int:
    return max(
        max(s.read_stride, s.write_stride) for s in summaries
    )


def _build_batched_side(
    chunks: tuple[tuple[int, ...], ...],
    *,
    batch_count: int,
    full_length: int,
    inverse: bool,
    simd_lanes: int,
    kernel_name_prefix: str,
    is_far_side: bool,
    spad_capacity_bytes: int | None,
    plain_boundary: bool = False,
) -> tuple[FFTCodegenPlan, ...]:
    """One side (N_A or N_B) of a make_balanced_plan split, as its own
    possibly-multi-kernel PEELED chain -- see AddressMappingKind.CROSSED.

    `batch_count`: size of the *other* side (this side's own chain runs
    `batch_count` independent copies, one per the other side's index).
    `is_far_side`: True for the side whose last kernel is the *overall*
    plan's last kernel (plain, unchanged `strided(elem_stride=a)` output,
    the inverse scale, no large_twiddle -- the near side never reaches
    this, its own last kernel is CROSSED instead, chained on through DRAM
    into the far side's own first kernel).

    Every internal (non-last) kernel here is exactly what
    make_multi_kernel_plan already builds for a single-chain FFT, with
    two differences threaded through by the caller rather than derived
    fresh: `a` starts at `batch_count` (not 1) instead of a real prior
    kernel's length, and every LargeTwiddlePlan/PEELED `full_length` is
    the *overall* N, not this side's own length -- both required for the
    cross-side twiddle to land correctly (verified: full_length must be
    the overall N, not this side's own length, or the numbers come out
    wrong despite the addressing alone still being a valid permutation --
    see the plan doc for the N_B=3 example that caught this).

    `plain_boundary` (default False -- every existing caller is byte-for-
    byte unaffected): for `make_balanced_transpose_plan` only. When True,
    the boundary this side touches is realized by a standalone tiled
    transpose kernel (see fft_transpose_codegen.py) instead of being fused
    into this side's own store/load -- so the near side's (`is_far_side=
    False`) terminal kernel writes its own plain `contiguous(row_stride=
    side_length)` (no CROSSED, no large_twiddle: the twiddle moves to the
    transpose kernel), and the far side's (`is_far_side=True`) first
    kernel reads its own plain `contiguous(row_stride=length)` (no strided
    transpose read). Everything else -- internal PEELED chain, ping-pong,
    the far side's own terminal inverse-scale -- is unchanged.
    """
    lengths = [prod(chunk) for chunk in chunks]
    side_length = prod(lengths)
    m = len(chunks)

    kernels: list[FFTCodegenPlan] = []
    a = batch_count
    for i, ki in enumerate(lengths):
        is_last = i == m - 1
        total_uthreads = (side_length // ki) * batch_count

        if i == 0:
            if plain_boundary and is_far_side:
                # Reads the transpose kernel's own plain output layout
                # (contiguous(row_stride=length)), not a transpose read --
                # the transpose kernel already did that.
                input_mapping = AddressMapping.contiguous(row_stride=ki)
            else:
                elem_stride0 = (side_length // ki) * batch_count
                input_mapping = AddressMapping.strided(row_stride=1, elem_stride=elem_stride0)
        else:
            input_mapping = AddressMapping.contiguous(row_stride=ki)

        if is_last:
            if is_far_side:
                output_mapping = AddressMapping.strided(row_stride=1, elem_stride=a)
                large_twiddle = None
                inverse_scale = (1.0 / full_length) if inverse else None
            elif plain_boundary:
                # Own plain layout for the transpose kernel to read -- row
                # is this kernel's own raw uthread id, elem its own just-
                # computed ki-sized digit (row_stride=ki, NOT side_length:
                # for a multi-kernel near side this is genuinely different,
                # verified by direct index simulation against CROSSED's
                # own formula before this was written -- see
                # FFTTransposePlan's docstring). No CROSSED, no fused
                # twiddle (the transpose kernel does both, tile-wise --
                # see fft_transpose_codegen.py).
                output_mapping = AddressMapping.contiguous(row_stride=ki)
                large_twiddle = None
                inverse_scale = None
            else:
                digit_multiplier = a // batch_count
                output_mapping = AddressMapping.crossed(
                    batch_count, side_length, digit_multiplier
                )
                large_twiddle = LargeTwiddlePlan(
                    full_length=full_length,
                    row_count=batch_count,
                    output_count=side_length,
                    inverse=inverse,
                )
                inverse_scale = None
        else:
            k_next = lengths[i + 1]
            tail_size = prod(lengths[i + 2 :]) if i + 2 < m else 1
            output_mapping = AddressMapping.peeled(a, k_next, tail_size)
            large_twiddle = LargeTwiddlePlan(
                full_length=full_length,
                row_count=total_uthreads // a,
                output_count=ki,
                inverse=inverse,
                a=a,
                k_next=k_next,
                tail_size=tail_size,
            )
            inverse_scale = None

        kernels.append(
            _build_plan(
                length=ki,
                inverse=inverse,
                total_uthreads=total_uthreads,
                simd_lanes=simd_lanes,
                use_pingpong=True,
                layouts=layouts_for_radices(ki, chunks[i], simd_lanes),
                kernel_name=f"{kernel_name_prefix}{i}",
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                large_twiddle=large_twiddle,
                inverse_scale=inverse_scale,
                spad_capacity_bytes=spad_capacity_bytes,
            )
        )
        a *= ki

    return tuple(kernels)


@dataclass(frozen=True)
class FFTTransposePlan:
    """Standalone tiled-transpose + fused large-twiddle boundary between a
    make_balanced_transpose_plan near side and far side -- see
    make_balanced_transpose_plan's own docstring for the full derivation.
    Terminates the near side's physical layout and creates the far side's
    physical layout from scratch; the fused twiddle operates on logical
    coordinates recovered at this boundary, so neither side needs to know
    the other's internal chunk/kernel decomposition or stride.

    `ki_near`/`ki_far`: the near side's own terminal kernel's length and
    the far side's own first kernel's length -- these two axes are what
    actually get swapped (transposed); everything else is a spectator that
    only multiplies the tile count.

    `digit_multiplier_near` (= n_a // ki_near) and `divisor_far` (= n_b //
    ki_far): both 1 when that side is single-kernel (bare radix), matching
    the simplest case exactly. `tile_uthreads = digit_multiplier_near *
    divisor_far` independent (ki_near x ki_far) tiles partition the whole
    N elements exactly once (verified by direct index simulation against
    AddressMapping.crossed's own formula before this was written).

    One microthread owns one whole (ki_near x ki_far) tile: `ki_far`
    contiguous vector reads of `ki_near` elements each from the near
    side's own plain layout (+ matching twiddle-table rows, addressed
    identically), a local transpose through one scratchpad buffer, then
    `ki_near` contiguous vector writes of `ki_far` elements each into the
    far side's own plain layout. No further sub-tiling by a smaller
    tile_size is done in this first version -- see fft_transpose_codegen.py
    and the design writeup's "remaining limitations" for the SIMD-width-
    alignment follow-up this defers.
    """

    n: int
    n_a: int
    n_b: int
    ki_near: int
    digit_multiplier_near: int
    ki_far: int
    divisor_far: int
    inverse: bool
    kernel_name: str
    simd_lanes: int
    total_uthreads: int
    max_uthread: int
    scratchpad_elements: int


def _choose_balanced_split(
    n: int, *, scratchpad_byte_budget: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Pick a contiguous split of n's supported prime factors into
    (factors_A, factors_B) minimizing max(prod(factors_A), prod(factors_B))
    subject to both sides being plannable under scratchpad_byte_budget (via
    _choose_side_chunks) -- see make_balanced_transpose_plan.

    Only contiguous splits of the factor list are searched (matches how
    factor_into_kernel_chunks's own chunks are always contiguous runs of
    this same list) -- kept as its own function so a future subset-
    partition search can replace just this one, without touching the
    feasibility check or the caller.
    """
    factors = _prime_factors_supported(n)
    num_factors = len(factors)
    if num_factors < 2:
        raise ValueError(
            f"n={n} has fewer than 2 supported prime factors; cannot split"
        )

    best: tuple[int, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]] | None = None
    for k in range(1, num_factors):
        factors_a = factors[:k]
        factors_b = factors[k:]
        n_a = prod(factors_a)
        n_b = prod(factors_b)
        try:
            chunks_a = _choose_side_chunks(
                factors_a, batch_count=n_b, scratchpad_byte_budget=scratchpad_byte_budget
            )
            chunks_b = _choose_side_chunks(
                factors_b, batch_count=n_a, scratchpad_byte_budget=scratchpad_byte_budget
            )
        except ValueError:
            continue
        cost = max(n_a, n_b)
        if best is None or cost < best[0]:
            best = (cost, chunks_a, chunks_b)

    if best is None:
        raise ValueError(
            f"scratchpad_byte_budget={scratchpad_byte_budget} is too small "
            f"to fit n={n} into any balanced two-side split"
        )
    return best[1], best[2]


@dataclass(frozen=True)
class BalancedTransposeFFTPlan:
    """N = N_A*N_B, near side (own PEELED chain, batched N_B times) then a
    standalone tiled transpose+twiddle boundary then far side (own PEELED
    chain, batched N_A times, ending in the overall output) -- see
    make_balanced_transpose_plan. Flat, ordered, heterogeneous sequence
    for codegen: `kernels_near + (transpose,) + kernels_far`.
    """

    n: int
    inverse: bool
    kernels_near: tuple[FFTCodegenPlan, ...]
    transpose: FFTTransposePlan
    kernels_far: tuple[FFTCodegenPlan, ...]
    host: MultiKernelHostPlan


def make_balanced_transpose_plan(
    n: int,
    *,
    scratchpad_byte_budget: int,
    simd_lanes: int = 8,
    inverse: bool = False,
    spad_capacity_bytes: int | None = None,
) -> BalancedTransposeFFTPlan:
    """N = N_A*N_B (N_A ~ N_B ~ sqrt(N) where the factorization allows,
    minimizing max(N_A,N_B) otherwise -- see _choose_balanced_split), each
    side its own possibly-multi-kernel PEELED chain bounded by that side's
    own size (see _build_batched_side, reused unmodified for everything
    except the boundary itself), joined by one standalone tiled transpose
    kernel instead of make_balanced_plan's scalar CROSSED-fused write.

    The near side's own terminal kernel and the far side's own first
    kernel both use `_build_batched_side(..., plain_boundary=True)`: each
    writes/reads its own plain contiguous layout, with no fused twiddle
    and no strided/scalar transpose access on either end -- see that
    parameter's own docstring. The transpose kernel (FFTTransposePlan)
    does the permutation and the twiddle, tile-wise, so both DRAM-facing
    ends of the boundary stay unit-stride, and neither side's plan needs
    to know the other side's internal chunk/kernel decomposition.
    """
    chunks_a, chunks_b = _choose_balanced_split(n, scratchpad_byte_budget=scratchpad_byte_budget)
    n_a = prod(prod(chunk) for chunk in chunks_a)
    n_b = prod(prod(chunk) for chunk in chunks_b)
    host = MultiKernelHostPlan(n=n, inverse=inverse, tolerance=1.0e-3)

    kernels_near = _build_batched_side(
        chunks_a, batch_count=n_b, full_length=n, inverse=inverse,
        simd_lanes=simd_lanes, kernel_name_prefix="FFTFP32Near",
        is_far_side=False, spad_capacity_bytes=spad_capacity_bytes,
        plain_boundary=True,
    )
    kernels_far = _build_batched_side(
        chunks_b, batch_count=n_a, full_length=n, inverse=inverse,
        simd_lanes=simd_lanes, kernel_name_prefix="FFTFP32Far",
        is_far_side=True, spad_capacity_bytes=spad_capacity_bytes,
        plain_boundary=True,
    )

    ki_near = kernels_near[-1].length
    digit_multiplier_near = n_a // ki_near
    ki_far = kernels_far[0].length
    divisor_far = n_b // ki_far
    tile_uthreads = digit_multiplier_near * divisor_far

    scratchpad_elements = 2 * ki_near * ki_far  # real+imag, one tile buffer
    if spad_capacity_bytes is None:
        max_uthread = tile_uthreads
    else:
        bytes_per_uthread = scratchpad_elements * 4
        if bytes_per_uthread > spad_capacity_bytes:
            raise ValueError(
                f"spad_capacity_bytes={spad_capacity_bytes} is too small "
                f"to fit even a single transpose tile "
                f"(needs {bytes_per_uthread} bytes)"
            )
        max_uthread = min(tile_uthreads, spad_capacity_bytes // bytes_per_uthread)

    transpose = FFTTransposePlan(
        n=n, n_a=n_a, n_b=n_b, ki_near=ki_near,
        digit_multiplier_near=digit_multiplier_near, ki_far=ki_far,
        divisor_far=divisor_far, inverse=inverse,
        kernel_name="FFTFP32Transpose", simd_lanes=simd_lanes,
        total_uthreads=tile_uthreads, max_uthread=max_uthread,
        scratchpad_elements=scratchpad_elements,
    )

    return BalancedTransposeFFTPlan(
        n=n, inverse=inverse, kernels_near=kernels_near, transpose=transpose,
        kernels_far=kernels_far, host=host,
    )


def make_balanced_plan(
    chunks_A: tuple[tuple[int, ...], ...],
    chunks_B: tuple[tuple[int, ...], ...],
    *,
    inverse: bool = False,
    simd_lanes: int = 8,
    spad_capacity_bytes: int | None = None,
) -> MultiKernelFFTPlan:
    """N = N_A * N_B (N_A = prod(chunks_A), N_B = prod(chunks_B)), each
    side its own possibly-multi-kernel PEELED chain, joined by one
    AddressMappingKind.CROSSED transpose fused into side A's own last
    kernel's store (never a standalone kernel) -- generalizes
    make_decomposed_plan from "each side is exactly one bare radix" to
    "each side is any chunk sequence make_multi_kernel_plan could build
    on its own", while keeping make_decomposed_plan itself untouched as
    the from-first-principles baseline this checks against (the M=1
    case on both sides is exactly make_decomposed_plan(N_A, N_B), field
    for field where the addressing coincides -- see AddressMappingKind
    .CROSSED's own docstring for why it degenerates to CONTIGUOUS there).

    The value of this over a single flat make_multi_kernel_plan chain
    over N's *whole* factor list: max effective DRAM stride bounded by
    max(N_A, N_B) rather than by N -- choosing N_A ~ N_B ~ sqrt(N) reaches
    the four-step FFT's classic O(sqrt(N)) bound, regardless of how deep
    either side's own internal chain has to go to fit the scratchpad
    budget (each side's own chain is bounded by that side's own size,
    never by N -- verified directly, not assumed: N=2^20 split into two
    balanced ~1024-element halves, each itself a deep 10-kernel all-
    radix-2 chain, gives max stride 1024 = sqrt(N); the same N run as one
    flat 20-kernel chain gives ~524288 ~ N/2). Choosing chunks_A/chunks_B
    to actually be balanced is the caller's job here (see the plan doc's
    "balanced split search", not yet implemented) -- this function only
    builds whatever split it's given.
    """
    n_a = prod(prod(chunk) for chunk in chunks_A)
    n_b = prod(prod(chunk) for chunk in chunks_B)
    n = n_a * n_b
    host = MultiKernelHostPlan(n=n, inverse=inverse, tolerance=1.0e-3)

    kernels_a = _build_batched_side(
        chunks_A,
        batch_count=n_b,
        full_length=n,
        inverse=inverse,
        simd_lanes=simd_lanes,
        kernel_name_prefix="FFTFP32KernelA",
        is_far_side=False,
        spad_capacity_bytes=spad_capacity_bytes,
    )
    kernels_b = _build_batched_side(
        chunks_B,
        batch_count=n_a,
        full_length=n,
        inverse=inverse,
        simd_lanes=simd_lanes,
        kernel_name_prefix="FFTFP32KernelB",
        is_far_side=True,
        spad_capacity_bytes=spad_capacity_bytes,
    )

    return MultiKernelFFTPlan(
        n=n, inverse=inverse, kernels=kernels_a + kernels_b, host=host
    )


# ------------------------------------------------- recursive tiled-transpose
#
# Generalizes make_balanced_transpose_plan's single 2-way split to a full
# recursive (six-step-FFT-style) decomposition: FFT chunk size and physical
# transpose tile size are two completely independent choices (see module
# docstring's design writeup for this session). Additive: does not touch
# make_multi_kernel_plan, make_balanced_plan, make_balanced_transpose_plan,
# or any AddressMapping kind -- those remain the flat/2-way regression
# baseline and numeric reference.
#
# One recursive node FFTNode(M, R) owns exactly the contract "R independent
# M-point transforms, natural contiguous order in (addr = q*M + n) both on
# entry and on exit" -- see FFTLeafPlan/FFTRecursiveNodePlan. A node is
# either a leaf (M fits one fused multi-radix kernel) or splits M = A*B and
# emits PRE transpose -> B-point FFT (batched R*A times) -> MIDDLE transpose
# with the current node's own W_M twiddle -> recursive FFTNode(A, R*B) ->
# POST transpose. Every junction address was independently re-derived and
# cross-checked (not just transcribed) against this contract before being
# implemented, then verified again by direct index/numeric simulation
# (bijection of the generic tiled transpose; multi-level recursion vs.
# numpy.fft) before any codegen was written -- see verify_fft_plan.py.


@dataclass(frozen=True)
class PhysicalTransposePlan:
    """One standalone tiled-transpose kernel: `replica_count` independent
    `rows x cols` matrices, each transposed to `cols x rows`, tile-wise.
    Completely independent of any FFT chunk/radix length -- `rows`/`cols`
    here are PRE/POST's (B,A) or MIDDLE's (A,B), never `ki_near`/`ki_far`.

    `twiddle_modulus`: None for PRE/POST (plain transpose). For MIDDLE,
    the *current recursive node's own* M (never the top-level N) -- see
    AddressMappingKind-adjacent docstrings elsewhere in this module for why
    a per-level modulus, not a global one, is what makes 3+-factor
    decomposition correct. The twiddle table is dense, `rows*cols` (=M)
    elements, addressed identically to this kernel's own source matrix
    (`table[r*cols+c]`) -- no per-replica duplication (see fft_transpose_
    codegen.py's precompute).

    `apply_inverse_scale`: True only for the *root* node's own POST
    transpose (or the root leaf, if the whole FFT fits in one kernel) when
    `inverse` -- the overall 1/N lands exactly once, fused into the final
    store, not per-node.

    One microthread owns one whole `tile_rows x tile_cols` physical tile
    (independent of `simd_lanes` up to this plan's own choice -- default is
    `min(simd_lanes, rows/cols)`, see make_recursive_transpose_plan). Every
    DRAM vector load/store is unit-stride (contiguous `tile_cols`-wide rows
    on the source side, `tile_rows`-wide rows on the destination side);
    row-to-row jumps are allowed and expected. Tail tiles (rows/cols not a
    multiple of tile_rows/tile_cols) are masked, never out-of-bounds -- see
    fft_transpose_codegen.py.
    """

    kernel_name: str
    rows: int
    cols: int
    replica_count: int
    tile_rows: int
    tile_cols: int
    grid_rows: int
    grid_cols: int
    total_uthreads: int
    max_uthread: int
    scratchpad_elements: int  # per-uthread: 2 * tile_rows * tile_cols
    twiddle_modulus: int | None
    inverse: bool
    simd_lanes: int
    apply_inverse_scale: bool = False


@dataclass(frozen=True)
class FFTLeafPlan:
    """FFTNode(m, r) that fits one fused multi-radix kernel outright (no
    transpose boundary at all) -- `kernel` is an ordinary FFTCodegenPlan
    with `length=m`, `total_uthreads=r`, both DRAM mappings plain
    `contiguous(row_stride=m)`, built via the existing layouts_for_radices/
    _build_plan exactly as every other single-kernel FFT in this module.
    """

    m: int
    r: int
    kernel: FFTCodegenPlan


@dataclass(frozen=True)
class FFTRecursiveNodePlan:
    """FFTNode(m, r) with m = a*b, too large for one fused kernel: PRE
    transpose -> near_fft (b-point, batched r*a times) -> middle_transpose
    (W_m twiddle fused) -> far_child (FFTNode(a, r*b)) -> post_transpose.
    See the module-level design writeup for the full derivation of why
    this exact shape sequence (b x a -> a x b -> [twiddle] -> b x a ->
    a x b) reproduces FFTNode's own natural-order contract at every
    junction, cross-checked address by address before implementation.
    """

    m: int
    r: int
    a: int
    b: int
    pre_transpose: PhysicalTransposePlan
    near_fft: FFTLeafPlan
    middle_transpose: PhysicalTransposePlan
    far_child: "FFTNode"
    post_transpose: PhysicalTransposePlan


FFTNode = Union[FFTLeafPlan, FFTRecursiveNodePlan]


@dataclass(frozen=True)
class RecursiveFFTPlan:
    n: int
    inverse: bool
    root: FFTNode
    host: MultiKernelHostPlan


def _choose_recursive_split(
    m: int, *, scratchpad_byte_budget: int
) -> int | None:
    """Returns b (the near_fft's own length) if m needs splitting, or None
    if m already fits one fused leaf kernel outright. b is chosen as the
    *largest* suffix-factor-product of m's own supported prime
    factorization that still fits scratchpad_byte_budget as a single
    leaf -- larger b means fewer recursion levels, fewer transpose kernels,
    and a leaf that fuses as many radix stages as it can (see the module
    design writeup's own reasoning for this heuristic; candidate
    generation is kept in this one function so a benchmark-driven cost
    model can replace just this later, same discipline
    factor_into_kernel_chunks's own docstring already established).
    """
    if scratchpad_byte_budget <= 0:
        raise ValueError("scratchpad_byte_budget must be positive")
    cap = scratchpad_byte_budget // 16
    if m <= cap:
        return None

    factors = _prime_factors_supported(m)
    suffix = [1] * (len(factors) + 1)
    for i in range(len(factors) - 1, -1, -1):
        suffix[i] = suffix[i + 1] * factors[i]

    for i in range(1, len(factors) + 1):
        if suffix[i] <= cap:
            if suffix[i] <= 1:
                raise ValueError(
                    f"scratchpad_byte_budget={scratchpad_byte_budget} is too "
                    f"small to make progress on m={m} (factors={factors}: "
                    f"not even the smallest one fits)"
                )
            return suffix[i]
    raise ValueError(
        f"scratchpad_byte_budget={scratchpad_byte_budget} is too small "
        f"for m={m} (factors={factors})"
    )


def _build_physical_transpose(
    *,
    rows: int,
    cols: int,
    replica_count: int,
    tile_rows: int,
    tile_cols: int,
    twiddle_modulus: int | None,
    inverse: bool,
    kernel_name: str,
    simd_lanes: int,
    spad_capacity_bytes: int | None,
    apply_inverse_scale: bool,
) -> PhysicalTransposePlan:
    grid_rows = -(-rows // tile_rows)
    grid_cols = -(-cols // tile_cols)
    total_uthreads = replica_count * grid_rows * grid_cols
    scratchpad_elements = 2 * tile_rows * tile_cols

    if spad_capacity_bytes is None:
        max_uthread = total_uthreads
    else:
        bytes_per_uthread = scratchpad_elements * 4
        if bytes_per_uthread > spad_capacity_bytes:
            raise ValueError(
                f"spad_capacity_bytes={spad_capacity_bytes} is too small to "
                f"fit even a single {tile_rows}x{tile_cols} transpose tile "
                f"(needs {bytes_per_uthread} bytes)"
            )
        max_uthread = min(total_uthreads, spad_capacity_bytes // bytes_per_uthread)

    return PhysicalTransposePlan(
        kernel_name=kernel_name,
        rows=rows,
        cols=cols,
        replica_count=replica_count,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        total_uthreads=total_uthreads,
        max_uthread=max_uthread,
        scratchpad_elements=scratchpad_elements,
        twiddle_modulus=twiddle_modulus,
        inverse=inverse,
        simd_lanes=simd_lanes,
        apply_inverse_scale=apply_inverse_scale,
    )


def _build_recursive_node(
    m: int,
    r: int,
    *,
    scratchpad_byte_budget: int,
    simd_lanes: int,
    inverse: bool,
    spad_capacity_bytes: int | None,
    tile_rows: int | None,
    tile_cols: int | None,
    is_root: bool,
    node_id: list[int],
) -> FFTNode:
    idx = node_id[0]
    node_id[0] += 1

    split_b = _choose_recursive_split(m, scratchpad_byte_budget=scratchpad_byte_budget)
    if split_b is None:
        radices = tuple(_prime_factors_supported(m))
        inverse_scale = (1.0 / m) if (inverse and is_root) else None
        kernel = _build_plan(
            length=m,
            inverse=inverse,
            total_uthreads=r,
            simd_lanes=simd_lanes,
            use_pingpong=True,
            layouts=layouts_for_radices(m, radices, simd_lanes),
            kernel_name=f"FFTRecLeaf{idx}",
            input_mapping=AddressMapping.contiguous(row_stride=m),
            output_mapping=AddressMapping.contiguous(row_stride=m),
            large_twiddle=None,
            inverse_scale=inverse_scale,
            spad_capacity_bytes=spad_capacity_bytes,
        )
        return FFTLeafPlan(m=m, r=r, kernel=kernel)

    b = split_b
    a = m // b
    tr = tile_rows if tile_rows is not None else min(simd_lanes, b, a)
    tc = tile_cols if tile_cols is not None else min(simd_lanes, a, b)
    tr = max(tr, 1)
    tc = max(tc, 1)

    pre = _build_physical_transpose(
        rows=b, cols=a, replica_count=r, tile_rows=tr, tile_cols=tc,
        twiddle_modulus=None, inverse=inverse, kernel_name=f"FFTRecPre{idx}",
        simd_lanes=simd_lanes, spad_capacity_bytes=spad_capacity_bytes,
        apply_inverse_scale=False,
    )

    near_radices = tuple(_prime_factors_supported(b))
    near_kernel = _build_plan(
        length=b,
        inverse=inverse,
        total_uthreads=r * a,
        simd_lanes=simd_lanes,
        use_pingpong=True,
        layouts=layouts_for_radices(b, near_radices, simd_lanes),
        kernel_name=f"FFTRecNear{idx}",
        input_mapping=AddressMapping.contiguous(row_stride=b),
        output_mapping=AddressMapping.contiguous(row_stride=b),
        large_twiddle=None,
        inverse_scale=None,
        spad_capacity_bytes=spad_capacity_bytes,
    )
    near_fft = FFTLeafPlan(m=b, r=r * a, kernel=near_kernel)

    middle = _build_physical_transpose(
        rows=a, cols=b, replica_count=r, tile_rows=tr, tile_cols=tc,
        twiddle_modulus=m, inverse=inverse, kernel_name=f"FFTRecMid{idx}",
        simd_lanes=simd_lanes, spad_capacity_bytes=spad_capacity_bytes,
        apply_inverse_scale=False,
    )

    far_child = _build_recursive_node(
        a, r * b, scratchpad_byte_budget=scratchpad_byte_budget,
        simd_lanes=simd_lanes, inverse=inverse,
        spad_capacity_bytes=spad_capacity_bytes, tile_rows=tile_rows,
        tile_cols=tile_cols, is_root=False, node_id=node_id,
    )

    post = _build_physical_transpose(
        rows=b, cols=a, replica_count=r, tile_rows=tr, tile_cols=tc,
        twiddle_modulus=None, inverse=inverse, kernel_name=f"FFTRecPost{idx}",
        simd_lanes=simd_lanes, spad_capacity_bytes=spad_capacity_bytes,
        apply_inverse_scale=(inverse and is_root),
    )

    return FFTRecursiveNodePlan(
        m=m, r=r, a=a, b=b, pre_transpose=pre, near_fft=near_fft,
        middle_transpose=middle, far_child=far_child, post_transpose=post,
    )


def make_recursive_transpose_plan(
    n: int,
    *,
    scratchpad_byte_budget: int,
    simd_lanes: int = 8,
    tile_rows: int | None = None,
    tile_cols: int | None = None,
    inverse: bool = False,
    spad_capacity_bytes: int | None = None,
) -> RecursiveFFTPlan:
    """N decomposed recursively (six-step-FFT style): each node either
    fuses into one multi-radix leaf kernel (see FFTLeafPlan) or splits
    M=A*B and emits PRE transpose -> B-point FFT -> MIDDLE transpose
    (W_M twiddle) -> recursive FFTNode(A, R*B) -> POST transpose (see
    FFTRecursiveNodePlan). Physical transpose tile size (tile_rows/
    tile_cols, default min(simd_lanes, the matrix's own two dimensions))
    is chosen completely independently of any FFT chunk length -- see the
    module design writeup. Additive: make_multi_kernel_plan/
    make_balanced_plan/make_balanced_transpose_plan are untouched.
    """
    node_id = [0]
    root = _build_recursive_node(
        n, 1, scratchpad_byte_budget=scratchpad_byte_budget,
        simd_lanes=simd_lanes, inverse=inverse,
        spad_capacity_bytes=spad_capacity_bytes, tile_rows=tile_rows,
        tile_cols=tile_cols, is_root=True, node_id=node_id,
    )
    host = MultiKernelHostPlan(n=n, inverse=inverse, tolerance=1.0e-3)
    return RecursiveFFTPlan(n=n, inverse=inverse, root=root, host=host)
