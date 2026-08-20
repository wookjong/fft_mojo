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
from typing import Literal

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
    """

    CONTIGUOUS = "contiguous"
    STRIDED = "strided"
    SPLIT = "split"
    PEELED = "peeled"


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
