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
    """

    CONTIGUOUS = "contiguous"
    STRIDED = "strided"


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

    @staticmethod
    def contiguous(row_stride: int, base: int = 0) -> "AddressMapping":
        return AddressMapping(AddressMappingKind.CONTIGUOUS, row_stride, 1, base)

    @staticmethod
    def strided(row_stride: int, elem_stride: int, base: int = 0) -> "AddressMapping":
        if elem_stride == 1:
            raise ValueError("a strided mapping needs elem_stride != 1")
        return AddressMapping(AddressMappingKind.STRIDED, row_stride, elem_stride, base)


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
    DFT. `row_count`*`output_count` == `full_length`, and the table is laid
    out to match this kernel's own `output_mapping` (row_stride, elem_stride)
    so the fetch reuses the store's own already-computed address, rather
    than being a second, independent address computation.

    A full twiddle table is the direct approach; it is what's needed here.
    rocFFT/clFFT-style digit-split reconstruction (W_N^u = T0[u0]*T1[u1]*...)
    would trade table size for extra multiplies on very large N -- worth
    adding if a table of size N stops being affordable, not before.
    """

    full_length: int
    row_count: int
    output_count: int
    inverse: bool
    table_name: str = "large_twiddle"


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
    max_uthread: int
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
    """

    length: int
    inverse: bool
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
    max_uthread: int,
    simd_lanes: int,
    layouts: tuple[_StageLayout, ...],
) -> None:
    if length <= 0:
        raise ValueError("FFT length must be positive")
    if max_uthread <= 0:
        raise ValueError("max_uthread must be positive")
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
        if simd_lanes % stage.twiddle_lane_divisor != 0:
            raise ValueError(
                f"stage {sid}: twiddle_lane_divisor must divide simd_lanes"
            )
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
    max_uthread: int,
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
        total_elems=length * max_uthread,
        pool_elems=simd_lanes * max_uthread,
        length=length,
        max_uthread=max_uthread,
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
    max_uthread: int,
    simd_lanes: int,
    use_pingpong: bool,
    layouts: tuple[_StageLayout, ...],
    kernel_name: str = "FFTFP32",
    input_mapping: AddressMapping | None = None,
    output_mapping: AddressMapping | None = None,
    large_twiddle: LargeTwiddlePlan | None = None,
    inverse_scale: float | None | _Default = _DEFAULT,
) -> FFTCodegenPlan:
    _check_layouts(
        length=length,
        max_uthread=max_uthread,
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
            max_uthread=max_uthread,
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


def make_444_plan(*, inverse: bool = False, max_uthread: int = 1) -> FFTCodegenPlan:
    """Create the fully lowered N=64, radix-4 x radix-4 x radix-4 plan."""

    return _build_plan(
        length=64,
        inverse=inverse,
        max_uthread=max_uthread,
        simd_lanes=8,
        use_pingpong=True,
        layouts=layouts_for_radices(64, (4, 4, 4), simd_lanes=8),
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
        max_uthread=n0,
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
        max_uthread=n1,
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


def _max_pow2_exponent(simd_lanes: int) -> int:
    """Largest e such that 2**e divides simd_lanes."""
    e = 0
    while simd_lanes % (2 ** (e + 1)) == 0:
        e += 1
    return e


def factor_into_kernel_chunks(
    n: int, *, scratchpad_byte_budget: int, simd_lanes: int = 8
) -> tuple[tuple[int, ...], ...]:
    """Factor n into a sequence of per-kernel radix chunks -- chunk i
    becomes kernel i's radix sequence for layouts_for_radices, processed
    in order and chained through DRAM by make_multi_kernel_plan.

    Each chunk is built greedily under two independent caps:
      * scratchpad: `16 * chunk_product <= scratchpad_byte_budget` (the
        `scratchpad_uthread_stride = 2*length` times 2 ping-pong buffers
        times 4 bytes/float convention `_build_plan` already uses). No
        default is offered: the real M2NDP per-uthread scratchpad size
        isn't established in this codebase, so callers pick a value.
      * layouts_for_radices/_check_layouts's real constraint (confirmed
        in Stage 2's tests by bypassing it and watching correct results
        go to ~O(1) wrong -- not overly conservative, not relaxable):
        every non-last *stage*'s cumulative radix product must divide
        simd_lanes. A chunk may carry at most one prime that isn't a
        power of two, and it always goes last within the chunk; a leading
        run of radix-2 stages is capped so its own cumulative products
        (1, 2, 4, ...) stay within simd_lanes too.
    """
    if scratchpad_byte_budget <= 0:
        raise ValueError("scratchpad_byte_budget must be positive")

    factors = _prime_factors_supported(n)
    twos_left = factors.count(2)
    others_left = [f for f in factors if f != 2]

    max_e = _max_pow2_exponent(simd_lanes)
    cap = scratchpad_byte_budget // 16

    chunks: list[tuple[int, ...]] = []
    while twos_left > 0 or others_left:
        if others_left:
            other = others_left.pop(0)
            if other > cap:
                raise ValueError(
                    f"scratchpad_byte_budget={scratchpad_byte_budget} is "
                    f"too small to fit even a single radix-{other} stage"
                )
            max_j = max_e + 1  # leading 2's, then this trailing non-2 prime
            j = 0
            product = other
            while j < max_j and twos_left > 0 and product * 2 <= cap:
                j += 1
                twos_left -= 1
                product *= 2
            chunks.append(tuple([2] * j + [other]))
        else:
            if 2 > cap:
                raise ValueError(
                    f"scratchpad_byte_budget={scratchpad_byte_budget} is "
                    "too small to fit even a single radix-2 stage"
                )
            max_j = max_e + 2  # pure radix-2 tower, its last stage exempt
            j = 0
            product = 1
            while j < max_j and twos_left > 0 and product * 2 <= cap:
                j += 1
                twos_left -= 1
                product *= 2
            chunks.append(tuple([2] * j))

    return tuple(chunks)


@dataclass(frozen=True)
class MultiKernelHostPlan:
    n: int
    inverse: bool
    tolerance: float


@dataclass(frozen=True)
class MultiKernelFFTPlan:
    """N run as a chain of M>=1 kernels (see factor_into_kernel_chunks /
    make_multi_kernel_plan), each itself a layouts_for_radices multi-stage
    FFT. M=1 is exactly a single-kernel plan; M=2 exactly matches
    make_decomposed_plan's addressing (verified in verify_fft_plan.py).
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
) -> MultiKernelFFTPlan:
    """Build the FFTCodegenPlan chain for an already-decided chunk
    sequence (see factor_into_kernel_chunks): kernel i processes chunks[i]
    -- its own layouts_for_radices multi-stage FFT -- in order, chained
    through DRAM.

    M=1 and M=2 only for now: the M=2 formulas are make_decomposed_plan's,
    generalized from a single bare radix per kernel to a full
    layouts_for_radices chunk (chunks[0] plays make_decomposed_plan's
    `n1` -- the first kernel's own length -- and chunks[1] its `n0`).
    M>=3 needs LargeTwiddlePlan's exponent generalized for a kernel whose
    own uthread id mixes an already- and not-yet-transformed digit -- not
    yet derived; see the plan doc's Stage 4.
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
            max_uthread=1,
            simd_lanes=simd_lanes,
            use_pingpong=True,
            layouts=layouts_for_radices(n, chunks[0], simd_lanes),
        )
        return MultiKernelFFTPlan(n=n, inverse=inverse, kernels=(kernel,), host=host)

    if m == 2:
        k0, k1 = lengths
        kernel0 = _build_plan(
            length=k0,
            inverse=inverse,
            max_uthread=k1,
            simd_lanes=simd_lanes,
            use_pingpong=True,
            layouts=layouts_for_radices(k0, chunks[0], simd_lanes),
            kernel_name="FFTFP32Kernel0",
            input_mapping=AddressMapping.strided(row_stride=1, elem_stride=k1),
            output_mapping=AddressMapping.contiguous(row_stride=k0),
            large_twiddle=LargeTwiddlePlan(
                full_length=n, row_count=k1, output_count=k0, inverse=inverse
            ),
            inverse_scale=None,
        )
        kernel1 = _build_plan(
            length=k1,
            inverse=inverse,
            max_uthread=k0,
            simd_lanes=simd_lanes,
            use_pingpong=True,
            layouts=layouts_for_radices(k1, chunks[1], simd_lanes),
            kernel_name="FFTFP32Kernel1",
            input_mapping=AddressMapping.strided(row_stride=1, elem_stride=k0),
            output_mapping=AddressMapping.strided(row_stride=1, elem_stride=k0),
            large_twiddle=None,
            inverse_scale=(1.0 / n) if inverse else None,
        )
        return MultiKernelFFTPlan(
            n=n, inverse=inverse, kernels=(kernel0, kernel1), host=host
        )

    raise NotImplementedError(
        f"{m} kernels: chaining more than 2 needs LargeTwiddlePlan's "
        "exponent generalized for a kernel whose own uthread id mixes an "
        "already-transformed digit with not-yet-transformed ones -- see "
        "plan Stage 4, not yet derived/verified."
    )
