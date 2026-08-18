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
from math import cos, pi, sin
from typing import Literal

from fft_butterflies import SUPPORTED_RADICES


LoadSource = Literal["input", "scratchpad"]
LoadMode = Literal["vector", "scalar_pack"]
StoreDestination = Literal["output", "scratchpad"]
StoreMode = Literal["vector", "scalar_lanes"]


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
class OutputPlan:
    output: int
    twiddle: TwiddlePlan | None
    scale: float | None
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
    """Fully lowered plan consumed by fft_codegen.py."""

    length: int
    inverse: bool
    max_uthread: int
    simd_lanes: int

    # Runtime address strides are also planner decisions.
    batch_stride: int
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
    store_layout: str = "linear"


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
        if stage.store_layout not in (
            "linear",
            "stage0_b_k2",
            "stage1_a_k1_k2",
        ):
            raise ValueError(
                f"stage {sid}: unknown store_layout {stage.store_layout!r}"
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
) -> LoadPlan:
    source: LoadSource = "input" if first_stage else "scratchpad"
    buffer_name = None if first_stage else read_buffer

    if source == "scratchpad" and buffer_name is None:
        raise ValueError("scratchpad load requires a resolved buffer name")

    if valid_lanes == simd_lanes:
        return LoadPlan(
            operand=operand,
            source=source,
            buffer_name=buffer_name,
            mode="vector",
            base_offset=base_offset,
        )

    return LoadPlan(
        operand=operand,
        source=source,
        buffer_name=buffer_name,
        mode="scalar_pack",
        packed_lane_offsets=tuple(
            (base_offset + lane) if lane < valid_lanes else None
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
) -> StorePlan:
    output_batch_base = simd_it * layout.output_batch_width

    if last_stage:
        base = output_batch_base + output * layout.output_stride
        if valid_lanes == simd_lanes:
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
            lane_offsets=tuple(base + lane for lane in range(valid_lanes)),
        )

    if write_buffer is None:
        raise ValueError("intermediate stage requires a resolved write buffer")

    if layout.store_layout == "stage0_b_k2":
        offsets = []
        for lane in range(valid_lanes):
            global_bfly = simd_it * simd_lanes + lane
            offsets.append(global_bfly * layout.output_batch_width + output)
        return StorePlan(
            destination="scratchpad",
            buffer_name=write_buffer,
            mode="scalar_lanes",
            lane_offsets=tuple(offsets),
        )

    if layout.store_layout == "stage1_a_k1_k2":
        divisor = layout.twiddle_lane_divisor
        group_stride = layout.radix * divisor
        offsets = []
        for lane in range(valid_lanes):
            global_bfly = simd_it * simd_lanes + lane
            a = global_bfly // divisor
            k2 = global_bfly % divisor
            offsets.append(a * group_stride + output * divisor + k2)
        return StorePlan(
            destination="scratchpad",
            buffer_name=write_buffer,
            mode="scalar_lanes",
            lane_offsets=tuple(offsets),
        )

    base = output_batch_base + output * layout.output_stride
    if valid_lanes == simd_lanes:
        return StorePlan(
            destination="scratchpad",
            buffer_name=write_buffer,
            mode="vector",
            base_offset=base,
        )
    return StorePlan(
        destination="scratchpad",
        buffer_name=write_buffer,
        mode="scalar_lanes",
        lane_offsets=tuple(base + lane for lane in range(valid_lanes)),
    )


def _lower_stages(
    *,
    length: int,
    inverse: bool,
    simd_lanes: int,
    layouts: tuple[_StageLayout, ...],
    buffer_names: tuple[str, ...],
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
                        scale=(1.0 / length) if (last_stage and inverse) else None,
                        store=_make_store(
                            last_stage=last_stage,
                            write_buffer=write_buffer,
                            layout=layout,
                            simd_it=simd_it,
                            output=output,
                            valid_lanes=valid_lanes,
                            simd_lanes=simd_lanes,
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


def _build_plan(
    *,
    length: int,
    inverse: bool,
    max_uthread: int,
    simd_lanes: int,
    use_pingpong: bool,
    layouts: tuple[_StageLayout, ...],
) -> FFTCodegenPlan:
    _check_layouts(
        length=length,
        max_uthread=max_uthread,
        simd_lanes=simd_lanes,
        layouts=layouts,
    )

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
        batch_stride=length,
        scratchpad_uthread_stride=scratchpad_stride,
        scratchpad_buffers=scratchpad_buffers,
        stages=_lower_stages(
            length=length,
            inverse=inverse,
            simd_lanes=simd_lanes,
            layouts=layouts,
            buffer_names=buffer_names,
        ),
        host=_make_host_plan(
            length=length,
            inverse=inverse,
            max_uthread=max_uthread,
            simd_lanes=simd_lanes,
        ),
    )


def make_444_plan(*, inverse: bool = False, max_uthread: int = 1) -> FFTCodegenPlan:
    """Create the fully lowered N=64, radix-4 x radix-4 x radix-4 plan."""

    layouts = (
        _StageLayout(
            radix=4,
            butterfly_count=16,
            input_batch_width=8,
            input_stride=16,
            output_batch_width=4,
            output_stride=4,
            twiddle_modulus=64,
            twiddle_stride=1,
            twiddle_lane_divisor=1,
            store_layout="stage0_b_k2",
        ),
        _StageLayout(
            radix=4,
            butterfly_count=16,
            input_batch_width=8,
            input_stride=16,
            output_batch_width=8,
            output_stride=16,
            twiddle_modulus=16,
            twiddle_stride=1,
            twiddle_lane_divisor=4,
            store_layout="stage1_a_k1_k2",
        ),
        _StageLayout(
            radix=4,
            butterfly_count=16,
            input_batch_width=8,
            input_stride=16,
            output_batch_width=8,
            output_stride=16,
            twiddle_modulus=None,
            store_layout="linear",
        ),
    )

    return _build_plan(
        length=64,
        inverse=inverse,
        max_uthread=max_uthread,
        simd_lanes=8,
        use_pingpong=True,
        layouts=layouts,
    )