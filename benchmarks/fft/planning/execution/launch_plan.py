"""FFT launch coverage shared by standalone, recursive and hybrid codegen."""
from dataclasses import dataclass
from typing import TYPE_CHECKING

from planning.core.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile
from planning.execution.launch_support import validate_fft_launch_granule

if TYPE_CHECKING:
    from planning.core.fft_plan_core import FFTCodegenPlan


@dataclass(frozen=True)
class FFTLaunchRound:
    fft_base: int
    fft_count: int
    uthread_count: int
    pool_elems: int


@dataclass(frozen=True)
class FFTLaunchPlan:
    total_ffts: int
    workers_per_fft: int
    chunk_uthreads: int
    slots_per_chunk: int
    round_fft_capacity: int
    round_uthread_capacity: int
    simd_lanes: int
    alignment_bytes: int
    target_units: int
    padded: bool
    spread_across_units: bool

    @property
    def uses_bounds_params(self) -> bool:
        return self.padded or self.total_ffts > self.round_fft_capacity

    def physical_count(self, fft_count: int) -> int:
        if self.padded:
            return -(-fft_count // self.slots_per_chunk) * self.chunk_uthreads
        return fft_count * self.workers_per_fft

    def rounds(self):
        for first in range(0, self.total_ffts, self.round_fft_capacity):
            count = min(self.round_fft_capacity, self.total_ffts - first)
            physical = self.physical_count(count)
            yield FFTLaunchRound(first, count, physical, physical * self.simd_lanes)

    def local_fft_expr(self) -> str:
        if self.padded:
            return (f"(global_uthread_id() // {self.chunk_uthreads}) * {self.slots_per_chunk} "
                    f"+ (global_uthread_id() % {self.chunk_uthreads}) // {self.workers_per_fft}")
        if self.workers_per_fft == 1:
            return "global_uthread_id()"
        return f"global_uthread_id() // {self.workers_per_fft}"


def resolve_fft_launch(plan: "FFTCodegenPlan", *, target: TargetProfile = DEFAULT_TARGET_PROFILE,
                       spread_across_units: bool = False) -> FFTLaunchPlan:
    if plan.persistent is not None:
        raise ValueError("persistent leaves own their internal rounds")
    workers = plan.cooperation.workers_per_fft if plan.cooperation is not None else 1
    chunk = target.interleave_chunk_uthreads
    for name, value in (("workers_per_fft", workers), ("chunk_uthreads", chunk),
                        ("num_ndp_units", target.num_ndp_units), ("max_uthread", plan.max_uthread)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if chunk % workers or plan.max_uthread % workers or plan.total_uthreads % workers:
        raise ValueError("FFT workers must divide the chunk, local cap and total launch")
    if target.mapping_stride_bytes != chunk * target.uthread_bytes:
        raise ValueError("target mapping stride must match its microthread interleave chunk")
    validate_fft_launch_granule(plan.simd_lanes, spread_across_units=spread_across_units)
    slots = plan.max_uthread // workers
    per_chunk = chunk // workers
    padded = spread_across_units and slots < per_chunk
    if spread_across_units:
        slots_per_chunk = min(slots, per_chunk)
        stripes = max(1, plan.max_uthread // chunk)
        capacity = slots_per_chunk * stripes * target.num_ndp_units
    else:
        slots_per_chunk = per_chunk
        capacity = slots
    physical_capacity = (-(-capacity // slots_per_chunk) * chunk if padded else capacity * workers)
    return FFTLaunchPlan(
        total_ffts=plan.total_uthreads // workers, workers_per_fft=workers,
        chunk_uthreads=chunk, slots_per_chunk=slots_per_chunk,
        round_fft_capacity=capacity, round_uthread_capacity=physical_capacity,
        simd_lanes=plan.simd_lanes, alignment_bytes=target.mapping_stride_bytes,
        target_units=target.num_ndp_units, padded=padded, spread_across_units=spread_across_units,
    )
