from __future__ import annotations

"""Controlled radix-swap ablation helpers -- see docs/radix_swap_ablation.md
for the full experiment this module exists for.

Question: after Mode C (physical-lane strip-mining, see
codegen.fft_persistent_codegen) already removed the logical-worker/wave
execution-lowering costs, GPU-derived planners (clfft/rocfft-default/vkfft)
still fail on every large-N (>=768) case in the 27-N sweep while m2ndp-native
succeeds through N=4096. Is that because of the GPU planner's own RADIX
DECOMPOSITION specifically, or some other downstream planning decision
(kernel partition, worker-count policy, execution strategy)?

This module does NOT invent a new planner, does NOT modify any existing
GPU-baseline module (clfft.py/rocfft_default.py/rocfft.py/vkfft.py) or any
M2NDP-native planning file, and does NOT rewrite an already-built
`RecursiveFFTPlan` tree after the fact. It isolates the "radix decomposition"
variable by calling the SAME two mid-pipeline entry points every existing
planner in this project already calls, with one input swapped:

* `planning.strategies.fft_plan_recursive._build_leaf_kernel` -- the exact
  function `_build_recursive_node`'s own terminal-leaf branch calls for
  every M2NDP-native leaf. Native's own downstream policy for a leaf is
  simply `persistent_leaf=False, cooperative_workers=None` (see that
  function's own docstring) -- calling it directly with a foreign `radices`
  tuple (instead of native's own `coalesce_radices(_prime_factors_
  supported(m))`) gives "GPU radix decomposition, M2NDP-native downstream"
  (kernel partition [trivially one kernel, since a whole-N radix list only
  makes sense as one leaf], stage layout, and execution strategy) with
  zero other code paths touched.

* `planning.gpu_baseline.common.map_cooperative_kernel` -- the ONE shared
  "GPU cooperative-kernel decision -> M2NDP plan" function every GPU
  baseline (clfft/rocfft_default/rocfft/vkfft) already calls, unmodified.
  Calling it directly with M2NDP-native's own radix tuple (instead of a
  GPU-derived one) but the corresponding GPU baseline's own
  `workers_per_fft` gives "M2NDP-native radix decomposition, GPU downstream
  worker-count/execution-strategy policy".

Both helpers return a `RecursiveFFTPlan` (via `wrap_leaf_as_recursive_plan`,
also unmodified/reused) that flows through the exact same production
`codegen.fft_transpose_codegen.generate_recursive_fft_kernels` ->
`planning.diagnostics.spill_probe.probe_spill_free` path every other plan in
this project already uses -- Mode C (`persistent_mode="physical"`) applies
identically, since nothing about how that path dispatches on
`stage.persistent`/`stage.cooperation` changes here.
"""

from dataclasses import dataclass

from planning.core.fft_plan_core import FFTCodegenPlan, coalesce_radices, _prime_factors_supported
from planning.core.target_profile import TargetProfile
from planning.strategies.fft_plan_recursive import FFTLeafPlan, RecursiveFFTPlan, _build_leaf_kernel
from planning.gpu_baseline.common import (
    BaselineStatus,
    GPUKernelConfig,
    map_cooperative_kernel,
    wrap_leaf_as_recursive_plan,
)


def native_radix_for_length(n: int) -> tuple[int, ...]:
    """M2NDP-native's own single-leaf radix choice for length `n` -- the
    exact expression `_build_recursive_node`'s own terminal-leaf branch
    uses (`allowed=None`, i.e. `_DEFAULT_COALESCE_ALLOWED` = radix-4-only
    composite coalescing, the one composite this codebase's own history
    -- see `coalesce_radices`'s own docstring -- has confirmed spill-free
    on real hardware in every configuration tried). Not re-derived, not
    guessed: copied verbatim from that call site.
    """
    return coalesce_radices(_prime_factors_supported(n))


def extract_gpu_radix_and_workers(plan: RecursiveFFTPlan) -> tuple[tuple[int, ...], int | None]:
    """`(radices, workers_per_fft)` off an already-built GPU-baseline
    `RecursiveFFTPlan` (from `get_plan("gpu-...", n, ...)`), read straight
    off the plan's own fields -- never re-derived from the GPU algorithm's
    formulas a second time. Every GPU baseline this ablation targets
    (control N and large N alike) is a single-leaf plan (`num_kernels ==
    1`, confirmed against the 27-N sweep's own data for every N this
    ablation uses) -- asserted, not silently assumed, so a future N outside
    that confirmed range fails loudly instead of silently reading the
    wrong stages. `workers_per_fft` is `None` when the leaf has no
    `.persistent` (a chunk-divisor cooperative leaf, or a plain leaf) --
    this ablation's own N range is persistent-only, but the accessor
    itself does not assume that.
    """
    assert isinstance(plan.root, FFTLeafPlan), (
        f"extract_gpu_radix_and_workers assumes a single-leaf GPU-baseline "
        f"plan (every N this ablation targets is confirmed single-leaf) -- "
        f"got a recursive tree instead for length {plan.n}"
    )
    kernel = plan.root.kernel
    radices = tuple(stage.radix for stage in kernel.stages)
    workers_per_fft = kernel.persistent.workers_per_fft if kernel.persistent is not None else None
    return radices, workers_per_fft


def build_gpu_radix_native_downstream(
    *,
    n: int,
    radices: tuple[int, ...],
    target: TargetProfile,
    inverse: bool = False,
    kernel_name: str = "AblationGpuRadixNativeDownstream",
) -> RecursiveFFTPlan:
    """Experiment C (the task's central ablation): the GPU planner's own
    radix decomposition, M2NDP-native's own downstream -- one variable
    (`radices`) changed from native's own default, everything else exactly
    native's own terminal-leaf call (`persistent_leaf=False,
    cooperative_workers=None`, `total_uthreads=1` matching this whole
    study's batch=1/no-replicas convention).
    """
    built = _build_leaf_kernel(
        length=n,
        radices=radices,
        total_uthreads=1,
        simd_lanes=8,
        inverse=inverse,
        inverse_scale=None,
        kernel_name=kernel_name,
        spad_capacity_bytes=target.spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=target.max_concurrent_scratchpad_bytes,
        cooperative_workers=None,
        interleave_chunk_uthreads=target.interleave_chunk_uthreads,
        persistent_leaf=False,
    )
    return wrap_leaf_as_recursive_plan(length=n, total_ffts=1, inverse=inverse, built_plan=built)


@dataclass(frozen=True)
class DownstreamRefusal:
    """`map_cooperative_kernel`'s own refusal, preserved (never silently
    swallowed) -- e.g. `RESOURCE_INFEASIBLE` if native's own radix list
    needs more scratchpad than the GPU planner's own cooperative-slot
    capacity check allows for that `workers_per_fft`."""

    status: BaselineStatus
    diagnostics: str


def build_native_radix_gpu_downstream(
    *,
    n: int,
    radices: tuple[int, ...],
    workers_per_fft: int,
    target: TargetProfile,
    inverse: bool = False,
    kernel_name: str = "AblationNativeRadixGpuDownstream",
) -> RecursiveFFTPlan | DownstreamRefusal:
    """Experiment D (reverse ablation, where structurally meaningful):
    M2NDP-native's own radix decomposition, the GPU planner's own
    downstream mapping (`map_cooperative_kernel` -- the identical shared
    function clfft.py/rocfft_default.py/rocfft.py/vkfft.py already call,
    completely unmodified). `workers_per_fft` is copied verbatim from the
    corresponding GPU-Full plan (`extract_gpu_radix_and_workers`) -- this
    ablation isolates radix alone, never re-derives the GPU's own
    worker-count policy. `fft_slots_wanted=1`/`total_ffts=1` match batch=1.
    """
    gpu_config = GPUKernelConfig(source="ablation-native-radix", length=n, radices=radices)
    result = map_cooperative_kernel(
        length=n,
        radices=radices,
        workers_per_fft=workers_per_fft,
        fft_slots_wanted=1,
        total_ffts=1,
        inverse=inverse,
        inverse_scale=None,
        kernel_name=kernel_name,
        target=target,
        gpu_config=gpu_config,
    )
    if result.status is not BaselineStatus.OK:
        return DownstreamRefusal(status=result.status, diagnostics=result.diagnostics)
    assert isinstance(result.plan, FFTCodegenPlan)
    return wrap_leaf_as_recursive_plan(length=n, total_ffts=1, inverse=inverse, built_plan=result.plan)
