from __future__ import annotations

"""Shared infrastructure for the three GPU-derived BASELINE planners
(clfft.py, rocfft.py, vkfft.py) -- ported as faithfully as reasonably
possible from clFFT, rocFFT, and VkFFT's own real, public planning source
(not from generic descriptions of "how GPU FFTs usually work"), to give
future M2NDP-specific research a fixed, independently-inspectable
reference point: "what does a straight GPU planning philosophy produce on
this hardware, before any M2NDP-aware optimization touches it."

NON-NEGOTIABLE RULE (see the task this package was built against): faithful
GPU port > M2NDP performance. Nothing in this package may import from, or
be influenced by, `planning.search.fft_cost_model` (the fitted, M2NDP-hardware-
calibrated cost model every other planner in this project uses to rank
candidates), `planning.execution.fft_plan_cooperative`'s worker-count heuristic,
`planning.execution.fft_plan_persistent`'s persistent-workgroup model, or
`planning.execution.fft_plan_lanes`'s compute_lanes narrowing heuristic. Those are
all M2NDP-specific optimizations this baseline must stay independent of --
see each baseline module's own docstring for exactly which upstream GPU
mechanism it replicates instead.

These three baseline planners are DELIBERATELY NOT unified. clFFT is a
static specialization-table/heuristic planner. rocFFT is a two-phase
empirical-tuning planner over a generated candidate space, benchmarked on
real measured execution time. VkFFT is a deterministic register/shared-
memory scheduler. They stay three independent modules on purpose (see the
project instructions this package was built from) -- do not fold them into
one "best of all three" heuristic; that would no longer be three baselines,
it would be a fourth, new one this task explicitly forbids inventing.

Every baseline terminates at the SAME plan representation this project's
existing planners already use (`FFTCodegenPlan` for a single fused kernel,
`RecursiveFFTPlan` for anything needing more than one kernel/transpose
boundary -- see planning.core.fft_plan_core / planning.strategies.fft_plan_recursive).
Planning still decides every implementation choice before codegen ever
runs (this project's own pre-existing invariant, unchanged): a GPU-baseline
planner's whole job is to decide a *GPU-faithful* radix sequence, stage
count, and execution-mapping choice, then call the SAME low-level lowering
machinery (`layouts_for_radices`, `_build_plan`/`make_cooperative_leaf_plan`,
`make_recursive_transpose_plan`'s tree builder) every other strategy in
this project already uses to actually render that choice into a
`FFTCodegenPlan`/`RecursiveFFTPlan` -- never inventing a second lowering
path, and never letting codegen make a choice a GPU baseline was supposed
to have made already.
"""

from dataclasses import dataclass, field
from enum import Enum

from planning.execution.fft_plan_cooperative import make_cooperative_leaf_plan
from planning.execution.fft_plan_persistent import (
    make_persistent_leaf_plan,
    num_rounds,
    persistent_leaf_scratchpad_bytes,
    physical_lane_workload,
)
from planning.core.fft_plan_core import FFTCodegenPlan, MultiKernelHostPlan, pingpong_needed
from planning.strategies.fft_plan_recursive import (
    FFTLeafPlan,
    PhysicalTransposePlan,
    RecursiveFFTPlan,
    flatten_recursive_node,
)
from planning.core.target_profile import TargetProfile


class BaselineStatus(Enum):
    """What happened when a GPU-derived baseline configuration was mapped
    onto this repository's actual M2NDP plan/codegen/toolchain surface.

    REVISED 2026-09-08 (baseline fidelity/mapping audit): the original
    version of this enum had one broad `UNSUPPORTED_MAPPING` bucket that
    conflated three genuinely different situations -- a real architectural
    impossibility, a gap in this repository's own current codegen/planner
    (not the hardware), and plain resource exhaustion. That conflation was
    traced and resolved by direct inspection of the real M2NDP-Detour
    simulator source (third_party/m2ndp-detour/src/{m2ndp_config.h,
    uthread_generator.cc,register_unit.cc}), not by inference from this
    project's own FFT-planner comments -- see docs/
    gpu_baseline_hardware_mapping_audit.md for the full trace, including
    concrete global-id -> (physical NDP unit, local_uthread_id()) examples
    at 8/16/64/256 uthreads. Summary of what that trace proved and how it
    is now reflected in `map_cooperative_kernel`:

    * `get_matched_unit_id(addr) = (addr / m_stride_size) % m_num_ndp_units`
      (m2ndp_config.h) routes uthreads to physical units in periodic
      chunks of `interleave_chunk_uthreads = m_stride_size / UTHREAD_
      SPAWN_UNIT` (8, for this project's own checked-in performance/M2NDP
      config) -- a value that IS parser-configurable (`ndp_stride=` in
      m2ndp_parser.cc) but which this project's own config file never
      overrides, so 8 is a real, binding fact about the SPECIFIC target
      this whole baseline effort models, not an implementation bug.
    * The period of that routing is exactly `interleave_chunk_uthreads *
      num_ndp_units` (256 uthreads): global id `g` and `g + 256` always
      land on the SAME physical unit, with `local_uthread_id()` values
      exactly `interleave_chunk_uthreads` apart. Because M2NDP's own
      global stage barrier (`launch_parallel[Self.stage_N]()`) already
      guarantees every uthread of a stage finishes before the next stage
      starts -- confirmed by this project's own existing execution model,
      not a new assumption -- a cooperative group whose `workers_per_fft`
      is an EXACT MULTIPLE of `interleave_chunk_uthreads` COULD, in
      principle, be assembled from several of these periodic "waves" on
      one physical unit and would be numerically correct. Nothing in this
      repository's current `AddressMapping`/codegen implements that
      striped, multi-wave DRAM layout, though -- it is a real
      architectural capability with no current implementation, hence
      `UNSUPPORTED_CURRENT_CODEGEN`, not a hardware wall.
    * A `workers_per_fft` that is NEITHER a divisor NOR a multiple of
      `interleave_chunk_uthreads` cannot be assembled from any whole
      number of periodic chunks at all -- that is a genuine, proven
      `UNSUPPORTED_HARDWARE_MAPPING` for this target.

    UPDATE 2026-09-12 (worker-wave virtualization): the "exact multiple"
    case above is no longer `UNSUPPORTED_CURRENT_CODEGEN` -- see
    `map_cooperative_kernel`'s own docstring for the mechanism that closes
    it (`planning.execution.fft_plan_persistent`'s round engine, given a
    second, nested "worker wave" dimension: `workers_per_fft` LOGICAL
    workers executed as `workers_per_fft // interleave_chunk_uthreads`
    sequential passes of the physical `interleave_chunk_uthreads` workers
    over the SAME physical unit, one whole pass over every wave completing
    before the FFT advances to its next stage). `workers_per_fft` itself
    is never clamped or substituted -- exactly the GPU planner's own
    chosen value becomes `PersistentWorkgroupPlan.workers_per_fft`,
    verbatim. `UNSUPPORTED_CURRENT_CODEGEN` is kept as an enum member (a
    still-live status for a genuinely different gap -- e.g. a plan shape
    the persistent lowering itself cannot build, see `RESOURCE_INFEASIBLE`
    below for the capacity case), just no longer reached by this
    particular case.

    UPDATE 2026-09-13 (ragged-wave generalization -- see docs/
    ragged_worker_wave_generalization.md for the full investigation): the
    THIRD bullet above ("neither a divisor nor a multiple ->
    UNSUPPORTED_HARDWARE_MAPPING") has been re-examined and found to be
    WRONG, not merely incomplete. The periodic-chunk argument that makes
    an exact multiple of 8 reachable via temporal multiplexing (bullet
    two) never actually required an exact multiple at all -- it only
    required that `worker_waves(workers_per_fft, 8) = ceil(workers_per_fft
    / 8)` sequential waves of 8 physical lanes be assembled, with the last
    wave's surplus physical lanes (`workers_per_fft` not evenly dividing
    8) simply computing a `logical_worker_id >= workers_per_fft` that
    matches no dispatch branch in the generated code (`codegen.
    fft_persistent_codegen._emit_worker_dispatch`'s own `if/elif` chain
    has no trailing `else`) -- a safe, inert dummy lane that runs the
    SAME compiled function as every active lane in its group (so it
    reaches every barrier that function reaches) but executes zero FFT
    arithmetic and touches zero scratchpad addresses. There is therefore
    NO `workers_per_fft` value this target's own address interleaving
    makes impossible -- `UNSUPPORTED_HARDWARE_MAPPING` for a cooperation-
    width reason no longer exists; every positive `workers_per_fft` is
    architecturally reachable, confirmed on the real M2NDP-Detour
    toolchain (not just the Python-level planner) at W=3/5/6/7/10/12/18/
    20/36 -- see that doc's own validation section. The member is kept
    in this enum (still reachable for an UNRELATED reason -- e.g. a
    negative or zero `workers_per_fft`, an actual programming error, not
    an architectural one) but `map_cooperative_kernel` no longer produces
    it for any real GPU-planner-chosen cooperation width.

    OK: the GPU planner's own chosen configuration mapped onto M2NDP with
    no substitution at all -- the resulting plan renders EXACTLY the radix
    sequence / stage grouping / worker mapping the GPU algorithm itself
    picked, nothing "helped" or rounded to a nearby M2NDP-friendlier value.

    Every other member is a *refusal*, per section 2 of the task this
    package was built from ("DO NOT silently choose a nearby better value.
    ... Instead return an explicit status..."). A refusal always keeps the
    GPU planner's own original chosen configuration in `diagnostics` (see
    BaselineResult.gpu_config) -- a failure here is itself a scientifically
    useful baseline result, never hidden.
    """

    OK = "ok"
    # The GPU algorithm itself needs functionality this FFT implementation
    # does not have AT ALL -- Rader's algorithm, Bluestein's algorithm,
    # clFFT's own true block-compute (SBCC) pipeline, or a length whose
    # prime factorization falls outside even the SOURCE GPU library's own
    # supported-radix set (so no M2NDP question is even reached). Distinct
    # from the M2NDP-specific categories below: this failure would occur
    # for ANY target, not just M2NDP.
    UNSUPPORTED_GPU_ALGORITHM = "unsupported_gpu_algorithm"
    # REVISED 2026-09-13: no longer produced for a `workers_per_fft`
    # cooperation-width reason at all -- see this enum's own "ragged-wave
    # generalization" docstring update. Every positive `workers_per_fft`
    # is reachable via `_map_worker_wave_kernel`'s ragged waves now. Kept
    # for a genuinely different reason (e.g. `workers_per_fft <= 0`, an
    # actual malformed-input case, not an architectural one).
    UNSUPPORTED_HARDWARE_MAPPING = "unsupported_hardware_mapping"
    # The M2NDP architecture CAN represent this configuration (proven via
    # the periodic-chunk argument in this enum's own docstring, or via
    # some other real but unimplemented mechanism a baseline module's own
    # docstring documents), but this repository's CURRENT planner/codegen
    # does not implement the mechanism needed -- e.g. a striped multi-wave
    # cooperative DRAM layout, or a lookup table this baseline's own
    # research pass did not finish extracting from GPU source. A future,
    # non-M2NDP-performance-driven codegen extension could close this gap
    # without contradicting anything about the GPU algorithm.
    UNSUPPORTED_CURRENT_CODEGEN = "unsupported_current_codegen"
    # The GPU planner's chosen configuration needs more of some physical
    # M2NDP resource (scratchpad bytes per cooperative group, register
    # count, concurrently active uthreads) than this target actually has,
    # honestly computed from the GPU algorithm's own resource-sizing
    # formula -- never silently shrunk to fit. Renamed from the original
    # `SCRATCHPAD_INFEASIBLE` (kept as an alias below) to cover register/
    # uthread exhaustion too, not only scratchpad.
    RESOURCE_INFEASIBLE = "resource_infeasible"
    # The mapped plan was built, but real Mojo -> llc -> M2NDP-Detour
    # compilation failed for it (see planning.diagnostics.spill_probe). Only ever
    # produced by a caller that actually invoked the real toolchain --
    # never guessed statically.
    COMPILE_FAILURE = "compile_failure"
    # The mapped plan compiled but the real toolchain reported a register
    # spill (planning.diagnostics.spill_probe.SpillProbeResult.spill_free is False).
    # Per the task's own instruction ("If preserving a GPU rule causes
    # spill, report the spill"), a baseline planner must NEVER silently
    # substitute a narrower compute_lanes or a different radix tier to
    # avoid this the way planning.execution.fft_plan_lanes does for the M2NDP-aware
    # planner -- that heuristic belongs to the OTHER planner, not this one.
    SPILLING = "spilling"
    # The mapped plan compiled and ran but crashed, hung, or otherwise
    # failed at runtime for a reason other than a register spill (e.g. a
    # simulator assertion) -- only ever produced by a caller that actually
    # invoked the real toolchain.
    RUNTIME_FAILURE = "runtime_failure"
    # The mapped plan ran but its output did not match the reference DFT
    # within tolerance.
    NUMERICAL_MISMATCH = "numerical_mismatch"


@dataclass(frozen=True)
class GPUKernelConfig:
    """The GPU planner's own chosen configuration, preserved verbatim (as
    plain data, independent of whether M2NDP could represent it) so a
    refusal's `diagnostics` always show "what the GPU algorithm actually
    wanted," per the task's own instruction to preserve the originally
    selected GPU configuration in diagnostics rather than silently
    dropping it. Every baseline populates this the same way regardless of
    outcome -- a caller comparing baselines never has to guess what a
    failed candidate would have been.

    Deliberately a free-form-ish, GPU-vocabulary record (not reshaped into
    M2NDP terms) -- `radices` is Cooley-Tukey/Stockham stage order in the
    GPU planner's own chosen sequence (may differ from a length's mere
    prime factorization: clFFT's specialization tables and rocFFT's
    permutation search both choose a specific ORDER, not just a multiset
    of factors). `extra` carries whatever fields are specific to one GPU
    family (clFFT's workGroupSize/numTransforms; rocFFT's threads_per_
    transform/transforms_per_block/half_lds/...; VkFFT's registers_per_
    thread/axis-split shape) -- see each baseline's own config dataclass,
    which every one of these dicts is built from via `dataclasses.asdict`
    equivalent, not hand-duplicated.
    """

    source: str  # "clfft" | "rocfft" | "vkfft"
    length: int
    radices: tuple[int, ...]
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselineResult:
    """One GPU-baseline planning attempt's full, honest outcome -- the
    return type every `plan_*` entry point in clfft.py/rocfft.py/vkfft.py
    produces. `status == BaselineStatus.OK` is the only case where `plan`
    is populated; every other status leaves it `None` and explains why in
    `diagnostics`, per section 2's "return an explicit status ... and
    preserve the originally selected GPU configuration in diagnostics"
    instruction -- never raise an exception for an ordinary "this GPU
    configuration doesn't map here" outcome (an actual programming error
    -- e.g. a malformed length -- still raises normally; that is not what
    this dataclass is for).
    """

    status: BaselineStatus
    gpu_config: GPUKernelConfig
    # `FFTCodegenPlan | RecursiveFFTPlan | None` -- left as `object` here
    # (rather than importing either type) so this module has zero
    # dependency on fft_plan_core/fft_plan_recursive, keeping it safe for
    # every baseline (and any future comparison script) to import cheaply.
    plan: object | None = None
    diagnostics: str = ""
    # Populated only once a caller has actually run the real toolchain
    # (planning.diagnostics.spill_probe-style probe) -- `None` means "not measured,"
    # the same "None means not probed" discipline planning.search.fft_cost_model.
    # PlanMetrics.spill_free/ndp_cycles already use, deliberately reused
    # here rather than inventing a different convention.
    ndp_cycles: int | None = None
    spill_free: bool | None = None


def unsupported(
    status: BaselineStatus, gpu_config: GPUKernelConfig, reason: str
) -> BaselineResult:
    """The one constructor every baseline's own refusal path should go
    through -- so every refusal's `diagnostics` string is built the same
    documented way (status name + the GPU config's own repr + the specific
    reason), never freehand-formatted differently per call site."""
    if status is BaselineStatus.OK:
        raise ValueError("unsupported() must not be called with BaselineStatus.OK")
    return BaselineResult(
        status=status,
        gpu_config=gpu_config,
        plan=None,
        diagnostics=(
            f"[{status.value}] {reason}\n"
            f"original GPU configuration: source={gpu_config.source} "
            f"length={gpu_config.length} radices={gpu_config.radices} "
            f"extra={gpu_config.extra}"
        ),
    )


# ---------------------------------------------------------------------------
# GPU concept -> M2NDP concept translation layer (section 2 of the task this
# package was built from). Documented here, in ONE place every baseline's own
# module docstring can point back to, rather than re-explained three times.
#
# GPU local/shared memory / LDS         -> M2NDP scratchpad
#     (bytes; target_profile.TargetProfile.spad_capacity_bytes is this
#     repo's one real number for "one NDP unit's own scratchpad, bytes" --
#     the same number every existing M2NDP-aware planner in this project
#     already budgets against, via _cap_max_uthread. A baseline's own
#     LDS-sizing FORMULA is still the GPU algorithm's own formula --
#     Section 2 forbids inventing a new NDP-specific one -- only the
#     capacity BUDGET it's checked against reuses this existing number,
#     since M2NDP has exactly one real scratchpad size regardless of which
#     baseline is asking.)
#
# GPU work item                         -> M2NDP microthread
#
# threads per transform (rocFFT TPT /
#   VkFFT's implicit per-axis thread
#   count)                              -> cooperative microthreads assigned
#                                          to one FFT (planning.
#                                          fft_plan_cooperative.
#                                          CooperationPlan.workers_per_fft --
#                                          reused as the EXECUTION vehicle
#                                          once a baseline has already
#                                          decided TPT on its own terms;
#                                          the *value itself* must still
#                                          come from the GPU algorithm's
#                                          own rule, never from
#                                          choose_workers_per_fft's
#                                          M2NDP-tuned heuristic)
#
# transforms per workgroup/block        -> FFT slots per cooperative group
#                                          (CooperationPlan.fft_slots_per_
#                                          group)
#
# total workgroup size                  -> total microthreads in that group
#                                          (workers_per_fft * fft_slots_per_
#                                          group)
#
# GPU kernel                            -> M2NDP kernel launch
#                                          (one FFTCodegenPlan)
#
# Every mapping above is MECHANICAL ONLY -- it authorizes reusing an
# existing M2NDP data structure to carry a GPU-decided number through to
# codegen, never authorizes changing what number that GPU algorithm itself
# would have chosen. See BaselineStatus's own docstring for what happens
# when a GPU concept has no such mechanical mapping at all.
# ---------------------------------------------------------------------------


def leaf_scratchpad_bytes(length: int, radices: tuple[int, ...]) -> int:
    """Scratchpad (LDS-equivalent) bytes ONE cooperating FFT slot of this
    shape needs -- mirrors planning.strategies.fft_plan_recursive._leaf_scratchpad_
    bytes' own ping-pong-bank rule exactly (same bytes-per-uthread formula
    every M2NDP leaf kernel is sized against, see fft_plan_core._build_plan),
    reimplemented here rather than imported: that helper is private to a
    different, M2NDP-specific planning module, and every GPU baseline in
    this package needs the identical byte-cost arithmetic without
    depending on an M2NDP-tuning module for it."""
    stage_count = len(radices)
    if stage_count <= 1:
        return 0
    buffers = 2 if pingpong_needed(stage_count) else 1
    return buffers * 2 * length * 4


def m2ndp_resource_complex_bytes() -> int:
    """M2NDP RESOURCE ADAPTER (section 9 of the task this was built from):
    bytes ONE complex FP32 element costs in M2NDP scratchpad once a leaf's
    own GPU-planner-chosen cooperation width needs worker-wave
    virtualization (`_map_worker_wave_kernel` above -- reached whenever
    `chunk % workers_per_fft != 0`, the common case for a real GPU
    planner's own wide `threads_per_transform`/`workers_per_fft` choice,
    e.g. VkFFT's `axisblock_threads_per_transform`). `make_persistent_leaf_
    plan` requires exactly `persistent_leaf_scratchpad_bytes(length) = 16 *
    length` bytes UNCONDITIONALLY there (two full ping-pong banks,
    independent of stage count -- see that function's own docstring), so
    this returns `persistent_leaf_scratchpad_bytes(1) = 16`: bytes per
    element, single-sourced from the SAME formula `_map_worker_wave_
    kernel`'s own plan-building call already enforces, rather than a
    second, independently-typed literal.

    Why a VkFFT-style baseline needs this at all (see docs/
    vkfft_m2ndp_scratchpad_resource_model.md): `vkfft.py`'s own shared-
    memory-sizing formulas (`max_sequence_length_shared_memory` et al.)
    are a faithful port of VkFFT's real `usedSharedMemory / complexSize`
    convention -- `complexSize = VKFFT_COMPLEX_SIZE_BYTES = 8`, a real GPU
    shared-memory single-buffer assumption. That assumption is CORRECT for
    `vkfft.plan()` (the frozen, source-faithful baseline -- a real GPU
    keeps using its own real shared-memory-reuse scheme regardless of what
    M2NDP does) but WRONG as a feasibility input for `vkfft.plan_m2ndp()`
    (the M2NDP-adapted baseline): once that baseline's own chosen
    `workers_per_fft` doesn't divide `target.interleave_chunk_uthreads`
    (the common case), the leaf that configuration maps to is actually
    lowered via `_map_worker_wave_kernel` -> `make_persistent_leaf_plan`,
    which needs 16, not 8, bytes per element -- a `plan_m2ndp` pass-count/
    axis-split decision made against the 8-byte assumption can therefore
    pick a shape (e.g. `numPasses=1`) that LOOKS scratchpad-feasible by
    VkFFT's own real formula but is actually `RESOURCE_INFEASIBLE` once
    `map_cooperative_kernel`/`_map_worker_wave_kernel` are reached --
    exactly the late, avoidable mismatch section 8 of the task this was
    built from describes (concretely: N=8192 against this project's own
    `target.spad_capacity_bytes=122880` -- `122880 // 8 = 15360 >= 8192`
    looks one-pass-feasible, but `16 * 8192 = 131072 > 122880` is not).
    `vkfft.plan_m2ndp` threads this value through as its own `complex_
    size_bytes` override (see that function's own docstring) so its pass-
    count/axis-split DECISION -- not just the eventual `map_cooperative_
    kernel` refusal -- already reflects M2NDP's real per-element cost,
    while `vkfft.plan()` keeps the real VkFFT default (8) completely
    untouched (section 11's own "never mix frozen and adapted baseline
    behavior" rule).
    """
    return persistent_leaf_scratchpad_bytes(1)


def map_cooperative_kernel(
    *,
    length: int,
    radices: tuple[int, ...],
    workers_per_fft: int,
    fft_slots_wanted: int,
    total_ffts: int,
    inverse: bool,
    inverse_scale: float | None,
    kernel_name: str,
    target: TargetProfile,
    gpu_config: GPUKernelConfig,
) -> BaselineResult:
    """The one shared "GPU cooperative-kernel decision -> M2NDP plan"
    mapping every baseline in this package uses (see the translation-layer
    comment block above): `workers_per_fft` work items cooperate on ONE
    transform (clFFT's `workgroup_size/num_transforms`, rocFFT's `threads_
    per_transform`), `fft_slots_wanted` transforms share one physical
    group (clFFT's `num_transforms`, rocFFT's `transforms_per_block`).

    Classification (see BaselineStatus's own docstring for the full
    hardware trace this is based on -- third_party/m2ndp-detour/src/
    {m2ndp_config.h,uthread_generator.cc,register_unit.cc} -- and docs/
    ragged_worker_wave_generalization.md for the 2026-09-13 investigation
    that replaced case 3 below):

    Let `chunk = target.interleave_chunk_uthreads`.

    1. `chunk % workers_per_fft == 0` (workers_per_fft divides one whole
       interleave chunk, i.e. `workers_per_fft <= chunk` and a divisor of
       it -- 1, 2, 4, 8 when chunk=8): OK via the ORIGINAL cooperative
       path -- every cooperating worker's global id falls in the SAME
       chunk, hence the SAME physical NDP unit, in a single "wave", AND
       (unlike case 2) several independent FFT slots can share that same
       physical unit concurrently (`fft_slots_wanted`). This is the
       mechanism M2NDP's own `fft_plan_cooperative.py`/
       `fft_cooperative_codegen.py` already implement -- untouched by the
       ragged-wave generalization, so every `workers_per_fft` value this
       branch already handled keeps its EXACT prior behavior (no
       regression).
    2. Every other positive `workers_per_fft` (a divisor test failure,
       whether or not it also happens to be an exact multiple of `chunk`):
       legalized via worker-wave virtualization (`_map_worker_wave_kernel`
       below, `fft_plan_persistent.worker_waves`'s ceiling-division
       generalization) -- see its own docstring. `workers_per_fft` itself
       is never clamped, rounded, or lowered; it becomes
       `PersistentWorkgroupPlan.workers_per_fft` verbatim, executed as
       `ceil(workers_per_fft / chunk)` sequential waves of the physical
       `chunk` workers over the SAME physical unit's scratchpad, the last
       (or only, when `workers_per_fft < chunk`) wave ragged whenever
       `workers_per_fft` doesn't evenly divide `chunk` -- its surplus
       physical lanes are safe dummy lanes (see `_map_worker_wave_
       kernel`'s own docstring). Architecturally sound for ANY positive
       `workers_per_fft`, not just an exact multiple of `chunk`: the
       address decoder's chunk-to-unit assignment repeats with period
       `chunk * target.num_ndp_units` regardless of how many of one
       wave's `chunk` physical lanes correspond to real logical work --
       see BaselineStatus's own docstring. Before 2026-09-12 an exact
       multiple returned `UNSUPPORTED_CURRENT_CODEGEN` (no codegen existed
       yet); before 2026-09-13 anything else in this bucket returned
       `UNSUPPORTED_HARDWARE_MAPPING` (believed, incorrectly, to be a real
       architectural wall -- see the doc above for why that belief was
       itself the limit, not the hardware). Neither status is reachable
       from a `workers_per_fft` reason any more.
    3. RESOURCE_INFEASIBLE when the GPU planner's own chosen
       `fft_slots_wanted` worth of cooperating FFT slots does not fit in
       one NDP unit's own scratchpad (`target.spad_capacity_bytes`),
       computed from `leaf_scratchpad_bytes` above -- never from M2NDP's
       own `_cap_max_uthread`'s auto-shrinking behavior, which would
       silently replace the GPU's own chosen value with a smaller,
       M2NDP-convenient one (exactly what section 2 forbids). Only
       reachable from case 1 above; case 2's own worker-wave path has a
       different, unconditional capacity check instead (`make_persistent_
       leaf_plan`'s own `16 * length` byte requirement) -- see
       `_map_worker_wave_kernel`'s own docstring for why `fft_slots_
       wanted` does not apply there.
    """
    if workers_per_fft <= 0 or fft_slots_wanted <= 0:
        return unsupported(
            BaselineStatus.UNSUPPORTED_HARDWARE_MAPPING, gpu_config,
            f"workers_per_fft={workers_per_fft} and fft_slots_wanted="
            f"{fft_slots_wanted} must both be positive",
        )
    chunk = target.interleave_chunk_uthreads
    if chunk % workers_per_fft != 0:
        return _map_worker_wave_kernel(
            length=length,
            radices=radices,
            workers_per_fft=workers_per_fft,
            total_ffts=total_ffts,
            inverse=inverse,
            inverse_scale=inverse_scale,
            kernel_name=kernel_name,
            target=target,
            gpu_config=gpu_config,
        )

    bytes_per_fft = leaf_scratchpad_bytes(length, radices)
    forced_spad_capacity = None
    if bytes_per_fft > 0:
        needed = fft_slots_wanted * bytes_per_fft
        if needed > target.spad_capacity_bytes:
            return unsupported(
                BaselineStatus.RESOURCE_INFEASIBLE, gpu_config,
                f"the GPU planner's own {fft_slots_wanted} cooperating FFT "
                f"slots of length={length} (radices={radices}) need {needed} "
                f"bytes of scratchpad (LDS-equivalent), but target."
                f"spad_capacity_bytes={target.spad_capacity_bytes} allows less "
                f"-- refusing to silently shrink this to fit",
            )
        forced_spad_capacity = needed

    try:
        built_plan = make_cooperative_leaf_plan(
            length=length,
            radices=radices,
            workers_per_fft=workers_per_fft,
            total_ffts=total_ffts,
            inverse=inverse,
            simd_lanes=8,
            kernel_name=kernel_name,
            inverse_scale=inverse_scale,
            spad_capacity_bytes=forced_spad_capacity,
            # Never M2NDP's own contention-throttle -- see this function's
            # own docstring; the only capacity check here is the one just
            # performed above, against the GPU planner's own real requirement.
            max_concurrent_scratchpad_bytes=None,
        )
    except ValueError as exc:
        return unsupported(BaselineStatus.RESOURCE_INFEASIBLE, gpu_config, str(exc))

    return BaselineResult(status=BaselineStatus.OK, gpu_config=gpu_config, plan=built_plan)


def _map_worker_wave_kernel(
    *,
    length: int,
    radices: tuple[int, ...],
    workers_per_fft: int,
    total_ffts: int,
    inverse: bool,
    inverse_scale: float | None,
    kernel_name: str,
    target: TargetProfile,
    gpu_config: GPUKernelConfig,
) -> BaselineResult:
    """Legalize ANY positive GPU-planner-chosen `workers_per_fft` that
    fails the `chunk % workers_per_fft == 0` divisor test above --
    whether it's a whole multiple of `target.interleave_chunk_uthreads`
    (16, 32, 64, ...) or "ragged" (3, 5, 6, 7, 10, 12, 18, 20, 36, ...,
    neither a divisor nor a multiple) -- by generalizing `planning.
    execution.fft_plan_persistent`'s existing round engine with a second,
    nested "worker wave" dimension, `worker_waves = ceil(workers_per_fft /
    chunk)`. See `PersistentWorkgroupPlan.workers_per_fft`'s own docstring
    for the mechanism and docs/ragged_worker_wave_generalization.md for
    the full investigation/validation; docs/gpu_baseline_hardware_mapping_
    audit.md section 4 for why the underlying periodic-chunk fact is
    architecturally sound in the first place (the address decoder's
    chunk-to-unit assignment repeats every `chunk * num_ndp_units` global
    ids, always landing back on the same physical unit) -- that fact never
    actually depended on `workers_per_fft` being an exact multiple of
    `chunk`, only on `worker_waves` sequential passes existing at all; a
    ragged last wave's surplus physical lanes are safe, inert dummy lanes
    (see `codegen.fft_persistent_codegen._emit_worker_dispatch`'s own
    docstring for why: they run the exact same compiled stage function as
    every active lane in their group, so they reach every barrier that
    function reaches, but match no dispatch branch, hence execute zero FFT
    arithmetic and touch zero scratchpad addresses).

    Every one of `total_ffts` independent length-`length` transforms
    becomes one persistent "logical block", executed across
    `make_persistent_leaf_plan`'s own rounds -- `workers_per_fft` LOGICAL
    workers cooperate on each block via `worker_waves` sequential waves of
    the physical `chunk` workers, exactly as the GPU planner chose, never
    clamped, never rounded up or down to a "convenient" nearby value.

    Why this ignores `fft_slots_wanted` (unlike the `chunk % workers_per_
    fft == 0` cooperative path above): once `workers_per_fft > chunk` (or
    even `workers_per_fft <= chunk` but not a clean divisor, which already
    consumes the group's own scalar-tail-worker slot asymmetrically), the
    physical `chunk` workers on one NDP unit are already fully consumed by
    ONE logical FFT's own wave dispatch -- there is no spare physical
    worker left on that unit to also run a second, concurrent FFT slot
    (`fft_slots_per_group` is architecturally forced to 1 in this regime,
    the same "one software group per physical unit" model `fft_plan_
    persistent.py`'s own module docstring already documents). A GPU
    planner's own `fft_slots_wanted > 1` in this regime is honored by
    running those transforms one after another (more persistent rounds)
    rather than concurrently -- an honest execution-SCHEDULE difference
    from the GPU's own concurrent placement, not a change to any
    algorithmic decision (radix sequence, worker count, per-stage
    partition are all identical either way): the task's own "GPU planner
    logical plan vs. M2NDP physical execution mapping must be clearly
    separated" instruction is exactly what licenses this -- concurrency
    is a physical scheduling choice, not part of the GPU algorithm itself.

    Capacity: `make_persistent_leaf_plan` has its own unconditional
    scratchpad check (`16 * length` bytes, two ping-pong banks -- see its
    own docstring), independent of `workers_per_fft`/waves entirely
    (scratchpad is sized by `length` alone, never duplicated per logical
    worker -- see PersistentWorkgroupPlan's own module docstring), so no
    separate capacity computation is needed here the way the cooperative
    path above needs one from `leaf_scratchpad_bytes`.
    """
    try:
        built_plan = make_persistent_leaf_plan(
            length,
            radices,
            num_logical_blocks=total_ffts,
            inverse=inverse,
            kernel_name=kernel_name,
            target=target,
            inverse_scale=inverse_scale,
            workers_per_fft=workers_per_fft,
        )
    except (ValueError, NotImplementedError) as exc:
        return unsupported(BaselineStatus.RESOURCE_INFEASIBLE, gpu_config, str(exc))

    return BaselineResult(status=BaselineStatus.OK, gpu_config=gpu_config, plan=built_plan)


@dataclass(frozen=True)
class BaselineProvenance:
    """Immutable record of exactly which upstream GPU source a baseline
    module was derived from -- required once these baselines become a
    versioned research reference (gpu-baseline-v1, see docs/
    gpu_baseline_v1_freeze.md) so a later reader/reproducer never has to
    guess which revision's behavior a given rule actually reflects.

    Pure static data, filled in by hand from the research passes already
    performed -- never a runtime GitHub query (this package must remain
    usable with no network access at all).

    `upstream_commit`: a real pinned commit SHA where the original
    research pass captured one (rocFFT: yes, via the GitHub API alongside
    the raw-content fetch); otherwise the floating branch name plus the
    fetch date the research was actually performed on (clFFT, VkFFT: both
    research passes fetched `master` without also recording a commit SHA
    -- an honest gap in the ORIGINAL research methodology, not something
    this metadata should paper over by inventing a SHA after the fact).

    `source_functions`: maps a short, stable rule name (referenced from
    this baseline's own docstrings) to the exact upstream function/struct
    it was ported from -- lets a future reader jump from "why does this
    baseline do X" straight to "which real function to re-check."
    """

    library: str
    upstream_repository: str
    upstream_commit: str
    source_files: tuple[str, ...]
    baseline_version: str
    source_functions: dict[str, str] = field(default_factory=dict)
    notes: str = ""


def wrap_leaf_as_recursive_plan(
    *, length: int, total_ffts: int, inverse: bool, built_plan: FFTCodegenPlan,
) -> RecursiveFFTPlan:
    """Wrap a single-kernel `FFTCodegenPlan` (the only kind
    `map_cooperative_kernel` ever produces) as a one-leaf `RecursiveFFTPlan`
    -- so every baseline's own top-level `plan()` entry point returns the
    SAME plan type (`RecursiveFFTPlan`) whether the GPU algorithm needed
    one kernel or several, matching what `planning.diagnostics.spill_probe.
    probe_spill_free` and `codegen.fft_transpose_codegen.
    generate_recursive_fft_kernels` both already expect as their input,
    and what every other planner in this project (fft_plan_recursive.py)
    already returns from its own top-level entry point."""
    leaf = FFTLeafPlan(m=length, r=total_ffts, kernel=built_plan)
    host = MultiKernelHostPlan(n=length, inverse=inverse, tolerance=1.0e-3)
    return RecursiveFFTPlan(n=length, inverse=inverse, root=leaf, host=host, batch=total_ffts)


# =============================================================================
# GPU LOGICAL EXECUTION vs. M2NDP PHYSICAL EXECUTION -- an explicit boundary
# (2026-09-23, "Step 2" of the task this section was built from).
#
# WHY THIS EXISTS: Step 1 (docs/logical_vs_physical_cost_model.md) fixed the
# COST MODEL's own confusion between a GPU planner's chosen `workers_per_fft`
# (LOGICAL) and M2NDP's real physical lane count (PHYSICAL) -- but a plan/
# report READER still had no single place to see both descriptions side by
# side. `RecursiveFFTPlan`/`FFTCodegenPlan` are this project's own COMMON
# codegen IR: once a GPU baseline's `map_cooperative_kernel`/`_map_worker_
# wave_kernel` call returns, `workers_per_fft` etc. are stored on
# `CooperationPlan`/`PersistentWorkgroupPlan` -- M2NDP execution vocabulary,
# not GPU planning vocabulary (no `gpu_scheme`, no `gpu_workgroup_size`
# label, no way to tell "clFFT single Stockham kernel" from "VkFFT single-
# pass axis-block kernel" just by looking at the plan tree). This section
# reconstructs BOTH descriptions from that one already-built plan tree --
# deliberately a RECONSTRUCTION, not a new field threaded through all four
# baseline modules' own recursive builders (see docs/
# gpu_m2ndp_execution_boundary.md's own "G. Remaining mismatches" section
# for exactly what this reconstruction cannot recover: per-non-root-leaf
# GPU-native metadata like VkFFT's own `upload_id`/`register_boost`, which
# `BaselineResult.gpu_config` only ever captured for the ROOT node).
#
# The reconstruction is provably faithful for every field that DOES survive
# onto the plan tree, because `map_cooperative_kernel`/`_map_worker_wave_
# kernel` NEVER clamp or round a GPU planner's own `workers_per_fft`/
# `fft_slots_wanted` choice (see either function's own docstring) --
# `CooperationPlan.workers_per_fft`/`PersistentWorkgroupPlan.workers_per_fft`
# on the built plan ARE, verbatim, the GPU planner's own logical choice.
# =============================================================================


@dataclass(frozen=True)
class GPULogicalKernelPlan:
    """One kernel/pass in the ORIGINAL GPU library's own execution model --
    descriptive only (section 4 of the task this was built from: "these
    fields must NOT automatically control M2NDP physical scheduling," and
    indeed nothing in this package reads a `GPULogicalKernelPlan` back
    into a planning decision -- it is built AFTER a plan already exists,
    purely for reporting/comparison).

    `gpu_threads_per_transform`/`gpu_transforms_per_workgroup`: GPU
    vocabulary (clFFT's `workgroup_size // num_transforms` and
    `num_transforms`; rocFFT's `threads_per_transform`/`transforms_per_
    block`; VkFFT's `axisblock_threads_per_transform`/the AxisBlockSplitter
    batch) -- `None` for a transpose kernel (PRE/MIDDLE/POST), which is not
    a "transform" in this sense at all (see `gpu_transpose_role`).
    `gpu_transforms_per_workgroup` (only) is ALSO `None` for an FFT kernel
    lowered via M2NDP's persistent/worker-wave-virtualization path: the
    GPU planner's own chosen value there is discarded before the M2NDP
    plan is even built (`gpu_baseline.common._map_worker_wave_kernel`
    forces exactly one FFT slot per physical group, architecturally, and
    never records what the GPU's own `fft_slots_wanted` had asked for) --
    genuinely NOT RECOVERABLE from the plan tree, not merely omitted here;
    see `reconstruct_gpu_logical_plan`'s own comment at that branch.
    `gpu_threads_per_transform` is always recoverable (the GPU's own
    cooperation-width choice is never clamped -- see this section's own
    top docstring).

    `gpu_shared_memory_bytes`: this kernel's own one workgroup/group's LDS-
    equivalent footprint (`gpu_baseline.common`'s own documented GPU-LDS ->
    M2NDP-scratchpad translation -- the byte NUMBER a real GPU and this
    plan's own M2NDP scratchpad allocation agree on, even though the two
    architectures reach it differently, e.g. persistent's own forced two-
    ping-pong-bank scheme vs. a real GPU's in-place shared-memory reuse --
    see docs/vkfft_m2ndp_scratchpad_resource_model.md for exactly where
    that difference matters).

    `gpu_transpose_role`: `"PRE"`/`"MIDDLE"`/`"POST"` for a transpose
    kernel (derived from this project's own `kernel_name` convention --
    every baseline's transpose-kernel builder names it `..Pre{idx}`/
    `..Mid{idx}`/`..Post{idx}`, see e.g. `vkfft._recursive_result`), `None`
    for an FFT kernel.

    `gpu_fused_operations`: descriptive tags only (`"twiddle"` when a
    transpose kernel also applies a cross-block twiddle -- clFFT/VkFFT's
    own `fft_3StepTwiddle`/large-twiddle-table mechanism; `"large_twiddle"`
    when an FFT kernel's own last stage fuses one -- see `LargeTwiddlePlan`).
    """

    planner_name: str
    gpu_scheme: str
    gpu_kernel_id: int
    pass_id: int
    fft_length: int
    radix_sequence: tuple[int, ...]
    gpu_workgroup_size: int | None
    gpu_threads_per_transform: int | None
    gpu_transforms_per_workgroup: int | None
    gpu_shared_memory_bytes: int
    gpu_fused_operations: tuple[str, ...]
    gpu_transpose_role: str | None
    gpu_global_read_bytes: int
    gpu_global_write_bytes: int
    # ADDED for P1.2 (rocFFT half_lds audit -- see docs/rocfft_half_lds_
    # classification.md). `gpu_memory_optimization`: a short name for a
    # GPU-side memory-usage optimization this kernel's own config applied
    # (currently only `"half_lds"`, rocFFT-specific) that materially
    # changes the GPU's OWN LDS byte accounting -- `None` for every other
    # baseline/kernel (no such optimization tracked). `gpu_logical_
    # shared_memory_bytes`: the GPU's own byte accounting UNDER that
    # optimization (e.g. `rocfft.gpu_logical_lds_bytes`'s own result,
    # read back from `gpu_config.extra` -- single-sourced, never
    # recomputed here) -- `None` when `gpu_memory_optimization is None`
    # (nothing to report separately from `gpu_shared_memory_bytes`).
    # `m2ndp_memory_optimization_status`: one of `"EXACT_EQUIVALENT"` /
    # `"M2NDP_ADAPTATION"` / `"NO_EQUIVALENT"` / `"UNSUPPORTED"` -- whether
    # M2NDP's OWN scratchpad model has a corresponding mechanism. Today
    # this is always `"NO_EQUIVALENT"` when `gpu_memory_optimization` is
    # set (M2NDP's own scratchpad-halving mechanism, `pingpong_needed`, is
    # keyed on stage count, a completely different axis from rocFFT's own
    # per-batch LDS halving -- see `rocfft.gpu_logical_lds_bytes`'s own
    # docstring) -- `gpu_shared_memory_bytes` (M2NDP's REAL scratchpad
    # allocation, what resource feasibility actually uses) is completely
    # unaffected by this field either way.
    gpu_memory_optimization: str | None = None
    gpu_logical_shared_memory_bytes: int | None = None
    m2ndp_memory_optimization_status: str | None = None


def _transpose_role_from_kernel_name(kernel_name: str) -> str:
    lowered = kernel_name.lower()
    for tag, role in (("pre", "PRE"), ("mid", "MIDDLE"), ("post", "POST")):
        if tag in lowered:
            return role
    raise ValueError(
        f"kernel_name={kernel_name!r} does not contain a recognized Pre/Mid/Post "
        f"tag -- every transpose-kernel builder in this package's own baselines "
        f"names its PRE/MIDDLE/POST kernels this way (see e.g. vkfft."
        f"_recursive_result); an untagged name means this reconstruction cannot "
        f"determine the transpose's own role"
    )


def _gpu_scheme_label(gpu_config: GPUKernelConfig) -> str:
    """A short, human-readable scheme label from this baseline's own root
    `GPUKernelConfig` -- e.g. `"vkfft: num_passes=2 is_pow2=True"`,
    `"clfft-m2ndp: decomposition=large1D_4step"`. Every baseline's own
    `extra` dict already carries this information (populated at each
    baseline's own top-level `plan`/`plan_m2ndp` entry point) -- this just
    formats it consistently instead of a caller reaching into `extra` by
    hand with baseline-specific key knowledge."""
    parts = [f"{k}={v}" for k, v in gpu_config.extra.items()]
    return f"{gpu_config.source}: " + (", ".join(parts) if parts else "(no scheme metadata)")


def reconstruct_gpu_logical_plan(
    result: BaselineResult,
) -> tuple[GPULogicalKernelPlan, ...]:
    """Walk `result.plan` (already built) and reconstruct one
    `GPULogicalKernelPlan` per kernel, in execution order -- see this
    section's own top docstring for why this is a RECONSTRUCTION (from
    `CooperationPlan`/`PersistentWorkgroupPlan`/stage radices/host byte
    counts already on the plan) rather than a value threaded through from
    each baseline's own recursive builder, and what that costs (some per-
    leaf GPU-native metadata, e.g. VkFFT's own per-leaf `upload_id`, is not
    recoverable this way -- only the ROOT `gpu_config` survives).
    `result.status` must be `BaselineStatus.OK` (a refusal has no plan to
    walk)."""
    if result.status is not BaselineStatus.OK:
        raise ValueError(
            f"reconstruct_gpu_logical_plan requires status=OK, got {result.status}"
        )
    assert result.plan is not None
    scheme = _gpu_scheme_label(result.gpu_config)
    planner_name = result.gpu_config.source

    entries: list[GPULogicalKernelPlan] = []
    for kernel_id, node in enumerate(flatten_recursive_node(result.plan.root)):
        if isinstance(node, PhysicalTransposePlan):
            bytes_moved = node.replica_count * node.rows * node.cols * 2 * 4
            entries.append(
                GPULogicalKernelPlan(
                    planner_name=planner_name, gpu_scheme=scheme, gpu_kernel_id=kernel_id,
                    pass_id=kernel_id, fft_length=node.rows * node.cols,
                    radix_sequence=(), gpu_workgroup_size=None,
                    gpu_threads_per_transform=None, gpu_transforms_per_workgroup=None,
                    gpu_shared_memory_bytes=node.scratchpad_elements * 4,
                    gpu_fused_operations=("twiddle",) if node.twiddle_modulus is not None else (),
                    gpu_transpose_role=_transpose_role_from_kernel_name(node.kernel_name),
                    gpu_global_read_bytes=bytes_moved, gpu_global_write_bytes=bytes_moved,
                )
            )
            continue

        assert isinstance(node, FFTCodegenPlan)
        radices = tuple(stage.radix for stage in node.stages)
        if node.cooperation is not None:
            threads_per_transform = node.cooperation.workers_per_fft
            transforms_per_workgroup = node.cooperation.fft_slots_per_group
        elif node.persistent is not None:
            # `threads_per_transform`: the GPU planner's own LOGICAL
            # choice, verbatim -- see this section's own top docstring.
            #
            # `transforms_per_workgroup`: DELIBERATELY `None`, not `1`.
            # `_map_worker_wave_kernel` (this module, reached whenever
            # cooperation width needs worker-wave virtualization) DISCARDS
            # the GPU planner's own `fft_slots_wanted`/`num_transforms`
            # choice entirely before building the M2NDP plan at all (see
            # that function's own docstring: "fft_slots_per_group is
            # architecturally forced to 1 in this regime... a GPU
            # planner's own fft_slots_wanted > 1 in this regime is honored
            # by running those transforms one after another"). The GPU's
            # own intended value is therefore NOT RECOVERABLE from the
            # plan tree at all here -- reporting `1` would silently
            # substitute M2NDP's own forced PHYSICAL value for the GPU's
            # own LOGICAL intent, exactly the conflation section 7 of the
            # task this was built from forbids ("do NOT change the GPU
            # logical plan... to fit M2NDP"). See `M2NDPPhysicalKernel
            # Execution.fft_slots_per_group` (always `1` here) for the
            # M2NDP-physical side of this same fact, and docs/
            # gpu_m2ndp_execution_boundary.md's own "G. Remaining
            # mismatches" for this exact gap.
            threads_per_transform = node.persistent.workers_per_fft
            transforms_per_workgroup = None
        else:
            # Non-cooperative: one physical microthread owns one whole
            # sub-FFT outright -- GPU vocabulary's "one work-item per
            # transform," `max_uthread` transforms sharing physical launch
            # width (not a single "workgroup" in the GPU sense, but the
            # closest available count of co-resident transforms).
            threads_per_transform = 1
            transforms_per_workgroup = node.max_uthread
        workgroup_size = (
            threads_per_transform * transforms_per_workgroup
            if transforms_per_workgroup is not None else None
        )
        shared_bytes = sum(buf.elements for buf in node.scratchpad_buffers) * 4
        bytes_moved = node.host.total_elems * 2 * 4
        # P1.2 (rocFFT half_lds -- see docs/rocfft_half_lds_classification.md):
        # only ever present in `result.gpu_config.extra` for a rocFFT-
        # sourced root config (rocfft.py's own `_map_config`) -- `None`
        # for every other baseline/kernel. rocfft.py's own single-kernel-
        # leaf scope (see that module's own SCOPE LIMIT) means there is
        # always exactly one kernel, so this ROOT-level metadata IS this
        # one kernel's own metadata -- no per-leaf-loss the way VkFFT's
        # own upload_id has (see this section's own top docstring).
        half_lds = result.gpu_config.extra.get("half_lds")
        memory_optimization = "half_lds" if half_lds else None
        gpu_logical_shared_bytes = (
            result.gpu_config.extra.get("gpu_logical_lds_bytes") if half_lds else None
        )
        memory_optimization_status = (
            result.gpu_config.extra.get("m2ndp_half_lds_status") if half_lds else None
        )
        entries.append(
            GPULogicalKernelPlan(
                planner_name=planner_name, gpu_scheme=scheme, gpu_kernel_id=kernel_id,
                pass_id=kernel_id, fft_length=node.length, radix_sequence=radices,
                gpu_workgroup_size=workgroup_size,
                gpu_threads_per_transform=threads_per_transform,
                gpu_transforms_per_workgroup=transforms_per_workgroup,
                gpu_shared_memory_bytes=shared_bytes,
                gpu_fused_operations=("large_twiddle",) if node.large_twiddle is not None else (),
                gpu_transpose_role=None,
                gpu_global_read_bytes=bytes_moved, gpu_global_write_bytes=bytes_moved,
                gpu_memory_optimization=memory_optimization,
                gpu_logical_shared_memory_bytes=gpu_logical_shared_bytes,
                m2ndp_memory_optimization_status=memory_optimization_status,
            )
        )
    return tuple(entries)


@dataclass(frozen=True)
class M2NDPPhysicalKernelExecution:
    """One kernel's own ACTUAL M2NDP physical execution shape -- the
    counterpart `GPULogicalKernelPlan` is deliberately never merged with
    (section 9 of the task this was built from: "Do not merge
    corresponding GPU/M2NDP fields into one value").

    `strategy`: `"non_cooperative"` | `"cooperative"` | `"persistent"` |
    `"transpose"` -- see this module's own `map_cooperative_kernel`/`_map_
    worker_wave_kernel` for how a GPU-chosen cooperation width resolves to
    one of the first three (never invented separately here; this field is
    read straight off which of `node.cooperation`/`node.persistent` is
    set, or neither).

    `logical_workers_per_fft`/`physical_workers_per_group`: `None` for
    `"transpose"`/`"non_cooperative"` (no cooperation-width concept
    applies). For `"cooperative"`, the two are ALWAYS EQUAL (cooperative
    `workers_per_fft` is, by construction, a divisor of `target.
    interleave_chunk_uthreads` -- see `fft_plan_cooperative.worker_
    candidates_per_fft`'s own docstring -- so logical and physical
    coincide, no virtualization needed). For `"persistent"`, `logical_
    workers_per_fft` is the GPU planner's own unclamped choice and
    `physical_workers_per_group` is always `target.interleave_chunk_
    uthreads` (8) -- the two DIFFER exactly when the GPU planner asked for
    more cooperation width than M2NDP can run simultaneously, which is
    the whole reason `"persistent"` (worker-wave virtualization) exists.

    `stage_physical_work`: one human-readable line per FFT stage
    summarizing its own physical-lane batch distribution (`persistent`
    only -- built via `fft_plan_persistent.physical_lane_workload`, the
    SAME function `fft_cost_model.compute_stage_metrics` now scores
    against, see docs/logical_vs_physical_cost_model.md). Empty for every
    other strategy (a cooperative/non-cooperative stage's own batch split
    is comparatively simple -- see `CooperationPlan`'s own module note --
    and already summarized by `fft_slots_per_group` alone).
    """

    strategy: str
    gpu_kernel_id: int
    logical_workers_per_fft: int | None
    physical_workers_per_group: int | None
    fft_slots_per_group: int | None
    logical_worker_groups: int | None
    stage_launches: int | None
    persistent_rounds: int | None
    stage_physical_work: tuple[str, ...]
    scratchpad_bytes: int
    lowering_mode: str | None
    needs_round_split: bool | None


def reconstruct_m2ndp_physical_plan(
    result: BaselineResult, *, target: TargetProfile,
) -> tuple[M2NDPPhysicalKernelExecution, ...]:
    """The M2NDP-physical-execution counterpart to `reconstruct_gpu_
    logical_plan` -- same kernel order, same `gpu_kernel_id` indexing, so
    a caller (`format_gpu_m2ndp_execution_report`) can zip the two
    sequences by index without any further lookup."""
    if result.status is not BaselineStatus.OK:
        raise ValueError(
            f"reconstruct_m2ndp_physical_plan requires status=OK, got {result.status}"
        )
    assert result.plan is not None

    entries: list[M2NDPPhysicalKernelExecution] = []
    for kernel_id, node in enumerate(flatten_recursive_node(result.plan.root)):
        if isinstance(node, PhysicalTransposePlan):
            entries.append(
                M2NDPPhysicalKernelExecution(
                    strategy="transpose", gpu_kernel_id=kernel_id,
                    logical_workers_per_fft=None, physical_workers_per_group=None,
                    fft_slots_per_group=None, logical_worker_groups=None,
                    stage_launches=None, persistent_rounds=None, stage_physical_work=(),
                    scratchpad_bytes=node.scratchpad_elements * 4, lowering_mode=None,
                    needs_round_split=node.needs_round_split,
                )
            )
            continue

        assert isinstance(node, FFTCodegenPlan)
        scratchpad_bytes = sum(buf.elements for buf in node.scratchpad_buffers) * 4
        if node.cooperation is not None:
            entries.append(
                M2NDPPhysicalKernelExecution(
                    strategy="cooperative", gpu_kernel_id=kernel_id,
                    logical_workers_per_fft=node.cooperation.workers_per_fft,
                    physical_workers_per_group=node.cooperation.workers_per_fft,
                    fft_slots_per_group=node.cooperation.fft_slots_per_group,
                    logical_worker_groups=1, stage_launches=1, persistent_rounds=None,
                    stage_physical_work=(), scratchpad_bytes=scratchpad_bytes,
                    lowering_mode=None, needs_round_split=None,
                )
            )
        elif node.persistent is not None:
            pw = node.persistent
            num_logical_blocks = node.host.total_elems // node.length
            rounds = num_rounds(num_logical_blocks, pw.software_group_count)
            stage_lines: list[str] = []
            for stage in node.stages:
                if stage.persistent_vector_batches is None:
                    continue
                workload = physical_lane_workload(
                    stage.persistent_vector_batches, stage.persistent_scalar_batches or (),
                    workers_per_fft=pw.workers_per_fft, workers_per_group=pw.workers_per_group,
                )
                stage_lines.append(
                    f"stage {stage.stage_id} (radix={stage.radix}): "
                    f"physical_lane_batches={[len(l) for l in workload.physical_lane_batches]} "
                    f"active_physical_lanes={workload.active_physical_lanes} "
                    f"max_physical_work_per_lane={workload.max_batches_per_lane}"
                )
            entries.append(
                M2NDPPhysicalKernelExecution(
                    strategy="persistent", gpu_kernel_id=kernel_id,
                    logical_workers_per_fft=pw.logical_workers_per_fft,
                    physical_workers_per_group=pw.physical_workers_per_group,
                    fft_slots_per_group=1, logical_worker_groups=pw.logical_worker_groups,
                    stage_launches=pw.stage_launches, persistent_rounds=rounds,
                    stage_physical_work=tuple(stage_lines), scratchpad_bytes=scratchpad_bytes,
                    lowering_mode=pw.lowering_mode, needs_round_split=None,
                )
            )
        else:
            entries.append(
                M2NDPPhysicalKernelExecution(
                    strategy="non_cooperative", gpu_kernel_id=kernel_id,
                    logical_workers_per_fft=None, physical_workers_per_group=None,
                    fft_slots_per_group=node.max_uthread, logical_worker_groups=None,
                    stage_launches=1, persistent_rounds=None, stage_physical_work=(),
                    scratchpad_bytes=scratchpad_bytes, lowering_mode=None,
                    needs_round_split=None,
                )
            )
    return tuple(entries)


def format_gpu_m2ndp_execution_report(
    result: BaselineResult, *, target: TargetProfile,
) -> str:
    """Unified GPU-logical / M2NDP-physical execution report for one
    `BaselineResult` -- section 9 of the task this was built from. Two
    clearly separated sections per kernel; no field is ever merged across
    them (a reader sees the GPU planner's own choice and M2NDP's own
    physical realization of it side by side, never collapsed into one
    number). `result.status` must be `BaselineStatus.OK`."""
    gpu_plan = reconstruct_gpu_logical_plan(result)
    m2ndp_plan = reconstruct_m2ndp_physical_plan(result, target=target)
    assert len(gpu_plan) == len(m2ndp_plan)

    lines: list[str] = []
    lines.append(f"BASELINE RESULT: {result.gpu_config.source} length={result.gpu_config.length}")
    lines.append(f"status: {result.status.value}")
    lines.append("")

    for gpu_kernel, m2ndp_kernel in zip(gpu_plan, m2ndp_plan):
        lines.append(f"{'=' * 50}")
        lines.append(
            f"KERNEL {gpu_kernel.gpu_kernel_id}"
            + (f" ({gpu_kernel.gpu_transpose_role} transpose)" if gpu_kernel.gpu_transpose_role else " (FFT)")
        )
        lines.append(f"{'=' * 50}")
        lines.append("-- GPU LOGICAL EXECUTION --")
        lines.append(f"  planner:                       {gpu_kernel.planner_name}")
        lines.append(f"  scheme:                        {gpu_kernel.gpu_scheme}")
        lines.append(f"  pass_id:                       {gpu_kernel.pass_id}")
        lines.append(f"  fft_length:                    {gpu_kernel.fft_length}")
        lines.append(f"  radices:                       {gpu_kernel.radix_sequence}")
        lines.append(f"  gpu_workgroup_size:            {gpu_kernel.gpu_workgroup_size}")
        lines.append(f"  gpu_threads_per_transform:     {gpu_kernel.gpu_threads_per_transform}")
        lines.append(f"  gpu_transforms_per_workgroup:  {gpu_kernel.gpu_transforms_per_workgroup}")
        lines.append(f"  gpu_shared_memory_bytes:       {gpu_kernel.gpu_shared_memory_bytes}")
        lines.append(f"  gpu_fused_operations:          {gpu_kernel.gpu_fused_operations}")
        lines.append(f"  gpu_transpose_role:            {gpu_kernel.gpu_transpose_role}")
        lines.append(f"  gpu_global_read_bytes:         {gpu_kernel.gpu_global_read_bytes}")
        lines.append(f"  gpu_global_write_bytes:        {gpu_kernel.gpu_global_write_bytes}")
        if gpu_kernel.gpu_memory_optimization is not None:
            # P1.2 (rocFFT half_lds) -- section format matches the task's
            # own worked example exactly: GPU logical value, then M2NDP's
            # own equivalence status, kept as separate lines from (never
            # merged into) `gpu_shared_memory_bytes`/`scratchpad_bytes`.
            lines.append(f"  gpu_memory_optimization:       {gpu_kernel.gpu_memory_optimization}")
            lines.append(f"  gpu_logical_shared_memory_bytes: {gpu_kernel.gpu_logical_shared_memory_bytes}")
            lines.append(f"  m2ndp_memory_optimization_status: {gpu_kernel.m2ndp_memory_optimization_status}")
        lines.append("")
        lines.append("-- M2NDP PHYSICAL EXECUTION --")
        lines.append(f"  strategy:                      {m2ndp_kernel.strategy}")
        lines.append(f"  logical_workers_per_fft:       {m2ndp_kernel.logical_workers_per_fft}")
        lines.append(f"  physical_workers_per_group:    {m2ndp_kernel.physical_workers_per_group}")
        lines.append(f"  fft_slots_per_group:           {m2ndp_kernel.fft_slots_per_group}")
        lines.append(f"  logical_worker_groups:         {m2ndp_kernel.logical_worker_groups}")
        lines.append(f"  stage_launches:                {m2ndp_kernel.stage_launches}")
        lines.append(f"  persistent_rounds:             {m2ndp_kernel.persistent_rounds}")
        lines.append(f"  lowering_mode:                 {m2ndp_kernel.lowering_mode}")
        lines.append(f"  scratchpad_bytes:              {m2ndp_kernel.scratchpad_bytes}")
        lines.append(f"  needs_round_split:             {m2ndp_kernel.needs_round_split}")
        for stage_line in m2ndp_kernel.stage_physical_work:
            lines.append(f"    {stage_line}")
        lines.append("")

    lines.append(f"{'=' * 50}")
    lines.append("OVERALL")
    lines.append(f"{'=' * 50}")
    lines.append(f"  kernel_count:                  {len(gpu_plan)}")
    lines.append(
        f"  fft_kernels:                   "
        f"{sum(1 for k in gpu_plan if k.gpu_transpose_role is None)}"
    )
    lines.append(
        f"  transpose_kernels:             "
        f"{sum(1 for k in gpu_plan if k.gpu_transpose_role is not None)}"
    )
    lines.append(f"  resource feasibility:          {result.status.value}")
    lines.append(f"  spill_free:                    {result.spill_free}")
    lines.append(f"  ndp_cycles:                    {result.ndp_cycles}")
    return "\n".join(lines)
