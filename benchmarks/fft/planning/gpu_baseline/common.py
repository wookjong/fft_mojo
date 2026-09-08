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
from planning.core.fft_plan_core import FFTCodegenPlan, MultiKernelHostPlan, pingpong_needed
from planning.strategies.fft_plan_recursive import FFTLeafPlan, RecursiveFFTPlan
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
    # Proven impossible on this M2NDP target's own address-interleaving
    # architecture (see this enum's own docstring) -- a `workers_per_fft`
    # that is neither a divisor nor a multiple of `target.
    # interleave_chunk_uthreads`. Traced from real simulator source, not
    # inferred from this project's own FFT-planner comments.
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
    {m2ndp_config.h,uthread_generator.cc,register_unit.cc}):

    Let `chunk = target.interleave_chunk_uthreads`.

    1. `chunk % workers_per_fft == 0` (workers_per_fft divides one whole
       interleave chunk): OK -- every cooperating worker's global id falls
       in the SAME chunk, hence the SAME physical NDP unit, in a single
       "wave". This is the mechanism M2NDP's own `fft_plan_cooperative.py`/
       `fft_cooperative_codegen.py` already implement.
    2. `workers_per_fft % chunk == 0` (workers_per_fft is a whole multiple
       of one chunk, e.g. 16, 32, 64 when chunk=8): UNSUPPORTED_CURRENT_
       CODEGEN. Proven architecturally reachable (the address decoder's
       chunk-to-unit assignment repeats with period `chunk *
       target.num_ndp_units`, so global ids `g` and `g + chunk*num_ndp_
       units` always land on the same physical unit with local_uthread_id()
       values exactly `chunk` apart -- see BaselineStatus's own docstring),
       but no `AddressMapping` kind in fft_plan_core.py, and no codegen in
       this repository, implements the striped/multi-wave DRAM layout a
       cooperative group built this way would need. Not implemented here
       either (section 7 of the task this package was built from: this
       audit reclassifies failures, it does not add new M2NDP-specific
       codegen).
    3. Neither of the above (e.g. workers_per_fft=15 with chunk=8):
       UNSUPPORTED_HARDWARE_MAPPING. No whole number of periodic chunks
       can ever produce this exact count on this target.
    4. RESOURCE_INFEASIBLE when the GPU planner's own chosen
       `fft_slots_wanted` worth of cooperating FFT slots does not fit in
       one NDP unit's own scratchpad (`target.spad_capacity_bytes`),
       computed from `leaf_scratchpad_bytes` above -- never from M2NDP's
       own `_cap_max_uthread`'s auto-shrinking behavior, which would
       silently replace the GPU's own chosen value with a smaller,
       M2NDP-convenient one (exactly what section 2 forbids).
    """
    if workers_per_fft <= 0 or fft_slots_wanted <= 0:
        return unsupported(
            BaselineStatus.UNSUPPORTED_HARDWARE_MAPPING, gpu_config,
            f"workers_per_fft={workers_per_fft} and fft_slots_wanted="
            f"{fft_slots_wanted} must both be positive",
        )
    chunk = target.interleave_chunk_uthreads
    if chunk % workers_per_fft != 0:
        if workers_per_fft % chunk == 0:
            return unsupported(
                BaselineStatus.UNSUPPORTED_CURRENT_CODEGEN, gpu_config,
                f"the GPU planner wants {workers_per_fft} work items cooperating "
                f"per transform -- an exact multiple of target.interleave_chunk_"
                f"uthreads={chunk}. This is architecturally reachable on M2NDP "
                f"via a striped, multi-wave DRAM layout (the address decoder's "
                f"chunk-to-unit assignment repeats every chunk*num_ndp_units="
                f"{chunk * target.num_ndp_units} global ids, always landing back "
                f"on the same physical unit -- see BaselineStatus's own hardware-"
                f"trace docstring), but no AddressMapping kind or codegen in this "
                f"repository implements that layout today.",
            )
        return unsupported(
            BaselineStatus.UNSUPPORTED_HARDWARE_MAPPING, gpu_config,
            f"the GPU planner wants {workers_per_fft} work items cooperating "
            f"per transform, which is neither a divisor nor a multiple of "
            f"target.interleave_chunk_uthreads={chunk} -- no whole number of "
            f"the M2NDP address decoder's periodic interleave chunks can ever "
            f"produce this exact count on one physical NDP unit (proven via "
            f"direct simulator source trace, see BaselineStatus's own docstring)",
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
