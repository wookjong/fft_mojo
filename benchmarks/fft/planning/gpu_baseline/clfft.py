from __future__ import annotations

"""clFFT-style BASELINE planner.

Ported from the REAL, public clFFT source (clMathLibraries/clFFT), fetched
and read directly (not recalled from memory) as part of this baseline
port. Every table entry, formula, and threshold below carries a citation
into that source. See `docs/gpu_baseline_clfft_research.md` (the original
research report this module was built from) and `docs/
gpu_baseline_clfft_large1d_split_deepdive.md` (a 2026-09-08 follow-up that
found the large-1D split-selection algorithm -- BitScanF bit-balancing and
the literal 490-entry non-po2 table, see `choose_large1d_split` -- is
EXACT_SOURCE_SELECTION: a 100% pure, byte-for-byte-reproducible function
of length given one assumed device LDS value, not merely
ALGORITHM_EQUIVALENT as first thought) for the full verification trail,
including two places the initial draft this project started from was
WRONG and had to be corrected against the real source:

* Only length 4096 is genuinely single-precision-only in `KernelCoreSpecs`.
  1024 is identical in both precision tables; 128 and 8 exist in BOTH
  tables with DIFFERENT radix decompositions.
* For double precision, `DetermineSizes`'s mixed-prime branch halves only
  `maxWorkGroupSize`, never `leastNumPerWI` (that halving line exists in
  the real source but is commented out).

This project is FP32-only (every kernel in this repo is `FFTFP32`), so
this module only ports clFFT's SINGLE-PRECISION tables/formulas --
double-precision facts are recorded in the research report for posterity
but not implemented here (there is nothing on this target to run them on).

NON-NEGOTIABLE (see gpu_baseline/common.py's own module docstring): this
module never imports from planning.search.fft_cost_model, planning.
fft_plan_cooperative's own worker-count heuristic (choose_workers_per_fft),
planning.execution.fft_plan_persistent, or planning.execution.fft_plan_lanes. Every radix,
workgroup-size, and split decision below comes from clFFT's own real
algorithm, never from an M2NDP-fitted formula.

============================================================================
FIXED IMPLEMENTATION PARAMETERS (section 3 of the task this was built from:
"Fixed implementation parameters required only because the current code
generator needs a value must remain FIXED for all GPU baseline candidates
and must be documented.")
============================================================================

CLFFT_MAX_WGS = 256
    clFFT's own `DetermineSizes`/`GetRadices` query the REAL OpenCL
    device's `CL_DEVICE_MAX_WORK_GROUP_SIZE` capability (`params.
    fft_MaxWorkGroupSize` / `envelope.limit_WorkGroupSize`) -- M2NDP has no
    such device to query, and this project has no OpenCL runtime at all.
    256 is used here because it is clFFT's own table gating threshold
    (every `KernelCoreSpecs` entry requires a device capability >= 256 to
    even be consulted -- see the research report's section 1) and every
    single-precision table WGS value tops out at exactly 256, so this
    value maximizes how much of clFFT's own real behavior is exercised
    rather than silently falling through to `DetermineSizes` for every
    length. This is a representative "generic capable GPU" assumption,
    not an M2NDP-tuned number, and is held IDENTICAL across every clFFT
    baseline candidate this module ever plans.

CLFFT_LDS_BYTES = 32768, CLFFT_ELEM_BYTES = 8 (single-precision complex)
    clFFT's own `FFTPlan::SetEnvelope` default (`envelope.
    limit_LocalMemSize = 32768`), later clamped to the real device's
    `CL_DEVICE_LOCAL_MEM_SIZE` -- again, no real device exists here, so
    the un-clamped default (32 KiB, a typical real GPU's shared-memory
    size) is used unconditionally. This determines
    `CLFFT_LARGE1D_THRESHOLD_SINGLE = floor_po2(32768/8) = 4096`, exactly
    matching the research report's derivation.
============================================================================
"""

from dataclasses import dataclass

from planning.core.fft_plan_core import MultiKernelHostPlan
from planning.strategies.fft_plan_recursive import (
    FFTLeafPlan,
    FFTNode,
    FFTRecursiveNodePlan,
    RecursiveFFTPlan,
    _build_physical_transpose,
)
from planning.core.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile
from radix_spec import SUPPORTED_RADICES

from .common import (
    BaselineProvenance,
    BaselineResult,
    BaselineStatus,
    GPUKernelConfig,
    map_cooperative_kernel,
    unsupported,
    wrap_leaf_as_recursive_plan,
)

# ---------------------------------------------------------------------------
# Source provenance (section 7 of the baseline-freeze task this was built
# from) -- see BaselineProvenance's own docstring for why `upstream_commit`
# is a floating-branch-plus-fetch-date here, not a pinned SHA: the original
# clFFT research pass fetched `master` without also recording a commit SHA
# via the GitHub API (unlike the rocFFT research pass, which did) -- an
# honest methodology gap, not something invented after the fact.
# ---------------------------------------------------------------------------
PROVENANCE = BaselineProvenance(
    library="clFFT",
    upstream_repository="https://github.com/clMathLibraries/clFFT",
    upstream_commit="master (unpinned; fetched 2026-09-08 -- no commit SHA recorded, see docs/gpu_baseline_clfft_research.md)",
    source_files=(
        "src/library/generator.stockham.cpp",
        "src/library/generator.stockham.h",
        "src/library/plan.cpp",
        "src/library/plan.h",
        "src/library/transform.cpp",
        "src/library/action.transpose.cpp",
        "src/library/generator.transpose.cpp",
        "src/library/generator.transpose.gcn.cpp",
        "src/library/fft_binary_lookup.cpp",
    ),
    baseline_version="gpu-baseline-v1",
    source_functions={
        "SPEC_TABLE": "generator.stockham.cpp: KernelCoreSpecs<P_SINGLE>/SpecRecord (RADIX_TABLE_COMMON + P_SINGLE block)",
        "determine_sizes": "generator.stockham.cpp: FFTGeneratedStockhamAction::DetermineSizes",
        "fallback_radix_decomposition": "generator.stockham.cpp: GetRadices's own fallback loop (cRad[])",
        "get_max_1d_length": "generator.stockham.cpp: FFTPlan::GetMax1DLengthStockham",
        "is_1d_possible": "plan.h: FFTPlan::Is1DPossible",
        "choose_large1d_split": "plan.cpp: clfftBakePlan's po2 (BitScanF bit-balance + block-compute table) and non-po2 (literal 490-entry supported[] table) branches -- EXACT_SOURCE_SELECTION, see docs/gpu_baseline_clfft_large1d_split_deepdive.md",
        "_plan_leaf_or_recurse": "plan.cpp: clfftBakePlan's recursive planTX/planX/planTY/planY/planTZ construction",
    },
)

# ---------------------------------------------------------------------------
# Fixed baseline-wide parameters (see module docstring).
# ---------------------------------------------------------------------------
CLFFT_MAX_WGS = 256
CLFFT_LDS_BYTES = 32768
CLFFT_ELEM_BYTES = 8  # sizeof(std::complex<float>) -- this project is FP32-only

# generator.stockham.cpp DetermineSizes, lines 399-417: only these primes are
# ever considered when factoring a length; anything else trips the source's
# own `assert(l == 1)` -- ported here as an explicit unsupported-length error.
_DETERMINE_SIZES_PRIMES = (13, 11, 7, 5, 3, 2)

# generator.stockham.cpp's fallback radix-decomposition loop, lines 3053-3099:
# `size_t cRad[] = {13,11,10,8,7,6,5,4,3,2,1};` -- checked in this descending
# order every pass.
_FALLBACK_RADIX_ORDER = (13, 11,10, 8, 7, 6, 5, 4, 3, 2, 1)

assert set(r for r in _FALLBACK_RADIX_ORDER if r != 1) <= SUPPORTED_RADICES, (
    "every non-trivial clFFT generator radix must be one this M2NDP butterfly "
    "generator implements -- if this ever fires, clFFT's own radix vocabulary "
    "has grown past radix_spec.SUPPORTED_RADICES and needs an unsupported_"
    "radix path added, not a silent substitution"
)


class ClfftUnsupportedLengthError(Exception):
    """Raised internally when a length's prime factorization contains a
    prime clFFT's own generator does not support at all (anything outside
    {2,3,5,7,11,13}) -- exactly clFFT's own `assert(l == 1)` in
    `DetermineSizes`. Always caught by this module's own top-level `plan()`
    and converted to `BaselineStatus.UNSUPPORTED_GPU_ALGORITHM`, never
    surfaced to a caller directly."""


@dataclass(frozen=True)
class SpecEntry:
    """One `KernelCoreSpecs`/`SpecRecord` row (generator.stockham.cpp lines
    284-354), single precision only (see module docstring)."""

    workgroup_size: int
    num_transforms: int
    radices: tuple[int, ...]


# The real, single-precision `KernelCoreSpecs<P_SINGLE>` table: the
# RADIX_TABLE_COMMON block (lines 284-291, shared with double precision) plus
# the P_SINGLE-only additions (lines 322-325) -- see the research report's
# section 1 for the exact citations and the two corrections against the
# initial draft (1024 is NOT single-only; 128/8 exist in both tables with
# different radices per precision -- this table is single precision only,
# so it carries single's own 128/8 rows, not double's).
SPEC_TABLE: dict[int, SpecEntry] = {
    4096: SpecEntry(256, 1, (8, 8, 8, 8)),
    2048: SpecEntry(256, 1, (8, 8, 8, 4)),
    1024: SpecEntry(128, 1, (8, 8, 4, 4)),
    512: SpecEntry(64, 1, (8, 8, 8)),
    256: SpecEntry(64, 1, (4, 4, 4, 4)),
    128: SpecEntry(64, 4, (8, 4, 4)),
    64: SpecEntry(64, 4, (4, 4, 4)),
    32: SpecEntry(64, 16, (8, 4)),
    16: SpecEntry(64, 16, (4, 4)),
    8: SpecEntry(64, 32, (4, 2)),
    4: SpecEntry(64, 32, (2, 2)),
    2: SpecEntry(64, 64, (2,)),
}


def _floor_po2(x: int) -> int:
    if x <= 0:
        return 0
    return 1 << (x.bit_length() - 1)


def get_max_1d_length() -> int:
    """`FFTPlan::GetMax1DLengthStockham` (generator.stockham.cpp lines
    4670-4690): `FloorPo2(limit_LocalMemSize / ElementSize())`, single
    precision only -- see module docstring for why the un-clamped
    32768-byte default is used unconditionally on this target."""
    return _floor_po2(CLFFT_LDS_BYTES // CLFFT_ELEM_BYTES)


def is_1d_possible(length: int, large1d_threshold: int) -> bool:
    """`FFTPlan::Is1DPossible` (plan.h lines 608-625), verbatim."""
    if length > large1d_threshold:
        return False
    if length % 7 == 0 and length % 5 == 0 and length % 3 == 0:
        return False
    if length % 11 == 0 and (
        length % 13 == 0 or length % 7 == 0 or length % 5 == 0 or length % 3 == 0
    ):
        return False
    if length % 13 == 0 and (
        length % 11 == 0 or length % 7 == 0 or length % 5 == 0 or length % 3 == 0
    ):
        return False
    return True


def _factor_supported_primes(length: int) -> dict[int, int]:
    """`DetermineSizes`'s own factoring loop (lines 399-417): for each of
    {13,11,7,5,3,2} (checked in that order), pull out the full power of
    that prime dividing `length`. Returns `{prime: prime**exponent}` --
    the exact `primeFactorsExpanded[p]` semantics the real source uses
    (this holds p^k, not the bare exponent k). Raises
    ClfftUnsupportedLengthError (the source's own `assert(l==1)`) if
    anything is left over."""
    remaining = length
    expanded: dict[int, int] = {}
    for p in _DETERMINE_SIZES_PRIMES:
        e = 1
        while remaining % p == 0:
            remaining //= p
            e *= p
        expanded[p] = e
    if remaining != 1:
        raise ClfftUnsupportedLengthError(
            f"length={length} has a prime factor outside clFFT's own supported "
            f"set {{2,3,5,7,11,13}} (remaining={remaining}) -- DetermineSizes's "
            f"own assert(l==1) would fire here in the real source"
        )
    return expanded


# generator.stockham.cpp DetermineSizes, mixed-prime branch (lines 451-520) --
# ported verbatim, including the P_DOUBLE-only `leastNumPerWI /= 2` line kept
# here ONLY as a comment (it is commented out in the real source and is never
# applied even for double precision -- see module docstring). This project is
# single precision only, so the branch below never needs that line at all.
def _mixed_prime_least_num_per_wi(length: int, pf: dict[int, int]) -> tuple[int, int]:
    """Returns (leastNumPerWI, maxWorkGroupSize) before the elements-per-
    work-item search loop -- the per-shape table from DetermineSizes lines
    451-520, checked in the same order the real source does (each branch
    is an exact-product equality check against `length`, and clFFT's own
    `if/else if` chain means the FIRST matching branch wins -- ported as
    the same ordered chain, not a set of independent conditions)."""
    p2, p3, p5, p7, p11, p13 = pf[2], pf[3], pf[5], pf[7], pf[11], pf[13]
    if p2 * p3 == length:
        return (12, 128) if length % 12 == 0 else (6, 256)
    if p2 * p5 == length:
        return (20, 64) if length % 20 == 0 else (10, 128)
    if p2 * p7 == length:
        return (14, 64)
    if p3 * p5 == length:
        return (15, 128)
    if p3 * p7 == length:
        return (21, 128)
    if p5 * p7 == length:
        return (35, 64)
    if p2 * p3 * p5 == length:
        return (30, 64)
    if p2 * p3 * p7 == length:
        return (42, 60)
    if p2 * p5 * p7 == length:
        return (70, 36)
    if p3 * p5 * p7 == length:
        return (105, 24)
    if p2 * p11 == length:
        return (22, 128)
    if p2 * p13 == length:
        return (26, 128)
    return (210, 12)


def determine_sizes(length: int, max_wgs: int = CLFFT_MAX_WGS) -> tuple[int, int, int]:
    """`FFTGeneratedStockhamAction::DetermineSizes` (generator.stockham.cpp
    lines 388-521), single precision, ported in full: the pure-prime-power
    branches (lines 419-450), the mixed-prime branch + elements-per-work-
    item search loop (lines 451-520), verified exactly against the research
    report. Returns `(workgroup_size, num_transforms, cn_per_wi)` --
    `cn_per_wi` (elements one work item handles, == the finally-chosen
    `leastNumPerWI`/its pure-prime-power equivalent) is derived, not part
    of the real function's own return value, but is exactly what the real
    source's radix-decomposition loop (`GetRadices`'s fallback branch)
    consumes as `cnPerWI` -- see `fallback_radix_decomposition`.

    Raises ClfftUnsupportedLengthError for a length outside clFFT's own
    supported prime set.
    """
    if length == 1:
        return 64, 64, 1  # DetermineSizes lines 392-397

    pf = _factor_supported_primes(length)
    p2, p3, p5, p7, p11, p13 = pf[2], pf[3], pf[5], pf[7], pf[11], pf[13]

    if p2 == length:
        if length >= 1024:
            wgs, num_trans = min(256, max_wgs), 1
        elif length == 512:
            wgs, num_trans = 64, 1
        elif length >= 16:
            wgs, num_trans = 64, 256 // length
        else:
            wgs, num_trans = 64, 128 // length
    elif p3 == length:
        wgs = 243 if max_wgs >= 256 else 27
        num_trans = 1 if length >= 3 * wgs else (3 * wgs) // length
    elif p5 == length:
        wgs = 125 if max_wgs >= 128 else 25
        num_trans = 1 if length >= 5 * wgs else (5 * wgs) // length
    elif p7 == length:
        wgs = 49
        num_trans = 1 if length >= 7 * wgs else (7 * wgs) // length
    elif p11 == length:
        wgs = 121
        num_trans = 1 if length >= 11 * wgs else (11 * wgs) // length
    elif p13 == length:
        wgs = 169
        num_trans = 1 if length >= 13 * wgs else (13 * wgs) // length
    else:
        least_num_per_wi, max_workgroup_size = _mixed_prime_least_num_per_wi(length, pf)
        # P_DOUBLE would halve max_workgroup_size here (never leastNumPerWI --
        # see module docstring); single precision (this project) does neither.
        max_workgroup_size = min(max_workgroup_size, max_wgs)
        assert least_num_per_wi > 0 and length % least_num_per_wi == 0

        lnpi = least_num_per_wi
        while lnpi <= length:
            if length % lnpi == 0 and length // lnpi <= max_wgs:
                least_num_per_wi = lnpi
                break
            lnpi += least_num_per_wi

        num_trans = max(1, max_workgroup_size // (length // least_num_per_wi))
        wgs = num_trans * (length // least_num_per_wi)

    cn_per_wi = (num_trans * length) // wgs
    assert num_trans * length % wgs == 0, (
        f"internal error porting DetermineSizes for length={length}: "
        f"numTrans*length not evenly divisible by workGroupSize"
    )
    return wgs, num_trans, cn_per_wi


def fallback_radix_decomposition(length: int, cn_per_wi: int) -> tuple[int, ...]:
    """The real `GetRadices` fallback loop (generator.stockham.cpp lines
    3053-3099) for a length with no `SPEC_TABLE` entry (or one gated off
    by a device-capability check this baseline doesn't need -- see
    `get_radices`): greedily, pass by pass, pick the LARGEST radix from
    `{13,11,10,8,7,6,5,4,3,2,1}` such that it does not exceed `cn_per_wi`,
    evenly divides `cn_per_wi`, and evenly divides the remaining factor of
    `length`. Repeat until the remaining factor is 1."""
    radices: list[int] = []
    remaining = length
    while remaining != 1:
        chosen = None
        for rad in _FALLBACK_RADIX_ORDER:
            if rad > cn_per_wi or cn_per_wi % rad != 0:
                continue
            if remaining % rad == 0:
                chosen = rad
                break
        if chosen is None:
            raise ClfftUnsupportedLengthError(
                f"length={length}: fallback radix decomposition stuck at "
                f"remaining={remaining} cn_per_wi={cn_per_wi} -- no radix in "
                f"{_FALLBACK_RADIX_ORDER} both divides cn_per_wi and the "
                f"remaining factor"
            )
        radices.append(chosen)
        remaining //= chosen
    return tuple(r for r in radices if r != 1)


@dataclass(frozen=True)
class ClfftRadixChoice:
    radices: tuple[int, ...]
    workgroup_size: int
    num_transforms: int
    source: str  # "table" | "determine_sizes"


def get_radices(length: int, max_wgs: int = CLFFT_MAX_WGS) -> ClfftRadixChoice:
    """`FFTGeneratedStockhamAction::GetRadices`-equivalent (generator.
    stockham.cpp lines 3018-3099): table lookup (gated on `max_wgs >= 256`,
    exactly the real source's own `params.fft_MaxWorkGroupSize >= 256`
    check) with the radix sequence taken verbatim in stored array order,
    else `DetermineSizes` + the fallback greedy radix loop."""
    if max_wgs >= 256 and length in SPEC_TABLE:
        entry = SPEC_TABLE[length]
        return ClfftRadixChoice(entry.radices, entry.workgroup_size, entry.num_transforms, "table")
    wgs, num_trans, cn_per_wi = determine_sizes(length, max_wgs=max_wgs)
    radices = fallback_radix_decomposition(length, cn_per_wi)
    return ClfftRadixChoice(radices, wgs, num_trans, "determine_sizes")


# ---------------------------------------------------------------------------
# M2NDP translation layer
#
# clFFT's own "threads per transform" is `workgroup_size // num_transforms`
# (the work items cooperating on ONE transform); "transforms per workgroup"
# is `num_transforms` itself. Both feed gpu_baseline.common.
# map_cooperative_kernel, the one shared GPU-cooperative-kernel-to-M2NDP
# mapping every baseline in this package uses -- see that function's own
# docstring for the possible refusals (UNSUPPORTED_HARDWARE_MAPPING /
# UNSUPPORTED_CURRENT_CODEGEN when threads-per-transform doesn't divide
# target.interleave_chunk_uthreads; RESOURCE_INFEASIBLE when num_transforms
# worth of cooperating slots doesn't fit one NDP unit's scratchpad).
# ---------------------------------------------------------------------------


def _map_stockham_kernel(
    *,
    length: int,
    choice: ClfftRadixChoice,
    total_ffts: int,
    inverse: bool,
    inverse_scale: float | None,
    kernel_name: str,
    target: TargetProfile,
    gpu_config: GPUKernelConfig,
) -> BaselineResult:
    if choice.workgroup_size % choice.num_transforms != 0:
        return unsupported(
            BaselineStatus.UNSUPPORTED_CURRENT_CODEGEN, gpu_config,
            f"clFFT's own workgroup_size={choice.workgroup_size} is not evenly "
            f"divisible by num_transforms={choice.num_transforms} -- internal "
            f"inconsistency in the ported DetermineSizes/table result, cannot map",
        )
    workers_per_fft = choice.workgroup_size // choice.num_transforms
    return map_cooperative_kernel(
        length=length, radices=choice.radices, workers_per_fft=workers_per_fft,
        fft_slots_wanted=choice.num_transforms, total_ffts=total_ffts,
        inverse=inverse, inverse_scale=inverse_scale, kernel_name=kernel_name,
        target=target, gpu_config=gpu_config,
    )


def plan_single_kernel(
    length: int,
    *,
    total_ffts: int = 1,
    inverse: bool = False,
    kernel_name: str = "FFTClfft",
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
) -> BaselineResult:
    """One clFFT Stockham kernel for `length` (no large-1D decomposition --
    see `plan` for the top-level entry point that decides when this
    applies at all)."""
    gpu_config = GPUKernelConfig(source="clfft", length=length, radices=())
    try:
        choice = get_radices(length)
    except ClfftUnsupportedLengthError as exc:
        return unsupported(BaselineStatus.UNSUPPORTED_GPU_ALGORITHM, gpu_config, str(exc))

    gpu_config = GPUKernelConfig(
        source="clfft", length=length, radices=choice.radices,
        extra={
            "workgroup_size": choice.workgroup_size,
            "num_transforms": choice.num_transforms,
            "source": choice.source,
            "max_wgs": CLFFT_MAX_WGS,
        },
    )
    inverse_scale = (1.0 / length) if inverse else None
    result = _map_stockham_kernel(
        length=length, choice=choice, total_ffts=total_ffts, inverse=inverse,
        inverse_scale=inverse_scale, kernel_name=kernel_name, target=target,
        gpu_config=gpu_config,
    )
    if result.status is not BaselineStatus.OK:
        return result
    recursive_plan = wrap_leaf_as_recursive_plan(
        length=length, total_ffts=total_ffts, inverse=inverse, built_plan=result.plan,
    )
    return BaselineResult(status=BaselineStatus.OK, gpu_config=result.gpu_config, plan=recursive_plan)


# ---------------------------------------------------------------------------
# Large-1D decomposition (clfftBakePlan's planTX/planX/planTY/planY/planTZ --
# see the research report's section 4). Structurally identical in shape to
# this project's OWN FFTRecursiveNodePlan (PRE transpose -> near_fft ->
# MIDDLE transpose+twiddle -> far_child -> POST transpose) -- see this
# module's own note below for the correspondence -- so this baseline reuses
# that exact plan dataclass and its PhysicalTransposePlan/_build_physical_
# transpose lowering machinery (section 1D of the project audit: mechanical
# reuse of already-decided-plan-to-codegen-IR machinery), while the actual
# DECISIONS (split point, each leaf's own radix sequence) come from clFFT's
# own algorithm below, never from fft_plan_recursive's own M2NDP-specific
# _choose_recursive_split/_prime_factors_supported/coalesce_radices.
#
# Correspondence: clFFT's planX (row FFT, size clLengths[1], batched over
# clLengths[0], large1D=0 "twiddling is done in row2") is FFTRecursiveNodePlan.
# near_fft (b-point, batched r*a times); clFFT's planTY (transpose with
# large1D=N triggering fft_3StepTwiddle) is .middle_transpose; clFFT's planY
# (row FFT, size clLengths[0], recursively baked if still too large) is
# .far_child; clFFT's planTX/planTZ are .pre_transpose/.post_transpose.
# So clLengths[1] == B (near/inner), clLengths[0] == A (far/outer, the one
# that may recurse again) in this project's own (m = a*b) notation.
#
# EXACT SOURCE SELECTION (upgraded from an earlier ALGORITHM_EQUIVALENT
# approximation during the 2026-09-08 baseline-completion audit, per
# docs/gpu_baseline_clfft_large1d_split_deepdive.md's own finding that
# clFFT's split-selection is, given a stated Large1DThreshold, a 100%
# pure, byte-for-byte-reproducible function of length/precision/plan-flags
# -- BitScanF is `ctz` (count-trailing-zeros; it behaves like log2 only
# because every call site restricts it to power-of-2 arguments), and the
# non-po2 branch is a literal, fully static 490-integer ascending array
# checked by plain `%` divisibility, not a generated primality/factor
# test as the earlier approximation assumed.
# ---------------------------------------------------------------------------

# plan.cpp lines 633-710 (po2 branch): the block-compute (SBCC) literal
# column-length table, single precision -- gates on
# `length <= 262144/PrecisionWidth(precision)` (262144 for single),
# contiguous unit strides, C2C, 1-D-only, and not (no-mem-alloc AND
# in-place) -- this baseline is always C2C/1-D/unit-stride with no-mem-
# alloc off (CLFFT_REQUEST_LIB_NOMEMALLOC unset, its own real default --
# see the research report's own section 4C), so eligibility here reduces
# to exactly `length <= 262144`. 524288/1048576 are real table rows that
# are architecturally UNREACHABLE through this exact gate (confirmed dead
# code in the real source, not an omission here) -- kept anyway, as the
# literal source has them, never silently dropped.
CLFFT_BLOCK_COMPUTE_TABLE_SINGLE: dict[int, int] = {
    8192: 64, 16384: 64, 32768: 128, 65536: 256,
    131072: 64, 262144: 64, 524288: 256, 1048576: 256,
}
CLFFT_BLOCK_COMPUTE_GATE_SINGLE = 262144

# plan.cpp lines 715-749: the literal, fully static, ascending 490-integer
# "supported" array for the non-power-of-2 branch -- transcribed exactly
# from the deep-dive's own verbatim quote of the live source (counted:
# 490 entries, first=1, last=4096).
CLFFT_NON_PO2_SUPPORTED: tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 24,
    25, 26, 27, 28, 30, 32, 33, 35, 36, 39, 40, 42, 44, 45, 48, 49, 50, 52, 54,
    55, 56, 60, 63, 64, 65, 66, 70, 72, 75, 77, 78, 80, 81, 84, 88, 90, 91, 96,
    98, 99, 100, 104, 105, 108, 110, 112, 117, 120, 121, 125, 126, 128, 130, 132,
    135, 140, 143, 144, 147, 150, 154, 156, 160, 162, 165, 168, 169, 175, 176,
    180, 182, 189, 192, 195, 196, 198, 200, 208, 210, 216, 220, 224, 225, 231,
    234, 240, 242, 243, 245, 250, 252, 256, 260, 264, 270, 273, 275, 280, 286,
    288, 294, 297, 300, 308, 312, 315, 320, 324, 325, 330, 336, 338, 343, 350,
    351, 352, 360, 363, 364, 375, 378, 384, 385, 390, 392, 396, 400, 405, 416,
    420, 429, 432, 440, 441, 448, 450, 455, 462, 468, 480, 484, 486, 490, 495,
    500, 504, 507, 512, 520, 525, 528, 539, 540, 546, 550, 560, 567, 572, 576,
    585, 588, 594, 600, 605, 616, 624, 625, 630, 637, 640, 648, 650, 660, 672,
    675, 676, 686, 693, 700, 702, 704, 715, 720, 726, 728, 729, 735, 750, 756,
    768, 770, 780, 784, 792, 800, 810, 819, 825, 832, 840, 845, 847, 858, 864,
    875, 880, 882, 891, 896, 900, 910, 924, 936, 945, 960, 968, 972, 975, 980,
    990, 1000, 1001, 1008, 1014, 1024, 1029, 1040, 1050, 1053, 1056, 1078, 1080,
    1089, 1092, 1100, 1120, 1125, 1134, 1144, 1152, 1155, 1170, 1176, 1183, 1188,
    1200, 1210, 1215, 1225, 1232, 1248, 1250, 1260, 1274, 1280, 1287, 1296, 1300,
    1320, 1323, 1331, 1344, 1350, 1352, 1365, 1372, 1375, 1386, 1400, 1404, 1408,
    1430, 1440, 1452, 1456, 1458, 1470, 1485, 1500, 1512, 1521, 1536, 1540, 1560,
    1568, 1573, 1575, 1584, 1600, 1617, 1620, 1625, 1638, 1650, 1664, 1680, 1690,
    1694, 1701, 1715, 1716, 1728, 1750, 1755, 1760, 1764, 1782, 1792, 1800, 1815,
    1820, 1848, 1859, 1872, 1875, 1890, 1911, 1920, 1925, 1936, 1944, 1950, 1960,
    1980, 2000, 2002, 2016, 2025, 2028, 2048, 2058, 2079, 2080, 2100, 2106, 2112,
    2145, 2156, 2160, 2178, 2184, 2187, 2197, 2200, 2205, 2240, 2250, 2268, 2275,
    2288, 2304, 2310, 2340, 2352, 2366, 2376, 2400, 2401, 2420, 2430, 2450, 2457,
    2464, 2475, 2496, 2500, 2520, 2535, 2541, 2548, 2560, 2574, 2592, 2600, 2625,
    2640, 2646, 2662, 2673, 2688, 2695, 2700, 2704, 2730, 2744, 2750, 2772, 2800,
    2808, 2816, 2835, 2860, 2880, 2904, 2912, 2916, 2925, 2940, 2970, 3000, 3003,
    3024, 3025, 3042, 3072, 3080, 3087, 3120, 3125, 3136, 3146, 3150, 3159, 3168,
    3185, 3200, 3234, 3240, 3250, 3267, 3276, 3300, 3328, 3360, 3375, 3380, 3388,
    3402, 3430, 3432, 3456, 3465, 3500, 3510, 3520, 3528, 3549, 3564, 3575, 3584,
    3600, 3630, 3640, 3645, 3675, 3696, 3718, 3744, 3750, 3773, 3780, 3822, 3840,
    3850, 3861, 3872, 3888, 3900, 3920, 3960, 3969, 3993, 4000, 4004, 4032, 4050,
    4056, 4095, 4096,
)
assert len(CLFFT_NON_PO2_SUPPORTED) == 490
assert list(CLFFT_NON_PO2_SUPPORTED) == sorted(CLFFT_NON_PO2_SUPPORTED)


def _ceil_po2(n: int) -> int:
    """`StockhamGenerator::CeilPo2` (generator.stockham.h:90-100): smallest
    `t` with `2**t >= n`, ported directly from the real `while` loop
    (equivalent to, but kept as the same loop shape as, `(n-1).bit_length()`
    for `n > 1`, `0` for `n <= 1`)."""
    v, t = 1, 0
    while v < n:
        v <<= 1
        t += 1
    return t


def _bit_scan_f(n: int) -> int:
    """`BitScanF` (private.h:322-336): count of trailing zero bits (`ctz`)
    -- equals `log2(n)` only for a power-of-2 `n`, exactly like the real
    source (every real call site restricts arguments to powers of two)."""
    assert n != 0
    return (n & -n).bit_length() - 1


def choose_large1d_split(length: int, threshold: int) -> tuple[int, int]:
    """`clfftBakePlan`'s CLFFT_1D large-1D split selection (plan.cpp
    lines 633-771), ported EXACTLY -- see docs/
    gpu_baseline_clfft_large1d_split_deepdive.md for the full derivation.
    Returns `(a, b) = (clLengths[0], clLengths[1])`: `b` is the inner/
    near/batched-first row transform, `a = length // b` the outer/far one
    (potentially recursed further -- see this module's own near/far
    convention, unchanged).

    Assumes this baseline's own fixed defaults throughout (all real,
    documented, section-3 "fixed implementation parameters," matching
    every other assumption this whole module already makes): complex-to-
    complex, out-of-place, unit strides, single `CLFFT_1D` plan,
    `CLFFT_REQUEST_LIB_NOMEMALLOC` unset (clFFT's own real default). Under
    these, block-compute eligibility (plan.cpp:646) reduces to exactly
    `length <= 262144`.

    Raises ClfftUnsupportedLengthError only when the non-po2 branch's own
    downward scan finds no divisor at all -- mirroring a GENUINE, CONFIRMED
    gap in clFFT itself (research report section 4D): the real source
    silently leaves `clLengths[1] = 1` in this case and recurses into
    `clfftBakePlan` on the SAME length/threshold, which does not terminate
    (unbounded recursion for e.g. a large prime length). This baseline
    raises a clear, catchable error instead of reproducing that infinite
    loop -- an explicitly-documented, deliberate departure from literal
    byte-for-byte behavior for an input clFFT itself does not handle
    correctly, not a silent improvement to the ALGORITHM'S real answers.
    """
    if (length & (length - 1)) == 0:
        if length <= CLFFT_BLOCK_COMPUTE_GATE_SINGLE and length in CLFFT_BLOCK_COMPUTE_TABLE_SINGLE:
            b = CLFFT_BLOCK_COMPUTE_TABLE_SINGLE[length]
            return length // b, b
        # Not block-compute-eligible. CLFFT_REQUEST_LIB_NOMEMALLOC's own
        # half-split branch is skipped (fixed default: unset -- see this
        # function's own docstring).
        if length > threshold * threshold:
            b = length // threshold
            return length // b, b
        # The exact BitScanF bit-balancing formula (plan.cpp:695-708).
        in_1d = _bit_scan_f(threshold)  # t = log2(threshold)
        in_x = _bit_scan_f(length)      # x = log2(length)
        assert in_1d > 0
        count = in_x // in_1d
        if count * in_1d < in_x:
            count += 1
            in_1d = in_x // count
            if in_1d * count < in_x:
                in_1d += 1
        b = 1 << in_1d
        return length // b, b

    # Non-power-of-2: the literal 490-entry table + exact downward scan
    # (plan.cpp:711-770).
    max_factored_length = min(CLFFT_NON_PO2_SUPPORTED[-1], threshold)
    half_power_length = 1 << ((_ceil_po2(length) + 1) // 2)
    factored_length_start = min(half_power_length, max_factored_length)

    index_start = 0
    while CLFFT_NON_PO2_SUPPORTED[index_start] < factored_length_start:
        index_start += 1

    for i in range(index_start, 0, -1):
        candidate = CLFFT_NON_PO2_SUPPORTED[i]
        if length % candidate == 0 and is_1d_possible(candidate, threshold):
            b = candidate
            return length // b, b

    raise ClfftUnsupportedLengthError(
        f"length={length}: clFFT's own non-power-of-2 large-1D split scan (the "
        f"literal 490-entry supported[] table) found no divisor -- in the REAL "
        f"clFFT source this is a genuine, confirmed bug (clLengths[1] silently "
        f"stays 1 and clfftBakePlan recurses into itself on the same length/"
        f"threshold without terminating; see docs/"
        f"gpu_baseline_clfft_large1d_split_deepdive.md section 4D) -- this "
        f"baseline raises instead of reproducing that infinite loop"
    )


def _plan_leaf_or_recurse(
    m: int,
    r: int,
    *,
    inverse: bool,
    is_root: bool,
    node_id: list[int],
    target: TargetProfile,
    threshold: int,
) -> tuple[FFTNode | None, BaselineResult | None]:
    """Returns `(node, None)` on success or `(None, failure_result)` on any
    refusal anywhere in the tree -- a single refusal anywhere aborts the
    whole large-1D plan (there is no partial baseline result), same as a
    real clFFT bake either fully succeeds or fails outright."""
    idx = node_id[0]
    node_id[0] += 1

    if is_1d_possible(m, threshold):
        try:
            choice = get_radices(m)
        except ClfftUnsupportedLengthError as exc:
            gpu_config = GPUKernelConfig(source="clfft", length=m, radices=())
            return None, unsupported(BaselineStatus.UNSUPPORTED_GPU_ALGORITHM, gpu_config, str(exc))
        gpu_config = GPUKernelConfig(
            source="clfft", length=m, radices=choice.radices,
            extra={"workgroup_size": choice.workgroup_size, "num_transforms": choice.num_transforms},
        )
        inverse_scale = (1.0 / m) if (inverse and is_root) else None
        result = _map_stockham_kernel(
            length=m, choice=choice, total_ffts=r, inverse=inverse,
            inverse_scale=inverse_scale, kernel_name=f"FFTClfftLeaf{idx}",
            target=target, gpu_config=gpu_config,
        )
        if result.status is not BaselineStatus.OK:
            return None, result
        return FFTLeafPlan(m=m, r=r, kernel=result.plan), None

    try:
        a, b = choose_large1d_split(m, threshold)
    except ClfftUnsupportedLengthError as exc:
        gpu_config = GPUKernelConfig(source="clfft", length=m, radices=())
        return None, unsupported(BaselineStatus.UNSUPPORTED_GPU_ALGORITHM, gpu_config, str(exc))

    tile = max(1, min(8, a, b))
    pre = _build_physical_transpose(
        rows=b, cols=a, replica_count=r, tile_rows=tile, tile_cols=tile,
        twiddle_modulus=None, inverse=inverse, kernel_name=f"FFTClfftPre{idx}",
        simd_lanes=8, spad_capacity_bytes=target.spad_capacity_bytes,
        apply_inverse_scale=False,
    )

    near_node, failure = _plan_leaf_or_recurse(
        b, r * a, inverse=inverse, is_root=False, node_id=node_id,
        target=target, threshold=threshold,
    )
    if failure is not None:
        return None, failure
    assert isinstance(near_node, FFTLeafPlan), (
        "clFFT's own planX (the inner/near row transform) is never itself "
        "recursively split -- Is1DPossible(b, threshold) was already checked "
        "as part of choose_large1d_split's own legality"
    )

    middle = _build_physical_transpose(
        rows=a, cols=b, replica_count=r, tile_rows=tile, tile_cols=tile,
        twiddle_modulus=m, inverse=inverse, kernel_name=f"FFTClfftMid{idx}",
        simd_lanes=8, spad_capacity_bytes=target.spad_capacity_bytes,
        apply_inverse_scale=False,
    )

    far_node, failure = _plan_leaf_or_recurse(
        a, r * b, inverse=inverse, is_root=False, node_id=node_id,
        target=target, threshold=threshold,
    )
    if failure is not None:
        return None, failure

    post = _build_physical_transpose(
        rows=b, cols=a, replica_count=r, tile_rows=tile, tile_cols=tile,
        twiddle_modulus=None, inverse=inverse, kernel_name=f"FFTClfftPost{idx}",
        simd_lanes=8, spad_capacity_bytes=target.spad_capacity_bytes,
        apply_inverse_scale=(inverse and is_root),
    )

    return (
        FFTRecursiveNodePlan(
            m=m, r=r, a=a, b=b, pre_transpose=pre, near_fft=near_node,
            middle_transpose=middle, far_child=far_node, post_transpose=post,
        ),
        None,
    )


def plan_large1d(
    length: int,
    *,
    batch: int = 1,
    inverse: bool = False,
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
) -> BaselineResult:
    """clFFT's own large-1D (four-step Bailey) decomposition -- see this
    section's own module-level docstring for the FFTRecursiveNodePlan
    correspondence and the documented split-selection fidelity gap."""
    threshold = get_max_1d_length()
    node_id = [0]
    node, failure = _plan_leaf_or_recurse(
        length, batch, inverse=inverse, is_root=True, node_id=node_id,
        target=target, threshold=threshold,
    )
    if failure is not None:
        return failure
    assert node is not None
    gpu_config = GPUKernelConfig(
        source="clfft", length=length, radices=(),
        extra={"decomposition": "large1D_4step", "threshold": threshold},
    )
    recursive_plan = RecursiveFFTPlan(
        n=length, inverse=inverse, root=node,
        host=MultiKernelHostPlan(n=length, inverse=inverse, tolerance=1.0e-3),
        batch=batch,
    )
    return BaselineResult(status=BaselineStatus.OK, gpu_config=gpu_config, plan=recursive_plan)


def plan(
    length: int,
    *,
    batch: int = 1,
    inverse: bool = False,
    kernel_name: str = "FFTClfft",
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
) -> BaselineResult:
    """Top-level clFFT-style baseline entry point -- decides single-kernel
    vs. large-1D exactly the way `clfftBakePlan` does (section 4a-4b of the
    research report): a single Stockham kernel whenever
    `Is1DPossible(length, GetMax1DLength())`, else the large-1D pipeline."""
    threshold = get_max_1d_length()
    if is_1d_possible(length, threshold):
        return plan_single_kernel(
            length, total_ffts=batch, inverse=inverse, kernel_name=kernel_name, target=target,
        )
    return plan_large1d(length, batch=batch, inverse=inverse, target=target)
