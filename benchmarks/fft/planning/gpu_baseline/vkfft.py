from __future__ import annotations

"""Deterministic VkFFT baseline at 066a17c17068c0f11c9298d848c2976c71fad1c1.

FP32 direct leaves use VkFFTGetRegistersPerThread's ordinary precision
branch, thread-cap register scaling, VkFFTOptimizeRadixKernels, final
MinMaxRegisterCheck and first-stage swap. Prime coverage is 2/3/5/7;
register boost is fixed at one. The retained Quad table is a separate
helper and is never used for FP32 plans. Source-generated C++ fixtures
verify registers, radix order, and upload-zero axis layout.

Axis splitting and upload order follow the pinned scheduler. AxisBlock
reports logical threads per transform and transforms per workgroup;
bank-conflict X/Y swapping is recorded independently in local_size and
axis_swapped. It does not exchange the logical worker and FFT counts.

Fixed device assumptions: NVIDIA vendor policy, warp 32, aimThreads 128,
maxThreadsNum 1024, coalesced memory 32 bytes, complex FP32 8 bytes.
Shared memory capacity comes from the supplied target. The portable
backend's power-of-two grouping bound is three; CUDA-only widening is
outside this baseline's scope. plan_m2ndp substitutes the documented
M2NDP resource inputs without searching alternative GPU decisions.

Original radix choices, including 12/14/15, survive physical mapping
failures. A missing M2NDP butterfly yields UNSUPPORTED_CURRENT_CODEGEN.
Primes 11/13 and Rader/Bluestein paths remain outside the ported planner
scope and yield UNSUPPORTED_GPU_ALGORITHM with residual classification.
isGoodSequence is preserved for padded-length classification and is not
used to reject ordinary direct FFTs. No native cost or tuning heuristic
participates in the GPU planner's decisions.
"""

import math
from dataclasses import dataclass

from .vkfft_registers import fp32_base_registers, plan_direct_radices, prime_multiplicities

from planning.strategies.fft_plan_recursive import (
    FFTLeafPlan,
    FFTNode,
    FFTRecursiveNodePlan,
    RecursiveFFTPlan,
    _build_physical_transpose,
)
from planning.core.fft_plan_core import MultiKernelHostPlan
from planning.core.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile
from radix_spec import SUPPORTED_RADICES

from .logical_plan import preserve_gpu_plan
from .common import (
    BaselineProvenance,
    BaselineResult,
    BaselineStatus,
    GPUKernelConfig,
    m2ndp_resource_complex_bytes,
    map_cooperative_kernel,
    unsupported,
)

# ---------------------------------------------------------------------------
# Source provenance (section 7 of the baseline-freeze task this was built
# from). Earlier extraction did not record a SHA. The scheduler/upload
# correction below was checked against the explicitly pinned revision.
# ---------------------------------------------------------------------------
PROVENANCE = BaselineProvenance(
    library="VkFFT",
    upstream_repository="https://github.com/DTolm/VkFFT",
    upstream_commit="066a17c17068c0f11c9298d848c2976c71fad1c1 (FP32 register/radix and axis corrections verified 2026-10-08; earlier extraction did not record a SHA)",
    source_files=(
        "vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h",
        "vkFFT/vkFFT/vkFFT_Structs/vkFFT_Structs.h",
        "vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h",
        "vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_ManageLUT.h",
        "vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h",
        "vkFFT/vkFFT/vkFFT_AppManagement/vkFFT_RunApp.h",
        "vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RaderKernels.h",
        "vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/PrePostProcessing/vkFFT_Bluestein.h",
    ),
    baseline_version="gpu-baseline-v1",
    source_functions={
        "registers_per_thread_for": "vkFFT_Scheduler.h: VkFFTGetRegistersPerThread (full classification)",
        "registers_per_thread_base_table": "vkFFT_Scheduler.h: VkFFTGetRegistersPerThread, ordinary FP32 base {2,3,5,7} table",
        "choose_pow2_grouping_radix": "vkFFT_Scheduler.h: the pow2 stage/radix grouping scheduler (active_threads_y/x, final_loc_multipliers_pow2)",
        "choose_num_passes": "vkFFT_Scheduler.h: the numPasses estimate (reorderFourStep=True default branch)",
        "_scheduler_limits": "vkFFT_Scheduler.h: usedSharedMemory (pow2 versus raw budget)",
        "schedule_axis_factors": "vkFFT_Scheduler.h: failed two-pass divisor search promotes numPasses to three",
        "split_pow2_2pass/_3pass": "vkFFT_Scheduler.h: reordered strided pow2 branches and NVIDIA >262144 divisor-search exception",
        "split_non_pow2_2pass/_3pass": "vkFFT_Scheduler.h: non-pow2 axis-split branches (sqrt/cbrt downward divisor search)",
        "apply_four_step_reordering": "vkFFT_Scheduler.h: the %2/%4/%8 locAxisSplit[0] preference loop",
        "direct_radix_sequence": "vkFFT_Scheduler.h: VkFFTOptimizeRadixKernels + thread-cap scaling + VkFFTMinMaxRegisterCheck + first-stage swap (boost one)",
        "axisblock_threads_per_transform/axisblock_batch_*": "vkFFT_AxisBlockSplitter.h: VkFFTSplitAxisBlock's automatic-batch, axis_id==0 branches",
        "_recursive_result (upload order)": "vkFFT_RunApp.h: descending C2C upload loops; vkFFT_Plan_FFT.h: stageStartSize product",
    },
)

# ---------------------------------------------------------------------------
# Fixed baseline-wide parameters (see module docstring).
# ---------------------------------------------------------------------------
VKFFT_COMPLEX_SIZE_BYTES = 8
VKFFT_COALESCED_MEMORY_BYTES = 32
VKFFT_FIX_MAX_CHECK_RADIX2 = 3
VKFFT_ACTIVE_THREADS_X_FLOOR = 128  # vkFFT_Scheduler.h: `active_threads_x >= 128`
VKFFT_MIN_SPLIT_DIM = 64  # `if (locAxisSplit[1] < 64) locAxisSplit[1] = 64`

# Prime coverage of the direct FP32 register scheduler. Composite radix
# choices are made by the original optimizer, independently of codegen.
_DIRECT_RADIX_ORDER = (10, 9, 8, 7, 6, 5, 4, 3, 2)
assert set(_DIRECT_RADIX_ORDER) <= SUPPORTED_RADICES


class VkfftUnsupportedError(Exception):
    """Raised internally for anything needing Rader/Bluestein (a residual
    prime factor outside `_DIRECT_RADIX_ORDER`) -- always caught by this
    module's own `plan()` and converted to UNSUPPORTED_GPU_ALGORITHM."""

    def __init__(self, message: str, *, residual: int | None = None):
        super().__init__(message)
        self.residual = residual


# ---------------------------------------------------------------------------
# Rader-vs-Bluestein PLANNING DECISION (task section 5C, 2026-09-09):
# vkFFT_AppManagement/vkFFT_InitializeApp.h's own real vendor/precision-keyed
# defaults for the boundary between VkFFT's built-in direct-radix kernels,
# Rader's algorithm, and Bluestein's algorithm -- for this baseline's own
# fixed NVIDIA/single-precision choice (VKFFT_VENDOR_IS_NVIDIA, this whole
# project's FP32-only nature): `fixMinRaderPrimeMult=17`,
# `fixMaxRaderPrimeMult=89` (NVIDIA-specific; AMD is also 89, every other
# vendor is 17), `fixMinRaderPrimeFFT=17` (non-AMD default; AMD's own
# precision-keyed values are 17/29/19 and irrelevant to this fixed choice),
# `fixMaxRaderPrimeFFT=16384`.
#
# `VkFFTConstructRaderTree`'s own two-pass loop structure (lines 1733-1809)
# only ever takes its "direct multiplication Rader" branch when `i <
# fixMinRaderPrimeFFT`, which for this baseline's own NVIDIA/single-
# precision fixed choice (fixMinRaderPrimeMult == fixMinRaderPrimeFFT == 17)
# is never true for any `i` the loop actually reaches (the loop itself
# starts at `i = fixMinRaderPrimeMult = 17`) -- so under this baseline's own
# fixed vendor/precision choice, EVERY prime from 17 up to fixMaxRaderPrimeFFT
# uses the FFT-convolution Rader variant, never the direct-multiplication
# one. This baseline therefore reports a single "Rader" scheme (not
# distinguishing Rader-Mult from Rader-FFT sub-variants) for that whole
# range -- a real, source-confirmed simplification for this baseline's own
# fixed vendor choice, not an approximation of an undetermined boundary.
# Neither Rader nor Bluestein is implemented in this repository's own
# codegen in ANY form (radix_spec.SUPPORTED_RADICES has no prime-length
# convolution path at all), so this classification is metadata only -- it
# never changes this baseline's own UNSUPPORTED_GPU_ALGORITHM outcome.
# ---------------------------------------------------------------------------
VKFFT_DIRECT_KERNEL_PRIMES = frozenset({2, 3, 5, 7, 11, 13})  # real VkFFT's own built-in kernels
VKFFT_FIX_MIN_RADER_PRIME_MULT = 17
VKFFT_FIX_MAX_RADER_PRIME_FFT = 16384


def _prime_factors(n: int) -> list[int]:
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def classify_vkfft_residual_scheme(residual: int) -> dict:
    """The real VkFFT scheme (`"direct"` | `"rader"` | `"bluestein"`) for a
    residual factor left over after this baseline's own direct prime coverage -- factors `residual` into primes and classifies
    each against real VkFFT's own built-in-kernel/Rader/Bluestein boundary
    (see module comment above), then reports the single WORST outcome
    (Bluestein dominates Rader dominates direct, matching how Bluestein
    operates on the whole sequence once any prime factor needs it -- see
    `VkFFTConstructRaderTree`'s own `tempSequence`-wide bookkeeping).
    `"direct"` here means real VkFFT would NOT need Rader/Bluestein at all
    for this residual (every prime factor is one of VkFFT's own built-in
    kernels, {2,3,5,7,11,13}) -- it is this BASELINE's own narrower
    `_DIRECT_RADIX_ORDER` (missing 11/13) that cannot represent it, not a
    real VkFFT limitation; still reported as a scope gap, never silently
    treated as Rader/Bluestein just because this port doesn't implement it.
    """
    primes = _prime_factors(residual)
    worst = "direct"
    for p in primes:
        if p in VKFFT_DIRECT_KERNEL_PRIMES:
            classification = "direct"
        elif p < VKFFT_FIX_MAX_RADER_PRIME_FFT:
            classification = "rader"
        else:
            classification = "bluestein"
        if classification == "bluestein" or (classification == "rader" and worst == "direct"):
            worst = classification
    return {"scheme": worst, "residual": residual, "residual_prime_factors": tuple(primes)}


# ---------------------------------------------------------------------------
# 1. Register scheduling -- vkFFT_Scheduler.h VkFFTGetRegistersPerThread.
#
# Ported faithfully (the exact classification rule), but -- per the module
# docstring's CORRECTED UNDERSTANDING -- never used to gate this baseline's
# own radix decomposition (the real source never uses it that way either).
# Exposed for diagnostic/fidelity purposes and for anyone extending this
# baseline toward VkFFT's own Bluestein padded-length search later.
# ---------------------------------------------------------------------------

# The real table (vkFFT_Scheduler.h lines 32-283 at the pinned commit
# 066a17c17068c0f11c9298d848c2976c71fad1c1) is a nested-if cascade keyed
# on PRESENCE of factors of 2/3/5/7 (plus, in two branches, the actual
# COUNT of factor-2's: 1, 2, or >=3) -- fetched and read in full during
# the 2026-09-09 source-fidelity re-audit (an earlier version of this
# module only ported the all-four-present branch, a single worked example
# quoted in the original research report, and fell back to a `min(radices)`
# proxy for every other combination on the M2NDP-mapping path -- a proxy
# reachable from a BaselineStatus.OK result, which is no longer acceptable
# once the real table is available). `registers_per_thread_base_table`
# below ports every branch literally.
#
# The real source also derives composite-radix entries (registers_per_
# thread_per_radix[4,6,8,9,10,12,14,15,16,32], lines 285-297) from these
# four base values. They are deliberately NOT reproduced here: every one
# is provably either 0 or EXACTLY EQUAL to one of the four base entries
# (each derivation is a plain `min(a, b)` of two base entries, or a
# modulus-gated copy of a single base entry -- never a genuinely new
# value) -- so including or excluding them cannot change the min/max this
# module's own callers compute. Concretely: `r[4]=r[8]=r[16]=r[32]` are
# each either 0 or exactly `r[2]`; `r[6]=min(r[2],r[3])`; `r[9]` is 0 or
# exactly `r[3]`; `r[10]=min(r[2],r[5])`; `r[12]` (when its own >=12 gate
# passes) equals `r[6]`; `r[14]=min(r[2],r[7])`; `r[15]=min(r[3],r[5])`.
def quad_registers_per_thread_base_table(
    count2: int, count3: int, count5: int, count7: int,
) -> dict[int, int]:
    """`VkFFTGetRegistersPerThreadQuad`'s own base `{2,3,5,7}` register
    table, ported literally in full (see module comment above for why the
    derived composite entries are provably redundant and omitted).
    `count2`/`count3`/`count5`/`count7`: how many times each prime factor
    appears across this sequence's own radix stages (`loc_multipliers[p]`
    in the real source) -- only presence (`> 0`) matters, EXCEPT in the
    two branches noted below, which switch on the actual count of 2's.

    Returns `{2: ..., 3: ..., 5: ..., 7: ...}` (0 where the real table
    leaves that entry unset for this exact combination). Does NOT handle
    the pure-power-of-two case (only 2 present) -- that has its own exact
    scheduler-search formula, already ported as
    `min_registers_per_thread_pow2`/`choose_pow2_grouping_radix`; nor the
    "none of 2/3/5/7 present" Rader-only case (`min_registers_per_thread`
    hardcoded to 2 in the real source) -- this baseline never reaches
    either through this function, since its own direct-radix vocabulary
    (`_DIRECT_RADIX_ORDER`) always contributes at least one prime factor
    of 2, 3, 5, or 7 alongside at least one other prime when this function
    is actually called (see `min_registers_per_thread_for`'s own dispatch).
    """
    has2, has3, has5, has7 = count2 > 0, count3 > 0, count5 > 0, count7 > 0
    r = {2: 0, 3: 0, 5: 0, 7: 0}
    if has2:
        if has3:
            if has5:
                if has7:
                    r[2], r[3], r[5], r[7] = 6, 6, 5, 7
                else:
                    r[2], r[3], r[5] = 6, 6, 5
            elif has7:
                if count2 in (1, 2):
                    r[2], r[3], r[7] = 6, 6, 7
                else:
                    r[2], r[3], r[7] = 8, 6, 7
            else:
                r[2], r[3] = 6, 6
        elif has5:
            if has7:
                r[2], r[5], r[7] = (6, 5, 7) if count2 == 1 else (8, 5, 7)
            else:
                r[2], r[5] = 4, 5
        elif has7:
            r[2], r[7] = 8, 7
        else:
            raise ValueError(
                "registers_per_thread_base_table: pure power-of-2 (only factor "
                "2 present) is handled by min_registers_per_thread_pow2 instead"
            )
    else:
        if has3:
            if has5:
                if has7:
                    r[3], r[5], r[7] = 6, 5, 7
                else:
                    r[3], r[5] = 3, 5
            elif has7:
                r[3], r[7] = 6, 7
            else:
                r[3] = 3 if count3 == 1 else 9
        elif has5:
            r[5], r[7] = (5, 7) if has7 else (5, 0)
        elif has7:
            r[7] = 7
        else:
            raise ValueError(
                "registers_per_thread_base_table: no factor of 2/3/5/7 present "
                "at all is a Rader-only sequence in the real source "
                "(min_registers_per_thread fixed at 2) -- this baseline's own "
                "direct-radix vocabulary never produces one"
            )
    return r


def registers_per_thread_base_table(
    count2: int, count3: int, count5: int, count7: int,
) -> dict[int, int]:
    """FP32 VkFFTGetRegistersPerThread; Quad's table is separate."""
    return fp32_base_registers(count2, count3, count5, count7)


def registers_per_thread_for(loc_multipliers: dict[int, int]) -> tuple[int, int, bool]:
    """`VkFFTGetRegistersPerThread`'s own classification (vkFFT_Scheduler.h
    lines 299-304, verbatim): scans the per-radix register-requirement
    table for the min and max non-zero entries, then `isGoodSequence = not
    (registers_per_thread > 16 or registers_per_thread >= 2 *
    min_registers_per_thread)`. `loc_multipliers`: `{prime: count}` for
    whichever of 2/3/5/7 appear (a pure power-of-2 sequence, or one with
    no factor of 2/3/5/7 at all, must go through
    `min_registers_per_thread_pow2` / the real source's own Rader-only
    fixed value instead -- see `registers_per_thread_base_table`'s own
    docstring)."""
    table = registers_per_thread_base_table(
        loc_multipliers.get(2, 0), loc_multipliers.get(3, 0),
        loc_multipliers.get(5, 0), loc_multipliers.get(7, 0),
    )
    nonzero = [v for v in table.values() if v != 0]
    min_r, max_r = min(nonzero), max(nonzero)
    is_good = not (max_r > 16 or max_r >= 2 * min_r)
    return max_r, min_r, is_good


# ---------------------------------------------------------------------------
# 2. Stage/radix grouping (power-of-two case) --
# vkFFT_Scheduler.h lines 146-179 (non-quad path, byte-identical).
# ---------------------------------------------------------------------------


def choose_pow2_grouping_radix(
    fft_length: int, max_rhs: int, *, num_compute_units: int = 64,
) -> int:
    """`final_loc_multipliers_pow2`, exactly: estimate workload balance
    across `num_compute_units` independent physical compute units
    (`active_threads_y = max_rhs/num_compute_units`, the real source's own
    comment, quoted verbatim, default 64 -- a representative real GPU's
    CU count), search radix groupings `2^i` for `i` in
    `1..VKFFT_FIX_MAX_CHECK_RADIX2` for the one keeping `active_threads_x
    >= 128`, floor the result at `2^3=8`, then pick whichever grouping
    `2..that bound` truly minimizes the number of Stockham stages
    (`ceil(log2(fft_length)/i)`, ties won by the smaller `i` since the
    source uses strict `<`). Returns the CHOSEN GROUPING RADIX ITSELF
    (`2**final_loc_multipliers_pow2`), not the exponent.

    `max_rhs`: VkFFT's own name for the total batch width this axis plan
    serves -- mapped here to this baseline's own `total_ffts` parameter
    (see `plan`).

    `num_compute_units`: source-faithful default 64 (a representative real
    GPU's own CU count, held fixed for the frozen baseline). The M2NDP-
    adapted baseline (`plan_m2ndp`) passes `target.num_ndp_units` (32)
    instead -- both answer "how many independent physical compute units
    exist to keep busy," the identical mapping already applied to rocFFT-
    default's own `multiprocessor_count` (docs/gpu_planner_m2ndp_target_
    mapping.md). Unlike `warp_size` (see `plan_m2ndp`'s own docstring for
    why that one has NO verified M2NDP equivalent), this one does: both
    64 and `num_ndp_units` answer the identical "count of independent
    physical compute units" question.
    """
    active_threads_y = max(1, max_rhs // num_compute_units)

    test_min_stages = 10_000_000
    max_radix_min_stages = 1
    for i in range(1, VKFFT_FIX_MAX_CHECK_RADIX2 + 1):
        num_stages = math.ceil(math.log2(fft_length) / i)
        if num_stages < test_min_stages:
            test_min_stages = num_stages
            max_radix_min_stages = i

    max_loc_multipliers_pow2 = 0
    for i in range(max_radix_min_stages, 0, -1):
        active_threads_x = (active_threads_y * fft_length) // (2 ** i)
        if active_threads_x >= VKFFT_ACTIVE_THREADS_X_FLOOR:
            max_loc_multipliers_pow2 = i
            break
    if max_loc_multipliers_pow2 < 3:
        max_loc_multipliers_pow2 = 3

    final_loc_multipliers_pow2 = 1
    num_stages_min = int(math.log2(fft_length))
    for i in range(2, max_loc_multipliers_pow2 + 1):
        num_stages = math.ceil(int(math.log2(fft_length)) / i)
        if num_stages < num_stages_min:
            final_loc_multipliers_pow2 = i
            num_stages_min = num_stages

    return 2 ** final_loc_multipliers_pow2


def pow2_radix_sequence(
    fft_length: int, max_rhs: int, *, num_compute_units: int = 64,
) -> tuple[int, ...]:
    """The radix sequence one VkFFT-style leaf kernel renders for a
    power-of-2 `fft_length`: repeat the grouping radix
    (`choose_pow2_grouping_radix`) as many times as it evenly divides
    `log2(fft_length)`, with one smaller final radix for the remainder
    bits (if any) -- exactly what minimizing `ceil(log2(N)/i)` stages
    means concretely. `num_compute_units`: forwarded to `choose_pow2_
    grouping_radix` -- see that function's own docstring."""
    if fft_length <= 1:
        return ()
    grouping_radix = choose_pow2_grouping_radix(fft_length, max_rhs, num_compute_units=num_compute_units)
    total_bits = fft_length.bit_length() - 1
    group_bits = grouping_radix.bit_length() - 1
    full_stages, remainder_bits = divmod(total_bits, group_bits)
    radices = [grouping_radix] * full_stages
    if remainder_bits > 0:
        radices.append(2 ** remainder_bits)
    return tuple(radices)


# ---------------------------------------------------------------------------
# Direct leaf radix decomposition: pinned register-driven optimizer.
# Unsupported original composite butterflies are not replaced by native radices.
# ---------------------------------------------------------------------------


def direct_radix_sequence(length: int) -> tuple[int, ...]:
    """Pinned portable FP32 register-driven radix optimizer, boost one.

    Original radix 12/14/15 choices are retained even when M2NDP has no
    corresponding butterfly; physical mapping reports that gap separately.
    """
    return _direct_radix_plan(length, 1).radices


def _direct_radix_plan(
    length: int, max_rhs: int, *, num_compute_units: int = 64,
    num_passes: int = 1, upload_id: int = 0,
):
    try:
        prime_multiplicities(length)
    except ValueError as exc:
        residual = length
        for prime in (2, 3, 5, 7):
            while residual > 1 and residual % prime == 0:
                residual //= prime
        raise VkfftUnsupportedError(str(exc), residual=residual) from exc
    pow2 = (min_registers_per_thread_pow2(length, max_rhs, num_compute_units=num_compute_units)
            if length & (length - 1) == 0 else None)
    return plan_direct_radices(length, pow2_registers=pow2,
        max_batch_coalesced=1 if num_passes == 1 else _MAX_BATCH_COALESCED,
        upload_id=upload_id, max_threads=VKFFT_MAX_THREADS_NUM)


def leaf_radix_sequence(
    length: int, max_rhs: int, *, num_compute_units: int = 64,
    num_passes: int = 1, upload_id: int = 0,
) -> tuple[int, ...]:
    return _direct_radix_plan(length, max_rhs, num_compute_units=num_compute_units,
                              num_passes=num_passes, upload_id=upload_id).radices


# ---------------------------------------------------------------------------
# 3. Shared memory sizing / registerBoost (fixed to 1 -- see module
# docstring) -- vkFFT_Scheduler.h lines 2240-2241, 2588.
# ---------------------------------------------------------------------------


def max_sequence_length_shared_memory(
    target: TargetProfile, *, complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES,
) -> int:
    """Non-strided max single-kernel length -- `usedSharedMemory /
    complexSize`, LDS-equivalent mapped to `target.spad_capacity_bytes`
    (see gpu_baseline/common.py's own documented GPU-LDS -> M2NDP-
    scratchpad translation).

    `complex_size_bytes`: the "bytes one complex element costs" RESOURCE
    INPUT -- defaults to `VKFFT_COMPLEX_SIZE_BYTES` (8), real VkFFT's own
    single-buffer shared-memory convention, so every existing caller
    (`vkfft.plan()`, the frozen/source-faithful baseline) computes byte-
    identically to before this parameter existed. `vkfft.plan_m2ndp` (the
    M2NDP-adapted baseline) passes `gpu_baseline.common.
    m2ndp_resource_complex_bytes()` (16) instead -- see that function's
    own docstring for why a real M2NDP persistent-leaf lowering needs
    double this many bytes per element once cooperation width is wide
    enough to require worker-wave virtualization, and why substituting it
    HERE (the algorithm's own resource INPUT), not by hand-patching this
    formula's shape, is what section 9 of the task this was built from
    asks for."""
    return target.spad_capacity_bytes // complex_size_bytes


def max_sequence_length_shared_memory_strided(
    target: TargetProfile, *, complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES,
) -> int:
    """Strided max single-kernel length -- `usedSharedMemory /
    coalescedMemory` (never smaller a divisor than `complexSize` -- the
    real source's own `> complexSize` guard, ported verbatim).

    `complex_size_bytes`: see `max_sequence_length_shared_memory`'s own
    docstring -- same default, same M2NDP-adapted override convention."""
    if VKFFT_COALESCED_MEMORY_BYTES > complex_size_bytes:
        return target.spad_capacity_bytes // VKFFT_COALESCED_MEMORY_BYTES
    return target.spad_capacity_bytes // complex_size_bytes


def max_sequence_length_shared_memory_pow2(
    target: TargetProfile, *, complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES,
) -> int:
    """`maxSequenceLengthSharedMemoryPow2` (vkFFT_Plan_FFT.h line 123):
    `allowedSharedMemoryPow2 / complexSize`, where `allowedSharedMemoryPow2
    = configuration.sharedMemorySizePow2` -- Structs.h's own documented
    "power of 2 which is less or equal to sharedMemorySize" (line 293).
    Genuinely distinct from `max_sequence_length_shared_memory` above (the
    real source keeps both bounds simultaneously, using each in different
    `VkFFTSplitAxisBlock` checks -- see `_postprocess_axis_upload0`).
    Derived from `target.spad_capacity_bytes` the same GPU-LDS -> M2NDP-
    scratchpad mapping every other sizing function in this module uses.

    `complex_size_bytes`: see `max_sequence_length_shared_memory`'s own
    docstring -- same default, same M2NDP-adapted override convention."""
    shared_bytes = target.spad_capacity_bytes
    pow2_bytes = (1 << (shared_bytes.bit_length() - 1)) if shared_bytes > 0 else 0
    return pow2_bytes // complex_size_bytes


# ---------------------------------------------------------------------------
# 4. Number of passes -- vkFFT_Scheduler.h lines 2590-2654 (registerBoost
# fixed at 1 throughout, per module docstring, and swapTo2Stage4Step/
# swapTo3Stage4Step left at their real documented defaults, 0 = disabled).
# ---------------------------------------------------------------------------


def _scheduler_limits(length: int, target: TargetProfile, complex_size_bytes: int) -> tuple[int, int, int]:
    """Scheduler usedSharedMemory: sharedMemorySizePow2 for a pow2 axis.

    Domain: 1D C2C, reorderFourStep=True, no Bluestein/convolution,
    registerBoost=1 and no bandwidth boost. AxisBlock retains its separate
    raw/pow2 bounds; these limits belong to the scheduler only.
    """
    shared = target.spad_capacity_bytes
    if length > 0 and length & (length - 1) == 0:
        shared = 1 << (shared.bit_length() - 1) if shared > 0 else 0
    non_strided = shared // complex_size_bytes
    strided = shared // max(VKFFT_COALESCED_MEMORY_BYTES, complex_size_bytes)
    if non_strided < 1 or strided <= 1:
        raise VkfftUnsupportedError("shared-memory budget is too small for VkFFT axis scheduling")
    return shared, non_strided, strided


def choose_num_passes(
    length: int, *, non_strided: bool, target: TargetProfile,
    complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES,
) -> int:
    """Pinned Scheduler.h 2593-2604, default reordered C2C branch.

    Integer powers implement ceil(log2(N)/log2(maxSingleSizeStrided))
    without boundary rounding. A failed 2-pass divisor search may promote
    this estimate to 3 passes in schedule_axis_factors.
    """
    _, max_non_strided, max_strided = _scheduler_limits(length, target, complex_size_bytes)
    if length <= (max_non_strided if non_strided else max_strided):
        return 1
    for passes in (2, 3):
        if length <= max_strided ** passes:
            return passes
    raise VkfftUnsupportedError(f"length={length}: requires more than 3 upload passes")


def _pow8_size(max_len: int) -> int:
    return 1 << (3 * ((max_len.bit_length() - 1) // 3)) if max_len > 0 else 1


def _ceil_sqrt(value: int) -> int:
    root = math.isqrt(value)
    return root if root * root == value else root + 1


def _ceil_cbrt(value: int) -> int:
    # Match ceil(cbrt(N)) mathematically, including exact cube boundaries.
    low, high = 0, 1 << ((value.bit_length() + 2) // 3)
    while low + 1 < high:
        mid = (low + high) // 2
        if mid ** 3 < value:
            low = mid
        else:
            high = mid
    return high


def split_pow2_2pass(
    length: int, target: TargetProfile, *, complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES,
) -> tuple[int, int] | None:
    """Scheduler.h 2655-2749: reordered pow2 branch and NVIDIA exception.

    Returns axisSplit[0:2] in upstream order, before four-step reordering.
    NVIDIA N>262144 uses the exact-divisor search even for pow2 inputs.
    """
    if VKFFT_VENDOR_IS_NVIDIA and length > 262144:
        return split_non_pow2_2pass(length, target, complex_size_bytes=complex_size_bytes)
    _, _, strided = _scheduler_limits(length, target, complex_size_bytes)
    pow8 = min(_pow8_size(strided), 512)
    split0 = pow8 if length // pow8 <= strided else strided
    split1 = length // split0
    if split1 < VKFFT_MIN_SPLIT_DIM:
        split1 = VKFFT_MIN_SPLIT_DIM
        split0 = length // split1
    return (max(split0, split1), min(split0, split1))


def split_non_pow2_2pass(
    length: int, target: TargetProfile, *, complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES,
) -> tuple[int, int] | None:
    """Scheduler.h 2734-2749: both factors use the strided bound.

    This is also the NVIDIA pow2 N>262144 branch. Failed search promotes
    numPasses to 3 rather than rejecting the axis immediately.
    """
    _, _, strided = _scheduler_limits(length, target, complex_size_bytes)
    for d in range(min(_ceil_sqrt(length), strided), 0, -1):
        if length % d == 0 and length // d <= strided:
            return length // d, d
    return None


def split_pow2_3pass(
    length: int, target: TargetProfile, *, complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES,
) -> tuple[int, int, int] | None:
    """Scheduler.h 2751-2844: dedicated reordered 3-upload algorithm.

    The first factor targets 128-byte coalescing, then the remaining two
    factors use power-of-eight/strided bounds. NVIDIA N>262144 uses the
    divisor-search branch instead, exactly as for two uploads.
    """
    if VKFFT_VENDOR_IS_NVIDIA and length > 262144:
        return split_non_pow2_3pass(length, target, complex_size_bytes=complex_size_bytes)
    shared, _, strided = _scheduler_limits(length, target, complex_size_bytes)
    pow8 = _pow8_size(strided)
    strided128 = shared // 128
    pow8_128 = _pow8_size(strided128)
    if length // pow8_128 <= pow8 * strided:
        split0 = pow8_128
    elif length // (pow8_128 * 2) <= pow8 * strided and pow8_128 * 2 <= strided128:
        split0 = pow8_128 * 2
    elif length // (pow8_128 * 4) <= pow8 * strided and pow8_128 * 4 <= strided128:
        split0 = pow8_128 * 4
    elif length // strided <= strided * strided:
        split0 = strided128
        while length // split0 > strided * strided:
            split0 *= 2
    else:
        split0 = strided
    remaining = length // split0
    if remaining < pow8:
        split1 = 1 << ((remaining.bit_length() - 1) // 2)
        split2 = remaining // split1
    else:
        split1 = pow8 if remaining // pow8 <= strided else strided
        split2 = remaining // split1
        if split2 < VKFFT_MIN_SPLIT_DIM:
            split2 = VKFFT_MIN_SPLIT_DIM
            split1 = remaining // split2
    return split0, max(split1, split2), min(split1, split2)


def split_non_pow2_3pass(
    length: int, target: TargetProfile, *, complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES,
) -> tuple[int, int, int] | None:
    """Scheduler.h 2867-2888: cbrt/sqrt search, all three factors bounded.

    Returns (quotient, cbrt divisor, sqrt divisor), preserving axisSplit
    order before apply_four_step_reordering; no sorting or truncation.
    """
    _, _, strided = _scheduler_limits(length, target, complex_size_bytes)
    for d0 in range(min(_ceil_cbrt(length), strided), 0, -1):
        if length % d0:
            continue
        remain = length // d0
        for d1 in range(min(_ceil_sqrt(remain), strided), 0, -1):
            if remain % d1 == 0 and remain // d1 <= strided:
                return remain // d1, d0, d1
    return None


def validate_axis_factors(length: int, factors: tuple[int, ...], target: TargetProfile,
                          *, complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES) -> None:
    """Reject invalid splits before constructing any FFT leaf/codegen plan."""
    if not factors or len(factors) > 3 or math.prod(factors) != length:
        raise VkfftUnsupportedError(f"length={length}: invalid axis split {factors}")
    remaining = length
    for factor in factors:
        if factor <= 0 or remaining % factor:
            raise VkfftUnsupportedError(f"length={length}: non-divisible axis split {factors}")
        if length & (length - 1) == 0 and factor & (factor - 1):
            raise VkfftUnsupportedError(f"length={length}: non-pow2 factor in split {factors}")
        remaining //= factor
    _, non_strided, strided = _scheduler_limits(length, target, complex_size_bytes)
    bound = non_strided if len(factors) == 1 else strided
    if any(factor > bound for factor in factors):
        raise VkfftUnsupportedError(f"length={length}: axis split {factors} exceeds per-upload limit {bound}")


def schedule_axis_factors(length: int, target: TargetProfile, *,
                          complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES) -> tuple[int, ...]:
    """Estimate passes, split, promote failed two-upload search, reorder."""
    passes = choose_num_passes(length, non_strided=True, target=target,
                               complex_size_bytes=complex_size_bytes)
    pow2 = length & (length - 1) == 0
    factors = (length,) if passes == 1 else None
    if passes == 2:
        splitter = split_pow2_2pass if pow2 else split_non_pow2_2pass
        factors = splitter(length, target, complex_size_bytes=complex_size_bytes)
        if factors is None:
            passes = 3
    if passes == 3:
        splitter = split_pow2_3pass if pow2 else split_non_pow2_3pass
        factors = splitter(length, target, complex_size_bytes=complex_size_bytes)
    if factors is None:
        raise VkfftUnsupportedError(f"length={length}: no legal axis split within three uploads")
    factors = apply_four_step_reordering(factors)
    validate_axis_factors(length, factors, target, complex_size_bytes=complex_size_bytes)
    return factors


def apply_four_step_reordering(loc_axis_split: tuple[int, ...]) -> tuple[int, ...]:
    """`vkFFT_Scheduler.h` lines 2945-2966, ported exactly (verified against
    docs/gpu_baseline_vkfft_research.md section 5's own quoted source):

        if (reorderFourStep && !useBluesteinFFT[axis_id]) {
            for (i=0;i<numPasses;i++) if (split[0]%2!=0 && split[i]%2==0) swap(split[0],split[i]);
            for (i=0;i<numPasses;i++) if (split[0]%4!=0 && split[i]%4==0) swap(split[0],split[i]);
            for (i=0;i<numPasses;i++) if (split[0]%8!=0 && split[i]%8==0) swap(split[0],split[i]);
        }

    Three SEQUENTIAL passes (2, then 4, then 8 -- this exact order,
    checked directly against source, not assumed), each greedily
    preferring `locAxisSplit[0]` (the pass transposed via the temp buffer
    under four-step -- this baseline's own `factors[0]`, matching the
    construction order `split_pow2_2pass`/`_3pass` already use, "[0] is
    always the larger/pow8-preferred value") to be divisible by that
    divisor, swapping in whichever later index already has that property
    if `[0]` doesn't. This is a REAL swap of which physical pass gets
    which split length -- observable in this baseline's own `factors`
    tuple, not a no-op bookkeeping change (see verify_gpu_baseline.py's
    own regression test constructing two splits with the SAME factor
    multiset but different orders and confirming this function picks the
    source-mandated one deterministically).

    Real VkFFT gates this on `configuration.reorderFourStep` (Structs.h:
    "Default 1") and `!useBluesteinFFT[axis_id]`. This baseline treats
    both as always true by the time this function is called:
    `choose_num_passes` already assumes `reorderFourStep=True` as VkFFT's
    real default (see that function's own docstring), and Bluestein never
    applies to a multi-pass split this baseline reaches (a length needing
    it was already rejected earlier as UNSUPPORTED_GPU_ALGORITHM) -- so
    the gate is unconditionally "on" here, matching the real default
    configuration this whole baseline otherwise assumes throughout
    (registerBoost=1, swapTo2/3Stage4Step=0-disabled, etc.).
    """
    split = list(loc_axis_split)
    n = len(split)
    for divisor in (2, 4, 8):
        for i in range(n):
            if split[0] % divisor != 0 and split[i] % divisor == 0:
                split[0], split[i] = split[i], split[0]
    return tuple(split)


# ---------------------------------------------------------------------------
# 6. Axis-block / per-transform thread-batching --
# vkFFT_AxisBlockSplitter.h (docs/gpu_baseline_vkfft_axisblock_deepdive.md).
#
# Replaces the earlier "one M2NDP microthread per widest-stage butterfly,
# one FFT slot per group" placeholder (this module's own former KNOWN
# FIDELITY GAP) with a real port of VkFFT's own thread/batch formulas --
# the "automatic batch" branch (`configuration.groupedBatch[axis_id]==0`,
# the real default), `axis_id==0` only (this whole repository is 1D-only,
# so VkFFT's own `axis_id>=1` strided-axis branch, which only applies to
# multi-dimensional plans, never applies here).
#
# 2026-09-09 source-fidelity re-audit: re-read the complete
# `VkFFTSplitAxisBlock` (vkFFT_AxisBlockSplitter.h lines 264-367, the
# `axis_id==0`/`axis_upload_id==0` branch) at the pinned commit, line by
# line, to resolve the two refinements this module's own docstring used to
# say were omitted because they "key on a whole-plan state this baseline's
# per-leaf model doesn't track." That justification turns out to be WRONG
# for both -- reproduced in full below via `_postprocess_axis_upload0`
# (shared by `axisblock_batch_single_pass` and
# `axisblock_batch_multipass_first`, exactly matching how the real source
# applies the same continuation to both of ITS OWN seed branches):
#
# * The "divisibility-fix loop" (lines 301-307) is a CONFIRMED NO-OP in
#   the real source at this pinned commit: its own guard condition and its
#   one possible assignment both key off the SAME pre-loop `axisBlock[1]`
#   value, never the loop variable `i`, and the loop unconditionally
#   terminates on its first pass through that guard regardless of which
#   way it evaluates -- so `axisBlock[1]` after the loop is provably
#   identical to its value before the loop, in every case. Not ported
#   (there is nothing to port); proven by exhaustive case analysis in
#   `verify_gpu_baseline_source_fidelity.py`, not merely asserted.
# * The task's own name for the second refinement ("power-of-two
#   bank-conflict axis swap") turns out to describe TWO SEPARATE real
#   mechanisms, not one: lines 308-311 round `axisBlock[1]` UP to the next
#   power of two (the real comment there says "we plan to swap" -- a
#   genuine comment/code mismatch, reported here rather than silently
#   "corrected" to match the comment); the actual axisBlock[0]<->
#   axisBlock[1] SWAP is a separate, later check at lines 350-364, applied
#   to the fully-processed `axisBlock[1]` (after several more steps this
#   module did not previously port at all: a per-axis-size cap keyed on
#   `original_length` (line 312) or `max_rhs` (line 329) -- both already
#   tracked exactly by this module's own existing parameters, contrary to
#   the old "whole-plan state" claim -- the NVIDIA vendor halving loop
#   (330-335), a `maxComputeWorkGroupSize` cap (336), and a max-thread-num
#   divisor search (338-347)). Both refinements are now ported exactly.
# ---------------------------------------------------------------------------
VKFFT_AIM_THREADS = 128  # Structs.h: "aim at this many threads per block. Default 128"
VKFFT_WARP_SIZE = 32  # Structs.h: "number of threads per warp/wavefront" (portable default)
VKFFT_MAX_THREADS_NUM = 1024  # VkPhysicalDeviceLimits-derived; no real device here, fixed
VKFFT_MAX_COMPUTE_WORKGROUP_SIZE = 1024  # same real-device caveat as above
VKFFT_NUM_SHARED_BANKS = 32  # Structs.h line 199: "how many banks shared memory has. Default 32"
# `maxBatchCoalesced = coalescedMemory / complexSize` (AxisBlockSplitter.h
# line 27) -- both already fixed baseline-wide parameters above (32 / 8 = 4,
# matching the deepdive report's own worked-example assumption exactly).
_MAX_BATCH_COALESCED = VKFFT_COALESCED_MEMORY_BYTES // VKFFT_COMPLEX_SIZE_BYTES
# The deepdive report's own recommendation: the NVIDIA vendor-ID branch
# (`vendorID==0x10DE`) is the only live vendor differentiation in this
# file, and is the internally-consistent choice given this module's own
# pre-existing CUDA-backend convention (VKFFT_FIX_MAX_CHECK_RADIX2 etc.).
VKFFT_VENDOR_IS_NVIDIA = True


def min_registers_per_thread_pow2(
    fft_length: int, max_rhs: int, *, num_compute_units: int = 64,
) -> int:
    """`registers_per_thread_per_radix[2]` for a PURE power-of-2 sequence
    (vkFFT_Scheduler.h, quoted in the original VkFFT research report):
    `= grouping_radix if loc_multipliers[2] > final_loc_multipliers_pow2
    else fft_length` -- since a pure-pow2 sequence's own register table
    has exactly one nonzero radix entry, `min == max == registers_per_
    thread_per_radix[2]` here, so this doubles as this baseline's own
    `min_registers_per_thread` for the pow2 case. `num_compute_units`:
    forwarded to `choose_pow2_grouping_radix` -- see its own docstring."""
    grouping_radix = choose_pow2_grouping_radix(fft_length, max_rhs, num_compute_units=num_compute_units)
    total_bits = fft_length.bit_length() - 1
    group_bits = grouping_radix.bit_length() - 1
    return grouping_radix if total_bits > group_bits else fft_length


def _prime_multiplicities(radices: tuple[int, ...]) -> dict[int, int]:
    return prime_multiplicities(math.prod(radices))


def min_registers_per_thread_for(
    length: int, radices: tuple[int, ...], max_rhs: int, *, num_compute_units: int = 64,
    num_passes: int = 1, upload_id: int = 0,
) -> int:
    """Final register minimum of the selected stages, after optimization."""
    decision = _direct_radix_plan(length, max_rhs, num_compute_units=num_compute_units,
                                  num_passes=num_passes, upload_id=upload_id)
    table = dict(decision.registers_per_radix)
    return min(table[r] for r in radices)


def axisblock_threads_per_transform(fft_dim: int, min_registers_per_thread: int) -> int:
    """`AxisBlockSplitter.h` section 2.1's universal formula: `max(1,
    ceil(fftDim/min_registers_per_thread) // registerBoost)` --
    `registerBoost` fixed at 1 throughout this baseline (module
    docstring), so this reduces to `max(1, ceil(fftDim/min_registers_
    per_thread))`."""
    return max(1, -(-fft_dim // min_registers_per_thread))


def _grouped_batch_seed(
    fft_dim: int, target: TargetProfile, *, single_upload: bool,
    complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES,
) -> int:
    """`AxisBlockSplitter.h` section 2.0's shared top-of-function seed for
    `axis->groupedBatch`, `axis_id==0` only. `single_upload=True` is the
    real source's own `(numAxisUploads[axis_id]==1 && axis_id==0)` case
    (non-strided branch); `single_upload=False` covers every multi-pass
    upload (this baseline's own `reorderFourStep=True` fixed default
    makes the real source's `!reorderFourStep && axis_upload_id==0`
    alternative for reaching the non-strided branch unreachable, so
    `single_upload` here is equivalent to `num_passes(this plan) == 1`,
    not merely `upload_id==0`).

    `complex_size_bytes`: see `choose_num_passes`'s own docstring."""
    if single_upload:
        shared_mem_max = max_sequence_length_shared_memory(target, complex_size_bytes=complex_size_bytes)
        scaled = shared_mem_max // fft_dim
        return scaled if scaled > _MAX_BATCH_COALESCED else _MAX_BATCH_COALESCED
    strided_max = max_sequence_length_shared_memory_strided(target, complex_size_bytes=complex_size_bytes)
    scaled = strided_max // fft_dim
    return scaled * _MAX_BATCH_COALESCED if scaled > 1 else _MAX_BATCH_COALESCED


def axisblock_batch_single_pass(
    fft_dim: int, threads_per_transform: int, target: TargetProfile, *,
    use_rader: bool = False, warp_size: int = VKFFT_WARP_SIZE,
) -> int:
    """`AxisBlockSplitter.h` lines 294-299's `aimThreads`/`warpSize`
    estimate -- the real source's `else` branch of `reorderFourStep &&
    numAxisUploads>1`, i.e. the SEED `axisBlock[1]` a `num_passes==1`
    (single-kernel) leaf starts from. This is only the seed: the real
    source applies a long shared continuation (lines 301-364) to this
    value afterward -- see `_postprocess_axis_upload0`, called by
    `axisblock_for_leaf` right after this function, never skipped.

    `warp_size`: the SIMT lockstep-execution-granularity hardware input
    this formula asks for ("how many threads run in genuine lockstep") --
    defaults to, and stays fixed at, `VKFFT_WARP_SIZE` (32) for BOTH the
    source-faithful and M2NDP-adapted baselines. An earlier revision
    mapped this to `target.interleave_chunk_uthreads` (8), but that field
    is a pure DRAM address-interleaving stride, not a SIMT lockstep width
    -- M2NDP has no verified quantity answering the same question this
    parameter asks, so per this task's own "no meaningful equivalent ->
    preserve the algorithm" rule it is NOT adapted (see docs/
    gpu_planner_m2ndp_target_mapping.md's own VkFFT row, NO_EQUIVALENT
    row). Kept as an explicit parameter (rather than removed back to a
    bare module constant) so a future, actually-verified M2NDP quantity
    can be wired in without another signature change. `VKFFT_AIM_THREADS`
    is likewise NOT adapted -- VkFFT's own hand-tuned occupancy target (a
    fixed algorithm parameter, not a hardware query).
    """
    if threads_per_transform // warp_size == 1 and threads_per_transform / warp_size < 1.5:
        estimate = VKFFT_AIM_THREADS // warp_size
    else:
        estimate = VKFFT_AIM_THREADS // threads_per_transform
    estimate = max(estimate, 1)
    if threads_per_transform < VKFFT_AIM_THREADS and (threads_per_transform < warp_size or use_rader):
        return estimate
    return 1


@dataclass(frozen=True)
class AxisBlockLayout:
    threads_per_transform: int
    transforms_per_block: int
    local_size: tuple[int, int, int]
    axis_swapped: bool = False


def _axis_layout_upload0(
    axis_block0: int, seed_batch: int, fft_dim: int, target: TargetProfile, *,
    num_passes: int, max_rhs: int, original_length: int,
    complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES,
) -> AxisBlockLayout:
    """`AxisBlockSplitter.h` lines 301-364: the complete shared
    continuation the real source applies to EITHER `axis_upload_id==0`
    seed branch (`axisblock_batch_single_pass`'s `num_passes==1` case, or
    `axisblock_batch_multipass_first`'s own unchanged-`groupedBatch` case)
    before this leaf's own final `(threads_per_transform,
    transforms_per_block)` pair is settled. See this section's own
    docstring for the full derivation and why the two "whole-plan-state"
    refinements it used to say were unported turn out to both be exactly
    expressible here: `num_passes` is `FFTPlan->numAxisUploads[0]`;
    `original_length` is `actualFFTSizePerAxis[0][0]` (the axis's own
    overall un-split size, already tracked throughout this module exactly
    under that name); `max_rhs` is `actualFFTSizePerAxis[0][1]` (the total
    batch/replica count this leaf's own FFT is one of, already tracked
    throughout this module exactly under that name) -- both real per-axis
    sizes this project's own always-1D, C2C-only domain maps onto directly,
    not whole-plan bookkeeping this baseline genuinely lacks.

    Returns logical workers/batch plus the final physical local size.
    The X/Y swap at the end changes local_size, not logical FFT counts.
    """
    batch = seed_batch

    # Lines 308-311: round batch UP to the next power of two. (The real
    # comment here says "we plan to swap" -- the CODE rounds up, not
    # swaps; the actual swap is the separate check at the end of this
    # function. A genuine comment/code mismatch in the real source,
    # reported here rather than silently "corrected" to match the
    # comment -- see this project's own established practice for such
    # mismatches, e.g. clFFT's "largest 33%" / rocFFT's utilization-rate
    # comments.)
    shared_mem_pow2 = max_sequence_length_shared_memory_pow2(target, complex_size_bytes=complex_size_bytes)
    if (
        (fft_dim % 2 == 0 or axis_block0 < VKFFT_NUM_SHARED_BANKS // 4)
        and batch > 1
        and batch * fft_dim < shared_mem_pow2
    ):
        batch = 1 << (batch - 1).bit_length()

    # Line 312: numAxisUploads[0] > 1 (first upload of a multi-pass plan)
    # -- cap by how many fft_dim-sized blocks the axis's own un-split
    # length actually has.
    if num_passes > 1:
        cap = -(-original_length // fft_dim)  # ceil division
        if cap < batch:
            batch = cap

    # Lines 313-328 (R2C merge-sequence bookkeeping): never applicable --
    # this project is C2C-only (gpu_baseline_v1_freeze.md's own "Exact
    # domain covered").

    # Line 329: numAxisUploads[0] == 1 (single-pass) -- cap by the total
    # batch count (r2cmult == 1 always for C2C).
    if num_passes == 1 and max_rhs < batch:
        batch = max_rhs

    # Lines 330-335: NVIDIA vendor halving loop (this baseline's own fixed
    # vendorID==0x10DE choice -- see module docstring/VKFFT_VENDOR_IS_NVIDIA).
    if VKFFT_VENDOR_IS_NVIDIA:
        while batch * axis_block0 >= 2 * VKFFT_AIM_THREADS and batch > _MAX_BATCH_COALESCED:
            batch //= 2
            if batch < _MAX_BATCH_COALESCED:
                batch = _MAX_BATCH_COALESCED

    # Line 336.
    if batch > VKFFT_MAX_COMPUTE_WORKGROUP_SIZE:
        batch = VKFFT_MAX_COMPUTE_WORKGROUP_SIZE

    # Lines 338-347: divisor search if axisBlock[0]*axisBlock[1] exceeds
    # maxThreadNum (== VKFFT_MAX_THREADS_NUM -- no real per-device query
    # exists here, matching every other "real device" caveat in this
    # module).
    if axis_block0 * batch > VKFFT_MAX_THREADS_NUM:
        for i in range(1, batch + 1):
            if (batch // i) * axis_block0 <= VKFFT_MAX_THREADS_NUM:
                batch //= i
                break

    # Line 348: register-boost-aware (fixed at 1 throughout this baseline)
    # shared-memory cap -- this is the SAME formula this function used to
    # apply immediately after the seed; the real source applies it HERE,
    # after every step above, which can produce a different final `batch`.
    max_seq_shared = max_sequence_length_shared_memory(target, complex_size_bytes=complex_size_bytes)
    while batch * fft_dim > max_seq_shared and batch > 1:
        batch //= 2

    # Lines 350-364: the real axisBlock[0] <-> axisBlock[1] SWAP, gated on
    # the non-Pow2 shared-memory bound -- genuinely distinct from the
    # round-up-to-po2 step above both in formula and in which (much more
    # processed) `batch` value it inspects.
    axis_swapped = (
        (fft_dim % 2 == 0 or axis_block0 < VKFFT_NUM_SHARED_BANKS // 4)
        and batch > 1
        and batch * fft_dim < max_seq_shared
    )
    batch = max(batch, 1)
    local_x, local_y = (batch, axis_block0) if axis_swapped else (axis_block0, batch)
    return AxisBlockLayout(axis_block0, batch, (local_x, local_y, 1), axis_swapped)


def _postprocess_axis_upload0(*args, **kwargs) -> tuple[int, int]:
    """Compatibility view of physical local X/Y, not logical FFT workers."""
    layout = _axis_layout_upload0(*args, **kwargs)
    return layout.local_size[:2]


def axisblock_batch_multipass_first(
    fft_dim: int, target: TargetProfile, *, complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES,
) -> int:
    """Upload-zero groupedBatch seed (upload 0 executes last).

    Kept under its existing helper name for callers; this is a GPU upload
    index, not the first FFT executed by the M2NDP plan.
    """
    return _grouped_batch_seed(
        fft_dim, target, single_upload=False, complex_size_bytes=complex_size_bytes,
    )


def axisblock_batch_multipass_later(
    fft_dim: int, threads_per_transform: int, target: TargetProfile, *, stage_start_size: int,
    complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES,
) -> int:
    """Upload IDs >0: grow groupedBatch, cap by stageStartSize and threads.

    stageStartSize is the product of all LOWER upload-index factors, as
    initialized by VkFFTPlanAxis. GPU upload IDs do not count execution
    order; ID numPasses-1 executes first in the baseline's C2C domain.
    """
    grouped_batch = _grouped_batch_seed(
        fft_dim, target, single_upload=False, complex_size_bytes=complex_size_bytes,
    )
    scale = VKFFT_AIM_THREADS // max(threads_per_transform * grouped_batch, 1)
    max_seq_shared = max_sequence_length_shared_memory(target, complex_size_bytes=complex_size_bytes)
    if scale > 1 and fft_dim * grouped_batch * scale <= max_seq_shared:
        grouped_batch *= scale

    axis_block0 = min(grouped_batch, stage_start_size) if stage_start_size > 0 else grouped_batch
    if VKFFT_VENDOR_IS_NVIDIA:
        while (
            axis_block0 * threads_per_transform >= 2 * VKFFT_AIM_THREADS
            and axis_block0 > _MAX_BATCH_COALESCED
        ):
            axis_block0 //= 2
            if axis_block0 < _MAX_BATCH_COALESCED:
                axis_block0 = _MAX_BATCH_COALESCED

    axis_block0 = min(axis_block0, VKFFT_MAX_COMPUTE_WORKGROUP_SIZE)
    if axis_block0 * threads_per_transform > VKFFT_MAX_THREADS_NUM:
        for i in range(1, axis_block0 + 1):
            if (axis_block0 // i) * threads_per_transform <= VKFFT_MAX_THREADS_NUM:
                axis_block0 = axis_block0 // i
                break
    return max(axis_block0, 1)


def axisblock_layout_for_leaf(
    length: int, radices: tuple[int, ...], *, max_rhs: int, num_passes: int, upload_id: int,
    original_length: int, target: TargetProfile, warp_size: int = VKFFT_WARP_SIZE,
    num_compute_units: int = 64, complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES,
    stage_start_size: int | None = None,
) -> AxisBlockLayout:
    """Map the original GPU upload ID to AxisBlock settings.

    Upload 0 executes last for both forward and inverse reordered C2C.
    It uses the upload-zero postprocessing branch; IDs >0 use the strided
    branch. For three uploads, callers must supply stage_start_size (the
    product of axisSplit[:upload_id]); original_length / length would
    wrongly include higher upload IDs too.
    """
    min_regs = min_registers_per_thread_for(length, radices, max_rhs, num_compute_units=num_compute_units,
                                            num_passes=num_passes, upload_id=upload_id)
    threads_per_transform = axisblock_threads_per_transform(length, min_regs)
    if num_passes == 1:
        seed = axisblock_batch_single_pass(length, threads_per_transform, target, warp_size=warp_size)
        return _axis_layout_upload0(
            threads_per_transform, seed, length, target,
            num_passes=num_passes, max_rhs=max_rhs, original_length=original_length,
            complex_size_bytes=complex_size_bytes,
        )
    elif upload_id == 0:
        seed = axisblock_batch_multipass_first(length, target, complex_size_bytes=complex_size_bytes)
        return _axis_layout_upload0(
            threads_per_transform, seed, length, target,
            num_passes=num_passes, max_rhs=max_rhs, original_length=original_length,
            complex_size_bytes=complex_size_bytes,
        )
    else:
        if stage_start_size is None:
            if num_passes > 2:
                raise ValueError("three-upload AxisBlock requires stage_start_size")
            stage_start_size = original_length // length
        batch = axisblock_batch_multipass_later(
            length, threads_per_transform, target, stage_start_size=stage_start_size,
            complex_size_bytes=complex_size_bytes,
        )
    # Strided uploads use local X for FFT slots and local Y for workers.
    return AxisBlockLayout(threads_per_transform, batch, (batch, threads_per_transform, 1))


def axisblock_for_leaf(*args, **kwargs) -> tuple[int, int]:
    """Logical FFT workers/slots, independent of local-axis arrangement."""
    layout = axisblock_layout_for_leaf(*args, **kwargs)
    return layout.threads_per_transform, layout.transforms_per_block


# ---------------------------------------------------------------------------
# M2NDP translation layer + recursive plan construction. Reuses this
# project's own FFTRecursiveNodePlan/PhysicalTransposePlan (the same
# structural shape gpu_baseline/clfft.py's own large-1D path already
# established -- PRE transpose -> near leaf -> MIDDLE transpose+twiddle ->
# far (leaf or further split) -> POST transpose): VkFFT's own multi-pass
# axis decomposition (a first-pass factor, a transpose-with-twiddle, and a
# second/third-pass factor) is the SAME classic four-step shape clFFT's
# large-1D pipeline uses, so the same M2NDP plan dataclass represents it
# exactly -- only the split-selection RULE (this module's own axis-
# splitting functions above) differs per baseline.
# ---------------------------------------------------------------------------


def _leaf_result(
    m: int, r: int, *, inverse: bool, is_root: bool, node_id: list[int],
    target: TargetProfile, num_passes: int, upload_id: int, original_length: int,
    warp_size: int = VKFFT_WARP_SIZE, num_compute_units: int = 64,
    complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES,
    stage_start_size: int = 1, leaf_configs: list[dict] | None = None,
) -> tuple[FFTLeafPlan | None, BaselineResult | None]:
    idx = node_id[0]
    node_id[0] += 1
    try:
        radices = leaf_radix_sequence(m, r, num_compute_units=num_compute_units,
                                     num_passes=num_passes, upload_id=upload_id)
    except VkfftUnsupportedError as exc:
        extra = classify_vkfft_residual_scheme(exc.residual) if exc.residual is not None else {}
        gpu_config = GPUKernelConfig(source="vkfft", length=m, radices=(), extra=extra)
        return None, unsupported(BaselineStatus.UNSUPPORTED_GPU_ALGORITHM, gpu_config, str(exc))

    if not radices:
        gpu_config = GPUKernelConfig(source="vkfft", length=m, radices=())
        return None, unsupported(
            BaselineStatus.UNSUPPORTED_GPU_ALGORITHM, gpu_config, f"length={m}: no radix sequence produced"
        )

    layout = axisblock_layout_for_leaf(
        m, radices, max_rhs=r, num_passes=num_passes, upload_id=upload_id,
        original_length=original_length, target=target, warp_size=warp_size,
        num_compute_units=num_compute_units, complex_size_bytes=complex_size_bytes,
        stage_start_size=stage_start_size,
    )
    workers_per_fft, fft_slots_wanted = layout.threads_per_transform, layout.transforms_per_block
    registers = _direct_radix_plan(m, r, num_compute_units=num_compute_units,
                                   num_passes=num_passes, upload_id=upload_id)
    gpu_config = GPUKernelConfig(
        source="vkfft", length=m, radices=radices,
        extra={
            "workers_per_fft": workers_per_fft, "transforms_per_block": fft_slots_wanted,
            "register_boost": 1, "upload_id": upload_id,
            "stage_start_size": stage_start_size,
            "axis_swapped": layout.axis_swapped, "local_size": layout.local_size,
            "min_registers_per_thread": registers.min_registers,
            "registers_per_thread": registers.max_registers,
            "registers_per_radix": registers.registers_per_radix,
        },
    )
    if leaf_configs is not None:
        leaf_configs.append(dict(length=m, replicas=r, radices=radices, **gpu_config.extra))
    inverse_scale = (1.0 / m) if (inverse and is_root) else None
    mapping = map_cooperative_kernel(
        length=m, radices=radices, workers_per_fft=workers_per_fft, fft_slots_wanted=fft_slots_wanted,
        total_ffts=r, inverse=inverse, inverse_scale=inverse_scale,
        kernel_name=f"FFTVkfftLeaf{idx}", target=target, gpu_config=gpu_config,
    )
    if mapping.status is not BaselineStatus.OK:
        return None, mapping
    return FFTLeafPlan(m=m, r=r, kernel=mapping.plan), None


def _recursive_result(
    m: int, r: int, factors: tuple[int, ...], *, inverse: bool, is_root: bool,
    node_id: list[int], target: TargetProfile, num_passes: int, upload_id: int, original_length: int,
    warp_size: int = VKFFT_WARP_SIZE, num_compute_units: int = 64,
    complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES,
    leaf_configs: list[dict] | None = None,
) -> tuple[FFTNode | None, BaselineResult | None]:
    """Execute axisSplit from its last index down to zero.

    factors[-1] is the next FFT upload. Each far-child recursion decrements
    upload_id. stageStartSize is the product of lower-index split factors,
    matching Plan_FFT.h, independently of execution direction.
    """
    if len(factors) == 1:
        return _leaf_result(
            m, r, inverse=inverse, is_root=is_root, node_id=node_id, target=target,
            num_passes=num_passes, upload_id=upload_id, original_length=original_length,
            warp_size=warp_size, num_compute_units=num_compute_units,
            complex_size_bytes=complex_size_bytes, stage_start_size=1,
            leaf_configs=leaf_configs,
        )

    idx = node_id[0]
    node_id[0] += 1
    b = factors[-1]
    a = m // b
    tile = max(1, min(8, a, b))

    pre = _build_physical_transpose(
        rows=b, cols=a, replica_count=r, tile_rows=tile, tile_cols=tile,
        twiddle_modulus=None, inverse=inverse, kernel_name=f"FFTVkfftPre{idx}",
        simd_lanes=8, spad_capacity_bytes=target.spad_capacity_bytes, apply_inverse_scale=False,
    )
    near_node, failure = _leaf_result(
        b, r * a, inverse=inverse, is_root=False, node_id=node_id, target=target,
        num_passes=num_passes, upload_id=upload_id, original_length=original_length,
        warp_size=warp_size, num_compute_units=num_compute_units,
        complex_size_bytes=complex_size_bytes, stage_start_size=math.prod(factors[:-1]),
        leaf_configs=leaf_configs,
    )
    if failure is not None:
        return None, failure

    middle = _build_physical_transpose(
        rows=a, cols=b, replica_count=r, tile_rows=tile, tile_cols=tile,
        twiddle_modulus=m, inverse=inverse, kernel_name=f"FFTVkfftMid{idx}",
        simd_lanes=8, spad_capacity_bytes=target.spad_capacity_bytes, apply_inverse_scale=False,
    )
    far_node, failure = _recursive_result(
        a, r * b, factors[:-1], inverse=inverse, is_root=False, node_id=node_id, target=target,
        num_passes=num_passes, upload_id=upload_id - 1, original_length=original_length,
        warp_size=warp_size, num_compute_units=num_compute_units,
        complex_size_bytes=complex_size_bytes, leaf_configs=leaf_configs,
    )
    if failure is not None:
        return None, failure

    post = _build_physical_transpose(
        rows=b, cols=a, replica_count=r, tile_rows=tile, tile_cols=tile,
        twiddle_modulus=None, inverse=inverse, kernel_name=f"FFTVkfftPost{idx}",
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


@preserve_gpu_plan
def plan(
    length: int,
    *,
    batch: int = 1,
    inverse: bool = False,
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
    warp_size: int = VKFFT_WARP_SIZE,
    num_compute_units: int = 64,
    complex_size_bytes: int = VKFFT_COMPLEX_SIZE_BYTES,
) -> BaselineResult:
    """Plan reordered 1D C2C FFTs using the pinned VkFFT scheduler.

    plan() uses 8-byte complex elements and the representative GPU CU
    count; plan_m2ndp() supplies 16-byte scratchpad costs and M2NDP units.
    Scheduler pow2 inputs use floor_pow2(target.spad_capacity_bytes);
    non-pow2 inputs use the full budget. The 2/3-upload split, NVIDIA
    exception, search promotion and upload IDs follow upstream control
    flow. Shared lowering still expands memory rearrangements into separate
    transposes (finding 3), independently of the scheduling correction.
    """
    is_po2 = (length & (length - 1)) == 0
    try:
        factors = schedule_axis_factors(length, target, complex_size_bytes=complex_size_bytes)
        num_passes = len(factors)
    except VkfftUnsupportedError as exc:
        gpu_config = GPUKernelConfig(source="vkfft", length=length, radices=())
        return unsupported(BaselineStatus.UNSUPPORTED_GPU_ALGORITHM, gpu_config, str(exc))

    node_id = [0]
    leaf_configs: list[dict] = []
    # factors is far-to-near ascending-nest order for _recursive_result's own
    # "pop from the end" convention: factors[-1] is this level's near/b.
    node, failure = _recursive_result(
        length, batch, factors, inverse=inverse, is_root=True, node_id=node_id, target=target,
        num_passes=num_passes, upload_id=num_passes - 1, original_length=length, warp_size=warp_size,
        num_compute_units=num_compute_units, complex_size_bytes=complex_size_bytes,
        leaf_configs=leaf_configs,
    )
    if failure is not None:
        return failure
    assert node is not None
    # For a single-leaf (num_passes==1) plan, surface the leaf's own real
    # radix sequence at the top level too -- a caller comparing baselines
    # (section 6 of the task this package was built from) shouldn't have
    # to reach into the plan tree just to see what radices were chosen.
    top_radices: tuple[int, ...] = ()
    if isinstance(node, FFTLeafPlan):
        top_radices = tuple(stage.radix for stage in node.kernel.stages)
    gpu_config = GPUKernelConfig(
        source="vkfft" if num_compute_units == 64 else "vkfft-m2ndp",
        length=length, radices=top_radices,
        extra={"num_passes": num_passes, "factors": factors, "is_pow2": is_po2,
               "upload_execution_order": tuple(range(num_passes - 1, -1, -1)),
               "leaf_configs": leaf_configs},
    )
    recursive_plan = RecursiveFFTPlan(
        n=length, inverse=inverse, root=node,
        host=MultiKernelHostPlan(n=length, inverse=inverse, tolerance=1.0e-3), batch=batch,
    )
    return BaselineResult(status=BaselineStatus.OK, gpu_config=gpu_config, plan=recursive_plan)


def plan_m2ndp(
    length: int,
    *,
    batch: int = 1,
    inverse: bool = False,
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
) -> BaselineResult:
    """M2NDP-ADAPTED top-level VkFFT-style entry point -- the `gpu-vkfft-
    m2ndp` CLI baseline. See docs/gpu_planner_m2ndp_target_mapping.md's
    own VkFFT row for the full, audited mapping table this function's own
    parameter choices below implement.

    `warp_size`: DELIBERATELY LEFT AT THE SOURCE-FAITHFUL DEFAULT (32) --
    an earlier revision of this function mapped it to `target.
    interleave_chunk_uthreads` (8), but that field is documented (`target_
    profile.py`'s own comment) as a pure DRAM ADDRESS-INTERLEAVING stride
    ("which physical NDP unit does this address land on"), not a SIMT
    lockstep execution width -- M2NDP's own microthreads are each
    independently generated/retired hardware FGMT, never executing in the
    per-instruction lockstep a GPU warp implies. There is no verified M2NDP
    quantity that answers the same question `VKFFT_WARP_SIZE` does; per
    this task's own "no meaningful equivalent -> preserve the algorithm"
    rule, this stays NO_EQUIVALENT, unchanged from the source-faithful
    baseline, rather than substituting a number whose meaning does not
    actually match.

    `num_compute_units=target.num_ndp_units` (32): `choose_pow2_grouping_
    radix`'s own `active_threads_y = max_rhs // num_compute_units`
    estimates workload balance across independent physical compute units
    keeping busy -- VkFFT's own real source assumes 64 (a representative
    GPU's CU count) here; M2NDP's own independent-physical-unit count is
    `num_ndp_units=32`. Both answer the identical question, exactly the
    same mapping already applied to rocFFT-default's own `multiprocessor_
    count` (docs/gpu_planner_m2ndp_target_mapping.md).

    `complex_size_bytes=m2ndp_resource_complex_bytes()` (16, ADDED
    2026-09-22): see `plan`'s own docstring for the full derivation --
    this is the ONE difference (beyond `num_compute_units`) between this
    function and the frozen `plan()` it wraps, resource INPUT only, same
    VkFFT algorithm. `plan()` itself keeps `complex_size_bytes=VKFFT_
    COMPLEX_SIZE_BYTES` (8) as its own default, so the `gpu-vkfft` (frozen)
    CLI baseline's behavior is completely unaffected by this parameter's
    existence (section 11 of the task this was built from -- frozen and
    M2NDP-adapted baselines never share a resource assumption that differs
    between them).
    """
    return plan(
        length, batch=batch, inverse=inverse, target=target,
        num_compute_units=target.num_ndp_units,
        complex_size_bytes=m2ndp_resource_complex_bytes(),
    )
