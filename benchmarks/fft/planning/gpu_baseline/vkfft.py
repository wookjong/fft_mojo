from __future__ import annotations

"""VkFFT-style BASELINE planner.

Ported from the REAL, public VkFFT source (DTolm/VkFFT, `master` branch),
fetched and read directly (not recalled from memory). See `docs/
gpu_baseline_vkfft_research.md` (a copy of the research report this module
was built from) for the full verification trail. Important path
correction the research uncovered: the library root inside the repo is
`vkFFT/vkFFT/...` (doubled), not the single `vkFFT/...` the initial draft
assumed -- recorded here so a future re-verification pass doesn't repeat
the 404.

This baseline is DETERMINISTIC -- no benchmarking, no empirical tuning
(unlike gpu_baseline/rocfft.py). See gpu_baseline/common.py's own module
docstring for the shared non-negotiable rule (no import from planning.
fft_cost_model / fft_plan_cooperative's own heuristic / fft_plan_persistent
/ fft_plan_lanes).

============================================================================
CORRECTED UNDERSTANDING vs. the initial draft (do not re-introduce):
============================================================================
`isGoodSequence` (the `registers_per_thread > 16 || >= 2*min_registers_per_
thread` classification) is NOT a general accept/reject gate on radix
sequences in the real source -- its only two call sites are both inside
VkFFT's own Bluestein PADDED-CONVOLUTION-LENGTH search (picking a
composite-of-{2,3,5,7} padding size register-balanced for the *internal*
convolution kernel, not the user's own requested axis length). This
module still ports the classification function faithfully (it is real,
verified source), but does not use it to gate this baseline's own radix
decomposition -- doing so would be inventing a rule VkFFT itself does not
apply there, exactly the "silently changing the algorithm" section 0/1C
of the task this package was built from forbids.

============================================================================
FIXED IMPLEMENTATION PARAMETERS (section 3 -- "must remain FIXED for all
GPU baseline candidates and must be documented"):
============================================================================
VKFFT_COMPLEX_SIZE_BYTES = 8 (single precision, this project is FP32-only)
VKFFT_COALESCED_MEMORY_BYTES = 32
    `Structs.h`'s own documented convention: "for Nvidia and AMD is equal
    to 32, Intel is equal 64" -- a user/vendor-set config value in real
    VkFFT, not auto-detected (confirmed by the research report: no
    vendor-ID-keyed lookup table was found). 32 (the Nvidia/AMD
    convention) is used here, fixed, since M2NDP has no such vendor to
    query.
VKFFT_FIX_MAX_CHECK_RADIX2 = 3
    The real source's own default search bound for the power-of-2 stage-
    grouping scheduler (section 2 of the research report). The CUDA-
    backend-only widening to 5 (`VKFFT_BACKEND==1`, certain lengths) is
    deliberately NOT ported: M2NDP is not a CUDA backend, and a grouping
    radix of 2^5=32 is not in radix_spec.SUPPORTED_RADICES anyway (M2NDP's
    largest supported radix is 17) -- widening the search would only ever
    produce an UNSUPPORTED_GPU_ALGORITHM result, never a usable one, so this
    baseline holds the bound at its real, portable default instead.
============================================================================

AXIS-BLOCK / THREAD-BATCHING (completed 2026-09-08, see docs/
gpu_baseline_vkfft_axisblock_deepdive.md): VkFFT's own per-transform
thread-count / batching formula (`vkFFT_AxisBlockSplitter.h`'s
`VkFFTSplitAxisBlock` -- see `axisblock_for_leaf` and the functions
around it below) is now ported directly from source: `threads_per_
transform = max(1, ceil(fftDim/min_registers_per_thread))`, and
`transforms_per_block` via the real `aimThreads`/`warpSize` estimate
(single-pass leaves) or the real seeded-batch/`scale`-growth formula
(multi-pass leaves, first vs. later upload). Two real refinements on top
of that core formula are NOT ported (documented at the axis-block
section's own docstring, not silently dropped): a divisibility-fix loop
and a power-of-2 bank-conflict axis-swap, both keyed on a "total
remaining sequence count" this baseline's per-leaf model does not track
the same way VkFFT's own whole-FFTPlan model does. The register input
(`min_registers_per_thread`) is EXACT for both power-of-2 lengths
(`min_registers_per_thread_pow2`, reusing this module's own already-exact
pow2 stage-grouping scheduler) and every other length this baseline's own
direct-radix vocabulary can produce, since the 2026-09-09 source-fidelity
re-audit ported `VkFFTGetRegistersPerThreadQuad`'s complete base `{2,3,5,
7}` table (`registers_per_thread_base_table`) -- replacing an earlier
`min(radices)` proxy that was reachable from a `BaselineStatus.OK` result
and is no longer used anywhere.

`registerBoost`/`registerBoost4Step` (deliberately using MORE registers
than the arithmetic minimum to emulate more shared memory -- Structs.h's
own doc comment) has NO M2NDP mechanical equivalent (project audit section
E: M2NDP's own register file is documented, in docs/STATUS.md, as
unconstrained-until-it-panics, not a tunable per-thread budget to trade
against scratchpad) -- this baseline fixes `registerBoost = 1` always
(VkFFT's own documented default), and never attempts to search or apply a
boost > 1.

Rader's algorithm and Bluestein's algorithm are UNIMPLEMENTED in this
repository entirely (radix_spec.SUPPORTED_RADICES is a small fixed set;
there is no prime-length convolution path anywhere in planning/ or
codegen/) -- any length whose factorization (after stripping VkFFT's own
directly-supported small radices) leaves a residual factor is reported
BaselineStatus.UNSUPPORTED_GPU_ALGORITHM, mirroring VkFFT's own real
`tempSequence != 1 => Bluestein` trigger condition (research report
section 7) as the exact boundary of what this baseline attempts.
"""

import math

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

from .common import (
    BaselineProvenance,
    BaselineResult,
    BaselineStatus,
    GPUKernelConfig,
    map_cooperative_kernel,
    unsupported,
)

# ---------------------------------------------------------------------------
# Source provenance (section 7 of the baseline-freeze task this was built
# from). See BaselineProvenance's own docstring for why `upstream_commit`
# is a floating-branch-plus-fetch-date here: neither VkFFT research pass
# also recorded a pinned commit SHA when fetching `master`.
# ---------------------------------------------------------------------------
PROVENANCE = BaselineProvenance(
    library="VkFFT",
    upstream_repository="https://github.com/DTolm/VkFFT",
    upstream_commit="master (unpinned; fetched 2026-09-08, re-fetched 2026-09-08 for the axis-block deep dive -- no commit SHA recorded, see docs/gpu_baseline_vkfft_research.md and docs/gpu_baseline_vkfft_axisblock_deepdive.md)",
    source_files=(
        "vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h",
        "vkFFT/vkFFT/vkFFT_Structs/vkFFT_Structs.h",
        "vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h",
        "vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_ManageLUT.h",
        "vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h",
        "vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RaderKernels.h",
        "vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/PrePostProcessing/vkFFT_Bluestein.h",
    ),
    baseline_version="gpu-baseline-v1",
    source_functions={
        "registers_per_thread_for": "vkFFT_Scheduler.h: VkFFTGetRegistersPerThread (full classification)",
        "registers_per_thread_base_table": "vkFFT_Scheduler.h: VkFFTGetRegistersPerThreadQuad, base {2,3,5,7} table (lines 32-283)",
        "choose_pow2_grouping_radix": "vkFFT_Scheduler.h: the pow2 stage/radix grouping scheduler (active_threads_y/x, final_loc_multipliers_pow2)",
        "choose_num_passes": "vkFFT_Scheduler.h: the numPasses estimate (reorderFourStep=True default branch)",
        "split_pow2_2pass/_3pass": "vkFFT_Scheduler.h: pow2 axis-split branches (maxPow8SharedMemory preference)",
        "split_non_pow2_2pass/_3pass": "vkFFT_Scheduler.h: non-pow2 axis-split branches (sqrt/cbrt downward divisor search)",
        "apply_four_step_reordering": "vkFFT_Scheduler.h: the %2/%4/%8 locAxisSplit[0] preference loop",
        "direct_radix_sequence": "vkFFT_Scheduler.h: direct Stockham-radix stripping (a conservative subset; Rader/Bluestein not ported)",
        "axisblock_threads_per_transform/axisblock_batch_*": "vkFFT_AxisBlockSplitter.h: VkFFTSplitAxisBlock's automatic-batch, axis_id==0 branches",
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

# Direct-radix-kernel vocabulary this baseline attempts for the non-power-
# of-2 leaf case: every composite of the register table's own {2,3,5,7}
# prime coverage that is also in M2NDP's own SUPPORTED_RADICES. This is
# still a real, documented SCOPE LIMIT (not the register-table gap, which
# is now fully ported -- see registers_per_thread_base_table): real VkFFT
# can also plan radices 11, 13, 17 directly, and this baseline cannot
# (11/13/17 are outside the {2,3,5,7} register table this function's own
# min_registers_per_thread_for now ports exactly, and this baseline's
# direct_radix_sequence's own greedy scan is not a faithful port of the
# real scheduler's own decision between direct-radix/Rader/Bluestein for
# such residuals -- see task section 5B, not yet addressed). Anything
# needing a larger prime factor than this vocabulary covers is reported
# UNSUPPORTED_GPU_ALGORITHM, matching VkFFT's own real `tempSequence != 1`
# boundary in spirit, though not (yet) its own real Rader-vs-Bluestein
# selection logic.
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
    residual factor left over after this baseline's own `_DIRECT_RADIX_
    ORDER` greedy strip -- factors `residual` into primes and classifies
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
def registers_per_thread_base_table(
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


def choose_pow2_grouping_radix(fft_length: int, max_rhs: int) -> int:
    """`final_loc_multipliers_pow2`, exactly: estimate workload balance
    across an assumed 64 compute units (`active_threads_y = max_rhs/64`,
    the real source's own comment, quoted verbatim), search radix
    groupings `2^i` for `i` in `1..VKFFT_FIX_MAX_CHECK_RADIX2` for the one
    keeping `active_threads_x >= 128`, floor the result at `2^3=8`, then
    pick whichever grouping `2..that bound` truly minimizes the number of
    Stockham stages (`ceil(log2(fft_length)/i)`, ties won by the smaller
    `i` since the source uses strict `<`). Returns the CHOSEN GROUPING
    RADIX ITSELF (`2**final_loc_multipliers_pow2`), not the exponent.

    `max_rhs`: VkFFT's own name for the total batch width this axis plan
    serves -- mapped here to this baseline's own `total_ffts` parameter
    (see `plan`).
    """
    active_threads_y = max(1, max_rhs // 64)

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


def pow2_radix_sequence(fft_length: int, max_rhs: int) -> tuple[int, ...]:
    """The radix sequence one VkFFT-style leaf kernel renders for a
    power-of-2 `fft_length`: repeat the grouping radix
    (`choose_pow2_grouping_radix`) as many times as it evenly divides
    `log2(fft_length)`, with one smaller final radix for the remainder
    bits (if any) -- exactly what minimizing `ceil(log2(N)/i)` stages
    means concretely."""
    if fft_length <= 1:
        return ()
    grouping_radix = choose_pow2_grouping_radix(fft_length, max_rhs)
    total_bits = fft_length.bit_length() - 1
    group_bits = grouping_radix.bit_length() - 1
    full_stages, remainder_bits = divmod(total_bits, group_bits)
    radices = [grouping_radix] * full_stages
    if remainder_bits > 0:
        radices.append(2 ** remainder_bits)
    return tuple(radices)


# ---------------------------------------------------------------------------
# Non-power-of-2 leaf radix decomposition -- see module docstring's KNOWN
# FIDELITY GAP: a conservative direct-radix greedy factorization, not the
# real source's full Rader-aware table.
# ---------------------------------------------------------------------------


def direct_radix_sequence(length: int) -> tuple[int, ...]:
    """Greedily factor `length` using only `_DIRECT_RADIX_ORDER` (largest
    first). Raises VkfftUnsupportedError if a residual factor remains --
    exactly VkFFT's own real `tempSequence != 1` Rader/Bluestein boundary
    (research report section 7), ported as this baseline's own scope
    limit rather than an attempt at Rader/Bluestein itself."""
    radices: list[int] = []
    remaining = length
    for rad in _DIRECT_RADIX_ORDER:
        while remaining % rad == 0:
            radices.append(rad)
            remaining //= rad
    if remaining != 1:
        classification = classify_vkfft_residual_scheme(remaining)
        raise VkfftUnsupportedError(
            f"length={length}: residual factor {remaining} (prime factors "
            f"{classification['residual_prime_factors']}) after stripping "
            f"{_DIRECT_RADIX_ORDER} -- real VkFFT's own planning decision here is "
            f"scheme={classification['scheme']!r} (see classify_vkfft_residual_scheme's "
            f"own docstring for the exact boundary this was derived from); neither "
            f"Rader nor Bluestein is implemented in this repository's codegen at all "
            f"(radix_spec.SUPPORTED_RADICES has no prime-length convolution path), and "
            f"even a 'direct' classification here only means real VkFFT's OWN built-in "
            f"kernels cover it -- this baseline's own narrower _DIRECT_RADIX_ORDER "
            f"(missing 11/13) still cannot represent it",
            residual=remaining,
        )
    return tuple(sorted(radices, reverse=True))


def leaf_radix_sequence(length: int, max_rhs: int) -> tuple[int, ...]:
    if length & (length - 1) == 0:
        return pow2_radix_sequence(length, max_rhs)
    return direct_radix_sequence(length)


# ---------------------------------------------------------------------------
# 3. Shared memory sizing / registerBoost (fixed to 1 -- see module
# docstring) -- vkFFT_Scheduler.h lines 2240-2241, 2588.
# ---------------------------------------------------------------------------


def max_sequence_length_shared_memory(target: TargetProfile) -> int:
    """Non-strided max single-kernel length -- `usedSharedMemory /
    complexSize`, LDS-equivalent mapped to `target.spad_capacity_bytes`
    (see gpu_baseline/common.py's own documented GPU-LDS -> M2NDP-
    scratchpad translation)."""
    return target.spad_capacity_bytes // VKFFT_COMPLEX_SIZE_BYTES


def max_sequence_length_shared_memory_strided(target: TargetProfile) -> int:
    """Strided max single-kernel length -- `usedSharedMemory /
    coalescedMemory` (never smaller a divisor than `complexSize` -- the
    real source's own `> complexSize` guard, ported verbatim)."""
    if VKFFT_COALESCED_MEMORY_BYTES > VKFFT_COMPLEX_SIZE_BYTES:
        return target.spad_capacity_bytes // VKFFT_COALESCED_MEMORY_BYTES
    return target.spad_capacity_bytes // VKFFT_COMPLEX_SIZE_BYTES


def max_sequence_length_shared_memory_pow2(target: TargetProfile) -> int:
    """`maxSequenceLengthSharedMemoryPow2` (vkFFT_Plan_FFT.h line 123):
    `allowedSharedMemoryPow2 / complexSize`, where `allowedSharedMemoryPow2
    = configuration.sharedMemorySizePow2` -- Structs.h's own documented
    "power of 2 which is less or equal to sharedMemorySize" (line 293).
    Genuinely distinct from `max_sequence_length_shared_memory` above (the
    real source keeps both bounds simultaneously, using each in different
    `VkFFTSplitAxisBlock` checks -- see `_postprocess_axis_upload0`).
    Derived from `target.spad_capacity_bytes` the same GPU-LDS -> M2NDP-
    scratchpad mapping every other sizing function in this module uses."""
    shared_bytes = target.spad_capacity_bytes
    pow2_bytes = (1 << (shared_bytes.bit_length() - 1)) if shared_bytes > 0 else 0
    return pow2_bytes // VKFFT_COMPLEX_SIZE_BYTES


# ---------------------------------------------------------------------------
# 4. Number of passes -- vkFFT_Scheduler.h lines 2590-2654 (registerBoost
# fixed at 1 throughout, per module docstring, and swapTo2Stage4Step/
# swapTo3Stage4Step left at their real documented defaults, 0 = disabled).
# ---------------------------------------------------------------------------


def choose_num_passes(length: int, *, non_strided: bool, target: TargetProfile) -> int:
    """FIDELITY FIX (found during the Phase-1-through-5 baseline audit,
    2026-09-08): the first version of this function computed `numPasses =
    ceil(log2(temp) / log2(max_strided)) + 1` unconditionally. Re-checked
    directly against the real source (vkFFT_Scheduler.h lines 2593-2234,
    quoted in full in docs/gpu_baseline_vkfft_research.md's own "Base pass
    count estimate" section): that formula is NEITHER of the two real
    branches. The real source picks one of:

        reorderFourStep and not Bluestein (VkFFT's own real DEFAULT --
        Structs.h: "Default 1" -- and Bluestein never applies by the time
        this baseline reaches here, since a length needing it was already
        rejected earlier as UNSUPPORTED_GPU_ALGORITHM):
            numPasses = ceil(log2(N) / log2(maxSingleSizeStrided))
        else (reorderFourStep disabled, a non-default VkFFT configuration
        this baseline does not model -- see module docstring):
            numPasses = 1 + ceil(log2(temp) / log2(maxSingleSizeStrided))

    where `temp` in the second branch is itself recomputed with a
    different `max_single` than the outer gate's own `temp`. The original
    version of this function used neither: it always took the shape of
    the second (non-default) branch, but computed `temp` with the WRONG
    divisor for that branch and against the WRONG base value (this
    function's own outer `temp`, already divided by `max_single`, not
    `length` itself) -- concretely different from either real formula for
    large N (they only coincidentally agreed at the two N this was first
    tested against). This is a baseline-fidelity fix, not an M2NDP-
    performance-driven change: it makes the port match VkFFT's own real,
    default-enabled code path, independent of anything M2NDP-specific.
    """
    max_single = (
        max_sequence_length_shared_memory(target) if non_strided
        else max_sequence_length_shared_memory_strided(target)
    )
    temp = math.ceil(length / max_single)
    if temp <= 1:
        return 1
    max_strided = max_sequence_length_shared_memory_strided(target)
    # reorderFourStep=True, no Bluestein (VkFFT's own real default path --
    # see this function's own docstring).
    num_passes = math.ceil(math.log2(length) / math.log2(max_strided))
    if num_passes > 3:
        raise VkfftUnsupportedError(
            f"length={length}: computed numPasses={num_passes} > 3 -- real "
            f"VkFFT hard-errors here too (VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH), "
            f"it does not support more than 3 upload passes"
        )
    return num_passes


# ---------------------------------------------------------------------------
# 5. Axis splitting -- vkFFT_Scheduler.h lines 2656-2888.
# ---------------------------------------------------------------------------


def _pow8_size(max_len: int) -> int:
    if max_len < 1:
        return 1
    return 8 ** (int(math.log2(max_len)) // 3)


def split_pow2_2pass(length: int, target: TargetProfile) -> tuple[int, int]:
    """Power-of-2, 2-pass split (lines 2656-2708): prefer a power-of-8
    first factor (`maxPow8SharedMemory`), falling back to the plain
    shared-memory-limited max sequence length, with a floor of
    `VKFFT_MIN_SPLIT_DIM` on the smaller split dimension and the larger
    factor always placed first. Returns `(a, b)` with `a >= b`,
    `a * b == length` -- `a` == clFFT/M2NDP notation's "far" (outer,
    potentially-recursed) factor, `b` == "near" (inner, batched-first).
    """
    shared_mem_max = max_sequence_length_shared_memory(target)
    strided_max = max_sequence_length_shared_memory_strided(target)
    pow8 = _pow8_size(shared_mem_max)
    if length // pow8 <= strided_max:
        split0 = pow8
    elif length // shared_mem_max <= strided_max:
        split0 = shared_mem_max
    else:
        split0 = shared_mem_max
    split1 = length // split0
    if split1 < VKFFT_MIN_SPLIT_DIM:
        split1 = VKFFT_MIN_SPLIT_DIM
        split0 = length // split1
    a, b = (split0, split1) if split0 >= split1 else (split1, split0)
    return a, b


def split_non_pow2_2pass(length: int, target: TargetProfile) -> tuple[int, int] | None:
    """Non-power-of-2, 2-pass split (lines 2709-2749): exact-divisor
    search starting at `ceil(sqrt(length))`, walking DOWNWARD, for the
    first divisor `d` with `d <= maxSingleSizeStrided` and `length/d <=
    maxSequenceLengthSharedMemory`. Returns `None` if no such divisor
    exists (the real source falls through to a 3-pass attempt in that
    case -- see `plan_multi_pass`)."""
    strided_max = max_sequence_length_shared_memory_strided(target)
    shared_mem_max = max_sequence_length_shared_memory(target)
    sqrt_seq = math.ceil(math.sqrt(length))
    for i in range(sqrt_seq):
        d = sqrt_seq - i
        if d <= 0:
            break
        if length % d != 0:
            continue
        if d <= strided_max and (length // d) <= shared_mem_max:
            return length // d, d
    return None


def split_pow2_3pass(length: int, target: TargetProfile) -> tuple[int, int, int]:
    """Power-of-2, 3-pass split (lines 2751-2844): the same power-of-8
    preference applied to pick the first factor, remainder split 2-pass-
    style between the other two."""
    a, remainder = split_pow2_2pass(length, target)
    remainder_a, remainder_b = split_pow2_2pass(remainder, target)
    return a, remainder_a, remainder_b


def split_non_pow2_3pass(length: int, target: TargetProfile) -> tuple[int, int, int] | None:
    """Non-power-of-2, 3-pass split (lines 2845-2888): outer divisor
    search near `ceil(N^(1/3))` walking downward, then an inner sqrt-style
    search on the remaining quotient (mirroring the 2-pass search's own
    downward walk) -- `None` if no legal triple exists (the real source's
    own `numPasses=4` -> immediate hard error, ported here as this
    function returning `None`, converted to UNSUPPORTED_GPU_ALGORITHM by the
    caller)."""
    strided_max = max_sequence_length_shared_memory_strided(target)
    cbrt_seq = math.ceil(round(length ** (1.0 / 3.0), 9))
    while cbrt_seq ** 3 < length:
        cbrt_seq += 1
    for i in range(cbrt_seq):
        d0 = cbrt_seq - i
        if d0 <= 0:
            break
        if length % d0 != 0:
            continue
        remain = length // d0
        sqrt_seq = math.ceil(math.sqrt(remain))
        for j in range(sqrt_seq):
            d1 = sqrt_seq - j
            if d1 <= 0:
                break
            if remain % d1 != 0:
                continue
            d2 = remain // d1
            if d0 <= strided_max and d1 <= strided_max:
                return d0, d1, d2
    return None


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


def min_registers_per_thread_pow2(fft_length: int, max_rhs: int) -> int:
    """`registers_per_thread_per_radix[2]` for a PURE power-of-2 sequence
    (vkFFT_Scheduler.h, quoted in the original VkFFT research report):
    `= grouping_radix if loc_multipliers[2] > final_loc_multipliers_pow2
    else fft_length` -- since a pure-pow2 sequence's own register table
    has exactly one nonzero radix entry, `min == max == registers_per_
    thread_per_radix[2]` here, so this doubles as this baseline's own
    `min_registers_per_thread` for the pow2 case."""
    grouping_radix = choose_pow2_grouping_radix(fft_length, max_rhs)
    total_bits = fft_length.bit_length() - 1
    group_bits = grouping_radix.bit_length() - 1
    return grouping_radix if total_bits > group_bits else fft_length


def _prime_multiplicities(radices: tuple[int, ...]) -> dict[int, int]:
    """`loc_multipliers[p]` -- how many times each prime factor of 2, 3, 5,
    or 7 appears across every radix stage in `radices` (this baseline's
    own direct-radix vocabulary, `_DIRECT_RADIX_ORDER = {2,3,4,5,6,7,8,9,
    10}`, is exactly the set of composites of these four primes it ever
    plans, so no other prime can appear here)."""
    counts = {2: 0, 3: 0, 5: 0, 7: 0}
    factor_map = {2: {2: 1}, 3: {3: 1}, 4: {2: 2}, 5: {5: 1}, 6: {2: 1, 3: 1},
                  7: {7: 1}, 8: {2: 3}, 9: {3: 2}, 10: {2: 1, 5: 1}}
    for radix in radices:
        for prime, count in factor_map[radix].items():
            counts[prime] += count
    return counts


def min_registers_per_thread_for(length: int, radices: tuple[int, ...], max_rhs: int) -> int:
    """`min_registers_per_thread`, VkFFT's own per-leaf register-count
    input to `axisblock_threads_per_transform` below. Exact for pure
    power-of-2 (`min_registers_per_thread_pow2`, reusing this module's own
    already-exact pow2 stage-grouping scheduler) and, since the 2026-09-09
    source-fidelity re-audit, exact for every other combination too: the
    full base `{2,3,5,7}` register table (`registers_per_thread_base_
    table`) is now ported literally (replacing the earlier `min(radices)`
    proxy, which sat on a path this baseline can return
    `BaselineStatus.OK` from and was never acceptable there)."""
    if length & (length - 1) == 0:
        return min_registers_per_thread_pow2(length, max_rhs)
    counts = _prime_multiplicities(radices)
    table = registers_per_thread_base_table(counts[2], counts[3], counts[5], counts[7])
    return min(v for v in table.values() if v != 0)


def axisblock_threads_per_transform(fft_dim: int, min_registers_per_thread: int) -> int:
    """`AxisBlockSplitter.h` section 2.1's universal formula: `max(1,
    ceil(fftDim/min_registers_per_thread) // registerBoost)` --
    `registerBoost` fixed at 1 throughout this baseline (module
    docstring), so this reduces to `max(1, ceil(fftDim/min_registers_
    per_thread))`."""
    return max(1, -(-fft_dim // min_registers_per_thread))


def _grouped_batch_seed(fft_dim: int, target: TargetProfile, *, single_upload: bool) -> int:
    """`AxisBlockSplitter.h` section 2.0's shared top-of-function seed for
    `axis->groupedBatch`, `axis_id==0` only. `single_upload=True` is the
    real source's own `(numAxisUploads[axis_id]==1 && axis_id==0)` case
    (non-strided branch); `single_upload=False` covers every multi-pass
    upload (this baseline's own `reorderFourStep=True` fixed default
    makes the real source's `!reorderFourStep && axis_upload_id==0`
    alternative for reaching the non-strided branch unreachable, so
    `single_upload` here is equivalent to `num_passes(this plan) == 1`,
    not merely `upload_id==0`)."""
    if single_upload:
        shared_mem_max = max_sequence_length_shared_memory(target)
        scaled = shared_mem_max // fft_dim
        return scaled if scaled > _MAX_BATCH_COALESCED else _MAX_BATCH_COALESCED
    strided_max = max_sequence_length_shared_memory_strided(target)
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
    defaults to `VKFFT_WARP_SIZE` (32, source-faithful). The M2NDP-adapted
    baseline (`plan_m2ndp`) passes `target.interleave_chunk_uthreads` (8)
    instead -- see docs/gpu_planner_m2ndp_target_mapping.md's own VkFFT
    row. `VKFFT_AIM_THREADS` is NOT adapted here -- it is VkFFT's own
    hand-tuned occupancy target (a fixed algorithm parameter, not a
    hardware query), unchanged either way.
    """
    if threads_per_transform // warp_size == 1 and threads_per_transform / warp_size < 1.5:
        estimate = VKFFT_AIM_THREADS // warp_size
    else:
        estimate = VKFFT_AIM_THREADS // threads_per_transform
    estimate = max(estimate, 1)
    if threads_per_transform < VKFFT_AIM_THREADS and (threads_per_transform < warp_size or use_rader):
        return estimate
    return 1


def _postprocess_axis_upload0(
    axis_block0: int, seed_batch: int, fft_dim: int, target: TargetProfile, *,
    num_passes: int, max_rhs: int, original_length: int,
) -> tuple[int, int]:
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

    Returns the FINAL `(axis_block0, batch)` pair -- note the swap at the
    end (lines 350-364) can return `axis_block0` different from the value
    passed in.
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
    shared_mem_pow2 = max_sequence_length_shared_memory_pow2(target)
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
    max_seq_shared = max_sequence_length_shared_memory(target)
    while batch * fft_dim > max_seq_shared and batch > 1:
        batch //= 2

    # Lines 350-364: the real axisBlock[0] <-> axisBlock[1] SWAP, gated on
    # the non-Pow2 shared-memory bound -- genuinely distinct from the
    # round-up-to-po2 step above both in formula and in which (much more
    # processed) `batch` value it inspects.
    if (
        (fft_dim % 2 == 0 or axis_block0 < VKFFT_NUM_SHARED_BANKS // 4)
        and batch > 1
        and batch * fft_dim < max_seq_shared
    ):
        axis_block0, batch = batch, axis_block0

    return axis_block0, max(batch, 1)


def axisblock_batch_multipass_first(fft_dim: int, target: TargetProfile) -> int:
    """`AxisBlockSplitter.h` section 2.2: `reorderFourStep(True, fixed) &&
    numAxisUploads>1` -> `axisBlock[1] = axis->groupedBatch` (the seeded
    value, unchanged) -- what upload 0 (this baseline's own `near_fft`
    leaf, first-processed pass) of a multi-pass plan actually uses."""
    return _grouped_batch_seed(fft_dim, target, single_upload=False)


def axisblock_batch_multipass_later(
    fft_dim: int, threads_per_transform: int, target: TargetProfile, *, stage_start_size: int,
) -> int:
    """`AxisBlockSplitter.h` section 2.3: a later upload (`axis_upload_id
    > 0` -- every leaf after this baseline's own `near_fft`) grows the
    seeded batch by `scale = aimThreads // (threads_per_transform *
    groupedBatch)` when that still fits shared memory, then caps the
    result by `stageStartSize` (the product of every PRIOR upload's own
    split factor -- `original_length // fft_dim` at this leaf's own
    recursion depth), the NVIDIA vendor halving loop (section 2.3,
    `vendorID==0x10DE` -- this baseline's own fixed choice, see module
    constant), `maxComputeWorkGroupSize`, and finally the largest-divisor
    search bringing the product under `maxThreadsNum`."""
    grouped_batch = _grouped_batch_seed(fft_dim, target, single_upload=False)
    scale = VKFFT_AIM_THREADS // max(threads_per_transform * grouped_batch, 1)
    max_seq_shared = max_sequence_length_shared_memory(target)
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


def axisblock_for_leaf(
    length: int, radices: tuple[int, ...], *, max_rhs: int, num_passes: int, upload_id: int,
    original_length: int, target: TargetProfile, warp_size: int = VKFFT_WARP_SIZE,
) -> tuple[int, int]:
    """One leaf's own `(threads_per_transform, transforms_per_block)` --
    the M2NDP translation's `(workers_per_fft, fft_slots_wanted)` pair --
    dispatching to the single-pass or multi-pass (first-upload vs. later-
    upload) formula above depending on this plan's own `num_passes` and
    this specific leaf's own `upload_id` (0 = this baseline's own
    `near_fft`, the first-processed pass; >0 = any leaf reached only after
    at least one PRE/MIDDLE transpose). `warp_size`: forwarded to
    `axisblock_batch_single_pass` only -- see that function's own
    docstring; the multi-pass formulas below have no warp-size input at
    all in the real source."""
    min_regs = min_registers_per_thread_for(length, radices, max_rhs)
    threads_per_transform = axisblock_threads_per_transform(length, min_regs)
    if num_passes == 1:
        seed = axisblock_batch_single_pass(length, threads_per_transform, target, warp_size=warp_size)
        threads_per_transform, batch = _postprocess_axis_upload0(
            threads_per_transform, seed, length, target,
            num_passes=num_passes, max_rhs=max_rhs, original_length=original_length,
        )
    elif upload_id == 0:
        seed = axisblock_batch_multipass_first(length, target)
        threads_per_transform, batch = _postprocess_axis_upload0(
            threads_per_transform, seed, length, target,
            num_passes=num_passes, max_rhs=max_rhs, original_length=original_length,
        )
    else:
        stage_start_size = original_length // length
        batch = axisblock_batch_multipass_later(
            length, threads_per_transform, target, stage_start_size=stage_start_size,
        )
    return threads_per_transform, batch


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
    warp_size: int = VKFFT_WARP_SIZE,
) -> tuple[FFTLeafPlan | None, BaselineResult | None]:
    idx = node_id[0]
    node_id[0] += 1
    try:
        radices = leaf_radix_sequence(m, r)
    except VkfftUnsupportedError as exc:
        extra = classify_vkfft_residual_scheme(exc.residual) if exc.residual is not None else {}
        gpu_config = GPUKernelConfig(source="vkfft", length=m, radices=(), extra=extra)
        return None, unsupported(BaselineStatus.UNSUPPORTED_GPU_ALGORITHM, gpu_config, str(exc))

    if not radices:
        gpu_config = GPUKernelConfig(source="vkfft", length=m, radices=())
        return None, unsupported(
            BaselineStatus.UNSUPPORTED_GPU_ALGORITHM, gpu_config, f"length={m}: no radix sequence produced"
        )

    workers_per_fft, fft_slots_wanted = axisblock_for_leaf(
        m, radices, max_rhs=r, num_passes=num_passes, upload_id=upload_id,
        original_length=original_length, target=target, warp_size=warp_size,
    )
    gpu_config = GPUKernelConfig(
        source="vkfft", length=m, radices=radices,
        extra={
            "workers_per_fft": workers_per_fft, "transforms_per_block": fft_slots_wanted,
            "register_boost": 1, "upload_id": upload_id,
        },
    )
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
    warp_size: int = VKFFT_WARP_SIZE,
) -> tuple[FFTNode | None, BaselineResult | None]:
    """`factors`: this level's own (a, b) or (a, b, c) split, near-to-far
    (b == innermost/near, matching this module's own axis-split return
    convention of "largest first" reinterpreted as far-first here -- see
    call site for the exact ordering used). `upload_id`: this level's own
    position in VkFFT's real upload/pass ordering (0 = first-processed --
    see `axisblock_for_leaf`'s own docstring); each recursive step into
    `far_child` advances it by one, matching VkFFT's real per-upload
    numbering."""
    if len(factors) == 1:
        return _leaf_result(
            m, r, inverse=inverse, is_root=is_root, node_id=node_id, target=target,
            num_passes=num_passes, upload_id=upload_id, original_length=original_length,
            warp_size=warp_size,
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
        warp_size=warp_size,
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
        num_passes=num_passes, upload_id=upload_id + 1, original_length=original_length,
        warp_size=warp_size,
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


def plan(
    length: int,
    *,
    batch: int = 1,
    inverse: bool = False,
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
    warp_size: int = VKFFT_WARP_SIZE,
) -> BaselineResult:
    """Top-level VkFFT-style baseline entry point: decides 1/2/3-pass via
    `choose_num_passes`, then splits via this module's own axis-splitting
    functions (pow2 vs. non-pow2), building an M2NDP plan with the same
    PRE/near/MIDDLE/far/POST shape clfft.py's large-1D path already
    establishes.

    SOURCE-FAITHFUL (default `warp_size=VKFFT_WARP_SIZE`): `choose_num_
    passes`/`split_pow2_*`/`split_non_pow2_*` already consistently use
    `target.spad_capacity_bytes` for every LDS-equivalent quantity (this
    baseline's pre-existing target adaptation, unchanged) -- `warp_size`
    is the ONE remaining fixed-GPU hardware input in this module
    (`axisblock_batch_single_pass`'s own SIMT-lockstep-granularity
    estimate). See `plan_m2ndp` for the M2NDP-adapted sibling and docs/
    gpu_planner_m2ndp_target_mapping.md.
    """
    is_po2 = (length & (length - 1)) == 0
    try:
        num_passes = choose_num_passes(length, non_strided=True, target=target)
    except VkfftUnsupportedError as exc:
        gpu_config = GPUKernelConfig(source="vkfft", length=length, radices=())
        return unsupported(BaselineStatus.UNSUPPORTED_GPU_ALGORITHM, gpu_config, str(exc))

    if num_passes == 1:
        factors: tuple[int, ...] = (length,)
    elif num_passes == 2:
        if is_po2:
            a, b = split_pow2_2pass(length, target)
        else:
            split = split_non_pow2_2pass(length, target)
            if split is None:
                gpu_config = GPUKernelConfig(source="vkfft", length=length, radices=())
                return unsupported(
                    BaselineStatus.UNSUPPORTED_GPU_ALGORITHM, gpu_config,
                    f"length={length}: no legal non-power-of-2 2-pass axis split found "
                    f"under this target's scratchpad budget",
                )
            a, b = split
        factors = apply_four_step_reordering((a, b))
    else:
        if is_po2:
            a, r1, r2 = split_pow2_3pass(length, target)
        else:
            split = split_non_pow2_3pass(length, target)
            if split is None:
                gpu_config = GPUKernelConfig(source="vkfft", length=length, radices=())
                return unsupported(
                    BaselineStatus.UNSUPPORTED_GPU_ALGORITHM, gpu_config,
                    f"length={length}: no legal non-power-of-2 3-pass axis split found "
                    f"under this target's scratchpad budget (real VkFFT would hard-error "
                    f"VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH here too)",
                )
            a, r1, r2 = split
        factors = apply_four_step_reordering((a, r1, r2))

    node_id = [0]
    # factors is far-to-near ascending-nest order for _recursive_result's own
    # "pop from the end" convention: factors[-1] is this level's near/b.
    node, failure = _recursive_result(
        length, batch, factors, inverse=inverse, is_root=True, node_id=node_id, target=target,
        num_passes=num_passes, upload_id=0, original_length=length, warp_size=warp_size,
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
        source="vkfft" if warp_size == VKFFT_WARP_SIZE else "vkfft-m2ndp",
        length=length, radices=top_radices,
        extra={"num_passes": num_passes, "factors": factors, "is_pow2": is_po2},
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
    m2ndp` CLI baseline. Identical to `plan` in every respect except
    `warp_size`: `target.interleave_chunk_uthreads` (8, M2NDP's own
    physical concurrent-microthread width -- the closest thing this target
    has to "threads executing in genuine lockstep") instead of VkFFT's own
    representative-GPU `VKFFT_WARP_SIZE` (32) -- see docs/
    gpu_planner_m2ndp_target_mapping.md's own VkFFT row. Every other
    hardware input in this module (`choose_num_passes`/`split_pow2_*`/
    `split_non_pow2_*`'s own `target.spad_capacity_bytes` usage) was
    already M2NDP-adapted before this task -- this is the one remaining
    gap this task's own audit found.
    """
    return plan(length, batch=batch, inverse=inverse, target=target, warp_size=target.interleave_chunk_uthreads)
