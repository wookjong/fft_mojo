from __future__ import annotations

"""rocFFT PRODUCTION (non-tuned) planner -- a SEPARATE baseline from
gpu_baseline/rocfft.py.

Phase 4 of the 2026-09-08 baseline fidelity/mapping audit found that
`rocfft.py` (the offline-tuner port) is NOT what an ordinary
`rocfft_plan_create()` call ever runs -- see that module's own docstring
and docs/gpu_baseline_rocfft_default_research.md for the original
call-graph evidence. A dedicated follow-up deep dive (docs/
gpu_baseline_rocfft_default_deepdive.md) then extracted the REMAINING
production mechanism this baseline was missing: `NodeFactory::
Decide1DScheme` (node_factory.cpp:628-819), traced and quoted in full,
covering:

* `config_sbrr.py`'s COMPLETE compiled-in single-kernel (`CS_KERNEL_
  STOCKHAM`) table -- every length from 2 through 4096 that the real
  library ships an AOT/RTC kernel for (see `SBRR_TABLE`), not just the
  11 rows the first research pass happened to quote.
* `map1DLengthSingle`/`map1DLengthDouble` -- literal hardcoded
  `length -> divLength1` maps (node_factory.cpp:42-207) selecting
  `CS_L1D_CC` (a fused block-column/block-row two-kernel decomposition,
  `SBCC`+`SBRC`) for lengths beyond the single-kernel table, up to
  262144 for power-of-2 lengths.
* `CS_L1D_TRTRT`'s own fallback chain (`get_largest_pow2_length`,
  `get_explicitly_supported_factor`, `get_largest_supported_factor`) for
  whatever `map1DLengthSingle` itself doesn't cover.
* The real `length > 4096` occupancy heuristic
  (`totalBatch/transforms_per_block >= multiProcessorCount`) that can
  still route even a length WITH a compiled single kernel through the
  multi-kernel path instead, at low batch.

SCOPE LIMIT, faithful not evasive (section 3 of the task this was built
from: "If an exact GPU configuration cannot be represented by current
M2NDP codegen: preserve the ... configuration as metadata, return
UNSUPPORTED_CURRENT_CODEGEN. Do NOT change the ... selection to make it
representable."): `CS_L1D_CC`'s own SBCC/SBRC kernels are FUSED block-
tiled-transpose-plus-FFT kernels -- a genuinely different mechanism from
this project's own PRE-transpose/leaf-FFT/POST-transpose six-step shape
(clfft.py's and vkfft.py's large-1D paths both reuse that shape because
their own upstream algorithms really do decompose that way; rocFFT's
SBCC/SBRC do NOT -- they read/write shared memory in column/row TILES
directly inside the FFT kernel itself, with no separate transpose stage
at all). Forcing SBCC/SBRC onto the PRE/near/MIDDLE/far/POST mechanism
would be silently CHANGING rocFFT's own algorithm to fit what this
repository's codegen already has, exactly what this whole baseline effort
is not supposed to do. So: this module DECIDES `CS_L1D_CC`/`CS_L1D_TRTRT`
correctly and faithfully (the exact scheme, `divLength1`, and each
sub-kernel's own real config from `config_sbcc.py`/`config_sbrc.py`), and
reports that decision honestly in `GPUKernelConfig.extra`, but returns
`UNSUPPORTED_CURRENT_CODEGEN` rather than building an M2NDP plan for it --
a real, unimplemented mechanism this repository's `AddressMapping`/codegen
has no equivalent for, not a hardware impossibility and not an algorithm
this baseline declined to understand.

Every length in the REQUIRED test set (2, 4, 8, 16, 32, 64, 128, 256, 512,
1024, 2048, 4096) is directly covered by the compiled single-kernel table
(`SBRR_TABLE`) and needs none of the CC/TRTRT machinery at all -- verified
directly against `config_sbrr.py`'s own real rows (research deep-dive
section 1).

============================================================================
2026-09-09: THE REAL SOLUTION-MAP LAYER (`rocfft_upstream_solution_map.py`)
============================================================================
Until this pass, `plan()` below only reproduced `Decide1DScheme` -- but
real production rocFFT ALSO probes `ApplySolution` (the shipped, per-arch
solution-map file) before settling on a scheme, and a non-dummy match can
OVERRIDE what `Decide1DScheme` picked (see that module's own docstring for
the full derivation from `plan.cpp`/`node_factory.cpp`/`compute_scheme.cpp`,
all at the SAME pinned commit as `rocfft.py`'s own PROVENANCE). `plan()` now
calls `apply_solution` first; a match is used if found, else `decide_scheme`
runs exactly as before.

A DECISIVE finding from implementing this layer: this project's own
consistent out-of-place assumption (M2NDP plans always use separate
input/output buffers) means `apply_solution` is PROVEN -- by exhaustively
scanning the entire shipped gfx908 file, not merely assumed -- to find ZERO
matches for this baseline's own FP32 domain (every single-precision complex
entry in that file is in-place-only). So `plan()`'s own observable behavior
is unchanged by this layer for every length this baseline can be called
with; what changed is that this is now DEMONSTRATED, not assumed by
omission -- see `rocfft_upstream_solution_map.py`'s own module docstring
for the full proof and `verify_gpu_baseline_source_fidelity.py` for the
exhaustive-scan test and a direct test of the lookup/resolution machinery
against the file's own one real (in-place) non-dummy entry.

NON-NEGOTIABLE (see gpu_baseline/common.py's own module docstring): no
import from planning.search.fft_cost_model / planning.execution.fft_plan_cooperative's own
worker-count heuristic / planning.execution.fft_plan_persistent / planning.
fft_plan_lanes. This module has no search and no benchmarking at all --
every decision is a table lookup or the same deterministic fallback chain
production rocFFT itself runs, never a search over alternatives (that is
`gpu-rocfft`/`gpu-rocfft-tuned`'s own, separate job).

============================================================================
FIXED IMPLEMENTATION PARAMETERS (section 3: "must remain FIXED for all GPU
baseline candidates and must be documented"):
============================================================================
ROCFFT_DEFAULT_MULTIPROCESSOR_COUNT = 120
    The real `>4096` occupancy heuristic
    (`totalBatch/transforms_per_block >= multiProcessorCount`) needs a real
    GPU's own compute-unit count, which M2NDP has no equivalent hardware
    query for. 120 is gfx908 (AMD Instinct MI100)'s own real CU count --
    one of only three archs confirmed (research deep-dive, and the earlier
    rocfft_default_research.md) to ship a populated rocFFT solution-map
    file, making it a genuinely representative "real rocFFT target," not
    an arbitrary guess. Held fixed across every length/batch this baseline
    ever plans.
============================================================================
"""

from dataclasses import dataclass

from planning.core.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile
from radix_spec import SUPPORTED_RADICES

from . import rocfft_upstream_solution_map as upstream_sol_map
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
# from). Same repository/commit as rocfft.py's own PROVENANCE, but a
# DISJOINT set of source files/functions -- confirming these two baselines
# really do port two separate production code paths, not two views of one.
# ---------------------------------------------------------------------------
PROVENANCE = BaselineProvenance(
    library="rocFFT",
    upstream_repository="https://github.com/ROCm/rocm-libraries",
    upstream_commit="bee97df517907c771de17189cb867d3c401285ae (develop, projects/rocfft/) -- "
    "pinned 2026-09-09, resolving the earlier floating 'develop HEAD as of 2026-09-08' note: "
    "re-fetched every source file below directly at this exact SHA (the SAME commit the "
    "SIBLING gpu-rocfft-tuned baseline already pinned) and confirmed all are live there, "
    "so this baseline and rocfft.py now cite one single, consistent revision, never two.",
    source_files=(
        "projects/rocfft/library/src/node_factory.cpp",
        "projects/rocfft/library/src/include/node_factory.h",
        "projects/rocfft/library/src/tree_node_1D.cpp",
        "projects/rocfft/library/src/include/tree_node.h",
        "projects/rocfft/library/src/include/function_pool.h",
        "projects/rocfft/library/src/device/kernel-generator.py",
        "projects/rocfft/library/src/device/kernels/configs/config_sbrr.py",
        "projects/rocfft/library/src/device/kernels/configs/config_sbcc.py",
        "projects/rocfft/library/src/device/kernels/configs/config_sbrc.py",
        "projects/rocfft/library/src/plan.cpp",
        "projects/rocfft/library/src/solution_map.cpp",
        "projects/rocfft/library/src/include/solution_map.h",
        "projects/rocfft/library/src/compute_scheme.cpp",
        "projects/rocfft/library/solution_map/gfx908_rocfft_solution_map.dat",
    ),
    baseline_version="gpu-baseline-v1",
    source_functions={
        "SBRR_TABLE": "device/kernels/configs/config_sbrr.py: sbrr_kernels (compiled-in single-kernel table, full 2..4096)",
        "SBCC_TABLE/SBRC_TABLE": "device/kernels/configs/config_sbcc.py, config_sbrc.py (CS_L1D_CC's own two sub-kernel tables)",
        "MAP_1D_LENGTH_SINGLE": "node_factory.cpp: NodeFactory::map1DLengthSingle",
        "decide_scheme": "node_factory.cpp: NodeFactory::Decide1DScheme",
        "get_largest_pow2_length": "function_pool.h: function_pool::get_largest_pow2_length",
        "get_explicitly_supported_factor/get_largest_supported_factor": "node_factory.cpp: search_pool + get_explicitly_supported_factor + get_largest_supported_factor",
        "plan (transforms_per_block)": "device/kernel-generator.py: generate_kernel_functions's own workgroup_size // threads_per_transform line",
        "rocfft_upstream_solution_map.apply_solution": "plan.cpp: ApplySolution/RecursivelyApplySol/GenerateProbKeys/GetNodeToken",
    },
    notes=(
        "Ports the compiled-in single-kernel (config_sbrr.py) table, the "
        "CS_L1D_CC/CS_L1D_TRTRT DECISION logic (which scheme, which "
        "divLength1/sub-kernel configs), AND (2026-09-09) the real "
        "solution-map override layer (rocfft_upstream_solution_map.py) that "
        "runs BEFORE this decision chain in real production rocFFT -- see "
        "that module's own docstring. Only ever BUILDS an M2NDP plan for "
        "the single-kernel (CS_KERNEL_STOCKHAM) case, whether that came "
        "from the compiled table, the solution map, or Decide1DScheme's "
        "own fallback formulas. CS_L1D_CC/TRTRT decisions (from either "
        "source) are reported faithfully in diagnostics but return "
        "UNSUPPORTED_CURRENT_CODEGEN, since SBCC/SBRC's fused block-tiled "
        "transpose+FFT kernels have no equivalent AddressMapping/codegen "
        "mechanism in this repository (see module docstring's own SCOPE "
        "LIMIT) -- forcing them onto the unrelated PRE/MIDDLE/POST six-step "
        "shape would silently change rocFFT's own real algorithm."
    ),
)

# ---------------------------------------------------------------------------
# Fixed baseline-wide parameter (see module docstring).
# ---------------------------------------------------------------------------
ROCFFT_DEFAULT_MULTIPROCESSOR_COUNT = 120


@dataclass(frozen=True)
class DefaultConfig:
    workgroup_size: int
    threads_per_transform: int
    factors: tuple[int, ...]


# ---------------------------------------------------------------------------
# library/src/device/kernels/configs/config_sbrr.py -- the COMPLETE
# compiled-in single-kernel (CS_KERNEL_STOCKHAM) table for lengths 2
# through 4096, transcribed row-for-row from docs/
# gpu_baseline_rocfft_default_deepdive.md's own full-file quote (which
# itself fetched and read config_sbrr.py, 509 lines, in full -- not the
# 11-row partial excerpt the first research pass had to work from).
#
# length=1 is omitted: it is a degenerate "FFT" needing zero radix stages,
# out of scope for every planner in this repository (see make_fft_kernel.
# py's own `n < 2` rejection).
#
# `transforms_per_block = workgroup_size // threads_per_transform`
# (kernel-generator.py's own one-line formula, computed in `plan()`, never
# stored here) -- NOT always an exact division in the real table (e.g.
# length=9: wgs=64/tpt=3 -> 64/3 not integral; real GPU hardware can
# launch a workgroup wider than tpt*transforms_per_block strictly needs,
# wasting a few threads to wavefront/warp-size rounding). Floor division
# is exactly what the real generator does.
# ---------------------------------------------------------------------------
SBRR_TABLE: dict[int, DefaultConfig] = {
    2: DefaultConfig(64, 1, (2,)),
    3: DefaultConfig(64, 1, (3,)),
    4: DefaultConfig(128, 1, (4,)),
    5: DefaultConfig(128, 1, (5,)),
    6: DefaultConfig(128, 1, (6,)),
    7: DefaultConfig(64, 1, (7,)),
    8: DefaultConfig(64, 4, (4, 2)),
    9: DefaultConfig(64, 3, (3, 3)),
    10: DefaultConfig(64, 1, (10,)),
    11: DefaultConfig(128, 1, (11,)),
    12: DefaultConfig(128, 6, (6, 2)),
    13: DefaultConfig(64, 1, (13,)),
    14: DefaultConfig(128, 7, (7, 2)),
    15: DefaultConfig(128, 5, (3, 5)),
    16: DefaultConfig(64, 4, (4, 4)),
    17: DefaultConfig(256, 1, (17,)),
    18: DefaultConfig(64, 6, (3, 6)),
    20: DefaultConfig(256, 10, (5, 4)),
    21: DefaultConfig(128, 7, (3, 7)),
    22: DefaultConfig(64, 2, (11, 2)),
    24: DefaultConfig(256, 8, (8, 3)),
    25: DefaultConfig(256, 5, (5, 5)),
    26: DefaultConfig(64, 2, (13, 2)),
    27: DefaultConfig(256, 9, (3, 3, 3)),
    28: DefaultConfig(64, 4, (7, 4)),
    30: DefaultConfig(128, 10, (10, 3)),
    32: DefaultConfig(128, 16, (8, 4)),
    33: DefaultConfig(256, 11, (11, 3)),
    34: DefaultConfig(256, 17, (17, 2)),
    35: DefaultConfig(256, 7, (5, 7)),
    36: DefaultConfig(64, 6, (6, 6)),
    39: DefaultConfig(256, 13, (13, 3)),
    40: DefaultConfig(128, 10, (10, 4)),
    42: DefaultConfig(256, 7, (7, 6)),
    44: DefaultConfig(64, 4, (11, 4)),
    45: DefaultConfig(128, 15, (5, 3, 3)),
    48: DefaultConfig(64, 16, (4, 3, 4)),
    49: DefaultConfig(64, 7, (7, 7)),
    50: DefaultConfig(256, 10, (10, 5)),
    51: DefaultConfig(256, 17, (17, 3)),
    52: DefaultConfig(64, 4, (13, 4)),
    54: DefaultConfig(256, 18, (6, 3, 3)),
    55: DefaultConfig(256, 11, (5, 11)),
    56: DefaultConfig(128, 8, (7, 8)),
    60: DefaultConfig(64, 10, (6, 10)),
    63: DefaultConfig(256, 21, (3, 3, 7)),
    64: DefaultConfig(64, 16, (4, 4, 4)),
    65: DefaultConfig(256, 13, (13, 5)),
    66: DefaultConfig(256, 11, (6, 11)),
    68: DefaultConfig(256, 17, (17, 4)),
    70: DefaultConfig(256, 14, (2, 5, 7)),
    72: DefaultConfig(64, 9, (8, 3, 3)),
    75: DefaultConfig(256, 25, (5, 5, 3)),
    77: DefaultConfig(256, 11, (7, 11)),
    78: DefaultConfig(256, 13, (6, 13)),
    80: DefaultConfig(64, 10, (5, 2, 8)),
    81: DefaultConfig(128, 27, (3, 3, 3, 3)),
    84: DefaultConfig(128, 12, (7, 2, 6)),
    85: DefaultConfig(256, 17, (17, 5)),
    88: DefaultConfig(128, 11, (11, 8)),
    90: DefaultConfig(64, 9, (3, 3, 10)),
    91: DefaultConfig(256, 13, (7, 13)),
    96: DefaultConfig(128, 16, (6, 16)),
    98: DefaultConfig(256, 14, (2, 7, 7)),
    99: DefaultConfig(256, 11, (3, 3, 11)),
    100: DefaultConfig(64, 10, (10, 10)),
    102: DefaultConfig(128, 17, (17, 6)),
    104: DefaultConfig(64, 8, (13, 8)),
    105: DefaultConfig(256, 21, (7, 3, 5)),
    108: DefaultConfig(256, 36, (6, 6, 3)),
    110: DefaultConfig(256, 11, (2, 5, 11)),
    112: DefaultConfig(256, 16, (16, 7)),
    117: DefaultConfig(64, 13, (13, 9)),
    119: DefaultConfig(256, 17, (17, 7)),
    120: DefaultConfig(64, 12, (6, 10, 2)),
    121: DefaultConfig(128, 11, (11, 11)),
    125: DefaultConfig(256, 25, (5, 5, 5)),
    126: DefaultConfig(256, 42, (6, 7, 3)),
    128: DefaultConfig(256, 16, (16, 8)),
    130: DefaultConfig(64, 13, (13, 10)),
    132: DefaultConfig(128, 22, (11, 6, 2)),
    135: DefaultConfig(128, 9, (5, 3, 3, 3)),
    136: DefaultConfig(128, 17, (17, 8)),
    140: DefaultConfig(64, 28, (7, 5, 4)),
    143: DefaultConfig(256, 13, (13, 11)),
    144: DefaultConfig(128, 12, (6, 6, 4)),
    147: DefaultConfig(64, 21, (7, 7, 3)),
    150: DefaultConfig(64, 5, (10, 5, 3)),
    153: DefaultConfig(128, 17, (17, 9)),
    154: DefaultConfig(128, 22, (11, 7, 2)),
    156: DefaultConfig(128, 13, (3, 4, 13)),
    160: DefaultConfig(256, 16, (16, 10)),
    162: DefaultConfig(256, 27, (6, 3, 3, 3)),
    165: DefaultConfig(64, 11, (11, 5, 3)),
    168: DefaultConfig(256, 56, (8, 7, 3)),
    169: DefaultConfig(256, 13, (13, 13)),
    170: DefaultConfig(128, 17, (17, 10)),
    175: DefaultConfig(256, 35, (5, 5, 7)),
    176: DefaultConfig(64, 16, (11, 16)),
    180: DefaultConfig(256, 60, (10, 6, 3)),
    182: DefaultConfig(64, 13, (13, 2, 7)),
    187: DefaultConfig(128, 17, (17, 11)),
    189: DefaultConfig(64, 21, (7, 3, 3, 3)),
    192: DefaultConfig(128, 16, (6, 4, 4, 2)),
    195: DefaultConfig(64, 13, (13, 5, 3)),
    196: DefaultConfig(64, 28, (4, 7, 7)),
    198: DefaultConfig(128, 22, (11, 2, 9)),
    200: DefaultConfig(64, 20, (10, 10, 2)),
    204: DefaultConfig(128, 17, (17, 4, 3)),
    208: DefaultConfig(64, 16, (13, 16)),
    210: DefaultConfig(64, 30, (10, 7, 3)),
    216: DefaultConfig(256, 36, (6, 6, 6)),
    220: DefaultConfig(128, 22, (10, 2, 11)),
    221: DefaultConfig(128, 17, (17, 13)),
    224: DefaultConfig(64, 16, (7, 2, 2, 2, 2, 2)),
    225: DefaultConfig(256, 75, (5, 5, 3, 3)),
    231: DefaultConfig(256, 33, (11, 7, 3)),
    234: DefaultConfig(64, 26, (13, 9, 2)),
    238: DefaultConfig(64, 17, (17, 7, 2)),
    240: DefaultConfig(128, 48, (8, 5, 6)),
    242: DefaultConfig(128, 22, (11, 2, 11)),
    243: DefaultConfig(256, 81, (3, 3, 3, 3, 3)),
    245: DefaultConfig(256, 35, (7, 5, 7)),
    250: DefaultConfig(128, 25, (10, 5, 5)),
    252: DefaultConfig(64, 63, (7, 3, 3, 4)),
    255: DefaultConfig(64, 17, (17, 5, 3)),
    256: DefaultConfig(64, 64, (4, 4, 4, 4)),
    260: DefaultConfig(64, 26, (13, 10, 2)),
    264: DefaultConfig(256, 33, (8, 3, 11)),
    270: DefaultConfig(128, 27, (10, 3, 3, 3)),
    272: DefaultConfig(128, 17, (16, 17)),
    273: DefaultConfig(64, 13, (13, 3, 7)),
    275: DefaultConfig(64, 55, (11, 5, 5)),
    280: DefaultConfig(64, 56, (8, 7, 5)),
    286: DefaultConfig(64, 26, (13, 11, 2)),
    288: DefaultConfig(128, 24, (6, 6, 4, 2)),
    289: DefaultConfig(128, 17, (17, 17)),
    294: DefaultConfig(128, 42, (6, 7, 7)),
    297: DefaultConfig(256, 33, (9, 3, 11)),
    300: DefaultConfig(64, 30, (10, 10, 3)),
    306: DefaultConfig(256, 34, (17, 2, 9)),
    308: DefaultConfig(64, 44, (11, 7, 4)),
    312: DefaultConfig(64, 26, (13, 4, 3, 2)),
    315: DefaultConfig(64, 63, (7, 3, 3, 5)),
    320: DefaultConfig(64, 16, (10, 4, 4, 2)),
    324: DefaultConfig(64, 54, (3, 6, 6, 3)),
    325: DefaultConfig(64, 13, (13, 5, 5)),
    330: DefaultConfig(128, 33, (11, 10, 3)),
    336: DefaultConfig(128, 56, (8, 7, 6)),
    338: DefaultConfig(64, 26, (13, 2, 13)),
    340: DefaultConfig(128, 34, (17, 2, 10)),
    343: DefaultConfig(256, 49, (7, 7, 7)),
    350: DefaultConfig(64, 50, (5, 7, 10)),
    351: DefaultConfig(128, 39, (13, 3, 9)),
    352: DefaultConfig(64, 32, (11, 2, 16)),
    357: DefaultConfig(256, 17, (17, 3, 7)),
    360: DefaultConfig(256, 60, (10, 6, 6)),
    363: DefaultConfig(128, 33, (11, 3, 11)),
    364: DefaultConfig(64, 52, (13, 7, 4)),
    374: DefaultConfig(256, 34, (17, 2, 11)),
    375: DefaultConfig(128, 25, (5, 5, 5, 3)),
    378: DefaultConfig(128, 126, (6, 3, 3, 7)),
    384: DefaultConfig(128, 32, (6, 4, 4, 4)),
    385: DefaultConfig(64, 55, (11, 7, 5)),
    390: DefaultConfig(128, 39, (13, 3, 10)),
    392: DefaultConfig(64, 56, (8, 7, 7)),
    396: DefaultConfig(64, 44, (11, 9, 4)),
    400: DefaultConfig(128, 40, (4, 10, 10)),
    405: DefaultConfig(128, 27, (5, 3, 3, 3, 3)),
    408: DefaultConfig(64, 17, (17, 3, 8)),
    416: DefaultConfig(64, 32, (13, 2, 16)),
    420: DefaultConfig(64, 60, (10, 7, 6)),
    425: DefaultConfig(64, 17, (17, 5, 5)),
    429: DefaultConfig(128, 39, (13, 3, 11)),
    432: DefaultConfig(64, 27, (3, 16, 3, 3)),
    440: DefaultConfig(64, 55, (11, 8, 5)),
    441: DefaultConfig(64, 63, (9, 7, 7)),
    442: DefaultConfig(256, 34, (17, 2, 13)),
    448: DefaultConfig(128, 64, (8, 7, 8)),
    450: DefaultConfig(128, 30, (10, 5, 3, 3)),
    455: DefaultConfig(256, 65, (13, 5, 7)),
    459: DefaultConfig(256, 51, (17, 3, 9)),
    462: DefaultConfig(256, 77, (11, 6, 7)),
    468: DefaultConfig(64, 52, (13, 9, 4)),
    476: DefaultConfig(128, 34, (17, 2, 7, 2)),
    480: DefaultConfig(64, 16, (10, 8, 6)),
    484: DefaultConfig(64, 44, (4, 11, 11)),
    486: DefaultConfig(256, 162, (6, 3, 3, 3, 3)),
    490: DefaultConfig(256, 70, (10, 7, 7)),
    495: DefaultConfig(64, 55, (11, 9, 5)),
    500: DefaultConfig(128, 100, (10, 5, 10)),
    504: DefaultConfig(64, 63, (7, 9, 4, 2)),
    507: DefaultConfig(128, 39, (13, 3, 13)),
    510: DefaultConfig(256, 34, (17, 2, 3, 5)),
    512: DefaultConfig(64, 64, (8, 8, 8)),
    520: DefaultConfig(64, 52, (13, 10, 4)),
    525: DefaultConfig(128, 105, (7, 3, 5, 5)),
    528: DefaultConfig(64, 48, (4, 4, 3, 11)),
    539: DefaultConfig(256, 77, (11, 7, 7)),
    540: DefaultConfig(256, 54, (3, 10, 6, 3)),
    544: DefaultConfig(128, 34, (17, 2, 16)),
    546: DefaultConfig(128, 39, (13, 3, 7, 2)),
    550: DefaultConfig(64, 55, (11, 10, 5)),
    560: DefaultConfig(64, 56, (8, 7, 5, 2)),
    561: DefaultConfig(256, 51, (17, 3, 11)),
    567: DefaultConfig(64, 63, (7, 9, 3, 3)),
    572: DefaultConfig(64, 52, (13, 11, 4)),
    576: DefaultConfig(128, 96, (16, 6, 6)),
    578: DefaultConfig(256, 34, (17, 17, 2)),
    585: DefaultConfig(256, 65, (13, 5, 9)),
    588: DefaultConfig(256, 84, (7, 3, 4, 7)),
    594: DefaultConfig(128, 99, (11, 3, 6, 3)),
    595: DefaultConfig(64, 17, (7, 17, 5)),
    600: DefaultConfig(64, 60, (10, 6, 10)),
    605: DefaultConfig(64, 55, (11, 5, 11)),
    612: DefaultConfig(64, 51, (17, 3, 6, 2)),
    616: DefaultConfig(128, 88, (11, 7, 8)),
    624: DefaultConfig(64, 52, (13, 4, 6, 2)),
    625: DefaultConfig(128, 125, (5, 5, 5, 5)),
    630: DefaultConfig(64, 63, (3, 3, 5, 7, 2)),
    637: DefaultConfig(128, 91, (13, 7, 7)),
    640: DefaultConfig(128, 64, (8, 10, 8)),
    648: DefaultConfig(256, 216, (8, 3, 3, 3, 3)),
    650: DefaultConfig(256, 65, (10, 5, 13)),
    660: DefaultConfig(128, 110, (11, 6, 10)),
    663: DefaultConfig(64, 51, (17, 13, 3)),
    672: DefaultConfig(64, 56, (2, 2, 2, 2, 2, 3, 7)),
    675: DefaultConfig(256, 225, (5, 5, 3, 3, 3)),
    676: DefaultConfig(64, 52, (13, 13, 4)),
    680: DefaultConfig(256, 68, (17, 4, 10)),
    686: DefaultConfig(64, 49, (7, 7, 7, 2)),
    693: DefaultConfig(128, 99, (11, 7, 9)),
    700: DefaultConfig(128, 100, (10, 7, 10)),
    702: DefaultConfig(128, 117, (13, 3, 6, 3)),
    704: DefaultConfig(256, 88, (2, 2, 2, 2, 11, 2, 2)),
    714: DefaultConfig(64, 51, (3, 17, 7, 2)),
    715: DefaultConfig(256, 65, (13, 5, 11)),
    720: DefaultConfig(256, 120, (10, 3, 8, 3)),
    726: DefaultConfig(256, 66, (11, 6, 11)),
    728: DefaultConfig(128, 104, (13, 7, 8)),
    729: DefaultConfig(256, 243, (3, 3, 3, 3, 3, 3)),
    735: DefaultConfig(256, 147, (7, 3, 5, 7)),
    748: DefaultConfig(256, 68, (17, 4, 11)),
    750: DefaultConfig(256, 250, (10, 5, 3, 5)),
    756: DefaultConfig(64, 63, (2, 2, 3, 3, 3, 7)),
    765: DefaultConfig(256, 51, (17, 3, 5, 3)),
    768: DefaultConfig(64, 48, (16, 3, 16)),
    770: DefaultConfig(256, 110, (11, 10, 7)),
    780: DefaultConfig(256, 78, (2, 3, 13, 5, 2)),
    784: DefaultConfig(64, 56, (2, 2, 2, 2, 7, 7)),
    792: DefaultConfig(256, 88, (2, 2, 2, 3, 3, 11)),
    800: DefaultConfig(256, 160, (16, 5, 10)),
    810: DefaultConfig(128, 81, (3, 10, 3, 3, 3)),
    816: DefaultConfig(64, 51, (17, 2, 3, 2, 2, 2)),
    819: DefaultConfig(128, 117, (9, 7, 13)),
    825: DefaultConfig(64, 55, (11, 5, 5, 3)),
    832: DefaultConfig(128, 104, (13, 2, 2, 2, 2, 2, 2)),
    833: DefaultConfig(128, 119, (17, 7, 7)),
    840: DefaultConfig(64, 56, (2, 2, 2, 3, 5, 7)),
    845: DefaultConfig(256, 65, (13, 5, 13)),
    847: DefaultConfig(256, 77, (11, 7, 11)),
    850: DefaultConfig(128, 85, (10, 5, 17)),
    858: DefaultConfig(256, 78, (13, 11, 6)),
    864: DefaultConfig(64, 54, (3, 6, 16, 3)),
    867: DefaultConfig(64, 51, (17, 17, 3)),
    875: DefaultConfig(256, 175, (7, 5, 5, 5)),
    880: DefaultConfig(256, 88, (2, 2, 2, 2, 11, 5)),
    882: DefaultConfig(64, 63, (9, 7, 7, 2)),
    884: DefaultConfig(256, 68, (13, 4, 17)),
    891: DefaultConfig(256, 99, (9, 11, 3, 3)),
    896: DefaultConfig(128, 112, (2, 2, 2, 2, 2, 2, 2, 7)),
    900: DefaultConfig(256, 90, (10, 10, 3, 3)),
    910: DefaultConfig(256, 91, (13, 2, 7, 5)),
    918: DefaultConfig(128, 102, (17, 9, 2, 3)),
    924: DefaultConfig(64, 44, (2, 2, 3, 7, 11)),
    935: DefaultConfig(256, 85, (17, 11, 5)),
    936: DefaultConfig(256, 78, (2, 2, 13, 2, 3, 3)),
    945: DefaultConfig(64, 63, (3, 3, 3, 5, 7)),
    952: DefaultConfig(256, 68, (17, 4, 2, 7)),
    960: DefaultConfig(256, 160, (16, 10, 6)),
    968: DefaultConfig(256, 88, (2, 2, 2, 11, 11)),
    972: DefaultConfig(256, 162, (3, 6, 3, 6, 3)),
    975: DefaultConfig(128, 39, (13, 5, 3, 5)),
    980: DefaultConfig(256, 196, (7, 5, 7, 4)),
    990: DefaultConfig(128, 110, (2, 3, 3, 5, 11)),
    1000: DefaultConfig(128, 100, (10, 10, 10)),
    1001: DefaultConfig(256, 91, (13, 7, 11)),
    1008: DefaultConfig(64, 56, (2, 2, 2, 2, 3, 3, 7)),
    1014: DefaultConfig(256, 78, (13, 6, 13)),
    1020: DefaultConfig(256, 68, (2, 17, 2, 3, 5)),
    1024: DefaultConfig(128, 128, (8, 8, 4, 4)),
    1040: DefaultConfig(256, 208, (13, 16, 5)),
    1050: DefaultConfig(256, 210, (2, 3, 5, 5, 7)),
    1053: DefaultConfig(128, 117, (3, 3, 13, 3, 3)),
    1056: DefaultConfig(256, 176, (2, 2, 2, 2, 11, 6)),
    1071: DefaultConfig(128, 119, (17, 7, 9)),
    1078: DefaultConfig(256, 77, (2, 11, 7, 7)),
    1080: DefaultConfig(256, 108, (6, 10, 6, 3)),
    1088: DefaultConfig(256, 68, (17, 4, 4, 2, 2)),
    1089: DefaultConfig(128, 121, (3, 11, 3, 11)),
    1092: DefaultConfig(64, 52, (2, 2, 13, 7, 3)),
    1100: DefaultConfig(128, 110, (2, 2, 11, 5, 5)),
    1105: DefaultConfig(256, 85, (17, 13, 5)),
    1120: DefaultConfig(256, 224, (2, 2, 2, 2, 2, 5, 7)),
    1122: DefaultConfig(256, 102, (17, 11, 6)),
    1125: DefaultConfig(256, 225, (5, 5, 3, 3, 5)),
    1134: DefaultConfig(128, 126, (2, 3, 3, 3, 3, 7)),
    1144: DefaultConfig(128, 104, (13, 11, 8)),
    1152: DefaultConfig(256, 144, (4, 3, 8, 3, 4)),
    1155: DefaultConfig(64, 55, (11, 5, 7, 3)),
    1156: DefaultConfig(256, 68, (17, 2, 17, 2)),
    1170: DefaultConfig(256, 117, (2, 13, 3, 5, 3)),
    1176: DefaultConfig(64, 56, (2, 2, 2, 3, 7, 7)),
    1183: DefaultConfig(256, 91, (7, 13, 13)),
    1188: DefaultConfig(256, 66, (6, 11, 2, 3, 3)),
    1190: DefaultConfig(256, 85, (17, 2, 5, 7)),
    1200: DefaultConfig(256, 75, (5, 5, 16, 3)),
    1210: DefaultConfig(128, 110, (2, 5, 11, 11)),
    1215: DefaultConfig(256, 243, (5, 3, 3, 3, 3, 3)),
    1224: DefaultConfig(256, 102, (17, 3, 4, 6)),
    1225: DefaultConfig(256, 175, (5, 5, 7, 7)),
    1232: DefaultConfig(256, 176, (2, 2, 2, 2, 11, 7)),
    1248: DefaultConfig(64, 52, (2, 2, 13, 2, 3, 2, 2)),
    1250: DefaultConfig(256, 250, (5, 10, 5, 5)),
    1260: DefaultConfig(64, 63, (2, 2, 3, 3, 5, 7)),
    1274: DefaultConfig(256, 182, (2, 13, 7, 7)),
    1275: DefaultConfig(256, 85, (17, 3, 5, 5)),
    1280: DefaultConfig(128, 80, (16, 5, 16)),
    1287: DefaultConfig(128, 117, (3, 13, 3, 11)),
    1296: DefaultConfig(128, 108, (6, 6, 6, 6)),
    1300: DefaultConfig(256, 130, (10, 10, 13)),
    1309: DefaultConfig(128, 119, (17, 7, 11)),
    1320: DefaultConfig(256, 165, (11, 2, 3, 5, 4)),
    1323: DefaultConfig(256, 189, (3, 3, 3, 7, 7)),
    1326: DefaultConfig(256, 102, (17, 6, 13)),
    1331: DefaultConfig(256, 121, (11, 11, 11)),
    1344: DefaultConfig(256, 224, (2, 2, 2, 2, 2, 2, 3, 7)),
    1350: DefaultConfig(256, 135, (5, 10, 3, 3, 3)),
    1352: DefaultConfig(64, 52, (2, 13, 13, 4)),
    1360: DefaultConfig(256, 85, (17, 5, 16)),
    1365: DefaultConfig(256, 91, (13, 7, 5, 3)),
    1372: DefaultConfig(256, 98, (2, 2, 7, 7, 7)),
    1375: DefaultConfig(64, 55, (11, 5, 5, 5)),
    1377: DefaultConfig(64, 51, (17, 3, 9, 3)),
    1386: DefaultConfig(256, 231, (2, 7, 3, 11, 3)),
    1400: DefaultConfig(64, 56, (2, 2, 2, 5, 7, 5)),
    1404: DefaultConfig(128, 117, (2, 2, 3, 13, 3, 3)),
    1408: DefaultConfig(256, 176, (2, 2, 2, 2, 2, 2, 11, 2)),
    1428: DefaultConfig(128, 119, (17, 2, 7, 6)),
    1430: DefaultConfig(256, 143, (13, 11, 10)),
    1440: DefaultConfig(128, 90, (10, 16, 3, 3)),
    1445: DefaultConfig(128, 85, (17, 5, 17)),
    1452: DefaultConfig(256, 132, (11, 3, 11, 4)),
    1456: DefaultConfig(256, 182, (13, 4, 7, 2, 2)),
    1458: DefaultConfig(256, 243, (6, 3, 3, 3, 3, 3)),
    1470: DefaultConfig(256, 210, (2, 3, 5, 7, 7)),
    1485: DefaultConfig(256, 165, (3, 5, 11, 3, 3)),
    1496: DefaultConfig(256, 187, (17, 8, 11)),
    1500: DefaultConfig(256, 150, (5, 10, 10, 3)),
    1512: DefaultConfig(64, 63, (2, 2, 2, 3, 3, 3, 7)),
    1521: DefaultConfig(128, 117, (13, 3, 3, 13)),
    1530: DefaultConfig(128, 102, (17, 3, 6, 5)),
    1536: DefaultConfig(256, 256, (16, 16, 6)),
    1540: DefaultConfig(256, 154, (11, 2, 7, 5, 2)),
    1547: DefaultConfig(128, 119, (17, 7, 13)),
    1560: DefaultConfig(256, 156, (13, 2, 2, 10, 3)),
    1568: DefaultConfig(256, 224, (2, 2, 2, 2, 2, 7, 7)),
    1573: DefaultConfig(256, 143, (13, 11, 11)),
    1575: DefaultConfig(64, 63, (3, 3, 5, 7, 5)),
    1584: DefaultConfig(256, 176, (4, 2, 2, 11, 3, 3)),
    1600: DefaultConfig(256, 100, (10, 16, 10)),
    1617: DefaultConfig(256, 231, (3, 7, 7, 11)),
    1620: DefaultConfig(256, 162, (10, 3, 3, 6, 3)),
    1625: DefaultConfig(256, 65, (13, 5, 5, 5)),
    1632: DefaultConfig(128, 102, (17, 2, 2, 3, 8)),
    1638: DefaultConfig(256, 182, (13, 2, 3, 7, 3)),
    1650: DefaultConfig(128, 110, (11, 2, 3, 5, 5)),
    1664: DefaultConfig(256, 208, (13, 2, 2, 4, 2, 2, 2)),
    1666: DefaultConfig(128, 119, (17, 2, 7, 7)),
    1680: DefaultConfig(128, 112, (2, 2, 2, 2, 3, 7, 5)),
    1683: DefaultConfig(64, 51, (17, 3, 11, 3)),
    1690: DefaultConfig(256, 169, (13, 10, 13)),
    1694: DefaultConfig(256, 154, (11, 2, 11, 7)),
    1700: DefaultConfig(256, 170, (17, 10, 10)),
    1701: DefaultConfig(64, 63, (3, 3, 3, 3, 3, 7)),
    1715: DefaultConfig(256, 245, (5, 7, 7, 7)),
    1716: DefaultConfig(256, 156, (13, 2, 6, 11)),
    1728: DefaultConfig(128, 108, (3, 6, 6, 16)),
    1734: DefaultConfig(128, 102, (17, 17, 6)),
    1750: DefaultConfig(256, 175, (2, 5, 5, 7, 5)),
    1755: DefaultConfig(128, 117, (13, 3, 3, 3, 5)),
    1760: DefaultConfig(256, 176, (2, 2, 2, 2, 2, 11, 5)),
    1764: DefaultConfig(128, 126, (2, 2, 3, 3, 7, 7)),
    1768: DefaultConfig(256, 136, (17, 13, 8)),
    1782: DefaultConfig(128, 99, (11, 3, 3, 3, 3, 2)),
    1785: DefaultConfig(128, 119, (17, 3, 5, 7)),
    1792: DefaultConfig(256, 224, (4, 4, 4, 4, 7)),
    1800: DefaultConfig(256, 180, (10, 6, 10, 3)),
    1815: DefaultConfig(256, 165, (11, 3, 5, 11)),
    1820: DefaultConfig(256, 182, (10, 13, 7, 2)),
    1836: DefaultConfig(256, 153, (17, 3, 3, 2, 6)),
    1848: DefaultConfig(256, 231, (3, 11, 7, 4, 2)),
    1859: DefaultConfig(256, 169, (13, 11, 13)),
    1870: DefaultConfig(256, 187, (17, 10, 11)),
    1872: DefaultConfig(256, 156, (13, 3, 4, 6, 2)),
    1875: DefaultConfig(256, 125, (5, 5, 5, 5, 3)),
    1890: DefaultConfig(128, 126, (2, 3, 3, 3, 7, 5)),
    1904: DefaultConfig(128, 119, (17, 2, 2, 7, 4)),
    1911: DefaultConfig(128, 91, (13, 7, 7, 3)),
    1920: DefaultConfig(256, 120, (10, 6, 16, 2)),
    1925: DefaultConfig(64, 55, (7, 11, 5, 5)),
    1936: DefaultConfig(256, 176, (2, 2, 4, 11, 11)),
    1944: DefaultConfig(256, 243, (3, 3, 3, 3, 8, 3)),
    1950: DefaultConfig(256, 195, (13, 5, 10, 3)),
    1960: DefaultConfig(64, 56, (4, 7, 2, 7, 5)),
    1980: DefaultConfig(256, 198, (11, 2, 3, 3, 5, 2)),
    1989: DefaultConfig(256, 153, (17, 13, 9)),
    2000: DefaultConfig(128, 125, (5, 5, 5, 16)),
    2002: DefaultConfig(256, 182, (2, 13, 7, 11)),
    2016: DefaultConfig(256, 112, (2, 2, 2, 2, 2, 3, 3, 7)),
    2023: DefaultConfig(128, 119, (17, 7, 17)),
    2025: DefaultConfig(256, 135, (3, 3, 5, 5, 3, 3)),
    2028: DefaultConfig(256, 156, (13, 4, 3, 13)),
    2040: DefaultConfig(256, 170, (17, 4, 3, 10)),
    2048: DefaultConfig(256, 256, (16, 16, 8)),
    2160: DefaultConfig(256, 60, (10, 6, 6, 6)),
    2187: DefaultConfig(256, 243, (3, 3, 3, 3, 3, 3, 3)),
    2197: DefaultConfig(256, 169, (13, 13, 13)),
    2250: DefaultConfig(256, 90, (10, 3, 5, 3, 5)),
    2304: DefaultConfig(256, 192, (6, 6, 4, 4, 4)),
    2400: DefaultConfig(256, 240, (4, 10, 10, 6)),
    2401: DefaultConfig(256, 49, (7, 7, 7, 7)),
    2430: DefaultConfig(256, 81, (10, 3, 3, 3, 3, 3)),
    2500: DefaultConfig(256, 250, (10, 5, 10, 5)),
    2560: DefaultConfig(128, 128, (4, 4, 4, 10, 4)),
    2592: DefaultConfig(256, 216, (6, 6, 6, 6, 2)),
    2700: DefaultConfig(128, 90, (3, 10, 10, 3, 3)),
    2880: DefaultConfig(256, 96, (10, 6, 6, 2, 2, 2)),
    2916: DefaultConfig(256, 243, (6, 6, 3, 3, 3, 3)),
    3000: DefaultConfig(128, 100, (10, 3, 10, 10)),
    3072: DefaultConfig(256, 256, (6, 4, 4, 4, 4, 2)),
    3125: DefaultConfig(128, 125, (5, 5, 5, 5, 5)),
    3200: DefaultConfig(256, 160, (10, 10, 4, 4, 2)),
    3240: DefaultConfig(128, 108, (3, 3, 10, 6, 6)),
    3375: DefaultConfig(256, 225, (5, 5, 5, 3, 3, 3)),
    3456: DefaultConfig(256, 144, (6, 6, 6, 4, 4)),
    3600: DefaultConfig(256, 120, (10, 10, 6, 6)),
    3645: DefaultConfig(256, 243, (5, 3, 3, 3, 3, 3, 3)),
    3750: DefaultConfig(256, 125, (3, 5, 5, 10, 5)),
    3840: DefaultConfig(256, 128, (10, 6, 2, 2, 2, 2, 2, 2)),
    3888: DefaultConfig(512, 324, (16, 3, 3, 3, 3, 3)),
    4000: DefaultConfig(256, 200, (10, 10, 10, 4)),
    4050: DefaultConfig(256, 135, (10, 5, 3, 3, 3, 3)),
    4096: DefaultConfig(256, 256, (16, 16, 16)),
    # Beyond 4096: single/half precision only (`precision=['sp','hp']` in
    # the real config -- excluded from double precision, moot here since
    # this whole project is FP32-only). Included since this baseline IS
    # FP32-only and these rows are real, verified table entries -- not
    # part of the REQUIRED test domain (2..4096) but real nonetheless.
    4704: DefaultConfig(256, 224, (8, 4, 7, 7, 3)),
    5488: DefaultConfig(256, 196, (7, 4, 7, 4, 7)),
    6144: DefaultConfig(512, 512, (16, 4, 8, 3, 4)),
    6561: DefaultConfig(256, 243, (3, 3, 3, 3, 3, 3, 3, 3)),
    8192: DefaultConfig(512, 512, (16, 4, 4, 4, 8)),
}

assert all(set(c.factors) <= SUPPORTED_RADICES for c in SBRR_TABLE.values()), (
    "every rocFFT default-table radix must be one this M2NDP butterfly "
    "generator implements"
)

# ---------------------------------------------------------------------------
# node_factory.cpp:42-123 -- map1DLengthSingle, the literal length ->
# divLength1 table CS_L1D_CC consults for a length beyond SBRR_TABLE's own
# coverage. `divLength1` is the SBCC ("column") sub-kernel's own length;
# the companion SBRC ("row") sub-kernel's length is `length // divLength1`
# (enforced upstream by NodeFactory::Large1DLengthsValid). Transcribed
# directly from the deep-dive's own full quote of the real source array.
# ---------------------------------------------------------------------------
MAP_1D_LENGTH_SINGLE: dict[int, int] = {
    # pow2 lengths
    8192: 64, 16384: 64, 32768: 128, 65536: 256, 131072: 256, 262144: 512,
    # non-pow2, (4096, 8192)
    4704: 96, 4913: 289, 5488: 112, 6144: 96, 6561: 81,
    # non-pow2, (8192, 16384)
    9216: 72, 10000: 100, 10240: 160, 10752: 96, 11200: 224,
    12288: 192, 15625: 125,
    # non-pow2, (16384, 32768)
    16807: 343, 17576: 104, 18816: 168, 19200: 192, 19683: 243,
    20480: 160, 21504: 168, 21952: 343, 23232: 192, 24576: 192,
    26000: 208, 28672: 256, 32256: 168,
    # non-pow2, (32768, 65536)
    34969: 289, 36864: 192, 38880: 160, 40000: 200, 40960: 160,
    43008: 168, 46080: 240, 48000: 240, 49152: 256, 51200: 512,
    53248: 208, 57344: 512,
    # non-pow2, (65536, 131072)
    68600: 343, 71344: 208, 73984: 289, 76832: 224, 79860: 60,
    81920: 160, 83521: 289, 87808: 343, 95832: 72, 98304: 512,
    102400: 512, 106496: 208, 110592: 216, 114688: 224,
}

# node_factory.cpp:209-213 -- the one-entry CS_L1D_TRTRT exception table
# (3^18 prefers a 3^7 kernel even though 3^8 is itself compiled in).
MAP_1D_LENGTH_TRTRT: dict[int, int] = {387420489: 177147}

# `reverse_factors` exception set (node_factory.cpp:221-225).
_REVERSE_FACTORS = frozenset({32256, 43008})

# CS_L1D_CC block threshold (node_factory.cpp:641) and the >4096 occupancy-
# heuristic threshold (node_factory.cpp:617) -- both real, both exact.
CS_L1D_CC_POW2_THRESHOLD = 262144
SINGLE_KERNEL_OCCUPANCY_THRESHOLD = 4096


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def get_largest_pow2_length() -> int:
    """`function_pool::get_largest_pow2_length` (function_pool.h:391-399):
    the largest power-of-2 key in the compiled single-kernel table."""
    pow2_lengths = [n for n in SBRR_TABLE if is_pow2(n)]
    return max(pow2_lengths) if pow2_lengths else 0


def _search_pool(length: int, is_supported_factor) -> int:
    """`search_pool` (node_factory.cpp:227-244): scan `SBRR_TABLE`'s own
    keys, sorted descending, starting near `sqrt(length)`, for the first
    (i.e. largest, since we never need to walk past the sqrt-vicinity
    starting point for the lengths this baseline's own domain covers)
    candidate satisfying `is_supported_factor`. Ported as a direct
    descending linear scan from the sqrt-anchored start, matching the real
    source's own `std::find_if` over a descending-sorted vector."""
    supported = sorted(SBRR_TABLE.keys(), reverse=True)
    if not supported:
        return 0
    import bisect

    # supported is descending; find the first element <= sqrt(length),
    # mirroring the real source's `lower_bound` + "step back one if we
    # overshot" adjustment against a `std::greater<>`-ordered vector.
    v = int(length ** 0.5)
    ascending = list(reversed(supported))
    idx = bisect.bisect_right(ascending, v)
    start = len(ascending) - idx  # position of the first elem <= v, in `supported`
    if start > 0 and supported[start - 1] < v:
        start -= 1
    for candidate in supported[max(0, start):]:
        if is_supported_factor(candidate):
            return candidate
    return 0


def get_explicitly_supported_factor(length: int) -> int:
    """`get_explicitly_supported_factor` (node_factory.cpp:246-256):
    largest compiled-single-kernel factor of `length` whose COMPLEMENT
    (`length // factor`) also has its own compiled single kernel."""

    def supported(factor: int) -> bool:
        return length % factor == 0 and (length // factor) in SBRR_TABLE

    factor = _search_pool(length, supported)
    if factor > 0 and length in _REVERSE_FACTORS:
        return length // factor
    return factor


def get_largest_supported_factor(length: int) -> int:
    """`get_largest_supported_factor` (node_factory.cpp:258-264): largest
    compiled-single-kernel factor of `length`, complement not required to
    itself have a kernel."""
    return _search_pool(length, lambda factor: length % factor == 0)


@dataclass(frozen=True)
class SchemeDecision:
    scheme: str  # "CS_KERNEL_STOCKHAM" | "CS_L1D_CC" | "CS_L1D_TRTRT" | "CS_BLUESTEIN"
    div_length1: int | None = None  # set for CS_L1D_CC / CS_L1D_TRTRT
    single_kernel: DefaultConfig | None = None  # set for CS_KERNEL_STOCKHAM


def decide_scheme(length: int, *, batch: int = 1) -> SchemeDecision:
    """`NodeFactory::Decide1DScheme` (node_factory.cpp:628-819), ported in
    full for the 1D C2C single-precision case this project supports (no
    multi-dimensional `totalBatch` beyond the plain `batch` parameter, no
    `parent->scheme == CS_BLUESTEIN` special case since this baseline
    never builds a Bluestein parent node at all). `SupportedLength`'s own
    real definition (a separate function not fully extracted from source
    in this baseline's own research pass) is approximated by "every real
    fallback below also fails" -- functionally identical, since that is
    also exactly the condition under which real rocFFT itself would fall
    through to Bluestein.
    """
    single = SBRR_TABLE.get(length)
    if single is not None:
        if length > SINGLE_KERNEL_OCCUPANCY_THRESHOLD:
            transforms_per_block = single.workgroup_size // single.threads_per_transform
            total_batch = batch
            if total_batch // max(transforms_per_block, 1) >= ROCFFT_DEFAULT_MULTIPROCESSOR_COUNT:
                return SchemeDecision(scheme="CS_KERNEL_STOCKHAM", single_kernel=single)
            # else: fall through to the multi-kernel (CC/TRTRT) path below,
            # exactly as the real source's own "otherwise, fall through to
            # multi-kernel plan" comment describes.
        else:
            return SchemeDecision(scheme="CS_KERNEL_STOCKHAM", single_kernel=single)

    if is_pow2(length):
        if length <= CS_L1D_CC_POW2_THRESHOLD:
            div_length1 = MAP_1D_LENGTH_SINGLE.get(length)
            if div_length1 is not None:
                return SchemeDecision(scheme="CS_L1D_CC", div_length1=div_length1)
            return SchemeDecision(scheme="CS_BLUESTEIN")
        largest = get_largest_pow2_length()
        if largest <= 1:
            return SchemeDecision(scheme="CS_BLUESTEIN")
        if length > largest * largest:
            div_length1 = length // largest
        else:
            in_x = 0
            remaining = length
            while remaining != 1:
                remaining >>= 1
                in_x += 1
            in_x //= 2
            div_length1 = 1 << in_x
        return SchemeDecision(scheme="CS_L1D_TRTRT", div_length1=div_length1)

    # Non-power-of-2.
    div_length1 = MAP_1D_LENGTH_SINGLE.get(length)
    if div_length1 is not None:
        return SchemeDecision(scheme="CS_L1D_CC", div_length1=div_length1)

    trtrt = MAP_1D_LENGTH_TRTRT.get(length)
    if trtrt is not None:
        return SchemeDecision(scheme="CS_L1D_TRTRT", div_length1=trtrt)

    div_length1 = get_explicitly_supported_factor(length)
    if div_length1 == 0:
        div_length0 = get_largest_supported_factor(length)
        div_length1 = 0 if div_length0 <= 1 else length // div_length0
    if div_length1 == 0:
        return SchemeDecision(scheme="CS_BLUESTEIN")
    return SchemeDecision(scheme="CS_L1D_TRTRT", div_length1=div_length1)


def _describe_solution_tree(node: "upstream_sol_map.ResolvedSchemeNode") -> dict:
    """A JSON-safe nested dict of a resolved solution-map tree, for
    diagnostics only (`GPUKernelConfig.extra` values must stay simple/
    inspectable, never a live dataclass graph)."""
    desc: dict = {
        "token": node.token, "option": node.option,
        "sol_node_type": node.sol_node_type, "using_scheme": node.using_scheme,
    }
    if node.kernel_key is not None:
        kc = node.kernel_key.kernel_config
        desc["kernel_key"] = {
            "lengths": node.kernel_key.lengths, "precision": node.kernel_key.precision,
            "scheme": node.kernel_key.scheme, "factors": kc.factors,
            "workgroup_size": kc.wgs, "threads_per_transform": kc.tpt, "transforms_per_block": kc.tpb,
            "half_lds": kc.half_lds,
        }
    if node.children:
        desc["children"] = [_describe_solution_tree(c) for c in node.children]
    return desc


def _plan_from_solution_match(
    node: "upstream_sol_map.ResolvedSchemeNode", length: int, *, batch: int, inverse: bool,
    target: TargetProfile, kernel_name: str,
) -> BaselineResult:
    """Build a `BaselineResult` from a real, non-dummy `ApplySolution`
    match. Mappable only when the match resolves to exactly the same
    shape `decide_scheme`'s own `CS_KERNEL_STOCKHAM` case builds (a single
    `SOL_LEAF_NODE` pointing at one tunable `FMKey`/`KernelConfig`) --
    anything else (a `SOL_INTERNAL_NODE` tree, e.g. a solution-map-selected
    `CS_L1D_CC`/`CS_L1D_TRTRT`, or a leaf pointing at a builtin/non-
    Stockham kernel) is preserved in full and refused, exactly mirroring
    how `decide_scheme`'s own CC/TRTRT outcomes are handled below -- never
    force-mapped just because the solution map (rather than Decide1DScheme)
    is what produced it.
    """
    if (
        node.sol_node_type == "SOL_LEAF_NODE"
        and node.using_scheme == "CS_KERNEL_STOCKHAM"
        and node.kernel_key is not None
    ):
        kc = node.kernel_key.kernel_config
        transforms_per_block = kc.wgs // kc.tpt[0]
        gpu_config = GPUKernelConfig(
            source="rocfft-default", length=length, radices=kc.factors,
            extra={
                "scheme": "CS_KERNEL_STOCKHAM",
                "workgroup_size": kc.wgs,
                "threads_per_transform": kc.tpt[0],
                "transforms_per_block": transforms_per_block,
                "mechanism": "solution-map override (ApplySolution)",
                "solution_token": node.token,
                "solution_option": node.option,
            },
        )
        inverse_scale = (1.0 / length) if inverse else None
        mapping = map_cooperative_kernel(
            length=length, radices=kc.factors, workers_per_fft=kc.tpt[0],
            fft_slots_wanted=transforms_per_block, total_ffts=batch, inverse=inverse,
            inverse_scale=inverse_scale, kernel_name=kernel_name, target=target,
            gpu_config=gpu_config,
        )
        if mapping.status is not BaselineStatus.OK:
            return mapping
        recursive_plan = wrap_leaf_as_recursive_plan(
            length=length, total_ffts=batch, inverse=inverse, built_plan=mapping.plan,
        )
        return BaselineResult(status=BaselineStatus.OK, gpu_config=mapping.gpu_config, plan=recursive_plan)

    gpu_config = GPUKernelConfig(
        source="rocfft-default", length=length, radices=(),
        extra={
            "scheme": node.using_scheme,
            "mechanism": "solution-map override (ApplySolution)",
            "solution_token": node.token,
            "solution_option": node.option,
            "solution_tree": _describe_solution_tree(node),
        },
    )
    return unsupported(
        BaselineStatus.UNSUPPORTED_CURRENT_CODEGEN, gpu_config,
        f"length={length}: the real gfx908 solution map overrides Decide1DScheme with "
        f"{node.using_scheme} (token={node.token!r}, option={node.option}) -- this shape "
        f"is not a single tunable Stockham leaf, so it needs the same fused/multi-node "
        f"mechanism (SBCC/SBRC/transpose chains) this repository's own AddressMapping/"
        f"codegen has no equivalent for (see module docstring's own SCOPE LIMIT). The "
        f"full resolved solution tree is preserved in gpu_config.extra['solution_tree'].",
    )


def plan(
    length: int,
    *,
    batch: int = 1,
    inverse: bool = False,
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
    kernel_name: str = "FFTRocfftDefault",
) -> BaselineResult:
    """Top-level rocFFT-PRODUCTION-DEFAULT baseline entry point. Real
    production rocFFT probes the shipped solution map (`ApplySolution`)
    BEFORE `Decide1DScheme` even runs (see `rocfft_upstream_solution_map`'s
    own module docstring) -- a non-dummy match there can override what
    `Decide1DScheme` would otherwise pick, so this function checks it
    first and only falls through to the table lookup + `Decide1DScheme`
    fallback chain below when no match exists. No search, no benchmarking
    either way. Only ever builds an M2NDP plan for the `CS_KERNEL_
    STOCKHAM` (single fused kernel) outcome -- see module docstring's own
    SCOPE LIMIT for why `CS_L1D_CC`/`CS_L1D_TRTRT` (from either source) are
    decided faithfully but reported `UNSUPPORTED_CURRENT_CODEGEN` rather
    than force-mapped onto this repository's unrelated PRE/MIDDLE/POST
    six-step mechanism.
    """
    sol_match = upstream_sol_map.apply_solution(
        length, placement="op", inverse=inverse, batch=batch,
        in_stride=(1,), out_stride=(1,), i_dist=length, o_dist=length,
    )
    if sol_match is not None:
        return _plan_from_solution_match(
            sol_match, length, batch=batch, inverse=inverse, target=target, kernel_name=kernel_name,
        )

    decision = decide_scheme(length, batch=batch)

    if decision.scheme == "CS_BLUESTEIN":
        gpu_config = GPUKernelConfig(source="rocfft-default", length=length, radices=())
        return unsupported(
            BaselineStatus.UNSUPPORTED_GPU_ALGORITHM, gpu_config,
            f"length={length}: Decide1DScheme's own real fallback chain (compiled "
            f"single kernel -> CS_L1D_CC via map1DLengthSingle -> CS_L1D_TRTRT via "
            f"get_explicitly_supported_factor/get_largest_supported_factor) found no "
            f"decomposition -- real rocFFT falls back to CS_BLUESTEIN here too",
        )

    if decision.scheme == "CS_KERNEL_STOCKHAM":
        config = decision.single_kernel
        assert config is not None
        transforms_per_block = config.workgroup_size // config.threads_per_transform
        gpu_config = GPUKernelConfig(
            source="rocfft-default", length=length, radices=config.factors,
            extra={
                "scheme": "CS_KERNEL_STOCKHAM",
                "workgroup_size": config.workgroup_size,
                "threads_per_transform": config.threads_per_transform,
                "transforms_per_block": transforms_per_block,
                "mechanism": "compiled-in function_pool default (config_sbrr.py)",
            },
        )
        inverse_scale = (1.0 / length) if inverse else None
        mapping = map_cooperative_kernel(
            length=length, radices=config.factors, workers_per_fft=config.threads_per_transform,
            fft_slots_wanted=transforms_per_block, total_ffts=batch, inverse=inverse,
            inverse_scale=inverse_scale, kernel_name=kernel_name, target=target,
            gpu_config=gpu_config,
        )
        if mapping.status is not BaselineStatus.OK:
            return mapping
        recursive_plan = wrap_leaf_as_recursive_plan(
            length=length, total_ffts=batch, inverse=inverse, built_plan=mapping.plan,
        )
        return BaselineResult(status=BaselineStatus.OK, gpu_config=mapping.gpu_config, plan=recursive_plan)

    # CS_L1D_CC / CS_L1D_TRTRT: decided faithfully, but not built -- see
    # module docstring's own SCOPE LIMIT.
    assert decision.div_length1 is not None
    div1 = decision.div_length1
    div0 = length // div1
    gpu_config = GPUKernelConfig(
        source="rocfft-default", length=length, radices=(),
        extra={
            "scheme": decision.scheme,
            "div_length1": div1,
            "div_length0": div0,
            "sbcc_length": div1 if decision.scheme == "CS_L1D_CC" else None,
            "sbrc_length": div0 if decision.scheme == "CS_L1D_CC" else None,
        },
    )
    scheme_explanation = (
        "SBCC/SBRC are fused block-tiled transpose+FFT kernels"
        if decision.scheme == "CS_L1D_CC"
        else "CS_L1D_TRTRT recursively builds a transpose-row-transpose-row-transpose "
        "chain whose row kernels are themselves further Decide1DScheme calls"
    )
    return unsupported(
        BaselineStatus.UNSUPPORTED_CURRENT_CODEGEN, gpu_config,
        f"length={length}: real rocFFT chooses {decision.scheme} (divLength1={div1}, "
        f"the other factor={div0}). {scheme_explanation} -- this repository's own "
        f"AddressMapping/codegen has no equivalent mechanism (see module docstring's "
        f"own SCOPE LIMIT: forcing this onto the unrelated PRE/MIDDLE/POST six-step "
        f"shape clfft.py/vkfft.py use for their own, differently-structured upstream "
        f"algorithms would silently change what rocFFT itself actually does here).",
    )
