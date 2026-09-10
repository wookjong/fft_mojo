from __future__ import annotations

"""The REAL, shipped rocFFT solution-map layer -- `ApplySolution` (library/
src/plan.cpp), `solution_map` (library/src/solution_map.cpp, library/src/
include/solution_map.h), and the compiled-in gfx908 data file itself
(library/solution_map/gfx908_rocfft_solution_map.dat), fetched and read
directly at the SAME pinned commit `bee97df517907c771de17189cb867d3c
401285ae` (ROCm/rocm-libraries, projects/rocfft/) as gpu_baseline/rocfft.py
and gpu_baseline/rocfft_default.py's own PROVENANCE -- confirmed still live
at that exact SHA for every file this module cites, so no separate/floating
commit is introduced for this layer.

============================================================================
NOT THE SAME THING AS planning/gpu_baseline/solution_map.py
============================================================================
That module is an M2NDP-SIDE measured-result CACHE this project's own
baseline-planning code writes to and reads from, deliberately modeled after
rocFFT's own solution_map's *role* (a flat key -> stored-result store) but
containing exclusively M2NDP measurements. THIS module is the opposite
direction entirely: a read-only, immutable transcription of the REAL
upstream GPU library's own shipped tuning results, used to reproduce what
real rocFFT itself would plan for gfx908 -- never written to, never mixed
with M2NDP measurements. Keep them conceptually and mechanically separate.

============================================================================
WHAT THIS MODULE PORTS
============================================================================
Real production rocFFT planning is NOT merely `NodeFactory::Decide1DScheme`
(the compiled-table + fallback-formula chain `rocfft_default.py` already
ports). The real sequence (`BuildSingleDevicePlan`, plan.cpp lines
1682-1780) is:

    execPlan.rootPlan = CreateExplicitNode(rootPlanData, nullptr)
        -- this ALREADY calls Decide1DScheme internally (parent==nullptr,
           determined_scheme defaults to CS_NONE) to pick the root's own
           scheme (CS_KERNEL_STOCKHAM / CS_L1D_CC / CS_L1D_TRTRT /
           CS_BLUESTEIN) -- confirmed directly from node_factory.cpp's own
           comment: "root-node calls the decide function anyway, before we
           try looking up solutions."
        v
    execPlan.rootScheme = ApplySolution(execPlan)
        -- GenerateProbKeys builds a token from the root's CURRENT state
           (length/precision/placement/complex-or-real/direction/batch/
           strides/dist/offset) -- crucially, `ComputeSchemeIsAProblem`
           (compute_scheme.cpp) returns true for EVERY scheme this
           project's own domain can produce as a root (CS_KERNEL_STOCKHAM,
           CS_L1D_CC, CS_L1D_TRTRT, CS_BLUESTEIN are ALL in its own
           `ProblemScheme` set), so the token NEVER gets a scheme prefix --
           meaning the solution-map lookup is scheme-INDEPENDENT: whatever
           `sol_node.using_scheme` the map stores for this exact problem
           shape can OVERRIDE what Decide1DScheme just picked, if a
           non-dummy entry exists.
        v
    if a non-dummy match is found (RecursivelyApplySol succeeds):
        execPlan.rootPlan REBUILT using the solution's own scheme
        (CreateExplicitNode(..., determined_scheme=matched_scheme) skips
        DecideNodeScheme entirely for the root)
    else:
        the original Decide1DScheme-derived rootPlan is kept as-is
        (this is `rocfft_default.py`'s own existing fallback path)

This module ports `GetNodeToken`/`GenerateProbKeys` (the exact string
token format, byte-for-byte) and `ApplySolution`/`RecursivelyApplySol`
(the exact 4-key search order and per-node tree resolution), plus a
faithful parse of the real, immutable gfx908 solution-map data file
(`data/gfx908_rocfft_solution_map.dat`, copied verbatim from the pinned
commit -- valid JSON, `solution_map::VERSION == 3`'s own on-disk format;
no "any"-arch file ships for rocFFT at all in this repository, confirmed
by directly listing `library/solution_map/` at the pinned commit, so this
module's own `(arch="any", ...)` search keys are implemented for fidelity
but never match anything against this one shipped file).

============================================================================
A DECISIVE, VERIFIED FINDING FOR THIS BASELINE'S OWN FIXED DOMAIN
============================================================================
This project's baselines are all FP32-only and consistently model
OUT-OF-PLACE transforms (M2NDP plans always use separate input/output
buffers -- there is no "in-place" mode anywhere in this repository's own
codegen). Exhaustively scanning the ENTIRE shipped gfx908 file (81 entries)
for any token containing `_op_` alongside `single`/`sp`+`complex` finds
ZERO matches -- every single-precision complex entry in this file (root or
kernel-level) is `_ip_` (in-place). This means: under this baseline's own
consistent, documented out-of-place assumption, `apply_solution` is
PROVEN (not merely assumed) to find no match for ANY length -- every call
from `rocfft_default.plan()` falls through to the existing `decide_scheme`
path, unchanged. This is verified directly (see
`verify_gpu_baseline_source_fidelity.py`'s own exhaustive scan), not
inferred from the absence of a counter-example.

The lookup/resolution machinery itself is still fully implemented and
tested against the file's own real (in-place) non-dummy entries -- the
ONE root-level, option-0 entry in the entire file for single precision
that is NOT a dummy is `16777216_sp_ip_complex` (`CS_L1D_TRTRT`, a real
5-node Transpose/4096-leaf/Transpose/4096-leaf/Transpose tree, the two
4096-length leaves using two DIFFERENT tuned `kernel_len4096_single_sbrr`
configs) -- proving this module's parser/resolver reproduces a real,
non-trivial upstream solution tree exactly, even though this specific
entry can never be reached through `plan()`'s own out-of-place calling
convention.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

_DATA_PATH = Path(__file__).resolve().parent / "data" / "gfx908_rocfft_solution_map.dat"

# solution_map.cpp's own literal constants.
KERNEL_TOKEN_BUILTIN_KERNEL = "kernel_token_builtin_kernel"
LEAFNODE_TOKEN_BUILTIN_KERNEL = "leafnode_token_builtin_kernel"

# compute_scheme.cpp's own ProblemScheme() set, restricted to the schemes
# this project's own 1D C2C domain can ever produce as a root (CS_NONE is
# not a real scheme; included only so GetNodeToken's own logic -- "is this
# scheme a problem-scheme, i.e. no prefix" -- is total over every scheme
# this module ever inspects).
_PROBLEM_SCHEMES = frozenset({"CS_KERNEL_STOCKHAM", "CS_L1D_CC", "CS_L1D_TRTRT", "CS_BLUESTEIN"})

# compute_scheme.cpp's own PrintKernelSchemeAbbr -- only the entries this
# project's own domain could ever need a prefix for (every scheme in
# _PROBLEM_SCHEMES needs none; kept for completeness/fidelity in case a
# future extension inspects a non-problem scheme).
_KERNEL_SCHEME_ABBR = {
    "CS_KERNEL_STOCKHAM": "sbrr",
    "CS_KERNEL_STOCKHAM_BLOCK_CC": "sbcc",
    "CS_KERNEL_STOCKHAM_BLOCK_CR": "sbcr",
    "CS_KERNEL_STOCKHAM_BLOCK_RC": "sbrc",
}


def compute_scheme_is_a_problem(scheme: str) -> bool:
    """`ComputeSchemeIsAProblem` (compute_scheme.cpp lines 113-145)."""
    return scheme in _PROBLEM_SCHEMES


# ---------------------------------------------------------------------------
# GetNodeToken / GenerateProbKeys -- plan.cpp lines 6122-6217, ported for
# this project's own fixed 1D C2C domain (no real-transform branch: this
# repository has no R2C/C2R path at all -- see gpu_baseline_v1_freeze.md's
# own "Exact domain covered").
# ---------------------------------------------------------------------------


def get_node_token(
    length: int, *, precision: str, placement: str, inverse: bool,
    batch: int, in_stride: tuple[int, ...], out_stride: tuple[int, ...],
    i_dist: int, o_dist: int, i_offset: int = 0, o_offset: int = 0,
    scheme: str = "CS_KERNEL_STOCKHAM",
) -> tuple[str, str]:
    """`GetNodeToken` (plan.cpp lines 6122-6197), ported exactly for a 1D
    complex-to-complex root problem (`probNode.dimension == 1`, never a
    real-transform, never itself a builtin-kernel leaf -- this function is
    only ever called on the ROOT node here, which is never a leaf).
    `scheme`: the root's OWN already-decided scheme (see module docstring
    -- `Decide1DScheme` runs before this token is built); every scheme this
    project's own domain can produce is in `_PROBLEM_SCHEMES`, so this
    parameter never actually changes the resulting token, but is threaded
    through for fidelity with the real function's own signature/logic.
    `precision`: "single" | "double" (this project only ever calls with
    "single" -- see module docstring's FP32-only note). `placement`:
    "ip" | "op". Returns `(min_token, full_token)`.
    """
    token = "" if compute_scheme_is_a_problem(scheme) else _KERNEL_SCHEME_ABBR[scheme] + "_"
    token += f"{length}_"

    precision_str = {"single": "sp_", "double": "dp_", "half": "half_"}[precision]
    token += precision_str
    token += "ip_" if placement == "ip" else "op_"

    # is_real_trans is always False for this project's own C2C-only domain.
    token += "complex"
    min_token = token
    token += "_fwd" if inverse is False else "_bwd"

    token += f"_batch_{batch}"
    token += "_istride"
    for s in in_stride:
        token += f"_{s}"
    token += "_ostride"
    for s in out_stride:
        token += f"_{s}"
    token += f"_idist_{i_dist}"
    token += f"_odist_{o_dist}"
    token += f"_ioffset_{i_offset}"
    token += f"_ooffset_{o_offset}"

    return min_token, token


@dataclass(frozen=True)
class ProblemKey:
    arch: str
    prob_token: str


def generate_prob_keys(
    length: int, *, arch_name: str, precision: str, placement: str, inverse: bool,
    batch: int, in_stride: tuple[int, ...], out_stride: tuple[int, ...],
    i_dist: int, o_dist: int, i_offset: int = 0, o_offset: int = 0,
) -> list[ProblemKey]:
    """`GenerateProbKeys` (plan.cpp lines 6200-6217), ported exactly: try
    `(archName, full_token)`, `(archName, min_token)`, `(any, full_token)`,
    `(any, min_token)`, in that order -- the first non-dummy match wins
    (see `apply_solution`)."""
    min_token, full_token = get_node_token(
        length, precision=precision, placement=placement, inverse=inverse, batch=batch,
        in_stride=in_stride, out_stride=out_stride, i_dist=i_dist, o_dist=o_dist,
        i_offset=i_offset, o_offset=o_offset,
    )
    keys = []
    for arch in (arch_name, "any"):
        for prob_token in (full_token, min_token):
            keys.append(ProblemKey(arch, prob_token))
    return keys


# ---------------------------------------------------------------------------
# Immutable data model -- solution_map.h's SolutionNodeType/SolutionNode/
# SolutionPtr, plus function_map_key.h's FMKey/KernelConfig, restricted to
# the fields the shipped gfx908 file actually populates.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelConfig:
    use_3steps: bool
    half_lds: bool
    dir_reg: bool
    buffer_inst: bool
    tpb: int
    wgs: int
    tpt: tuple[int, ...]
    factors: tuple[int, ...]


@dataclass(frozen=True)
class FMKey:
    lengths: tuple[int, ...]
    precision: str
    scheme: str
    sbrc_trans: str
    kernel_config: KernelConfig


@dataclass(frozen=True)
class SolutionPtr:
    child_token: str
    child_option: int


@dataclass(frozen=True)
class SolutionNode:
    sol_node_type: str  # "SOL_DUMMY" | "SOL_BUILTIN_KERNEL" | "SOL_KERNEL_ONLY" | "SOL_LEAF_NODE" | "SOL_INTERNAL_NODE"
    using_scheme: str = "CS_NONE"
    kernel_key: FMKey | None = None
    solution_childnodes: tuple[SolutionPtr, ...] = field(default_factory=tuple)


def _parse_kernel_config(raw: dict) -> KernelConfig:
    return KernelConfig(
        use_3steps=raw["use_3steps"], half_lds=raw["half_lds"], dir_reg=raw["dir_reg"],
        buffer_inst=raw["buffer_inst"], tpb=raw["tpb"], wgs=raw["wgs"],
        tpt=tuple(raw["tpt"]), factors=tuple(raw.get("factors", ())),
    )


def _parse_fmkey(raw: dict) -> FMKey:
    return FMKey(
        lengths=tuple(raw["lengths"]), precision=raw["precision"], scheme=raw["scheme"],
        sbrc_trans=raw["sbrc_trans"], kernel_config=_parse_kernel_config(raw["kernelConfig"]),
    )


def _parse_solution_node(raw: dict) -> SolutionNode:
    kernel_key = _parse_fmkey(raw["kernel_key"]) if "kernel_key" in raw else None
    children = tuple(
        SolutionPtr(c["child_token"], c["child_option"]) for c in raw.get("solution_childnodes", ())
    )
    return SolutionNode(
        sol_node_type=raw["sol_node_type"], using_scheme=raw.get("using_scheme", "CS_NONE"),
        kernel_key=kernel_key, solution_childnodes=children,
    )


def _load_solution_map(path: Path) -> dict[tuple[str, str], tuple[SolutionNode, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["Version"] == 3, f"unexpected solution-map format version {raw['Version']}"
    table: dict[tuple[str, str], tuple[SolutionNode, ...]] = {}
    for entry in raw["Data"]:
        key = (entry["Problem"]["arch"], entry["Problem"]["token"])
        table[key] = tuple(_parse_solution_node(s) for s in entry["Solutions"])
    return table


_GFX908_MAP: dict[tuple[str, str], tuple[SolutionNode, ...]] = _load_solution_map(_DATA_PATH)


def has_solution_node(key: ProblemKey, option_id: int = 0) -> bool:
    """`solution_map::has_solution_node` (solution_map.cpp lines 385-396) --
    only the gfx908 map is shipped/loaded (see module docstring), so any
    `key.arch` other than `"gfx908"` never matches, matching this baseline's
    own fixed-target assumption exactly (there is no data to consult for a
    different arch, real or "any")."""
    solutions = _GFX908_MAP.get((key.arch, key.prob_token))
    return solutions is not None and len(solutions) > option_id


def get_solution_node(key: ProblemKey, option_id: int = 0) -> SolutionNode:
    if not has_solution_node(key, option_id):
        raise KeyError(f"no solution node for {key} option {option_id}")
    return _GFX908_MAP[(key.arch, key.prob_token)][option_id]


@dataclass(frozen=True)
class ResolvedSchemeNode:
    """The result of recursively resolving one matched `SolutionNode` tree
    -- mirrors `SchemeTree` (plan.cpp) closely enough for this baseline's
    own diagnostic/mapping purposes: `scheme` for an internal/leaf node,
    or the underlying `FMKey` for a kernel-only leaf's own concrete config.
    `token`/`option`: which map entry this node came from, kept for exact
    diagnostics and for tests to check specific child_option values."""

    token: str
    option: int
    sol_node_type: str
    using_scheme: str
    kernel_key: FMKey | None
    children: tuple["ResolvedSchemeNode", ...] = field(default_factory=tuple)


def recursively_apply_sol(key: ProblemKey, option_id: int = 0) -> ResolvedSchemeNode | None:
    """`RecursivelyApplySol` (plan.cpp lines 6219-6351), ported exactly for
    this project's own scope (no tuning-mode elaborated-token branch: this
    baseline never runs rocFFT's own offline tuner against real hardware).
    Returns `None` for a dummy solution (`using_scheme == CS_NONE`) or a
    missing node -- exactly mirroring the real function's own two "give up"
    conditions, both of which mean "no override here, let the caller try
    the next key or fall back."
    """
    if not has_solution_node(key, option_id):
        return None

    sol_node = get_solution_node(key, option_id)

    if sol_node.using_scheme == "CS_NONE":
        return None  # a dummy root-solution

    if sol_node.sol_node_type == "SOL_INTERNAL_NODE":
        if not sol_node.solution_childnodes:
            return None
        children: list[ResolvedSchemeNode] = []
        for child_ptr in sol_node.solution_childnodes:
            child_key = ProblemKey(key.arch, child_ptr.child_token)
            child_resolved = recursively_apply_sol(child_key, child_ptr.child_option)
            if child_resolved is None:
                return None
            children.append(child_resolved)
        return ResolvedSchemeNode(
            token=key.prob_token, option=option_id, sol_node_type=sol_node.sol_node_type,
            using_scheme=sol_node.using_scheme, kernel_key=None, children=tuple(children),
        )

    if sol_node.sol_node_type == "SOL_LEAF_NODE":
        if len(sol_node.solution_childnodes) != 1:
            return None
        kernel_ptr = sol_node.solution_childnodes[0]
        kernel_key_probkey = ProblemKey(key.arch, kernel_ptr.child_token)
        if not has_solution_node(kernel_key_probkey, kernel_ptr.child_option):
            return None
        kernel_node = get_solution_node(kernel_key_probkey, kernel_ptr.child_option)
        return ResolvedSchemeNode(
            token=key.prob_token, option=option_id, sol_node_type=sol_node.sol_node_type,
            using_scheme=sol_node.using_scheme, kernel_key=kernel_node.kernel_key,
        )

    raise ValueError(f"Tree-Decomposition in solution map is invalid: {sol_node.sol_node_type}")


def apply_solution(
    length: int, *, arch_name: str = "gfx908", precision: str = "single", placement: str,
    inverse: bool, batch: int, in_stride: tuple[int, ...], out_stride: tuple[int, ...],
    i_dist: int, o_dist: int, i_offset: int = 0, o_offset: int = 0,
) -> ResolvedSchemeNode | None:
    """`ApplySolution` (plan.cpp lines 6353-6370), ported exactly: try each
    of the 4 possible keys (in `generate_prob_keys`'s own order) at option
    0, returning the first non-null resolved tree; `None` if every key is
    either absent or a dummy solution -- the caller (`rocfft_default.plan`)
    must then fall back to `decide_scheme`, exactly as real
    `BuildSingleDevicePlan` keeps its `Decide1DScheme`-derived rootPlan
    when `ApplySolution` returns null."""
    for key in generate_prob_keys(
        length, arch_name=arch_name, precision=precision, placement=placement, inverse=inverse,
        batch=batch, in_stride=in_stride, out_stride=out_stride, i_dist=i_dist, o_dist=o_dist,
        i_offset=i_offset, o_offset=o_offset,
    ):
        resolved = recursively_apply_sol(key, 0)
        if resolved is not None:
            return resolved
    return None
