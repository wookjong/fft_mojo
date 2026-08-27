from __future__ import annotations

"""Real build+run probe for whether a plan's own FFT kernels spill on the
actual Mojo -> llc -> M2NDP-Detour toolchain -- the one PlanMetrics signal
fft_cost_model.py cannot compute statically (see PlanMetrics.spill_free's
own docstring for why: unlike radix_risk_score and every other term there,
whether a *specific* stage spills depends on LLVM's real register
allocator, not on anything this repo's own planning/codegen layer decides
-- see this session's own N=630 isolation, where compute_lanes=4/2 spilled
identically in byte count and location but only one of them corrupted the
answer).

Deliberately its own module, never imported by fft_cost_model.py or
fft_plan_search.py themselves: those two stay pure-Python and toolchain-
free (every other PlanMetrics term is a fast, static function of the plan
alone -- see fft_cost_model.py's own module docstring), and probing here
needs the real Mojo/llc/M2NDP-Detour toolchain (scripts/env.sh) plus tens
of seconds to minutes per plan (this project's own benchmark_fft_
candidates.sh has the same cost profile) -- something no caller should pay
merely by importing the cost model. A caller who wants this signal calls
probe_spill_free explicitly and folds the result back into a PlanMetrics
via apply_spill_probe; generate_candidates/estimate_cost/rank_candidates
are unaffected unless a caller does exactly that.
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from planning.fft_cost_model import CostWeights, DEFAULT_COST_WEIGHTS, PlanMetrics, estimate_cost
from planning.fft_plan_recursive import RecursiveFFTPlan
from planning.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# `warning: <unknown>:0:0: in function fft_fp32_N630_generated::FFTRecNear0::
# stage_1() void (): M2NDP kernel spills to memory (128-byte frame); spills
# go to DRAM` -- see docs/STATUS.md's own note on this warning (a frame
# appearing at all means a spill; this target has no callee-saved
# registers, so the whole register file being free is the only reason a
# frame ever exists otherwise). `(\w+)::(\w+\(\))` pulls the kernel struct
# name and its own stage method name; the byte count isn't needed by any
# caller today but is kept for anyone printing a fuller diagnostic later.
_SPILL_RE = re.compile(
    r"in function \S*::(\w+)::(\w+\(\)) void \(\): "
    r"M2NDP kernel spills to memory \((\d+)-byte frame\)"
)


@dataclass(frozen=True)
class SpillProbeResult:
    """One real build+run's own answer for one plan/compute_lanes/
    narrow_middle_stages combination -- `spilling_kernels` is
    `(kernel_name, stage_name)` pairs, deduplicated, in first-seen order;
    `spill_free` is just `not spilling_kernels`, kept as its own field so a
    caller never has to remember that equivalence. `build_ok`/`run_ok`
    both `False` only on a toolchain-level failure (a real compile error,
    a timeout, a missing toolchain) -- distinguished from `spill_free`
    itself so a caller can tell "this plan is clean" from "this probe
    could not answer," which `fft_cost_model.estimate_cost` must never
    conflate (a failed probe is not evidence of a spill-free plan).
    """

    spill_free: bool
    spilling_kernels: tuple[tuple[str, str], ...]
    build_ok: bool
    run_ok: bool
    log: str


def _toolchain_env(*, mojo_root: str | None, m2ndp_root: str | None) -> dict[str, str]:
    """Mirrors scripts/env.sh's own MOJO_ROOT/MOJO_BIN resolution (see that
    script for the authoritative version -- duplicated here in Python
    since this probe has to run without a shell wrapper around every
    subprocess call, not because the logic itself is expected to diverge).
    `mojo_root`/`m2ndp_root`: `None` picks the same defaults env.sh does
    (`<repo>/toolchain`, `<repo>` itself) -- pass explicitly only to probe
    against a different build, e.g. a `git worktree` whose own toolchain
    dir doesn't exist (see fft-benchmark-workflow's own note on this).
    """
    root = Path(mojo_root) if mojo_root else _REPO_ROOT / "toolchain"
    m2ndp = Path(m2ndp_root) if m2ndp_root else _REPO_ROOT
    mojo_bin = root / "bin" / "mojo.real"
    if not mojo_bin.exists():
        mojo_bin = root / "bin" / "mojo"

    env = dict(os.environ)
    env["MOJO_ROOT"] = str(root)
    env["LD_LIBRARY_PATH"] = str(root / "lib") + (
        (":" + env["LD_LIBRARY_PATH"]) if env.get("LD_LIBRARY_PATH") else ""
    )
    env["MODULAR_MOJO_MAX_PACKAGE_ROOT"] = str(root)
    env["MODULAR_MOJO_MAX_IMPORT_PATH"] = str(root / "lib" / "mojo")
    env["MODULAR_HOME"] = str(root)
    env["MOJO_BIN"] = str(mojo_bin)
    env["MODULAR_MOJO_MAX_DRIVER_PATH"] = str(mojo_bin)
    env["M2NDP_ROOT"] = str(m2ndp)
    env["M2NDP_CONFIG"] = str(
        m2ndp / "third_party" / "m2ndp-detour" / "config" / "performance" / "M2NDP" / "m2ndp.config"
    )
    return env


def probe_spill_free(
    plan: RecursiveFFTPlan,
    *,
    compute_lanes: int | None = None,
    simd_lanes: int = 8,
    narrow_middle_stages: bool = True,
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
    mojo_root: str | None = None,
    m2ndp_root: str | None = None,
    build_timeout: float = 120.0,
    run_timeout: float = 200.0,
) -> SpillProbeResult:
    """Render `plan` (exactly as make_fft_kernel.py would, at this
    compute_lanes/narrow_middle_stages), build it against the real
    toolchain, run it once with `--no-reference-check`-equivalent
    behavior (reference_check=False -- the O(N^2) host DFT is skipped, but
    every kernel still launches and runs on the real M2NDP-Detour
    simulator, which is what actually prints a spill warning -- see
    `docs/STATUS.md`), and report every kernel/stage that spilled.

    `compute_lanes=None`/`narrow_middle_stages=True` are the real defaults
    -- resolved the *same way* `make_fft_kernel()` resolves them
    (`min(simd_lanes, target.lmul1_float32_lanes)`) before this function
    ever calls `generate_recursive_fft_kernels`, not passed through as a
    raw `None`. This matters: `generate_recursive_fft_kernels`/`_emit_
    stage`/`_chunk_batch` treat a raw `compute_lanes=None` as "no
    narrowing at all, use the full `simd_lanes` width" (see `_chunk_
    batch`'s own `compute_lanes if compute_lanes is not None else plan.
    simd_lanes`) -- a materially *wider*, more register-pressured
    rendering than the actually-shipped default of 4. Before this
    resolution step existed here, `probe_spill_free(plan, compute_lanes=
    None)` silently probed compute_lanes=8 while every real caller
    (make_fft_kernel.py, run_fft_test.sh, benchmark_fft_candidates.sh)
    ships compute_lanes=4 -- confirmed the hard way: N=1024's own
    baseline plan (verified spill-free via run_fft_test.sh's real
    default path) came back `spill_free=False` here before this fix,
    purely from testing an unrepresentative width nobody actually ships.
    Pass an explicit `compute_lanes` int to deliberately probe a
    non-default width (mirrors make_fft_kernel.py's own `--compute-lanes`
    override) -- only the `None` case's *meaning* changed, not its
    availability as an override mechanism.

    Correctness is *not* checked here (reference_check=False, and this
    never parses the INPUT_BEGIN/OUTPUT_BEGIN dump either) -- this answers
    one question only, "does anything spill," not "is the answer right."
    A caller that also wants correctness for this exact configuration
    should run `run_fft_test.sh` (or `reference_check=True` through
    make_fft_kernel.py directly) separately -- see fft-benchmark-workflow.

    Mirrors this project's own manual build+run steps (see the
    fft-benchmark-workflow memory / run_fft_test.sh) rather than shelling
    out to that script, so a caller only pays for one temp directory and
    one subprocess round trip per probe, not a second Python startup.
    """
    if compute_lanes is None:
        compute_lanes = min(simd_lanes, target.lmul1_float32_lanes)
    env = _toolchain_env(mojo_root=mojo_root, m2ndp_root=m2ndp_root)
    source = generate_recursive_fft_kernels(
        plan, compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
        reference_check=False,
    )

    with tempfile.TemporaryDirectory(prefix="fft_spill_probe_") as work_str:
        work = Path(work_str)
        stage = work / "stage"
        stage.mkdir()
        for f in (_REPO_ROOT / "src").glob("*.mojo"):
            (stage / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
        gen_path = stage / "gen.mojo"
        gen_path.write_text(source, encoding="utf-8")

        host_stubs_o = work / "host_stubs.o"
        cc = subprocess.run(
            ["cc", "-c", "-O2", str(_REPO_ROOT / "sim" / "host_stubs.c"), "-o", str(host_stubs_o)],
            capture_output=True, text=True, timeout=build_timeout,
        )
        if cc.returncode != 0:
            return SpillProbeResult(
                spill_free=False, spilling_kernels=(), build_ok=False, run_ok=False,
                log="host_stubs.c compile failed:\n" + cc.stdout + cc.stderr,
            )

        bin_path = work / "bin"
        try:
            build = subprocess.run(
                [env["MOJO_BIN"], "build", "gen.mojo", "-o", str(bin_path),
                 "-Xlinker", str(host_stubs_o), "-Xlinker", "-lm"],
                cwd=stage, env=env, capture_output=True, text=True, timeout=build_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return SpillProbeResult(
                spill_free=False, spilling_kernels=(), build_ok=False, run_ok=False,
                log=f"build timed out after {build_timeout}s:\n{exc.stdout or ''}{exc.stderr or ''}",
            )
        if build.returncode != 0 or not bin_path.exists():
            return SpillProbeResult(
                spill_free=False, spilling_kernels=(), build_ok=False, run_ok=False,
                log="build failed:\n" + build.stdout + build.stderr,
            )

        try:
            run = subprocess.run(
                [str(bin_path)], env=env, capture_output=True, text=True, timeout=run_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return SpillProbeResult(
                spill_free=False, spilling_kernels=(), build_ok=True, run_ok=False,
                log=f"run timed out after {run_timeout}s:\n{exc.stdout or ''}{exc.stderr or ''}",
            )

        log = run.stdout + run.stderr
        spilling: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for kernel_name, stage_name, _bytes in _SPILL_RE.findall(log):
            key = (kernel_name, stage_name)
            if key not in seen:
                seen.add(key)
                spilling.append(key)

        return SpillProbeResult(
            spill_free=not spilling, spilling_kernels=tuple(spilling),
            build_ok=True, run_ok=True, log=log,
        )


def apply_spill_probe(
    metrics: PlanMetrics, result: SpillProbeResult, *, weights: CostWeights = DEFAULT_COST_WEIGHTS
) -> PlanMetrics:
    """`metrics` with `spill_free` set from `result` and `estimated_cost`
    recomputed against it. `result.build_ok`/`run_ok` both `False` (a
    toolchain-level failure, not an answer about the plan itself -- see
    `SpillProbeResult`'s own docstring) leaves `spill_free` at `None`
    rather than reading a failed probe as either "spill-free" or
    "spills": `estimate_cost` already treats `None` as "not probed," so a
    toolchain hiccup silently falls back to ranking this candidate on
    every other term instead of misleading the search.
    """
    spill_free = result.spill_free if (result.build_ok and result.run_ok) else None
    probed = replace(metrics, spill_free=spill_free)
    return replace(probed, estimated_cost=estimate_cost(probed, weights))


class NoSpillFreeCandidateError(RuntimeError):
    """Raised by `probe_and_rerank_candidates` when it ran out of
    candidates before finding `top_k` confirmed spill-free ones. Carries
    the partial probing work done so far (`kept`/`excluded`/`unresolved`,
    same meaning as `ProbeAndRerankResult`'s own fields) so a caller can
    inspect what was found -- e.g. fall back to a smaller `top_k`, widen
    `scratchpad_byte_budget`/try a different `compute_lanes`, or just
    report the situation -- rather than losing every probe result along
    with the exception.
    """

    def __init__(self, message: str, *, kept, excluded, unresolved):
        super().__init__(message)
        self.kept = kept
        self.excluded = excluded
        self.unresolved = unresolved


@dataclass(frozen=True)
class ProbeAndRerankResult:
    """`candidates`: up to `top_k` real-build+run-CONFIRMED spill-free
    candidates, ranked by `estimated_cost` -- see `probe_and_rerank_
    candidates`'s own docstring for why this is a hard filter, not a cost
    term, despite `fft_cost_model.CostWeights.spill_penalty` still
    existing (that penalty is for a candidate nobody has probed yet, so
    `estimate_cost` still has *some* signal to rank unprobed candidates
    against each other -- once a candidate is actually probed here,
    confirmed-spilling is disqualifying, period).

    `excluded_for_spill`: candidates dropped because probing confirmed a
    real spill, in the order encountered -- kept for visibility (this
    project's own history, e.g. N=630's compute_lanes=4/ReadCsr/vlenb
    bug, is why "it happened to still pass its reference check" is not
    good enough to keep a spilling candidate around).

    `unresolved`: candidates whose probe itself failed (toolchain
    error/timeout, `SpillProbeResult.build_ok`/`run_ok` False) -- neither
    confirmed safe nor confirmed spilling, so these count toward neither
    `candidates` nor `excluded_for_spill`; a caller that wants to treat
    "couldn't tell" as acceptable can inspect this list and decide for
    itself, rather than this function silently picking a side.
    """

    candidates: list
    excluded_for_spill: tuple
    unresolved: tuple
    probed_count: int


def probe_and_rerank_candidates(
    candidates,
    *,
    compute_lanes: int | None = None,
    simd_lanes: int = 8,
    narrow_middle_stages: bool = True,
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
    top_k: int = 5,
    weights: CostWeights = DEFAULT_COST_WEIGHTS,
    mojo_root: str | None = None,
    m2ndp_root: str | None = None,
    build_timeout: float = 120.0,
    run_timeout: float = 200.0,
) -> ProbeAndRerankResult:
    """`compute_lanes`/`simd_lanes`/`target`: passed straight through to
    `probe_spill_free`, which is where `compute_lanes=None`'s actual
    resolution to `make_fft_kernel()`'s own real default lives -- see that
    function's own docstring for why this matters (a raw `None` means
    something wider and more spill-prone at the codegen layer than what
    every real caller actually ships).

    Walk `candidates` in the order given (should already be
    `rank_candidates`'s own output, cheapest-estimated first, so the
    candidates probed first are the ones that would otherwise have been
    recommended) and probe each one for real until `top_k` are CONFIRMED
    spill-free -- a confirmed spill is a hard disqualification, never
    just a cost penalty (this project's own architecture assumes spill
    basically never happens -- no callee-saved registers -- and a real
    spill has previously produced a silently wrong answer on this
    simulator, see fft_cost_model.py's own _NON_FIRST_STAGE_RISKY_RADICES
    comment and this module's own N=630 reference above; "it happened to
    pass its reference check this time" is not a basis for recommending a
    plan). Unlike the version of this function that existed before this
    discipline (spill_penalty as a soft cost term, every candidate
    returned regardless), this keeps probing PAST `top_k` into the rest
    of `candidates` whenever a spill knocks one out, so the caller still
    gets `top_k` genuinely safe options rather than fewer.

    Raises `NoSpillFreeCandidateError` if every candidate in `candidates`
    is exhausted without finding `top_k` confirmed spill-free ones --
    silently returning fewer than asked would look like "there just
    weren't more good candidates," when the real situation is "every
    remaining one actually spills," a materially different (and more
    urgent) fact for the caller. See that exception's own docstring for
    what it carries.
    """
    from planning.fft_plan_search import rank_candidates

    kept: list = []
    excluded: list = []
    unresolved: list = []
    probed_count = 0

    for candidate in candidates:
        if len(kept) >= top_k:
            break
        probed_count += 1
        result = probe_spill_free(
            candidate.plan, compute_lanes=compute_lanes, simd_lanes=simd_lanes,
            narrow_middle_stages=narrow_middle_stages, target=target,
            mojo_root=mojo_root, m2ndp_root=m2ndp_root,
            build_timeout=build_timeout, run_timeout=run_timeout,
        )
        new_metrics = apply_spill_probe(candidate.metrics, result, weights=weights)
        probed_candidate = replace(candidate, metrics=new_metrics)
        if new_metrics.spill_free is True:
            kept.append(probed_candidate)
        elif new_metrics.spill_free is False:
            excluded.append(probed_candidate)
        else:
            unresolved.append(probed_candidate)

    if len(kept) < top_k:
        raise NoSpillFreeCandidateError(
            f"probe_and_rerank_candidates: exhausted all {len(candidates)} "
            f"candidate(s), probed {probed_count}, and found only "
            f"{len(kept)} confirmed spill-free -- short of the requested "
            f"top_k={top_k}. {len(excluded)} confirmed spilling, "
            f"{len(unresolved)} unresolved (probe itself failed). No plan "
            f"is safe to recommend at this compute_lanes="
            f"{compute_lanes}/narrow_middle_stages={narrow_middle_stages} "
            f"combination under this discipline -- spilling is a hard "
            f"disqualification here, not a risk to weigh against other terms.",
            kept=kept, excluded=excluded, unresolved=unresolved,
        )

    return ProbeAndRerankResult(
        candidates=rank_candidates(kept), excluded_for_spill=tuple(excluded),
        unresolved=tuple(unresolved), probed_count=probed_count,
    )
