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
    compute_lanes: int | None,
    narrow_middle_stages: bool = False,
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


def probe_and_rerank_candidates(
    candidates,
    *,
    compute_lanes: int | None,
    narrow_middle_stages: bool = True,
    top_k: int = 5,
    weights: CostWeights = DEFAULT_COST_WEIGHTS,
    mojo_root: str | None = None,
    m2ndp_root: str | None = None,
    build_timeout: float = 120.0,
    run_timeout: float = 200.0,
):
    """Probe only the `top_k` already-ranked candidates (real build+run is
    the expensive part of this whole module -- see its own docstring --
    so this never probes a candidate `rank_candidates` already ranked
    behind the cut, matching `benchmark_fft_candidates.sh`'s own "only the
    top-K get built" discipline) and re-rank with each one's own
    `spill_free` folded in. A candidate that turns out to spill can still
    end up ranked above a spill-free one further down `candidates` --
    `spill_penalty` is a cost term, not a hard filter (see its own
    comment in fft_cost_model.py for why a spill isn't an automatic
    disqualification) -- so this returns every candidate given, re-sorted,
    never a filtered subset.

    `candidates` should already be `rank_candidates`'s own output (or
    anything sorted the same way) -- this re-sorts its result again after
    probing, so passing an unsorted list still produces a correctly
    ranked answer, just probes an arbitrary `top_k` slice of it instead of
    the `top_k` cheapest-by-estimate ones.
    """
    from planning.fft_plan_search import rank_candidates

    head = list(candidates[:top_k])
    tail = list(candidates[top_k:])
    probed_head = []
    for candidate in head:
        result = probe_spill_free(
            candidate.plan, compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
            mojo_root=mojo_root, m2ndp_root=m2ndp_root,
            build_timeout=build_timeout, run_timeout=run_timeout,
        )
        new_metrics = apply_spill_probe(candidate.metrics, result, weights=weights)
        probed_head.append(replace(candidate, metrics=new_metrics))
    return rank_candidates(probed_head + tail)
