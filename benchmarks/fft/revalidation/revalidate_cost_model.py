from __future__ import annotations

"""Reproducible cost-model revalidation harness (2026-08-31, Phase A of
docs/active_ndp_units_cost_task.md's "physical-parallelism validation and
production integration" task). Replaces the old 55-candidate/14-N dataset
that generated `docs/execution_cost_model_validation.md`'s own Phase 3/6
table -- that dataset's own generating script does not survive in the
repo, so this is a NEW, deterministic, explicitly-specified candidate list
(not a reconstruction) meant to stay in the repo and be re-run whenever a
cost-model weight changes, not a one-off.

Every candidate here is an explicit, hand-specified `(N, label,
plan_kwargs)` triple -- no dependency on `generate_candidates`'s own
search order or internal ranking, so this dataset's own composition
cannot silently drift if that search changes. Real M2NDP-Detour toolchain
required (see scripts/env.sh); one row is one real build+run, tens of
seconds each.

Usage:
    python3 revalidate_cost_model.py [--out-prefix docs/cost_model_revalidation]

Writes `<prefix>.csv` and `<prefix>.json`, one row per candidate, columns
matching this module's own `CandidateResult` fields.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.search.fft_cost_model import (
    DEFAULT_COST_WEIGHTS,
    CostWeights,
    _execution_cost,
    _memory_cost,
    estimate_cost,
    estimate_metrics,
)
from planning.strategies.fft_plan_recursive import make_recursive_transpose_plan
from planning.diagnostics.fft_unit_utilization import compute_unit_utilization
from planning.diagnostics.spill_probe import _parse_ndp_cycles
from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from codegen.fft_transpose_codegen import generate_recursive_fft_kernels

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_M2NDP_ROOT = _REPO_ROOT
_HOST_STUBS = _REPO_ROOT / "sim" / "host_stubs.c"
_SRC_DIR = _REPO_ROOT / "src"


@dataclass(frozen=True)
class CandidateSpec:
    """One deterministic, hand-specified candidate -- see module
    docstring for why this replaces a search-generated list."""

    n: int
    label: str
    plan_kwargs: dict = field(default_factory=dict)
    scratchpad_byte_budget: int = 4096


@dataclass
class CandidateResult:
    n: int
    label: str
    plan_kwargs: str  # repr, for a human-readable CSV cell
    estimated_cost: float
    memory_cost: float
    execution_cost: float
    total_worker_stage_batches: int
    radix_risk_score: float
    spill_free: bool | None
    near_active_units_best: int | None
    far_active_units_best: int | None
    build_ok: bool
    run_ok: bool
    measured_cycles: int | None
    error: str | None = None


def _leaf_active_units(plan) -> tuple[int | None, int | None]:
    """(near_fft, far_child) active_units_best -- None if this plan has no
    split (single fused leaf, only one leaf estimate exists)."""
    estimates = [
        e for e in compute_unit_utilization(plan, DEFAULT_TARGET_PROFILE)
        if e.leaf_index is not None
    ]
    if len(estimates) == 1:
        return estimates[0].active_units_best, None
    if len(estimates) >= 2:
        return estimates[0].active_units_best, estimates[-1].active_units_best
    return None, None


def _build_and_run(source: str, *, timeout: int = 240) -> tuple[bool, bool, str]:
    """Returns (build_ok, run_ok, combined_log)."""
    mojo_bin = _find_mojo_bin()
    with tempfile.TemporaryDirectory() as work_str:
        work = Path(work_str)
        stage = work / "stage"
        stage.mkdir()
        for f in _SRC_DIR.glob("*.mojo"):
            (stage / f.name).write_text(f.read_text())
        gen_path = stage / "gen.mojo"
        gen_path.write_text(source)

        host_obj = work / "host_stubs.o"
        subprocess.run(
            ["cc", "-c", "-O2", str(_HOST_STUBS), "-o", str(host_obj)],
            check=True, capture_output=True,
        )

        bin_path = work / "bin"
        build = subprocess.run(
            [mojo_bin, "build", "gen.mojo", "-o", str(bin_path),
             "-Xlinker", str(host_obj), "-Xlinker", "-lm"],
            cwd=stage, capture_output=True, text=True, timeout=timeout,
        )
        if build.returncode != 0:
            return False, False, build.stdout + build.stderr

        try:
            run = subprocess.run(
                [str(bin_path)], capture_output=True, text=True, timeout=timeout,
                env=_run_env(),
            )
        except subprocess.TimeoutExpired as e:
            return True, False, (e.stdout or "") + (e.stderr or "") + "\nTIMEOUT"
        return True, run.returncode == 0, run.stdout + run.stderr


def _find_mojo_bin() -> str:
    import os

    mojo_root = os.environ.get("MOJO_ROOT", str(_REPO_ROOT / "toolchain"))
    real = Path(mojo_root) / "bin" / "mojo.real"
    plain = Path(mojo_root) / "bin" / "mojo"
    return str(real if real.exists() else plain)


def _run_env() -> dict:
    import os

    env = dict(os.environ)
    env.setdefault("M2NDP_ROOT", str(_M2NDP_ROOT))
    env.setdefault(
        "M2NDP_CONFIG",
        str(_M2NDP_ROOT / "third_party" / "m2ndp-detour" / "config" / "performance" / "M2NDP" / "m2ndp.config"),
    )
    mojo_root = env.get("MOJO_ROOT", str(_REPO_ROOT / "toolchain"))
    env["LD_LIBRARY_PATH"] = f"{mojo_root}/lib:" + env.get("LD_LIBRARY_PATH", "")
    return env


def run_candidate(spec: CandidateSpec) -> CandidateResult:
    try:
        plan = make_recursive_transpose_plan(
            spec.n, scratchpad_byte_budget=spec.scratchpad_byte_budget,
            simd_lanes=8,
            spad_capacity_bytes=DEFAULT_TARGET_PROFILE.spad_capacity_bytes,
            max_concurrent_scratchpad_bytes=DEFAULT_TARGET_PROFILE.max_concurrent_scratchpad_bytes,
            **spec.plan_kwargs,
        )
    except Exception as e:  # noqa: BLE001 -- report, don't crash the sweep
        return CandidateResult(
            n=spec.n, label=spec.label, plan_kwargs=repr(spec.plan_kwargs),
            estimated_cost=float("nan"), memory_cost=float("nan"),
            execution_cost=float("nan"), total_worker_stage_batches=-1,
            radix_risk_score=float("nan"), spill_free=None,
            near_active_units_best=None, far_active_units_best=None,
            build_ok=False, run_ok=False, measured_cycles=None,
            error=f"plan build failed: {type(e).__name__}: {e}",
        )

    metrics = estimate_metrics(plan, DEFAULT_TARGET_PROFILE)
    cost = estimate_cost(metrics, DEFAULT_COST_WEIGHTS)
    mem = _memory_cost(metrics, DEFAULT_COST_WEIGHTS)
    ex = _execution_cost(metrics, DEFAULT_COST_WEIGHTS)
    near_units, far_units = _leaf_active_units(plan)

    source = generate_recursive_fft_kernels(
        plan, target=DEFAULT_TARGET_PROFILE, reference_check=False,
        compute_lanes=4, narrow_middle_stages=True,
    )
    build_ok, run_ok, log = _build_and_run(source)
    cycles = _parse_ndp_cycles(log) if run_ok else None
    spill_free = (not _has_spill(log)) if run_ok else None

    return CandidateResult(
        n=spec.n, label=spec.label, plan_kwargs=repr(spec.plan_kwargs),
        estimated_cost=cost, memory_cost=mem, execution_cost=ex,
        total_worker_stage_batches=metrics.total_worker_stage_batches,
        radix_risk_score=metrics.radix_risk_score, spill_free=spill_free,
        near_active_units_best=near_units, far_active_units_best=far_units,
        build_ok=build_ok, run_ok=run_ok, measured_cycles=cycles,
        error=None if run_ok else "build or run failed -- see kept log" if build_ok else "build failed",
    )


def _has_spill(log: str) -> bool:
    return "spills to memory" in log.lower()


def add_ranks_and_regret(results: list[CandidateResult]) -> None:
    """In-place: for each N, rank by estimated_cost (model_rank) and by
    measured_cycles (measured_rank), plus per-candidate regret relative to
    that N's own real best. Candidates with no measured_cycles are
    excluded from ranking (kept in the dataset with rank=None)."""
    by_n: dict[int, list[CandidateResult]] = {}
    for r in results:
        by_n.setdefault(r.n, []).append(r)

    for n, group in by_n.items():
        measured = [r for r in group if r.measured_cycles is not None]
        if not measured:
            continue
        best_cycles = min(r.measured_cycles for r in measured)
        model_order = sorted(measured, key=lambda r: r.estimated_cost)
        measured_order = sorted(measured, key=lambda r: r.measured_cycles)
        for rank, r in enumerate(model_order, start=1):
            r.model_rank = rank  # type: ignore[attr-defined]
        for rank, r in enumerate(measured_order, start=1):
            r.measured_rank = rank  # type: ignore[attr-defined]
        for r in measured:
            r.regret = r.measured_cycles / best_cycles - 1.0  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-prefix", default="docs/cost_model_revalidation")
    parser.add_argument("--only-n", type=int, default=None)
    parser.add_argument(
        "--candidates-module", default="revalidate_candidates",
        help="module (importable from this file's own directory) exporting a CANDIDATES "
             "list of CandidateSpec -- e.g. revalidate_holdout_candidates for the Phase 2 "
             "unseen-N holdout set, kept separate from the calibration set this defaults to",
    )
    args = parser.parse_args()

    import importlib

    CANDIDATES = importlib.import_module(args.candidates_module).CANDIDATES

    specs = CANDIDATES if args.only_n is None else [c for c in CANDIDATES if c.n == args.only_n]
    results: list[CandidateResult] = []
    for i, spec in enumerate(specs):
        print(f"[{i+1}/{len(specs)}] N={spec.n} {spec.label} ...", flush=True)
        result = run_candidate(spec)
        results.append(result)
        print(f"    -> measured_cycles={result.measured_cycles} "
              f"estimated_cost={result.estimated_cost:.1f} "
              f"build_ok={result.build_ok} run_ok={result.run_ok} "
              f"{result.error or ''}", flush=True)

    add_ranks_and_regret(results)

    rows = [asdict(r) for r in results]
    for r, row in zip(results, rows):
        row["model_rank"] = getattr(r, "model_rank", None)
        row["measured_rank"] = getattr(r, "measured_rank", None)
        row["regret"] = getattr(r, "regret", None)

    out_csv = Path(args.out_prefix + ".csv")
    out_json = Path(args.out_prefix + ".json")
    out_json.write_text(json.dumps(rows, indent=2))

    import csv

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {out_csv} and {out_json} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
