"""Record planner completion gaps without changing production code.

Run from benchmarks/fft: python -m verification.audit_fft_planner_completion
The findings are observations, not a passing correctness certificate. The
optional --refresh-vkfft-oracle compiles pinned upstream host functions; it
requires the sibling VkFFT checkout and a host C++ compiler, but no GPU.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from gpu_vs_m2ndp_benchmark import run_one, static_diagnostics
from planning.core.target_profile import DEFAULT_TARGET_PROFILE as TARGET
from planning.diagnostics.fft_unit_utilization import compute_unit_utilization
from planning.execution.fft_plan_lanes import (
    apply_all_scalar_lanes_to_plan, apply_compute_lanes_to_plan,
)
from planning.gpu_baseline import rocfft_default, vkfft
from planning.gpu_baseline.common import reconstruct_gpu_logical_plan
from planning.search.fft_cost_model import estimate_cost, estimate_metrics
from planning.search.fft_plan_search import generate_candidates
from planning.strategies.fft_plan_recursive import flatten_recursive_node
from verification.verify_fft_recursive import run_recursive_plan


DEFAULT_OUTPUT = Path(__file__).parent / "results" / "planner_completion_2026-10-07"


def refresh_oracle(output: Path, compiler: str) -> None:
    from verification import audit_fft_planner_porting as upstream_audit

    @contextmanager
    def ordinary_workspace_directory(*args, **kwargs):
        # TemporaryDirectory's mode-700 ACL is unreadable in this Windows
        # sandbox. Retain the original C++ source and binary as audit artifacts.
        work = output / "oracle_work"
        work.mkdir(parents=True, exist_ok=True)
        yield str(work.resolve())

    with patch.object(upstream_audit.tempfile, "TemporaryDirectory", ordinary_workspace_directory):
        with (output / "vkfft_oracle.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(log):
                upstream_audit.audit(compiler, output / "vkfft_current_oracle.json")


def vkfft_findings(oracle_path: Path) -> dict:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    current_hash = hashlib.sha256(Path(vkfft.__file__).read_bytes()).hexdigest()
    if current_hash != oracle["port_sha256"]:
        raise ValueError("VkFFT changed since the oracle capture; use --refresh-vkfft-oracle")
    fp32 = [row for row in oracle["cases"] if row["precision"] == "fp32"]
    # Match the existing audit's default benchmark domain (27 of its 32 sizes).
    default_lengths = {row["length"] for row in fp32
                       if row["length"] not in {15, 30, 60, 210, 630}}
    sweep = [row for row in fp32 if row["batch"] == 1 and row["length"] in default_lengths]
    normalized = {
        name: [row["length"] for row in sweep if row[upstream] != row[port]]
        for name, upstream, port in (
            ("different_radices", "upstream_radices", "port_radices"),
            ("different_registers", "upstream_final_min_registers", "port_min_registers"),
            ("different_workers", "upstream_workers", "port_workers"),
        )
    }
    # JSON normalizes tuple/list types. The older audit compared them before
    # serialization, spuriously counting equal radix sequences as different.
    examples = {(row["length"], row["batch"]): row for row in fp32}
    return {
        "upstream_commit": oracle["upstream_commit"],
        "scheduler_sha256": oracle["scheduler_sha256"],
        "port_sha256": current_hash,
        "total_oracle_cases": len(oracle["cases"]),
        "fp32_cases": len(fp32),
        "default_sweep_size": len(sweep),
        "normalized_default_sweep_summary": normalized,
        "older_summary_radix_type_false_positives": sorted(
            set(oracle["default_sweep_summary"]["different_radices"])
            - set(normalized["different_radices"])),
        "register_examples": [examples[n, 1] for n in (12, 15, 20, 60, 120)],
        "axis_swap_examples": [examples[key] for key in ((16, 16), (64, 32), (128, 32))],
        "radix_examples": [examples[n, 1] for n in (12, 30, 216, 120)],
        "scope": oracle["scope"],
    }


def native_width_findings() -> list[dict]:
    records = []
    for n in (64, 128, 960):
        original = generate_candidates(n)[0].plan
        resolved = apply_compute_lanes_to_plan(original, compute_lanes=4, narrow_middle_stages=True)
        options = dict(compute_lanes=4, narrow_middle_stages=True, reference_check=False)
        sources = [generate_recursive_fft_kernels(p, **options) for p in (original, resolved)]
        metrics = [estimate_metrics(p, TARGET) for p in (original, resolved)]
        records.append(dict(
            n=n, byte_identical_source=sources[0] == sources[1],
            source_sha256=hashlib.sha256(sources[0].encode()).hexdigest(),
            physical_worker_stage_batches_from_model=[m.physical_total_worker_stage_batches for m in metrics],
            estimated_costs=[estimate_cost(m) for m in metrics],
            unset_widths=[[s.compute_lanes for s in k.stages] for k in flatten_recursive_node(original.root)
                          if hasattr(k, "stages")],
            resolved_widths=[[s.compute_lanes for s in k.stages] for k in flatten_recursive_node(resolved.root)
                             if hasattr(k, "stages")],
        ))
    return records


def hybrid_findings(candidates) -> dict:
    plan = next(c.plan for c in candidates if getattr(c.plan.root, "tail_hybrid", None))
    scalar = apply_all_scalar_lanes_to_plan(plan)
    options = dict(compute_lanes=4, reference_check=False)
    return dict(
        n=plan.n, batch=plan.batch,
        rewritten_root_widths=[s.compute_lanes for s in scalar.root.kernel.stages],
        actual_task_widths=[[s.compute_lanes for s in k.stages] for k in flatten_recursive_node(scalar.root)],
        byte_identical_source=generate_recursive_fft_kernels(plan, **options)
                              == generate_recursive_fft_kernels(scalar, **options),
        actual_task_count=len(flatten_recursive_node(plan.root)),
        diagnostic_task_count=len(compute_unit_utilization(plan, TARGET)),
        diagnostic_tasks=[asdict(x) for x in compute_unit_utilization(plan, TARGET)],
    )


def target_findings() -> dict:
    target = replace(TARGET, spad_capacity_bytes=512, max_concurrent_scratchpad_bytes=512)
    record = dict(n=64, target=asdict(target), default_scratchpad_byte_budget=4096)
    try:
        candidates = generate_candidates(64, target=target)
        record["default_search"] = dict(status="ok", count=len(candidates))
    except ValueError as exc:
        record["default_search"] = dict(status="error", diagnostic=str(exc))
    feasible = generate_candidates(64, target=target, scratchpad_byte_budget=512)
    plan = feasible[0].plan
    source = generate_recursive_fft_kernels(plan, target=target, compute_lanes=4, reference_check=False)
    rng = np.random.default_rng(21)
    x = rng.normal(size=64) + 1j*rng.normal(size=64)
    error = float(np.max(np.abs(run_recursive_plan(plan, x, compute_lanes=4) - np.fft.fft(x))))
    record["explicit_budget_search"] = dict(status="ok", count=len(feasible), budget=512,
        source_generated=bool(source), root_type=type(plan.root).__name__, numeric_max_error=error)
    return record


def benchmark_findings(candidates) -> dict:
    batch_checks = []
    for batch in (1, 2, 33):
        for inverse in (False, True):
            result = run_one("m2ndp-native", 64, TARGET, batch=batch, inverse=inverse,
                skip_toolchain=True, mojo_root=None, m2ndp_root=None,
                build_timeout=1, run_timeout=1, seed=1)
            batch_checks.append(dict(n=64, batch=batch, inverse=inverse,
                correct=result.correct, numeric_max_error=result.numeric_max_error,
                diagnostics=result.diagnostics))
    selected = [
        next(c for c in candidates if c.choices.workers_per_fft == 2),
        next(c for c in candidates if c.choices.tail_strategy == "all_persistent"),
        next(c for c in candidates if getattr(c.plan.root, "tail_hybrid", None)),
    ]
    traffic = []
    for c in selected:
        tasks = flatten_recursive_node(c.plan.root)
        assert all(hasattr(k, "host") for k in tasks)
        expected = sum(k.host.total_elems for k in tasks) * 8 * 2
        reported = static_diagnostics(c.plan)
        actual = reported["dram_read_bytes"] + reported["dram_write_bytes"]
        traffic.append(dict(n=c.plan.n, batch=c.plan.batch,
            choices=asdict(c.choices), expected_read_plus_write_bytes=expected,
            search_cost_model_bytes=estimate_metrics(c.plan, TARGET).estimated_dram_bytes,
            benchmark_reported_bytes=actual, overcount_factor=actual/expected,
            tasks=[dict(length=k.length, logical_elements=k.host.total_elems,
                        physical_uthreads=k.total_uthreads) for k in tasks]))
    return dict(batch_correctness=batch_checks, traffic=traffic)


def rocfft_metadata_findings() -> list[dict]:
    # Expected flags follow the pinned generic SBRR configs and generator:
    # explicit False at N=64; ordinary multi-radix Stockham defaults True.
    records = []
    for n, expected in ((64, False), (128, True), (4096, True)):
        result = rocfft_default.plan(n)
        kernel = reconstruct_gpu_logical_plan(result)[0]
        records.append(dict(n=n, expected_half_lds=expected,
            actual_half_lds=dict(kernel.gpu_metadata).get("half_lds"),
            actual_memory_optimization=kernel.gpu_memory_optimization,
            factors=list(rocfft_default.SBRR_TABLE[n].factors)))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-vkfft-oracle", action="store_true")
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    oracle_path = args.out_dir / "vkfft_current_oracle.json"
    if args.refresh_vkfft_oracle:
        refresh_oracle(args.out_dir, args.compiler)
    if not oracle_path.is_file():
        parser.error("No captured oracle; pass --refresh-vkfft-oracle")
    candidates = generate_candidates(64, batch=33, max_candidates=500)
    findings = dict(
        scope="Current working tree; production planner/codegen files are not modified by this audit.",
        vkfft=vkfft_findings(oracle_path),
        native_compute_width_cost=native_width_findings(),
        hybrid_lane_and_diagnostic=hybrid_findings(candidates),
        target_budget_search=target_findings(),
        benchmark=benchmark_findings(candidates),
        rocfft_default_metadata=rocfft_metadata_findings(),
    )
    output = args.out_dir / "completion_findings.json"
    output.write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")
    print("Recorded planner completion observations:", output)
    print("VkFFT normalized differences:", findings["vkfft"]["normalized_default_sweep_summary"])
    print("Native identical-code/different-cost cases:", len(findings["native_compute_width_cost"]))
    print("Hybrid scalar rewrite changes source:", not findings["hybrid_lane_and_diagnostic"]["byte_identical_source"])
    print("Benchmark batch checks:", [(row["batch"], row["inverse"], row["correct"])
                                     for row in findings["benchmark"]["batch_correctness"]])
    print("Benchmark traffic overcounts:", [row["overcount_factor"] for row in findings["benchmark"]["traffic"]])


if __name__ == "__main__":
    main()
