from __future__ import annotations

"""Controlled radix-swap ablation -- isolates whether the 27-N GPU-baseline
sweep's large-N (>=768) failures trace to the GPU planner's own RADIX
DECOMPOSITION specifically, or to some other downstream planning decision
(kernel partition, worker-count policy, execution strategy). Mode C
(persistent_mode="physical") already removed the logical-worker/wave
execution-lowering cost this task's earlier phase investigated -- that is
NOT the variable here; every arm below uses Mode C.

Four arms per (N, GPU planner) pair:

  native_full                  -- m2ndp-native's own plan, unmodified
                                   (get_plan("m2ndp-native", n, ...))
  gpu_full                      -- the GPU planner's own plan, unmodified
                                   (get_plan(planner, n, ...))
  gpu_radix_native_downstream   -- the GPU planner's own radix sequence,
                                   M2NDP-native's own downstream (kernel
                                   partition/execution strategy) -- see
                                   planning.diagnostics.radix_swap_ablation.
                                   build_gpu_radix_native_downstream
  native_radix_gpu_downstream   -- M2NDP-native's own radix sequence for
                                   this length, the GPU planner's own
                                   downstream (workers_per_fft, execution
                                   strategy) -- see that module's
                                   build_native_radix_gpu_downstream

`native_full` is computed once per N (planner-independent) and repeated in
every row for that N. Every arm goes through the exact same production
codegen/probe path (`codegen.fft_transpose_codegen.
generate_recursive_fft_kernels` -> `planning.diagnostics.spill_probe.
probe_spill_free`) as every other plan in this project.

Reproduction:
    python3 radix_swap_ablation_experiment.py --out ablation.csv
    python3 radix_swap_ablation_experiment.py --out retry.csv --n 1024 2048 \\
        --build-timeout 900 --run-timeout 400   # extended-timeout retry
"""

import argparse
import csv
import time
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

from gpu_vs_m2ndp_benchmark import DEFAULT_TARGET_PROFILE, get_plan, static_diagnostics
from planning.core.target_profile import TargetProfile
from planning.diagnostics.radix_swap_ablation import (
    DownstreamRefusal,
    build_gpu_radix_native_downstream,
    build_native_radix_gpu_downstream,
    extract_gpu_radix_and_workers,
    native_radix_for_length,
)
from planning.diagnostics.spill_probe import SpillProbeResult, probe_spill_free
from planning.strategies.fft_plan_recursive import RecursiveFFTPlan
from verification.verify_fft_recursive import run_recursive_plan

GPU_PLANNERS = ("gpu-clfft", "gpu-rocfft-default", "gpu-vkfft")
CONTROL_N = (216, 512)
LARGE_N = (768, 1024, 1536, 2048, 3072, 4096)


def classify_outcome(probe: SpillProbeResult) -> str:
    """Per the task's own explicit instruction: never conflate a build
    TIMEOUT with a real compiler error, and never conflate a CSRRS
    simulator panic with an ordinary runtime failure -- each gets its own
    label, read straight off the probe's own log text (the exact strings
    `spill_probe.probe_source_spill_free` itself writes for each distinct
    failure branch -- see that function's own timeout/returncode/PANIC
    checks), never guessed from timing alone.
    """
    if not probe.build_ok:
        if "build timed out after" in probe.log:
            return "build_timeout"
        return "compile_error"
    if not probe.run_ok:
        if "run timed out after" in probe.log:
            return "run_timeout"
        if "unmapped opcode: CSRRS" in probe.log:
            return "runtime_crash_csrrs"
        if "M2NDP PANIC" in probe.log:
            return "runtime_crash_other"
        return "runtime_error_other"
    if not probe.spill_free:
        return "spilling"
    return "ok"


@dataclass
class ArmResult:
    n: int
    planner: str
    arm: str
    radix_sequence: str
    workers_per_fft: object
    num_kernels: object
    outcome: str
    build_ok: object
    run_ok: object
    spill_free: object
    spilling_kernels: str
    cycles: object
    max_error: object
    elapsed_s: float
    diagnostics_tail: str


CSV_FIELDS = [f.name for f in fields(ArmResult)]


def _correctness(plan: RecursiveFFTPlan, n: int, seed: int = 1234) -> object:
    try:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)
        got = run_recursive_plan(plan, x)
        expected = np.fft.fft(x)
        return float(np.max(np.abs(got - expected)))
    except Exception as exc:  # noqa: BLE001 -- record, never crash the sweep
        return f"error:{type(exc).__name__}:{exc}"


def _probe_and_record(
    *, n: int, planner: str, arm: str, plan: RecursiveFFTPlan,
    target: TargetProfile, build_timeout: float, run_timeout: float,
) -> ArmResult:
    diags = static_diagnostics(plan)
    workers_per_fft = None
    try:
        _, workers_per_fft = extract_gpu_radix_and_workers(plan)
    except AssertionError:
        pass  # a recursive (multi-leaf) plan, e.g. native_full at large N
    err = _correctness(plan, n)
    t0 = time.time()
    probe = probe_spill_free(
        plan, target=target, mojo_root=None, m2ndp_root=None,
        build_timeout=build_timeout, run_timeout=run_timeout, persistent_mode="physical",
    )
    elapsed = time.time() - t0
    return ArmResult(
        n=n, planner=planner, arm=arm, radix_sequence=str(diags["radix_sequence"]),
        workers_per_fft=workers_per_fft, num_kernels=diags["num_kernels"],
        outcome=classify_outcome(probe), build_ok=probe.build_ok, run_ok=probe.run_ok,
        spill_free=probe.spill_free, spilling_kernels=str(probe.spilling_kernels),
        cycles=probe.ndp_cycles, max_error=err, elapsed_s=round(elapsed, 1),
        diagnostics_tail=probe.log[-600:] if not (probe.build_ok and probe.run_ok and probe.spill_free) else "",
    )


def _refused(n: int, planner: str, arm: str, refusal: DownstreamRefusal) -> ArmResult:
    return ArmResult(
        n=n, planner=planner, arm=arm, radix_sequence="", workers_per_fft=None,
        num_kernels=None, outcome=f"plan_refused:{refusal.status.value}",
        build_ok=None, run_ok=None, spill_free=None, spilling_kernels="",
        cycles=None, max_error=None, elapsed_s=0.0, diagnostics_tail=refusal.diagnostics[:600],
    )


def run_one_n(
    n: int, target: TargetProfile, build_timeout: float, run_timeout: float,
    writer: csv.DictWriter, flush_fh,
) -> None:
    ok, native_plan, diag = get_plan("m2ndp-native", n, target, batch=1, inverse=False)
    if not ok:
        row = ArmResult(
            n=n, planner="(all)", arm="native_full", radix_sequence="", workers_per_fft=None,
            num_kernels=None, outcome="plan_unsupported", build_ok=None, run_ok=None,
            spill_free=None, spilling_kernels="", cycles=None, max_error=None,
            elapsed_s=0.0, diagnostics_tail=diag[:600],
        )
        writer.writerow(vars(row)); flush_fh.flush()
        print(f"N={n:5d} native_full  status=plan_unsupported: {diag[:150]}", flush=True)
        native_plan = None
    else:
        row = _probe_and_record(
            n=n, planner="(shared)", arm="native_full", plan=native_plan,
            target=target, build_timeout=build_timeout, run_timeout=run_timeout,
        )
        writer.writerow(vars(row)); flush_fh.flush()
        print(f"N={n:5d} native_full  outcome={row.outcome:20s} cycles={row.cycles} ({row.elapsed_s}s)", flush=True)

    native_radices = native_radix_for_length(n)

    for planner in GPU_PLANNERS:
        ok, gpu_plan, diag = get_plan(planner, n, target, batch=1, inverse=False)
        if not ok:
            row = ArmResult(
                n=n, planner=planner, arm="gpu_full", radix_sequence="", workers_per_fft=None,
                num_kernels=None, outcome="plan_unsupported", build_ok=None, run_ok=None,
                spill_free=None, spilling_kernels="", cycles=None, max_error=None,
                elapsed_s=0.0, diagnostics_tail=diag[:600],
            )
            writer.writerow(vars(row)); flush_fh.flush()
            print(f"N={n:5d} {planner:18s} gpu_full: plan_unsupported: {diag[:150]}", flush=True)
            continue

        row = _probe_and_record(
            n=n, planner=planner, arm="gpu_full", plan=gpu_plan,
            target=target, build_timeout=build_timeout, run_timeout=run_timeout,
        )
        writer.writerow(vars(row)); flush_fh.flush()
        print(f"N={n:5d} {planner:18s} gpu_full     outcome={row.outcome:20s} cycles={row.cycles} ({row.elapsed_s}s)", flush=True)

        gpu_radices, gpu_workers = extract_gpu_radix_and_workers(gpu_plan)

        c_plan = build_gpu_radix_native_downstream(n=n, radices=gpu_radices, target=target)
        row = _probe_and_record(
            n=n, planner=planner, arm="gpu_radix_native_downstream", plan=c_plan,
            target=target, build_timeout=build_timeout, run_timeout=run_timeout,
        )
        writer.writerow(vars(row)); flush_fh.flush()
        print(f"N={n:5d} {planner:18s} gpu_radix+native outcome={row.outcome:20s} cycles={row.cycles} ({row.elapsed_s}s)", flush=True)

        if gpu_workers is None:
            row = _refused(n, planner, "native_radix_gpu_downstream", DownstreamRefusal(
                status=None, diagnostics="gpu_full leaf has no workers_per_fft (not persistent) -- D not attempted",
            ))
            writer.writerow(vars(row)); flush_fh.flush()
            print(f"N={n:5d} {planner:18s} native_radix+gpu skipped (no workers_per_fft)", flush=True)
        else:
            d_plan_or_refusal = build_native_radix_gpu_downstream(
                n=n, radices=native_radices, workers_per_fft=gpu_workers, target=target,
            )
            if isinstance(d_plan_or_refusal, DownstreamRefusal):
                row = _refused(n, planner, "native_radix_gpu_downstream", d_plan_or_refusal)
                writer.writerow(vars(row)); flush_fh.flush()
                print(f"N={n:5d} {planner:18s} native_radix+gpu refused: {d_plan_or_refusal.status}", flush=True)
            else:
                row = _probe_and_record(
                    n=n, planner=planner, arm="native_radix_gpu_downstream", plan=d_plan_or_refusal,
                    target=target, build_timeout=build_timeout, run_timeout=run_timeout,
                )
                writer.writerow(vars(row)); flush_fh.flush()
                print(f"N={n:5d} {planner:18s} native_radix+gpu outcome={row.outcome:20s} cycles={row.cycles} ({row.elapsed_s}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, nargs="*", default=None, help="override N sweep (default: control + large)")
    parser.add_argument("--build-timeout", type=float, default=450.0)
    parser.add_argument("--run-timeout", type=float, default=300.0)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    n_values = tuple(args.n) if args.n else (CONTROL_N + LARGE_N)
    target = DEFAULT_TARGET_PROFILE

    mode = "a" if args.append else "w"
    with open(args.out, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if mode == "w":
            writer.writeheader()
        for n in n_values:
            run_one_n(n, target, args.build_timeout, args.run_timeout, writer, f)

    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
