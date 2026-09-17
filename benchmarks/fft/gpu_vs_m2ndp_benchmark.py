from __future__ import annotations

"""GPU-derived FFT planning policy vs. the M2NDP-native planner -- fair
comparison harness.

Compares each of `gpu-clfft`, `gpu-rocfft-default`, `gpu-rocfft-tuned`,
`gpu-vkfft` (see planning/gpu_baseline/*.py) against this project's own
M2NDP-native planner (planning.strategies.fft_plan_recursive.
make_recursive_transpose_plan), all rendered through the exact same M2NDP
codegen (codegen.fft_transpose_codegen.generate_recursive_fft_kernels via
planning.diagnostics.spill_probe.probe_spill_free) and the same real Mojo ->
LLVM(M2NDP fork) -> M2NDP-Detour toolchain. The only thing that varies
between rows for the same N is the planner.

This is NOT a new measurement mechanism: every real-toolchain call here is
`planning.diagnostics.spill_probe.probe_spill_free` (spill detection + real
`ndp_cycles`, already used throughout this repo's own regression/
revalidation scripts), and every correctness check is `verification.
verify_fft_recursive.run_recursive_plan` (the Python-side numeric oracle
this repo already uses to verify its own recursive planner, re-executing
the *actual emitted stage formulas*, not a separate approximation) against
`numpy.fft`. No new toolchain-facing code exists here beyond a thin
per-kernel-struct cycle-breakdown parser (`_parse_per_kernel_cycles`,
reusing spill_probe's own regexes) that spill_probe's own `_parse_ndp_cycles`
does not expose (it only returns the plan-wide *total*).

Usage (from benchmarks/fft/, with the real toolchain environment already
sourced -- see spill_probe.py's own `_toolchain_env` / this repo's
scripts/env.sh for what that means):

    python3 gpu_vs_m2ndp_benchmark.py --out results.csv
    python3 gpu_vs_m2ndp_benchmark.py --out results.csv --n 64 128 960 1024
    python3 gpu_vs_m2ndp_benchmark.py --out results.csv --planners m2ndp-native gpu-vkfft
    python3 gpu_vs_m2ndp_benchmark.py --out results.csv --skip-toolchain   # coverage+correctness only, no real build/run

`--mojo-root`/`--m2ndp-root` forward straight to `probe_spill_free` -- see
that function's own docstring; pass these to point at a toolchain that
isn't at this repo's own default `./toolchain`/`.` (e.g. this project's own
`ghcr.io/psal-postech/mojo-m2ndp:main` dev image, which installs Mojo at
`/opt/mojo` instead of `<repo>/toolchain`, and additionally needs `LD_
LIBRARY_PATH` to include `<repo>/third_party/m2ndp-detour/build/lib` for
`libNDPSim_lib.so` -- export that yourself before running this script,
`probe_spill_free`'s own env construction only ever *prepends* to whatever
`LD_LIBRARY_PATH` this process itself already has, never invents the
Detour build's own lib dir on its own).
"""

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from planning.core.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile
from planning.strategies.fft_plan_recursive import (
    PhysicalTransposePlan,
    RecursiveFFTPlan,
    flatten_recursive_node,
    make_recursive_transpose_plan,
)
from planning.gpu_baseline import clfft, rocfft, rocfft_default, vkfft
from planning.gpu_baseline.common import BaselineResult, BaselineStatus
from planning.diagnostics.spill_probe import (
    _NDP_CYCLE_RE,
    _TASK_REGISTERED_RE,
    SpillProbeResult,
    probe_spill_free,
)
from verification.verify_fft_recursive import run_recursive_plan

PLANNERS = ("m2ndp-native", "gpu-clfft", "gpu-rocfft-default", "gpu-rocfft-tuned", "gpu-vkfft")

# The task's own required N sweep (small/pow2/mixed-radix/multi-factorable/
# split-needing/previously-problematic/large) -- kept verbatim; domain
# support (or lack of it) is a per-(planner,N) COVERAGE result, never a
# reason to drop an N from the sweep itself.
DEFAULT_N_SWEEP = (
    12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 120, 128, 192, 216, 240, 256,
    320, 384, 480, 512, 768, 960, 1024, 1536, 2048, 3072, 4096,
)

_KERNEL_STRUCT_NAME_RE = re.compile(r"finished NDP kernel \S*::(\w+)::")


def _parse_per_kernel_cycles(log: str) -> dict[str, int]:
    """Per-top-level-kernel-struct cycle totals -- the same 'a new
    "Registered task" line starts a fresh process/clock group; within a
    group only the LAST ndp-cycle value is real; the true total is the SUM
    of each group's own last value' rule as `spill_probe._parse_ndp_cycles`
    (see that function's own 2026-08-31 fix comment for the full
    derivation), but keeping each group's own kernel-struct name (the
    `::Name::` segment of the Gantt line, e.g. `FFTRecPre0`) instead of
    only a single plan-wide sum. Reuses spill_probe's own `_TASK_
    REGISTERED_RE`/`_NDP_CYCLE_RE` directly rather than re-deriving them,
    so a future change to the real log format only needs updating once.
    """
    per_kernel: dict[str, int] = {}
    current_name: str | None = None
    current_last: int | None = None
    for line in log.splitlines():
        if _TASK_REGISTERED_RE.search(line):
            if current_name is not None and current_last is not None:
                per_kernel[current_name] = per_kernel.get(current_name, 0) + current_last
            current_name, current_last = None, None
            continue
        m = _NDP_CYCLE_RE.search(line)
        if m:
            current_last = int(m.group(1))
            name_m = _KERNEL_STRUCT_NAME_RE.search(line)
            if name_m:
                current_name = name_m.group(1)
    if current_name is not None and current_last is not None:
        per_kernel[current_name] = per_kernel.get(current_name, 0) + current_last
    return per_kernel


@dataclass
class PlanResult:
    n: int
    planner: str
    status: str  # spill_free_correct | spilling | compile_failure | runtime_failure | numerical_mismatch | unsupported
    correct: object = "not_available"  # True | False | "not_available"
    spill: object = "not_available"  # True | False | "not_available"
    measured_total_cycles: object = "not_available"
    radix_sequence: object = "not_available"
    kernel_partition: object = "not_available"
    num_kernels: object = "not_available"
    execution_strategy: object = "not_available"
    kernel_cycles: object = "not_available"
    scratchpad_uthreads: object = "not_available"
    dram_read_bytes: object = "not_available"
    dram_write_bytes: object = "not_available"
    transpose_count: object = "not_available"
    large_twiddle_count: object = "not_available"
    numeric_max_error: object = "not_available"
    diagnostics: str = ""


BYTES_PER_COMPLEX_FP32 = 8  # sizeof(Float32)*2 -- every plan in this repo is FP32-only


def static_diagnostics(plan: RecursiveFFTPlan) -> dict:
    """Everything about `plan` derivable WITHOUT the real toolchain --
    structural facts read directly off the already-fully-decided plan
    tree, never guessed. `dram_read_bytes`/`dram_write_bytes`: the
    structural minimum DRAM traffic each kernel's own launch implies
    (`total_uthreads * length` complex elements read once and written once
    -- every kernel in this repo reads its whole input region from DRAM
    into scratchpad and writes its whole output region back out, exactly
    once, regardless of AddressMapping shape) -- NOT a simulator-measured
    memory-controller counter (Detour's own per-channel ramulator
    utilization percentages in its log are a real but much coarser/noisier
    signal, mixing every concurrent channel/kernel; not attributable to one
    kernel's own traffic without additional instrumentation this repo
    doesn't have, so not used here).
    """
    stages = flatten_recursive_node(plan.root)

    kernel_partition = []
    radix_sequence = []
    execution_strategies = set()
    scratchpad_uthreads = []
    dram_read_bytes = 0
    dram_write_bytes = 0
    large_twiddle_count = 0
    transpose_count = 0

    for stage in stages:
        if isinstance(stage, PhysicalTransposePlan):
            transpose_count += 1
            kernel_partition.append(f"TRANSPOSE(rows={stage.rows},cols={stage.cols},replicas={stage.replica_count})")
            elems = stage.replica_count * stage.rows * stage.cols
            dram_read_bytes += elems * BYTES_PER_COMPLEX_FP32
            dram_write_bytes += elems * BYTES_PER_COMPLEX_FP32
            if stage.twiddle_modulus is not None:
                large_twiddle_count += 1
        else:
            kernel_partition.append(f"FFT(length={stage.length},total_uthreads={stage.total_uthreads})")
            radix_sequence.append(tuple(s.radix for s in stage.stages))
            scratchpad_uthreads.append(stage.max_uthread)
            elems = stage.total_uthreads * stage.length
            dram_read_bytes += elems * BYTES_PER_COMPLEX_FP32
            dram_write_bytes += elems * BYTES_PER_COMPLEX_FP32
            if stage.large_twiddle is not None:
                large_twiddle_count += 1
            if stage.cooperation is not None:
                execution_strategies.add("cooperative")
            elif stage.persistent is not None:
                execution_strategies.add("persistent")
            else:
                execution_strategies.add("plain")

    return dict(
        radix_sequence=tuple(radix_sequence),
        kernel_partition=" -> ".join(kernel_partition),
        num_kernels=len(stages),
        execution_strategy=",".join(sorted(execution_strategies)) if execution_strategies else "n/a",
        scratchpad_uthreads=tuple(scratchpad_uthreads),
        dram_read_bytes=dram_read_bytes,
        dram_write_bytes=dram_write_bytes,
        transpose_count=transpose_count,
        large_twiddle_count=large_twiddle_count,
    )


def get_plan(
    planner: str, n: int, target: TargetProfile, *, batch: int, inverse: bool,
) -> tuple[bool, RecursiveFFTPlan | None, str]:
    """Returns `(ok, plan_or_None, diagnostics)` -- `diagnostics` explains
    an unsupported/failed planning attempt; empty string on success.
    `planner="gpu-rocfft-tuned"` uses `rocfft.plan`'s own DEFAULT
    `benchmark_fn` (`default_benchmark`, itself `probe_spill_free`-based --
    see rocfft.py's own module docstring) for its internal winner
    selection: real M2NDP measurement, never `estimate_cost` -- this is
    the one planner whose OWN planning call already does real toolchain
    build+run rounds, once per phase-0/phase-1 candidate that maps onto
    M2NDP at all (see this script's own module docstring / the final
    report's "rocFFT-tuned" section for why this planner's own wall-clock
    cost is structurally different from the other four).
    """
    if planner == "m2ndp-native":
        try:
            plan = make_recursive_transpose_plan(
                n, scratchpad_byte_budget=4096, simd_lanes=8, inverse=inverse, batch=batch,
                spad_capacity_bytes=target.spad_capacity_bytes,
                max_concurrent_scratchpad_bytes=target.max_concurrent_scratchpad_bytes,
                cooperative_workers=None,
                interleave_chunk_uthreads=target.interleave_chunk_uthreads,
            )
            return True, plan, ""
        except Exception as exc:  # noqa: BLE001 -- record, never crash the sweep
            return False, None, f"m2ndp-native planning raised {type(exc).__name__}: {exc}"

    module = {
        "gpu-clfft": clfft, "gpu-rocfft-default": rocfft_default,
        "gpu-rocfft-tuned": rocfft, "gpu-vkfft": vkfft,
    }[planner]
    try:
        result: BaselineResult = module.plan(n, batch=batch, inverse=inverse, target=target)
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{planner} planning raised {type(exc).__name__}: {exc}"
    if result.status is not BaselineStatus.OK:
        return False, None, result.diagnostics
    assert isinstance(result.plan, RecursiveFFTPlan)
    return True, result.plan, ""


def run_one(
    planner: str, n: int, target: TargetProfile, *, batch: int, inverse: bool,
    skip_toolchain: bool, mojo_root: str | None, m2ndp_root: str | None,
    build_timeout: float, run_timeout: float, seed: int,
    persistent_mode: str = "physical",
) -> PlanResult:
    ok, plan, diag = get_plan(planner, n, target, batch=batch, inverse=inverse)
    if not ok:
        return PlanResult(n=n, planner=planner, status="unsupported", diagnostics=diag)

    result = PlanResult(n=n, planner=planner, status="unsupported")
    result.diagnostics = ""
    diags = static_diagnostics(plan)
    for k, v in diags.items():
        setattr(result, k, v)

    # Correctness: the Python-side numeric oracle, no toolchain needed --
    # verification.verify_fft_recursive.run_recursive_plan re-executes the
    # ACTUAL emitted stage formulas (same discipline as every other
    # correctness check in this repo's own verification/ suite).
    try:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)
        got = run_recursive_plan(plan, x)
        expected = np.fft.ifft(x) * n if inverse else np.fft.fft(x)
        max_err = float(np.max(np.abs(got - expected)))
        result.numeric_max_error = max_err
        result.correct = max_err < 1e-2
    except Exception as exc:  # noqa: BLE001
        result.correct = "not_available"
        result.numeric_max_error = "not_available"
        result.diagnostics += f"correctness oracle raised {type(exc).__name__}: {exc}; "

    if skip_toolchain:
        result.status = "unsupported"  # no real measurement attempted -- not a planning failure
        result.diagnostics += "real-toolchain probe skipped (--skip-toolchain)"
        return result

    probe: SpillProbeResult = probe_spill_free(
        plan, target=target, mojo_root=mojo_root, m2ndp_root=m2ndp_root,
        build_timeout=build_timeout, run_timeout=run_timeout,
        persistent_mode=persistent_mode,
    )
    if not probe.build_ok:
        result.status = "compile_failure"
        result.diagnostics += "build failed: " + probe.log[-2000:]
        return result
    if not probe.run_ok:
        result.status = "runtime_failure"
        result.diagnostics += "run failed/timed out: " + probe.log[-2000:]
        return result

    result.spill = probe.spill_free is False
    result.measured_total_cycles = probe.ndp_cycles if probe.ndp_cycles is not None else "not_available"
    result.kernel_cycles = json.dumps(_parse_per_kernel_cycles(probe.log))

    if probe.spill_free is False:
        result.status = "spilling"
    elif result.correct is False:
        result.status = "numerical_mismatch"
    elif result.correct == "not_available":
        result.status = "runtime_failure"
        result.diagnostics += "spill-free real run succeeded but the Python correctness oracle could not run; "
    else:
        result.status = "spill_free_correct"
    return result


CSV_FIELDS = [
    "n", "planner", "status", "correct", "spill", "measured_total_cycles",
    "radix_sequence", "kernel_partition", "num_kernels", "execution_strategy",
    "kernel_cycles", "scratchpad_uthreads", "dram_read_bytes", "dram_write_bytes",
    "transpose_count", "large_twiddle_count", "numeric_max_error", "diagnostics",
]


def _serialize(value: object) -> str:
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(value)
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument("--n", type=int, nargs="*", default=None, help="override the N sweep (default: the task's required 27-point sweep)")
    parser.add_argument("--planners", nargs="*", default=list(PLANNERS), choices=PLANNERS)
    parser.add_argument("--batch", type=int, default=1, help="replica/batch count (default: 1, isolating plan quality -- see report's fairness section)")
    parser.add_argument("--inverse", action="store_true")
    parser.add_argument("--skip-toolchain", action="store_true", help="coverage+correctness only, no real build/run (fast, no toolchain needed)")
    parser.add_argument("--mojo-root", default=None)
    parser.add_argument("--m2ndp-root", default=None)
    parser.add_argument("--build-timeout", type=float, default=180.0)
    parser.add_argument("--run-timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--append", action="store_true", help="append to --out instead of overwriting (for resuming a long sweep)")
    parser.add_argument(
        "--persistent-mode", default="physical", choices=("wave", "fused", "physical"),
        help="execution lowering for persistent leaf kernels: wave=Mode A (original worker-wave), "
             "fused=Mode B (fused logical-worker loop), physical=Mode C (direct physical-lane strip-mining, default)",
    )
    args = parser.parse_args()

    n_sweep = tuple(args.n) if args.n else DEFAULT_N_SWEEP
    target = DEFAULT_TARGET_PROFILE

    mode = "a" if args.append else "w"
    out_path = Path(args.out)
    write_header = not (args.append and out_path.exists())
    with out_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_FIELDS)
        f.flush()
        for n in n_sweep:
            for planner in args.planners:
                t0 = time.time()
                result = run_one(
                    planner, n, target, batch=args.batch, inverse=args.inverse,
                    skip_toolchain=args.skip_toolchain, mojo_root=args.mojo_root,
                    m2ndp_root=args.m2ndp_root, build_timeout=args.build_timeout,
                    run_timeout=args.run_timeout, seed=args.seed,
                    persistent_mode=args.persistent_mode,
                )
                elapsed = time.time() - t0
                row = [_serialize(getattr(result, field_name)) for field_name in CSV_FIELDS]
                writer.writerow(row)
                f.flush()
                print(
                    f"N={n:5d} planner={planner:20s} status={result.status:20s} "
                    f"correct={result.correct!s:5s} spill={result.spill!s:5s} "
                    f"cycles={result.measured_total_cycles!s:12s} ({elapsed:.1f}s)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
