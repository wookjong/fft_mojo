from __future__ import annotations

"""Priority-2 PHYSICAL validation -- real Mojo -> llc -> M2NDP-Detour
toolchain, not the structural/translation-based checks in
verify_stage_active_workers.py / verify_fft_cooperative_partition_modes.py /
verify_fft_persistent_copy_modes.py / verify_fft_persistent_tail_hybrid.py /
priority2_ab_harness.py (all of which remain valid and unchanged -- this
file ADDS a real-execution layer, it does not replace them).

This file is LOCAL-ONLY (never committed) -- it is meant to run inside the
Docker container where the real toolchain is available (see the task's own
Docker/GHCR environment-setup instructions), using
`planning.diagnostics.spill_probe.probe_source_spill_free` (the EXISTING
production spill-detection mechanism) for every build+run, never an ad hoc
regex.

Every row records `estimated_cost` and `measured_cycles` in SEPARATE
columns, `median_cycles` (of N repetitions) as the primary reported number,
and raw per-repetition cycles are kept too. No comparison across different
plans is used to attribute a difference to a single execution option unless
every other axis is held fixed (mirrors priority2_ab_harness.py's own
"GPU logical plan held fixed" discipline).
"""

import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.diagnostics.spill_probe import probe_source_spill_free, _toolchain_env
from planning.diagnostics.stage_active_workers import (
    compute_stage_active_worker_reports,
    format_stage_active_worker_report,
)
from planning.execution.fft_plan_cooperative import make_cooperative_leaf_plan
from planning.execution.fft_plan_persistent import (
    make_persistent_leaf_plan,
    make_persistent_tail_hybrid_plan,
    num_rounds,
    round_active_groups,
)
from planning.gpu_baseline.common import wrap_leaf_as_recursive_plan
from planning.search.fft_cost_model import (
    DEFAULT_COST_WEIGHTS,
    PRIORITY2_COST_WEIGHTS,
    estimate_cost,
    estimate_metrics,
)
from codegen.fft_codegen import generate_fft_kernel
from codegen.fft_cooperative_codegen import generate_cooperative_fft_kernel
from codegen.fft_persistent_codegen import (
    generate_persistent_fft_kernel,
    generate_persistent_tail_hybrid_kernel,
)

_T = DEFAULT_TARGET_PROFILE
_MOJO_ROOT = "/opt/mojo"
_M2NDP_ROOT = "/work"
_REPS = 3


@dataclass
class Row:
    experiment: str
    label: str
    build_ok: bool | None = None
    run_ok: bool | None = None
    spill_free: bool | None = None
    spilling_kernels: tuple = ()
    cycles_raw: list = field(default_factory=list)
    median_cycles: float | None = None
    estimated_cost_legacy: float | None = None
    estimated_cost_priority2: float | None = None
    host_verification: str = "unknown"
    notes: str = ""


ROWS: list[Row] = []

_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _flush_results() -> None:
    _RESULTS_DIR.mkdir(exist_ok=True)
    json_path = _RESULTS_DIR / "priority2_physical_validation.json"
    json_path.write_text(json.dumps([asdict(r) for r in ROWS], indent=2), encoding="utf-8")
    csv_path = _RESULTS_DIR / "priority2_physical_validation.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("experiment,label,build_ok,run_ok,spill_free,spilling_kernels,cycles_raw,median_cycles,"
                 "estimated_cost_legacy,estimated_cost_priority2,notes\n")
        for r in ROWS:
            f.write(f'{r.experiment},{r.label},{r.build_ok},{r.run_ok},{r.spill_free},'
                    f'"{r.spilling_kernels}","{r.cycles_raw}",{r.median_cycles},'
                    f'{r.estimated_cost_legacy},{r.estimated_cost_priority2},"{r.notes}"\n')


def _done_labels(experiment: str) -> set[str]:
    """Labels already present in ROWS for `experiment` with a definite
    (non-crashed) outcome -- lets a long experiment (A/C) resume after an
    interrupted run instead of re-running every already-completed build."""
    return {r.label for r in ROWS if r.experiment == experiment}


def _median_probe(source: str, *, label: str, experiment: str, reps: int = _REPS) -> Row:
    row = Row(experiment=experiment, label=label)
    cycles: list[int] = []
    last_result = None
    for i in range(reps):
        result = probe_source_spill_free(
            source, mojo_root=_MOJO_ROOT, m2ndp_root=_M2NDP_ROOT,
            build_timeout=180, run_timeout=150,
        )
        last_result = result
        if not (result.build_ok and result.run_ok):
            break
        if result.ndp_cycles is not None:
            cycles.append(result.ndp_cycles)
    assert last_result is not None
    row.build_ok = last_result.build_ok
    row.run_ok = last_result.run_ok
    row.spill_free = last_result.spill_free
    row.spilling_kernels = tuple(last_result.spilling_kernels)
    row.cycles_raw = cycles
    row.median_cycles = statistics.median(cycles) if cycles else None
    if not (last_result.build_ok and last_result.run_ok):
        row.notes = "compile_failure" if not last_result.build_ok else "runtime_failure"
        row.notes += " | tail of log: " + last_result.log[-800:]
        row.host_verification = "not_run"
    else:
        mismatches = last_result.log.count(" mismatch at")
        passed = "verification passed" in last_result.log
        if mismatches:
            row.host_verification = "numerical_mismatch"
            row.notes += f" | HOST VERIFICATION MISMATCH ({mismatches} occurrence(s)) -- see log excerpt"
        elif passed:
            row.host_verification = "passed"
        else:
            row.host_verification = "unknown_no_marker_found"
    print(f"  [{experiment}/{label}] build_ok={row.build_ok} run_ok={row.run_ok} "
          f"spill_free={row.spill_free} spilling={row.spilling_kernels} "
          f"cycles_raw={row.cycles_raw} median={row.median_cycles} "
          f"host_verification={row.host_verification}")
    return row


# ---------------------------------------------------------------------------
# Experiment D -- active-worker diagnostic sanity check
# ---------------------------------------------------------------------------

def experiment_d() -> None:
    print("\n=== Experiment D: active-worker diagnostic vs real execution ===")
    cases = [
        ("N64_444_W16", 64, (4, 4, 4), 16),
        ("N105_357_W10_ragged", 105, (3, 5, 7), 10),
    ]
    done = _done_labels("D")
    for label, length, radices, workers in cases:
        if label in done:
            print(f"  [D/{label}] already done, skipping")
            continue
        plan = make_persistent_leaf_plan(length, radices, num_logical_blocks=1, target=_T, workers_per_fft=workers)
        rec = wrap_leaf_as_recursive_plan(length=length, total_ffts=1, inverse=False, built_plan=plan)
        reports = compute_stage_active_worker_reports(rec)
        for r in reports:
            print(f"    diag: {format_stage_active_worker_report(r)}")
        src = generate_persistent_fft_kernel(plan, num_logical_blocks=1)
        row = _median_probe(src, label=label, experiment="D")
        row.notes += (f" | diag_stage0_active={reports[0].active_physical_workers} "
                       f"diag_stage0_idle={reports[0].idle_physical_workers} "
                       f"diag_batches_per_lane={reports[0].physical_batches_per_lane}")
        ROWS.append(row)
        _flush_results()


# ---------------------------------------------------------------------------
# Experiment A -- cooperative partition_mode
# ---------------------------------------------------------------------------

def experiment_a() -> None:
    print("\n=== Experiment A: cooperative partition_mode (real execution) ===")
    cases = [
        ("regular_W4_total8", 64, (4, 4, 4), 4, 8),
        # workers_per_fft MUST be a divisor of target.interleave_chunk_uthreads
        # (8) -- see fft_plan_cooperative.worker_candidates_per_fft's own
        # docstring/divisor-list construction. workers_per_fft=3 is never a
        # real planner candidate (found the hard way: an earlier version of
        # this case used W=3 and got a real host_verification=numerical_
        # mismatch on every partition_mode -- not a P2.2 defect, just an
        # out-of-precondition config outside what the cooperative path was
        # ever built/verified for). Raggedness here is in total_ffts %
        # workers_per_fft != 0 instead, with a VALID workers_per_fft=4.
        ("ragged_W4_total17", 64, (4, 4, 4), 4, 17),
        ("workers_lt_physical_W2_total5", 64, (4, 4, 4), 2, 5),
    ]
    done = _done_labels("A")
    for case_label, length, radices, workers, total_ffts in cases:
        for mode in ("round_robin", "contiguous", "balanced_contiguous"):
            label_check = f"{case_label}/{mode}"
            if label_check in done:
                print(f"  [A/{label_check}] already done, skipping")
                continue
            plan = make_cooperative_leaf_plan(
                length, radices, workers_per_fft=workers, total_ffts=total_ffts, partition_mode=mode,
            )
            assert plan.cooperation is not None and plan.cooperation.partition_mode == mode
            src = generate_cooperative_fft_kernel(plan)
            row = _median_probe(src, label=f"{case_label}/{mode}", experiment="A")
            recursive = wrap_leaf_as_recursive_plan(length=length, total_ffts=total_ffts, inverse=False, built_plan=plan)
            metrics = estimate_metrics(recursive, _T)
            row.estimated_cost_legacy = estimate_cost(metrics, DEFAULT_COST_WEIGHTS)
            row.estimated_cost_priority2 = estimate_cost(metrics, PRIORITY2_COST_WEIGHTS)
            ROWS.append(row)
            _flush_results()


# ---------------------------------------------------------------------------
# Experiment B -- persistent copy_mode: scalar vs vectorized_contiguous,
# WITH real assembly inspection (never source-text-only).
# ---------------------------------------------------------------------------

def _build_ir_and_asm(source: str, *, tag: str) -> tuple[str | None, str | None, str]:
    """Real build via mojo, dump device IR via --emit-ir, lower to real
    RISC-V/xm2ndp assembly via the project's own llc. Returns
    (ir_text_or_None, asm_text_or_None, log)."""
    import tempfile
    log_parts = []
    with tempfile.TemporaryDirectory(prefix=f"fft_asm_{tag}_") as work_str:
        work = Path(work_str)
        stage = work / "stage"
        stage.mkdir()
        repo_root = Path(_M2NDP_ROOT)
        for f in (repo_root / "src").glob("*.mojo"):
            (stage / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
        (stage / "gen.mojo").write_text(source, encoding="utf-8")
        env = _toolchain_env(mojo_root=_MOJO_ROOT, m2ndp_root=_M2NDP_ROOT)
        mojo_bin = env["MOJO_BIN"]
        host_stubs_o = work / "host_stubs.o"
        cc = subprocess.run(["cc", "-c", "-O2", str(repo_root / "sim" / "host_stubs.c"), "-o", str(host_stubs_o)],
                             capture_output=True, text=True, timeout=120)
        log_parts.append("cc:" + cc.stdout + cc.stderr)
        if cc.returncode != 0:
            return None, None, "\n".join(log_parts)
        bin_path = work / "bin"
        build = subprocess.run(
            [mojo_bin, "build", "gen.mojo", "-o", str(bin_path), "-Xlinker", str(host_stubs_o), "-Xlinker", "-lm"],
            cwd=stage, env=env, capture_output=True, text=True, timeout=180,
        )
        log_parts.append("build:" + build.stdout + build.stderr)
        if build.returncode != 0 or not bin_path.exists():
            return None, None, "\n".join(log_parts)
        ir = subprocess.run([str(bin_path), "--emit-ir"], env=env, capture_output=True, text=True, timeout=60)
        log_parts.append("emit-ir rc=" + str(ir.returncode))
        if ir.returncode != 0 or not ir.stdout.strip():
            return None, None, "\n".join(log_parts) + "\nemit-ir stderr:" + ir.stderr
        ir_text = ir.stdout
        ir_path = work / "gen.ll"
        ir_path.write_text(ir_text, encoding="utf-8")
        llc = env.get("LLC", "/work/build/llvm/bin/llc")
        asm_path = work / "gen.s"
        lower = subprocess.run(
            [llc, "-O2", "-mtriple=riscv64", "-mattr=+m,+a,+f,+d,+v,+zvl128b,+zfh,+zvfh,+xm2ndp",
             str(ir_path), "-o", str(asm_path)],
            capture_output=True, text=True, timeout=90,
        )
        log_parts.append("llc rc=" + str(lower.returncode) + " " + lower.stdout + lower.stderr)
        if lower.returncode != 0 or not asm_path.exists():
            return ir_text, None, "\n".join(log_parts)
        return ir_text, asm_path.read_text(encoding="utf-8"), "\n".join(log_parts)


def _classify_vectorization(asm_text: str) -> tuple[str, dict]:
    lines = asm_text.splitlines()
    vector_mnemonics = ("vsetvli", "vle32", "vle64", "vse32", "vse64", "vfadd", "vfmul",
                         "vfsub", "vmv.v", "vfmacc", "vfmv")
    hits = {}
    for mnemonic in vector_mnemonics:
        count = sum(1 for l in lines if mnemonic in l)
        if count:
            hits[mnemonic] = count
    verdict = "VECTORIZATION_PHYSICALLY_CONFIRMED" if hits else "VECTORIZATION_NOT_PHYSICALLY_REALIZED"
    return verdict, hits


def experiment_b() -> None:
    print("\n=== Experiment B: persistent copy_mode -- real asm/IR inspection ===")
    length, radices, workers = 64, (4, 4, 4), 16
    done = _done_labels("B")
    for copy_mode in ("scalar", "vectorized_contiguous"):
        label_check = f"copy_mode={copy_mode}"
        if label_check in done:
            print(f"  [B/{label_check}] already done, skipping")
            continue
        plan = make_persistent_leaf_plan(length, radices, num_logical_blocks=1, target=_T,
                                          workers_per_fft=workers, copy_mode=copy_mode)
        assert plan.persistent is not None and plan.persistent.copy_mode == copy_mode
        src = generate_persistent_fft_kernel(plan, num_logical_blocks=1, copy_mode=copy_mode)
        row = _median_probe(src, label=f"copy_mode={copy_mode}", experiment="B")
        ir_text, asm_text, build_log = _build_ir_and_asm(src, tag=copy_mode)
        if asm_text is not None:
            verdict, hits = _classify_vectorization(asm_text)
            row.notes += f" | asm_verdict={verdict} vector_mnemonic_hits={hits}"
            print(f"    asm_verdict={verdict} hits={hits}")
        else:
            row.notes += " | asm inspection FAILED: " + build_log[-500:]
            print(f"    asm inspection FAILED (see notes)")
        ROWS.append(row)
        _flush_results()


# ---------------------------------------------------------------------------
# Experiment C -- persistent tail-round hybrid (HIGHEST PRIORITY): real
# end-to-end compile+execution at N_blocks = 31,32,33,63,64,65.
# ---------------------------------------------------------------------------

def experiment_c() -> None:
    print("\n=== Experiment C: persistent tail-round hybrid (real execution) ===")
    length, radices, workers = 64, (4, 4, 4), 16
    done = _done_labels("C")
    for n_blocks in (31, 32, 33, 63, 64, 65):
        for tail_strategy in ("all_persistent", "noncoop_tail", "cooperative_tail"):
            label_check = f"N{n_blocks}/{tail_strategy}"
            if label_check in done:
                print(f"  [C/{label_check}] already done, skipping")
                continue
            hybrid = make_persistent_tail_hybrid_plan(
                length, radices, num_logical_blocks=n_blocks, target=_T,
                tail_strategy=tail_strategy, workers_per_fft=workers,
            )
            assert hybrid.full_blocks + hybrid.tail_blocks == n_blocks
            label = f"N{n_blocks}/{tail_strategy}"
            try:
                src = generate_persistent_tail_hybrid_kernel(hybrid)
            except Exception as exc:  # noqa: BLE001 -- genuine codegen-time failure is data here
                row = Row(experiment="C", label=label, build_ok=False, run_ok=False,
                          notes=f"codegen raised: {exc!r}")
                print(f"  [C/{label}] codegen FAILED: {exc!r}")
                ROWS.append(row)
                _flush_results()
                continue
            row = _median_probe(src, label=label, experiment="C")
            row.notes += (f" | full_blocks={hybrid.full_blocks} tail_blocks={hybrid.tail_blocks} "
                           f"tail_plan_is_none={hybrid.tail_plan is None}")
            ROWS.append(row)
            _flush_results()


_EXPERIMENTS = {"D": experiment_d, "A": experiment_a, "B": experiment_b, "C": experiment_c}


def _load_existing() -> None:
    json_path = _RESULTS_DIR / "priority2_physical_validation.json"
    if not json_path.exists():
        return
    for d in json.loads(json_path.read_text(encoding="utf-8")):
        ROWS.append(Row(**d))


def main() -> None:
    t0 = time.time()
    only = sys.argv[1:] or list(_EXPERIMENTS.keys())
    _load_existing()
    for key in only:
        _EXPERIMENTS[key]()
    _flush_results()
    print(f"\n[priority2_physical_validation] wrote {len(ROWS)} rows to {_RESULTS_DIR} "
          f"in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
