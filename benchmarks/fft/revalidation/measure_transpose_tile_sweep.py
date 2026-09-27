from __future__ import annotations

"""Transpose tile-size re-calibration sweep (sections 12-15 of the task
this script was built from -- see docs/transpose_cost_model_audit.md).

WHY THIS EXISTS: `fft_cost_model.CostWeights.transpose_tile_count` (-2.0)
and `.tile_oversaturation_penalty` (10.0) are both marked "FLAGGED
SUSPECT, NOT YET RE-VERIFIED" in their own comments -- derived from cycle
numbers this project's own `_parse_ndp_cycles` fix (2026-08-31) later
found were very likely measured with the retired `tail -1`-on-Gantt-log
convention (a confirmed-wrong reading for the N=1024 point both weights
cite: the "2111 cycles" figure was actually the LAST kernel's own
duration, not the true end-to-end total of 49061). Nobody has re-measured
against the corrected parsing pipeline (`spill_probe._parse_ndp_cycles`,
already fixed) since. This script is that re-measurement's own generation
half: hold N/split/radix/execution-strategy/compute_lanes FIXED, vary only
`tile_rows`/`tile_cols`, and record one row per (N, tile_rows, tile_cols)
combination with BOTH a per-kernel and a true end-to-end total cycle
count (see `_parse_ndp_cycles_by_task` below) -- so a parsing mistake like
the original one is structurally harder to repeat (both numbers are in
the same row, so "kernel" and "total" are never confused downstream).

Per the task's own section 14 instruction ("measurement이 없으면
coefficient를 바꾸지 말 것, synthetic 값으로 만들지 말 것"): this script
does NOT invent, guess, or synthesize any cycle number. It either runs the
real Mojo -> llc -> M2NDP-Detour toolchain (see `--execute`, requires
`<repo>/toolchain/bin/mojo(.real)` -- NOT present in every environment
this cost model's own source tree is checked out into, confirmed absent
in the environment this script was authored in) or, without `--execute`,
only DRY-RUNS the plan-building/codegen-generation half (confirms every
candidate plan actually builds and its transpose stage shape is what this
script's own row claims) and prints the exact command to actually measure.

Usage (dry run, no toolchain required -- confirms candidates build):
    python3 revalidation/measure_transpose_tile_sweep.py

Usage (real measurement, requires the toolchain):
    python3 revalidation/measure_transpose_tile_sweep.py --execute \\
        --out revalidation/transpose_tile_sweep.csv

Writes one CSV row per (N, tile_rows, tile_cols) candidate, columns
matching this module's own `TileSweepRow` fields -- exactly the field
list section 13 of the task this was built from specifies (N, split,
stage type, tile_rows, tile_cols, tile_count, tail_tile_count,
active_ndp_units, bytes_read, bytes_written, measured kernel/total
cycles, correctness, spill status).
"""

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.diagnostics.fft_unit_utilization import compute_unit_utilization
from planning.diagnostics.spill_probe import _toolchain_env
from planning.strategies.fft_plan_recursive import (
    FFTRecursiveNodePlan,
    PhysicalTransposePlan,
    flatten_recursive_node,
    make_recursive_transpose_plan,
)
from codegen.fft_transpose_codegen import generate_recursive_fft_kernels

# Representative N from this project's own currently-supported set,
# deliberately re-using the two N the SUSPECT coefficients' own comments
# cite (N=1024, N=630) so a re-measurement is directly comparable to the
# old (suspect) numbers, plus one more (N=960) already used elsewhere in
# this project's own revalidation suite (revalidate_cost_model.py) for
# cross-script consistency.
DEFAULT_N_VALUES = (630, 960, 1024)

# Tile-size candidates to sweep per N: 1x1 (maximally many, smallest
# tiles -- the shape `transpose_tile_count`'s own comment claims wins),
# 2x2, 4x4, and `None` (this project's own `_default_tile_size` --
# "the largest divisor of gcd(rows,cols) that still fits under
# min(simd_lanes, rows, cols)", i.e. today's shipped default, so this
# sweep always includes the actual production baseline as one point).
TILE_CANDIDATES: tuple[tuple[int, int] | None, ...] = (None, (1, 1), (2, 2), (4, 4))

_SCRATCHPAD_BUDGET = 4096
_COMPUTE_LANES = 4  # this project's own shipped default (make_fft_kernel.py)


@dataclass
class TileSweepRow:
    n: int
    split_near_length: int
    stage_type: str  # "PRE" | "MIDDLE" | "POST"
    tile_rows: int
    tile_cols: int
    tile_count: int
    tail_tile_count: int  # 0 or 1 -- whether this stage's own tile has a tail (rows/cols not evenly divided)
    active_ndp_units: int
    bytes_read: int
    bytes_written: int
    measured_kernel_cycles: int | None = None
    measured_total_cycles: int | None = None
    correctness: str = "not_measured"  # "not_measured" | "pass" | "fail"
    spill_status: str = "not_measured"  # "not_measured" | "spill_free" | "spilling"
    build_ok: bool | None = None
    run_ok: bool | None = None
    error: str = ""


def _stage_rows(plan_root, n: int) -> list[TileSweepRow]:
    rows: list[TileSweepRow] = []
    assert isinstance(plan_root, FFTRecursiveNodePlan), (
        f"N={n}: expected a single-split recursive node (this sweep's own tile_rows/"
        f"tile_cols override only reaches the TOP-level PRE/MIDDLE/POST triple -- see "
        f"make_recursive_transpose_plan's own docstring); got {type(plan_root)}"
    )
    for stage_type, stage in (
        ("PRE", plan_root.pre_transpose),
        ("MIDDLE", plan_root.middle_transpose),
        ("POST", plan_root.post_transpose),
    ):
        assert isinstance(stage, PhysicalTransposePlan)
        tail = 1 if (stage.rows % stage.tile_rows != 0 or stage.cols % stage.tile_cols != 0) else 0
        # One real+imag FP32 element read, and one written, per array
        # position -- see fft_cost_model.estimate_metrics's own "DRAM full-
        # array passes" accounting for the same convention applied here to
        # ONE transpose stage instead of a whole plan.
        bytes_moved = stage.replica_count * stage.rows * stage.cols * 2 * 4
        rows.append(
            TileSweepRow(
                n=n, split_near_length=plan_root.b, stage_type=stage_type,
                tile_rows=stage.tile_rows, tile_cols=stage.tile_cols,
                tile_count=stage.total_uthreads, tail_tile_count=tail,
                active_ndp_units=0,  # filled in by the caller via compute_unit_utilization
                bytes_read=bytes_moved, bytes_written=bytes_moved,
            )
        )
    return rows


def _fill_active_ndp_units(rows: list[TileSweepRow], plan) -> None:
    estimates = compute_unit_utilization(plan, DEFAULT_TARGET_PROFILE)
    by_family = {
        "transpose_pre": "PRE", "transpose_middle": "MIDDLE", "transpose_post": "POST",
    }
    family_to_units = {
        by_family[e.family]: e.active_units_best for e in estimates if e.family in by_family
    }
    for row in rows:
        if row.stage_type in family_to_units:
            row.active_ndp_units = family_to_units[row.stage_type]


# Mirrors spill_probe._parse_ndp_cycles's own two regexes exactly (kept in
# sync by hand -- this script is deliberately independent of that private
# helper so a caller can see BOTH the per-kernel and the corrected total
# in one row; see this module's own docstring for why conflating them was
# the original mistake). Returns (per_task_cycles, total_cycles) where
# per_task_cycles is one entry per registered-task boundary in launch
# order -- for this sweep's own single-split plan, in `flatten_recursive_
# node` order: PRE, near_fft, MIDDLE, far_child, POST.
import re

_TASK_REGISTERED_RE = re.compile(r"Task Registered")
_NDP_CYCLE_RE = re.compile(r"NDP Cycle[: ]+(\d+)")


def _parse_ndp_cycles_by_task(log: str) -> tuple[list[int], int | None]:
    per_task: list[int] = []
    current_group_last: int | None = None
    any_seen = False
    for line in log.splitlines():
        if _TASK_REGISTERED_RE.search(line):
            if current_group_last is not None:
                per_task.append(current_group_last)
            current_group_last = None
            continue
        m = _NDP_CYCLE_RE.search(line)
        if m:
            current_group_last = int(m.group(1))
            any_seen = True
    if current_group_last is not None:
        per_task.append(current_group_last)
    total = sum(per_task) if any_seen else None
    return per_task, total


def _find_mojo_bin(mojo_root: Path) -> Path | None:
    for name in ("mojo.real", "mojo"):
        candidate = mojo_root / "bin" / name
        if candidate.exists():
            return candidate
    return None


def _build_and_run(source: str, *, repo_root: Path, timeout: int = 240) -> tuple[bool, bool, str]:
    import subprocess
    import tempfile

    env = _toolchain_env(mojo_root=None, m2ndp_root=None)
    mojo_root = Path(env["MOJO_ROOT"])
    mojo_bin = _find_mojo_bin(mojo_root)
    if mojo_bin is None:
        return False, False, f"no mojo binary found under {mojo_root / 'bin'}"

    src_dir = repo_root / "src"
    host_stubs = repo_root / "sim" / "host_stubs.c"
    with tempfile.TemporaryDirectory(prefix="fft_transpose_sweep_") as work_str:
        work = Path(work_str)
        stage = work / "stage"
        stage.mkdir()
        for f in src_dir.glob("*.mojo"):
            (stage / f.name).write_text(f.read_text())
        (stage / "gen.mojo").write_text(source)

        host_obj = work / "host_stubs.o"
        subprocess.run(
            ["cc", "-c", "-O2", str(host_stubs), "-o", str(host_obj)],
            check=True, capture_output=True,
        )

        bin_path = work / "bin"
        build = subprocess.run(
            [str(mojo_bin), "build", "gen.mojo", "-o", str(bin_path),
             "-Xlinker", str(host_obj), "-Xlinker", "-lm"],
            cwd=stage, env=env, capture_output=True, text=True, timeout=timeout,
        )
        if build.returncode != 0:
            return False, False, build.stdout + build.stderr

        run = subprocess.run(
            [str(bin_path)], cwd=stage, env=env, capture_output=True, text=True, timeout=timeout,
        )
        return True, run.returncode == 0, build.stdout + build.stderr + run.stdout + run.stderr


def measure_one_n(n: int, *, execute: bool, repo_root: Path) -> list[TileSweepRow]:
    all_rows: list[TileSweepRow] = []
    for tile in TILE_CANDIDATES:
        tile_rows, tile_cols = tile if tile is not None else (None, None)
        try:
            plan = make_recursive_transpose_plan(
                n, scratchpad_byte_budget=_SCRATCHPAD_BUDGET,
                tile_rows=tile_rows, tile_cols=tile_cols,
            )
        except Exception as exc:  # noqa: BLE001 -- report, don't crash the sweep
            print(f"    SKIP N={n} tile={tile}: plan build failed: {exc}")
            continue
        rows = _stage_rows(plan.root, n)
        _fill_active_ndp_units(rows, plan)

        if not execute:
            for row in rows:
                row.error = "dry run (pass --execute to actually build+run)"
            all_rows.extend(rows)
            continue

        source = generate_recursive_fft_kernels(
            plan, compute_lanes=_COMPUTE_LANES, narrow_middle_stages=True,
            reference_check=True, spread_across_units=True,
        )
        build_ok, run_ok, log = _build_and_run(source, repo_root=repo_root)
        per_task_cycles, total_cycles = _parse_ndp_cycles_by_task(log) if (build_ok and run_ok) else ([], None)
        spilling = "vs1r.v" in log or "vs2r.v" in log or "Unsupported Instruction" in log
        passed = "[PASS]" in log
        failed = "[FAIL]" in log

        # This sweep's own single-split plan renders as (in flatten_
        # recursive_node order): PRE, near_fft, MIDDLE, far_child, POST --
        # 5 registered tasks. Map each transpose row to its own index.
        stage_order = {"PRE": 0, "MIDDLE": 2, "POST": 4}
        for row in rows:
            row.build_ok, row.run_ok = build_ok, run_ok
            row.measured_total_cycles = total_cycles
            idx = stage_order[row.stage_type]
            row.measured_kernel_cycles = per_task_cycles[idx] if idx < len(per_task_cycles) else None
            row.spill_status = "spilling" if spilling else ("spill_free" if (build_ok and run_ok) else "not_measured")
            row.correctness = "pass" if passed else ("fail" if failed else "not_measured")
            if not build_ok:
                row.error = "build failed"
            elif not run_ok:
                row.error = "run failed"
        all_rows.extend(rows)
        print(
            f"    N={n} tile={tile_rows}x{tile_cols}: build_ok={build_ok} run_ok={run_ok} "
            f"total_cycles={total_cycles}"
        )
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="*", default=list(DEFAULT_N_VALUES))
    parser.add_argument("--execute", action="store_true", help="actually build+run via the real toolchain")
    parser.add_argument("--out", type=str, default="revalidation/transpose_tile_sweep.csv")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    if args.execute:
        mojo_root = repo_root / "toolchain"
        if _find_mojo_bin(mojo_root) is None:
            print(
                f"[measure_transpose_tile_sweep] --execute requested but no mojo "
                f"binary found under {mojo_root / 'bin'} -- this environment has no "
                f"built Mojo/M2NDP-Detour toolchain. Falling back to a dry run "
                f"(plan-building only, no real measurement). To actually measure, "
                f"build the toolchain per this repo's own scripts/build.sh / "
                f"scripts/setup.sh, then re-run:\n"
                f"    python3 revalidation/measure_transpose_tile_sweep.py --execute "
                f"--n {' '.join(str(n) for n in args.n)} --out {args.out}\n"
            )
            args.execute = False

    print(f"  Transpose tile-size sweep: N={args.n}, tiles={TILE_CANDIDATES}, execute={args.execute}")
    all_rows: list[TileSweepRow] = []
    for n in args.n:
        all_rows.extend(measure_one_n(n, execute=args.execute, repo_root=repo_root))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()) if all_rows else [])
        if all_rows:
            writer.writeheader()
            for row in all_rows:
                writer.writerow(asdict(row))
    print(f"  Wrote {len(all_rows)} rows to {out_path}")
    if not args.execute:
        print(
            "  NOTE: this was a DRY RUN -- every row's own measured_kernel_cycles/"
            "measured_total_cycles/correctness/spill_status is 'not_measured'/None. "
            "No coefficient should be derived from this CSV. Re-run with --execute "
            "on a machine with the real toolchain built to get actual measurements."
        )


if __name__ == "__main__":
    main()
