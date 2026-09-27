from __future__ import annotations

"""Analysis half of the transpose tile-size re-calibration (section 15 of
the task this script was built from -- see docs/transpose_cost_model_audit.md
and `measure_transpose_tile_sweep.py`'s own module docstring for the
measurement half this reads).

Answers ONE question: which transpose metric actually explains cycle
variation -- `tile_count`, `tail_tile_count`, `active_ndp_units`, bytes
moved, or the (tile_rows, tile_cols) pair itself -- NOT "what coefficient
best fits this data" (overfitting a handful of real-hardware points to an
exact multi-term formula is exactly what produced the original SUSPECT
`transpose_tile_count=-2.0`/`tile_oversaturation_penalty=10.0` weights in
the first place -- see fft_cost_model.CostWeights' own comments). Splits
rows into a calibration and a held-out validation set (deterministic,
seeded) and reports each single-variable correlation/fit on calibration
only, then how well that same single-variable fit predicts the held-out
rows -- so a "this metric explains cycles" claim is never just a curve
laid on top of the exact points it was read off of.

Usage:
    python3 revalidation/analyze_transpose_tile_sweep.py \\
        revalidation/transpose_tile_sweep.csv

Refuses to run on a CSV whose rows are all `measured_total_cycles=` empty
(i.e. a dry-run-only sweep, see `measure_transpose_tile_sweep.py`) --
there is nothing to analyze yet, and section 14's own "no synthetic
coefficient" rule applies here too: this script must never quietly
fabricate a trend line from absent data.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _measured_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("measured_total_cycles")]


def _split_calibration_validation(rows: list[dict], *, val_fraction: float = 0.3, seed: int = 0):
    """Deterministic pseudo-random split (Python's own `random.Random(seed)`,
    not numpy, to keep this script dependency-free beyond the stdlib) --
    stratified by `n` so every N contributes to both sets whenever it has
    enough rows, rather than one N landing entirely in validation by luck."""
    import random

    by_n: dict[str, list[dict]] = {}
    for r in rows:
        by_n.setdefault(r["n"], []).append(r)

    rng = random.Random(seed)
    calibration: list[dict] = []
    validation: list[dict] = []
    for n, group in by_n.items():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
        validation.extend(shuffled[:n_val])
        calibration.extend(shuffled[n_val:])
    return calibration, validation


def _simple_linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Ordinary least squares, one variable: returns (slope, intercept).
    Stdlib-only (no numpy dependency for this script)."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return 0.0, mean_y
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        return None


def _rmse(xs: list[float], ys: list[float], *, slope: float, intercept: float) -> float:
    if not xs:
        return float("nan")
    errs = [(slope * x + intercept - y) ** 2 for x, y in zip(xs, ys)]
    return (sum(errs) / len(errs)) ** 0.5


def analyze_metric(
    metric_name: str, calibration: list[dict], validation: list[dict], *, target_field: str,
) -> None:
    cal_xs = [float(r[metric_name]) for r in calibration]
    cal_ys = [float(r[target_field]) for r in calibration]
    slope, intercept = _simple_linear_fit(cal_xs, cal_ys)
    r = _pearson(cal_xs, cal_ys)
    cal_rmse = _rmse(cal_xs, cal_ys, slope=slope, intercept=intercept)

    val_xs = [float(r[metric_name]) for r in validation]
    val_ys = [float(r[target_field]) for r in validation]
    val_rmse = _rmse(val_xs, val_ys, slope=slope, intercept=intercept) if val_xs else float("nan")

    r_str = f"{r:+.3f}" if r is not None else "n/a (degenerate)"
    print(
        f"    {metric_name:20s}  pearson_r={r_str:>16s}  fit: cycles ~= {slope:.4f}*x + "
        f"{intercept:.1f}  cal_rmse={cal_rmse:.1f}  val_rmse={val_rmse:.1f} "
        f"(n_cal={len(cal_xs)}, n_val={len(val_xs)})"
    )


def analyze_by_tile_shape(rows: list[dict], *, target_field: str) -> None:
    by_shape: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        key = (r["tile_rows"], r["tile_cols"])
        by_shape.setdefault(key, []).append(float(r[target_field]))
    print(f"    mean {target_field} by (tile_rows, tile_cols):")
    for (tr, tc), values in sorted(by_shape.items(), key=lambda kv: (int(kv[0][0]), int(kv[0][1]))):
        print(f"      {tr}x{tc}: mean={statistics.mean(values):.1f} n={len(values)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=str)
    parser.add_argument(
        "--target", type=str, default="measured_total_cycles",
        choices=("measured_total_cycles", "measured_kernel_cycles"),
        help="which cycle column to explain (default: measured_total_cycles, the "
             "corrected end-to-end measurement -- see this project's own "
             "_parse_ndp_cycles fix). measured_kernel_cycles isolates just the ONE "
             "transpose kernel under test, excluding the rest of the chain.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.csv_path)
    rows = _read_rows(path)
    measured = _measured_rows(rows)
    if not measured:
        print(
            f"[analyze_transpose_tile_sweep] {path} has {len(rows)} row(s), none with a "
            f"measured_total_cycles value -- this looks like a dry-run-only sweep (see "
            f"measure_transpose_tile_sweep.py's own docstring). Refusing to analyze: "
            f"there is no real measurement here to explain, and per section 14 of the "
            f"task this was built from, this script must never fabricate a trend from "
            f"absent data. Re-run the sweep with --execute on a machine with the real "
            f"Mojo/M2NDP-Detour toolchain built, then point this script at that CSV."
        )
        sys.exit(1)

    target_field = args.target
    measured = [r for r in measured if r.get(target_field)]
    calibration, validation = _split_calibration_validation(
        measured, val_fraction=args.val_fraction, seed=args.seed,
    )
    print(
        f"  Transpose tile-size analysis: {len(measured)} measured row(s) "
        f"({len(calibration)} calibration / {len(validation)} validation), "
        f"target={target_field}"
    )

    print("  Single-variable fits (calibration-fit, both-set RMSE):")
    for metric in ("tile_count", "tail_tile_count", "active_ndp_units", "bytes_read"):
        analyze_metric(metric, calibration, validation, target_field=target_field)

    print()
    analyze_by_tile_shape(measured, target_field=target_field)

    print()
    print(
        "  Interpretation: the metric with the strongest |pearson_r| on calibration "
        "AND a validation RMSE not much worse than its calibration RMSE is the one "
        "worth a real coefficient -- a metric that fits calibration well but has a "
        "much larger validation RMSE is overfit to this specific sweep and should not "
        "be trusted for a production coefficient. This script does NOT pick a winner "
        "automatically or write anything back to fft_cost_model.py -- see section 14 "
        "of the task this was built from."
    )


if __name__ == "__main__":
    main()
