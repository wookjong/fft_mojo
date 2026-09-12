from __future__ import annotations

"""Analysis / report / graph generation for gpu_vs_m2ndp_benchmark.py's own
CSV output. Pure post-processing -- reads whatever CSV(s) the benchmark
harness wrote, never re-runs anything or invents a measurement. Kept
separate from the harness itself so the (slow, toolchain-dependent)
measurement step and the (fast, toolchain-free) analysis step can be run
independently -- e.g. re-run this after copying results out of the real-
toolchain container the measurements actually ran in.

Usage:
    python3 analyze_gpu_vs_m2ndp.py results_main.csv results_rocfft_tuned.csv \
        --out-dir report/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STATUS_ORDER = [
    "spill_free_correct", "spilling", "numerical_mismatch",
    "compile_failure", "runtime_failure", "unsupported",
]

NATIVE = "m2ndp-native"
GPU_PLANNERS = ["gpu-clfft", "gpu-rocfft-default", "gpu-rocfft-tuned", "gpu-vkfft"]
ALL_PLANNERS = [NATIVE] + GPU_PLANNERS


def load(csv_paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype=str) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    df["n"] = df["n"].astype(int)
    df["num_kernels_num"] = pd.to_numeric(df["num_kernels"], errors="coerce")
    df["transpose_count_num"] = pd.to_numeric(df["transpose_count"], errors="coerce")
    df["cycles_num"] = pd.to_numeric(df["measured_total_cycles"], errors="coerce")
    df["correct_bool"] = df["correct"].map({"True": True, "False": False})
    df["spill_bool"] = df["spill"].map({"True": True, "False": False})
    # De-dup: keep the LAST row for a given (n, planner) -- lets a
    # resumed/appended sweep's re-run of a row supersede an earlier one.
    df = df.drop_duplicates(subset=["n", "planner"], keep="last")
    return df


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    total_n = df["n"].nunique()
    rows = []
    for planner in ALL_PLANNERS:
        sub = df[df["planner"] == planner]
        counts = sub["status"].value_counts()
        row = {"planner": planner, "total_n_tested": len(sub), "total_n_in_sweep": total_n}
        for s in STATUS_ORDER:
            row[s] = int(counts.get(s, 0))
        row["supported_correct"] = row["spill_free_correct"]
        rows.append(row)
    return pd.DataFrame(rows)


def unsupported_reason_summary(df: pd.DataFrame) -> pd.DataFrame:
    """First-line-of-diagnostics summary per planner for every non-spill-
    free-correct row -- a coarse but real ("what did the actual refusal
    say") categorization, not invented buckets."""
    bad = df[df["status"] != "spill_free_correct"].copy()

    def first_reason(diag: str) -> str:
        if not isinstance(diag, str) or not diag:
            return "(no diagnostics)"
        first_line = diag.splitlines()[0]
        return first_line[:140]

    bad["reason"] = bad["diagnostics"].map(first_reason)
    return bad.groupby(["planner", "status", "reason"]).size().reset_index(name="count").sort_values(
        ["planner", "count"], ascending=[True, False]
    )


def performance_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per_n_cycles_wide, summary_ratios). Only `spill_free_
    correct` rows are ever used for ranking/ratio purposes (task's own
    explicit rule) -- a spilling or incorrect row's cycle number, even if
    present, never enters a ratio."""
    ok = df[df["status"] == "spill_free_correct"][["n", "planner", "cycles_num"]]
    wide = ok.pivot(index="n", columns="planner", values="cycles_num").sort_index()

    summary_rows = []
    if NATIVE in wide.columns:
        native = wide[NATIVE]
        for planner in GPU_PLANNERS:
            if planner not in wide.columns:
                continue
            both = wide[[NATIVE, planner]].dropna()
            if both.empty:
                continue
            ratio = both[planner] / both[NATIVE]
            summary_rows.append({
                "gpu_planner": planner,
                "comparable_n_count": len(both),
                "geomean_ratio": float(np.exp(np.mean(np.log(ratio)))),
                "median_ratio": float(np.median(ratio)),
                "best_ratio_for_gpu_planner": float(ratio.min()),
                "best_ratio_at_n": int(ratio.idxmin()),
                "worst_ratio_for_gpu_planner": float(ratio.max()),
                "worst_ratio_at_n": int(ratio.idxmax()),
            })
    summary = pd.DataFrame(summary_rows)
    return wide, summary


def ratio_table(wide: pd.DataFrame) -> pd.DataFrame:
    if NATIVE not in wide.columns:
        return pd.DataFrame()
    out = pd.DataFrame(index=wide.index)
    for planner in GPU_PLANNERS:
        if planner in wide.columns:
            out[planner] = wide[planner] / wide[NATIVE]
    return out


def case_studies(df: pd.DataFrame, wide: pd.DataFrame, ratios: pd.DataFrame, n_focus: list[int]) -> dict:
    cases: dict[str, list[int]] = {}
    all_ratios = ratios.stack()
    if not all_ratios.empty:
        cases["native_much_faster"] = [int(all_ratios.idxmax()[0])]
        cases["closest_to_parity"] = [int((all_ratios - 1).abs().idxmin()[0])]
        gpu_faster = all_ratios[all_ratios < 1.0]
        cases["gpu_faster_than_native"] = sorted(set(int(i[0]) for i in gpu_faster.index)) if not gpu_faster.empty else []
    cases["explicitly_flagged"] = [n for n in n_focus if n in df["n"].unique()]
    # A representative "someone is spilling/unsupported here" N: whichever
    # tested N has the most non-OK planners.
    non_ok = df[df["status"] != "spill_free_correct"]
    if not non_ok.empty:
        worst_n = non_ok.groupby("n").size().idxmax()
        cases["most_refusals"] = [int(worst_n)]
    return cases


def plan_diff_report(df: pd.DataFrame, n: int) -> str:
    sub = df[df["n"] == n]
    lines = [f"### N={n}\n"]
    lines.append("| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, row in sub.iterrows():
        lines.append(
            f"| {row['planner']} | {row['status']} | {row['correct']} | {row['spill']} | "
            f"{row['measured_total_cycles']} | {row['num_kernels']} | {row['transpose_count']} | "
            f"{row['radix_sequence']} | {row['execution_strategy']} |"
        )
    lines.append("")
    for _, row in sub.iterrows():
        if row["status"] == "spill_free_correct":
            lines.append(f"**{row['planner']}** kernel partition: `{row['kernel_partition']}`")
            lines.append(f"  scratchpad_uthreads={row['scratchpad_uthreads']}, "
                          f"dram_read_bytes={row['dram_read_bytes']}, dram_write_bytes={row['dram_write_bytes']}, "
                          f"large_twiddle_count={row['large_twiddle_count']}")
            kc = row.get("kernel_cycles")
            if isinstance(kc, str) and kc not in ("not_available", ""):
                try:
                    kc_d = json.loads(kc)
                    lines.append(f"  per-kernel cycles: {kc_d}")
                except (json.JSONDecodeError, TypeError):
                    pass
        elif row["status"] != "unsupported" or True:
            lines.append(f"**{row['planner']}**: {str(row['diagnostics'])[:400]}")
        lines.append("")
    return "\n".join(lines)


def make_graphs(df: pd.DataFrame, wide: pd.DataFrame, ratios: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Graph 1: measured cycles vs N, one curve per planner.
    fig, ax = plt.subplots(figsize=(10, 6))
    for planner in ALL_PLANNERS:
        if planner in wide.columns:
            series = wide[planner].dropna()
            if not series.empty:
                ax.plot(series.index, series.values, marker="o", label=planner)
    ax.set_xlabel("FFT length N")
    ax.set_ylabel("measured NDP cycles (spill_free_correct only)")
    ax.set_title("Measured M2NDP cycles vs N, by planner")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "graph1_cycles_vs_n.png", dpi=130)
    plt.close(fig)

    # Graph 2: ratio vs N, one curve per GPU planner, 1.0 baseline.
    fig, ax = plt.subplots(figsize=(10, 6))
    for planner in GPU_PLANNERS:
        if planner in ratios.columns:
            series = ratios[planner].dropna()
            if not series.empty:
                ax.plot(series.index, series.values, marker="o", label=planner)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="parity (1.0)")
    ax.set_xlabel("FFT length N")
    ax.set_ylabel("GPU-derived cycles / M2NDP-native cycles")
    ax.set_title("Speed ratio vs M2NDP-native, by GPU-derived planner")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "graph2_ratio_vs_n.png", dpi=130)
    plt.close(fig)

    # Graph 3: coverage stacked bar per planner.
    cov = coverage_table(df).set_index("planner")
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(cov))
    colors = {
        "spill_free_correct": "#2ca02c", "spilling": "#ff7f0e",
        "numerical_mismatch": "#d62728", "compile_failure": "#9467bd",
        "runtime_failure": "#8c564b", "unsupported": "#7f7f7f",
    }
    for status in STATUS_ORDER:
        vals = cov[status].values.astype(float)
        ax.bar(cov.index, vals, bottom=bottom, label=status, color=colors.get(status))
        bottom += vals
    ax.set_ylabel("count of N values")
    ax.set_title("Coverage by planner and status")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "graph3_coverage.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_paths", nargs="+")
    parser.add_argument("--out-dir", default="report")
    parser.add_argument("--n-focus", type=int, nargs="*", default=[960, 1024])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(args.csv_paths)
    df.to_csv(out_dir / "combined_raw.csv", index=False)

    cov = coverage_table(df)
    cov.to_csv(out_dir / "coverage.csv", index=False)

    reasons = unsupported_reason_summary(df)
    reasons.to_csv(out_dir / "unsupported_reasons.csv", index=False)

    wide, summary = performance_table(df)
    wide.to_csv(out_dir / "cycles_by_n_wide.csv")
    summary.to_csv(out_dir / "performance_summary.csv", index=False)

    ratios = ratio_table(wide)
    ratios.to_csv(out_dir / "ratios_by_n.csv")

    cases = case_studies(df, wide, ratios, args.n_focus)
    with (out_dir / "case_studies.json").open("w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    case_report_lines = ["# Representative plan differences\n"]
    seen_n = set()
    for label, ns in cases.items():
        case_report_lines.append(f"## {label}: N={ns}\n")
        for n in ns:
            if n in seen_n:
                case_report_lines.append(f"(see N={n} above)\n")
                continue
            seen_n.add(n)
            case_report_lines.append(plan_diff_report(df, n))
    (out_dir / "case_studies.md").write_text("\n".join(case_report_lines), encoding="utf-8")

    try:
        make_graphs(df, wide, ratios, out_dir)
    except ImportError as exc:
        print(f"skipping graphs (matplotlib unavailable): {exc}")

    print(f"wrote analysis to {out_dir}/")
    print("\n=== COVERAGE ===")
    print(cov.to_string(index=False))
    print("\n=== PERFORMANCE SUMMARY (spill_free_correct only) ===")
    print(summary.to_string(index=False) if not summary.empty else "(no comparable spill_free_correct rows yet)")


if __name__ == "__main__":
    main()
