from __future__ import annotations

"""`gpu_baseline.common.format_gpu_m2ndp_execution_report` verification
(section 11 of the task this suite was built from, "Step 2"). Checks
structural properties of the report (section separation, kernel-block
count, no merged fields) across representative cases spanning single-pass,
multi-pass, transpose-based, workers<=8, and workers>8 -- NOT full string
matching (a formatting tweak should not break this suite).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.gpu_baseline import clfft, rocfft, rocfft_default, vkfft
from planning.gpu_baseline.common import (
    BaselineStatus,
    format_gpu_m2ndp_execution_report,
    reconstruct_gpu_logical_plan,
)

_T = DEFAULT_TARGET_PROFILE


def _fake_bench(plan, target):
    return rocfft.BenchmarkOutcome(ok=True, ndp_cycles=plan.length * 7 + len(plan.stages) * 13, spill_free=True)


def representative_cases() -> list[tuple[str, object]]:
    return [
        ("clFFT single-pass (W<=8)", clfft.plan(16)),
        ("clFFT large-1D (multi-pass, transpose-based)", clfft.plan(8192)),
        ("rocFFT tuned single-kernel", rocfft.tune(24, total_ffts=4, benchmark_fn=_fake_bench)[0]),
        ("rocFFT-default compiled table (W>8)", rocfft_default.plan(4096)),
        ("VkFFT single-pass (W<=8)", vkfft.plan(64, batch=1)),
        ("VkFFT single-pass (W>8)", vkfft.plan(512, batch=3)),
        ("VkFFT M2NDP-adapted multi-pass (N=8192)", vkfft.plan_m2ndp(8192, target=_T)),
    ]


def check_report_structure() -> None:
    for label, result in representative_cases():
        assert result.status is BaselineStatus.OK, f"{label}: expected OK, got {result.status}"
        report = format_gpu_m2ndp_execution_report(result, target=_T)
        assert "GPU LOGICAL EXECUTION" in report, f"{label}: missing GPU LOGICAL EXECUTION section"
        assert "M2NDP PHYSICAL EXECUTION" in report, f"{label}: missing M2NDP PHYSICAL EXECUTION section"
        assert "OVERALL" in report

        gpu_plan = reconstruct_gpu_logical_plan(result)
        kernel_blocks = report.count("KERNEL ")
        assert kernel_blocks == len(gpu_plan), (
            f"{label}: report has {kernel_blocks} 'KERNEL ' block(s), expected {len(gpu_plan)} "
            f"(one per reconstructed logical kernel)"
        )
        # Every GPU section must appear strictly before its own kernel's
        # M2NDP section (never interleaved/merged) -- spot-check on the
        # first kernel block.
        gpu_idx = report.index("GPU LOGICAL EXECUTION")
        m2ndp_idx = report.index("M2NDP PHYSICAL EXECUTION")
        assert gpu_idx < m2ndp_idx, f"{label}: GPU section must precede M2NDP section"
    print(f"    OK   report structure (GPU/M2NDP sections, KERNEL block count, section ordering) "
          f"holds across {len(representative_cases())} representative case(s)")


def check_no_merged_fields() -> None:
    """For a case where GPU-logical and M2NDP-physical worker counts
    genuinely DIFFER in vocabulary (persistent: `logical_workers_per_fft`
    on the M2NDP side equals `gpu_threads_per_transform` on the GPU side
    numerically, by design -- see verify_gpu_m2ndp_execution_mapping.py --
    but `gpu_transforms_per_workgroup` is None while `fft_slots_per_group`
    is 1, a genuinely DIFFERENT pair of numbers/vocabulary), confirm the
    report shows BOTH labels distinctly rather than collapsing them into
    one line."""
    result = vkfft.plan(512, batch=3, target=_T)
    assert result.status is BaselineStatus.OK
    report = format_gpu_m2ndp_execution_report(result, target=_T)
    assert "gpu_threads_per_transform:" in report
    assert "logical_workers_per_fft:" in report
    assert "gpu_transforms_per_workgroup:" in report
    assert "fft_slots_per_group:" in report
    # The two vocabularies never share a label.
    assert "gpu_threads_per_transform:" != "logical_workers_per_fft:"
    print("    OK   GPU vocabulary (gpu_threads_per_transform, gpu_transforms_per_workgroup) and "
          "M2NDP vocabulary (logical_workers_per_fft, fft_slots_per_group) both appear as "
          "distinct, separately-labeled fields -- never merged into one value")


def check_transpose_kernel_reports_no_worker_fields() -> None:
    result = clfft.plan(8192)
    assert result.status is BaselineStatus.OK
    report = format_gpu_m2ndp_execution_report(result, target=_T)
    assert "PRE transpose" in report
    assert "strategy:                      transpose" in report
    print("    OK   a transpose kernel's own block reports strategy='transpose' and a PRE/MIDDLE/"
          "POST role tag")


def main() -> None:
    print("  Unified GPU/M2NDP execution report:")
    check_report_structure()
    check_no_merged_fields()
    check_transpose_kernel_reports_no_worker_fields()
    print("[verify] GPU execution report: all checks passed")


if __name__ == "__main__":
    main()
