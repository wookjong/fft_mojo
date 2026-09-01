from __future__ import annotations

"""Regression coverage for `planning.spill_probe._parse_ndp_cycles`'s
2026-08-31 fix -- see that function's own comment and
docs/active_ndp_units_cost_task.md's "Phase 1.5" section for the full
root-cause writeup. Synthetic log snippets only (no real toolchain
needed): a multi-kernel (split) plan spawns a fresh `m2ndp_run`
subprocess per top-level kernel struct, each restarting M2NDPConfig's own
`ndp_cycle` at 0 -- confirmed by reading `src/m2ndp.mojo`'s `Self.launch()`
and a real run log's own "Registered task ... ndp cycle 0" lines, one per
distinct struct. The plan's true total is the SUM of each struct's own
final cycle value, not the single last Gantt line in the whole log (the
old, wrong convention -- captured only the last struct's own standalone
duration, ~19x too low on a real N=630 non-cooperative run: 1867 vs the
true 35246).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.spill_probe import _parse_ndp_cycles


def _registered(path: str) -> str:
    return f"[info] Host 0 Registered task {path} id 0 at core cycle 0 ndp cycle 0 to CXL 0"


def _finished(kernel: str, launch_id: int, cycle: int) -> str:
    return (
        f"[info] Gantt info: Host 0 finished NDP kernel {kernel}() "
        f"launch id {launch_id} at core cycle 0 ndp cycle {cycle} at CXL 0"
    )


def check_single_kernel_plan_unaffected() -> None:
    """A single-kernel (no split) plan has exactly one 'Registered task'
    group -- the old and new parsing must agree here, since this is the
    case that was never actually broken."""
    log = "\n".join([
        _registered("/tmp/x/task.elf"),
        _finished("gen::FFTRec::stage_0", 0, 517),
    ])
    assert _parse_ndp_cycles(log) == 517
    print("    OK   single-kernel plan: total == its own one Gantt value (517), unaffected by the fix")


def check_multi_kernel_plan_sums_per_group() -> None:
    """A real N=630 non-cooperative shape: 5 top-level kernel structs
    (Pre0, Near0 x4 launches, Mid0, Leaf1 x2 launches, Post0), values taken
    verbatim from an actual run log. Old convention (last Gantt line only)
    gave 1867; correct total is 35246."""
    log = "\n".join([
        _registered("/tmp/a/task.elf"),
        _finished("gen::FFTRecPre0::stage_0", 0, 1872),
        _registered("/tmp/b/task.elf"),
        _finished("gen::FFTRecNear0::stage_0", 0, 5511),
        _finished("gen::FFTRecNear0::stage_0_tail", 1, 6725),
        _finished("gen::FFTRecNear0::stage_1", 2, 19133),
        _finished("gen::FFTRecNear0::stage_2", 3, 26517),
        _registered("/tmp/c/task.elf"),
        _finished("gen::FFTRecMid0::stage_0", 0, 2428),
        _registered("/tmp/d/task.elf"),
        _finished("gen::FFTRecLeaf1::stage_0", 0, 1657),
        _finished("gen::FFTRecLeaf1::stage_1", 1, 2562),
        _registered("/tmp/e/task.elf"),
        _finished("gen::FFTRecPost0::stage_0", 0, 1867),
    ])
    old_wrong_convention = 1867  # last Gantt line only -- what this project used to compute
    correct_total = 1872 + 26517 + 2428 + 2562 + 1867
    assert correct_total == 35246
    result = _parse_ndp_cycles(log)
    assert result == correct_total, f"expected {correct_total}, got {result}"
    assert result != old_wrong_convention
    print(f"    OK   5-struct split plan (real N=630 non-cooperative values): "
          f"total={result} (old wrong convention would have said {old_wrong_convention}, "
          f"~{result / old_wrong_convention:.0f}x too low)")


def check_no_gantt_lines_returns_none() -> None:
    """A build/run failure with no Gantt output at all must return None,
    not 0 or an empty-string crash -- a caller (spill_probe.probe_spill_
    free) treats None as 'could not measure', never as a real zero."""
    log = "some build error\nno kernels ran"
    assert _parse_ndp_cycles(log) is None
    print("    OK   log with no Gantt lines -> None, not 0")


def check_trailing_registered_task_with_no_finish() -> None:
    """A 'Registered task' with no matching 'finished' line before the log
    ends (e.g. the run crashed mid-kernel) must not silently inflate the
    total with a stale value from an earlier group."""
    log = "\n".join([
        _registered("/tmp/a/task.elf"),
        _finished("gen::FFTRecPre0::stage_0", 0, 1872),
        _registered("/tmp/b/task.elf"),
        # crashed here -- no finished line for this second task
    ])
    assert _parse_ndp_cycles(log) == 1872
    print("    OK   trailing unfinished task group contributes 0, not a stale carry-over")


def main() -> None:
    print("  spill_probe.py: _parse_ndp_cycles multi-kernel-plan fix (2026-08-31):")
    check_single_kernel_plan_unaffected()
    check_multi_kernel_plan_sums_per_group()
    check_no_gantt_lines_returns_none()
    check_trailing_registered_task_with_no_finish()
    print("[verify] ndp_cycles parsing fix: all checks passed")


if __name__ == "__main__":
    main()
