"""Parses a `probes/probe_pool_alignment.py`-driven sweep log (one `===
N=<n> strategy=<s> ===` block per run, `[pool_align] ...` lines, then a
`cycles=... spill=... panic=...` summary line) and cross-checks each
observed runtime pool address against `planning.diagnostics.fft_unit_utilization`'s
best/worst active-unit predictions -- part of the Phase 1 runtime-
alignment audit, see docs/active_ndp_units_cost_task.md.

Usage: python3 analyze_pool_probe.py RESULTS.log
"""
import re
import sys
from pathlib import Path

# This file lives in benchmarks/fft/revalidation/ -- `planning`/`codegen`
# are importable from benchmarks/fft/ itself (one level up), which Python
# does not add to sys.path automatically once a script is no longer
# directly in that directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.strategies.fft_plan_recursive import make_recursive_transpose_plan
from planning.diagnostics.fft_unit_utilization import compute_unit_utilization, distinct_active_units
from planning.core.target_profile import DEFAULT_TARGET_PROFILE as TARGET

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pool_probe_results2.log"
text = open(path).read()
blocks = re.split(r"^=== (N=\d+ strategy=\S+) ===$", text, flags=re.M)

print(f"{'N':>6} {'strategy':>13} {'stage':>5} {'kind':>16} {'mod256':>7} "
      f"{'boff(uthreads)':>15} {'active(off=0)':>14} {'active(off=6)':>14} "
      f"{'active(observed)':>17} {'differs?':>9}")

for i in range(1, len(blocks), 2):
    header, body = blocks[i], blocks[i + 1]
    n_str, strategy = re.match(r"N=(\d+) strategy=(\S+)", header).groups()
    n = int(n_str)
    plan_kwargs = dict(scratchpad_byte_budget=4096, simd_lanes=8)
    if strategy == "cooperative":
        plan_kwargs["cooperative_workers"] = 4
    elif strategy == "persistent":
        plan_kwargs["persistent_leaf"] = True
    plan = make_recursive_transpose_plan(n, **plan_kwargs)
    estimates = compute_unit_utilization(plan, TARGET)

    pool_lines = [l for l in body.splitlines() if l.startswith("[pool_align]")]
    for estimate, line in zip(estimates, pool_lines):
        kind = re.search(r"kind (\S+)", line).group(1)
        mod256 = int(re.search(r"mod256 (\d+)", line).group(1))
        boff = mod256 // TARGET.uthread_bytes
        stage_idx = re.search(r"stage (\d+)", line).group(1)
        if kind == "persistent":
            # distinct_active_units (address-interleave chunk counting) is
            # the wrong mechanism for a persistent leaf -- its own active-
            # unit count comes from round_active_groups over `replicas`
            # (fft_unit_utilization._persistent_leaf_estimate), a 1:1
            # software-group-to-unit mapping, not an address-interleave
            # computation over `logical_work_count`. Its alignment is
            # already confirmed exact (boff==0 always observed) so there is
            # no best/worst ambiguity to cross-check here anyway.
            print(f"{n:>6} {strategy:>13} {stage_idx:>5} {kind:>16} {mod256:>7} {boff:>15} "
                  f"{'n/a (round_active_groups, not addr-interleave)':>63}")
            continue
        observed = distinct_active_units(
            estimate.logical_work_count, base_offset_uthreads=boff,
            interleave_chunk_uthreads=TARGET.interleave_chunk_uthreads,
            num_ndp_units=TARGET.num_ndp_units,
        )
        differs = observed != estimate.active_units_best
        print(f"{n:>6} {strategy:>13} {stage_idx:>5} {kind:>16} {mod256:>7} {boff:>15} "
              f"{estimate.active_units_best:>14} {estimate.active_units_worst:>14} "
              f"{observed:>17} {'YES' if differs else '':>9}")

    cyc = re.search(r"cycles=(\d+) spill=(\S+) panic=(\S+)", body)
    if cyc:
        print(f"       -> cycles={cyc.group(1)} spill={cyc.group(2)} panic={cyc.group(3)}")
