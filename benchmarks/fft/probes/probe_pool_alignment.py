"""Ad-hoc probe -- NOT part of the FFT planner/codegen, same "probe
kernel" status as id_dump.mojo (see that file's own docstring). Builds a
real recursive plan for a chosen N/execution-strategy combination and
writes the generated Mojo source to OUT.mojo for a separate build+run
step.

Originally built for the Phase 1 runtime-alignment audit
(docs/active_ndp_units_cost_task.md) -- `debug_print_pool_alignment=True`
is always on, so every run still emits `[pool_align]` lines even when the
caller only cares about cycles (Phase 2's fair split-level execution-
strategy comparison, same doc). Harmless either way: the flag only adds
print statements, changes nothing about which pool a stage gets.

Usage: python3 probe_pool_alignment.py N {noncoop,coop2,coop4,coop8,persistent} OUT.mojo [--reference-check]
"""
import re
import sys
from pathlib import Path

# This file lives in benchmarks/fft/probes/ -- `planning`/`codegen` are
# importable from benchmarks/fft/ itself (one level up).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.strategies.fft_plan_recursive import make_recursive_transpose_plan
from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from codegen.fft_transpose_codegen import generate_recursive_fft_kernels

n = int(sys.argv[1])
strategy = sys.argv[2]
out_path = sys.argv[3]
reference_check = "--reference-check" in sys.argv[4:]

kwargs = dict(scratchpad_byte_budget=4096, simd_lanes=8)
coop_match = re.match(r"coop(\d+)$", strategy)
if coop_match:
    kwargs["cooperative_workers"] = int(coop_match.group(1))
elif strategy == "persistent":
    kwargs["persistent_leaf"] = True
elif strategy != "noncoop":
    raise SystemExit(f"unknown strategy {strategy}")

plan = make_recursive_transpose_plan(n, **kwargs)
source = generate_recursive_fft_kernels(
    plan, target=DEFAULT_TARGET_PROFILE, debug_print_pool_alignment=True,
    reference_check=reference_check,
    # Match make_fft_kernel.py's own shipped default (compute_lanes=4,
    # narrow_middle_stages=True) -- an early probe run omitted these and
    # every config spilled/panicked as a result (VFMV_S_F), producing
    # meaningless "cycles" numbers.
    compute_lanes=4, narrow_middle_stages=True,
)
with open(out_path, "w") as f:
    f.write(source)
print(f"wrote {out_path} (N={n} strategy={strategy} reference_check={reference_check})", file=sys.stderr)
