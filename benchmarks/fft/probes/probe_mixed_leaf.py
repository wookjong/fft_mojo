"""Ad-hoc probe (same status as probe_pool_alignment.py / id_dump.mojo) --
builds a real N=960-shaped plan with an explicit per-leaf forced_worker_
sequence (Phase 6, docs/active_ndp_units_cost_task.md) and writes the
generated Mojo source to OUT.mojo.

Usage: python3 probe_mixed_leaf.py N near_entry far_entry OUT.mojo
  entry: "none" | "persistent" | an int (cooperative workers)
"""
import sys
from pathlib import Path

# This file lives in benchmarks/fft/probes/ -- `planning`/`codegen` are
# importable from benchmarks/fft/ itself (one level up).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.strategies.fft_plan_recursive import make_recursive_transpose_plan
from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from codegen.fft_transpose_codegen import generate_recursive_fft_kernels

n = int(sys.argv[1])


def _parse_entry(s: str):
    if s == "none":
        return None
    if s == "persistent":
        return "persistent"
    return int(s)


near_entry = _parse_entry(sys.argv[2])
far_entry = _parse_entry(sys.argv[3])
out_path = sys.argv[4]

plan = make_recursive_transpose_plan(
    n, scratchpad_byte_budget=4096, simd_lanes=8,
    forced_worker_sequence=(near_entry, far_entry),
)
source = generate_recursive_fft_kernels(
    plan, target=DEFAULT_TARGET_PROFILE, debug_print_pool_alignment=True,
    reference_check=False, compute_lanes=4, narrow_middle_stages=True,
)
with open(out_path, "w") as f:
    f.write(source)
print(f"wrote {out_path} (N={n} near={near_entry} far={far_entry})", file=sys.stderr)
