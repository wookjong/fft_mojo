"""Ad-hoc standalone-cooperative-leaf generator for the workers_per_fft=8
correctness investigation -- bypasses make_fft_kernel.py's CLI (which only
ever offers `choose_workers_per_fft`'s auto-selected/capped value, never a
raw override) and fft_plan_cooperative.worker_candidates_per_fft's default
`exclude_full_interleave_chunk=True` gate entirely, calling
`make_cooperative_leaf_plan`/`generate_cooperative_fft_kernel` directly --
same as verification/verify_fft_cooperative.py's own Python-harness path,
but rendered to a real .mojo file for the actual M2NDP-Detour simulator.

    python3 gen_coop_repro.py 128 4,4,4,2 --workers 8 --total-ffts 4 -o gen.mojo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codegen.fft_cooperative_codegen import generate_cooperative_fft_kernel
from planning.execution.fft_plan_cooperative import make_cooperative_leaf_plan
from planning.core.target_profile import DEFAULT_TARGET_PROFILE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("length", type=int)
    ap.add_argument("radices", type=str, help="comma-separated, e.g. 4,4,4,2")
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--total-ffts", type=int, default=1)
    ap.add_argument("--inverse", action="store_true")
    ap.add_argument("--compute-lanes", type=int, default=None)
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument(
        "--no-spad-cap", action="store_true",
        help="omit spad_capacity_bytes/max_concurrent_scratchpad_bytes (uncapped fft_slots_per_group)",
    )
    ap.add_argument("-o", "--output", type=str, default="gen.mojo")
    args = ap.parse_args()

    radices = tuple(int(x) for x in args.radices.split(","))

    plan = make_cooperative_leaf_plan(
        length=args.length,
        radices=radices,
        workers_per_fft=args.workers,
        total_ffts=args.total_ffts,
        inverse=args.inverse,
        kernel_name="FFTCoopRepro",
        spad_capacity_bytes=None if args.no_spad_cap else DEFAULT_TARGET_PROFILE.spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=None if args.no_spad_cap else DEFAULT_TARGET_PROFILE.max_concurrent_scratchpad_bytes,
    )
    print(
        f"plan: length={args.length} radices={radices} workers_per_fft={args.workers} "
        f"total_ffts={args.total_ffts} fft_slots_per_group={plan.cooperation.fft_slots_per_group} "
        f"max_uthread(physical)={plan.max_uthread} total_uthreads(physical)={plan.total_uthreads}"
    )

    text = generate_cooperative_fft_kernel(
        plan, compute_lanes=args.compute_lanes, in_place=args.in_place
    )
    Path(args.output).write_text(text)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
