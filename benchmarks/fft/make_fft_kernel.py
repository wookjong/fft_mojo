from __future__ import annotations

"""Generate a single, working M2NDP FFT Mojo kernel from nothing but a
length -- the same way every other benchmark in this repo is one
self-contained `.mojo` file (see scripts/build.sh's own comment: "A
benchmark is one file -- kernels, device_main and the host main that
launches them").

    python3 make_fft_kernel.py 960
    python3 make_fft_kernel.py 65536 --inverse -o out.mojo

This is a thin, deliberately boring wrapper around planning/fft_plan_recursive
.make_recursive_transpose_plan + codegen/fft_transpose_codegen.
generate_recursive_fft_kernels -- both already extensively verified (see
verification/) -- not a new implementation. That one planning strategy was
picked as the sole backend here because it is the only one that handles
*any* N on its own: it degenerates to a single fused kernel when N already
fits one uthread's own scratchpad, and recurses into a tiled-transpose
boundary only when it doesn't, so "just give it a number" always produces
something. The other strategies (make_444_plan, make_decomposed_plan,
make_multi_kernel_plan, make_balanced_plan, make_balanced_transpose_plan)
stay put in planning/ as the regression baseline/numeric reference they've
always been -- this file does not replace them, it's a convenience front
end for the one strategy general enough to be a sensible default.

The generated file's own shape is meant to be readable at a glance:
every kernel implementation first (one Mojo `struct` per stage -- see
generate_recursive_fft_kernels's own "KERNEL IMPLEMENTATIONS" banner),
then exactly one host `def main()` that allocates buffers, launches them
in order, and checks the result against an independent reference DFT
computed at runtime.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from planning.fft_plan_recursive import make_recursive_transpose_plan



# This target's guaranteed vector register length (+zvl128b, see
# scripts/build.sh's FEATURES) in Float32 elements: 128 bits / 32 bits.
# Above this width RVV needs LMUL>1 register grouping, which is exactly
# what makes a heavily-unrolled kernel spill (see the `compute_lanes`
# discussion below and docs/STATUS.md).
_LMUL1_FLOAT32_LANES = 4

# One NDP unit's own scratchpad, bytes -- matches `spad_size` in
# third_party/m2ndp-detour/config/performance/M2NDP/m2ndp.config and
# `spad (rw) : ORIGIN = 0, LENGTH = 128K` in scripts/m2ndp.lds (both
# 131072); duplicated here for the same reason `_LMUL1_FLOAT32_LANES` is --
# there's no single place in this repo both the C++ simulator config and
# this Python planner could read it from. A margin below the real 131072
# (rather than that exact figure) leaves room for a kernel's own Params
# struct and other scratchpad-resident globals (see scripts/m2ndp.lds's own
# docstring: "the globals are laid out from" the scratchpad base), which
# _cap_max_uthread has no visibility into and so cannot budget for itself.
_SPAD_CAPACITY_BYTES = 120 * 1024

# A cap on `max_uthread * bytes_per_uthread` -- how many bytes of
# scratchpad may be concurrently active across every uthread resident on
# one NDP unit at once, independent of how many total bytes
# _SPAD_CAPACITY_BYTES alone would allow. Found by hand on N=8192's
# FFTRecNear0 (4096 bytes/uthread -- length 256, ping-pong-doubled: see
# fft_plan_core._build_plan's own `bytes_per_uthread = len(buffer_names) *
# scratchpad_stride * 4`, not FFTCodegenPlan.scratchpad_uthread_stride
# alone, which is only one buffer's share): 16 concurrent uthreads (65536
# bytes) finishes in ~90,000 simulated cycles; 30 (122880 bytes) blew
# *past* the simulator's fixed 20,000,000-cycle-per-launch budget for the
# exact same kernel and data. 16 * 4096 = 65536 is the largest
# *confirmed-safe* point on that line, so that's the budget here -- not a
# fitted formula. A flat uthread-*count* cap (always 16) was tried first
# and also fixes this, but then wrongly re-caps kernels whose own
# footprint was never at risk (e.g. N=1024's FFTRecLeaf1 at 64 bytes/
# uthread, fine at 256 concurrent uthreads -- 16384 bytes total), forcing
# them into extra small launches that only add per-launch overhead.
# Capping the byte product instead leaves those uncapped while still
# catching Near0-shaped kernels at any N. See _cap_max_uthread's own
# docstring.
_MAX_CONCURRENT_SCRATCHPAD_BYTES = 16 * 4096


def make_fft_kernel(
    n: int,
    *,
    inverse: bool = False,
    scratchpad_byte_budget: int | None = None,
    simd_lanes: int = 8,
    compute_lanes: int | None = None,
    tile_rows: int | None = None,
    tile_cols: int | None = None,
    cooperative_workers: int | str | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Plan and render a length-`n` FFT kernel, write it to `output_path`
    (default: `fft_fp32_N{n}{_inverse}_generated.mojo` next to this file),
    and return the path written.

    `scratchpad_byte_budget`: how many bytes of one NDP unit's own
    scratchpad one leaf kernel may use -- the planner recurses (adds a
    tiled-transpose boundary) only once N would exceed this as a single
    fused kernel. `None` (the default) picks `4096` regardless of
    `cooperative_workers` -- tried making this cooperation-aware (a
    `16*sqrt(n)`-ish guess, see fft_plan_cooperative.
    default_cooperative_scratchpad_budget's own docstring) and measured it
    against the flat constant at three N: it won at two (N=4096, N=16384)
    but lost by ~26% at a third (N=1024, where keeping the *flat* 4096
    default outright beat every budget tried except one even bigger one
    that hit a register-spill wall) -- no formula found so far is safe to
    default to over just keeping `4096`, so this still does that. Pass a
    number explicitly to try something else for a specific N; see
    default_cooperative_scratchpad_budget's own docstring for the numbers
    behind this and what a real per-N search would need.
    `tile_rows`/`tile_cols`: physical transpose tile size, default
    `min(simd_lanes, ...)` picked by the planner itself (see
    make_recursive_transpose_plan) -- pass explicitly only to compare
    tile-size choices.

    `cooperative_workers`: `None` (the default) keeps every leaf exactly
    what this project produced before cooperative leaves existed (see
    CooperationPlan / fft_plan_cooperative.py). `"auto"` lets
    `choose_workers_per_fft` decide each leaf's own worker count from its
    own shape; a positive int caps that choice instead of overriding it
    outright (see `_build_leaf_kernel`'s own docstring in fft_plan_
    recursive.py for why it's a cap, not a raw override).

    `simd_lanes` is the hardware launch granule (ties to `PooledRange`/
    `VECTOR_WIDTH` in src/m2ndp.mojo -- do not change it to tune register
    pressure, that miscounts how many microthreads get launched and
    silently produces an all-zero result). `compute_lanes` is the
    separate, safe knob for that: how wide a vector *instruction* each
    stage's arithmetic actually emits (see fft_codegen._chunk_batch).
    Defaults to `min(simd_lanes, 4)` -- this target's LMUL=1 width -- since
    a full `simd_lanes=8` butterfly, fully unrolled the way this generator
    always renders one, needs more RVV registers (LMUL=2) than this
    target's simulator can spill through: it emits `csrr t, vlenb` to size
    the scalable-vector spill slot, an instruction the simulator's decoder
    does not implement, and the kernel panics the moment register pressure
    forces a spill. Pass `compute_lanes=simd_lanes` to opt back into the
    old, unchunked (and, on any kernel large enough to spill, simulator-
    incompatible) code shape.
    """
    if n < 2:
        # A length-1 "FFT" needs zero radix stages, which _build_plan/
        # _check_layouts (planning/fft_plan_core.py) has never supported --
        # not something introduced here, and not worth adding a zero-stage
        # passthrough kernel for. Fail clearly instead of surfacing that
        # internal ValueError.
        raise ValueError(f"n={n} is too small: make_fft_kernel needs n >= 2")
    if compute_lanes is None:
        compute_lanes = min(simd_lanes, _LMUL1_FLOAT32_LANES)
    if scratchpad_byte_budget is None:
        scratchpad_byte_budget = 4096
    plan = make_recursive_transpose_plan(
        n,
        scratchpad_byte_budget=scratchpad_byte_budget,
        simd_lanes=simd_lanes,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        inverse=inverse,
        spad_capacity_bytes=_SPAD_CAPACITY_BYTES,
        max_concurrent_scratchpad_bytes=_MAX_CONCURRENT_SCRATCHPAD_BYTES,
        cooperative_workers=cooperative_workers,
    )
    source = generate_recursive_fft_kernels(plan, compute_lanes=compute_lanes)

    if output_path is None:
        suffix = "_inverse" if inverse else ""
        output_path = Path(__file__).resolve().parent / f"fft_fp32_N{n}{suffix}_generated.mojo"
    else:
        output_path = Path(output_path)

    output_path.write_text(source, encoding="utf-8")
    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a working M2NDP FFT Mojo kernel for length N."
    )
    parser.add_argument("n", type=int, help="FFT length (any product of supported radices)")
    parser.add_argument("--inverse", action="store_true", help="generate the inverse FFT")
    parser.add_argument(
        "--scratchpad-byte-budget", type=int, default=None,
        help="bytes of one NDP unit's own scratchpad one leaf kernel may use "
        "(default: 4096, same whether or not --cooperative-workers is given -- "
        "see make_fft_kernel's own docstring on why a cooperation-aware "
        "default isn't safe yet)",
    )
    parser.add_argument(
        "--cooperative-workers", type=str, default=None,
        help='"auto" to let each leaf pick its own cooperative worker count '
        "(see choose_workers_per_fft), or an integer to cap it -- default: "
        "off, one uthread per sub-FFT (this project's original behavior)",
    )
    parser.add_argument("--simd-lanes", type=int, default=8, help="SIMD lane count (default: 8)")
    parser.add_argument(
        "--compute-lanes", type=int, default=None,
        help="SIMD width the emitted arithmetic actually uses, independent of "
        "--simd-lanes (the launch granule -- do not use this to change that). "
        f"Default: min(simd_lanes, {_LMUL1_FLOAT32_LANES}), this target's LMUL=1 "
        "width, to avoid register-spill instructions the simulator can't run. "
        "Pass --compute-lanes matching --simd-lanes to opt back into the old, "
        "unchunked code shape.",
    )
    parser.add_argument("--tile-rows", type=int, default=None, help="transpose tile rows (default: planner's own choice)")
    parser.add_argument("--tile-cols", type=int, default=None, help="transpose tile cols (default: planner's own choice)")
    parser.add_argument("-o", "--output", type=str, default=None, help="output .mojo path")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    cooperative_workers: int | str | None = args.cooperative_workers
    if cooperative_workers is not None and cooperative_workers != "auto":
        cooperative_workers = int(cooperative_workers)
    path = make_fft_kernel(
        args.n,
        inverse=args.inverse,
        scratchpad_byte_budget=args.scratchpad_byte_budget,
        simd_lanes=args.simd_lanes,
        compute_lanes=args.compute_lanes,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
        cooperative_workers=cooperative_workers,
        output_path=args.output,
    )
    print(f"generated: {path}")


if __name__ == "__main__":
    main()
