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
from planning.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile


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
    target: TargetProfile = DEFAULT_TARGET_PROFILE,
) -> Path:
    """Plan and render a length-`n` FFT kernel, write it to `output_path`
    (default: `fft_fp32_N{n}{_inverse}_generated.mojo` next to this file),
    and return the path written.

    `target`: the hardware/simulator constants this generator otherwise has
    no single source for (scratchpad capacity, LMUL=1 vector width, the
    interleave chunk cooperative leaves must divide evenly) -- see
    `TargetProfile`'s own docstring. Defaults to `DEFAULT_TARGET_PROFILE`,
    this repo's one real target today; pass a different profile only to
    plan against a different config.

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
    Defaults to `min(simd_lanes, target.lmul1_float32_lanes)` -- this target's
    LMUL=1 width -- since
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
        compute_lanes = min(simd_lanes, target.lmul1_float32_lanes)
    if scratchpad_byte_budget is None:
        scratchpad_byte_budget = 4096
    plan = make_recursive_transpose_plan(
        n,
        scratchpad_byte_budget=scratchpad_byte_budget,
        simd_lanes=simd_lanes,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        inverse=inverse,
        spad_capacity_bytes=target.spad_capacity_bytes,
        max_concurrent_scratchpad_bytes=target.max_concurrent_scratchpad_bytes,
        cooperative_workers=cooperative_workers,
        interleave_chunk_uthreads=target.interleave_chunk_uthreads,
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
        f"Default: min(simd_lanes, {DEFAULT_TARGET_PROFILE.lmul1_float32_lanes}), this target's LMUL=1 "
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
