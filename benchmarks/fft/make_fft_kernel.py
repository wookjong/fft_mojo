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


def make_fft_kernel(
    n: int,
    *,
    inverse: bool = False,
    scratchpad_byte_budget: int = 4096,
    simd_lanes: int = 8,
    tile_rows: int | None = None,
    tile_cols: int | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Plan and render a length-`n` FFT kernel, write it to `output_path`
    (default: `fft_fp32_N{n}{_inverse}_generated.mojo` next to this file),
    and return the path written.

    `scratchpad_byte_budget`: how many bytes of one NDP unit's own
    scratchpad one leaf kernel may use -- the planner recurses (adds a
    tiled-transpose boundary) only once N would exceed this as a single
    fused kernel. `tile_rows`/`tile_cols`: physical transpose tile size,
    default `min(simd_lanes, ...)` picked by the planner itself (see
    make_recursive_transpose_plan) -- pass explicitly only to compare
    tile-size choices.
    """
    if n < 2:
        # A length-1 "FFT" needs zero radix stages, which _build_plan/
        # _check_layouts (planning/fft_plan_core.py) has never supported --
        # not something introduced here, and not worth adding a zero-stage
        # passthrough kernel for. Fail clearly instead of surfacing that
        # internal ValueError.
        raise ValueError(f"n={n} is too small: make_fft_kernel needs n >= 2")
    plan = make_recursive_transpose_plan(
        n,
        scratchpad_byte_budget=scratchpad_byte_budget,
        simd_lanes=simd_lanes,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        inverse=inverse,
    )
    source = generate_recursive_fft_kernels(plan)

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
        "--scratchpad-byte-budget", type=int, default=4096,
        help="bytes of one NDP unit's own scratchpad one leaf kernel may use (default: 4096)",
    )
    parser.add_argument("--simd-lanes", type=int, default=8, help="SIMD lane count (default: 8)")
    parser.add_argument("--tile-rows", type=int, default=None, help="transpose tile rows (default: planner's own choice)")
    parser.add_argument("--tile-cols", type=int, default=None, help="transpose tile cols (default: planner's own choice)")
    parser.add_argument("-o", "--output", type=str, default=None, help="output .mojo path")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    path = make_fft_kernel(
        args.n,
        inverse=args.inverse,
        scratchpad_byte_budget=args.scratchpad_byte_budget,
        simd_lanes=args.simd_lanes,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
        output_path=args.output,
    )
    print(f"generated: {path}")


if __name__ == "__main__":
    main()
