"""Supported FFT pool launch granules, separate from arithmetic lane counts."""


def validate_fft_launch_granule(simd_lanes: int, *, spread_across_units: bool) -> None:
    if not isinstance(simd_lanes, int) or isinstance(simd_lanes, bool) or simd_lanes < 1:
        raise ValueError("simd_lanes must be a positive integer launch granule")
    if spread_across_units and simd_lanes != 8:
        raise ValueError(
            "spread_across_units requires simd_lanes=8 (32-byte launch granule); "
            f"got simd_lanes={simd_lanes}. For 4-lane arithmetic use simd_lanes=8 "
            "with compute_lanes=4. Non-spreading experiments require "
            "spread_across_units=False (--no-spread-across-units)."
        )
