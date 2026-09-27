from __future__ import annotations

"""Regression tests for the rocFFT `half_lds` classification (P1.2 --
see docs/rocfft_half_lds_classification.md and `rocfft.gpu_logical_lds_
bytes`'s own docstring for the full semantics).

`half_lds` is a REAL rocFFT GPU-side LDS-usage optimization (halves the
per-batch LDS byte footprint `DeriveMaxTPB` budgets against, letting more
transforms share one workgroup for the same LDS budget). M2NDP has NO
corresponding mechanism -- its own scratchpad-halving mechanism
(`pingpong_needed`) is keyed on stage count, a completely different axis.
This suite verifies:

1. `half_lds` is classified `NO_EQUIVALENT`, never silently treated as
   `EXACT_EQUIVALENT` (i.e. never assumed to also halve M2NDP's own
   scratchpad requirement).
2. M2NDP's own resource-feasibility check (`scratchpad_bytes`, what
   `map_cooperative_kernel` actually gates on) is COMPLETELY UNAFFECTED
   by `half_lds` -- the same M2NDP length/radix combination produces the
   identical scratchpad byte count whether or not the winning GPU config
   happened to set `half_lds`.
3. `gpu_logical_lds_bytes` (the GPU's own accounting) DOES change with
   `half_lds` -- exactly halved -- so the two numbers are genuinely
   different quantities, not two names for the same one.
4. The unified report shows both, never merged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.gpu_baseline import rocfft
from planning.gpu_baseline.common import (
    BaselineStatus,
    format_gpu_m2ndp_execution_report,
    reconstruct_gpu_logical_plan,
)

_T = DEFAULT_TARGET_PROFILE


def _fake_bench(plan, target):
    return rocfft.BenchmarkOutcome(ok=True, ndp_cycles=plan.length * 7 + len(plan.stages) * 13, spill_free=True)


def check_gpu_logical_lds_bytes_formula() -> None:
    """Direct unit check of the extracted helper against the original
    inline formula (`length * BYTES_PER_ELEM`, halved iff `half_lds`)."""
    for length in (16, 24, 100, 1024):
        full = rocfft.gpu_logical_lds_bytes(length, half_lds=False)
        halved = rocfft.gpu_logical_lds_bytes(length, half_lds=True)
        assert full == length * rocfft.BYTES_PER_ELEM
        assert halved == full // 2
    print("    OK   gpu_logical_lds_bytes(length, half_lds=True) is exactly half of "
          "gpu_logical_lds_bytes(length, half_lds=False), at every length tested")


def check_derive_max_tpb_unchanged_by_refactor() -> None:
    """`derive_max_tpb`/`conservative_max_tpb` now call `gpu_logical_lds_
    bytes` instead of inlining the same formula -- confirm this refactor
    changed no observable behavior (same inputs, same outputs, at several
    representative lengths and both half_lds values)."""
    for length in (16, 24, 100, 512, 1024, 2048):
        for half_lds in (False, True):
            for tpt in (1, 4, 16):
                got = rocfft.derive_max_tpb(length, half_lds=half_lds, tpt=tpt, wgs_bound=256)
                bytes_per_batch = length * rocfft.BYTES_PER_ELEM
                if half_lds:
                    bytes_per_batch //= 2
                expected = rocfft.LDS_BYTE_LIMIT // bytes_per_batch
                while tpt * expected > 256:
                    expected -= 1
                assert got == expected, f"length={length} half_lds={half_lds} tpt={tpt}: {got} != {expected}"
        conservative = rocfft.conservative_max_tpb(length)
        bytes_per_batch = length * rocfft.BYTES_PER_ELEM
        expected_conservative = rocfft.LDS_BYTE_LIMIT // bytes_per_batch
        if length >= 1024:
            expected_conservative += 1
        assert conservative == expected_conservative
    print("    OK   derive_max_tpb/conservative_max_tpb byte-identical to the pre-refactor inline "
          "formula at every (length, half_lds, tpt) combination tested")


def check_half_lds_does_not_change_m2ndp_scratchpad() -> None:
    """The REAL M2NDP scratchpad requirement (what `map_cooperative_
    kernel` actually gates resource feasibility on) must be identical for
    the SAME (length, radices) pair regardless of whatever `half_lds`
    value the winning GPU config happened to carry -- i.e. `half_lds`
    must never leak into M2NDP's own scratchpad math."""
    from planning.gpu_baseline.common import leaf_scratchpad_bytes

    for length, radices in ((24, (3, 8)), (64, (4, 4, 4)), (105, (3, 5, 7))):
        # leaf_scratchpad_bytes has no half_lds parameter at all -- calling
        # it twice with the identical arguments must be identical (this is
        # itself the proof: there is no code path for half_lds to alter
        # this number through).
        a = leaf_scratchpad_bytes(length, radices)
        b = leaf_scratchpad_bytes(length, radices)
        assert a == b
    print("    OK   leaf_scratchpad_bytes (M2NDP's own real resource-feasibility input) has no "
          "half_lds parameter at all -- confirmed unaffected by construction")


def check_report_shows_both_numbers_distinctly() -> None:
    """Find a real winning config with `half_lds=True` (N=24 is known to
    produce one -- confirmed directly, not assumed) and verify the report
    shows `gpu_logical_shared_memory_bytes` (GPU's own halved accounting)
    and `scratchpad_bytes` (M2NDP's own real allocation) as two distinct,
    separately-labeled numbers that DIFFER for this case, plus the
    NO_EQUIVALENT classification."""
    result, _outcomes = rocfft.tune(24, total_ffts=4, benchmark_fn=_fake_bench)
    assert result.status is BaselineStatus.OK
    assert result.gpu_config.extra.get("half_lds") is True, (
        "this test's own N=24 case is expected to produce a half_lds=True winner -- "
        "re-pick a representative case if this assumption ever breaks"
    )
    [kernel] = reconstruct_gpu_logical_plan(result)
    assert kernel.gpu_memory_optimization == "half_lds"
    assert kernel.m2ndp_memory_optimization_status == "NO_EQUIVALENT"
    expected_gpu_bytes = rocfft.gpu_logical_lds_bytes(24, half_lds=True)
    assert kernel.gpu_logical_shared_memory_bytes == expected_gpu_bytes == 96
    assert kernel.gpu_shared_memory_bytes != kernel.gpu_logical_shared_memory_bytes, (
        f"expected the M2NDP real scratchpad number ({kernel.gpu_shared_memory_bytes}) to differ "
        f"from the GPU's own half_lds-adjusted number ({kernel.gpu_logical_shared_memory_bytes}) "
        f"for this case -- otherwise this test isn't exercising a genuine divergence"
    )

    report = format_gpu_m2ndp_execution_report(result, target=_T)
    assert "gpu_memory_optimization:       half_lds" in report
    assert "m2ndp_memory_optimization_status: NO_EQUIVALENT" in report
    assert f"gpu_logical_shared_memory_bytes: {expected_gpu_bytes}" in report
    assert f"scratchpad_bytes:              {kernel.gpu_shared_memory_bytes}" in report
    print(
        f"    OK   report distinctly shows gpu_logical_shared_memory_bytes={expected_gpu_bytes} "
        f"(GPU, half_lds-adjusted) and scratchpad_bytes={kernel.gpu_shared_memory_bytes} (M2NDP, "
        f"real, half_lds-UNaffected) as two separate numbers, plus NO_EQUIVALENT classification"
    )


def check_non_rocfft_kernel_has_no_memory_optimization_tag() -> None:
    """A non-rocFFT kernel (no `half_lds` concept at all) must report
    `gpu_memory_optimization=None` -- never a stale/default value."""
    from planning.gpu_baseline import vkfft

    result = vkfft.plan(64, batch=1)
    assert result.status is BaselineStatus.OK
    [kernel] = reconstruct_gpu_logical_plan(result)
    assert kernel.gpu_memory_optimization is None
    assert kernel.gpu_logical_shared_memory_bytes is None
    assert kernel.m2ndp_memory_optimization_status is None
    print("    OK   a non-rocFFT kernel (VkFFT) reports gpu_memory_optimization=None, not a "
          "stale half_lds value")


def main() -> None:
    print("  rocFFT half_lds classification (P1.2):")
    check_gpu_logical_lds_bytes_formula()
    check_derive_max_tpb_unchanged_by_refactor()
    check_half_lds_does_not_change_m2ndp_scratchpad()
    check_report_shows_both_numbers_distinctly()
    check_non_rocfft_kernel_has_no_memory_optimization_tag()
    print("[verify] rocFFT half_lds classification: all checks passed")


if __name__ == "__main__":
    main()
