from __future__ import annotations

"""GOLDEN fidelity tests for the GPU-derived FFT baselines (Phase 8 of the
2026-09-08 baseline-completion/freeze task).

Unlike verify_gpu_baseline.py's own tests (which check individual helper
functions -- `Factorize`, `choose_pow2_grouping_radix`, etc.), every test
here asserts the COMPLETE planning result for a representative length --
scheme/decomposition, exact radix sequence, WGS/TPT/TPB-equivalent
fields, and OK-vs-refusal status -- exactly as a real caller of
`planning.gpu_baseline.{clfft,rocfft_default,vkfft}.plan()` would observe
it. The purpose (the task's own words): "The test must fail if future
M2NDP research accidentally modifies the GPU baseline behavior."

Golden values below were captured directly from this repository's own
`gpu-baseline-v1` implementation on 2026-09-08 (see docs/
gpu_baseline_v1_freeze.md) -- they are DELIBERATELY not re-derived by hand
here; if a value looks surprising, that surprise is itself the point: any
future change to a GOLDEN value must come with an explicit, documented,
GPU-source-fidelity justification (a real upstream research finding), the
same discipline every fix in this baseline's own history already follows
-- never an M2NDP-performance-driven change (see gpu_baseline/common.py's
own non-negotiable-rule docstring).

Deliberately excludes measured M2NDP cycles/spill status entirely (the
task's own instruction) -- every field here is a pure planning-time
decision, never a real-hardware measurement.

Run directly: `python3 verify_gpu_baseline_golden.py` from benchmarks/fft/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.gpu_baseline import clfft, rocfft_default, vkfft

_FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        _FAILURES.append(message)
        print(f"  FAIL: {message}")


REQUIRED_LENGTHS = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

# ============================================================================
# clFFT -- golden (status, radices, workgroup_size, num_transforms) per
# length. clfft.plan_single_kernel is used directly (not the large-1D
# `plan()` dispatcher) since every required length is single-kernel-
# eligible (Is1DPossible(n, 4096) is true for all of 2..4096).
# ============================================================================
CLFFT_GOLDEN: dict[int, dict[str, object]] = {
    2: dict(status="ok", radices=(2,), workgroup_size=64, num_transforms=64),
    4: dict(status="ok", radices=(2, 2), workgroup_size=64, num_transforms=32),
    8: dict(status="ok", radices=(4, 2), workgroup_size=64, num_transforms=32),
    16: dict(status="ok", radices=(4, 4), workgroup_size=64, num_transforms=16),
    32: dict(status="ok", radices=(8, 4), workgroup_size=64, num_transforms=16),
    64: dict(status="unsupported_current_codegen", radices=(4, 4, 4), workgroup_size=64, num_transforms=4),
    128: dict(status="unsupported_current_codegen", radices=(8, 4, 4), workgroup_size=64, num_transforms=4),
    256: dict(status="unsupported_current_codegen", radices=(4, 4, 4, 4), workgroup_size=64, num_transforms=1),
    512: dict(status="unsupported_current_codegen", radices=(8, 8, 8), workgroup_size=64, num_transforms=1),
    1024: dict(status="unsupported_current_codegen", radices=(8, 8, 4, 4), workgroup_size=128, num_transforms=1),
    2048: dict(status="unsupported_current_codegen", radices=(8, 8, 8, 4), workgroup_size=256, num_transforms=1),
    4096: dict(status="unsupported_current_codegen", radices=(8, 8, 8, 8), workgroup_size=256, num_transforms=1),
}


def verify_clfft_golden() -> None:
    print("GOLDEN: clFFT complete planning result, N=2..4096")
    for n in REQUIRED_LENGTHS:
        golden = CLFFT_GOLDEN[n]
        result = clfft.plan_single_kernel(n, total_ffts=4)
        check(result.status.value == golden["status"], f"clfft N={n}: status={result.status.value}, golden={golden['status']}")
        check(result.gpu_config.radices == golden["radices"], f"clfft N={n}: radices={result.gpu_config.radices}, golden={golden['radices']}")
        check(
            result.gpu_config.extra.get("workgroup_size") == golden["workgroup_size"]
            and result.gpu_config.extra.get("num_transforms") == golden["num_transforms"],
            f"clfft N={n}: wgs/nt={result.gpu_config.extra.get('workgroup_size')}/"
            f"{result.gpu_config.extra.get('num_transforms')}, golden={golden['workgroup_size']}/{golden['num_transforms']}",
        )


# ============================================================================
# rocFFT-default -- golden (scheme, workgroup_size, threads_per_transform,
# factors) per length. `decide_scheme` used directly (batch=1, avoiding
# the >4096 occupancy branch, irrelevant for this required-length range).
# ============================================================================
ROCFFT_DEFAULT_GOLDEN: dict[int, dict[str, object]] = {
    2: dict(scheme="CS_KERNEL_STOCKHAM", workgroup_size=64, threads_per_transform=1, factors=(2,)),
    4: dict(scheme="CS_KERNEL_STOCKHAM", workgroup_size=128, threads_per_transform=1, factors=(4,)),
    8: dict(scheme="CS_KERNEL_STOCKHAM", workgroup_size=64, threads_per_transform=4, factors=(4, 2)),
    16: dict(scheme="CS_KERNEL_STOCKHAM", workgroup_size=64, threads_per_transform=4, factors=(4, 4)),
    32: dict(scheme="CS_KERNEL_STOCKHAM", workgroup_size=128, threads_per_transform=16, factors=(8, 4)),
    64: dict(scheme="CS_KERNEL_STOCKHAM", workgroup_size=64, threads_per_transform=16, factors=(4, 4, 4)),
    128: dict(scheme="CS_KERNEL_STOCKHAM", workgroup_size=256, threads_per_transform=16, factors=(16, 8)),
    256: dict(scheme="CS_KERNEL_STOCKHAM", workgroup_size=64, threads_per_transform=64, factors=(4, 4, 4, 4)),
    512: dict(scheme="CS_KERNEL_STOCKHAM", workgroup_size=64, threads_per_transform=64, factors=(8, 8, 8)),
    1024: dict(scheme="CS_KERNEL_STOCKHAM", workgroup_size=128, threads_per_transform=128, factors=(8, 8, 4, 4)),
    2048: dict(scheme="CS_KERNEL_STOCKHAM", workgroup_size=256, threads_per_transform=256, factors=(16, 16, 8)),
    4096: dict(scheme="CS_KERNEL_STOCKHAM", workgroup_size=256, threads_per_transform=256, factors=(16, 16, 16)),
}


def verify_rocfft_default_golden() -> None:
    print("GOLDEN: rocFFT-default complete planning result, N=2..4096")
    for n in REQUIRED_LENGTHS:
        golden = ROCFFT_DEFAULT_GOLDEN[n]
        d = rocfft_default.decide_scheme(n, batch=1)
        check(d.scheme == golden["scheme"], f"rocfft-default N={n}: scheme={d.scheme}, golden={golden['scheme']}")
        sk = d.single_kernel
        check(sk is not None, f"rocfft-default N={n}: expected a single_kernel config")
        if sk is not None:
            check(
                sk.workgroup_size == golden["workgroup_size"] and sk.threads_per_transform == golden["threads_per_transform"],
                f"rocfft-default N={n}: wgs/tpt={sk.workgroup_size}/{sk.threads_per_transform}, "
                f"golden={golden['workgroup_size']}/{golden['threads_per_transform']}",
            )
            check(sk.factors == golden["factors"], f"rocfft-default N={n}: factors={sk.factors}, golden={golden['factors']}")


# ============================================================================
# VkFFT -- golden (status, radices, num_passes, extra worker/batch fields)
# per length.
# ============================================================================
VKFFT_GOLDEN: dict[int, dict[str, object]] = {
    2: dict(status="ok", radices=(2,), num_passes=1),
    4: dict(status="ok", radices=(4,), num_passes=1),
    8: dict(status="ok", radices=(8,), num_passes=1),
    16: dict(status="ok", radices=(4, 4), num_passes=1),
    32: dict(status="ok", radices=(8, 4), num_passes=1),
    64: dict(status="ok", radices=(8, 8), num_passes=1),
    # 2026-09-09 source-fidelity re-audit: N=128 flipped from refused to OK
    # after porting VkFFTSplitAxisBlock's real max_rhs cap (line 329) and
    # axisBlock[0]<->axisBlock[1] swap (lines 350-364) -- both previously
    # entirely unported (see vkfft.py's own `_postprocess_axis_upload0`).
    # The cap (batch=8 > max_rhs=4) then the swap together produce
    # workers_per_fft=4/transforms_per_block=16 instead of the old,
    # never-capped/never-swapped 16/8 -- which M2NDP's own hardware-mapping
    # constraints happen to accept where the old pair did not. radices are
    # unaffected (same leaf_radix_sequence result either way).
    128: dict(status="ok", radices=(8, 8, 2), num_passes=1),
    256: dict(status="unsupported_current_codegen", radices=(8, 8, 4), workers_per_fft=32, transforms_per_block=1),
    512: dict(status="unsupported_current_codegen", radices=(8, 8, 8), workers_per_fft=64, transforms_per_block=1),
    1024: dict(status="unsupported_current_codegen", radices=(8, 8, 8, 2), workers_per_fft=128, transforms_per_block=1),
    2048: dict(status="unsupported_current_codegen", radices=(8, 8, 8, 4), workers_per_fft=256, transforms_per_block=1),
    4096: dict(status="unsupported_current_codegen", radices=(8, 8, 8, 8), workers_per_fft=512, transforms_per_block=1),
}


def verify_vkfft_golden() -> None:
    print("GOLDEN: VkFFT complete planning result, N=2..4096")
    for n in REQUIRED_LENGTHS:
        golden = VKFFT_GOLDEN[n]
        result = vkfft.plan(n, batch=4)
        check(result.status.value == golden["status"], f"vkfft N={n}: status={result.status.value}, golden={golden['status']}")
        check(result.gpu_config.radices == golden["radices"], f"vkfft N={n}: radices={result.gpu_config.radices}, golden={golden['radices']}")
        if golden["status"] == "ok":
            check(
                result.gpu_config.extra.get("num_passes") == golden["num_passes"],
                f"vkfft N={n}: num_passes={result.gpu_config.extra.get('num_passes')}, golden={golden['num_passes']}",
            )
        else:
            check(
                result.gpu_config.extra.get("workers_per_fft") == golden["workers_per_fft"]
                and result.gpu_config.extra.get("transforms_per_block") == golden["transforms_per_block"],
                f"vkfft N={n}: workers/tpb={result.gpu_config.extra.get('workers_per_fft')}/"
                f"{result.gpu_config.extra.get('transforms_per_block')}, golden={golden['workers_per_fft']}/{golden['transforms_per_block']}",
            )


# ============================================================================
# rocFFT-tuned: the search itself is non-deterministic in the sense that it
# depends on an injected `benchmark_fn` (real hardware measurement in
# production), so a full golden PLAN result isn't meaningful here the same
# way -- instead, golden-test the deterministic CANDIDATE-GENERATION shape
# (which is exactly what the task's own Phase 8 asks golden tests to
# protect: decomposition/radix-sequence/WGS/TPT/TPB fields, computed
# identically regardless of which candidate a benchmark eventually picks).
# ============================================================================
def verify_rocfft_tuned_golden() -> None:
    from planning.gpu_baseline import rocfft

    # 2026-09-09 source-fidelity re-audit: these golden numbers were
    # recomputed after `_supported_kernel_configs` was rewritten to port
    # SupportedKernelConfigs (tuning_kernel_tuner.cpp) literally -- adding
    # the `tpt < wgs` guard (line 552) and the min_wgs 64-rounding (line
    # 491) it was missing, and rescoping the tpbs_to_remove/bad-utilization/
    # "largest half of TPTs" pruning to the WHOLE phase-0 call instead of
    # per-ordering (see that function's own docstring). Cross-checked via
    # an independent from-scratch re-transliteration of the same pinned
    # source (not derived from this production code) that reproduces the
    # same count and factor-set shape for N=24 (and N=8/16/64/336/1024).
    # The (4, 6) factor multiset -- present in the pre-fix golden value --
    # is now correctly pruned entirely: with pruning scoped globally across
    # every N=24 factorization at once, (4,6)'s own surviving TPTs land
    # among the globally-largest half removed by the phase-0 "largest half
    # of TPTs" step, which a per-ordering-scoped view could not detect.
    print("GOLDEN: rocFFT-tuned deterministic candidate-generation shape, N=24")
    configs = rocfft.phase0_candidates(24)
    check(len(configs) == 105, f"rocfft.phase0_candidates(24) should produce exactly 105 configs, got {len(configs)}")
    factor_sets = sorted({tuple(sorted(c.factors)) for c in configs})
    check(
        factor_sets == [(2, 2, 2, 3), (2, 2, 6), (2, 3, 4), (3, 8)],
        f"rocfft.phase0_candidates(24) factor multisets = {factor_sets}",
    )


def main() -> None:
    verify_clfft_golden()
    verify_rocfft_default_golden()
    verify_vkfft_golden()
    verify_rocfft_tuned_golden()

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S):")
        for f in _FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL GOLDEN CHECKS PASSED")


if __name__ == "__main__":
    main()
