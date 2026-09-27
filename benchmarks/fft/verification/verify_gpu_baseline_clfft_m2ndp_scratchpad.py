from __future__ import annotations

"""Regression tests for the clFFT M2NDP-adapted scratchpad resource model
(P1.1 -- see docs/clfft_m2ndp_scratchpad_resource_model.md and
`gpu_baseline.clfft.get_max_1d_length_m2ndp`'s own docstring for the full
mechanism). Mirrors `verify_gpu_baseline_vkfft_m2ndp_scratchpad.py`'s own
structure exactly -- same class of fix, same verification shape.

Root cause: `clfft.get_max_1d_length_m2ndp` (the single-kernel-vs-large-1D
split DECISION threshold for the M2NDP-adapted baseline) divided by
`CLFFT_ELEM_BYTES=8` (real clFFT's own single-buffer LDS convention) even
though the M2NDP leaf that decision selects is, for almost every length,
actually lowered via the persistent/worker-wave path (`gpu_baseline.
common._map_worker_wave_kernel`), which needs 16 bytes per element
UNCONDITIONALLY. Before this fix, `clfft.plan_m2ndp` could judge a length
single-kernel-feasible by the 8-byte convention and only discover the
mismatch late, inside `map_cooperative_kernel`/`_map_worker_wave_kernel`'s
own resource check (`RESOURCE_INFEASIBLE`).

N=8192 against this project's own `DEFAULT_TARGET_PROFILE.spad_capacity_
bytes=122880` is this suite's own headline regression case -- the same N
the VkFFT scratchpad fix (Step 1) used, chosen again here deliberately so
the two fixes are directly comparable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.execution.fft_plan_persistent import persistent_leaf_scratchpad_bytes
from planning.gpu_baseline import clfft
from planning.gpu_baseline.common import BaselineStatus, m2ndp_resource_complex_bytes
from verification.verify_fft_recursive import run_recursive_plan

_N = 8192
_T = DEFAULT_TARGET_PROFILE


def check_scratchpad_constant_agrees() -> None:
    assert m2ndp_resource_complex_bytes() == persistent_leaf_scratchpad_bytes(1) == 16
    print("    OK   m2ndp_resource_complex_bytes()=16 (shared with the VkFFT fix, single-sourced "
          "from persistent_leaf_scratchpad_bytes)")


def check_n8192_before_after_threshold() -> None:
    assert _T.spad_capacity_bytes == 122880, (
        f"this regression's own numbers assume spad_capacity_bytes=122880, got "
        f"{_T.spad_capacity_bytes} -- DEFAULT_TARGET_PROFILE changed; re-derive"
    )
    old_threshold = clfft.get_max_1d_length(lds_bytes=_T.spad_capacity_bytes, elem_bytes=8)
    assert old_threshold == 8192
    assert clfft.is_1d_possible(_N, old_threshold), (
        f"expected the 8-byte convention to (wrongly, for M2NDP) judge N={_N} "
        f"single-kernel-feasible at threshold={old_threshold}"
    )

    real_requirement = persistent_leaf_scratchpad_bytes(_N)
    assert real_requirement == 16 * _N == 131072
    assert real_requirement > _T.spad_capacity_bytes

    new_threshold = clfft.get_max_1d_length_m2ndp(_T)
    assert new_threshold == 4096
    assert not clfft.is_1d_possible(_N, new_threshold), (
        f"expected the corrected 16-byte convention to require the large-1D split for "
        f"N={_N}, got single-kernel-feasible at threshold={new_threshold}"
    )
    print(
        f"    OK   N={_N}: 8-byte convention -> threshold={old_threshold} (looks single-kernel-"
        f"feasible) but real M2NDP leaf needs {real_requirement} > {_T.spad_capacity_bytes} bytes; "
        f"16-byte-corrected convention -> threshold={new_threshold} (correctly routes into the "
        f"large-1D split up front)"
    )


def check_frozen_plan_unaffected() -> None:
    """`clfft.plan(N=8192)` (frozen, source-faithful) must still use
    `CLFFT_LDS_BYTES` (the real clFFT representative-GPU constant, NEVER
    `target`-derived) for its own single-kernel-vs-large-1D decision --
    unchanged before and after this fix, since `plan()` never calls
    `get_max_1d_length_m2ndp` at all."""
    result = clfft.plan(_N, target=_T)
    assert result.status is BaselineStatus.OK
    frozen_threshold = clfft.get_max_1d_length()
    assert frozen_threshold == clfft.get_max_1d_length(lds_bytes=clfft.CLFFT_LDS_BYTES, elem_bytes=8)
    print(
        f"    OK   frozen clfft.plan(N={_N}) unaffected: its own threshold (get_max_1d_length()) "
        f"still uses CLFFT_LDS_BYTES/CLFFT_ELEM_BYTES=8 unconditionally, independent of `target` "
        f"and of this fix"
    )


def check_m2ndp_adapted_plan_now_feasible() -> None:
    result = clfft.plan_m2ndp(_N, target=_T)
    assert result.status is BaselineStatus.OK, (
        f"expected clfft.plan_m2ndp(N={_N}) to now build an OK plan (avoiding the infeasible "
        f"single-kernel attempt up front), got {result.status}: {result.diagnostics[:500]}"
    )
    assert result.gpu_config.extra.get("decomposition") == "large1D_4step"
    recursive_plan = result.plan
    assert recursive_plan is not None
    rng = np.random.default_rng(1234)
    x = rng.uniform(-1, 1, _N) + 1j * rng.uniform(-1, 1, _N)
    got = run_recursive_plan(recursive_plan, x)
    expected = np.fft.fft(x)
    max_err = float(np.max(np.abs(got - expected)))
    assert max_err < 1e-3, f"clfft.plan_m2ndp(N={_N}): numerically wrong, max_err={max_err}"
    print(
        f"    OK   clfft.plan_m2ndp(N={_N}) now builds an OK, numerically correct plan "
        f"(decomposition=large1D_4step, max_err={max_err:.3e}) -- no more late "
        f"RESOURCE_INFEASIBLE mismatch"
    )


def check_frozen_baseline_regression_at_other_lengths() -> None:
    """`get_max_1d_length`'s own default (`elem_bytes` omitted) must be
    byte-identical to explicitly passing `elem_bytes=CLFFT_ELEM_BYTES` --
    confirms adding the parameter did not change any existing caller's
    behavior, at several representative lengths."""
    for n in (1024, 4096, 8192, 65536):
        implicit = clfft.plan(n, target=_T)
        explicit_threshold = clfft.get_max_1d_length(elem_bytes=clfft.CLFFT_ELEM_BYTES)
        default_threshold = clfft.get_max_1d_length()
        assert explicit_threshold == default_threshold
        assert implicit.status in (BaselineStatus.OK, BaselineStatus.UNSUPPORTED_CURRENT_CODEGEN)
    print(
        "    OK   clfft.get_max_1d_length()'s own default (elem_bytes omitted) == explicit "
        "elem_bytes=CLFFT_ELEM_BYTES=8 -- adding this parameter did not change the frozen "
        "baseline's default behavior"
    )


def main() -> None:
    print("  clFFT M2NDP-adapted scratchpad resource model (N=8192 regression, P1.1):")
    check_scratchpad_constant_agrees()
    check_n8192_before_after_threshold()
    check_frozen_plan_unaffected()
    check_m2ndp_adapted_plan_now_feasible()
    check_frozen_baseline_regression_at_other_lengths()
    print("[verify] clFFT M2NDP-adapted scratchpad resource model: all checks passed")


if __name__ == "__main__":
    main()
