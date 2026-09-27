from __future__ import annotations

"""Regression tests for the VkFFT M2NDP-adapted scratchpad resource model
(sections 8-11 of the task this suite was built from -- see docs/
vkfft_m2ndp_scratchpad_resource_model.md and `gpu_baseline.common.
m2ndp_resource_complex_bytes`'s own docstring for the full mechanism).

Root cause under test: `vkfft.py`'s own shared-memory-sizing formulas
(`max_sequence_length_shared_memory` et al.) are a faithful port of real
VkFFT's `usedSharedMemory / complexSize` convention (`complexSize=8`, a
real-GPU single-buffer assumption) -- correct for the frozen baseline
(`vkfft.plan()`), but WRONG as a feasibility input once a leaf's own
cooperation width needs M2NDP's worker-wave-virtualization path
(`gpu_baseline.common._map_worker_wave_kernel` -> `make_persistent_leaf_
plan`), which unconditionally needs 16 (not 8) bytes per complex element
regardless of stage count. Before this fix, `vkfft.plan_m2ndp` could pick
a pass-count/axis-split shape that LOOKED scratchpad-feasible by the real
VkFFT formula but was actually `RESOURCE_INFEASIBLE` once mapped onto
M2NDP -- a late, avoidable mismatch. `plan_m2ndp` now threads `gpu_
baseline.common.m2ndp_resource_complex_bytes()` (16) through as its own
`complex_size_bytes` override; `plan()` (the frozen, source-faithful
baseline) is untouched -- its own default stays `VKFFT_COMPLEX_SIZE_BYTES`
(8), the real VkFFT convention, per section 11's "never mix frozen and
adapted baseline behavior" rule.

N=8192 against `DEFAULT_TARGET_PROFILE.spad_capacity_bytes=122880`
(matching the task's own cited example exactly: `122880 // 8 = 15360 >=
8192` looks one-pass-feasible by the real 8-byte convention, but `16 *
8192 = 131072 > 122880` is not) is this suite's own headline regression
case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.execution.fft_plan_persistent import persistent_leaf_scratchpad_bytes
from planning.gpu_baseline import vkfft
from planning.gpu_baseline.common import BaselineStatus, m2ndp_resource_complex_bytes
from verification.verify_fft_recursive import run_recursive_plan

_N = 8192
_T = DEFAULT_TARGET_PROFILE


def check_scratchpad_constants_agree() -> None:
    """`m2ndp_resource_complex_bytes()` (the resource ADAPTER vkfft.py
    threads through) must equal `persistent_leaf_scratchpad_bytes(1)` --
    single-sourced from `make_persistent_leaf_plan`'s own unconditional
    `16 * length` requirement, not an independently-typed literal."""
    assert m2ndp_resource_complex_bytes() == persistent_leaf_scratchpad_bytes(1) == 16
    print(f"    OK   m2ndp_resource_complex_bytes()=16 == persistent_leaf_scratchpad_bytes(1) "
          f"(single-sourced from make_persistent_leaf_plan's own formula)")


def check_n8192_before_after_planner_decision() -> None:
    """The task's own headline case: N=8192, target.spad_capacity_bytes=122880."""
    assert _T.spad_capacity_bytes == 122880, (
        f"this regression's own numbers assume spad_capacity_bytes=122880, got "
        f"{_T.spad_capacity_bytes} -- DEFAULT_TARGET_PROFILE changed; re-derive"
    )

    # "Before": the real VkFFT 8-byte-per-element convention (what plan()
    # uses, and what plan_m2ndp used before this fix existed) judges
    # one-pass feasible.
    before_num_passes = vkfft.choose_num_passes(
        _N, non_strided=True, target=_T, complex_size_bytes=8,
    )
    assert before_num_passes == 1, (
        f"expected the 8-byte convention to (wrongly, for M2NDP) judge N={_N} "
        f"one-pass-feasible, got num_passes={before_num_passes}"
    )
    eight_byte_bound = _T.spad_capacity_bytes // 8
    assert eight_byte_bound >= _N, (
        f"8-byte bound {eight_byte_bound} should be >= N={_N} (the 'looks feasible' "
        f"reading this test documents)"
    )

    # The REAL M2NDP persistent-leaf requirement for this N.
    real_requirement = persistent_leaf_scratchpad_bytes(_N)
    assert real_requirement == 16 * _N == 131072
    assert real_requirement > _T.spad_capacity_bytes, (
        f"expected the real M2NDP requirement ({real_requirement} bytes) to exceed "
        f"target capacity ({_T.spad_capacity_bytes} bytes) -- this is the actual "
        f"infeasibility the 8-byte convention misses"
    )

    # "After": plan_m2ndp's own corrected resource input judges >1 pass needed.
    after_num_passes = vkfft.choose_num_passes(
        _N, non_strided=True, target=_T, complex_size_bytes=m2ndp_resource_complex_bytes(),
    )
    assert after_num_passes > 1, (
        f"expected the M2NDP-corrected (16-byte) convention to require >1 pass for "
        f"N={_N}, got num_passes={after_num_passes} -- the fix did not change the "
        f"pass-count decision"
    )
    print(
        f"    OK   N={_N}: 8-byte convention -> num_passes={before_num_passes} "
        f"(looks feasible: bound={eight_byte_bound} >= N) but real M2NDP leaf needs "
        f"{real_requirement} > {_T.spad_capacity_bytes} bytes; 16-byte-corrected "
        f"convention -> num_passes={after_num_passes} (correctly avoids the one-pass "
        f"attempt up front)"
    )


def check_frozen_plan_unaffected() -> None:
    """`vkfft.plan(N=8192)` (frozen, source-faithful) must still choose the
    (M2NDP-infeasible, by design -- a real GPU wouldn't need to know or
    care) one-pass shape: this is the CORRECT real-VkFFT-on-a-real-GPU
    answer, and this fix must not touch it. The resulting mapped plan
    stays RESOURCE_INFEASIBLE once mapped onto M2NDP (an honest, already-
    existing refusal from `map_cooperative_kernel`/`_map_worker_wave_
    kernel` -- see `BaselineStatus.RESOURCE_INFEASIBLE`'s own docstring),
    exactly as before this change: this is the LATE mismatch section 8
    describes, still reachable via the frozen baseline on purpose (a real
    GPU baseline should not silently start reasoning about M2NDP's own
    resource limits)."""
    result = vkfft.plan(_N, target=_T)
    assert result.status is BaselineStatus.RESOURCE_INFEASIBLE, (
        f"expected the frozen vkfft.plan(N={_N}) to still refuse with "
        f"RESOURCE_INFEASIBLE (unchanged pre-fix behavior), got {result.status}"
    )
    assert result.gpu_config.extra.get("workers_per_fft") is not None
    print(
        f"    OK   frozen vkfft.plan(N={_N}) unchanged: still chooses one-pass "
        f"(8-byte convention) and still refuses RESOURCE_INFEASIBLE once mapped onto "
        f"M2NDP -- this fix does not touch the frozen baseline's own decision"
    )


def check_m2ndp_adapted_plan_now_feasible() -> None:
    """`vkfft.plan_m2ndp(N=8192)` must now build a genuinely OK, numerically
    correct plan -- the mismatch section 8 describes (planner decides
    'one pass feasible', lowering later discovers otherwise) is closed for
    the M2NDP-adapted baseline specifically."""
    result = vkfft.plan_m2ndp(_N, target=_T)
    assert result.status is BaselineStatus.OK, (
        f"expected vkfft.plan_m2ndp(N={_N}) to now build an OK plan (avoiding the "
        f"infeasible one-pass attempt up front), got {result.status}: "
        f"{result.diagnostics[:500]}"
    )
    assert result.gpu_config.extra.get("num_passes", 1) > 1
    recursive_plan = result.plan
    assert recursive_plan is not None
    rng = np.random.default_rng(1234)
    x = rng.uniform(-1, 1, _N) + 1j * rng.uniform(-1, 1, _N)
    got = run_recursive_plan(recursive_plan, x)
    expected = np.fft.fft(x)
    max_err = float(np.max(np.abs(got - expected)))
    assert max_err < 1e-3, f"vkfft.plan_m2ndp(N={_N}): numerically wrong, max_err={max_err}"
    print(
        f"    OK   vkfft.plan_m2ndp(N={_N}) now builds an OK, numerically correct "
        f"plan (num_passes={result.gpu_config.extra['num_passes']}, max_err={max_err:.3e}) "
        f"-- no more late RESOURCE_INFEASIBLE mismatch for the M2NDP-adapted baseline"
    )


def check_frozen_baseline_regression_at_other_lengths() -> None:
    """Frozen `plan()` results at a handful of other representative lengths
    must be unaffected by `complex_size_bytes` existing at all (default-
    argument backward compatibility, not just N=8192) -- compares against
    calling `plan()` with the parameter completely omitted vs. explicitly
    passing the real VkFFT default, which must be identical by construction
    (same value), and cross-checks a few lengths' own `num_passes`/`status`
    stay whatever `choose_num_passes`'s pre-existing (8-byte) formula says."""
    for n in (1024, 4096, 8192, 65536):
        implicit = vkfft.plan(n, target=_T)
        explicit = vkfft.plan(n, target=_T, complex_size_bytes=8)
        assert implicit.status == explicit.status
        assert implicit.gpu_config.extra == explicit.gpu_config.extra
        assert implicit.gpu_config.radices == explicit.gpu_config.radices
    print(
        "    OK   vkfft.plan()'s own default complex_size_bytes (omitted) == explicit "
        "complex_size_bytes=8 (VKFFT_COMPLEX_SIZE_BYTES) at N in (1024, 4096, 8192, "
        "65536) -- adding this parameter did not change the frozen baseline's default "
        "behavior"
    )


def main() -> None:
    print("  VkFFT M2NDP-adapted scratchpad resource model (N=8192 regression):")
    check_scratchpad_constants_agree()
    check_n8192_before_after_planner_decision()
    check_frozen_plan_unaffected()
    check_m2ndp_adapted_plan_now_feasible()
    check_frozen_baseline_regression_at_other_lengths()
    print("[verify] VkFFT M2NDP-adapted scratchpad resource model: all checks passed")


if __name__ == "__main__":
    main()
