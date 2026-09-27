from __future__ import annotations

"""Semantic verification for `gpu_baseline.common.reconstruct_gpu_logical_
plan` (section 11 of the task this suite was built from -- "Step 2":
establishing an exact GPU-logical vs. M2NDP-physical execution boundary).

Does NOT test string formatting -- every check here cross-references the
reconstructed `GPULogicalKernelPlan` against the SAME baseline module's
own raw planning function (`clfft.get_radices`, `rocfft_default.
SBRR_TABLE`, `vkfft.axisblock_for_leaf`, `rocfft.tune`'s own winning
`KernelConfig`), independently of the plan tree the reconstruction reads
from -- so a bug that made the reconstruction merely SELF-consistent
(agreeing with the plan tree it was built from, but not with what the GPU
planner actually decided) would still be caught.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.gpu_baseline import clfft, rocfft, rocfft_default, vkfft
from planning.gpu_baseline.common import BaselineStatus, reconstruct_gpu_logical_plan

_T = DEFAULT_TARGET_PROFILE


def check_clfft_single_kernel_matches_get_radices() -> None:
    """clFFT single-kernel: `gpu_threads_per_transform` must equal
    `get_radices(length)`'s own `workgroup_size // num_transforms` -- the
    raw table/DetermineSizes answer, read independently of the plan tree
    -- for EVERY length, cooperative-mapped or persistent-mapped alike
    (this field is always recoverable, see `GPULogicalKernelPlan`'s own
    docstring).

    `gpu_transforms_per_workgroup` is checked only for the lengths that
    map to M2NDP's COOPERATIVE path (`workers_per_fft` a divisor of the
    physical lane count, N=16/32 here -- see clfft.get_radices): there it
    must equal `get_radices`'s own `num_transforms` exactly (`total_
    ffts=32` keeps `CooperationPlan.fft_slots_per_group` from being capped
    below that -- N=16/32's own `num_transforms=16` is the largest among
    the cooperative-path lengths tested here -- see `make_cooperative_
    leaf_plan`'s own `min(total_ffts, capacity)` realization). For the lengths that map to the PERSISTENT
    path instead (N=64/256/1024, `workers_per_fft` not a divisor of 8),
    `gpu_transforms_per_workgroup` must be `None` -- the GPU's own `num_
    transforms` choice is discarded before the M2NDP plan is built at all
    (see `reconstruct_gpu_logical_plan`'s own comment on that branch) and
    is NOT recoverable, so this checks the reconstruction correctly
    reports "not recoverable" rather than fabricating a number."""
    cooperative_lengths = (16, 32)
    persistent_lengths = (64, 256, 1024)
    for length in cooperative_lengths + persistent_lengths:
        result = clfft.plan_single_kernel(length, total_ffts=32)
        assert result.status is BaselineStatus.OK
        choice = clfft.get_radices(length)
        [kernel] = reconstruct_gpu_logical_plan(result)
        assert kernel.gpu_threads_per_transform == choice.workgroup_size // choice.num_transforms, (
            f"N={length}: gpu_threads_per_transform={kernel.gpu_threads_per_transform} != "
            f"get_radices's own workgroup_size//num_transforms="
            f"{choice.workgroup_size // choice.num_transforms}"
        )
        if length in cooperative_lengths:
            assert result.plan.root.kernel.cooperation is not None, f"N={length} expected cooperative path"
            assert kernel.gpu_transforms_per_workgroup == choice.num_transforms, (
                f"N={length}: gpu_transforms_per_workgroup={kernel.gpu_transforms_per_workgroup} != "
                f"get_radices's own num_transforms={choice.num_transforms}"
            )
            assert kernel.gpu_workgroup_size == choice.workgroup_size
        else:
            assert result.plan.root.kernel.persistent is not None, f"N={length} expected persistent path"
            assert kernel.gpu_transforms_per_workgroup is None, (
                f"N={length}: expected gpu_transforms_per_workgroup=None (not recoverable on the "
                f"persistent path), got {kernel.gpu_transforms_per_workgroup}"
            )
            assert kernel.gpu_workgroup_size is None
        assert kernel.radix_sequence == choice.radices
        assert kernel.fft_length == length
        assert kernel.gpu_transpose_role is None
    print(f"    OK   clFFT single-kernel: gpu_threads_per_transform matches get_radices() at every "
          f"N in {cooperative_lengths + persistent_lengths}; gpu_transforms_per_workgroup matches "
          f"exactly on the cooperative path ({cooperative_lengths}) and is honestly None on the "
          f"persistent path ({persistent_lengths}, where it is genuinely not recoverable)")


def check_rocfft_default_matches_sbrr_table() -> None:
    """N=2048/4096 both have `threads_per_transform=256`, which does not
    divide the physical lane count -- both map to M2NDP's persistent
    path, so `gpu_workgroup_size`/`gpu_transforms_per_workgroup` are
    honestly `None` (see `check_clfft_single_kernel_matches_get_radices`'s
    own docstring for why); `gpu_threads_per_transform` (always
    recoverable) and `radix_sequence` are checked directly against
    `SBRR_TABLE`."""
    for length in (2048, 4096):
        result = rocfft_default.plan(length)
        assert result.status is BaselineStatus.OK
        assert result.plan.root.kernel.persistent is not None, f"N={length} expected persistent path"
        config = rocfft_default.SBRR_TABLE[length]
        [kernel] = reconstruct_gpu_logical_plan(result)
        assert kernel.gpu_threads_per_transform == config.threads_per_transform, (
            f"N={length}: gpu_threads_per_transform={kernel.gpu_threads_per_transform} != "
            f"SBRR_TABLE's own threads_per_transform={config.threads_per_transform}"
        )
        assert kernel.gpu_workgroup_size is None
        assert kernel.radix_sequence == config.factors
    print("    OK   rocFFT-default: gpu_threads_per_transform matches SBRR_TABLE directly at "
          "N in (2048, 4096) (both persistent-path; gpu_workgroup_size honestly None)")


def check_vkfft_matches_axisblock_for_leaf() -> None:
    """VkFFT single-pass leaf: `gpu_threads_per_transform` must equal
    `axisblock_for_leaf`'s own returned `threads_per_transform` for the
    SAME length/radices/max_rhs -- read independently via a fresh call,
    not the value stashed on the plan tree. `gpu_transforms_per_workgroup`
    is checked against `axisblock_for_leaf`'s own `transforms_per_block`
    ONLY on the cooperative path (N=64/batch=1, tpt=8 divides the physical
    lane count); on the persistent path (N=512/batch=3, tpt=64 does not),
    it must be `None` -- same "not recoverable once worker-wave
    virtualization applies" reason as clFFT's own check above."""
    result = vkfft.plan(64, batch=1)
    assert result.status is BaselineStatus.OK
    assert result.plan.root.kernel.cooperation is not None
    [kernel] = reconstruct_gpu_logical_plan(result)
    radices = vkfft.leaf_radix_sequence(64, 1, num_compute_units=64)
    expected_tpt, expected_tpb = vkfft.axisblock_for_leaf(
        64, radices, max_rhs=1, num_passes=1, upload_id=0, original_length=64, target=_T,
    )
    assert kernel.gpu_threads_per_transform == expected_tpt
    assert kernel.gpu_transforms_per_workgroup == expected_tpb
    assert kernel.radix_sequence == radices

    result2 = vkfft.plan(512, batch=3)
    assert result2.status is BaselineStatus.OK
    assert result2.plan.root.kernel.persistent is not None
    [kernel2] = reconstruct_gpu_logical_plan(result2)
    radices2 = vkfft.leaf_radix_sequence(512, 3, num_compute_units=64)
    expected_tpt2, _expected_tpb2 = vkfft.axisblock_for_leaf(
        512, radices2, max_rhs=3, num_passes=1, upload_id=0, original_length=512, target=_T,
    )
    assert kernel2.gpu_threads_per_transform == expected_tpt2
    assert kernel2.gpu_transforms_per_workgroup is None, (
        f"expected None on the persistent path (tpt={expected_tpt2} does not divide the "
        f"physical lane count), got {kernel2.gpu_transforms_per_workgroup}"
    )
    print("    OK   VkFFT: gpu_threads_per_transform matches axisblock_for_leaf() on both the "
          "cooperative path (N=64,batch=1) and the persistent path (N=512,batch=3); "
          "gpu_transforms_per_workgroup matches exactly on the cooperative path and is "
          "honestly None on the persistent path")


def check_rocfft_tuned_matches_winning_config() -> None:
    """rocFFT (offline-tuner baseline): gpu_threads_per_transform/gpu_
    transforms_per_workgroup must equal the WINNING `KernelConfig`'s own
    threads_per_transform/transforms_per_block -- read back from the
    `BaselineResult.gpu_config.extra` this baseline's own `_map_config`
    already stores (independent of the reconstruction's own tree walk,
    which reads `CooperationPlan` instead)."""
    def fake_bench(plan, target):
        return rocfft.BenchmarkOutcome(
            ok=True, ndp_cycles=plan.length * 7 + len(plan.stages) * 13, spill_free=True,
        )

    result, _outcomes = rocfft.tune(24, total_ffts=4, benchmark_fn=fake_bench)
    assert result.status is BaselineStatus.OK
    [kernel] = reconstruct_gpu_logical_plan(result)
    winning_tpt = result.gpu_config.extra["threads_per_transform"]
    winning_tpb = result.gpu_config.extra["transforms_per_block"]
    assert kernel.gpu_threads_per_transform == winning_tpt, (
        f"gpu_threads_per_transform={kernel.gpu_threads_per_transform} != the winning "
        f"KernelConfig's own threads_per_transform={winning_tpt} (stored in gpu_config.extra "
        f"by rocfft.py's own _map_config)"
    )
    # N=24's own winning threads_per_transform=3 does not divide the
    # physical lane count -- persistent path, gpu_transforms_per_workgroup
    # honestly None (winning_tpb, the GPU's own real intent, is still
    # visible in result.gpu_config.extra -- just not recoverable via the
    # plan-tree reconstruction; see check_clfft_single_kernel_matches_
    # get_radices's own docstring for why).
    if result.plan.root.kernel.persistent is not None:
        assert kernel.gpu_transforms_per_workgroup is None
    else:
        assert kernel.gpu_transforms_per_workgroup == winning_tpb
    print(f"    OK   rocFFT (tuned): gpu_threads_per_transform={winning_tpt} matches the winning "
          f"KernelConfig stored independently in gpu_config.extra (transforms_per_block="
          f"{winning_tpb} in extra, persistent-path so honestly None in the reconstruction)")


def check_transpose_role_tagging() -> None:
    """clFFT/VkFFT large-1D/multi-pass plans: PRE/MIDDLE/POST transpose
    kernels must be tagged in the SAME order the plan tree actually
    chains them (PRE, near_fft, MIDDLE, far_child, POST), and no FFT
    kernel is ever mistagged as a transpose or vice versa."""
    result = vkfft.plan_m2ndp(8192, target=_T)
    assert result.status is BaselineStatus.OK
    kernels = reconstruct_gpu_logical_plan(result)
    roles = [k.gpu_transpose_role for k in kernels]
    assert roles == ["PRE", None, "MIDDLE", None, "POST"], f"unexpected role sequence: {roles}"
    for k in kernels:
        if k.gpu_transpose_role is not None:
            assert k.radix_sequence == () and k.gpu_threads_per_transform is None
        else:
            assert k.radix_sequence != () and k.gpu_threads_per_transform is not None
    print(f"    OK   vkfft.plan_m2ndp(8192): transpose-role sequence is exactly "
          f"['PRE', FFT, 'MIDDLE', FFT, 'POST'], no cross-tagging")


def main() -> None:
    print("  GPU execution semantics reconstruction (cross-checked against each baseline's own raw planner):")
    check_clfft_single_kernel_matches_get_radices()
    check_rocfft_default_matches_sbrr_table()
    check_vkfft_matches_axisblock_for_leaf()
    check_rocfft_tuned_matches_winning_config()
    check_transpose_role_tagging()
    print("[verify] GPU execution semantics: all checks passed")


if __name__ == "__main__":
    main()
