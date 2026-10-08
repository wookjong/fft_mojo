"""Check pre-lowering metadata against independent raw planner choices."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.gpu_baseline import clfft, rocfft, rocfft_default, vkfft
from planning.gpu_baseline.common import reconstruct_gpu_logical_plan


def main():
    for n in (2, 16, 32, 64, 256, 1024):
        for batch in (1, 3, 32):
            [k] = reconstruct_gpu_logical_plan(clfft.plan_single_kernel(n, total_ffts=batch))
            c = clfft.get_radices(n)
            assert (k.radix_sequence, k.gpu_workgroup_size, k.gpu_threads_per_transform,
                    k.gpu_transforms_per_workgroup) == (c.radices, c.workgroup_size,
                                                       c.workgroup_size // c.num_transforms,
                                                       c.num_transforms)
    for n in (2048, 4096):
        [k] = reconstruct_gpu_logical_plan(rocfft_default.plan(n))
        c = rocfft_default.SBRR_TABLE[n]
        assert (k.radix_sequence, k.gpu_workgroup_size, k.gpu_threads_per_transform,
                k.gpu_transforms_per_workgroup) == (c.factors, c.workgroup_size,
                                                   c.threads_per_transform,
                                                   c.workgroup_size // c.threads_per_transform)
    for n, batch in ((64, 1), (512, 3)):
        [k] = reconstruct_gpu_logical_plan(vkfft.plan(n, batch=batch))
        radices = vkfft.leaf_radix_sequence(n, batch, num_compute_units=64)
        tpt, tpb = vkfft.axisblock_for_leaf(n, radices, max_rhs=batch, num_passes=1,
                                           upload_id=0, original_length=n, target=DEFAULT_TARGET_PROFILE)
        assert (k.radix_sequence, k.gpu_workgroup_size, k.gpu_threads_per_transform,
                k.gpu_transforms_per_workgroup) == (radices, tpt * tpb, tpt, tpb)
    def fake_bench(plan, target):
        return rocfft.BenchmarkOutcome(ok=True, ndp_cycles=plan.length * 7 + len(plan.stages) * 13,
                                       spill_free=True)
    result, _ = rocfft.tune(24, total_ffts=4, benchmark_fn=fake_bench)
    [k] = reconstruct_gpu_logical_plan(result)
    assert k.gpu_threads_per_transform == result.gpu_config.extra["threads_per_transform"]
    assert k.gpu_transforms_per_workgroup == result.gpu_config.extra["transforms_per_block"]
    kernels = reconstruct_gpu_logical_plan(vkfft.plan_m2ndp(8192))
    assert len(kernels) == 2 and [k.pass_id for k in kernels] == [1, 0]
    assert all(k.gpu_transpose_role is None for k in kernels)
    print("[verify] original GPU execution metadata: all checks passed")


if __name__ == "__main__":
    main()
