"""Regression checks for audit findings 3/4: GPU IR and lowering overhead."""
from dataclasses import FrozenInstanceError, asdict, replace
import unittest

from planning.core.target_profile import DEFAULT_TARGET_PROFILE as TARGET
from planning.gpu_baseline import clfft, rocfft_default, vkfft
from planning.gpu_baseline.common import (
    BaselineStatus, compute_lowering_overhead, format_gpu_m2ndp_execution_report,
    reconstruct_gpu_logical_plan, reconstruct_m2ndp_physical_plan,
)


class OriginalGPUPlanTests(unittest.TestCase):
    def test_fused_kernel_count_and_batch_traffic(self):
        for planner in (clfft.plan, rocfft_default.plan, vkfft.plan_m2ndp):
            for batch in (1, 3):
                with self.subTest(planner=planner.__name__, batch=batch):
                    r = planner(8192, batch=batch)
                    self.assertEqual(r.status, BaselineStatus.OK)
                    o = compute_lowering_overhead(r)
                    self.assertEqual((o.gpu_kernel_count, o.physical_plan_nodes,
                                      o.added_plan_nodes, o.added_transpose_nodes), (2, 5, 3, 3))
                    self.assertEqual(o.gpu_full_array_bytes, 2 * 8192 * batch * 16)
                    self.assertEqual(o.m2ndp_full_array_bytes, 5 * 8192 * batch * 16)
                    self.assertEqual(o.additional_full_array_bytes, 3 * 8192 * batch * 16)
                    self.assertEqual(o.full_array_traffic_ratio, 2.5)

    def test_raw_metadata_survives_slot_clipping_and_persistence(self):
        for n in (2, 16, 64, 1024):
            for batch in (1, 3):
                r = clfft.plan(n, batch=batch)
                [gpu] = reconstruct_gpu_logical_plan(r)
                [physical] = reconstruct_m2ndp_physical_plan(r, target=TARGET)
                c = clfft.get_radices(n)
                self.assertEqual(gpu.gpu_workgroup_size, c.workgroup_size)
                self.assertEqual(gpu.gpu_transforms_per_workgroup, c.num_transforms)
                self.assertEqual(gpu.gpu_threads_per_transform, c.workgroup_size // c.num_transforms)
                self.assertLessEqual(physical.fft_slots_per_group, c.num_transforms)
                self.assertIsNone(gpu.gpu_shared_memory_bytes)
                self.assertGreaterEqual(physical.scratchpad_bytes, 0)

    def test_native_transposes_are_not_counted_as_unfusing_overhead(self):
        r = rocfft_default.plan(524288)
        self.assertEqual(r.status, BaselineStatus.OK)
        gpu = reconstruct_gpu_logical_plan(r)
        self.assertEqual([k.gpu_transpose_role for k in gpu], ["PRE", None, "MIDDLE", None, "POST"])
        o = compute_lowering_overhead(r)
        self.assertEqual((o.gpu_kernel_count, o.physical_plan_nodes), (5, 5))
        self.assertEqual((o.added_plan_nodes, o.added_transpose_nodes, o.additional_full_array_bytes), (0, 0, 0))
        self.assertEqual(o.full_array_traffic_ratio, 1.0)

    def test_clfft_large_row_keeps_its_separate_post_transpose(self):
        r = clfft.plan(131072)
        self.assertEqual(r.status, BaselineStatus.OK)
        gpu = reconstruct_gpu_logical_plan(r)
        self.assertEqual(len(gpu), 3)
        self.assertEqual(gpu[-1].gpu_transpose_role, "POST")
        self.assertEqual(gpu[1].radix_sequence, clfft.get_radices(2048).radices)
        o = compute_lowering_overhead(r)
        self.assertEqual((o.added_plan_nodes, o.added_transpose_nodes), (2, 2))
        self.assertAlmostEqual(o.full_array_traffic_ratio, 5 / 3)

    def test_snapshot_independent_of_physical_tree(self):
        r = rocfft_default.plan(8192)
        original = reconstruct_gpu_logical_plan(r)
        self.assertEqual(reconstruct_gpu_logical_plan(replace(r, plan=None)), original)
        unrelated_tree = clfft.plan(16).plan
        self.assertEqual(reconstruct_gpu_logical_plan(replace(r, plan=unrelated_tree)), original)
        with self.assertRaises(FrozenInstanceError):
            original[0].gpu_workgroup_size = 1

    def test_original_plan_retained_on_mapping_refusal(self):
        # 8192's single-upload GPU decision survives insufficient M2NDP scratchpad.
        r = vkfft.plan(8192)
        self.assertEqual(r.status, BaselineStatus.OK)
        self.assertEqual(r.plan.root.kernel.persistent.storage_mode, "global")
        [gpu] = reconstruct_gpu_logical_plan(r)
        self.assertEqual(gpu.fft_length, 8192)
        self.assertIsNotNone(gpu.gpu_workgroup_size)
        self.assertIn("KERNEL 0", format_gpu_m2ndp_execution_report(r, target=TARGET))
        self.assertEqual(compute_lowering_overhead(r).full_array_traffic_ratio,
                         len(r.plan.root.kernel.stages))

    def test_three_uploads_are_not_nine_gpu_kernels(self):
        t = replace(TARGET, spad_capacity_bytes=4096)
        r = vkfft.plan_m2ndp(32768, target=t)
        self.assertEqual(r.status, BaselineStatus.OK)
        gpu = reconstruct_gpu_logical_plan(r)
        self.assertEqual([k.pass_id for k in gpu], [2, 1, 0])
        factors = r.gpu_config.extra["factors"]
        for k in gpu:
            from math import prod
            self.assertEqual(dict(k.gpu_metadata)["stage_start_size"], prod(factors[:k.pass_id]))
        o = compute_lowering_overhead(r)
        self.assertEqual((o.gpu_kernel_count, o.physical_plan_nodes), (3, 9))
        self.assertEqual(o.full_array_traffic_ratio, 3.0)

    def test_report_independent_sequences_and_unknown_fields(self):
        r = clfft.plan(8192)
        report = format_gpu_m2ndp_execution_report(r, target=TARGET)
        gpu_section, physical_section = report.split("-- M2NDP PHYSICAL EXECUTION --")
        self.assertEqual(gpu_section.count("KERNEL "), 2)
        self.assertEqual(physical_section.count("NODE "), 5)
        self.assertNotIn("PRE transpose", gpu_section)
        self.assertIn("PRE transpose", physical_section)
        self.assertIn("gpu_workgroup_size:            128", gpu_section)
        self.assertIn("not measured traffic/cycles", report)
        physical = reconstruct_m2ndp_physical_plan(r, target=TARGET)
        self.assertEqual([k.physical_node_id for k in physical], list(range(5)))

    def test_missing_capture_never_fabricates_zero_gpu_traffic(self):
        r = clfft.plan(16)
        extra = dict(r.gpu_config.extra, original_gpu_plan=(), gpu_plan_capture_error="not extracted")
        r = replace(r, gpu_config=replace(r.gpu_config, extra=extra))
        with self.assertRaises(ValueError):
            compute_lowering_overhead(r)
        report = format_gpu_m2ndp_execution_report(r, target=TARGET)
        self.assertIn("original_gpu_plan: unavailable", report)
        self.assertNotIn("full_array_traffic_ratio:", report)

    def test_benchmark_exports_independent_accounting(self):
        from gpu_vs_m2ndp_benchmark import CSV_FIELDS, run_one
        r = run_one("gpu-clfft", 8192, TARGET, batch=1, inverse=False,
                    skip_toolchain=True, mojo_root=None, m2ndp_root=None,
                    build_timeout=30, run_timeout=30, seed=1)
        self.assertEqual((r.gpu_kernel_count, r.physical_plan_nodes), (2, 5))
        self.assertEqual(r.full_array_traffic_ratio, 2.5)
        self.assertEqual(r.measured_total_cycles, "not_available")
        for key in asdict(compute_lowering_overhead(clfft.plan(8192))):
            self.assertIn(key, CSV_FIELDS)
        refused = run_one("gpu-vkfft", 8192, TARGET, batch=1, inverse=False,
                          skip_toolchain=True, mojo_root=None, m2ndp_root=None,
                          build_timeout=30, run_timeout=30, seed=1)
        self.assertEqual(refused.gpu_kernel_count, 1)
        self.assertEqual(refused.gpu_full_array_bytes, 8192 * 16)
        self.assertEqual(refused.physical_plan_nodes, 1)
        self.assertEqual(refused.full_array_traffic_ratio,
                         len(vkfft.plan(8192).plan.root.kernel.stages))


if __name__ == "__main__":
    unittest.main(verbosity=2)
