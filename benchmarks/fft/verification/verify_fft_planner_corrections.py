"""Planner-only regressions against independent pinned GPU host decisions.

No GPU measurement runner is imported. Numeric checks use the IR evaluator
and NumPy; source checks exercise the existing emitter without modifying it.
"""
from dataclasses import replace
import json
from pathlib import Path
import unittest

import numpy as np

from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from planning.core.fft_plan_core import FFTCodegenPlan
from planning.core.target_profile import DEFAULT_TARGET_PROFILE as TARGET
from planning.diagnostics.fft_unit_utilization import compute_unit_utilization
from planning.execution.fft_plan_lanes import (
    apply_all_scalar_lanes_to_plan, apply_compute_lanes_to_plan,
    resolve_compute_lanes_to_plan,
)
from planning.execution.fft_plan_persistent import persistent_tracker_scratchpad_bytes
from planning.gpu_baseline import clfft, rocfft_default, vkfft
from planning.gpu_baseline.common import BaselineStatus, reconstruct_gpu_logical_plan
from planning.gpu_baseline.vkfft_registers import fp32_base_registers
from planning.search.fft_cost_model import estimate_cost, estimate_metrics
from planning.search.fft_plan_search import format_plan_summary, generate_candidates
from planning.strategies.fft_plan_recursive import flatten_recursive_node
from verification.verify_fft_recursive import run_recursive_plan

FIXTURES = Path(__file__).parent / "fixtures"


def kernels(plan):
    return [k for k in flatten_recursive_node(plan.root) if isinstance(k, FFTCodegenPlan)]


class PinnedGPUDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vk = json.loads((FIXTURES / "vkfft_fp32_upstream.json").read_text())
        cls.roc = json.loads((FIXTURES / "rocfft_sbrr_fp32_metadata.json").read_text())

    def test_fp32_base_table_from_original_cpp(self):
        self.assertEqual(len(self.vk["base_register_cases"]), 777)
        for row in self.vk["base_register_cases"]:
            with self.subTest(counts=row["counts"]):
                registers = fp32_base_registers(*row["counts"])
                self.assertEqual([registers[p] for p in (2,3,5,7)], row["registers"])

    def test_radix_order_and_thread_cap_scaling_from_original_cpp(self):
        self.assertEqual(len(self.vk["direct_cases"]), 585)
        for row in self.vk["direct_cases"]:
            with self.subTest(n=row["length"], batch=row["batch"],
                              passes=row["num_passes"], upload=row["upload_id"]):
                p = vkfft._direct_radix_plan(row["length"], row["batch"],
                    num_passes=row["num_passes"], upload_id=row["upload_id"])
                self.assertEqual(list(p.radices), row["radices"])
                if row["batch"] == 1 and row["num_passes"] == 1:
                    self.assertEqual(list(vkfft.direct_radix_sequence(row["length"])),row["radices"])
                self.assertEqual(p.min_registers, row["min_registers"])
                self.assertEqual(p.max_registers, row["max_registers"])
                self.assertEqual([list(pair) for pair in p.registers_per_radix], row["registers_per_radix"])

    def test_logical_workers_and_physical_axis_swap_from_original_cpp(self):
        self.assertEqual(len(self.vk["axis_cases"]), 128)
        for row in self.vk["axis_cases"]:
            with self.subTest(n=row["length"], batch=row["batch"]):
                radices = vkfft.leaf_radix_sequence(row["length"], row["batch"])
                self.assertEqual(list(radices), row["radices"])
                layout = vkfft.axisblock_layout_for_leaf(row["length"], radices,
                    max_rhs=row["batch"], num_passes=1, upload_id=0,
                    original_length=row["length"], target=TARGET)
                self.assertEqual(layout.threads_per_transform, row["workers"])
                self.assertEqual(layout.transforms_per_block, row["slots"])
                self.assertEqual(layout.axis_swapped, row["axis_swapped"])
                axes = (row["slots"], row["workers"]) if row["axis_swapped"] else (row["workers"], row["slots"])
                self.assertEqual(layout.local_size, axes + (1,))
        result = vkfft.plan(128, batch=4)
        [gpu] = reconstruct_gpu_logical_plan(result)
        metadata = dict(gpu.gpu_metadata)
        self.assertEqual((gpu.gpu_threads_per_transform, gpu.gpu_transforms_per_workgroup), (16,4))
        self.assertEqual(metadata["local_size"], (4,16,1))
        self.assertTrue(metadata["axis_swapped"])
        self.assertEqual(result.plan.root.kernel.persistent.workers_per_fft, 16)

    def test_original_composite_butterflies_survive_mapping(self):
        for planner in (vkfft.plan, vkfft.plan_m2ndp):
            for n, expected in ((15,(15,)), (24,(12,2)), (28,(14,2))):
                with self.subTest(planner=planner.__name__, n=n):
                    result = planner(n, batch=3)
                    self.assertEqual(result.status, BaselineStatus.OK)
                    self.assertEqual(tuple(s.radix for s in result.plan.root.kernel.stages), expected)
                    self.assertEqual(result.gpu_config.radices, expected)
                    [gpu] = reconstruct_gpu_logical_plan(result)
                    self.assertEqual(gpu.radix_sequence, expected)

    def test_supported_corrected_mixed_radices_numeric_forward_inverse(self):
        rng = np.random.default_rng(31)
        for n in (20,30,216):
            x = rng.normal(size=3*n) + 1j*rng.normal(size=3*n)
            for inverse in (False,True):
                with self.subTest(n=n, inverse=inverse):
                    result = vkfft.plan(n, batch=3, inverse=inverse)
                    self.assertEqual(result.status, BaselineStatus.OK)
                    expected = (np.fft.ifft if inverse else np.fft.fft)(x.reshape(3,n),axis=1).ravel()
                    np.testing.assert_allclose(run_recursive_plan(result.plan,x,compute_lanes=4),
                                               expected,rtol=1e-5,atol=1e-4)

    def test_complete_rocfft_generic_sbrr_metadata_from_pinned_source(self):
        self.assertEqual(len(self.roc["configs"]),459)
        self.assertEqual(set(rocfft_default.SBRR_TABLE),{r["length"] for r in self.roc["configs"]})
        for row in self.roc["configs"]:
            with self.subTest(n=row["length"]):
                c = rocfft_default.SBRR_TABLE[row["length"]]
                self.assertEqual(c.workgroup_size,row["workgroup_size"])
                self.assertEqual(c.threads_per_transform,row["threads_per_transform"])
                self.assertEqual(c.factors,tuple(row["factors"]))
                self.assertEqual(c.half_lds,row["half_lds"])
                self.assertEqual(c.direct_to_from_reg,row["direct_to_from_reg"])
        for n, half in ((64,False),(128,True),(4096,True)):
            result = rocfft_default.plan(n)
            [gpu] = reconstruct_gpu_logical_plan(result)
            self.assertEqual(dict(gpu.gpu_metadata)["half_lds"],half)
            self.assertEqual(result.gpu_config.extra["half_lds"],half)
            self.assertEqual(gpu.gpu_memory_optimization,"half_lds" if half else None)
            self.assertEqual(gpu.m2ndp_memory_optimization_status,"NO_EQUIVALENT" if half else None)
            self.assertIsNone(gpu.gpu_logical_shared_memory_bytes)

    def test_clfft_block_sizes_and_reused_worker_groups(self):
        # Independently extracted from pinned generator.stockham.cpp BlockSizes
        # and its block-compute pass loop, FP32. WGS is not TPT*block_width.
        expected = {8:(64,128,2,32,4,8192),16:(64,64,4,16,4,8192),
                    32:(64,32,4,16,2,8192),64:(128,16,16,8,2,8192),
                    128:(128,8,16,8,1,8192),256:(256,8,64,4,2,16384)}
        for n, values in expected.items():
            with self.subTest(n=n):
                c = clfft.get_block_config(n)
                self.assertEqual((c.workgroup_size,c.block_width,c.threads_per_transform,
                                  c.active_transforms,c.compute_rounds,c.lds_bytes),values)
        result = clfft.plan(8192)
        gpu = reconstruct_gpu_logical_plan(result)
        self.assertEqual(len(gpu),2)
        self.assertEqual([k.gpu_workgroup_size for k in gpu],[128,128])
        self.assertEqual([k.gpu_threads_per_transform for k in gpu],[16,16])
        self.assertEqual([k.gpu_transforms_per_workgroup for k in gpu],[16,8])
        self.assertEqual([k.gpu_shared_memory_bytes for k in gpu],[8192,8192])
        self.assertEqual([dict(k.gpu_metadata)["block_compute_rounds"] for k in gpu],[2,1])


class NativePlannerCorrectionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates = generate_candidates(64,batch=33,max_candidates=500)

    def test_widths_resolved_before_costing_and_explicit_scalar_retained(self):
        for candidate in self.candidates:
            for kernel in kernels(candidate.plan):
                self.assertTrue(all(s.compute_lanes is not None for s in kernel.stages))
            self.assertEqual(candidate.plan,resolve_compute_lanes_to_plan(candidate.plan,compute_lanes=4))
        scalar = next(c for c in self.candidates if c.choices.lane_variant == "all_scalar")
        self.assertTrue(all(s.compute_lanes == 1 for k in kernels(scalar.plan) for s in k.stages))

    def test_equal_rendered_widths_have_equal_costs(self):
        for n in (30,128,960):
            with self.subTest(n=n):
                original = generate_candidates(n,max_candidates=1)[0].plan
                resolved = apply_compute_lanes_to_plan(original,compute_lanes=4,narrow_middle_stages=True)
                before = estimate_metrics(original,TARGET)
                after = estimate_metrics(resolved,TARGET)
                self.assertEqual(before,after)
                self.assertEqual(estimate_cost(before),estimate_cost(after))
                self.assertEqual(generate_recursive_fft_kernels(original,compute_lanes=4,reference_check=False),
                                 generate_recursive_fft_kernels(resolved,compute_lanes=4,reference_check=False))

    def test_hybrid_rewrites_and_diagnostics_use_both_actual_tasks(self):
        candidate = next(c for c in self.candidates if getattr(c.plan.root,"tail_hybrid",None) is not None)
        original = candidate.plan
        scalar = apply_all_scalar_lanes_to_plan(original)
        self.assertEqual(len(kernels(scalar)),2)
        self.assertTrue(all(s.compute_lanes == 1 for k in kernels(scalar) for s in k.stages))
        self.assertNotEqual(generate_recursive_fft_kernels(original,compute_lanes=4,reference_check=False),
                            generate_recursive_fft_kernels(scalar,compute_lanes=4,reference_check=False))
        usage = compute_unit_utilization(original,TARGET)
        self.assertEqual(len(usage),len(kernels(original)))
        h = original.root.tail_hybrid
        self.assertEqual(usage[0].logical_work_count,h.full_blocks)
        self.assertEqual(usage[0].active_units_worst,TARGET.num_ndp_units)
        self.assertEqual(usage[1].logical_work_count,h.tail_plan.total_uthreads)
        report = format_plan_summary(candidate)
        self.assertIn(h.persistent_plan.kernel_name,report)
        self.assertIn(h.tail_plan.kernel_name,report)
        self.assertEqual(candidate.metrics.max_scratchpad_bytes,1056)
        self.assertIn("estimated_bytes=1056",report)

    def test_default_leaf_budget_respects_small_target(self):
        target = replace(TARGET,spad_capacity_bytes=512,max_concurrent_scratchpad_bytes=512)
        rng = np.random.default_rng(32)
        x = rng.normal(size=2*64) + 1j*rng.normal(size=2*64)
        for inverse in (False,True):
            candidates = generate_candidates(64,batch=2,target=target,inverse=inverse)
            self.assertTrue(candidates)
            for c in candidates:
                for k in kernels(c.plan):
                    used = sum(b.elements*4 for b in k.scratchpad_buffers)
                    if k.persistent is not None:
                        p = k.persistent
                        used += persistent_tracker_scratchpad_bytes(p.workers_per_group,
                            mode=p.lowering_mode,workers_per_fft=p.workers_per_fft)
                    self.assertLessEqual(used,512)
            plan = candidates[0].plan
            self.assertTrue(generate_recursive_fft_kernels(plan,target=target,compute_lanes=4,reference_check=False))
            expected = (np.fft.ifft if inverse else np.fft.fft)(x.reshape(2,64),axis=1).ravel()
            np.testing.assert_allclose(run_recursive_plan(plan,x,compute_lanes=4),expected,rtol=1e-5,atol=1e-4)


if __name__ == "__main__":
    unittest.main()
