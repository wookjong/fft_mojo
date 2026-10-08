"""Execution feasibility: actual distinct launch targets, including tails."""
from dataclasses import replace
import re
import unittest

from codegen.fft_codegen import generate_fft_kernel
from codegen.fft_persistent_codegen import generate_persistent_fft_kernel
from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from planning.core.target_profile import DEFAULT_TARGET_PROFILE as TARGET
from planning.execution.fft_plan_persistent import make_persistent_leaf_plan
from planning.execution.kernel_registration import (
    KernelRegistrationError, registered_kernel_names, validate_kernel_registration,
)
from planning.search.fft_plan_search import generate_candidates
from planning.strategies.fft_plan_recursive import flatten_recursive_node, make_recursive_transpose_plan


def source_names(source, name):
    return set(re.findall(r"launch_parallel\[" + re.escape(name) + r"\.(\w+)\]", source))


class KernelRegistrationTests(unittest.TestCase):
    def test_analysis_matches_plain_and_cooperative_emission(self):
        # A larger cap permits observing the original over-limit source for comparison.
        target = replace(TARGET, max_kernel_register=20)
        for n in (64, 105, 729, 2401):
            for workers in (None, 2, 8):
                plan = make_recursive_transpose_plan(n, scratchpad_byte_budget=120 * 1024,
                                                     cooperative_workers=workers)
                kernel = plan.root.kernel
                for loops in (False, True):
                    source = generate_recursive_fft_kernels(plan, target=target, loop_stages=loops,
                                                           compute_lanes=4, reference_check=False)
                    self.assertEqual(set(registered_kernel_names(kernel, loop_stages=loops)),
                                     source_names(source, kernel.kernel_name))

    def test_n729_is_six_stages_but_nine_functions(self):
        for workers in (None, 2, 8):
            plan = make_recursive_transpose_plan(729, scratchpad_byte_budget=16384,
                                                 cooperative_workers=workers)
            kernel = plan.root.kernel
            self.assertEqual(len(kernel.stages), 6)
            self.assertEqual(len(registered_kernel_names(kernel, loop_stages=True)), 9)
            with self.assertRaisesRegex(KernelRegistrationError, "9 distinct.*max_kernel_register=8"):
                generate_recursive_fft_kernels(plan, compute_lanes=4, reference_check=False)
            # Loop-free emission registers only six functions; no unconditional length ban.
            generate_recursive_fft_kernels(plan, loop_stages=False, compute_lanes=4,
                                           reference_check=False)

    def test_search_prunes_every_over_limit_task(self):
        for batch in (1, 33):
            candidates = generate_candidates(729, scratchpad_byte_budget=16384,
                                             batch=batch, max_candidates=80)
            self.assertTrue(candidates)
            for candidate in candidates:
                source = generate_recursive_fft_kernels(candidate.plan, compute_lanes=4,
                                                         reference_check=False)
                for node in flatten_recursive_node(candidate.plan.root):
                    if hasattr(node, "stages"):
                        self.assertLessEqual(len(source_names(source, node.kernel_name)), 8)

    def test_persistent_rounds_and_waves_do_not_multiply_registrations(self):
        for blocks in (1, 33, 65):
            kernel = make_persistent_leaf_plan(64, (4, 4, 4), num_logical_blocks=blocks,
                                              workers_per_fft=16)
            self.assertEqual(len(registered_kernel_names(kernel)), 5)
            for mode in ("wave", "fused", "physical"):
                source = generate_persistent_fft_kernel(kernel, num_logical_blocks=blocks, mode=mode)
                self.assertEqual(set(registered_kernel_names(kernel)),
                                 source_names(source, kernel.kernel_name))
            with self.assertRaisesRegex(KernelRegistrationError, "max_kernel_register=4"):
                generate_persistent_fft_kernel(kernel, num_logical_blocks=blocks,
                                               target=replace(TARGET, max_kernel_register=4))

    def test_plain_standalone_checks_stage_only_limit(self):
        plan = make_recursive_transpose_plan(512, scratchpad_byte_budget=16384,
                                             allowed_radix_composites=frozenset())
        self.assertEqual(len(plan.root.kernel.stages), 9)
        with self.assertRaises(KernelRegistrationError):
            generate_fft_kernel(plan.root.kernel, compute_lanes=4)

    def test_boundary_and_target_cap(self):
        plan = make_recursive_transpose_plan(256, scratchpad_byte_budget=4096,
                                             allowed_radix_composites=frozenset())
        self.assertEqual(len(validate_kernel_registration(plan.root.kernel)), 8)
        generate_fft_kernel(plan.root.kernel, compute_lanes=4)
        with self.assertRaises(KernelRegistrationError):
            generate_fft_kernel(plan.root.kernel, target=replace(TARGET, max_kernel_register=7))

    def test_gpu_mapping_refuses_over_limit_without_losing_original_choice(self):
        from planning.gpu_baseline.common import (
            GPUKernelConfig, BaselineStatus, map_cooperative_kernel, reconstruct_gpu_logical_plan,
        )
        config = GPUKernelConfig(source="test", length=729, radices=(3,) * 6)
        result = map_cooperative_kernel(
            length=729, radices=(3,) * 6, workers_per_fft=2, fft_slots_wanted=1,
            total_ffts=1, inverse=False, inverse_scale=None, kernel_name="OverLimitGPU",
            target=TARGET, gpu_config=config,
        )
        self.assertEqual(result.status, BaselineStatus.RESOURCE_INFEASIBLE)
        self.assertIsNone(result.plan)
        self.assertIn("9 distinct", result.diagnostics)
        [original] = reconstruct_gpu_logical_plan(result)
        self.assertEqual(original.radix_sequence, (3,) * 6)
        self.assertEqual(original.gpu_threads_per_transform, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
