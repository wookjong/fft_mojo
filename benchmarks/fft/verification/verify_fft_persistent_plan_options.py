"""Default rendering follows persistent plan options; overrides remain explicit."""
from dataclasses import replace
import re
import unittest
from unittest.mock import patch
import numpy as np
from codegen.fft_persistent_codegen import generate_persistent_fft_kernel, generate_persistent_tail_hybrid_kernel
from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from planning.execution.fft_plan_persistent import make_persistent_leaf_plan, make_persistent_tail_hybrid_plan
from planning.strategies.fft_plan_recursive import make_recursive_transpose_plan
from planning.diagnostics.spill_probe import probe_spill_free
from verification.verify_fft_recursive import run_recursive_plan
from verification.verify_fft_persistent import run_persistent_kernel
from verification.verify_fft_harness import Ptr


def launch_count(source, name):
    return len(re.findall(r"launch_parallel\[" + re.escape(name) + r"\.stage_\d+\]", source))


def task(source, name):
    return source.split("struct " + name + "(NDPTask):", 1)[1].split("\nstruct ", 1)[0]


class PersistentPlanOptionsTests(unittest.TestCase):
    def leaf(self, mode="wave", copy="vectorized_contiguous", blocks=33, inverse=False):
        return make_persistent_leaf_plan(64, (4, 4, 4), num_logical_blocks=blocks,
                    workers_per_fft=16, lowering_mode=mode, copy_mode=copy, inverse=inverse)

    def test_standalone_default_and_independent_overrides(self):
        plan = self.leaf()
        def render(**kwargs):
            return generate_persistent_fft_kernel(plan, num_logical_blocks=33, **kwargs)
        default = render()
        self.assertEqual(default, render(mode="wave", copy_mode="vectorized_contiguous"))
        self.assertIn("wave_tracker =", default)
        self.assertIn("var chunk_start", default)
        self.assertEqual(launch_count(default, plan.kernel_name), 12)
        mode_override = render(mode="physical")
        self.assertNotIn("wave_tracker =", mode_override)
        self.assertIn("var chunk_start", mode_override)
        self.assertEqual(launch_count(mode_override, plan.kernel_name), 6)
        copy_override = render(copy_mode="scalar")
        self.assertIn("wave_tracker =", copy_override)
        self.assertNotIn("var chunk_start", copy_override)
        self.assertEqual(plan.persistent.lowering_mode, "wave")
        self.assertEqual(plan.persistent.copy_mode, "vectorized_contiguous")

    def mixed(self, inverse=False):
        plan = make_recursive_transpose_plan(960, scratchpad_byte_budget=512,
                                            persistent_leaf=True, inverse=inverse)
        root = plan.root
        near = root.near_fft.kernel
        far = root.far_child.kernel
        near = make_persistent_leaf_plan(near.length, tuple(s.radix for s in near.stages),
            num_logical_blocks=near.host.total_elems // near.length, kernel_name=near.kernel_name,
            workers_per_fft=16, lowering_mode="wave", copy_mode="vectorized_contiguous",
            inverse=inverse, inverse_scale=None)
        far = replace(far, persistent=replace(far.persistent, lowering_mode="fused", copy_mode="scalar"))
        return replace(plan, root=replace(root, near_fft=replace(root.near_fft, kernel=near),
                                         far_child=replace(root.far_child, kernel=far)))

    def test_recursive_per_leaf_and_global_overrides(self):
        plan = self.mixed()
        near = plan.root.near_fft.kernel.kernel_name
        far = plan.root.far_child.kernel.kernel_name
        source = generate_recursive_fft_kernels(plan, compute_lanes=4)
        self.assertIn("wave_tracker =", task(source, near))
        self.assertIn("var chunk_start", task(source, near))
        self.assertNotIn("wave_tracker =", task(source, far))
        self.assertNotIn("var chunk_start", task(source, far))
        override = generate_recursive_fft_kernels(plan, compute_lanes=4,
                            persistent_mode="physical", persistent_copy_mode="vectorized_contiguous",
                            persistent_vector_width=2)
        for name in (near, far):
            self.assertNotIn("wave_tracker =", task(override, name))
            self.assertIn("var chunk_start", task(override, name))
            self.assertIn("while i + 2 <= chunk_end", task(override, name))

    def test_numeric_standalone_all_modes_and_copies(self):
        rng = np.random.default_rng(4)
        blocks = 33
        x = rng.normal(size=64*blocks) + 1j*rng.normal(size=64*blocks)
        for inverse in (False, True):
            expected = (np.fft.ifft if inverse else np.fft.fft)(x.reshape(blocks, 64), axis=1).ravel()
            for mode in ("wave", "fused", "physical"):
                for copy in ("scalar", "vectorized_contiguous"):
                    plan = self.leaf(mode, copy, blocks=blocks, inverse=inverse)
                    inputs = [Ptr(len(x)) for _ in range(4)]
                    inputs[0].arr[:] = x.real
                    inputs[1].arr[:] = x.imag
                    run_persistent_kernel(plan, num_logical_blocks=blocks,
                        input_real=inputs[0], input_imag=inputs[1], output_real=inputs[2], output_imag=inputs[3])
                    actual = inputs[2].arr + 1j*inputs[3].arr
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_numeric_mixed_defaults_and_overrides(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=960) + 1j*rng.normal(size=960)
        for inverse in (False, True):
            plan = self.mixed(inverse)
            expected = (np.fft.ifft if inverse else np.fft.fft)(x)
            for options in ({}, dict(persistent_mode="physical", persistent_copy_mode="vectorized_contiguous")):
                np.testing.assert_allclose(run_recursive_plan(plan, x, compute_lanes=4, **options),
                                           expected, rtol=1e-5, atol=1e-4)

    def test_hybrid_inherits_plan_options(self):
        for strategy in ("all_persistent", "cooperative_tail", "noncoop_tail"):
            hybrid = make_persistent_tail_hybrid_plan(64, (4, 4, 4), num_logical_blocks=35,
                workers_per_fft=16, lowering_mode="wave", copy_mode="vectorized_contiguous", tail_strategy=strategy)
            self.assertEqual(generate_persistent_tail_hybrid_kernel(hybrid),
                generate_persistent_tail_hybrid_kernel(hybrid, mode="wave", copy_mode="vectorized_contiguous"))
            source = generate_persistent_tail_hybrid_kernel(hybrid)
            self.assertIn("wave_tracker =", source)
            self.assertIn("var chunk_start", source)

    def test_probe_uses_same_rendering(self):
        plan = self.mixed()
        with patch("planning.diagnostics.spill_probe.probe_source_spill_free", return_value="checked") as probe:
            self.assertEqual(probe_spill_free(plan), "checked")
            source = probe.call_args.args[0]
            self.assertEqual(source, generate_recursive_fft_kernels(plan, compute_lanes=4,
                narrow_middle_stages=True, reference_check=False, spread_across_units=True))
            self.assertIn("wave_tracker =", source)
            self.assertIn("var chunk_start", source)

    def test_invalid_planned_options_rejected_and_override_resolves(self):
        plan = self.leaf()
        for field, value, override in (("lowering_mode", "invalid", {"mode": "physical"}),
                                        ("copy_mode", "invalid", {"copy_mode": "scalar"})):
            invalid = replace(plan, persistent=replace(plan.persistent, **{field: value}))
            with self.assertRaises(ValueError):
                generate_persistent_fft_kernel(invalid, num_logical_blocks=33)
            generate_persistent_fft_kernel(invalid, num_logical_blocks=33, **override)

    def test_default_plan_legacy_source(self):
        plan = self.leaf("physical", "scalar")
        self.assertEqual(generate_persistent_fft_kernel(plan, num_logical_blocks=33),
            generate_persistent_fft_kernel(plan, num_logical_blocks=33, mode="physical", copy_mode="scalar"))


if __name__ == "__main__":
    unittest.main()
