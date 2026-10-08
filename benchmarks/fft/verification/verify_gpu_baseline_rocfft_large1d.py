"""Source-order/config regressions for rocFFT CC and TRTRT large-1D plans.

Run from benchmarks/fft: python -m verification.verify_gpu_baseline_rocfft_large1d
Expected values come from pinned tree_node_1D.cpp and config_sbcc/sbrc.py,
not from the M2NDP planner's own decisions.
"""

from dataclasses import replace
from unittest.mock import patch
import unittest

import numpy as np

from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.gpu_baseline import rocfft_default as roc
from planning.gpu_baseline.common import BaselineStatus
from planning.search.fft_cost_model import estimate_metrics
from planning.strategies.fft_plan_recursive import (
    FFTLeafPlan, FFTRecursiveNodePlan, flatten_recursive_node,
)
from verification.verify_fft_recursive import run_recursive_plan, verify_recursive_tree_index_only


class RocfftLarge1DTests(unittest.TestCase):
    def test_cc_source_order_and_block_settings(self):
        # length, first (length, radices, WGS, TPT, TPB), second (...)
        cases = (
            (8192, (64, (8, 8), 256, 8, 32),
             (128, (8, 4, 4), 128, 16, 8)),
            (4704, (96, (8, 3, 4), 256, 32, 8),
             (49, (7, 7), 196, 7, 28)),
            (32768, (128, (16, 8), 256, 16, 16),
             (256, (4, 4, 4, 4), 256, 32, 8)),
        )
        for planner in (roc.plan, roc.plan_m2ndp):
            for n, first, second in cases:
                for batch in (1, 3):
                    with self.subTest(planner=planner.__name__, n=n, batch=batch):
                        # Neither CC child may call Decide1DScheme again.
                        with patch.object(roc, "decide_scheme", wraps=roc.decide_scheme) as decide:
                            result = planner(n, batch=batch)
                        self.assertEqual(result.status, BaselineStatus.OK)
                        self.assertTrue(all(call.args[0] == n for call in decide.call_args_list))
                        configs = result.gpu_config.extra["leaf_configs"]
                        self.assertEqual([c["scheme"] for c in configs], [
                            "CS_KERNEL_STOCKHAM_BLOCK_CC", "CS_KERNEL_STOCKHAM_BLOCK_RC"])
                        for config, expected in zip(configs, (first, second)):
                            self.assertEqual(tuple(config[k] for k in (
                                "length", "radices", "workgroup_size",
                                "threads_per_transform", "transforms_per_block")), expected)
                            self.assertEqual(config["replicas"], batch * n // config["length"])
                        root = result.plan.root
                        self.assertEqual((root.b, root.a), (first[0], second[0]))
                        self.assertEqual((root.pre_transpose.rows, root.pre_transpose.cols),
                                         (first[0], second[0]))
                        for leaf, expected in zip((root.near_fft, root.far_child), (first, second)):
                            self.assertEqual(tuple(s.radix for s in leaf.kernel.stages), expected[1])
                            execution = leaf.kernel.cooperation or leaf.kernel.persistent
                            self.assertEqual(execution.workers_per_fft, expected[3])

    def test_block_table_launch_sizing(self):
        self.assertEqual((len(roc.SBCC_TABLE), len(roc.SBRC_TABLE)), (32, 20))
        # Wide flavour, half-LDS workgroup expansion, non-divisible WGS,
        # and LDS-limited TPB exercise the actual generator rules.
        for table, length, expected in (
            (roc.SBCC_TABLE, 160, (192, 16)),
            (roc.SBCC_TABLE, 100, (200, 25)),
            (roc.SBCC_TABLE, 168, (252, 42)),
            (roc.SBRC_TABLE, 289, (119, 17)),
            (roc.SBRC_TABLE, 512, (512, 128)),
        ):
            c = table[length]
            self.assertEqual((c.workgroup_size, c.threads_per_transform), expected)
        for n, div1 in roc.MAP_1D_LENGTH_SINGLE.items():
            self.assertIn(div1, roc.SBCC_TABLE)
            self.assertIn(n // div1, roc.SBRC_TABLE)

    def test_trtrt_forces_only_second_child(self):
        for planner in (roc.plan, roc.plan_m2ndp):
            with patch.object(roc, "decide_scheme", wraps=roc.decide_scheme) as decide:
                result = planner(2058)
            self.assertEqual(result.status, BaselineStatus.OK)
            self.assertEqual([c["length"] for c in result.gpu_config.extra["leaf_configs"]], [42, 49])
            self.assertIn(42, [c.args[0] for c in decide.call_args_list])
            self.assertNotIn(49, [c.args[0] for c in decide.call_args_list])

    def test_real_nested_first_child_flattens_and_generates(self):
        # Upstream large power-of-two split: row1(16384) itself uses CC.
        # Relax only scratchpad capacity for this structural/codegen test;
        # default hardware refusal remains legitimate and is not bypassed.
        target = replace(DEFAULT_TARGET_PROFILE, spad_capacity_bytes=1024 * 1024)
        result = roc.plan(134217728, target=target)
        self.assertEqual(result.status, BaselineStatus.OK, result.diagnostics)
        root = result.plan.root
        self.assertIsInstance(root.near_fft, FFTRecursiveNodePlan)
        self.assertIsInstance(root.far_child, FFTLeafPlan)
        self.assertEqual((root.near_fft.m, root.far_child.m), (16384, 8192))
        stages = flatten_recursive_node(root)
        self.assertGreater(len(stages), 5)
        metrics = estimate_metrics(result.plan, target)
        self.assertEqual(metrics.recursion_depth, 2)
        self.assertEqual(metrics.leaf_kernel_count, 3)
        text = generate_recursive_fft_kernels(result.plan, reference_check=False, target=target)
        for stage in stages:
            self.assertIn(stage.kernel_name, text)

    def test_nested_first_child_numerical_contract(self):
        # A small controlled tree exercises the recursive-first-child math
        # without allocating a real N=2^27 input. Source selection itself
        # is tested separately above using the real upstream split.
        original_decide = roc.decide_scheme

        def decide(length, **kwargs):
            if length in (48, 12):
                return roc.SchemeDecision("CS_L1D_TRTRT", div_length1=12 if length == 48 else 3)
            return original_decide(length, **kwargs)

        for inverse in (False, True):
            with patch.object(roc, "decide_scheme", side_effect=decide):
                result = roc.plan(48, inverse=inverse)
            self.assertEqual(result.status, BaselineStatus.OK)
            self.assertIsInstance(result.plan.root.near_fft, FFTRecursiveNodePlan)
            verify_recursive_tree_index_only(result.plan.root)
            x = np.random.default_rng(48).normal(size=48) + 1j * np.arange(48)
            expected = np.fft.ifft(x) if inverse else np.fft.fft(x)
            np.testing.assert_allclose(run_recursive_plan(result.plan, x), expected,
                                       rtol=1e-5, atol=1e-2)

    def test_forward_and_inverse_numerical(self):
        for planner, lengths in ((roc.plan, (2058, 4704, 8192)),
                                 (roc.plan_m2ndp, (8192,))):
            for n in lengths:
                x = np.random.default_rng(n).normal(size=n) + 1j * np.random.default_rng(n + 1).normal(size=n)
                for inverse in (False, True):
                    with self.subTest(planner=planner.__name__, n=n, inverse=inverse):
                        result = planner(n, inverse=inverse)
                        self.assertEqual(result.status, BaselineStatus.OK)
                        got = run_recursive_plan(result.plan, x)
                        expected = np.fft.ifft(x) if inverse else np.fft.fft(x)
                        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
