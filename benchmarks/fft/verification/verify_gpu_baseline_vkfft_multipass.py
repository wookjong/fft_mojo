"""VkFFT finding-2 regressions: source decisions, upload order and numerics.

Run: python -m verification.verify_gpu_baseline_vkfft_multipass
The scheduler fixture was captured by compiling the pinned upstream C++
split/reordering blocks, independently of the Python implementation.
"""

from dataclasses import replace
import json
from math import prod
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from planning.core.target_profile import DEFAULT_TARGET_PROFILE as TARGET
from planning.gpu_baseline import vkfft as vk
from planning.gpu_baseline.common import BaselineStatus
from planning.strategies.fft_plan_recursive import flatten_recursive_node
from verification.verify_fft_recursive import run_recursive_plan, verify_recursive_tree_index_only


class VkfftMultipassTests(unittest.TestCase):
    def test_captured_upstream_cpp_decisions(self):
        fixture = json.loads((Path(__file__).parent / "fixtures" / "vkfft_multipass_upstream.json").read_text())
        self.assertIn(fixture["upstream_commit"], vk.PROVENANCE.upstream_commit)
        for case in fixture["cases"]:
            with self.subTest(**case):
                target = replace(TARGET, spad_capacity_bytes=case["shared_bytes"])
                with patch.object(vk, "VKFFT_VENDOR_IS_NVIDIA", case["nvidia"]):
                    if case["factors"] is None:
                        with self.assertRaises(vk.VkfftUnsupportedError):
                            vk.schedule_axis_factors(case["length"], target,
                                                     complex_size_bytes=case["complex_size_bytes"])
                    else:
                        got = vk.schedule_axis_factors(case["length"], target,
                                                      complex_size_bytes=case["complex_size_bytes"])
                        self.assertEqual(got, tuple(case["factors"]))

    def test_reported_split_regressions(self):
        for n, expected in ((65536, (512, 128)), (131072, (512, 256)),
                            (262144, (512, 512))):
            for elem_bytes in (8, 16):
                self.assertEqual(vk.split_pow2_2pass(n, TARGET, complex_size_bytes=elem_bytes), expected)
        for elem_bytes in (8, 16):
            factors = vk.split_pow2_3pass(2**24, TARGET, complex_size_bytes=elem_bytes)
            self.assertEqual(factors, (256, 256, 256))
            self.assertEqual(prod(factors), 2**24)
            result = vk.plan(2**24, complex_size_bytes=elem_bytes)
            self.assertEqual(result.status, BaselineStatus.OK, result.diagnostics)
        self.assertEqual(vk._scheduler_limits(65536, TARGET, 8), (65536, 8192, 2048))
        self.assertEqual(vk._scheduler_limits(30000, TARGET, 8), (122880, 15360, 3840))

    def test_vendor_exception_and_failed_two_upload_search(self):
        # Exactly at the NVIDIA threshold the pow2 branch is active;
        # above it the original source's sqrt-divisor branch is used.
        with patch.object(vk, "split_non_pow2_2pass", wraps=vk.split_non_pow2_2pass) as search:
            vk.split_pow2_2pass(262144, TARGET)
            search.assert_not_called()
            vk.split_pow2_2pass(524288, TARGET)
            search.assert_called_once()
        # 101^2 * 5 has no pair bounded by 256, while its three factors
        # fit. Scheduling does not imply support for the subsequent Rader FFT.
        target = replace(TARGET, spad_capacity_bytes=8192)
        n = 101**2 * 5
        self.assertEqual(vk.choose_num_passes(n, non_strided=True, target=target), 2)
        self.assertIsNone(vk.split_non_pow2_2pass(n, target))
        self.assertEqual(prod(vk.schedule_axis_factors(n, target)), n)

    def test_batched_upload_replicas(self):
        for batch in (1, 3):
            for n, target in ((8192, TARGET), (32768, replace(TARGET, spad_capacity_bytes=4096))):
                result = vk.plan_m2ndp(n, batch=batch, target=target)
                self.assertEqual(result.status, BaselineStatus.OK, result.diagnostics)
                factors = result.gpu_config.extra["factors"]
                for config in result.gpu_config.extra["leaf_configs"]:
                    self.assertEqual(config["replicas"], batch * n // config["length"])
                    self.assertEqual(config["length"], factors[config["upload_id"]])

    def test_invalid_split_refused_before_lowering(self):
        for factors in ((15360, 64, 17), (256, 256, 0), (256, 256, 512)):
            with self.assertRaises(vk.VkfftUnsupportedError):
                vk.validate_axis_factors(2**24, factors, TARGET)
        with self.assertRaises(vk.VkfftUnsupportedError):
            vk.validate_axis_factors(65536, (8192, 8), TARGET)
        with patch.object(vk, "split_pow2_3pass", return_value=(15360, 64, 17)), \
             patch.object(vk, "_recursive_result") as lower:
            result = vk.plan(2**24)
            self.assertEqual(result.status, BaselineStatus.UNSUPPORTED_GPU_ALGORITHM)
            lower.assert_not_called()

    def test_upload_order_and_stage_start_size(self):
        cases = ((8192, TARGET), (32768, replace(TARGET, spad_capacity_bytes=4096)))
        for n, target in cases:
            for inverse in (False, True):
                result = vk.plan_m2ndp(n, inverse=inverse, target=target)
                self.assertEqual(result.status, BaselineStatus.OK, result.diagnostics)
                extra = result.gpu_config.extra
                factors, leaves = extra["factors"], extra["leaf_configs"]
                self.assertEqual([c["upload_id"] for c in leaves], list(range(len(factors) - 1, -1, -1)))
                self.assertEqual(tuple(c["upload_id"] for c in leaves), extra["upload_execution_order"])
                for c in leaves:
                    upload = c["upload_id"]
                    self.assertEqual(c["length"], factors[upload])
                    self.assertEqual(c["stage_start_size"], prod(factors[:upload]))
                    self.assertEqual(c["replicas"], n // c["length"])
                fft_stages = [k for k in flatten_recursive_node(result.plan.root) if hasattr(k, "stages")]
                self.assertEqual([k.length for k in fft_stages], list(reversed(factors)))
        # The middle upload's stageStartSize must exclude upload 2.
        result = vk.plan_m2ndp(32768, target=cases[1][1])
        middle = result.gpu_config.extra["leaf_configs"][1]
        self.assertEqual(middle["stage_start_size"], 8)
        self.assertNotEqual(middle["stage_start_size"], 32768 // middle["length"])

    def test_two_three_upload_numerics_and_codegen(self):
        for planner, n, target in (
            (vk.plan, 16384, TARGET),
            (vk.plan_m2ndp, 8192, TARGET),
            (vk.plan_m2ndp, 8192, replace(TARGET, spad_capacity_bytes=2048)),
            (vk.plan_m2ndp, 10000, TARGET),
        ):
            for inverse in (False, True):
                with self.subTest(planner=planner.__name__, n=n, inverse=inverse,
                                  shared=target.spad_capacity_bytes):
                    result = planner(n, inverse=inverse, target=target)
                    self.assertEqual(result.status, BaselineStatus.OK, result.diagnostics)
                    x = np.random.default_rng(12).uniform(-1, 1, n) + 1j * np.random.default_rng(13).uniform(-1, 1, n)
                    expected = np.fft.ifft(x) if inverse else np.fft.fft(x)
                    np.testing.assert_allclose(run_recursive_plan(result.plan, x), expected, rtol=1e-5, atol=1e-2)
                    verify_recursive_tree_index_only(result.plan.root)
                    source = generate_recursive_fft_kernels(result.plan, reference_check=False, target=target)
                    for stage in flatten_recursive_node(result.plan.root):
                        self.assertIn(stage.kernel_name, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
