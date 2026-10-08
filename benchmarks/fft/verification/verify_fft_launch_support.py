"""Reject unsupported unit-spreading launch granules before work begins."""
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from make_fft_kernel import make_fft_kernel, _parse_args
from planning.search.fft_plan_search import generate_candidates
from planning.strategies.fft_plan_recursive import make_recursive_transpose_plan
from verification.verify_fft_recursive import run_recursive_plan


class LaunchSupportTests(unittest.TestCase):
    def test_recursive_planner_rejects_spreading_granule4(self):
        with self.assertRaisesRegex(ValueError, "compute_lanes=4"):
            make_recursive_transpose_plan(64, scratchpad_byte_budget=4096,
                                         simd_lanes=4, spread_across_units=True)

    def test_search_rejects_before_costing(self):
        with patch("planning.search.fft_plan_search.estimate_metrics") as metrics:
            with self.assertRaisesRegex(ValueError, "simd_lanes=8"):
                generate_candidates(64, simd_lanes=4, spread_across_units=True)
            metrics.assert_not_called()
        self.assertTrue(generate_candidates(64, simd_lanes=4, spread_across_units=False))

    def test_api_rejects_before_planning_and_preserves_output(self):
        output = Path(__file__).resolve().parent / "existing.mojo"
        with patch("make_fft_kernel.make_recursive_transpose_plan") as builder, patch.object(Path, "write_text") as write:
            with self.assertRaisesRegex(ValueError, "32-byte launch granule"):
                make_fft_kernel(64, simd_lanes=4, output_path=output)
            builder.assert_not_called()
            write.assert_not_called()

    def test_cli_checks_dump_paths_too(self):
        for dump in ([], ["--dump-plan"], ["--dump-candidates"]):
            errors = StringIO()
            with redirect_stderr(errors), self.assertRaises(SystemExit) as error:
                _parse_args(["64", "--simd-lanes", "4"] + dump)
            self.assertEqual(error.exception.code, 2)
            self.assertIn("compute_lanes=4", errors.getvalue())
        accepted = _parse_args(["64", "--simd-lanes", "4", "--no-spread-across-units"])
        self.assertTrue(accepted.no_spread_across_units)
        self.assertEqual(_parse_args(["64", "--compute-lanes", "4"]).simd_lanes, 8)

    def test_codegen_checks_before_emitting(self):
        plan = make_recursive_transpose_plan(64, scratchpad_byte_budget=4096, simd_lanes=4)
        with patch("codegen.fft_transpose_codegen._emit_kernel") as emitter:
            with self.assertRaisesRegex(ValueError, "simd_lanes=8"):
                generate_recursive_fft_kernels(plan, spread_across_units=True)
            emitter.assert_not_called()

    def test_arithmetic4_default_and_nonspreading_experiment(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=64) + 1j*rng.normal(size=64)
        for simd, spread in ((8, True), (4, False)):
            for inverse in (False, True):
                plan = make_recursive_transpose_plan(64, scratchpad_byte_budget=4096,
                                simd_lanes=simd, spread_across_units=spread, inverse=inverse)
                generate_recursive_fft_kernels(plan, compute_lanes=4, spread_across_units=spread)
                expected = (np.fft.ifft if inverse else np.fft.fft)(x)
                np.testing.assert_allclose(run_recursive_plan(plan, x, compute_lanes=4),
                                           expected, rtol=1e-5, atol=1e-5)
        output = Path(__file__).resolve().parent / "experiment.mojo"
        with patch.object(Path, "write_text") as write:
            make_fft_kernel(64, simd_lanes=4, spread_across_units=False, output_path=output)
            write.assert_called_once()
            self.assertIn("struct FFTRecLeaf0(NDPTask):", write.call_args.args[0])

    def test_invalid_granule_rejected_before_division(self):
        for granule in (0, -1, True, 4.0):
            with self.assertRaises(ValueError):
                make_recursive_transpose_plan(64, scratchpad_byte_budget=4096, simd_lanes=granule)


if __name__ == "__main__":
    unittest.main()
