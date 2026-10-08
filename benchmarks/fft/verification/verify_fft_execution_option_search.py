"""Execution search options are realized, costed, rendered and probed as actual tasks."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from planning.core.target_profile import DEFAULT_TARGET_PROFILE as TARGET
from planning.core.fft_plan_core import FFTCodegenPlan
from planning.search.fft_cost_model import estimate_metrics, estimate_cost, DEFAULT_COST_WEIGHTS
from planning.search.fft_plan_search import (generate_candidates, generate_execution_option_candidates,
    _plan_signature, format_plan_summary, rank_candidates)
from planning.strategies.fft_plan_recursive import flatten_recursive_node
from planning.execution.kernel_registration import validate_recursive_kernel_registration
from planning.diagnostics.spill_probe import _probe_plan
from verification.verify_fft_recursive import run_recursive_plan
from make_fft_kernel import make_fft_kernel
from pathlib import Path


class ExecutionOptionSearchTests(unittest.TestCase):
    def candidates(self, n=64, batch=33, **kwargs):
        return generate_candidates(n, batch=batch, max_candidates=500, **kwargs)

    def test_all_axes_reachable_and_physically_deduplicated(self):
        candidates = self.candidates(630, scratchpad_byte_budget=16384)
        partitions = {mode for c in candidates for _, mode in c.choices.cooperative_partitions}
        copies = {mode for c in candidates for _, mode in c.choices.persistent_copies}
        tails = {c.choices.tail_strategy for c in candidates}
        self.assertEqual(partitions, {"round_robin", "contiguous", "balanced_contiguous"})
        self.assertEqual(copies, {"scalar", "vectorized_contiguous"})
        self.assertTrue({"all_persistent", "noncoop_tail", "cooperative_tail"} <= tails)
        self.assertEqual(len(candidates), len({_plan_signature(c.plan) for c in candidates}))
        for c in candidates:
            validate_recursive_kernel_registration(c.plan, target=TARGET)
            generate_recursive_fft_kernels(c.plan, compute_lanes=4, reference_check=False)
            self.assertEqual(c.metrics, replace(estimate_metrics(c.plan, TARGET),
                estimated_cost=estimate_cost(estimate_metrics(c.plan, TARGET))))

    def test_bounded_search_and_baseline_retained(self):
        seeds = self.candidates(630, scratchpad_byte_budget=16384, max_execution_option_candidates=0)
        for cap in (0, 1, 3, 7):
            variants = generate_execution_option_candidates(seeds, target=TARGET, max_variants=cap)
            self.assertLessEqual(len(variants), cap)
        limited = generate_candidates(630, batch=33, scratchpad_byte_budget=16384, max_candidates=8)
        self.assertEqual(len(limited), 8)
        self.assertEqual(_plan_signature(limited[0].plan), _plan_signature(seeds[0].plan))
        with self.assertRaises(ValueError):
            generate_execution_option_candidates(seeds, target=TARGET, max_variants=-1)

    def test_batch_boundaries_and_actual_tail_accounting(self):
        for batch in (1, 31, 32, 33, 40, 63, 64, 65):
            candidates = self.candidates(batch=batch)
            hybrids = [c for c in candidates if getattr(c.plan.root, "tail_hybrid", None) is not None]
            self.assertEqual(bool(hybrids), batch > 32 and batch % 32 != 0)
            for candidate in hybrids:
                h = candidate.plan.root.tail_hybrid
                self.assertEqual(h.full_blocks + h.tail_blocks, batch)
                self.assertEqual(sum(k.host.total_elems for k in flatten_recursive_node(candidate.plan.root)), 64*batch)
                self.assertEqual(candidate.metrics.estimated_dram_bytes, 64*batch*16)
                self.assertEqual(candidate.metrics.leaf_kernel_count, 2)
                self.assertEqual(candidate.metrics.persistent_extra_rounds, h.full_blocks//32 - 1)
                source = generate_recursive_fft_kernels(candidate.plan, compute_lanes=4, reference_check=False)
                self.assertEqual(source.count('print("INPUT_BEGIN")'), 1)
                self.assertEqual(source.count('print("OUTPUT_BEGIN")'), 1)
                self.assertIn("for i in range(p_total_elems)", source)
                self.assertIn("for i in range(t_total_elems)", source)
                self.assertNotIn("var p_pi", source)
                self.assertIn("full_blocks=", format_plan_summary(candidate))

    def test_partition_and_copy_variants_numeric(self):
        rng = np.random.default_rng(10)
        for inverse in (False, True):
            candidates = self.candidates(630, batch=1, scratchpad_byte_budget=16384, inverse=inverse)
            selected = []
            for mode in ("contiguous", "balanced_contiguous"):
                selected.append(next(c for c in candidates if mode in dict(c.choices.cooperative_partitions).values()))
            selected.append(next(c for c in candidates if "vectorized_contiguous" in dict(c.choices.persistent_copies).values()))
            x = rng.normal(size=630) + 1j*rng.normal(size=630)
            expected = (np.fft.ifft if inverse else np.fft.fft)(x)
            for c in selected:
                np.testing.assert_allclose(run_recursive_plan(c.plan, x, compute_lanes=4),
                                           expected, rtol=1e-5, atol=1e-4)

    def test_hybrid_forward_inverse_with_resource_capped_tail_rounds(self):
        # FP32 FFT64 data (1024 bytes) plus four-byte trackers for eight workers.
        target = replace(TARGET, num_ndp_units=4, spad_capacity_bytes=1056,
                         max_concurrent_scratchpad_bytes=1024)
        rng = np.random.default_rng(14)
        x = rng.normal(size=7*64) + 1j*rng.normal(size=7*64)
        for inverse in (False, True):
            expected = (np.fft.ifft if inverse else np.fft.fft)(x.reshape(7,64), axis=1).ravel()
            hybrids = [c for c in self.candidates(batch=7, inverse=inverse, target=target)
                       if getattr(c.plan.root, "tail_hybrid", None) is not None]
            self.assertEqual({c.choices.tail_strategy for c in hybrids}, {"noncoop_tail", "cooperative_tail"})
            for c in hybrids:
                tail = c.plan.root.tail_hybrid.tail_plan
                self.assertLess(tail.max_uthread, tail.total_uthreads)
                source = generate_recursive_fft_kernels(c.plan, target=target, compute_lanes=4, reference_check=False)
                self.assertIn("var t_rc_2", source)
                self.assertIn(f"{tail.kernel_name}Params(t_input_real, t_input_imag, t_output_real, t_output_imag, 2, 1)", source)
                np.testing.assert_allclose(run_recursive_plan(c.plan, x, compute_lanes=4),
                                           expected, rtol=1e-5, atol=1e-4)

    def test_copy_instruction_proxy_and_report_matches_plan(self):
        candidates = self.candidates()
        scalar = next(c for c in candidates if c.choices.tail_strategy == "all_persistent"
                      and c.choices.persistent_copies == ((0, "scalar"),))
        vector = next(c for c in candidates if c.choices.tail_strategy == "all_persistent"
                      and c.choices.persistent_copies == ((0, "vectorized_contiguous"),)
                      and len(c.plan.root.kernel.stages) == len(scalar.plan.root.kernel.stages))
        self.assertEqual(scalar.metrics.estimated_dram_bytes, vector.metrics.estimated_dram_bytes)
        self.assertGreater(scalar.metrics.persistent_copy_instructions, vector.metrics.persistent_copy_instructions)
        self.assertEqual(vector.metrics.estimated_cost, scalar.metrics.estimated_cost)
        weights = replace(DEFAULT_COST_WEIGHTS, persistent_copy_instruction=0.1)
        self.assertLess(estimate_cost(vector.metrics, weights), estimate_cost(scalar.metrics, weights))
        self.assertIn("vectorized_contiguous", format_plan_summary(vector))

    def test_hybrid_probe_renders_both_tasks(self):
        hybrid = next(c for c in self.candidates() if getattr(c.plan.root, "tail_hybrid", None) is not None)
        with patch("planning.diagnostics.spill_probe.probe_source_spill_free", return_value="checked") as probe:
            result = _probe_plan(hybrid.plan, compute_lanes=4, simd_lanes=8,
                narrow_middle_stages=True, target=TARGET, mojo_root=None, m2ndp_root=None,
                build_timeout=1, run_timeout=1)
            self.assertEqual(result, "checked")
            source = probe.call_args.args[0]
            for task in flatten_recursive_node(hybrid.plan.root):
                self.assertIn(f"struct {task.kernel_name}(NDPTask):", source)
            self.assertIn('print("OUTPUT_BEGIN")', source)

    def test_high_level_plan_index_renders_hybrid(self):
        ranked = rank_candidates(generate_candidates(64, batch=33))
        index = next(i for i, candidate in enumerate(ranked)
                     if getattr(candidate.plan.root, "tail_hybrid", None) is not None)
        with patch.object(Path, "write_text") as write:
            make_fft_kernel(64, batch=33, plan_index=index, reference_check=False,
                            output_path=Path(__file__).parent / "unused.mojo")
            source = write.call_args.args[0]
            self.assertIn("var p_rc", source)
            self.assertIn("var t_rc", source)
            self.assertEqual(source.count("def main() raises:"), 1)

    def test_nested_hybrid_is_rejected(self):
        from planning.strategies.fft_plan_recursive import make_recursive_transpose_plan
        hybrid = next(c for c in self.candidates() if getattr(c.plan.root, "tail_hybrid", None))
        split = make_recursive_transpose_plan(960, scratchpad_byte_budget=512)
        invalid = replace(split, root=replace(split.root, near_fft=hybrid.plan.root))
        with self.assertRaisesRegex(ValueError, "root only"):
            generate_recursive_fft_kernels(invalid, compute_lanes=4)

    def test_no_tail_bypass_of_existing_persistent_gate(self):
        candidates = self.candidates(960, scratchpad_byte_budget=512)
        self.assertFalse(any(getattr(c.plan.root, "tail_hybrid", None) for c in candidates))
        self.assertTrue(any(c.choices.persistent_copies for c in candidates))


if __name__ == "__main__":
    unittest.main()
