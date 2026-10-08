"""Cooperative hardware grouping validation at public factory boundaries."""
from dataclasses import replace
import unittest
import numpy as np
from planning.core.target_profile import DEFAULT_TARGET_PROFILE as DEFAULT
from planning.execution.fft_plan_cooperative import (
    make_cooperative_leaf_plan, generate_partition_mode_candidates, worker_candidates_per_fft,
)
from planning.execution.fft_plan_persistent import make_persistent_tail_hybrid_plan
from planning.strategies.fft_plan_recursive import make_recursive_transpose_plan
from verification.verify_fft_recursive import run_recursive_plan


class CooperativeWorkersTests(unittest.TestCase):
    def test_direct_factory_rejects_invalid_workers(self):
        for workers in (0, -1, 3, 6, 16, 2.0, True):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "positive integer divisor"):
                make_cooperative_leaf_plan(128, (4, 4, 4, 2), workers_per_fft=workers)

    def test_explicit_tail_rejected_even_without_tail(self):
        for blocks in (3, 32, 35, 64):
            for workers in (0, 3, 16):
                with self.subTest(blocks=blocks, workers=workers), self.assertRaises(ValueError):
                    make_persistent_tail_hybrid_plan(64, (4, 4, 4), num_logical_blocks=blocks,
                        tail_strategy="cooperative_tail", tail_cooperative_workers=workers)

    def test_custom_interleave_propagates(self):
        target = replace(DEFAULT, interleave_chunk_uthreads=4, mapping_stride_bytes=128)
        for workers in (1, 2, 4):
            make_cooperative_leaf_plan(128, (4, 4, 4, 2), workers_per_fft=workers,
                                       interleave_chunk_uthreads=4)
        with self.assertRaises(ValueError):
            make_cooperative_leaf_plan(128, (4, 4, 4, 2), workers_per_fft=8,
                                       interleave_chunk_uthreads=4)
        hybrid = make_persistent_tail_hybrid_plan(128, (4, 4, 4, 2), num_logical_blocks=35,
                                                 target=target, tail_strategy="cooperative_tail")
        self.assertEqual(hybrid.tail_plan.cooperation.workers_per_fft, 4)
        with self.assertRaises(ValueError):
            make_persistent_tail_hybrid_plan(128, (4, 4, 4, 2), num_logical_blocks=35,
                target=target, tail_strategy="cooperative_tail", tail_cooperative_workers=8)
        self.assertTrue(generate_partition_mode_candidates(128, (4, 4, 4, 2),
                        workers_per_fft=4, interleave_chunk_uthreads=4))

    def test_invalid_chunk_is_rejected(self):
        for chunk in (0, -1, 4.0, True):
            with self.assertRaises(ValueError):
                worker_candidates_per_fft(128, (4, 4, 4, 2), interleave_chunk_uthreads=chunk)
            with self.assertRaises(ValueError):
                make_cooperative_leaf_plan(128, (4, 4, 4, 2), workers_per_fft=2,
                                           interleave_chunk_uthreads=chunk)

    def test_valid_workers_all_partitions_numeric(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=128) + 1j * rng.normal(size=128)
        for inverse in (False, True):
            wrapper = make_recursive_transpose_plan(128, scratchpad_byte_budget=4096, inverse=inverse)
            for workers in (1, 2, 4, 8):
                for mode in ("round_robin", "contiguous", "balanced_contiguous"):
                    kernel = make_cooperative_leaf_plan(128, (4, 4, 4, 2),
                                workers_per_fft=workers, partition_mode=mode, inverse=inverse)
                    plan = replace(wrapper, root=replace(wrapper.root, kernel=kernel))
                    actual = run_recursive_plan(plan, x, compute_lanes=4)
                    expected = np.fft.ifft(x) if inverse else np.fft.fft(x)
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
