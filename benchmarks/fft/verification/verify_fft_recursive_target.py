"""Caller resource profiles must survive recursive and mixed leaf lowering."""
from dataclasses import replace
import unittest
import numpy as np
from planning.core.target_profile import DEFAULT_TARGET_PROFILE as DEFAULT
from planning.core.fft_plan_core import FFTCodegenPlan
from planning.execution.fft_plan_persistent import make_persistent_leaf_plan
from planning.strategies.fft_plan_recursive import make_recursive_transpose_plan, flatten_recursive_node
from planning.search.fft_plan_search import generate_candidates
from verification.verify_fft_recursive import run_recursive_plan


def leaves(plan):
    return [s for s in flatten_recursive_node(plan.root) if isinstance(s, FFTCodegenPlan)]


class RecursiveTargetTests(unittest.TestCase):
    def setUp(self):
        self.target = replace(DEFAULT, num_ndp_units=4, spad_capacity_bytes=2048)

    def build(self, n, **kwargs):
        return make_recursive_transpose_plan(n, scratchpad_byte_budget=kwargs.pop("budget", 4096),
                                            target=kwargs.pop("target", self.target), **kwargs)

    def check_groups(self, plan, count=4, workers=8):
        persistent = [k for k in leaves(plan) if k.persistent is not None]
        self.assertTrue(persistent)
        for k in persistent:
            self.assertEqual(k.persistent.software_group_count, count)
            self.assertEqual(k.persistent.workers_per_group, workers)
            self.assertEqual(k.total_uthreads, count * workers)
        return persistent

    def test_root_matches_standalone(self):
        k = self.check_groups(self.build(64, persistent_leaf=True))[0]
        standalone = make_persistent_leaf_plan(length=64, radices=tuple(s.radix for s in k.stages),
            num_logical_blocks=1, target=self.target, kernel_name=k.kernel_name,
            inverse_scale=None)
        self.assertEqual(k, standalone)

    def test_deep_recursion(self):
        plan = self.build(4096, budget=128, persistent_leaf=True)
        self.assertGreater(len(self.check_groups(plan)), 2)

    def test_mixed_numeric_forward_and_inverse(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=960) + 1j * rng.normal(size=960)
        for sequence in (("persistent", None), (2, "persistent")):
            for inverse in (False, True):
                plan = self.build(960, budget=512, forced_worker_sequence=sequence, inverse=inverse)
                self.check_groups(plan)
                actual = run_recursive_plan(plan, x, compute_lanes=4)
                expected = np.fft.ifft(x) if inverse else np.fft.fft(x)
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)

    def test_scratchpad_cap_and_legacy_override(self):
        small = replace(self.target, spad_capacity_bytes=512)
        # Resource-limited recursive execution now decomposes before a leaf
        # could consume the space reserved for worker-private trackers.
        constrained = self.build(64, target=small, persistent_leaf=True)
        self.check_groups(constrained)
        for leaf in leaves(constrained):
            self.assertLessEqual(sum(b.elements * 4 for b in leaf.scratchpad_buffers) + 32, 512)
        with self.assertRaises((ValueError, NotImplementedError)):
            self.build(64, target=small, forced_worker_sequence=("persistent",))
        self.check_groups(self.build(64, target=small, spad_capacity_bytes=1056, persistent_leaf=True))
        self.check_groups(self.build(960, target=small, budget=512, persistent_leaf=True))

    def test_registration_limit_and_search(self):
        target = replace(self.target, max_kernel_register=4)
        with self.assertRaises((ValueError, NotImplementedError)):
            self.build(64, target=target, persistent_leaf=True)
        self.check_groups(self.build(16, target=target, persistent_leaf=True))
        for n in (16, 64):
            candidates = generate_candidates(n, target=target, scratchpad_byte_budget=4096)
            self.assertTrue(candidates)
            for c in candidates:
                for k in leaves(c.plan):
                    if k.persistent is not None:
                        self.assertEqual(k.persistent.software_group_count, 4)
                        self.assertLessEqual(2 + len(k.stages), 4)

    def test_search_preserves_custom_group_count(self):
        for n, budget in ((64, 4096), (960, 512)):
            persistent_count = 0
            for candidate in generate_candidates(n, target=self.target, scratchpad_byte_budget=budget):
                for k in leaves(candidate.plan):
                    if k.persistent is not None:
                        persistent_count += 1
                        self.assertEqual(k.persistent.software_group_count, 4)
                        self.assertEqual(k.total_uthreads, 32)
            self.assertGreater(persistent_count, 0)

    def test_custom_interleave_and_invalid_mapping(self):
        target = replace(self.target, interleave_chunk_uthreads=4, mapping_stride_bytes=128)
        self.check_groups(self.build(64, target=target, persistent_leaf=True), workers=4)
        with self.assertRaises(NotImplementedError):
            self.build(64, target=replace(target, mapping_stride_bytes=256), persistent_leaf=True)

    def test_default_profile_compatibility(self):
        kwargs = dict(scratchpad_byte_budget=512, spad_capacity_bytes=DEFAULT.spad_capacity_bytes,
                      max_concurrent_scratchpad_bytes=DEFAULT.max_concurrent_scratchpad_bytes)
        for persistent in (False, True):
            self.assertEqual(make_recursive_transpose_plan(960, persistent_leaf=persistent, **kwargs),
                make_recursive_transpose_plan(960, persistent_leaf=persistent, target=DEFAULT, **kwargs))


if __name__ == "__main__":
    unittest.main()
