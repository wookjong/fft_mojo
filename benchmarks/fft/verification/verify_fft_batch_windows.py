"""Batch window resources, tail addressing and emitted FFT numeric checks."""
import unittest
import numpy as np
from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from planning.execution.task_fusion import make_task_fusion_plan, window_execution_plan
from planning.strategies.fft_plan_recursive import flatten_recursive_node, PhysicalTransposePlan
from verification.verify_fft_task_fusion import TARGET, example, run_fused


class BatchWindowTests(unittest.TestCase):
    def fusion(self, plan, width='auto', storage='spad'):
        return make_task_fusion_plan(plan, target=TARGET, compute_lanes=4,
            narrow_middle_stages=True, leaf_storage=storage, batch_window=width)

    def test_constant_resources_and_single_window_identity(self):
        for lib in ('clfft', 'vkfft'):
            for n in (8192, 16384):
                previous = None
                for b in (8, 16, 64):
                    p = example(lib, n, b)
                    f = self.fusion(p, '2')
                    self.assertTrue(f.supported, f.reason)
                    resources = f.common_uthreads, f.arena_elements, f.reserved_bytes
                    self.assertEqual(resources, previous or resources)
                    previous = resources
                    self.assertEqual(f.total_batch, b)
                    self.assertTrue(all(len(g.functions) <= 8 for g in f.groups))
                p = example(lib, n)
                options = dict(target=TARGET, compute_lanes=4, narrow_middle_stages=True,
                               reference_check=False, task_fusion='auto', leaf_storage='spad')
                self.assertEqual(generate_recursive_fft_kernels(p, **options),
                                 generate_recursive_fft_kernels(p, batch_window='auto', **options))
                f = self.fusion(example(lib, n, 64))
                self.assertTrue(f.supported)
                self.assertLessEqual(f.reserved_bytes, 32768)

    def test_window_geometry_matches_independent_plan(self):
        for lib in ('clfft', 'vkfft'):
            for n in (8192, 16384):
                resized = window_execution_plan(example(lib, n, 64), 2)
                for a, b in zip(flatten_recursive_node(resized.root),
                                flatten_recursive_node(example(lib, n, 2).root)):
                    self.assertEqual(a, b)

    def test_invalid_windows_and_nonfused_use(self):
        p = example()
        for w in ('0', '-1', 'bad'):
            with self.assertRaises(ValueError):
                self.fusion(p, w)
        with self.assertRaises(ValueError):
            generate_recursive_fft_kernels(p, batch_window='2')

    def test_tail_numerics_and_private_spad_writers(self):
        # Covers two task groups, CXL/SPAD banks, inverse normalization, idle
        # transpose workers and a pool beginning on a nonzero physical unit.
        cases = [('vkfft',8192,3,2,False,'spad',1),
                 ('clfft',8192,5,2,True,'spad',27),
                 ('vkfft',16384,3,2,True,'spad',27),
                 ('clfft',16384,3,2,False,'spad',1),
                 ('vkfft',8192,3,2,False,'cxl',0),
                 ('clfft',8192,1,4,False,'spad',0)]
        for lib,n,b,w,inverse,storage,unit in cases:
            with self.subTest(lib=lib,n=n,b=b,storage=storage):
                rng = np.random.default_rng(31)
                x = rng.normal(size=(b,n)) + 1j*rng.normal(size=(b,n))
                p = example(lib,n,b,inverse)
                y = run_fused(p,x.ravel(),base_unit=unit,leaf_storage=storage,batch_window=str(w))
                ref = (np.fft.ifft if inverse else np.fft.fft)(x,axis=1)
                np.testing.assert_allclose(y.reshape(b,n),ref,atol=1e-5,rtol=1e-5)

    def test_exact_windows_elide_tail_math(self):
        for lib in ('clfft', 'vkfft'):
            p = example(lib, 8192, 4)
            source = generate_recursive_fft_kernels(p, target=TARGET, compute_lanes=4,
                narrow_middle_stages=True, reference_check=False, task_fusion='auto',
                leaf_storage='spad', batch_window='2')
            self.assertNotIn('var live_batch', source)
            rng = np.random.default_rng(61)
            x = rng.normal(size=(4,8192))+1j*rng.normal(size=(4,8192))
            y = run_fused(p,x.ravel(),base_unit=27,leaf_storage='spad',batch_window='2')
            np.testing.assert_allclose(y.reshape(4,-1),np.fft.fft(x,axis=1),atol=1e-5,rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
