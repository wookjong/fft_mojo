"""POST fusion: independent transpose oracle, real emitted execution and ABI wiring."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch
import numpy as np

from codegen.fft_codegen import _mapping_base_expr
from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from planning.core.fft_plan_core import AddressMapping
from planning.execution.fft_boundary_fusion import make_boundary_fusion_plan
from planning.execution.fft_unit_parallel import distribute_plan
from planning.execution.task_fusion import make_task_fusion_plan, resolve_task_execution
from planning.strategies.fft_plan_recursive import make_recursive_transpose_plan
from verification.verify_fft_task_fusion import TARGET, example, run_fused


def fixture(n, near, batch=1, inverse=False):
    plan = make_recursive_transpose_plan(n, scratchpad_byte_budget=4096,
        forced_split_near_length=near, batch=batch, inverse=inverse, target=TARGET)
    return distribute_plan(plan, target=TARGET, units_per_leaf=1)


class BoundaryFusionTests(unittest.TestCase):
    def assert_numeric(self, plan, width='off', unit=0, storage='spad', signal='random'):
        rng = np.random.default_rng(91)
        x = rng.normal(size=(plan.batch, plan.n)) + 1j*rng.normal(size=(plan.batch, plan.n))
        if signal == 'impulse':
            x[:] = 0
            x[:, 1] = np.arange(1, plan.batch+1) + 0.5j
        elif signal == 'tone':
            x = np.exp(2j*np.pi*np.arange(plan.n)[None,:]*np.arange(1,plan.batch+1)[:,None]/plan.n)
        y = run_fused(plan, x.ravel(), base_unit=unit, leaf_storage=storage,
                      batch_window=width, boundary_fusion='post').reshape(x.shape)
        expected = (np.fft.ifft if plan.inverse else np.fft.fft)(x, axis=1)
        np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)
        errors = np.abs(y-expected)
        nrms = np.sqrt(np.mean(errors**2,axis=1))/np.maximum(np.sqrt(np.mean(np.abs(expected)**2,axis=1)),1e-30)
        self.assertTrue(np.all(nrms <= 1e-5))
        self.assertTrue(np.all(errors.max(axis=1) <= 1e-4+5e-5*np.abs(expected).max(axis=1)))

    def test_integer_oracle_and_renderer(self):
        for a,b,batch in ((8,8,3),(16,8,8),(16,16,3),(128,64,3),(256,64,8)):
            n=a*b
            # Labels are source offsets. NumPy transpose is independent of the
            # new planner's row/element formula and includes every batch.
            source=np.arange(batch*n).reshape(batch,b,a)
            destination=np.empty(batch*n,dtype=int)
            mapping=AddressMapping.batched_strided(rows_per_batch=b,batch_stride=n,row_stride=1,elem_stride=b)
            row_expr=_mapping_base_expr(mapping,a,row_index='row')
            addresses=[]
            for row in range(batch*b):
                base=eval(row_expr,{'row':row})
                for elem in range(a):
                    addr=mapping.batched_element_index(row,elem)
                    self.assertEqual(addr,base+elem*b)
                    addresses.append(addr)
                    destination[addr]=source.reshape(-1)[row*a+elem]
            self.assertEqual(sorted(addresses),list(range(batch*n)))
            np.testing.assert_array_equal(destination,source.transpose(0,2,1).ravel())
        with self.assertRaises(ValueError):
            AddressMapping.batched_strided(rows_per_batch=0,batch_stride=64,row_stride=1,elem_stride=8)

    def test_final_stage_only_and_resource_schedule(self):
        for n in (8192,16384):
            for batch,width in ((1,'off'),(8,'4')):
                for inverse in (False,True):
                    with self.subTest(n=n,batch=batch,inverse=inverse):
                        p=example('vkfft',n,batch,inverse)
                        f=make_task_fusion_plan(p,target=TARGET,compute_lanes=4,narrow_middle_stages=True,
                            leaf_storage='spad',batch_window=width,boundary_fusion='post')
                        self.assertTrue(f.supported,f.reason)
                        boundary=resolve_task_execution(p,f)
                        far=boundary.nodes[3]
                        original=p.root.far_child.kernel
                        self.assertEqual(far.stages[:-1],original.stages[:-1])
                        self.assertEqual(far.distributed,original.distributed)
                        self.assertEqual(tuple(s.radix for s in far.stages),tuple(s.radix for s in original.stages))
                        self.assertEqual(len(f.groups),1)
                        self.assertEqual(len(f.groups[0].functions),7)
                        self.assertEqual(f.groups[0].node_ids,(0,1,2,3))
                        self.assertLessEqual(f.reserved_bytes,TARGET.spad_capacity_bytes)
                        for b in far.stages[-1].batches:
                            for o in b.outputs:
                                self.assertEqual(o.scale,1/n if inverse else None)
                                self.assertEqual(o.store.mode,'scalar_lanes')
                        source=generate_recursive_fft_kernels(p,target=TARGET,compute_lanes=4,
                            narrow_middle_stages=True,reference_check=False,task_fusion='auto',
                            leaf_storage='spad',batch_window=width,boundary_fusion='post')
                        self.assertTrue(p.root.post_transpose.kernel_name not in source)
                        self.assertEqual(source.count('launch_parallel['),7)
                        self.assertEqual(source.count('.launch('),1)
                        self.assertTrue(f'{far.kernel_name}Params(work0_real, work0_imag, output_real, output_imag' in source,
                                        'far host Params must bind the final output arrays')
                        metadata=json.loads(next(line.split(' ',2)[2] for line in source.splitlines()
                            if line.startswith('# FFT_BOUNDARY_FUSION ')))
                        self.assertEqual(metadata['active_node_ids'],[0,1,2,3])
                        self.assertEqual(metadata['removed_node_ids'],[4])

    def test_small_emitted_numerics(self):
        for n,near in ((64,8),(128,8),(256,16)):
            for inverse in (False,True):
                for batch,width in ((1,'off'),(3,'2'),(3,'4'),(8,'4')):
                    with self.subTest(n=n,inverse=inverse,batch=batch,width=width):
                        self.assert_numeric(fixture(n,near,batch,inverse),width,unit=(batch+near)%28)
        for signal in ('impulse','tone'):
            self.assert_numeric(fixture(128,8,3,True),'2',unit=27,signal=signal)

    def test_representative_forward_inverse_and_tails(self):
        for n in (8192,16384):
            for inverse in (False,True):
                for batch,width,unit in ((1,'off',0),(3,'2',27),(3,'4',1),(8,'4',27)):
                    with self.subTest(n=n,inverse=inverse,batch=batch,width=width):
                        self.assert_numeric(example('vkfft',n,batch,inverse),width,unit)
        self.assert_numeric(example('vkfft',16384,8),'2',unit=1)
        self.assert_numeric(example('vkfft',8192,3,True),'2',unit=27,storage='cxl')

    def test_explicit_rejections(self):
        p=example('vkfft',8192)
        cases=[(example('vkfft',1024),{}),
               (distribute_plan(p,target=TARGET,units_per_leaf=2),{}),
               (distribute_plan(p,target=TARGET,units_per_leaf=1,fuse_stages=2),{}),
               (p,dict(task_fusion_control=True)),(p,dict(task_fusion='off'))]
        options=dict(target=TARGET,task_fusion='auto',boundary_fusion='post')
        for plan,extra in cases:
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                generate_recursive_fft_kernels(plan,**(options|extra))
        with self.assertRaises(ValueError):
            make_boundary_fusion_plan(p,mode='invalid')

    def test_corrupted_window_offset_is_detected(self):
        from codegen.common import TaskKernelBinding
        original=TaskKernelBinding.emit_window_prelude
        def broken(binding,emitter):
            original(binding,emitter)
            if binding.window_batch:
                emitter.add('        window_base = 0')
        with patch.object(TaskKernelBinding,'emit_window_prelude',broken):
            with self.assertRaises(AssertionError):
                self.assert_numeric(fixture(128,8,3),'2')

    def test_wrong_scale_is_detected(self):
        from planning.execution import fft_boundary_fusion as module
        original=module.rebind_final_output
        def broken(plan,**options):
            return original(plan,**(options|dict(inverse_scale=1.0)))
        with patch.object(module,'rebind_final_output',broken):
            with self.assertRaises(AssertionError):
                self.assert_numeric(fixture(128,8,3,True),'2')


if __name__ == '__main__':
    unittest.main()
