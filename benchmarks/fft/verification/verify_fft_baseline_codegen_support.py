"""Execute the emitted code for baseline composites and global FFT staging."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codegen.fft_persistent_codegen import generate_persistent_fft_kernel
from codegen.common import Emitter,emit_loop_twiddle_table
from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from planning.core.target_profile import DEFAULT_TARGET_PROFILE as TARGET
from planning.execution.fft_plan_persistent import make_persistent_leaf_plan
from planning.execution.kernel_registration import registered_kernel_names
from planning.gpu_baseline import clfft, rocfft_default, vkfft
from planning.gpu_baseline.common import (BaselineStatus, compute_lowering_overhead,
                                         reconstruct_gpu_logical_plan,
                                         reconstruct_m2ndp_physical_plan)
from planning.search.fft_cost_model import estimate_metrics
from verification.verify_fft_butterflies import verify_butterflies
from verification.verify_fft_harness import Ptr,_translate_emitted_lines
from verification.verify_fft_persistent import run_persistent_kernel
from verification.verify_fft_recursive import run_recursive_plan


class BaselineCodegenSupportTests(unittest.TestCase):
    def test_generated_twiddle_initialization_preserves_fp32_bits(self):
        class HostPtr:
            def __init__(self,n):
                self.arr = np.zeros(n,dtype=np.float32)
                self.bitcast = {int:lambda:self.arr.view(np.uint32)}
            def __setitem__(self,i,v):
                self.arr[i] = v
        for table in ([(0.5,-0.75)]*9,
                      list(zip(np.cos(np.arange(1024)*0.317),np.sin(np.arange(1024)*0.317)))):
            emitter = Emitter()
            emitter.add("    def initialize():")
            body = Emitter()
            emit_loop_twiddle_table(body,table=table,real_name="tw_real",imag_name="tw_imag")
            for line in body.lines:
                emitter.add("    "+line)
            emitter.add("        return tw_real, tw_imag")
            scope = dict(cxl_alloc={float:HostPtr},Float32=float,UInt32=int,Int=int,
                         String=lambda value:SimpleNamespace(unsafe_ptr=lambda:value.encode()))
            exec(_translate_emitted_lines(emitter.lines),scope)
            real,imag = scope["initialize"]()
            expected = np.asarray(table,dtype=np.float32)
            np.testing.assert_array_equal(real.arr.view(np.uint32),expected[:,0].copy().view(np.uint32))
            np.testing.assert_array_equal(imag.arr.view(np.uint32),expected[:,1].copy().view(np.uint32))

    def test_all_supported_butterfly_dft_matrices(self):
        verify_butterflies()

    def test_vkfft_original_composites_numeric(self):
        rng = np.random.default_rng(420)
        for n in (12, 14, 15, 24, 28, 60, 120, 240, 768, 960):
            x = rng.normal(size=3*n) + 1j*rng.normal(size=3*n)
            for inverse in (False, True):
                for planner in (vkfft.plan, vkfft.plan_m2ndp):
                    with self.subTest(n=n, inverse=inverse, planner=planner.__name__):
                        result = planner(n, batch=3, inverse=inverse)
                        self.assertEqual(result.status, BaselineStatus.OK, result.diagnostics)
                        [gpu] = reconstruct_gpu_logical_plan(result)
                        self.assertEqual(tuple(s.radix for s in result.plan.root.kernel.stages), gpu.radix_sequence)
                        expected = (np.fft.ifft if inverse else np.fft.fft)(x.reshape(3,n), axis=1).ravel()
                        np.testing.assert_allclose(run_recursive_plan(result.plan,x), expected, atol=1e-4,rtol=1e-5)

    def test_global_round_reuse_and_ragged_workers_in_all_modes(self):
        rng = np.random.default_rng(81)
        target = replace(TARGET, num_ndp_units=3, spad_capacity_bytes=96)
        for n, radices in ((15,(15,)), (28,(14,2)), (64,(4,4,4))):
            for workers, mode in ((3,"physical"), (10,"wave"), (17,"fused"), (512,"physical")):
                for inverse in (False,True):
                    with self.subTest(n=n, workers=workers, mode=mode, inverse=inverse):
                        batch = 8
                        plan = make_persistent_leaf_plan(n,radices,num_logical_blocks=batch,
                            target=target,workers_per_fft=workers,lowering_mode=mode,
                            storage_mode="global",inverse=inverse)
                        x = rng.normal(size=batch*n) + 1j*rng.normal(size=batch*n)
                        real, imag = Ptr(batch*n), Ptr(batch*n)
                        real.arr[:], imag.arr[:] = x.real, x.imag
                        # Input/output aliasing must remain safe while slots are
                        # reused over three rounds, including the last partial one.
                        def worker_order(phase, count):
                            return np.random.default_rng(phase+700).permutation(count)
                        run_persistent_kernel(plan,num_logical_blocks=batch,input_real=real,
                            input_imag=imag,output_real=real,output_imag=imag,
                            phase_worker_order=worker_order)
                        expected = (np.fft.ifft if inverse else np.fft.fft)(x.reshape(batch,n),axis=1).ravel()
                        np.testing.assert_allclose(real.arr+1j*imag.arr,expected,atol=1e-4,rtol=1e-5)
                        self.assertEqual(plan.scratchpad_buffers,())
                        self.assertTrue(all(b.elements == 2*n*3 for b in plan.workspace_buffers))
                        self.assertEqual(len(registered_kernel_names(plan)),len(radices)+1)

    def test_original_large_kernel_executes_without_changing_gpu_choice(self):
        n, batch = 8192, 2
        rng = np.random.default_rng(18)
        x = rng.normal(size=n*batch)+1j*rng.normal(size=n*batch)
        # NDP-adapted rocFFT selects the same large SBRR kernel at batch=32.
        for planner, actual_batch in ((vkfft.plan,batch),(rocfft_default.plan_m2ndp,32)):
            for inverse in (False,True):
                result = planner(n,batch=actual_batch,inverse=inverse)
                self.assertEqual(result.status,BaselineStatus.OK,result.diagnostics)
                kernel = result.plan.root.kernel
                self.assertEqual(kernel.persistent.storage_mode,"global")
                [gpu] = reconstruct_gpu_logical_plan(result)
                self.assertEqual(tuple(s.radix for s in kernel.stages),gpu.radix_sequence)
                self.assertEqual(kernel.persistent.workers_per_fft,gpu.gpu_threads_per_transform)
                # Run two replicas with exactly that kernel's radices and worker
                # mapping, reducing only the independent batch size for CPU QA.
                small = make_persistent_leaf_plan(n,tuple(s.radix for s in kernel.stages),
                    num_logical_blocks=batch,workers_per_fft=kernel.persistent.workers_per_fft,
                    storage_mode="global",inverse=inverse,loop_stages=kernel.persistent.loop_stages)
                expected = (np.fft.ifft if inverse else np.fft.fft)(x.reshape(batch,n),axis=1).ravel()
                np.testing.assert_allclose(run_recursive_plan(replace(result.plan,batch=batch,
                    root=replace(result.plan.root,r=batch,kernel=small)),x),expected,atol=1e-3,rtol=1e-5)

    def test_persistent_loops_preserve_banks_workers_and_rounds(self):
        rng = np.random.default_rng(160)
        target = replace(TARGET,num_ndp_units=3)
        n, batch, radices = 4096, 4, (4,4,4,4,4,4)
        x = rng.normal(size=n*batch)+1j*rng.normal(size=n*batch)
        for storage,mode,workers in (("global","wave",10),("global","fused",10),
                                      ("global","physical",16),("scratchpad","physical",3)):
            with self.subTest(storage=storage,mode=mode,workers=workers):
                plan = make_persistent_leaf_plan(n,radices,num_logical_blocks=batch,target=target,
                    workers_per_fft=workers,lowering_mode=mode,storage_mode=storage,loop_stages=True)
                real,imag,out_real,out_imag = (Ptr(n*batch) for _ in range(4))
                real.arr[:],imag.arr[:] = x.real,x.imag
                run_persistent_kernel(plan,num_logical_blocks=batch,input_real=real,input_imag=imag,
                    output_real=out_real,output_imag=out_imag,
                    phase_worker_order=lambda phase,count:np.random.default_rng(phase+18).permutation(count))
                np.testing.assert_allclose(out_real.arr+1j*out_imag.arr,np.fft.fft(x.reshape(batch,n),axis=1).ravel(),
                                           atol=1e-3,rtol=1e-5)
                # Assert the arithmetic actually looped; passing an unused
                # loop option would not establish correctness of this path.
                code = generate_persistent_fft_kernel(plan,num_logical_blocks=batch,target=target)
                self.assertTrue("while outer_it <" in code)

    def test_workspace_and_traffic_are_visible_in_codegen_and_reports(self):
        result = rocfft_default.plan_m2ndp(8192,batch=128)
        self.assertEqual(result.status,BaselineStatus.OK,result.diagnostics)
        kernel = result.plan.root.kernel
        code = generate_recursive_fft_kernels(result.plan)
        self.assertNotIn("def preload():",code)
        self.assertIn("var rc0_buf_a = cxl_alloc[Float32](524288)",code)
        self.assertIn("p.buf_b.load[width=",code)
        self.assertNotIn("scratchpad[16384, Float32",code)
        [record] = [json.loads(s.split(" ",2)[2]) for s in code.splitlines() if s.startswith("# FFT_EXECUTION ")]
        self.assertEqual(record["storage_mode"],"global")
        self.assertEqual(record["scratchpad_bytes"],32)
        self.assertEqual(record["phase_launches"],4*(len(kernel.stages)+1))
        self.assertEqual(record["workspace_bytes"],4*sum(b.elements for b in kernel.workspace_buffers))
        self.assertEqual(record["dram_bytes"],16*8192*128*len(kernel.stages))
        self.assertEqual(compute_lowering_overhead(result).full_array_traffic_ratio,len(kernel.stages))
        self.assertEqual(estimate_metrics(result.plan,TARGET).estimated_dram_bytes,record["dram_bytes"])
        self.assertEqual(estimate_metrics(result.plan,TARGET).persistent_copy_instructions,0)
        [physical] = reconstruct_m2ndp_physical_plan(result,target=TARGET)
        self.assertEqual(physical.storage_mode,"global")
        self.assertEqual(physical.workspace_bytes,record["workspace_bytes"])
        standalone = generate_persistent_fft_kernel(kernel,num_logical_blocks=128)
        self.assertTrue("Params(input_real, input_imag, output_real, output_imag, loop_twiddle_real, loop_twiddle_imag, buf_a, buf_b)" in standalone)
        self.assertTrue("while outer_it <" in standalone)

    def test_capacity_fallback_preserves_divisor_worker_count(self):
        target = replace(TARGET,spad_capacity_bytes=96)
        from planning.gpu_baseline.common import GPUKernelConfig, map_cooperative_kernel
        config = GPUKernelConfig("test",64,(4,4,4),{})
        result = map_cooperative_kernel(length=64,radices=(4,4,4),workers_per_fft=2,
            fft_slots_wanted=4,total_ffts=9,inverse=False,inverse_scale=None,
            kernel_name="GlobalTwoWorkers",target=target,gpu_config=config)
        self.assertEqual(result.status,BaselineStatus.OK,result.diagnostics)
        self.assertEqual(result.plan.persistent.workers_per_fft,2)
        self.assertEqual(result.plan.persistent.storage_mode,"global")
        tiny = replace(target,spad_capacity_bytes=16)
        refused = map_cooperative_kernel(length=64,radices=(4,4,4),workers_per_fft=2,
            fft_slots_wanted=4,total_ffts=9,inverse=False,inverse_scale=None,
            kernel_name="TooSmallTrackers",target=tiny,gpu_config=config)
        self.assertEqual(refused.status,BaselineStatus.RESOURCE_INFEASIBLE)


if __name__ == "__main__":
    unittest.main()
