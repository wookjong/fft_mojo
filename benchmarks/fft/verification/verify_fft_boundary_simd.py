"""Reference math for new plan objects, not machine/FP32 emitter verification."""
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch
import random
import unittest
import numpy as np
from planning.execution.fft_boundary_simd import make_boundary_simd_plan, extract_scalar_stages
from planning.execution.fft_leaf_ownership import LeafOwnershipPlan
from planning.execution.fft_boundary_fusion import make_boundary_fusion_plan
from verification.verify_fft_leaf_ownership import reference_owners
from verification.verify_fft_task_fusion import example

CASE_RECORDS=[]


def original_stage(stage, data):
    """Execute the old SIMD IR as independent scalar radix DFTs."""
    result=np.full_like(data,complex(np.nan,np.nan))
    for batch in stage.batches:
        for lane in range(batch.valid_lanes):
            indices=[x.base_offset+lane if x.mode=='vector' else x.packed_lane_offsets[lane] for x in batch.loads]
            values=(np.fft.ifft(data[indices])*stage.radix if stage.inverse else np.fft.fft(data[indices]))
            for output in batch.outputs:
                value=values[output.output]
                if output.twiddle is not None:
                    value*=complex(output.twiddle.real[lane],output.twiddle.imag[lane])
                if output.scale is not None:value*=output.scale
                store=output.store
                index=store.base_offset+lane if store.mode=='vector' else store.lane_offsets[lane]
                result[index]=value
    assert np.isfinite(result).all()
    return result


def exercise_boundary(plan,kernel,*,window_base=0,live_batch=None):
    """Drive concrete global segments/local addresses from actual new plan objects."""
    live=min(plan.ownership.window_batch,plan.total_batch-window_base) if live_batch is None else live_batch
    rng=np.random.default_rng(601)
    source=rng.normal(size=plan.total_batch*plan.n)+1j*rng.normal(size=plan.total_batch*plan.n)
    p=plan.ownership
    expected_owners,unused=reference_owners(p,live)
    full_owners,max_slots=reference_owners(p,p.window_batch)
    bank_elements=max_slots*2*p.leaf_length
    arena=np.full((p.num_units,p.total_elements),np.nan)
    output=np.full(plan.total_batch*plan.n,complex(np.nan,np.nan))
    stage=kernel.stages[0] if plan.boundary=='pre' else kernel.stages[-1]

    def reference_offset(replica,element,bank,plane):
        unit,slot=expected_owners[replica]
        return unit,p.bank_names.index(bank)*bank_elements+slot*2*p.leaf_length+plane*p.leaf_length+element

    if plan.boundary=='post':
        bank=stage.batches[0].loads[0].buffer_name
        for replica in expected_owners:
            for element in range(p.leaf_length):
                value=source[replica*p.leaf_length+element]
                for plane,part in ((0,value.real),(1,value.imag)):
                    unit,off=reference_offset(replica,element,bank,plane)
                    arena[unit,off]=part
    resolved=list(plan.iter_resolved_jobs(window_base=window_base,live_batch=live))
    random.Random(17).shuffle(resolved)  # No hidden dependency on another worker in this phase.
    written=set()
    for concrete in resolved:
        job=plan.jobs[concrete.job_id]
        operands=np.full((stage.radix,plan.arithmetic_width),complex(np.nan,np.nan))
        if plan.boundary=='pre':
            for segment in concrete.global_segments:
                operands[segment.value_id,segment.lane_start:segment.lane_start+segment.width]=source[list(segment.offsets)]
        else:
            for access in concrete.local_accesses:
                old=operands[access.value_id,access.vector_lane]
                value=arena[concrete.unit,access.resident_offset]
                if access.plane==0:
                    operands[access.value_id,access.vector_lane]=complex(value,old.imag)
                else:
                    operands[access.value_id,access.vector_lane]=complex(old.real,value)
        assert np.isfinite(operands).all(), 'Uninitialized local read'
        values=np.fft.ifft(operands,axis=0)*stage.radix if stage.inverse else np.fft.fft(operands,axis=0)
        for lane in job.lanes:
            for out in lane.butterfly.outputs:
                if out.twiddle is not None:
                    values[out.output,lane.vector_lane]*=complex(*out.twiddle)
        if plan.inverse_scale is not None:values*=plan.inverse_scale
        if plan.boundary=='pre':
            for access in concrete.local_accesses:
                key=concrete.unit,access.resident_offset
                assert key not in written,'Duplicate resident writer'
                written.add(key)
                value=values[access.value_id,access.vector_lane]
                arena[key]=value.real if access.plane==0 else value.imag
        else:
            for segment in concrete.global_segments:
                for delta,index in enumerate(segment.offsets):
                    assert index not in written,'Duplicate global writer'
                    written.add(index)
                    output[index]=values[segment.value_id,segment.lane_start+delta]
    if plan.boundary=='pre':
        bank=stage.batches[0].outputs[0].store.buffer_name
        for replica in expected_owners:
            q,j=divmod(replica,plan.a)
            indices=[(window_base+q)*plan.n+t*plan.a+j for t in range(plan.b)]
            expected=original_stage(stage,source[indices])
            actual=[]
            for element in range(plan.b):
                u,r=reference_offset(replica,element,bank,0)
                _,i=reference_offset(replica,element,bank,1)
                actual.append(complex(arena[u,r],arena[u,i]))
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
        assert len(written)==live*plan.n*2
    else:
        expected=np.empty(live*plan.n,dtype=complex)
        for replica in expected_owners:
            q,k=divmod(replica,plan.b)
            values=original_stage(stage,source[replica*plan.a:(replica+1)*plan.a])
            if stage.inverse:values/=plan.n
            expected[q*plan.n+k:(q+1)*plan.n:plan.b]=values
        np.testing.assert_allclose(output[window_base*plan.n:(window_base+live)*plan.n],expected,rtol=1e-12,atol=1e-12)
        assert np.isnan(output[:window_base*plan.n]).all()
        assert np.isnan(output[(window_base+live)*plan.n:]).all()
        assert len(written)==live*plan.n
    CASE_RECORDS.append(dict(n=plan.n,boundary=plan.boundary,group=p.group_leaves,
        inverse=stage.inverse,total_batch=plan.total_batch,window=p.window_batch,
        window_base=window_base,live_batch=live,reference='complex128 CPU stage DFT',status='PASS'))


def case(n,batch,inverse,boundary,c,window):
    root=example('vkfft',n,batch,inverse).root
    k=root.near_fft.kernel if boundary=='pre' else root.far_child.kernel
    p=make_boundary_simd_plan(k,boundary=boundary,a=root.a,b=root.b,total_batch=batch,
                             group_leaves=c,window_batch=window)
    return p,k


class BoundarySIMDTests(unittest.TestCase):
    def test_reference_math_and_random_worker_order(self):
        for n in (8192,16384):
            for inverse in (False,True):
                for boundary,c in (('pre',4),('post',2),('post',4)):
                    with self.subTest(n=n,inverse=inverse,boundary=boundary,c=c):
                        p,k=case(n,1,inverse,boundary,c,1)
                        exercise_boundary(p,k)

    def test_window_base_and_tail(self):
        for boundary,c in (('pre',4),('post',2),('post',4)):
            for inverse in (False,True):
                p,k=case(16384,3,inverse,boundary,c,2)
                exercise_boundary(p,k,window_base=0,live_batch=2)
                exercise_boundary(p,k,window_base=2,live_batch=1)
                p,k=case(16384,8,inverse,boundary,c,4)
                exercise_boundary(p,k,window_base=4)

    def test_c2d2_keeps_four_arithmetic_lanes_and_two_segments(self):
        p,k=case(16384,1,False,'post',2,1)
        self.assertEqual(p.arithmetic_width,4)
        self.assertEqual(p.butterflies_per_vector,2)
        self.assertEqual([(x.leaf_lane,x.butterfly.butterfly_id) for x in p.jobs[0].lanes],[(0,0),(1,0),(0,1),(1,1)])
        job=p.resolve_job(0,0)
        self.assertEqual(len(job.global_segments),8)
        self.assertEqual([x.width for x in job.global_segments],[2]*8)
        self.assertEqual(job.global_segments[0].offsets,(0,1))
        self.assertEqual(job.global_segments[4].offsets,(64,65))

    def test_stage_ir_is_exact_and_immutable(self):
        p,k=case(16384,1,False,'pre',4,1)
        stages=extract_scalar_stages(k)
        for s,original in zip(stages,k.stages):
            for bf in s.butterflies:
                b=next(x for x in original.batches if x.batch_id==bf.original_batch)
                for out,old in zip(bf.outputs,b.outputs):
                    tw=None if old.twiddle is None else (old.twiddle.real[bf.original_lane],old.twiddle.imag[bf.original_lane])
                    self.assertEqual(out.twiddle,tw)
        with self.assertRaises(FrozenInstanceError):p.jobs[0].lanes[0].leaf_lane=2
        with self.assertRaises(FrozenInstanceError):stages[0].butterflies[0].outputs[0].offset=4

    def test_rebound_or_unsupported_ir_rejected(self):
        root=example('vkfft',16384).root
        original=example('vkfft',16384)
        rebound=make_boundary_fusion_plan(original,mode='pre')
        near=next(x for x in rebound.nodes if not hasattr(x,'tile_rows'))
        with self.assertRaisesRegex(ValueError,'canonical offsets'):extract_scalar_stages(near)
        k=root.near_fft.kernel
        for bad in (replace(k,distributed=None),replace(k,boundary_twiddle=('store',256,64))):
            with self.assertRaises(ValueError):extract_scalar_stages(bad)
        stage=k.stages[0]
        batch=stage.batches[0]
        bad_load=replace(batch.loads[0],base_offset=1)
        bad_stage=replace(stage,batches=(replace(batch,loads=(bad_load,*batch.loads[1:])),))
        with self.assertRaisesRegex(ValueError,'cover'):extract_scalar_stages(replace(k,stages=(bad_stage,*k.stages[1:])))

    def test_invalid_shapes_windows_jobs(self):
        p,k=case(16384,3,False,'pre',4,2)
        for kwargs in (dict(a=128),dict(group_leaves=2),dict(a=255),dict(window_batch=4),dict(boundary='middle')):
            args=dict(boundary='pre',a=256,b=64,total_batch=3,group_leaves=4,window_batch=2);args.update(kwargs)
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):make_boundary_simd_plan(k,**args)
        with self.assertRaises(ValueError):p.resolve_job(0,0,window_base=3)
        with self.assertRaises(ValueError):p.resolve_job(0,0,window_base=2,live_batch=2)
        with self.assertRaises(ValueError):p.resolve_job(64,0,window_base=2)
        with self.assertRaises(ValueError):p.resolve_job(0,99)

    def test_unwritten_bank_and_inconsistent_direction_rejected(self):
        p,k=case(16384,1,False,'pre',4,1)
        stage=k.stages[1]
        bad_batches=tuple(replace(b,loads=tuple(replace(x,buffer_name='buf_b') for x in b.loads)) for b in stage.batches)
        with self.assertRaisesRegex(ValueError,'source/bank'):
            extract_scalar_stages(replace(k,stages=(k.stages[0],replace(stage,batches=bad_batches))))
        with self.assertRaises(ValueError):extract_scalar_stages(replace(k,inverse=True))

    def test_resident_budget_and_approved_model(self):
        for boundary,c,batch,w,expected in (('pre',4,1,1,12288),('pre',4,8,4,40960),
                                           ('post',2,1,1,16384),('post',4,8,4,49152)):
            p,k=case(16384,batch,False,boundary,c,w)
            self.assertEqual(p.ownership.arena_bytes,expected)
        with self.assertRaisesRegex(ValueError,'scratchpad budget'):case(16384,8,False,'post',4,8)

    def test_negative_wrong_legacy_slot_is_detected(self):
        p,k=case(8192,1,False,'pre',4,1)
        original=LeafOwnershipPlan.resident_index
        def wrong(self,replica,element,*,bank,plane,live_batch=None):
            owner=self.resolve(replica,live_batch=live_batch)
            return original(self,replica,element,bank=bank,plane=plane,live_batch=live_batch)+(replica//self.num_units-owner.leaf_slot)*2*self.leaf_length
        with patch.object(LeafOwnershipPlan,'resident_index',wrong):
            with self.assertRaises((AssertionError,IndexError)):exercise_boundary(p,k)

    def test_negative_wrong_lane_ownership_is_detected(self):
        p,k=case(8192,1,False,'pre',4,1)
        job=p.jobs[0]
        bad_lanes=(replace(job.lanes[0],leaf_lane=1),replace(job.lanes[1],leaf_lane=0),*job.lanes[2:])
        bad=replace(p,jobs=(replace(job,lanes=bad_lanes),*p.jobs[1:]))
        with self.assertRaises(AssertionError):exercise_boundary(bad,k)

    def test_negative_wrong_twiddle_is_detected(self):
        p,k=case(8192,1,False,'pre',4,1)
        job=p.jobs[1]
        bf=job.lanes[0].butterfly
        out=replace(bf.outputs[1],twiddle=(0.0,0.0))
        bad_bf=replace(bf,outputs=(bf.outputs[0],out,*bf.outputs[2:]))
        bad_job=replace(job,lanes=tuple(replace(l,butterfly=bad_bf) for l in job.lanes))
        bad=replace(p,jobs=(p.jobs[0],bad_job,*p.jobs[2:]))
        with self.assertRaises(AssertionError):exercise_boundary(bad,k)

    def test_negative_missing_inverse_scale_is_detected(self):
        p,k=case(8192,1,True,'post',2,1)
        with self.assertRaises(AssertionError):exercise_boundary(replace(p,inverse_scale=None),k)


if __name__=='__main__':
    unittest.main()
