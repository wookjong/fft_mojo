"""Execute the actual fused wrappers/schedule with private per-unit arenas."""
from dataclasses import replace
import json
import random
import re
import types
import unittest
import numpy as np

from codegen.common import Emitter
from codegen.fft_task_fusion_codegen import emit_fused_task_structs
from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from make_fft_kernel import _plan_gpu_baseline_fft
from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.execution.fft_unit_parallel import distribute_plan
from planning.execution.task_fusion import make_task_fusion_plan, node_parameter_bytes, resolve_task_execution
from planning.strategies.fft_plan_recursive import PhysicalTransposePlan, flatten_recursive_node
from verification.verify_fft_harness import Ptr, _simd, _translate_emitted_lines

TARGET = replace(DEFAULT_TARGET_PROFILE, num_ndp_units=28)


def example(library='vkfft', n=8192, batch=1, inverse=False):
    return _plan_gpu_baseline_fft(n, planner=f'gpu-{library}-m2ndp', inverse=inverse,
        batch=batch, target=TARGET, gpu_execution='distributed', units_per_leaf='auto')


def run_fused(plan, values, *, base_unit=0, control=False, leaf_storage="cxl", batch_window="off",
              boundary_fusion="off", boundary_layout="legacy", near_group=1, far_group=1, leaf_phase_policy="stages", leaf_codegen_schedule="existing", spad_allocation="declared", far12_codelet="off"):
    fusion = make_task_fusion_plan(plan, target=TARGET, compute_lanes=4,
                                  narrow_middle_stages=True, control=control,
                                  leaf_storage=leaf_storage, batch_window=batch_window,
                                  boundary_fusion=boundary_fusion,boundary_layout=boundary_layout,near_group=near_group,far_group=far_group,leaf_phase_policy=leaf_phase_policy,leaf_codegen_schedule=leaf_codegen_schedule,spad_allocation=spad_allocation,far12_codelet=far12_codelet)
    assert fusion.supported, fusion.reason
    if fusion.window_batch:
        from planning.execution.task_fusion import window_execution_plan
        execution = window_execution_plan(plan, fusion.window_batch)
    else:
        execution = plan
    nodes = flatten_recursive_node(execution.root)
    if boundary_fusion != 'off' or boundary_layout == 'grouped':
        nodes = resolve_task_execution(plan, fusion).nodes
    # Consume the emitted host Params arguments, rather than inventing the
    # expected pointers in the checker. This catches a correct device mapping
    # paired with an incorrect host output pointer.
    host_source = generate_recursive_fft_kernels(plan, target=TARGET, compute_lanes=4,
        narrow_middle_stages=True, reference_check=False, task_fusion='auto',
        task_fusion_control=control, leaf_storage=leaf_storage, batch_window=batch_window,
        boundary_fusion=boundary_fusion,boundary_layout=boundary_layout,near_group=near_group,far_group=far_group,leaf_phase_policy=leaf_phase_policy,leaf_codegen_schedule=leaf_codegen_schedule,spad_allocation=spad_allocation,far12_codelet=far12_codelet)
    host_values = {i: re.findall(re.escape(node.kernel_name) + r'Params\(([^()]*)\)', host_source)[-1].split(', ')
                   for i, node in enumerate(nodes)}
    emitter = Emitter()
    tables = emit_fused_task_structs(emitter, nodes=nodes, fusion=fusion,
                    compute_lanes=4, narrow_middle_stages=True, target=TARGET)
    writers, final_writers, state = set(), set(), {'gid': 0, 'private_offset': None, 'private_end': None, 'active_regions': None, 'phase': None}
    private_owners = {}

    class Checked(Ptr):
        def __init__(self, size, track=True):
            super().__init__(size)
            self.arr[:] = np.nan
            self.track = track

        def arena_access(self, offset, width, write):
            if not getattr(self,'is_arena',False) or state['active_regions'] is None:return
            candidates=[r for r in state['active_regions'] if r.offset<=offset and offset+width<=r.end]
            assert len(candidates)==1,('access outside live region',state['phase'],offset,width)
            region=candidates[0]
            assert state['phase'] in (region.writes if write else region.reads),('wrong access kind',region.key,write)
            if region.kind=='private':
                assert (offset-region.offset)//region.worker_stride == state['gid']%8,('private worker mismatch',offset)
                assert (offset+width-1-region.offset)//region.worker_stride == state['gid']%8

        def load(self, offset, width):
            self.arena_access(int(offset),int(width),False)
            assert 0 <= int(offset) <= len(self.arr) - int(width), ('load bounds', offset, width)
            result = super().load(offset, width)
            assert np.isfinite(result).all(), ('uninitialized load', offset, width)
            return result

        def store(self, offset, value):
            size = np.asarray(value).size
            self.arena_access(int(offset),int(size),True)
            assert 0 <= int(offset) <= len(self.arr) - size, ('store bounds', offset, size)
            for index in range(int(offset), int(offset) + size) if self.track else ():
                key = id(self), index
                if state['private_offset'] is not None and getattr(self,'is_arena',False) and state['private_offset'] <= index < state['private_end']:
                    assert key not in private_owners or private_owners[key] == state['gid'], ('private scratch owner collision',index)
                    private_owners[key] = state['gid']
                else:
                    assert key not in writers, ('multiple writers', state['gid'], index)
                writers.add(key)
                if getattr(self, 'final_output', False):
                    assert key not in final_writers, ('multiple final writers across windows', index)
                    final_writers.add(key)
            super().store(offset, value)

    total = len(values)
    buffers = {}
    for name in ('input', 'work0', 'work1', 'output'):
        for part in ('real', 'imag'):
            buffers[f'{name}_{part}'] = Checked(total)
            buffers[f'{name}_{part}'].final_output = name == 'output'
    buffers['input_real'].arr[:] = values.real
    buffers['input_imag'].arr[:] = values.imag

    def buffer(index, part):
        name = 'input' if index < 0 else 'output' if index == len(nodes)-1 else f'work{index % 2}'
        return buffers[f'{name}_{part}']

    if any(not isinstance(node, PhysicalTransposePlan) and node.boundary_twiddle for node in nodes):
        a,b = execution.root.a, execution.root.b
        j,k = np.indices((a,b))
        tw = np.exp((2j if plan.inverse else -2j)*np.pi*j*k/plan.n).ravel()
        for part,data in (('real',tw.real),('imag',tw.imag)):
            ptr = Checked(plan.n, False)
            ptr.arr[:] = data
            buffers[f'boundary_twiddle_{part}'] = ptr
    params = types.SimpleNamespace()
    for index, node in enumerate(nodes):
        p = types.SimpleNamespace()
        if isinstance(node, PhysicalTransposePlan):
            for field, value in zip(('src_real_base', 'src_imag_base', 'dst_real_base', 'dst_imag_base'), host_values[index][:4]):
                setattr(p, field, buffers[value])
            p.round_offset = 0
            if node.twiddle_modulus:
                r, c = np.indices((node.rows, node.cols))
                tw = np.exp((2j if node.inverse else -2j) * np.pi * r*c / node.twiddle_modulus).ravel()
                for part, data in [('real', tw.real), ('imag', tw.imag)]:
                    ptr = Checked(len(tw), False)
                    ptr.arr[:] = data
                    setattr(p, f'twiddle_{part}_base', ptr)
        else:
            for field, value in zip(('input_real_base', 'input_imag_base', 'output_real_base', 'output_imag_base'), host_values[index][:4]):
                setattr(p, field, buffers[value])
            if node.boundary_twiddle:
                for part,value in zip(('real','imag'),host_values[index][4:6]):
                    setattr(p, f'boundary_twiddle_{part}_base', buffers[value])
            for b in node.workspace_buffers:
                setattr(p, b.name, Checked(b.elements))
            table = tables[index] or [(0, 0)]
            for component, part in enumerate(('real', 'imag')):
                ptr = Checked(len(table), False)
                ptr.arr[:] = [pair[component] for pair in table]
                setattr(p, f'loop_twiddle_{part}_base', ptr)
        setattr(params, f'node_{index}', p)

    # Extract the real emitted method bodies and real device_main call order.
    methods, owner, name, body = {}, None, None, []
    def finish():
        if name is not None:
            methods[(owner, name)] = list(body)
    for line in emitter.lines:
        struct = re.match(r'struct (\w+)\(NDPTask\):', line)
        definition = re.match(r'    def (\w+)\(\):', line)
        if struct:
            finish()
            owner, name, body = struct[1], None, []
        elif definition:
            finish()
            name, body = definition[1], [line]
        elif name is not None and (line.startswith('        ') or not line.strip()):
            body.append(line)
    finish()
    rng = random.Random(29)
    for group in fusion.groups:
        arenas = [Checked(fusion.arena_elements, True) for _ in range(28)]
        for arena in arenas: arena.is_arena = True
        task = types.SimpleNamespace(params=params, arena=arenas[0], batch_base=Checked(1, False))
        namespace = dict(Float32=float, Int=int, SIMD=_simd, global_uthread_id=lambda: state['gid'],
                         **{group.name: task})
        for function in group.functions:
            lines = methods[(group.name, function)]
            # Keep the real Params binding, adapting only Mojo reference syntax.
            lines = [line.replace('ref p = ', 'var p = ').replace('.params[].', '.params.') for line in lines]
            lines = [line.replace('DType.int32, 1', 'DType.float32, 1') for line in lines]
            translated = _translate_emitted_lines(lines)
            exec(compile(translated, f'<{group.name}.{function}>', 'exec'), namespace)
        schedule = '\n'.join(methods[(group.name, 'device_main')])
        calls = re.findall(r'launch_parallel\[' + group.name + r'\.(\w+)\]', schedule)
        assert tuple(calls) == group.functions
        for batch_base in range(0, plan.batch, fusion.window_batch or plan.batch):
            task.batch_base.arr[0] = batch_base
            for phase_index,function in enumerate(calls):
                writers.clear()
                private_owners.clear()
                state['private_offset'] = (next(p.scratch_offset for p in fusion.private_phases if p)
                    if fusion.private_phases and function.endswith('stage_1_2') else None)
                current_private=next((p for p in fusion.private_phases if p),None)
                state['private_end']=(state['private_offset']+current_private.scratch_elements
                    if state['private_offset'] is not None else None)
                state['phase']=phase_index
                memory=fusion.memory_allocation
                if memory is not None:
                    assert memory.phases[phase_index].name==function
                    state['active_regions']=[r for r in memory.regions if r.first<=phase_index<=r.last]
                    # New object epochs must not inherit values from a dead
                    # overlapping object or a prior batch window.
                    for r in state['active_regions']:
                        if r.first==phase_index:
                            for arena in arenas:arena.arr[r.offset:r.end]=np.nan
                order = list(range(fusion.common_uthreads))
                rng.shuffle(order)
                for gid in order:
                    state['gid'] = gid
                    unit = (base_unit + gid // 8) % 28
                    task.arena = arenas[unit]
                    namespace[function]()
    result = buffers['output_real'].arr + 1j * buffers['output_imag'].arr
    assert np.isfinite(result).all()
    assert len(final_writers) == 2 * total, 'each final output must be written exactly once'
    np.testing.assert_array_equal(buffers['input_real'].arr, values.real)
    np.testing.assert_array_equal(buffers['input_imag'].arr, values.imag)
    return result


class TaskFusionTests(unittest.TestCase):
    def test_resources_and_registration_partition(self):
        for lib in ('clfft', 'vkfft'):
            for n in (8192, 16384):
                plan = example(lib, n)
                fused = make_task_fusion_plan(plan, target=TARGET, compute_lanes=4, narrow_middle_stages=True)
                self.assertTrue(fused.supported, fused.reason)
                self.assertEqual(len(fused.groups), 1 if lib == 'vkfft' else 2)
                self.assertEqual(fused.common_uthreads, 224 if n == 8192 else 448)
                self.assertLessEqual(fused.reserved_bytes, 122880)
                self.assertTrue(all(len(g.functions) <= 8 for g in fused.groups))
                source = generate_recursive_fft_kernels(plan, target=TARGET, compute_lanes=4,
                    narrow_middle_stages=True, reference_check=False, task_fusion='auto')
                self.assertEqual(source.count('.launch('), len(fused.groups))
                self.assertEqual(source.count('comptime arena = scratchpad['), len(fused.groups))
                self.assertNotIn('comptime tile_buf = scratchpad[', source)
                self.assertNotIn('set_task_range', source)
                self.assertEqual(sum(node_parameter_bytes(x) for x in flatten_recursive_node(plan.root)),
                                 fused.params_bytes)
                control = make_task_fusion_plan(plan, target=TARGET, compute_lanes=4,
                    narrow_middle_stages=True, control=True)
                self.assertEqual(control.control_pool_ids,
                                 (0, 0, 0, 0, 0) if lib == 'vkfft' else (0, 0, 0, 1, 1))
                control_source = generate_recursive_fft_kernels(plan, target=TARGET, compute_lanes=4,
                    narrow_middle_stages=True, reference_check=False, task_fusion='auto',
                    task_fusion_control=True)
                self.assertEqual(control_source.count('.launch('), 5)
                self.assertEqual(control_source.count('_elems = '), source.count('_elems = '))
                self.assertEqual(control_source.count('reserved_arena.store('), 2)

    def test_real_emitted_wrappers_forward_inverse(self):
        for lib in ('clfft', 'vkfft'):
            for n in (8192, 16384):
                for inverse in (False, True):
                    with self.subTest(lib=lib, n=n, inverse=inverse):
                        rng = np.random.default_rng(72)
                        values = rng.normal(size=n) + 1j*rng.normal(size=n)
                        actual = run_fused(example(lib, n, inverse=inverse), values, base_unit=17)
                        expected = (np.fft.ifft if inverse else np.fft.fft)(values)
                        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-5)

    def test_leaf_resident_spad_matches_cxl_and_sizes_the_shared_arena(self):
        # Exact bytes_per_unit from the leaf-SPAD feasibility report (N,B=1):
        # near/far banks peak at 6144B (N=8192) / 12288B (N=16384) either lib.
        for lib in ('clfft', 'vkfft'):
            for n, arena_bytes in ((8192, 6144), (16384, 12288)):
                with self.subTest(lib=lib, n=n):
                    plan = example(lib, n)
                    fusion = make_task_fusion_plan(plan, target=TARGET, compute_lanes=4,
                                     narrow_middle_stages=True, leaf_storage='spad')
                    self.assertTrue(fusion.supported, fusion.reason)
                    self.assertEqual(fusion.arena_elements * 4, arena_bytes)
                    self.assertEqual(sum(r is not None for r in fusion.leaf_storage), 2)
                    source = generate_recursive_fft_kernels(plan, target=TARGET, compute_lanes=4,
                        narrow_middle_stages=True, reference_check=False, task_fusion='auto',
                        leaf_storage='spad')
                    self.assertNotIn('buf_a: UnsafePointer', source)
                    self.assertNotIn('buf_b: UnsafePointer', source)
                    for inverse in (False, True):
                        rng = np.random.default_rng(72)
                        values = rng.normal(size=n) + 1j*rng.normal(size=n)
                        actual = run_fused(example(lib, n, inverse=inverse), values,
                                          base_unit=17, leaf_storage='spad')
                        expected = (np.fft.ifft if inverse else np.fft.fft)(values)
                        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-5)
        for lib in ('clfft', 'vkfft'):
            plan = example(lib, 8192, batch=4)
            rng = np.random.default_rng(93)
            values = rng.normal(size=32768) + 1j*rng.normal(size=32768)
            expected = np.fft.fft(values.reshape(4, 8192), axis=1).ravel()
            actual = run_fused(plan, values, base_unit=27, leaf_storage='spad')
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=3e-5)

    def test_leaf_resident_spad_rejects_multi_unit_leaves_auto_falls_back(self):
        for lib in ('clfft', 'vkfft'):
            forced = distribute_plan(example(lib, 8192), target=TARGET, units_per_leaf=2)
            spad = make_task_fusion_plan(forced, target=TARGET, compute_lanes=4,
                                        narrow_middle_stages=True, leaf_storage='spad')
            self.assertFalse(spad.supported)
            self.assertIn('units_per_leaf', spad.reason)
            auto = make_task_fusion_plan(forced, target=TARGET, compute_lanes=4,
                                        narrow_middle_stages=True, leaf_storage='auto')
            self.assertTrue(auto.supported, auto.reason)
            self.assertTrue(all(r is None for r in auto.leaf_storage))
        with self.assertRaisesRegex(ValueError, 'task_fusion=auto'):
            generate_recursive_fft_kernels(example(), task_fusion='off', leaf_storage='spad')
        with self.assertRaisesRegex(ValueError, 'not supported'):
            generate_recursive_fft_kernels(
                distribute_plan(example('clfft', 8192), target=TARGET, units_per_leaf=2),
                target=TARGET, task_fusion='auto', leaf_storage='spad')

    def test_leaf_storage_auto_preserves_fusion_when_spad_exceeds_budget(self):
        for lib in ('clfft', 'vkfft'):
            plan = example(lib, 16384, batch=8)
            options = dict(target=TARGET, compute_lanes=4, narrow_middle_stages=True)
            strict = make_task_fusion_plan(plan, leaf_storage='spad', **options)
            self.assertFalse(strict.supported)
            self.assertIn('scratchpad budget', strict.reason)
            for control in (False, True):
                cxl = make_task_fusion_plan(plan, leaf_storage='cxl', control=control, **options)
                auto = make_task_fusion_plan(plan, leaf_storage='auto', control=control, **options)
                self.assertTrue(auto.supported, auto.reason)
                self.assertEqual(auto, cxl)
                for spread in (False, True):
                    render = dict(**options, reference_check=False, task_fusion='auto',
                                  task_fusion_control=control, spread_across_units=spread)
                    expected = generate_recursive_fft_kernels(plan, leaf_storage='cxl', **render)
                    actual = generate_recursive_fft_kernels(plan, leaf_storage='auto', **render)
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual.count('.launch('), 5 if control else
                                     (1 if lib == 'vkfft' else 2))
            # A CXL plan that also exceeds the budget must still reject fusion.
            oversized = example(lib, 16384, batch=64)
            self.assertFalse(make_task_fusion_plan(oversized, leaf_storage='auto', **options).supported)

    def test_layout_matched_control_and_batch(self):
        plan = example('clfft', 8192, batch=4)
        rng = np.random.default_rng(9)
        values = rng.normal(size=32768) + 1j*rng.normal(size=32768)
        expected = np.fft.fft(values.reshape(4, 8192), axis=1).ravel()
        for control in (False, True):
            actual = run_fused(plan, values, base_unit=27, control=control)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=3e-5)

    def test_explicit_fallback_and_invalid_mode(self):
        for plan in (example(n=1024), example(n=8192, batch=64)):
            fusion = make_task_fusion_plan(plan, target=TARGET)
            self.assertFalse(fusion.supported)
            source = generate_recursive_fft_kernels(plan, target=TARGET, compute_lanes=4,
                narrow_middle_stages=True, reference_check=False, task_fusion='auto')
            record = next(json.loads(line.split(' ', 2)[2]) for line in source.splitlines()
                          if line.startswith('# FFT_TASK_FUSION '))
            self.assertFalse(record['supported'])
            self.assertTrue(record['reason'])
        with self.assertRaisesRegex(ValueError, 'off or auto'):
            generate_recursive_fft_kernels(example(), task_fusion='unknown')


if __name__ == '__main__':
    unittest.main()
