"""Regression checks using emitted stages and the physical address interleave."""
from dataclasses import replace
import json
import types
import unittest

import numpy as np

from codegen.fft_persistent_codegen import generate_persistent_fft_kernel
from planning.core.fft_plan_core import _build_plan, layouts_for_radices, pingpong_needed
from planning.core.target_profile import DEFAULT_TARGET_PROFILE as DEFAULT
from planning.execution.fft_plan_cooperative import make_cooperative_leaf_plan
from planning.execution.fft_plan_persistent import make_persistent_leaf_plan
from planning.execution.launch_plan import resolve_fft_launch
from verification.verify_fft_harness import Ptr, _simd, _translate_stage, _translate_emitted_lines
from verification.verify_fft_persistent import run_persistent_kernel


def plain_plan(n=64, batch=65, radices=(4, 4, 4), cap=1024, inverse=False):
    return _build_plan(length=n, inverse=inverse, total_uthreads=batch, simd_lanes=8,
                       use_pingpong=pingpong_needed(len(radices)),
                       layouts=layouts_for_radices(n, radices, 8),
                       spad_capacity_bytes=cap)


def run_mapped_kernel(plan, x, *, target=DEFAULT, spread=True, in_place=False, loop=False):
    """One scratchpad per physical unit; no virtual max_uthread-sized groups."""
    launch = resolve_fft_launch(plan, target=target, spread_across_units=spread)
    plan = replace(plan, launch_mapping=launch)
    ir, ii = Ptr(x.size), Ptr(x.size)
    ir.arr[:], ii.arr[:] = x.real.ravel(), x.imag.ravel()
    out_r, out_i = (ir, ii) if in_place else (Ptr(x.size), Ptr(x.size))
    p = types.SimpleNamespace(input_real_base=ir, input_imag_base=ii,
                              output_real_base=out_r, output_imag_base=out_i)
    units = []
    for _ in range(target.num_ndp_units):
        ns = types.SimpleNamespace()
        for buf in plan.scratchpad_buffers:
            setattr(ns, buf.name, Ptr(buf.elements))
        units.append(ns)
    current = dict(gid=0, local=0)
    table, phases = [], []
    for stage in plan.stages:
        src, tail = _translate_stage(plan, stage, compute_lanes=4,
                                     loop_stages=loop, min_loop_batches=1,
                                     twiddle_table=table)
        ns = dict(Float32=float, Int=int, SIMD=_simd, N=plan.length, W=8,
                  WORKERS_PER_FFT=launch.workers_per_fft, p=p,
                  global_uthread_id=lambda: current['gid'],
                  local_uthread_id=lambda: current['local'])
        ns[f'MAX_UTHREAD_{plan.kernel_name}'] = plan.max_uthread
        exec(compile(src, '<mapped FFT stage>', 'exec'), ns)
        names = [f'stage_{stage.stage_id}']
        if tail:
            names.append(f'stage_{stage.stage_id}_tail')
        phases.extend((ns, ns[name]) for name in names)
    p.loop_twiddle_real_base, p.loop_twiddle_imag_base = Ptr(len(table)), Ptr(len(table))
    if table:
        p.loop_twiddle_real_base.arr[:] = [v[0] for v in table]
        p.loop_twiddle_imag_base.arr[:] = [v[1] for v in table]
    chunk, count = target.interleave_chunk_uthreads, target.num_ndp_units
    for round_ in launch.rounds():
        p.launch_fft_base, p.launch_fft_count = round_.fft_base, round_.fft_count
        for ns, fn in phases:
            for gid in range(round_.uthread_count):
                unit = (gid // chunk) % count
                current.update(gid=gid, local=(gid // (chunk * count)) * chunk + gid % chunk)
                ns[plan.kernel_name] = units[unit]
                fn()
    return (out_r.arr + 1j * out_i.arr).reshape(x.shape)


class ExecutionCodegenFixTests(unittest.TestCase):
    def data(self, batch, n):
        rng = np.random.default_rng(37)
        return rng.normal(size=(batch, n)) + 1j * rng.normal(size=(batch, n))

    def test_persistent_generated_host_setup_allocates_valid_launch_pool(self):
        # Execute the generated host allocation/initialization prefix, as well
        # as device stages. Stage-only tests cannot catch an unbound host name.
        for target in (DEFAULT, replace(DEFAULT, num_ndp_units=4,
                                       interleave_chunk_uthreads=16, mapping_stride_bytes=512)):
            plan = make_persistent_leaf_plan(64, (4, 4, 4), num_logical_blocks=9, target=target)
            source = generate_persistent_fft_kernel(plan, num_logical_blocks=9, target=target)
            prefix = source[source.index('def main() raises:'):source.index('    var rc = ')]
            prefix = prefix.replace('def main() raises:', 'def main():')
            lines = ['    ' + line for line in (prefix + '    return locals()\n').splitlines()]
            allocations = []
            class HostPtr(Ptr):
                def __int__(self):
                    return self.address
            def allocate(count):
                ptr = HostPtr(count)
                ptr.address = (allocations[-1].address + allocations[-1].arr.size * 4 + 64
                               if allocations else 0x10040)
                allocations.append(ptr)
                return ptr
            def from_address(*, unsafe_from_address):
                ptr = HostPtr(0)
                ptr.address = unsafe_from_address
                raw = next(p for p in allocations
                           if p.address <= ptr.address < p.address + p.arr.size * 4)
                ptr.arr = raw.arr[(ptr.address - raw.address) // 4:]
                return ptr
            class GenericCallable:
                def __init__(self, function):
                    self.function = function
                def __getitem__(self, _):
                    return self.function
            namespace = dict(Float32=float, Int=int, MutAnyOrigin=object,
                             cxl_alloc=GenericCallable(allocate),
                             UnsafePointer=GenericCallable(from_address),
                             seed=lambda _: None, random_float64=lambda low, high: 0.25)
            namespace[plan.kernel_name] = types.SimpleNamespace(emit_ir_if_asked=lambda: False)
            exec(compile(_translate_emitted_lines(lines), '<generated host setup>', 'exec'), namespace)
            host = namespace['main']()
            self.assertEqual(host['pool_elems'], plan.total_uthreads * plan.simd_lanes)
            self.assertEqual(int(host['uthread_pool']) % target.mapping_stride_bytes, 0)
            self.assertGreaterEqual(host['uthread_pool'].arr.size, host['pool_elems'])
            self.assertEqual(host['input_real'].arr.size, 64 * 9)

    def test_capped_plain_and_cooperative_cover_all_batches(self):
        for workers in (1, 2, 4, 8):
            for spread in (False, True):
                for inverse in (False, True):
                    with self.subTest(workers=workers, spread=spread, inverse=inverse):
                        plan = (plain_plan(inverse=inverse) if workers == 1 else
                                make_cooperative_leaf_plan(64, (4, 4, 4), workers_per_fft=workers,
                                                           total_ffts=65, inverse=inverse,
                                                           spad_capacity_bytes=1024))
                        x = self.data(65, 64)
                        actual = run_mapped_kernel(plan, x, spread=spread, in_place=workers > 1)
                        expected = np.fft.ifft(x) if inverse else np.fft.fft(x)
                        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)

    def test_loop_tail_and_custom_mapping(self):
        target = replace(DEFAULT, num_ndp_units=4, interleave_chunk_uthreads=16,
                         mapping_stride_bytes=512)
        for workers in (1, 4, 8):
            plan = (plain_plan(105, 33, (3, 5, 7), 4096) if workers == 1 else
                    make_cooperative_leaf_plan(105, (3, 5, 7), workers_per_fft=workers,
                                               total_ffts=33, spad_capacity_bytes=4096,
                                               interleave_chunk_uthreads=16))
            x = self.data(33, 105)
            np.testing.assert_allclose(run_mapped_kernel(plan, x, target=target, loop=True),
                                       np.fft.fft(x), rtol=1e-5, atol=1e-4)

    def test_cooperative_round_spreads_across_all_units(self):
        for workers in (2, 8):
            plan = make_cooperative_leaf_plan(1024, (8, 8, 16), workers_per_fft=workers,
                                              total_ffts=128, spad_capacity_bytes=65536,
                                              max_concurrent_scratchpad_bytes=65536)
            launch = resolve_fft_launch(plan, spread_across_units=True)
            self.assertEqual(len(list(launch.rounds())), 1)
            self.assertEqual(launch.round_fft_capacity, 128)
            self.assertEqual(launch.round_uthread_capacity, workers * 128)
            x = self.data(128, 1024)
            np.testing.assert_allclose(run_mapped_kernel(plan, x, loop=True),
                                       np.fft.fft(x), rtol=1e-5, atol=1e-4)

    def test_private_trackers_ignore_phase_worker_order(self):
        target = replace(DEFAULT, num_ndp_units=4)
        for workers in (3, 10, 16, 17):
            for mode in ('physical', 'fused', 'wave'):
                with self.subTest(workers=workers, mode=mode):
                    plan = make_persistent_leaf_plan(64, (4, 4, 4), num_logical_blocks=9,
                                                     workers_per_fft=workers, target=target,
                                                     lowering_mode=mode, inverse=workers == 17)
                    x = self.data(9, 64)
                    ir, ii, out_r, out_i = (Ptr(x.size) for _ in range(4))
                    ir.arr[:], ii.arr[:] = x.real.ravel(), x.imag.ravel()
                    rng = np.random.default_rng(5)
                    order = lambda phase, count: rng.permutation(count)
                    run_persistent_kernel(plan, num_logical_blocks=9, input_real=ir, input_imag=ii,
                                          output_real=out_r, output_imag=out_i, phase_worker_order=order,
                                          copy_mode='vectorized_contiguous', vector_width=3)
                    actual = (out_r.arr + 1j * out_i.arr).reshape(x.shape)
                    expected = np.fft.ifft(x) if workers == 17 else np.fft.fft(x)
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)

    def test_tracker_resource_boundary_and_codegen_override(self):
        for mode, required in (('physical', 1056), ('wave', 1088)):
            with self.assertRaises(ValueError):
                make_persistent_leaf_plan(64, (4, 4, 4), num_logical_blocks=1,
                                          workers_per_fft=16, lowering_mode=mode,
                                          target=replace(DEFAULT, spad_capacity_bytes=required - 1))
            plan = make_persistent_leaf_plan(64, (4, 4, 4), num_logical_blocks=1,
                                             workers_per_fft=16, lowering_mode=mode,
                                             target=replace(DEFAULT, spad_capacity_bytes=required))
            generate_persistent_fft_kernel(plan, num_logical_blocks=1,
                                            target=replace(DEFAULT, spad_capacity_bytes=required))
            if mode == 'physical':
                with self.assertRaises(ValueError):
                    generate_persistent_fft_kernel(plan, num_logical_blocks=1, mode='wave',
                                                    target=replace(DEFAULT, spad_capacity_bytes=required))

    def test_alignment_invalid_copy_width_and_actual_options(self):
        target = replace(DEFAULT, num_ndp_units=4, interleave_chunk_uthreads=16,
                         mapping_stride_bytes=512)
        plan = make_persistent_leaf_plan(64, (4, 4, 4), num_logical_blocks=9,
                                         workers_per_fft=32, target=target)
        for bad in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                generate_persistent_fft_kernel(plan, num_logical_blocks=9, target=target,
                                                copy_mode='vectorized_contiguous', vector_width=bad)
        code = generate_persistent_fft_kernel(plan, num_logical_blocks=9, target=target,
                                              mode='wave', copy_mode='vectorized_contiguous',
                                              vector_width=8)
        self.assertIn('// 512 * 512', code)
        record = json.loads(next(line.split(' ', 2)[2] for line in code.splitlines()
                                 if line.startswith('# FFT_EXECUTION ')))
        self.assertEqual(record['persistent_mode'], 'wave')
        self.assertEqual(record['vector_width'], 8)
        self.assertEqual(record['arithmetic_lanes'], [4, 2, 4])
        self.assertEqual(plan.persistent.lowering_mode, 'physical')
        self.assertIsNone(plan.launch_mapping)


if __name__ == '__main__':
    unittest.main()
