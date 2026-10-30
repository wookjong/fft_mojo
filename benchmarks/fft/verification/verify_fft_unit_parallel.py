"""Numerically execute emitted cross-unit stages, including exchange and tails."""
import random
import types
import unittest

import numpy as np

from codegen.common import Emitter
from codegen.fft_distributed_codegen import emit_distributed_phase
from planning.execution.distributed_phases import resolve_distributed_phases
from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.execution.fft_unit_parallel import (
    apply_gpu_execution, distribute_plan, make_split_leaf_plan,
)
from planning.gpu_baseline import clfft, rocfft_default, vkfft
from planning.strategies.fft_plan_recursive import flatten_recursive_node, PhysicalTransposePlan
from verification.verify_fft_harness import Ptr, _simd, _translate_emitted_lines
from verification.verify_fft_recursive import run_recursive_plan
from dataclasses import replace

TARGET = replace(DEFAULT_TARGET_PROFILE, num_ndp_units=28)


def run_distributed_kernel(plan, values, *, compute_lanes=4, base_unit=0,
                           narrow_middle_stages=False):
    """Reexecute actual emitted source with shuffled workers per global phase.

    CXL banks are shared, unlike private per-unit scratchpads. Bounds and
    duplicate writers are checked independently of the generated dispatch.
    """
    current = {"gid": 0}
    writers = {}
    class CheckedPtr(Ptr):
        def __init__(self, size, *, track_writers=True):
            super().__init__(size)
            self.track_writers = track_writers
        def load(self, offset, width):
            if not 0 <= int(offset) <= len(self.arr) - int(width):
                raise AssertionError(f"out-of-bounds load {offset} + {width}")
            result = super().load(offset, width)
            if not np.all(np.isfinite(result)):
                raise AssertionError(f"uninitialized load {offset} + {width}")
            return result
        def store(self, offset, value):
            size = np.asarray(value).size
            if not 0 <= int(offset) <= len(self.arr) - size:
                raise AssertionError(f"out-of-bounds store {offset} + {size}")
            for index in range(int(offset), int(offset) + size) if self.track_writers else ():
                key = (id(self), index)
                if key in writers:
                    raise AssertionError(f"two writers for element {index}")
                writers[key] = current["gid"]
            super().store(offset, value)
    params = types.SimpleNamespace()
    for name in ("input_real_base", "input_imag_base", "output_real_base", "output_imag_base"):
        pointer = CheckedPtr(len(values))
        pointer.arr[:] = np.nan
        setattr(params, name, pointer)
    params.input_real_base.arr[:] = values.real
    params.input_imag_base.arr[:] = values.imag
    for buffer in plan.workspace_buffers:
        pointer = CheckedPtr(buffer.elements)
        pointer.arr[:] = np.nan
        setattr(params, buffer.name, pointer)
    table = []
    namespaces = []
    phases = resolve_distributed_phases(plan, compute_lanes=compute_lanes,
                                        narrow_middle_stages=narrow_middle_stages)
    local_elements = max(p.local_elements for p in phases)
    private = []
    for _ in range(plan.distributed.num_units):
        pointer = CheckedPtr(2 * local_elements * plan.distributed.workers_per_unit,
                             track_writers=False)
        private.append(types.SimpleNamespace(phase_scratch=pointer))
    for phase in phases:
        e = Emitter()
        emit_distributed_phase(e, plan=plan, phase=phase, compute_lanes=compute_lanes,
                               narrow_middle_stages=narrow_middle_stages, twiddle_table=table)
        namespace = {"Float32": float, "SIMD": _simd, "p": params,
                     "global_uthread_id": lambda: current["gid"]}
        exec(compile(_translate_emitted_lines(e.lines), "<distributed FFT>", "exec"), namespace)
        namespaces.append(namespace)
    for component, name in enumerate(("loop_twiddle_real_base", "loop_twiddle_imag_base")):
        pointer = CheckedPtr(max(1, len(table)))
        pointer.arr[:len(table)] = [pair[component] for pair in table]
        setattr(params, name, pointer)
    participation = []
    for phase, namespace in zip(phases, namespaces):
        writers.clear()
        for unit in private:
            unit.phase_scratch.arr[:] = np.nan
        ids = list(range(plan.total_uthreads))
        random.Random(100 + phase.stage_ids[0]).shuffle(ids)
        for gid in ids:
            current["gid"] = gid
            namespace[plan.kernel_name] = private[gid // plan.distributed.workers_per_unit]
            namespace[phase.name]()
        units = {(base_unit + gid // plan.distributed.workers_per_unit) % plan.distributed.num_units
                 for gid in writers.values()}
        participation.append(units)
        # The completion barrier happens here, after EVERY worker's stores.
    return params.output_real_base.arr + 1j * params.output_imag_base.arr, participation


class UnitParallelTests(unittest.TestCase):
    def test_leaf_groups_preserve_physical_decoder_and_remainder_units(self):
        rng = np.random.default_rng(76)
        for units in (1, 4, 8, 14, 28, "auto"):
            with self.subTest(units=units):
                original = clfft.plan_m2ndp(256, batch=4, target=TARGET).plan
                plan = distribute_plan(original, target=TARGET, units_per_leaf=units)
                kernel = plan.root.kernel
                mapping = kernel.distributed
                self.assertEqual(mapping.num_units, 28)
                self.assertEqual(kernel.total_uthreads, 224)
                values = rng.normal(size=1024) + 1j * rng.normal(size=1024)
                actual, participation = run_distributed_kernel(kernel, values, base_unit=25,
                                                               narrow_middle_stages=True)
                np.testing.assert_allclose(actual, np.fft.fft(values.reshape(4, 256)).ravel(),
                                           rtol=1e-5, atol=1e-5)
                allowed = {(25 + i) % 28 for i in range(
                    min(4, mapping.leaf_groups) * mapping.leaf_units)}
                self.assertTrue(all(p <= allowed for p in participation))
                if units == "auto":
                    self.assertEqual(mapping.leaf_units, 7)
                    self.assertEqual(len(set.union(*participation)), 28)

    def test_fused_phases_private_memory_aliases_and_widths(self):
        rng = np.random.default_rng(81)
        for module, n, batch, units, width, narrow in (
                (clfft, 1024, 1, 4, 4, True),
                (clfft, 1024, 3, 14, 4, True),
                (vkfft, 1024, 2, 14, 4, True),
                (rocfft_default, 256, 3, 4, 2, False),
                (vkfft, 96, 3, 1, 1, False),
                (clfft, 256, 2, 1, 8, False)):
            for inverse in (False, True):
                with self.subTest(planner=module.__name__, n=n, units=units,
                                  inverse=inverse, width=width):
                    original = module.plan_m2ndp(n, batch=batch, inverse=inverse, target=TARGET).plan
                    plan = distribute_plan(original, target=TARGET, units_per_leaf=units,
                                           fuse_stages=2).root.kernel
                    values = rng.normal(size=n * batch) + 1j * rng.normal(size=n * batch)
                    actual, _ = run_distributed_kernel(plan, values, compute_lanes=width,
                                                       narrow_middle_stages=narrow)
                    expected = (np.fft.ifft if inverse else np.fft.fft)(values.reshape(batch, n)).ravel()
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
                    self.assertEqual(tuple(s.radix for s in plan.stages),
                                     tuple(s.radix for s in original.root.kernel.stages))

    def test_fusion_budget_parallelism_and_registration(self):
        from planning.execution.kernel_registration import registered_kernel_names
        original = clfft.plan_m2ndp(1024, target=TARGET).plan
        for units, budget, expected in ((4, 256, 2), (14, 64, 3), (28, 256, 4), (4, 1, 4)):
            plan = distribute_plan(original, target=TARGET, units_per_leaf=units,
                                   fuse_stages=2, fusion_max_elements=budget)
            kernel = plan.root.kernel
            phases = resolve_distributed_phases(kernel, compute_lanes=4, narrow_middle_stages=True)
            self.assertEqual(len(phases), expected)
            names = registered_kernel_names(kernel, compute_lanes=4, narrow_middle_stages=True)
            source = generate_recursive_fft_kernels(plan, compute_lanes=4,
                     narrow_middle_stages=True, target=TARGET, spread_across_units=True)
            self.assertEqual(source.count("launch_parallel["), expected)
            self.assertEqual(names, tuple(p.name for p in phases))
            if units == 14 and budget == 64:
                # The middle ping-pong pair aliases its boundary bank and must
                # not race with other components still reading that old bank.
                self.assertEqual([p.stage_ids for p in phases], [(0,), (1,), (2, 3)])
        tiny = replace(TARGET, max_concurrent_scratchpad_bytes=32)
        kernel = distribute_plan(original, target=tiny, units_per_leaf=1, fuse_stages=2).root.kernel
        self.assertEqual(len(resolve_distributed_phases(kernel)), len(kernel.stages))

    def test_invalid_leaf_and_fusion_settings(self):
        original = clfft.plan_m2ndp(256, target=TARGET).plan
        for options in (dict(units_per_leaf=0), dict(units_per_leaf=29),
                        dict(units_per_leaf=True), dict(units_per_leaf="bad"),
                        dict(fuse_stages=3), dict(fuse_stages=2.0),
                        dict(fusion_max_elements=0)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                distribute_plan(original, target=TARGET, **options)
        with self.assertRaisesRegex(ValueError, "require distributed"):
            apply_gpu_execution(original, units_per_leaf=4)
        plan = distribute_plan(original, target=TARGET, units_per_leaf=4)
        with self.assertRaisesRegex(ValueError, 'physical target'):
            generate_recursive_fft_kernels(plan, target=DEFAULT_TARGET_PROFILE)

    def test_fused_recursive_inverse_preserves_exchange_and_scale(self):
        rng = np.random.default_rng(89)
        original = clfft.plan_m2ndp(8192, batch=2, inverse=True, target=TARGET).plan
        plan = distribute_plan(original, target=TARGET, units_per_leaf=4, fuse_stages=2)
        values = rng.normal(size=16384) + 1j * rng.normal(size=16384)
        actual = run_recursive_plan(plan, values, compute_lanes=4, narrow_middle_stages=True)
        np.testing.assert_allclose(actual, np.fft.ifft(values.reshape(2, 8192)).ravel(),
                                   rtol=1e-5, atol=1e-5)

    def test_distributed_forward_inverse_batches_and_ragged_stages(self):
        rng = np.random.default_rng(18)
        for module in (clfft, rocfft_default, vkfft):
            for n, batch in ((128, 1), (256, 2), (1024, 1), (96, 3)):
                for inverse in (False, True):
                    with self.subTest(planner=module.__name__, n=n, batch=batch, inverse=inverse):
                        original = module.plan_m2ndp(n, batch=batch, inverse=inverse, target=TARGET)
                        self.assertEqual(original.status.value, "ok")
                        plan = distribute_plan(original.plan, target=TARGET)
                        self.assertEqual(tuple(s.radix for s in plan.root.kernel.stages),
                                         tuple(s.radix for s in original.plan.root.kernel.stages))
                        values = rng.normal(size=n * batch) + 1j * rng.normal(size=n * batch)
                        actual, units = run_distributed_kernel(plan.root.kernel, values, base_unit=11)
                        expected = (np.fft.ifft if inverse else np.fft.fft)(values.reshape(batch, n)).ravel()
                        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
                        self.assertTrue(all(len(u) > 1 for u in units))
                        if n == 1024:
                            self.assertEqual(len(set.union(*units)), 28)

    def test_split_leaves_reuses_exchange_and_normalizes_once(self):
        rng = np.random.default_rng(20)
        for module in (clfft, rocfft_default, vkfft):
            for inverse in (False, True):
                plan = make_split_leaf_plan(1024, leaf_planner=module.plan_m2ndp,
                         inverse=inverse, batch=2, leaf_byte_budget=1024, target=TARGET)
                stages = flatten_recursive_node(plan.root)
                self.assertTrue(any(isinstance(s, PhysicalTransposePlan) for s in stages))
                leaves = [s for s in stages if not isinstance(s, PhysicalTransposePlan)]
                self.assertTrue(all(s.length <= 64 and s.host.total_elems > s.length for s in leaves))
                values = rng.normal(size=2048) + 1j * rng.normal(size=2048)
                actual = run_recursive_plan(plan, values, compute_lanes=4)
                expected = (np.fft.ifft if inverse else np.fft.fft)(values.reshape(2, 1024)).ravel()
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_codegen_barriers_workspace_and_legacy_identity(self):
        original = clfft.plan_m2ndp(1024, target=TARGET).plan
        self.assertIs(apply_gpu_execution(original), original)
        plan = distribute_plan(original, target=TARGET)
        source = generate_recursive_fft_kernels(plan, compute_lanes=4, target=TARGET,
                                               spread_across_units=True)
        self.assertEqual(source.count("launch_parallel["), len(plan.root.kernel.stages))
        self.assertIn('"exchange": "cxl_pingpong"', source)
        self.assertIn("var pool0_elems = 1792", source)
        self.assertNotIn("remaining_blocks = 1", source)
        self.assertFalse(plan.root.kernel.scratchpad_buffers)
        self.assertEqual(plan.root.kernel.estimated_dram_bytes,
                         16 * 1024 * len(plan.root.kernel.stages))

    def test_distributed_recursive_plan_and_arithmetic_widths(self):
        rng = np.random.default_rng(32)
        original = clfft.plan_m2ndp(8192, target=TARGET).plan
        plan = distribute_plan(original, target=TARGET)
        values = rng.normal(size=8192) + 1j * rng.normal(size=8192)
        np.testing.assert_allclose(run_recursive_plan(plan, values, compute_lanes=4),
                                   np.fft.fft(values), rtol=1e-5, atol=1e-5)
        leaf = distribute_plan(vkfft.plan_m2ndp(96, target=TARGET).plan, target=TARGET).root.kernel
        for width in (1, 2, 8):
            values = rng.normal(size=96) + 1j * rng.normal(size=96)
            actual, _ = run_distributed_kernel(leaf, values, compute_lanes=width)
            np.testing.assert_allclose(actual, np.fft.fft(values), rtol=1e-5, atol=1e-5)

    def test_cli_api_modes_and_invalid_native_choice(self):
        from pathlib import Path
        from unittest.mock import patch
        from make_fft_kernel import make_fft_kernel, _parse_args
        self.assertEqual(_parse_args(['256', '--planner', 'gpu-clfft-m2ndp',
                         '--gpu-execution', 'distributed']).gpu_execution, 'distributed')
        options = _parse_args(['256', '--planner', 'gpu-clfft-m2ndp',
                              '--gpu-execution', 'distributed', '--units-per-leaf', 'auto',
                              '--fuse-stages', '2'])
        self.assertEqual(options.units_per_leaf, 'auto')
        self.assertEqual(options.fuse_stages, 2)
        with patch.object(Path, 'write_text') as write:
            for mode in ('distributed', 'split-leaves'):
                path = make_fft_kernel(512, planner='gpu-clfft-m2ndp', gpu_execution=mode,
                       split_leaf_byte_budget=1024, output_path=Path(mode + '.mojo'))
                self.assertEqual(path, Path(mode + '.mojo'))
                self.assertIn('launch_parallel[', write.call_args.args[0])
            with self.assertRaisesRegex(ValueError, 'require a GPU planner'):
                make_fft_kernel(512, gpu_execution='distributed')
            make_fft_kernel(256, planner='gpu-clfft-m2ndp', gpu_execution='distributed',
                            units_per_leaf=4, fuse_stages=2, output_path=Path('shared.mojo'))
            self.assertIn('stage_0_1', write.call_args.args[0])
            with self.assertRaisesRegex(ValueError, 'require distributed'):
                make_fft_kernel(256, units_per_leaf=4)

    def test_work_slots_cover_all_units_first_and_are_bijective(self):
        plan = distribute_plan(clfft.plan_m2ndp(256, target=TARGET).plan, target=TARGET)
        mapping = plan.root.kernel.distributed
        self.assertEqual(sorted(mapping.work_slot(i) for i in range(224)), list(range(224)))
        active = {gid // 8 for gid in range(224) if mapping.work_slot(gid) < 28}
        self.assertEqual(active, set(range(28)))


if __name__ == "__main__":
    unittest.main()
