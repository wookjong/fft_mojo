"""Per-length tuning (codegen.length_tuning): default reproduces the measured sources, other paths unchanged.

The SHA-256 values are those of the sources built and timed on 2026-10-31 /
2026-11-01 (implementations/all_lengths_20261031, verification/generated/
<experiment>/variant.original.mojo; rounds 16-21 bests, report_r16_ko.md),
rendered here with the measurement harness's options.
"""
import hashlib
import unittest

from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from codegen.length_tuning import MEASURED, MEASURED_TARGET, resolve_length_tuning, select_leaf_spec
from planning.core.target_profile import DEFAULT_TARGET_PROFILE, TargetProfile
from planning.execution.fft_unit_parallel import apply_gpu_execution
from planning.gpu_baseline import vkfft

TARGET = TargetProfile(**MEASURED_TARGET)
MEASURED_SHA256 = {
    (64, 1): '2ff0a11f713a1ada8879dd34ac5377b59cc8a0c38a333b3d93b38ce8d94a723d',
    (64, 256): '1c91e7152055da52881fa386845952073a0ef1eff6d537582eb5d5910535910e',
    (128, 1): '141c0af5d3d72be627ce548a8e0cfa45000124579429b0be35d2dacfd979c867',
    (256, 1): '12490d04d0adbb05d3e790ac59a1b4171363f41ddbd8a6179113053af14ea04d',
    (256, 4): 'ff15d135d8109a6596e87fb647b427439245c49786bfa6c84bc50c1df1555712',
    (256, 16): 'e7662d6a34c2fde5a7e4cd459c2c7a30d736736f3f1fc25ca3f9cb076a9befd1',
    (256, 64): 'd42b8ddaa933e0fee943135cc24fc3960d68a32cec34ad03435c38246a16badf',
    (512, 1): 'd7bbbb7ceab083bed7b1628c243bc696a119bd9a6e36a5e909411b7e8aafa777',
    (1024, 1): '7f59d7ec986a5c34460c35eac1ea4597a6b60f43d1992a43caf1230f18b9b2bd',
    (1024, 4): '4a8ed98c7146f58c71362a45ee7c3afed58003316da94072bd8e5f8c486049bb',
    (1024, 16): '2106224120fe5e2284ddbcb0fda1ffb219bc67c44b655eed0b5fbf1c483d5e21',
    (1024, 64): '8072d6611d752c63fd2e819c35e987eeb6388a5be080df79f6a04bb7437af79c',
    (2048, 1): '04f02020090f9525bd3624c9f57ece76ce123a58e4e30169a02ab6d5d709c02d',
    (4096, 1): '90a59a80c25cbc3ac828dc5df443dd3e553e072159a4ede8d388b31948500a20',
    (4096, 4): '537e3240a2bcc39bdafc791fdb7c7257fa05c9073df811081d3d5139e663c997',
    (8192, 1): '7586adbea181cd4ac207fde8b6076d46235de22b8bedfdfe222127b1b7ee18d5',
    (16384, 1): 'e58e1d88e6d0e882a22c88477dbcd1bba6f51306b37b8d08ad4b394a9f09c6fd',
    (16384, 8): '2c3753546154f17c147a7a98927029f7ec5159eb09f532833695436f2d428858',
}


def plan_for(n, batch, mode='distributed', target=TARGET, inverse=False):
    result = vkfft.plan_m2ndp(n, batch=batch, inverse=inverse, target=target)
    extra = dict(units_per_leaf='auto', fuse_stages=1) if mode == 'distributed' else {}
    return apply_gpu_execution(result.plan, mode=mode, target=target, **extra)


def harness_options(n, batch, target=TARGET):
    common = dict(compute_lanes=4, narrow_middle_stages=True, reference_check=False, target=target,
                  spread_across_units=True, debug_print_pool_alignment=True)
    if n <= 4096:
        return common
    allocation, codelet, window = (('phase-packed', 'native4-control', '4') if batch == 1
                                   else ('live-banks', 'native4-latch-forward', '8'))
    return dict(common, task_fusion='auto', task_fusion_control=False, leaf_storage='spad', batch_window=window,
                boundary_fusion='pre', boundary_layout='grouped', near_group=4, far_group=1,
                leaf_phase_policy='far12-private', leaf_codegen_schedule='far0-short-live',
                spad_allocation=allocation, far12_codelet=codelet,
                layout_rewrite='n8+f8+sp' if n == 16384 else 'off')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class LengthTuningTests(unittest.TestCase):
    def test_default_reproduces_measured_sources(self):
        self.assertEqual(set(MEASURED), set(MEASURED_SHA256))
        for (n, batch), digest in sorted(MEASURED_SHA256.items()):
            with self.subTest(n=n, batch=batch):
                source = generate_recursive_fft_kernels(plan_for(n, batch), **harness_options(n, batch))
                self.assertEqual(sha(source), digest)

    def test_layout_off_16384_reaches_the_same_source(self):
        for batch in (1, 8):
            options = dict(harness_options(16384, batch), layout_rewrite='off')
            source = generate_recursive_fft_kernels(plan_for(16384, batch), **options)
            self.assertEqual(sha(source), MEASURED_SHA256[(16384, batch)])

    def test_unmeasured_configurations_are_unchanged(self):
        cases = [
            (plan_for(256, 1), dict(harness_options(256, 1), target=TARGET)),
            (plan_for(256, 1, mode='worker-wave'), harness_options(256, 1)),
            (plan_for(256, 1, inverse=True), harness_options(256, 1)),
            (plan_for(256, 1, target=DEFAULT_TARGET_PROFILE), harness_options(256, 1, DEFAULT_TARGET_PROFILE)),
            (plan_for(16384, 4), harness_options(16384, 4)),
            (vkfft.plan_m2ndp(1024, batch=3, inverse=False, target=TARGET).plan, {}),
        ]
        for index, (plan, options) in enumerate(cases[1:], 1):
            with self.subTest(case=index):
                self.assertEqual(generate_recursive_fft_kernels(plan, **options),
                                 generate_recursive_fft_kernels(plan, length_tuning='off', **options))
        plan, options = cases[0]
        self.assertNotEqual(generate_recursive_fft_kernels(plan, **options),
                            generate_recursive_fft_kernels(plan, length_tuning='off', **options))

    def test_leaf_spec_selection(self):
        self.assertEqual(select_leaf_spec(256, 4), 'D1+TR+R8x4x8+IX+PB')
        self.assertEqual(select_leaf_spec(256, 2), 'D1+TR+R8x4x8+IX+PB')
        self.assertEqual(select_leaf_spec(64, 100), 'O4+IX+PB')           # nearest 256 -> O, ceil(100/28) owners
        self.assertEqual(select_leaf_spec(1024, 32), 'S48+J+TR+R8x4x8x4+IX+W16')
        self.assertEqual(select_leaf_spec(1024, 256), 'D4+TR+R8x4x8x4+IX+W16')  # S arena would not fit
        self.assertEqual(select_leaf_spec(4096, 2), 'D4+TR+R8x4x8x4x4+IX+PL+PB')  # tie -> larger measured batch

    def test_explicit_specs_are_checked(self):
        options = harness_options(256, 1)
        with self.assertRaises(ValueError):
            generate_recursive_fft_kernels(plan_for(256, 1, mode='worker-wave'), length_tuning='D1+TR', **options)
        with self.assertRaises(ValueError):
            generate_recursive_fft_kernels(plan_for(256, 1), length_tuning='Q3', **options)
        spec, reason = resolve_length_tuning(plan_for(256, 1), dict(options, task_fusion='off'), 'S8+TR')
        self.assertEqual((spec, reason), ('S8+TR', 'explicit'))


if __name__ == '__main__':
    unittest.main()
