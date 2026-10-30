"""Regressions for transpose task reservations and per-unit launch coverage."""
import unittest

from codegen.common import Emitter
from codegen.fft_transpose_codegen import (
    _emit_physical_transpose_params_struct, _safe_round_size,
)
from planning.strategies.fft_plan_recursive import _build_physical_transpose


class TaskScratchpadTests(unittest.TestCase):
    def test_full_task_fits_and_rounds_cover_tiles(self):
        # Full, partial, and non-power-of-two grids exercise all Params fields.
        for rows, cols, replicas in ((128, 128, 1), (128, 128, 4), (129, 131, 3)):
            for twiddle in (None, rows * cols):
                with self.subTest(rows=rows, cols=cols, replicas=replicas, twiddle=twiddle):
                    plan = _build_physical_transpose(
                        rows=rows, cols=cols, replica_count=replicas,
                        tile_rows=8, tile_cols=8, twiddle_modulus=twiddle,
                        inverse=False, kernel_name="Transpose", simd_lanes=8,
                        spad_capacity_bytes=122880, apply_inverse_scale=False)
                    params = Emitter()
                    _emit_physical_transpose_params_struct(params, plan=plan)
                    params_bytes = 8 * sum(line.startswith("    var ") for line in params.lines)
                    self.assertLessEqual(plan.max_uthread * plan.scratchpad_elements * 4
                                         + params_bytes + 8, 122880)
                    self.assertEqual(plan.max_uthread, 239)
                    for spread in (False, True):
                        width = (_safe_round_size(plan.max_uthread, num_ndp_units=28,
                                 interleave_chunk_uthreads=8) if spread else plan.max_uthread)
                        seen = []
                        for offset in range(0, plan.total_uthreads, width):
                            counts = [0] * 28
                            for gid in range(min(width, plan.total_uthreads - offset)):
                                counts[(gid // 8) % 28] += 1
                                seen.append(offset + gid)
                            self.assertLessEqual(max(counts), plan.max_uthread)
                        self.assertEqual(seen, list(range(plan.total_uthreads)))

    def test_params_can_make_even_one_tile_infeasible(self):
        with self.assertRaisesRegex(ValueError, "too small to fit"):
            _build_physical_transpose(
                rows=8, cols=8, replica_count=1, tile_rows=8, tile_cols=8,
                twiddle_modulus=None, inverse=False, kernel_name="Transpose",
                simd_lanes=8, spad_capacity_bytes=512, apply_inverse_scale=False)


if __name__ == "__main__":
    unittest.main()
