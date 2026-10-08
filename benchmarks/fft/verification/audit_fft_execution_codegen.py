"""Post-fix audit of emitted FFT execution; historical pre-fix fixture is retained.

This runs translated device code with ordinary memory reads and models the
physical interleave. Actual Docker results are recorded separately.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from codegen.fft_persistent_codegen import generate_persistent_fft_kernel
from planning.core.target_profile import DEFAULT_TARGET_PROFILE as TARGET
from planning.execution.fft_plan_cooperative import make_cooperative_leaf_plan
from planning.execution.fft_plan_persistent import make_persistent_leaf_plan
from planning.execution.launch_plan import resolve_fft_launch
from verification.verify_fft_execution_codegen_fixes import plain_plan, run_mapped_kernel
from verification.verify_fft_harness import Ptr
from verification.verify_fft_persistent import run_persistent_kernel


def main():
    root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(17)
    trackers = []
    for mode in ("physical", "fused", "wave"):
        for batch in (1, 33):
            plan = make_persistent_leaf_plan(64, (4, 4, 4), num_logical_blocks=batch,
                                             workers_per_fft=16, lowering_mode=mode)
            x = rng.normal(size=(batch, 64)) + 1j * rng.normal(size=(batch, 64))
            ir, ii, outr, outi = (Ptr(x.size) for _ in range(4))
            ir.arr[:], ii.arr[:] = x.real.ravel(), x.imag.ravel()
            run_persistent_kernel(plan, num_logical_blocks=batch, input_real=ir, input_imag=ii,
                                  output_real=outr, output_imag=outi,
                                  phase_worker_order=lambda phase, count: rng.permutation(count))
            err = float(np.max(np.abs((outr.arr + 1j * outi.arr).reshape(x.shape) - np.fft.fft(x))))
            trackers.append(dict(mode=mode, batch=batch, workers=16, max_error=err,
                                 tracker_reads="ordinary memory", worker_order="random per phase"))
    coverage = []
    for workers in (1, 8):
        plan = (plain_plan(batch=64) if workers == 1 else
                make_cooperative_leaf_plan(64, (4, 4, 4), total_ffts=64,
                                           workers_per_fft=workers, spad_capacity_bytes=1024))
        x = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
        actual = run_mapped_kernel(plan, x)
        launch = resolve_fft_launch(plan, spread_across_units=True)
        errors = np.max(np.abs(actual - np.fft.fft(x)), axis=1)
        coverage.append(dict(workers=workers, batch=64, max_uthread=plan.max_uthread,
                             host_launches=len(list(launch.rounds())),
                             omitted_fft_ids=np.flatnonzero(errors > 1e-4).tolist(),
                             max_error=float(max(errors)), padded=launch.padded))
    rounds = []
    for workers in (2, 8):
        plan = make_cooperative_leaf_plan(1024, (8, 8, 16), total_ffts=128,
                                          workers_per_fft=workers,
                                          spad_capacity_bytes=TARGET.spad_capacity_bytes,
                                          max_concurrent_scratchpad_bytes=TARGET.max_concurrent_scratchpad_bytes)
        launch = resolve_fft_launch(plan, spread_across_units=True)
        rounds.append(dict(workers=workers, host_launches=len(list(launch.rounds())),
                           ffts_per_launch=launch.round_fft_capacity,
                           physical_uthreads=launch.round_uthread_capacity,
                           active_units=min(TARGET.num_ndp_units,
                                            -(-launch.round_uthread_capacity // TARGET.interleave_chunk_uthreads))))
    target = replace(TARGET, interleave_chunk_uthreads=16, mapping_stride_bytes=512, num_ndp_units=4)
    plan = make_persistent_leaf_plan(64, (4, 4, 4), num_logical_blocks=5, target=target)
    source = generate_persistent_fft_kernel(plan, num_logical_blocks=5, target=target)
    widths = []
    for width in (0, -1):
        try:
            generate_persistent_fft_kernel(plan, num_logical_blocks=5, target=target,
                                            copy_mode="vectorized_contiguous", vector_width=width)
        except ValueError as exc:
            widths.append(dict(vector_width=width, accepted=False, reason=str(exc)))
        else:
            raise AssertionError("Invalid copy width was accepted")
    paths = ("codegen/common.py", "codegen/fft_codegen.py", "codegen/fft_cooperative_codegen.py",
             "codegen/fft_persistent_codegen.py", "codegen/fft_transpose_codegen.py",
             "planning/execution/launch_plan.py", "planning/execution/fft_plan_persistent.py",
             "planning/strategies/fft_plan_recursive.py", "planning/core/fft_plan_core.py",
             "verification/verify_fft_persistent.py")
    result = dict(scope="Post-fix execution/codegen audit; translated code, not a simulator measurement",
                  source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
                  tracker_ordering=trackers, standalone_coverage=coverage, cooperative_rounds=rounds,
                  custom_persistent_alignment=dict(planned_bytes=512, emits_512_alignment="// 512 * 512" in source),
                  invalid_copy_widths=widths)
    assert all(t["max_error"] < 1e-4 for t in trackers)
    assert all(not t["omitted_fft_ids"] for t in coverage)
    output = root / "verification/fixtures/fft_execution_codegen_fixes_2026-10-08.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(root)}")


if __name__ == "__main__":
    main()
