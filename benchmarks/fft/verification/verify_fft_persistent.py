from __future__ import annotations

"""Numeric verification for the persistent-software-workgroup FFT leaf
(planning/fft_plan_persistent.py, codegen/fft_persistent_codegen.py) --
runs the *actual* emitted phase text (via each phase's own `emit_stage_
phase`/`emit_bulk_copy_phase`) through verify_fft_harness's existing
Mojo-to-Python translator (`_translate_emitted_lines`) and numpy-backed
`Ptr`/`SimdVec` stand-ins, the same discipline every other verify_fft_*.py
module in this package follows: no FFT mathematics is reimplemented here,
only re-executed from the real generator output.

`run_persistent_kernel` cannot reuse `verify_fft_harness.run_kernel`
outright -- that function's own group/round model (`group = global_id //
max_uthread`, one stage launched once total) doesn't match the persistent
model (`software_group_id = global_id // workers_per_group`, many rounds,
three phase kinds per round) -- but it reuses that same module's
`_translate_emitted_lines`/`Ptr`/`SimdVec`/`_simd` outright, and mirrors
its "rebind the kernel-name namespace entry per group before calling"
idiom (see `run_kernel`'s own docstring) with `software_group_id` in
place of `global_id // max_uthread`.
"""

import sys
import types
from pathlib import Path

import numpy as np

# See verify_fft_plan.py's own comment on this line -- lets this script run
# directly (`python3 verification/verify_fft_persistent.py`) without `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codegen.fft_persistent_codegen import emit_bulk_copy_phase, emit_stage_phase
from planning.fft_plan_core import FFTCodegenPlan
from planning.fft_plan_persistent import (
    make_persistent_leaf_plan,
    num_rounds,
    round_active_groups,
)
from verification.verify_fft_harness import Ptr, _simd, _translate_emitted_lines


def run_persistent_kernel(
    plan: FFTCodegenPlan,
    *,
    num_logical_blocks: int,
    input_real: Ptr,
    input_imag: Ptr,
    output_real: Ptr,
    output_imag: Ptr,
    compute_lanes: int | None = 4,
    narrow_middle_stages: bool = True,
) -> None:
    if plan.persistent is None:
        raise ValueError("run_persistent_kernel requires a persistent plan")
    pw = plan.persistent
    workers_per_group = pw.workers_per_group
    software_group_count = pw.software_group_count
    launch_uthreads = workers_per_group * software_group_count
    assert launch_uthreads == plan.total_uthreads

    group_namespaces: list[types.SimpleNamespace] = []
    for _ in range(software_group_count):
        ns = types.SimpleNamespace()
        for buf in plan.scratchpad_buffers:
            setattr(ns, buf.name, Ptr(buf.elements))
        group_namespaces.append(ns)

    p_ns = types.SimpleNamespace(
        input_real_base=input_real,
        input_imag_base=input_imag,
        output_real_base=output_real,
        output_imag_base=output_imag,
    )

    current = {"global_id": 0}

    def run_phase(name: str, lines: list[str]) -> None:
        src = _translate_emitted_lines(lines)
        namespace = {
            "Float32": float,
            "SIMD": _simd,
            "global_uthread_id": lambda: current["global_id"],
            "N": plan.length,
            "W": plan.simd_lanes,
            "p": p_ns,
        }
        code = compile(src, f"<{plan.kernel_name} {name}>", "exec")
        exec(code, namespace)
        fn = namespace[name]
        for global_id in range(launch_uthreads):
            current["global_id"] = global_id
            software_group_id = global_id // workers_per_group
            namespace[plan.kernel_name] = group_namespaces[software_group_id]
            fn()

    buffer_names = tuple(b.name for b in plan.scratchpad_buffers)
    final_buffer = buffer_names[len(plan.stages) % 2]
    rounds = num_rounds(num_logical_blocks, software_group_count)

    for r in range(rounds):
        active_groups = round_active_groups(r, num_logical_blocks, software_group_count)

        name, lines = emit_bulk_copy_phase(
            plan=plan, name="preload", round_index=r, active_groups=active_groups,
            workers_per_group=workers_per_group, to_scratchpad=True,
            buffer_name=buffer_names[0],
        )
        run_phase(name, lines)

        for stage in plan.stages:
            name, lines = emit_stage_phase(
                plan=plan, stage=stage, round_index=r, active_groups=active_groups,
                workers_per_group=workers_per_group,
                compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
            )
            run_phase(name, lines)

        name, lines = emit_bulk_copy_phase(
            plan=plan, name="writeback", round_index=r, active_groups=active_groups,
            workers_per_group=workers_per_group, to_scratchpad=False,
            buffer_name=final_buffer,
        )
        run_phase(name, lines)


def verify_persistent_leaf(
    length: int,
    radices: tuple[int, ...],
    *,
    num_logical_blocks: int,
    inverse: bool = False,
    scalar_worker_mode: str = "adaptive",
    compute_lanes: int | None = 4,
    narrow_middle_stages: bool = True,
    seed: int = 0,
) -> float:
    """Random input, `num_logical_blocks` independent length-`length`
    blocks back to back (the same DRAM layout `codegen.common.
    emit_reference_check`'s own `batch_count` convention uses) -- runs
    the actual generated persistent kernel via `run_persistent_kernel`
    and compares against `numpy.fft`. Returns the max abs error across
    both real and imaginary parts.
    """
    plan = make_persistent_leaf_plan(
        length, radices, num_logical_blocks=num_logical_blocks, inverse=inverse,
        scalar_worker_mode=scalar_worker_mode,
    )

    rng = np.random.default_rng(seed)
    total = length * num_logical_blocks
    in_r = rng.uniform(-1, 1, total)
    in_i = rng.uniform(-1, 1, total)

    input_real, input_imag = Ptr(total), Ptr(total)
    input_real.arr[:] = in_r
    input_imag.arr[:] = in_i
    output_real, output_imag = Ptr(total), Ptr(total)

    run_persistent_kernel(
        plan, num_logical_blocks=num_logical_blocks,
        input_real=input_real, input_imag=input_imag,
        output_real=output_real, output_imag=output_imag,
        compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
    )

    # This project's own inverse convention (fft_plan_core._build_plan's
    # `inverse_scale = 1/length`, confirmed against codegen.common.
    # emit_reference_check's own O(N^2) host check, which divides by N too)
    # is the normalized inverse -- matches numpy.fft.ifft exactly, no extra
    # *length multiply needed.
    x = (in_r + 1j * in_i).reshape(num_logical_blocks, length)
    ref = np.fft.ifft(x, axis=1) if inverse else np.fft.fft(x, axis=1)
    got = (output_real.arr + 1j * output_imag.arr).reshape(num_logical_blocks, length)
    return float(np.max(np.abs(got - ref)))


def check_plan_equivalence(
    length: int, radices: tuple[int, ...], *, inverse: bool = False
) -> None:
    """Structural comparison: the persistent lowering's own stages must
    carry *exactly* the same mathematics as the ordinary (non-persistent)
    lowering for the same (length, radices, inverse) -- radix, stage
    position, load/store base_offset, valid_lanes, twiddle metadata,
    output_stride/permutation, scale -- everything except `source`/
    `destination`/`buffer_name` (which are intentionally different:
    scratchpad vs. DRAM). See docs/persistent_leaf_design.md's
    "Mathematical plan-equivalence test" section. Raises AssertionError
    with a specific mismatch on failure; returns normally on success.
    """
    from dataclasses import replace as _replace

    from planning.fft_plan_core import _build_plan, layouts_for_radices, pingpong_needed

    normal = _build_plan(
        length=length, inverse=inverse, total_uthreads=1, simd_lanes=8,
        use_pingpong=pingpong_needed(len(radices)),
        layouts=layouts_for_radices(length, radices, 8),
        kernel_name="Normal",
    )
    persistent = make_persistent_leaf_plan(
        length, radices, num_logical_blocks=1, inverse=inverse,
    )

    assert len(normal.stages) == len(persistent.stages)
    for ns, ps in zip(normal.stages, persistent.stages):
        assert ns.stage_id == ps.stage_id
        assert ns.radix == ps.radix
        assert ns.inverse == ps.inverse
        assert ns.simd_iteration_count == ps.simd_iteration_count
        assert len(ns.batches) == len(ps.batches)
        for nb, pb in zip(ns.batches, ps.batches):
            assert nb.batch_id == pb.batch_id
            assert nb.valid_lanes == pb.valid_lanes
            assert len(nb.loads) == len(pb.loads)
            for nl, pl in zip(nb.loads, pb.loads):
                assert nl.operand == pl.operand
                assert nl.mode == pl.mode
                assert nl.base_offset == pl.base_offset
                assert nl.packed_lane_offsets == pl.packed_lane_offsets
                # source/buffer_name deliberately NOT compared -- see docstring.
            assert len(nb.outputs) == len(pb.outputs)
            for no, po in zip(nb.outputs, pb.outputs):
                assert no.output == po.output
                assert no.twiddle == po.twiddle
                assert no.scale == po.scale
                assert no.store.mode == po.store.mode
                assert no.store.base_offset == po.store.base_offset
                assert no.store.lane_offsets == po.store.lane_offsets
                # destination/buffer_name deliberately NOT compared.


def _run_case(
    length: int, radices: tuple[int, ...], *, num_logical_blocks: int,
    scalar_worker_mode: str = "adaptive",
) -> None:
    for inverse in (False, True):
        check_plan_equivalence(length, radices, inverse=inverse)
        err = verify_persistent_leaf(
            length, radices, num_logical_blocks=num_logical_blocks, inverse=inverse,
            scalar_worker_mode=scalar_worker_mode,
        )
        status = "OK" if err < 1e-6 else "FAIL"
        print(
            f"    {status}   persistent N={length} radices={radices} "
            f"blocks={num_logical_blocks} mode={scalar_worker_mode} inverse={inverse}: "
            f"max error {err:.3e}"
        )
        assert err < 1e-6, f"N={length} radices={radices} inverse={inverse}: err={err}"


def _check_target_invariant_rejections() -> None:
    """docs/persistent_leaf_design.md's own "Target-mapping invariant
    checks" section: construct incompatible TargetProfiles and confirm
    make_persistent_leaf_plan rejects each, rather than silently building
    a mapping that doesn't match the target."""
    from dataclasses import replace as _replace

    from planning.target_profile import DEFAULT_TARGET_PROFILE

    # Only these two are independently-supplied TargetProfile fields whose
    # *mutual* consistency make_persistent_leaf_plan can actually check --
    # workers_per_group/software_group_count/stripes_per_group are always
    # derived directly from target.interleave_chunk_uthreads/num_ndp_units/
    # a hardcoded 1 in this implementation (never independently supplied),
    # so their own "!= target.X" checks in the planner are unreachable by
    # construction today; kept there as documentation of the invariant a
    # future caller-overridable version would need to re-check for real,
    # not exercised here since there is no way to actually violate them
    # through this function's own parameters.
    cases = {
        "mapping_stride_bytes not a multiple of uthread_bytes":
            _replace(DEFAULT_TARGET_PROFILE, mapping_stride_bytes=257),
        "interleave_chunk_uthreads inconsistent with mapping_stride/uthread_bytes":
            _replace(DEFAULT_TARGET_PROFILE, interleave_chunk_uthreads=7),
    }
    for label, bad_target in cases.items():
        try:
            make_persistent_leaf_plan(64, (4, 4, 4), num_logical_blocks=1, target=bad_target)
        except NotImplementedError:
            print(f"    OK   rejected: {label}")
        else:
            raise AssertionError(f"expected NotImplementedError for: {label}")


def main() -> None:
    print("  Persistent-workgroup leaf: target-invariant rejection checks:")
    _check_target_invariant_rejections()
    print("  Persistent-workgroup leaf: plan-equivalence + numeric checks:")
    # No-tail-anywhere case (N=64 = 4*4*4, every stage's own butterfly_count
    # divides simd_lanes=8 evenly).
    _run_case(64, (4, 4, 4), num_logical_blocks=1)
    # Scalar-tail case, single round (num_logical_blocks < software_group_count).
    _run_case(105, (3, 5, 7), num_logical_blocks=1)
    # Scalar-tail case, exactly at the round boundary (32 == software_group_count).
    _run_case(105, (3, 5, 7), num_logical_blocks=32)
    # Multi-round case (40 blocks, 32 groups -> 2 rounds, tail round ACTIVE_GROUPS=8).
    _run_case(105, (3, 5, 7), num_logical_blocks=40)
    # A different radix/tail shape.
    _run_case(30, (2, 3, 5), num_logical_blocks=40)
    # reserved scalar_worker_mode: worker 7 always idle/scalar, even on a
    # no-tail stage (N=64's own (4,4,4) has no tail at all).
    _run_case(64, (4, 4, 4), num_logical_blocks=40, scalar_worker_mode="reserved")
    _run_case(105, (3, 5, 7), num_logical_blocks=40, scalar_worker_mode="reserved")
    # ACTIVE_GROUPS=1 tail round (33 = software_group_count(32) + 1).
    _run_case(105, (3, 5, 7), num_logical_blocks=33)
    print("[verify] persistent leaf: all cases matched numpy's FFT")


if __name__ == "__main__":
    main()
