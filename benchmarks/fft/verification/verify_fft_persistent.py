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
each phase function called once per round) -- but it reuses that same
module's `_translate_emitted_lines`/`Ptr`/`SimdVec`/`_simd` outright, and
mirrors its "rebind the kernel-name namespace entry per group before
calling" idiom (see `run_kernel`'s own docstring) with `software_group_id`
in place of `global_id // max_uthread`.

Each phase is translated/compiled *once* (not once per round -- see
codegen.fft_persistent_codegen's own top docstring for why round-specific
functions don't work on the real simulator) and called `rounds` times in
a row, exactly mirroring device_main's own repeated `launch_parallel`
calls to the same function; round state threads through the same
per-physical-unit `round_tracker` scratchpad cell the real generated code
uses, not a Python-level round counter this harness invents separately.

REVISED 2026-09-14 (worker-wave FUSION -- see docs/worker_wave_fusion.md):
a stage phase is called exactly ONCE per round now, regardless of
`workers_per_fft` -- `codegen.fft_persistent_codegen.emit_stage_phase`
moved the "visit every logical worker" loop INSIDE the compiled stage
function itself (a plain runtime `while` loop over `logical_worker_id`),
so this harness's own translated-and-exec'd Python function already
performs that loop when called, the same way it already handles
`emit_bulk_copy_phase`'s own pre-existing `while i < length: ...`
strided-copy loop -- no new translation machinery, no `wave_tracker`
scratchpad mirror needed any more (removed, since the real generated code
no longer has one).

REVISED 2026-09-14 (physical-lane strip-mining -- see docs/
physical_lane_strip_mining.md): `run_persistent_kernel` now takes a
`mode` parameter (`"wave"`/`"fused"`/`"physical"`, forwarded straight to
`codegen.fft_persistent_codegen.emit_stage_phase` -- see that module's
own `ExecutionMode` docstring) so this harness can exercise and cross-
check all three execution-lowering strategies against the SAME plan.
`wave_tracker` is restored here (mirroring `round_tracker`) but only
matters for `mode="wave"` -- harmless, unused scratchpad for the other
two modes, the same "always present, only sometimes referenced by the
translated source" discipline this file already used before Mode B's
own `wave_tracker` was removed from production codegen.
"""

import re
import sys
import types
from pathlib import Path

import numpy as np

# See verify_fft_plan.py's own comment on this line -- lets this script run
# directly (`python3 verification/verify_fft_persistent.py`) without `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codegen.fft_persistent_codegen import (
    emit_bulk_copy_phase,
    emit_stage_phase,
    generate_persistent_fft_kernel,
)
from planning.core.fft_plan_core import FFTCodegenPlan
from planning.execution.fft_plan_persistent import (
    make_persistent_leaf_plan,
    num_rounds,
    worker_waves as _worker_waves,
)
from verification.verify_fft_harness import Ptr, SimdVec, _simd, _translate_emitted_lines


class _FrozenReadPtr:
    """Wraps one group's `round_tracker` for the duration of a single
    phase's own `launch_parallel` call: every worker's `.load(...)` sees
    the value as of the *start* of this call, regardless of the order
    this harness happens to iterate workers in -- matching real
    hardware's own apparent guarantee (confirmed via a from-scratch
    real-hardware experiment this session, not kept in-repo: every one
    of 256 workers across 32 independent physical units read the
    identical tracker value within one `launch_parallel` call, even
    though worker_id==0 of each group also writes that same tracker
    within the same call). `.store(...)` writes straight through to the
    real `Ptr`, so the write is visible starting the *next* phase call --
    only same-call visibility is frozen. This harness runs workers in a
    plain sequential Python loop (`run_kernel`'s own established
    idiom -- see its docstring), which has no such same-call ordering
    guarantee on its own; without this wrapper, worker 0's own bump
    would leak into worker 1..7's reads later in the same loop, purely
    an artifact of this harness's own execution order, not a real race.
    Only `round_tracker` needs this: `buf_a`/`buf_b` are never read and
    written by different workers within the same phase call (preload
    writes buf_a but never reads it; a stage reads buf[i%2]/writes
    buf[(i+1)%2], never the same bank in one call) -- see
    codegen.fft_persistent_codegen's own top docstring.
    """

    def __init__(self, snapshot: "np.ndarray", real: Ptr) -> None:
        self._snapshot = snapshot
        self._real = real

    def load(self, offset: int, width: int) -> SimdVec:
        offset = int(offset)
        return self._snapshot[offset : offset + int(width)].copy().view(SimdVec)

    def store(self, offset: int, value) -> None:
        self._real.store(offset, value)


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
    mode: str = "physical",
) -> None:
    if plan.persistent is None:
        raise ValueError("run_persistent_kernel requires a persistent plan")
    pw = plan.persistent
    workers_per_group = pw.workers_per_group
    workers_per_fft = pw.workers_per_fft
    software_group_count = pw.software_group_count
    launch_uthreads = workers_per_group * software_group_count
    assert launch_uthreads == plan.total_uthreads

    group_namespaces: list[types.SimpleNamespace] = []
    for _ in range(software_group_count):
        ns = types.SimpleNamespace()
        for buf in plan.scratchpad_buffers:
            setattr(ns, buf.name, Ptr(buf.elements))
        # One independent round_tracker per physical unit, zero-initialized
        # -- matches codegen.fft_persistent_codegen's own scratchpad
        # round_tracker exactly (see that module's own top docstring for
        # why round state lives here instead of a compile-time constant).
        ns.round_tracker = Ptr(1)
        # Only referenced by mode="wave"'s own translated source -- see
        # this module's own top docstring for why it's harmless to always
        # allocate regardless of mode.
        ns.wave_tracker = Ptr(1)
        group_namespaces.append(ns)

    p_ns = types.SimpleNamespace(
        input_real_base=input_real,
        input_imag_base=input_imag,
        output_real_base=output_real,
        output_imag_base=output_imag,
    )

    current = {"global_id": 0}

    def compile_phase(name: str, lines: list[str]):
        src = _translate_emitted_lines(lines)
        namespace = {
            "Float32": float,
            "Int": int,
            "SIMD": _simd,
            "global_uthread_id": lambda: current["global_id"],
            "N": plan.length,
            "W": plan.simd_lanes,
            "p": p_ns,
        }
        code = compile(src, f"<{plan.kernel_name} {name}>", "exec")
        exec(code, namespace)
        return namespace, namespace[name]

    def run_phase(namespace: dict, fn) -> None:
        # Fresh frozen round_tracker snapshot for THIS phase call only --
        # see _FrozenReadPtr's own docstring. buf_a/buf_b pass straight
        # through (same real Ptr every group already holds).
        phase_views = [
            types.SimpleNamespace(
                round_tracker=_FrozenReadPtr(ns.round_tracker.arr.copy(), ns.round_tracker),
                wave_tracker=_FrozenReadPtr(ns.wave_tracker.arr.copy(), ns.wave_tracker),
                **{buf.name: getattr(ns, buf.name) for buf in plan.scratchpad_buffers},
            )
            for ns in group_namespaces
        ]
        for global_id in range(launch_uthreads):
            current["global_id"] = global_id
            software_group_id = global_id // workers_per_group
            namespace[plan.kernel_name] = phase_views[software_group_id]
            fn()

    buffer_names = tuple(b.name for b in plan.scratchpad_buffers)
    final_buffer = buffer_names[len(plan.stages) % 2]
    rounds = num_rounds(num_logical_blocks, software_group_count)
    waves = _worker_waves(workers_per_fft, workers_per_group)
    stage_repeat = waves if (mode == "wave" and waves > 1) else 1

    # Compile every phase exactly once -- mirrors the real generated
    # device_main, which registers exactly 2+len(stages) kernel functions
    # total. Modes "fused"/"physical" call each stage phase exactly ONCE
    # per round regardless of `workers_per_fft`; mode "wave" calls it
    # `waves` times in a row (`stage_repeat`), mirroring device_main's own
    # per-mode call-count decision (see codegen.fft_persistent_codegen.
    # emit_persistent_kernel_struct).
    preload_phase = compile_phase(*emit_bulk_copy_phase(
        plan=plan, name="preload", software_group_count=software_group_count,
        num_logical_blocks=num_logical_blocks, workers_per_group=workers_per_group,
        to_scratchpad=True, buffer_name=buffer_names[0], bump_round=False,
    ))

    stage_phases: list[tuple[dict, object]] = []
    for stage in plan.stages:
        name, lines = emit_stage_phase(
            plan=plan, stage=stage, software_group_count=software_group_count,
            num_logical_blocks=num_logical_blocks, workers_per_group=workers_per_group,
            workers_per_fft=workers_per_fft,
            compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
            mode=mode,
        )
        stage_phases.append(compile_phase(name, lines))

    writeback_phase = compile_phase(*emit_bulk_copy_phase(
        plan=plan, name="writeback", software_group_count=software_group_count,
        num_logical_blocks=num_logical_blocks, workers_per_group=workers_per_group,
        to_scratchpad=False, buffer_name=final_buffer, bump_round=True,
    ))

    for _ in range(rounds):
        run_phase(*preload_phase)
        for namespace, fn in stage_phases:
            for _ in range(stage_repeat):
                run_phase(namespace, fn)
        run_phase(*writeback_phase)


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
    workers_per_fft: int | None = None,
    mode: str = "physical",
) -> float:
    """Random input, `num_logical_blocks` independent length-`length`
    blocks back to back (the same DRAM layout `codegen.common.
    emit_reference_check`'s own `batch_count` convention uses) -- runs
    the actual generated persistent kernel via `run_persistent_kernel`
    and compares against `numpy.fft`. Returns the max abs error across
    both real and imaginary parts.

    `workers_per_fft`: `None` (the default) is the plain one-wave case;
    a larger multiple of `target.interleave_chunk_uthreads` exercises the
    worker-wave virtualization path end to end (partition -> dispatch ->
    physical lowering) through this same Python-level oracle, not just
    real hardware -- see `make_persistent_leaf_plan`'s own parameter.

    `mode`: forwarded to `run_persistent_kernel` -- `"wave"`/`"fused"`/
    `"physical"`, see `codegen.fft_persistent_codegen`'s own
    `ExecutionMode` docstring.
    """
    plan = make_persistent_leaf_plan(
        length, radices, num_logical_blocks=num_logical_blocks, inverse=inverse,
        scalar_worker_mode=scalar_worker_mode, workers_per_fft=workers_per_fft,
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
        mode=mode,
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
    from planning.core.fft_plan_core import _build_plan, layouts_for_radices, pingpong_needed

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
    scalar_worker_mode: str = "adaptive", workers_per_fft: int | None = None,
    exec_mode: str = "physical",
) -> None:
    for inverse in (False, True):
        check_plan_equivalence(length, radices, inverse=inverse)
        err = verify_persistent_leaf(
            length, radices, num_logical_blocks=num_logical_blocks, inverse=inverse,
            scalar_worker_mode=scalar_worker_mode, workers_per_fft=workers_per_fft,
            mode=exec_mode,
        )
        status = "OK" if err < 1e-6 else "FAIL"
        print(
            f"    {status}   persistent N={length} radices={radices} "
            f"blocks={num_logical_blocks} mode={scalar_worker_mode} "
            f"workers_per_fft={workers_per_fft} exec_mode={exec_mode} inverse={inverse}: "
            f"max error {err:.3e}"
        )
        assert err < 1e-6, f"N={length} radices={radices} inverse={inverse}: err={err}"


def _check_target_invariant_rejections() -> None:
    """docs/persistent_leaf_design.md's own "Target-mapping invariant
    checks" section: construct incompatible TargetProfiles and confirm
    make_persistent_leaf_plan rejects each, rather than silently building
    a mapping that doesn't match the target."""
    from dataclasses import replace as _replace

    from planning.core.target_profile import DEFAULT_TARGET_PROFILE

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


def _check_registered_kernel_count_stable_across_rounds() -> None:
    """docs/persistent_leaf_design.md's own "POST-IMPLEMENTATION
    CORRECTION" section, and Task 5 of docs/
    persistent_vs_cooperative_comparison_task.md: 2, 8, and 32 rounds
    must all register the *same* distinct phase-function set
    (`preload`, `stage_0`, ..., `writeback`) -- round count must only
    change how many times `device_main` calls them, never how many
    distinct kernels exist. A regression back to one-function-per-round
    would crash real hardware (`max_kernel_register=8`, see
    target_profile.TargetProfile's own docstring) long before it showed
    up here, so this test catches it structurally, in Python, without a
    real build+run."""
    length, radices = 64, (4, 4, 4)
    software_group_count = make_persistent_leaf_plan(
        length, radices, num_logical_blocks=1
    ).persistent.software_group_count
    expected = {"preload", "stage_0", "stage_1", "stage_2", "writeback"}
    for blocks in (2 * software_group_count, 8 * software_group_count, 32 * software_group_count):
        plan = make_persistent_leaf_plan(length, radices, num_logical_blocks=blocks)
        src = generate_persistent_fft_kernel(plan, num_logical_blocks=blocks)
        device_main = src.split("def device_main():")[1].split("def main() raises:")[0]
        calls = re.findall(r"launch_parallel\[PersistentFFT\.(\w+)\]", device_main)
        distinct = set(calls)
        rounds = num_rounds(blocks, software_group_count)
        assert distinct == expected, f"blocks={blocks}: got {distinct}, expected {expected}"
        assert len(calls) == rounds * len(expected), (
            f"blocks={blocks}: expected {rounds * len(expected)} launch_parallel calls "
            f"({rounds} rounds x {len(expected)} phases), got {len(calls)}"
        )
        print(f"    OK   blocks={blocks} ({rounds} rounds): {len(distinct)} distinct "
              f"kernel functions ({sorted(distinct)}), {len(calls)} launch_parallel calls")


def _check_single_invocation_per_stage_regardless_of_workers_per_fft() -> None:
    """REVISED 2026-09-14 (worker-wave FUSION, then physical-lane strip-
    mining -- see docs/worker_wave_fusion.md and docs/physical_lane_
    strip_mining.md): checks the per-mode `launch_parallel` call-count/
    structural invariant for all three `ExecutionMode`s against the SAME
    plans. Mode "wave" (Mode A, the ORIGINAL mechanism) calls each stage
    phase `worker_waves(...)` times per round, threaded via a scratchpad
    `wave_tracker`; modes "fused" (Mode B) and "physical" (Mode C, the
    default) both call each stage phase EXACTLY ONCE per round regardless
    of `workers_per_fft` -- the `launch_parallel` count for either does
    not scale with `workers_per_fft` at all, only with `rounds * (2 +
    len(stages))`. Mode "fused" additionally has the runtime `while
    logical_worker_id <` loop and no `wave_tracker`; Mode "physical" has
    NEITHER the loop NOR `wave_tracker`, and -- the key Mode C invariant --
    every dispatch branch id is `< workers_per_group` (8), never `<
    workers_per_fft`, however large `workers_per_fft` is."""
    length, radices = 64, (4, 4, 4)
    workers_per_group = make_persistent_leaf_plan(
        length, radices, num_logical_blocks=1
    ).persistent.workers_per_group
    expected = {"preload", "stage_0", "stage_1", "stage_2", "writeback"}
    for workers_per_fft, blocks in ((8, 40), (10, 40), (16, 40), (18, 40), (32, 40), (36, 96), (64, 96)):
        waves = _worker_waves(workers_per_fft, workers_per_group)
        plan = make_persistent_leaf_plan(
            length, radices, num_logical_blocks=blocks, workers_per_fft=workers_per_fft,
        )
        for exec_mode in ("wave", "fused", "physical"):
            src = generate_persistent_fft_kernel(plan, num_logical_blocks=blocks, mode=exec_mode)
            device_main = src.split("def device_main():")[1].split("def main() raises:")[0]
            calls = re.findall(r"launch_parallel\[PersistentFFT\.(\w+)\]", device_main)
            distinct = set(calls)
            rounds = num_rounds(blocks, plan.persistent.software_group_count)
            stage_repeat = waves if (exec_mode == "wave" and waves > 1) else 1
            expected_calls = rounds * (1 + len(radices) * stage_repeat + 1)
            assert distinct == expected, (
                f"mode={exec_mode} workers_per_fft={workers_per_fft}: got {distinct}, expected {expected}"
            )
            assert len(calls) == expected_calls, (
                f"mode={exec_mode} workers_per_fft={workers_per_fft} (waves={waves}): expected "
                f"{expected_calls} launch_parallel calls, got {len(calls)}"
            )
            one_round = calls[: 1 + len(radices) * stage_repeat + 1]
            expected_one_round = (
                ["preload"]
                + [f"stage_{s}" for s in range(len(radices)) for _ in range(stage_repeat)]
                + ["writeback"]
            )
            assert one_round == expected_one_round, (
                f"mode={exec_mode} workers_per_fft={workers_per_fft}: phase ordering mismatch: "
                f"got {one_round}, expected {expected_one_round}"
            )
            has_wave_tracker = "wave_tracker" in src
            has_loop = "while logical_worker_id <" in src
            expect_wave_tracker = exec_mode == "wave" and waves > 1
            expect_loop = exec_mode == "fused" and waves > 1
            assert has_wave_tracker == expect_wave_tracker, (
                f"mode={exec_mode} workers_per_fft={workers_per_fft}: wave_tracker "
                f"presence={has_wave_tracker}, expected {expect_wave_tracker}"
            )
            assert has_loop == expect_loop, (
                f"mode={exec_mode} workers_per_fft={workers_per_fft}: while-loop "
                f"presence={has_loop}, expected {expect_loop}"
            )
            if exec_mode == "physical":
                branch_ids = [int(x) for x in re.findall(r"worker_id == (\d+)", src)]
                assert all(b < workers_per_group for b in branch_ids), (
                    f"Mode C workers_per_fft={workers_per_fft}: dispatch branch id(s) "
                    f"{branch_ids} should never reach workers_per_group={workers_per_group}, "
                    f"regardless of workers_per_fft -- this is the whole point of "
                    f"flattening to physical lanes at codegen time"
                )
            print(
                f"    OK   mode={exec_mode:9s} workers_per_fft={workers_per_fft} (waves={waves}) "
                f"blocks={blocks} ({rounds} rounds): {len(calls)} launch_parallel calls, "
                f"wave_tracker={has_wave_tracker}, loop={has_loop}"
            )


def check_logical_worker_coverage(workers_per_fft: int, workers_per_group: int) -> None:
    """Referenced from `fft_plan_persistent.worker_waves`'s own docstring:
    `(loop_iteration, physical_lane)` for `loop_iteration in range(waves)`,
    `physical_lane in range(workers_per_group)` computes `logical_worker_id
    = loop_iteration * workers_per_group + physical_lane`, a mixed-radix
    bijection onto `0 .. waves*workers_per_group - 1`. Restricting to
    `logical_worker_id < workers_per_fft` must therefore cover exactly
    `{0, ..., workers_per_fft - 1}` -- no missing logical worker, no
    duplicate -- for ANY positive `workers_per_fft`, not just an exact
    multiple of `workers_per_group`. This arithmetic fact is unaffected by
    worker-wave fusion (2026-09-14): `loop_iteration` used to mean "which
    separate `launch_parallel` call," now means "which pass through one
    physical lane's own `while logical_worker_id < workers_per_fft:` loop
    body inside a single call" -- the bijection itself, and this test, are
    identical either way. Raises AssertionError on any violation."""
    waves = _worker_waves(workers_per_fft, workers_per_group)
    seen: list[int] = []
    dummy_count = 0
    for wave in range(waves):
        for lane in range(workers_per_group):
            logical_id = wave * workers_per_group + lane
            if logical_id < workers_per_fft:
                seen.append(logical_id)
            else:
                dummy_count += 1
    assert sorted(seen) == list(range(workers_per_fft)), (
        f"workers_per_fft={workers_per_fft}: coverage mismatch, got {sorted(seen)}"
    )
    assert len(seen) == len(set(seen)), f"workers_per_fft={workers_per_fft}: duplicate logical id"
    expected_dummy = waves * workers_per_group - workers_per_fft
    assert dummy_count == expected_dummy, (
        f"workers_per_fft={workers_per_fft}: expected {expected_dummy} dummy "
        f"(wave, lane) slots, got {dummy_count}"
    )


def _check_dispatch_never_exceeds_w_and_has_no_else() -> None:
    """Direct source-level confirmation of the claim `codegen.fft_
    persistent_codegen._emit_worker_dispatch`'s own docstring makes for a
    ragged `workers_per_fft`: every emitted `if/elif <dispatch_var> == k:`
    branch has `k < workers_per_fft` (never a dummy logical id), and there
    is never a trailing `else:` that could otherwise accidentally give a
    dummy lane real work. Checked directly against the ACTUAL generated
    source (not inferred from reading the emitter's own Python code), for
    every stage of several ragged and non-ragged W values, across all
    three `ExecutionMode`s.

    Mode "physical" (Mode C) additionally must satisfy the STRICTER bound
    `k < workers_per_group` (8) -- the entire point of flattening at
    codegen time is that the emitted dispatch never grows past 8 branches
    no matter how large `workers_per_fft` is."""
    length, radices = 105, (3, 5, 7)
    for workers_per_fft in (3, 5, 6, 7, 10, 12, 18, 20, 36, 8, 16):
        plan = make_persistent_leaf_plan(
            length, radices, num_logical_blocks=1, workers_per_fft=workers_per_fft,
        )
        workers_per_group = plan.persistent.workers_per_group
        for exec_mode in ("wave", "fused", "physical"):
            src = generate_persistent_fft_kernel(plan, num_logical_blocks=1, mode=exec_mode)
            for stage_id in range(len(radices)):
                m = re.search(
                    rf"def stage_{stage_id}\(\):.*?(?=\n    @staticmethod|\ndef main)", src, re.S,
                )
                assert m is not None, f"mode={exec_mode} W={workers_per_fft}: stage_{stage_id} not found"
                body = m.group(0)
                branch_ids = [
                    int(x) for x in re.findall(r"(?:worker_id|logical_worker_id) == (\d+)", body)
                ]
                assert all(b < workers_per_fft for b in branch_ids), (
                    f"mode={exec_mode} W={workers_per_fft} stage_{stage_id}: branch id(s) "
                    f"{branch_ids} exceed workers_per_fft"
                )
                if exec_mode == "physical":
                    assert all(b < workers_per_group for b in branch_ids), (
                        f"Mode C W={workers_per_fft} stage_{stage_id}: branch id(s) "
                        f"{branch_ids} exceed workers_per_group={workers_per_group} -- "
                        f"flattening should have bounded this regardless of W"
                    )
                assert not re.search(r"\n\s*else:\s*\n", body), (
                    f"mode={exec_mode} W={workers_per_fft} stage_{stage_id}: unexpected "
                    f"trailing else -- a dummy lane must never match a catch-all branch"
                )
        print(f"    OK   workers_per_fft={workers_per_fft}: every dispatch branch < W (all "
              f"modes), Mode C additionally < workers_per_group, no else anywhere")


def _check_mode_equivalence(
    length: int, radices: tuple[int, ...], *, num_logical_blocks: int, workers_per_fft: int,
    scalar_worker_mode: str = "adaptive", inverse: bool = False, seed: int = 0,
) -> None:
    """Direct, bit-for-bit cross-check that Modes "wave"/"fused"/
    "physical" compute IDENTICAL output for the SAME plan and SAME random
    input -- not just "each matches numpy within tolerance" (which
    `_run_case`/`verify_persistent_leaf` already establish per mode) but
    "all three produce the exact same floating-point bits," the strongest
    possible confirmation that Mode C's compile-time flattening changes
    nothing about WHAT gets computed, only how the dispatch controlling it
    is expressed."""
    plan = make_persistent_leaf_plan(
        length, radices, num_logical_blocks=num_logical_blocks, inverse=inverse,
        scalar_worker_mode=scalar_worker_mode, workers_per_fft=workers_per_fft,
    )
    rng = np.random.default_rng(seed)
    total = length * num_logical_blocks
    in_r = rng.uniform(-1, 1, total)
    in_i = rng.uniform(-1, 1, total)

    outputs = {}
    for exec_mode in ("wave", "fused", "physical"):
        input_real, input_imag = Ptr(total), Ptr(total)
        input_real.arr[:] = in_r
        input_imag.arr[:] = in_i
        output_real, output_imag = Ptr(total), Ptr(total)
        run_persistent_kernel(
            plan, num_logical_blocks=num_logical_blocks,
            input_real=input_real, input_imag=input_imag,
            output_real=output_real, output_imag=output_imag, mode=exec_mode,
        )
        outputs[exec_mode] = output_real.arr + 1j * output_imag.arr

    diff_wave_fused = float(np.max(np.abs(outputs["wave"] - outputs["fused"])))
    diff_wave_physical = float(np.max(np.abs(outputs["wave"] - outputs["physical"])))
    status = "OK" if diff_wave_fused == 0.0 and diff_wave_physical == 0.0 else "FAIL"
    print(
        f"    {status}   N={length} radices={radices} W={workers_per_fft} blocks="
        f"{num_logical_blocks} inverse={inverse}: wave-vs-fused diff={diff_wave_fused:.3e}, "
        f"wave-vs-physical diff={diff_wave_physical:.3e}"
    )
    assert diff_wave_fused == 0.0, f"W={workers_per_fft}: wave vs fused mismatch {diff_wave_fused}"
    assert diff_wave_physical == 0.0, f"W={workers_per_fft}: wave vs physical mismatch {diff_wave_physical}"


def main() -> None:
    print("  Persistent-workgroup leaf: target-invariant rejection checks:")
    _check_target_invariant_rejections()
    print("  Persistent-workgroup leaf: registered-kernel-count stability across rounds:")
    _check_registered_kernel_count_stable_across_rounds()
    print("  Persistent-workgroup leaf: single stage invocation regardless of workers_per_fft (worker-wave fusion):")
    _check_single_invocation_per_stage_regardless_of_workers_per_fft()
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
    print("  Persistent-workgroup leaf: worker-wave (workers_per_fft > "
          "workers_per_group) numeric checks:")
    # Single round, waves=2 (workers_per_fft=16, no tail).
    _run_case(64, (4, 4, 4), num_logical_blocks=1, workers_per_fft=16)
    # Multi-round, waves=4 (workers_per_fft=32, no tail).
    _run_case(64, (4, 4, 4), num_logical_blocks=40, workers_per_fft=32)
    # Tail batch + waves=2 together.
    _run_case(105, (3, 5, 7), num_logical_blocks=40, workers_per_fft=16)
    # Tail batch + waves=4 + reserved scalar worker, all three at once.
    _run_case(
        105, (3, 5, 7), num_logical_blocks=40, workers_per_fft=32,
        scalar_worker_mode="reserved",
    )
    # Largest waves this target's max_kernel_register/candidate space
    # exercises (workers_per_fft=64, waves=8) -- matches a GPU-planner-
    # chosen workers_per_fft of 64 (e.g. clFFT at larger N).
    _run_case(64, (4, 4, 4), num_logical_blocks=96, workers_per_fft=64)

    print("  Persistent-workgroup leaf: ragged worker-wave (workers_per_fft "
          "not a divisor or multiple of workers_per_group) logical-worker "
          "coverage checks:")
    for w in (3, 5, 6, 7, 10, 12, 18, 20, 36):
        check_logical_worker_coverage(w, workers_per_group=8)
        print(f"    OK   workers_per_fft={w}: exact 0..{w - 1} coverage, no gap/duplicate")

    print("  Persistent-workgroup leaf: ragged dispatch source-level checks:")
    _check_dispatch_never_exceeds_w_and_has_no_else()

    print("  Persistent-workgroup leaf: ragged worker-wave numeric checks "
          "(GPU-baseline-observed W values: 3/5/6/7/10/12/18/20/36):")
    # Ragged single wave (W < workers_per_group=8), no tail.
    _run_case(64, (4, 4, 4), num_logical_blocks=1, workers_per_fft=3)
    _run_case(64, (4, 4, 4), num_logical_blocks=1, workers_per_fft=5)
    _run_case(64, (4, 4, 4), num_logical_blocks=1, workers_per_fft=7)
    # Ragged single wave, multi-round, reserved scalar worker.
    _run_case(
        64, (4, 4, 4), num_logical_blocks=40, workers_per_fft=6,
        scalar_worker_mode="reserved",
    )
    # Ragged multi-wave (W > workers_per_group, not a multiple) -- the
    # exact W values observed from real GPU baselines at N=12/20/40/80/
    # 120/216 (see docs/ragged_worker_wave_generalization.md).
    _run_case(64, (4, 4, 4), num_logical_blocks=1, workers_per_fft=10)
    _run_case(64, (4, 4, 4), num_logical_blocks=40, workers_per_fft=12)
    _run_case(64, (4, 4, 4), num_logical_blocks=40, workers_per_fft=18)
    _run_case(64, (4, 4, 4), num_logical_blocks=1, workers_per_fft=20)
    _run_case(64, (4, 4, 4), num_logical_blocks=96, workers_per_fft=36)
    # Ragged W combined with a real scalar tail batch (105 = 3*5*7).
    _run_case(105, (3, 5, 7), num_logical_blocks=40, workers_per_fft=6)
    _run_case(105, (3, 5, 7), num_logical_blocks=40, workers_per_fft=10)
    _run_case(105, (3, 5, 7), num_logical_blocks=40, workers_per_fft=18)

    print("  Persistent-workgroup leaf: worker-wave FUSION matrix -- "
          "W = 3,5,6,7,8,10,12,16,18,20,24,32,36 across single/multi "
          "replica, tail/no-tail, mixed-radix/power-of-two, single/multi "
          "round (docs/worker_wave_fusion.md's own required coverage):")
    fusion_w_values = (3, 5, 6, 7, 8, 10, 12, 16, 18, 20, 24, 32, 36)
    for w in fusion_w_values:
        check_logical_worker_coverage(w, workers_per_group=8)
    # N=64 = 4*4*4 (power-of-two, mixed within powers of 2), single replica.
    for w in fusion_w_values:
        _run_case(64, (4, 4, 4), num_logical_blocks=1, workers_per_fft=w)
    # N=64, multiple replicas -> multiple persistent rounds (40 blocks,
    # 32 software groups -> 2 rounds), reserved scalar worker mode too.
    for w in fusion_w_values:
        _run_case(64, (4, 4, 4), num_logical_blocks=40, workers_per_fft=w)
        _run_case(
            64, (4, 4, 4), num_logical_blocks=40, workers_per_fft=w,
            scalar_worker_mode="reserved",
        )
    # N=105 = 3*5*7 (genuinely mixed-radix, has a real scalar tail batch
    # every stage), single and multiple replicas.
    for w in fusion_w_values:
        _run_case(105, (3, 5, 7), num_logical_blocks=1, workers_per_fft=w)
        _run_case(105, (3, 5, 7), num_logical_blocks=40, workers_per_fft=w)
    # N=216 = 6*6*6 (the exact clFFT-observed ragged W=18 case) and its
    # own rocFFT-default/VkFFT W=36 sibling, both previously-tested
    # real-toolchain lengths.
    _run_case(216, (6, 6, 6), num_logical_blocks=1, workers_per_fft=18)
    _run_case(216, (6, 6, 6), num_logical_blocks=40, workers_per_fft=18)
    _run_case(216, (6, 6, 6), num_logical_blocks=1, workers_per_fft=36)
    _run_case(216, (6, 6, 6), num_logical_blocks=40, workers_per_fft=36)

    print("  Persistent-workgroup leaf: explicit exec_mode='wave'/'fused' "
          "regression (mode-parameter routing, not the default 'physical' "
          "path the matrix above already exercised exhaustively):")
    for w in fusion_w_values:
        _run_case(64, (4, 4, 4), num_logical_blocks=40, workers_per_fft=w, exec_mode="wave")
        _run_case(64, (4, 4, 4), num_logical_blocks=40, workers_per_fft=w, exec_mode="fused")

    print("  Persistent-workgroup leaf: physical-lane strip-mining (Mode C) "
          "bit-for-bit cross-mode equivalence -- wave vs fused vs physical "
          "on the SAME plan/input, not just each vs numpy:")
    for w in (3, 5, 6, 7, 8, 10, 12, 16, 18, 20, 24, 32, 36):
        _check_mode_equivalence(64, (4, 4, 4), num_logical_blocks=40, workers_per_fft=w)
        _check_mode_equivalence(105, (3, 5, 7), num_logical_blocks=40, workers_per_fft=w)
    for w in (10, 18, 36):
        _check_mode_equivalence(216, (6, 6, 6), num_logical_blocks=1, workers_per_fft=w)
        _check_mode_equivalence(216, (6, 6, 6), num_logical_blocks=40, workers_per_fft=w)
        _check_mode_equivalence(216, (6, 6, 6), num_logical_blocks=40, workers_per_fft=w, inverse=True)
    print("[verify] persistent leaf: all cases matched numpy's FFT")


if __name__ == "__main__":
    main()
