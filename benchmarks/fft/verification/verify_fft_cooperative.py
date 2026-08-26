from __future__ import annotations

"""Numeric verification for cooperative-worker leaf FFTs (see
planning/fft_plan_cooperative.py and CooperationPlan in fft_plan_core.py).

`run_cooperative_kernel` mirrors `verify_fft_harness.run_kernel` exactly --
same discipline (re-exec the *actual emitted* stage text, never reimplement
the algorithm), same group/local split convention (`group = global_id //
max_uthread`, `local_id = global_id % max_uthread` -- block-major, one of
many interleavings the real hardware could pick, but the *only* one that
matters here is that it satisfies the same precondition
_emit_cooperative_stage documents: `max_uthread` is a multiple of
`workers_per_fft`, so this split's own `fft_slot`/`worker_id` grouping
(local-id-based) and `logical_fft_id`'s grouping (global-id-based) partition
the same physical microthreads into the same groups -- see
fft_codegen._emit_cooperative_stage's own docstring for why that alignment,
not the specific interleaving, is what correctness actually depends on).
`num_groups` is a parameter (how many `max_uthread`-sized blocks `total_uthreads`
splits into), mirroring `run_kernel`'s own `plan.total_uthreads // plan.max_uthread`.
"""

import types

import numpy as np

from planning.fft_plan_core import FFTCodegenPlan
from verification.verify_fft_harness import Ptr, _simd, _translate_stage


def run_cooperative_kernel(
    plan: FFTCodegenPlan,
    *,
    input_real: Ptr,
    input_imag: Ptr,
    output_real: Ptr,
    output_imag: Ptr,
) -> None:
    assert plan.cooperation is not None
    max_uthread = plan.max_uthread  # physical cap per group -- see CooperationPlan
    num_groups = -(-plan.total_uthreads // max_uthread)  # ceil div, mirrors run_kernel

    group_namespaces: list[types.SimpleNamespace] = []
    for _ in range(num_groups):
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

    stage_sources = {
        stage.stage_id: _translate_stage(plan, stage) for stage in plan.stages
    }

    current = {"global_id": 0, "local_id": 0}

    for stage in plan.stages:
        src = stage_sources[stage.stage_id]
        namespace = {
            "Float32": float,
            "SIMD": _simd,
            "local_uthread_id": lambda: current["local_id"],
            "global_uthread_id": lambda: current["global_id"],
            "N": plan.length,
            "W": plan.simd_lanes,
            f"MAX_UTHREAD_{plan.kernel_name}": plan.max_uthread,
            "p": p_ns,
        }
        code = compile(src, f"<{plan.kernel_name} stage {stage.stage_id}>", "exec")
        exec(code, namespace)
        stage_fn = namespace[f"stage_{stage.stage_id}"]
        for global_id in range(plan.total_uthreads):
            current["global_id"] = global_id
            current["local_id"] = global_id % max_uthread
            namespace[plan.kernel_name] = group_namespaces[global_id // max_uthread]
            stage_fn()


def verify_cooperative_leaf(
    length: int,
    radices: tuple[int, ...],
    *,
    workers_per_fft: int,
    total_ffts: int,
    inverse: bool = False,
    simd_lanes: int = 8,
    seed: int = 0,
    in_place: bool = False,
) -> float:
    """Plan + run a standalone cooperative leaf against `total_ffts`
    independent random length-`length` signals -- return the max abs error
    against numpy.fft. `in_place`: model the same buffer for input and
    output (see fft_codegen.generate_cooperative_fft_kernel's own docstring
    for why this is race-free) by handing `run_cooperative_kernel` one `Ptr`
    for both.
    """
    from planning.fft_plan_cooperative import make_cooperative_leaf_plan

    plan = make_cooperative_leaf_plan(
        length=length,
        radices=radices,
        workers_per_fft=workers_per_fft,
        total_ffts=total_ffts,
        inverse=inverse,
        simd_lanes=simd_lanes,
        kernel_name="FFTCoopVerify",
    )

    n = length * total_ffts
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)

    in_r, in_i = Ptr(n), Ptr(n)
    in_r.arr[:] = x.real
    in_i.arr[:] = x.imag
    if in_place:
        out_r, out_i = in_r, in_i
    else:
        out_r, out_i = Ptr(n), Ptr(n)

    run_cooperative_kernel(
        plan,
        input_real=in_r, input_imag=in_i, output_real=out_r, output_imag=out_i,
    )

    got = out_r.arr + 1j * out_i.arr
    pieces = []
    for i in range(total_ffts):
        chunk = x[i * length : (i + 1) * length]
        pieces.append(np.fft.ifft(chunk) if inverse else np.fft.fft(chunk))
    expected = np.concatenate(pieces)
    return float(np.max(np.abs(got - expected)))
