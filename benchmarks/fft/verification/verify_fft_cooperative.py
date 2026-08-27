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

import numpy as np

from planning.fft_plan_core import FFTCodegenPlan
from verification.verify_fft_harness import Ptr, run_kernel


def run_cooperative_kernel(
    plan: FFTCodegenPlan,
    *,
    input_real: Ptr,
    input_imag: Ptr,
    output_real: Ptr,
    output_imag: Ptr,
    compute_lanes: int | None = None,
    narrow_middle_stages: bool = False,
    loop_stages: bool = False,
) -> None:
    """Thin wrapper over `verify_fft_harness.run_kernel` -- this module used
    to carry its own parallel reimplementation of the group/local_id split
    and stage exec loop, byte-for-byte the same generic logic `run_kernel`
    already does for *any* `FFTCodegenPlan`, cooperative or not (that
    function's own `num_groups`/`local_id` handling never assumed
    non-cooperative in the first place). Kept as a distinct name for this
    module's own callers (`verify_cooperative_leaf`), not because the
    behavior actually differs from calling `run_kernel` directly.

    That duplication had silently fallen behind `run_kernel`'s own
    `compute_lanes`/`narrow_middle_stages`/`loop_stages` support (all
    added 2026-08-27, the same day as `fft_cooperative_codegen.
    _emit_cooperative_stage`'s own per-worker `loop_stages` fix) until
    this delegation replaced it -- every `verify_cooperative_leaf` call
    was silently testing full-width (`compute_lanes=None`), fully-unrolled
    (`loop_stages=False`) rendering, not the `compute_lanes=4`/
    `narrow_middle_stages=True`/`loop_stages=True` shape `make_fft_kernel.
    py` actually ships by default -- the same class of "the harness tests
    a code path nobody ships" gap `verify_fft_harness.run_kernel`'s own
    tail-batch fix closed for the non-cooperative case earlier the same
    day. See `verify_cooperative_leaf`'s own docstring for this function's
    new real defaults.
    """
    run_kernel(
        plan, input_real=input_real, input_imag=input_imag,
        output_real=output_real, output_imag=output_imag,
        compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
        loop_stages=loop_stages,
    )


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
    compute_lanes: int | None = None,
    narrow_middle_stages: bool = True,
    loop_stages: bool = True,
) -> float:
    """Plan + run a standalone cooperative leaf against `total_ffts`
    independent random length-`length` signals -- return the max abs error
    against numpy.fft. `in_place`: model the same buffer for input and
    output (see fft_codegen.generate_cooperative_fft_kernel's own docstring
    for why this is race-free) by handing `run_cooperative_kernel` one `Ptr`
    for both.

    `compute_lanes`/`narrow_middle_stages`/`loop_stages`: `True`/`True`
    and `compute_lanes=None` resolved the same way `make_fft_kernel()`
    resolves it (`min(simd_lanes, DEFAULT_TARGET_PROFILE.
    lmul1_float32_lanes)`) are the *real* shipped defaults -- unlike
    `run_cooperative_kernel`'s own former defaults (`False`/`False`/full
    width), which silently tested a code shape nobody ships until this
    function's own signature gained these. Pass `False`/explicit widths to
    compare against the old shape deliberately.
    """
    from planning.fft_plan_cooperative import make_cooperative_leaf_plan
    from planning.target_profile import DEFAULT_TARGET_PROFILE

    if compute_lanes is None:
        compute_lanes = min(simd_lanes, DEFAULT_TARGET_PROFILE.lmul1_float32_lanes)

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
        compute_lanes=compute_lanes, narrow_middle_stages=narrow_middle_stages,
        loop_stages=loop_stages,
    )

    got = out_r.arr + 1j * out_i.arr
    pieces = []
    for i in range(total_ffts):
        chunk = x[i * length : (i + 1) * length]
        pieces.append(np.fft.ifft(chunk) if inverse else np.fft.fft(chunk))
    expected = np.concatenate(pieces)
    return float(np.max(np.abs(got - expected)))
