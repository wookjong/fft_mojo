from __future__ import annotations

"""Verification for the persistent tail-round hybrid (P2.4 -- see docs/
priority2_execution_strategies.md and `fft_plan_persistent.make_
persistent_tail_hybrid_plan`'s own docstring).

Two things this suite proves, at the task's own required boundary batch
counts (1, 31, 32, 33, 40, 63, 64, 65 -- `target.num_ndp_units=32`):

A. STRUCTURAL: `full_blocks + tail_blocks == num_logical_blocks` always;
   `tail_blocks == 0` exactly when `num_logical_blocks` is already a
   multiple of 32 (or `tail_strategy="all_persistent"`); each of the
   `num_logical_blocks` FFT replicas is covered by EXACTLY ONE of the two
   sub-plans (never both, never neither) -- checked directly against the
   replica-count fields on each sub-plan, not assumed from the arithmetic
   alone.

B. NUMERIC: both `persistent_plan` (via `verify_fft_persistent.run_
   persistent_kernel`) and `tail_plan` (via `verify_fft_harness.run_
   kernel` for `noncoop_tail`, `verify_fft_cooperative.run_cooperative_
   kernel` for `cooperative_tail`) independently compute a numpy-correct
   FFT for their own respective replica ranges -- the actual emitted
   stage text, re-executed, never a re-implementation.

The HOST-LEVEL Mojo text `codegen.fft_persistent_codegen.generate_
persistent_tail_hybrid_kernel` emits (buffer allocation, two independent
`.launch()` calls in one shared `main()`) is checked only STRUCTURALLY
here (contains both kernel names/launches, well-formed) -- actually
compiling and running it requires the real Mojo/M2NDP-Detour toolchain,
UNMEASURED in this environment (confirmed absent, see docs/
transpose_cost_model_audit.md's own toolchain-availability check) -- see
this suite's own module docstring in the final report for why the
buffer-alloc/launch pattern used is not new (a byte-for-byte reuse of
`generate_persistent_fft_kernel`'s/`generate_fft_kernel`'s/`generate_
cooperative_fft_kernel`'s own already-shipped host blocks, just namespaced
by prefix).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from codegen.fft_persistent_codegen import generate_persistent_tail_hybrid_kernel
from planning.core.target_profile import DEFAULT_TARGET_PROFILE
from planning.execution.fft_plan_persistent import make_persistent_tail_hybrid_plan
from verification.verify_fft_cooperative import run_cooperative_kernel
from verification.verify_fft_harness import Ptr, run_kernel
from verification.verify_fft_persistent import run_persistent_kernel

_T = DEFAULT_TARGET_PROFILE
_BOUNDARY_BATCH_COUNTS = (1, 31, 32, 33, 40, 63, 64, 65)


def check_full_blocks_plus_tail_equals_total() -> None:
    for num_blocks in _BOUNDARY_BATCH_COUNTS:
        for tail_strategy in ("all_persistent", "noncoop_tail", "cooperative_tail"):
            hybrid = make_persistent_tail_hybrid_plan(
                64, (4, 4, 4), num_logical_blocks=num_blocks, target=_T, tail_strategy=tail_strategy,
            )
            assert hybrid.full_blocks + hybrid.tail_blocks == num_blocks, (
                f"num_blocks={num_blocks} strategy={tail_strategy}: full_blocks("
                f"{hybrid.full_blocks}) + tail_blocks({hybrid.tail_blocks}) != {num_blocks}"
            )
            if tail_strategy == "all_persistent" or num_blocks % _T.num_ndp_units == 0:
                assert hybrid.tail_blocks == 0, (
                    f"num_blocks={num_blocks} strategy={tail_strategy}: expected no tail "
                    f"(already a multiple of {_T.num_ndp_units}, or strategy forces none), "
                    f"got tail_blocks={hybrid.tail_blocks}"
                )
                assert hybrid.tail_plan is None
            else:
                assert hybrid.tail_blocks == num_blocks % _T.num_ndp_units
                assert hybrid.tail_plan is not None
    print(f"    OK   full_blocks + tail_blocks == num_logical_blocks at every boundary count in "
          f"{_BOUNDARY_BATCH_COUNTS}, for all 3 tail_strategy values")


def check_each_replica_covered_exactly_once() -> None:
    for num_blocks in _BOUNDARY_BATCH_COUNTS:
        for tail_strategy in ("noncoop_tail", "cooperative_tail"):
            hybrid = make_persistent_tail_hybrid_plan(
                64, (4, 4, 4), num_logical_blocks=num_blocks, target=_T, tail_strategy=tail_strategy,
            )
            if hybrid.persistent_plan is not None:
                persistent_replicas = hybrid.persistent_plan.host.total_elems // hybrid.persistent_plan.length
                assert persistent_replicas == hybrid.full_blocks
            else:
                assert hybrid.full_blocks == 0
                persistent_replicas = 0
            if hybrid.tail_plan is not None:
                if hybrid.tail_plan.cooperation is not None:
                    tail_replicas = (
                        hybrid.tail_plan.total_uthreads // hybrid.tail_plan.cooperation.workers_per_fft
                    )
                else:
                    tail_replicas = hybrid.tail_plan.total_uthreads
                assert tail_replicas == hybrid.tail_blocks, (
                    f"num_blocks={num_blocks} strategy={tail_strategy}: tail_plan's own replica "
                    f"count ({tail_replicas}) != hybrid.tail_blocks ({hybrid.tail_blocks})"
                )
            total_covered = persistent_replicas + (hybrid.tail_blocks if hybrid.tail_plan is not None else 0)
            assert total_covered == num_blocks, (
                f"num_blocks={num_blocks} strategy={tail_strategy}: total covered replicas "
                f"({total_covered}) != {num_blocks}"
            )
    print(f"    OK   each of the {len(_BOUNDARY_BATCH_COUNTS)} boundary batch counts' own FFT "
          f"replicas is covered exactly once, split correctly between persistent_plan and tail_plan")


def _numpy_reference(xr, xi, length, num_blocks, inverse=False):
    sign = 1 if inverse else -1
    out = []
    for b in range(num_blocks):
        block = xr[b * length:(b + 1) * length] + 1j * xi[b * length:(b + 1) * length]
        if inverse:
            out.append(np.fft.ifft(block) * length)  # unnormalized, matches this project's own convention
        else:
            out.append(np.fft.fft(block))
    return np.concatenate(out)


def check_persistent_portion_numeric_correctness() -> None:
    for num_blocks in (33, 65):
        hybrid = make_persistent_tail_hybrid_plan(
            64, (4, 4, 4), num_logical_blocks=num_blocks, target=_T, tail_strategy="noncoop_tail",
        )
        total = hybrid.full_blocks * 64
        rng = np.random.default_rng(3)
        xr, xi = rng.uniform(-1, 1, total), rng.uniform(-1, 1, total)
        in_r, in_i = Ptr(total), Ptr(total)
        for i in range(total):
            in_r.arr[i] = xr[i]
            in_i.arr[i] = xi[i]
        out_r, out_i = Ptr(total), Ptr(total)
        run_persistent_kernel(
            hybrid.persistent_plan, num_logical_blocks=hybrid.full_blocks,
            input_real=in_r, input_imag=in_i, output_real=out_r, output_imag=out_i,
        )
        got = out_r.arr + 1j * out_i.arr
        expected = _numpy_reference(xr, xi, 64, hybrid.full_blocks)
        err = float(np.max(np.abs(got - expected)))
        assert err < 1e-3, f"num_blocks={num_blocks}: persistent portion max_err={err}"
    print("    OK   persistent_plan's own full_blocks replicas compute a numpy-correct FFT "
          "(num_blocks=33 and 65)")


def check_noncoop_tail_numeric_correctness() -> None:
    for num_blocks in (33, 65, 1):
        hybrid = make_persistent_tail_hybrid_plan(
            64, (4, 4, 4), num_logical_blocks=num_blocks, target=_T, tail_strategy="noncoop_tail",
        )
        if hybrid.tail_plan is None:
            continue
        total = hybrid.tail_blocks * 64
        rng = np.random.default_rng(5)
        xr, xi = rng.uniform(-1, 1, total), rng.uniform(-1, 1, total)
        in_r, in_i = Ptr(total), Ptr(total)
        for i in range(total):
            in_r.arr[i] = xr[i]
            in_i.arr[i] = xi[i]
        out_r, out_i = Ptr(total), Ptr(total)
        run_kernel(hybrid.tail_plan, input_real=in_r, input_imag=in_i, output_real=out_r, output_imag=out_i)
        got = out_r.arr + 1j * out_i.arr
        expected = _numpy_reference(xr, xi, 64, hybrid.tail_blocks)
        err = float(np.max(np.abs(got - expected)))
        assert err < 1e-3, f"num_blocks={num_blocks}: noncoop tail max_err={err}"
    print("    OK   tail_plan (noncoop_tail strategy) computes a numpy-correct FFT for its own "
          "tail_blocks replicas (num_blocks=33, 65, 1)")


def check_cooperative_tail_numeric_correctness() -> None:
    for num_blocks in (33, 65):
        hybrid = make_persistent_tail_hybrid_plan(
            64, (4, 4, 4), num_logical_blocks=num_blocks, target=_T, tail_strategy="cooperative_tail",
        )
        assert hybrid.tail_plan is not None
        assert hybrid.tail_plan.cooperation is not None
        workers = hybrid.tail_plan.cooperation.workers_per_fft
        total = hybrid.tail_blocks * 64
        rng = np.random.default_rng(9)
        xr, xi = rng.uniform(-1, 1, total), rng.uniform(-1, 1, total)
        in_r, in_i = Ptr(total), Ptr(total)
        for i in range(total):
            in_r.arr[i] = xr[i]
            in_i.arr[i] = xi[i]
        out_r, out_i = Ptr(total), Ptr(total)
        run_cooperative_kernel(
            hybrid.tail_plan, input_real=in_r, input_imag=in_i, output_real=out_r, output_imag=out_i,
        )
        got = out_r.arr + 1j * out_i.arr
        expected = _numpy_reference(xr, xi, 64, hybrid.tail_blocks)
        err = float(np.max(np.abs(got - expected)))
        assert err < 1e-3, f"num_blocks={num_blocks}: cooperative tail (workers={workers}) max_err={err}"
    print("    OK   tail_plan (cooperative_tail strategy) computes a numpy-correct FFT for its own "
          "tail_blocks replicas (num_blocks=33, 65)")


def check_all_persistent_is_unchanged_default() -> None:
    """`tail_strategy='all_persistent'` must produce a `persistent_plan`
    IDENTICAL to calling `make_persistent_leaf_plan` directly (byte-for-
    byte, via the same construction path) -- confirms the default
    strategy is a true no-op relative to every plan built before P2.4
    existed."""
    from planning.execution.fft_plan_persistent import make_persistent_leaf_plan

    for num_blocks in (33, 65, 1):
        hybrid = make_persistent_tail_hybrid_plan(
            64, (4, 4, 4), num_logical_blocks=num_blocks, target=_T, tail_strategy="all_persistent",
        )
        direct = make_persistent_leaf_plan(64, (4, 4, 4), num_logical_blocks=num_blocks, target=_T)
        assert hybrid.tail_plan is None
        assert hybrid.full_blocks == num_blocks
        assert hybrid.persistent_plan == direct, (
            f"num_blocks={num_blocks}: all_persistent's own persistent_plan differs from a direct "
            f"make_persistent_leaf_plan call"
        )
    print("    OK   tail_strategy='all_persistent' produces a persistent_plan byte-identical to "
          "make_persistent_leaf_plan called directly -- true no-op default")


def check_generated_source_structure() -> None:
    hybrid = make_persistent_tail_hybrid_plan(
        64, (4, 4, 4), num_logical_blocks=33, target=_T, tail_strategy="noncoop_tail",
    )
    src = generate_persistent_tail_hybrid_kernel(hybrid, target=_T)
    assert "struct PersistentFFT(NDPTask):" in src
    assert "struct TailFFT(NDPTask):" in src
    assert "PersistentFFT.launch(" in src
    assert "TailFFT.launch(" in src
    assert src.count("def main() raises:") == 1, "expected exactly one shared host main()"

    # all_persistent degenerates to the plain single-kernel generator.
    hybrid_default = make_persistent_tail_hybrid_plan(64, (4, 4, 4), num_logical_blocks=33, target=_T)
    from codegen.fft_persistent_codegen import generate_persistent_fft_kernel
    src_default = generate_persistent_tail_hybrid_kernel(hybrid_default, target=_T)
    src_direct = generate_persistent_fft_kernel(hybrid_default.persistent_plan, num_logical_blocks=33)
    assert src_default == src_direct, "all_persistent must byte-match generate_persistent_fft_kernel directly"
    print("    OK   generated source contains both kernel structs/launches and exactly one shared "
          "main() for a real hybrid case; all_persistent byte-matches the plain single-kernel "
          "generator (UNMEASURED: real compilation/execution requires the Mojo/M2NDP-Detour "
          "toolchain, not available in this environment)")


def check_no_host_variable_collision() -> None:
    """Regression test for a REAL toolchain compile failure found during
    physical validation (see verification/results/priority2_physical_
    validation.json, experiment C, N33/noncoop_tail and N63/N65's own
    noncoop_tail/cooperative_tail rows): `codegen.common.emit_reference_
    check` used to always emit unprefixed `var pi`/`var sign`/`var tol`,
    and `generate_persistent_tail_hybrid_kernel` calls it TWICE into one
    shared host `main()` whenever BOTH `full_blocks > 0` and `tail_blocks
    > 0` (e.g. num_logical_blocks=33/63/65 at num_ndp_units=32) -- the
    real Mojo compiler rejected this with "invalid redefinition of 'pi'"
    (confirmed on the actual toolchain, not assumed). `full_blocks == 0`
    (num_blocks=31: no persistent portion) or `tail_blocks == 0`
    (num_blocks=32/64: `tail_plan is None`, degenerates to one kernel)
    never triggered it -- only a genuine two-real-kernel hybrid does,
    which is exactly why this went undetected by every earlier
    STRUCTURAL-only check in this file (`check_generated_source_
    structure` never parsed/compiled the source, just substring-checked
    it). The fix threads `emit_reference_check`'s own new `var_prefix`
    parameter through the two hybrid host-body helpers (`"p_"`/`"t_"`,
    the SAME prefix every other host variable in that block already
    used) -- every other existing caller keeps the default `var_prefix=
    ""`, so this test also confirms the single-kernel path is untouched."""
    for num_blocks, tail_strategy in ((33, "noncoop_tail"), (65, "cooperative_tail")):
        hybrid = make_persistent_tail_hybrid_plan(
            64, (4, 4, 4), num_logical_blocks=num_blocks, target=_T, tail_strategy=tail_strategy,
        )
        assert hybrid.full_blocks > 0 and hybrid.tail_blocks > 0, (
            f"num_blocks={num_blocks}: expected a genuine two-kernel hybrid for this regression test"
        )
        src = generate_persistent_tail_hybrid_kernel(hybrid, target=_T)
        assert "    var pi = " not in src, (
            f"num_blocks={num_blocks}/{tail_strategy}: found an unprefixed 'var pi' -- the exact "
            f"host-variable collision that made this real hybrid fail to compile on the actual "
            f"Mojo toolchain ('invalid redefinition of pi')"
        )
        assert "    var sign = " not in src and "    var tol = " not in src, (
            f"num_blocks={num_blocks}/{tail_strategy}: found an unprefixed 'var sign'/'var tol'"
        )
        for prefix in ("p_", "t_"):
            for name in ("pi", "sign", "tol"):
                assert f"    var {prefix}{name} = " in src, (
                    f"num_blocks={num_blocks}/{tail_strategy}: missing expected {prefix}{name}"
                )

    # Single-kernel paths (no second reference-check block in the same
    # scope) must still use the plain, unprefixed names -- byte-identical
    # to every plan rendered before `var_prefix` existed.
    from codegen.fft_persistent_codegen import generate_persistent_fft_kernel
    plain_hybrid = make_persistent_tail_hybrid_plan(
        64, (4, 4, 4), num_logical_blocks=33, target=_T, tail_strategy="all_persistent",
    )
    plain_src = generate_persistent_fft_kernel(plain_hybrid.persistent_plan, num_logical_blocks=33)
    assert "    var pi = " in plain_src and "    var sign = " in plain_src and "    var tol = " in plain_src
    print("    OK   the real toolchain 'invalid redefinition of pi/sign/tol' compile failure "
          "(N33/noncoop_tail, N65/cooperative_tail) does not reproduce after the var_prefix fix; "
          "single-kernel paths remain byte-identical (unprefixed)")


def main() -> None:
    print("  Persistent tail-round hybrid (P2.4):")
    check_full_blocks_plus_tail_equals_total()
    check_each_replica_covered_exactly_once()
    check_persistent_portion_numeric_correctness()
    check_noncoop_tail_numeric_correctness()
    check_cooperative_tail_numeric_correctness()
    check_all_persistent_is_unchanged_default()
    check_generated_source_structure()
    check_no_host_variable_collision()
    print("[verify] persistent tail-round hybrid: all checks passed")


if __name__ == "__main__":
    main()
