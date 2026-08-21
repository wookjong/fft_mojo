from __future__ import annotations

"""Shared numeric-execution harness for every verify_fft_*.py module:
translates the *actual emitted Mojo-ish stage text* (via each codegen
module's own `_emit_*` functions) into executable Python and re-runs it,
rather than reimplementing the algorithm a second time -- see
verify_fft_plan.py's own module docstring for the full rationale and the
translation rules this covers.

`run_kernel`/`_make_large_twiddle_table`/`_run_kernel_chain`/
`_run_multi_kernel_plan` operate on plain `FFTCodegenPlan`/
`MultiKernelFFTPlan` shapes, so they're reused by more than one strategy
(multikernel and balanced both return a `MultiKernelFFTPlan`) -- kept here
rather than in any one verify_fft_*.py module, the same "used by 2+
strategies -> shared module" rule fft_plan_core.py itself follows.
Plan-type-specific translate/run/twiddle-table trios (for FFTTransposePlan,
PhysicalTransposePlan) live in their own strategy's verify_fft_*.py file
instead, since nothing outside that one strategy needs them.
"""

import re
import types

import numpy as np

from codegen.fft_codegen import Emitter, _emit_stage
from planning.fft_plan_core import FFTCodegenPlan

_VAR_RE = re.compile(r"^(\s*)var ")
_COMPTIME_RE = re.compile(r"^(\s*)comptime ")
_LOAD_W_RE = re.compile(r"\.load\[width=(\w+)\]\(([^()]*)\)")
_LOAD_DT_RE = re.compile(r"\.load\[DType\.float32,\s*(\w+)\]\(([^()]*)\)")
_SIMD_RE = re.compile(r"SIMD\[DType\.float32,\s*\w+\]\(")


def _translate_emitted_lines(lines: list[str]) -> str:
    """Shared by every _translate_*_stage below: turn one stage's actual
    emitted lines (class-body indented, Mojo-flavored) into executable
    top-level Python -- strip declarations/comments/the `ref p =` binding,
    rewrite the handful of Mojo-only syntax forms (`var`/`comptime`
    declarations, `.load[width=W](...)`, `SIMD[DType.float32, W](...)`)
    into plain Python, then dedent by the one `class-body -> top-level def`
    level every caller's own `_emit_*_stage` uses. Only the emit call
    (which lines get produced) differs between callers -- this is purely
    text translation, so it's the same regardless of which plan/address
    space the emitted stage belongs to."""
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if s == "" or s.startswith("#") or s == "@staticmethod":
            continue
        if s.startswith("ref p ="):
            continue
        line = _VAR_RE.sub(r"\1", line)
        line = _COMPTIME_RE.sub(r"\1", line)
        line = _LOAD_W_RE.sub(r".load(\2, \1)", line)
        line = _LOAD_DT_RE.sub(r".load(\2, \1)", line)
        line = _SIMD_RE.sub("SIMD(", line)
        assert line.startswith("    "), line
        out.append(line[4:])
    return "\n".join(out)



def _translate_stage(plan: FFTCodegenPlan, stage) -> str:
    e = Emitter()
    _emit_stage(e, plan=plan, stage=stage)
    return _translate_emitted_lines(e.lines)



class SimdVec(np.ndarray):
    """A numpy array with Mojo's SIMD value semantics instead of numpy's.

    `var or0 = rr0` (a bare name-to-name assignment -- e.g.
    fft_butterflies._emit_symmetric_odd_radix's `or0 = rr0; or0 += ...`
    accumulation pattern) is a *copy* in Mojo: SIMD is a value type, so
    later mutating `or0` never touches `rr0`. Plain numpy aliases the same
    buffer and `+=` mutates it in place, silently corrupting `rr0` for
    every later line that reads it -- caught by the radix-7/11/13/17 cases
    in Stage 2 (radix-2/3/4/6/8/9/10/16's butterflies never do a bare
    copy-then-accumulate, so this stayed latent through every earlier
    verified case). Disabling the in-place dunders makes Python's `+=`
    fall back to `self = self + other`, which rebinds instead of
    mutating -- exactly Mojo's copy behavior.
    """

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __itruediv__(self, other):
        return self / other


class Ptr:
    """Stand-in for an UnsafePointer[Float32]: a flat float64 buffer."""

    def __init__(self, n: int) -> None:
        self.arr = np.zeros(n, dtype=np.float64)

    def load(self, offset: int, width: int) -> SimdVec:
        offset = int(offset)
        return self.arr[offset : offset + int(width)].copy().view(SimdVec)

    def store(self, offset: int, value) -> None:
        offset = int(offset)
        value = np.asarray(value, dtype=np.float64)
        if value.ndim == 0:
            self.arr[offset] = float(value)
        else:
            flat = value.reshape(-1)
            self.arr[offset : offset + flat.size] = flat


def _simd(*args: float) -> SimdVec:
    return np.array(args, dtype=np.float64).view(SimdVec)


def run_kernel(
    plan: FFTCodegenPlan,
    *,
    input_real: Ptr,
    input_imag: Ptr,
    output_real: Ptr,
    output_imag: Ptr,
    large_twiddle_real: Ptr | None = None,
    large_twiddle_imag: Ptr | None = None,
) -> None:
    """Run every stage of `plan`'s device_main, for every uthread, by
    exec()ing the actual text fft_codegen.py emits for each stage -- the
    same sequencing `launch_parallel[Self.stage_N]()` describes: every
    uthread finishes stage N before stage N+1 starts.

    `plan.total_uthreads` (the whole launch) may exceed `plan.max_uthread`
    (how many of this kernel's own uthreads share one NDP unit's
    scratchpad -- see FFTCodegenPlan's docstring): group `g =
    global_id // max_uthread` gets its own independent scratchpad
    instance ("one instance per core," not one shared array for the whole
    launch -- docs/INTERFACE.md), and `local_uthread_id() = global_id %
    max_uthread` indexes within it.
    """
    num_groups = -(-plan.total_uthreads // plan.max_uthread)  # ceil div
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
    if plan.large_twiddle is not None:
        assert large_twiddle_real is not None and large_twiddle_imag is not None
        p_ns.large_twiddle_real_base = large_twiddle_real
        p_ns.large_twiddle_imag_base = large_twiddle_imag

    current = {"global_id": 0, "local_id": 0}

    for stage in plan.stages:
        src = _translate_stage(plan, stage)
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
            current["local_id"] = global_id % plan.max_uthread
            namespace[plan.kernel_name] = group_namespaces[global_id // plan.max_uthread]
            stage_fn()



def _make_large_twiddle_table(lt) -> tuple[np.ndarray, np.ndarray]:
    """Built directly over the fetch address (0..full_length-1), matching
    the host precompute this plan's kernel actually emits (see
    fft_codegen._emit_large_twiddle_table_precompute) -- the twiddle
    *value* only ever depends on (b, digit) (b = the not-yet-transformed
    remainder row//a, digit = this kernel's own output), never on which
    address formula placed it, so only the addr -> (b, digit) inverse
    below differs between mapping kinds.

    The `a>1, k_next=0` case (addr = out_a + digit*a + b*a*output_count,
    out_a in [0,a), redundant) is SPLIT's own now-unused table layout --
    no current planner ever constructs a LargeTwiddlePlan with that
    combination (see AddressMappingKind.SPLIT's own docstring), so this
    branch stays untested by every case below; a=1 (the only combination
    any planner actually produces alongside k_next=0) degenerates to
    large_twiddle[row*output_count+c1] = W_full_length^(row*c1).

    PEELED (see AddressMapping.peeled / AddressMappingKind.PEELED):
    addr = rest*(a*output_count*k_next) + (out_a+digit*a)*k_next + d_next
    (out_a in [0,a), redundant); b = remaining = d_next*tail_size + rest.
    """
    sign = 1.0 if lt.inverse else -1.0
    addr = np.arange(lt.full_length)
    if lt.k_next:
        d_next = addr % lt.k_next
        combined_ao = (addr // lt.k_next) % (lt.a * lt.output_count)
        rest = addr // (lt.k_next * lt.a * lt.output_count)
        digit = combined_ao // lt.a
        b = d_next * lt.tail_size + rest
    else:
        a_ki = lt.a * lt.output_count
        b = addr // a_ki
        digit = (addr % a_ki) // lt.a
    angle = sign * 2.0 * np.pi * (b * digit * lt.a) / lt.full_length
    return np.cos(angle), np.sin(angle)



def _run_kernel_chain(
    kernels: tuple[FFTCodegenPlan, ...], *, input_real: Ptr, input_imag: Ptr
) -> tuple[Ptr, Ptr]:
    """Chains `kernels` through DRAM -- one large-twiddle table per
    non-last kernel, exactly the M-kernel PEELED-chain contract
    make_multi_kernel_plan documents -- each re-executing its own actual
    emitted stage text (run_kernel). Generic over how/which planner built
    `kernels`: shared by _run_multi_kernel_plan (a whole MultiKernelFFTPlan)
    and verify_balanced_transpose_plan (just one side's own kernels, with a
    standalone transpose kernel run separately in between the two sides).
    """
    n = input_real.arr.size
    cur_r, cur_i = input_real, input_imag
    for kernel in kernels[:-1]:
        next_r, next_i = Ptr(n), Ptr(n)
        lt = kernel.large_twiddle
        assert lt is not None
        lt_real_vals, lt_imag_vals = _make_large_twiddle_table(lt)
        lt_r, lt_i = Ptr(n), Ptr(n)
        lt_r.arr[:] = lt_real_vals
        lt_i.arr[:] = lt_imag_vals
        run_kernel(
            kernel,
            input_real=cur_r, input_imag=cur_i,
            output_real=next_r, output_imag=next_i,
            large_twiddle_real=lt_r, large_twiddle_imag=lt_i,
        )
        cur_r, cur_i = next_r, next_i

    out_r, out_i = Ptr(n), Ptr(n)
    run_kernel(
        kernels[-1],
        input_real=cur_r, input_imag=cur_i, output_real=out_r, output_imag=out_i,
    )
    return out_r, out_i


def _run_multi_kernel_plan(plan, *, inverse: bool, seed: int) -> float:
    """Compares plan.kernels' chained-through-DRAM result (_run_kernel_chain)
    to numpy. Generic over how the plan was built -- shared by
    verify_multi_kernel_plan and verify_balanced_plan, which differ only in
    which planner function produces `plan`.
    """
    n = plan.n
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)

    in_r, in_i = Ptr(n), Ptr(n)
    in_r.arr[:] = x.real
    in_i.arr[:] = x.imag

    out_r, out_i = _run_kernel_chain(plan.kernels, input_real=in_r, input_imag=in_i)

    got = out_r.arr + 1j * out_i.arr
    expected = np.fft.ifft(x) if inverse else np.fft.fft(x)
    return float(np.max(np.abs(got - expected)))

