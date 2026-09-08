import re
import sys
from pathlib import Path

import numpy as np

# benchmarks/fft/ (this file's grandparent) holds the role directories
# (planning/, codegen/, verification/) as importable packages -- see
# verify_fft_plan.py's own bootstrap comment for why only entry-point
# scripts need this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codegen.fft_butterflies import emit_butterfly
from codegen.fft_codegen import generate_fft_kernel, generate_multi_kernel_fft_kernels
from planning.strategies.fft_plan_simple import make_444_plan
from planning.strategies.fft_plan_multikernel import factor_into_kernel_chunks, make_multi_kernel_plan
from radix_spec import SUPPORTED_RADICES

_VAR_RE = re.compile(r"^(\s*)var ")


class _TextEmitter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, line: str = "") -> None:
        self.lines.append(line)


def _butterfly_matrix(radix: int, *, inverse: bool) -> np.ndarray:
    """The DFT matrix emit_butterfly is supposed to compute.

    `fft_butterflies._emit_complex_twiddle` fixes the sign convention:
    forward multiplies by W_modulus**exponent with angle = -2*pi*exponent/modulus,
    inverse flips that sign -- and never divides by radix (that scaling is
    the codegen's job elsewhere). So the reference is the *unnormalized*
    DFT/IDFT matrix, not numpy.fft's normalized ifft.
    """
    sign = 1.0 if inverse else -1.0
    n = np.arange(radix)
    exponent = np.outer(n, n)
    return np.exp(sign * 2j * np.pi * exponent / radix)


def _run_butterfly(radix: int, *, inverse: bool, x: np.ndarray) -> np.ndarray:
    """Emit one radix's butterfly as Mojo text and actually execute it.

    The emitted lines (`var name = expr`) are already valid Python once
    `var ` is stripped -- `emit_butterfly`'s docstring promises it "knows
    nothing about scratchpad layout," just SIMD-lane arithmetic on already
    loaded values, so running that arithmetic straight in Python with real
    numbers bound to rr{k}/ii{k} is a faithful re-execution, not a
    reimplementation that could hide the same bug twice.
    """
    e = _TextEmitter()
    emit_butterfly(e, indent="", radix=radix, inverse=inverse)

    body_lines = [
        _VAR_RE.sub(r"\1", line)
        for line in e.lines
        if line and not line.lstrip().startswith("#")
    ]
    code = compile("\n".join(body_lines), f"<butterfly radix={radix}>", "exec")

    scope: dict[str, float] = {"Float32": float}
    for k in range(radix):
        scope[f"rr{k}"] = float(x[k].real)
        scope[f"ii{k}"] = float(x[k].imag)
    exec(code, scope)

    return np.array(
        [scope[f"or{k}"] + 1j * scope[f"oi{k}"] for k in range(radix)]
    )


def verify_butterflies(*, seed: int = 0, tolerance: float = 1.0e-5) -> None:
    """Actually verify every supported radix's butterfly, forward and
    inverse, against its DFT matrix -- what this file's name has always
    promised and, until now, never did (see fft_codegen.py's own
    docstring: this module only emits code, nothing runs it).
    """
    rng = np.random.default_rng(seed)
    failures: list[str] = []

    for radix in sorted(SUPPORTED_RADICES):
        for inverse in (False, True):
            x = rng.uniform(-1, 1, radix) + 1j * rng.uniform(-1, 1, radix)
            got = _run_butterfly(radix, inverse=inverse, x=x)
            expected = _butterfly_matrix(radix, inverse=inverse) @ x
            err = np.max(np.abs(got - expected))
            tag = f"radix={radix:<2} inverse={inverse}"
            if err > tolerance:
                failures.append(f"{tag}: max error {err:.3e}")
                print(f"  FAIL {tag}: max error {err:.3e}")
            else:
                print(f"  OK   {tag}: max error {err:.3e}")

    if failures:
        raise AssertionError(
            f"{len(failures)} butterfly radix/direction combination(s) failed:\n"
            + "\n".join(failures)
        )
    print(f"[verify] all {2 * len(SUPPORTED_RADICES)} butterflies passed")


def main() -> None:
    here = Path(__file__).resolve().parent / "fixtures"

    print("Verifying FFT butterflies against their DFT matrices...")
    verify_butterflies()

    # Test case A: single-kernel N=64, radix 4x4x4 -- whole FFT in one
    # uthread's own scratchpad, no decomposition. simd_lanes=4 (half the
    # hardware's 8-lane width): at simd_lanes=8 this kernel's stages spill
    # to DRAM (LLVM runs out of vector registers -- three radix-4 stages
    # fused via scratchpad ping-pong need more live SIMD values than fit at
    # LMUL=2); simd_lanes=4 halves LMUL to 1 and clears it, confirmed by
    # rebuilding through the real toolchain and checking llc's spill
    # warning, not by inspection.
    plan = make_444_plan(simd_lanes=4)
    source = generate_fft_kernel(plan)
    output = here / "fft_fp32_generated.mojo"
    output.write_text(source, encoding="utf-8")
    print(f"generated: {output}")

    # Test case B: N=256 = 16*16, too large for one uthread's scratchpad --
    # kernel0 (16 uthreads, a 4x4 multi-stage sub-FFT via scratchpad
    # ping-pong + large twiddle + permuted store each) then kernel1 (16
    # uthreads, another 4x4 multi-stage sub-FFT each), chained through DRAM.
    #
    # Built via make_multi_kernel_plan(((4, 4), (4, 4))) rather than
    # make_decomposed_plan(16, 16) -- both produce a working M=2, N0=N1=16
    # decomposition, but no longer field-for-field identical addressing:
    # make_multi_kernel_plan's non-last kernel now uses
    # AddressMappingKind.PEELED (kernel1's read is a real vector load
    # instead of the uniform scalar strided(elem_stride=n//K) both used to
    # share), while make_decomposed_plan is untouched, still SPLIT/
    # contiguous-based. Both are independently numpy-verified correct.
    # Also: an atomic radix-16 butterfly spills to DRAM regardless of
    # simd_lanes (confirmed down to simd_lanes=1): _emit_factorized's
    # first-stage "groups" for a=4,b=4 needs 16 pairs live simultaneously
    # before any final output exists, independent of store timing or SIMD
    # width. Two chained radix-4 *stages* need only one stage's 4 pairs
    # live at a time -- the same structure that already clears kernel0 of
    # test case C. See fft_plan_multikernel.make_multi_kernel_plan /
    # factor_into_kernel_chunks.
    decomposed = make_multi_kernel_plan(((4, 4), (4, 4)), simd_lanes=4)
    decomposed_source = generate_multi_kernel_fft_kernels(decomposed)
    decomposed_output = here / "fft_fp32_decomposed_generated.mojo"
    decomposed_output.write_text(decomposed_source, encoding="utf-8")
    print(f"generated: {decomposed_output}")

    # Test case C: N=960, three kernels chained through DRAM. Exercises
    # AddressMappingKind.SPLIT (the runtime %/// a middle kernel's output
    # needs) end to end. See fft_plan_multikernel.make_multi_kernel_plan.
    #
    # Chunks come from factor_into_kernel_chunks rather than a hand-picked
    # radix split: every kernel in the chain pays both a *read* stride
    # (n // its own length -- large whenever that one kernel is small,
    # regardless of chain position) and a *write* stride (`a`, the product
    # of every earlier kernel's length, which only grows kernel to kernel
    # -- the last kernel included). factor_into_kernel_chunks picks split
    # points to minimize the largest of the two across the whole chain --
    # see its docstring. A hand-picked ((4, 4, 4), (3,), (5,)) has a
    # write-side max of 192; scratchpad_byte_budget=256 is the smallest
    # budget that still yields three kernels (preserving the SPLIT exercise
    # this test case exists for -- larger budgets collapse it to M=2 or
    # M=1) and gets max_effective_stride=120, including the read side the
    # hand-picked split never accounted for.
    #
    # simd_lanes=4: a fused multi-stage kernel spills at simd_lanes=8 for
    # the same reason as test case A -- see its comment above.
    chunks = factor_into_kernel_chunks(960, scratchpad_byte_budget=256)
    multi = make_multi_kernel_plan(chunks, simd_lanes=4)
    multi_source = generate_multi_kernel_fft_kernels(multi)
    multi_output = here / "fft_fp32_multikernel_generated.mojo"
    multi_output.write_text(multi_source, encoding="utf-8")
    print(f"generated: {multi_output}")


if __name__ == "__main__":
    main()
