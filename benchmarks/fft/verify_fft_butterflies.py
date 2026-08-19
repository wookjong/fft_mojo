import re
from pathlib import Path

import numpy as np

from fft_butterflies import SUPPORTED_RADICES, emit_butterfly
from fft_codegen import generate_decomposed_fft_kernels, generate_fft_kernel
from fft_plangen import make_444_plan, make_decomposed_plan

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
    here = Path(__file__).resolve().parent

    print("Verifying FFT butterflies against their DFT matrices...")
    verify_butterflies()

    # Test case A: single-kernel N=64, radix 4x4x4 -- whole FFT in one
    # uthread's own scratchpad, no decomposition.
    plan = make_444_plan()
    source = generate_fft_kernel(plan)
    output = here / "fft_fp32_generated.mojo"
    output.write_text(source, encoding="utf-8")
    print(f"generated: {output}")

    # Test case B: N=256 = 16*16, too large for one uthread's scratchpad --
    # kernel0 (16 uthreads, one 16-point sub-FFT + large twiddle + permuted
    # store each) then kernel1 (16 uthreads, one 16-point sub-FFT each),
    # chained through DRAM. See fft_plangen.make_decomposed_plan.
    decomposed = make_decomposed_plan(16, 16)
    decomposed_source = generate_decomposed_fft_kernels(decomposed)
    decomposed_output = here / "fft_fp32_decomposed_generated.mojo"
    decomposed_output.write_text(decomposed_source, encoding="utf-8")
    print(f"generated: {decomposed_output}")


if __name__ == "__main__":
    main()
