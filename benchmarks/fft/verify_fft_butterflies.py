from pathlib import Path

from fft_codegen import generate_decomposed_fft_kernels, generate_fft_kernel
from fft_plangen import make_444_plan, make_decomposed_plan


def main() -> None:
    here = Path(__file__).resolve().parent

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
