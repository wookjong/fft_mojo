from pathlib import Path

from fft_codegen import generate_fft_kernel
from fft_plangen import make_444_plan


def main() -> None:
    plan = make_444_plan()
    source = generate_fft_kernel(plan)
    output = Path(__file__).resolve().parent / "fft_fp32_generated.mojo"
    output.write_text(source, encoding="utf-8")
    print(f"generated: {output}")


if __name__ == "__main__":
    main()
