"""Capture pinned upstream C++ split decisions without a GPU/backend.

python -m verification.generate_vkfft_multipass_fixture --scheduler PATH
The input is VkFFT's original Scheduler.h at UPSTREAM_SHA. Its split and
reordering blocks are compiled verbatim inside a minimal C++ harness;
no Python baseline functions are imported. Requires a host C++ compiler.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

UPSTREAM_SHA = "066a17c17068c0f11c9298d848c2976c71fad1c1"
SOURCE_SHA256 = "9b37b786a70cd416be468e637d21bf463ad307e98b6217b2c0149878460e5809"
SOURCE_PATH = "vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h"


def generate(scheduler: Path, output: Path, compiler: str):
    raw = scheduler.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Scheduler.h does not match the pinned upstream source bytes")
    source = raw.decode()
    start = source.index("\tpfUINT* locAxisSplit = FFTPlan->axisSplit[axis_id];")
    end = source.index("\tif ((numPasses > 1) &&", start)
    split = source[start:end]
    start = source.index("\tif (((app->configuration.reorderFourStep) && (!app->useBluesteinFFT[axis_id])))", end)
    end = source.index("\tFFTPlan->numAxisUploads[axis_id] = numPasses;", start)
    reorder = source[start:end]
    harness = r'''
#include <cmath>
#include <cstdint>
#include <iostream>
using pfUINT = uint64_t;
#define pfceil std::ceil
#define VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH 1
struct Configuration { int vendorID; bool reorderFourStep = true; };
struct Application { Configuration configuration; bool useBluesteinFFT[1] = {false}; };
struct Plan { pfUINT actualFFTSizePerAxis[1][1]; pfUINT axisSplit[1][3] = {}; };
int schedule(pfUINT length, int shared, int complexSize, bool nvidia, Plan* FFTPlan, int& numPasses) {
    Application application; Application* app = &application;
    app->configuration.vendorID = nvidia ? 0x10DE : 0x1002;
    FFTPlan->actualFFTSizePerAxis[0][0] = length;
    int axis_id = 0, nonStridedAxisId = 0, registerBoost = 1;
    int isPowOf2 = (length & (length - 1)) == 0;
    int usedSharedMemory = shared;
    if (isPowOf2) usedSharedMemory = int(pow(2, int(log2(shared))));
    int maxSequenceLengthSharedMemory = usedSharedMemory / complexSize;
    int maxSingleSizeStrided = usedSharedMemory / ((32 > complexSize) ? 32 : complexSize);
    int maxSingleSizeStridedHalfBandwidth = maxSingleSizeStrided;
    numPasses = (length <= pfUINT(maxSequenceLengthSharedMemory)) ? 1
        : int(pfceil(log2(length) / log2(maxSingleSizeStrided)));
''' + split + reorder + r'''
    return 0;
}
int main() {
    uint64_t n; int shared, bytes, nvidia;
    while (std::cin >> n >> shared >> bytes >> nvidia) {
        Plan plan; int passes;
        int error = schedule(n, shared, bytes, nvidia, &plan, passes);
        std::cout << error << " " << passes;
        if (!error) for (int i=0; i<passes; ++i) std::cout << " " << plan.axisSplit[0][i];
        std::cout << "\n";
    }
}
'''
    cases = []
    for shared in (4096, 65536, 122880):
        for complex_size in (8, 16):
            for nvidia in (True, False):
                for n in (8192, 65536, 262144, 524288, 2**24, 2**34,
                          10000, 21000, 30000, 127**2):
                    cases.append(dict(length=n, shared_bytes=shared,
                                      complex_size_bytes=complex_size, nvidia=nvidia))
    # Non-power-of-two byte budget and a failed two-upload divisor search.
    cases += [dict(length=51005, shared_bytes=8192, complex_size_bytes=16, nvidia=True),
              dict(length=2**24, shared_bytes=98304, complex_size_bytes=16, nvidia=False)]
    stdin = "".join(f"{c['length']} {c['shared_bytes']} {c['complex_size_bytes']} {int(c['nvidia'])}\n"
                    for c in cases)
    with tempfile.TemporaryDirectory(prefix="vkfft-source-fixture-") as temp:
        cpp, exe = Path(temp) / "schedule.cpp", Path(temp) / "schedule.exe"
        cpp.write_text(harness, encoding="utf-8")
        subprocess.run([compiler, "-std=c++17", str(cpp), "-O2", "-o", str(exe)], check=True)
        result = subprocess.run([str(exe)], input=stdin, capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    assert len(lines) == len(cases)
    for case, line in zip(cases, lines):
        values = [int(v) for v in line.split()]
        case["factors"] = None if values[0] else values[2:]
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(upstream_commit=UPSTREAM_SHA, source_path=SOURCE_PATH,
                    source_sha256=SOURCE_SHA256,
                    capture="Original C++ split/reordering blocks; host scheduler only, no GPU execution")
    lines = ["{"] + [f"  {json.dumps(k)}: {json.dumps(v)}," for k, v in metadata.items()]
    lines += ['  "cases": [', ",\n".join("    " + json.dumps(c) for c in cases), "  ]", "}"]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Captured {len(cases)} upstream C++ cases into {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", type=Path, required=True)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "fixtures" / "vkfft_multipass_upstream.json")
    args = parser.parse_args()
    generate(args.scheduler, args.out, args.compiler)
