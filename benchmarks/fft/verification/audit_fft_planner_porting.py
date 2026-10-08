"""Compare the current port with compiled, pinned VkFFT host code.

Run from benchmarks/fft: python -m verification.audit_fft_planner_porting
Requires the sibling VkFFT checkout and a host C++ compiler (no GPU).
The oracle compiles original register, radix-optimization, and upload-zero
AxisBlock blocks. It covers direct single-upload FFTs <=4096 with register
boost one, NVIDIA defaults, and the portable backend (VKFFT_BACKEND=0).
Quad-flag rows identify the alternate register table with an otherwise
fixed FP32 layout; they are not complete quad-precision library plans.
This is a review artifact: discrepancies are recorded, not auto-corrected.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from planning.core.target_profile import DEFAULT_TARGET_PROFILE as TARGET
from planning.gpu_baseline import vkfft as vk


def function(source, name):
    start = source.index("static inline VkFFTResult " + name + "(")
    body = source.index("{", start)
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def build_oracle(scheduler, splitter):
    functions = "\n".join(function(scheduler, name) for name in (
        "VkFFTGetRegistersPerThreadQuad", "VkFFTGetRegistersPerThread",
        "VkFFTOptimizeRadixKernels", "VkFFTMinMaxRegisterCheck"))
    start = splitter.index("\t\t\tpfUINT currentAxisBlock1 = axis->axisBlock[1];")
    end = splitter.index("\t\t\taxis->axisBlock[2] = 1;", start)
    postprocess = splitter[start:end]
    start = scheduler.index("\t\telse {\n\t\t\t//try to read directly to registers")
    end = scheduler.index("\n\t}\n\treturn VKFFT_SUCCESS;", start)
    swap = scheduler[start:end].replace("axes[k].specializationConstants.", "sc.")
    return r'''
#include <cmath>
#include <cstdint>
#include <iostream>
using pfUINT = uint64_t;
using pfINT = int64_t;
using VkFFTResult = int;
#define VKFFT_SUCCESS 0
#define VKFFT_BACKEND 0
#define pfceil std::ceil
struct Configuration {
    bool quadDoubleDoublePrecision = false, quadDoubleDoublePrecisionDoubleMemory = false;
    pfUINT numSharedBanks = 32, vendorID = 0x10DE, aimThreads = 128;
    pfUINT maxComputeWorkGroupSize[3] = {1024, 1024, 1024};
    bool performZeropadding[3] = {false, false, false};
};
struct VkFFTApplication { Configuration configuration; };
struct VkFFTRaderContainer {
    int registers_per_thread_per_radix[33] = {}, loc_multipliers[33] = {};
    int numSubPrimes = 0, type = 0, prime = 0, numStages = 0;
    int stageRadix[40] = {}, stage_rader_generator[40] = {};
    VkFFTRaderContainer* container = nullptr;
};
struct Integer { struct { pfUINT i = 0; } data; };
struct Spec {
    Integer fftDim;
    pfUINT complexSize = 8, registerBoost = 1;
    int mergeSequencesR2C = 0, useBluesteinFFT = 0, axisSwapped = 0;
    int numStages = 0, stageRadix[40] = {}, rader_generator[40] = {}, useRader = 0;
    int registers_per_thread_per_radix[33] = {};
    bool reorderFourStep = true;
};
struct Axis { Spec specializationConstants; pfUINT axisBlock[4] = {}, groupedBatch = 0; };
struct Plan { int numAxisUploads[1] = {1}; pfUINT actualFFTSizePerAxis[1][2] = {}; };
''' + functions + r'''
int main() {
    int n, batch, quad;
    while (std::cin >> n >> batch >> quad) {
        VkFFTApplication application; auto* app = &application;
        application.configuration.quadDoubleDoublePrecision = quad;
        int multipliers[33] = {}, regs[33] = {}, maxregs = 0, minregs = 10000000, good = 0;
        int remaining = n;
        for (int i = 2; i < 17; ++i) while (remaining % i == 0) { remaining /= i; ++multipliers[i]; }
        if (remaining != 1 || n > 4096) return 2;
        VkFFTGetRegistersPerThread(app, n, 1, batch, 0, multipliers, regs, &maxregs, &minregs, &good);
        // For this restricted single-upload domain the scheduler's thread-cap
        // scaling and all Rader branches are inactive. Check that precondition.
        if (std::ceil(double(n)/minregs) > 1024) return 3;
        int initial_min = minregs, maxradix = 1, usedregs = 1;
        VkFFTOptimizeRadixKernels(regs, multipliers, 1, &maxradix, &usedregs, nullptr, 0);
        Spec sc;
        for (int i=2; i<33; ++i) sc.registers_per_thread_per_radix[i] = regs[i];
        for (int i=32; i>1; --i) while (multipliers[i]>0) { sc.stageRadix[sc.numStages++] = i; --multipliers[i]; }
        minregs = 10000000; maxregs = 0;
        VkFFTMinMaxRegisterCheck(sc.numStages, sc.stageRadix, &minregs, &maxregs,
                                sc.registers_per_thread_per_radix, nullptr, 0, sc.rader_generator);
        int min_registers_per_thread = minregs, registers_per_thread = maxregs;
        int switchRegisterBoost = 0;
        if (switchRegisterBoost > 0) {}
''' + swap + r'''
        Axis axisValue; auto* axis = &axisValue;
        Plan planValue; auto* FFTPlan = &planValue;
        axis->specializationConstants.fftDim.data.i = n;
        FFTPlan->actualFFTSizePerAxis[0][0] = n;
        FFTPlan->actualFFTSizePerAxis[0][1] = batch;
        pfUINT axis_id = 0, r2cmult = 1, maxThreadNum = 1024;
        pfUINT allowedSharedMemory = 122880, maxSequenceLengthSharedMemory = 15360;
        pfUINT maxSequenceLengthSharedMemoryPow2 = 8192, maxBatchCoalesced = 4;
        axis->axisBlock[0] = std::ceil(double(n)/minregs);
        pfUINT estimate_batch = ((axis->axisBlock[0]/32 == 1) && (axis->axisBlock[0]/32.0 < 1.5))
            ? 128/32 : 128/axis->axisBlock[0];
        if (!estimate_batch) estimate_batch = 1;
        axis->axisBlock[1] = (axis->axisBlock[0]<128 && axis->axisBlock[0]<32) ? estimate_batch : 1;
''' + postprocess + r'''
        pfUINT tpt = axis->axisBlock[axis->specializationConstants.axisSwapped ? 1 : 0];
        pfUINT tpb = axis->axisBlock[axis->specializationConstants.axisSwapped ? 0 : 1];
        std::cout << n << " " << batch << " " << quad << " " << initial_min << " " << minregs;
        std::cout << " " << tpt << " " << tpb << " " << axis->specializationConstants.axisSwapped;
        std::cout << " " << sc.numStages;
        for (int i=0; i<sc.numStages; ++i) std::cout << " " << sc.stageRadix[i];
        std::cout << "\n";
    }
}
'''


def audit(compiler, output):
    if TARGET.spad_capacity_bytes != 122880:
        raise ValueError("This fixed-budget oracle expects 120 KiB shared memory")
    root = Path(__file__).resolve().parents[4]
    source_dir = root / "VkFFT/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions"
    scheduler_path = source_dir / "vkFFT_Scheduler.h"
    # Windows Git checkouts may have CRLF; the pinned fixture hashes LF blobs.
    raw = scheduler_path.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(raw).hexdigest()
    from verification.generate_vkfft_multipass_fixture import SOURCE_SHA256, UPSTREAM_SHA
    if digest != SOURCE_SHA256:
        raise ValueError("Local VkFFT scheduler is not the pinned source")
    splitter_raw = (source_dir / "vkFFT_AxisBlockSplitter.h").read_bytes().replace(b"\r\n", b"\n")
    lengths = (12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 64, 80, 96, 120, 128,
               192, 210, 216, 240, 256, 320, 384, 480, 512, 630, 768, 960,
               1024, 1536, 2048, 3072, 4096)
    cases = [(n, batch, quad) for n in lengths
             for batch in (1, 16, 32) for quad in (0, 1)]
    with tempfile.TemporaryDirectory(prefix="fft-port-review-") as temporary:
        cpp = Path(temporary) / "oracle.cpp"
        exe = Path(temporary) / "oracle.exe"
        cpp.write_text(build_oracle(raw.decode(), splitter_raw.decode()), encoding="utf-8")
        subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(exe)], check=True)
        result = subprocess.run([str(exe)], input="".join(f"{n} {b} {q}\n" for n, b, q in cases),
                                text=True, capture_output=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        n, batch, quad, initial, final, tpt, tpb, swapped, stages, *radices = map(int, line.split())
        assert stages == len(radices)
        port_radices = vk.leaf_radix_sequence(n, batch)
        port_min = vk.min_registers_per_thread_for(n, port_radices, batch)
        port_workers, port_slots = vk.axisblock_for_leaf(n, port_radices, max_rhs=batch,
            num_passes=1, upload_id=0, original_length=n, target=TARGET)
        plan = vk.plan(n, batch=batch)
        rows.append(dict(length=n, batch=batch, precision="quad" if quad else "fp32",
            upstream_initial_min_registers=initial, upstream_final_min_registers=final,
            upstream_radices=radices, upstream_workers=tpt, upstream_slots=tpb,
            upstream_axis_swapped=bool(swapped), port_min_registers=port_min,
            port_radices=port_radices, port_workers=port_workers, port_slots=port_slots,
            port_status=plan.status.name))
    baseline_lengths = set(lengths) - {15, 30, 60, 210, 630}
    sweep = [row for row in rows if row["precision"] == "fp32" and row["batch"] == 1
             and row["length"] in baseline_lengths]
    summary = {name: [row["length"] for row in sweep if row[upstream] != row[port]]
               for name, upstream, port in (
                   ("different_radices", "upstream_radices", "port_radices"),
                   ("different_registers", "upstream_final_min_registers", "port_min_registers"),
                   ("different_workers", "upstream_workers", "port_workers"))}
    data = dict(upstream_commit=UPSTREAM_SHA, scheduler_sha256=digest,
                axisblock_sha256=hashlib.sha256(splitter_raw).hexdigest(),
                port_sha256=hashlib.sha256(Path(vk.__file__).read_bytes()).hexdigest(),
                shared_bytes=TARGET.spad_capacity_bytes,
                complex_size_bytes=8,
                scope=__doc__, default_sweep_summary=summary, cases=rows)
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        if row["precision"] == "fp32" and row["batch"] == 1:
            print(f"N={row['length']}: registers {row['upstream_final_min_registers']} vs "
                  f"{row['port_min_registers']}; radices {row['upstream_radices']} vs "
                  f"{row['port_radices']}; workers {row['upstream_workers']} vs {row['port_workers']}")
    print(f"Recorded {len(rows)} cases: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "fixtures" /
                        "fft_planner_porting_audit_2026-10-07.json")
    args = parser.parse_args()
    audit(args.compiler, args.out)
