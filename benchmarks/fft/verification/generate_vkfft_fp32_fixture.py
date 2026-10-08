"""Compile pinned original host decisions; no GPU or Python planner oracle.

Run from benchmarks/fft with --compiler pointing to a C++17 compiler.
The direct domain is FP32, portable backend, boost one, primes 2/3/5/7.
Original register scaling and radix optimization are compiled verbatim.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

from verification.audit_fft_planner_porting import build_oracle
from verification.generate_vkfft_multipass_fixture import SOURCE_SHA256, UPSTREAM_SHA


def direct_oracle(scheduler, splitter):
    header = build_oracle(scheduler, splitter).split("int main() {")[0]
    header = header.replace("pfUINT numSharedBanks = 32", "pfUINT maxThreadsNum = 1024;\n    pfUINT numSharedBanks = 32")
    header = header.replace("Integer fftDim;", "Integer fftDim;\n    int numRaderPrimes = 0; VkFFTRaderContainer* raderContainer = nullptr;")
    start = scheduler.index("\t\tif ((int)pfceil((maxBatchCoalesced * locAxisSplit[k]")
    end = scheduler.index("\n\t\t//second optimizer pass", start)
    scaling = scheduler[start:end].replace("axes[k].specializationConstants.", "sc.")
    start = scheduler.index("\t\telse {\n\t\t\t//try to read directly to registers")
    end = scheduler.index("\n\t}\n\treturn VKFFT_SUCCESS;", start)
    swap = scheduler[start:end].replace("axes[k].specializationConstants.", "sc.")
    return header + r'''
int main() {
    char mode; int n, batch, passes, k;
    while (std::cin >> mode >> n >> batch >> passes >> k) {
        VkFFTApplication application; auto* app = &application;
        int loc_multipliers[33] = {}, registers_per_thread_per_radix[33] = {};
        int registers_per_thread = 0, min_registers_per_thread = 10000000, good = 0;
        int remaining = n;
        for (int p: {2,3,5,7}) while (remaining % p == 0) { remaining /= p; ++loc_multipliers[p]; }
        if (remaining != 1) return 2;
        VkFFTGetRegistersPerThread(app, n, 1, batch, 0, loc_multipliers,
            registers_per_thread_per_radix, &registers_per_thread, &min_registers_per_thread, &good);
        if (mode == 'R') {
            for (int p: {2,3,5,7}) std::cout << registers_per_thread_per_radix[p] << " ";
            std::cout << "\n"; continue;
        }
        Spec sc;
        pfUINT axis_id = 0, maxBatchCoalesced = passes == 1 ? 1 : 4;
        pfUINT locAxisSplit[3] = {pfUINT(n), pfUINT(n), pfUINT(n)};
        pfUINT registerBoost = 1, estimate_rader_threadnum = 0, temp = 0;
''' + scaling + r'''
        int maxradix = 1, usedregs = 1;
        VkFFTOptimizeRadixKernels(registers_per_thread_per_radix, loc_multipliers,
                                 1, &maxradix, &usedregs, nullptr, 0);
        for (int i=2; i<33; ++i) sc.registers_per_thread_per_radix[i] = registers_per_thread_per_radix[i];
        for (int i=32; i>1; --i) while (loc_multipliers[i]>0) {
            sc.stageRadix[sc.numStages++] = i; --loc_multipliers[i];
        }
        min_registers_per_thread = 10000000; registers_per_thread = 0;
        VkFFTMinMaxRegisterCheck(sc.numStages, sc.stageRadix, &min_registers_per_thread,
            &registers_per_thread, sc.registers_per_thread_per_radix, nullptr, 0, sc.rader_generator);
        int switchRegisterBoost = 0;
        if (switchRegisterBoost > 0) {}
''' + swap + r'''
        std::cout << min_registers_per_thread << " " << registers_per_thread << " " << sc.numStages;
        for (int i=0; i<sc.numStages; ++i) std::cout << " " << sc.stageRadix[i];
        for (int i=2; i<33; ++i) std::cout << " " << sc.registers_per_thread_per_radix[i];
        std::cout << "\n";
    }
}
'''


def compile_run(compiler, work, name, source, stdin):
    cpp, exe = work / (name + ".cpp"), work / (name + ".exe")
    cpp.write_text(source, encoding="utf-8")
    subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(exe)], check=True)
    return subprocess.run([str(exe)], input=stdin, text=True,
                          capture_output=True, check=True).stdout.splitlines()


def generate(compiler, output, work):
    root = Path(__file__).resolve().parents[4]
    upstream = root / "VkFFT/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions"
    scheduler_raw = (upstream / "vkFFT_Scheduler.h").read_bytes().replace(b"\r\n", b"\n")
    splitter_raw = (upstream / "vkFFT_AxisBlockSplitter.h").read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(scheduler_raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Local VkFFT scheduler does not match pinned revision")
    scheduler, splitter = scheduler_raw.decode(), splitter_raw.decode()
    work.mkdir(parents=True, exist_ok=True)
    # Equality of c2/c3 at 2..6 exercises the special six-register branch.
    counts = [c for c in itertools.product(range(7), range(7), range(4), range(4))
              if any(c[1:])]
    base_inputs = [(2**a * 3**b * 5**c * 7**d) for a, b, c, d in counts]
    lengths = (2,3,4,5,6,7,8,9,12,15,16,18,20,24,25,27,28,30,32,35,40,42,48,
               49,60,64,80,96,120,128,192,210,216,240,256,320,384,480,512,630,
               768,960,1024,1296,1536,2048,2187,3072,4096,6561,7776,8192,12288,
               15000,16384,16807,19683,32768,46656,59049,62500,65536,98304,
               103680,131072)
    contexts = [(batch, passes, upload) for batch in (1,32,256)
                for passes, upload in ((1,0),(2,0),(2,1))]
    direct_cases = [(n, b, p, u) for n in lengths for b, p, u in contexts]
    stdin = "".join(f"R {n} 1 1 0\n" for n in base_inputs)
    stdin += "".join(f"D {n} {b} {p} {u}\n" for n,b,p,u in direct_cases)
    lines = compile_run(compiler, work, "fp32_direct", direct_oracle(scheduler, splitter), stdin)
    if len(lines) != len(counts) + len(direct_cases):
        raise ValueError("Oracle row count differs from inputs")
    base_rows = [dict(counts=c, registers=list(map(int, line.split())))
                 for c, line in zip(counts, lines[:len(counts)])]
    direct_rows = []
    for (n,b,p,u), line in zip(direct_cases, lines[len(counts):]):
        minimum, maximum, stages, *values = map(int, line.split())
        radices, registers = values[:stages], values[stages:]
        assert len(registers) == 31
        direct_rows.append(dict(length=n, batch=b, num_passes=p, upload_id=u,
            min_registers=minimum, max_registers=maximum, radices=radices,
            registers_per_radix=[[i,v] for i,v in enumerate(registers,2) if v]))
    axis_lengths = (12,15,16,20,24,30,32,40,48,60,64,80,96,120,128,192,210,216,
                    240,256,320,384,480,512,630,768,960,1024,1536,2048,3072,4096)
    axis_inputs = [(n,b) for n in axis_lengths for b in (1,4,16,32)]
    lines = compile_run(compiler, work, "fp32_axis", build_oracle(scheduler, splitter),
                        "".join(f"{n} {b} 0\n" for n,b in axis_inputs))
    axis_rows = []
    for line in lines:
        n,b,quad,initial,final,tpt,tpb,swapped,stages,*radices = map(int,line.split())
        assert quad == 0 and stages == len(radices)
        axis_rows.append(dict(length=n,batch=b,radices=radices,min_registers=final,
            workers=tpt,slots=tpb,axis_swapped=bool(swapped)))
    output.parent.mkdir(parents=True, exist_ok=True)
    data = dict(upstream_commit=UPSTREAM_SHA, scheduler_sha256=SOURCE_SHA256,
        axisblock_sha256=hashlib.sha256(splitter_raw).hexdigest(),
        scope=__doc__, base_register_cases=base_rows, direct_cases=direct_rows, axis_cases=axis_rows)
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Pinned original C++: {len(base_rows)} base, {len(direct_rows)} direct, {len(axis_rows)} axis cases -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "fixtures/vkfft_fp32_upstream.json")
    parser.add_argument("--work", type=Path, default=Path(__file__).parent / "results/planner_fix_2026-10-07/fp32_oracle_work")
    args = parser.parse_args()
    generate(args.compiler, args.out, args.work)
