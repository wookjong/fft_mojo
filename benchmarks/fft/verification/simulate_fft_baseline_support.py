"""Compile and simulate the newly supported baseline execution paths.

Uses the existing local image and the host-only compatibility bridge from
simulate_fft_execution_fixes. No simulator configuration is modified.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verification import simulate_fft_execution_fixes as sim

sim.RESULTS = ROOT / "verification/results/baseline_support_2026-10-08"


def generate():
    from codegen.fft_persistent_codegen import generate_persistent_fft_kernel
    from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
    from planning.execution.fft_plan_persistent import make_persistent_leaf_plan
    from planning.gpu_baseline import rocfft_default, vkfft
    sim.RESULTS.mkdir(parents=True,exist_ok=True)
    cases = []
    def add(name, source):
        (sim.RESULTS/f"{name}.mojo").write_text(sim.domain_bridge(source),encoding="utf-8")
        cases.append(dict(name=name,config="M2NDP",execution=[json.loads(s.split(" ",2)[2])
            for s in source.splitlines() if s.startswith("# FFT_EXECUTION ")]))
    for n in (12,14,15):
        result = vkfft.plan(n,batch=3,inverse=n == 14)
        add(f"vkfft_r{n}_b3",generate_recursive_fft_kernels(result.plan))
    plan = make_persistent_leaf_plan(64,(4,4,4),num_logical_blocks=65,workers_per_fft=17,
                                     storage_mode="global")
    add("global_n64_w17_b65",generate_persistent_fft_kernel(plan,num_logical_blocks=65))
    result = rocfft_default.plan_m2ndp(8192,batch=33)
    original = result.plan.root.kernel
    smoke = make_persistent_leaf_plan(8192,tuple(s.radix for s in original.stages),num_logical_blocks=2,
        workers_per_fft=original.persistent.workers_per_fft,kernel_name=original.kernel_name,
        storage_mode="global",loop_stages=True)
    source = generate_persistent_fft_kernel(smoke,num_logical_blocks=2,
                                             reference_check=False)
    source = source.replace("input_real[i] = Float32(random_float64(-1.0, 1.0))",
        "input_real[i] = Float32(0)\n        if i % 8192 == 0:\n"
        "            input_real[i] = Float32(i // 8192 % 7 + 1)")
    source = source.replace("input_imag[i] = Float32(random_float64(-1.0, 1.0))",
        "input_imag[i] = Float32(0)\n        if i % 8192 == 2048:\n"
        "            input_imag[i] = Float32(0.25)")
    # Independently known spectrum: a real impulse at zero plus an imaginary
    # quarter-period impulse. Vary the real impulse across replicas to detect
    # stale workspace/round addressing without an O(batch*N^2) host DFT.
    source = source.split('    print("INPUT_BEGIN")')[0] + '''
    for i in range(total_elems):
        var want_real = Float32(i // 8192 % 7 + 1)
        var want_imag = Float32(0)
        if i % 4 == 0:
            want_imag = Float32(0.25)
        elif i % 4 == 1:
            want_real += Float32(0.25)
        elif i % 4 == 2:
            want_imag = Float32(-0.25)
        else:
            want_real -= Float32(0.25)
        if abs(output_real[i] - want_real) > Float32(0.002) or abs(output_imag[i] - want_imag) > Float32(0.002):
            print("[host] FFT mismatch at", i)
            return
    print("[host] large FFT verification passed")
'''
    add("rocfft_global_n8192_b2",source)
    cases[-1]['run_timeout'] = 900
    cases[-1]['scope'] = 'Original rocFFT batch=33 radix/worker choice; two replicas for functional simulation only'
    (sim.RESULTS/"cases.json").write_text(json.dumps(cases,indent=2)+"\n",encoding="utf-8")
    print(f"Generated {len(cases)} cases in {sim.RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate",action="store_true")
    parser.add_argument("--run",nargs="*")
    parser.add_argument("--summarize",action="store_true")
    args = parser.parse_args()
    if args.generate:
        generate()
    if args.run is not None:
        sys.exit(sim.run(args.run))
    if args.summarize:
        sim.summarize()
