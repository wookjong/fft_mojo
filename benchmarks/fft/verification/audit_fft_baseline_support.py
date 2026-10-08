"""Reproducible plan/registration coverage matrix for the C2C FP32 baseline."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from planning.core.target_profile import DEFAULT_TARGET_PROFILE as TARGET
from planning.execution.kernel_registration import validate_kernel_registration
from planning.gpu_baseline import clfft, rocfft_default, vkfft
from planning.gpu_baseline.common import (BaselineStatus,reconstruct_gpu_logical_plan,
    reconstruct_m2ndp_physical_plan,compute_lowering_overhead)
from planning.strategies.fft_plan_recursive import flatten_recursive_node,PhysicalTransposePlan


def main():
    records = []
    lengths = (12,14,15,20,24,28,30,64,128,216,256,512,768,960,1024,2048,4096,8192,16384)
    for module in (clfft,rocfft_default,vkfft):
        for method in ("plan","plan_m2ndp"):
            for n in lengths:
                for batch in (1,32,128):
                    for inverse in (False,True):
                        result = getattr(module,method)(n,batch=batch,inverse=inverse)
                        row = dict(planner=module.__name__.rsplit(".",1)[-1],method=method,
                            n=n,batch=batch,inverse=inverse,status=result.status.value,
                            diagnostics=result.diagnostics,gpu=[asdict(k) for k in reconstruct_gpu_logical_plan(result)])
                        if result.status is BaselineStatus.OK:
                            assert result.plan.batch == batch and result.plan.inverse == inverse
                            for kernel in flatten_recursive_node(result.plan.root):
                                if not isinstance(kernel,PhysicalTransposePlan):
                                    validate_kernel_registration(kernel,max_kernel_register=TARGET.max_kernel_register,
                                        loop_stages=True)
                            row["physical"] = [asdict(k) for k in reconstruct_m2ndp_physical_plan(result,target=TARGET)]
                            row["lowering_overhead"] = asdict(compute_lowering_overhead(result))
                        records.append(row)
    output = ROOT/"verification/results/baseline_support_2026-10-08"
    output.mkdir(parents=True,exist_ok=True)
    counts = {f"{p}/{m}":dict(Counter(r["status"] for r in records if r["planner"]==p and r["method"]==m))
              for p in ("clfft","rocfft_default","vkfft") for m in ("plan","plan_m2ndp")}
    report = dict(scope="C2C FP32 contiguous batches; planner and distinct registration validation, not simulation",
                  lengths=lengths,batches=(1,32,128),target=asdict(TARGET),cases=len(records),counts=counts,records=records)
    (output/"planner_matrix.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(cases=len(records),counts=counts),indent=2))


if __name__ == "__main__":
    main()
