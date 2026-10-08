"""Read-only planner/codegen audit; does not compile or execute Mojo binaries.

Run from benchmarks/fft with python -m verification.audit_fft_execution_support.
Observed gaps are data, not failing assertions: this is a review reproducer.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import re

from codegen.fft_persistent_codegen import (
    generate_persistent_fft_kernel, generate_persistent_tail_hybrid_kernel,
)
from codegen.fft_transpose_codegen import generate_recursive_fft_kernels
from planning.core.target_profile import DEFAULT_TARGET_PROFILE as TARGET
from planning.execution.fft_plan_cooperative import make_cooperative_leaf_plan
from planning.execution.fft_plan_persistent import (
    make_persistent_leaf_plan, make_persistent_tail_hybrid_plan,
)
from planning.search.fft_plan_search import generate_candidates
from planning.strategies.fft_plan_recursive import (
    FFTLeafPlan, RecursiveFFTPlan, flatten_recursive_node, make_recursive_transpose_plan,
)


def render(plan, target=TARGET):
    return generate_recursive_fft_kernels(
        plan, target=target, compute_lanes=4, narrow_middle_stages=True,
        reference_check=False, spread_across_units=True,
    )


def registered_functions(source):
    # Count distinct launch targets, not rounds, method definitions or FFT nodes.
    chunks = re.split(r"^struct (\w+)\(NDPTask\):", source, flags=re.M)
    return {chunks[i]: sorted(set(re.findall(
        r"launch_parallel\[" + re.escape(chunks[i]) + r"\.(\w+)\]", chunks[i + 1]
    ))) for i in range(1, len(chunks), 2)}


def audit():
    over_cap = []
    candidates = generate_candidates(729, target=TARGET, scratchpad_byte_budget=16384,
                                     max_candidates=24)
    for index, candidate in enumerate(candidates):
        for task, functions in registered_functions(render(candidate.plan)).items():
            if len(functions) > TARGET.max_kernel_register:
                over_cap.append(dict(candidate=index, task=task, functions=functions,
                                     workers=candidate.choices.workers_per_fft,
                                     lane_variant=candidate.choices.lane_variant))

    target = replace(TARGET, spad_capacity_bytes=2048, max_kernel_register=4, num_ndp_units=4)
    wrong_target = []
    for candidate in generate_candidates(64, target=target, max_candidates=80):
        for node in flatten_recursive_node(candidate.plan.root):
            if getattr(node, "persistent", None) is not None:
                wrong_target.append(dict(worker_sequence=candidate.choices.worker_sequence,
                                         expected_units=target.num_ndp_units,
                                         actual_groups=node.persistent.software_group_count,
                                         expected_kernel_cap=target.max_kernel_register,
                                         actual_functions=2 + len(node.stages)))
    over_spad = make_recursive_transpose_plan(
        64, scratchpad_byte_budget=4096, spad_capacity_bytes=512, persistent_leaf=True)

    persistent = make_persistent_leaf_plan(
        64, (4, 4, 4), num_logical_blocks=33, workers_per_fft=16,
        lowering_mode="wave", copy_mode="vectorized_contiguous", kernel_name="AuditPersistent")
    default_source = generate_persistent_fft_kernel(persistent, num_logical_blocks=33)
    explicit_source = generate_persistent_fft_kernel(
        persistent, num_logical_blocks=33, mode="wave", copy_mode="vectorized_contiguous")
    scalar_metadata = replace(persistent, persistent=replace(
        persistent.persistent, copy_mode="scalar", lowering_mode="physical"))
    host_plan = make_recursive_transpose_plan(64, scratchpad_byte_budget=4096).host
    wrapped = RecursiveFFTPlan(n=64, inverse=False, root=FFTLeafPlan(m=64, r=33, kernel=persistent),
                               host=host_plan, batch=33)
    wrapped_scalar = replace(wrapped, root=replace(wrapped.root, kernel=scalar_metadata))

    accepted_workers = []
    for workers in (3, 16):
        leaf = make_cooperative_leaf_plan(64, (4, 4, 4), workers_per_fft=workers, total_ffts=3)
        hybrid = make_persistent_tail_hybrid_plan(
            64, (4, 4, 4), num_logical_blocks=35, tail_strategy="cooperative_tail",
            tail_cooperative_workers=workers)
        source = generate_persistent_tail_hybrid_kernel(hybrid)
        accepted_workers.append(dict(workers=workers, interleave_chunk=8,
                                     leaf_uthreads=leaf.total_uthreads,
                                     tail_blocks=hybrid.tail_blocks, codegen_succeeded=bool(source)))

    simd4 = make_recursive_transpose_plan(64, scratchpad_byte_budget=4096, simd_lanes=4)
    try:
        render(simd4)
    except ValueError as exc:
        simd4_error = str(exc)
    else:
        simd4_error = None

    matrix = []
    for n in (60, 128, 216, 630, 960, 1024):
        for batch in (1, 33):
            rows = generate_candidates(n, target=TARGET, batch=batch, max_candidates=12)
            strategies, errors = set(), []
            for index, candidate in enumerate(rows):
                for node in flatten_recursive_node(candidate.plan.root):
                    if hasattr(node, "stages"):
                        strategies.add("persistent" if node.persistent else
                                       "cooperative" if node.cooperation else "non_cooperative")
                try:
                    functions = registered_functions(render(candidate.plan))
                    for task, names in functions.items():
                        if len(names) > TARGET.max_kernel_register:
                            errors.append(dict(candidate=index, task=task, kernel_count=len(names)))
                except (ValueError, AssertionError) as exc:
                    errors.append(dict(candidate=index, error=str(exc)))
            matrix.append(dict(n=n, batch=batch, candidates=len(rows),
                               strategies=sorted(strategies), errors=errors))

    return dict(
        validation="Python planner and generated-source inspection; no binary execution",
        registry=dict(n=729, scratchpad_byte_budget=16384, max_kernel_register=8,
                      candidates=len(candidates), violations=over_cap),
        target_propagation=dict(mixed_candidates=wrong_target,
                                requested_scratchpad_bytes=512,
                                accepted_scratchpad_bytes=sum(
                                    b.elements for b in over_spad.root.kernel.scratchpad_buffers) * 4),
        persistent_options=dict(plan_mode="wave", plan_copy="vectorized_contiguous",
                                default_wave_tracker="wave_tracker" in default_source,
                                explicit_wave_tracker="wave_tracker" in explicit_source,
                                default_stage_calls=default_source.count("launch_parallel[AuditPersistent.stage_"),
                                explicit_stage_calls=explicit_source.count("launch_parallel[AuditPersistent.stage_"),
                                recursive_ignores_plan_options=render(wrapped) == render(wrapped_scalar)),
        unsafe_cooperative_workers=accepted_workers,
        cooperative_lane_counterexample=dict(global_id=8, local_id=0, workers=3,
                                             emitted_worker=0 % 3, required_worker=8 % 3),
        simd4=dict(planner_accepted=True, codegen_error=simd4_error),
        default_matrix=matrix,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
        print(f"Wrote audit to {args.output}; "
              f"{len(result['registry']['violations'])} registry violations; "
              f"{sum(r['candidates'] for r in result['default_matrix'])} default candidates inspected")
    else:
        print(encoded, end="")
