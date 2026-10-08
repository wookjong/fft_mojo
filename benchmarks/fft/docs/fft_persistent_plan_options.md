# Persistent plan options drive default execution

Execution implementation step 4 (2026-10-07).

Previously a plan with `lowering_mode="wave"` and
`copy_mode="vectorized_contiguous"` was rendered as physical/scalar unless the
caller repeated both options in codegen arguments. Recursive rendering could
only override the execution mode globally and offered no copy selection.

Standalone, struct-only, hybrid and recursive persistent rendering now follow
each leaf's `PersistentWorkgroupPlan.lowering_mode` and `copy_mode` by default.
`None` means use the planned value. Explicit options override independently and
do not mutate the plan. One tree may therefore render different persistent
leaves with different modes and copy strategies. Default physical/scalar plans
retain the same generated source.

The common `resolve_persistent_options` helper resolves and checks both values
before emitting a persistent struct. Low-level stage/copy phase emitters still
accept concrete options for isolated experiments.

```python
# Follow this standalone leaf's mode and copy strategy.
generate_persistent_fft_kernel(leaf, num_logical_blocks=33)

# Override mode while retaining the leaf's planned copy strategy.
generate_persistent_fft_kernel(leaf, num_logical_blocks=33, mode="physical")

# Follow each recursive leaf independently.
generate_recursive_fft_kernels(tree, compute_lanes=4)

# Override all persistent leaves for an experiment.
generate_recursive_fft_kernels(
    tree, compute_lanes=4, persistent_mode="physical",
    persistent_copy_mode="vectorized_contiguous", persistent_vector_width=2,
)
```

`probe_spill_free` follows the same defaults and exposes the same recursive
options. Numerical verification helpers resolve the same settings, so they
exercise the default rendering rather than silently testing physical/scalar.

Cost/report fields continue to describe the planned configuration. An explicit
A/B codegen override does not rewrite those fields or trigger a rerank; use a
plan with the corresponding settings when comparing planner cost/report results.
This step connects existing execution choices and does not add search axes.

Verification: `python -m verification.verify_fft_persistent_plan_options`
contains eight tests covering standalone inheritance, independent overrides,
per-leaf mixed settings, global recursive overrides, all three modes and both
copy strategies across forward/inverse transforms with 33 blocks, hybrid
inheritance, probe source consistency, invalid option values and default source
compatibility. Numerical checks interpret emitted bodies and compare against
NumPy. Actual Mojo compilation and simulator execution were unavailable.
