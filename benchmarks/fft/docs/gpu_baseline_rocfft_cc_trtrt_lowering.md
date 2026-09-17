# rocFFT-default CS_L1D_CC / CS_L1D_TRTRT lowering onto M2NDP

## The question

`_plan_from_scheme_decision` previously refused (`UNSUPPORTED_CURRENT_
CODEGEN`) every length where real rocFFT-default selects `CS_L1D_CC`
(fused SBCC+SBRC) or `CS_L1D_TRTRT` (transpose-row-transpose-row-
transpose), on the stated belief that these fused block-tiled kernels
have "no equivalent AddressMapping/codegen mechanism in this repository."
Before writing any new codegen, the real upstream source was fetched
directly at this module's own pinned commit
(`bee97df517907c771de17189cb867d3c401285ae`,
`projects/rocfft/library/src/tree_node_1D.cpp`) and read line-by-line.

## What the real source actually shows

**`TRTRT1DNode::BuildTree_internal`** builds exactly 5 child kernels, in
this order: `trans1Plan` (CS_KERNEL_TRANSPOSE) → `row1Plan` (recursively
built via `RecursiveBuildTree` -- i.e. re-invokes the SAME rocFFT
planning algorithm) → `trans2Plan` (CS_KERNEL_TRANSPOSE, `large1D =
length[0]` -- the twiddle-multiply flag) → `row2Plan` (forced straight to
`CS_KERNEL_STOCKHAM`, never re-decided) → `trans3Plan` (CS_KERNEL_
TRANSPOSE). The name itself says it: Transpose-Row-Transpose-Row-
Transpose. `lenFactor1 = length.back()` (`= Decide1DScheme`'s own
`divLength1`, confirmed directly in `node_factory.cpp`'s own
`nodeData.length.emplace_back(divLength1)`), `lenFactor0 = length[0] /
lenFactor1`. `row1Plan` gets length `lenFactor1`/batch `lenFactor0`;
`row2Plan` gets length `lenFactor0`/batch `lenFactor1`. Two OPTIONAL
"fuse shims" (`FT_TRANS_WITH_STOCKHAM`, `FT_STOCKHAM_WITH_TRANS`) may
merge adjacent kernel pairs when profitable -- a GPU-specific
micro-optimization with no dataflow effect.

**`CC1DNode::BuildTree_internal`** builds exactly 2 child kernels:
`col2colPlan` (`CS_KERNEL_STOCKHAM_BLOCK_CC`, length `(lenFactor1,
lenFactor0)`, `large1D = length[0]`, reading the ORIGINAL buffer with a
column stride and writing packed/transposed -- a transpose+FFT+twiddle
kernel fused into one) and `row2colPlan` (`CS_KERNEL_STOCKHAM_BLOCK_RC`,
length `(lenFactor0, lenFactor1)`, reading the packed intermediate and
writing the final, transposed-back output -- an FFT+transpose kernel
fused into one). Both are ALWAYS leaves (`CreateNodeFromScheme`, never
`RecursiveBuildTree`) -- consistent with `Decide1DScheme` only choosing
`CS_L1D_CC` when `MAP_1D_LENGTH_SINGLE` already guarantees both factors
are compiled-single-kernel-safe.

**Both schemes are the identical `(lenFactor1, lenFactor0)` Cooley-Tukey
split** `Decide1DScheme` already computes (the same `divLength1` this
repo's own `rocfft_default.py` already names `div_length1`) -- `CS_L1D_
CC` simply fuses what `CS_L1D_TRTRT`'s 3 transpose kernels do into the 2
FFT kernels' own strided access instead. This is dataflow-identical to
this project's own existing `_build_recursive_node`-shaped five-kernel
PRE-transpose/near-FFT/MIDDLE-transpose/far-FFT/POST-transpose structure
(already used for clFFT's large-1D path, and for clFFT's SBCC -- see
docs/gpu_baseline_clfft_sbcc_lowering.md, the same finding as there) --
with `near = row2Plan`/`row2colPlan` (length `div_length0 = lenFactor0`,
ALWAYS a plain leaf) and `far = row1Plan`/`col2colPlan`'s counterpart
(length `div_length1 = lenFactor1`, the ONLY one that may recurse, and
only for TRTRT).

## What changed

New `_rocfft_leaf_or_recurse` (`planning/gpu_baseline/rocfft_default.py`)
mirrors clFFT's own `_plan_leaf_or_recurse`: given `decide_scheme`'s own
`(div_length1, div_length0)`, it builds the same PRE/near/MIDDLE/far/POST
`FFTRecursiveNodePlan` -- `near` always a plain `CS_KERNEL_STOCKHAM` leaf
(matching real rocFFT's own `row2Plan`/`row2colPlan`, which never
consults `Decide1DScheme` at all), `far` recursing back into `_rocfft_
leaf_or_recurse` itself when still too large (matching real rocFFT's own
`RecursiveBuildTree` on `row1Plan` -- this task's own section 4B
requirement: the child row FFT plan calls the SAME rocFFT planning
algorithm with M2NDP target inputs, never the M2NDP-native recursive
splitter). `_plan_from_scheme_decision`'s `CS_L1D_CC`/`CS_L1D_TRTRT`
branch now calls this instead of refusing. No new plan IR node type was
needed -- `FFTRecursiveNodePlan` already represents this dataflow.

If `decide_scheme` applied to the `near`/`div_length0` factor does NOT
itself resolve to `CS_KERNEL_STOCKHAM` (an upstream behavior real rocFFT
sidesteps by never calling `Decide1DScheme` there at all), this baseline
refuses honestly (`UNSUPPORTED_CURRENT_CODEGEN`) rather than guessing --
not observed in any length tested so far.

`multiprocessor_count` (docs/gpu_planner_m2ndp_target_mapping.md) is
threaded into every recursive `decide_scheme` call too, so the M2NDP-
adapted baseline's own mapping never silently reverts to the
source-faithful constant partway through a recursive CS_L1D_CC/TRTRT
tree.

## Verification

- `CS_L1D_CC`: 5 representative lengths (4704, 4913, 5488, 6144, 6561)
  all build and match `numpy.fft.fft` to within 2.6e-7.
- `CS_L1D_TRTRT`: 8 representative lengths (1029, 2057, 2058, 2079, 2080,
  2100, 2106, 2112) all build and match BOTH `numpy.fft.fft` (to within
  1.6e-7) and `numpy.fft.ifft` (to within 7.4e-11).
- Power-of-2 `CS_L1D_TRTRT` (only reachable above `CS_L1D_CC_POW2_
  THRESHOLD=262144`): N=524288, 1048576, and 16777216 (this repo's own
  existing solution-map test's named representative) all build
  successfully, confirming the recursive construction terminates
  correctly even for a length two levels beyond the compiled-single-
  kernel table.
- Some sampled `CS_L1D_TRTRT` lengths (e.g. N=1002-1006) recurse into a
  PRIME `div_length1` (167, 59, 251, 67, 503) with no compiled single
  kernel -- these correctly fall through to `UNSUPPORTED_GPU_ALGORITHM`
  (Bluestein territory, explicitly out of this task's scope), not a bug.

## What did NOT change

The `CS_KERNEL_STOCKHAM`/`CS_BLUESTEIN` branches, `Decide1DScheme`'s own
table/formula logic, and the solution-map override layer are all
untouched. `map_cooperative_kernel`/`_build_physical_transpose` are
reused verbatim, mechanically -- no new lowering utility was needed
beyond the recursive orchestration itself.
