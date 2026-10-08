# rocFFT-default large-1D lowering onto M2NDP

Updated 2026-10-07 for finding 1 in the GPU baseline review.

## Upstream contract

Pinned source: `ROCm/rocm-libraries` commit
`bee97df517907c771de17189cb867d3c401285ae`, under
`projects/rocfft/library/src/`.

`tree_node_1D.cpp` constructs these children in execution order:

| Scheme | First FFT (`near_fft`) | Second FFT (`far_child`) |
| --- | --- | --- |
| `CS_L1D_CC` | `div1`, forced `CS_KERNEL_STOCKHAM_BLOCK_CC` | `div0`, forced `CS_KERNEL_STOCKHAM_BLOCK_RC` |
| `CS_L1D_TRTRT` | `div1`, `RecursiveBuildTree` | `div0`, forced `CS_KERNEL_STOCKHAM` |

Here `div1 = div_length1` and `div0 = N / div1`. CC does not re-run
`Decide1DScheme` for either child. TRTRT re-runs it only for the first
child, with the caller's multiprocessor count and multiplied batch.
The previous implementation reversed the children and substituted plain
SBRR configs for SBCC/SBRC. Numerical agreement alone did not detect this:
both factor orders can compute a correct FFT but select different kernels.

## Implemented mapping

`rocfft_default._rocfft_leaf_or_recurse` now sets the common IR split to
`a = div0, b = div1`, yielding:

1. PRE transpose: `div1 x div0`.
2. First FFT: length `div1`, replicas `batch * div0`.
3. MIDDLE transpose: `div0 x div1`, with the node's `W_N` twiddle.
4. Second FFT: length `div0`, replicas `batch * div1`.
5. POST transpose: `div1 x div0`, applying `1/N` only at the inverse root.

Both children in `FFTRecursiveNodePlan` can now contain recursive nodes.
Flattening, cost-model replica collection, diagnostics, and index verification
walk both children. Native planners retain their existing tree shapes.

`rocfft_block_configs.py` ports all 32 generic SBCC and 20 generic SBRC
rows from `device/kernels/configs/config_sbcc.py` and `config_sbrc.py`.
Workgroup sizing follows `device/kernel-generator.py` and
`device/generator/stockham_gen.cpp`, `stockham_gen_base.h`:

- SBCC's default desired WGS is `16 * length / min(radices)`; an explicit
  half-LDS setting doubles it, capped at 1024.
- TPT defaults to `length / min(radices)` (`uwide`), or
  `length / max(radices)` (`wide`), unless explicitly specified.
- Generic kernels use 64 KiB LDS with occupancy two, hence a 32 KiB budget.
  The shared SP/DP kernel key sizes transforms using 16-byte elements.
- TPB is limited by LDS and `floor(desired_WGS / TPT)`; final WGS is
  `TPT * TPB`, which can differ from the config's requested WGS.

Radices and TPT feed the existing M2NDP mapping. Half-LDS and direct-register
flags are retained as upstream metadata and do not halve M2NDP scratchpad
requirements. `gpu_config.extra['leaf_configs']` records the original leaf
settings in execution order for both `plan()` and `plan_m2ndp()`.

| N | Correct first FFT | Correct second FFT |
| --- | --- | --- |
| 8192 | SBCC 64, radices `(8,8)`, TPT 8, WGS 256 | SBRC 128, radices `(8,4,4)`, TPT 16, WGS 128 |
| 4704 | SBCC 96, radices `(8,3,4)`, TPT 32, WGS 256 | SBRC 49, radices `(7,7)`, TPT 7, WGS 196 |
| 2058 | Stockham 42 | Forced Stockham 49 |

## Verification

Run from `benchmarks/fft`:

```
python -m verification.verify_gpu_baseline_rocfft_large1d
```

The regression suite checks upstream expected order/radices/worker settings,
batches 1 and 3, forced-child decision calls, generator sizing edge cases,
and table coverage for every hardcoded CC split. It checks forward/inverse
numerical output against NumPy for N=2058, 4704, and 8192, including the
M2NDP-adapted entry point. A real N=2^27 split exercises a recursive first
child through flattening, cost estimation, and code generation with a larger
scratchpad profile; it does not allocate or numerically execute that large FFT.
All 52 block-table rows were also compared with the pinned upstream files.

## Remaining performance limitation

CC still expands upstream's two fused SBCC/SBRC kernels into five M2NDP
kernels. Extra launches and full-array memory passes affect performance.
This correction resolves execution order and selected block configurations;
it does not resolve finding 3 (fused-kernel lowering). Python numerical
verification does not establish compiled simulator execution correctness.
