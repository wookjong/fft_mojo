# Recursive FFT host task fusion

`--task-fusion auto` groups supported recursive distributed FFT nodes into fewer
host tasks. The default is `off`. Each radix/transpose kernel still completes
before the next kernel begins; global work buffers and FFT arithmetic remain.
This is separate from `--fuse-stages 2`, which removes a local radix boundary.

`--batch-window auto` bounds leaf banks and transpose scratch together by
processing small batches inside each existing task. An explicit positive
integer selects the window width; the default `off` preserves prior behavior.
Full CXL boundary arrays remain `N * batch` elements. Only execution extents,
leaf workspaces, the common pool and SPAD arena use the window width. Workers
add the controller's batch offset to global data addresses; SPAD and twiddles
retain local addresses. Partial final windows are guarded before data loads.
For exact windows the tail arithmetic and dynamic bounds are omitted.

Auto considers power-of-two widths under a 32 KiB total SPAD tier; explicit
widths can use the existing target arena budget. This is a resource heuristic,
not a guarantee of minimum latency. Internal phase calls grow with the number
of windows, while registered functions and host task counts remain fixed.
The controller uses a runtime loop, so generated code does not grow linearly
with batch. Unsupported explicit window policies reject instead of silently
reverting to whole-batch execution. Single-window cases keep the prior source.

Example: `--batch 8 --task-fusion auto --leaf-storage spad --batch-window 4`.
The `leaf_storage_benchmark.py` runner accepts the same `--batch-window` flag
and always measures once per configuration.

For windowed execution, `FFT_EXECUTION.estimated_array_bytes` describes one
full window, not the complete logical batch. `FFT_TASK_FUSION` records the
window and total batch extents. Scale linear array traffic by `total_batch /
window_batch` (including the partial final window), rather than counting a
padded tail as a full window. Actual simulator counters also include traffic
from tables, spills and the runtime.

With `--leaf-storage auto`, an oversized resident SPAD arena falls back to
CXL placement and recomputes Params and task grouping. Fusion is retained
whenever the CXL plan fits. Explicit `--leaf-storage spad` still rejects an
unsupported placement. The focused `leaf_storage_benchmark.py` runner accepts
only `--leaf-storage cxl` with `--variant off`; SPAD/auto require fused or control.

```powershell
python benchmarks/fft/make_fft_kernel.py 16384 --planner gpu-vkfft-m2ndp --gpu-execution distributed --units-per-leaf auto --task-fusion auto --no-reference-check -o fused.mojo
```

Direct generation uses the default physical target (32 units). For the paired
RTX3060 model (28 units), use the benchmark wrapper, which reads the config:

```powershell
powershell -File experiments/m2ndp_fft/scripts/run_benchmark.ps1 -Planner gpu-vkfft-m2ndp -GpuExecution distributed -UnitsPerLeaf auto -TaskFusion auto -Sizes 8192,16384 -ExperimentId <new-id>
```

The implementation currently supports a single recursive level with two
distributed leaves, unfused radix stages, full power-of-two transpose tiles,
and a common launch arena fitting the target's budgets. Other ordinary plans
retain their existing execution and emit `FFT_TASK_FUSION` with the rejection
reason. The specialized root persistent-tail-hybrid generator retains its
existing path. `auto` uses static resource checks, not measured latency search.

The group planner preserves node order and keeps at most eight distinct kernel
functions per task. Current clFFT 8192/16384 use two tasks; VkFFT uses one.
For 28 units their common ranges are 224/448 microthreads. Leaf wrappers bound
their active range before any Params access; transpose wrappers bound tile IDs.
The per-unit arena slot is `(gid // (units * 8)) * 8 + gid % 8`.

One aggregate Params struct contains the original node Params. Sequential
transpose phases reuse one arena. Actual ELF `.spad` and `.m2ndp.kinfo` checks
remain necessary: source-level arithmetic is only a conservative preflight.

The focused experiment runner `experiments/m2ndp_fft/scripts/task_fusion_benchmark.py`
runs inside the pinned image, with read-only `/checkout` and writable
`/experiment`. It supports `fused`, `control` (five tasks using the same common
dispatch), and `off`, plus `--build-only`. Each timing configuration runs once,
with NumPy validation on that execution. Existing successful executions are
retained. A new experiment ID is required after changing the FFT checkout.

The runner's link wrapper invokes the original linker unchanged and copies the
actual device IR/object/ELF to `device_artifacts`; it checks ELF64 sections and
registered kernel records before launching. No simulator constant or device
compiler optimization flag is relaxed. The control reuses the fused plan's
launch pools, preserving allocation order and workspace placement. Leaf-only
control tasks issue one controller scratchpad store each (two total) to retain
the otherwise unused arena; actual ELF reservations are checked before treating
adaptive L1D capacity as matched. This small control-only cost and different
code placement remain limitations of the comparison.

Verification:

```powershell
$env:PYTHONPATH='benchmarks/fft'
python -B -m unittest verification.verify_fft_task_fusion verification.verify_fft_task_scratchpad verification.verify_fft_unit_parallel verification.verify_fft_kernel_registration
```

The numeric checker executes the actual emitted fused methods and device_main
schedule, randomizes worker order, gives every unit a private NaN-initialized
arena, and checks bounds, initialization, and duplicate global writers.
