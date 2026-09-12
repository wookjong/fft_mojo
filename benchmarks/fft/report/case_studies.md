# Representative plan differences

## native_much_faster: N=[12]

### N=12

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 3011 | 1 | 0 | [[4, 3]] | plain |
| gpu-clfft | spill_free_correct | True | False | 3028 | 1 | 0 | [[6, 2]] | cooperative |
| gpu-rocfft-default | unsupported | not_available | not_available | not_available | not_available | not_available | not_available | not_available |
| gpu-vkfft | spill_free_correct | True | False | 3472 | 1 | 0 | [[6, 2]] | cooperative |

**m2ndp-native** kernel partition: `FFT(length=12,total_uthreads=1)`
  scratchpad_uthreads=[1], dram_read_bytes=96, dram_write_bytes=96, large_twiddle_count=0
  per-kernel cycles: {'FFTRecLeaf0': 3011}

**gpu-clfft** kernel partition: `FFT(length=12,total_uthreads=1)`
  scratchpad_uthreads=[1], dram_read_bytes=96, dram_write_bytes=96, large_twiddle_count=0
  per-kernel cycles: {'FFTClfft': 3028}

**gpu-rocfft-default**: [unsupported_hardware_mapping] the GPU planner wants 6 work items cooperating per transform, which is neither a divisor nor a multiple of target.interleave_chunk_uthreads=8 -- no whole number of the M2NDP address decoder's periodic interleave chunks can ever produce this exact count on one physical NDP unit (proven via direct simulator source trace, see BaselineStatus's own docstring)
original GPU

**gpu-vkfft** kernel partition: `FFT(length=12,total_uthreads=2)`
  scratchpad_uthreads=[2], dram_read_bytes=192, dram_write_bytes=192, large_twiddle_count=0
  per-kernel cycles: {'FFTVkfftLeaf0': 3472}

## closest_to_parity: N=[12]

(see N=12 above)

## gpu_faster_than_native: N=[16, 24, 48, 96]

### N=16

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 2902 | 1 | 0 | [[4, 4]] | plain |
| gpu-clfft | spill_free_correct | True | False | 2570 | 1 | 0 | [[4, 4]] | cooperative |
| gpu-rocfft-default | spill_free_correct | True | False | 2570 | 1 | 0 | [[4, 4]] | cooperative |
| gpu-vkfft | spill_free_correct | True | False | 2570 | 1 | 0 | [[4, 4]] | cooperative |

**m2ndp-native** kernel partition: `FFT(length=16,total_uthreads=1)`
  scratchpad_uthreads=[1], dram_read_bytes=128, dram_write_bytes=128, large_twiddle_count=0
  per-kernel cycles: {'FFTRecLeaf0': 2902}

**gpu-clfft** kernel partition: `FFT(length=16,total_uthreads=4)`
  scratchpad_uthreads=[4], dram_read_bytes=512, dram_write_bytes=512, large_twiddle_count=0
  per-kernel cycles: {'FFTClfft': 2570}

**gpu-rocfft-default** kernel partition: `FFT(length=16,total_uthreads=4)`
  scratchpad_uthreads=[4], dram_read_bytes=512, dram_write_bytes=512, large_twiddle_count=0
  per-kernel cycles: {'FFTRocfftDefault': 2570}

**gpu-vkfft** kernel partition: `FFT(length=16,total_uthreads=4)`
  scratchpad_uthreads=[4], dram_read_bytes=512, dram_write_bytes=512, large_twiddle_count=0
  per-kernel cycles: {'FFTVkfftLeaf0': 2570}

### N=24

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 5533 | 1 | 0 | [[4, 2, 3]] | plain |
| gpu-clfft | spill_free_correct | True | False | 4389 | 1 | 0 | [[6, 4]] | cooperative |
| gpu-rocfft-default | spilling | True | True | not_available | 1 | 0 | [[8, 3]] | cooperative |
| gpu-vkfft | spilling | True | True | not_available | 1 | 0 | [[8, 3]] | cooperative |

**m2ndp-native** kernel partition: `FFT(length=24,total_uthreads=1)`
  scratchpad_uthreads=[1], dram_read_bytes=192, dram_write_bytes=192, large_twiddle_count=0
  per-kernel cycles: {'FFTRecLeaf0': 5533}

**gpu-clfft** kernel partition: `FFT(length=24,total_uthreads=2)`
  scratchpad_uthreads=[2], dram_read_bytes=384, dram_write_bytes=384, large_twiddle_count=0
  per-kernel cycles: {'FFTClfft': 4389}

**gpu-rocfft-default**: nan

**gpu-vkfft**: nan

### N=48

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 10188 | 1 | 0 | [[4, 4, 3]] | plain |
| gpu-clfft | spill_free_correct | True | False | 6811 | 1 | 0 | [[6, 4, 2]] | cooperative |
| gpu-rocfft-default | unsupported | not_available | not_available | not_available | not_available | not_available | not_available | not_available |
| gpu-vkfft | spilling | True | True | not_available | 1 | 0 | [[8, 6]] | cooperative |

**m2ndp-native** kernel partition: `FFT(length=48,total_uthreads=1)`
  scratchpad_uthreads=[1], dram_read_bytes=384, dram_write_bytes=384, large_twiddle_count=0
  per-kernel cycles: {'FFTRecLeaf0': 10188}

**gpu-clfft** kernel partition: `FFT(length=48,total_uthreads=4)`
  scratchpad_uthreads=[4], dram_read_bytes=1536, dram_write_bytes=1536, large_twiddle_count=0
  per-kernel cycles: {'FFTClfft': 6811}

**gpu-rocfft-default**: [unsupported_current_codegen] the GPU planner wants 16 work items cooperating per transform -- an exact multiple of target.interleave_chunk_uthreads=8. This is architecturally reachable on M2NDP via a striped, multi-wave DRAM layout (the address decoder's chunk-to-unit assignment repeats every chunk*num_ndp_units=256 global ids, always landing back on the same physical unit -- see BaselineStatus's

**gpu-vkfft**: nan

### N=96

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 16233 | 1 | 0 | [[4, 4, 2, 3]] | plain |
| gpu-clfft | spill_free_correct | True | False | 7886 | 1 | 0 | [[6, 4, 4]] | cooperative |
| gpu-rocfft-default | unsupported | not_available | not_available | not_available | not_available | not_available | not_available | not_available |
| gpu-vkfft | unsupported | not_available | not_available | not_available | not_available | not_available | not_available | not_available |

**m2ndp-native** kernel partition: `FFT(length=96,total_uthreads=1)`
  scratchpad_uthreads=[1], dram_read_bytes=768, dram_write_bytes=768, large_twiddle_count=0
  per-kernel cycles: {'FFTRecLeaf0': 16233}

**gpu-clfft** kernel partition: `FFT(length=96,total_uthreads=8)`
  scratchpad_uthreads=[8], dram_read_bytes=6144, dram_write_bytes=6144, large_twiddle_count=0
  per-kernel cycles: {'FFTClfft': 7886}

**gpu-rocfft-default**: [unsupported_current_codegen] the GPU planner wants 16 work items cooperating per transform -- an exact multiple of target.interleave_chunk_uthreads=8. This is architecturally reachable on M2NDP via a striped, multi-wave DRAM layout (the address decoder's chunk-to-unit assignment repeats every chunk*num_ndp_units=256 global ids, always landing back on the same physical unit -- see BaselineStatus's

**gpu-vkfft**: [unsupported_current_codegen] the GPU planner wants 16 work items cooperating per transform -- an exact multiple of target.interleave_chunk_uthreads=8. This is architecturally reachable on M2NDP via a striped, multi-wave DRAM layout (the address decoder's chunk-to-unit assignment repeats every chunk*num_ndp_units=256 global ids, always landing back on the same physical unit -- see BaselineStatus's

## explicitly_flagged: N=[960, 1024]

### N=960

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 49238 | 5 | 3 | [[4, 4, 3, 5], [4]] | plain |
| gpu-clfft | unsupported | not_available | not_available | not_available | not_available | not_available | not_available | not_available |
| gpu-rocfft-default | unsupported | not_available | not_available | not_available | not_available | not_available | not_available | not_available |
| gpu-vkfft | unsupported | not_available | not_available | not_available | not_available | not_available | not_available | not_available |
| gpu-rocfft-tuned | unsupported | not_available | not_available | not_available | not_available | not_available | not_available | not_available |

**m2ndp-native** kernel partition: `TRANSPOSE(rows=240,cols=4,replicas=1) -> FFT(length=240,total_uthreads=4) -> TRANSPOSE(rows=4,cols=240,replicas=1) -> FFT(length=4,total_uthreads=240) -> TRANSPOSE(rows=240,cols=4,replicas=1)`
  scratchpad_uthreads=[4, 240], dram_read_bytes=38400, dram_write_bytes=38400, large_twiddle_count=1
  per-kernel cycles: {'FFTRecPre0': 2241, 'FFTRecNear0': 40378, 'FFTRecMid0': 3086, 'FFTRecLeaf1': 1290, 'FFTRecPost0': 2243}

**gpu-clfft**: [unsupported_current_codegen] the GPU planner wants 32 work items cooperating per transform -- an exact multiple of target.interleave_chunk_uthreads=8. This is architecturally reachable on M2NDP via a striped, multi-wave DRAM layout (the address decoder's chunk-to-unit assignment repeats every chunk*num_ndp_units=256 global ids, always landing back on the same physical unit -- see BaselineStatus's

**gpu-rocfft-default**: [unsupported_current_codegen] the GPU planner wants 160 work items cooperating per transform -- an exact multiple of target.interleave_chunk_uthreads=8. This is architecturally reachable on M2NDP via a striped, multi-wave DRAM layout (the address decoder's chunk-to-unit assignment repeats every chunk*num_ndp_units=256 global ids, always landing back on the same physical unit -- see BaselineStatus'

**gpu-vkfft**: [unsupported_current_codegen] the GPU planner wants 192 work items cooperating per transform -- an exact multiple of target.interleave_chunk_uthreads=8. This is architecturally reachable on M2NDP via a striped, multi-wave DRAM layout (the address decoder's chunk-to-unit assignment repeats every chunk*num_ndp_units=256 global ids, always landing back on the same physical unit -- see BaselineStatus'

**gpu-rocfft-tuned**: [unsupported_current_codegen] the GPU planner wants 16 work items cooperating per transform -- an exact multiple of target.interleave_chunk_uthreads=8. This is architecturally reachable on M2NDP via a striped, multi-wave DRAM layout (the address decoder's chunk-to-unit assignment repeats every chunk*num_ndp_units=256 global ids, always landing back on the same physical unit -- see BaselineStatus's

### N=1024

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 49097 | 5 | 3 | [[4, 4, 4, 4], [4]] | plain |
| gpu-clfft | unsupported | not_available | not_available | not_available | not_available | not_available | not_available | not_available |
| gpu-rocfft-default | unsupported | not_available | not_available | not_available | not_available | not_available | not_available | not_available |
| gpu-vkfft | unsupported | not_available | not_available | not_available | not_available | not_available | not_available | not_available |
| gpu-rocfft-tuned | unsupported | not_available | not_available | not_available | not_available | not_available | not_available | not_available |

**m2ndp-native** kernel partition: `TRANSPOSE(rows=256,cols=4,replicas=1) -> FFT(length=256,total_uthreads=4) -> TRANSPOSE(rows=4,cols=256,replicas=1) -> FFT(length=4,total_uthreads=256) -> TRANSPOSE(rows=256,cols=4,replicas=1)`
  scratchpad_uthreads=[4, 256], dram_read_bytes=40960, dram_write_bytes=40960, large_twiddle_count=1
  per-kernel cycles: {'FFTRecPre0': 2246, 'FFTRecNear0': 40264, 'FFTRecMid0': 3038, 'FFTRecLeaf1': 1303, 'FFTRecPost0': 2246}

**gpu-clfft**: [unsupported_current_codegen] the GPU planner wants 128 work items cooperating per transform -- an exact multiple of target.interleave_chunk_uthreads=8. This is architecturally reachable on M2NDP via a striped, multi-wave DRAM layout (the address decoder's chunk-to-unit assignment repeats every chunk*num_ndp_units=256 global ids, always landing back on the same physical unit -- see BaselineStatus'

**gpu-rocfft-default**: [unsupported_current_codegen] the GPU planner wants 128 work items cooperating per transform -- an exact multiple of target.interleave_chunk_uthreads=8. This is architecturally reachable on M2NDP via a striped, multi-wave DRAM layout (the address decoder's chunk-to-unit assignment repeats every chunk*num_ndp_units=256 global ids, always landing back on the same physical unit -- see BaselineStatus'

**gpu-vkfft**: [unsupported_current_codegen] the GPU planner wants 128 work items cooperating per transform -- an exact multiple of target.interleave_chunk_uthreads=8. This is architecturally reachable on M2NDP via a striped, multi-wave DRAM layout (the address decoder's chunk-to-unit assignment repeats every chunk*num_ndp_units=256 global ids, always landing back on the same physical unit -- see BaselineStatus'

**gpu-rocfft-tuned**: [unsupported_current_codegen] the GPU planner wants 16 work items cooperating per transform -- an exact multiple of target.interleave_chunk_uthreads=8. This is architecturally reachable on M2NDP via a striped, multi-wave DRAM layout (the address decoder's chunk-to-unit assignment repeats every chunk*num_ndp_units=256 global ids, always landing back on the same physical unit -- see BaselineStatus's

## most_refusals: N=[64]

### N=64

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spilling | True | True | 13126 | 1 | 0 | [[4, 4, 4]] | plain |
| gpu-clfft | unsupported | not_available | not_available | not_available | not_available | not_available | not_available | not_available |
| gpu-rocfft-default | unsupported | not_available | not_available | not_available | not_available | not_available | not_available | not_available |
| gpu-vkfft | spilling | True | True | not_available | 1 | 0 | [[8, 8]] | cooperative |
| gpu-rocfft-tuned | spill_free_correct | True | False | 5649 | 1 | 0 | [[4, 4, 4]] | cooperative |

**m2ndp-native**: nan

**gpu-clfft**: [unsupported_current_codegen] the GPU planner wants 16 work items cooperating per transform -- an exact multiple of target.interleave_chunk_uthreads=8. This is architecturally reachable on M2NDP via a striped, multi-wave DRAM layout (the address decoder's chunk-to-unit assignment repeats every chunk*num_ndp_units=256 global ids, always landing back on the same physical unit -- see BaselineStatus's

**gpu-rocfft-default**: [unsupported_current_codegen] the GPU planner wants 16 work items cooperating per transform -- an exact multiple of target.interleave_chunk_uthreads=8. This is architecturally reachable on M2NDP via a striped, multi-wave DRAM layout (the address decoder's chunk-to-unit assignment repeats every chunk*num_ndp_units=256 global ids, always landing back on the same physical unit -- see BaselineStatus's

**gpu-vkfft**: nan

**gpu-rocfft-tuned** kernel partition: `FFT(length=64,total_uthreads=4)`
  scratchpad_uthreads=[4], dram_read_bytes=2048, dram_write_bytes=2048, large_twiddle_count=0
  per-kernel cycles: {'FFTRocfft': 5649}
