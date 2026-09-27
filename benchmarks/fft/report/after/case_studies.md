# Representative plan differences

## native_much_faster: N=[2048]

### N=2048

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 64646 | 5 | 3 | [[4, 4, 4, 4], [4, 2]] | plain |
| gpu-clfft | spill_free_correct | True | False | 305382 | 1 | 0 | [[8, 8, 8, 4]] | persistent |
| gpu-rocfft-default | spilling | True | True | 37764 | 1 | 0 | [[16, 16, 8]] | persistent |
| gpu-vkfft | spill_free_correct | True | False | 305382 | 1 | 0 | [[8, 8, 8, 4]] | persistent |

**m2ndp-native** kernel partition: `TRANSPOSE(rows=256,cols=8,replicas=1) -> FFT(length=256,total_uthreads=8) -> TRANSPOSE(rows=8,cols=256,replicas=1) -> FFT(length=8,total_uthreads=256) -> TRANSPOSE(rows=256,cols=8,replicas=1)`
  scratchpad_uthreads=[8, 256], dram_read_bytes=81920, dram_write_bytes=81920, large_twiddle_count=1
  per-kernel cycles: {'FFTRecPre0': 7195, 'FFTRecNear0': 40264, 'FFTRecMid0': 7455, 'FFTRecLeaf1': 2539, 'FFTRecPost0': 7193}

**gpu-clfft** kernel partition: `FFT(length=2048,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=4194304, dram_write_bytes=4194304, large_twiddle_count=0
  per-kernel cycles: {'FFTClfft': 305382}

**gpu-rocfft-default**: nan

**gpu-vkfft** kernel partition: `FFT(length=2048,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=4194304, dram_write_bytes=4194304, large_twiddle_count=0
  per-kernel cycles: {'FFTVkfftLeaf0': 305382}

## closest_to_parity: N=[12]

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

## gpu_faster_than_native: N=[16, 24, 32, 48, 96, 128, 192, 256]

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

### N=32

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 6401 | 1 | 0 | [[4, 4, 2]] | plain |
| gpu-clfft | spilling | True | True | not_available | 1 | 0 | [[8, 4]] | cooperative |
| gpu-rocfft-default | spill_free_correct | True | False | 6359 | 1 | 0 | [[8, 4]] | persistent |
| gpu-vkfft | spilling | True | True | not_available | 1 | 0 | [[8, 4]] | cooperative |

**m2ndp-native** kernel partition: `FFT(length=32,total_uthreads=1)`
  scratchpad_uthreads=[1], dram_read_bytes=256, dram_write_bytes=256, large_twiddle_count=0
  per-kernel cycles: {'FFTRecLeaf0': 6401}

**gpu-clfft**: nan

**gpu-rocfft-default** kernel partition: `FFT(length=32,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=65536, dram_write_bytes=65536, large_twiddle_count=0
  per-kernel cycles: {'FFTRocfftDefault': 6359}

**gpu-vkfft**: nan

### N=48

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 10188 | 1 | 0 | [[4, 4, 3]] | plain |
| gpu-clfft | spill_free_correct | True | False | 6811 | 1 | 0 | [[6, 4, 2]] | cooperative |
| gpu-rocfft-default | spill_free_correct | True | False | 9788 | 1 | 0 | [[4, 3, 4]] | persistent |
| gpu-vkfft | spilling | True | True | not_available | 1 | 0 | [[8, 6]] | cooperative |

**m2ndp-native** kernel partition: `FFT(length=48,total_uthreads=1)`
  scratchpad_uthreads=[1], dram_read_bytes=384, dram_write_bytes=384, large_twiddle_count=0
  per-kernel cycles: {'FFTRecLeaf0': 10188}

**gpu-clfft** kernel partition: `FFT(length=48,total_uthreads=4)`
  scratchpad_uthreads=[4], dram_read_bytes=1536, dram_write_bytes=1536, large_twiddle_count=0
  per-kernel cycles: {'FFTClfft': 6811}

**gpu-rocfft-default** kernel partition: `FFT(length=48,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=98304, dram_write_bytes=98304, large_twiddle_count=0
  per-kernel cycles: {'FFTRocfftDefault': 9788}

**gpu-vkfft**: nan

### N=96

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 16233 | 1 | 0 | [[4, 4, 2, 3]] | plain |
| gpu-clfft | spill_free_correct | True | False | 7886 | 1 | 0 | [[6, 4, 4]] | cooperative |
| gpu-rocfft-default | spilling | True | True | 7241 | 1 | 0 | [[6, 16]] | persistent |
| gpu-vkfft | spilling | True | True | 2579 | 1 | 0 | [[8, 6, 2]] | persistent |

**m2ndp-native** kernel partition: `FFT(length=96,total_uthreads=1)`
  scratchpad_uthreads=[1], dram_read_bytes=768, dram_write_bytes=768, large_twiddle_count=0
  per-kernel cycles: {'FFTRecLeaf0': 16233}

**gpu-clfft** kernel partition: `FFT(length=96,total_uthreads=8)`
  scratchpad_uthreads=[8], dram_read_bytes=6144, dram_write_bytes=6144, large_twiddle_count=0
  per-kernel cycles: {'FFTClfft': 7886}

**gpu-rocfft-default**: nan

**gpu-vkfft**: nan

### N=128

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 22587 | 1 | 0 | [[4, 4, 4, 2]] | plain |
| gpu-clfft | spill_free_correct | True | False | 14433 | 1 | 0 | [[8, 4, 4]] | persistent |
| gpu-rocfft-default | spilling | True | True | 3159 | 1 | 0 | [[16, 8]] | persistent |
| gpu-vkfft | spill_free_correct | True | False | 15992 | 1 | 0 | [[8, 8, 2]] | persistent |

**m2ndp-native** kernel partition: `FFT(length=128,total_uthreads=1)`
  scratchpad_uthreads=[1], dram_read_bytes=1024, dram_write_bytes=1024, large_twiddle_count=0
  per-kernel cycles: {'FFTRecLeaf0': 22587}

**gpu-clfft** kernel partition: `FFT(length=128,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=262144, dram_write_bytes=262144, large_twiddle_count=0
  per-kernel cycles: {'FFTClfft': 14433}

**gpu-rocfft-default**: nan

**gpu-vkfft** kernel partition: `FFT(length=128,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=262144, dram_write_bytes=262144, large_twiddle_count=0
  per-kernel cycles: {'FFTVkfftLeaf0': 15992}

### N=192

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 30463 | 1 | 0 | [[4, 4, 4, 3]] | plain |
| gpu-clfft | spill_free_correct | True | False | 20795 | 1 | 0 | [[6, 4, 4, 2]] | persistent |
| gpu-rocfft-default | spill_free_correct | True | False | 20795 | 1 | 0 | [[6, 4, 4, 2]] | persistent |
| gpu-vkfft | spill_free_correct | True | False | 20283 | 1 | 0 | [[8, 8, 3]] | persistent |

**m2ndp-native** kernel partition: `FFT(length=192,total_uthreads=1)`
  scratchpad_uthreads=[1], dram_read_bytes=1536, dram_write_bytes=1536, large_twiddle_count=0
  per-kernel cycles: {'FFTRecLeaf0': 30463}

**gpu-clfft** kernel partition: `FFT(length=192,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=393216, dram_write_bytes=393216, large_twiddle_count=0
  per-kernel cycles: {'FFTClfft': 20795}

**gpu-rocfft-default** kernel partition: `FFT(length=192,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=393216, dram_write_bytes=393216, large_twiddle_count=0
  per-kernel cycles: {'FFTRocfftDefault': 20795}

**gpu-vkfft** kernel partition: `FFT(length=192,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=393216, dram_write_bytes=393216, large_twiddle_count=0
  per-kernel cycles: {'FFTVkfftLeaf0': 20283}

### N=256

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 38804 | 1 | 0 | [[4, 4, 4, 4]] | plain |
| gpu-clfft | spill_free_correct | True | False | 33705 | 1 | 0 | [[4, 4, 4, 4]] | persistent |
| gpu-rocfft-default | spill_free_correct | True | False | 33705 | 1 | 0 | [[4, 4, 4, 4]] | persistent |
| gpu-vkfft | spill_free_correct | True | False | 23897 | 1 | 0 | [[8, 8, 4]] | persistent |

**m2ndp-native** kernel partition: `FFT(length=256,total_uthreads=1)`
  scratchpad_uthreads=[1], dram_read_bytes=2048, dram_write_bytes=2048, large_twiddle_count=0
  per-kernel cycles: {'FFTRecLeaf0': 38804}

**gpu-clfft** kernel partition: `FFT(length=256,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=524288, dram_write_bytes=524288, large_twiddle_count=0
  per-kernel cycles: {'FFTClfft': 33705}

**gpu-rocfft-default** kernel partition: `FFT(length=256,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=524288, dram_write_bytes=524288, large_twiddle_count=0
  per-kernel cycles: {'FFTRocfftDefault': 33705}

**gpu-vkfft** kernel partition: `FFT(length=256,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=524288, dram_write_bytes=524288, large_twiddle_count=0
  per-kernel cycles: {'FFTVkfftLeaf0': 23897}

## explicitly_flagged: N=[960, 1024]

### N=960

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 49238 | 5 | 3 | [[4, 4, 3, 5], [4]] | plain |
| gpu-clfft | spilling | True | True | 163657 | 1 | 0 | [[10, 6, 2, 2, 2, 2]] | persistent |
| gpu-rocfft-default | spilling | True | True | 18515 | 1 | 0 | [[16, 10, 6]] | persistent |
| gpu-vkfft | spilling | True | True | 185179 | 1 | 0 | [[10, 8, 6, 2]] | persistent |

**m2ndp-native** kernel partition: `TRANSPOSE(rows=240,cols=4,replicas=1) -> FFT(length=240,total_uthreads=4) -> TRANSPOSE(rows=4,cols=240,replicas=1) -> FFT(length=4,total_uthreads=240) -> TRANSPOSE(rows=240,cols=4,replicas=1)`
  scratchpad_uthreads=[4, 240], dram_read_bytes=38400, dram_write_bytes=38400, large_twiddle_count=1
  per-kernel cycles: {'FFTRecPre0': 2241, 'FFTRecNear0': 40378, 'FFTRecMid0': 3086, 'FFTRecLeaf1': 1290, 'FFTRecPost0': 2243}

**gpu-clfft**: nan

**gpu-rocfft-default**: nan

**gpu-vkfft**: nan

### N=1024

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 49097 | 5 | 3 | [[4, 4, 4, 4], [4]] | plain |
| gpu-clfft | spill_free_correct | True | False | 137474 | 1 | 0 | [[8, 8, 4, 4]] | persistent |
| gpu-rocfft-default | spill_free_correct | True | False | 137474 | 1 | 0 | [[8, 8, 4, 4]] | persistent |
| gpu-vkfft | spill_free_correct | True | False | 141395 | 1 | 0 | [[8, 8, 8, 2]] | persistent |

**m2ndp-native** kernel partition: `TRANSPOSE(rows=256,cols=4,replicas=1) -> FFT(length=256,total_uthreads=4) -> TRANSPOSE(rows=4,cols=256,replicas=1) -> FFT(length=4,total_uthreads=256) -> TRANSPOSE(rows=256,cols=4,replicas=1)`
  scratchpad_uthreads=[4, 256], dram_read_bytes=40960, dram_write_bytes=40960, large_twiddle_count=1
  per-kernel cycles: {'FFTRecPre0': 2246, 'FFTRecNear0': 40264, 'FFTRecMid0': 3038, 'FFTRecLeaf1': 1303, 'FFTRecPost0': 2246}

**gpu-clfft** kernel partition: `FFT(length=1024,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=2097152, dram_write_bytes=2097152, large_twiddle_count=0
  per-kernel cycles: {'FFTClfft': 137474}

**gpu-rocfft-default** kernel partition: `FFT(length=1024,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=2097152, dram_write_bytes=2097152, large_twiddle_count=0
  per-kernel cycles: {'FFTRocfftDefault': 137474}

**gpu-vkfft** kernel partition: `FFT(length=1024,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=2097152, dram_write_bytes=2097152, large_twiddle_count=0
  per-kernel cycles: {'FFTVkfftLeaf0': 141395}

## most_refusals: N=[80]

### N=80

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spilling | True | True | 17407 | 1 | 0 | [[4, 4, 5]] | plain |
| gpu-clfft | spilling | True | True | 16988 | 1 | 0 | [[10, 4, 2]] | cooperative |
| gpu-rocfft-default | unsupported | not_available | not_available | not_available | not_available | not_available | not_available | not_available |
| gpu-vkfft | unsupported | not_available | not_available | not_available | not_available | not_available | not_available | not_available |

**m2ndp-native**: nan

**gpu-clfft**: nan

**gpu-rocfft-default**: [unsupported_hardware_mapping] the GPU planner wants 10 work items cooperating per transform, which is neither a divisor nor a multiple of target.interleave_chunk_uthreads=8 -- no whole number of the M2NDP address decoder's periodic interleave chunks can ever produce this exact count on one physical NDP unit (proven via direct simulator source trace, see BaselineStatus's own docstring)
original GP

**gpu-vkfft**: [unsupported_hardware_mapping] the GPU planner wants 20 work items cooperating per transform, which is neither a divisor nor a multiple of target.interleave_chunk_uthreads=8 -- no whole number of the M2NDP address decoder's periodic interleave chunks can ever produce this exact count on one physical NDP unit (proven via direct simulator source trace, see BaselineStatus's own docstring)
original GP
