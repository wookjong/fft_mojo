# Representative plan differences

## native_much_faster: N=[3072]

### N=3072

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 56328 | 5 | 3 | [[4, 4, 4, 3], [4, 4]] | plain |
| gpu-clfft | compile_failure | True | not_available | not_available | 1 | 0 | [[6, 4, 4, 4, 4, 2]] | persistent |
| gpu-rocfft-default | compile_failure | True | not_available | not_available | 1 | 0 | [[6, 4, 4, 4, 4, 2]] | persistent |
| gpu-vkfft | spill_free_correct | True | False | 553045 | 1 | 0 | [[8, 8, 8, 6]] | persistent |

**m2ndp-native** kernel partition: `TRANSPOSE(rows=192,cols=16,replicas=1) -> FFT(length=192,total_uthreads=16) -> TRANSPOSE(rows=16,cols=192,replicas=1) -> FFT(length=16,total_uthreads=192) -> TRANSPOSE(rows=192,cols=16,replicas=1)`
  scratchpad_uthreads=[16, 192], dram_read_bytes=122880, dram_write_bytes=122880, large_twiddle_count=1
  per-kernel cycles: {'FFTRecPre0': 7019, 'FFTRecNear0': 32030, 'FFTRecMid0': 7261, 'FFTRecLeaf1': 2997, 'FFTRecPost0': 7021}

**gpu-clfft**: build failed: e/gen.mojo:112960:16: warning: 'if' condition always evaluates to 'True'; 'else' branch is unreachable\n            if True:  # batch 191 scope\n               ^\n/tmp/fft_spill_probe_25x1sihp/stage/gen.mojo:112979:16: warning: 'if' condition always evaluates to 'True'; 'else' branch is unreachable\n            if True:  # batch 191 scope\n               ^\nIncluded from /tmp/fft_spi

**gpu-rocfft-default**: build failed: ways evaluates to 'True'; 'else' branch is unreachable\n            if True:  # batch 191 scope\n               ^\n/tmp/fft_spill_probe_vvoob7o0/stage/gen.mojo:112979:16: warning: 'if' condition always evaluates to 'True'; 'else' branch is unreachable\n            if True:  # batch 191 scope\n               ^\nIncluded from /tmp/fft_spill_probe_vvoob7o0/stage/gen.mojo:6:\n/tmp/fft_sp

**gpu-vkfft** kernel partition: `FFT(length=3072,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=6291456, dram_write_bytes=6291456, large_twiddle_count=0
  per-kernel cycles: {'FFTVkfftLeaf0': 553045}

## closest_to_parity: N=[12]

### N=12

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 3011 | 1 | 0 | [[4, 3]] | plain |
| gpu-clfft | spill_free_correct | True | False | 3028 | 1 | 0 | [[6, 2]] | cooperative |
| gpu-rocfft-default | spill_free_correct | True | False | 3965 | 1 | 0 | [[6, 2]] | persistent |
| gpu-vkfft | spill_free_correct | True | False | 3472 | 1 | 0 | [[6, 2]] | cooperative |

**m2ndp-native** kernel partition: `FFT(length=12,total_uthreads=1)`
  scratchpad_uthreads=[1], dram_read_bytes=96, dram_write_bytes=96, large_twiddle_count=0
  per-kernel cycles: {'FFTRecLeaf0': 3011}

**gpu-clfft** kernel partition: `FFT(length=12,total_uthreads=1)`
  scratchpad_uthreads=[1], dram_read_bytes=96, dram_write_bytes=96, large_twiddle_count=0
  per-kernel cycles: {'FFTClfft': 3028}

**gpu-rocfft-default** kernel partition: `FFT(length=12,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=24576, dram_write_bytes=24576, large_twiddle_count=0
  per-kernel cycles: {'FFTRocfftDefault': 3965}

**gpu-vkfft** kernel partition: `FFT(length=12,total_uthreads=2)`
  scratchpad_uthreads=[2], dram_read_bytes=192, dram_write_bytes=192, large_twiddle_count=0
  per-kernel cycles: {'FFTVkfftLeaf0': 3472}

## gpu_faster_than_native: N=[16, 24, 32, 48, 96, 128, 192, 216, 256]

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

### N=216

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 43735 | 1 | 0 | [[4, 2, 3, 3, 3]] | plain |
| gpu-clfft | spill_free_correct | True | False | 23632 | 1 | 0 | [[6, 6, 6]] | persistent |
| gpu-rocfft-default | spill_free_correct | True | False | 25912 | 1 | 0 | [[6, 6, 6]] | persistent |
| gpu-vkfft | spilling | True | True | 4707 | 1 | 0 | [[9, 8, 3]] | persistent |

**m2ndp-native** kernel partition: `FFT(length=216,total_uthreads=1)`
  scratchpad_uthreads=[1], dram_read_bytes=1728, dram_write_bytes=1728, large_twiddle_count=0
  per-kernel cycles: {'FFTRecLeaf0': 43735}

**gpu-clfft** kernel partition: `FFT(length=216,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=442368, dram_write_bytes=442368, large_twiddle_count=0
  per-kernel cycles: {'FFTClfft': 23632}

**gpu-rocfft-default** kernel partition: `FFT(length=216,total_uthreads=256)`
  scratchpad_uthreads=[256], dram_read_bytes=442368, dram_write_bytes=442368, large_twiddle_count=0
  per-kernel cycles: {'FFTRocfftDefault': 25912}

**gpu-vkfft**: nan

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

## most_refusals: N=[40]

### N=40

| planner | status | correct | spill | cycles | kernels | transposes | radix_sequence | execution_strategy |
|---|---|---|---|---|---|---|---|---|
| m2ndp-native | spill_free_correct | True | False | 9366 | 1 | 0 | [[4, 2, 5]] | plain |
| gpu-clfft | spilling | True | True | 8605 | 1 | 0 | [[10, 4]] | cooperative |
| gpu-rocfft-default | spilling | True | True | 13374 | 1 | 0 | [[10, 4]] | persistent |
| gpu-vkfft | spilling | True | True | 13374 | 1 | 0 | [[10, 4]] | persistent |

**m2ndp-native** kernel partition: `FFT(length=40,total_uthreads=1)`
  scratchpad_uthreads=[1], dram_read_bytes=320, dram_write_bytes=320, large_twiddle_count=0
  per-kernel cycles: {'FFTRecLeaf0': 9366}

**gpu-clfft**: nan

**gpu-rocfft-default**: nan

**gpu-vkfft**: nan
