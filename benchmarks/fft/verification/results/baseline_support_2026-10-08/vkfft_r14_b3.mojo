from std.sys import size_of
from std.random import random_float64, seed
from std.math import cos as host_cos, sin as host_sin

from m2ndp import VECTOR_WIDTH, NDPTask, Domain, MappedDomain, UTHREAD_BYTES, global_uthread_id, local_uthread_id, group_id, num_groups, launch_parallel, scratchpad
from m2ndp_host import cxl_alloc

comptime W = VECTOR_WIDTH // size_of[Float32]()

struct PooledRange:
    @staticmethod
    def over(data: UnsafePointer[Float32, MutAnyOrigin], count: Int) -> MappedDomain:
        return MappedDomain(data, Domain(count * size_of[Float32]() // UTHREAD_BYTES))

comptime N = 14

# ============================================================================
# KERNEL IMPLEMENTATIONS
# ============================================================================
# 1 stage(s), chained through DRAM in this order:
#   0: FFTVkfftLeaf0 (FFT)

# ---- stage 0: FFTVkfftLeaf0 ----
# FFT_EXECUTION {"arithmetic_lanes": [8], "compute_lanes": null, "kernel": "FFTVkfftLeaf0", "launch": {"alignment_bytes": 256, "chunk_uthreads": 8, "padded": false, "round_fft_capacity": 3, "round_uthread_capacity": 3, "simd_lanes": 8, "slots_per_chunk": 8, "spread_across_units": false, "target_units": 32, "total_ffts": 3, "workers_per_fft": 1}, "loop_stages": true, "narrow_middle_stages": false}
comptime MAX_UTHREAD_FFTVkfftLeaf0 = 3

@fieldwise_init
struct FFTVkfftLeaf0Params(Movable):
    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var loop_twiddle_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var loop_twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]


struct FFTVkfftLeaf0(NDPTask):
    comptime Params = FFTVkfftLeaf0Params

    @staticmethod
    def stage_0():
        ref p = FFTVkfftLeaf0.params[]
        comptime RADIX = 14
        comptime SIMD_ITERS = 1

        comptime WORKERS_PER_FFT = 1
        var local_id = local_uthread_id()
        var fft_slot = local_id // WORKERS_PER_FFT
        var worker_id = local_id % WORKERS_PER_FFT
        if fft_slot >= 3:
            return
        var logical_fft_id = global_uthread_id() // WORKERS_PER_FFT
        var in_batch_base = logical_fft_id * 14
        var out_batch_base = logical_fft_id * 14

        if worker_id == 0:
            # ===== stage 0, SIMD batch 0 (valid lanes: 1/8) =====
            var rr0_lane0 = p.input_real_base.load[width=1](in_batch_base + 0)
            var ii0_lane0 = p.input_imag_base.load[width=1](in_batch_base + 0)
            var rr0 = SIMD[DType.float32, 8](rr0_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var ii0 = SIMD[DType.float32, 8](ii0_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var rr1_lane0 = p.input_real_base.load[width=1](in_batch_base + 1)
            var ii1_lane0 = p.input_imag_base.load[width=1](in_batch_base + 1)
            var rr1 = SIMD[DType.float32, 8](rr1_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var ii1 = SIMD[DType.float32, 8](ii1_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var rr2_lane0 = p.input_real_base.load[width=1](in_batch_base + 2)
            var ii2_lane0 = p.input_imag_base.load[width=1](in_batch_base + 2)
            var rr2 = SIMD[DType.float32, 8](rr2_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var ii2 = SIMD[DType.float32, 8](ii2_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var rr3_lane0 = p.input_real_base.load[width=1](in_batch_base + 3)
            var ii3_lane0 = p.input_imag_base.load[width=1](in_batch_base + 3)
            var rr3 = SIMD[DType.float32, 8](rr3_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var ii3 = SIMD[DType.float32, 8](ii3_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var rr4_lane0 = p.input_real_base.load[width=1](in_batch_base + 4)
            var ii4_lane0 = p.input_imag_base.load[width=1](in_batch_base + 4)
            var rr4 = SIMD[DType.float32, 8](rr4_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var ii4 = SIMD[DType.float32, 8](ii4_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var rr5_lane0 = p.input_real_base.load[width=1](in_batch_base + 5)
            var ii5_lane0 = p.input_imag_base.load[width=1](in_batch_base + 5)
            var rr5 = SIMD[DType.float32, 8](rr5_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var ii5 = SIMD[DType.float32, 8](ii5_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var rr6_lane0 = p.input_real_base.load[width=1](in_batch_base + 6)
            var ii6_lane0 = p.input_imag_base.load[width=1](in_batch_base + 6)
            var rr6 = SIMD[DType.float32, 8](rr6_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var ii6 = SIMD[DType.float32, 8](ii6_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var rr7_lane0 = p.input_real_base.load[width=1](in_batch_base + 7)
            var ii7_lane0 = p.input_imag_base.load[width=1](in_batch_base + 7)
            var rr7 = SIMD[DType.float32, 8](rr7_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var ii7 = SIMD[DType.float32, 8](ii7_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var rr8_lane0 = p.input_real_base.load[width=1](in_batch_base + 8)
            var ii8_lane0 = p.input_imag_base.load[width=1](in_batch_base + 8)
            var rr8 = SIMD[DType.float32, 8](rr8_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var ii8 = SIMD[DType.float32, 8](ii8_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var rr9_lane0 = p.input_real_base.load[width=1](in_batch_base + 9)
            var ii9_lane0 = p.input_imag_base.load[width=1](in_batch_base + 9)
            var rr9 = SIMD[DType.float32, 8](rr9_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var ii9 = SIMD[DType.float32, 8](ii9_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var rr10_lane0 = p.input_real_base.load[width=1](in_batch_base + 10)
            var ii10_lane0 = p.input_imag_base.load[width=1](in_batch_base + 10)
            var rr10 = SIMD[DType.float32, 8](rr10_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var ii10 = SIMD[DType.float32, 8](ii10_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var rr11_lane0 = p.input_real_base.load[width=1](in_batch_base + 11)
            var ii11_lane0 = p.input_imag_base.load[width=1](in_batch_base + 11)
            var rr11 = SIMD[DType.float32, 8](rr11_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var ii11 = SIMD[DType.float32, 8](ii11_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var rr12_lane0 = p.input_real_base.load[width=1](in_batch_base + 12)
            var ii12_lane0 = p.input_imag_base.load[width=1](in_batch_base + 12)
            var rr12 = SIMD[DType.float32, 8](rr12_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var ii12 = SIMD[DType.float32, 8](ii12_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var rr13_lane0 = p.input_real_base.load[width=1](in_batch_base + 13)
            var ii13_lane0 = p.input_imag_base.load[width=1](in_batch_base + 13)
            var rr13 = SIMD[DType.float32, 8](rr13_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
            var ii13 = SIMD[DType.float32, 8](ii13_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))

            # fixed radix-14 butterfly
            var ctg0_p1r = rr2 + rr12
            var ctg0_p1i = ii2 + ii12
            var ctg0_m1r = rr2 - rr12
            var ctg0_m1i = ii2 - ii12
            var ctg0_p2r = rr4 + rr10
            var ctg0_p2i = ii4 + ii10
            var ctg0_m2r = rr4 - rr10
            var ctg0_m2i = ii4 - ii10
            var ctg0_p3r = rr6 + rr8
            var ctg0_p3i = ii6 + ii8
            var ctg0_m3r = rr6 - rr8
            var ctg0_m3i = ii6 - ii8
            var ctg0_r0 = rr0
            var ctg0_i0 = ii0
            ctg0_r0 += ctg0_p1r
            ctg0_i0 += ctg0_p1i
            ctg0_r0 += ctg0_p2r
            ctg0_i0 += ctg0_p2i
            ctg0_r0 += ctg0_p3r
            ctg0_i0 += ctg0_p3i
            var ctg0_r1 = rr0
            var ctg0_i1 = ii0
            var ctg0_r6 = rr0
            var ctg0_i6 = ii0
            ctg0_r1 += ctg0_p1r * Float32(0.623489802) - ctg0_m1i * Float32(0.781831482)
            ctg0_i1 += ctg0_p1i * Float32(0.623489802) + ctg0_m1r * Float32(0.781831482)
            ctg0_r6 += ctg0_p1r * Float32(0.623489802) + ctg0_m1i * Float32(0.781831482)
            ctg0_i6 += ctg0_p1i * Float32(0.623489802) - ctg0_m1r * Float32(0.781831482)
            ctg0_r1 += ctg0_p2r * Float32(-0.222520934) - ctg0_m2i * Float32(0.974927912)
            ctg0_i1 += ctg0_p2i * Float32(-0.222520934) + ctg0_m2r * Float32(0.974927912)
            ctg0_r6 += ctg0_p2r * Float32(-0.222520934) + ctg0_m2i * Float32(0.974927912)
            ctg0_i6 += ctg0_p2i * Float32(-0.222520934) - ctg0_m2r * Float32(0.974927912)
            ctg0_r1 += ctg0_p3r * Float32(-0.900968868) - ctg0_m3i * Float32(0.433883739)
            ctg0_i1 += ctg0_p3i * Float32(-0.900968868) + ctg0_m3r * Float32(0.433883739)
            ctg0_r6 += ctg0_p3r * Float32(-0.900968868) + ctg0_m3i * Float32(0.433883739)
            ctg0_i6 += ctg0_p3i * Float32(-0.900968868) - ctg0_m3r * Float32(0.433883739)
            var ctg0_r2 = rr0
            var ctg0_i2 = ii0
            var ctg0_r5 = rr0
            var ctg0_i5 = ii0
            ctg0_r2 += ctg0_p1r * Float32(-0.222520934) - ctg0_m1i * Float32(0.974927912)
            ctg0_i2 += ctg0_p1i * Float32(-0.222520934) + ctg0_m1r * Float32(0.974927912)
            ctg0_r5 += ctg0_p1r * Float32(-0.222520934) + ctg0_m1i * Float32(0.974927912)
            ctg0_i5 += ctg0_p1i * Float32(-0.222520934) - ctg0_m1r * Float32(0.974927912)
            ctg0_r2 += ctg0_p2r * Float32(-0.900968868) - ctg0_m2i * Float32(-0.433883739)
            ctg0_i2 += ctg0_p2i * Float32(-0.900968868) + ctg0_m2r * Float32(-0.433883739)
            ctg0_r5 += ctg0_p2r * Float32(-0.900968868) + ctg0_m2i * Float32(-0.433883739)
            ctg0_i5 += ctg0_p2i * Float32(-0.900968868) - ctg0_m2r * Float32(-0.433883739)
            ctg0_r2 += ctg0_p3r * Float32(0.623489802) - ctg0_m3i * Float32(-0.781831482)
            ctg0_i2 += ctg0_p3i * Float32(0.623489802) + ctg0_m3r * Float32(-0.781831482)
            ctg0_r5 += ctg0_p3r * Float32(0.623489802) + ctg0_m3i * Float32(-0.781831482)
            ctg0_i5 += ctg0_p3i * Float32(0.623489802) - ctg0_m3r * Float32(-0.781831482)
            var ctg0_r3 = rr0
            var ctg0_i3 = ii0
            var ctg0_r4 = rr0
            var ctg0_i4 = ii0
            ctg0_r3 += ctg0_p1r * Float32(-0.900968868) - ctg0_m1i * Float32(0.433883739)
            ctg0_i3 += ctg0_p1i * Float32(-0.900968868) + ctg0_m1r * Float32(0.433883739)
            ctg0_r4 += ctg0_p1r * Float32(-0.900968868) + ctg0_m1i * Float32(0.433883739)
            ctg0_i4 += ctg0_p1i * Float32(-0.900968868) - ctg0_m1r * Float32(0.433883739)
            ctg0_r3 += ctg0_p2r * Float32(0.623489802) - ctg0_m2i * Float32(-0.781831482)
            ctg0_i3 += ctg0_p2i * Float32(0.623489802) + ctg0_m2r * Float32(-0.781831482)
            ctg0_r4 += ctg0_p2r * Float32(0.623489802) + ctg0_m2i * Float32(-0.781831482)
            ctg0_i4 += ctg0_p2i * Float32(0.623489802) - ctg0_m2r * Float32(-0.781831482)
            ctg0_r3 += ctg0_p3r * Float32(-0.222520934) - ctg0_m3i * Float32(0.974927912)
            ctg0_i3 += ctg0_p3i * Float32(-0.222520934) + ctg0_m3r * Float32(0.974927912)
            ctg0_r4 += ctg0_p3r * Float32(-0.222520934) + ctg0_m3i * Float32(0.974927912)
            ctg0_i4 += ctg0_p3i * Float32(-0.222520934) - ctg0_m3r * Float32(0.974927912)
            var ctg1_p1r = rr3 + rr13
            var ctg1_p1i = ii3 + ii13
            var ctg1_m1r = rr3 - rr13
            var ctg1_m1i = ii3 - ii13
            var ctg1_p2r = rr5 + rr11
            var ctg1_p2i = ii5 + ii11
            var ctg1_m2r = rr5 - rr11
            var ctg1_m2i = ii5 - ii11
            var ctg1_p3r = rr7 + rr9
            var ctg1_p3i = ii7 + ii9
            var ctg1_m3r = rr7 - rr9
            var ctg1_m3i = ii7 - ii9
            var ctg1_r0 = rr1
            var ctg1_i0 = ii1
            ctg1_r0 += ctg1_p1r
            ctg1_i0 += ctg1_p1i
            ctg1_r0 += ctg1_p2r
            ctg1_i0 += ctg1_p2i
            ctg1_r0 += ctg1_p3r
            ctg1_i0 += ctg1_p3i
            var ctg1_r1 = rr1
            var ctg1_i1 = ii1
            var ctg1_r6 = rr1
            var ctg1_i6 = ii1
            ctg1_r1 += ctg1_p1r * Float32(0.623489802) - ctg1_m1i * Float32(0.781831482)
            ctg1_i1 += ctg1_p1i * Float32(0.623489802) + ctg1_m1r * Float32(0.781831482)
            ctg1_r6 += ctg1_p1r * Float32(0.623489802) + ctg1_m1i * Float32(0.781831482)
            ctg1_i6 += ctg1_p1i * Float32(0.623489802) - ctg1_m1r * Float32(0.781831482)
            ctg1_r1 += ctg1_p2r * Float32(-0.222520934) - ctg1_m2i * Float32(0.974927912)
            ctg1_i1 += ctg1_p2i * Float32(-0.222520934) + ctg1_m2r * Float32(0.974927912)
            ctg1_r6 += ctg1_p2r * Float32(-0.222520934) + ctg1_m2i * Float32(0.974927912)
            ctg1_i6 += ctg1_p2i * Float32(-0.222520934) - ctg1_m2r * Float32(0.974927912)
            ctg1_r1 += ctg1_p3r * Float32(-0.900968868) - ctg1_m3i * Float32(0.433883739)
            ctg1_i1 += ctg1_p3i * Float32(-0.900968868) + ctg1_m3r * Float32(0.433883739)
            ctg1_r6 += ctg1_p3r * Float32(-0.900968868) + ctg1_m3i * Float32(0.433883739)
            ctg1_i6 += ctg1_p3i * Float32(-0.900968868) - ctg1_m3r * Float32(0.433883739)
            var ctg1_r2 = rr1
            var ctg1_i2 = ii1
            var ctg1_r5 = rr1
            var ctg1_i5 = ii1
            ctg1_r2 += ctg1_p1r * Float32(-0.222520934) - ctg1_m1i * Float32(0.974927912)
            ctg1_i2 += ctg1_p1i * Float32(-0.222520934) + ctg1_m1r * Float32(0.974927912)
            ctg1_r5 += ctg1_p1r * Float32(-0.222520934) + ctg1_m1i * Float32(0.974927912)
            ctg1_i5 += ctg1_p1i * Float32(-0.222520934) - ctg1_m1r * Float32(0.974927912)
            ctg1_r2 += ctg1_p2r * Float32(-0.900968868) - ctg1_m2i * Float32(-0.433883739)
            ctg1_i2 += ctg1_p2i * Float32(-0.900968868) + ctg1_m2r * Float32(-0.433883739)
            ctg1_r5 += ctg1_p2r * Float32(-0.900968868) + ctg1_m2i * Float32(-0.433883739)
            ctg1_i5 += ctg1_p2i * Float32(-0.900968868) - ctg1_m2r * Float32(-0.433883739)
            ctg1_r2 += ctg1_p3r * Float32(0.623489802) - ctg1_m3i * Float32(-0.781831482)
            ctg1_i2 += ctg1_p3i * Float32(0.623489802) + ctg1_m3r * Float32(-0.781831482)
            ctg1_r5 += ctg1_p3r * Float32(0.623489802) + ctg1_m3i * Float32(-0.781831482)
            ctg1_i5 += ctg1_p3i * Float32(0.623489802) - ctg1_m3r * Float32(-0.781831482)
            var ctg1_r3 = rr1
            var ctg1_i3 = ii1
            var ctg1_r4 = rr1
            var ctg1_i4 = ii1
            ctg1_r3 += ctg1_p1r * Float32(-0.900968868) - ctg1_m1i * Float32(0.433883739)
            ctg1_i3 += ctg1_p1i * Float32(-0.900968868) + ctg1_m1r * Float32(0.433883739)
            ctg1_r4 += ctg1_p1r * Float32(-0.900968868) + ctg1_m1i * Float32(0.433883739)
            ctg1_i4 += ctg1_p1i * Float32(-0.900968868) - ctg1_m1r * Float32(0.433883739)
            ctg1_r3 += ctg1_p2r * Float32(0.623489802) - ctg1_m2i * Float32(-0.781831482)
            ctg1_i3 += ctg1_p2i * Float32(0.623489802) + ctg1_m2r * Float32(-0.781831482)
            ctg1_r4 += ctg1_p2r * Float32(0.623489802) + ctg1_m2i * Float32(-0.781831482)
            ctg1_i4 += ctg1_p2i * Float32(0.623489802) - ctg1_m2r * Float32(-0.781831482)
            ctg1_r3 += ctg1_p3r * Float32(-0.222520934) - ctg1_m3i * Float32(0.974927912)
            ctg1_i3 += ctg1_p3i * Float32(-0.222520934) + ctg1_m3r * Float32(0.974927912)
            ctg1_r4 += ctg1_p3r * Float32(-0.222520934) + ctg1_m3i * Float32(0.974927912)
            ctg1_i4 += ctg1_p3i * Float32(-0.222520934) - ctg1_m3r * Float32(0.974927912)
            var ctf0_r0 = ctg0_r0 + ctg1_r0
            var ctf0_i0 = ctg0_i0 + ctg1_i0
            var ctf0_r1 = ctg0_r0 - ctg1_r0
            var ctf0_i1 = ctg0_i0 - ctg1_i0
            var or0 = ctf0_r0
            var oi0 = ctf0_i0
            or0 *= Float32(0.0714285714)
            oi0 *= Float32(0.0714285714)
            p.output_real_base.store(out_batch_base + 0, or0[0])
            p.output_imag_base.store(out_batch_base + 0, oi0[0])

            var or7 = ctf0_r1
            var oi7 = ctf0_i1
            or7 *= Float32(0.0714285714)
            oi7 *= Float32(0.0714285714)
            p.output_real_base.store(out_batch_base + 7, or7[0])
            p.output_imag_base.store(out_batch_base + 7, oi7[0])

            var cttw_1_1r = ctg1_r1 * Float32(0.900968868) - ctg1_i1 * Float32(0.433883739)
            var cttw_1_1i = ctg1_r1 * Float32(0.433883739) + ctg1_i1 * Float32(0.900968868)
            var ctf1_r0 = ctg0_r1 + cttw_1_1r
            var ctf1_i0 = ctg0_i1 + cttw_1_1i
            var ctf1_r1 = ctg0_r1 - cttw_1_1r
            var ctf1_i1 = ctg0_i1 - cttw_1_1i
            var or1 = ctf1_r0
            var oi1 = ctf1_i0
            or1 *= Float32(0.0714285714)
            oi1 *= Float32(0.0714285714)
            p.output_real_base.store(out_batch_base + 1, or1[0])
            p.output_imag_base.store(out_batch_base + 1, oi1[0])

            var or8 = ctf1_r1
            var oi8 = ctf1_i1
            or8 *= Float32(0.0714285714)
            oi8 *= Float32(0.0714285714)
            p.output_real_base.store(out_batch_base + 8, or8[0])
            p.output_imag_base.store(out_batch_base + 8, oi8[0])

            var cttw_2_1r = ctg1_r2 * Float32(0.623489802) - ctg1_i2 * Float32(0.781831482)
            var cttw_2_1i = ctg1_r2 * Float32(0.781831482) + ctg1_i2 * Float32(0.623489802)
            var ctf2_r0 = ctg0_r2 + cttw_2_1r
            var ctf2_i0 = ctg0_i2 + cttw_2_1i
            var ctf2_r1 = ctg0_r2 - cttw_2_1r
            var ctf2_i1 = ctg0_i2 - cttw_2_1i
            var or2 = ctf2_r0
            var oi2 = ctf2_i0
            or2 *= Float32(0.0714285714)
            oi2 *= Float32(0.0714285714)
            p.output_real_base.store(out_batch_base + 2, or2[0])
            p.output_imag_base.store(out_batch_base + 2, oi2[0])

            var or9 = ctf2_r1
            var oi9 = ctf2_i1
            or9 *= Float32(0.0714285714)
            oi9 *= Float32(0.0714285714)
            p.output_real_base.store(out_batch_base + 9, or9[0])
            p.output_imag_base.store(out_batch_base + 9, oi9[0])

            var cttw_3_1r = ctg1_r3 * Float32(0.222520934) - ctg1_i3 * Float32(0.974927912)
            var cttw_3_1i = ctg1_r3 * Float32(0.974927912) + ctg1_i3 * Float32(0.222520934)
            var ctf3_r0 = ctg0_r3 + cttw_3_1r
            var ctf3_i0 = ctg0_i3 + cttw_3_1i
            var ctf3_r1 = ctg0_r3 - cttw_3_1r
            var ctf3_i1 = ctg0_i3 - cttw_3_1i
            var or3 = ctf3_r0
            var oi3 = ctf3_i0
            or3 *= Float32(0.0714285714)
            oi3 *= Float32(0.0714285714)
            p.output_real_base.store(out_batch_base + 3, or3[0])
            p.output_imag_base.store(out_batch_base + 3, oi3[0])

            var or10 = ctf3_r1
            var oi10 = ctf3_i1
            or10 *= Float32(0.0714285714)
            oi10 *= Float32(0.0714285714)
            p.output_real_base.store(out_batch_base + 10, or10[0])
            p.output_imag_base.store(out_batch_base + 10, oi10[0])

            var cttw_4_1r = ctg1_r4 * Float32(-0.222520934) - ctg1_i4 * Float32(0.974927912)
            var cttw_4_1i = ctg1_r4 * Float32(0.974927912) + ctg1_i4 * Float32(-0.222520934)
            var ctf4_r0 = ctg0_r4 + cttw_4_1r
            var ctf4_i0 = ctg0_i4 + cttw_4_1i
            var ctf4_r1 = ctg0_r4 - cttw_4_1r
            var ctf4_i1 = ctg0_i4 - cttw_4_1i
            var or4 = ctf4_r0
            var oi4 = ctf4_i0
            or4 *= Float32(0.0714285714)
            oi4 *= Float32(0.0714285714)
            p.output_real_base.store(out_batch_base + 4, or4[0])
            p.output_imag_base.store(out_batch_base + 4, oi4[0])

            var or11 = ctf4_r1
            var oi11 = ctf4_i1
            or11 *= Float32(0.0714285714)
            oi11 *= Float32(0.0714285714)
            p.output_real_base.store(out_batch_base + 11, or11[0])
            p.output_imag_base.store(out_batch_base + 11, oi11[0])

            var cttw_5_1r = ctg1_r5 * Float32(-0.623489802) - ctg1_i5 * Float32(0.781831482)
            var cttw_5_1i = ctg1_r5 * Float32(0.781831482) + ctg1_i5 * Float32(-0.623489802)
            var ctf5_r0 = ctg0_r5 + cttw_5_1r
            var ctf5_i0 = ctg0_i5 + cttw_5_1i
            var ctf5_r1 = ctg0_r5 - cttw_5_1r
            var ctf5_i1 = ctg0_i5 - cttw_5_1i
            var or5 = ctf5_r0
            var oi5 = ctf5_i0
            or5 *= Float32(0.0714285714)
            oi5 *= Float32(0.0714285714)
            p.output_real_base.store(out_batch_base + 5, or5[0])
            p.output_imag_base.store(out_batch_base + 5, oi5[0])

            var or12 = ctf5_r1
            var oi12 = ctf5_i1
            or12 *= Float32(0.0714285714)
            oi12 *= Float32(0.0714285714)
            p.output_real_base.store(out_batch_base + 12, or12[0])
            p.output_imag_base.store(out_batch_base + 12, oi12[0])

            var cttw_6_1r = ctg1_r6 * Float32(-0.900968868) - ctg1_i6 * Float32(0.433883739)
            var cttw_6_1i = ctg1_r6 * Float32(0.433883739) + ctg1_i6 * Float32(-0.900968868)
            var ctf6_r0 = ctg0_r6 + cttw_6_1r
            var ctf6_i0 = ctg0_i6 + cttw_6_1i
            var ctf6_r1 = ctg0_r6 - cttw_6_1r
            var ctf6_i1 = ctg0_i6 - cttw_6_1i
            var or6 = ctf6_r0
            var oi6 = ctf6_i0
            or6 *= Float32(0.0714285714)
            oi6 *= Float32(0.0714285714)
            p.output_real_base.store(out_batch_base + 6, or6[0])
            p.output_imag_base.store(out_batch_base + 6, oi6[0])

            var or13 = ctf6_r1
            var oi13 = ctf6_i1
            or13 *= Float32(0.0714285714)
            oi13 *= Float32(0.0714285714)
            p.output_real_base.store(out_batch_base + 13, or13[0])
            p.output_imag_base.store(out_batch_base + 13, oi13[0])



    @staticmethod
    def device_main():
        launch_parallel[FFTVkfftLeaf0.stage_0]()


# ============================================================================
# HOST MAIN -- allocates DRAM buffers/twiddle tables, launches every
# kernel above in order, checks the result against an independent
# reference DFT computed at runtime.
# ============================================================================

def main() raises:
    if FFTVkfftLeaf0.emit_ir_if_asked():
        return

    var total_elems = 42
    var input_real = cxl_alloc[Float32](total_elems)
    var input_imag = cxl_alloc[Float32](total_elems)
    var work0_real = cxl_alloc[Float32](total_elems)
    var work0_imag = cxl_alloc[Float32](total_elems)
    var work1_real = cxl_alloc[Float32](total_elems)
    var work1_imag = cxl_alloc[Float32](total_elems)
    var output_real = cxl_alloc[Float32](total_elems)
    var output_imag = cxl_alloc[Float32](total_elems)
    var ref_real = cxl_alloc[Float32](total_elems)
    var ref_imag = cxl_alloc[Float32](total_elems)

    var looptw0_real = cxl_alloc[Float32](1)
    var looptw0_imag = cxl_alloc[Float32](1)

    var pool0_elems = 24
    var pool0_raw = cxl_alloc[Float32](pool0_elems + 64)
    var pool0_raw_addr = Int(pool0_raw)
    var pool0_addr = (pool0_raw_addr + 255) // 256 * 256
    if pool0_addr % 256 != 0:
        print("[host] pool alignment assertion failed")
        return
    var pool0 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=pool0_addr)

    seed(0)
    for i in range(total_elems):
        input_real[i] = Float32(random_float64(-1.0, 1.0))
        input_imag[i] = Float32(random_float64(-1.0, 1.0))
        work0_real[i] = Float32(0)
        work0_imag[i] = Float32(0)
        work1_real[i] = Float32(0)
        work1_imag[i] = Float32(0)
        output_real[i] = Float32(0)
        output_imag[i] = Float32(0)
        ref_real[i] = Float32(0)
        ref_imag[i] = Float32(0)

    var rc0 = FFTVkfftLeaf0.launch(
        PooledRange.over(pool0, 24),
        FFTVkfftLeaf0Params(input_real, input_imag, output_real, output_imag, looptw0_real, looptw0_imag),
    )
    if rc0 != 0:
        print("[host] recursive FFT stage 0 (FFTVkfftLeaf0) failed, exit", rc0)
        return

    var pi = Float64(3.141592653589793)
    var sign = Float64(1.0)
    for batch in range(3):
        var batch_base = batch * 14
        for k in range(14):
            var acc_r = Float64(0)
            var acc_i = Float64(0)
            for n_ in range(14):
                var angle = sign * 2.0 * pi * Float64(n_) * Float64(k) / Float64(14)
                var c = host_cos(angle)
                var s = host_sin(angle)
                var xr = Float64(input_real[batch_base + n_])
                var xi = Float64(input_imag[batch_base + n_])
                acc_r += xr * c - xi * s
                acc_i += xr * s + xi * c
            acc_r /= Float64(14)
            acc_i /= Float64(14)
            ref_real[batch_base + k] = Float32(acc_r)
            ref_imag[batch_base + k] = Float32(acc_i)

    var tol = Float32(0.001)
    for i in range(42):
        var err_r = output_real[i] - ref_real[i]
        var err_i = output_imag[i] - ref_imag[i]
        if err_r < Float32(0):
            err_r = -err_r
        if err_i < Float32(0):
            err_i = -err_i
        if err_r > tol or err_i > tol:
            print("[host] recursive tiled-transpose FFT mismatch at", i)
            print("  expected:", ref_real[i], ref_imag[i])
            print("  actual:  ", output_real[i], output_imag[i])
            print("  error:   ", err_r, err_i)
            return

    print("[host] recursive tiled-transpose FFT verification passed")
