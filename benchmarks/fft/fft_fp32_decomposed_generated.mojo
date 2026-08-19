from std.sys import size_of
from std.random import random_float64, seed
from std.math import cos as host_cos, sin as host_sin

from m2ndp import VECTOR_WIDTH, NDPTask, PooledRange, global_uthread_id, local_uthread_id, launch_parallel, scratchpad
from m2ndp_host import cxl_alloc

comptime W = VECTOR_WIDTH // size_of[Float32]()
comptime N0 = 16
comptime N1 = 16
comptime N = 256

comptime MAX_UTHREAD_FFTFP32Kernel0 = 16

@fieldwise_init
struct FFTFP32Kernel0Params(Movable):
    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var large_twiddle_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var large_twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]


struct FFTFP32Kernel0(NDPTask):
    comptime Params = FFTFP32Kernel0Params

    @staticmethod
    def stage_0():
        ref p = FFTFP32Kernel0.params[]
        comptime RADIX = 16
        comptime SIMD_ITERS = 1

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel0:
            return
        var in_batch_base = global_uthread_id() * 1
        var out_batch_base = global_uthread_id() * 16

        # ===== stage 0, SIMD batch 0 (valid lanes: 1/8) =====
        var rr0_lane0 = p.input_real_base.load[width=1](in_batch_base + 0)
        var ii0_lane0 = p.input_imag_base.load[width=1](in_batch_base + 0)
        var rr0 = SIMD[DType.float32, W](rr0_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii0 = SIMD[DType.float32, W](ii0_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr1_lane0 = p.input_real_base.load[width=1](in_batch_base + 16)
        var ii1_lane0 = p.input_imag_base.load[width=1](in_batch_base + 16)
        var rr1 = SIMD[DType.float32, W](rr1_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii1 = SIMD[DType.float32, W](ii1_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr2_lane0 = p.input_real_base.load[width=1](in_batch_base + 32)
        var ii2_lane0 = p.input_imag_base.load[width=1](in_batch_base + 32)
        var rr2 = SIMD[DType.float32, W](rr2_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii2 = SIMD[DType.float32, W](ii2_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr3_lane0 = p.input_real_base.load[width=1](in_batch_base + 48)
        var ii3_lane0 = p.input_imag_base.load[width=1](in_batch_base + 48)
        var rr3 = SIMD[DType.float32, W](rr3_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii3 = SIMD[DType.float32, W](ii3_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr4_lane0 = p.input_real_base.load[width=1](in_batch_base + 64)
        var ii4_lane0 = p.input_imag_base.load[width=1](in_batch_base + 64)
        var rr4 = SIMD[DType.float32, W](rr4_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii4 = SIMD[DType.float32, W](ii4_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr5_lane0 = p.input_real_base.load[width=1](in_batch_base + 80)
        var ii5_lane0 = p.input_imag_base.load[width=1](in_batch_base + 80)
        var rr5 = SIMD[DType.float32, W](rr5_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii5 = SIMD[DType.float32, W](ii5_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr6_lane0 = p.input_real_base.load[width=1](in_batch_base + 96)
        var ii6_lane0 = p.input_imag_base.load[width=1](in_batch_base + 96)
        var rr6 = SIMD[DType.float32, W](rr6_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii6 = SIMD[DType.float32, W](ii6_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr7_lane0 = p.input_real_base.load[width=1](in_batch_base + 112)
        var ii7_lane0 = p.input_imag_base.load[width=1](in_batch_base + 112)
        var rr7 = SIMD[DType.float32, W](rr7_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii7 = SIMD[DType.float32, W](ii7_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr8_lane0 = p.input_real_base.load[width=1](in_batch_base + 128)
        var ii8_lane0 = p.input_imag_base.load[width=1](in_batch_base + 128)
        var rr8 = SIMD[DType.float32, W](rr8_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii8 = SIMD[DType.float32, W](ii8_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr9_lane0 = p.input_real_base.load[width=1](in_batch_base + 144)
        var ii9_lane0 = p.input_imag_base.load[width=1](in_batch_base + 144)
        var rr9 = SIMD[DType.float32, W](rr9_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii9 = SIMD[DType.float32, W](ii9_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr10_lane0 = p.input_real_base.load[width=1](in_batch_base + 160)
        var ii10_lane0 = p.input_imag_base.load[width=1](in_batch_base + 160)
        var rr10 = SIMD[DType.float32, W](rr10_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii10 = SIMD[DType.float32, W](ii10_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr11_lane0 = p.input_real_base.load[width=1](in_batch_base + 176)
        var ii11_lane0 = p.input_imag_base.load[width=1](in_batch_base + 176)
        var rr11 = SIMD[DType.float32, W](rr11_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii11 = SIMD[DType.float32, W](ii11_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr12_lane0 = p.input_real_base.load[width=1](in_batch_base + 192)
        var ii12_lane0 = p.input_imag_base.load[width=1](in_batch_base + 192)
        var rr12 = SIMD[DType.float32, W](rr12_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii12 = SIMD[DType.float32, W](ii12_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr13_lane0 = p.input_real_base.load[width=1](in_batch_base + 208)
        var ii13_lane0 = p.input_imag_base.load[width=1](in_batch_base + 208)
        var rr13 = SIMD[DType.float32, W](rr13_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii13 = SIMD[DType.float32, W](ii13_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr14_lane0 = p.input_real_base.load[width=1](in_batch_base + 224)
        var ii14_lane0 = p.input_imag_base.load[width=1](in_batch_base + 224)
        var rr14 = SIMD[DType.float32, W](rr14_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii14 = SIMD[DType.float32, W](ii14_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr15_lane0 = p.input_real_base.load[width=1](in_batch_base + 240)
        var ii15_lane0 = p.input_imag_base.load[width=1](in_batch_base + 240)
        var rr15 = SIMD[DType.float32, W](rr15_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii15 = SIMD[DType.float32, W](ii15_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))

        # fixed radix-16 butterfly
        var ctg0_a0r = rr0 + rr8
        var ctg0_a0i = ii0 + ii8
        var ctg0_a1r = rr0 - rr8
        var ctg0_a1i = ii0 - ii8
        var ctg0_b0r = rr4 + rr12
        var ctg0_b0i = ii4 + ii12
        var ctg0_b1r = rr4 - rr12
        var ctg0_b1i = ii4 - ii12
        var ctg0_r0 = ctg0_a0r + ctg0_b0r
        var ctg0_i0 = ctg0_a0i + ctg0_b0i
        var ctg0_r2 = ctg0_a0r - ctg0_b0r
        var ctg0_i2 = ctg0_a0i - ctg0_b0i
        var ctg0_r1 = ctg0_a1r + ctg0_b1i
        var ctg0_i1 = ctg0_a1i - ctg0_b1r
        var ctg0_r3 = ctg0_a1r - ctg0_b1i
        var ctg0_i3 = ctg0_a1i + ctg0_b1r
        var ctg1_a0r = rr1 + rr9
        var ctg1_a0i = ii1 + ii9
        var ctg1_a1r = rr1 - rr9
        var ctg1_a1i = ii1 - ii9
        var ctg1_b0r = rr5 + rr13
        var ctg1_b0i = ii5 + ii13
        var ctg1_b1r = rr5 - rr13
        var ctg1_b1i = ii5 - ii13
        var ctg1_r0 = ctg1_a0r + ctg1_b0r
        var ctg1_i0 = ctg1_a0i + ctg1_b0i
        var ctg1_r2 = ctg1_a0r - ctg1_b0r
        var ctg1_i2 = ctg1_a0i - ctg1_b0i
        var ctg1_r1 = ctg1_a1r + ctg1_b1i
        var ctg1_i1 = ctg1_a1i - ctg1_b1r
        var ctg1_r3 = ctg1_a1r - ctg1_b1i
        var ctg1_i3 = ctg1_a1i + ctg1_b1r
        var ctg2_a0r = rr2 + rr10
        var ctg2_a0i = ii2 + ii10
        var ctg2_a1r = rr2 - rr10
        var ctg2_a1i = ii2 - ii10
        var ctg2_b0r = rr6 + rr14
        var ctg2_b0i = ii6 + ii14
        var ctg2_b1r = rr6 - rr14
        var ctg2_b1i = ii6 - ii14
        var ctg2_r0 = ctg2_a0r + ctg2_b0r
        var ctg2_i0 = ctg2_a0i + ctg2_b0i
        var ctg2_r2 = ctg2_a0r - ctg2_b0r
        var ctg2_i2 = ctg2_a0i - ctg2_b0i
        var ctg2_r1 = ctg2_a1r + ctg2_b1i
        var ctg2_i1 = ctg2_a1i - ctg2_b1r
        var ctg2_r3 = ctg2_a1r - ctg2_b1i
        var ctg2_i3 = ctg2_a1i + ctg2_b1r
        var ctg3_a0r = rr3 + rr11
        var ctg3_a0i = ii3 + ii11
        var ctg3_a1r = rr3 - rr11
        var ctg3_a1i = ii3 - ii11
        var ctg3_b0r = rr7 + rr15
        var ctg3_b0i = ii7 + ii15
        var ctg3_b1r = rr7 - rr15
        var ctg3_b1i = ii7 - ii15
        var ctg3_r0 = ctg3_a0r + ctg3_b0r
        var ctg3_i0 = ctg3_a0i + ctg3_b0i
        var ctg3_r2 = ctg3_a0r - ctg3_b0r
        var ctg3_i2 = ctg3_a0i - ctg3_b0i
        var ctg3_r1 = ctg3_a1r + ctg3_b1i
        var ctg3_i1 = ctg3_a1i - ctg3_b1r
        var ctg3_r3 = ctg3_a1r - ctg3_b1i
        var ctg3_i3 = ctg3_a1i + ctg3_b1r
        var ctf0_a0r = ctg0_r0 + ctg2_r0
        var ctf0_a0i = ctg0_i0 + ctg2_i0
        var ctf0_a1r = ctg0_r0 - ctg2_r0
        var ctf0_a1i = ctg0_i0 - ctg2_i0
        var ctf0_b0r = ctg1_r0 + ctg3_r0
        var ctf0_b0i = ctg1_i0 + ctg3_i0
        var ctf0_b1r = ctg1_r0 - ctg3_r0
        var ctf0_b1i = ctg1_i0 - ctg3_i0
        var ctf0_r0 = ctf0_a0r + ctf0_b0r
        var ctf0_i0 = ctf0_a0i + ctf0_b0i
        var ctf0_r2 = ctf0_a0r - ctf0_b0r
        var ctf0_i2 = ctf0_a0i - ctf0_b0i
        var ctf0_r1 = ctf0_a1r + ctf0_b1i
        var ctf0_i1 = ctf0_a1i - ctf0_b1r
        var ctf0_r3 = ctf0_a1r - ctf0_b1i
        var ctf0_i3 = ctf0_a1i + ctf0_b1r
        var or0 = ctf0_r0
        var oi0 = ctf0_i0
        var or4 = ctf0_r1
        var oi4 = ctf0_i1
        var or8 = ctf0_r2
        var oi8 = ctf0_i2
        var or12 = ctf0_r3
        var oi12 = ctf0_i3
        var cttw_1_1r = ctg1_r1 * Float32(0.923879533) - ctg1_i1 * Float32(-0.382683432)
        var cttw_1_1i = ctg1_r1 * Float32(-0.382683432) + ctg1_i1 * Float32(0.923879533)
        var cttw_1_2r = ctg2_r1 * Float32(0.707106781) - ctg2_i1 * Float32(-0.707106781)
        var cttw_1_2i = ctg2_r1 * Float32(-0.707106781) + ctg2_i1 * Float32(0.707106781)
        var cttw_1_3r = ctg3_r1 * Float32(0.382683432) - ctg3_i1 * Float32(-0.923879533)
        var cttw_1_3i = ctg3_r1 * Float32(-0.923879533) + ctg3_i1 * Float32(0.382683432)
        var ctf1_a0r = ctg0_r1 + cttw_1_2r
        var ctf1_a0i = ctg0_i1 + cttw_1_2i
        var ctf1_a1r = ctg0_r1 - cttw_1_2r
        var ctf1_a1i = ctg0_i1 - cttw_1_2i
        var ctf1_b0r = cttw_1_1r + cttw_1_3r
        var ctf1_b0i = cttw_1_1i + cttw_1_3i
        var ctf1_b1r = cttw_1_1r - cttw_1_3r
        var ctf1_b1i = cttw_1_1i - cttw_1_3i
        var ctf1_r0 = ctf1_a0r + ctf1_b0r
        var ctf1_i0 = ctf1_a0i + ctf1_b0i
        var ctf1_r2 = ctf1_a0r - ctf1_b0r
        var ctf1_i2 = ctf1_a0i - ctf1_b0i
        var ctf1_r1 = ctf1_a1r + ctf1_b1i
        var ctf1_i1 = ctf1_a1i - ctf1_b1r
        var ctf1_r3 = ctf1_a1r - ctf1_b1i
        var ctf1_i3 = ctf1_a1i + ctf1_b1r
        var or1 = ctf1_r0
        var oi1 = ctf1_i0
        var or5 = ctf1_r1
        var oi5 = ctf1_i1
        var or9 = ctf1_r2
        var oi9 = ctf1_i2
        var or13 = ctf1_r3
        var oi13 = ctf1_i3
        var cttw_2_1r = ctg1_r2 * Float32(0.707106781) - ctg1_i2 * Float32(-0.707106781)
        var cttw_2_1i = ctg1_r2 * Float32(-0.707106781) + ctg1_i2 * Float32(0.707106781)
        var cttw_2_2r = ctg2_i2
        var cttw_2_2i = -ctg2_r2
        var cttw_2_3r = ctg3_r2 * Float32(-0.707106781) - ctg3_i2 * Float32(-0.707106781)
        var cttw_2_3i = ctg3_r2 * Float32(-0.707106781) + ctg3_i2 * Float32(-0.707106781)
        var ctf2_a0r = ctg0_r2 + cttw_2_2r
        var ctf2_a0i = ctg0_i2 + cttw_2_2i
        var ctf2_a1r = ctg0_r2 - cttw_2_2r
        var ctf2_a1i = ctg0_i2 - cttw_2_2i
        var ctf2_b0r = cttw_2_1r + cttw_2_3r
        var ctf2_b0i = cttw_2_1i + cttw_2_3i
        var ctf2_b1r = cttw_2_1r - cttw_2_3r
        var ctf2_b1i = cttw_2_1i - cttw_2_3i
        var ctf2_r0 = ctf2_a0r + ctf2_b0r
        var ctf2_i0 = ctf2_a0i + ctf2_b0i
        var ctf2_r2 = ctf2_a0r - ctf2_b0r
        var ctf2_i2 = ctf2_a0i - ctf2_b0i
        var ctf2_r1 = ctf2_a1r + ctf2_b1i
        var ctf2_i1 = ctf2_a1i - ctf2_b1r
        var ctf2_r3 = ctf2_a1r - ctf2_b1i
        var ctf2_i3 = ctf2_a1i + ctf2_b1r
        var or2 = ctf2_r0
        var oi2 = ctf2_i0
        var or6 = ctf2_r1
        var oi6 = ctf2_i1
        var or10 = ctf2_r2
        var oi10 = ctf2_i2
        var or14 = ctf2_r3
        var oi14 = ctf2_i3
        var cttw_3_1r = ctg1_r3 * Float32(0.382683432) - ctg1_i3 * Float32(-0.923879533)
        var cttw_3_1i = ctg1_r3 * Float32(-0.923879533) + ctg1_i3 * Float32(0.382683432)
        var cttw_3_2r = ctg2_r3 * Float32(-0.707106781) - ctg2_i3 * Float32(-0.707106781)
        var cttw_3_2i = ctg2_r3 * Float32(-0.707106781) + ctg2_i3 * Float32(-0.707106781)
        var cttw_3_3r = ctg3_r3 * Float32(-0.923879533) - ctg3_i3 * Float32(0.382683432)
        var cttw_3_3i = ctg3_r3 * Float32(0.382683432) + ctg3_i3 * Float32(-0.923879533)
        var ctf3_a0r = ctg0_r3 + cttw_3_2r
        var ctf3_a0i = ctg0_i3 + cttw_3_2i
        var ctf3_a1r = ctg0_r3 - cttw_3_2r
        var ctf3_a1i = ctg0_i3 - cttw_3_2i
        var ctf3_b0r = cttw_3_1r + cttw_3_3r
        var ctf3_b0i = cttw_3_1i + cttw_3_3i
        var ctf3_b1r = cttw_3_1r - cttw_3_3r
        var ctf3_b1i = cttw_3_1i - cttw_3_3i
        var ctf3_r0 = ctf3_a0r + ctf3_b0r
        var ctf3_i0 = ctf3_a0i + ctf3_b0i
        var ctf3_r2 = ctf3_a0r - ctf3_b0r
        var ctf3_i2 = ctf3_a0i - ctf3_b0i
        var ctf3_r1 = ctf3_a1r + ctf3_b1i
        var ctf3_i1 = ctf3_a1i - ctf3_b1r
        var ctf3_r3 = ctf3_a1r - ctf3_b1i
        var ctf3_i3 = ctf3_a1i + ctf3_b1r
        var or3 = ctf3_r0
        var oi3 = ctf3_i0
        var or7 = ctf3_r1
        var oi7 = ctf3_i1
        var or11 = ctf3_r2
        var oi11 = ctf3_i2
        var or15 = ctf3_r3
        var oi15 = ctf3_i3

        var ltwr0_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 0)
        var ltwi0_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 0)
        var ltwr0 = SIMD[DType.float32, W](ltwr0_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi0 = SIMD[DType.float32, W](ltwi0_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr0 = or0 * ltwr0 - oi0 * ltwi0
        var lti0 = or0 * ltwi0 + oi0 * ltwr0
        or0 = ltr0
        oi0 = lti0
        p.output_real_base.store(out_batch_base + 0, or0[0])
        p.output_imag_base.store(out_batch_base + 0, oi0[0])

        var ltwr1_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 1)
        var ltwi1_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 1)
        var ltwr1 = SIMD[DType.float32, W](ltwr1_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi1 = SIMD[DType.float32, W](ltwi1_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr1 = or1 * ltwr1 - oi1 * ltwi1
        var lti1 = or1 * ltwi1 + oi1 * ltwr1
        or1 = ltr1
        oi1 = lti1
        p.output_real_base.store(out_batch_base + 1, or1[0])
        p.output_imag_base.store(out_batch_base + 1, oi1[0])

        var ltwr2_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 2)
        var ltwi2_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 2)
        var ltwr2 = SIMD[DType.float32, W](ltwr2_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi2 = SIMD[DType.float32, W](ltwi2_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr2 = or2 * ltwr2 - oi2 * ltwi2
        var lti2 = or2 * ltwi2 + oi2 * ltwr2
        or2 = ltr2
        oi2 = lti2
        p.output_real_base.store(out_batch_base + 2, or2[0])
        p.output_imag_base.store(out_batch_base + 2, oi2[0])

        var ltwr3_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 3)
        var ltwi3_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 3)
        var ltwr3 = SIMD[DType.float32, W](ltwr3_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi3 = SIMD[DType.float32, W](ltwi3_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr3 = or3 * ltwr3 - oi3 * ltwi3
        var lti3 = or3 * ltwi3 + oi3 * ltwr3
        or3 = ltr3
        oi3 = lti3
        p.output_real_base.store(out_batch_base + 3, or3[0])
        p.output_imag_base.store(out_batch_base + 3, oi3[0])

        var ltwr4_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 4)
        var ltwi4_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 4)
        var ltwr4 = SIMD[DType.float32, W](ltwr4_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi4 = SIMD[DType.float32, W](ltwi4_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr4 = or4 * ltwr4 - oi4 * ltwi4
        var lti4 = or4 * ltwi4 + oi4 * ltwr4
        or4 = ltr4
        oi4 = lti4
        p.output_real_base.store(out_batch_base + 4, or4[0])
        p.output_imag_base.store(out_batch_base + 4, oi4[0])

        var ltwr5_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 5)
        var ltwi5_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 5)
        var ltwr5 = SIMD[DType.float32, W](ltwr5_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi5 = SIMD[DType.float32, W](ltwi5_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr5 = or5 * ltwr5 - oi5 * ltwi5
        var lti5 = or5 * ltwi5 + oi5 * ltwr5
        or5 = ltr5
        oi5 = lti5
        p.output_real_base.store(out_batch_base + 5, or5[0])
        p.output_imag_base.store(out_batch_base + 5, oi5[0])

        var ltwr6_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 6)
        var ltwi6_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 6)
        var ltwr6 = SIMD[DType.float32, W](ltwr6_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi6 = SIMD[DType.float32, W](ltwi6_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr6 = or6 * ltwr6 - oi6 * ltwi6
        var lti6 = or6 * ltwi6 + oi6 * ltwr6
        or6 = ltr6
        oi6 = lti6
        p.output_real_base.store(out_batch_base + 6, or6[0])
        p.output_imag_base.store(out_batch_base + 6, oi6[0])

        var ltwr7_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 7)
        var ltwi7_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 7)
        var ltwr7 = SIMD[DType.float32, W](ltwr7_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi7 = SIMD[DType.float32, W](ltwi7_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr7 = or7 * ltwr7 - oi7 * ltwi7
        var lti7 = or7 * ltwi7 + oi7 * ltwr7
        or7 = ltr7
        oi7 = lti7
        p.output_real_base.store(out_batch_base + 7, or7[0])
        p.output_imag_base.store(out_batch_base + 7, oi7[0])

        var ltwr8_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 8)
        var ltwi8_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 8)
        var ltwr8 = SIMD[DType.float32, W](ltwr8_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi8 = SIMD[DType.float32, W](ltwi8_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr8 = or8 * ltwr8 - oi8 * ltwi8
        var lti8 = or8 * ltwi8 + oi8 * ltwr8
        or8 = ltr8
        oi8 = lti8
        p.output_real_base.store(out_batch_base + 8, or8[0])
        p.output_imag_base.store(out_batch_base + 8, oi8[0])

        var ltwr9_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 9)
        var ltwi9_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 9)
        var ltwr9 = SIMD[DType.float32, W](ltwr9_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi9 = SIMD[DType.float32, W](ltwi9_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr9 = or9 * ltwr9 - oi9 * ltwi9
        var lti9 = or9 * ltwi9 + oi9 * ltwr9
        or9 = ltr9
        oi9 = lti9
        p.output_real_base.store(out_batch_base + 9, or9[0])
        p.output_imag_base.store(out_batch_base + 9, oi9[0])

        var ltwr10_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 10)
        var ltwi10_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 10)
        var ltwr10 = SIMD[DType.float32, W](ltwr10_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi10 = SIMD[DType.float32, W](ltwi10_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr10 = or10 * ltwr10 - oi10 * ltwi10
        var lti10 = or10 * ltwi10 + oi10 * ltwr10
        or10 = ltr10
        oi10 = lti10
        p.output_real_base.store(out_batch_base + 10, or10[0])
        p.output_imag_base.store(out_batch_base + 10, oi10[0])

        var ltwr11_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 11)
        var ltwi11_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 11)
        var ltwr11 = SIMD[DType.float32, W](ltwr11_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi11 = SIMD[DType.float32, W](ltwi11_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr11 = or11 * ltwr11 - oi11 * ltwi11
        var lti11 = or11 * ltwi11 + oi11 * ltwr11
        or11 = ltr11
        oi11 = lti11
        p.output_real_base.store(out_batch_base + 11, or11[0])
        p.output_imag_base.store(out_batch_base + 11, oi11[0])

        var ltwr12_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 12)
        var ltwi12_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 12)
        var ltwr12 = SIMD[DType.float32, W](ltwr12_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi12 = SIMD[DType.float32, W](ltwi12_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr12 = or12 * ltwr12 - oi12 * ltwi12
        var lti12 = or12 * ltwi12 + oi12 * ltwr12
        or12 = ltr12
        oi12 = lti12
        p.output_real_base.store(out_batch_base + 12, or12[0])
        p.output_imag_base.store(out_batch_base + 12, oi12[0])

        var ltwr13_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 13)
        var ltwi13_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 13)
        var ltwr13 = SIMD[DType.float32, W](ltwr13_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi13 = SIMD[DType.float32, W](ltwi13_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr13 = or13 * ltwr13 - oi13 * ltwi13
        var lti13 = or13 * ltwi13 + oi13 * ltwr13
        or13 = ltr13
        oi13 = lti13
        p.output_real_base.store(out_batch_base + 13, or13[0])
        p.output_imag_base.store(out_batch_base + 13, oi13[0])

        var ltwr14_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 14)
        var ltwi14_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 14)
        var ltwr14 = SIMD[DType.float32, W](ltwr14_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi14 = SIMD[DType.float32, W](ltwi14_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr14 = or14 * ltwr14 - oi14 * ltwi14
        var lti14 = or14 * ltwi14 + oi14 * ltwr14
        or14 = ltr14
        oi14 = lti14
        p.output_real_base.store(out_batch_base + 14, or14[0])
        p.output_imag_base.store(out_batch_base + 14, oi14[0])

        var ltwr15_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 15)
        var ltwi15_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 15)
        var ltwr15 = SIMD[DType.float32, W](ltwr15_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi15 = SIMD[DType.float32, W](ltwi15_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr15 = or15 * ltwr15 - oi15 * ltwi15
        var lti15 = or15 * ltwi15 + oi15 * ltwr15
        or15 = ltr15
        oi15 = lti15
        p.output_real_base.store(out_batch_base + 15, or15[0])
        p.output_imag_base.store(out_batch_base + 15, oi15[0])

    @staticmethod
    def device_main():
        launch_parallel[FFTFP32Kernel0.stage_0]()


comptime MAX_UTHREAD_FFTFP32Kernel1 = 16

@fieldwise_init
struct FFTFP32Kernel1Params(Movable):
    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]


struct FFTFP32Kernel1(NDPTask):
    comptime Params = FFTFP32Kernel1Params

    @staticmethod
    def stage_0():
        ref p = FFTFP32Kernel1.params[]
        comptime RADIX = 16
        comptime SIMD_ITERS = 1

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel1:
            return
        var in_batch_base = global_uthread_id() * 16
        var out_batch_base = global_uthread_id() * 1

        # ===== stage 0, SIMD batch 0 (valid lanes: 1/8) =====
        var rr0_lane0 = p.input_real_base.load[width=1](in_batch_base + 0)
        var ii0_lane0 = p.input_imag_base.load[width=1](in_batch_base + 0)
        var rr0 = SIMD[DType.float32, W](rr0_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii0 = SIMD[DType.float32, W](ii0_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr1_lane0 = p.input_real_base.load[width=1](in_batch_base + 1)
        var ii1_lane0 = p.input_imag_base.load[width=1](in_batch_base + 1)
        var rr1 = SIMD[DType.float32, W](rr1_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii1 = SIMD[DType.float32, W](ii1_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr2_lane0 = p.input_real_base.load[width=1](in_batch_base + 2)
        var ii2_lane0 = p.input_imag_base.load[width=1](in_batch_base + 2)
        var rr2 = SIMD[DType.float32, W](rr2_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii2 = SIMD[DType.float32, W](ii2_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr3_lane0 = p.input_real_base.load[width=1](in_batch_base + 3)
        var ii3_lane0 = p.input_imag_base.load[width=1](in_batch_base + 3)
        var rr3 = SIMD[DType.float32, W](rr3_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii3 = SIMD[DType.float32, W](ii3_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr4_lane0 = p.input_real_base.load[width=1](in_batch_base + 4)
        var ii4_lane0 = p.input_imag_base.load[width=1](in_batch_base + 4)
        var rr4 = SIMD[DType.float32, W](rr4_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii4 = SIMD[DType.float32, W](ii4_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr5_lane0 = p.input_real_base.load[width=1](in_batch_base + 5)
        var ii5_lane0 = p.input_imag_base.load[width=1](in_batch_base + 5)
        var rr5 = SIMD[DType.float32, W](rr5_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii5 = SIMD[DType.float32, W](ii5_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr6_lane0 = p.input_real_base.load[width=1](in_batch_base + 6)
        var ii6_lane0 = p.input_imag_base.load[width=1](in_batch_base + 6)
        var rr6 = SIMD[DType.float32, W](rr6_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii6 = SIMD[DType.float32, W](ii6_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr7_lane0 = p.input_real_base.load[width=1](in_batch_base + 7)
        var ii7_lane0 = p.input_imag_base.load[width=1](in_batch_base + 7)
        var rr7 = SIMD[DType.float32, W](rr7_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii7 = SIMD[DType.float32, W](ii7_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr8_lane0 = p.input_real_base.load[width=1](in_batch_base + 8)
        var ii8_lane0 = p.input_imag_base.load[width=1](in_batch_base + 8)
        var rr8 = SIMD[DType.float32, W](rr8_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii8 = SIMD[DType.float32, W](ii8_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr9_lane0 = p.input_real_base.load[width=1](in_batch_base + 9)
        var ii9_lane0 = p.input_imag_base.load[width=1](in_batch_base + 9)
        var rr9 = SIMD[DType.float32, W](rr9_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii9 = SIMD[DType.float32, W](ii9_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr10_lane0 = p.input_real_base.load[width=1](in_batch_base + 10)
        var ii10_lane0 = p.input_imag_base.load[width=1](in_batch_base + 10)
        var rr10 = SIMD[DType.float32, W](rr10_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii10 = SIMD[DType.float32, W](ii10_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr11_lane0 = p.input_real_base.load[width=1](in_batch_base + 11)
        var ii11_lane0 = p.input_imag_base.load[width=1](in_batch_base + 11)
        var rr11 = SIMD[DType.float32, W](rr11_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii11 = SIMD[DType.float32, W](ii11_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr12_lane0 = p.input_real_base.load[width=1](in_batch_base + 12)
        var ii12_lane0 = p.input_imag_base.load[width=1](in_batch_base + 12)
        var rr12 = SIMD[DType.float32, W](rr12_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii12 = SIMD[DType.float32, W](ii12_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr13_lane0 = p.input_real_base.load[width=1](in_batch_base + 13)
        var ii13_lane0 = p.input_imag_base.load[width=1](in_batch_base + 13)
        var rr13 = SIMD[DType.float32, W](rr13_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii13 = SIMD[DType.float32, W](ii13_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr14_lane0 = p.input_real_base.load[width=1](in_batch_base + 14)
        var ii14_lane0 = p.input_imag_base.load[width=1](in_batch_base + 14)
        var rr14 = SIMD[DType.float32, W](rr14_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii14 = SIMD[DType.float32, W](ii14_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr15_lane0 = p.input_real_base.load[width=1](in_batch_base + 15)
        var ii15_lane0 = p.input_imag_base.load[width=1](in_batch_base + 15)
        var rr15 = SIMD[DType.float32, W](rr15_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii15 = SIMD[DType.float32, W](ii15_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))

        # fixed radix-16 butterfly
        var ctg0_a0r = rr0 + rr8
        var ctg0_a0i = ii0 + ii8
        var ctg0_a1r = rr0 - rr8
        var ctg0_a1i = ii0 - ii8
        var ctg0_b0r = rr4 + rr12
        var ctg0_b0i = ii4 + ii12
        var ctg0_b1r = rr4 - rr12
        var ctg0_b1i = ii4 - ii12
        var ctg0_r0 = ctg0_a0r + ctg0_b0r
        var ctg0_i0 = ctg0_a0i + ctg0_b0i
        var ctg0_r2 = ctg0_a0r - ctg0_b0r
        var ctg0_i2 = ctg0_a0i - ctg0_b0i
        var ctg0_r1 = ctg0_a1r + ctg0_b1i
        var ctg0_i1 = ctg0_a1i - ctg0_b1r
        var ctg0_r3 = ctg0_a1r - ctg0_b1i
        var ctg0_i3 = ctg0_a1i + ctg0_b1r
        var ctg1_a0r = rr1 + rr9
        var ctg1_a0i = ii1 + ii9
        var ctg1_a1r = rr1 - rr9
        var ctg1_a1i = ii1 - ii9
        var ctg1_b0r = rr5 + rr13
        var ctg1_b0i = ii5 + ii13
        var ctg1_b1r = rr5 - rr13
        var ctg1_b1i = ii5 - ii13
        var ctg1_r0 = ctg1_a0r + ctg1_b0r
        var ctg1_i0 = ctg1_a0i + ctg1_b0i
        var ctg1_r2 = ctg1_a0r - ctg1_b0r
        var ctg1_i2 = ctg1_a0i - ctg1_b0i
        var ctg1_r1 = ctg1_a1r + ctg1_b1i
        var ctg1_i1 = ctg1_a1i - ctg1_b1r
        var ctg1_r3 = ctg1_a1r - ctg1_b1i
        var ctg1_i3 = ctg1_a1i + ctg1_b1r
        var ctg2_a0r = rr2 + rr10
        var ctg2_a0i = ii2 + ii10
        var ctg2_a1r = rr2 - rr10
        var ctg2_a1i = ii2 - ii10
        var ctg2_b0r = rr6 + rr14
        var ctg2_b0i = ii6 + ii14
        var ctg2_b1r = rr6 - rr14
        var ctg2_b1i = ii6 - ii14
        var ctg2_r0 = ctg2_a0r + ctg2_b0r
        var ctg2_i0 = ctg2_a0i + ctg2_b0i
        var ctg2_r2 = ctg2_a0r - ctg2_b0r
        var ctg2_i2 = ctg2_a0i - ctg2_b0i
        var ctg2_r1 = ctg2_a1r + ctg2_b1i
        var ctg2_i1 = ctg2_a1i - ctg2_b1r
        var ctg2_r3 = ctg2_a1r - ctg2_b1i
        var ctg2_i3 = ctg2_a1i + ctg2_b1r
        var ctg3_a0r = rr3 + rr11
        var ctg3_a0i = ii3 + ii11
        var ctg3_a1r = rr3 - rr11
        var ctg3_a1i = ii3 - ii11
        var ctg3_b0r = rr7 + rr15
        var ctg3_b0i = ii7 + ii15
        var ctg3_b1r = rr7 - rr15
        var ctg3_b1i = ii7 - ii15
        var ctg3_r0 = ctg3_a0r + ctg3_b0r
        var ctg3_i0 = ctg3_a0i + ctg3_b0i
        var ctg3_r2 = ctg3_a0r - ctg3_b0r
        var ctg3_i2 = ctg3_a0i - ctg3_b0i
        var ctg3_r1 = ctg3_a1r + ctg3_b1i
        var ctg3_i1 = ctg3_a1i - ctg3_b1r
        var ctg3_r3 = ctg3_a1r - ctg3_b1i
        var ctg3_i3 = ctg3_a1i + ctg3_b1r
        var ctf0_a0r = ctg0_r0 + ctg2_r0
        var ctf0_a0i = ctg0_i0 + ctg2_i0
        var ctf0_a1r = ctg0_r0 - ctg2_r0
        var ctf0_a1i = ctg0_i0 - ctg2_i0
        var ctf0_b0r = ctg1_r0 + ctg3_r0
        var ctf0_b0i = ctg1_i0 + ctg3_i0
        var ctf0_b1r = ctg1_r0 - ctg3_r0
        var ctf0_b1i = ctg1_i0 - ctg3_i0
        var ctf0_r0 = ctf0_a0r + ctf0_b0r
        var ctf0_i0 = ctf0_a0i + ctf0_b0i
        var ctf0_r2 = ctf0_a0r - ctf0_b0r
        var ctf0_i2 = ctf0_a0i - ctf0_b0i
        var ctf0_r1 = ctf0_a1r + ctf0_b1i
        var ctf0_i1 = ctf0_a1i - ctf0_b1r
        var ctf0_r3 = ctf0_a1r - ctf0_b1i
        var ctf0_i3 = ctf0_a1i + ctf0_b1r
        var or0 = ctf0_r0
        var oi0 = ctf0_i0
        var or4 = ctf0_r1
        var oi4 = ctf0_i1
        var or8 = ctf0_r2
        var oi8 = ctf0_i2
        var or12 = ctf0_r3
        var oi12 = ctf0_i3
        var cttw_1_1r = ctg1_r1 * Float32(0.923879533) - ctg1_i1 * Float32(-0.382683432)
        var cttw_1_1i = ctg1_r1 * Float32(-0.382683432) + ctg1_i1 * Float32(0.923879533)
        var cttw_1_2r = ctg2_r1 * Float32(0.707106781) - ctg2_i1 * Float32(-0.707106781)
        var cttw_1_2i = ctg2_r1 * Float32(-0.707106781) + ctg2_i1 * Float32(0.707106781)
        var cttw_1_3r = ctg3_r1 * Float32(0.382683432) - ctg3_i1 * Float32(-0.923879533)
        var cttw_1_3i = ctg3_r1 * Float32(-0.923879533) + ctg3_i1 * Float32(0.382683432)
        var ctf1_a0r = ctg0_r1 + cttw_1_2r
        var ctf1_a0i = ctg0_i1 + cttw_1_2i
        var ctf1_a1r = ctg0_r1 - cttw_1_2r
        var ctf1_a1i = ctg0_i1 - cttw_1_2i
        var ctf1_b0r = cttw_1_1r + cttw_1_3r
        var ctf1_b0i = cttw_1_1i + cttw_1_3i
        var ctf1_b1r = cttw_1_1r - cttw_1_3r
        var ctf1_b1i = cttw_1_1i - cttw_1_3i
        var ctf1_r0 = ctf1_a0r + ctf1_b0r
        var ctf1_i0 = ctf1_a0i + ctf1_b0i
        var ctf1_r2 = ctf1_a0r - ctf1_b0r
        var ctf1_i2 = ctf1_a0i - ctf1_b0i
        var ctf1_r1 = ctf1_a1r + ctf1_b1i
        var ctf1_i1 = ctf1_a1i - ctf1_b1r
        var ctf1_r3 = ctf1_a1r - ctf1_b1i
        var ctf1_i3 = ctf1_a1i + ctf1_b1r
        var or1 = ctf1_r0
        var oi1 = ctf1_i0
        var or5 = ctf1_r1
        var oi5 = ctf1_i1
        var or9 = ctf1_r2
        var oi9 = ctf1_i2
        var or13 = ctf1_r3
        var oi13 = ctf1_i3
        var cttw_2_1r = ctg1_r2 * Float32(0.707106781) - ctg1_i2 * Float32(-0.707106781)
        var cttw_2_1i = ctg1_r2 * Float32(-0.707106781) + ctg1_i2 * Float32(0.707106781)
        var cttw_2_2r = ctg2_i2
        var cttw_2_2i = -ctg2_r2
        var cttw_2_3r = ctg3_r2 * Float32(-0.707106781) - ctg3_i2 * Float32(-0.707106781)
        var cttw_2_3i = ctg3_r2 * Float32(-0.707106781) + ctg3_i2 * Float32(-0.707106781)
        var ctf2_a0r = ctg0_r2 + cttw_2_2r
        var ctf2_a0i = ctg0_i2 + cttw_2_2i
        var ctf2_a1r = ctg0_r2 - cttw_2_2r
        var ctf2_a1i = ctg0_i2 - cttw_2_2i
        var ctf2_b0r = cttw_2_1r + cttw_2_3r
        var ctf2_b0i = cttw_2_1i + cttw_2_3i
        var ctf2_b1r = cttw_2_1r - cttw_2_3r
        var ctf2_b1i = cttw_2_1i - cttw_2_3i
        var ctf2_r0 = ctf2_a0r + ctf2_b0r
        var ctf2_i0 = ctf2_a0i + ctf2_b0i
        var ctf2_r2 = ctf2_a0r - ctf2_b0r
        var ctf2_i2 = ctf2_a0i - ctf2_b0i
        var ctf2_r1 = ctf2_a1r + ctf2_b1i
        var ctf2_i1 = ctf2_a1i - ctf2_b1r
        var ctf2_r3 = ctf2_a1r - ctf2_b1i
        var ctf2_i3 = ctf2_a1i + ctf2_b1r
        var or2 = ctf2_r0
        var oi2 = ctf2_i0
        var or6 = ctf2_r1
        var oi6 = ctf2_i1
        var or10 = ctf2_r2
        var oi10 = ctf2_i2
        var or14 = ctf2_r3
        var oi14 = ctf2_i3
        var cttw_3_1r = ctg1_r3 * Float32(0.382683432) - ctg1_i3 * Float32(-0.923879533)
        var cttw_3_1i = ctg1_r3 * Float32(-0.923879533) + ctg1_i3 * Float32(0.382683432)
        var cttw_3_2r = ctg2_r3 * Float32(-0.707106781) - ctg2_i3 * Float32(-0.707106781)
        var cttw_3_2i = ctg2_r3 * Float32(-0.707106781) + ctg2_i3 * Float32(-0.707106781)
        var cttw_3_3r = ctg3_r3 * Float32(-0.923879533) - ctg3_i3 * Float32(0.382683432)
        var cttw_3_3i = ctg3_r3 * Float32(0.382683432) + ctg3_i3 * Float32(-0.923879533)
        var ctf3_a0r = ctg0_r3 + cttw_3_2r
        var ctf3_a0i = ctg0_i3 + cttw_3_2i
        var ctf3_a1r = ctg0_r3 - cttw_3_2r
        var ctf3_a1i = ctg0_i3 - cttw_3_2i
        var ctf3_b0r = cttw_3_1r + cttw_3_3r
        var ctf3_b0i = cttw_3_1i + cttw_3_3i
        var ctf3_b1r = cttw_3_1r - cttw_3_3r
        var ctf3_b1i = cttw_3_1i - cttw_3_3i
        var ctf3_r0 = ctf3_a0r + ctf3_b0r
        var ctf3_i0 = ctf3_a0i + ctf3_b0i
        var ctf3_r2 = ctf3_a0r - ctf3_b0r
        var ctf3_i2 = ctf3_a0i - ctf3_b0i
        var ctf3_r1 = ctf3_a1r + ctf3_b1i
        var ctf3_i1 = ctf3_a1i - ctf3_b1r
        var ctf3_r3 = ctf3_a1r - ctf3_b1i
        var ctf3_i3 = ctf3_a1i + ctf3_b1r
        var or3 = ctf3_r0
        var oi3 = ctf3_i0
        var or7 = ctf3_r1
        var oi7 = ctf3_i1
        var or11 = ctf3_r2
        var oi11 = ctf3_i2
        var or15 = ctf3_r3
        var oi15 = ctf3_i3

        p.output_real_base.store(out_batch_base + 0, or0[0])
        p.output_imag_base.store(out_batch_base + 0, oi0[0])

        p.output_real_base.store(out_batch_base + 16, or1[0])
        p.output_imag_base.store(out_batch_base + 16, oi1[0])

        p.output_real_base.store(out_batch_base + 32, or2[0])
        p.output_imag_base.store(out_batch_base + 32, oi2[0])

        p.output_real_base.store(out_batch_base + 48, or3[0])
        p.output_imag_base.store(out_batch_base + 48, oi3[0])

        p.output_real_base.store(out_batch_base + 64, or4[0])
        p.output_imag_base.store(out_batch_base + 64, oi4[0])

        p.output_real_base.store(out_batch_base + 80, or5[0])
        p.output_imag_base.store(out_batch_base + 80, oi5[0])

        p.output_real_base.store(out_batch_base + 96, or6[0])
        p.output_imag_base.store(out_batch_base + 96, oi6[0])

        p.output_real_base.store(out_batch_base + 112, or7[0])
        p.output_imag_base.store(out_batch_base + 112, oi7[0])

        p.output_real_base.store(out_batch_base + 128, or8[0])
        p.output_imag_base.store(out_batch_base + 128, oi8[0])

        p.output_real_base.store(out_batch_base + 144, or9[0])
        p.output_imag_base.store(out_batch_base + 144, oi9[0])

        p.output_real_base.store(out_batch_base + 160, or10[0])
        p.output_imag_base.store(out_batch_base + 160, oi10[0])

        p.output_real_base.store(out_batch_base + 176, or11[0])
        p.output_imag_base.store(out_batch_base + 176, oi11[0])

        p.output_real_base.store(out_batch_base + 192, or12[0])
        p.output_imag_base.store(out_batch_base + 192, oi12[0])

        p.output_real_base.store(out_batch_base + 208, or13[0])
        p.output_imag_base.store(out_batch_base + 208, oi13[0])

        p.output_real_base.store(out_batch_base + 224, or14[0])
        p.output_imag_base.store(out_batch_base + 224, oi14[0])

        p.output_real_base.store(out_batch_base + 240, or15[0])
        p.output_imag_base.store(out_batch_base + 240, oi15[0])

    @staticmethod
    def device_main():
        launch_parallel[FFTFP32Kernel1.stage_0]()


def main() raises:
    if FFTFP32Kernel0.emit_ir_if_asked():
        return

    var n = 256
    var input_real = cxl_alloc[Float32](n)
    var input_imag = cxl_alloc[Float32](n)
    var mid_real = cxl_alloc[Float32](n)
    var mid_imag = cxl_alloc[Float32](n)
    var output_real = cxl_alloc[Float32](n)
    var output_imag = cxl_alloc[Float32](n)
    var ref_real = cxl_alloc[Float32](n)
    var ref_imag = cxl_alloc[Float32](n)

    var large_twiddle_real = cxl_alloc[Float32](n)
    var large_twiddle_imag = cxl_alloc[Float32](n)
    var pi = Float64(3.141592653589793)
    var lt_sign = Float64(-1.0)
    var r = 0
    while r < 16:
        var c1 = 0
        while c1 < 16:
            var angle = lt_sign * 2.0 * pi * Float64(r) * Float64(c1) / Float64(256)
            large_twiddle_real[r * 16 + c1] = Float32(host_cos(angle))
            large_twiddle_imag[r * 16 + c1] = Float32(host_sin(angle))
            c1 += 1
        r += 1

    var pool0_elems = 128
    var pool1_elems = 128
    var pool0 = cxl_alloc[Float32](pool0_elems)
    var pool1 = cxl_alloc[Float32](pool1_elems)

    seed(0)
    for i in range(n):
        input_real[i] = Float32(random_float64(-1.0, 1.0))
        input_imag[i] = Float32(random_float64(-1.0, 1.0))
        mid_real[i] = Float32(0)
        mid_imag[i] = Float32(0)
        output_real[i] = Float32(0)
        output_imag[i] = Float32(0)
        ref_real[i] = Float32(0)
        ref_imag[i] = Float32(0)

    var rc0 = FFTFP32Kernel0.launch(
        PooledRange.over(pool0, pool0_elems),
        FFTFP32Kernel0Params(input_real, input_imag, mid_real, mid_imag, large_twiddle_real, large_twiddle_imag),
    )
    if rc0 != 0:
        print("[host] FFT kernel0 failed, exit", rc0)
        return

    var rc1 = FFTFP32Kernel1.launch(
        PooledRange.over(pool1, pool1_elems),
        FFTFP32Kernel1Params(mid_real, mid_imag, output_real, output_imag),
    )
    if rc1 != 0:
        print("[host] FFT kernel1 failed, exit", rc1)
        return

    var pi = Float64(3.141592653589793)
    var sign = Float64(-1.0)
    for batch in range(1):
        var batch_base = batch * 256
        for k in range(256):
            var acc_r = Float64(0)
            var acc_i = Float64(0)
            for n_ in range(256):
                var angle = sign * 2.0 * pi * Float64(n_) * Float64(k) / Float64(256)
                var c = host_cos(angle)
                var s = host_sin(angle)
                var xr = Float64(input_real[batch_base + n_])
                var xi = Float64(input_imag[batch_base + n_])
                acc_r += xr * c - xi * s
                acc_i += xr * s + xi * c
            ref_real[batch_base + k] = Float32(acc_r)
            ref_imag[batch_base + k] = Float32(acc_i)

    var tol = Float32(0.001)
    for i in range(256):
        var err_r = output_real[i] - ref_real[i]
        var err_i = output_imag[i] - ref_imag[i]
        if err_r < Float32(0):
            err_r = -err_r
        if err_i < Float32(0):
            err_i = -err_i
        if err_r > tol or err_i > tol:
            print("[host] decomposed FFT mismatch at", i)
            print("  expected:", ref_real[i], ref_imag[i])
            print("  actual:  ", output_real[i], output_imag[i])
            print("  error:   ", err_r, err_i)
            return

    print("[host] decomposed FFT verification passed")
