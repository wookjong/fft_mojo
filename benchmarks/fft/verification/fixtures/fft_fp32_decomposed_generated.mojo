from std.sys import size_of
from std.random import random_float64, seed
from std.math import cos as host_cos, sin as host_sin

from m2ndp import VECTOR_WIDTH, NDPTask, PooledRange, global_uthread_id, local_uthread_id, group_id, num_groups, launch_parallel, scratchpad
from m2ndp_host import cxl_alloc

comptime W = VECTOR_WIDTH // size_of[Float32]()
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

    comptime buf = scratchpad[512, Float32, name="fftfp32kernel0_buf"]()

    @staticmethod
    def stage_0():
        ref p = FFTFP32Kernel0.params[]
        comptime RADIX = 4
        comptime SIMD_ITERS = 1

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel0:
            return
        var spad_base = local_id * 32
        var in_batch_base = global_uthread_id() * 1

        # ===== stage 0, SIMD batch 0 (valid lanes: 4/4) =====
        var rr0_lane0 = p.input_real_base.load[width=1](in_batch_base + 0)
        var ii0_lane0 = p.input_imag_base.load[width=1](in_batch_base + 0)
        var rr0_lane1 = p.input_real_base.load[width=1](in_batch_base + 16)
        var ii0_lane1 = p.input_imag_base.load[width=1](in_batch_base + 16)
        var rr0_lane2 = p.input_real_base.load[width=1](in_batch_base + 32)
        var ii0_lane2 = p.input_imag_base.load[width=1](in_batch_base + 32)
        var rr0_lane3 = p.input_real_base.load[width=1](in_batch_base + 48)
        var ii0_lane3 = p.input_imag_base.load[width=1](in_batch_base + 48)
        var rr0 = SIMD[DType.float32, 4](rr0_lane0[0], rr0_lane1[0], rr0_lane2[0], rr0_lane3[0])
        var ii0 = SIMD[DType.float32, 4](ii0_lane0[0], ii0_lane1[0], ii0_lane2[0], ii0_lane3[0])
        var rr1_lane0 = p.input_real_base.load[width=1](in_batch_base + 64)
        var ii1_lane0 = p.input_imag_base.load[width=1](in_batch_base + 64)
        var rr1_lane1 = p.input_real_base.load[width=1](in_batch_base + 80)
        var ii1_lane1 = p.input_imag_base.load[width=1](in_batch_base + 80)
        var rr1_lane2 = p.input_real_base.load[width=1](in_batch_base + 96)
        var ii1_lane2 = p.input_imag_base.load[width=1](in_batch_base + 96)
        var rr1_lane3 = p.input_real_base.load[width=1](in_batch_base + 112)
        var ii1_lane3 = p.input_imag_base.load[width=1](in_batch_base + 112)
        var rr1 = SIMD[DType.float32, 4](rr1_lane0[0], rr1_lane1[0], rr1_lane2[0], rr1_lane3[0])
        var ii1 = SIMD[DType.float32, 4](ii1_lane0[0], ii1_lane1[0], ii1_lane2[0], ii1_lane3[0])
        var rr2_lane0 = p.input_real_base.load[width=1](in_batch_base + 128)
        var ii2_lane0 = p.input_imag_base.load[width=1](in_batch_base + 128)
        var rr2_lane1 = p.input_real_base.load[width=1](in_batch_base + 144)
        var ii2_lane1 = p.input_imag_base.load[width=1](in_batch_base + 144)
        var rr2_lane2 = p.input_real_base.load[width=1](in_batch_base + 160)
        var ii2_lane2 = p.input_imag_base.load[width=1](in_batch_base + 160)
        var rr2_lane3 = p.input_real_base.load[width=1](in_batch_base + 176)
        var ii2_lane3 = p.input_imag_base.load[width=1](in_batch_base + 176)
        var rr2 = SIMD[DType.float32, 4](rr2_lane0[0], rr2_lane1[0], rr2_lane2[0], rr2_lane3[0])
        var ii2 = SIMD[DType.float32, 4](ii2_lane0[0], ii2_lane1[0], ii2_lane2[0], ii2_lane3[0])
        var rr3_lane0 = p.input_real_base.load[width=1](in_batch_base + 192)
        var ii3_lane0 = p.input_imag_base.load[width=1](in_batch_base + 192)
        var rr3_lane1 = p.input_real_base.load[width=1](in_batch_base + 208)
        var ii3_lane1 = p.input_imag_base.load[width=1](in_batch_base + 208)
        var rr3_lane2 = p.input_real_base.load[width=1](in_batch_base + 224)
        var ii3_lane2 = p.input_imag_base.load[width=1](in_batch_base + 224)
        var rr3_lane3 = p.input_real_base.load[width=1](in_batch_base + 240)
        var ii3_lane3 = p.input_imag_base.load[width=1](in_batch_base + 240)
        var rr3 = SIMD[DType.float32, 4](rr3_lane0[0], rr3_lane1[0], rr3_lane2[0], rr3_lane3[0])
        var ii3 = SIMD[DType.float32, 4](ii3_lane0[0], ii3_lane1[0], ii3_lane2[0], ii3_lane3[0])

        # fixed radix-4 butterfly
        var oa0r = rr0 + rr2
        var oa0i = ii0 + ii2
        var oa1r = rr0 - rr2
        var oa1i = ii0 - ii2
        var ob0r = rr1 + rr3
        var ob0i = ii1 + ii3
        var ob1r = rr1 - rr3
        var ob1i = ii1 - ii3
        var or0 = oa0r + ob0r
        var oi0 = oa0i + ob0i
        var or2 = oa0r - ob0r
        var oi2 = oa0i - ob0i
        var or1 = oa1r + ob1i
        var oi1 = oa1i - ob1r
        var or3 = oa1r - ob1i
        var oi3 = oa1i + ob1r
        FFTFP32Kernel0.buf.store(spad_base + 0, or0[0])
        FFTFP32Kernel0.buf.store(spad_base + 16 + 0, oi0[0])
        FFTFP32Kernel0.buf.store(spad_base + 4, or0[1])
        FFTFP32Kernel0.buf.store(spad_base + 16 + 4, oi0[1])
        FFTFP32Kernel0.buf.store(spad_base + 8, or0[2])
        FFTFP32Kernel0.buf.store(spad_base + 16 + 8, oi0[2])
        FFTFP32Kernel0.buf.store(spad_base + 12, or0[3])
        FFTFP32Kernel0.buf.store(spad_base + 16 + 12, oi0[3])

        var twr1 = SIMD[DType.float32, 4](Float32(1), Float32(0.923879533), Float32(0.707106781), Float32(0.382683432))
        var twi1 = SIMD[DType.float32, 4](Float32(0), Float32(-0.382683432), Float32(-0.707106781), Float32(-0.923879533))
        var tr1 = or1 * twr1 - oi1 * twi1
        var ti1 = or1 * twi1 + oi1 * twr1
        or1 = tr1
        oi1 = ti1
        FFTFP32Kernel0.buf.store(spad_base + 1, or1[0])
        FFTFP32Kernel0.buf.store(spad_base + 16 + 1, oi1[0])
        FFTFP32Kernel0.buf.store(spad_base + 5, or1[1])
        FFTFP32Kernel0.buf.store(spad_base + 16 + 5, oi1[1])
        FFTFP32Kernel0.buf.store(spad_base + 9, or1[2])
        FFTFP32Kernel0.buf.store(spad_base + 16 + 9, oi1[2])
        FFTFP32Kernel0.buf.store(spad_base + 13, or1[3])
        FFTFP32Kernel0.buf.store(spad_base + 16 + 13, oi1[3])

        var twr2 = SIMD[DType.float32, 4](Float32(1), Float32(0.707106781), Float32(0), Float32(-0.707106781))
        var twi2 = SIMD[DType.float32, 4](Float32(0), Float32(-0.707106781), Float32(-1), Float32(-0.707106781))
        var tr2 = or2 * twr2 - oi2 * twi2
        var ti2 = or2 * twi2 + oi2 * twr2
        or2 = tr2
        oi2 = ti2
        FFTFP32Kernel0.buf.store(spad_base + 2, or2[0])
        FFTFP32Kernel0.buf.store(spad_base + 16 + 2, oi2[0])
        FFTFP32Kernel0.buf.store(spad_base + 6, or2[1])
        FFTFP32Kernel0.buf.store(spad_base + 16 + 6, oi2[1])
        FFTFP32Kernel0.buf.store(spad_base + 10, or2[2])
        FFTFP32Kernel0.buf.store(spad_base + 16 + 10, oi2[2])
        FFTFP32Kernel0.buf.store(spad_base + 14, or2[3])
        FFTFP32Kernel0.buf.store(spad_base + 16 + 14, oi2[3])

        var twr3 = SIMD[DType.float32, 4](Float32(1), Float32(0.382683432), Float32(-0.707106781), Float32(-0.923879533))
        var twi3 = SIMD[DType.float32, 4](Float32(0), Float32(-0.923879533), Float32(-0.707106781), Float32(0.382683432))
        var tr3 = or3 * twr3 - oi3 * twi3
        var ti3 = or3 * twi3 + oi3 * twr3
        or3 = tr3
        oi3 = ti3
        FFTFP32Kernel0.buf.store(spad_base + 3, or3[0])
        FFTFP32Kernel0.buf.store(spad_base + 16 + 3, oi3[0])
        FFTFP32Kernel0.buf.store(spad_base + 7, or3[1])
        FFTFP32Kernel0.buf.store(spad_base + 16 + 7, oi3[1])
        FFTFP32Kernel0.buf.store(spad_base + 11, or3[2])
        FFTFP32Kernel0.buf.store(spad_base + 16 + 11, oi3[2])
        FFTFP32Kernel0.buf.store(spad_base + 15, or3[3])
        FFTFP32Kernel0.buf.store(spad_base + 16 + 15, oi3[3])


    @staticmethod
    def stage_1():
        ref p = FFTFP32Kernel0.params[]
        comptime RADIX = 4
        comptime SIMD_ITERS = 1

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel0:
            return
        var spad_base = local_id * 32
        var out_batch_base = (((global_uthread_id() // 1) % 1) * 256) + ((global_uthread_id() % 1) * 16) + ((global_uthread_id() // 1) // 1)

        # ===== stage 1, SIMD batch 0 (valid lanes: 4/4) =====
        var rr0 = FFTFP32Kernel0.buf.load[DType.float32, 4](spad_base + 0)
        var ii0 = FFTFP32Kernel0.buf.load[DType.float32, 4](spad_base + 16 + 0)
        var rr1 = FFTFP32Kernel0.buf.load[DType.float32, 4](spad_base + 4)
        var ii1 = FFTFP32Kernel0.buf.load[DType.float32, 4](spad_base + 16 + 4)
        var rr2 = FFTFP32Kernel0.buf.load[DType.float32, 4](spad_base + 8)
        var ii2 = FFTFP32Kernel0.buf.load[DType.float32, 4](spad_base + 16 + 8)
        var rr3 = FFTFP32Kernel0.buf.load[DType.float32, 4](spad_base + 12)
        var ii3 = FFTFP32Kernel0.buf.load[DType.float32, 4](spad_base + 16 + 12)

        # fixed radix-4 butterfly
        var oa0r = rr0 + rr2
        var oa0i = ii0 + ii2
        var oa1r = rr0 - rr2
        var oa1i = ii0 - ii2
        var ob0r = rr1 + rr3
        var ob0i = ii1 + ii3
        var ob1r = rr1 - rr3
        var ob1i = ii1 - ii3
        var or0 = oa0r + ob0r
        var oi0 = oa0i + ob0i
        var or2 = oa0r - ob0r
        var oi2 = oa0i - ob0i
        var or1 = oa1r + ob1i
        var oi1 = oa1i - ob1r
        var or3 = oa1r - ob1i
        var oi3 = oa1i + ob1r
        var ltwr0_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 0)
        var ltwi0_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 0)
        var ltwr0_lane1 = p.large_twiddle_real_base.load[width=1](out_batch_base + 16)
        var ltwi0_lane1 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 16)
        var ltwr0_lane2 = p.large_twiddle_real_base.load[width=1](out_batch_base + 32)
        var ltwi0_lane2 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 32)
        var ltwr0_lane3 = p.large_twiddle_real_base.load[width=1](out_batch_base + 48)
        var ltwi0_lane3 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 48)
        var ltwr0 = SIMD[DType.float32, 4](ltwr0_lane0[0], ltwr0_lane1[0], ltwr0_lane2[0], ltwr0_lane3[0])
        var ltwi0 = SIMD[DType.float32, 4](ltwi0_lane0[0], ltwi0_lane1[0], ltwi0_lane2[0], ltwi0_lane3[0])
        var ltr0 = or0 * ltwr0 - oi0 * ltwi0
        var lti0 = or0 * ltwi0 + oi0 * ltwr0
        or0 = ltr0
        oi0 = lti0
        p.output_real_base.store(out_batch_base + 0, or0[0])
        p.output_imag_base.store(out_batch_base + 0, oi0[0])
        p.output_real_base.store(out_batch_base + 16, or0[1])
        p.output_imag_base.store(out_batch_base + 16, oi0[1])
        p.output_real_base.store(out_batch_base + 32, or0[2])
        p.output_imag_base.store(out_batch_base + 32, oi0[2])
        p.output_real_base.store(out_batch_base + 48, or0[3])
        p.output_imag_base.store(out_batch_base + 48, oi0[3])

        var ltwr1_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 64)
        var ltwi1_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 64)
        var ltwr1_lane1 = p.large_twiddle_real_base.load[width=1](out_batch_base + 80)
        var ltwi1_lane1 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 80)
        var ltwr1_lane2 = p.large_twiddle_real_base.load[width=1](out_batch_base + 96)
        var ltwi1_lane2 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 96)
        var ltwr1_lane3 = p.large_twiddle_real_base.load[width=1](out_batch_base + 112)
        var ltwi1_lane3 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 112)
        var ltwr1 = SIMD[DType.float32, 4](ltwr1_lane0[0], ltwr1_lane1[0], ltwr1_lane2[0], ltwr1_lane3[0])
        var ltwi1 = SIMD[DType.float32, 4](ltwi1_lane0[0], ltwi1_lane1[0], ltwi1_lane2[0], ltwi1_lane3[0])
        var ltr1 = or1 * ltwr1 - oi1 * ltwi1
        var lti1 = or1 * ltwi1 + oi1 * ltwr1
        or1 = ltr1
        oi1 = lti1
        p.output_real_base.store(out_batch_base + 64, or1[0])
        p.output_imag_base.store(out_batch_base + 64, oi1[0])
        p.output_real_base.store(out_batch_base + 80, or1[1])
        p.output_imag_base.store(out_batch_base + 80, oi1[1])
        p.output_real_base.store(out_batch_base + 96, or1[2])
        p.output_imag_base.store(out_batch_base + 96, oi1[2])
        p.output_real_base.store(out_batch_base + 112, or1[3])
        p.output_imag_base.store(out_batch_base + 112, oi1[3])

        var ltwr2_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 128)
        var ltwi2_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 128)
        var ltwr2_lane1 = p.large_twiddle_real_base.load[width=1](out_batch_base + 144)
        var ltwi2_lane1 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 144)
        var ltwr2_lane2 = p.large_twiddle_real_base.load[width=1](out_batch_base + 160)
        var ltwi2_lane2 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 160)
        var ltwr2_lane3 = p.large_twiddle_real_base.load[width=1](out_batch_base + 176)
        var ltwi2_lane3 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 176)
        var ltwr2 = SIMD[DType.float32, 4](ltwr2_lane0[0], ltwr2_lane1[0], ltwr2_lane2[0], ltwr2_lane3[0])
        var ltwi2 = SIMD[DType.float32, 4](ltwi2_lane0[0], ltwi2_lane1[0], ltwi2_lane2[0], ltwi2_lane3[0])
        var ltr2 = or2 * ltwr2 - oi2 * ltwi2
        var lti2 = or2 * ltwi2 + oi2 * ltwr2
        or2 = ltr2
        oi2 = lti2
        p.output_real_base.store(out_batch_base + 128, or2[0])
        p.output_imag_base.store(out_batch_base + 128, oi2[0])
        p.output_real_base.store(out_batch_base + 144, or2[1])
        p.output_imag_base.store(out_batch_base + 144, oi2[1])
        p.output_real_base.store(out_batch_base + 160, or2[2])
        p.output_imag_base.store(out_batch_base + 160, oi2[2])
        p.output_real_base.store(out_batch_base + 176, or2[3])
        p.output_imag_base.store(out_batch_base + 176, oi2[3])

        var ltwr3_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 192)
        var ltwi3_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 192)
        var ltwr3_lane1 = p.large_twiddle_real_base.load[width=1](out_batch_base + 208)
        var ltwi3_lane1 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 208)
        var ltwr3_lane2 = p.large_twiddle_real_base.load[width=1](out_batch_base + 224)
        var ltwi3_lane2 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 224)
        var ltwr3_lane3 = p.large_twiddle_real_base.load[width=1](out_batch_base + 240)
        var ltwi3_lane3 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 240)
        var ltwr3 = SIMD[DType.float32, 4](ltwr3_lane0[0], ltwr3_lane1[0], ltwr3_lane2[0], ltwr3_lane3[0])
        var ltwi3 = SIMD[DType.float32, 4](ltwi3_lane0[0], ltwi3_lane1[0], ltwi3_lane2[0], ltwi3_lane3[0])
        var ltr3 = or3 * ltwr3 - oi3 * ltwi3
        var lti3 = or3 * ltwi3 + oi3 * ltwr3
        or3 = ltr3
        oi3 = lti3
        p.output_real_base.store(out_batch_base + 192, or3[0])
        p.output_imag_base.store(out_batch_base + 192, oi3[0])
        p.output_real_base.store(out_batch_base + 208, or3[1])
        p.output_imag_base.store(out_batch_base + 208, oi3[1])
        p.output_real_base.store(out_batch_base + 224, or3[2])
        p.output_imag_base.store(out_batch_base + 224, oi3[2])
        p.output_real_base.store(out_batch_base + 240, or3[3])
        p.output_imag_base.store(out_batch_base + 240, oi3[3])


    @staticmethod
    def device_main():
        launch_parallel[FFTFP32Kernel0.stage_0]()
        launch_parallel[FFTFP32Kernel0.stage_1]()


comptime MAX_UTHREAD_FFTFP32Kernel1 = 16

@fieldwise_init
struct FFTFP32Kernel1Params(Movable):
    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]


struct FFTFP32Kernel1(NDPTask):
    comptime Params = FFTFP32Kernel1Params

    comptime buf = scratchpad[512, Float32, name="fftfp32kernel1_buf"]()

    @staticmethod
    def stage_0():
        ref p = FFTFP32Kernel1.params[]
        comptime RADIX = 4
        comptime SIMD_ITERS = 1

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel1:
            return
        var spad_base = local_id * 32
        var in_batch_base = global_uthread_id() * 16

        # ===== stage 0, SIMD batch 0 (valid lanes: 4/4) =====
        var rr0 = p.input_real_base.load[width=4](in_batch_base + 0)
        var ii0 = p.input_imag_base.load[width=4](in_batch_base + 0)
        var rr1 = p.input_real_base.load[width=4](in_batch_base + 4)
        var ii1 = p.input_imag_base.load[width=4](in_batch_base + 4)
        var rr2 = p.input_real_base.load[width=4](in_batch_base + 8)
        var ii2 = p.input_imag_base.load[width=4](in_batch_base + 8)
        var rr3 = p.input_real_base.load[width=4](in_batch_base + 12)
        var ii3 = p.input_imag_base.load[width=4](in_batch_base + 12)

        # fixed radix-4 butterfly
        var oa0r = rr0 + rr2
        var oa0i = ii0 + ii2
        var oa1r = rr0 - rr2
        var oa1i = ii0 - ii2
        var ob0r = rr1 + rr3
        var ob0i = ii1 + ii3
        var ob1r = rr1 - rr3
        var ob1i = ii1 - ii3
        var or0 = oa0r + ob0r
        var oi0 = oa0i + ob0i
        var or2 = oa0r - ob0r
        var oi2 = oa0i - ob0i
        var or1 = oa1r + ob1i
        var oi1 = oa1i - ob1r
        var or3 = oa1r - ob1i
        var oi3 = oa1i + ob1r
        FFTFP32Kernel1.buf.store(spad_base + 0, or0[0])
        FFTFP32Kernel1.buf.store(spad_base + 16 + 0, oi0[0])
        FFTFP32Kernel1.buf.store(spad_base + 4, or0[1])
        FFTFP32Kernel1.buf.store(spad_base + 16 + 4, oi0[1])
        FFTFP32Kernel1.buf.store(spad_base + 8, or0[2])
        FFTFP32Kernel1.buf.store(spad_base + 16 + 8, oi0[2])
        FFTFP32Kernel1.buf.store(spad_base + 12, or0[3])
        FFTFP32Kernel1.buf.store(spad_base + 16 + 12, oi0[3])

        var twr1 = SIMD[DType.float32, 4](Float32(1), Float32(0.923879533), Float32(0.707106781), Float32(0.382683432))
        var twi1 = SIMD[DType.float32, 4](Float32(0), Float32(-0.382683432), Float32(-0.707106781), Float32(-0.923879533))
        var tr1 = or1 * twr1 - oi1 * twi1
        var ti1 = or1 * twi1 + oi1 * twr1
        or1 = tr1
        oi1 = ti1
        FFTFP32Kernel1.buf.store(spad_base + 1, or1[0])
        FFTFP32Kernel1.buf.store(spad_base + 16 + 1, oi1[0])
        FFTFP32Kernel1.buf.store(spad_base + 5, or1[1])
        FFTFP32Kernel1.buf.store(spad_base + 16 + 5, oi1[1])
        FFTFP32Kernel1.buf.store(spad_base + 9, or1[2])
        FFTFP32Kernel1.buf.store(spad_base + 16 + 9, oi1[2])
        FFTFP32Kernel1.buf.store(spad_base + 13, or1[3])
        FFTFP32Kernel1.buf.store(spad_base + 16 + 13, oi1[3])

        var twr2 = SIMD[DType.float32, 4](Float32(1), Float32(0.707106781), Float32(0), Float32(-0.707106781))
        var twi2 = SIMD[DType.float32, 4](Float32(0), Float32(-0.707106781), Float32(-1), Float32(-0.707106781))
        var tr2 = or2 * twr2 - oi2 * twi2
        var ti2 = or2 * twi2 + oi2 * twr2
        or2 = tr2
        oi2 = ti2
        FFTFP32Kernel1.buf.store(spad_base + 2, or2[0])
        FFTFP32Kernel1.buf.store(spad_base + 16 + 2, oi2[0])
        FFTFP32Kernel1.buf.store(spad_base + 6, or2[1])
        FFTFP32Kernel1.buf.store(spad_base + 16 + 6, oi2[1])
        FFTFP32Kernel1.buf.store(spad_base + 10, or2[2])
        FFTFP32Kernel1.buf.store(spad_base + 16 + 10, oi2[2])
        FFTFP32Kernel1.buf.store(spad_base + 14, or2[3])
        FFTFP32Kernel1.buf.store(spad_base + 16 + 14, oi2[3])

        var twr3 = SIMD[DType.float32, 4](Float32(1), Float32(0.382683432), Float32(-0.707106781), Float32(-0.923879533))
        var twi3 = SIMD[DType.float32, 4](Float32(0), Float32(-0.923879533), Float32(-0.707106781), Float32(0.382683432))
        var tr3 = or3 * twr3 - oi3 * twi3
        var ti3 = or3 * twi3 + oi3 * twr3
        or3 = tr3
        oi3 = ti3
        FFTFP32Kernel1.buf.store(spad_base + 3, or3[0])
        FFTFP32Kernel1.buf.store(spad_base + 16 + 3, oi3[0])
        FFTFP32Kernel1.buf.store(spad_base + 7, or3[1])
        FFTFP32Kernel1.buf.store(spad_base + 16 + 7, oi3[1])
        FFTFP32Kernel1.buf.store(spad_base + 11, or3[2])
        FFTFP32Kernel1.buf.store(spad_base + 16 + 11, oi3[2])
        FFTFP32Kernel1.buf.store(spad_base + 15, or3[3])
        FFTFP32Kernel1.buf.store(spad_base + 16 + 15, oi3[3])


    @staticmethod
    def stage_1():
        ref p = FFTFP32Kernel1.params[]
        comptime RADIX = 4
        comptime SIMD_ITERS = 1

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel1:
            return
        var spad_base = local_id * 32
        var out_batch_base = global_uthread_id() * 1

        # ===== stage 1, SIMD batch 0 (valid lanes: 4/4) =====
        var rr0 = FFTFP32Kernel1.buf.load[DType.float32, 4](spad_base + 0)
        var ii0 = FFTFP32Kernel1.buf.load[DType.float32, 4](spad_base + 16 + 0)
        var rr1 = FFTFP32Kernel1.buf.load[DType.float32, 4](spad_base + 4)
        var ii1 = FFTFP32Kernel1.buf.load[DType.float32, 4](spad_base + 16 + 4)
        var rr2 = FFTFP32Kernel1.buf.load[DType.float32, 4](spad_base + 8)
        var ii2 = FFTFP32Kernel1.buf.load[DType.float32, 4](spad_base + 16 + 8)
        var rr3 = FFTFP32Kernel1.buf.load[DType.float32, 4](spad_base + 12)
        var ii3 = FFTFP32Kernel1.buf.load[DType.float32, 4](spad_base + 16 + 12)

        # fixed radix-4 butterfly
        var oa0r = rr0 + rr2
        var oa0i = ii0 + ii2
        var oa1r = rr0 - rr2
        var oa1i = ii0 - ii2
        var ob0r = rr1 + rr3
        var ob0i = ii1 + ii3
        var ob1r = rr1 - rr3
        var ob1i = ii1 - ii3
        var or0 = oa0r + ob0r
        var oi0 = oa0i + ob0i
        var or2 = oa0r - ob0r
        var oi2 = oa0i - ob0i
        var or1 = oa1r + ob1i
        var oi1 = oa1i - ob1r
        var or3 = oa1r - ob1i
        var oi3 = oa1i + ob1r
        p.output_real_base.store(out_batch_base + 0, or0[0])
        p.output_imag_base.store(out_batch_base + 0, oi0[0])
        p.output_real_base.store(out_batch_base + 16, or0[1])
        p.output_imag_base.store(out_batch_base + 16, oi0[1])
        p.output_real_base.store(out_batch_base + 32, or0[2])
        p.output_imag_base.store(out_batch_base + 32, oi0[2])
        p.output_real_base.store(out_batch_base + 48, or0[3])
        p.output_imag_base.store(out_batch_base + 48, oi0[3])

        p.output_real_base.store(out_batch_base + 64, or1[0])
        p.output_imag_base.store(out_batch_base + 64, oi1[0])
        p.output_real_base.store(out_batch_base + 80, or1[1])
        p.output_imag_base.store(out_batch_base + 80, oi1[1])
        p.output_real_base.store(out_batch_base + 96, or1[2])
        p.output_imag_base.store(out_batch_base + 96, oi1[2])
        p.output_real_base.store(out_batch_base + 112, or1[3])
        p.output_imag_base.store(out_batch_base + 112, oi1[3])

        p.output_real_base.store(out_batch_base + 128, or2[0])
        p.output_imag_base.store(out_batch_base + 128, oi2[0])
        p.output_real_base.store(out_batch_base + 144, or2[1])
        p.output_imag_base.store(out_batch_base + 144, oi2[1])
        p.output_real_base.store(out_batch_base + 160, or2[2])
        p.output_imag_base.store(out_batch_base + 160, oi2[2])
        p.output_real_base.store(out_batch_base + 176, or2[3])
        p.output_imag_base.store(out_batch_base + 176, oi2[3])

        p.output_real_base.store(out_batch_base + 192, or3[0])
        p.output_imag_base.store(out_batch_base + 192, oi3[0])
        p.output_real_base.store(out_batch_base + 208, or3[1])
        p.output_imag_base.store(out_batch_base + 208, oi3[1])
        p.output_real_base.store(out_batch_base + 224, or3[2])
        p.output_imag_base.store(out_batch_base + 224, oi3[2])
        p.output_real_base.store(out_batch_base + 240, or3[3])
        p.output_imag_base.store(out_batch_base + 240, oi3[3])


    @staticmethod
    def device_main():
        launch_parallel[FFTFP32Kernel1.stage_0]()
        launch_parallel[FFTFP32Kernel1.stage_1]()


def main() raises:
    if FFTFP32Kernel0.emit_ir_if_asked():
        return

    var n = 256
    var input_real = cxl_alloc[Float32](n)
    var input_imag = cxl_alloc[Float32](n)
    var mid0_real = cxl_alloc[Float32](n)
    var mid0_imag = cxl_alloc[Float32](n)
    var output_real = cxl_alloc[Float32](n)
    var output_imag = cxl_alloc[Float32](n)
    var ref_real = cxl_alloc[Float32](n)
    var ref_imag = cxl_alloc[Float32](n)

    var large_twiddle0_real = cxl_alloc[Float32](n)
    var large_twiddle0_imag = cxl_alloc[Float32](n)
    var lt_pi_0 = Float64(3.141592653589793)
    var lt_sign_0 = Float64(-1.0)
    var lt_r_0 = 0
    while lt_r_0 < 16:
        var lt_c1_0 = 0
        while lt_c1_0 < 16:
            var lt_angle_0 = lt_sign_0 * 2.0 * lt_pi_0 * Float64(lt_r_0) * Float64(lt_c1_0) * Float64(1) / Float64(256)
            var lt_val_r_0 = Float32(host_cos(lt_angle_0))
            var lt_val_i_0 = Float32(host_sin(lt_angle_0))
            var lt_dnext_0 = lt_r_0 // 1
            var lt_rest_0 = lt_r_0 % 1
            var lt_out_a_0 = 0
            while lt_out_a_0 < 1:
                var lt_addr_0 = lt_rest_0 * 256 + (lt_out_a_0 + lt_c1_0 * 1) * 16 + lt_dnext_0
                large_twiddle0_real[lt_addr_0] = lt_val_r_0
                large_twiddle0_imag[lt_addr_0] = lt_val_i_0
                lt_out_a_0 += 1
            lt_c1_0 += 1
        lt_r_0 += 1

    var pool0_elems = 64
    var pool0 = cxl_alloc[Float32](pool0_elems)
    var pool1_elems = 64
    var pool1 = cxl_alloc[Float32](pool1_elems)

    seed(0)
    for i in range(n):
        input_real[i] = Float32(random_float64(-1.0, 1.0))
        input_imag[i] = Float32(random_float64(-1.0, 1.0))
        mid0_real[i] = Float32(0)
        mid0_imag[i] = Float32(0)
        output_real[i] = Float32(0)
        output_imag[i] = Float32(0)
        ref_real[i] = Float32(0)
        ref_imag[i] = Float32(0)

    var rc0 = FFTFP32Kernel0.launch(
        PooledRange.over(pool0, pool0_elems),
        FFTFP32Kernel0Params(input_real, input_imag, mid0_real, mid0_imag, large_twiddle0_real, large_twiddle0_imag),
    )
    if rc0 != 0:
        print("[host] FFT kernel0 failed, exit", rc0)
        return

    var rc1 = FFTFP32Kernel1.launch(
        PooledRange.over(pool1, pool1_elems),
        FFTFP32Kernel1Params(mid0_real, mid0_imag, output_real, output_imag),
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
            print("[host] multi-kernel FFT mismatch at", i)
            print("  expected:", ref_real[i], ref_imag[i])
            print("  actual:  ", output_real[i], output_imag[i])
            print("  error:   ", err_r, err_i)
            return

    print("[host] multi-kernel FFT verification passed")
