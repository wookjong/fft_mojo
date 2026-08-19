from std.sys import size_of
from std.random import random_float64, seed
from std.math import cos as host_cos, sin as host_sin

from m2ndp import VECTOR_WIDTH, NDPTask, PooledRange, global_uthread_id, local_uthread_id, launch_parallel, scratchpad
from m2ndp_host import cxl_alloc

comptime W = VECTOR_WIDTH // size_of[Float32]()
comptime N = 960

comptime MAX_UTHREAD_FFTFP32Kernel0 = 15

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

    comptime buf_a = scratchpad[1920, Float32, name="fftfp32kernel0_buf_a"]()
    comptime buf_b = scratchpad[1920, Float32, name="fftfp32kernel0_buf_b"]()

    @staticmethod
    def stage_0():
        ref p = FFTFP32Kernel0.params[]
        comptime RADIX = 4
        comptime SIMD_ITERS = 2

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel0:
            return
        var spad_base = local_id * 128
        var in_batch_base = global_uthread_id() * 1

        if True:  # batch 0 scope
            # ===== stage 0, SIMD batch 0 (valid lanes: 8/8) =====
            var rr0_lane0 = p.input_real_base.load[width=1](in_batch_base + 0)
            var ii0_lane0 = p.input_imag_base.load[width=1](in_batch_base + 0)
            var rr0_lane1 = p.input_real_base.load[width=1](in_batch_base + 15)
            var ii0_lane1 = p.input_imag_base.load[width=1](in_batch_base + 15)
            var rr0_lane2 = p.input_real_base.load[width=1](in_batch_base + 30)
            var ii0_lane2 = p.input_imag_base.load[width=1](in_batch_base + 30)
            var rr0_lane3 = p.input_real_base.load[width=1](in_batch_base + 45)
            var ii0_lane3 = p.input_imag_base.load[width=1](in_batch_base + 45)
            var rr0_lane4 = p.input_real_base.load[width=1](in_batch_base + 60)
            var ii0_lane4 = p.input_imag_base.load[width=1](in_batch_base + 60)
            var rr0_lane5 = p.input_real_base.load[width=1](in_batch_base + 75)
            var ii0_lane5 = p.input_imag_base.load[width=1](in_batch_base + 75)
            var rr0_lane6 = p.input_real_base.load[width=1](in_batch_base + 90)
            var ii0_lane6 = p.input_imag_base.load[width=1](in_batch_base + 90)
            var rr0_lane7 = p.input_real_base.load[width=1](in_batch_base + 105)
            var ii0_lane7 = p.input_imag_base.load[width=1](in_batch_base + 105)
            var rr0 = SIMD[DType.float32, W](rr0_lane0[0], rr0_lane1[0], rr0_lane2[0], rr0_lane3[0], rr0_lane4[0], rr0_lane5[0], rr0_lane6[0], rr0_lane7[0])
            var ii0 = SIMD[DType.float32, W](ii0_lane0[0], ii0_lane1[0], ii0_lane2[0], ii0_lane3[0], ii0_lane4[0], ii0_lane5[0], ii0_lane6[0], ii0_lane7[0])
            var rr1_lane0 = p.input_real_base.load[width=1](in_batch_base + 240)
            var ii1_lane0 = p.input_imag_base.load[width=1](in_batch_base + 240)
            var rr1_lane1 = p.input_real_base.load[width=1](in_batch_base + 255)
            var ii1_lane1 = p.input_imag_base.load[width=1](in_batch_base + 255)
            var rr1_lane2 = p.input_real_base.load[width=1](in_batch_base + 270)
            var ii1_lane2 = p.input_imag_base.load[width=1](in_batch_base + 270)
            var rr1_lane3 = p.input_real_base.load[width=1](in_batch_base + 285)
            var ii1_lane3 = p.input_imag_base.load[width=1](in_batch_base + 285)
            var rr1_lane4 = p.input_real_base.load[width=1](in_batch_base + 300)
            var ii1_lane4 = p.input_imag_base.load[width=1](in_batch_base + 300)
            var rr1_lane5 = p.input_real_base.load[width=1](in_batch_base + 315)
            var ii1_lane5 = p.input_imag_base.load[width=1](in_batch_base + 315)
            var rr1_lane6 = p.input_real_base.load[width=1](in_batch_base + 330)
            var ii1_lane6 = p.input_imag_base.load[width=1](in_batch_base + 330)
            var rr1_lane7 = p.input_real_base.load[width=1](in_batch_base + 345)
            var ii1_lane7 = p.input_imag_base.load[width=1](in_batch_base + 345)
            var rr1 = SIMD[DType.float32, W](rr1_lane0[0], rr1_lane1[0], rr1_lane2[0], rr1_lane3[0], rr1_lane4[0], rr1_lane5[0], rr1_lane6[0], rr1_lane7[0])
            var ii1 = SIMD[DType.float32, W](ii1_lane0[0], ii1_lane1[0], ii1_lane2[0], ii1_lane3[0], ii1_lane4[0], ii1_lane5[0], ii1_lane6[0], ii1_lane7[0])
            var rr2_lane0 = p.input_real_base.load[width=1](in_batch_base + 480)
            var ii2_lane0 = p.input_imag_base.load[width=1](in_batch_base + 480)
            var rr2_lane1 = p.input_real_base.load[width=1](in_batch_base + 495)
            var ii2_lane1 = p.input_imag_base.load[width=1](in_batch_base + 495)
            var rr2_lane2 = p.input_real_base.load[width=1](in_batch_base + 510)
            var ii2_lane2 = p.input_imag_base.load[width=1](in_batch_base + 510)
            var rr2_lane3 = p.input_real_base.load[width=1](in_batch_base + 525)
            var ii2_lane3 = p.input_imag_base.load[width=1](in_batch_base + 525)
            var rr2_lane4 = p.input_real_base.load[width=1](in_batch_base + 540)
            var ii2_lane4 = p.input_imag_base.load[width=1](in_batch_base + 540)
            var rr2_lane5 = p.input_real_base.load[width=1](in_batch_base + 555)
            var ii2_lane5 = p.input_imag_base.load[width=1](in_batch_base + 555)
            var rr2_lane6 = p.input_real_base.load[width=1](in_batch_base + 570)
            var ii2_lane6 = p.input_imag_base.load[width=1](in_batch_base + 570)
            var rr2_lane7 = p.input_real_base.load[width=1](in_batch_base + 585)
            var ii2_lane7 = p.input_imag_base.load[width=1](in_batch_base + 585)
            var rr2 = SIMD[DType.float32, W](rr2_lane0[0], rr2_lane1[0], rr2_lane2[0], rr2_lane3[0], rr2_lane4[0], rr2_lane5[0], rr2_lane6[0], rr2_lane7[0])
            var ii2 = SIMD[DType.float32, W](ii2_lane0[0], ii2_lane1[0], ii2_lane2[0], ii2_lane3[0], ii2_lane4[0], ii2_lane5[0], ii2_lane6[0], ii2_lane7[0])
            var rr3_lane0 = p.input_real_base.load[width=1](in_batch_base + 720)
            var ii3_lane0 = p.input_imag_base.load[width=1](in_batch_base + 720)
            var rr3_lane1 = p.input_real_base.load[width=1](in_batch_base + 735)
            var ii3_lane1 = p.input_imag_base.load[width=1](in_batch_base + 735)
            var rr3_lane2 = p.input_real_base.load[width=1](in_batch_base + 750)
            var ii3_lane2 = p.input_imag_base.load[width=1](in_batch_base + 750)
            var rr3_lane3 = p.input_real_base.load[width=1](in_batch_base + 765)
            var ii3_lane3 = p.input_imag_base.load[width=1](in_batch_base + 765)
            var rr3_lane4 = p.input_real_base.load[width=1](in_batch_base + 780)
            var ii3_lane4 = p.input_imag_base.load[width=1](in_batch_base + 780)
            var rr3_lane5 = p.input_real_base.load[width=1](in_batch_base + 795)
            var ii3_lane5 = p.input_imag_base.load[width=1](in_batch_base + 795)
            var rr3_lane6 = p.input_real_base.load[width=1](in_batch_base + 810)
            var ii3_lane6 = p.input_imag_base.load[width=1](in_batch_base + 810)
            var rr3_lane7 = p.input_real_base.load[width=1](in_batch_base + 825)
            var ii3_lane7 = p.input_imag_base.load[width=1](in_batch_base + 825)
            var rr3 = SIMD[DType.float32, W](rr3_lane0[0], rr3_lane1[0], rr3_lane2[0], rr3_lane3[0], rr3_lane4[0], rr3_lane5[0], rr3_lane6[0], rr3_lane7[0])
            var ii3 = SIMD[DType.float32, W](ii3_lane0[0], ii3_lane1[0], ii3_lane2[0], ii3_lane3[0], ii3_lane4[0], ii3_lane5[0], ii3_lane6[0], ii3_lane7[0])

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

            FFTFP32Kernel0.buf_a.store(spad_base + 0, or0[0])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 0, oi0[0])
            FFTFP32Kernel0.buf_a.store(spad_base + 4, or0[1])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 4, oi0[1])
            FFTFP32Kernel0.buf_a.store(spad_base + 8, or0[2])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 8, oi0[2])
            FFTFP32Kernel0.buf_a.store(spad_base + 12, or0[3])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 12, oi0[3])
            FFTFP32Kernel0.buf_a.store(spad_base + 16, or0[4])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 16, oi0[4])
            FFTFP32Kernel0.buf_a.store(spad_base + 20, or0[5])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 20, oi0[5])
            FFTFP32Kernel0.buf_a.store(spad_base + 24, or0[6])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 24, oi0[6])
            FFTFP32Kernel0.buf_a.store(spad_base + 28, or0[7])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 28, oi0[7])

            var twr1 = SIMD[DType.float32, W](Float32(1), Float32(0.995184727), Float32(0.98078528), Float32(0.956940336), Float32(0.923879533), Float32(0.881921264), Float32(0.831469612), Float32(0.773010453))
            var twi1 = SIMD[DType.float32, W](Float32(0), Float32(-0.0980171403), Float32(-0.195090322), Float32(-0.290284677), Float32(-0.382683432), Float32(-0.471396737), Float32(-0.555570233), Float32(-0.634393284))
            var tr1 = or1 * twr1 - oi1 * twi1
            var ti1 = or1 * twi1 + oi1 * twr1
            or1 = tr1
            oi1 = ti1
            FFTFP32Kernel0.buf_a.store(spad_base + 1, or1[0])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 1, oi1[0])
            FFTFP32Kernel0.buf_a.store(spad_base + 5, or1[1])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 5, oi1[1])
            FFTFP32Kernel0.buf_a.store(spad_base + 9, or1[2])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 9, oi1[2])
            FFTFP32Kernel0.buf_a.store(spad_base + 13, or1[3])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 13, oi1[3])
            FFTFP32Kernel0.buf_a.store(spad_base + 17, or1[4])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 17, oi1[4])
            FFTFP32Kernel0.buf_a.store(spad_base + 21, or1[5])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 21, oi1[5])
            FFTFP32Kernel0.buf_a.store(spad_base + 25, or1[6])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 25, oi1[6])
            FFTFP32Kernel0.buf_a.store(spad_base + 29, or1[7])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 29, oi1[7])

            var twr2 = SIMD[DType.float32, W](Float32(1), Float32(0.98078528), Float32(0.923879533), Float32(0.831469612), Float32(0.707106781), Float32(0.555570233), Float32(0.382683432), Float32(0.195090322))
            var twi2 = SIMD[DType.float32, W](Float32(0), Float32(-0.195090322), Float32(-0.382683432), Float32(-0.555570233), Float32(-0.707106781), Float32(-0.831469612), Float32(-0.923879533), Float32(-0.98078528))
            var tr2 = or2 * twr2 - oi2 * twi2
            var ti2 = or2 * twi2 + oi2 * twr2
            or2 = tr2
            oi2 = ti2
            FFTFP32Kernel0.buf_a.store(spad_base + 2, or2[0])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 2, oi2[0])
            FFTFP32Kernel0.buf_a.store(spad_base + 6, or2[1])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 6, oi2[1])
            FFTFP32Kernel0.buf_a.store(spad_base + 10, or2[2])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 10, oi2[2])
            FFTFP32Kernel0.buf_a.store(spad_base + 14, or2[3])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 14, oi2[3])
            FFTFP32Kernel0.buf_a.store(spad_base + 18, or2[4])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 18, oi2[4])
            FFTFP32Kernel0.buf_a.store(spad_base + 22, or2[5])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 22, oi2[5])
            FFTFP32Kernel0.buf_a.store(spad_base + 26, or2[6])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 26, oi2[6])
            FFTFP32Kernel0.buf_a.store(spad_base + 30, or2[7])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 30, oi2[7])

            var twr3 = SIMD[DType.float32, W](Float32(1), Float32(0.956940336), Float32(0.831469612), Float32(0.634393284), Float32(0.382683432), Float32(0.0980171403), Float32(-0.195090322), Float32(-0.471396737))
            var twi3 = SIMD[DType.float32, W](Float32(0), Float32(-0.290284677), Float32(-0.555570233), Float32(-0.773010453), Float32(-0.923879533), Float32(-0.995184727), Float32(-0.98078528), Float32(-0.881921264))
            var tr3 = or3 * twr3 - oi3 * twi3
            var ti3 = or3 * twi3 + oi3 * twr3
            or3 = tr3
            oi3 = ti3
            FFTFP32Kernel0.buf_a.store(spad_base + 3, or3[0])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 3, oi3[0])
            FFTFP32Kernel0.buf_a.store(spad_base + 7, or3[1])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 7, oi3[1])
            FFTFP32Kernel0.buf_a.store(spad_base + 11, or3[2])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 11, oi3[2])
            FFTFP32Kernel0.buf_a.store(spad_base + 15, or3[3])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 15, oi3[3])
            FFTFP32Kernel0.buf_a.store(spad_base + 19, or3[4])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 19, oi3[4])
            FFTFP32Kernel0.buf_a.store(spad_base + 23, or3[5])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 23, oi3[5])
            FFTFP32Kernel0.buf_a.store(spad_base + 27, or3[6])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 27, oi3[6])
            FFTFP32Kernel0.buf_a.store(spad_base + 31, or3[7])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 31, oi3[7])

        if True:  # batch 1 scope
            # ===== stage 0, SIMD batch 1 (valid lanes: 8/8) =====
            var rr0_lane0 = p.input_real_base.load[width=1](in_batch_base + 120)
            var ii0_lane0 = p.input_imag_base.load[width=1](in_batch_base + 120)
            var rr0_lane1 = p.input_real_base.load[width=1](in_batch_base + 135)
            var ii0_lane1 = p.input_imag_base.load[width=1](in_batch_base + 135)
            var rr0_lane2 = p.input_real_base.load[width=1](in_batch_base + 150)
            var ii0_lane2 = p.input_imag_base.load[width=1](in_batch_base + 150)
            var rr0_lane3 = p.input_real_base.load[width=1](in_batch_base + 165)
            var ii0_lane3 = p.input_imag_base.load[width=1](in_batch_base + 165)
            var rr0_lane4 = p.input_real_base.load[width=1](in_batch_base + 180)
            var ii0_lane4 = p.input_imag_base.load[width=1](in_batch_base + 180)
            var rr0_lane5 = p.input_real_base.load[width=1](in_batch_base + 195)
            var ii0_lane5 = p.input_imag_base.load[width=1](in_batch_base + 195)
            var rr0_lane6 = p.input_real_base.load[width=1](in_batch_base + 210)
            var ii0_lane6 = p.input_imag_base.load[width=1](in_batch_base + 210)
            var rr0_lane7 = p.input_real_base.load[width=1](in_batch_base + 225)
            var ii0_lane7 = p.input_imag_base.load[width=1](in_batch_base + 225)
            var rr0 = SIMD[DType.float32, W](rr0_lane0[0], rr0_lane1[0], rr0_lane2[0], rr0_lane3[0], rr0_lane4[0], rr0_lane5[0], rr0_lane6[0], rr0_lane7[0])
            var ii0 = SIMD[DType.float32, W](ii0_lane0[0], ii0_lane1[0], ii0_lane2[0], ii0_lane3[0], ii0_lane4[0], ii0_lane5[0], ii0_lane6[0], ii0_lane7[0])
            var rr1_lane0 = p.input_real_base.load[width=1](in_batch_base + 360)
            var ii1_lane0 = p.input_imag_base.load[width=1](in_batch_base + 360)
            var rr1_lane1 = p.input_real_base.load[width=1](in_batch_base + 375)
            var ii1_lane1 = p.input_imag_base.load[width=1](in_batch_base + 375)
            var rr1_lane2 = p.input_real_base.load[width=1](in_batch_base + 390)
            var ii1_lane2 = p.input_imag_base.load[width=1](in_batch_base + 390)
            var rr1_lane3 = p.input_real_base.load[width=1](in_batch_base + 405)
            var ii1_lane3 = p.input_imag_base.load[width=1](in_batch_base + 405)
            var rr1_lane4 = p.input_real_base.load[width=1](in_batch_base + 420)
            var ii1_lane4 = p.input_imag_base.load[width=1](in_batch_base + 420)
            var rr1_lane5 = p.input_real_base.load[width=1](in_batch_base + 435)
            var ii1_lane5 = p.input_imag_base.load[width=1](in_batch_base + 435)
            var rr1_lane6 = p.input_real_base.load[width=1](in_batch_base + 450)
            var ii1_lane6 = p.input_imag_base.load[width=1](in_batch_base + 450)
            var rr1_lane7 = p.input_real_base.load[width=1](in_batch_base + 465)
            var ii1_lane7 = p.input_imag_base.load[width=1](in_batch_base + 465)
            var rr1 = SIMD[DType.float32, W](rr1_lane0[0], rr1_lane1[0], rr1_lane2[0], rr1_lane3[0], rr1_lane4[0], rr1_lane5[0], rr1_lane6[0], rr1_lane7[0])
            var ii1 = SIMD[DType.float32, W](ii1_lane0[0], ii1_lane1[0], ii1_lane2[0], ii1_lane3[0], ii1_lane4[0], ii1_lane5[0], ii1_lane6[0], ii1_lane7[0])
            var rr2_lane0 = p.input_real_base.load[width=1](in_batch_base + 600)
            var ii2_lane0 = p.input_imag_base.load[width=1](in_batch_base + 600)
            var rr2_lane1 = p.input_real_base.load[width=1](in_batch_base + 615)
            var ii2_lane1 = p.input_imag_base.load[width=1](in_batch_base + 615)
            var rr2_lane2 = p.input_real_base.load[width=1](in_batch_base + 630)
            var ii2_lane2 = p.input_imag_base.load[width=1](in_batch_base + 630)
            var rr2_lane3 = p.input_real_base.load[width=1](in_batch_base + 645)
            var ii2_lane3 = p.input_imag_base.load[width=1](in_batch_base + 645)
            var rr2_lane4 = p.input_real_base.load[width=1](in_batch_base + 660)
            var ii2_lane4 = p.input_imag_base.load[width=1](in_batch_base + 660)
            var rr2_lane5 = p.input_real_base.load[width=1](in_batch_base + 675)
            var ii2_lane5 = p.input_imag_base.load[width=1](in_batch_base + 675)
            var rr2_lane6 = p.input_real_base.load[width=1](in_batch_base + 690)
            var ii2_lane6 = p.input_imag_base.load[width=1](in_batch_base + 690)
            var rr2_lane7 = p.input_real_base.load[width=1](in_batch_base + 705)
            var ii2_lane7 = p.input_imag_base.load[width=1](in_batch_base + 705)
            var rr2 = SIMD[DType.float32, W](rr2_lane0[0], rr2_lane1[0], rr2_lane2[0], rr2_lane3[0], rr2_lane4[0], rr2_lane5[0], rr2_lane6[0], rr2_lane7[0])
            var ii2 = SIMD[DType.float32, W](ii2_lane0[0], ii2_lane1[0], ii2_lane2[0], ii2_lane3[0], ii2_lane4[0], ii2_lane5[0], ii2_lane6[0], ii2_lane7[0])
            var rr3_lane0 = p.input_real_base.load[width=1](in_batch_base + 840)
            var ii3_lane0 = p.input_imag_base.load[width=1](in_batch_base + 840)
            var rr3_lane1 = p.input_real_base.load[width=1](in_batch_base + 855)
            var ii3_lane1 = p.input_imag_base.load[width=1](in_batch_base + 855)
            var rr3_lane2 = p.input_real_base.load[width=1](in_batch_base + 870)
            var ii3_lane2 = p.input_imag_base.load[width=1](in_batch_base + 870)
            var rr3_lane3 = p.input_real_base.load[width=1](in_batch_base + 885)
            var ii3_lane3 = p.input_imag_base.load[width=1](in_batch_base + 885)
            var rr3_lane4 = p.input_real_base.load[width=1](in_batch_base + 900)
            var ii3_lane4 = p.input_imag_base.load[width=1](in_batch_base + 900)
            var rr3_lane5 = p.input_real_base.load[width=1](in_batch_base + 915)
            var ii3_lane5 = p.input_imag_base.load[width=1](in_batch_base + 915)
            var rr3_lane6 = p.input_real_base.load[width=1](in_batch_base + 930)
            var ii3_lane6 = p.input_imag_base.load[width=1](in_batch_base + 930)
            var rr3_lane7 = p.input_real_base.load[width=1](in_batch_base + 945)
            var ii3_lane7 = p.input_imag_base.load[width=1](in_batch_base + 945)
            var rr3 = SIMD[DType.float32, W](rr3_lane0[0], rr3_lane1[0], rr3_lane2[0], rr3_lane3[0], rr3_lane4[0], rr3_lane5[0], rr3_lane6[0], rr3_lane7[0])
            var ii3 = SIMD[DType.float32, W](ii3_lane0[0], ii3_lane1[0], ii3_lane2[0], ii3_lane3[0], ii3_lane4[0], ii3_lane5[0], ii3_lane6[0], ii3_lane7[0])

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

            FFTFP32Kernel0.buf_a.store(spad_base + 32, or0[0])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 32, oi0[0])
            FFTFP32Kernel0.buf_a.store(spad_base + 36, or0[1])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 36, oi0[1])
            FFTFP32Kernel0.buf_a.store(spad_base + 40, or0[2])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 40, oi0[2])
            FFTFP32Kernel0.buf_a.store(spad_base + 44, or0[3])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 44, oi0[3])
            FFTFP32Kernel0.buf_a.store(spad_base + 48, or0[4])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 48, oi0[4])
            FFTFP32Kernel0.buf_a.store(spad_base + 52, or0[5])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 52, oi0[5])
            FFTFP32Kernel0.buf_a.store(spad_base + 56, or0[6])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 56, oi0[6])
            FFTFP32Kernel0.buf_a.store(spad_base + 60, or0[7])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 60, oi0[7])

            var twr1 = SIMD[DType.float32, W](Float32(0.707106781), Float32(0.634393284), Float32(0.555570233), Float32(0.471396737), Float32(0.382683432), Float32(0.290284677), Float32(0.195090322), Float32(0.0980171403))
            var twi1 = SIMD[DType.float32, W](Float32(-0.707106781), Float32(-0.773010453), Float32(-0.831469612), Float32(-0.881921264), Float32(-0.923879533), Float32(-0.956940336), Float32(-0.98078528), Float32(-0.995184727))
            var tr1 = or1 * twr1 - oi1 * twi1
            var ti1 = or1 * twi1 + oi1 * twr1
            or1 = tr1
            oi1 = ti1
            FFTFP32Kernel0.buf_a.store(spad_base + 33, or1[0])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 33, oi1[0])
            FFTFP32Kernel0.buf_a.store(spad_base + 37, or1[1])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 37, oi1[1])
            FFTFP32Kernel0.buf_a.store(spad_base + 41, or1[2])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 41, oi1[2])
            FFTFP32Kernel0.buf_a.store(spad_base + 45, or1[3])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 45, oi1[3])
            FFTFP32Kernel0.buf_a.store(spad_base + 49, or1[4])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 49, oi1[4])
            FFTFP32Kernel0.buf_a.store(spad_base + 53, or1[5])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 53, oi1[5])
            FFTFP32Kernel0.buf_a.store(spad_base + 57, or1[6])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 57, oi1[6])
            FFTFP32Kernel0.buf_a.store(spad_base + 61, or1[7])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 61, oi1[7])

            var twr2 = SIMD[DType.float32, W](Float32(0), Float32(-0.195090322), Float32(-0.382683432), Float32(-0.555570233), Float32(-0.707106781), Float32(-0.831469612), Float32(-0.923879533), Float32(-0.98078528))
            var twi2 = SIMD[DType.float32, W](Float32(-1), Float32(-0.98078528), Float32(-0.923879533), Float32(-0.831469612), Float32(-0.707106781), Float32(-0.555570233), Float32(-0.382683432), Float32(-0.195090322))
            var tr2 = or2 * twr2 - oi2 * twi2
            var ti2 = or2 * twi2 + oi2 * twr2
            or2 = tr2
            oi2 = ti2
            FFTFP32Kernel0.buf_a.store(spad_base + 34, or2[0])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 34, oi2[0])
            FFTFP32Kernel0.buf_a.store(spad_base + 38, or2[1])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 38, oi2[1])
            FFTFP32Kernel0.buf_a.store(spad_base + 42, or2[2])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 42, oi2[2])
            FFTFP32Kernel0.buf_a.store(spad_base + 46, or2[3])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 46, oi2[3])
            FFTFP32Kernel0.buf_a.store(spad_base + 50, or2[4])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 50, oi2[4])
            FFTFP32Kernel0.buf_a.store(spad_base + 54, or2[5])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 54, oi2[5])
            FFTFP32Kernel0.buf_a.store(spad_base + 58, or2[6])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 58, oi2[6])
            FFTFP32Kernel0.buf_a.store(spad_base + 62, or2[7])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 62, oi2[7])

            var twr3 = SIMD[DType.float32, W](Float32(-0.707106781), Float32(-0.881921264), Float32(-0.98078528), Float32(-0.995184727), Float32(-0.923879533), Float32(-0.773010453), Float32(-0.555570233), Float32(-0.290284677))
            var twi3 = SIMD[DType.float32, W](Float32(-0.707106781), Float32(-0.471396737), Float32(-0.195090322), Float32(0.0980171403), Float32(0.382683432), Float32(0.634393284), Float32(0.831469612), Float32(0.956940336))
            var tr3 = or3 * twr3 - oi3 * twi3
            var ti3 = or3 * twi3 + oi3 * twr3
            or3 = tr3
            oi3 = ti3
            FFTFP32Kernel0.buf_a.store(spad_base + 35, or3[0])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 35, oi3[0])
            FFTFP32Kernel0.buf_a.store(spad_base + 39, or3[1])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 39, oi3[1])
            FFTFP32Kernel0.buf_a.store(spad_base + 43, or3[2])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 43, oi3[2])
            FFTFP32Kernel0.buf_a.store(spad_base + 47, or3[3])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 47, oi3[3])
            FFTFP32Kernel0.buf_a.store(spad_base + 51, or3[4])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 51, oi3[4])
            FFTFP32Kernel0.buf_a.store(spad_base + 55, or3[5])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 55, oi3[5])
            FFTFP32Kernel0.buf_a.store(spad_base + 59, or3[6])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 59, oi3[6])
            FFTFP32Kernel0.buf_a.store(spad_base + 63, or3[7])
            FFTFP32Kernel0.buf_a.store(spad_base + N + 63, oi3[7])

    @staticmethod
    def stage_1():
        ref p = FFTFP32Kernel0.params[]
        comptime RADIX = 4
        comptime SIMD_ITERS = 2

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel0:
            return
        var spad_base = local_id * 128

        if True:  # batch 0 scope
            # ===== stage 1, SIMD batch 0 (valid lanes: 8/8) =====
            var rr0 = FFTFP32Kernel0.buf_a.load[DType.float32, W](spad_base + 0)
            var ii0 = FFTFP32Kernel0.buf_a.load[DType.float32, W](spad_base + N + 0)
            var rr1 = FFTFP32Kernel0.buf_a.load[DType.float32, W](spad_base + 16)
            var ii1 = FFTFP32Kernel0.buf_a.load[DType.float32, W](spad_base + N + 16)
            var rr2 = FFTFP32Kernel0.buf_a.load[DType.float32, W](spad_base + 32)
            var ii2 = FFTFP32Kernel0.buf_a.load[DType.float32, W](spad_base + N + 32)
            var rr3 = FFTFP32Kernel0.buf_a.load[DType.float32, W](spad_base + 48)
            var ii3 = FFTFP32Kernel0.buf_a.load[DType.float32, W](spad_base + N + 48)

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

            FFTFP32Kernel0.buf_b.store(spad_base + 0, or0[0])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 0, oi0[0])
            FFTFP32Kernel0.buf_b.store(spad_base + 1, or0[1])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 1, oi0[1])
            FFTFP32Kernel0.buf_b.store(spad_base + 2, or0[2])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 2, oi0[2])
            FFTFP32Kernel0.buf_b.store(spad_base + 3, or0[3])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 3, oi0[3])
            FFTFP32Kernel0.buf_b.store(spad_base + 16, or0[4])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 16, oi0[4])
            FFTFP32Kernel0.buf_b.store(spad_base + 17, or0[5])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 17, oi0[5])
            FFTFP32Kernel0.buf_b.store(spad_base + 18, or0[6])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 18, oi0[6])
            FFTFP32Kernel0.buf_b.store(spad_base + 19, or0[7])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 19, oi0[7])

            var twr1 = SIMD[DType.float32, W](Float32(1), Float32(1), Float32(1), Float32(1), Float32(0.923879533), Float32(0.923879533), Float32(0.923879533), Float32(0.923879533))
            var twi1 = SIMD[DType.float32, W](Float32(0), Float32(0), Float32(0), Float32(0), Float32(-0.382683432), Float32(-0.382683432), Float32(-0.382683432), Float32(-0.382683432))
            var tr1 = or1 * twr1 - oi1 * twi1
            var ti1 = or1 * twi1 + oi1 * twr1
            or1 = tr1
            oi1 = ti1
            FFTFP32Kernel0.buf_b.store(spad_base + 4, or1[0])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 4, oi1[0])
            FFTFP32Kernel0.buf_b.store(spad_base + 5, or1[1])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 5, oi1[1])
            FFTFP32Kernel0.buf_b.store(spad_base + 6, or1[2])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 6, oi1[2])
            FFTFP32Kernel0.buf_b.store(spad_base + 7, or1[3])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 7, oi1[3])
            FFTFP32Kernel0.buf_b.store(spad_base + 20, or1[4])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 20, oi1[4])
            FFTFP32Kernel0.buf_b.store(spad_base + 21, or1[5])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 21, oi1[5])
            FFTFP32Kernel0.buf_b.store(spad_base + 22, or1[6])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 22, oi1[6])
            FFTFP32Kernel0.buf_b.store(spad_base + 23, or1[7])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 23, oi1[7])

            var twr2 = SIMD[DType.float32, W](Float32(1), Float32(1), Float32(1), Float32(1), Float32(0.707106781), Float32(0.707106781), Float32(0.707106781), Float32(0.707106781))
            var twi2 = SIMD[DType.float32, W](Float32(0), Float32(0), Float32(0), Float32(0), Float32(-0.707106781), Float32(-0.707106781), Float32(-0.707106781), Float32(-0.707106781))
            var tr2 = or2 * twr2 - oi2 * twi2
            var ti2 = or2 * twi2 + oi2 * twr2
            or2 = tr2
            oi2 = ti2
            FFTFP32Kernel0.buf_b.store(spad_base + 8, or2[0])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 8, oi2[0])
            FFTFP32Kernel0.buf_b.store(spad_base + 9, or2[1])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 9, oi2[1])
            FFTFP32Kernel0.buf_b.store(spad_base + 10, or2[2])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 10, oi2[2])
            FFTFP32Kernel0.buf_b.store(spad_base + 11, or2[3])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 11, oi2[3])
            FFTFP32Kernel0.buf_b.store(spad_base + 24, or2[4])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 24, oi2[4])
            FFTFP32Kernel0.buf_b.store(spad_base + 25, or2[5])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 25, oi2[5])
            FFTFP32Kernel0.buf_b.store(spad_base + 26, or2[6])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 26, oi2[6])
            FFTFP32Kernel0.buf_b.store(spad_base + 27, or2[7])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 27, oi2[7])

            var twr3 = SIMD[DType.float32, W](Float32(1), Float32(1), Float32(1), Float32(1), Float32(0.382683432), Float32(0.382683432), Float32(0.382683432), Float32(0.382683432))
            var twi3 = SIMD[DType.float32, W](Float32(0), Float32(0), Float32(0), Float32(0), Float32(-0.923879533), Float32(-0.923879533), Float32(-0.923879533), Float32(-0.923879533))
            var tr3 = or3 * twr3 - oi3 * twi3
            var ti3 = or3 * twi3 + oi3 * twr3
            or3 = tr3
            oi3 = ti3
            FFTFP32Kernel0.buf_b.store(spad_base + 12, or3[0])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 12, oi3[0])
            FFTFP32Kernel0.buf_b.store(spad_base + 13, or3[1])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 13, oi3[1])
            FFTFP32Kernel0.buf_b.store(spad_base + 14, or3[2])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 14, oi3[2])
            FFTFP32Kernel0.buf_b.store(spad_base + 15, or3[3])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 15, oi3[3])
            FFTFP32Kernel0.buf_b.store(spad_base + 28, or3[4])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 28, oi3[4])
            FFTFP32Kernel0.buf_b.store(spad_base + 29, or3[5])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 29, oi3[5])
            FFTFP32Kernel0.buf_b.store(spad_base + 30, or3[6])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 30, oi3[6])
            FFTFP32Kernel0.buf_b.store(spad_base + 31, or3[7])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 31, oi3[7])

        if True:  # batch 1 scope
            # ===== stage 1, SIMD batch 1 (valid lanes: 8/8) =====
            var rr0 = FFTFP32Kernel0.buf_a.load[DType.float32, W](spad_base + 8)
            var ii0 = FFTFP32Kernel0.buf_a.load[DType.float32, W](spad_base + N + 8)
            var rr1 = FFTFP32Kernel0.buf_a.load[DType.float32, W](spad_base + 24)
            var ii1 = FFTFP32Kernel0.buf_a.load[DType.float32, W](spad_base + N + 24)
            var rr2 = FFTFP32Kernel0.buf_a.load[DType.float32, W](spad_base + 40)
            var ii2 = FFTFP32Kernel0.buf_a.load[DType.float32, W](spad_base + N + 40)
            var rr3 = FFTFP32Kernel0.buf_a.load[DType.float32, W](spad_base + 56)
            var ii3 = FFTFP32Kernel0.buf_a.load[DType.float32, W](spad_base + N + 56)

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

            FFTFP32Kernel0.buf_b.store(spad_base + 32, or0[0])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 32, oi0[0])
            FFTFP32Kernel0.buf_b.store(spad_base + 33, or0[1])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 33, oi0[1])
            FFTFP32Kernel0.buf_b.store(spad_base + 34, or0[2])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 34, oi0[2])
            FFTFP32Kernel0.buf_b.store(spad_base + 35, or0[3])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 35, oi0[3])
            FFTFP32Kernel0.buf_b.store(spad_base + 48, or0[4])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 48, oi0[4])
            FFTFP32Kernel0.buf_b.store(spad_base + 49, or0[5])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 49, oi0[5])
            FFTFP32Kernel0.buf_b.store(spad_base + 50, or0[6])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 50, oi0[6])
            FFTFP32Kernel0.buf_b.store(spad_base + 51, or0[7])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 51, oi0[7])

            var twr1 = SIMD[DType.float32, W](Float32(0.707106781), Float32(0.707106781), Float32(0.707106781), Float32(0.707106781), Float32(0.382683432), Float32(0.382683432), Float32(0.382683432), Float32(0.382683432))
            var twi1 = SIMD[DType.float32, W](Float32(-0.707106781), Float32(-0.707106781), Float32(-0.707106781), Float32(-0.707106781), Float32(-0.923879533), Float32(-0.923879533), Float32(-0.923879533), Float32(-0.923879533))
            var tr1 = or1 * twr1 - oi1 * twi1
            var ti1 = or1 * twi1 + oi1 * twr1
            or1 = tr1
            oi1 = ti1
            FFTFP32Kernel0.buf_b.store(spad_base + 36, or1[0])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 36, oi1[0])
            FFTFP32Kernel0.buf_b.store(spad_base + 37, or1[1])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 37, oi1[1])
            FFTFP32Kernel0.buf_b.store(spad_base + 38, or1[2])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 38, oi1[2])
            FFTFP32Kernel0.buf_b.store(spad_base + 39, or1[3])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 39, oi1[3])
            FFTFP32Kernel0.buf_b.store(spad_base + 52, or1[4])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 52, oi1[4])
            FFTFP32Kernel0.buf_b.store(spad_base + 53, or1[5])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 53, oi1[5])
            FFTFP32Kernel0.buf_b.store(spad_base + 54, or1[6])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 54, oi1[6])
            FFTFP32Kernel0.buf_b.store(spad_base + 55, or1[7])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 55, oi1[7])

            var twr2 = SIMD[DType.float32, W](Float32(0), Float32(0), Float32(0), Float32(0), Float32(-0.707106781), Float32(-0.707106781), Float32(-0.707106781), Float32(-0.707106781))
            var twi2 = SIMD[DType.float32, W](Float32(-1), Float32(-1), Float32(-1), Float32(-1), Float32(-0.707106781), Float32(-0.707106781), Float32(-0.707106781), Float32(-0.707106781))
            var tr2 = or2 * twr2 - oi2 * twi2
            var ti2 = or2 * twi2 + oi2 * twr2
            or2 = tr2
            oi2 = ti2
            FFTFP32Kernel0.buf_b.store(spad_base + 40, or2[0])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 40, oi2[0])
            FFTFP32Kernel0.buf_b.store(spad_base + 41, or2[1])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 41, oi2[1])
            FFTFP32Kernel0.buf_b.store(spad_base + 42, or2[2])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 42, oi2[2])
            FFTFP32Kernel0.buf_b.store(spad_base + 43, or2[3])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 43, oi2[3])
            FFTFP32Kernel0.buf_b.store(spad_base + 56, or2[4])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 56, oi2[4])
            FFTFP32Kernel0.buf_b.store(spad_base + 57, or2[5])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 57, oi2[5])
            FFTFP32Kernel0.buf_b.store(spad_base + 58, or2[6])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 58, oi2[6])
            FFTFP32Kernel0.buf_b.store(spad_base + 59, or2[7])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 59, oi2[7])

            var twr3 = SIMD[DType.float32, W](Float32(-0.707106781), Float32(-0.707106781), Float32(-0.707106781), Float32(-0.707106781), Float32(-0.923879533), Float32(-0.923879533), Float32(-0.923879533), Float32(-0.923879533))
            var twi3 = SIMD[DType.float32, W](Float32(-0.707106781), Float32(-0.707106781), Float32(-0.707106781), Float32(-0.707106781), Float32(0.382683432), Float32(0.382683432), Float32(0.382683432), Float32(0.382683432))
            var tr3 = or3 * twr3 - oi3 * twi3
            var ti3 = or3 * twi3 + oi3 * twr3
            or3 = tr3
            oi3 = ti3
            FFTFP32Kernel0.buf_b.store(spad_base + 44, or3[0])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 44, oi3[0])
            FFTFP32Kernel0.buf_b.store(spad_base + 45, or3[1])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 45, oi3[1])
            FFTFP32Kernel0.buf_b.store(spad_base + 46, or3[2])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 46, oi3[2])
            FFTFP32Kernel0.buf_b.store(spad_base + 47, or3[3])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 47, oi3[3])
            FFTFP32Kernel0.buf_b.store(spad_base + 60, or3[4])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 60, oi3[4])
            FFTFP32Kernel0.buf_b.store(spad_base + 61, or3[5])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 61, oi3[5])
            FFTFP32Kernel0.buf_b.store(spad_base + 62, or3[6])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 62, oi3[6])
            FFTFP32Kernel0.buf_b.store(spad_base + 63, or3[7])
            FFTFP32Kernel0.buf_b.store(spad_base + N + 63, oi3[7])

    @staticmethod
    def stage_2():
        ref p = FFTFP32Kernel0.params[]
        comptime RADIX = 4
        comptime SIMD_ITERS = 2

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel0:
            return
        var spad_base = local_id * 128
        var out_batch_base = (global_uthread_id() % 1) + (global_uthread_id() // 1) * 64

        if True:  # batch 0 scope
            # ===== stage 2, SIMD batch 0 (valid lanes: 8/8) =====
            var rr0 = FFTFP32Kernel0.buf_b.load[DType.float32, W](spad_base + 0)
            var ii0 = FFTFP32Kernel0.buf_b.load[DType.float32, W](spad_base + N + 0)
            var rr1 = FFTFP32Kernel0.buf_b.load[DType.float32, W](spad_base + 16)
            var ii1 = FFTFP32Kernel0.buf_b.load[DType.float32, W](spad_base + N + 16)
            var rr2 = FFTFP32Kernel0.buf_b.load[DType.float32, W](spad_base + 32)
            var ii2 = FFTFP32Kernel0.buf_b.load[DType.float32, W](spad_base + N + 32)
            var rr3 = FFTFP32Kernel0.buf_b.load[DType.float32, W](spad_base + 48)
            var ii3 = FFTFP32Kernel0.buf_b.load[DType.float32, W](spad_base + N + 48)

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

            var ltwr0 = p.large_twiddle_real_base.load[width=W](out_batch_base + 0)
            var ltwi0 = p.large_twiddle_imag_base.load[width=W](out_batch_base + 0)
            var ltr0 = or0 * ltwr0 - oi0 * ltwi0
            var lti0 = or0 * ltwi0 + oi0 * ltwr0
            or0 = ltr0
            oi0 = lti0
            p.output_real_base.store(out_batch_base + 0, or0)
            p.output_imag_base.store(out_batch_base + 0, oi0)

            var ltwr1 = p.large_twiddle_real_base.load[width=W](out_batch_base + 16)
            var ltwi1 = p.large_twiddle_imag_base.load[width=W](out_batch_base + 16)
            var ltr1 = or1 * ltwr1 - oi1 * ltwi1
            var lti1 = or1 * ltwi1 + oi1 * ltwr1
            or1 = ltr1
            oi1 = lti1
            p.output_real_base.store(out_batch_base + 16, or1)
            p.output_imag_base.store(out_batch_base + 16, oi1)

            var ltwr2 = p.large_twiddle_real_base.load[width=W](out_batch_base + 32)
            var ltwi2 = p.large_twiddle_imag_base.load[width=W](out_batch_base + 32)
            var ltr2 = or2 * ltwr2 - oi2 * ltwi2
            var lti2 = or2 * ltwi2 + oi2 * ltwr2
            or2 = ltr2
            oi2 = lti2
            p.output_real_base.store(out_batch_base + 32, or2)
            p.output_imag_base.store(out_batch_base + 32, oi2)

            var ltwr3 = p.large_twiddle_real_base.load[width=W](out_batch_base + 48)
            var ltwi3 = p.large_twiddle_imag_base.load[width=W](out_batch_base + 48)
            var ltr3 = or3 * ltwr3 - oi3 * ltwi3
            var lti3 = or3 * ltwi3 + oi3 * ltwr3
            or3 = ltr3
            oi3 = lti3
            p.output_real_base.store(out_batch_base + 48, or3)
            p.output_imag_base.store(out_batch_base + 48, oi3)

        if True:  # batch 1 scope
            # ===== stage 2, SIMD batch 1 (valid lanes: 8/8) =====
            var rr0 = FFTFP32Kernel0.buf_b.load[DType.float32, W](spad_base + 8)
            var ii0 = FFTFP32Kernel0.buf_b.load[DType.float32, W](spad_base + N + 8)
            var rr1 = FFTFP32Kernel0.buf_b.load[DType.float32, W](spad_base + 24)
            var ii1 = FFTFP32Kernel0.buf_b.load[DType.float32, W](spad_base + N + 24)
            var rr2 = FFTFP32Kernel0.buf_b.load[DType.float32, W](spad_base + 40)
            var ii2 = FFTFP32Kernel0.buf_b.load[DType.float32, W](spad_base + N + 40)
            var rr3 = FFTFP32Kernel0.buf_b.load[DType.float32, W](spad_base + 56)
            var ii3 = FFTFP32Kernel0.buf_b.load[DType.float32, W](spad_base + N + 56)

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

            var ltwr0 = p.large_twiddle_real_base.load[width=W](out_batch_base + 8)
            var ltwi0 = p.large_twiddle_imag_base.load[width=W](out_batch_base + 8)
            var ltr0 = or0 * ltwr0 - oi0 * ltwi0
            var lti0 = or0 * ltwi0 + oi0 * ltwr0
            or0 = ltr0
            oi0 = lti0
            p.output_real_base.store(out_batch_base + 8, or0)
            p.output_imag_base.store(out_batch_base + 8, oi0)

            var ltwr1 = p.large_twiddle_real_base.load[width=W](out_batch_base + 24)
            var ltwi1 = p.large_twiddle_imag_base.load[width=W](out_batch_base + 24)
            var ltr1 = or1 * ltwr1 - oi1 * ltwi1
            var lti1 = or1 * ltwi1 + oi1 * ltwr1
            or1 = ltr1
            oi1 = lti1
            p.output_real_base.store(out_batch_base + 24, or1)
            p.output_imag_base.store(out_batch_base + 24, oi1)

            var ltwr2 = p.large_twiddle_real_base.load[width=W](out_batch_base + 40)
            var ltwi2 = p.large_twiddle_imag_base.load[width=W](out_batch_base + 40)
            var ltr2 = or2 * ltwr2 - oi2 * ltwi2
            var lti2 = or2 * ltwi2 + oi2 * ltwr2
            or2 = ltr2
            oi2 = lti2
            p.output_real_base.store(out_batch_base + 40, or2)
            p.output_imag_base.store(out_batch_base + 40, oi2)

            var ltwr3 = p.large_twiddle_real_base.load[width=W](out_batch_base + 56)
            var ltwi3 = p.large_twiddle_imag_base.load[width=W](out_batch_base + 56)
            var ltr3 = or3 * ltwr3 - oi3 * ltwi3
            var lti3 = or3 * ltwi3 + oi3 * ltwr3
            or3 = ltr3
            oi3 = lti3
            p.output_real_base.store(out_batch_base + 56, or3)
            p.output_imag_base.store(out_batch_base + 56, oi3)

    @staticmethod
    def device_main():
        launch_parallel[FFTFP32Kernel0.stage_0]()
        launch_parallel[FFTFP32Kernel0.stage_1]()
        launch_parallel[FFTFP32Kernel0.stage_2]()


comptime MAX_UTHREAD_FFTFP32Kernel1 = 320

@fieldwise_init
struct FFTFP32Kernel1Params(Movable):
    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var large_twiddle_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var large_twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]


struct FFTFP32Kernel1(NDPTask):
    comptime Params = FFTFP32Kernel1Params

    @staticmethod
    def stage_0():
        ref p = FFTFP32Kernel1.params[]
        comptime RADIX = 3
        comptime SIMD_ITERS = 1

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel1:
            return
        var in_batch_base = global_uthread_id() * 1
        var out_batch_base = (global_uthread_id() % 64) + (global_uthread_id() // 64) * 192

        # ===== stage 0, SIMD batch 0 (valid lanes: 1/8) =====
        var rr0_lane0 = p.input_real_base.load[width=1](in_batch_base + 0)
        var ii0_lane0 = p.input_imag_base.load[width=1](in_batch_base + 0)
        var rr0 = SIMD[DType.float32, W](rr0_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii0 = SIMD[DType.float32, W](ii0_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr1_lane0 = p.input_real_base.load[width=1](in_batch_base + 320)
        var ii1_lane0 = p.input_imag_base.load[width=1](in_batch_base + 320)
        var rr1 = SIMD[DType.float32, W](rr1_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii1 = SIMD[DType.float32, W](ii1_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr2_lane0 = p.input_real_base.load[width=1](in_batch_base + 640)
        var ii2_lane0 = p.input_imag_base.load[width=1](in_batch_base + 640)
        var rr2 = SIMD[DType.float32, W](rr2_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii2 = SIMD[DType.float32, W](ii2_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))

        # fixed radix-3 butterfly
        var osum12r = rr1 + rr2
        var osum12i = ii1 + ii2
        var odiff12r = rr1 - rr2
        var odiff12i = ii1 - ii2
        var obaser = rr0 - osum12r * Float32(0.5)
        var obasei = ii0 - osum12i * Float32(0.5)
        var or0 = rr0 + osum12r
        var oi0 = ii0 + osum12i
        var or1 = obaser + odiff12i * Float32(0.866025404)
        var oi1 = obasei - odiff12r * Float32(0.866025404)
        var or2 = obaser - odiff12i * Float32(0.866025404)
        var oi2 = obasei + odiff12r * Float32(0.866025404)

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

        var ltwr1_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 64)
        var ltwi1_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 64)
        var ltwr1 = SIMD[DType.float32, W](ltwr1_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi1 = SIMD[DType.float32, W](ltwi1_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr1 = or1 * ltwr1 - oi1 * ltwi1
        var lti1 = or1 * ltwi1 + oi1 * ltwr1
        or1 = ltr1
        oi1 = lti1
        p.output_real_base.store(out_batch_base + 64, or1[0])
        p.output_imag_base.store(out_batch_base + 64, oi1[0])

        var ltwr2_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 128)
        var ltwi2_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 128)
        var ltwr2 = SIMD[DType.float32, W](ltwr2_lane0[0], Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var ltwi2 = SIMD[DType.float32, W](ltwi2_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ltr2 = or2 * ltwr2 - oi2 * ltwi2
        var lti2 = or2 * ltwi2 + oi2 * ltwr2
        or2 = ltr2
        oi2 = lti2
        p.output_real_base.store(out_batch_base + 128, or2[0])
        p.output_imag_base.store(out_batch_base + 128, oi2[0])

    @staticmethod
    def device_main():
        launch_parallel[FFTFP32Kernel1.stage_0]()


comptime MAX_UTHREAD_FFTFP32Kernel2 = 192

@fieldwise_init
struct FFTFP32Kernel2Params(Movable):
    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]


struct FFTFP32Kernel2(NDPTask):
    comptime Params = FFTFP32Kernel2Params

    @staticmethod
    def stage_0():
        ref p = FFTFP32Kernel2.params[]
        comptime RADIX = 5
        comptime SIMD_ITERS = 1

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel2:
            return
        var in_batch_base = global_uthread_id() * 1
        var out_batch_base = global_uthread_id() * 1

        # ===== stage 0, SIMD batch 0 (valid lanes: 1/8) =====
        var rr0_lane0 = p.input_real_base.load[width=1](in_batch_base + 0)
        var ii0_lane0 = p.input_imag_base.load[width=1](in_batch_base + 0)
        var rr0 = SIMD[DType.float32, W](rr0_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii0 = SIMD[DType.float32, W](ii0_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr1_lane0 = p.input_real_base.load[width=1](in_batch_base + 192)
        var ii1_lane0 = p.input_imag_base.load[width=1](in_batch_base + 192)
        var rr1 = SIMD[DType.float32, W](rr1_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii1 = SIMD[DType.float32, W](ii1_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr2_lane0 = p.input_real_base.load[width=1](in_batch_base + 384)
        var ii2_lane0 = p.input_imag_base.load[width=1](in_batch_base + 384)
        var rr2 = SIMD[DType.float32, W](rr2_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii2 = SIMD[DType.float32, W](ii2_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr3_lane0 = p.input_real_base.load[width=1](in_batch_base + 576)
        var ii3_lane0 = p.input_imag_base.load[width=1](in_batch_base + 576)
        var rr3 = SIMD[DType.float32, W](rr3_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii3 = SIMD[DType.float32, W](ii3_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var rr4_lane0 = p.input_real_base.load[width=1](in_batch_base + 768)
        var ii4_lane0 = p.input_imag_base.load[width=1](in_batch_base + 768)
        var rr4 = SIMD[DType.float32, W](rr4_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var ii4 = SIMD[DType.float32, W](ii4_lane0[0], Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))

        # fixed radix-5 butterfly
        var os14r = rr1 + rr4
        var os14i = ii1 + ii4
        var os23r = rr2 + rr3
        var os23i = ii2 + ii3
        var od14r = rr1 - rr4
        var od14i = ii1 - ii4
        var od23r = rr2 - rr3
        var od23i = ii2 - ii3
        var or0 = rr0 + os14r + os23r
        var oi0 = ii0 + os14i + os23i
        var ob1r = rr0 - os23r * Float32(0.5) + ((rr1 - rr2) + (rr4 - rr3)) * Float32(0.309016994)
        var ob1i = ii0 - os23i * Float32(0.5) + ((ii1 - ii2) + (ii4 - ii3)) * Float32(0.309016994)
        var ob2r = rr0 - os14r * Float32(0.5) + ((rr2 - rr1) + (rr3 - rr4)) * Float32(0.309016994)
        var ob2i = ii0 - os14i * Float32(0.5) + ((ii2 - ii1) + (ii3 - ii4)) * Float32(0.309016994)
        var oc1r = od14i * Float32(0.951056516) + od23i * Float32(0.587785252)
        var oc1i = od14r * Float32(0.951056516) + od23r * Float32(0.587785252)
        var oc2r = -od23i * Float32(0.951056516) + od14i * Float32(0.587785252)
        var oc2i = -od23r * Float32(0.951056516) + od14r * Float32(0.587785252)
        var or1 = ob1r + oc1r
        var oi1 = ob1i - oc1i
        var or4 = ob1r - oc1r
        var oi4 = ob1i + oc1i
        var or2 = ob2r + oc2r
        var oi2 = ob2i - oc2i
        var or3 = ob2r - oc2r
        var oi3 = ob2i + oc2i

        p.output_real_base.store(out_batch_base + 0, or0[0])
        p.output_imag_base.store(out_batch_base + 0, oi0[0])

        p.output_real_base.store(out_batch_base + 192, or1[0])
        p.output_imag_base.store(out_batch_base + 192, oi1[0])

        p.output_real_base.store(out_batch_base + 384, or2[0])
        p.output_imag_base.store(out_batch_base + 384, oi2[0])

        p.output_real_base.store(out_batch_base + 576, or3[0])
        p.output_imag_base.store(out_batch_base + 576, oi3[0])

        p.output_real_base.store(out_batch_base + 768, or4[0])
        p.output_imag_base.store(out_batch_base + 768, oi4[0])

    @staticmethod
    def device_main():
        launch_parallel[FFTFP32Kernel2.stage_0]()


def main() raises:
    if FFTFP32Kernel0.emit_ir_if_asked():
        return

    var n = 960
    var input_real = cxl_alloc[Float32](n)
    var input_imag = cxl_alloc[Float32](n)
    var mid0_real = cxl_alloc[Float32](n)
    var mid0_imag = cxl_alloc[Float32](n)
    var mid1_real = cxl_alloc[Float32](n)
    var mid1_imag = cxl_alloc[Float32](n)
    var output_real = cxl_alloc[Float32](n)
    var output_imag = cxl_alloc[Float32](n)
    var ref_real = cxl_alloc[Float32](n)
    var ref_imag = cxl_alloc[Float32](n)

    var large_twiddle0_real = cxl_alloc[Float32](n)
    var large_twiddle0_imag = cxl_alloc[Float32](n)
    var lt_pi_0 = Float64(3.141592653589793)
    var lt_sign_0 = Float64(-1.0)
    var lt_r_0 = 0
    while lt_r_0 < 15:
        var lt_c1_0 = 0
        while lt_c1_0 < 64:
            var lt_angle_0 = lt_sign_0 * 2.0 * lt_pi_0 * Float64(lt_r_0) * Float64(lt_c1_0) * Float64(1) / Float64(960)
            var lt_val_r_0 = Float32(host_cos(lt_angle_0))
            var lt_val_i_0 = Float32(host_sin(lt_angle_0))
            large_twiddle0_real[lt_r_0 * 64 + lt_c1_0] = lt_val_r_0
            large_twiddle0_imag[lt_r_0 * 64 + lt_c1_0] = lt_val_i_0
            lt_c1_0 += 1
        lt_r_0 += 1

    var large_twiddle1_real = cxl_alloc[Float32](n)
    var large_twiddle1_imag = cxl_alloc[Float32](n)
    var lt_pi_1 = Float64(3.141592653589793)
    var lt_sign_1 = Float64(-1.0)
    var lt_r_1 = 0
    while lt_r_1 < 5:
        var lt_c1_1 = 0
        while lt_c1_1 < 3:
            var lt_angle_1 = lt_sign_1 * 2.0 * lt_pi_1 * Float64(lt_r_1) * Float64(lt_c1_1) * Float64(64) / Float64(960)
            var lt_val_r_1 = Float32(host_cos(lt_angle_1))
            var lt_val_i_1 = Float32(host_sin(lt_angle_1))
            var lt_out_a_1 = 0
            while lt_out_a_1 < 64:
                var lt_addr_1 = lt_out_a_1 + lt_c1_1 * 64 + lt_r_1 * 192
                large_twiddle1_real[lt_addr_1] = lt_val_r_1
                large_twiddle1_imag[lt_addr_1] = lt_val_i_1
                lt_out_a_1 += 1
            lt_c1_1 += 1
        lt_r_1 += 1

    var pool0_elems = 120
    var pool0 = cxl_alloc[Float32](pool0_elems)
    var pool1_elems = 2560
    var pool1 = cxl_alloc[Float32](pool1_elems)
    var pool2_elems = 1536
    var pool2 = cxl_alloc[Float32](pool2_elems)

    seed(0)
    for i in range(n):
        input_real[i] = Float32(random_float64(-1.0, 1.0))
        input_imag[i] = Float32(random_float64(-1.0, 1.0))
        mid0_real[i] = Float32(0)
        mid0_imag[i] = Float32(0)
        mid1_real[i] = Float32(0)
        mid1_imag[i] = Float32(0)
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
        FFTFP32Kernel1Params(mid0_real, mid0_imag, mid1_real, mid1_imag, large_twiddle1_real, large_twiddle1_imag),
    )
    if rc1 != 0:
        print("[host] FFT kernel1 failed, exit", rc1)
        return

    var rc2 = FFTFP32Kernel2.launch(
        PooledRange.over(pool2, pool2_elems),
        FFTFP32Kernel2Params(mid1_real, mid1_imag, output_real, output_imag),
    )
    if rc2 != 0:
        print("[host] FFT kernel2 failed, exit", rc2)
        return

    var pi = Float64(3.141592653589793)
    var sign = Float64(-1.0)
    for batch in range(1):
        var batch_base = batch * 960
        for k in range(960):
            var acc_r = Float64(0)
            var acc_i = Float64(0)
            for n_ in range(960):
                var angle = sign * 2.0 * pi * Float64(n_) * Float64(k) / Float64(960)
                var c = host_cos(angle)
                var s = host_sin(angle)
                var xr = Float64(input_real[batch_base + n_])
                var xi = Float64(input_imag[batch_base + n_])
                acc_r += xr * c - xi * s
                acc_i += xr * s + xi * c
            ref_real[batch_base + k] = Float32(acc_r)
            ref_imag[batch_base + k] = Float32(acc_i)

    var tol = Float32(0.001)
    for i in range(960):
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
