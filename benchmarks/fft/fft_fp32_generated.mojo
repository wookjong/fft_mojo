from std.sys import size_of

from m2ndp import VECTOR_WIDTH, NDPTask, PooledRange, global_uthread_id, local_uthread_id, launch_parallel, scratchpad
from m2ndp_host import cxl_alloc

comptime W = VECTOR_WIDTH // size_of[Float32]()
comptime N = 64
comptime MAX_UTHREAD = 1

@fieldwise_init
struct FFTFP32Params(Movable):
    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]


struct FFTFP32(NDPTask):
    comptime Params = FFTFP32Params

    comptime buf = scratchpad[2 * N * MAX_UTHREAD, Float32, name="fft_buffer"]()

    @staticmethod
    def stage_0():
        ref p = FFTFP32.params[]
        comptime RADIX = 8
        comptime SIMD_ITERS = 1
        comptime INPUT_STRIDE = 8
        comptime OUTPUT_STRIDE = 8

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD:
            return
        var spad_base = local_id * (2 * N)
        var batch_base = global_uthread_id() * N

        # ===== stage 0, SIMD batch 0 =====
        var rr0 = p.input_real_base.load[width=W](batch_base + 0)
        var ii0 = p.input_imag_base.load[width=W](batch_base + 0)
        var rr1 = p.input_real_base.load[width=W](batch_base + 8)
        var ii1 = p.input_imag_base.load[width=W](batch_base + 8)
        var rr2 = p.input_real_base.load[width=W](batch_base + 16)
        var ii2 = p.input_imag_base.load[width=W](batch_base + 16)
        var rr3 = p.input_real_base.load[width=W](batch_base + 24)
        var ii3 = p.input_imag_base.load[width=W](batch_base + 24)
        var rr4 = p.input_real_base.load[width=W](batch_base + 32)
        var ii4 = p.input_imag_base.load[width=W](batch_base + 32)
        var rr5 = p.input_real_base.load[width=W](batch_base + 40)
        var ii5 = p.input_imag_base.load[width=W](batch_base + 40)
        var rr6 = p.input_real_base.load[width=W](batch_base + 48)
        var ii6 = p.input_imag_base.load[width=W](batch_base + 48)
        var rr7 = p.input_real_base.load[width=W](batch_base + 56)
        var ii7 = p.input_imag_base.load[width=W](batch_base + 56)

        # radix-8 output 0
        var or0 = SIMD[DType.float32, W](0)
        var oi0 = SIMD[DType.float32, W](0)
        or0 += rr0
        oi0 += ii0
        or0 += rr1
        oi0 += ii1
        or0 += rr2
        oi0 += ii2
        or0 += rr3
        oi0 += ii3
        or0 += rr4
        oi0 += ii4
        or0 += rr5
        oi0 += ii5
        or0 += rr6
        oi0 += ii6
        or0 += rr7
        oi0 += ii7
        var twr0 = SIMD[DType.float32, W](Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var twi0 = SIMD[DType.float32, W](Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var tr0 = or0 * twr0 - oi0 * twi0
        var ti0 = or0 * twi0 + oi0 * twr0
        or0 = tr0
        oi0 = ti0
        FFTFP32.buf.store(spad_base + 0, or0[0])
        FFTFP32.buf.store(spad_base + N + 0, oi0[0])
        FFTFP32.buf.store(spad_base + 8, or0[1])
        FFTFP32.buf.store(spad_base + N + 8, oi0[1])
        FFTFP32.buf.store(spad_base + 16, or0[2])
        FFTFP32.buf.store(spad_base + N + 16, oi0[2])
        FFTFP32.buf.store(spad_base + 24, or0[3])
        FFTFP32.buf.store(spad_base + N + 24, oi0[3])
        FFTFP32.buf.store(spad_base + 32, or0[4])
        FFTFP32.buf.store(spad_base + N + 32, oi0[4])
        FFTFP32.buf.store(spad_base + 40, or0[5])
        FFTFP32.buf.store(spad_base + N + 40, oi0[5])
        FFTFP32.buf.store(spad_base + 48, or0[6])
        FFTFP32.buf.store(spad_base + N + 48, oi0[6])
        FFTFP32.buf.store(spad_base + 56, or0[7])
        FFTFP32.buf.store(spad_base + N + 56, oi0[7])

        # radix-8 output 1
        var or1 = SIMD[DType.float32, W](0)
        var oi1 = SIMD[DType.float32, W](0)
        or1 += rr0
        oi1 += ii0
        or1 += rr1 * Float32(0.707106781) - ii1 * Float32(-0.707106781)
        oi1 += rr1 * Float32(-0.707106781) + ii1 * Float32(0.707106781)
        or1 += rr2 * Float32(0) - ii2 * Float32(-1)
        oi1 += rr2 * Float32(-1) + ii2 * Float32(0)
        or1 += rr3 * Float32(-0.707106781) - ii3 * Float32(-0.707106781)
        oi1 += rr3 * Float32(-0.707106781) + ii3 * Float32(-0.707106781)
        or1 -= rr4
        oi1 -= ii4
        or1 += rr5 * Float32(-0.707106781) - ii5 * Float32(0.707106781)
        oi1 += rr5 * Float32(0.707106781) + ii5 * Float32(-0.707106781)
        or1 += rr6 * Float32(0) - ii6 * Float32(1)
        oi1 += rr6 * Float32(1) + ii6 * Float32(0)
        or1 += rr7 * Float32(0.707106781) - ii7 * Float32(0.707106781)
        oi1 += rr7 * Float32(0.707106781) + ii7 * Float32(0.707106781)
        var twr1 = SIMD[DType.float32, W](Float32(1), Float32(0.995184727), Float32(0.98078528), Float32(0.956940336), Float32(0.923879533), Float32(0.881921264), Float32(0.831469612), Float32(0.773010453))
        var twi1 = SIMD[DType.float32, W](Float32(0), Float32(-0.0980171403), Float32(-0.195090322), Float32(-0.290284677), Float32(-0.382683432), Float32(-0.471396737), Float32(-0.555570233), Float32(-0.634393284))
        var tr1 = or1 * twr1 - oi1 * twi1
        var ti1 = or1 * twi1 + oi1 * twr1
        or1 = tr1
        oi1 = ti1
        FFTFP32.buf.store(spad_base + 1, or1[0])
        FFTFP32.buf.store(spad_base + N + 1, oi1[0])
        FFTFP32.buf.store(spad_base + 9, or1[1])
        FFTFP32.buf.store(spad_base + N + 9, oi1[1])
        FFTFP32.buf.store(spad_base + 17, or1[2])
        FFTFP32.buf.store(spad_base + N + 17, oi1[2])
        FFTFP32.buf.store(spad_base + 25, or1[3])
        FFTFP32.buf.store(spad_base + N + 25, oi1[3])
        FFTFP32.buf.store(spad_base + 33, or1[4])
        FFTFP32.buf.store(spad_base + N + 33, oi1[4])
        FFTFP32.buf.store(spad_base + 41, or1[5])
        FFTFP32.buf.store(spad_base + N + 41, oi1[5])
        FFTFP32.buf.store(spad_base + 49, or1[6])
        FFTFP32.buf.store(spad_base + N + 49, oi1[6])
        FFTFP32.buf.store(spad_base + 57, or1[7])
        FFTFP32.buf.store(spad_base + N + 57, oi1[7])

        # radix-8 output 2
        var or2 = SIMD[DType.float32, W](0)
        var oi2 = SIMD[DType.float32, W](0)
        or2 += rr0
        oi2 += ii0
        or2 += rr1 * Float32(0) - ii1 * Float32(-1)
        oi2 += rr1 * Float32(-1) + ii1 * Float32(0)
        or2 -= rr2
        oi2 -= ii2
        or2 += rr3 * Float32(0) - ii3 * Float32(1)
        oi2 += rr3 * Float32(1) + ii3 * Float32(0)
        or2 += rr4
        oi2 += ii4
        or2 += rr5 * Float32(0) - ii5 * Float32(-1)
        oi2 += rr5 * Float32(-1) + ii5 * Float32(0)
        or2 -= rr6
        oi2 -= ii6
        or2 += rr7 * Float32(0) - ii7 * Float32(1)
        oi2 += rr7 * Float32(1) + ii7 * Float32(0)
        var twr2 = SIMD[DType.float32, W](Float32(1), Float32(0.98078528), Float32(0.923879533), Float32(0.831469612), Float32(0.707106781), Float32(0.555570233), Float32(0.382683432), Float32(0.195090322))
        var twi2 = SIMD[DType.float32, W](Float32(0), Float32(-0.195090322), Float32(-0.382683432), Float32(-0.555570233), Float32(-0.707106781), Float32(-0.831469612), Float32(-0.923879533), Float32(-0.98078528))
        var tr2 = or2 * twr2 - oi2 * twi2
        var ti2 = or2 * twi2 + oi2 * twr2
        or2 = tr2
        oi2 = ti2
        FFTFP32.buf.store(spad_base + 2, or2[0])
        FFTFP32.buf.store(spad_base + N + 2, oi2[0])
        FFTFP32.buf.store(spad_base + 10, or2[1])
        FFTFP32.buf.store(spad_base + N + 10, oi2[1])
        FFTFP32.buf.store(spad_base + 18, or2[2])
        FFTFP32.buf.store(spad_base + N + 18, oi2[2])
        FFTFP32.buf.store(spad_base + 26, or2[3])
        FFTFP32.buf.store(spad_base + N + 26, oi2[3])
        FFTFP32.buf.store(spad_base + 34, or2[4])
        FFTFP32.buf.store(spad_base + N + 34, oi2[4])
        FFTFP32.buf.store(spad_base + 42, or2[5])
        FFTFP32.buf.store(spad_base + N + 42, oi2[5])
        FFTFP32.buf.store(spad_base + 50, or2[6])
        FFTFP32.buf.store(spad_base + N + 50, oi2[6])
        FFTFP32.buf.store(spad_base + 58, or2[7])
        FFTFP32.buf.store(spad_base + N + 58, oi2[7])

        # radix-8 output 3
        var or3 = SIMD[DType.float32, W](0)
        var oi3 = SIMD[DType.float32, W](0)
        or3 += rr0
        oi3 += ii0
        or3 += rr1 * Float32(-0.707106781) - ii1 * Float32(-0.707106781)
        oi3 += rr1 * Float32(-0.707106781) + ii1 * Float32(-0.707106781)
        or3 += rr2 * Float32(0) - ii2 * Float32(1)
        oi3 += rr2 * Float32(1) + ii2 * Float32(0)
        or3 += rr3 * Float32(0.707106781) - ii3 * Float32(-0.707106781)
        oi3 += rr3 * Float32(-0.707106781) + ii3 * Float32(0.707106781)
        or3 -= rr4
        oi3 -= ii4
        or3 += rr5 * Float32(0.707106781) - ii5 * Float32(0.707106781)
        oi3 += rr5 * Float32(0.707106781) + ii5 * Float32(0.707106781)
        or3 += rr6 * Float32(0) - ii6 * Float32(-1)
        oi3 += rr6 * Float32(-1) + ii6 * Float32(0)
        or3 += rr7 * Float32(-0.707106781) - ii7 * Float32(0.707106781)
        oi3 += rr7 * Float32(0.707106781) + ii7 * Float32(-0.707106781)
        var twr3 = SIMD[DType.float32, W](Float32(1), Float32(0.956940336), Float32(0.831469612), Float32(0.634393284), Float32(0.382683432), Float32(0.0980171403), Float32(-0.195090322), Float32(-0.471396737))
        var twi3 = SIMD[DType.float32, W](Float32(0), Float32(-0.290284677), Float32(-0.555570233), Float32(-0.773010453), Float32(-0.923879533), Float32(-0.995184727), Float32(-0.98078528), Float32(-0.881921264))
        var tr3 = or3 * twr3 - oi3 * twi3
        var ti3 = or3 * twi3 + oi3 * twr3
        or3 = tr3
        oi3 = ti3
        FFTFP32.buf.store(spad_base + 3, or3[0])
        FFTFP32.buf.store(spad_base + N + 3, oi3[0])
        FFTFP32.buf.store(spad_base + 11, or3[1])
        FFTFP32.buf.store(spad_base + N + 11, oi3[1])
        FFTFP32.buf.store(spad_base + 19, or3[2])
        FFTFP32.buf.store(spad_base + N + 19, oi3[2])
        FFTFP32.buf.store(spad_base + 27, or3[3])
        FFTFP32.buf.store(spad_base + N + 27, oi3[3])
        FFTFP32.buf.store(spad_base + 35, or3[4])
        FFTFP32.buf.store(spad_base + N + 35, oi3[4])
        FFTFP32.buf.store(spad_base + 43, or3[5])
        FFTFP32.buf.store(spad_base + N + 43, oi3[5])
        FFTFP32.buf.store(spad_base + 51, or3[6])
        FFTFP32.buf.store(spad_base + N + 51, oi3[6])
        FFTFP32.buf.store(spad_base + 59, or3[7])
        FFTFP32.buf.store(spad_base + N + 59, oi3[7])

        # radix-8 output 4
        var or4 = SIMD[DType.float32, W](0)
        var oi4 = SIMD[DType.float32, W](0)
        or4 += rr0
        oi4 += ii0
        or4 -= rr1
        oi4 -= ii1
        or4 += rr2
        oi4 += ii2
        or4 -= rr3
        oi4 -= ii3
        or4 += rr4
        oi4 += ii4
        or4 -= rr5
        oi4 -= ii5
        or4 += rr6
        oi4 += ii6
        or4 -= rr7
        oi4 -= ii7
        var twr4 = SIMD[DType.float32, W](Float32(1), Float32(0.923879533), Float32(0.707106781), Float32(0.382683432), Float32(0), Float32(-0.382683432), Float32(-0.707106781), Float32(-0.923879533))
        var twi4 = SIMD[DType.float32, W](Float32(0), Float32(-0.382683432), Float32(-0.707106781), Float32(-0.923879533), Float32(-1), Float32(-0.923879533), Float32(-0.707106781), Float32(-0.382683432))
        var tr4 = or4 * twr4 - oi4 * twi4
        var ti4 = or4 * twi4 + oi4 * twr4
        or4 = tr4
        oi4 = ti4
        FFTFP32.buf.store(spad_base + 4, or4[0])
        FFTFP32.buf.store(spad_base + N + 4, oi4[0])
        FFTFP32.buf.store(spad_base + 12, or4[1])
        FFTFP32.buf.store(spad_base + N + 12, oi4[1])
        FFTFP32.buf.store(spad_base + 20, or4[2])
        FFTFP32.buf.store(spad_base + N + 20, oi4[2])
        FFTFP32.buf.store(spad_base + 28, or4[3])
        FFTFP32.buf.store(spad_base + N + 28, oi4[3])
        FFTFP32.buf.store(spad_base + 36, or4[4])
        FFTFP32.buf.store(spad_base + N + 36, oi4[4])
        FFTFP32.buf.store(spad_base + 44, or4[5])
        FFTFP32.buf.store(spad_base + N + 44, oi4[5])
        FFTFP32.buf.store(spad_base + 52, or4[6])
        FFTFP32.buf.store(spad_base + N + 52, oi4[6])
        FFTFP32.buf.store(spad_base + 60, or4[7])
        FFTFP32.buf.store(spad_base + N + 60, oi4[7])

        # radix-8 output 5
        var or5 = SIMD[DType.float32, W](0)
        var oi5 = SIMD[DType.float32, W](0)
        or5 += rr0
        oi5 += ii0
        or5 += rr1 * Float32(-0.707106781) - ii1 * Float32(0.707106781)
        oi5 += rr1 * Float32(0.707106781) + ii1 * Float32(-0.707106781)
        or5 += rr2 * Float32(0) - ii2 * Float32(-1)
        oi5 += rr2 * Float32(-1) + ii2 * Float32(0)
        or5 += rr3 * Float32(0.707106781) - ii3 * Float32(0.707106781)
        oi5 += rr3 * Float32(0.707106781) + ii3 * Float32(0.707106781)
        or5 -= rr4
        oi5 -= ii4
        or5 += rr5 * Float32(0.707106781) - ii5 * Float32(-0.707106781)
        oi5 += rr5 * Float32(-0.707106781) + ii5 * Float32(0.707106781)
        or5 += rr6 * Float32(0) - ii6 * Float32(1)
        oi5 += rr6 * Float32(1) + ii6 * Float32(0)
        or5 += rr7 * Float32(-0.707106781) - ii7 * Float32(-0.707106781)
        oi5 += rr7 * Float32(-0.707106781) + ii7 * Float32(-0.707106781)
        var twr5 = SIMD[DType.float32, W](Float32(1), Float32(0.881921264), Float32(0.555570233), Float32(0.0980171403), Float32(-0.382683432), Float32(-0.773010453), Float32(-0.98078528), Float32(-0.956940336))
        var twi5 = SIMD[DType.float32, W](Float32(0), Float32(-0.471396737), Float32(-0.831469612), Float32(-0.995184727), Float32(-0.923879533), Float32(-0.634393284), Float32(-0.195090322), Float32(0.290284677))
        var tr5 = or5 * twr5 - oi5 * twi5
        var ti5 = or5 * twi5 + oi5 * twr5
        or5 = tr5
        oi5 = ti5
        FFTFP32.buf.store(spad_base + 5, or5[0])
        FFTFP32.buf.store(spad_base + N + 5, oi5[0])
        FFTFP32.buf.store(spad_base + 13, or5[1])
        FFTFP32.buf.store(spad_base + N + 13, oi5[1])
        FFTFP32.buf.store(spad_base + 21, or5[2])
        FFTFP32.buf.store(spad_base + N + 21, oi5[2])
        FFTFP32.buf.store(spad_base + 29, or5[3])
        FFTFP32.buf.store(spad_base + N + 29, oi5[3])
        FFTFP32.buf.store(spad_base + 37, or5[4])
        FFTFP32.buf.store(spad_base + N + 37, oi5[4])
        FFTFP32.buf.store(spad_base + 45, or5[5])
        FFTFP32.buf.store(spad_base + N + 45, oi5[5])
        FFTFP32.buf.store(spad_base + 53, or5[6])
        FFTFP32.buf.store(spad_base + N + 53, oi5[6])
        FFTFP32.buf.store(spad_base + 61, or5[7])
        FFTFP32.buf.store(spad_base + N + 61, oi5[7])

        # radix-8 output 6
        var or6 = SIMD[DType.float32, W](0)
        var oi6 = SIMD[DType.float32, W](0)
        or6 += rr0
        oi6 += ii0
        or6 += rr1 * Float32(0) - ii1 * Float32(1)
        oi6 += rr1 * Float32(1) + ii1 * Float32(0)
        or6 -= rr2
        oi6 -= ii2
        or6 += rr3 * Float32(0) - ii3 * Float32(-1)
        oi6 += rr3 * Float32(-1) + ii3 * Float32(0)
        or6 += rr4
        oi6 += ii4
        or6 += rr5 * Float32(0) - ii5 * Float32(1)
        oi6 += rr5 * Float32(1) + ii5 * Float32(0)
        or6 -= rr6
        oi6 -= ii6
        or6 += rr7 * Float32(0) - ii7 * Float32(-1)
        oi6 += rr7 * Float32(-1) + ii7 * Float32(0)
        var twr6 = SIMD[DType.float32, W](Float32(1), Float32(0.831469612), Float32(0.382683432), Float32(-0.195090322), Float32(-0.707106781), Float32(-0.98078528), Float32(-0.923879533), Float32(-0.555570233))
        var twi6 = SIMD[DType.float32, W](Float32(0), Float32(-0.555570233), Float32(-0.923879533), Float32(-0.98078528), Float32(-0.707106781), Float32(-0.195090322), Float32(0.382683432), Float32(0.831469612))
        var tr6 = or6 * twr6 - oi6 * twi6
        var ti6 = or6 * twi6 + oi6 * twr6
        or6 = tr6
        oi6 = ti6
        FFTFP32.buf.store(spad_base + 6, or6[0])
        FFTFP32.buf.store(spad_base + N + 6, oi6[0])
        FFTFP32.buf.store(spad_base + 14, or6[1])
        FFTFP32.buf.store(spad_base + N + 14, oi6[1])
        FFTFP32.buf.store(spad_base + 22, or6[2])
        FFTFP32.buf.store(spad_base + N + 22, oi6[2])
        FFTFP32.buf.store(spad_base + 30, or6[3])
        FFTFP32.buf.store(spad_base + N + 30, oi6[3])
        FFTFP32.buf.store(spad_base + 38, or6[4])
        FFTFP32.buf.store(spad_base + N + 38, oi6[4])
        FFTFP32.buf.store(spad_base + 46, or6[5])
        FFTFP32.buf.store(spad_base + N + 46, oi6[5])
        FFTFP32.buf.store(spad_base + 54, or6[6])
        FFTFP32.buf.store(spad_base + N + 54, oi6[6])
        FFTFP32.buf.store(spad_base + 62, or6[7])
        FFTFP32.buf.store(spad_base + N + 62, oi6[7])

        # radix-8 output 7
        var or7 = SIMD[DType.float32, W](0)
        var oi7 = SIMD[DType.float32, W](0)
        or7 += rr0
        oi7 += ii0
        or7 += rr1 * Float32(0.707106781) - ii1 * Float32(0.707106781)
        oi7 += rr1 * Float32(0.707106781) + ii1 * Float32(0.707106781)
        or7 += rr2 * Float32(0) - ii2 * Float32(1)
        oi7 += rr2 * Float32(1) + ii2 * Float32(0)
        or7 += rr3 * Float32(-0.707106781) - ii3 * Float32(0.707106781)
        oi7 += rr3 * Float32(0.707106781) + ii3 * Float32(-0.707106781)
        or7 -= rr4
        oi7 -= ii4
        or7 += rr5 * Float32(-0.707106781) - ii5 * Float32(-0.707106781)
        oi7 += rr5 * Float32(-0.707106781) + ii5 * Float32(-0.707106781)
        or7 += rr6 * Float32(0) - ii6 * Float32(-1)
        oi7 += rr6 * Float32(-1) + ii6 * Float32(0)
        or7 += rr7 * Float32(0.707106781) - ii7 * Float32(-0.707106781)
        oi7 += rr7 * Float32(-0.707106781) + ii7 * Float32(0.707106781)
        var twr7 = SIMD[DType.float32, W](Float32(1), Float32(0.773010453), Float32(0.195090322), Float32(-0.471396737), Float32(-0.923879533), Float32(-0.956940336), Float32(-0.555570233), Float32(0.0980171403))
        var twi7 = SIMD[DType.float32, W](Float32(0), Float32(-0.634393284), Float32(-0.98078528), Float32(-0.881921264), Float32(-0.382683432), Float32(0.290284677), Float32(0.831469612), Float32(0.995184727))
        var tr7 = or7 * twr7 - oi7 * twi7
        var ti7 = or7 * twi7 + oi7 * twr7
        or7 = tr7
        oi7 = ti7
        FFTFP32.buf.store(spad_base + 7, or7[0])
        FFTFP32.buf.store(spad_base + N + 7, oi7[0])
        FFTFP32.buf.store(spad_base + 15, or7[1])
        FFTFP32.buf.store(spad_base + N + 15, oi7[1])
        FFTFP32.buf.store(spad_base + 23, or7[2])
        FFTFP32.buf.store(spad_base + N + 23, oi7[2])
        FFTFP32.buf.store(spad_base + 31, or7[3])
        FFTFP32.buf.store(spad_base + N + 31, oi7[3])
        FFTFP32.buf.store(spad_base + 39, or7[4])
        FFTFP32.buf.store(spad_base + N + 39, oi7[4])
        FFTFP32.buf.store(spad_base + 47, or7[5])
        FFTFP32.buf.store(spad_base + N + 47, oi7[5])
        FFTFP32.buf.store(spad_base + 55, or7[6])
        FFTFP32.buf.store(spad_base + N + 55, oi7[6])
        FFTFP32.buf.store(spad_base + 63, or7[7])
        FFTFP32.buf.store(spad_base + N + 63, oi7[7])

    @staticmethod
    def stage_1():
        ref p = FFTFP32.params[]
        comptime RADIX = 8
        comptime SIMD_ITERS = 1
        comptime INPUT_STRIDE = 8
        comptime OUTPUT_STRIDE = 8

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD:
            return
        var spad_base = local_id * (2 * N)
        var batch_base = global_uthread_id() * N

        # ===== stage 1, SIMD batch 0 =====
        var rr0 = FFTFP32.buf.load[DType.float32, W](spad_base + 0)
        var ii0 = FFTFP32.buf.load[DType.float32, W](spad_base + N + 0)
        var rr1 = FFTFP32.buf.load[DType.float32, W](spad_base + 8)
        var ii1 = FFTFP32.buf.load[DType.float32, W](spad_base + N + 8)
        var rr2 = FFTFP32.buf.load[DType.float32, W](spad_base + 16)
        var ii2 = FFTFP32.buf.load[DType.float32, W](spad_base + N + 16)
        var rr3 = FFTFP32.buf.load[DType.float32, W](spad_base + 24)
        var ii3 = FFTFP32.buf.load[DType.float32, W](spad_base + N + 24)
        var rr4 = FFTFP32.buf.load[DType.float32, W](spad_base + 32)
        var ii4 = FFTFP32.buf.load[DType.float32, W](spad_base + N + 32)
        var rr5 = FFTFP32.buf.load[DType.float32, W](spad_base + 40)
        var ii5 = FFTFP32.buf.load[DType.float32, W](spad_base + N + 40)
        var rr6 = FFTFP32.buf.load[DType.float32, W](spad_base + 48)
        var ii6 = FFTFP32.buf.load[DType.float32, W](spad_base + N + 48)
        var rr7 = FFTFP32.buf.load[DType.float32, W](spad_base + 56)
        var ii7 = FFTFP32.buf.load[DType.float32, W](spad_base + N + 56)

        # radix-8 output 0
        var or0 = SIMD[DType.float32, W](0)
        var oi0 = SIMD[DType.float32, W](0)
        or0 += rr0
        oi0 += ii0
        or0 += rr1
        oi0 += ii1
        or0 += rr2
        oi0 += ii2
        or0 += rr3
        oi0 += ii3
        or0 += rr4
        oi0 += ii4
        or0 += rr5
        oi0 += ii5
        or0 += rr6
        oi0 += ii6
        or0 += rr7
        oi0 += ii7
        p.output_real_base.store(batch_base + 0, or0)
        p.output_imag_base.store(batch_base + 0, oi0)

        # radix-8 output 1
        var or1 = SIMD[DType.float32, W](0)
        var oi1 = SIMD[DType.float32, W](0)
        or1 += rr0
        oi1 += ii0
        or1 += rr1 * Float32(0.707106781) - ii1 * Float32(-0.707106781)
        oi1 += rr1 * Float32(-0.707106781) + ii1 * Float32(0.707106781)
        or1 += rr2 * Float32(0) - ii2 * Float32(-1)
        oi1 += rr2 * Float32(-1) + ii2 * Float32(0)
        or1 += rr3 * Float32(-0.707106781) - ii3 * Float32(-0.707106781)
        oi1 += rr3 * Float32(-0.707106781) + ii3 * Float32(-0.707106781)
        or1 -= rr4
        oi1 -= ii4
        or1 += rr5 * Float32(-0.707106781) - ii5 * Float32(0.707106781)
        oi1 += rr5 * Float32(0.707106781) + ii5 * Float32(-0.707106781)
        or1 += rr6 * Float32(0) - ii6 * Float32(1)
        oi1 += rr6 * Float32(1) + ii6 * Float32(0)
        or1 += rr7 * Float32(0.707106781) - ii7 * Float32(0.707106781)
        oi1 += rr7 * Float32(0.707106781) + ii7 * Float32(0.707106781)
        p.output_real_base.store(batch_base + 8, or1)
        p.output_imag_base.store(batch_base + 8, oi1)

        # radix-8 output 2
        var or2 = SIMD[DType.float32, W](0)
        var oi2 = SIMD[DType.float32, W](0)
        or2 += rr0
        oi2 += ii0
        or2 += rr1 * Float32(0) - ii1 * Float32(-1)
        oi2 += rr1 * Float32(-1) + ii1 * Float32(0)
        or2 -= rr2
        oi2 -= ii2
        or2 += rr3 * Float32(0) - ii3 * Float32(1)
        oi2 += rr3 * Float32(1) + ii3 * Float32(0)
        or2 += rr4
        oi2 += ii4
        or2 += rr5 * Float32(0) - ii5 * Float32(-1)
        oi2 += rr5 * Float32(-1) + ii5 * Float32(0)
        or2 -= rr6
        oi2 -= ii6
        or2 += rr7 * Float32(0) - ii7 * Float32(1)
        oi2 += rr7 * Float32(1) + ii7 * Float32(0)
        p.output_real_base.store(batch_base + 16, or2)
        p.output_imag_base.store(batch_base + 16, oi2)

        # radix-8 output 3
        var or3 = SIMD[DType.float32, W](0)
        var oi3 = SIMD[DType.float32, W](0)
        or3 += rr0
        oi3 += ii0
        or3 += rr1 * Float32(-0.707106781) - ii1 * Float32(-0.707106781)
        oi3 += rr1 * Float32(-0.707106781) + ii1 * Float32(-0.707106781)
        or3 += rr2 * Float32(0) - ii2 * Float32(1)
        oi3 += rr2 * Float32(1) + ii2 * Float32(0)
        or3 += rr3 * Float32(0.707106781) - ii3 * Float32(-0.707106781)
        oi3 += rr3 * Float32(-0.707106781) + ii3 * Float32(0.707106781)
        or3 -= rr4
        oi3 -= ii4
        or3 += rr5 * Float32(0.707106781) - ii5 * Float32(0.707106781)
        oi3 += rr5 * Float32(0.707106781) + ii5 * Float32(0.707106781)
        or3 += rr6 * Float32(0) - ii6 * Float32(-1)
        oi3 += rr6 * Float32(-1) + ii6 * Float32(0)
        or3 += rr7 * Float32(-0.707106781) - ii7 * Float32(0.707106781)
        oi3 += rr7 * Float32(0.707106781) + ii7 * Float32(-0.707106781)
        p.output_real_base.store(batch_base + 24, or3)
        p.output_imag_base.store(batch_base + 24, oi3)

        # radix-8 output 4
        var or4 = SIMD[DType.float32, W](0)
        var oi4 = SIMD[DType.float32, W](0)
        or4 += rr0
        oi4 += ii0
        or4 -= rr1
        oi4 -= ii1
        or4 += rr2
        oi4 += ii2
        or4 -= rr3
        oi4 -= ii3
        or4 += rr4
        oi4 += ii4
        or4 -= rr5
        oi4 -= ii5
        or4 += rr6
        oi4 += ii6
        or4 -= rr7
        oi4 -= ii7
        p.output_real_base.store(batch_base + 32, or4)
        p.output_imag_base.store(batch_base + 32, oi4)

        # radix-8 output 5
        var or5 = SIMD[DType.float32, W](0)
        var oi5 = SIMD[DType.float32, W](0)
        or5 += rr0
        oi5 += ii0
        or5 += rr1 * Float32(-0.707106781) - ii1 * Float32(0.707106781)
        oi5 += rr1 * Float32(0.707106781) + ii1 * Float32(-0.707106781)
        or5 += rr2 * Float32(0) - ii2 * Float32(-1)
        oi5 += rr2 * Float32(-1) + ii2 * Float32(0)
        or5 += rr3 * Float32(0.707106781) - ii3 * Float32(0.707106781)
        oi5 += rr3 * Float32(0.707106781) + ii3 * Float32(0.707106781)
        or5 -= rr4
        oi5 -= ii4
        or5 += rr5 * Float32(0.707106781) - ii5 * Float32(-0.707106781)
        oi5 += rr5 * Float32(-0.707106781) + ii5 * Float32(0.707106781)
        or5 += rr6 * Float32(0) - ii6 * Float32(1)
        oi5 += rr6 * Float32(1) + ii6 * Float32(0)
        or5 += rr7 * Float32(-0.707106781) - ii7 * Float32(-0.707106781)
        oi5 += rr7 * Float32(-0.707106781) + ii7 * Float32(-0.707106781)
        p.output_real_base.store(batch_base + 40, or5)
        p.output_imag_base.store(batch_base + 40, oi5)

        # radix-8 output 6
        var or6 = SIMD[DType.float32, W](0)
        var oi6 = SIMD[DType.float32, W](0)
        or6 += rr0
        oi6 += ii0
        or6 += rr1 * Float32(0) - ii1 * Float32(1)
        oi6 += rr1 * Float32(1) + ii1 * Float32(0)
        or6 -= rr2
        oi6 -= ii2
        or6 += rr3 * Float32(0) - ii3 * Float32(-1)
        oi6 += rr3 * Float32(-1) + ii3 * Float32(0)
        or6 += rr4
        oi6 += ii4
        or6 += rr5 * Float32(0) - ii5 * Float32(1)
        oi6 += rr5 * Float32(1) + ii5 * Float32(0)
        or6 -= rr6
        oi6 -= ii6
        or6 += rr7 * Float32(0) - ii7 * Float32(-1)
        oi6 += rr7 * Float32(-1) + ii7 * Float32(0)
        p.output_real_base.store(batch_base + 48, or6)
        p.output_imag_base.store(batch_base + 48, oi6)

        # radix-8 output 7
        var or7 = SIMD[DType.float32, W](0)
        var oi7 = SIMD[DType.float32, W](0)
        or7 += rr0
        oi7 += ii0
        or7 += rr1 * Float32(0.707106781) - ii1 * Float32(0.707106781)
        oi7 += rr1 * Float32(0.707106781) + ii1 * Float32(0.707106781)
        or7 += rr2 * Float32(0) - ii2 * Float32(1)
        oi7 += rr2 * Float32(1) + ii2 * Float32(0)
        or7 += rr3 * Float32(-0.707106781) - ii3 * Float32(0.707106781)
        oi7 += rr3 * Float32(0.707106781) + ii3 * Float32(-0.707106781)
        or7 -= rr4
        oi7 -= ii4
        or7 += rr5 * Float32(-0.707106781) - ii5 * Float32(-0.707106781)
        oi7 += rr5 * Float32(-0.707106781) + ii5 * Float32(-0.707106781)
        or7 += rr6 * Float32(0) - ii6 * Float32(-1)
        oi7 += rr6 * Float32(-1) + ii6 * Float32(0)
        or7 += rr7 * Float32(0.707106781) - ii7 * Float32(-0.707106781)
        oi7 += rr7 * Float32(-0.707106781) + ii7 * Float32(0.707106781)
        p.output_real_base.store(batch_base + 56, or7)
        p.output_imag_base.store(batch_base + 56, oi7)

    @staticmethod
    def device_main():
        launch_parallel[FFTFP32.stage_0]()
        launch_parallel[FFTFP32.stage_1]()


def main() raises:
    if FFTFP32.emit_ir_if_asked():
        return

    var total_elems = N * MAX_UTHREAD
    var input_real = cxl_alloc[Float32](total_elems)
    var input_imag = cxl_alloc[Float32](total_elems)
    var output_real = cxl_alloc[Float32](total_elems)
    var output_imag = cxl_alloc[Float32](total_elems)
    var ref_real = cxl_alloc[Float32](total_elems)
    var ref_imag = cxl_alloc[Float32](total_elems)

    var pool_elems = W * MAX_UTHREAD
    var uthread_pool = cxl_alloc[Float32](pool_elems)

    for i in range(total_elems):
        input_real[i] = Float32(0)
        input_imag[i] = Float32(0)
        output_real[i] = Float32(0)
        output_imag[i] = Float32(0)
        ref_real[i] = Float32(0)
        ref_imag[i] = Float32(0)

    # batch 0
    input_real[1] = Float32(1)
    ref_real[0] = Float32(1)
    ref_imag[0] = Float32(0)
    ref_real[1] = Float32(0.995184727)
    ref_imag[1] = Float32(-0.0980171403)
    ref_real[2] = Float32(0.98078528)
    ref_imag[2] = Float32(-0.195090322)
    ref_real[3] = Float32(0.956940336)
    ref_imag[3] = Float32(-0.290284677)
    ref_real[4] = Float32(0.923879533)
    ref_imag[4] = Float32(-0.382683432)
    ref_real[5] = Float32(0.881921264)
    ref_imag[5] = Float32(-0.471396737)
    ref_real[6] = Float32(0.831469612)
    ref_imag[6] = Float32(-0.555570233)
    ref_real[7] = Float32(0.773010453)
    ref_imag[7] = Float32(-0.634393284)
    ref_real[8] = Float32(0.707106781)
    ref_imag[8] = Float32(-0.707106781)
    ref_real[9] = Float32(0.634393284)
    ref_imag[9] = Float32(-0.773010453)
    ref_real[10] = Float32(0.555570233)
    ref_imag[10] = Float32(-0.831469612)
    ref_real[11] = Float32(0.471396737)
    ref_imag[11] = Float32(-0.881921264)
    ref_real[12] = Float32(0.382683432)
    ref_imag[12] = Float32(-0.923879533)
    ref_real[13] = Float32(0.290284677)
    ref_imag[13] = Float32(-0.956940336)
    ref_real[14] = Float32(0.195090322)
    ref_imag[14] = Float32(-0.98078528)
    ref_real[15] = Float32(0.0980171403)
    ref_imag[15] = Float32(-0.995184727)
    ref_real[16] = Float32(0)
    ref_imag[16] = Float32(-1)
    ref_real[17] = Float32(-0.0980171403)
    ref_imag[17] = Float32(-0.995184727)
    ref_real[18] = Float32(-0.195090322)
    ref_imag[18] = Float32(-0.98078528)
    ref_real[19] = Float32(-0.290284677)
    ref_imag[19] = Float32(-0.956940336)
    ref_real[20] = Float32(-0.382683432)
    ref_imag[20] = Float32(-0.923879533)
    ref_real[21] = Float32(-0.471396737)
    ref_imag[21] = Float32(-0.881921264)
    ref_real[22] = Float32(-0.555570233)
    ref_imag[22] = Float32(-0.831469612)
    ref_real[23] = Float32(-0.634393284)
    ref_imag[23] = Float32(-0.773010453)
    ref_real[24] = Float32(-0.707106781)
    ref_imag[24] = Float32(-0.707106781)
    ref_real[25] = Float32(-0.773010453)
    ref_imag[25] = Float32(-0.634393284)
    ref_real[26] = Float32(-0.831469612)
    ref_imag[26] = Float32(-0.555570233)
    ref_real[27] = Float32(-0.881921264)
    ref_imag[27] = Float32(-0.471396737)
    ref_real[28] = Float32(-0.923879533)
    ref_imag[28] = Float32(-0.382683432)
    ref_real[29] = Float32(-0.956940336)
    ref_imag[29] = Float32(-0.290284677)
    ref_real[30] = Float32(-0.98078528)
    ref_imag[30] = Float32(-0.195090322)
    ref_real[31] = Float32(-0.995184727)
    ref_imag[31] = Float32(-0.0980171403)
    ref_real[32] = Float32(-1)
    ref_imag[32] = Float32(0)
    ref_real[33] = Float32(-0.995184727)
    ref_imag[33] = Float32(0.0980171403)
    ref_real[34] = Float32(-0.98078528)
    ref_imag[34] = Float32(0.195090322)
    ref_real[35] = Float32(-0.956940336)
    ref_imag[35] = Float32(0.290284677)
    ref_real[36] = Float32(-0.923879533)
    ref_imag[36] = Float32(0.382683432)
    ref_real[37] = Float32(-0.881921264)
    ref_imag[37] = Float32(0.471396737)
    ref_real[38] = Float32(-0.831469612)
    ref_imag[38] = Float32(0.555570233)
    ref_real[39] = Float32(-0.773010453)
    ref_imag[39] = Float32(0.634393284)
    ref_real[40] = Float32(-0.707106781)
    ref_imag[40] = Float32(0.707106781)
    ref_real[41] = Float32(-0.634393284)
    ref_imag[41] = Float32(0.773010453)
    ref_real[42] = Float32(-0.555570233)
    ref_imag[42] = Float32(0.831469612)
    ref_real[43] = Float32(-0.471396737)
    ref_imag[43] = Float32(0.881921264)
    ref_real[44] = Float32(-0.382683432)
    ref_imag[44] = Float32(0.923879533)
    ref_real[45] = Float32(-0.290284677)
    ref_imag[45] = Float32(0.956940336)
    ref_real[46] = Float32(-0.195090322)
    ref_imag[46] = Float32(0.98078528)
    ref_real[47] = Float32(-0.0980171403)
    ref_imag[47] = Float32(0.995184727)
    ref_real[48] = Float32(0)
    ref_imag[48] = Float32(1)
    ref_real[49] = Float32(0.0980171403)
    ref_imag[49] = Float32(0.995184727)
    ref_real[50] = Float32(0.195090322)
    ref_imag[50] = Float32(0.98078528)
    ref_real[51] = Float32(0.290284677)
    ref_imag[51] = Float32(0.956940336)
    ref_real[52] = Float32(0.382683432)
    ref_imag[52] = Float32(0.923879533)
    ref_real[53] = Float32(0.471396737)
    ref_imag[53] = Float32(0.881921264)
    ref_real[54] = Float32(0.555570233)
    ref_imag[54] = Float32(0.831469612)
    ref_real[55] = Float32(0.634393284)
    ref_imag[55] = Float32(0.773010453)
    ref_real[56] = Float32(0.707106781)
    ref_imag[56] = Float32(0.707106781)
    ref_real[57] = Float32(0.773010453)
    ref_imag[57] = Float32(0.634393284)
    ref_real[58] = Float32(0.831469612)
    ref_imag[58] = Float32(0.555570233)
    ref_real[59] = Float32(0.881921264)
    ref_imag[59] = Float32(0.471396737)
    ref_real[60] = Float32(0.923879533)
    ref_imag[60] = Float32(0.382683432)
    ref_real[61] = Float32(0.956940336)
    ref_imag[61] = Float32(0.290284677)
    ref_real[62] = Float32(0.98078528)
    ref_imag[62] = Float32(0.195090322)
    ref_real[63] = Float32(0.995184727)
    ref_imag[63] = Float32(0.0980171403)

    var rc = FFTFP32.launch(
        PooledRange.over(uthread_pool, pool_elems),
        FFTFP32Params(input_real, input_imag, output_real, output_imag),
    )

    if rc != 0:
        print("[host] FFT failed, exit", rc)
        return

    var tol = Float32(1.0e-3)
    for i in range(total_elems):
        var err_r = output_real[i] - ref_real[i]
        var err_i = output_imag[i] - ref_imag[i]

        if err_r < Float32(0):
            err_r = -err_r
        if err_i < Float32(0):
            err_i = -err_i

        if err_r > tol or err_i > tol:
            print("[host] FFT mismatch at", i)
            print("  expected:", ref_real[i], ref_imag[i])
            print("  actual:  ", output_real[i], output_imag[i])
            print("  error:   ", err_r, err_i)
            return

    print("[host] FFT verification passed")
