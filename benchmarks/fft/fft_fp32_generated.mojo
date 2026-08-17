from std.sys import size_of

from m2ndp import VECTOR_WIDTH, NDPTask, PooledRange, global_uthread_id, launch_parallel, scratchpad
from m2ndp_host import cxl_alloc

comptime W = VECTOR_WIDTH // size_of[Float32]()
comptime N = 64

@fieldwise_init
struct FFTFP32Params(Movable):
    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]


struct FFTFP32(NDPTask):
    comptime Params = FFTFP32Params

    comptime buf = scratchpad[2 * 64, Float32, name="fft_buffer"]()

    @staticmethod
    def stage_0():
        ref p = FFTFP32.params[]
        comptime RADIX = 8
        comptime SIMD_ITERS = 1
        comptime INPUT_STRIDE = 8
        comptime OUTPUT_STRIDE = 8

        var uid = global_uthread_id()
        if uid != 0:
            return

        # ===== stage 0, SIMD batch 0 =====
        var rr0 = p.input_real_base.load[width=W](0)
        var ii0 = p.input_imag_base.load[width=W](0)
        var rr1 = p.input_real_base.load[width=W](8)
        var ii1 = p.input_imag_base.load[width=W](8)
        var rr2 = p.input_real_base.load[width=W](16)
        var ii2 = p.input_imag_base.load[width=W](16)
        var rr3 = p.input_real_base.load[width=W](24)
        var ii3 = p.input_imag_base.load[width=W](24)
        var rr4 = p.input_real_base.load[width=W](32)
        var ii4 = p.input_imag_base.load[width=W](32)
        var rr5 = p.input_real_base.load[width=W](40)
        var ii5 = p.input_imag_base.load[width=W](40)
        var rr6 = p.input_real_base.load[width=W](48)
        var ii6 = p.input_imag_base.load[width=W](48)
        var rr7 = p.input_real_base.load[width=W](56)
        var ii7 = p.input_imag_base.load[width=W](56)

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

        # Cooley-Tukey twiddle multiplication
        var twr0 = SIMD[DType.float32, W](Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1), Float32(1))
        var twi0 = SIMD[DType.float32, W](Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
        var tr0 = or0 * twr0 - oi0 * twi0
        var ti0 = or0 * twi0 + oi0 * twr0
        or0 = tr0
        oi0 = ti0
        var twr1 = SIMD[DType.float32, W](Float32(1), Float32(0.995184727), Float32(0.98078528), Float32(0.956940336), Float32(0.923879533), Float32(0.881921264), Float32(0.831469612), Float32(0.773010453))
        var twi1 = SIMD[DType.float32, W](Float32(0), Float32(-0.0980171403), Float32(-0.195090322), Float32(-0.290284677), Float32(-0.382683432), Float32(-0.471396737), Float32(-0.555570233), Float32(-0.634393284))
        var tr1 = or1 * twr1 - oi1 * twi1
        var ti1 = or1 * twi1 + oi1 * twr1
        or1 = tr1
        oi1 = ti1
        var twr2 = SIMD[DType.float32, W](Float32(1), Float32(0.98078528), Float32(0.923879533), Float32(0.831469612), Float32(0.707106781), Float32(0.555570233), Float32(0.382683432), Float32(0.195090322))
        var twi2 = SIMD[DType.float32, W](Float32(0), Float32(-0.195090322), Float32(-0.382683432), Float32(-0.555570233), Float32(-0.707106781), Float32(-0.831469612), Float32(-0.923879533), Float32(-0.98078528))
        var tr2 = or2 * twr2 - oi2 * twi2
        var ti2 = or2 * twi2 + oi2 * twr2
        or2 = tr2
        oi2 = ti2
        var twr3 = SIMD[DType.float32, W](Float32(1), Float32(0.956940336), Float32(0.831469612), Float32(0.634393284), Float32(0.382683432), Float32(0.0980171403), Float32(-0.195090322), Float32(-0.471396737))
        var twi3 = SIMD[DType.float32, W](Float32(0), Float32(-0.290284677), Float32(-0.555570233), Float32(-0.773010453), Float32(-0.923879533), Float32(-0.995184727), Float32(-0.98078528), Float32(-0.881921264))
        var tr3 = or3 * twr3 - oi3 * twi3
        var ti3 = or3 * twi3 + oi3 * twr3
        or3 = tr3
        oi3 = ti3
        var twr4 = SIMD[DType.float32, W](Float32(1), Float32(0.923879533), Float32(0.707106781), Float32(0.382683432), Float32(0), Float32(-0.382683432), Float32(-0.707106781), Float32(-0.923879533))
        var twi4 = SIMD[DType.float32, W](Float32(0), Float32(-0.382683432), Float32(-0.707106781), Float32(-0.923879533), Float32(-1), Float32(-0.923879533), Float32(-0.707106781), Float32(-0.382683432))
        var tr4 = or4 * twr4 - oi4 * twi4
        var ti4 = or4 * twi4 + oi4 * twr4
        or4 = tr4
        oi4 = ti4
        var twr5 = SIMD[DType.float32, W](Float32(1), Float32(0.881921264), Float32(0.555570233), Float32(0.0980171403), Float32(-0.382683432), Float32(-0.773010453), Float32(-0.98078528), Float32(-0.956940336))
        var twi5 = SIMD[DType.float32, W](Float32(0), Float32(-0.471396737), Float32(-0.831469612), Float32(-0.995184727), Float32(-0.923879533), Float32(-0.634393284), Float32(-0.195090322), Float32(0.290284677))
        var tr5 = or5 * twr5 - oi5 * twi5
        var ti5 = or5 * twi5 + oi5 * twr5
        or5 = tr5
        oi5 = ti5
        var twr6 = SIMD[DType.float32, W](Float32(1), Float32(0.831469612), Float32(0.382683432), Float32(-0.195090322), Float32(-0.707106781), Float32(-0.98078528), Float32(-0.923879533), Float32(-0.555570233))
        var twi6 = SIMD[DType.float32, W](Float32(0), Float32(-0.555570233), Float32(-0.923879533), Float32(-0.98078528), Float32(-0.707106781), Float32(-0.195090322), Float32(0.382683432), Float32(0.831469612))
        var tr6 = or6 * twr6 - oi6 * twi6
        var ti6 = or6 * twi6 + oi6 * twr6
        or6 = tr6
        oi6 = ti6
        var twr7 = SIMD[DType.float32, W](Float32(1), Float32(0.773010453), Float32(0.195090322), Float32(-0.471396737), Float32(-0.923879533), Float32(-0.956940336), Float32(-0.555570233), Float32(0.0980171403))
        var twi7 = SIMD[DType.float32, W](Float32(0), Float32(-0.634393284), Float32(-0.98078528), Float32(-0.881921264), Float32(-0.382683432), Float32(0.290284677), Float32(0.831469612), Float32(0.995184727))
        var tr7 = or7 * twr7 - oi7 * twi7
        var ti7 = or7 * twi7 + oi7 * twr7
        or7 = tr7
        oi7 = ti7

        # transpose into scratchpad for the next FFT stage
        FFTFP32.buf.store(0, or0[0])
        FFTFP32.buf.store(N + 0, oi0[0])
        FFTFP32.buf.store(1, or1[0])
        FFTFP32.buf.store(N + 1, oi1[0])
        FFTFP32.buf.store(2, or2[0])
        FFTFP32.buf.store(N + 2, oi2[0])
        FFTFP32.buf.store(3, or3[0])
        FFTFP32.buf.store(N + 3, oi3[0])
        FFTFP32.buf.store(4, or4[0])
        FFTFP32.buf.store(N + 4, oi4[0])
        FFTFP32.buf.store(5, or5[0])
        FFTFP32.buf.store(N + 5, oi5[0])
        FFTFP32.buf.store(6, or6[0])
        FFTFP32.buf.store(N + 6, oi6[0])
        FFTFP32.buf.store(7, or7[0])
        FFTFP32.buf.store(N + 7, oi7[0])
        FFTFP32.buf.store(8, or0[1])
        FFTFP32.buf.store(N + 8, oi0[1])
        FFTFP32.buf.store(9, or1[1])
        FFTFP32.buf.store(N + 9, oi1[1])
        FFTFP32.buf.store(10, or2[1])
        FFTFP32.buf.store(N + 10, oi2[1])
        FFTFP32.buf.store(11, or3[1])
        FFTFP32.buf.store(N + 11, oi3[1])
        FFTFP32.buf.store(12, or4[1])
        FFTFP32.buf.store(N + 12, oi4[1])
        FFTFP32.buf.store(13, or5[1])
        FFTFP32.buf.store(N + 13, oi5[1])
        FFTFP32.buf.store(14, or6[1])
        FFTFP32.buf.store(N + 14, oi6[1])
        FFTFP32.buf.store(15, or7[1])
        FFTFP32.buf.store(N + 15, oi7[1])
        FFTFP32.buf.store(16, or0[2])
        FFTFP32.buf.store(N + 16, oi0[2])
        FFTFP32.buf.store(17, or1[2])
        FFTFP32.buf.store(N + 17, oi1[2])
        FFTFP32.buf.store(18, or2[2])
        FFTFP32.buf.store(N + 18, oi2[2])
        FFTFP32.buf.store(19, or3[2])
        FFTFP32.buf.store(N + 19, oi3[2])
        FFTFP32.buf.store(20, or4[2])
        FFTFP32.buf.store(N + 20, oi4[2])
        FFTFP32.buf.store(21, or5[2])
        FFTFP32.buf.store(N + 21, oi5[2])
        FFTFP32.buf.store(22, or6[2])
        FFTFP32.buf.store(N + 22, oi6[2])
        FFTFP32.buf.store(23, or7[2])
        FFTFP32.buf.store(N + 23, oi7[2])
        FFTFP32.buf.store(24, or0[3])
        FFTFP32.buf.store(N + 24, oi0[3])
        FFTFP32.buf.store(25, or1[3])
        FFTFP32.buf.store(N + 25, oi1[3])
        FFTFP32.buf.store(26, or2[3])
        FFTFP32.buf.store(N + 26, oi2[3])
        FFTFP32.buf.store(27, or3[3])
        FFTFP32.buf.store(N + 27, oi3[3])
        FFTFP32.buf.store(28, or4[3])
        FFTFP32.buf.store(N + 28, oi4[3])
        FFTFP32.buf.store(29, or5[3])
        FFTFP32.buf.store(N + 29, oi5[3])
        FFTFP32.buf.store(30, or6[3])
        FFTFP32.buf.store(N + 30, oi6[3])
        FFTFP32.buf.store(31, or7[3])
        FFTFP32.buf.store(N + 31, oi7[3])
        FFTFP32.buf.store(32, or0[4])
        FFTFP32.buf.store(N + 32, oi0[4])
        FFTFP32.buf.store(33, or1[4])
        FFTFP32.buf.store(N + 33, oi1[4])
        FFTFP32.buf.store(34, or2[4])
        FFTFP32.buf.store(N + 34, oi2[4])
        FFTFP32.buf.store(35, or3[4])
        FFTFP32.buf.store(N + 35, oi3[4])
        FFTFP32.buf.store(36, or4[4])
        FFTFP32.buf.store(N + 36, oi4[4])
        FFTFP32.buf.store(37, or5[4])
        FFTFP32.buf.store(N + 37, oi5[4])
        FFTFP32.buf.store(38, or6[4])
        FFTFP32.buf.store(N + 38, oi6[4])
        FFTFP32.buf.store(39, or7[4])
        FFTFP32.buf.store(N + 39, oi7[4])
        FFTFP32.buf.store(40, or0[5])
        FFTFP32.buf.store(N + 40, oi0[5])
        FFTFP32.buf.store(41, or1[5])
        FFTFP32.buf.store(N + 41, oi1[5])
        FFTFP32.buf.store(42, or2[5])
        FFTFP32.buf.store(N + 42, oi2[5])
        FFTFP32.buf.store(43, or3[5])
        FFTFP32.buf.store(N + 43, oi3[5])
        FFTFP32.buf.store(44, or4[5])
        FFTFP32.buf.store(N + 44, oi4[5])
        FFTFP32.buf.store(45, or5[5])
        FFTFP32.buf.store(N + 45, oi5[5])
        FFTFP32.buf.store(46, or6[5])
        FFTFP32.buf.store(N + 46, oi6[5])
        FFTFP32.buf.store(47, or7[5])
        FFTFP32.buf.store(N + 47, oi7[5])
        FFTFP32.buf.store(48, or0[6])
        FFTFP32.buf.store(N + 48, oi0[6])
        FFTFP32.buf.store(49, or1[6])
        FFTFP32.buf.store(N + 49, oi1[6])
        FFTFP32.buf.store(50, or2[6])
        FFTFP32.buf.store(N + 50, oi2[6])
        FFTFP32.buf.store(51, or3[6])
        FFTFP32.buf.store(N + 51, oi3[6])
        FFTFP32.buf.store(52, or4[6])
        FFTFP32.buf.store(N + 52, oi4[6])
        FFTFP32.buf.store(53, or5[6])
        FFTFP32.buf.store(N + 53, oi5[6])
        FFTFP32.buf.store(54, or6[6])
        FFTFP32.buf.store(N + 54, oi6[6])
        FFTFP32.buf.store(55, or7[6])
        FFTFP32.buf.store(N + 55, oi7[6])
        FFTFP32.buf.store(56, or0[7])
        FFTFP32.buf.store(N + 56, oi0[7])
        FFTFP32.buf.store(57, or1[7])
        FFTFP32.buf.store(N + 57, oi1[7])
        FFTFP32.buf.store(58, or2[7])
        FFTFP32.buf.store(N + 58, oi2[7])
        FFTFP32.buf.store(59, or3[7])
        FFTFP32.buf.store(N + 59, oi3[7])
        FFTFP32.buf.store(60, or4[7])
        FFTFP32.buf.store(N + 60, oi4[7])
        FFTFP32.buf.store(61, or5[7])
        FFTFP32.buf.store(N + 61, oi5[7])
        FFTFP32.buf.store(62, or6[7])
        FFTFP32.buf.store(N + 62, oi6[7])
        FFTFP32.buf.store(63, or7[7])
        FFTFP32.buf.store(N + 63, oi7[7])

    @staticmethod
    def stage_1():
        ref p = FFTFP32.params[]
        comptime RADIX = 8
        comptime SIMD_ITERS = 1
        comptime INPUT_STRIDE = 8
        comptime OUTPUT_STRIDE = 8

        var uid = global_uthread_id()
        if uid != 0:
            return

        # ===== stage 1, SIMD batch 0 =====
        var rr0 = FFTFP32.buf.load[DType.float32, W](0)
        var ii0 = FFTFP32.buf.load[DType.float32, W](N + 0)
        var rr1 = FFTFP32.buf.load[DType.float32, W](8)
        var ii1 = FFTFP32.buf.load[DType.float32, W](N + 8)
        var rr2 = FFTFP32.buf.load[DType.float32, W](16)
        var ii2 = FFTFP32.buf.load[DType.float32, W](N + 16)
        var rr3 = FFTFP32.buf.load[DType.float32, W](24)
        var ii3 = FFTFP32.buf.load[DType.float32, W](N + 24)
        var rr4 = FFTFP32.buf.load[DType.float32, W](32)
        var ii4 = FFTFP32.buf.load[DType.float32, W](N + 32)
        var rr5 = FFTFP32.buf.load[DType.float32, W](40)
        var ii5 = FFTFP32.buf.load[DType.float32, W](N + 40)
        var rr6 = FFTFP32.buf.load[DType.float32, W](48)
        var ii6 = FFTFP32.buf.load[DType.float32, W](N + 48)
        var rr7 = FFTFP32.buf.load[DType.float32, W](56)
        var ii7 = FFTFP32.buf.load[DType.float32, W](N + 56)

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

        p.output_real_base.store(0, or0)
        p.output_imag_base.store(0, oi0)
        p.output_real_base.store(8, or1)
        p.output_imag_base.store(8, oi1)
        p.output_real_base.store(16, or2)
        p.output_imag_base.store(16, oi2)
        p.output_real_base.store(24, or3)
        p.output_imag_base.store(24, oi3)
        p.output_real_base.store(32, or4)
        p.output_imag_base.store(32, oi4)
        p.output_real_base.store(40, or5)
        p.output_imag_base.store(40, oi5)
        p.output_real_base.store(48, or6)
        p.output_imag_base.store(48, oi6)
        p.output_real_base.store(56, or7)
        p.output_imag_base.store(56, oi7)

    @staticmethod
    def device_main():
        launch_parallel[FFTFP32.stage_0]()
        launch_parallel[FFTFP32.stage_1]()


def main() raises:
    if FFTFP32.emit_ir_if_asked():
        return

    var input_real = cxl_alloc[Float32](N)
    var input_imag = cxl_alloc[Float32](N)
    var output_real = cxl_alloc[Float32](N)
    var output_imag = cxl_alloc[Float32](N)

    # TODO: host-side input initialization / reference check
    var pool_elems = W * 1
    var rc = FFTFP32.launch(
        PooledRange.over(input_real, pool_elems),
        FFTFP32Params(input_real, input_imag, output_real, output_imag),
    )
    if rc != 0:
        print("[host] FFT failed, exit", rc)
        return
    print("[host] FFT finished")
