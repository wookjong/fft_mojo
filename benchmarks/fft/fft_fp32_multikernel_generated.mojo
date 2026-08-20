from std.sys import size_of
from std.random import random_float64, seed
from std.math import cos as host_cos, sin as host_sin

from m2ndp import VECTOR_WIDTH, NDPTask, PooledRange, global_uthread_id, local_uthread_id, launch_parallel, scratchpad
from m2ndp_host import cxl_alloc

comptime W = VECTOR_WIDTH // size_of[Float32]()
comptime N = 960

comptime MAX_UTHREAD_FFTFP32Kernel0 = 120

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
        comptime RADIX = 2
        comptime SIMD_ITERS = 1

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel0:
            return
        var spad_base = local_id * 16
        var in_batch_base = global_uthread_id() * 1

        # ===== stage 0, SIMD batch 0 (valid lanes: 4/4) =====
        var rr0_lane0 = p.input_real_base.load[width=1](in_batch_base + 0)
        var ii0_lane0 = p.input_imag_base.load[width=1](in_batch_base + 0)
        var rr0_lane1 = p.input_real_base.load[width=1](in_batch_base + 120)
        var ii0_lane1 = p.input_imag_base.load[width=1](in_batch_base + 120)
        var rr0_lane2 = p.input_real_base.load[width=1](in_batch_base + 240)
        var ii0_lane2 = p.input_imag_base.load[width=1](in_batch_base + 240)
        var rr0_lane3 = p.input_real_base.load[width=1](in_batch_base + 360)
        var ii0_lane3 = p.input_imag_base.load[width=1](in_batch_base + 360)
        var rr0 = SIMD[DType.float32, 4](rr0_lane0[0], rr0_lane1[0], rr0_lane2[0], rr0_lane3[0])
        var ii0 = SIMD[DType.float32, 4](ii0_lane0[0], ii0_lane1[0], ii0_lane2[0], ii0_lane3[0])
        var rr1_lane0 = p.input_real_base.load[width=1](in_batch_base + 480)
        var ii1_lane0 = p.input_imag_base.load[width=1](in_batch_base + 480)
        var rr1_lane1 = p.input_real_base.load[width=1](in_batch_base + 600)
        var ii1_lane1 = p.input_imag_base.load[width=1](in_batch_base + 600)
        var rr1_lane2 = p.input_real_base.load[width=1](in_batch_base + 720)
        var ii1_lane2 = p.input_imag_base.load[width=1](in_batch_base + 720)
        var rr1_lane3 = p.input_real_base.load[width=1](in_batch_base + 840)
        var ii1_lane3 = p.input_imag_base.load[width=1](in_batch_base + 840)
        var rr1 = SIMD[DType.float32, 4](rr1_lane0[0], rr1_lane1[0], rr1_lane2[0], rr1_lane3[0])
        var ii1 = SIMD[DType.float32, 4](ii1_lane0[0], ii1_lane1[0], ii1_lane2[0], ii1_lane3[0])

        # fixed radix-2 butterfly
        var or0 = rr0 + rr1
        var oi0 = ii0 + ii1
        var or1 = rr0 - rr1
        var oi1 = ii0 - ii1
        FFTFP32Kernel0.buf_a.store(spad_base + 0, or0[0])
        FFTFP32Kernel0.buf_a.store(spad_base + N + 0, oi0[0])
        FFTFP32Kernel0.buf_a.store(spad_base + 2, or0[1])
        FFTFP32Kernel0.buf_a.store(spad_base + N + 2, oi0[1])
        FFTFP32Kernel0.buf_a.store(spad_base + 4, or0[2])
        FFTFP32Kernel0.buf_a.store(spad_base + N + 4, oi0[2])
        FFTFP32Kernel0.buf_a.store(spad_base + 6, or0[3])
        FFTFP32Kernel0.buf_a.store(spad_base + N + 6, oi0[3])

        var twr1 = SIMD[DType.float32, 4](Float32(1), Float32(0.707106781), Float32(0), Float32(-0.707106781))
        var twi1 = SIMD[DType.float32, 4](Float32(0), Float32(-0.707106781), Float32(-1), Float32(-0.707106781))
        var tr1 = or1 * twr1 - oi1 * twi1
        var ti1 = or1 * twi1 + oi1 * twr1
        or1 = tr1
        oi1 = ti1
        FFTFP32Kernel0.buf_a.store(spad_base + 1, or1[0])
        FFTFP32Kernel0.buf_a.store(spad_base + N + 1, oi1[0])
        FFTFP32Kernel0.buf_a.store(spad_base + 3, or1[1])
        FFTFP32Kernel0.buf_a.store(spad_base + N + 3, oi1[1])
        FFTFP32Kernel0.buf_a.store(spad_base + 5, or1[2])
        FFTFP32Kernel0.buf_a.store(spad_base + N + 5, oi1[2])
        FFTFP32Kernel0.buf_a.store(spad_base + 7, or1[3])
        FFTFP32Kernel0.buf_a.store(spad_base + N + 7, oi1[3])


    @staticmethod
    def stage_1():
        ref p = FFTFP32Kernel0.params[]
        comptime RADIX = 2
        comptime SIMD_ITERS = 1

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel0:
            return
        var spad_base = local_id * 16

        # ===== stage 1, SIMD batch 0 (valid lanes: 4/4) =====
        var rr0 = FFTFP32Kernel0.buf_a.load[DType.float32, 4](spad_base + 0)
        var ii0 = FFTFP32Kernel0.buf_a.load[DType.float32, 4](spad_base + N + 0)
        var rr1 = FFTFP32Kernel0.buf_a.load[DType.float32, 4](spad_base + 4)
        var ii1 = FFTFP32Kernel0.buf_a.load[DType.float32, 4](spad_base + N + 4)

        # fixed radix-2 butterfly
        var or0 = rr0 + rr1
        var oi0 = ii0 + ii1
        var or1 = rr0 - rr1
        var oi1 = ii0 - ii1
        FFTFP32Kernel0.buf_b.store(spad_base + 0, or0[0])
        FFTFP32Kernel0.buf_b.store(spad_base + N + 0, oi0[0])
        FFTFP32Kernel0.buf_b.store(spad_base + 1, or0[1])
        FFTFP32Kernel0.buf_b.store(spad_base + N + 1, oi0[1])
        FFTFP32Kernel0.buf_b.store(spad_base + 4, or0[2])
        FFTFP32Kernel0.buf_b.store(spad_base + N + 4, oi0[2])
        FFTFP32Kernel0.buf_b.store(spad_base + 5, or0[3])
        FFTFP32Kernel0.buf_b.store(spad_base + N + 5, oi0[3])

        var twr1 = SIMD[DType.float32, 4](Float32(1), Float32(1), Float32(0), Float32(0))
        var twi1 = SIMD[DType.float32, 4](Float32(0), Float32(0), Float32(-1), Float32(-1))
        var tr1 = or1 * twr1 - oi1 * twi1
        var ti1 = or1 * twi1 + oi1 * twr1
        or1 = tr1
        oi1 = ti1
        FFTFP32Kernel0.buf_b.store(spad_base + 2, or1[0])
        FFTFP32Kernel0.buf_b.store(spad_base + N + 2, oi1[0])
        FFTFP32Kernel0.buf_b.store(spad_base + 3, or1[1])
        FFTFP32Kernel0.buf_b.store(spad_base + N + 3, oi1[1])
        FFTFP32Kernel0.buf_b.store(spad_base + 6, or1[2])
        FFTFP32Kernel0.buf_b.store(spad_base + N + 6, oi1[2])
        FFTFP32Kernel0.buf_b.store(spad_base + 7, or1[3])
        FFTFP32Kernel0.buf_b.store(spad_base + N + 7, oi1[3])


    @staticmethod
    def stage_2():
        ref p = FFTFP32Kernel0.params[]
        comptime RADIX = 2
        comptime SIMD_ITERS = 1

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel0:
            return
        var spad_base = local_id * 16
        var out_batch_base = (global_uthread_id() % 1) + (global_uthread_id() // 1) * 8

        # ===== stage 2, SIMD batch 0 (valid lanes: 4/4) =====
        var rr0 = FFTFP32Kernel0.buf_b.load[DType.float32, 4](spad_base + 0)
        var ii0 = FFTFP32Kernel0.buf_b.load[DType.float32, 4](spad_base + N + 0)
        var rr1 = FFTFP32Kernel0.buf_b.load[DType.float32, 4](spad_base + 4)
        var ii1 = FFTFP32Kernel0.buf_b.load[DType.float32, 4](spad_base + N + 4)

        # fixed radix-2 butterfly
        var or0 = rr0 + rr1
        var oi0 = ii0 + ii1
        var or1 = rr0 - rr1
        var oi1 = ii0 - ii1
        var ltwr0 = p.large_twiddle_real_base.load[width=4](out_batch_base + 0)
        var ltwi0 = p.large_twiddle_imag_base.load[width=4](out_batch_base + 0)
        var ltr0 = or0 * ltwr0 - oi0 * ltwi0
        var lti0 = or0 * ltwi0 + oi0 * ltwr0
        or0 = ltr0
        oi0 = lti0
        p.output_real_base.store(out_batch_base + 0, or0)
        p.output_imag_base.store(out_batch_base + 0, oi0)

        var ltwr1 = p.large_twiddle_real_base.load[width=4](out_batch_base + 4)
        var ltwi1 = p.large_twiddle_imag_base.load[width=4](out_batch_base + 4)
        var ltr1 = or1 * ltwr1 - oi1 * ltwi1
        var lti1 = or1 * ltwi1 + oi1 * ltwr1
        or1 = ltr1
        oi1 = lti1
        p.output_real_base.store(out_batch_base + 4, or1)
        p.output_imag_base.store(out_batch_base + 4, oi1)


    @staticmethod
    def device_main():
        launch_parallel[FFTFP32Kernel0.stage_0]()
        launch_parallel[FFTFP32Kernel0.stage_1]()
        launch_parallel[FFTFP32Kernel0.stage_2]()


comptime MAX_UTHREAD_FFTFP32Kernel1 = 120

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

    comptime buf_a = scratchpad[1920, Float32, name="fftfp32kernel1_buf_a"]()
    comptime buf_b = scratchpad[1920, Float32, name="fftfp32kernel1_buf_b"]()

    @staticmethod
    def stage_0():
        ref p = FFTFP32Kernel1.params[]
        comptime RADIX = 2
        comptime SIMD_ITERS = 1

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel1:
            return
        var spad_base = local_id * 16
        var in_batch_base = global_uthread_id() * 1

        # ===== stage 0, SIMD batch 0 (valid lanes: 4/4) =====
        var rr0_lane0 = p.input_real_base.load[width=1](in_batch_base + 0)
        var ii0_lane0 = p.input_imag_base.load[width=1](in_batch_base + 0)
        var rr0_lane1 = p.input_real_base.load[width=1](in_batch_base + 120)
        var ii0_lane1 = p.input_imag_base.load[width=1](in_batch_base + 120)
        var rr0_lane2 = p.input_real_base.load[width=1](in_batch_base + 240)
        var ii0_lane2 = p.input_imag_base.load[width=1](in_batch_base + 240)
        var rr0_lane3 = p.input_real_base.load[width=1](in_batch_base + 360)
        var ii0_lane3 = p.input_imag_base.load[width=1](in_batch_base + 360)
        var rr0 = SIMD[DType.float32, 4](rr0_lane0[0], rr0_lane1[0], rr0_lane2[0], rr0_lane3[0])
        var ii0 = SIMD[DType.float32, 4](ii0_lane0[0], ii0_lane1[0], ii0_lane2[0], ii0_lane3[0])
        var rr1_lane0 = p.input_real_base.load[width=1](in_batch_base + 480)
        var ii1_lane0 = p.input_imag_base.load[width=1](in_batch_base + 480)
        var rr1_lane1 = p.input_real_base.load[width=1](in_batch_base + 600)
        var ii1_lane1 = p.input_imag_base.load[width=1](in_batch_base + 600)
        var rr1_lane2 = p.input_real_base.load[width=1](in_batch_base + 720)
        var ii1_lane2 = p.input_imag_base.load[width=1](in_batch_base + 720)
        var rr1_lane3 = p.input_real_base.load[width=1](in_batch_base + 840)
        var ii1_lane3 = p.input_imag_base.load[width=1](in_batch_base + 840)
        var rr1 = SIMD[DType.float32, 4](rr1_lane0[0], rr1_lane1[0], rr1_lane2[0], rr1_lane3[0])
        var ii1 = SIMD[DType.float32, 4](ii1_lane0[0], ii1_lane1[0], ii1_lane2[0], ii1_lane3[0])

        # fixed radix-2 butterfly
        var or0 = rr0 + rr1
        var oi0 = ii0 + ii1
        var or1 = rr0 - rr1
        var oi1 = ii0 - ii1
        FFTFP32Kernel1.buf_a.store(spad_base + 0, or0[0])
        FFTFP32Kernel1.buf_a.store(spad_base + N + 0, oi0[0])
        FFTFP32Kernel1.buf_a.store(spad_base + 2, or0[1])
        FFTFP32Kernel1.buf_a.store(spad_base + N + 2, oi0[1])
        FFTFP32Kernel1.buf_a.store(spad_base + 4, or0[2])
        FFTFP32Kernel1.buf_a.store(spad_base + N + 4, oi0[2])
        FFTFP32Kernel1.buf_a.store(spad_base + 6, or0[3])
        FFTFP32Kernel1.buf_a.store(spad_base + N + 6, oi0[3])

        var twr1 = SIMD[DType.float32, 4](Float32(1), Float32(0.707106781), Float32(0), Float32(-0.707106781))
        var twi1 = SIMD[DType.float32, 4](Float32(0), Float32(-0.707106781), Float32(-1), Float32(-0.707106781))
        var tr1 = or1 * twr1 - oi1 * twi1
        var ti1 = or1 * twi1 + oi1 * twr1
        or1 = tr1
        oi1 = ti1
        FFTFP32Kernel1.buf_a.store(spad_base + 1, or1[0])
        FFTFP32Kernel1.buf_a.store(spad_base + N + 1, oi1[0])
        FFTFP32Kernel1.buf_a.store(spad_base + 3, or1[1])
        FFTFP32Kernel1.buf_a.store(spad_base + N + 3, oi1[1])
        FFTFP32Kernel1.buf_a.store(spad_base + 5, or1[2])
        FFTFP32Kernel1.buf_a.store(spad_base + N + 5, oi1[2])
        FFTFP32Kernel1.buf_a.store(spad_base + 7, or1[3])
        FFTFP32Kernel1.buf_a.store(spad_base + N + 7, oi1[3])


    @staticmethod
    def stage_1():
        ref p = FFTFP32Kernel1.params[]
        comptime RADIX = 2
        comptime SIMD_ITERS = 1

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel1:
            return
        var spad_base = local_id * 16

        # ===== stage 1, SIMD batch 0 (valid lanes: 4/4) =====
        var rr0 = FFTFP32Kernel1.buf_a.load[DType.float32, 4](spad_base + 0)
        var ii0 = FFTFP32Kernel1.buf_a.load[DType.float32, 4](spad_base + N + 0)
        var rr1 = FFTFP32Kernel1.buf_a.load[DType.float32, 4](spad_base + 4)
        var ii1 = FFTFP32Kernel1.buf_a.load[DType.float32, 4](spad_base + N + 4)

        # fixed radix-2 butterfly
        var or0 = rr0 + rr1
        var oi0 = ii0 + ii1
        var or1 = rr0 - rr1
        var oi1 = ii0 - ii1
        FFTFP32Kernel1.buf_b.store(spad_base + 0, or0[0])
        FFTFP32Kernel1.buf_b.store(spad_base + N + 0, oi0[0])
        FFTFP32Kernel1.buf_b.store(spad_base + 1, or0[1])
        FFTFP32Kernel1.buf_b.store(spad_base + N + 1, oi0[1])
        FFTFP32Kernel1.buf_b.store(spad_base + 4, or0[2])
        FFTFP32Kernel1.buf_b.store(spad_base + N + 4, oi0[2])
        FFTFP32Kernel1.buf_b.store(spad_base + 5, or0[3])
        FFTFP32Kernel1.buf_b.store(spad_base + N + 5, oi0[3])

        var twr1 = SIMD[DType.float32, 4](Float32(1), Float32(1), Float32(0), Float32(0))
        var twi1 = SIMD[DType.float32, 4](Float32(0), Float32(0), Float32(-1), Float32(-1))
        var tr1 = or1 * twr1 - oi1 * twi1
        var ti1 = or1 * twi1 + oi1 * twr1
        or1 = tr1
        oi1 = ti1
        FFTFP32Kernel1.buf_b.store(spad_base + 2, or1[0])
        FFTFP32Kernel1.buf_b.store(spad_base + N + 2, oi1[0])
        FFTFP32Kernel1.buf_b.store(spad_base + 3, or1[1])
        FFTFP32Kernel1.buf_b.store(spad_base + N + 3, oi1[1])
        FFTFP32Kernel1.buf_b.store(spad_base + 6, or1[2])
        FFTFP32Kernel1.buf_b.store(spad_base + N + 6, oi1[2])
        FFTFP32Kernel1.buf_b.store(spad_base + 7, or1[3])
        FFTFP32Kernel1.buf_b.store(spad_base + N + 7, oi1[3])


    @staticmethod
    def stage_2():
        ref p = FFTFP32Kernel1.params[]
        comptime RADIX = 2
        comptime SIMD_ITERS = 1

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel1:
            return
        var spad_base = local_id * 16
        var out_batch_base = (global_uthread_id() % 8) + (global_uthread_id() // 8) * 64

        # ===== stage 2, SIMD batch 0 (valid lanes: 4/4) =====
        var rr0 = FFTFP32Kernel1.buf_b.load[DType.float32, 4](spad_base + 0)
        var ii0 = FFTFP32Kernel1.buf_b.load[DType.float32, 4](spad_base + N + 0)
        var rr1 = FFTFP32Kernel1.buf_b.load[DType.float32, 4](spad_base + 4)
        var ii1 = FFTFP32Kernel1.buf_b.load[DType.float32, 4](spad_base + N + 4)

        # fixed radix-2 butterfly
        var or0 = rr0 + rr1
        var oi0 = ii0 + ii1
        var or1 = rr0 - rr1
        var oi1 = ii0 - ii1
        var ltwr0_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 0)
        var ltwi0_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 0)
        var ltwr0_lane1 = p.large_twiddle_real_base.load[width=1](out_batch_base + 8)
        var ltwi0_lane1 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 8)
        var ltwr0_lane2 = p.large_twiddle_real_base.load[width=1](out_batch_base + 16)
        var ltwi0_lane2 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 16)
        var ltwr0_lane3 = p.large_twiddle_real_base.load[width=1](out_batch_base + 24)
        var ltwi0_lane3 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 24)
        var ltwr0 = SIMD[DType.float32, 4](ltwr0_lane0[0], ltwr0_lane1[0], ltwr0_lane2[0], ltwr0_lane3[0])
        var ltwi0 = SIMD[DType.float32, 4](ltwi0_lane0[0], ltwi0_lane1[0], ltwi0_lane2[0], ltwi0_lane3[0])
        var ltr0 = or0 * ltwr0 - oi0 * ltwi0
        var lti0 = or0 * ltwi0 + oi0 * ltwr0
        or0 = ltr0
        oi0 = lti0
        p.output_real_base.store(out_batch_base + 0, or0[0])
        p.output_imag_base.store(out_batch_base + 0, oi0[0])
        p.output_real_base.store(out_batch_base + 8, or0[1])
        p.output_imag_base.store(out_batch_base + 8, oi0[1])
        p.output_real_base.store(out_batch_base + 16, or0[2])
        p.output_imag_base.store(out_batch_base + 16, oi0[2])
        p.output_real_base.store(out_batch_base + 24, or0[3])
        p.output_imag_base.store(out_batch_base + 24, oi0[3])

        var ltwr1_lane0 = p.large_twiddle_real_base.load[width=1](out_batch_base + 32)
        var ltwi1_lane0 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 32)
        var ltwr1_lane1 = p.large_twiddle_real_base.load[width=1](out_batch_base + 40)
        var ltwi1_lane1 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 40)
        var ltwr1_lane2 = p.large_twiddle_real_base.load[width=1](out_batch_base + 48)
        var ltwi1_lane2 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 48)
        var ltwr1_lane3 = p.large_twiddle_real_base.load[width=1](out_batch_base + 56)
        var ltwi1_lane3 = p.large_twiddle_imag_base.load[width=1](out_batch_base + 56)
        var ltwr1 = SIMD[DType.float32, 4](ltwr1_lane0[0], ltwr1_lane1[0], ltwr1_lane2[0], ltwr1_lane3[0])
        var ltwi1 = SIMD[DType.float32, 4](ltwi1_lane0[0], ltwi1_lane1[0], ltwi1_lane2[0], ltwi1_lane3[0])
        var ltr1 = or1 * ltwr1 - oi1 * ltwi1
        var lti1 = or1 * ltwi1 + oi1 * ltwr1
        or1 = ltr1
        oi1 = lti1
        p.output_real_base.store(out_batch_base + 32, or1[0])
        p.output_imag_base.store(out_batch_base + 32, oi1[0])
        p.output_real_base.store(out_batch_base + 40, or1[1])
        p.output_imag_base.store(out_batch_base + 40, oi1[1])
        p.output_real_base.store(out_batch_base + 48, or1[2])
        p.output_imag_base.store(out_batch_base + 48, oi1[2])
        p.output_real_base.store(out_batch_base + 56, or1[3])
        p.output_imag_base.store(out_batch_base + 56, oi1[3])


    @staticmethod
    def device_main():
        launch_parallel[FFTFP32Kernel1.stage_0]()
        launch_parallel[FFTFP32Kernel1.stage_1]()
        launch_parallel[FFTFP32Kernel1.stage_2]()


comptime MAX_UTHREAD_FFTFP32Kernel2 = 64

@fieldwise_init
struct FFTFP32Kernel2Params(Movable):
    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]


struct FFTFP32Kernel2(NDPTask):
    comptime Params = FFTFP32Kernel2Params

    comptime buf_a = scratchpad[1920, Float32, name="fftfp32kernel2_buf_a"]()
    comptime buf_b = scratchpad[1920, Float32, name="fftfp32kernel2_buf_b"]()

    @staticmethod
    def stage_0():
        ref p = FFTFP32Kernel2.params[]
        comptime RADIX = 3
        comptime SIMD_ITERS = 2

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel2:
            return
        var spad_base = local_id * 30
        var in_batch_base = global_uthread_id() * 1

        if True:  # batch 0 scope
            # ===== stage 0, SIMD batch 0 (valid lanes: 4/4) =====
            var rr0_lane0 = p.input_real_base.load[width=1](in_batch_base + 0)
            var ii0_lane0 = p.input_imag_base.load[width=1](in_batch_base + 0)
            var rr0_lane1 = p.input_real_base.load[width=1](in_batch_base + 64)
            var ii0_lane1 = p.input_imag_base.load[width=1](in_batch_base + 64)
            var rr0_lane2 = p.input_real_base.load[width=1](in_batch_base + 128)
            var ii0_lane2 = p.input_imag_base.load[width=1](in_batch_base + 128)
            var rr0_lane3 = p.input_real_base.load[width=1](in_batch_base + 192)
            var ii0_lane3 = p.input_imag_base.load[width=1](in_batch_base + 192)
            var rr0 = SIMD[DType.float32, 4](rr0_lane0[0], rr0_lane1[0], rr0_lane2[0], rr0_lane3[0])
            var ii0 = SIMD[DType.float32, 4](ii0_lane0[0], ii0_lane1[0], ii0_lane2[0], ii0_lane3[0])
            var rr1_lane0 = p.input_real_base.load[width=1](in_batch_base + 320)
            var ii1_lane0 = p.input_imag_base.load[width=1](in_batch_base + 320)
            var rr1_lane1 = p.input_real_base.load[width=1](in_batch_base + 384)
            var ii1_lane1 = p.input_imag_base.load[width=1](in_batch_base + 384)
            var rr1_lane2 = p.input_real_base.load[width=1](in_batch_base + 448)
            var ii1_lane2 = p.input_imag_base.load[width=1](in_batch_base + 448)
            var rr1_lane3 = p.input_real_base.load[width=1](in_batch_base + 512)
            var ii1_lane3 = p.input_imag_base.load[width=1](in_batch_base + 512)
            var rr1 = SIMD[DType.float32, 4](rr1_lane0[0], rr1_lane1[0], rr1_lane2[0], rr1_lane3[0])
            var ii1 = SIMD[DType.float32, 4](ii1_lane0[0], ii1_lane1[0], ii1_lane2[0], ii1_lane3[0])
            var rr2_lane0 = p.input_real_base.load[width=1](in_batch_base + 640)
            var ii2_lane0 = p.input_imag_base.load[width=1](in_batch_base + 640)
            var rr2_lane1 = p.input_real_base.load[width=1](in_batch_base + 704)
            var ii2_lane1 = p.input_imag_base.load[width=1](in_batch_base + 704)
            var rr2_lane2 = p.input_real_base.load[width=1](in_batch_base + 768)
            var ii2_lane2 = p.input_imag_base.load[width=1](in_batch_base + 768)
            var rr2_lane3 = p.input_real_base.load[width=1](in_batch_base + 832)
            var ii2_lane3 = p.input_imag_base.load[width=1](in_batch_base + 832)
            var rr2 = SIMD[DType.float32, 4](rr2_lane0[0], rr2_lane1[0], rr2_lane2[0], rr2_lane3[0])
            var ii2 = SIMD[DType.float32, 4](ii2_lane0[0], ii2_lane1[0], ii2_lane2[0], ii2_lane3[0])

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
            FFTFP32Kernel2.buf_a.store(spad_base + 0, or0[0])
            FFTFP32Kernel2.buf_a.store(spad_base + N + 0, oi0[0])
            FFTFP32Kernel2.buf_a.store(spad_base + 3, or0[1])
            FFTFP32Kernel2.buf_a.store(spad_base + N + 3, oi0[1])
            FFTFP32Kernel2.buf_a.store(spad_base + 6, or0[2])
            FFTFP32Kernel2.buf_a.store(spad_base + N + 6, oi0[2])
            FFTFP32Kernel2.buf_a.store(spad_base + 9, or0[3])
            FFTFP32Kernel2.buf_a.store(spad_base + N + 9, oi0[3])

            var twr1 = SIMD[DType.float32, 4](Float32(1), Float32(0.913545458), Float32(0.669130606), Float32(0.309016994))
            var twi1 = SIMD[DType.float32, 4](Float32(0), Float32(-0.406736643), Float32(-0.743144825), Float32(-0.951056516))
            var tr1 = or1 * twr1 - oi1 * twi1
            var ti1 = or1 * twi1 + oi1 * twr1
            or1 = tr1
            oi1 = ti1
            FFTFP32Kernel2.buf_a.store(spad_base + 1, or1[0])
            FFTFP32Kernel2.buf_a.store(spad_base + N + 1, oi1[0])
            FFTFP32Kernel2.buf_a.store(spad_base + 4, or1[1])
            FFTFP32Kernel2.buf_a.store(spad_base + N + 4, oi1[1])
            FFTFP32Kernel2.buf_a.store(spad_base + 7, or1[2])
            FFTFP32Kernel2.buf_a.store(spad_base + N + 7, oi1[2])
            FFTFP32Kernel2.buf_a.store(spad_base + 10, or1[3])
            FFTFP32Kernel2.buf_a.store(spad_base + N + 10, oi1[3])

            var twr2 = SIMD[DType.float32, 4](Float32(1), Float32(0.669130606), Float32(-0.104528463), Float32(-0.809016994))
            var twi2 = SIMD[DType.float32, 4](Float32(0), Float32(-0.743144825), Float32(-0.994521895), Float32(-0.587785252))
            var tr2 = or2 * twr2 - oi2 * twi2
            var ti2 = or2 * twi2 + oi2 * twr2
            or2 = tr2
            oi2 = ti2
            FFTFP32Kernel2.buf_a.store(spad_base + 2, or2[0])
            FFTFP32Kernel2.buf_a.store(spad_base + N + 2, oi2[0])
            FFTFP32Kernel2.buf_a.store(spad_base + 5, or2[1])
            FFTFP32Kernel2.buf_a.store(spad_base + N + 5, oi2[1])
            FFTFP32Kernel2.buf_a.store(spad_base + 8, or2[2])
            FFTFP32Kernel2.buf_a.store(spad_base + N + 8, oi2[2])
            FFTFP32Kernel2.buf_a.store(spad_base + 11, or2[3])
            FFTFP32Kernel2.buf_a.store(spad_base + N + 11, oi2[3])


        if True:  # batch 1 scope
            # ===== stage 0, SIMD batch 1 (valid lanes: 1/4) =====
            var rr0_lane0 = p.input_real_base.load[width=1](in_batch_base + 256)
            var ii0_lane0 = p.input_imag_base.load[width=1](in_batch_base + 256)
            var rr0 = SIMD[DType.float32, 4](rr0_lane0[0], Float32(0), Float32(0), Float32(0))
            var ii0 = SIMD[DType.float32, 4](ii0_lane0[0], Float32(0), Float32(0), Float32(0))
            var rr1_lane0 = p.input_real_base.load[width=1](in_batch_base + 576)
            var ii1_lane0 = p.input_imag_base.load[width=1](in_batch_base + 576)
            var rr1 = SIMD[DType.float32, 4](rr1_lane0[0], Float32(0), Float32(0), Float32(0))
            var ii1 = SIMD[DType.float32, 4](ii1_lane0[0], Float32(0), Float32(0), Float32(0))
            var rr2_lane0 = p.input_real_base.load[width=1](in_batch_base + 896)
            var ii2_lane0 = p.input_imag_base.load[width=1](in_batch_base + 896)
            var rr2 = SIMD[DType.float32, 4](rr2_lane0[0], Float32(0), Float32(0), Float32(0))
            var ii2 = SIMD[DType.float32, 4](ii2_lane0[0], Float32(0), Float32(0), Float32(0))

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
            FFTFP32Kernel2.buf_a.store(spad_base + 12, or0[0])
            FFTFP32Kernel2.buf_a.store(spad_base + N + 12, oi0[0])

            var twr1 = SIMD[DType.float32, 4](Float32(-0.104528463), Float32(1), Float32(1), Float32(1))
            var twi1 = SIMD[DType.float32, 4](Float32(-0.994521895), Float32(0), Float32(0), Float32(0))
            var tr1 = or1 * twr1 - oi1 * twi1
            var ti1 = or1 * twi1 + oi1 * twr1
            or1 = tr1
            oi1 = ti1
            FFTFP32Kernel2.buf_a.store(spad_base + 13, or1[0])
            FFTFP32Kernel2.buf_a.store(spad_base + N + 13, oi1[0])

            var twr2 = SIMD[DType.float32, 4](Float32(-0.978147601), Float32(1), Float32(1), Float32(1))
            var twi2 = SIMD[DType.float32, 4](Float32(0.207911691), Float32(0), Float32(0), Float32(0))
            var tr2 = or2 * twr2 - oi2 * twi2
            var ti2 = or2 * twi2 + oi2 * twr2
            or2 = tr2
            oi2 = ti2
            FFTFP32Kernel2.buf_a.store(spad_base + 14, or2[0])
            FFTFP32Kernel2.buf_a.store(spad_base + N + 14, oi2[0])


    @staticmethod
    def stage_1():
        ref p = FFTFP32Kernel2.params[]
        comptime RADIX = 5
        comptime SIMD_ITERS = 1

        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTFP32Kernel2:
            return
        var spad_base = local_id * 30
        var out_batch_base = global_uthread_id() * 1

        # ===== stage 1, SIMD batch 0 (valid lanes: 3/4) =====
        var rr0_lane0 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + 0)
        var ii0_lane0 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + N + 0)
        var rr0_lane1 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + 1)
        var ii0_lane1 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + N + 1)
        var rr0_lane2 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + 2)
        var ii0_lane2 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + N + 2)
        var rr0 = SIMD[DType.float32, 4](rr0_lane0[0], rr0_lane1[0], rr0_lane2[0], Float32(0))
        var ii0 = SIMD[DType.float32, 4](ii0_lane0[0], ii0_lane1[0], ii0_lane2[0], Float32(0))
        var rr1_lane0 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + 3)
        var ii1_lane0 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + N + 3)
        var rr1_lane1 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + 4)
        var ii1_lane1 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + N + 4)
        var rr1_lane2 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + 5)
        var ii1_lane2 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + N + 5)
        var rr1 = SIMD[DType.float32, 4](rr1_lane0[0], rr1_lane1[0], rr1_lane2[0], Float32(0))
        var ii1 = SIMD[DType.float32, 4](ii1_lane0[0], ii1_lane1[0], ii1_lane2[0], Float32(0))
        var rr2_lane0 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + 6)
        var ii2_lane0 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + N + 6)
        var rr2_lane1 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + 7)
        var ii2_lane1 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + N + 7)
        var rr2_lane2 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + 8)
        var ii2_lane2 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + N + 8)
        var rr2 = SIMD[DType.float32, 4](rr2_lane0[0], rr2_lane1[0], rr2_lane2[0], Float32(0))
        var ii2 = SIMD[DType.float32, 4](ii2_lane0[0], ii2_lane1[0], ii2_lane2[0], Float32(0))
        var rr3_lane0 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + 9)
        var ii3_lane0 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + N + 9)
        var rr3_lane1 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + 10)
        var ii3_lane1 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + N + 10)
        var rr3_lane2 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + 11)
        var ii3_lane2 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + N + 11)
        var rr3 = SIMD[DType.float32, 4](rr3_lane0[0], rr3_lane1[0], rr3_lane2[0], Float32(0))
        var ii3 = SIMD[DType.float32, 4](ii3_lane0[0], ii3_lane1[0], ii3_lane2[0], Float32(0))
        var rr4_lane0 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + 12)
        var ii4_lane0 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + N + 12)
        var rr4_lane1 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + 13)
        var ii4_lane1 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + N + 13)
        var rr4_lane2 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + 14)
        var ii4_lane2 = FFTFP32Kernel2.buf_a.load[DType.float32, 1](spad_base + N + 14)
        var rr4 = SIMD[DType.float32, 4](rr4_lane0[0], rr4_lane1[0], rr4_lane2[0], Float32(0))
        var ii4 = SIMD[DType.float32, 4](ii4_lane0[0], ii4_lane1[0], ii4_lane2[0], Float32(0))

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
        p.output_real_base.store(out_batch_base + 64, or0[1])
        p.output_imag_base.store(out_batch_base + 64, oi0[1])
        p.output_real_base.store(out_batch_base + 128, or0[2])
        p.output_imag_base.store(out_batch_base + 128, oi0[2])

        p.output_real_base.store(out_batch_base + 192, or1[0])
        p.output_imag_base.store(out_batch_base + 192, oi1[0])
        p.output_real_base.store(out_batch_base + 256, or1[1])
        p.output_imag_base.store(out_batch_base + 256, oi1[1])
        p.output_real_base.store(out_batch_base + 320, or1[2])
        p.output_imag_base.store(out_batch_base + 320, oi1[2])

        p.output_real_base.store(out_batch_base + 384, or2[0])
        p.output_imag_base.store(out_batch_base + 384, oi2[0])
        p.output_real_base.store(out_batch_base + 448, or2[1])
        p.output_imag_base.store(out_batch_base + 448, oi2[1])
        p.output_real_base.store(out_batch_base + 512, or2[2])
        p.output_imag_base.store(out_batch_base + 512, oi2[2])

        p.output_real_base.store(out_batch_base + 576, or3[0])
        p.output_imag_base.store(out_batch_base + 576, oi3[0])
        p.output_real_base.store(out_batch_base + 640, or3[1])
        p.output_imag_base.store(out_batch_base + 640, oi3[1])
        p.output_real_base.store(out_batch_base + 704, or3[2])
        p.output_imag_base.store(out_batch_base + 704, oi3[2])

        p.output_real_base.store(out_batch_base + 768, or4[0])
        p.output_imag_base.store(out_batch_base + 768, oi4[0])
        p.output_real_base.store(out_batch_base + 832, or4[1])
        p.output_imag_base.store(out_batch_base + 832, oi4[1])
        p.output_real_base.store(out_batch_base + 896, or4[2])
        p.output_imag_base.store(out_batch_base + 896, oi4[2])


    @staticmethod
    def device_main():
        launch_parallel[FFTFP32Kernel2.stage_0]()
        launch_parallel[FFTFP32Kernel2.stage_1]()


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
    while lt_r_0 < 120:
        var lt_c1_0 = 0
        while lt_c1_0 < 8:
            var lt_angle_0 = lt_sign_0 * 2.0 * lt_pi_0 * Float64(lt_r_0) * Float64(lt_c1_0) * Float64(1) / Float64(960)
            var lt_val_r_0 = Float32(host_cos(lt_angle_0))
            var lt_val_i_0 = Float32(host_sin(lt_angle_0))
            large_twiddle0_real[lt_r_0 * 8 + lt_c1_0] = lt_val_r_0
            large_twiddle0_imag[lt_r_0 * 8 + lt_c1_0] = lt_val_i_0
            lt_c1_0 += 1
        lt_r_0 += 1

    var large_twiddle1_real = cxl_alloc[Float32](n)
    var large_twiddle1_imag = cxl_alloc[Float32](n)
    var lt_pi_1 = Float64(3.141592653589793)
    var lt_sign_1 = Float64(-1.0)
    var lt_r_1 = 0
    while lt_r_1 < 15:
        var lt_c1_1 = 0
        while lt_c1_1 < 8:
            var lt_angle_1 = lt_sign_1 * 2.0 * lt_pi_1 * Float64(lt_r_1) * Float64(lt_c1_1) * Float64(8) / Float64(960)
            var lt_val_r_1 = Float32(host_cos(lt_angle_1))
            var lt_val_i_1 = Float32(host_sin(lt_angle_1))
            var lt_out_a_1 = 0
            while lt_out_a_1 < 8:
                var lt_addr_1 = lt_out_a_1 + lt_c1_1 * 8 + lt_r_1 * 64
                large_twiddle1_real[lt_addr_1] = lt_val_r_1
                large_twiddle1_imag[lt_addr_1] = lt_val_i_1
                lt_out_a_1 += 1
            lt_c1_1 += 1
        lt_r_1 += 1

    var pool0_elems = 480
    var pool0 = cxl_alloc[Float32](pool0_elems)
    var pool1_elems = 480
    var pool1 = cxl_alloc[Float32](pool1_elems)
    var pool2_elems = 256
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
