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

comptime N = 960

# ============================================================================
# KERNEL IMPLEMENTATIONS
# ============================================================================
# 5 stage(s), chained through DRAM in this order:
#   0: FFTRecPre0 (tiled transpose)
#   1: FFTRecNear0 (FFT)
#   2: FFTRecMid0 (tiled transpose)
#   3: FFTRecLeaf1 (FFT)
#   4: FFTRecPost0 (tiled transpose)

# ---- stage 0: FFTRecPre0 ----
comptime MAX_UTHREAD_FFTRecPre0 = 480

@fieldwise_init
struct FFTRecPre0Params(Movable):
    var src_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var src_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var dst_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var dst_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var q_table: UnsafePointer[Int, MutAnyOrigin]


struct FFTRecPre0(NDPTask):
    comptime Params = FFTRecPre0Params

    comptime tile_buf = scratchpad[3840, Float32, name="fftrecpre0_tile"]()

    @staticmethod
    def stage_0():
        ref p = FFTRecPre0.params[]
        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTRecPre0:
            return
        var spad_base = local_id * 8

        var tile_id = global_uthread_id()
        var q = p.q_table[tile_id]
        var local_tile = tile_id - q * 240
        var t_r = local_tile // 16
        var t_c = local_tile % 16

        var src_row_0 = t_r * 2 + 0
        var src_base_0 = q * 960 + src_row_0 * 32 + t_c * 2
        var rr0 = p.src_real_base.load[width=2](src_base_0)
        var ii0 = p.src_imag_base.load[width=2](src_base_0)
        FFTRecPre0.tile_buf.store(spad_base + 0, rr0[0])
        FFTRecPre0.tile_buf.store(spad_base + 4, ii0[0])
        FFTRecPre0.tile_buf.store(spad_base + 2, rr0[1])
        FFTRecPre0.tile_buf.store(spad_base + 6, ii0[1])

        var src_row_1 = t_r * 2 + 1
        var src_base_1 = q * 960 + src_row_1 * 32 + t_c * 2
        var rr1 = p.src_real_base.load[width=2](src_base_1)
        var ii1 = p.src_imag_base.load[width=2](src_base_1)
        FFTRecPre0.tile_buf.store(spad_base + 1, rr1[0])
        FFTRecPre0.tile_buf.store(spad_base + 5, ii1[0])
        FFTRecPre0.tile_buf.store(spad_base + 3, rr1[1])
        FFTRecPre0.tile_buf.store(spad_base + 7, ii1[1])

        var dst_row_0 = t_c * 2 + 0
        var dst_base_0 = q * 960 + dst_row_0 * 30 + t_r * 2
        var or0 = FFTRecPre0.tile_buf.load[DType.float32, 2](spad_base + 0)
        var oi0 = FFTRecPre0.tile_buf.load[DType.float32, 2](spad_base + 4)
        p.dst_real_base.store(dst_base_0, or0)
        p.dst_imag_base.store(dst_base_0, oi0)
        var dst_row_1 = t_c * 2 + 1
        var dst_base_1 = q * 960 + dst_row_1 * 30 + t_r * 2
        var or1 = FFTRecPre0.tile_buf.load[DType.float32, 2](spad_base + 2)
        var oi1 = FFTRecPre0.tile_buf.load[DType.float32, 2](spad_base + 6)
        p.dst_real_base.store(dst_base_1, or1)
        p.dst_imag_base.store(dst_base_1, oi1)

    @staticmethod
    def device_main():
        launch_parallel[FFTRecPre0.stage_0]()


# ---- stage 1: FFTRecNear0 ----
# FFT_EXECUTION {"arithmetic_lanes": [4, 2, 4], "compute_lanes": 4, "kernel": "FFTRecNear0", "launch": {"alignment_bytes": 256, "chunk_uthreads": 8, "padded": false, "round_fft_capacity": 2048, "round_uthread_capacity": 4096, "simd_lanes": 8, "slots_per_chunk": 4, "spread_across_units": true, "target_units": 32, "total_ffts": 64, "workers_per_fft": 2}, "loop_stages": true, "narrow_middle_stages": true}
comptime MAX_UTHREAD_FFTRecNear0 = 128

@fieldwise_init
struct FFTRecNear0Params(Movable):
    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var loop_twiddle_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var loop_twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]


struct FFTRecNear0(NDPTask):
    comptime Params = FFTRecNear0Params

    comptime buf_a = scratchpad[3840, Float32, name="fftrecnear0_buf_a"]()
    comptime buf_b = scratchpad[3840, Float32, name="fftrecnear0_buf_b"]()

    @staticmethod
    def stage_0():
        ref p = FFTRecNear0.params[]
        comptime RADIX = 2
        comptime SIMD_ITERS = 2

        comptime WORKERS_PER_FFT = 2
        var local_id = local_uthread_id()
        var fft_slot = local_id // WORKERS_PER_FFT
        var worker_id = local_id % WORKERS_PER_FFT
        if fft_slot >= 64:
            return
        var spad_base = fft_slot * 60
        var logical_fft_id = global_uthread_id() // WORKERS_PER_FFT
        var in_batch_base = logical_fft_id * 30

        if worker_id == 0:
            if True:  # batch 0 scope
                # ===== stage 0, SIMD batch 0 (valid lanes: 4/4) =====
                var rr0 = p.input_real_base.load[width=4](in_batch_base + 0)
                var ii0 = p.input_imag_base.load[width=4](in_batch_base + 0)
                var rr1 = p.input_real_base.load[width=4](in_batch_base + 15)
                var ii1 = p.input_imag_base.load[width=4](in_batch_base + 15)

                # fixed radix-2 butterfly
                var or0 = rr0 + rr1
                var oi0 = ii0 + ii1
                var or1 = rr0 - rr1
                var oi1 = ii0 - ii1
                FFTRecNear0.buf_a.store(spad_base + 0, or0[0])
                FFTRecNear0.buf_a.store(spad_base + 30 + 0, oi0[0])
                FFTRecNear0.buf_a.store(spad_base + 2, or0[1])
                FFTRecNear0.buf_a.store(spad_base + 30 + 2, oi0[1])
                FFTRecNear0.buf_a.store(spad_base + 4, or0[2])
                FFTRecNear0.buf_a.store(spad_base + 30 + 4, oi0[2])
                FFTRecNear0.buf_a.store(spad_base + 6, or0[3])
                FFTRecNear0.buf_a.store(spad_base + 30 + 6, oi0[3])

                var twr1 = SIMD[DType.float32, 4](Float32(1), Float32(0.978147601), Float32(0.913545458), Float32(0.809016994))
                var twi1 = SIMD[DType.float32, 4](Float32(0), Float32(-0.207911691), Float32(-0.406736643), Float32(-0.587785252))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                FFTRecNear0.buf_a.store(spad_base + 1, or1[0])
                FFTRecNear0.buf_a.store(spad_base + 30 + 1, oi1[0])
                FFTRecNear0.buf_a.store(spad_base + 3, or1[1])
                FFTRecNear0.buf_a.store(spad_base + 30 + 3, oi1[1])
                FFTRecNear0.buf_a.store(spad_base + 5, or1[2])
                FFTRecNear0.buf_a.store(spad_base + 30 + 5, oi1[2])
                FFTRecNear0.buf_a.store(spad_base + 7, or1[3])
                FFTRecNear0.buf_a.store(spad_base + 30 + 7, oi1[3])


            if True:  # batch 0 scope
                # ===== stage 0, SIMD batch 0 (valid lanes: 4/4) =====
                var rr0 = p.input_real_base.load[width=4](in_batch_base + 4)
                var ii0 = p.input_imag_base.load[width=4](in_batch_base + 4)
                var rr1 = p.input_real_base.load[width=4](in_batch_base + 19)
                var ii1 = p.input_imag_base.load[width=4](in_batch_base + 19)

                # fixed radix-2 butterfly
                var or0 = rr0 + rr1
                var oi0 = ii0 + ii1
                var or1 = rr0 - rr1
                var oi1 = ii0 - ii1
                FFTRecNear0.buf_a.store(spad_base + 8, or0[0])
                FFTRecNear0.buf_a.store(spad_base + 30 + 8, oi0[0])
                FFTRecNear0.buf_a.store(spad_base + 10, or0[1])
                FFTRecNear0.buf_a.store(spad_base + 30 + 10, oi0[1])
                FFTRecNear0.buf_a.store(spad_base + 12, or0[2])
                FFTRecNear0.buf_a.store(spad_base + 30 + 12, oi0[2])
                FFTRecNear0.buf_a.store(spad_base + 14, or0[3])
                FFTRecNear0.buf_a.store(spad_base + 30 + 14, oi0[3])

                var twr1 = SIMD[DType.float32, 4](Float32(0.669130606), Float32(0.5), Float32(0.309016994), Float32(0.104528463))
                var twi1 = SIMD[DType.float32, 4](Float32(-0.743144825), Float32(-0.866025404), Float32(-0.951056516), Float32(-0.994521895))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                FFTRecNear0.buf_a.store(spad_base + 9, or1[0])
                FFTRecNear0.buf_a.store(spad_base + 30 + 9, oi1[0])
                FFTRecNear0.buf_a.store(spad_base + 11, or1[1])
                FFTRecNear0.buf_a.store(spad_base + 30 + 11, oi1[1])
                FFTRecNear0.buf_a.store(spad_base + 13, or1[2])
                FFTRecNear0.buf_a.store(spad_base + 30 + 13, oi1[2])
                FFTRecNear0.buf_a.store(spad_base + 15, or1[3])
                FFTRecNear0.buf_a.store(spad_base + 30 + 15, oi1[3])


        if worker_id == 1:
            if True:  # batch 1 scope
                # ===== stage 0, SIMD batch 1 (valid lanes: 4/4) =====
                var rr0 = p.input_real_base.load[width=4](in_batch_base + 8)
                var ii0 = p.input_imag_base.load[width=4](in_batch_base + 8)
                var rr1 = p.input_real_base.load[width=4](in_batch_base + 23)
                var ii1 = p.input_imag_base.load[width=4](in_batch_base + 23)

                # fixed radix-2 butterfly
                var or0 = rr0 + rr1
                var oi0 = ii0 + ii1
                var or1 = rr0 - rr1
                var oi1 = ii0 - ii1
                FFTRecNear0.buf_a.store(spad_base + 16, or0[0])
                FFTRecNear0.buf_a.store(spad_base + 30 + 16, oi0[0])
                FFTRecNear0.buf_a.store(spad_base + 18, or0[1])
                FFTRecNear0.buf_a.store(spad_base + 30 + 18, oi0[1])
                FFTRecNear0.buf_a.store(spad_base + 20, or0[2])
                FFTRecNear0.buf_a.store(spad_base + 30 + 20, oi0[2])
                FFTRecNear0.buf_a.store(spad_base + 22, or0[3])
                FFTRecNear0.buf_a.store(spad_base + 30 + 22, oi0[3])

                var twr1 = SIMD[DType.float32, 4](Float32(-0.104528463), Float32(-0.309016994), Float32(-0.5), Float32(-0.669130606))
                var twi1 = SIMD[DType.float32, 4](Float32(-0.994521895), Float32(-0.951056516), Float32(-0.866025404), Float32(-0.743144825))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                FFTRecNear0.buf_a.store(spad_base + 17, or1[0])
                FFTRecNear0.buf_a.store(spad_base + 30 + 17, oi1[0])
                FFTRecNear0.buf_a.store(spad_base + 19, or1[1])
                FFTRecNear0.buf_a.store(spad_base + 30 + 19, oi1[1])
                FFTRecNear0.buf_a.store(spad_base + 21, or1[2])
                FFTRecNear0.buf_a.store(spad_base + 30 + 21, oi1[2])
                FFTRecNear0.buf_a.store(spad_base + 23, or1[3])
                FFTRecNear0.buf_a.store(spad_base + 30 + 23, oi1[3])


            if True:  # batch 1 scope
                # ===== stage 0, SIMD batch 1 (valid lanes: 3/4) =====
                var rr0_lane0 = p.input_real_base.load[width=1](in_batch_base + 12)
                var ii0_lane0 = p.input_imag_base.load[width=1](in_batch_base + 12)
                var rr0_lane1 = p.input_real_base.load[width=1](in_batch_base + 13)
                var ii0_lane1 = p.input_imag_base.load[width=1](in_batch_base + 13)
                var rr0_lane2 = p.input_real_base.load[width=1](in_batch_base + 14)
                var ii0_lane2 = p.input_imag_base.load[width=1](in_batch_base + 14)
                var rr0 = SIMD[DType.float32, 4](rr0_lane0[0], rr0_lane1[0], rr0_lane2[0], Float32(0))
                var ii0 = SIMD[DType.float32, 4](ii0_lane0[0], ii0_lane1[0], ii0_lane2[0], Float32(0))
                var rr1_lane0 = p.input_real_base.load[width=1](in_batch_base + 27)
                var ii1_lane0 = p.input_imag_base.load[width=1](in_batch_base + 27)
                var rr1_lane1 = p.input_real_base.load[width=1](in_batch_base + 28)
                var ii1_lane1 = p.input_imag_base.load[width=1](in_batch_base + 28)
                var rr1_lane2 = p.input_real_base.load[width=1](in_batch_base + 29)
                var ii1_lane2 = p.input_imag_base.load[width=1](in_batch_base + 29)
                var rr1 = SIMD[DType.float32, 4](rr1_lane0[0], rr1_lane1[0], rr1_lane2[0], Float32(0))
                var ii1 = SIMD[DType.float32, 4](ii1_lane0[0], ii1_lane1[0], ii1_lane2[0], Float32(0))

                # fixed radix-2 butterfly
                var or0 = rr0 + rr1
                var oi0 = ii0 + ii1
                var or1 = rr0 - rr1
                var oi1 = ii0 - ii1
                FFTRecNear0.buf_a.store(spad_base + 24, or0[0])
                FFTRecNear0.buf_a.store(spad_base + 30 + 24, oi0[0])
                FFTRecNear0.buf_a.store(spad_base + 26, or0[1])
                FFTRecNear0.buf_a.store(spad_base + 30 + 26, oi0[1])
                FFTRecNear0.buf_a.store(spad_base + 28, or0[2])
                FFTRecNear0.buf_a.store(spad_base + 30 + 28, oi0[2])

                var twr1 = SIMD[DType.float32, 4](Float32(-0.809016994), Float32(-0.913545458), Float32(-0.978147601), Float32(1))
                var twi1 = SIMD[DType.float32, 4](Float32(-0.587785252), Float32(-0.406736643), Float32(-0.207911691), Float32(0))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                FFTRecNear0.buf_a.store(spad_base + 25, or1[0])
                FFTRecNear0.buf_a.store(spad_base + 30 + 25, oi1[0])
                FFTRecNear0.buf_a.store(spad_base + 27, or1[1])
                FFTRecNear0.buf_a.store(spad_base + 30 + 27, oi1[1])
                FFTRecNear0.buf_a.store(spad_base + 29, or1[2])
                FFTRecNear0.buf_a.store(spad_base + 30 + 29, oi1[2])



    @staticmethod
    def stage_1():
        ref p = FFTRecNear0.params[]
        comptime RADIX = 3
        comptime SIMD_ITERS = 2

        comptime WORKERS_PER_FFT = 2
        var local_id = local_uthread_id()
        var fft_slot = local_id // WORKERS_PER_FFT
        var worker_id = local_id % WORKERS_PER_FFT
        if fft_slot >= 64:
            return
        var spad_base = fft_slot * 60
        var logical_fft_id = global_uthread_id() // WORKERS_PER_FFT

        if worker_id == 0:
            if True:  # batch 0 scope
                # ===== stage 1, SIMD batch 0 (valid lanes: 2/2) =====
                var rr0 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 0)
                var ii0 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 30 + 0)
                var rr1 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 10)
                var ii1 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 30 + 10)
                var rr2 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 20)
                var ii2 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 30 + 20)

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
                FFTRecNear0.buf_b.store(spad_base + 0, or0)
                FFTRecNear0.buf_b.store(spad_base + 30 + 0, oi0)

                FFTRecNear0.buf_b.store(spad_base + 2, or1)
                FFTRecNear0.buf_b.store(spad_base + 30 + 2, oi1)

                FFTRecNear0.buf_b.store(spad_base + 4, or2)
                FFTRecNear0.buf_b.store(spad_base + 30 + 4, oi2)


            if True:  # batch 0 scope
                # ===== stage 1, SIMD batch 0 (valid lanes: 2/2) =====
                var rr0 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 2)
                var ii0 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 30 + 2)
                var rr1 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 12)
                var ii1 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 30 + 12)
                var rr2 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 22)
                var ii2 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 30 + 22)

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
                FFTRecNear0.buf_b.store(spad_base + 6, or0)
                FFTRecNear0.buf_b.store(spad_base + 30 + 6, oi0)

                var twr1 = SIMD[DType.float32, 2](Float32(0.913545458), Float32(0.913545458))
                var twi1 = SIMD[DType.float32, 2](Float32(-0.406736643), Float32(-0.406736643))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                FFTRecNear0.buf_b.store(spad_base + 8, or1)
                FFTRecNear0.buf_b.store(spad_base + 30 + 8, oi1)

                var twr2 = SIMD[DType.float32, 2](Float32(0.669130606), Float32(0.669130606))
                var twi2 = SIMD[DType.float32, 2](Float32(-0.743144825), Float32(-0.743144825))
                var tr2 = or2 * twr2 - oi2 * twi2
                var ti2 = or2 * twi2 + oi2 * twr2
                or2 = tr2
                oi2 = ti2
                FFTRecNear0.buf_b.store(spad_base + 10, or2)
                FFTRecNear0.buf_b.store(spad_base + 30 + 10, oi2)


            if True:  # batch 0 scope
                # ===== stage 1, SIMD batch 0 (valid lanes: 2/2) =====
                var rr0 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 4)
                var ii0 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 30 + 4)
                var rr1 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 14)
                var ii1 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 30 + 14)
                var rr2 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 24)
                var ii2 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 30 + 24)

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
                FFTRecNear0.buf_b.store(spad_base + 12, or0)
                FFTRecNear0.buf_b.store(spad_base + 30 + 12, oi0)

                var twr1 = SIMD[DType.float32, 2](Float32(0.669130606), Float32(0.669130606))
                var twi1 = SIMD[DType.float32, 2](Float32(-0.743144825), Float32(-0.743144825))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                FFTRecNear0.buf_b.store(spad_base + 14, or1)
                FFTRecNear0.buf_b.store(spad_base + 30 + 14, oi1)

                var twr2 = SIMD[DType.float32, 2](Float32(-0.104528463), Float32(-0.104528463))
                var twi2 = SIMD[DType.float32, 2](Float32(-0.994521895), Float32(-0.994521895))
                var tr2 = or2 * twr2 - oi2 * twi2
                var ti2 = or2 * twi2 + oi2 * twr2
                or2 = tr2
                oi2 = ti2
                FFTRecNear0.buf_b.store(spad_base + 16, or2)
                FFTRecNear0.buf_b.store(spad_base + 30 + 16, oi2)


            if True:  # batch 0 scope
                # ===== stage 1, SIMD batch 0 (valid lanes: 2/2) =====
                var rr0 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 6)
                var ii0 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 30 + 6)
                var rr1 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 16)
                var ii1 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 30 + 16)
                var rr2 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 26)
                var ii2 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 30 + 26)

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
                FFTRecNear0.buf_b.store(spad_base + 18, or0)
                FFTRecNear0.buf_b.store(spad_base + 30 + 18, oi0)

                var twr1 = SIMD[DType.float32, 2](Float32(0.309016994), Float32(0.309016994))
                var twi1 = SIMD[DType.float32, 2](Float32(-0.951056516), Float32(-0.951056516))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                FFTRecNear0.buf_b.store(spad_base + 20, or1)
                FFTRecNear0.buf_b.store(spad_base + 30 + 20, oi1)

                var twr2 = SIMD[DType.float32, 2](Float32(-0.809016994), Float32(-0.809016994))
                var twi2 = SIMD[DType.float32, 2](Float32(-0.587785252), Float32(-0.587785252))
                var tr2 = or2 * twr2 - oi2 * twi2
                var ti2 = or2 * twi2 + oi2 * twr2
                or2 = tr2
                oi2 = ti2
                FFTRecNear0.buf_b.store(spad_base + 22, or2)
                FFTRecNear0.buf_b.store(spad_base + 30 + 22, oi2)


        if worker_id == 1:
            # ===== stage 1, SIMD batch 1 (valid lanes: 2/2) =====
            var rr0 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 8)
            var ii0 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 30 + 8)
            var rr1 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 18)
            var ii1 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 30 + 18)
            var rr2 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 28)
            var ii2 = FFTRecNear0.buf_a.load[DType.float32, 2](spad_base + 30 + 28)

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
            FFTRecNear0.buf_b.store(spad_base + 24, or0)
            FFTRecNear0.buf_b.store(spad_base + 30 + 24, oi0)

            var twr1 = SIMD[DType.float32, 2](Float32(-0.104528463), Float32(-0.104528463))
            var twi1 = SIMD[DType.float32, 2](Float32(-0.994521895), Float32(-0.994521895))
            var tr1 = or1 * twr1 - oi1 * twi1
            var ti1 = or1 * twi1 + oi1 * twr1
            or1 = tr1
            oi1 = ti1
            FFTRecNear0.buf_b.store(spad_base + 26, or1)
            FFTRecNear0.buf_b.store(spad_base + 30 + 26, oi1)

            var twr2 = SIMD[DType.float32, 2](Float32(-0.978147601), Float32(-0.978147601))
            var twi2 = SIMD[DType.float32, 2](Float32(0.207911691), Float32(0.207911691))
            var tr2 = or2 * twr2 - oi2 * twi2
            var ti2 = or2 * twi2 + oi2 * twr2
            or2 = tr2
            oi2 = ti2
            FFTRecNear0.buf_b.store(spad_base + 28, or2)
            FFTRecNear0.buf_b.store(spad_base + 30 + 28, oi2)



    @staticmethod
    def stage_2():
        ref p = FFTRecNear0.params[]
        comptime RADIX = 5
        comptime SIMD_ITERS = 1

        comptime WORKERS_PER_FFT = 2
        var local_id = local_uthread_id()
        var fft_slot = local_id // WORKERS_PER_FFT
        var worker_id = local_id % WORKERS_PER_FFT
        if fft_slot >= 64:
            return
        var spad_base = fft_slot * 60
        var logical_fft_id = global_uthread_id() // WORKERS_PER_FFT
        var out_batch_base = logical_fft_id * 30

        if worker_id == 0:
            if True:  # batch 0 scope
                # ===== stage 2, SIMD batch 0 (valid lanes: 4/4) =====
                var rr0 = FFTRecNear0.buf_b.load[DType.float32, 4](spad_base + 0)
                var ii0 = FFTRecNear0.buf_b.load[DType.float32, 4](spad_base + 30 + 0)
                var rr1 = FFTRecNear0.buf_b.load[DType.float32, 4](spad_base + 6)
                var ii1 = FFTRecNear0.buf_b.load[DType.float32, 4](spad_base + 30 + 6)
                var rr2 = FFTRecNear0.buf_b.load[DType.float32, 4](spad_base + 12)
                var ii2 = FFTRecNear0.buf_b.load[DType.float32, 4](spad_base + 30 + 12)
                var rr3 = FFTRecNear0.buf_b.load[DType.float32, 4](spad_base + 18)
                var ii3 = FFTRecNear0.buf_b.load[DType.float32, 4](spad_base + 30 + 18)
                var rr4 = FFTRecNear0.buf_b.load[DType.float32, 4](spad_base + 24)
                var ii4 = FFTRecNear0.buf_b.load[DType.float32, 4](spad_base + 30 + 24)

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
                p.output_real_base.store(out_batch_base + 0, or0)
                p.output_imag_base.store(out_batch_base + 0, oi0)

                p.output_real_base.store(out_batch_base + 6, or1)
                p.output_imag_base.store(out_batch_base + 6, oi1)

                p.output_real_base.store(out_batch_base + 12, or2)
                p.output_imag_base.store(out_batch_base + 12, oi2)

                p.output_real_base.store(out_batch_base + 18, or3)
                p.output_imag_base.store(out_batch_base + 18, oi3)

                p.output_real_base.store(out_batch_base + 24, or4)
                p.output_imag_base.store(out_batch_base + 24, oi4)


            if True:  # batch 0 scope
                # ===== stage 2, SIMD batch 0 (valid lanes: 2/4) =====
                var rr0_lane0 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 4)
                var ii0_lane0 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 30 + 4)
                var rr0_lane1 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 5)
                var ii0_lane1 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 30 + 5)
                var rr0 = SIMD[DType.float32, 4](rr0_lane0[0], rr0_lane1[0], Float32(0), Float32(0))
                var ii0 = SIMD[DType.float32, 4](ii0_lane0[0], ii0_lane1[0], Float32(0), Float32(0))
                var rr1_lane0 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 10)
                var ii1_lane0 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 30 + 10)
                var rr1_lane1 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 11)
                var ii1_lane1 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 30 + 11)
                var rr1 = SIMD[DType.float32, 4](rr1_lane0[0], rr1_lane1[0], Float32(0), Float32(0))
                var ii1 = SIMD[DType.float32, 4](ii1_lane0[0], ii1_lane1[0], Float32(0), Float32(0))
                var rr2_lane0 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 16)
                var ii2_lane0 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 30 + 16)
                var rr2_lane1 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 17)
                var ii2_lane1 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 30 + 17)
                var rr2 = SIMD[DType.float32, 4](rr2_lane0[0], rr2_lane1[0], Float32(0), Float32(0))
                var ii2 = SIMD[DType.float32, 4](ii2_lane0[0], ii2_lane1[0], Float32(0), Float32(0))
                var rr3_lane0 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 22)
                var ii3_lane0 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 30 + 22)
                var rr3_lane1 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 23)
                var ii3_lane1 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 30 + 23)
                var rr3 = SIMD[DType.float32, 4](rr3_lane0[0], rr3_lane1[0], Float32(0), Float32(0))
                var ii3 = SIMD[DType.float32, 4](ii3_lane0[0], ii3_lane1[0], Float32(0), Float32(0))
                var rr4_lane0 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 28)
                var ii4_lane0 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 30 + 28)
                var rr4_lane1 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 29)
                var ii4_lane1 = FFTRecNear0.buf_b.load[DType.float32, 1](spad_base + 30 + 29)
                var rr4 = SIMD[DType.float32, 4](rr4_lane0[0], rr4_lane1[0], Float32(0), Float32(0))
                var ii4 = SIMD[DType.float32, 4](ii4_lane0[0], ii4_lane1[0], Float32(0), Float32(0))

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
                p.output_real_base.store(out_batch_base + 4, or0[0])
                p.output_imag_base.store(out_batch_base + 4, oi0[0])
                p.output_real_base.store(out_batch_base + 5, or0[1])
                p.output_imag_base.store(out_batch_base + 5, oi0[1])

                p.output_real_base.store(out_batch_base + 10, or1[0])
                p.output_imag_base.store(out_batch_base + 10, oi1[0])
                p.output_real_base.store(out_batch_base + 11, or1[1])
                p.output_imag_base.store(out_batch_base + 11, oi1[1])

                p.output_real_base.store(out_batch_base + 16, or2[0])
                p.output_imag_base.store(out_batch_base + 16, oi2[0])
                p.output_real_base.store(out_batch_base + 17, or2[1])
                p.output_imag_base.store(out_batch_base + 17, oi2[1])

                p.output_real_base.store(out_batch_base + 22, or3[0])
                p.output_imag_base.store(out_batch_base + 22, oi3[0])
                p.output_real_base.store(out_batch_base + 23, or3[1])
                p.output_imag_base.store(out_batch_base + 23, oi3[1])

                p.output_real_base.store(out_batch_base + 28, or4[0])
                p.output_imag_base.store(out_batch_base + 28, oi4[0])
                p.output_real_base.store(out_batch_base + 29, or4[1])
                p.output_imag_base.store(out_batch_base + 29, oi4[1])



    @staticmethod
    def device_main():
        launch_parallel[FFTRecNear0.stage_0]()
        launch_parallel[FFTRecNear0.stage_1]()
        launch_parallel[FFTRecNear0.stage_2]()


# ---- stage 2: FFTRecMid0 ----
comptime MAX_UTHREAD_FFTRecMid0 = 480

@fieldwise_init
struct FFTRecMid0Params(Movable):
    var src_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var src_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var dst_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var dst_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var twiddle_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var q_table: UnsafePointer[Int, MutAnyOrigin]
    var t_r_table: UnsafePointer[Int, MutAnyOrigin]


struct FFTRecMid0(NDPTask):
    comptime Params = FFTRecMid0Params

    comptime tile_buf = scratchpad[3840, Float32, name="fftrecmid0_tile"]()

    @staticmethod
    def stage_0():
        ref p = FFTRecMid0.params[]
        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTRecMid0:
            return
        var spad_base = local_id * 8

        var tile_id = global_uthread_id()
        var q = p.q_table[tile_id]
        var local_tile = tile_id - q * 240
        var t_r = p.t_r_table[local_tile]
        var t_c = local_tile - t_r * 15

        var src_row_0 = t_r * 2 + 0
        var src_base_0 = q * 960 + src_row_0 * 30 + t_c * 2
        var rr0 = p.src_real_base.load[width=2](src_base_0)
        var ii0 = p.src_imag_base.load[width=2](src_base_0)
        var tw_base_0 = src_row_0 * 30 + t_c * 2
        var twr0 = p.twiddle_real_base.load[width=2](tw_base_0)
        var twi0 = p.twiddle_imag_base.load[width=2](tw_base_0)
        var tr0 = rr0 * twr0 - ii0 * twi0
        var ti0 = rr0 * twi0 + ii0 * twr0
        FFTRecMid0.tile_buf.store(spad_base + 0, tr0[0])
        FFTRecMid0.tile_buf.store(spad_base + 4, ti0[0])
        FFTRecMid0.tile_buf.store(spad_base + 2, tr0[1])
        FFTRecMid0.tile_buf.store(spad_base + 6, ti0[1])

        var src_row_1 = t_r * 2 + 1
        var src_base_1 = q * 960 + src_row_1 * 30 + t_c * 2
        var rr1 = p.src_real_base.load[width=2](src_base_1)
        var ii1 = p.src_imag_base.load[width=2](src_base_1)
        var tw_base_1 = src_row_1 * 30 + t_c * 2
        var twr1 = p.twiddle_real_base.load[width=2](tw_base_1)
        var twi1 = p.twiddle_imag_base.load[width=2](tw_base_1)
        var tr1 = rr1 * twr1 - ii1 * twi1
        var ti1 = rr1 * twi1 + ii1 * twr1
        FFTRecMid0.tile_buf.store(spad_base + 1, tr1[0])
        FFTRecMid0.tile_buf.store(spad_base + 5, ti1[0])
        FFTRecMid0.tile_buf.store(spad_base + 3, tr1[1])
        FFTRecMid0.tile_buf.store(spad_base + 7, ti1[1])

        var dst_row_0 = t_c * 2 + 0
        var dst_base_0 = q * 960 + dst_row_0 * 32 + t_r * 2
        var or0 = FFTRecMid0.tile_buf.load[DType.float32, 2](spad_base + 0)
        var oi0 = FFTRecMid0.tile_buf.load[DType.float32, 2](spad_base + 4)
        p.dst_real_base.store(dst_base_0, or0)
        p.dst_imag_base.store(dst_base_0, oi0)
        var dst_row_1 = t_c * 2 + 1
        var dst_base_1 = q * 960 + dst_row_1 * 32 + t_r * 2
        var or1 = FFTRecMid0.tile_buf.load[DType.float32, 2](spad_base + 2)
        var oi1 = FFTRecMid0.tile_buf.load[DType.float32, 2](spad_base + 6)
        p.dst_real_base.store(dst_base_1, or1)
        p.dst_imag_base.store(dst_base_1, oi1)

    @staticmethod
    def device_main():
        launch_parallel[FFTRecMid0.stage_0]()


# ---- stage 3: FFTRecLeaf1 ----
# FFT_EXECUTION {"arithmetic_lanes": [4, 2, 4], "compute_lanes": 4, "kernel": "FFTRecLeaf1", "launch": {"alignment_bytes": 256, "chunk_uthreads": 8, "padded": false, "round_fft_capacity": 1920, "round_uthread_capacity": 3840, "simd_lanes": 8, "slots_per_chunk": 4, "spread_across_units": true, "target_units": 32, "total_ffts": 60, "workers_per_fft": 2}, "loop_stages": true, "narrow_middle_stages": true}
comptime MAX_UTHREAD_FFTRecLeaf1 = 120

@fieldwise_init
struct FFTRecLeaf1Params(Movable):
    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var loop_twiddle_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var loop_twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]


struct FFTRecLeaf1(NDPTask):
    comptime Params = FFTRecLeaf1Params

    comptime buf_a = scratchpad[3840, Float32, name="fftrecleaf1_buf_a"]()
    comptime buf_b = scratchpad[3840, Float32, name="fftrecleaf1_buf_b"]()

    @staticmethod
    def stage_0():
        ref p = FFTRecLeaf1.params[]
        comptime RADIX = 4
        comptime SIMD_ITERS = 1

        comptime WORKERS_PER_FFT = 2
        var local_id = local_uthread_id()
        var fft_slot = local_id // WORKERS_PER_FFT
        var worker_id = local_id % WORKERS_PER_FFT
        if fft_slot >= 60:
            return
        var spad_base = fft_slot * 64
        var logical_fft_id = global_uthread_id() // WORKERS_PER_FFT
        var in_batch_base = logical_fft_id * 32

        if worker_id == 0:
            if True:  # batch 0 scope
                # ===== stage 0, SIMD batch 0 (valid lanes: 4/4) =====
                var rr0 = p.input_real_base.load[width=4](in_batch_base + 0)
                var ii0 = p.input_imag_base.load[width=4](in_batch_base + 0)
                var rr1 = p.input_real_base.load[width=4](in_batch_base + 8)
                var ii1 = p.input_imag_base.load[width=4](in_batch_base + 8)
                var rr2 = p.input_real_base.load[width=4](in_batch_base + 16)
                var ii2 = p.input_imag_base.load[width=4](in_batch_base + 16)
                var rr3 = p.input_real_base.load[width=4](in_batch_base + 24)
                var ii3 = p.input_imag_base.load[width=4](in_batch_base + 24)

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
                FFTRecLeaf1.buf_a.store(spad_base + 0, or0[0])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 0, oi0[0])
                FFTRecLeaf1.buf_a.store(spad_base + 4, or0[1])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 4, oi0[1])
                FFTRecLeaf1.buf_a.store(spad_base + 8, or0[2])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 8, oi0[2])
                FFTRecLeaf1.buf_a.store(spad_base + 12, or0[3])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 12, oi0[3])

                var twr1 = SIMD[DType.float32, 4](Float32(1), Float32(0.98078528), Float32(0.923879533), Float32(0.831469612))
                var twi1 = SIMD[DType.float32, 4](Float32(0), Float32(-0.195090322), Float32(-0.382683432), Float32(-0.555570233))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                FFTRecLeaf1.buf_a.store(spad_base + 1, or1[0])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 1, oi1[0])
                FFTRecLeaf1.buf_a.store(spad_base + 5, or1[1])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 5, oi1[1])
                FFTRecLeaf1.buf_a.store(spad_base + 9, or1[2])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 9, oi1[2])
                FFTRecLeaf1.buf_a.store(spad_base + 13, or1[3])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 13, oi1[3])

                var twr2 = SIMD[DType.float32, 4](Float32(1), Float32(0.923879533), Float32(0.707106781), Float32(0.382683432))
                var twi2 = SIMD[DType.float32, 4](Float32(0), Float32(-0.382683432), Float32(-0.707106781), Float32(-0.923879533))
                var tr2 = or2 * twr2 - oi2 * twi2
                var ti2 = or2 * twi2 + oi2 * twr2
                or2 = tr2
                oi2 = ti2
                FFTRecLeaf1.buf_a.store(spad_base + 2, or2[0])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 2, oi2[0])
                FFTRecLeaf1.buf_a.store(spad_base + 6, or2[1])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 6, oi2[1])
                FFTRecLeaf1.buf_a.store(spad_base + 10, or2[2])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 10, oi2[2])
                FFTRecLeaf1.buf_a.store(spad_base + 14, or2[3])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 14, oi2[3])

                var twr3 = SIMD[DType.float32, 4](Float32(1), Float32(0.831469612), Float32(0.382683432), Float32(-0.195090322))
                var twi3 = SIMD[DType.float32, 4](Float32(0), Float32(-0.555570233), Float32(-0.923879533), Float32(-0.98078528))
                var tr3 = or3 * twr3 - oi3 * twi3
                var ti3 = or3 * twi3 + oi3 * twr3
                or3 = tr3
                oi3 = ti3
                FFTRecLeaf1.buf_a.store(spad_base + 3, or3[0])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 3, oi3[0])
                FFTRecLeaf1.buf_a.store(spad_base + 7, or3[1])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 7, oi3[1])
                FFTRecLeaf1.buf_a.store(spad_base + 11, or3[2])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 11, oi3[2])
                FFTRecLeaf1.buf_a.store(spad_base + 15, or3[3])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 15, oi3[3])


            if True:  # batch 0 scope
                # ===== stage 0, SIMD batch 0 (valid lanes: 4/4) =====
                var rr0 = p.input_real_base.load[width=4](in_batch_base + 4)
                var ii0 = p.input_imag_base.load[width=4](in_batch_base + 4)
                var rr1 = p.input_real_base.load[width=4](in_batch_base + 12)
                var ii1 = p.input_imag_base.load[width=4](in_batch_base + 12)
                var rr2 = p.input_real_base.load[width=4](in_batch_base + 20)
                var ii2 = p.input_imag_base.load[width=4](in_batch_base + 20)
                var rr3 = p.input_real_base.load[width=4](in_batch_base + 28)
                var ii3 = p.input_imag_base.load[width=4](in_batch_base + 28)

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
                FFTRecLeaf1.buf_a.store(spad_base + 16, or0[0])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 16, oi0[0])
                FFTRecLeaf1.buf_a.store(spad_base + 20, or0[1])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 20, oi0[1])
                FFTRecLeaf1.buf_a.store(spad_base + 24, or0[2])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 24, oi0[2])
                FFTRecLeaf1.buf_a.store(spad_base + 28, or0[3])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 28, oi0[3])

                var twr1 = SIMD[DType.float32, 4](Float32(0.707106781), Float32(0.555570233), Float32(0.382683432), Float32(0.195090322))
                var twi1 = SIMD[DType.float32, 4](Float32(-0.707106781), Float32(-0.831469612), Float32(-0.923879533), Float32(-0.98078528))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                FFTRecLeaf1.buf_a.store(spad_base + 17, or1[0])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 17, oi1[0])
                FFTRecLeaf1.buf_a.store(spad_base + 21, or1[1])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 21, oi1[1])
                FFTRecLeaf1.buf_a.store(spad_base + 25, or1[2])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 25, oi1[2])
                FFTRecLeaf1.buf_a.store(spad_base + 29, or1[3])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 29, oi1[3])

                var twr2 = SIMD[DType.float32, 4](Float32(0), Float32(-0.382683432), Float32(-0.707106781), Float32(-0.923879533))
                var twi2 = SIMD[DType.float32, 4](Float32(-1), Float32(-0.923879533), Float32(-0.707106781), Float32(-0.382683432))
                var tr2 = or2 * twr2 - oi2 * twi2
                var ti2 = or2 * twi2 + oi2 * twr2
                or2 = tr2
                oi2 = ti2
                FFTRecLeaf1.buf_a.store(spad_base + 18, or2[0])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 18, oi2[0])
                FFTRecLeaf1.buf_a.store(spad_base + 22, or2[1])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 22, oi2[1])
                FFTRecLeaf1.buf_a.store(spad_base + 26, or2[2])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 26, oi2[2])
                FFTRecLeaf1.buf_a.store(spad_base + 30, or2[3])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 30, oi2[3])

                var twr3 = SIMD[DType.float32, 4](Float32(-0.707106781), Float32(-0.98078528), Float32(-0.923879533), Float32(-0.555570233))
                var twi3 = SIMD[DType.float32, 4](Float32(-0.707106781), Float32(-0.195090322), Float32(0.382683432), Float32(0.831469612))
                var tr3 = or3 * twr3 - oi3 * twi3
                var ti3 = or3 * twi3 + oi3 * twr3
                or3 = tr3
                oi3 = ti3
                FFTRecLeaf1.buf_a.store(spad_base + 19, or3[0])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 19, oi3[0])
                FFTRecLeaf1.buf_a.store(spad_base + 23, or3[1])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 23, oi3[1])
                FFTRecLeaf1.buf_a.store(spad_base + 27, or3[2])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 27, oi3[2])
                FFTRecLeaf1.buf_a.store(spad_base + 31, or3[3])
                FFTRecLeaf1.buf_a.store(spad_base + 32 + 31, oi3[3])



    @staticmethod
    def stage_1():
        ref p = FFTRecLeaf1.params[]
        comptime RADIX = 4
        comptime SIMD_ITERS = 1

        comptime WORKERS_PER_FFT = 2
        var local_id = local_uthread_id()
        var fft_slot = local_id // WORKERS_PER_FFT
        var worker_id = local_id % WORKERS_PER_FFT
        if fft_slot >= 60:
            return
        var spad_base = fft_slot * 64
        var logical_fft_id = global_uthread_id() // WORKERS_PER_FFT

        if worker_id == 0:
            if True:  # batch 0 scope
                # ===== stage 1, SIMD batch 0 (valid lanes: 2/2) =====
                var rr0 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 0)
                var ii0 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 32 + 0)
                var rr1 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 8)
                var ii1 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 32 + 8)
                var rr2 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 16)
                var ii2 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 32 + 16)
                var rr3 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 24)
                var ii3 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 32 + 24)

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
                FFTRecLeaf1.buf_b.store(spad_base + 0, or0)
                FFTRecLeaf1.buf_b.store(spad_base + 32 + 0, oi0)

                FFTRecLeaf1.buf_b.store(spad_base + 4, or1)
                FFTRecLeaf1.buf_b.store(spad_base + 32 + 4, oi1)

                FFTRecLeaf1.buf_b.store(spad_base + 8, or2)
                FFTRecLeaf1.buf_b.store(spad_base + 32 + 8, oi2)

                FFTRecLeaf1.buf_b.store(spad_base + 12, or3)
                FFTRecLeaf1.buf_b.store(spad_base + 32 + 12, oi3)


            if True:  # batch 0 scope
                # ===== stage 1, SIMD batch 0 (valid lanes: 2/2) =====
                var rr0 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 2)
                var ii0 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 32 + 2)
                var rr1 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 10)
                var ii1 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 32 + 10)
                var rr2 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 18)
                var ii2 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 32 + 18)
                var rr3 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 26)
                var ii3 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 32 + 26)

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
                FFTRecLeaf1.buf_b.store(spad_base + 2, or0)
                FFTRecLeaf1.buf_b.store(spad_base + 32 + 2, oi0)

                FFTRecLeaf1.buf_b.store(spad_base + 6, or1)
                FFTRecLeaf1.buf_b.store(spad_base + 32 + 6, oi1)

                FFTRecLeaf1.buf_b.store(spad_base + 10, or2)
                FFTRecLeaf1.buf_b.store(spad_base + 32 + 10, oi2)

                FFTRecLeaf1.buf_b.store(spad_base + 14, or3)
                FFTRecLeaf1.buf_b.store(spad_base + 32 + 14, oi3)


            if True:  # batch 0 scope
                # ===== stage 1, SIMD batch 0 (valid lanes: 2/2) =====
                var rr0 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 4)
                var ii0 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 32 + 4)
                var rr1 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 12)
                var ii1 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 32 + 12)
                var rr2 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 20)
                var ii2 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 32 + 20)
                var rr3 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 28)
                var ii3 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 32 + 28)

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
                FFTRecLeaf1.buf_b.store(spad_base + 16, or0)
                FFTRecLeaf1.buf_b.store(spad_base + 32 + 16, oi0)

                var twr1 = SIMD[DType.float32, 2](Float32(0.707106781), Float32(0.707106781))
                var twi1 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                FFTRecLeaf1.buf_b.store(spad_base + 20, or1)
                FFTRecLeaf1.buf_b.store(spad_base + 32 + 20, oi1)

                var tr2 = oi2
                oi2 = -or2
                or2 = tr2
                FFTRecLeaf1.buf_b.store(spad_base + 24, or2)
                FFTRecLeaf1.buf_b.store(spad_base + 32 + 24, oi2)

                var twr3 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var twi3 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var tr3 = or3 * twr3 - oi3 * twi3
                var ti3 = or3 * twi3 + oi3 * twr3
                or3 = tr3
                oi3 = ti3
                FFTRecLeaf1.buf_b.store(spad_base + 28, or3)
                FFTRecLeaf1.buf_b.store(spad_base + 32 + 28, oi3)


            if True:  # batch 0 scope
                # ===== stage 1, SIMD batch 0 (valid lanes: 2/2) =====
                var rr0 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 6)
                var ii0 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 32 + 6)
                var rr1 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 14)
                var ii1 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 32 + 14)
                var rr2 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 22)
                var ii2 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 32 + 22)
                var rr3 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 30)
                var ii3 = FFTRecLeaf1.buf_a.load[DType.float32, 2](spad_base + 32 + 30)

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
                FFTRecLeaf1.buf_b.store(spad_base + 18, or0)
                FFTRecLeaf1.buf_b.store(spad_base + 32 + 18, oi0)

                var twr1 = SIMD[DType.float32, 2](Float32(0.707106781), Float32(0.707106781))
                var twi1 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                FFTRecLeaf1.buf_b.store(spad_base + 22, or1)
                FFTRecLeaf1.buf_b.store(spad_base + 32 + 22, oi1)

                var tr2 = oi2
                oi2 = -or2
                or2 = tr2
                FFTRecLeaf1.buf_b.store(spad_base + 26, or2)
                FFTRecLeaf1.buf_b.store(spad_base + 32 + 26, oi2)

                var twr3 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var twi3 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var tr3 = or3 * twr3 - oi3 * twi3
                var ti3 = or3 * twi3 + oi3 * twr3
                or3 = tr3
                oi3 = ti3
                FFTRecLeaf1.buf_b.store(spad_base + 30, or3)
                FFTRecLeaf1.buf_b.store(spad_base + 32 + 30, oi3)



    @staticmethod
    def stage_2():
        ref p = FFTRecLeaf1.params[]
        comptime RADIX = 2
        comptime SIMD_ITERS = 2

        comptime WORKERS_PER_FFT = 2
        var local_id = local_uthread_id()
        var fft_slot = local_id // WORKERS_PER_FFT
        var worker_id = local_id % WORKERS_PER_FFT
        if fft_slot >= 60:
            return
        var spad_base = fft_slot * 64
        var logical_fft_id = global_uthread_id() // WORKERS_PER_FFT
        var out_batch_base = logical_fft_id * 32

        if worker_id == 0:
            if True:  # batch 0 scope
                # ===== stage 2, SIMD batch 0 (valid lanes: 4/4) =====
                var rr0 = FFTRecLeaf1.buf_b.load[DType.float32, 4](spad_base + 0)
                var ii0 = FFTRecLeaf1.buf_b.load[DType.float32, 4](spad_base + 32 + 0)
                var rr1 = FFTRecLeaf1.buf_b.load[DType.float32, 4](spad_base + 16)
                var ii1 = FFTRecLeaf1.buf_b.load[DType.float32, 4](spad_base + 32 + 16)

                # fixed radix-2 butterfly
                var or0 = rr0 + rr1
                var oi0 = ii0 + ii1
                var or1 = rr0 - rr1
                var oi1 = ii0 - ii1
                p.output_real_base.store(out_batch_base + 0, or0)
                p.output_imag_base.store(out_batch_base + 0, oi0)

                p.output_real_base.store(out_batch_base + 16, or1)
                p.output_imag_base.store(out_batch_base + 16, oi1)


            if True:  # batch 0 scope
                # ===== stage 2, SIMD batch 0 (valid lanes: 4/4) =====
                var rr0 = FFTRecLeaf1.buf_b.load[DType.float32, 4](spad_base + 4)
                var ii0 = FFTRecLeaf1.buf_b.load[DType.float32, 4](spad_base + 32 + 4)
                var rr1 = FFTRecLeaf1.buf_b.load[DType.float32, 4](spad_base + 20)
                var ii1 = FFTRecLeaf1.buf_b.load[DType.float32, 4](spad_base + 32 + 20)

                # fixed radix-2 butterfly
                var or0 = rr0 + rr1
                var oi0 = ii0 + ii1
                var or1 = rr0 - rr1
                var oi1 = ii0 - ii1
                p.output_real_base.store(out_batch_base + 4, or0)
                p.output_imag_base.store(out_batch_base + 4, oi0)

                p.output_real_base.store(out_batch_base + 20, or1)
                p.output_imag_base.store(out_batch_base + 20, oi1)


        if worker_id == 1:
            if True:  # batch 1 scope
                # ===== stage 2, SIMD batch 1 (valid lanes: 4/4) =====
                var rr0 = FFTRecLeaf1.buf_b.load[DType.float32, 4](spad_base + 8)
                var ii0 = FFTRecLeaf1.buf_b.load[DType.float32, 4](spad_base + 32 + 8)
                var rr1 = FFTRecLeaf1.buf_b.load[DType.float32, 4](spad_base + 24)
                var ii1 = FFTRecLeaf1.buf_b.load[DType.float32, 4](spad_base + 32 + 24)

                # fixed radix-2 butterfly
                var or0 = rr0 + rr1
                var oi0 = ii0 + ii1
                var or1 = rr0 - rr1
                var oi1 = ii0 - ii1
                p.output_real_base.store(out_batch_base + 8, or0)
                p.output_imag_base.store(out_batch_base + 8, oi0)

                p.output_real_base.store(out_batch_base + 24, or1)
                p.output_imag_base.store(out_batch_base + 24, oi1)


            if True:  # batch 1 scope
                # ===== stage 2, SIMD batch 1 (valid lanes: 4/4) =====
                var rr0 = FFTRecLeaf1.buf_b.load[DType.float32, 4](spad_base + 12)
                var ii0 = FFTRecLeaf1.buf_b.load[DType.float32, 4](spad_base + 32 + 12)
                var rr1 = FFTRecLeaf1.buf_b.load[DType.float32, 4](spad_base + 28)
                var ii1 = FFTRecLeaf1.buf_b.load[DType.float32, 4](spad_base + 32 + 28)

                # fixed radix-2 butterfly
                var or0 = rr0 + rr1
                var oi0 = ii0 + ii1
                var or1 = rr0 - rr1
                var oi1 = ii0 - ii1
                p.output_real_base.store(out_batch_base + 12, or0)
                p.output_imag_base.store(out_batch_base + 12, oi0)

                p.output_real_base.store(out_batch_base + 28, or1)
                p.output_imag_base.store(out_batch_base + 28, oi1)



    @staticmethod
    def device_main():
        launch_parallel[FFTRecLeaf1.stage_0]()
        launch_parallel[FFTRecLeaf1.stage_1]()
        launch_parallel[FFTRecLeaf1.stage_2]()


# ---- stage 4: FFTRecPost0 ----
comptime MAX_UTHREAD_FFTRecPost0 = 480

@fieldwise_init
struct FFTRecPost0Params(Movable):
    var src_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var src_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var dst_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var dst_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var q_table: UnsafePointer[Int, MutAnyOrigin]


struct FFTRecPost0(NDPTask):
    comptime Params = FFTRecPost0Params

    comptime tile_buf = scratchpad[3840, Float32, name="fftrecpost0_tile"]()

    @staticmethod
    def stage_0():
        ref p = FFTRecPost0.params[]
        var local_id = local_uthread_id()
        if local_id >= MAX_UTHREAD_FFTRecPost0:
            return
        var spad_base = local_id * 8

        var tile_id = global_uthread_id()
        var q = p.q_table[tile_id]
        var local_tile = tile_id - q * 240
        var t_r = local_tile // 16
        var t_c = local_tile % 16

        var src_row_0 = t_r * 2 + 0
        var src_base_0 = q * 960 + src_row_0 * 32 + t_c * 2
        var rr0 = p.src_real_base.load[width=2](src_base_0)
        var ii0 = p.src_imag_base.load[width=2](src_base_0)
        FFTRecPost0.tile_buf.store(spad_base + 0, rr0[0])
        FFTRecPost0.tile_buf.store(spad_base + 4, ii0[0])
        FFTRecPost0.tile_buf.store(spad_base + 2, rr0[1])
        FFTRecPost0.tile_buf.store(spad_base + 6, ii0[1])

        var src_row_1 = t_r * 2 + 1
        var src_base_1 = q * 960 + src_row_1 * 32 + t_c * 2
        var rr1 = p.src_real_base.load[width=2](src_base_1)
        var ii1 = p.src_imag_base.load[width=2](src_base_1)
        FFTRecPost0.tile_buf.store(spad_base + 1, rr1[0])
        FFTRecPost0.tile_buf.store(spad_base + 5, ii1[0])
        FFTRecPost0.tile_buf.store(spad_base + 3, rr1[1])
        FFTRecPost0.tile_buf.store(spad_base + 7, ii1[1])

        var dst_row_0 = t_c * 2 + 0
        var dst_base_0 = q * 960 + dst_row_0 * 30 + t_r * 2
        var or0 = FFTRecPost0.tile_buf.load[DType.float32, 2](spad_base + 0)
        var oi0 = FFTRecPost0.tile_buf.load[DType.float32, 2](spad_base + 4)
        p.dst_real_base.store(dst_base_0, or0)
        p.dst_imag_base.store(dst_base_0, oi0)
        var dst_row_1 = t_c * 2 + 1
        var dst_base_1 = q * 960 + dst_row_1 * 30 + t_r * 2
        var or1 = FFTRecPost0.tile_buf.load[DType.float32, 2](spad_base + 2)
        var oi1 = FFTRecPost0.tile_buf.load[DType.float32, 2](spad_base + 6)
        p.dst_real_base.store(dst_base_1, or1)
        p.dst_imag_base.store(dst_base_1, oi1)

    @staticmethod
    def device_main():
        launch_parallel[FFTRecPost0.stage_0]()


# ============================================================================
# HOST MAIN -- allocates DRAM buffers/twiddle tables, launches every
# kernel above in order, checks the result against an independent
# reference DFT computed at runtime.
# ============================================================================

def main() raises:
    if FFTRecPre0.emit_ir_if_asked():
        return

    var total_elems = 1920
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

    var twiddle2_real = cxl_alloc[Float32](960)
    var twiddle2_imag = cxl_alloc[Float32](960)
    var tw_pi2 = Float64(3.141592653589793)
    var tw_sign2 = Float64(-1.0)
    var tw_r2 = 0
    while tw_r2 < 32:
        var tw_c2 = 0
        while tw_c2 < 30:
            var tw_angle2 = tw_sign2 * 2.0 * tw_pi2 * Float64(tw_r2) * Float64(tw_c2) / Float64(960)
            var tw_addr2 = tw_r2 * 30 + tw_c2
            twiddle2_real[tw_addr2] = Float32(host_cos(tw_angle2))
            twiddle2_imag[tw_addr2] = Float32(host_sin(tw_angle2))
            tw_c2 += 1
        tw_r2 += 1

    var qtab0 = cxl_alloc[Int](480)
    qtab0[0] = 0
    qtab0[1] = 0
    qtab0[2] = 0
    qtab0[3] = 0
    qtab0[4] = 0
    qtab0[5] = 0
    qtab0[6] = 0
    qtab0[7] = 0
    qtab0[8] = 0
    qtab0[9] = 0
    qtab0[10] = 0
    qtab0[11] = 0
    qtab0[12] = 0
    qtab0[13] = 0
    qtab0[14] = 0
    qtab0[15] = 0
    qtab0[16] = 0
    qtab0[17] = 0
    qtab0[18] = 0
    qtab0[19] = 0
    qtab0[20] = 0
    qtab0[21] = 0
    qtab0[22] = 0
    qtab0[23] = 0
    qtab0[24] = 0
    qtab0[25] = 0
    qtab0[26] = 0
    qtab0[27] = 0
    qtab0[28] = 0
    qtab0[29] = 0
    qtab0[30] = 0
    qtab0[31] = 0
    qtab0[32] = 0
    qtab0[33] = 0
    qtab0[34] = 0
    qtab0[35] = 0
    qtab0[36] = 0
    qtab0[37] = 0
    qtab0[38] = 0
    qtab0[39] = 0
    qtab0[40] = 0
    qtab0[41] = 0
    qtab0[42] = 0
    qtab0[43] = 0
    qtab0[44] = 0
    qtab0[45] = 0
    qtab0[46] = 0
    qtab0[47] = 0
    qtab0[48] = 0
    qtab0[49] = 0
    qtab0[50] = 0
    qtab0[51] = 0
    qtab0[52] = 0
    qtab0[53] = 0
    qtab0[54] = 0
    qtab0[55] = 0
    qtab0[56] = 0
    qtab0[57] = 0
    qtab0[58] = 0
    qtab0[59] = 0
    qtab0[60] = 0
    qtab0[61] = 0
    qtab0[62] = 0
    qtab0[63] = 0
    qtab0[64] = 0
    qtab0[65] = 0
    qtab0[66] = 0
    qtab0[67] = 0
    qtab0[68] = 0
    qtab0[69] = 0
    qtab0[70] = 0
    qtab0[71] = 0
    qtab0[72] = 0
    qtab0[73] = 0
    qtab0[74] = 0
    qtab0[75] = 0
    qtab0[76] = 0
    qtab0[77] = 0
    qtab0[78] = 0
    qtab0[79] = 0
    qtab0[80] = 0
    qtab0[81] = 0
    qtab0[82] = 0
    qtab0[83] = 0
    qtab0[84] = 0
    qtab0[85] = 0
    qtab0[86] = 0
    qtab0[87] = 0
    qtab0[88] = 0
    qtab0[89] = 0
    qtab0[90] = 0
    qtab0[91] = 0
    qtab0[92] = 0
    qtab0[93] = 0
    qtab0[94] = 0
    qtab0[95] = 0
    qtab0[96] = 0
    qtab0[97] = 0
    qtab0[98] = 0
    qtab0[99] = 0
    qtab0[100] = 0
    qtab0[101] = 0
    qtab0[102] = 0
    qtab0[103] = 0
    qtab0[104] = 0
    qtab0[105] = 0
    qtab0[106] = 0
    qtab0[107] = 0
    qtab0[108] = 0
    qtab0[109] = 0
    qtab0[110] = 0
    qtab0[111] = 0
    qtab0[112] = 0
    qtab0[113] = 0
    qtab0[114] = 0
    qtab0[115] = 0
    qtab0[116] = 0
    qtab0[117] = 0
    qtab0[118] = 0
    qtab0[119] = 0
    qtab0[120] = 0
    qtab0[121] = 0
    qtab0[122] = 0
    qtab0[123] = 0
    qtab0[124] = 0
    qtab0[125] = 0
    qtab0[126] = 0
    qtab0[127] = 0
    qtab0[128] = 0
    qtab0[129] = 0
    qtab0[130] = 0
    qtab0[131] = 0
    qtab0[132] = 0
    qtab0[133] = 0
    qtab0[134] = 0
    qtab0[135] = 0
    qtab0[136] = 0
    qtab0[137] = 0
    qtab0[138] = 0
    qtab0[139] = 0
    qtab0[140] = 0
    qtab0[141] = 0
    qtab0[142] = 0
    qtab0[143] = 0
    qtab0[144] = 0
    qtab0[145] = 0
    qtab0[146] = 0
    qtab0[147] = 0
    qtab0[148] = 0
    qtab0[149] = 0
    qtab0[150] = 0
    qtab0[151] = 0
    qtab0[152] = 0
    qtab0[153] = 0
    qtab0[154] = 0
    qtab0[155] = 0
    qtab0[156] = 0
    qtab0[157] = 0
    qtab0[158] = 0
    qtab0[159] = 0
    qtab0[160] = 0
    qtab0[161] = 0
    qtab0[162] = 0
    qtab0[163] = 0
    qtab0[164] = 0
    qtab0[165] = 0
    qtab0[166] = 0
    qtab0[167] = 0
    qtab0[168] = 0
    qtab0[169] = 0
    qtab0[170] = 0
    qtab0[171] = 0
    qtab0[172] = 0
    qtab0[173] = 0
    qtab0[174] = 0
    qtab0[175] = 0
    qtab0[176] = 0
    qtab0[177] = 0
    qtab0[178] = 0
    qtab0[179] = 0
    qtab0[180] = 0
    qtab0[181] = 0
    qtab0[182] = 0
    qtab0[183] = 0
    qtab0[184] = 0
    qtab0[185] = 0
    qtab0[186] = 0
    qtab0[187] = 0
    qtab0[188] = 0
    qtab0[189] = 0
    qtab0[190] = 0
    qtab0[191] = 0
    qtab0[192] = 0
    qtab0[193] = 0
    qtab0[194] = 0
    qtab0[195] = 0
    qtab0[196] = 0
    qtab0[197] = 0
    qtab0[198] = 0
    qtab0[199] = 0
    qtab0[200] = 0
    qtab0[201] = 0
    qtab0[202] = 0
    qtab0[203] = 0
    qtab0[204] = 0
    qtab0[205] = 0
    qtab0[206] = 0
    qtab0[207] = 0
    qtab0[208] = 0
    qtab0[209] = 0
    qtab0[210] = 0
    qtab0[211] = 0
    qtab0[212] = 0
    qtab0[213] = 0
    qtab0[214] = 0
    qtab0[215] = 0
    qtab0[216] = 0
    qtab0[217] = 0
    qtab0[218] = 0
    qtab0[219] = 0
    qtab0[220] = 0
    qtab0[221] = 0
    qtab0[222] = 0
    qtab0[223] = 0
    qtab0[224] = 0
    qtab0[225] = 0
    qtab0[226] = 0
    qtab0[227] = 0
    qtab0[228] = 0
    qtab0[229] = 0
    qtab0[230] = 0
    qtab0[231] = 0
    qtab0[232] = 0
    qtab0[233] = 0
    qtab0[234] = 0
    qtab0[235] = 0
    qtab0[236] = 0
    qtab0[237] = 0
    qtab0[238] = 0
    qtab0[239] = 0
    qtab0[240] = 1
    qtab0[241] = 1
    qtab0[242] = 1
    qtab0[243] = 1
    qtab0[244] = 1
    qtab0[245] = 1
    qtab0[246] = 1
    qtab0[247] = 1
    qtab0[248] = 1
    qtab0[249] = 1
    qtab0[250] = 1
    qtab0[251] = 1
    qtab0[252] = 1
    qtab0[253] = 1
    qtab0[254] = 1
    qtab0[255] = 1
    qtab0[256] = 1
    qtab0[257] = 1
    qtab0[258] = 1
    qtab0[259] = 1
    qtab0[260] = 1
    qtab0[261] = 1
    qtab0[262] = 1
    qtab0[263] = 1
    qtab0[264] = 1
    qtab0[265] = 1
    qtab0[266] = 1
    qtab0[267] = 1
    qtab0[268] = 1
    qtab0[269] = 1
    qtab0[270] = 1
    qtab0[271] = 1
    qtab0[272] = 1
    qtab0[273] = 1
    qtab0[274] = 1
    qtab0[275] = 1
    qtab0[276] = 1
    qtab0[277] = 1
    qtab0[278] = 1
    qtab0[279] = 1
    qtab0[280] = 1
    qtab0[281] = 1
    qtab0[282] = 1
    qtab0[283] = 1
    qtab0[284] = 1
    qtab0[285] = 1
    qtab0[286] = 1
    qtab0[287] = 1
    qtab0[288] = 1
    qtab0[289] = 1
    qtab0[290] = 1
    qtab0[291] = 1
    qtab0[292] = 1
    qtab0[293] = 1
    qtab0[294] = 1
    qtab0[295] = 1
    qtab0[296] = 1
    qtab0[297] = 1
    qtab0[298] = 1
    qtab0[299] = 1
    qtab0[300] = 1
    qtab0[301] = 1
    qtab0[302] = 1
    qtab0[303] = 1
    qtab0[304] = 1
    qtab0[305] = 1
    qtab0[306] = 1
    qtab0[307] = 1
    qtab0[308] = 1
    qtab0[309] = 1
    qtab0[310] = 1
    qtab0[311] = 1
    qtab0[312] = 1
    qtab0[313] = 1
    qtab0[314] = 1
    qtab0[315] = 1
    qtab0[316] = 1
    qtab0[317] = 1
    qtab0[318] = 1
    qtab0[319] = 1
    qtab0[320] = 1
    qtab0[321] = 1
    qtab0[322] = 1
    qtab0[323] = 1
    qtab0[324] = 1
    qtab0[325] = 1
    qtab0[326] = 1
    qtab0[327] = 1
    qtab0[328] = 1
    qtab0[329] = 1
    qtab0[330] = 1
    qtab0[331] = 1
    qtab0[332] = 1
    qtab0[333] = 1
    qtab0[334] = 1
    qtab0[335] = 1
    qtab0[336] = 1
    qtab0[337] = 1
    qtab0[338] = 1
    qtab0[339] = 1
    qtab0[340] = 1
    qtab0[341] = 1
    qtab0[342] = 1
    qtab0[343] = 1
    qtab0[344] = 1
    qtab0[345] = 1
    qtab0[346] = 1
    qtab0[347] = 1
    qtab0[348] = 1
    qtab0[349] = 1
    qtab0[350] = 1
    qtab0[351] = 1
    qtab0[352] = 1
    qtab0[353] = 1
    qtab0[354] = 1
    qtab0[355] = 1
    qtab0[356] = 1
    qtab0[357] = 1
    qtab0[358] = 1
    qtab0[359] = 1
    qtab0[360] = 1
    qtab0[361] = 1
    qtab0[362] = 1
    qtab0[363] = 1
    qtab0[364] = 1
    qtab0[365] = 1
    qtab0[366] = 1
    qtab0[367] = 1
    qtab0[368] = 1
    qtab0[369] = 1
    qtab0[370] = 1
    qtab0[371] = 1
    qtab0[372] = 1
    qtab0[373] = 1
    qtab0[374] = 1
    qtab0[375] = 1
    qtab0[376] = 1
    qtab0[377] = 1
    qtab0[378] = 1
    qtab0[379] = 1
    qtab0[380] = 1
    qtab0[381] = 1
    qtab0[382] = 1
    qtab0[383] = 1
    qtab0[384] = 1
    qtab0[385] = 1
    qtab0[386] = 1
    qtab0[387] = 1
    qtab0[388] = 1
    qtab0[389] = 1
    qtab0[390] = 1
    qtab0[391] = 1
    qtab0[392] = 1
    qtab0[393] = 1
    qtab0[394] = 1
    qtab0[395] = 1
    qtab0[396] = 1
    qtab0[397] = 1
    qtab0[398] = 1
    qtab0[399] = 1
    qtab0[400] = 1
    qtab0[401] = 1
    qtab0[402] = 1
    qtab0[403] = 1
    qtab0[404] = 1
    qtab0[405] = 1
    qtab0[406] = 1
    qtab0[407] = 1
    qtab0[408] = 1
    qtab0[409] = 1
    qtab0[410] = 1
    qtab0[411] = 1
    qtab0[412] = 1
    qtab0[413] = 1
    qtab0[414] = 1
    qtab0[415] = 1
    qtab0[416] = 1
    qtab0[417] = 1
    qtab0[418] = 1
    qtab0[419] = 1
    qtab0[420] = 1
    qtab0[421] = 1
    qtab0[422] = 1
    qtab0[423] = 1
    qtab0[424] = 1
    qtab0[425] = 1
    qtab0[426] = 1
    qtab0[427] = 1
    qtab0[428] = 1
    qtab0[429] = 1
    qtab0[430] = 1
    qtab0[431] = 1
    qtab0[432] = 1
    qtab0[433] = 1
    qtab0[434] = 1
    qtab0[435] = 1
    qtab0[436] = 1
    qtab0[437] = 1
    qtab0[438] = 1
    qtab0[439] = 1
    qtab0[440] = 1
    qtab0[441] = 1
    qtab0[442] = 1
    qtab0[443] = 1
    qtab0[444] = 1
    qtab0[445] = 1
    qtab0[446] = 1
    qtab0[447] = 1
    qtab0[448] = 1
    qtab0[449] = 1
    qtab0[450] = 1
    qtab0[451] = 1
    qtab0[452] = 1
    qtab0[453] = 1
    qtab0[454] = 1
    qtab0[455] = 1
    qtab0[456] = 1
    qtab0[457] = 1
    qtab0[458] = 1
    qtab0[459] = 1
    qtab0[460] = 1
    qtab0[461] = 1
    qtab0[462] = 1
    qtab0[463] = 1
    qtab0[464] = 1
    qtab0[465] = 1
    qtab0[466] = 1
    qtab0[467] = 1
    qtab0[468] = 1
    qtab0[469] = 1
    qtab0[470] = 1
    qtab0[471] = 1
    qtab0[472] = 1
    qtab0[473] = 1
    qtab0[474] = 1
    qtab0[475] = 1
    qtab0[476] = 1
    qtab0[477] = 1
    qtab0[478] = 1
    qtab0[479] = 1

    var qtab2 = cxl_alloc[Int](480)
    qtab2[0] = 0
    qtab2[1] = 0
    qtab2[2] = 0
    qtab2[3] = 0
    qtab2[4] = 0
    qtab2[5] = 0
    qtab2[6] = 0
    qtab2[7] = 0
    qtab2[8] = 0
    qtab2[9] = 0
    qtab2[10] = 0
    qtab2[11] = 0
    qtab2[12] = 0
    qtab2[13] = 0
    qtab2[14] = 0
    qtab2[15] = 0
    qtab2[16] = 0
    qtab2[17] = 0
    qtab2[18] = 0
    qtab2[19] = 0
    qtab2[20] = 0
    qtab2[21] = 0
    qtab2[22] = 0
    qtab2[23] = 0
    qtab2[24] = 0
    qtab2[25] = 0
    qtab2[26] = 0
    qtab2[27] = 0
    qtab2[28] = 0
    qtab2[29] = 0
    qtab2[30] = 0
    qtab2[31] = 0
    qtab2[32] = 0
    qtab2[33] = 0
    qtab2[34] = 0
    qtab2[35] = 0
    qtab2[36] = 0
    qtab2[37] = 0
    qtab2[38] = 0
    qtab2[39] = 0
    qtab2[40] = 0
    qtab2[41] = 0
    qtab2[42] = 0
    qtab2[43] = 0
    qtab2[44] = 0
    qtab2[45] = 0
    qtab2[46] = 0
    qtab2[47] = 0
    qtab2[48] = 0
    qtab2[49] = 0
    qtab2[50] = 0
    qtab2[51] = 0
    qtab2[52] = 0
    qtab2[53] = 0
    qtab2[54] = 0
    qtab2[55] = 0
    qtab2[56] = 0
    qtab2[57] = 0
    qtab2[58] = 0
    qtab2[59] = 0
    qtab2[60] = 0
    qtab2[61] = 0
    qtab2[62] = 0
    qtab2[63] = 0
    qtab2[64] = 0
    qtab2[65] = 0
    qtab2[66] = 0
    qtab2[67] = 0
    qtab2[68] = 0
    qtab2[69] = 0
    qtab2[70] = 0
    qtab2[71] = 0
    qtab2[72] = 0
    qtab2[73] = 0
    qtab2[74] = 0
    qtab2[75] = 0
    qtab2[76] = 0
    qtab2[77] = 0
    qtab2[78] = 0
    qtab2[79] = 0
    qtab2[80] = 0
    qtab2[81] = 0
    qtab2[82] = 0
    qtab2[83] = 0
    qtab2[84] = 0
    qtab2[85] = 0
    qtab2[86] = 0
    qtab2[87] = 0
    qtab2[88] = 0
    qtab2[89] = 0
    qtab2[90] = 0
    qtab2[91] = 0
    qtab2[92] = 0
    qtab2[93] = 0
    qtab2[94] = 0
    qtab2[95] = 0
    qtab2[96] = 0
    qtab2[97] = 0
    qtab2[98] = 0
    qtab2[99] = 0
    qtab2[100] = 0
    qtab2[101] = 0
    qtab2[102] = 0
    qtab2[103] = 0
    qtab2[104] = 0
    qtab2[105] = 0
    qtab2[106] = 0
    qtab2[107] = 0
    qtab2[108] = 0
    qtab2[109] = 0
    qtab2[110] = 0
    qtab2[111] = 0
    qtab2[112] = 0
    qtab2[113] = 0
    qtab2[114] = 0
    qtab2[115] = 0
    qtab2[116] = 0
    qtab2[117] = 0
    qtab2[118] = 0
    qtab2[119] = 0
    qtab2[120] = 0
    qtab2[121] = 0
    qtab2[122] = 0
    qtab2[123] = 0
    qtab2[124] = 0
    qtab2[125] = 0
    qtab2[126] = 0
    qtab2[127] = 0
    qtab2[128] = 0
    qtab2[129] = 0
    qtab2[130] = 0
    qtab2[131] = 0
    qtab2[132] = 0
    qtab2[133] = 0
    qtab2[134] = 0
    qtab2[135] = 0
    qtab2[136] = 0
    qtab2[137] = 0
    qtab2[138] = 0
    qtab2[139] = 0
    qtab2[140] = 0
    qtab2[141] = 0
    qtab2[142] = 0
    qtab2[143] = 0
    qtab2[144] = 0
    qtab2[145] = 0
    qtab2[146] = 0
    qtab2[147] = 0
    qtab2[148] = 0
    qtab2[149] = 0
    qtab2[150] = 0
    qtab2[151] = 0
    qtab2[152] = 0
    qtab2[153] = 0
    qtab2[154] = 0
    qtab2[155] = 0
    qtab2[156] = 0
    qtab2[157] = 0
    qtab2[158] = 0
    qtab2[159] = 0
    qtab2[160] = 0
    qtab2[161] = 0
    qtab2[162] = 0
    qtab2[163] = 0
    qtab2[164] = 0
    qtab2[165] = 0
    qtab2[166] = 0
    qtab2[167] = 0
    qtab2[168] = 0
    qtab2[169] = 0
    qtab2[170] = 0
    qtab2[171] = 0
    qtab2[172] = 0
    qtab2[173] = 0
    qtab2[174] = 0
    qtab2[175] = 0
    qtab2[176] = 0
    qtab2[177] = 0
    qtab2[178] = 0
    qtab2[179] = 0
    qtab2[180] = 0
    qtab2[181] = 0
    qtab2[182] = 0
    qtab2[183] = 0
    qtab2[184] = 0
    qtab2[185] = 0
    qtab2[186] = 0
    qtab2[187] = 0
    qtab2[188] = 0
    qtab2[189] = 0
    qtab2[190] = 0
    qtab2[191] = 0
    qtab2[192] = 0
    qtab2[193] = 0
    qtab2[194] = 0
    qtab2[195] = 0
    qtab2[196] = 0
    qtab2[197] = 0
    qtab2[198] = 0
    qtab2[199] = 0
    qtab2[200] = 0
    qtab2[201] = 0
    qtab2[202] = 0
    qtab2[203] = 0
    qtab2[204] = 0
    qtab2[205] = 0
    qtab2[206] = 0
    qtab2[207] = 0
    qtab2[208] = 0
    qtab2[209] = 0
    qtab2[210] = 0
    qtab2[211] = 0
    qtab2[212] = 0
    qtab2[213] = 0
    qtab2[214] = 0
    qtab2[215] = 0
    qtab2[216] = 0
    qtab2[217] = 0
    qtab2[218] = 0
    qtab2[219] = 0
    qtab2[220] = 0
    qtab2[221] = 0
    qtab2[222] = 0
    qtab2[223] = 0
    qtab2[224] = 0
    qtab2[225] = 0
    qtab2[226] = 0
    qtab2[227] = 0
    qtab2[228] = 0
    qtab2[229] = 0
    qtab2[230] = 0
    qtab2[231] = 0
    qtab2[232] = 0
    qtab2[233] = 0
    qtab2[234] = 0
    qtab2[235] = 0
    qtab2[236] = 0
    qtab2[237] = 0
    qtab2[238] = 0
    qtab2[239] = 0
    qtab2[240] = 1
    qtab2[241] = 1
    qtab2[242] = 1
    qtab2[243] = 1
    qtab2[244] = 1
    qtab2[245] = 1
    qtab2[246] = 1
    qtab2[247] = 1
    qtab2[248] = 1
    qtab2[249] = 1
    qtab2[250] = 1
    qtab2[251] = 1
    qtab2[252] = 1
    qtab2[253] = 1
    qtab2[254] = 1
    qtab2[255] = 1
    qtab2[256] = 1
    qtab2[257] = 1
    qtab2[258] = 1
    qtab2[259] = 1
    qtab2[260] = 1
    qtab2[261] = 1
    qtab2[262] = 1
    qtab2[263] = 1
    qtab2[264] = 1
    qtab2[265] = 1
    qtab2[266] = 1
    qtab2[267] = 1
    qtab2[268] = 1
    qtab2[269] = 1
    qtab2[270] = 1
    qtab2[271] = 1
    qtab2[272] = 1
    qtab2[273] = 1
    qtab2[274] = 1
    qtab2[275] = 1
    qtab2[276] = 1
    qtab2[277] = 1
    qtab2[278] = 1
    qtab2[279] = 1
    qtab2[280] = 1
    qtab2[281] = 1
    qtab2[282] = 1
    qtab2[283] = 1
    qtab2[284] = 1
    qtab2[285] = 1
    qtab2[286] = 1
    qtab2[287] = 1
    qtab2[288] = 1
    qtab2[289] = 1
    qtab2[290] = 1
    qtab2[291] = 1
    qtab2[292] = 1
    qtab2[293] = 1
    qtab2[294] = 1
    qtab2[295] = 1
    qtab2[296] = 1
    qtab2[297] = 1
    qtab2[298] = 1
    qtab2[299] = 1
    qtab2[300] = 1
    qtab2[301] = 1
    qtab2[302] = 1
    qtab2[303] = 1
    qtab2[304] = 1
    qtab2[305] = 1
    qtab2[306] = 1
    qtab2[307] = 1
    qtab2[308] = 1
    qtab2[309] = 1
    qtab2[310] = 1
    qtab2[311] = 1
    qtab2[312] = 1
    qtab2[313] = 1
    qtab2[314] = 1
    qtab2[315] = 1
    qtab2[316] = 1
    qtab2[317] = 1
    qtab2[318] = 1
    qtab2[319] = 1
    qtab2[320] = 1
    qtab2[321] = 1
    qtab2[322] = 1
    qtab2[323] = 1
    qtab2[324] = 1
    qtab2[325] = 1
    qtab2[326] = 1
    qtab2[327] = 1
    qtab2[328] = 1
    qtab2[329] = 1
    qtab2[330] = 1
    qtab2[331] = 1
    qtab2[332] = 1
    qtab2[333] = 1
    qtab2[334] = 1
    qtab2[335] = 1
    qtab2[336] = 1
    qtab2[337] = 1
    qtab2[338] = 1
    qtab2[339] = 1
    qtab2[340] = 1
    qtab2[341] = 1
    qtab2[342] = 1
    qtab2[343] = 1
    qtab2[344] = 1
    qtab2[345] = 1
    qtab2[346] = 1
    qtab2[347] = 1
    qtab2[348] = 1
    qtab2[349] = 1
    qtab2[350] = 1
    qtab2[351] = 1
    qtab2[352] = 1
    qtab2[353] = 1
    qtab2[354] = 1
    qtab2[355] = 1
    qtab2[356] = 1
    qtab2[357] = 1
    qtab2[358] = 1
    qtab2[359] = 1
    qtab2[360] = 1
    qtab2[361] = 1
    qtab2[362] = 1
    qtab2[363] = 1
    qtab2[364] = 1
    qtab2[365] = 1
    qtab2[366] = 1
    qtab2[367] = 1
    qtab2[368] = 1
    qtab2[369] = 1
    qtab2[370] = 1
    qtab2[371] = 1
    qtab2[372] = 1
    qtab2[373] = 1
    qtab2[374] = 1
    qtab2[375] = 1
    qtab2[376] = 1
    qtab2[377] = 1
    qtab2[378] = 1
    qtab2[379] = 1
    qtab2[380] = 1
    qtab2[381] = 1
    qtab2[382] = 1
    qtab2[383] = 1
    qtab2[384] = 1
    qtab2[385] = 1
    qtab2[386] = 1
    qtab2[387] = 1
    qtab2[388] = 1
    qtab2[389] = 1
    qtab2[390] = 1
    qtab2[391] = 1
    qtab2[392] = 1
    qtab2[393] = 1
    qtab2[394] = 1
    qtab2[395] = 1
    qtab2[396] = 1
    qtab2[397] = 1
    qtab2[398] = 1
    qtab2[399] = 1
    qtab2[400] = 1
    qtab2[401] = 1
    qtab2[402] = 1
    qtab2[403] = 1
    qtab2[404] = 1
    qtab2[405] = 1
    qtab2[406] = 1
    qtab2[407] = 1
    qtab2[408] = 1
    qtab2[409] = 1
    qtab2[410] = 1
    qtab2[411] = 1
    qtab2[412] = 1
    qtab2[413] = 1
    qtab2[414] = 1
    qtab2[415] = 1
    qtab2[416] = 1
    qtab2[417] = 1
    qtab2[418] = 1
    qtab2[419] = 1
    qtab2[420] = 1
    qtab2[421] = 1
    qtab2[422] = 1
    qtab2[423] = 1
    qtab2[424] = 1
    qtab2[425] = 1
    qtab2[426] = 1
    qtab2[427] = 1
    qtab2[428] = 1
    qtab2[429] = 1
    qtab2[430] = 1
    qtab2[431] = 1
    qtab2[432] = 1
    qtab2[433] = 1
    qtab2[434] = 1
    qtab2[435] = 1
    qtab2[436] = 1
    qtab2[437] = 1
    qtab2[438] = 1
    qtab2[439] = 1
    qtab2[440] = 1
    qtab2[441] = 1
    qtab2[442] = 1
    qtab2[443] = 1
    qtab2[444] = 1
    qtab2[445] = 1
    qtab2[446] = 1
    qtab2[447] = 1
    qtab2[448] = 1
    qtab2[449] = 1
    qtab2[450] = 1
    qtab2[451] = 1
    qtab2[452] = 1
    qtab2[453] = 1
    qtab2[454] = 1
    qtab2[455] = 1
    qtab2[456] = 1
    qtab2[457] = 1
    qtab2[458] = 1
    qtab2[459] = 1
    qtab2[460] = 1
    qtab2[461] = 1
    qtab2[462] = 1
    qtab2[463] = 1
    qtab2[464] = 1
    qtab2[465] = 1
    qtab2[466] = 1
    qtab2[467] = 1
    qtab2[468] = 1
    qtab2[469] = 1
    qtab2[470] = 1
    qtab2[471] = 1
    qtab2[472] = 1
    qtab2[473] = 1
    qtab2[474] = 1
    qtab2[475] = 1
    qtab2[476] = 1
    qtab2[477] = 1
    qtab2[478] = 1
    qtab2[479] = 1

    var trtab2 = cxl_alloc[Int](240)
    trtab2[0] = 0
    trtab2[1] = 0
    trtab2[2] = 0
    trtab2[3] = 0
    trtab2[4] = 0
    trtab2[5] = 0
    trtab2[6] = 0
    trtab2[7] = 0
    trtab2[8] = 0
    trtab2[9] = 0
    trtab2[10] = 0
    trtab2[11] = 0
    trtab2[12] = 0
    trtab2[13] = 0
    trtab2[14] = 0
    trtab2[15] = 1
    trtab2[16] = 1
    trtab2[17] = 1
    trtab2[18] = 1
    trtab2[19] = 1
    trtab2[20] = 1
    trtab2[21] = 1
    trtab2[22] = 1
    trtab2[23] = 1
    trtab2[24] = 1
    trtab2[25] = 1
    trtab2[26] = 1
    trtab2[27] = 1
    trtab2[28] = 1
    trtab2[29] = 1
    trtab2[30] = 2
    trtab2[31] = 2
    trtab2[32] = 2
    trtab2[33] = 2
    trtab2[34] = 2
    trtab2[35] = 2
    trtab2[36] = 2
    trtab2[37] = 2
    trtab2[38] = 2
    trtab2[39] = 2
    trtab2[40] = 2
    trtab2[41] = 2
    trtab2[42] = 2
    trtab2[43] = 2
    trtab2[44] = 2
    trtab2[45] = 3
    trtab2[46] = 3
    trtab2[47] = 3
    trtab2[48] = 3
    trtab2[49] = 3
    trtab2[50] = 3
    trtab2[51] = 3
    trtab2[52] = 3
    trtab2[53] = 3
    trtab2[54] = 3
    trtab2[55] = 3
    trtab2[56] = 3
    trtab2[57] = 3
    trtab2[58] = 3
    trtab2[59] = 3
    trtab2[60] = 4
    trtab2[61] = 4
    trtab2[62] = 4
    trtab2[63] = 4
    trtab2[64] = 4
    trtab2[65] = 4
    trtab2[66] = 4
    trtab2[67] = 4
    trtab2[68] = 4
    trtab2[69] = 4
    trtab2[70] = 4
    trtab2[71] = 4
    trtab2[72] = 4
    trtab2[73] = 4
    trtab2[74] = 4
    trtab2[75] = 5
    trtab2[76] = 5
    trtab2[77] = 5
    trtab2[78] = 5
    trtab2[79] = 5
    trtab2[80] = 5
    trtab2[81] = 5
    trtab2[82] = 5
    trtab2[83] = 5
    trtab2[84] = 5
    trtab2[85] = 5
    trtab2[86] = 5
    trtab2[87] = 5
    trtab2[88] = 5
    trtab2[89] = 5
    trtab2[90] = 6
    trtab2[91] = 6
    trtab2[92] = 6
    trtab2[93] = 6
    trtab2[94] = 6
    trtab2[95] = 6
    trtab2[96] = 6
    trtab2[97] = 6
    trtab2[98] = 6
    trtab2[99] = 6
    trtab2[100] = 6
    trtab2[101] = 6
    trtab2[102] = 6
    trtab2[103] = 6
    trtab2[104] = 6
    trtab2[105] = 7
    trtab2[106] = 7
    trtab2[107] = 7
    trtab2[108] = 7
    trtab2[109] = 7
    trtab2[110] = 7
    trtab2[111] = 7
    trtab2[112] = 7
    trtab2[113] = 7
    trtab2[114] = 7
    trtab2[115] = 7
    trtab2[116] = 7
    trtab2[117] = 7
    trtab2[118] = 7
    trtab2[119] = 7
    trtab2[120] = 8
    trtab2[121] = 8
    trtab2[122] = 8
    trtab2[123] = 8
    trtab2[124] = 8
    trtab2[125] = 8
    trtab2[126] = 8
    trtab2[127] = 8
    trtab2[128] = 8
    trtab2[129] = 8
    trtab2[130] = 8
    trtab2[131] = 8
    trtab2[132] = 8
    trtab2[133] = 8
    trtab2[134] = 8
    trtab2[135] = 9
    trtab2[136] = 9
    trtab2[137] = 9
    trtab2[138] = 9
    trtab2[139] = 9
    trtab2[140] = 9
    trtab2[141] = 9
    trtab2[142] = 9
    trtab2[143] = 9
    trtab2[144] = 9
    trtab2[145] = 9
    trtab2[146] = 9
    trtab2[147] = 9
    trtab2[148] = 9
    trtab2[149] = 9
    trtab2[150] = 10
    trtab2[151] = 10
    trtab2[152] = 10
    trtab2[153] = 10
    trtab2[154] = 10
    trtab2[155] = 10
    trtab2[156] = 10
    trtab2[157] = 10
    trtab2[158] = 10
    trtab2[159] = 10
    trtab2[160] = 10
    trtab2[161] = 10
    trtab2[162] = 10
    trtab2[163] = 10
    trtab2[164] = 10
    trtab2[165] = 11
    trtab2[166] = 11
    trtab2[167] = 11
    trtab2[168] = 11
    trtab2[169] = 11
    trtab2[170] = 11
    trtab2[171] = 11
    trtab2[172] = 11
    trtab2[173] = 11
    trtab2[174] = 11
    trtab2[175] = 11
    trtab2[176] = 11
    trtab2[177] = 11
    trtab2[178] = 11
    trtab2[179] = 11
    trtab2[180] = 12
    trtab2[181] = 12
    trtab2[182] = 12
    trtab2[183] = 12
    trtab2[184] = 12
    trtab2[185] = 12
    trtab2[186] = 12
    trtab2[187] = 12
    trtab2[188] = 12
    trtab2[189] = 12
    trtab2[190] = 12
    trtab2[191] = 12
    trtab2[192] = 12
    trtab2[193] = 12
    trtab2[194] = 12
    trtab2[195] = 13
    trtab2[196] = 13
    trtab2[197] = 13
    trtab2[198] = 13
    trtab2[199] = 13
    trtab2[200] = 13
    trtab2[201] = 13
    trtab2[202] = 13
    trtab2[203] = 13
    trtab2[204] = 13
    trtab2[205] = 13
    trtab2[206] = 13
    trtab2[207] = 13
    trtab2[208] = 13
    trtab2[209] = 13
    trtab2[210] = 14
    trtab2[211] = 14
    trtab2[212] = 14
    trtab2[213] = 14
    trtab2[214] = 14
    trtab2[215] = 14
    trtab2[216] = 14
    trtab2[217] = 14
    trtab2[218] = 14
    trtab2[219] = 14
    trtab2[220] = 14
    trtab2[221] = 14
    trtab2[222] = 14
    trtab2[223] = 14
    trtab2[224] = 14
    trtab2[225] = 15
    trtab2[226] = 15
    trtab2[227] = 15
    trtab2[228] = 15
    trtab2[229] = 15
    trtab2[230] = 15
    trtab2[231] = 15
    trtab2[232] = 15
    trtab2[233] = 15
    trtab2[234] = 15
    trtab2[235] = 15
    trtab2[236] = 15
    trtab2[237] = 15
    trtab2[238] = 15
    trtab2[239] = 15

    var qtab4 = cxl_alloc[Int](480)
    qtab4[0] = 0
    qtab4[1] = 0
    qtab4[2] = 0
    qtab4[3] = 0
    qtab4[4] = 0
    qtab4[5] = 0
    qtab4[6] = 0
    qtab4[7] = 0
    qtab4[8] = 0
    qtab4[9] = 0
    qtab4[10] = 0
    qtab4[11] = 0
    qtab4[12] = 0
    qtab4[13] = 0
    qtab4[14] = 0
    qtab4[15] = 0
    qtab4[16] = 0
    qtab4[17] = 0
    qtab4[18] = 0
    qtab4[19] = 0
    qtab4[20] = 0
    qtab4[21] = 0
    qtab4[22] = 0
    qtab4[23] = 0
    qtab4[24] = 0
    qtab4[25] = 0
    qtab4[26] = 0
    qtab4[27] = 0
    qtab4[28] = 0
    qtab4[29] = 0
    qtab4[30] = 0
    qtab4[31] = 0
    qtab4[32] = 0
    qtab4[33] = 0
    qtab4[34] = 0
    qtab4[35] = 0
    qtab4[36] = 0
    qtab4[37] = 0
    qtab4[38] = 0
    qtab4[39] = 0
    qtab4[40] = 0
    qtab4[41] = 0
    qtab4[42] = 0
    qtab4[43] = 0
    qtab4[44] = 0
    qtab4[45] = 0
    qtab4[46] = 0
    qtab4[47] = 0
    qtab4[48] = 0
    qtab4[49] = 0
    qtab4[50] = 0
    qtab4[51] = 0
    qtab4[52] = 0
    qtab4[53] = 0
    qtab4[54] = 0
    qtab4[55] = 0
    qtab4[56] = 0
    qtab4[57] = 0
    qtab4[58] = 0
    qtab4[59] = 0
    qtab4[60] = 0
    qtab4[61] = 0
    qtab4[62] = 0
    qtab4[63] = 0
    qtab4[64] = 0
    qtab4[65] = 0
    qtab4[66] = 0
    qtab4[67] = 0
    qtab4[68] = 0
    qtab4[69] = 0
    qtab4[70] = 0
    qtab4[71] = 0
    qtab4[72] = 0
    qtab4[73] = 0
    qtab4[74] = 0
    qtab4[75] = 0
    qtab4[76] = 0
    qtab4[77] = 0
    qtab4[78] = 0
    qtab4[79] = 0
    qtab4[80] = 0
    qtab4[81] = 0
    qtab4[82] = 0
    qtab4[83] = 0
    qtab4[84] = 0
    qtab4[85] = 0
    qtab4[86] = 0
    qtab4[87] = 0
    qtab4[88] = 0
    qtab4[89] = 0
    qtab4[90] = 0
    qtab4[91] = 0
    qtab4[92] = 0
    qtab4[93] = 0
    qtab4[94] = 0
    qtab4[95] = 0
    qtab4[96] = 0
    qtab4[97] = 0
    qtab4[98] = 0
    qtab4[99] = 0
    qtab4[100] = 0
    qtab4[101] = 0
    qtab4[102] = 0
    qtab4[103] = 0
    qtab4[104] = 0
    qtab4[105] = 0
    qtab4[106] = 0
    qtab4[107] = 0
    qtab4[108] = 0
    qtab4[109] = 0
    qtab4[110] = 0
    qtab4[111] = 0
    qtab4[112] = 0
    qtab4[113] = 0
    qtab4[114] = 0
    qtab4[115] = 0
    qtab4[116] = 0
    qtab4[117] = 0
    qtab4[118] = 0
    qtab4[119] = 0
    qtab4[120] = 0
    qtab4[121] = 0
    qtab4[122] = 0
    qtab4[123] = 0
    qtab4[124] = 0
    qtab4[125] = 0
    qtab4[126] = 0
    qtab4[127] = 0
    qtab4[128] = 0
    qtab4[129] = 0
    qtab4[130] = 0
    qtab4[131] = 0
    qtab4[132] = 0
    qtab4[133] = 0
    qtab4[134] = 0
    qtab4[135] = 0
    qtab4[136] = 0
    qtab4[137] = 0
    qtab4[138] = 0
    qtab4[139] = 0
    qtab4[140] = 0
    qtab4[141] = 0
    qtab4[142] = 0
    qtab4[143] = 0
    qtab4[144] = 0
    qtab4[145] = 0
    qtab4[146] = 0
    qtab4[147] = 0
    qtab4[148] = 0
    qtab4[149] = 0
    qtab4[150] = 0
    qtab4[151] = 0
    qtab4[152] = 0
    qtab4[153] = 0
    qtab4[154] = 0
    qtab4[155] = 0
    qtab4[156] = 0
    qtab4[157] = 0
    qtab4[158] = 0
    qtab4[159] = 0
    qtab4[160] = 0
    qtab4[161] = 0
    qtab4[162] = 0
    qtab4[163] = 0
    qtab4[164] = 0
    qtab4[165] = 0
    qtab4[166] = 0
    qtab4[167] = 0
    qtab4[168] = 0
    qtab4[169] = 0
    qtab4[170] = 0
    qtab4[171] = 0
    qtab4[172] = 0
    qtab4[173] = 0
    qtab4[174] = 0
    qtab4[175] = 0
    qtab4[176] = 0
    qtab4[177] = 0
    qtab4[178] = 0
    qtab4[179] = 0
    qtab4[180] = 0
    qtab4[181] = 0
    qtab4[182] = 0
    qtab4[183] = 0
    qtab4[184] = 0
    qtab4[185] = 0
    qtab4[186] = 0
    qtab4[187] = 0
    qtab4[188] = 0
    qtab4[189] = 0
    qtab4[190] = 0
    qtab4[191] = 0
    qtab4[192] = 0
    qtab4[193] = 0
    qtab4[194] = 0
    qtab4[195] = 0
    qtab4[196] = 0
    qtab4[197] = 0
    qtab4[198] = 0
    qtab4[199] = 0
    qtab4[200] = 0
    qtab4[201] = 0
    qtab4[202] = 0
    qtab4[203] = 0
    qtab4[204] = 0
    qtab4[205] = 0
    qtab4[206] = 0
    qtab4[207] = 0
    qtab4[208] = 0
    qtab4[209] = 0
    qtab4[210] = 0
    qtab4[211] = 0
    qtab4[212] = 0
    qtab4[213] = 0
    qtab4[214] = 0
    qtab4[215] = 0
    qtab4[216] = 0
    qtab4[217] = 0
    qtab4[218] = 0
    qtab4[219] = 0
    qtab4[220] = 0
    qtab4[221] = 0
    qtab4[222] = 0
    qtab4[223] = 0
    qtab4[224] = 0
    qtab4[225] = 0
    qtab4[226] = 0
    qtab4[227] = 0
    qtab4[228] = 0
    qtab4[229] = 0
    qtab4[230] = 0
    qtab4[231] = 0
    qtab4[232] = 0
    qtab4[233] = 0
    qtab4[234] = 0
    qtab4[235] = 0
    qtab4[236] = 0
    qtab4[237] = 0
    qtab4[238] = 0
    qtab4[239] = 0
    qtab4[240] = 1
    qtab4[241] = 1
    qtab4[242] = 1
    qtab4[243] = 1
    qtab4[244] = 1
    qtab4[245] = 1
    qtab4[246] = 1
    qtab4[247] = 1
    qtab4[248] = 1
    qtab4[249] = 1
    qtab4[250] = 1
    qtab4[251] = 1
    qtab4[252] = 1
    qtab4[253] = 1
    qtab4[254] = 1
    qtab4[255] = 1
    qtab4[256] = 1
    qtab4[257] = 1
    qtab4[258] = 1
    qtab4[259] = 1
    qtab4[260] = 1
    qtab4[261] = 1
    qtab4[262] = 1
    qtab4[263] = 1
    qtab4[264] = 1
    qtab4[265] = 1
    qtab4[266] = 1
    qtab4[267] = 1
    qtab4[268] = 1
    qtab4[269] = 1
    qtab4[270] = 1
    qtab4[271] = 1
    qtab4[272] = 1
    qtab4[273] = 1
    qtab4[274] = 1
    qtab4[275] = 1
    qtab4[276] = 1
    qtab4[277] = 1
    qtab4[278] = 1
    qtab4[279] = 1
    qtab4[280] = 1
    qtab4[281] = 1
    qtab4[282] = 1
    qtab4[283] = 1
    qtab4[284] = 1
    qtab4[285] = 1
    qtab4[286] = 1
    qtab4[287] = 1
    qtab4[288] = 1
    qtab4[289] = 1
    qtab4[290] = 1
    qtab4[291] = 1
    qtab4[292] = 1
    qtab4[293] = 1
    qtab4[294] = 1
    qtab4[295] = 1
    qtab4[296] = 1
    qtab4[297] = 1
    qtab4[298] = 1
    qtab4[299] = 1
    qtab4[300] = 1
    qtab4[301] = 1
    qtab4[302] = 1
    qtab4[303] = 1
    qtab4[304] = 1
    qtab4[305] = 1
    qtab4[306] = 1
    qtab4[307] = 1
    qtab4[308] = 1
    qtab4[309] = 1
    qtab4[310] = 1
    qtab4[311] = 1
    qtab4[312] = 1
    qtab4[313] = 1
    qtab4[314] = 1
    qtab4[315] = 1
    qtab4[316] = 1
    qtab4[317] = 1
    qtab4[318] = 1
    qtab4[319] = 1
    qtab4[320] = 1
    qtab4[321] = 1
    qtab4[322] = 1
    qtab4[323] = 1
    qtab4[324] = 1
    qtab4[325] = 1
    qtab4[326] = 1
    qtab4[327] = 1
    qtab4[328] = 1
    qtab4[329] = 1
    qtab4[330] = 1
    qtab4[331] = 1
    qtab4[332] = 1
    qtab4[333] = 1
    qtab4[334] = 1
    qtab4[335] = 1
    qtab4[336] = 1
    qtab4[337] = 1
    qtab4[338] = 1
    qtab4[339] = 1
    qtab4[340] = 1
    qtab4[341] = 1
    qtab4[342] = 1
    qtab4[343] = 1
    qtab4[344] = 1
    qtab4[345] = 1
    qtab4[346] = 1
    qtab4[347] = 1
    qtab4[348] = 1
    qtab4[349] = 1
    qtab4[350] = 1
    qtab4[351] = 1
    qtab4[352] = 1
    qtab4[353] = 1
    qtab4[354] = 1
    qtab4[355] = 1
    qtab4[356] = 1
    qtab4[357] = 1
    qtab4[358] = 1
    qtab4[359] = 1
    qtab4[360] = 1
    qtab4[361] = 1
    qtab4[362] = 1
    qtab4[363] = 1
    qtab4[364] = 1
    qtab4[365] = 1
    qtab4[366] = 1
    qtab4[367] = 1
    qtab4[368] = 1
    qtab4[369] = 1
    qtab4[370] = 1
    qtab4[371] = 1
    qtab4[372] = 1
    qtab4[373] = 1
    qtab4[374] = 1
    qtab4[375] = 1
    qtab4[376] = 1
    qtab4[377] = 1
    qtab4[378] = 1
    qtab4[379] = 1
    qtab4[380] = 1
    qtab4[381] = 1
    qtab4[382] = 1
    qtab4[383] = 1
    qtab4[384] = 1
    qtab4[385] = 1
    qtab4[386] = 1
    qtab4[387] = 1
    qtab4[388] = 1
    qtab4[389] = 1
    qtab4[390] = 1
    qtab4[391] = 1
    qtab4[392] = 1
    qtab4[393] = 1
    qtab4[394] = 1
    qtab4[395] = 1
    qtab4[396] = 1
    qtab4[397] = 1
    qtab4[398] = 1
    qtab4[399] = 1
    qtab4[400] = 1
    qtab4[401] = 1
    qtab4[402] = 1
    qtab4[403] = 1
    qtab4[404] = 1
    qtab4[405] = 1
    qtab4[406] = 1
    qtab4[407] = 1
    qtab4[408] = 1
    qtab4[409] = 1
    qtab4[410] = 1
    qtab4[411] = 1
    qtab4[412] = 1
    qtab4[413] = 1
    qtab4[414] = 1
    qtab4[415] = 1
    qtab4[416] = 1
    qtab4[417] = 1
    qtab4[418] = 1
    qtab4[419] = 1
    qtab4[420] = 1
    qtab4[421] = 1
    qtab4[422] = 1
    qtab4[423] = 1
    qtab4[424] = 1
    qtab4[425] = 1
    qtab4[426] = 1
    qtab4[427] = 1
    qtab4[428] = 1
    qtab4[429] = 1
    qtab4[430] = 1
    qtab4[431] = 1
    qtab4[432] = 1
    qtab4[433] = 1
    qtab4[434] = 1
    qtab4[435] = 1
    qtab4[436] = 1
    qtab4[437] = 1
    qtab4[438] = 1
    qtab4[439] = 1
    qtab4[440] = 1
    qtab4[441] = 1
    qtab4[442] = 1
    qtab4[443] = 1
    qtab4[444] = 1
    qtab4[445] = 1
    qtab4[446] = 1
    qtab4[447] = 1
    qtab4[448] = 1
    qtab4[449] = 1
    qtab4[450] = 1
    qtab4[451] = 1
    qtab4[452] = 1
    qtab4[453] = 1
    qtab4[454] = 1
    qtab4[455] = 1
    qtab4[456] = 1
    qtab4[457] = 1
    qtab4[458] = 1
    qtab4[459] = 1
    qtab4[460] = 1
    qtab4[461] = 1
    qtab4[462] = 1
    qtab4[463] = 1
    qtab4[464] = 1
    qtab4[465] = 1
    qtab4[466] = 1
    qtab4[467] = 1
    qtab4[468] = 1
    qtab4[469] = 1
    qtab4[470] = 1
    qtab4[471] = 1
    qtab4[472] = 1
    qtab4[473] = 1
    qtab4[474] = 1
    qtab4[475] = 1
    qtab4[476] = 1
    qtab4[477] = 1
    qtab4[478] = 1
    qtab4[479] = 1

    var looptw1_real = cxl_alloc[Float32](1)
    var looptw1_imag = cxl_alloc[Float32](1)

    var looptw3_real = cxl_alloc[Float32](1)
    var looptw3_imag = cxl_alloc[Float32](1)

    var pool0_elems = 122880
    var pool0 = cxl_alloc[Float32](pool0_elems)
    var pool1_elems = 32768
    var pool1_raw = cxl_alloc[Float32](pool1_elems + 64)
    var pool1_raw_addr = Int(pool1_raw)
    var pool1_addr = (pool1_raw_addr + 255) // 256 * 256
    if pool1_addr % 256 != 0:
        print("[host] pool alignment assertion failed")
        return
    var pool1 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=pool1_addr)
    var pool2_elems = 122880
    var pool2 = cxl_alloc[Float32](pool2_elems)
    var pool3_elems = 30720
    var pool3_raw = cxl_alloc[Float32](pool3_elems + 64)
    var pool3_raw_addr = Int(pool3_raw)
    var pool3_addr = (pool3_raw_addr + 255) // 256 * 256
    if pool3_addr % 256 != 0:
        print("[host] pool alignment assertion failed")
        return
    var pool3 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=pool3_addr)
    var pool4_elems = 122880
    var pool4 = cxl_alloc[Float32](pool4_elems)

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

    var rc0 = FFTRecPre0.launch(
        PooledRange.over(pool0, 3840),
        FFTRecPre0Params(input_real, input_imag, work0_real, work0_imag, qtab0),
    )
    if rc0 != 0:
        print("[host] recursive FFT stage 0 (FFTRecPre0) failed, exit", rc0)
        return

    var rc1 = FFTRecNear0.launch(
        PooledRange.over(pool1, 1024),
        FFTRecNear0Params(work0_real, work0_imag, work1_real, work1_imag, looptw1_real, looptw1_imag),
    )
    if rc1 != 0:
        print("[host] recursive FFT stage 1 (FFTRecNear0) failed, exit", rc1)
        return

    var rc2 = FFTRecMid0.launch(
        PooledRange.over(pool2, 3840),
        FFTRecMid0Params(work1_real, work1_imag, work0_real, work0_imag, twiddle2_real, twiddle2_imag, qtab2, trtab2),
    )
    if rc2 != 0:
        print("[host] recursive FFT stage 2 (FFTRecMid0) failed, exit", rc2)
        return

    var rc3 = FFTRecLeaf1.launch(
        PooledRange.over(pool3, 960),
        FFTRecLeaf1Params(work0_real, work0_imag, work1_real, work1_imag, looptw3_real, looptw3_imag),
    )
    if rc3 != 0:
        print("[host] recursive FFT stage 3 (FFTRecLeaf1) failed, exit", rc3)
        return

    var rc4 = FFTRecPost0.launch(
        PooledRange.over(pool4, 3840),
        FFTRecPost0Params(work1_real, work1_imag, output_real, output_imag, qtab4),
    )
    if rc4 != 0:
        print("[host] recursive FFT stage 4 (FFTRecPost0) failed, exit", rc4)
        return

    var pi = Float64(3.141592653589793)
    var sign = Float64(-1.0)
    for batch in range(2):
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
    for i in range(1920):
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
