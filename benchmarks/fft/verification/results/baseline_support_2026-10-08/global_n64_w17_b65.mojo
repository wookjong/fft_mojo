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

comptime N = 64

# FFT_EXECUTION {"arithmetic_lanes": [4, 2, 4], "compute_lanes": 4, "copy_mode": "scalar", "dram_bytes": 199680, "host_launches": 1, "internal_rounds": 3, "kernel": "PersistentFFT", "launch": null, "loop_stages": false, "narrow_middle_stages": true, "persistent_mode": "physical", "phase_launches": 12, "registered_functions": 4, "scratchpad_bytes": 32, "storage_mode": "global", "tracker_bytes": 32, "vector_width": 1, "workspace_bytes": 32768}
@fieldwise_init
struct PersistentFFTParams(Movable):
    var input_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var input_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_real_base: UnsafePointer[Float32, MutAnyOrigin]
    var output_imag_base: UnsafePointer[Float32, MutAnyOrigin]
    var buf_a: UnsafePointer[Float32, MutAnyOrigin]
    var buf_b: UnsafePointer[Float32, MutAnyOrigin]


struct PersistentFFT(NDPTask):
    comptime Params = PersistentFFTParams

    comptime round_tracker = scratchpad[8, Float32, name="persistentfft_round_tracker"]()

    @staticmethod
    def stage_0():
        ref p = PersistentFFT.params[]
        comptime RADIX = 4
        var gid = global_uthread_id()
        var software_group_id = gid // 8
        var worker_id = gid % 8
        var round_index = Int(PersistentFFT.round_tracker.load[DType.float32, 1](worker_id)[0])
        var round_base = round_index * 32
        var active_groups = 32
        var remaining_blocks = 65 - round_base
        if remaining_blocks < active_groups:
            active_groups = remaining_blocks
        if software_group_id >= active_groups:
            return
        var logical_block = round_base + software_group_id
        var in_batch_base = logical_block * 64
        var out_batch_base = in_batch_base
        var spad_base = software_group_id * 128

        if worker_id == 0:
            if True:  # batch 0 scope
                # ===== stage 0, SIMD batch 0 (valid lanes: 4/4) =====
                var rr0 = p.input_real_base.load[width=4](in_batch_base + 0)
                var ii0 = p.input_imag_base.load[width=4](in_batch_base + 0)
                var rr1 = p.input_real_base.load[width=4](in_batch_base + 16)
                var ii1 = p.input_imag_base.load[width=4](in_batch_base + 16)
                var rr2 = p.input_real_base.load[width=4](in_batch_base + 32)
                var ii2 = p.input_imag_base.load[width=4](in_batch_base + 32)
                var rr3 = p.input_real_base.load[width=4](in_batch_base + 48)
                var ii3 = p.input_imag_base.load[width=4](in_batch_base + 48)

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
                p.buf_b.store(spad_base + 0, or0[0])
                p.buf_b.store(spad_base + 64 + 0, oi0[0])
                p.buf_b.store(spad_base + 4, or0[1])
                p.buf_b.store(spad_base + 64 + 4, oi0[1])
                p.buf_b.store(spad_base + 8, or0[2])
                p.buf_b.store(spad_base + 64 + 8, oi0[2])
                p.buf_b.store(spad_base + 12, or0[3])
                p.buf_b.store(spad_base + 64 + 12, oi0[3])

                var twr1 = SIMD[DType.float32, 4](Float32(1), Float32(0.995184727), Float32(0.98078528), Float32(0.956940336))
                var twi1 = SIMD[DType.float32, 4](Float32(0), Float32(-0.0980171403), Float32(-0.195090322), Float32(-0.290284677))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                p.buf_b.store(spad_base + 1, or1[0])
                p.buf_b.store(spad_base + 64 + 1, oi1[0])
                p.buf_b.store(spad_base + 5, or1[1])
                p.buf_b.store(spad_base + 64 + 5, oi1[1])
                p.buf_b.store(spad_base + 9, or1[2])
                p.buf_b.store(spad_base + 64 + 9, oi1[2])
                p.buf_b.store(spad_base + 13, or1[3])
                p.buf_b.store(spad_base + 64 + 13, oi1[3])

                var twr2 = SIMD[DType.float32, 4](Float32(1), Float32(0.98078528), Float32(0.923879533), Float32(0.831469612))
                var twi2 = SIMD[DType.float32, 4](Float32(0), Float32(-0.195090322), Float32(-0.382683432), Float32(-0.555570233))
                var tr2 = or2 * twr2 - oi2 * twi2
                var ti2 = or2 * twi2 + oi2 * twr2
                or2 = tr2
                oi2 = ti2
                p.buf_b.store(spad_base + 2, or2[0])
                p.buf_b.store(spad_base + 64 + 2, oi2[0])
                p.buf_b.store(spad_base + 6, or2[1])
                p.buf_b.store(spad_base + 64 + 6, oi2[1])
                p.buf_b.store(spad_base + 10, or2[2])
                p.buf_b.store(spad_base + 64 + 10, oi2[2])
                p.buf_b.store(spad_base + 14, or2[3])
                p.buf_b.store(spad_base + 64 + 14, oi2[3])

                var twr3 = SIMD[DType.float32, 4](Float32(1), Float32(0.956940336), Float32(0.831469612), Float32(0.634393284))
                var twi3 = SIMD[DType.float32, 4](Float32(0), Float32(-0.290284677), Float32(-0.555570233), Float32(-0.773010453))
                var tr3 = or3 * twr3 - oi3 * twi3
                var ti3 = or3 * twi3 + oi3 * twr3
                or3 = tr3
                oi3 = ti3
                p.buf_b.store(spad_base + 3, or3[0])
                p.buf_b.store(spad_base + 64 + 3, oi3[0])
                p.buf_b.store(spad_base + 7, or3[1])
                p.buf_b.store(spad_base + 64 + 7, oi3[1])
                p.buf_b.store(spad_base + 11, or3[2])
                p.buf_b.store(spad_base + 64 + 11, oi3[2])
                p.buf_b.store(spad_base + 15, or3[3])
                p.buf_b.store(spad_base + 64 + 15, oi3[3])


            if True:  # batch 0 scope
                # ===== stage 0, SIMD batch 0 (valid lanes: 4/4) =====
                var rr0 = p.input_real_base.load[width=4](in_batch_base + 4)
                var ii0 = p.input_imag_base.load[width=4](in_batch_base + 4)
                var rr1 = p.input_real_base.load[width=4](in_batch_base + 20)
                var ii1 = p.input_imag_base.load[width=4](in_batch_base + 20)
                var rr2 = p.input_real_base.load[width=4](in_batch_base + 36)
                var ii2 = p.input_imag_base.load[width=4](in_batch_base + 36)
                var rr3 = p.input_real_base.load[width=4](in_batch_base + 52)
                var ii3 = p.input_imag_base.load[width=4](in_batch_base + 52)

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
                p.buf_b.store(spad_base + 16, or0[0])
                p.buf_b.store(spad_base + 64 + 16, oi0[0])
                p.buf_b.store(spad_base + 20, or0[1])
                p.buf_b.store(spad_base + 64 + 20, oi0[1])
                p.buf_b.store(spad_base + 24, or0[2])
                p.buf_b.store(spad_base + 64 + 24, oi0[2])
                p.buf_b.store(spad_base + 28, or0[3])
                p.buf_b.store(spad_base + 64 + 28, oi0[3])

                var twr1 = SIMD[DType.float32, 4](Float32(0.923879533), Float32(0.881921264), Float32(0.831469612), Float32(0.773010453))
                var twi1 = SIMD[DType.float32, 4](Float32(-0.382683432), Float32(-0.471396737), Float32(-0.555570233), Float32(-0.634393284))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                p.buf_b.store(spad_base + 17, or1[0])
                p.buf_b.store(spad_base + 64 + 17, oi1[0])
                p.buf_b.store(spad_base + 21, or1[1])
                p.buf_b.store(spad_base + 64 + 21, oi1[1])
                p.buf_b.store(spad_base + 25, or1[2])
                p.buf_b.store(spad_base + 64 + 25, oi1[2])
                p.buf_b.store(spad_base + 29, or1[3])
                p.buf_b.store(spad_base + 64 + 29, oi1[3])

                var twr2 = SIMD[DType.float32, 4](Float32(0.707106781), Float32(0.555570233), Float32(0.382683432), Float32(0.195090322))
                var twi2 = SIMD[DType.float32, 4](Float32(-0.707106781), Float32(-0.831469612), Float32(-0.923879533), Float32(-0.98078528))
                var tr2 = or2 * twr2 - oi2 * twi2
                var ti2 = or2 * twi2 + oi2 * twr2
                or2 = tr2
                oi2 = ti2
                p.buf_b.store(spad_base + 18, or2[0])
                p.buf_b.store(spad_base + 64 + 18, oi2[0])
                p.buf_b.store(spad_base + 22, or2[1])
                p.buf_b.store(spad_base + 64 + 22, oi2[1])
                p.buf_b.store(spad_base + 26, or2[2])
                p.buf_b.store(spad_base + 64 + 26, oi2[2])
                p.buf_b.store(spad_base + 30, or2[3])
                p.buf_b.store(spad_base + 64 + 30, oi2[3])

                var twr3 = SIMD[DType.float32, 4](Float32(0.382683432), Float32(0.0980171403), Float32(-0.195090322), Float32(-0.471396737))
                var twi3 = SIMD[DType.float32, 4](Float32(-0.923879533), Float32(-0.995184727), Float32(-0.98078528), Float32(-0.881921264))
                var tr3 = or3 * twr3 - oi3 * twi3
                var ti3 = or3 * twi3 + oi3 * twr3
                or3 = tr3
                oi3 = ti3
                p.buf_b.store(spad_base + 19, or3[0])
                p.buf_b.store(spad_base + 64 + 19, oi3[0])
                p.buf_b.store(spad_base + 23, or3[1])
                p.buf_b.store(spad_base + 64 + 23, oi3[1])
                p.buf_b.store(spad_base + 27, or3[2])
                p.buf_b.store(spad_base + 64 + 27, oi3[2])
                p.buf_b.store(spad_base + 31, or3[3])
                p.buf_b.store(spad_base + 64 + 31, oi3[3])


        elif worker_id == 1:
            if True:  # batch 1 scope
                # ===== stage 0, SIMD batch 1 (valid lanes: 4/4) =====
                var rr0 = p.input_real_base.load[width=4](in_batch_base + 8)
                var ii0 = p.input_imag_base.load[width=4](in_batch_base + 8)
                var rr1 = p.input_real_base.load[width=4](in_batch_base + 24)
                var ii1 = p.input_imag_base.load[width=4](in_batch_base + 24)
                var rr2 = p.input_real_base.load[width=4](in_batch_base + 40)
                var ii2 = p.input_imag_base.load[width=4](in_batch_base + 40)
                var rr3 = p.input_real_base.load[width=4](in_batch_base + 56)
                var ii3 = p.input_imag_base.load[width=4](in_batch_base + 56)

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
                p.buf_b.store(spad_base + 32, or0[0])
                p.buf_b.store(spad_base + 64 + 32, oi0[0])
                p.buf_b.store(spad_base + 36, or0[1])
                p.buf_b.store(spad_base + 64 + 36, oi0[1])
                p.buf_b.store(spad_base + 40, or0[2])
                p.buf_b.store(spad_base + 64 + 40, oi0[2])
                p.buf_b.store(spad_base + 44, or0[3])
                p.buf_b.store(spad_base + 64 + 44, oi0[3])

                var twr1 = SIMD[DType.float32, 4](Float32(0.707106781), Float32(0.634393284), Float32(0.555570233), Float32(0.471396737))
                var twi1 = SIMD[DType.float32, 4](Float32(-0.707106781), Float32(-0.773010453), Float32(-0.831469612), Float32(-0.881921264))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                p.buf_b.store(spad_base + 33, or1[0])
                p.buf_b.store(spad_base + 64 + 33, oi1[0])
                p.buf_b.store(spad_base + 37, or1[1])
                p.buf_b.store(spad_base + 64 + 37, oi1[1])
                p.buf_b.store(spad_base + 41, or1[2])
                p.buf_b.store(spad_base + 64 + 41, oi1[2])
                p.buf_b.store(spad_base + 45, or1[3])
                p.buf_b.store(spad_base + 64 + 45, oi1[3])

                var twr2 = SIMD[DType.float32, 4](Float32(0), Float32(-0.195090322), Float32(-0.382683432), Float32(-0.555570233))
                var twi2 = SIMD[DType.float32, 4](Float32(-1), Float32(-0.98078528), Float32(-0.923879533), Float32(-0.831469612))
                var tr2 = or2 * twr2 - oi2 * twi2
                var ti2 = or2 * twi2 + oi2 * twr2
                or2 = tr2
                oi2 = ti2
                p.buf_b.store(spad_base + 34, or2[0])
                p.buf_b.store(spad_base + 64 + 34, oi2[0])
                p.buf_b.store(spad_base + 38, or2[1])
                p.buf_b.store(spad_base + 64 + 38, oi2[1])
                p.buf_b.store(spad_base + 42, or2[2])
                p.buf_b.store(spad_base + 64 + 42, oi2[2])
                p.buf_b.store(spad_base + 46, or2[3])
                p.buf_b.store(spad_base + 64 + 46, oi2[3])

                var twr3 = SIMD[DType.float32, 4](Float32(-0.707106781), Float32(-0.881921264), Float32(-0.98078528), Float32(-0.995184727))
                var twi3 = SIMD[DType.float32, 4](Float32(-0.707106781), Float32(-0.471396737), Float32(-0.195090322), Float32(0.0980171403))
                var tr3 = or3 * twr3 - oi3 * twi3
                var ti3 = or3 * twi3 + oi3 * twr3
                or3 = tr3
                oi3 = ti3
                p.buf_b.store(spad_base + 35, or3[0])
                p.buf_b.store(spad_base + 64 + 35, oi3[0])
                p.buf_b.store(spad_base + 39, or3[1])
                p.buf_b.store(spad_base + 64 + 39, oi3[1])
                p.buf_b.store(spad_base + 43, or3[2])
                p.buf_b.store(spad_base + 64 + 43, oi3[2])
                p.buf_b.store(spad_base + 47, or3[3])
                p.buf_b.store(spad_base + 64 + 47, oi3[3])


            if True:  # batch 1 scope
                # ===== stage 0, SIMD batch 1 (valid lanes: 4/4) =====
                var rr0 = p.input_real_base.load[width=4](in_batch_base + 12)
                var ii0 = p.input_imag_base.load[width=4](in_batch_base + 12)
                var rr1 = p.input_real_base.load[width=4](in_batch_base + 28)
                var ii1 = p.input_imag_base.load[width=4](in_batch_base + 28)
                var rr2 = p.input_real_base.load[width=4](in_batch_base + 44)
                var ii2 = p.input_imag_base.load[width=4](in_batch_base + 44)
                var rr3 = p.input_real_base.load[width=4](in_batch_base + 60)
                var ii3 = p.input_imag_base.load[width=4](in_batch_base + 60)

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
                p.buf_b.store(spad_base + 48, or0[0])
                p.buf_b.store(spad_base + 64 + 48, oi0[0])
                p.buf_b.store(spad_base + 52, or0[1])
                p.buf_b.store(spad_base + 64 + 52, oi0[1])
                p.buf_b.store(spad_base + 56, or0[2])
                p.buf_b.store(spad_base + 64 + 56, oi0[2])
                p.buf_b.store(spad_base + 60, or0[3])
                p.buf_b.store(spad_base + 64 + 60, oi0[3])

                var twr1 = SIMD[DType.float32, 4](Float32(0.382683432), Float32(0.290284677), Float32(0.195090322), Float32(0.0980171403))
                var twi1 = SIMD[DType.float32, 4](Float32(-0.923879533), Float32(-0.956940336), Float32(-0.98078528), Float32(-0.995184727))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                p.buf_b.store(spad_base + 49, or1[0])
                p.buf_b.store(spad_base + 64 + 49, oi1[0])
                p.buf_b.store(spad_base + 53, or1[1])
                p.buf_b.store(spad_base + 64 + 53, oi1[1])
                p.buf_b.store(spad_base + 57, or1[2])
                p.buf_b.store(spad_base + 64 + 57, oi1[2])
                p.buf_b.store(spad_base + 61, or1[3])
                p.buf_b.store(spad_base + 64 + 61, oi1[3])

                var twr2 = SIMD[DType.float32, 4](Float32(-0.707106781), Float32(-0.831469612), Float32(-0.923879533), Float32(-0.98078528))
                var twi2 = SIMD[DType.float32, 4](Float32(-0.707106781), Float32(-0.555570233), Float32(-0.382683432), Float32(-0.195090322))
                var tr2 = or2 * twr2 - oi2 * twi2
                var ti2 = or2 * twi2 + oi2 * twr2
                or2 = tr2
                oi2 = ti2
                p.buf_b.store(spad_base + 50, or2[0])
                p.buf_b.store(spad_base + 64 + 50, oi2[0])
                p.buf_b.store(spad_base + 54, or2[1])
                p.buf_b.store(spad_base + 64 + 54, oi2[1])
                p.buf_b.store(spad_base + 58, or2[2])
                p.buf_b.store(spad_base + 64 + 58, oi2[2])
                p.buf_b.store(spad_base + 62, or2[3])
                p.buf_b.store(spad_base + 64 + 62, oi2[3])

                var twr3 = SIMD[DType.float32, 4](Float32(-0.923879533), Float32(-0.773010453), Float32(-0.555570233), Float32(-0.290284677))
                var twi3 = SIMD[DType.float32, 4](Float32(0.382683432), Float32(0.634393284), Float32(0.831469612), Float32(0.956940336))
                var tr3 = or3 * twr3 - oi3 * twi3
                var ti3 = or3 * twi3 + oi3 * twr3
                or3 = tr3
                oi3 = ti3
                p.buf_b.store(spad_base + 51, or3[0])
                p.buf_b.store(spad_base + 64 + 51, oi3[0])
                p.buf_b.store(spad_base + 55, or3[1])
                p.buf_b.store(spad_base + 64 + 55, oi3[1])
                p.buf_b.store(spad_base + 59, or3[2])
                p.buf_b.store(spad_base + 64 + 59, oi3[2])
                p.buf_b.store(spad_base + 63, or3[3])
                p.buf_b.store(spad_base + 64 + 63, oi3[3])




    @staticmethod
    def stage_1():
        ref p = PersistentFFT.params[]
        comptime RADIX = 4
        var gid = global_uthread_id()
        var software_group_id = gid // 8
        var worker_id = gid % 8
        var round_index = Int(PersistentFFT.round_tracker.load[DType.float32, 1](worker_id)[0])
        var round_base = round_index * 32
        var active_groups = 32
        var remaining_blocks = 65 - round_base
        if remaining_blocks < active_groups:
            active_groups = remaining_blocks
        if software_group_id >= active_groups:
            return
        var logical_block = round_base + software_group_id
        var in_batch_base = logical_block * 64
        var out_batch_base = in_batch_base
        var spad_base = software_group_id * 128

        if worker_id == 0:
            if True:  # batch 0 scope
                # ===== stage 1, SIMD batch 0 (valid lanes: 2/2) =====
                var rr0 = p.buf_b.load[width=2](spad_base + 0)
                var ii0 = p.buf_b.load[width=2](spad_base + 64 + 0)
                var rr1 = p.buf_b.load[width=2](spad_base + 16)
                var ii1 = p.buf_b.load[width=2](spad_base + 64 + 16)
                var rr2 = p.buf_b.load[width=2](spad_base + 32)
                var ii2 = p.buf_b.load[width=2](spad_base + 64 + 32)
                var rr3 = p.buf_b.load[width=2](spad_base + 48)
                var ii3 = p.buf_b.load[width=2](spad_base + 64 + 48)

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
                p.buf_a.store(spad_base + 0, or0)
                p.buf_a.store(spad_base + 64 + 0, oi0)

                p.buf_a.store(spad_base + 4, or1)
                p.buf_a.store(spad_base + 64 + 4, oi1)

                p.buf_a.store(spad_base + 8, or2)
                p.buf_a.store(spad_base + 64 + 8, oi2)

                p.buf_a.store(spad_base + 12, or3)
                p.buf_a.store(spad_base + 64 + 12, oi3)


            if True:  # batch 0 scope
                # ===== stage 1, SIMD batch 0 (valid lanes: 2/2) =====
                var rr0 = p.buf_b.load[width=2](spad_base + 2)
                var ii0 = p.buf_b.load[width=2](spad_base + 64 + 2)
                var rr1 = p.buf_b.load[width=2](spad_base + 18)
                var ii1 = p.buf_b.load[width=2](spad_base + 64 + 18)
                var rr2 = p.buf_b.load[width=2](spad_base + 34)
                var ii2 = p.buf_b.load[width=2](spad_base + 64 + 34)
                var rr3 = p.buf_b.load[width=2](spad_base + 50)
                var ii3 = p.buf_b.load[width=2](spad_base + 64 + 50)

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
                p.buf_a.store(spad_base + 2, or0)
                p.buf_a.store(spad_base + 64 + 2, oi0)

                p.buf_a.store(spad_base + 6, or1)
                p.buf_a.store(spad_base + 64 + 6, oi1)

                p.buf_a.store(spad_base + 10, or2)
                p.buf_a.store(spad_base + 64 + 10, oi2)

                p.buf_a.store(spad_base + 14, or3)
                p.buf_a.store(spad_base + 64 + 14, oi3)


            if True:  # batch 0 scope
                # ===== stage 1, SIMD batch 0 (valid lanes: 2/2) =====
                var rr0 = p.buf_b.load[width=2](spad_base + 4)
                var ii0 = p.buf_b.load[width=2](spad_base + 64 + 4)
                var rr1 = p.buf_b.load[width=2](spad_base + 20)
                var ii1 = p.buf_b.load[width=2](spad_base + 64 + 20)
                var rr2 = p.buf_b.load[width=2](spad_base + 36)
                var ii2 = p.buf_b.load[width=2](spad_base + 64 + 36)
                var rr3 = p.buf_b.load[width=2](spad_base + 52)
                var ii3 = p.buf_b.load[width=2](spad_base + 64 + 52)

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
                p.buf_a.store(spad_base + 16, or0)
                p.buf_a.store(spad_base + 64 + 16, oi0)

                var twr1 = SIMD[DType.float32, 2](Float32(0.923879533), Float32(0.923879533))
                var twi1 = SIMD[DType.float32, 2](Float32(-0.382683432), Float32(-0.382683432))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                p.buf_a.store(spad_base + 20, or1)
                p.buf_a.store(spad_base + 64 + 20, oi1)

                var twr2 = SIMD[DType.float32, 2](Float32(0.707106781), Float32(0.707106781))
                var twi2 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var tr2 = or2 * twr2 - oi2 * twi2
                var ti2 = or2 * twi2 + oi2 * twr2
                or2 = tr2
                oi2 = ti2
                p.buf_a.store(spad_base + 24, or2)
                p.buf_a.store(spad_base + 64 + 24, oi2)

                var twr3 = SIMD[DType.float32, 2](Float32(0.382683432), Float32(0.382683432))
                var twi3 = SIMD[DType.float32, 2](Float32(-0.923879533), Float32(-0.923879533))
                var tr3 = or3 * twr3 - oi3 * twi3
                var ti3 = or3 * twi3 + oi3 * twr3
                or3 = tr3
                oi3 = ti3
                p.buf_a.store(spad_base + 28, or3)
                p.buf_a.store(spad_base + 64 + 28, oi3)


            if True:  # batch 0 scope
                # ===== stage 1, SIMD batch 0 (valid lanes: 2/2) =====
                var rr0 = p.buf_b.load[width=2](spad_base + 6)
                var ii0 = p.buf_b.load[width=2](spad_base + 64 + 6)
                var rr1 = p.buf_b.load[width=2](spad_base + 22)
                var ii1 = p.buf_b.load[width=2](spad_base + 64 + 22)
                var rr2 = p.buf_b.load[width=2](spad_base + 38)
                var ii2 = p.buf_b.load[width=2](spad_base + 64 + 38)
                var rr3 = p.buf_b.load[width=2](spad_base + 54)
                var ii3 = p.buf_b.load[width=2](spad_base + 64 + 54)

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
                p.buf_a.store(spad_base + 18, or0)
                p.buf_a.store(spad_base + 64 + 18, oi0)

                var twr1 = SIMD[DType.float32, 2](Float32(0.923879533), Float32(0.923879533))
                var twi1 = SIMD[DType.float32, 2](Float32(-0.382683432), Float32(-0.382683432))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                p.buf_a.store(spad_base + 22, or1)
                p.buf_a.store(spad_base + 64 + 22, oi1)

                var twr2 = SIMD[DType.float32, 2](Float32(0.707106781), Float32(0.707106781))
                var twi2 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var tr2 = or2 * twr2 - oi2 * twi2
                var ti2 = or2 * twi2 + oi2 * twr2
                or2 = tr2
                oi2 = ti2
                p.buf_a.store(spad_base + 26, or2)
                p.buf_a.store(spad_base + 64 + 26, oi2)

                var twr3 = SIMD[DType.float32, 2](Float32(0.382683432), Float32(0.382683432))
                var twi3 = SIMD[DType.float32, 2](Float32(-0.923879533), Float32(-0.923879533))
                var tr3 = or3 * twr3 - oi3 * twi3
                var ti3 = or3 * twi3 + oi3 * twr3
                or3 = tr3
                oi3 = ti3
                p.buf_a.store(spad_base + 30, or3)
                p.buf_a.store(spad_base + 64 + 30, oi3)


        elif worker_id == 1:
            if True:  # batch 1 scope
                # ===== stage 1, SIMD batch 1 (valid lanes: 2/2) =====
                var rr0 = p.buf_b.load[width=2](spad_base + 8)
                var ii0 = p.buf_b.load[width=2](spad_base + 64 + 8)
                var rr1 = p.buf_b.load[width=2](spad_base + 24)
                var ii1 = p.buf_b.load[width=2](spad_base + 64 + 24)
                var rr2 = p.buf_b.load[width=2](spad_base + 40)
                var ii2 = p.buf_b.load[width=2](spad_base + 64 + 40)
                var rr3 = p.buf_b.load[width=2](spad_base + 56)
                var ii3 = p.buf_b.load[width=2](spad_base + 64 + 56)

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
                p.buf_a.store(spad_base + 32, or0)
                p.buf_a.store(spad_base + 64 + 32, oi0)

                var twr1 = SIMD[DType.float32, 2](Float32(0.707106781), Float32(0.707106781))
                var twi1 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                p.buf_a.store(spad_base + 36, or1)
                p.buf_a.store(spad_base + 64 + 36, oi1)

                var tr2 = oi2
                oi2 = -or2
                or2 = tr2
                p.buf_a.store(spad_base + 40, or2)
                p.buf_a.store(spad_base + 64 + 40, oi2)

                var twr3 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var twi3 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var tr3 = or3 * twr3 - oi3 * twi3
                var ti3 = or3 * twi3 + oi3 * twr3
                or3 = tr3
                oi3 = ti3
                p.buf_a.store(spad_base + 44, or3)
                p.buf_a.store(spad_base + 64 + 44, oi3)


            if True:  # batch 1 scope
                # ===== stage 1, SIMD batch 1 (valid lanes: 2/2) =====
                var rr0 = p.buf_b.load[width=2](spad_base + 10)
                var ii0 = p.buf_b.load[width=2](spad_base + 64 + 10)
                var rr1 = p.buf_b.load[width=2](spad_base + 26)
                var ii1 = p.buf_b.load[width=2](spad_base + 64 + 26)
                var rr2 = p.buf_b.load[width=2](spad_base + 42)
                var ii2 = p.buf_b.load[width=2](spad_base + 64 + 42)
                var rr3 = p.buf_b.load[width=2](spad_base + 58)
                var ii3 = p.buf_b.load[width=2](spad_base + 64 + 58)

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
                p.buf_a.store(spad_base + 34, or0)
                p.buf_a.store(spad_base + 64 + 34, oi0)

                var twr1 = SIMD[DType.float32, 2](Float32(0.707106781), Float32(0.707106781))
                var twi1 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                p.buf_a.store(spad_base + 38, or1)
                p.buf_a.store(spad_base + 64 + 38, oi1)

                var tr2 = oi2
                oi2 = -or2
                or2 = tr2
                p.buf_a.store(spad_base + 42, or2)
                p.buf_a.store(spad_base + 64 + 42, oi2)

                var twr3 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var twi3 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var tr3 = or3 * twr3 - oi3 * twi3
                var ti3 = or3 * twi3 + oi3 * twr3
                or3 = tr3
                oi3 = ti3
                p.buf_a.store(spad_base + 46, or3)
                p.buf_a.store(spad_base + 64 + 46, oi3)


            if True:  # batch 1 scope
                # ===== stage 1, SIMD batch 1 (valid lanes: 2/2) =====
                var rr0 = p.buf_b.load[width=2](spad_base + 12)
                var ii0 = p.buf_b.load[width=2](spad_base + 64 + 12)
                var rr1 = p.buf_b.load[width=2](spad_base + 28)
                var ii1 = p.buf_b.load[width=2](spad_base + 64 + 28)
                var rr2 = p.buf_b.load[width=2](spad_base + 44)
                var ii2 = p.buf_b.load[width=2](spad_base + 64 + 44)
                var rr3 = p.buf_b.load[width=2](spad_base + 60)
                var ii3 = p.buf_b.load[width=2](spad_base + 64 + 60)

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
                p.buf_a.store(spad_base + 48, or0)
                p.buf_a.store(spad_base + 64 + 48, oi0)

                var twr1 = SIMD[DType.float32, 2](Float32(0.382683432), Float32(0.382683432))
                var twi1 = SIMD[DType.float32, 2](Float32(-0.923879533), Float32(-0.923879533))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                p.buf_a.store(spad_base + 52, or1)
                p.buf_a.store(spad_base + 64 + 52, oi1)

                var twr2 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var twi2 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var tr2 = or2 * twr2 - oi2 * twi2
                var ti2 = or2 * twi2 + oi2 * twr2
                or2 = tr2
                oi2 = ti2
                p.buf_a.store(spad_base + 56, or2)
                p.buf_a.store(spad_base + 64 + 56, oi2)

                var twr3 = SIMD[DType.float32, 2](Float32(-0.923879533), Float32(-0.923879533))
                var twi3 = SIMD[DType.float32, 2](Float32(0.382683432), Float32(0.382683432))
                var tr3 = or3 * twr3 - oi3 * twi3
                var ti3 = or3 * twi3 + oi3 * twr3
                or3 = tr3
                oi3 = ti3
                p.buf_a.store(spad_base + 60, or3)
                p.buf_a.store(spad_base + 64 + 60, oi3)


            if True:  # batch 1 scope
                # ===== stage 1, SIMD batch 1 (valid lanes: 2/2) =====
                var rr0 = p.buf_b.load[width=2](spad_base + 14)
                var ii0 = p.buf_b.load[width=2](spad_base + 64 + 14)
                var rr1 = p.buf_b.load[width=2](spad_base + 30)
                var ii1 = p.buf_b.load[width=2](spad_base + 64 + 30)
                var rr2 = p.buf_b.load[width=2](spad_base + 46)
                var ii2 = p.buf_b.load[width=2](spad_base + 64 + 46)
                var rr3 = p.buf_b.load[width=2](spad_base + 62)
                var ii3 = p.buf_b.load[width=2](spad_base + 64 + 62)

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
                p.buf_a.store(spad_base + 50, or0)
                p.buf_a.store(spad_base + 64 + 50, oi0)

                var twr1 = SIMD[DType.float32, 2](Float32(0.382683432), Float32(0.382683432))
                var twi1 = SIMD[DType.float32, 2](Float32(-0.923879533), Float32(-0.923879533))
                var tr1 = or1 * twr1 - oi1 * twi1
                var ti1 = or1 * twi1 + oi1 * twr1
                or1 = tr1
                oi1 = ti1
                p.buf_a.store(spad_base + 54, or1)
                p.buf_a.store(spad_base + 64 + 54, oi1)

                var twr2 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var twi2 = SIMD[DType.float32, 2](Float32(-0.707106781), Float32(-0.707106781))
                var tr2 = or2 * twr2 - oi2 * twi2
                var ti2 = or2 * twi2 + oi2 * twr2
                or2 = tr2
                oi2 = ti2
                p.buf_a.store(spad_base + 58, or2)
                p.buf_a.store(spad_base + 64 + 58, oi2)

                var twr3 = SIMD[DType.float32, 2](Float32(-0.923879533), Float32(-0.923879533))
                var twi3 = SIMD[DType.float32, 2](Float32(0.382683432), Float32(0.382683432))
                var tr3 = or3 * twr3 - oi3 * twi3
                var ti3 = or3 * twi3 + oi3 * twr3
                or3 = tr3
                oi3 = ti3
                p.buf_a.store(spad_base + 62, or3)
                p.buf_a.store(spad_base + 64 + 62, oi3)




    @staticmethod
    def stage_2():
        ref p = PersistentFFT.params[]
        comptime RADIX = 4
        var gid = global_uthread_id()
        var software_group_id = gid // 8
        var worker_id = gid % 8
        var round_index = Int(PersistentFFT.round_tracker.load[DType.float32, 1](worker_id)[0])
        var round_base = round_index * 32
        var active_groups = 32
        var remaining_blocks = 65 - round_base
        if remaining_blocks < active_groups:
            active_groups = remaining_blocks
        if software_group_id >= active_groups:
            return
        var logical_block = round_base + software_group_id
        var in_batch_base = logical_block * 64
        var out_batch_base = in_batch_base
        var spad_base = software_group_id * 128

        if worker_id == 0:
            if True:  # batch 0 scope
                # ===== stage 2, SIMD batch 0 (valid lanes: 4/4) =====
                var rr0 = p.buf_a.load[width=4](spad_base + 0)
                var ii0 = p.buf_a.load[width=4](spad_base + 64 + 0)
                var rr1 = p.buf_a.load[width=4](spad_base + 16)
                var ii1 = p.buf_a.load[width=4](spad_base + 64 + 16)
                var rr2 = p.buf_a.load[width=4](spad_base + 32)
                var ii2 = p.buf_a.load[width=4](spad_base + 64 + 32)
                var rr3 = p.buf_a.load[width=4](spad_base + 48)
                var ii3 = p.buf_a.load[width=4](spad_base + 64 + 48)

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
                p.output_real_base.store(out_batch_base + 0, or0)
                p.output_imag_base.store(out_batch_base + 0, oi0)

                p.output_real_base.store(out_batch_base + 16, or1)
                p.output_imag_base.store(out_batch_base + 16, oi1)

                p.output_real_base.store(out_batch_base + 32, or2)
                p.output_imag_base.store(out_batch_base + 32, oi2)

                p.output_real_base.store(out_batch_base + 48, or3)
                p.output_imag_base.store(out_batch_base + 48, oi3)


            if True:  # batch 0 scope
                # ===== stage 2, SIMD batch 0 (valid lanes: 4/4) =====
                var rr0 = p.buf_a.load[width=4](spad_base + 4)
                var ii0 = p.buf_a.load[width=4](spad_base + 64 + 4)
                var rr1 = p.buf_a.load[width=4](spad_base + 20)
                var ii1 = p.buf_a.load[width=4](spad_base + 64 + 20)
                var rr2 = p.buf_a.load[width=4](spad_base + 36)
                var ii2 = p.buf_a.load[width=4](spad_base + 64 + 36)
                var rr3 = p.buf_a.load[width=4](spad_base + 52)
                var ii3 = p.buf_a.load[width=4](spad_base + 64 + 52)

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
                p.output_real_base.store(out_batch_base + 4, or0)
                p.output_imag_base.store(out_batch_base + 4, oi0)

                p.output_real_base.store(out_batch_base + 20, or1)
                p.output_imag_base.store(out_batch_base + 20, oi1)

                p.output_real_base.store(out_batch_base + 36, or2)
                p.output_imag_base.store(out_batch_base + 36, oi2)

                p.output_real_base.store(out_batch_base + 52, or3)
                p.output_imag_base.store(out_batch_base + 52, oi3)


        elif worker_id == 1:
            if True:  # batch 1 scope
                # ===== stage 2, SIMD batch 1 (valid lanes: 4/4) =====
                var rr0 = p.buf_a.load[width=4](spad_base + 8)
                var ii0 = p.buf_a.load[width=4](spad_base + 64 + 8)
                var rr1 = p.buf_a.load[width=4](spad_base + 24)
                var ii1 = p.buf_a.load[width=4](spad_base + 64 + 24)
                var rr2 = p.buf_a.load[width=4](spad_base + 40)
                var ii2 = p.buf_a.load[width=4](spad_base + 64 + 40)
                var rr3 = p.buf_a.load[width=4](spad_base + 56)
                var ii3 = p.buf_a.load[width=4](spad_base + 64 + 56)

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
                p.output_real_base.store(out_batch_base + 8, or0)
                p.output_imag_base.store(out_batch_base + 8, oi0)

                p.output_real_base.store(out_batch_base + 24, or1)
                p.output_imag_base.store(out_batch_base + 24, oi1)

                p.output_real_base.store(out_batch_base + 40, or2)
                p.output_imag_base.store(out_batch_base + 40, oi2)

                p.output_real_base.store(out_batch_base + 56, or3)
                p.output_imag_base.store(out_batch_base + 56, oi3)


            if True:  # batch 1 scope
                # ===== stage 2, SIMD batch 1 (valid lanes: 4/4) =====
                var rr0 = p.buf_a.load[width=4](spad_base + 12)
                var ii0 = p.buf_a.load[width=4](spad_base + 64 + 12)
                var rr1 = p.buf_a.load[width=4](spad_base + 28)
                var ii1 = p.buf_a.load[width=4](spad_base + 64 + 28)
                var rr2 = p.buf_a.load[width=4](spad_base + 44)
                var ii2 = p.buf_a.load[width=4](spad_base + 64 + 44)
                var rr3 = p.buf_a.load[width=4](spad_base + 60)
                var ii3 = p.buf_a.load[width=4](spad_base + 64 + 60)

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
                p.output_real_base.store(out_batch_base + 12, or0)
                p.output_imag_base.store(out_batch_base + 12, oi0)

                p.output_real_base.store(out_batch_base + 28, or1)
                p.output_imag_base.store(out_batch_base + 28, oi1)

                p.output_real_base.store(out_batch_base + 44, or2)
                p.output_imag_base.store(out_batch_base + 44, oi2)

                p.output_real_base.store(out_batch_base + 60, or3)
                p.output_imag_base.store(out_batch_base + 60, oi3)




    @staticmethod
    def writeback():
        ref p = PersistentFFT.params[]
        var gid = global_uthread_id()
        var software_group_id = gid // 8
        var worker_id = gid % 8
        var round_index = Int(PersistentFFT.round_tracker.load[DType.float32, 1](worker_id)[0])
        var round_base = round_index * 32
        var active_groups = 32
        var remaining_blocks = 65 - round_base
        if remaining_blocks < active_groups:
            active_groups = remaining_blocks
        if software_group_id >= active_groups:
            return
        PersistentFFT.round_tracker.store(worker_id, Float32(round_index) + Float32(1))


    @staticmethod
    def device_main():
        launch_parallel[PersistentFFT.stage_0]()
        launch_parallel[PersistentFFT.stage_1]()
        launch_parallel[PersistentFFT.stage_2]()
        launch_parallel[PersistentFFT.writeback]()
        launch_parallel[PersistentFFT.stage_0]()
        launch_parallel[PersistentFFT.stage_1]()
        launch_parallel[PersistentFFT.stage_2]()
        launch_parallel[PersistentFFT.writeback]()
        launch_parallel[PersistentFFT.stage_0]()
        launch_parallel[PersistentFFT.stage_1]()
        launch_parallel[PersistentFFT.stage_2]()
        launch_parallel[PersistentFFT.writeback]()


def main() raises:
    if PersistentFFT.emit_ir_if_asked():
        return

    var total_elems = 4160
    var input_real = cxl_alloc[Float32](total_elems)
    var input_imag = cxl_alloc[Float32](total_elems)
    var output_real = cxl_alloc[Float32](total_elems)
    var output_imag = cxl_alloc[Float32](total_elems)
    var ref_real = cxl_alloc[Float32](total_elems)
    var ref_imag = cxl_alloc[Float32](total_elems)

    seed(0)
    for i in range(total_elems):
        input_real[i] = Float32(random_float64(-1.0, 1.0))
        input_imag[i] = Float32(random_float64(-1.0, 1.0))
        output_real[i] = Float32(0)
        output_imag[i] = Float32(0)
        ref_real[i] = Float32(0)
        ref_imag[i] = Float32(0)

    var pool_elems = 2048
    var uthread_pool_raw = cxl_alloc[Float32](pool_elems + 64)
    var uthread_pool_raw_addr = Int(uthread_pool_raw)
    var uthread_pool_addr = (uthread_pool_raw_addr + 255) // 256 * 256
    if uthread_pool_addr % 256 != 0:
        print("[host] pool alignment assertion failed")
        return
    var uthread_pool = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=uthread_pool_addr)

    var buf_a = cxl_alloc[Float32](4096)
    var buf_b = cxl_alloc[Float32](4096)
    var rc = PersistentFFT.launch(
        PooledRange.over(uthread_pool, pool_elems),
        PersistentFFTParams(input_real, input_imag, output_real, output_imag, buf_a, buf_b),
    )

    if rc != 0:
        print("[host] persistent FFT failed, exit", rc)
        return

    var pi = Float64(3.141592653589793)
    var sign = Float64(-1.0)
    for batch in range(65):
        var batch_base = batch * 64
        for k in range(64):
            var acc_r = Float64(0)
            var acc_i = Float64(0)
            for n_ in range(64):
                var angle = sign * 2.0 * pi * Float64(n_) * Float64(k) / Float64(64)
                var c = host_cos(angle)
                var s = host_sin(angle)
                var xr = Float64(input_real[batch_base + n_])
                var xi = Float64(input_imag[batch_base + n_])
                acc_r += xr * c - xi * s
                acc_i += xr * s + xi * c
            ref_real[batch_base + k] = Float32(acc_r)
            ref_imag[batch_base + k] = Float32(acc_i)

    var tol = Float32(0.001)
    for i in range(4160):
        var err_r = output_real[i] - ref_real[i]
        var err_i = output_imag[i] - ref_imag[i]
        if err_r < Float32(0):
            err_r = -err_r
        if err_i < Float32(0):
            err_i = -err_i
        if err_r > tol or err_i > tol:
            print("[host] persistent FFT mismatch at", i)
            print("  expected:", ref_real[i], ref_imag[i])
            print("  actual:  ", output_real[i], output_imag[i])
            print("  error:   ", err_r, err_i)
            return

    print("[host] persistent FFT verification passed")
