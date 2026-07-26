"""IMDB fp32 range filter — port of M2NDP-public imdb_gt_lt_FP32.

Reference kernel:

    KERNELBODY:
    vsetvli t0, a0, e32, m1
    vle32.v v1, x1
    li x7, spad_addr + 64
    vle32.v v4, (x7)            ; the two bounds, as floats
    vfmv.f.s f1, v4
    csrwi vstart, 1
    vfmv.f.s f2, v4
    vmsgt.vf v2, v1, f1         ; lo < x
    vmslt.vf v3, v1, f2         ; x < hi
    vmand.mm v2, v2, v3
    srli x5, x2, 5
    ...
    sb x3, (x6)                 ; two bitmap bytes, W being 16 lanes there
    sb x10, 1(x6)

The int64 range filter over floats: same two compares, `vmsgt.vf`/`vmslt.vf`
selected by the operand type. A µthread packs its `W` mask bits into `W / 8`
bitmap bytes -- one at the 32-byte packet this build uses, where fp32 gives
eight lanes, and two at the reference's 64.
"""

from std.sys import argv, size_of
from std.random import random_float64, seed

from m2ndp import PACKET, NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import Pool

comptime W = PACKET // size_of[Float32]()   # lanes in one packet
comptime BYTES = W // 8                     # bitmap bytes they fill


@fieldwise_init
struct GtLtFp32Params(Movable):
    var column: UnsafePointer[Float32, MutAnyOrigin]
    var bitmap: UnsafePointer[UInt8, MutAnyOrigin]
    var lo: Float32
    var hi: Float32


struct ImdbGtLtFp32(NDPTask):
    comptime Params = GtLtFp32Params

    @staticmethod
    def body():
        var i = global_uthread_id()
        ref p = ImdbGtLtFp32.params[]
        var v = p.column.load[width=W](i * W)
        var mask = v.gt(p.lo) & v.lt(p.hi)

        comptime for byte in range(BYTES):
            var bits = UInt8(0)
            comptime for lane in range(8):
                if mask[byte * 8 + lane]:
                    bits |= UInt8(1 << lane)
            p.bitmap[i * BYTES + byte] = bits

    @staticmethod
    def device_main():
        launch_parallel[ImdbGtLtFp32.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh imdb_gt_lt_fp32


def main() raises:
    if ImdbGtLtFp32.emit_ir_if_asked():
        return

    var rows = W * 64 * 8

    var pool = Pool()
    var column = pool.alloc[Float32](rows)
    var bitmap = pool.alloc[UInt8](rows // 8)
    var lo = Float32(3.0)
    var hi = Float32(5.0)

    seed(0)
    for i in range(rows):
        column[i] = Float32(random_float64(1.0, 10.0))

    var rc = ImdbGtLtFp32.launch(
        pool, PooledRange.over(column, rows),
        GtLtFp32Params(column, bitmap, lo, hi)
    )
    if rc != 0:
        print("[host] imdb_gt_lt_fp32 failed, exit", rc)
        return

    for i in range(rows // 8):
        var want = UInt8(0)
        for lane in range(8):
            var x = column[i * 8 + lane]
            if x > lo and x < hi:
                want |= UInt8(1 << lane)
        if bitmap[i] != want:
            print("[host] wrong at byte", i, ":", bitmap[i], "expected", want)
            return
    print("[host] imdb_gt_lt_fp32 ok")
