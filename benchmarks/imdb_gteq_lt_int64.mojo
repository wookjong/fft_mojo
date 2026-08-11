"""IMDB range filter — port of M2NDP-public imdb_gteq_lt_INT64.

Reference kernel:

    KERNELBODY:
    vsetvli t0, a0, e64, m1
    vle64.v v1, x1              ; this µthread's chunk of the column
    li x7, spad_addr
    vle64.v v4, (x7)            ; args: output_addr, min, max
    vmv.x.s x3, v4
    csrwi vstart, 1
    vmv.x.s x4, v4
    csrwi vstart, 2
    vmv.x.s x5, v4
    vmsge.vx v2, v1, x4         ; min <= x
    vmslt.vx v3, v1, x5         ; x < max
    vmand.mm v2, v2, v3
    srli x5, x2, 6
    add x6, x3, x5
    vsetvli t0, a0, e8, m1
    vmv.x.s x3, v2
    sb x3, (x6)

`imdb_lt_int64` with two bounds instead of one: two compares and'd together.
Both bounds go in by value, and packing the mask into a byte costs the same
lane-at-a-time loop for the same reason.
"""

from std.sys import argv, size_of
from std.random import random_si64, seed

from m2ndp import VECTOR_WIDTH, NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import cxl_alloc

comptime W = VECTOR_WIDTH // size_of[Int64]()   # lanes in one vector;
                                          # one bitmap byte covers them


@fieldwise_init
struct GteqLtParams(Movable):
    var column: UnsafePointer[Int64, MutAnyOrigin]
    var bitmap: UnsafePointer[UInt8, MutAnyOrigin]
    var lo: Int64
    var hi: Int64


struct ImdbGteqLtInt64(NDPTask):
    comptime Params = GteqLtParams

    @staticmethod
    def body():
        var i = global_uthread_id()
        ref p = ImdbGteqLtInt64.params[]
        var v = p.column.load[width=W](i * W)
        var mask = v.ge(p.lo) & v.lt(p.hi)

        var bits = UInt8(0)
        comptime for lane in range(W):
            if mask[lane]:
                bits |= UInt8(1 << lane)
        p.bitmap[i] = bits

    @staticmethod
    def device_main():
        launch_parallel[ImdbGteqLtInt64.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh imdb_gteq_lt_int64


def main() raises:
    if ImdbGteqLtInt64.emit_ir_if_asked():
        return

    var rows = W * 64 * 8        # one bitmap byte per W rows

    var column = cxl_alloc[Int64](rows)
    var bitmap = cxl_alloc[UInt8](rows // W)
    var lo = Int64(3)
    var hi = Int64(5)

    seed(0)
    for i in range(rows):
        column[i] = random_si64(1, 9)

    var rc = ImdbGteqLtInt64.launch(
        PooledRange.over(column, rows),
        GteqLtParams(column, bitmap, lo, hi)
    )
    if rc != 0:
        print("[host] imdb_gteq_lt_int64 failed, exit", rc)
        return

    for i in range(rows // W):
        var want = UInt8(0)
        for lane in range(W):
            var x = column[i * W + lane]
            if x >= lo and x < hi:
                want |= UInt8(1 << lane)
        if bitmap[i] != want:
            print("[host] wrong at byte", i, ":", bitmap[i], "expected", want)
            return
    print("[host] imdb_gteq_lt_int64 ok")
