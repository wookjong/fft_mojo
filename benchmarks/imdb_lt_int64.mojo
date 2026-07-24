"""IMDB lt-int64 filter — port of M2NDP-public imdb_lt_INT64.

Reference kernel:

    KERNELBODY:
    vsetvli t0, a0, e64, m1
    vle64.v v1, x1              ; this µthread's chunk of the column
    li x7, spad_addr
    vle64.v v4, (x7)            ; args: output_addr, predicate_value
    vmv.x.s x3, v4
    csrwi vstart, 1
    vmv.x.s x4, v4
    vmslt.vx v2, v1, x4         ; lanewise column < predicate -> mask register
    srli x5, x2, 6              ; µthread index / 64 -> byte index in the bitmap
    add x6, x3, x5
    vsetvli t0, a0, e8, m1
    vmv.x.s x3, v2              ; move the mask out as bits
    sb x3, (x6)                 ; one byte of bitmap per 8 rows

A database scan: compare a column against a constant and write a bitmap of
which rows passed, one bit per row.

The compare ports cleanly — `v.lt(predicate)` selects `vmslt.vx`. Packing the
mask into a byte does not. The reference gets it free from an RVV mask
register (`vmv.x.s` + `sb`), but `SIMD[bool, W]` has no conversion to an
integer bitmask, so the lanes are tested and OR'd back one at a time and the
compiler spends ~15 instructions rebuilding a bit pattern it already had.

Closing this needs a primitive, not a rewrite of the benchmark.
"""

from std.sys import argv, size_of
from std.random import random_si64, seed

from m2ndp import PACKET, NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import Pool

comptime W = PACKET // size_of[Int64]()   # lanes in one packet;
                                          # one bitmap byte covers them


@fieldwise_init
struct ImdbParams(Movable):
    """What the host passes. `predicate` comes as a one-element buffer,
    since the parameter block is addresses; the kernel dereferences it, so it
    stays an Int64 rather than being widened on the way through."""

    var column: UnsafePointer[Int64, MutAnyOrigin]
    var bitmap: UnsafePointer[UInt8, MutAnyOrigin]
    var predicate: UnsafePointer[Int64, MutAnyOrigin]


struct ImdbLtInt64(NDPTask):
    comptime Params = ImdbParams

    @staticmethod
    def body():
        var i = global_uthread_id()
        ref p = ImdbLtInt64.params[]
        var v = p.column.load[width=W](i * W)
        var mask = v.lt(p.predicate[0])          # SIMD[bool, W]

        # Pack the lanes into one bitmap byte.
        var bits = UInt8(0)
        comptime for lane in range(W):
            if mask[lane]:
                bits |= UInt8(1 << lane)
        p.bitmap[i] = bits

    @staticmethod
    def device_main():
        launch_parallel[ImdbLtInt64.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh imdb_lt_int64
#
# Each microthread compares W int64 rows against a constant and writes one
# bitmap byte, so the host's check is the bit pattern rather than a value per
# row -- which is the part the reference gets free from an RVV mask register
# and this does not. See the module docstring above.


def main() raises:
    if ImdbLtInt64.emit_ir_if_asked():
        return

    var rows = W * 64 * 8        # one bitmap byte per W rows

    var pool = Pool()
    var column = pool.alloc[Int64](rows)
    var bitmap = pool.alloc[UInt8](rows // W)
    var predicate = pool.alloc[Int64](1)
    predicate[0] = 0

    seed(0)
    for i in range(rows):
        column[i] = random_si64(-1000, 999)

    var rc = ImdbLtInt64.launch(
        pool, PooledRange.over(column, rows),
        ImdbParams(column, bitmap, predicate)
    )
    if rc != 0:
        print("[host] imdb_lt_int64 failed, exit", rc)
        return

    for i in range(rows // W):
        var want = UInt8(0)
        for lane in range(W):
            if column[i * W + lane] < predicate[0]:
                want |= UInt8(1 << lane)
        if bitmap[i] != want:
            print("[host] wrong at byte", i, ":", bitmap[i], "expected", want)
            return
    print("[host] imdb_lt_int64 ok")
