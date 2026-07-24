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
which rows passed. The interesting part is that the comparison result is a
*mask*, packed to bits rather than stored as one value per row.

The compare itself ports cleanly — `v.lt(predicate)` selects `vmslt.vx`, the
same instruction the reference uses. Packing the mask into a byte does not:
the reference gets it for free, because an RVV mask register already holds
one bit per lane, so `vmv.x.s` + `sb` is the whole job. Mojo has no way to
name that. `SIMD[bool, W]` cannot be converted to an integer bitmask —
`Int(mask)` only instantiates at width 1, and there is no movemask-style
primitive — so the lanes have to be tested and OR'd back together one at a
time. The mask register is still produced; the compiler then spends ~15
and/or instructions rebuilding the bit pattern it already had.

Like the vector atomic in histogram.mojo, closing this gap needs a
primitive, not a rewrite of the benchmark.
"""

from std.ffi import external_call
from std.sys import argv, size_of

from m2ndp import NDPTask, PooledRange, global_uthread_id
from m2ndp_host import Buffer

comptime W = 8   # int64 lanes per chunk; one bitmap byte covers exactly these


@fieldwise_init
struct ImdbParams(Copyable, Movable):
    """What the host passes. `predicate` comes as a one-element buffer,
    since the parameter block is addresses; `device_main` reads it and hands
    the kernel the value."""

    var column: UnsafePointer[Int64, MutAnyOrigin]
    var bitmap: UnsafePointer[UInt8, MutAnyOrigin]
    var predicate: UnsafePointer[Int64, MutAnyOrigin]


struct ImdbLtInt64(NDPTask):
    comptime packet = W * size_of[Int64]()

    @staticmethod
    def body(column: UnsafePointer[Int64, MutAnyOrigin],
             bitmap: UnsafePointer[UInt8, MutAnyOrigin],
             predicate: Int64):
        var i = global_uthread_id()
        var v = column.load[width=W](i * W)
        var mask = v.lt(predicate)            # SIMD[bool, W]

        # Pack the lanes into one bitmap byte.
        var bits = UInt8(0)
        comptime for lane in range(W):
            if mask[lane]:
                bits |= UInt8(1 << lane)
        bitmap[i] = bits

    @staticmethod
    def device_main(params: UnsafePointer[NoneType, MutAnyOrigin]):
        var p = params.bitcast[ImdbParams]()
        external_call["__m2ndp_launch_parallel", NoneType](
            ImdbLtInt64.body, Int(p[].column), Int(p[].bitmap),
            Int(p[].predicate[0]), Int(0), Int(0), Int(0)
        )


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

    var column = List[Int64](length=rows, fill=0)
    var bitmap = List[UInt8](length=rows // W, fill=0)
    var predicate = List[Int64](length=1, fill=0)
    predicate[0] = 0

    var state: Int = 20260724
    for i in range(rows):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        column[i] = Int64((state >> 8) % 2000 - 1000)

    var rc = ImdbLtInt64.launch(
        PooledRange.over(column),
        Buffer.input(column), Buffer.output(bitmap), Buffer.input(predicate)
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
