"""DLRM sparse-length sum — port of M2NDP-public examples/benchmarks/dlrm.

Reference kernel:

    KERNELBODY:
    vmv.v.i v1, 0
    ...
    srli x5, OFFSET, log2(emb_dim)      ; which batch item this µthread is in
    andi x6, OFFSET, emb_dim * 4 - 1    ; and where in the embedding row
    add x7, x4, x5
    lw x8, 4(x7)                        ; the item's index range,
    lw x7, (x7)                         ; from the offsets array
    .LOOP1
    ld x9, (x9)                         ; an index
    muli x9, x9, emb_dim * 4
    vle32.v v10, (x9)                   ; the row it names
    vfadd.vv v1, v1, v10
    blt x7, x8, .LOOP1
    vse32.v v1, (ADDR)

DLRM's embedding stage: each batch item names a list of rows and wants their
sum. The µthread is mapped over the *output*, so it owns one packet of one
item's embedding and walks that item's whole index list -- a loop whose length
is data, which is what makes this workload different from the elementwise
ones.
"""

from std.sys import argv, size_of
from std.random import random_si64, seed

from m2ndp import PACKET, NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import Pool

comptime W = PACKET // size_of[Float32]()   # lanes in one packet
comptime EMB_DIM = 256                      # elements in an embedding row
comptime PER_ROW = EMB_DIM // W             # packets it takes


@fieldwise_init
struct SlsParams(Movable):
    var table: UnsafePointer[Float32, MutAnyOrigin]
    var index: UnsafePointer[Int32, MutAnyOrigin]
    var offset: UnsafePointer[Int32, MutAnyOrigin]
    var output: UnsafePointer[Float32, MutAnyOrigin]


struct DlrmSls(NDPTask):
    comptime Params = SlsParams

    @staticmethod
    def body():
        ref p = DlrmSls.params[]
        var u = global_uthread_id()
        var item = u // PER_ROW               # which batch item
        var within = (u % PER_ROW) * W        # where in its row

        var acc = SIMD[DType.float32, W](0)
        var first = Int(p.offset[item])
        var last = Int(p.offset[item + 1])
        for k in range(first, last):
            var row = Int(p.index[k])
            acc += p.table.load[width=W](row * EMB_DIM + within)

        p.output.store(u * W, acc)

    @staticmethod
    def device_main():
        launch_parallel[DlrmSls.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh dlrm_sls
#
# The range is the output, one packet per µthread, as the reference has it.


def main() raises:
    if DlrmSls.emit_ir_if_asked():
        return

    var batch = 64
    var lookups = 16
    var rows = 4096
    var n = batch * EMB_DIM

    var pool = Pool()
    var table = pool.alloc[Float32](rows * EMB_DIM)
    var index = pool.alloc[Int32](batch * lookups)
    var offset = pool.alloc[Int32](batch + 1)
    var output = pool.alloc[Float32](n)

    seed(0)
    for i in range(rows * EMB_DIM):
        table[i] = Float32(random_si64(-100, 100)) / 100
    for i in range(batch * lookups):
        index[i] = Int32(random_si64(0, rows - 1))
    for i in range(batch + 1):
        offset[i] = Int32(i * lookups)

    var rc = DlrmSls.launch(
        pool, PooledRange.over(output, n),
        SlsParams(table, index, offset, output)
    )
    if rc != 0:
        print("[host] dlrm_sls failed, exit", rc)
        return

    for item in range(batch):
        for d in range(EMB_DIM):
            var want = Float32(0)
            for k in range(Int(offset[item]), Int(offset[item + 1])):
                want += table[Int(index[k]) * EMB_DIM + d]
            var got = output[item * EMB_DIM + d]
            if abs(got - want) > 1e-4 * abs(want) + 1e-5:
                print("[host] wrong at", item, d, ":", got, "expected", want)
                return
    print("[host] dlrm_sls ok")
