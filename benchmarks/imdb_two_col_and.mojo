"""IMDB two-column AND — port of M2NDP-public imdb_two_col_AND.

Reference kernel:

    KERNELBODY:
    vsetvli t0, a0, e64, m1
    vle64.v v1, (x1)        ; this µthread's chunk of column a
    li x3, spad_addr
    ld x3, (x3)             ; column b
    add x3, x3, OFFSET
    vle64.v v2, (x3)
    li x4, spad_addr + 8
    ld x4, (x4)             ; the output
    add x4, x4, OFFSET
    vand.vv v3, v1, v2
    vse64.v v3, (x4)

Combining two scan results: each column is already a bitmap, so a row passes
both predicates where the bits agree.
"""

from std.sys import argv, size_of
from std.random import random_si64, seed

from m2ndp import PACKET, NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import Pool

comptime W = PACKET // size_of[Int64]()   # lanes in one packet


@fieldwise_init
struct TwoColAndParams(Movable):
    var a: UnsafePointer[Int64, MutAnyOrigin]
    var b: UnsafePointer[Int64, MutAnyOrigin]
    var out_bits: UnsafePointer[Int64, MutAnyOrigin]


struct ImdbTwoColAnd(NDPTask):
    comptime Params = TwoColAndParams

    @staticmethod
    def body():
        ref p = ImdbTwoColAnd.params[]
        var i = global_uthread_id() * W
        p.out_bits.store(i, p.a.load[width=W](i) & p.b.load[width=W](i))

    @staticmethod
    def device_main():
        launch_parallel[ImdbTwoColAnd.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh imdb_two_col_and


def main() raises:
    if ImdbTwoColAnd.emit_ir_if_asked():
        return

    var words = W * 64 * 8          # bitmap words, 64 rows each

    var pool = Pool()
    var a = pool.alloc[Int64](words)
    var b = pool.alloc[Int64](words)
    var res = pool.alloc[Int64](words)

    seed(0)
    for i in range(words):
        a[i] = random_si64(1, 9)
        b[i] = random_si64(1, 9)

    var rc = ImdbTwoColAnd.launch(
        pool, PooledRange.over(a, words), TwoColAndParams(a, b, res)
    )
    if rc != 0:
        print("[host] imdb_two_col_and failed, exit", rc)
        return

    for i in range(words):
        if res[i] != (a[i] & b[i]):
            print("[host] wrong at", i, ":", res[i], "expected", a[i] & b[i])
            return
    print("[host] imdb_two_col_and ok")
