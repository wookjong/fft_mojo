"""IMDB three-column AND — port of M2NDP-public imdb_three_col_AND.

The two-column kernel with one more column folded in: three bitmap loads, two
`vand.vv`, one store. It is here because the third column is what makes the
µthread's parameter block outgrow a single pointer pair, so the scratchpad
carries four addresses rather than two.
"""

from std.sys import argv, size_of
from std.random import random_si64, seed

from m2ndp import VECTOR_WIDTH, NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import cxl_alloc

comptime W = VECTOR_WIDTH // size_of[Int64]()   # lanes in one vector


@fieldwise_init
struct ThreeColAndParams(Movable):
    var a: UnsafePointer[Int64, MutAnyOrigin]
    var b: UnsafePointer[Int64, MutAnyOrigin]
    var c: UnsafePointer[Int64, MutAnyOrigin]
    var out_bits: UnsafePointer[Int64, MutAnyOrigin]


struct ImdbThreeColAnd(NDPTask):
    comptime Params = ThreeColAndParams

    @staticmethod
    def body():
        ref p = ImdbThreeColAnd.params[]
        var i = global_uthread_id() * W
        p.out_bits.store(
            i,
            p.a.load[width=W](i) & p.b.load[width=W](i) & p.c.load[width=W](i),
        )

    @staticmethod
    def device_main():
        launch_parallel[ImdbThreeColAnd.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh imdb_three_col_and


def main() raises:
    if ImdbThreeColAnd.emit_ir_if_asked():
        return

    var words = W * 64 * 8          # bitmap words, 64 rows each

    var a = cxl_alloc[Int64](words)
    var b = cxl_alloc[Int64](words)
    var c = cxl_alloc[Int64](words)
    var res = cxl_alloc[Int64](words)

    seed(0)
    for i in range(words):
        a[i] = random_si64(1, 9)
        b[i] = random_si64(1, 9)
        c[i] = random_si64(1, 9)

    var rc = ImdbThreeColAnd.launch(
        PooledRange.over(a, words), ThreeColAndParams(a, b, c, res)
    )
    if rc != 0:
        print("[host] imdb_three_col_and failed, exit", rc)
        return

    for i in range(words):
        var want = a[i] & b[i] & c[i]
        if res[i] != want:
            print("[host] wrong at", i, ":", res[i], "expected", want)
            return
    print("[host] imdb_three_col_and ok")
