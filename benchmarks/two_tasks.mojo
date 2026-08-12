"""Two tasks, one host program: `Scale` then `AddB`, chained through the pool.

Each task is compiled and launched on its own, so what the second one reads is
what the first one left in the pool. Nothing is transferred between them.
"""

from std.sys import argv, size_of
from std.random import random_si64, seed

from m2ndp import VECTOR_WIDTH, NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import cxl_alloc

comptime W = VECTOR_WIDTH // size_of[Int32]()   # lanes in one vector


@fieldwise_init
struct ScaleParams(Movable):
    var a: UnsafePointer[Int32, MutAnyOrigin]
    var c: UnsafePointer[Int32, MutAnyOrigin]


@fieldwise_init
struct AddBParams(Movable):
    var c: UnsafePointer[Int32, MutAnyOrigin]
    var b: UnsafePointer[Int32, MutAnyOrigin]
    var d: UnsafePointer[Int32, MutAnyOrigin]


struct Scale(NDPTask):
    comptime Params = ScaleParams

    @staticmethod
    def body():
        ref p = Scale.params[]
        var i = global_uthread_id() * W
        p.c.store(i, p.a.load[width=W](i) + p.a.load[width=W](i))

    @staticmethod
    def device_main():
        launch_parallel[Scale.body]()


struct AddB(NDPTask):
    comptime Params = AddBParams

    @staticmethod
    def body():
        ref p = AddB.params[]
        var i = global_uthread_id() * W
        p.d.store(i, p.c.load[width=W](i) + p.b.load[width=W](i))

    @staticmethod
    def device_main():
        launch_parallel[AddB.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh two_tasks


def main() raises:
    if Scale.emit_ir_if_asked():
        return

    var n = W * 64 * 8

    var a = cxl_alloc[Int32](n)
    var b = cxl_alloc[Int32](n)
    var c = cxl_alloc[Int32](n)
    var d = cxl_alloc[Int32](n)
    var expect = List[Int32](length=n, fill=0)

    seed(0)
    for i in range(n):
        a[i] = Int32(random_si64(-1000, 999))
        b[i] = Int32(random_si64(-1000, 999))
        expect[i] = a[i] * 2 + b[i]

    var rc = Scale.launch(PooledRange.over(a, n), ScaleParams(a, c))
    if rc != 0:
        print("[host] two_tasks: Scale failed, exit", rc)
        return

    rc = AddB.launch(PooledRange.over(c, n), AddBParams(c, b, d))
    if rc != 0:
        print("[host] two_tasks: AddB failed, exit", rc)
        return

    for i in range(n):
        if d[i] != expect[i]:
            print("[host] wrong at", i, ":", d[i], "expected", expect[i])
            return
    print("[host] two_tasks ok")
