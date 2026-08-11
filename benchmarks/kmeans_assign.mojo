"""K-means assignment — port of M2NDP-public examples/benchmarks/kmeans (kernel1).

Reference kernel:

    KERNELBODY:
    vsetvli t0, a0, e32, m1
    vle32.v v1, ADDR        ; the distances to every cluster
    li x1, spad_addr
    ld x2, (x1)             ; the output
    vmset.m v0
    vmv.v.i v5, 0
    vredmin.vs v2, v1, v1, v0
    vmv.x.s x3, v2          ; the smallest
    vmseq.vx v3, v1, x3
    vfirst.m x4, v3, v3     ; and which cluster it was
    vmv.s.x v5, x4
    vse32.v v5, (x2)

Picking each point's cluster: reduce the distance row to its minimum, then
find the lane that holds it. The reference runs one µthread over one row; here
a row is one packet and every µthread does its own, which is the same kernel
over a whole dataset rather than a single point.
"""

from std.sys import argv, size_of
from std.random import random_si64, seed

from m2ndp import VECTOR_WIDTH, NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import cxl_alloc

comptime CLUSTERS = VECTOR_WIDTH // size_of[Int32]()   # distances in one vector


@fieldwise_init
struct KmeansParams(Movable):
    var distances: UnsafePointer[Int32, MutAnyOrigin]
    var assignment: UnsafePointer[Int32, MutAnyOrigin]


struct KmeansAssign(NDPTask):
    comptime Params = KmeansParams

    @staticmethod
    def body():
        ref p = KmeansAssign.params[]
        var point = global_uthread_id()
        var row = p.distances.load[width=CLUSTERS](point * CLUSTERS)

        var best = row.reduce_min()
        var mask = row.eq(best)
        var first = Int32(0)
        comptime for c in reversed(range(CLUSTERS)):
            if mask[c]:
                first = Int32(c)
        p.assignment[point] = first

    @staticmethod
    def device_main():
        launch_parallel[KmeansAssign.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh kmeans_assign


def main() raises:
    if KmeansAssign.emit_ir_if_asked():
        return

    var points = 64 * 8
    var n = points * CLUSTERS

    var distances = cxl_alloc[Int32](n)
    var assignment = cxl_alloc[Int32](points)

    seed(0)
    for i in range(n):
        distances[i] = Int32(random_si64(0, 9))

    var rc = KmeansAssign.launch(
        PooledRange.over(distances, n),
        KmeansParams(distances, assignment)
    )
    if rc != 0:
        print("[host] kmeans_assign failed, exit", rc)
        return

    for i in range(points):
        var best = distances[i * CLUSTERS]
        var want = 0
        for c in range(1, CLUSTERS):
            if distances[i * CLUSTERS + c] < best:
                best = distances[i * CLUSTERS + c]
                want = c
        if assignment[i] != Int32(want):
            print("[host] wrong at", i, ":", assignment[i], "expected", want)
            return
    print("[host] kmeans_assign ok")
