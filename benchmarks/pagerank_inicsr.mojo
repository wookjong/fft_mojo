"""PageRank CSR setup — port of M2NDP-public examples/benchmarks/pagerank (kernel1).

Reference kernel, inner loop:

    vadd.vx v3, v2, x7          ; start + [0..7]
    vmslt.vx v0, v3, x8         ; mask off the edges past this node's end
    vsll.vi v3, v3, 2
    vluxei32.v v4, (x3), v3, v0 ; gather col[e]
    vsll.vi v4, v4, 2
    vluxei32.v v5, (x4), v4, v0 ; gather cnt[col[e]]
    vfcvt.f.x.v v5, v5
    vfrdiv.vf v5, v5, f1        ; 1 / outdegree
    vsuxei32.v v5, (x5), v3, v0 ; scatter into data[e]

Weighting every edge by the out-degree of the node it points at, before
PageRank iterates. A µthread owns a packet of the row array -- a run of nodes
-- and walks their edges.

The edges of one node are a variable-length run, so the gather and scatter are
written as a loop over the run rather than the reference's masked
`vluxei32.v`/`vsuxei32.v` pair. Closing that needs a masked-gather primitive,
the same shape of gap `imdb_lt_int64` records for mask-to-bitmap.
"""

from std.sys import argv, size_of
from std.random import random_si64, seed

from m2ndp import PACKET, NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import cxl_alloc

comptime W = PACKET // size_of[Int32]()   # nodes in one packet of the row array


@fieldwise_init
struct InicsrParams(Movable):
    var rows: UnsafePointer[Int32, MutAnyOrigin]
    var cols: UnsafePointer[Int32, MutAnyOrigin]
    var counts: UnsafePointer[Int32, MutAnyOrigin]
    var data: UnsafePointer[Float32, MutAnyOrigin]


struct PagerankInicsr(NDPTask):
    comptime Params = InicsrParams

    @staticmethod
    def body():
        ref p = PagerankInicsr.params[]
        var first = global_uthread_id() * W
        for node in range(first, first + W):
            for e in range(Int(p.rows[node]), Int(p.rows[node + 1])):
                var col = Int(p.cols[e])
                p.data[e] = 1 / Float32(p.counts[col])

    @staticmethod
    def device_main():
        launch_parallel[PagerankInicsr.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh pagerank_inicsr
#
# The range is the row array, so the µthread count follows the node count. It
# is one longer than the graph has nodes -- CSR's end marker -- and the extra
# entry is left out of the range so no µthread reads past it.


def main() raises:
    if PagerankInicsr.emit_ir_if_asked():
        return

    var nodes = W * 64 * 2
    var degree = 8
    var edges = nodes * degree

    var rows = cxl_alloc[Int32](nodes + 1)
    var cols = cxl_alloc[Int32](edges)
    var counts = cxl_alloc[Int32](nodes)
    var data = cxl_alloc[Float32](edges)

    seed(0)
    for i in range(nodes + 1):
        rows[i] = Int32(i * degree)
    for e in range(edges):
        cols[e] = Int32(random_si64(0, nodes - 1))
    for i in range(nodes):
        counts[i] = Int32(random_si64(1, 16))

    var rc = PagerankInicsr.launch(
        PooledRange.over(rows, nodes),
        InicsrParams(rows, cols, counts, data)
    )
    if rc != 0:
        print("[host] pagerank_inicsr failed, exit", rc)
        return

    for e in range(edges):
        var want = 1 / Float32(counts[Int(cols[e])])
        if data[e] != want:
            print("[host] wrong at edge", e, ":", data[e], "expected", want)
            return
    print("[host] pagerank_inicsr ok")
