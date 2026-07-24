"""SSSP relaxation — port of M2NDP-public examples/benchmarks/sssp (kernel0).

Reference kernel, inner loop:

    vid.v v5
    vadd.vx v5, v5, x9              ; this node's edges, eight at a time
    vmslt.vx v0, v5, x11            ; masked off at the node's last edge
    vluxei32.v v6, (x4), v5, v0     ; col[e]
    vluxei32.v v7, (x5), v5, v0     ; weight[e]
    vluxei32.v v8, (x12), v6, v0    ; dist[col[e]]
    vadd.vv v6, v8, v7
    vredmin.vs v5, v6, v7, v0       ; against the node's own distance
    vmv.x.s x8, v5
    ...
    vse32.v v3, (x13)               ; the new distance vector

One Bellman-Ford relaxation pass: a node's new distance is the smallest of its
own and every neighbour's plus the edge between them. A µthread owns a packet
of the row array and relaxes those nodes.

The reduction is written over the edge run rather than the reference's masked
gather; see `pagerank_inicsr` for the same gap.
"""

from std.sys import argv, size_of
from std.random import random_si64, seed

from m2ndp import PACKET, NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import Pool

comptime W = PACKET // size_of[Int32]()   # nodes in one packet of the row array
comptime UNREACHED = Int32(0x3FFFFFFF)    # far enough that no path is longer


@fieldwise_init
struct SsspParams(Movable):
    var rows: UnsafePointer[Int32, MutAnyOrigin]
    var cols: UnsafePointer[Int32, MutAnyOrigin]
    var weights: UnsafePointer[Int32, MutAnyOrigin]
    var distance: UnsafePointer[Int32, MutAnyOrigin]
    var updated: UnsafePointer[Int32, MutAnyOrigin]


struct Sssp(NDPTask):
    comptime Params = SsspParams

    @staticmethod
    def body():
        ref p = Sssp.params[]
        var first = global_uthread_id() * W
        for node in range(first, first + W):
            var best = p.distance[node]
            for e in range(Int(p.rows[node]), Int(p.rows[node + 1])):
                var neighbour = p.distance[Int(p.cols[e])]
                if neighbour < UNREACHED:
                    var through = neighbour + p.weights[e]
                    if through < best:
                        best = through
            p.updated[node] = best

    @staticmethod
    def device_main():
        launch_parallel[Sssp.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh sssp
#
# One pass, into a second distance array: the reference writes `vector2` from
# `vector1` rather than in place, so a node relaxed early in the pass cannot
# feed a node relaxed later in it.


def main() raises:
    if Sssp.emit_ir_if_asked():
        return

    var nodes = W * 64 * 2
    var degree = 8
    var edges = nodes * degree

    var pool = Pool()
    var rows = pool.alloc[Int32](nodes + 1)
    var cols = pool.alloc[Int32](edges)
    var weights = pool.alloc[Int32](edges)
    var distance = pool.alloc[Int32](nodes)
    var updated = pool.alloc[Int32](nodes)

    seed(0)
    for i in range(nodes + 1):
        rows[i] = Int32(i * degree)
    for e in range(edges):
        cols[e] = Int32(random_si64(0, nodes - 1))
        weights[e] = Int32(random_si64(1, 100))
    distance[0] = 0
    for i in range(1, nodes):
        distance[i] = UNREACHED

    var rc = Sssp.launch(
        pool, PooledRange.over(rows, nodes),
        SsspParams(rows, cols, weights, distance, updated)
    )
    if rc != 0:
        print("[host] sssp failed, exit", rc)
        return

    for node in range(nodes):
        var want = distance[node]
        for e in range(Int(rows[node]), Int(rows[node + 1])):
            var neighbour = distance[Int(cols[e])]
            if neighbour < UNREACHED:
                var through = neighbour + weights[e]
                if through < want:
                    want = through
        if updated[node] != want:
            print("[host] wrong at node", node, ":", updated[node],
                  "expected", want)
            return
    print("[host] sssp ok")
