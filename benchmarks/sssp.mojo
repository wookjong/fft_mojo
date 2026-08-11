"""SSSP — port of M2NDP-Detour benchmarks/sssp, the device-side control flow one.

Bellman-Ford, iterated on the device. Two kernels and the loop that drives them:

    void SSSP::device_main() {
      do {
        flag = 0;
        min_dot_plus();         // relax every node from the current distances
        vector_diff_assign();   // did anything change? if so, take the new ones
      } while (flag);
    }

`min_dot_plus` is the relaxation: a node's new distance is the smallest of its
own and every neighbour's plus the edge between them. A µthread owns a packet
of the row array and relaxes those nodes, taking each node's edges a vector at
a time under a mask that closes at the last one:

    NDPMask<int> vm = LtS(I, Iota(I, i), row[idx+1]);
    NDPVec<int> v_col  = MaskedLoad(vm, I, col + i);
    NDPVec<int> v_data = MaskedLoad(vm, I, data + i);
    NDPVec<int> v_x    = MaskedIndexedLoad(I, x, I, v_col, vm);
    NDPVec<int> v_min  = MinS(I, Add(I, v_data, v_x), min);
    min = MaskedReduceMin(I, v_min, vm);

`UnsafePointer.gather` is the masked indexed load and reaches `vluxei64.v`
under `v0.t`; a masked reduction is `select` against the identity and then
`reduce_min`, which reaches `vredmin.vs`.

`vector_diff_assign` is the convergence test: compare the packet it owns before
and after, raise `flag` if any lane moved, and take the new distances.
"""

from std.math import iota
from std.sys import argv, size_of
from std.random import random_si64, seed

from m2ndp import (
    VECTOR_WIDTH,
    NDPTask,
    PooledRange,
    global_uthread_id,
    launch_parallel,
    spad_addr,
)
from m2ndp_host import cxl_alloc

comptime W = VECTOR_WIDTH // size_of[Int32]()   # nodes in one vector of the row array
comptime UNREACHED = Int32(0x3FFFFFFF)    # far enough that no path is longer


@fieldwise_init
struct SsspParams(Movable):
    var rows: UnsafePointer[Int32, MutAnyOrigin]
    var cols: UnsafePointer[Int32, MutAnyOrigin]
    var weights: UnsafePointer[Int32, MutAnyOrigin]
    var distance: UnsafePointer[Int32, MutAnyOrigin]
    var updated: UnsafePointer[Int32, MutAnyOrigin]
    var flag: UnsafePointer[Int32, MutAnyOrigin]
    """Raised by any µthread that moved a distance, cleared between passes."""


struct Sssp(NDPTask):
    comptime Params = SsspParams

    @staticmethod
    def min_dot_plus():
        """One relaxation pass, into `updated` from `distance`.

        Into a second array rather than in place, so a node relaxed early in
        the pass cannot feed a node relaxed later in it.
        """
        ref p = Sssp.params[]
        var first = global_uthread_id() * W
        for node in range(first, first + W):
            var best = p.distance[node]
            var last = Int(p.rows[node + 1])
            var e = Int(p.rows[node])
            while e < last:
                # This node's edges, a vector at a time, masked off at the last.
                var lane = iota[DType.int32, W]() + Int32(e)
                var vm = lane.lt(Int32(last))
                var zero = SIMD[DType.int32, W](0)
                var col = p.cols.gather[width=W, alignment=4](lane, vm, zero)
                var weight = p.weights.gather[width=W, alignment=4](
                    lane, vm, zero
                )
                var neighbour = p.distance.gather[width=W, alignment=4](
                    col, vm, SIMD[DType.int32, W](UNREACHED)
                )
                var through = min(
                    neighbour + weight, SIMD[DType.int32, W](best)
                )
                # Masked reduction: the lanes past the end reduce to nothing.
                var live = vm.select(through, SIMD[DType.int32, W](Int32.MAX))
                best = live.reduce_min()
                e += W
            p.updated[node] = best

    @staticmethod
    def vector_diff_assign():
        """Did this packet move? If so raise the flag and take the new distances."""
        ref p = Sssp.params[]
        var first = global_uthread_id() * W
        var was = p.distance.load[width=W](first)
        var now = p.updated.load[width=W](first)
        if was.ne(now).reduce_or():
            p.flag[0] = 1
            p.distance.store(first, now)

    @staticmethod
    def device_main():
        """Relax and take what moved, until a pass moves nothing.

        The parameters come through `spad_addr`, not `Self.params`: indexing a
        scratchpad global needs the base register a kernel is given and the
        controller has none. `flag` itself is in the pool, which the controller
        addresses directly.
        """
        ref p = spad_addr(Sssp.params, 0)[]
        while True:
            p.flag[0] = 0
            launch_parallel[Sssp.min_dot_plus]()
            launch_parallel[Sssp.vector_diff_assign]()
            if p.flag[0] == 0:
                break


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh sssp
#
# One launch: the device iterates to convergence itself, so the host only has
# to check the answer against its own Bellman-Ford.


def main() raises:
    if Sssp.emit_ir_if_asked():
        return

    var nodes = W * 64 * 2
    var degree = 8
    var edges = nodes * degree

    var rows = cxl_alloc[Int32](nodes + 1)
    var cols = cxl_alloc[Int32](edges)
    var weights = cxl_alloc[Int32](edges)
    var distance = cxl_alloc[Int32](nodes)
    var updated = cxl_alloc[Int32](nodes)
    var flag = cxl_alloc[Int32](1)

    seed(0)
    for i in range(nodes + 1):
        rows[i] = Int32(i * degree)
    for e in range(edges):
        cols[e] = Int32(random_si64(0, nodes - 1))
        weights[e] = Int32(random_si64(1, 100))
    distance[0] = 0
    for i in range(1, nodes):
        distance[i] = UNREACHED

    # The answer, before the device overwrites the distances it starts from.
    var want = List[Int32](capacity=nodes)
    for node in range(nodes):
        want.append(distance[node])
    while True:
        var moved = False
        for node in range(nodes):
            var best = want[node]
            for e in range(Int(rows[node]), Int(rows[node + 1])):
                var through = want[Int(cols[e])] + weights[e]
                if through < best:
                    best = through
            if best != want[node]:
                want[node] = best
                moved = True
        if not moved:
            break

    var rc = Sssp.launch(
        PooledRange.over(rows, nodes),
        SsspParams(rows, cols, weights, distance, updated, flag)
    )
    if rc != 0:
        print("[host] sssp failed, exit", rc)
        return

    for node in range(nodes):
        if distance[node] != want[node]:
            print("[host] wrong at node", node, ":", distance[node],
                  "expected", want[node])
            return
    print("[host] sssp ok")
