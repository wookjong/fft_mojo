"""GEMV aggregation — port of M2NDP-public examples/benchmarks/gemv/aggregation.

Reference kernel:

    KERNELBODY:
    vsetvli t0, a0, e32, m1
    li x1, spad_addr
    ld x2, (x1)             ; the accumulator
    vle32.v v1, (ADDR)      ; this µthread's chunk of the partial products
    add x2, x2, OFFSET
    vmset.m v0
    vid.v v2, v0
    vsll.vi v2, v2, 2       ; lane index -> byte offset
    vamoaddei32.v x0, (x2), v2, v1

The last step of a distributed GEMV: every NDP unit produced a partial vector
and they are summed in place. The combine is a vector atomic rather than a
load-add-store because the units write the same accumulator concurrently --
the same reason `spmv` uses one.
"""

from std.sys import argv, size_of
from std.random import random_float64, seed

from m2ndp import (
    PACKET,
    NDPTask,
    PooledRange,
    atomic_add_indexed,
    global_uthread_id,
    launch_parallel,
)
from m2ndp_host import cxl_alloc

comptime W = PACKET // size_of[Float32]()   # lanes in one packet


@fieldwise_init
struct AggregationParams(Movable):
    var partial: UnsafePointer[Float32, MutAnyOrigin]
    var acc: UnsafePointer[Float32, MutAnyOrigin]


struct GemvAggregation(NDPTask):
    comptime Params = AggregationParams

    @staticmethod
    def body():
        ref p = GemvAggregation.params[]
        var i = global_uthread_id() * W
        var v = p.partial.load[width=W](i)

        # Lane `k` lands at `acc[i + k]`, so the offsets are the lane indices
        # scaled to bytes -- `vid.v` + `vsll.vi` in the reference.
        var offsets = SIMD[DType.int32, W](0)
        comptime for lane in range(W):
            offsets[lane] = Int32(lane * size_of[Float32]())
        _ = atomic_add_indexed(p.acc + i, offsets, v)

    @staticmethod
    def device_main():
        launch_parallel[GemvAggregation.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh gemv_aggregation
#
# The accumulator starts at a value the host picks, so a run that dropped the
# combine and merely stored would be caught.


def main() raises:
    if GemvAggregation.emit_ir_if_asked():
        return

    var n = W * 64 * 8

    var partial = cxl_alloc[Float32](n)
    var acc = cxl_alloc[Float32](n)
    var expect = List[Float32](length=n, fill=0)

    seed(0)
    for i in range(n):
        partial[i] = Float32(random_float64(-1.0, 1.0))
        acc[i] = Float32(random_float64(-1.0, 1.0))
        expect[i] = acc[i] + partial[i]

    var rc = GemvAggregation.launch(
        PooledRange.over(partial, n), AggregationParams(partial, acc)
    )
    if rc != 0:
        print("[host] gemv_aggregation failed, exit", rc)
        return

    for i in range(n):
        if acc[i] != expect[i]:
            print("[host] wrong at", i, ":", acc[i], "expected", expect[i])
            return
    print("[host] gemv_aggregation ok")
