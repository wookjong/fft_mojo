"""Softmax — port of M2NDP-public examples/benchmarks/softmax (kernel0-2).

Three kernels, in order, because each needs the one before it to have
finished everywhere:

    kernel0   vamomaxei32.v   the maximum, for numerical stability
    kernel1   vfexp + vfredosum + vamoaddei32.v   exp(x - max), and its sum
    kernel2   vfmul.vf        divide through by that sum

The reference stages each reduction in the core's scratchpad and folds it into
memory in a FINALIZER, the way `histogram` does. Here the combine goes to
memory directly -- one atomic per µthread instead of one per core -- which is
the same answer with more traffic, and keeps the three-kernel shape the point
of this workload legible.

The exponent is `m2ndp.exp`, the `vfexp` the reference shows; see `vector_exp`.
"""

from std.sys import argv, size_of
from std.math import exp as host_exp
from std.random import random_float64, seed

from m2ndp import (
    PACKET,
    NDPTask,
    PooledRange,
    atomic_add,
    atomic_max,
    exp,
    global_uthread_id,
    launch_parallel,
)
from m2ndp_host import cxl_alloc

comptime W = PACKET // size_of[Float32]()   # lanes in one packet


@fieldwise_init
struct SoftmaxParams(Movable):
    var input: UnsafePointer[Float32, MutAnyOrigin]
    var output: UnsafePointer[Float32, MutAnyOrigin]
    var maximum: UnsafePointer[Float32, MutAnyOrigin]
    var total: UnsafePointer[Float32, MutAnyOrigin]


struct Softmax(NDPTask):
    comptime Params = SoftmaxParams

    @staticmethod
    def reduce_max():
        ref p = Softmax.params[]
        var i = global_uthread_id() * W
        atomic_max(p.maximum, p.input.load[width=W](i).reduce_max())

    @staticmethod
    def exponentiate():
        ref p = Softmax.params[]
        var i = global_uthread_id() * W
        var e = exp(p.input.load[width=W](i) - p.maximum[0])
        p.output.store(i, e)
        _ = atomic_add(p.total, e.reduce_add())

    @staticmethod
    def normalize():
        ref p = Softmax.params[]
        var i = global_uthread_id() * W
        p.output.store(i, p.output.load[width=W](i) / p.total[0])

    @staticmethod
    def device_main():
        """Launches are synchronous, so each kernel sees the whole of the one
        before it -- which is what makes a reduction expressible without a
        barrier."""
        launch_parallel[Softmax.reduce_max]()
        launch_parallel[Softmax.exponentiate]()
        launch_parallel[Softmax.normalize]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh softmax
#
# `maximum` starts at the lowest float the reduction could raise, and `total`
# at zero: both are accumulators the device folds into.


def main() raises:
    if Softmax.emit_ir_if_asked():
        return

    var n = W * 64 * 8

    var input = cxl_alloc[Float32](n)
    var output = cxl_alloc[Float32](n)
    var maximum = cxl_alloc[Float32](1)
    var total = cxl_alloc[Float32](1)

    seed(0)
    for i in range(n):
        input[i] = Float32(random_float64(-4.0, 4.0))
    maximum[0] = Float32.MIN
    total[0] = 0

    var rc = Softmax.launch(
        PooledRange.over(input, n),
        SoftmaxParams(input, output, maximum, total)
    )
    if rc != 0:
        print("[host] softmax failed, exit", rc)
        return

    var host_max = input[0]
    for i in range(n):
        if input[i] > host_max:
            host_max = input[i]
    var host_total = Float32(0)
    for i in range(n):
        host_total += host_exp(input[i] - host_max)

    for i in range(n):
        var want = host_exp(input[i] - host_max) / host_total
        if abs(output[i] - want) > 1e-4 * abs(want) + 1e-9:
            print("[host] wrong at", i, ":", output[i], "expected", want)
            return
    print("[host] softmax ok")
