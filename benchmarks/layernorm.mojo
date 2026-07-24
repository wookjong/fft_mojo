"""Layer normalization — port of M2NDP-public examples/benchmarks/layerNorm32.

Two kernels. The first accumulates both moments at once:

    vle32.v v1, (ADDR)
    vfmul.vv v4, v1, v1                 ; x^2 alongside x
    vamoaddei32.v x0, (x1), v6, v1      ; E[x] , in the scratchpad
    vamoaddei32.v x0, (x1), v6, v4      ; E[x^2]
    FINALIZER:
    vfredosum.vs ...                    ; fold this core's lanes
    famoadd.w x0, f1, (x30)             ; and add them to memory

the second rescales, with the reciprocal standard deviation and the shifted
mean already folded into two constants:

    vfmsub.vf v1, f1, v2                ; x / std - mean / std

Both moments come out of one pass over the input, which is why the kernel
carries `x` and `x^2` together rather than reading the input twice.
"""

from std.sys import argv, size_of
from std.math import sqrt
from std.random import random_float64, seed

from m2ndp import (
    PACKET,
    NDPTask,
    PooledRange,
    atomic_add,
    global_uthread_id,
    launch_parallel,
)
from m2ndp_host import Pool

comptime W = PACKET // size_of[Float32]()   # lanes in one packet
comptime EPS = Float32(1e-5)


@fieldwise_init
struct LayerNormParams(Movable):
    var data: UnsafePointer[Float32, MutAnyOrigin]
    var sums: UnsafePointer[Float32, MutAnyOrigin]   # [sum(x), sum(x^2)]
    var count: Float32


struct LayerNorm(NDPTask):
    comptime Params = LayerNormParams

    @staticmethod
    def moments():
        ref p = LayerNorm.params[]
        var i = global_uthread_id() * W
        var x = p.data.load[width=W](i)
        _ = atomic_add(p.sums, x.reduce_add())
        _ = atomic_add(p.sums + 1, (x * x).reduce_add())

    @staticmethod
    def rescale():
        ref p = LayerNorm.params[]
        var i = global_uthread_id() * W
        var mean = p.sums[0] / p.count
        var variance = p.sums[1] / p.count - mean * mean
        var inv_std = 1 / sqrt(variance + EPS)
        # The reference form: one multiply-subtract per lane, the two
        # constants having been combined before the kernel sees them.
        p.data.store(i, p.data.load[width=W](i) * inv_std - mean * inv_std)

    @staticmethod
    def device_main():
        launch_parallel[LayerNorm.moments]()
        launch_parallel[LayerNorm.rescale]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh layernorm
#
# In place, as the reference is: the second kernel stores back over (ADDR).


def main() raises:
    if LayerNorm.emit_ir_if_asked():
        return

    var n = W * 64 * 8

    var pool = Pool()
    var data = pool.alloc[Float32](n)
    var sums = pool.alloc[Float32](2)
    var original = List[Float32](length=n, fill=0)

    seed(0)
    for i in range(n):
        data[i] = Float32(random_float64(-2.0, 2.0))
        original[i] = data[i]
    sums[0] = 0
    sums[1] = 0

    var rc = LayerNorm.launch(
        pool, PooledRange.over(data, n),
        LayerNormParams(data, sums, Float32(n))
    )
    if rc != 0:
        print("[host] layernorm failed, exit", rc)
        return

    var total = Float32(0)
    var square = Float32(0)
    for i in range(n):
        total += original[i]
        square += original[i] * original[i]
    var mean = total / Float32(n)
    var inv_std = 1 / sqrt(square / Float32(n) - mean * mean + EPS)

    for i in range(n):
        var want = (original[i] - mean) * inv_std
        if abs(data[i] - want) > 1e-3 * abs(want) + 1e-3:
            print("[host] wrong at", i, ":", data[i], "expected", want)
            return
    print("[host] layernorm ok")
