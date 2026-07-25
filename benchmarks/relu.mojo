"""ReLU — port of M2NDP-public examples/benchmarks/opt/activation.

Reference kernel:

    KERNELBODY:
    vsetvli 0, 0, e32, m1, 0
    li x3, spad_addr
    ld x3, (x3)             ; the output base
    add x3, x3, OFFSET
    vmv.v.i v2, 0
    vle32.v v1, (ADDR)      ; this µthread's chunk of the input
    fmv.w.x f0, x0
    vmsge.vf v0, v1, f0     ; lanes that are >= 0
    vmv.v.v v2, v1, v0      ; keep those, leave the rest zero
    vse32.v v2, (x3)

An OPT feed-forward activation: elementwise max(x, 0) over fp32.
"""

from std.sys import argv, size_of
from std.random import random_float64, seed

from m2ndp import PACKET, NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import Pool

comptime W = PACKET // size_of[Float32]()   # lanes in one packet


@fieldwise_init
struct ReluParams(Movable):
    var input: UnsafePointer[Float32, MutAnyOrigin]
    var output: UnsafePointer[Float32, MutAnyOrigin]


struct Relu(NDPTask):
    comptime Params = ReluParams

    @staticmethod
    def body():
        ref p = Relu.params[]
        var i = global_uthread_id() * W
        var v = p.input.load[width=W](i)
        p.output.store(i, v.ge(0).select(v, SIMD[DType.float32, W](0)))

    @staticmethod
    def device_main():
        launch_parallel[Relu.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh relu


def main() raises:
    if Relu.emit_ir_if_asked():
        return

    var n = W * 64 * 8

    var pool = Pool()
    var input = pool.alloc[Float32](n)
    var output = pool.alloc[Float32](n)

    seed(0)
    for i in range(n):
        input[i] = Float32(random_float64(-1.0, 1.0))

    var rc = Relu.launch(
        pool, PooledRange.over(input, n), ReluParams(input, output)
    )
    if rc != 0:
        print("[host] relu failed, exit", rc)
        return

    for i in range(n):
        var want = input[i] if input[i] > 0 else Float32(0)
        if output[i] != want:
            print("[host] wrong at", i, ":", output[i], "expected", want)
            return
    print("[host] relu ok")
