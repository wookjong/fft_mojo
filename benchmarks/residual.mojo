"""Residual add — port of M2NDP-public examples/benchmarks/opt/residual.

Reference kernel:

    KERNELBODY:
    vsetvli 0, 0, e32, m1, 0
    li x3, spad_addr
    li x4, spad_addr + 8
    ld x3, (x3)             ; the residual
    ld x4, (x4)             ; the output
    add x3, x3, OFFSET
    add x4, x4, OFFSET
    vle32.v v1, (ADDR)
    vle32.v v2, (x3)
    vadd.vv v3, v1, v2
    vse32.v v3, (x4)

A transformer block's skip connection: the sublayer's output plus its input.
"""

from std.sys import argv, size_of
from std.random import random_float64, seed

from m2ndp import PACKET, NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import Pool

comptime W = PACKET // size_of[Float32]()   # lanes in one packet


@fieldwise_init
struct ResidualParams(Movable):
    var hidden: UnsafePointer[Float32, MutAnyOrigin]
    var residual: UnsafePointer[Float32, MutAnyOrigin]
    var output: UnsafePointer[Float32, MutAnyOrigin]


struct Residual(NDPTask):
    comptime Params = ResidualParams

    @staticmethod
    def body():
        ref p = Residual.params[]
        var i = global_uthread_id() * W
        p.output.store(
            i, p.hidden.load[width=W](i) + p.residual.load[width=W](i)
        )

    @staticmethod
    def device_main():
        launch_parallel[Residual.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh residual


def main() raises:
    if Residual.emit_ir_if_asked():
        return

    var n = W * 64 * 8

    var pool = Pool()
    var hidden = pool.alloc[Float32](n)
    var residual = pool.alloc[Float32](n)
    var output = pool.alloc[Float32](n)
    var expect = List[Float32](length=n, fill=0)

    seed(0)
    for i in range(n):
        hidden[i] = Float32(random_float64(-1.0, 1.0))
        residual[i] = Float32(random_float64(-1.0, 1.0))
        expect[i] = hidden[i] + residual[i]

    var rc = Residual.launch(
        pool, PooledRange.over(hidden, n),
        ResidualParams(hidden, residual, output)
    )
    if rc != 0:
        print("[host] residual failed, exit", rc)
        return

    for i in range(n):
        if output[i] != expect[i]:
            print("[host] wrong at", i, ":", output[i], "expected", expect[i])
            return
    print("[host] residual ok")
