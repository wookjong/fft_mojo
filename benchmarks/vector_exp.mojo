"""Vector exponent — port of M2NDP-public examples/benchmarks/exponent.

Reference kernel:

    KERNELBODY:
    vsetvli 0, 0, e32, m1, 0
    li x1, spad_addr
    ld x1, (x1)
    add x1, x1, OFFSET
    vle32.v v1, (ADDR)
    vfexp.v v2, v1          ; one instruction
    vse32.v v2, (x1)

`vfexp.v` is an M²NDP instruction, and the vendor extension in this repo does
not carry it: the extension covers the atomics, which are what the workloads
here need from it. So the exponent is the stdlib's -- a polynomial over the
same vector -- and the kernel comes out as arithmetic rather than one
instruction. Adding `vfexp` to the extension would close the gap; nothing in
the workload would change.
"""

from std.sys import argv, size_of
from std.math import exp
from std.random import random_float64, seed

from m2ndp import PACKET, NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import Pool

comptime W = PACKET // size_of[Float32]()   # lanes in one packet


@fieldwise_init
struct ExpParams(Movable):
    var input: UnsafePointer[Float32, MutAnyOrigin]
    var output: UnsafePointer[Float32, MutAnyOrigin]


struct VectorExp(NDPTask):
    comptime Params = ExpParams

    @staticmethod
    def body():
        ref p = VectorExp.params[]
        var i = global_uthread_id() * W
        p.output.store(i, exp(p.input.load[width=W](i)))

    @staticmethod
    def device_main():
        launch_parallel[VectorExp.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh vector_exp
#
# Checked against the host's own exp within a tolerance: the device's is a
# different implementation, so demanding the same bits would be testing that
# two polynomials agree exactly rather than that the kernel ran.


def main() raises:
    if VectorExp.emit_ir_if_asked():
        return

    var n = W * 64 * 8

    var pool = Pool()
    var input = pool.alloc[Float32](n)
    var output = pool.alloc[Float32](n)

    seed(0)
    for i in range(n):
        input[i] = Float32(random_float64(-2.0, 2.0))

    var rc = VectorExp.launch(
        pool, PooledRange.over(input, n), ExpParams(input, output)
    )
    if rc != 0:
        print("[host] vector_exp failed, exit", rc)
        return

    for i in range(n):
        var want = exp(input[i])
        var err = abs(output[i] - want)
        if err > 1e-5 * abs(want) + 1e-6:
            print("[host] wrong at", i, ":", output[i], "expected", want)
            return
    print("[host] vector_exp ok")
