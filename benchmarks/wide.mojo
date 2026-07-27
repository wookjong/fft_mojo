"""Wide — port of M2NDP-public examples/benchmarks/narrow_wide/wide.

Reference kernel:

    KERNELBODY:
    vsetvli 0, 0, e16, m1, 0
    li x1, spad_addr
    ld x1, (x1)             ; the fp32 output
    muli x2, OFFSET, 2
    add x1, x1, x2
    vle16.v v2, (ADDR)      ; this µthread's chunk of the fp16 input
    vsetvli 0, 0, e32, m2, 0
    vfwcvt.f.f.v v4, v2
    vse32.v v4, (x1)

`narrow` the other way. Here the µthread is mapped over the input, that being
the narrow side, and the output offset is the doubled one.
"""

from std.sys import argv, size_of
from std.random import random_float64, seed

from m2ndp import PACKET, NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import cxl_alloc

comptime W = PACKET // size_of[Float16]()   # fp16 lanes in one packet


@fieldwise_init
struct WideParams(Movable):
    var input: UnsafePointer[Float16, MutAnyOrigin]
    var output: UnsafePointer[Float32, MutAnyOrigin]


struct Wide(NDPTask):
    comptime Params = WideParams

    @staticmethod
    def body():
        ref p = Wide.params[]
        var i = global_uthread_id() * W
        p.output.store(i, p.input.load[width=W](i).cast[DType.float32]())

    @staticmethod
    def device_main():
        launch_parallel[Wide.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh wide


def main() raises:
    if Wide.emit_ir_if_asked():
        return

    var n = W * 64 * 8

    var input = cxl_alloc[Float16](n)
    var output = cxl_alloc[Float32](n)

    seed(0)
    for i in range(n):
        input[i] = Float16(random_float64(-1.0, 1.0))

    var rc = Wide.launch(
        PooledRange.over(input, n), WideParams(input, output)
    )
    if rc != 0:
        print("[host] wide failed, exit", rc)
        return

    for i in range(n):
        var want = input[i].cast[DType.float32]()
        if output[i] != want:
            print("[host] wrong at", i, ":", output[i], "expected", want)
            return
    print("[host] wide ok")
