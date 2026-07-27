"""Narrow — port of M2NDP-public examples/benchmarks/narrow_wide/narrow.

Reference kernel:

    KERNELBODY:
    vsetvli 0, 0, e32, m2, 0
    li x1, spad_addr
    ld x1, (x1)             ; the fp32 input
    muli x2, OFFSET, 2      ; twice as far in, its elements being twice as wide
    add x1, x1, x2
    vle32.v v2, (x1)
    vsetvli 0, 0, e16, m1, 0
    vfncvt.f.f.w v5, v2     ; fp32 -> fp16
    vse32.v v5, (ADDR)

A precision conversion. The µthread is mapped over the *output* -- the
reference sets base_addr to the output, not the input -- so its own packet is
the narrow one and the input is reached by doubling the offset.
"""

from std.sys import argv, size_of
from std.random import random_float64, seed

from m2ndp import PACKET, NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import cxl_alloc

comptime W = PACKET // size_of[Float16]()   # fp16 lanes in one packet


@fieldwise_init
struct NarrowParams(Movable):
    var input: UnsafePointer[Float32, MutAnyOrigin]
    var output: UnsafePointer[Float16, MutAnyOrigin]


struct Narrow(NDPTask):
    comptime Params = NarrowParams

    @staticmethod
    def body():
        ref p = Narrow.params[]
        var i = global_uthread_id() * W
        p.output.store(i, p.input.load[width=W](i).cast[DType.float16]())

    @staticmethod
    def device_main():
        launch_parallel[Narrow.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh narrow
#
# The range is the output, since that is what a µthread owns a packet of.


def main() raises:
    if Narrow.emit_ir_if_asked():
        return

    var n = W * 64 * 8

    var input = cxl_alloc[Float32](n)
    var output = cxl_alloc[Float16](n)

    seed(0)
    for i in range(n):
        input[i] = Float32(random_float64(-1.0, 1.0))

    var rc = Narrow.launch(
        PooledRange.over(output, n), NarrowParams(input, output)
    )
    if rc != 0:
        print("[host] narrow failed, exit", rc)
        return

    for i in range(n):
        var want = input[i].cast[DType.float16]()
        if output[i] != want:
            print("[host] wrong at", i, ":", output[i], "expected", want)
            return
    print("[host] narrow ok")
