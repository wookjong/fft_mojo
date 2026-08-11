"""GEMV — port of M2NDP-public examples/benchmarks/gemv.

Reference kernel, in outline:

    vle16.v v1, (x14)               ; the input vector, staged per core
    vsetvli t0, a0, e32, m2
    vmv.v.i v3, 0                   ; an fp32 accumulator
    vle16.v v5..v20, (x2)           ; sixteen rows of fp16 weights
    vfmv.f.s f{i}, v1               ; input[i]
    vfwmacc.vf v3, v{i+4}, f{i}     ; widening multiply-accumulate
    vslide1down.vx v1, v1, x0
    vamoaddei32.v x0, (x12), v4, v3 ; combine into the output

A matrix-vector product split across the NDP units: fp16 weights, fp32
accumulation, and the partial sums combined with the vector atomic -- which is
`gemv_aggregation` as its own kernel.

One packet of weights per µthread rather than the reference's sixteen. The
sixteen rows are register blocking: it amortises one atomic over sixteen
multiply-accumulates, and costs sixteen live vector registers to do it. The
arithmetic and the combine are the same either way.
"""

from std.sys import argv, size_of
from std.random import random_si64, seed

from m2ndp import (
    VECTOR_WIDTH,
    NDPTask,
    PooledRange,
    atomic_add_indexed,
    global_uthread_id,
    launch_parallel,
)
from m2ndp_host import cxl_alloc

comptime W = VECTOR_WIDTH // size_of[Float16]()   # weight lanes in one vector
comptime COLS = 256                         # output length
comptime ROWS = 256                         # input length
comptime TILES = COLS // W                  # packets across one weight row


@fieldwise_init
struct GemvParams(Movable):
    var weight: UnsafePointer[Float16, MutAnyOrigin]   # ROWS x COLS
    var input: UnsafePointer[Float16, MutAnyOrigin]    # ROWS
    var output: UnsafePointer[Float32, MutAnyOrigin]   # COLS


struct Gemv(NDPTask):
    comptime Params = GemvParams

    @staticmethod
    def body():
        ref p = Gemv.params[]
        var u = global_uthread_id()
        var row = u // TILES
        var col = (u % TILES) * W

        # One row's worth of this µthread's columns, scaled by that row's
        # input element -- the widening multiply of the reference.
        var w = p.weight.load[width=W](row * COLS + col)
        var contribution = w.cast[DType.float32]() * p.input[row].cast[
            DType.float32
        ]()

        var offsets = SIMD[DType.int32, W](0)
        comptime for lane in range(W):
            offsets[lane] = Int32(lane * size_of[Float32]())
        _ = atomic_add_indexed(p.output + col, offsets, contribution)

    @staticmethod
    def device_main():
        launch_parallel[Gemv.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh gemv
#
# The range is the weight matrix: a µthread owns a packet of it, which is what
# makes the atomic combine necessary -- every row lands on the same output.


def main() raises:
    if Gemv.emit_ir_if_asked():
        return

    var weight = cxl_alloc[Float16](ROWS * COLS)
    var input = cxl_alloc[Float16](ROWS)
    var output = cxl_alloc[Float32](COLS)

    seed(0)
    for i in range(ROWS * COLS):
        weight[i] = Float16(random_si64(-8, 8)) / 16
    for i in range(ROWS):
        input[i] = Float16(random_si64(-8, 8)) / 16

    var rc = Gemv.launch(
        PooledRange.over(weight, ROWS * COLS),
        GemvParams(weight, input, output)
    )
    if rc != 0:
        print("[host] gemv failed, exit", rc)
        return

    for c in range(COLS):
        var want = Float32(0)
        for r in range(ROWS):
            want += weight[r * COLS + c].cast[DType.float32]() * input[
                r
            ].cast[DType.float32]()
        if abs(output[c] - want) > 1e-3 * abs(want) + 1e-3:
            print("[host] wrong at", c, ":", output[c], "expected", want)
            return
    print("[host] gemv ok")
