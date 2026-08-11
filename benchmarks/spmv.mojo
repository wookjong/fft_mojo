"""CSR SpMV — a port of M2NDP-public examples/benchmarks/spmv.

One row to a µthread: it walks the row's nonzeros itself and stores the answer
once, so there is nothing to combine. The reference kernel does the same, from
its OFFSET; here the row is the µthread's index.

Also exercises indirect access (x[col_idx[k]]).
"""

from std.random import random_float64, random_si64, seed

from m2ndp import (
    VECTOR_WIDTH,
    NDPTask,
    PooledRange,
    global_uthread_id,
    launch_parallel,
)
from m2ndp_host import cxl_alloc


@fieldwise_init
struct SpmvParams(Movable):
    """The task's parameters, declared once for both sides."""

    var values: UnsafePointer[Float32, MutAnyOrigin]
    var col_idx: UnsafePointer[Int32, MutAnyOrigin]
    var x: UnsafePointer[Float32, MutAnyOrigin]
    var row_ptr: UnsafePointer[Int32, MutAnyOrigin]
    var y: UnsafePointer[Float32, MutAnyOrigin]


struct Spmv(NDPTask):
    comptime Params = SpmvParams

    @staticmethod
    def body():
        ref p = Spmv.params[]
        var values = p.values
        var col_idx = p.col_idx
        var x = p.x

        var row = global_uthread_id()
        var acc = Float32(0)
        for k in range(Int(p.row_ptr[row]), Int(p.row_ptr[row + 1])):
            acc += values[k] * x[Int(col_idx[k])]   # indirect access
        p.y[row] = acc

    @staticmethod
    def device_main():
        launch_parallel[Spmv.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh spmv
#
# The range is one packet per row, so there are as many µthreads as rows. It is
# not the length of any buffer -- hence of_bytes -- and it starts at the row
# pointers, which is where the reference anchors it too (base_addr =
# input_rows_addr, bound = num_rows * packet_size).


comptime ROWS = 64
comptime NNZ_PER_ROW = 5
comptime NCOLS = 32


def main() raises:
    if Spmv.emit_ir_if_asked():
        return

    comptime nnz = ROWS * NNZ_PER_ROW

    var row_ptr = cxl_alloc[Int32](ROWS + 1)
    var values = cxl_alloc[Float32](nnz)
    var col_idx = cxl_alloc[Int32](nnz)
    var x = cxl_alloc[Float32](NCOLS)
    var y = cxl_alloc[Float32](ROWS)

    seed(0)
    for i in range(NCOLS):
        x[i] = Float32(random_float64(-10.0, 10.0))
    for k in range(nnz):
        values[k] = Float32(random_float64(-10.0, 10.0))
        col_idx[k] = Int32(random_si64(0, NCOLS - 1))
    for r in range(ROWS + 1):
        row_ptr[r] = Int32(r * NNZ_PER_ROW)

    var rc = Spmv.launch(
        PooledRange.of_bytes(row_ptr, ROWS * VECTOR_WIDTH),
        SpmvParams(values, col_idx, x, row_ptr, y),
    )
    if rc != 0:
        print("[host] spmv failed, exit", rc)
        return

    for r in range(ROWS):
        var want = Float32(0)
        for k in range(Int(row_ptr[r]), Int(row_ptr[r + 1])):
            want += values[k] * x[Int(col_idx[k])]
        if y[r] != want:
            print("[host] wrong at row", r, ":", y[r], "expected", want)
            return
    print("[host] spmv ok, rows =", ROWS)
