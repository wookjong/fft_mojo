"""CSR SpMV — a memory-bound workload that fits M²NDP's character.

One group per row: its µthreads split the row's nonzeros, and the partial
sums are combined with an atomic add rather than a barrier and a tree
reduction. M²NDP has no barrier — µthreads are created and retired by
hardware FGMT, so there is no set to synchronize — which makes atomics the
way µthreads combine results.

Also exercises indirect access (x[col_idx[k]]).
"""

from std.sys import argv, size_of
from std.random import random_float64, random_si64, seed

from m2ndp import (
    NDPTask,
    launch_parallel,
    PooledRange,
    local_uthread_id,
    group_id,
    group_size,
    atomic_add,
)
from m2ndp_host import Config, Pool


@fieldwise_init
struct SpmvParams(Movable):
    """The task's parameters, declared once for both sides."""

    var values: UnsafePointer[Float32, MutAnyOrigin]
    var col_idx: UnsafePointer[Int32, MutAnyOrigin]
    var x: UnsafePointer[Float32, MutAnyOrigin]
    var row_ptr: UnsafePointer[Int32, MutAnyOrigin]
    var y: UnsafePointer[Float32, MutAnyOrigin]


struct Spmv(NDPTask):
    """One group per row.

    The one workload the launch model does not fit cleanly. The kernel keys
    off `group_id()` and a group is a core, so a run computes as many rows as
    there are cores -- the host has to set `cores` to the row count. What is
    missing is a launch that can say "spawn G groups of N".

    `packet` therefore says nothing about data: the kernel indexes by group and
    slot, so the range only settles the µthread count.
    """

    comptime Params = SpmvParams
    comptime packet = size_of[Float32]()

    @staticmethod
    def body():
        var values = Spmv.params()[].values
        var col_idx = Spmv.params()[].col_idx
        var x = Spmv.params()[].x
        var row_ptr = Spmv.params()[].row_ptr
        var y = Spmv.params()[].y

        var row = group_id()
        var tid = local_uthread_id()
        var start = Int(row_ptr[row])
        var end = Int(row_ptr[row + 1])

        # Each µthread takes a strided slice of the row's nonzeros.
        var acc = Float32(0)
        var k = start + tid
        while k < end:
            acc += values[k] * x[Int(col_idx[k])]   # indirect access
            k += group_size()

        # Combine without synchronizing.
        _ = atomic_add(y + row, acc)

    @staticmethod
    def device_main(params: UnsafePointer[SpmvParams, MutAnyOrigin]):
        launch_parallel[Spmv.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh spmv 4 1        # 4 cores, so 4 rows
#
# The one workload the launch model does not fit cleanly. The kernel keys off
# `group_id()`, a group is a core here, so a run computes exactly as many rows
# as the machine has cores. That is why this is the only benchmark that reads
# config/machine.conf: it cannot size its matrix without knowing the hardware.
# See the note on `struct Spmv`.
#
# Two other things follow from that. The range passed to the launch is not the
# length of any buffer: the kernel never indexes by `global_uthread_id()`, so
# the range exists only to say how many microthreads there are, which is rows
# times the microthreads that split each row. And the check allows a tolerance,
# because the partial sums are combined with an atomic in whatever order the
# microthreads happen to retire, and float addition is not associative.


comptime PER_ROW = 8    # microthreads splitting one row's nonzeros
comptime NNZ_PER_ROW = 5
comptime NCOLS = 32


def main() raises:
    if Spmv.emit_ir_if_asked():
        return

    # The one workload that has to know the machine. Its kernel keys off
    # group_id(), a group is a core, so the matrix has exactly as many rows as
    # the hardware has cores -- there is no way to ask for more. Every other
    # benchmark is free of this and never reads the config.
    var rows = Config.load().get("cores")
    var nnz = rows * NNZ_PER_ROW

    var pool = Pool()
    var values = pool.alloc[Float32](nnz)
    var col_idx = pool.alloc[Int32](nnz)
    var x = pool.alloc[Float32](NCOLS)
    var row_ptr = pool.alloc[Int32](rows + 1)
    var y = pool.alloc[Float32](rows)

    seed(0)
    for i in range(NCOLS):
        x[i] = Float32(random_float64(-10.0, 10.0))
    for k in range(nnz):
        values[k] = Float32(random_float64(-10.0, 10.0))
        col_idx[k] = Int32(random_si64(0, NCOLS - 1))
    for r in range(rows + 1):
        row_ptr[r] = Int32(r * NNZ_PER_ROW)

    # The range says how many microthreads, nothing about data: rows x PER_ROW
    # of them, one packet each. Hence of_bytes rather than over(): no buffer's
    # length is the right number here.
    var threads = rows * PER_ROW

    var rc = Spmv.launch(
        pool, PooledRange.of_bytes(values, threads * Spmv.packet),
        SpmvParams(values, col_idx, x, row_ptr, y),
    )
    if rc != 0:
        print("[host] spmv failed, exit", rc)
        return

    for r in range(rows):
        var want = Float32(0)
        for k in range(Int(row_ptr[r]), Int(row_ptr[r + 1])):
            want += values[k] * x[Int(col_idx[k])]
        var diff = y[r] - want
        if diff < Float32(-0.01) or diff > Float32(0.01):
            print("[host] wrong at row", r, ":", y[r], "expected", want)
            return
    print("[host] spmv ok, rows =", rows)
