"""CSR SpMV — a memory-bound workload that fits M²NDP's character.

One group per row: its µthreads split the row's nonzeros, and the partial
sums are combined with an atomic add rather than a barrier and a tree
reduction. M²NDP has no barrier — µthreads are created and retired by
hardware FGMT, so there is no set to synchronize — which makes atomics the
way µthreads combine results.

Also exercises indirect access (x[col_idx[k]]).
"""

from std.sys import argv, size_of

from m2ndp import (
    NDPTask,
    launch_parallel,
    PooledRange,
    local_uthread_id,
    group_id,
    group_size,
    atomic_add,
)
from m2ndp_host import In, Out, Config


@fieldwise_init
struct SpmvParams(Movable):
    """The task's parameters, declared once for both sides."""

    var values: In[Float32]
    var col_idx: In[Int32]
    var x: In[Float32]
    var row_ptr: In[Int32]
    var y: Out[Float32]


struct Spmv(NDPTask):
    """One group per row.

    This is the one workload whose shape the launch model does not fit
    cleanly. The kernel keys off `group_id()`, and in this model a group is a
    core, so a run computes exactly as many rows as there are cores -- the
    host has to set `cores` to the row count. A launch that could say "spawn
    G groups of N" independently of the core count is what the model is
    missing; until then this is the arrangement that expresses the workload.

    `packet` therefore says nothing about data here. The kernel indexes by
    group and slot, never by `global_uthread_id()`, so the range only settles
    how many µthreads there are: rows x µthreads-per-row.
    """

    comptime Params = SpmvParams
    comptime packet = size_of[Float32]()

    @staticmethod
    def body():
        var values = Spmv.params()[].values.ptr
        var col_idx = Spmv.params()[].col_idx.ptr
        var x = Spmv.params()[].x.ptr
        var row_ptr = Spmv.params()[].row_ptr.ptr
        var y = Spmv.params()[].y.ptr

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

    var values = List[Float32](length=nnz, fill=0)
    var col_idx = List[Int32](length=nnz, fill=0)
    var x = List[Float32](length=NCOLS, fill=0)
    var row_ptr = List[Int32](length=rows + 1, fill=0)
    var y = List[Float32](length=rows, fill=0)

    var state: Int = 20260724
    for i in range(NCOLS):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        x[i] = Float32((state >> 8) % 20 - 10)
    for k in range(nnz):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        values[k] = Float32((state >> 8) % 20 - 10)
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        col_idx[k] = Int32((state >> 8) % NCOLS)
    for r in range(rows + 1):
        row_ptr[r] = Int32(r * NNZ_PER_ROW)

    # The range says how many microthreads, nothing about data: rows x PER_ROW
    # of them, one packet each. Hence of_bytes rather than over(): no buffer's
    # length is the right number here.
    var threads = rows * PER_ROW

    var rc = Spmv.launch(
        PooledRange.of_bytes(threads * Spmv.packet),
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
