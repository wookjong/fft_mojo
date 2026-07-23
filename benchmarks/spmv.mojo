"""CSR SpMV — a memory-bound workload that fits M²NDP's character.

One group per row: its µthreads split the row's nonzeros, and the partial
sums are combined with an atomic add rather than a barrier and a tree
reduction. M²NDP has no barrier — µthreads are created and retired by
hardware FGMT, so there is no set to synchronize — which makes atomics the
way µthreads combine results.

Also exercises indirect access (x[col_idx[k]]).
"""

from m2ndp import (
    local_uthread_id,
    group_id,
    group_size,
    atomic_add,
)


@export
def spmv_row(values: UnsafePointer[Float32, MutAnyOrigin],
             col_idx: UnsafePointer[Int32, MutAnyOrigin],
             x: UnsafePointer[Float32, MutAnyOrigin],
             row_ptr: UnsafePointer[Int32, MutAnyOrigin],
             y: UnsafePointer[Float32, MutAnyOrigin]):
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


def main():
    pass
