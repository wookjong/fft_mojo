"""CSR SpMV — M²NDP 성격에 맞는 메모리 바운드 워크로드.

indirect access(x[col_idx[k]]) + 스크래치패드 + 그룹 배리어 +
tree reduction을 모두 포함한다. 그룹 하나가 행 하나를 담당.
"""

from m2ndp import (
    uthread_id,
    group_id,
    group_size,
    group_barrier,
    scratchpad,
    m2ndp_target,
)
from std.gpu.host.compile import _compile_code


def spmv_row(values: UnsafePointer[Float32, MutAnyOrigin],
             col_idx: UnsafePointer[Int32, MutAnyOrigin],
             x: UnsafePointer[Float32, MutAnyOrigin],
             row_ptr: UnsafePointer[Int32, MutAnyOrigin],
             y: UnsafePointer[Float32, MutAnyOrigin]):
    var tile = scratchpad[DType.float32]()
    var tid = uthread_id()
    var row = group_id()
    var start = Int(row_ptr[row])
    var end = Int(row_ptr[row + 1])

    # 각 µthread가 nonzero를 나눠 맡아 부분합 (indirect access)
    var acc = Float32(0)
    var k = start + tid
    while k < end:
        acc += values[k] * x[Int(col_idx[k])]
        k += group_size()
    tile[tid] = acc
    group_barrier()

    # 스크래치패드에서 그룹 단위 tree reduction
    var stride = group_size() // 2
    while stride > 0:
        if tid < stride:
            tile[tid] = tile[tid] + tile[tid + stride]
        group_barrier()
        stride //= 2

    if tid == 0:
        y[row] = tile[0]


def main():
    comptime t = m2ndp_target()
    print(_compile_code[spmv_row, emission_kind="llvm", target=t]().asm)
