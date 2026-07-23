"""Memset — port of M2NDP-public examples/benchmarks/memset.

Reference kernel:

    KERNELBODY:
    vsetvli 0, 0, e8, m1, 0
    li x1, spad_addr
    ld x1, (x1)             ; arg0 = the fill byte
    vmv.v.x v1, x1          ; splat it across the vector
    vse8.v v1, (ADDR)       ; store to this µthread's chunk

Note the reference works at e8 and splats a scalar; `SIMD[uint8, W](value)`
is the same splat.
"""

from m2ndp import global_uthread_id

comptime W = 32   # uint8 lanes per chunk


@export
def memset(dst: UnsafePointer[UInt8, MutAnyOrigin], value: UInt8):
    var i = global_uthread_id() * W
    dst.store(i, SIMD[DType.uint8, W](value))


def main():
    pass
