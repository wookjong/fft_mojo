"""Memcpy — port of M2NDP-public examples/benchmarks/memcpy.

Reference kernel:

    KERNELBODY:
    vsetvli 0, 0, e32, m1, 0
    li x1, spad_addr
    ld x1, (x1)             ; arg0 = destination
    add x1, x1, OFFSET
    vle32.v v1, (ADDR)      ; load this µthread's chunk of the source
    vse32.v v1, (x1)        ; store it at the matching offset in the dest

The smallest kernel in the suite: one vector load, one vector store, no
arithmetic. Useful as a floor for what the interface costs.
"""

from m2ndp import global_uthread_id

comptime W = 8   # int32 lanes per chunk


@export
def memcpy(src: UnsafePointer[Int32, MutAnyOrigin],
           dst: UnsafePointer[Int32, MutAnyOrigin]):
    var i = global_uthread_id() * W
    dst.store(i, src.load[width=W](i))


def main():
    pass
