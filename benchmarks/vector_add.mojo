"""Vector add — port of M2NDP-public examples/benchmarks/vector_add.

Reference kernel (hand-written M²NDP assembly):

    KERNELBODY:
    vsetvli 0, 0, e32, m1, 0
    li x1, spad_addr        ; kernel args arrive through the scratchpad
    li x2, spad_addr + 8
    ld x1, (x1)
    ld x2, (x2)
    add x1, x1, OFFSET      ; other arrays: their base + this µthread's offset
    add x2, x2, OFFSET
    vle32.v v1, (ADDR)      ; this µthread's own chunk
    vle32.v v2, (x1)
    vadd.vv v3, v1, v2
    vse32.v v3, (x2)

Written here with ordinary parameters and an index: arg marshalling and the
id-to-address mapping are the backend's job, not the workload's.
"""

from m2ndp import global_uthread_id

comptime W = 8   # int32 lanes per chunk


@export
def vector_add(a: UnsafePointer[Int32, MutAnyOrigin],
               b: UnsafePointer[Int32, MutAnyOrigin],
               c: UnsafePointer[Int32, MutAnyOrigin]):
    var i = global_uthread_id() * W
    c.store(i, a.load[width=W](i) + b.load[width=W](i))
