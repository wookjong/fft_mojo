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

Written with an index instead: the id-to-address mapping is the backend's job,
not the workload's.
"""

from std.sys import argv, size_of
from std.random import random_si64, seed

from m2ndp import NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import Pool

comptime W = 8   # int32 lanes per chunk, one packet's worth


@fieldwise_init
struct VectorAddParams(Movable):
    """The task's parameters, declared once for both sides. `main` builds one
    of these and the kernel reads it."""

    var a: UnsafePointer[Int32, MutAnyOrigin]
    var b: UnsafePointer[Int32, MutAnyOrigin]
    var c: UnsafePointer[Int32, MutAnyOrigin]


struct VectorAdd(NDPTask):
    # One packet is one chunk: W int32 lanes. Written in terms of W so it
    # cannot drift from what the kernel indexes by.
    comptime Params = VectorAddParams
    comptime packet = W * size_of[Int32]()

    @staticmethod
    def body():
        ref p = VectorAdd.params()
        var i = global_uthread_id() * W
        p.c.store(i, p.a.load[width=W](i) + p.b.load[width=W](i))

    @staticmethod
    def device_main(params: UnsafePointer[VectorAddParams, MutAnyOrigin]):
        """The task, as the device runs it: one kernel over the range.

        Launches are synchronous, so the order written is the order that
        happens. A launch names a kernel and nothing else -- no size, and no
        arguments, a kernel reading the task's parameters from the scratchpad.
        See docs/INTERFACE.md.
        """
        launch_parallel[VectorAdd.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh vector_add
#
# The host fills the inputs and checks the output against a result it computes
# itself, so the two answers do not come from the same place.
#
#     VectorAdd.launch(PooledRange.over(a), VectorAddParams(a, b, c))
#
# Naming the task compiles it, for the target it declares. What hardware it runs
# on is config/machine.conf's business.


def main() raises:
    if VectorAdd.emit_ir_if_asked():
        return

    # One packet is W lanes; the length has to be a whole number of them,
    # and to divide over whatever core count the machine config names.
    var n = W * 64 * 8

    # The pool is memory the device shares, so the host writes its inputs
    # straight into it and reads the results back out of it.
    var pool = Pool()
    var a = pool.alloc[Int32](n)
    var b = pool.alloc[Int32](n)
    var c = pool.alloc[Int32](n)
    var expect = List[Int32](length=n, fill=0)

    seed(0)
    for i in range(n):
        a[i] = Int32(random_si64(-1000, 999))
        b[i] = Int32(random_si64(-1000, 999))
        expect[i] = a[i] + b[i]

    var rc = VectorAdd.launch(
        pool, PooledRange.over(a, n), VectorAddParams(a, b, c)
    )
    if rc != 0:
        print("[host] vector_add failed, exit", rc)
        return

    for i in range(n):
        if c[i] != expect[i]:
            print("[host] wrong at", i, ":", c[i], "expected", expect[i])
            return
    print("[host] vector_add ok")
