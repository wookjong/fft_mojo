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

from std.sys import argv, size_of
from std.random import random_si64, seed

from m2ndp import NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import Pool

comptime W = 8   # int32 lanes per chunk


@fieldwise_init
struct MemcpyParams(Movable):
    """The task's parameters, declared once for both sides."""

    var src: UnsafePointer[Int32, MutAnyOrigin]
    var dst: UnsafePointer[Int32, MutAnyOrigin]


struct Memcpy(NDPTask):
    comptime Params = MemcpyParams
    comptime packet = W * size_of[Int32]()

    @staticmethod
    def body():
        var i = global_uthread_id() * W
        ref p = Memcpy.params[]
        p.dst.store(i, p.src.load[width=W](i))

    @staticmethod
    def device_main():
        launch_parallel[Memcpy.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh memcpy
#
# The floor of the suite: one vector load and one vector store per microthread,
# so what it really checks is that the launch path itself moves the right bytes
# to the right places.


def main() raises:
    if Memcpy.emit_ir_if_asked():
        return

    var n = W * 64 * 8

    var pool = Pool()
    var src = pool.alloc[Int32](n)
    var dst = pool.alloc[Int32](n)

    seed(0)
    for i in range(n):
        src[i] = Int32(random_si64(-1000, 999))

    var rc = Memcpy.launch(
        pool, PooledRange.over(src, n), MemcpyParams(src, dst)
    )
    if rc != 0:
        print("[host] memcpy failed, exit", rc)
        return

    for i in range(n):
        if dst[i] != src[i]:
            print("[host] wrong at", i, ":", dst[i], "expected", src[i])
            return
    print("[host] memcpy ok")
