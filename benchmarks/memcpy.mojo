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

from std.ffi import external_call
from std.sys import argv, size_of

from m2ndp import NDPTask, PooledRange, global_uthread_id
from m2ndp_host import Buffer

comptime W = 8   # int32 lanes per chunk


@fieldwise_init
struct MemcpyParams(Copyable, Movable):
    """What the host passes, in the order it hands the buffers over."""

    var src: UnsafePointer[Int32, MutAnyOrigin]
    var dst: UnsafePointer[Int32, MutAnyOrigin]


struct Memcpy(NDPTask):
    comptime packet = W * size_of[Int32]()

    @staticmethod
    def body(src: UnsafePointer[Int32, MutAnyOrigin],
             dst: UnsafePointer[Int32, MutAnyOrigin]):
        var i = global_uthread_id() * W
        dst.store(i, src.load[width=W](i))

    @staticmethod
    def device_main(params: UnsafePointer[NoneType, MutAnyOrigin]):
        var p = params.bitcast[MemcpyParams]()
        external_call["__m2ndp_launch_parallel", NoneType](
            Memcpy.body, Int(p[].src), Int(p[].dst), Int(0), Int(0), Int(0), Int(0)
        )


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

    var src = List[Int32](length=n, fill=0)
    var dst = List[Int32](length=n, fill=0)

    var state: Int = 20260724
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        src[i] = Int32((state >> 8) % 2000 - 1000)

    var rc = Memcpy.launch(
        PooledRange.over(src), Buffer.input(src), Buffer.output(dst)
    )
    if rc != 0:
        print("[host] memcpy failed, exit", rc)
        return

    for i in range(n):
        if dst[i] != src[i]:
            print("[host] wrong at", i, ":", dst[i], "expected", src[i])
            return
    print("[host] memcpy ok")
