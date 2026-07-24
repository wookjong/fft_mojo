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

from std.sys import argv

from m2ndp import NDPTask, PooledRange, global_uthread_id, launch_parallel
from m2ndp_host import Pool

comptime W = 32   # uint8 lanes per chunk


@fieldwise_init
struct MemsetParams(Movable):
    """What the host passes. `value` arrives as a one-element buffer rather
    than a scalar: the parameter block the host fills is addresses, so a
    scalar has to be somewhere to have an address. The kernel dereferences it
    where it needs it, keeping the byte a byte the whole way."""

    var dst: UnsafePointer[UInt8, MutAnyOrigin]
    var value: UnsafePointer[UInt8, MutAnyOrigin]


struct Memset(NDPTask):
    comptime Params = MemsetParams
    comptime packet = W   # W uint8 lanes is W bytes

    @staticmethod
    def body():
        var i = global_uthread_id() * W
        ref p = Memset.params[]
        p.dst.store(i, SIMD[DType.uint8, W](p.value[0]))

    @staticmethod
    def device_main():
        launch_parallel[Memset.body]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh memset
#
# The only workload here whose range is an *output* buffer -- there is no input
# to map over, just memory to fill -- and the only one that hands a kernel a
# scalar. The parameter block the host fills is addresses, so the fill byte
# travels as a one-element buffer and `device_main` reads it before passing the
# value on.


def main() raises:
    if Memset.emit_ir_if_asked():
        return

    var n = W * 64 * 8          # bytes; one packet is W of them

    var pool = Pool()
    var dst = pool.alloc[UInt8](n)
    # The scalar goes in the pool too: a parameter is an address, so a byte
    # the kernel reads has to be somewhere the device can address.
    var value = pool.alloc[UInt8](1)
    value[0] = 0xAB

    var rc = Memset.launch(
        pool, PooledRange.over(dst, n), MemsetParams(dst, value)
    )
    if rc != 0:
        print("[host] memset failed, exit", rc)
        return

    for i in range(n):
        if dst[i] != 0xAB:
            print("[host] wrong at", i, ":", dst[i], "expected 171")
            return
    print("[host] memset ok")
