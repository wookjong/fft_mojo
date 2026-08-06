"""device_main-only vector add through scratchpad.

device_main reads the task's params (via spad_addr -- it cannot index the params
scratchpad global directly), adds A+B into a scratchpad staging buffer, and
copies that to C. Exercises device_main reaching both params and scratchpad, and
computing, with no kernel launch.
"""
from std.random import random_si64, seed
from m2ndp import NDPTask, scratchpad, spad_addr, PooledRange
from m2ndp_host import cxl_alloc

comptime N = 64


@fieldwise_init
struct DevmainVaddParams(Movable):
    var a: UnsafePointer[Int32, MutAnyOrigin]
    var b: UnsafePointer[Int32, MutAnyOrigin]
    var c: UnsafePointer[Int32, MutAnyOrigin]


struct DevmainVadd(NDPTask):
    comptime Params = DevmainVaddParams
    comptime tmp = scratchpad[N, Int32, name="devmain_vadd_tmp"]()

    @staticmethod
    def device_main():
        # device_main reads its params out of scratchpad through spad_addr.
        ref p = spad_addr(DevmainVadd.params, 0)[]
        var stage = spad_addr(DevmainVadd.tmp, 0)
        for i in range(N):
            stage.store(i, p.a.load(i) + p.b.load(i))   # A+B -> scratchpad
        for i in range(N):
            p.c.store(i, stage.load(i))                 # scratchpad -> C


def main() raises:
    if DevmainVadd.emit_ir_if_asked():
        return

    var a = cxl_alloc[Int32](N)
    var b = cxl_alloc[Int32](N)
    var c = cxl_alloc[Int32](N)
    var expect = List[Int32](length=N, fill=0)
    seed(0)
    for i in range(N):
        a[i] = Int32(random_si64(-1000, 999))
        b[i] = Int32(random_si64(-1000, 999))
        expect[i] = a[i] + b[i]

    var rc = DevmainVadd.launch(
        PooledRange.over(a, N), DevmainVaddParams(a, b, c)
    )
    if rc != 0:
        print("[host] devmain_vadd failed, exit", rc)
        return

    for i in range(N):
        if c[i] != expect[i]:
            print("[host] devmain_vadd mismatch at", i, ":", c[i], "vs", expect[i])
            return
    print("[host] devmain_vadd ok")
