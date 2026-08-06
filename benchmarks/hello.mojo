"""Hello, world -- the smallest workload there is.

No kernels and no data: `device_main` runs on the controller and prints. It is
the shortest path from a checkout to something running on the simulator, and
what the README's getting started walks through.

    ./scripts/host-run.sh hello

`DeviceConsole` writes to the controller's UART, which the runtime streams to
the host stdout. `write` takes any `Writable`, so a value formats the same way
`print` does on the host. Only `device_main` has one -- a kernel runs on the
cores, which own no UART.
"""

from m2ndp import NDPTask, DeviceConsole, PooledRange
from m2ndp_host import cxl_alloc


@fieldwise_init
struct HelloParams(Movable):
    """Every task declares its parameters. This one has nothing to say, so it
    carries the one buffer the launch is measured over and never reads it."""

    var scratch: UnsafePointer[Int32, MutAnyOrigin]


struct Hello(NDPTask):
    comptime Params = HelloParams

    @staticmethod
    def device_main():
        var con = DeviceConsole()
        con.write("Hello, world!\n")
        con.write("2 + 2 = ", 2 + 2, "\n")


def main() raises:
    if Hello.emit_ir_if_asked():
        return

    # A launch is over a range of the pool, which is what settles the
    # microthread count. Nothing here launches a kernel, so one packet is
    # enough -- but a range there must be.
    var scratch = cxl_alloc[Int32](8)

    var rc = Hello.launch(PooledRange.over(scratch, 8), HelloParams(scratch))
    if rc != 0:
        print("[host] hello failed, exit", rc)
        return
    print("[host] hello ok")
