"""Device-main serial console — device_main prints to the host stdout.

No CXL buffers, no kernels: device_main writes through the UART transmit
register (a DeviceConsole) so its output lands on the host stdout, reusing the
standard formatting. It exercises the controller's UART device register.
"""

from m2ndp import NDPTask, DeviceConsole


@fieldwise_init
struct DevmainConsoleParams(Movable):
    var out: UnsafePointer[Int32, MutAnyOrigin]


struct DevmainConsole(NDPTask):
    comptime Params = DevmainConsoleParams

    @staticmethod
    def device_main():
        var con = DeviceConsole()
        con.write("hello from device_main\n")
        var sum = 0
        for i in range(1, 5):
            sum += i
        con.write("sum(1..4) = ", sum, "\n")


def main() raises:
    _ = DevmainConsole.emit_ir_if_asked()
