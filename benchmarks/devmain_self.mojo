"""Self-contained device-main — the controller computes on its own stack.

No CXL buffers, no scratchpad params: device_main fills a small stack array,
runs it through a non-inlined helper, and folds the result back into the array
so the work can't be optimised away. It exercises the controller's own
execution -- stack, a call, arithmetic -- launching nothing.
"""

from m2ndp import NDPTask


@fieldwise_init
struct DevmainSelfParams(Movable):
    var out: UnsafePointer[Int32, MutAnyOrigin]


struct DevmainSelf(NDPTask):
    comptime Params = DevmainSelfParams

    @staticmethod
    @no_inline
    def reduce(buf: UnsafePointer[Int32, MutAnyOrigin], n: Int) -> Int32:
        var s = Int32(0)
        for i in range(n):
            s += buf.load(i)
        return s

    @staticmethod
    def device_main():
        var buf = InlineArray[Int32, 64](fill=0)
        var p = buf.unsafe_ptr()
        for i in range(64):
            p.store(i, Int32(i))
        var total = DevmainSelf.reduce(p, 64)
        # Fold the result back so neither the fill nor the call is dead.
        p.store(0, total)


def main() raises:
    _ = DevmainSelf.emit_ir_if_asked()
