"""Histogram — port of M2NDP-public examples/benchmarks/histogram.

Reference kernel, abridged (hand-written M²NDP assembly):

    INITIALIZER:
    li x1, spad_addr + packet_size    ; per-core bin array
    vmv.v.i v1, 0
    .LOOP0                            ; zero the core-local bins
    vse32.v v1, (x4)
    bne NDPID, x0, .SKIP1             ; core 0 also zeros the output
    vse32.v v1, (x4)
    .SKIP1

    KERNELBODY:
    vle32.v v{u}, (x6)                ; load a chunk of samples
    vmul.vi v{u}, v{u}, 4             ; sample -> byte offset into bins
    vamoaddei32.v x0, (x1), v{u}, v30 ; bins[sample] += 1, indexed vector atomic

    FINALIZER:
    .LOOP1                            ; flush core-local bins to the output
    vle32.v v2, (x1)
    vamoaddei32.v x0, (x6), v1, v2

Everything the task is made of lives in one struct: the three kernels, the
scratchpad they share, and the `device_main` that launches them. That
grouping is the point -- a task is a unit, and its parts are not separately
meaningful. Conforming to `NDPTask` is the whole interface to the host; the
entry point it is launched through comes with it.

`bins` is declared at struct level rather than inside a kernel because a
comptime member is evaluated once and shared, which keeps all three kernels
on the same addrspace(3) global. Calling `scratchpad()` separately in each
would instead mint a fresh symbol per call site.

The reference uses `vamoaddei32.v` — an indexed *vector* atomic. Neither
`pop.atomic.rmw` nor LLVM's `atomicrmw` accepts a vector operand, so this
falls back to one scalar atomic per sample; see the atomics note in
src/m2ndp.mojo. Reaching that instruction needs a dedicated intrinsic.
"""

from std.ffi import external_call
from std.sys import argv, size_of

from m2ndp import (
    NDPTask,
    PooledRange,
    global_uthread_id,
    local_uthread_id,
    group_size,
    atomic_add,
    atomic_add_indexed,
    scratchpad,
)
from m2ndp_host import Buffer

comptime BINS = 256
comptime UNROLL = 16


@fieldwise_init
struct HistogramParams(Copyable, Movable):
    """What the host passes. The layout is the interface: `main` below hands
    the buffers over in this order, and nothing checks that the two agree."""

    var samples: UnsafePointer[Int32, MutAnyOrigin]
    var out_hist: UnsafePointer[Int32, MutAnyOrigin]


struct Histogram(NDPTask):
    # One packet is UNROLL samples. In terms of UNROLL so it stays in step
    # with what the body loads.
    comptime packet = UNROLL * size_of[Int32]()

    # Declared once, shared by all three kernels.
    comptime bins = scratchpad[BINS, Int32, name="hist_bins"]()

    @staticmethod
    def initialize():
        """INITIALIZER: zero this core's bins."""
        var i = local_uthread_id()
        while i < BINS:
            Histogram.bins[i] = 0
            i += group_size()

    @staticmethod
    def body(samples: UnsafePointer[Int32, MutAnyOrigin]):
        """KERNELBODY: tally this µthread's samples into the core-local bins.

        One indexed vector atomic over the whole chunk, as in the reference:
        load the samples, scale them to byte offsets, and let every lane hit
        its own bin.
        """
        var base = global_uthread_id() * UNROLL
        var chunk = (samples + base).load[width=UNROLL]()
        _ = atomic_add_indexed(
            Histogram.bins, chunk * 4, SIMD[DType.int32, UNROLL](1)
        )

    @staticmethod
    def finalize(out_hist: UnsafePointer[Int32, MutAnyOrigin]):
        """FINALIZER: fold this core's bins into the global histogram."""
        var i = local_uthread_id()
        while i < BINS:
            _ = atomic_add(out_hist + i, Histogram.bins[i])
            i += group_size()

    @staticmethod
    def device_main(params: UnsafePointer[NoneType, MutAnyOrigin]):
        """The task, as the device runs it.

        The three kernels are the reason the launch kind matters. The body is
        `parallel`: one µthread per chunk of samples, spread over the cores by
        whatever mapping the hardware uses. The initializer and the finalizer
        are `serial`: they walk this core's bins rather than the data, so what
        they need is one µthread on each core -- the work is per-core, not
        per-packet. Striding by `group_size()` from `local_uthread_id()` then
        covers the whole bin array, since that µthread is alone on its core.

        Correctness rests on the launches being synchronous. The bins must be
        zero before the first tally and complete before the fold, and there is
        no barrier to arrange that inside a kernel; a launch boundary is the
        only synchronization point the model has.
        """
        var p = params.bitcast[HistogramParams]()
        external_call["__m2ndp_launch_serial", NoneType](
            Histogram.initialize, Int(0), Int(0), Int(0), Int(0), Int(0), Int(0)
        )
        external_call["__m2ndp_launch_parallel", NoneType](
            Histogram.body, Int(p[].samples), Int(0), Int(0), Int(0), Int(0), Int(0)
        )
        external_call["__m2ndp_launch_serial", NoneType](
            Histogram.finalize, Int(p[].out_hist), Int(0), Int(0), Int(0), Int(0), Int(0)
        )


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh histogram
#
# The task the whole contract runs through: three kernels sharing a per-core
# scratchpad across launches, and an indexed vector atomic. The host fills the
# samples, launches, and folds its own histogram to check against -- and running
# it against several machine descriptions is what tests that the answer does not
# depend on the core count, which is the scratchpad-per-core claim.


def main() raises:
    if Histogram.emit_ir_if_asked():
        return

    # One packet is UNROLL samples; the count has to divide over the cores.
    var n = UNROLL * 64 * 8

    var samples = List[Int32](length=n, fill=0)
    var hist = List[Int32](length=BINS, fill=0)
    var expect = List[Int32](length=BINS, fill=0)

    var state: Int = 20260724
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        # Deliberately narrow, so bins collide and the atomic has to hold up.
        var s = Int((state >> 8) % 64)
        samples[i] = Int32(s)
        expect[s] += 1

    var rc = Histogram.launch(
        PooledRange.over(samples),
        Buffer.input(samples),
        Buffer.output(hist),
    )
    if rc != 0:
        print("[host] histogram failed, exit", rc)
        return

    for i in range(BINS):
        if hist[i] != expect[i]:
            print("[host] wrong at bin", i, ":", hist[i], "expected", expect[i])
            return
    print("[host] histogram ok")
