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

The three kernels, the scratchpad they share and the `device_main` that
launches them are one struct: a task is a unit and its parts are not
separately meaningful.

`bins` is at struct level because a comptime member is evaluated once and
shared, keeping all three kernels on the same addrspace(3) global; calling
`scratchpad()` in each would mint a fresh symbol per call site.
"""

from std.sys import argv, size_of
from std.random import random_si64, seed

from m2ndp import (
    PACKET,
    NDPTask,
    launch_parallel,
    launch_serial,
    PooledRange,
    global_uthread_id,
    local_uthread_id,
    group_size,
    atomic_add,
    atomic_add_indexed,
    scratchpad,
)
from m2ndp_host import Pool

comptime BINS = 256
comptime UNROLL = PACKET // size_of[Int32]()   # samples in one packet


@fieldwise_init
struct HistogramParams(Movable):
    """The task's parameters, declared once for both sides. `main` builds one
    of these and the kernels read it."""

    var samples: UnsafePointer[Int32, MutAnyOrigin]
    var out_hist: UnsafePointer[Int32, MutAnyOrigin]


struct Histogram(NDPTask):
    comptime Params = HistogramParams

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
    def body():
        """KERNELBODY: tally this µthread's samples into the core-local bins.

        One indexed vector atomic over the chunk, as in the reference: every
        lane hits its own bin.
        """
        var base = global_uthread_id() * UNROLL
        var chunk = (Histogram.params[].samples + base).load[width=UNROLL]()
        _ = atomic_add_indexed(
            Histogram.bins, chunk * 4, SIMD[DType.int32, UNROLL](1)
        )

    @staticmethod
    def finalize():
        """FINALIZER: fold this core's bins into the global histogram."""
        var i = local_uthread_id()
        while i < BINS:
            _ = atomic_add(Histogram.params[].out_hist + i, Histogram.bins[i])
            i += group_size()

    @staticmethod
    def device_main():
        """The task, as the device runs it.

        The body is `parallel`: one µthread per chunk of samples. The
        initializer and finalizer walk this core's bins rather than the data,
        so they are `serial` -- one µthread per core, striding by
        `group_size()`.

        Correctness rests on the launches being synchronous: the bins must be
        zero before the first tally and complete before the fold, and a launch
        boundary is the only synchronization point there is.
        """
        launch_serial[Histogram.initialize]()
        launch_parallel[Histogram.body]()
        launch_serial[Histogram.finalize]()


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh histogram
#
# The host folds its own histogram to check against. Running it at several core
# counts is what tests the scratchpad-per-core claim: the answer must not
# depend on how many there are.


def main() raises:
    if Histogram.emit_ir_if_asked():
        return

    # One packet is UNROLL samples; the count has to divide over the cores.
    var n = UNROLL * 64 * 8

    var pool = Pool()
    var samples = pool.alloc[Int32](n)
    var hist = pool.alloc[Int32](BINS)
    var expect = List[Int32](length=BINS, fill=0)

    seed(0)
    for i in range(n):
        # Deliberately narrow, so bins collide and the atomic has to hold up.
        var s = Int(random_si64(0, 63))
        samples[i] = Int32(s)
        expect[s] += 1

    var rc = Histogram.launch(
        pool, PooledRange.over(samples, n), HistogramParams(samples, hist)
    )
    if rc != 0:
        print("[host] histogram failed, exit", rc)
        return

    for i in range(BINS):
        if hist[i] != expect[i]:
            print("[host] wrong at bin", i, ":", hist[i], "expected", expect[i])
            return
    print("[host] histogram ok")
