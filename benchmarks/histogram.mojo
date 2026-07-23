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

Three phases of one kernel, so `Histogram.bins` is declared once at struct
level: a comptime member is evaluated once and shared, which keeps all three
functions on the same addrspace(3) global. Calling `scratchpad()` separately
in each function would instead mint a fresh symbol per call site.

The reference uses `vamoaddei32.v` — an indexed *vector* atomic. Neither
`pop.atomic.rmw` nor LLVM's `atomicrmw` accepts a vector operand, so this
falls back to one scalar atomic per sample; see the atomics note in
src/m2ndp.mojo. Reaching that instruction needs a dedicated intrinsic.
"""

from m2ndp import (
    global_uthread_id,
    local_uthread_id,
    group_size,
    atomic_add,
    scratchpad,
)

comptime BINS = 256
comptime UNROLL = 16


struct Histogram:
    # Declared once, shared by all three phases.
    comptime bins = scratchpad[BINS, Int32, name="hist_bins"]()


@export
def histogram_init():
    """INITIALIZER: zero this core's bins."""
    var i = local_uthread_id()
    while i < BINS:
        Histogram.bins[i] = 0
        i += group_size()


@export
def histogram_body(samples: UnsafePointer[Int32, MutAnyOrigin]):
    """KERNELBODY: tally this µthread's samples into the core-local bins."""
    var base = global_uthread_id() * UNROLL

    comptime for u in range(UNROLL):
        var bin = Int(samples[base + u])
        _ = atomic_add(Histogram.bins + bin, Int32(1))


@export
def histogram_final(out_hist: UnsafePointer[Int32, MutAnyOrigin]):
    """FINALIZER: fold this core's bins into the global histogram."""
    var i = local_uthread_id()
    while i < BINS:
        _ = atomic_add(out_hist + i, Histogram.bins[i])
        i += group_size()


def main():
    pass
