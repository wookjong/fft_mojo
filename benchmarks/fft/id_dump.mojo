"""Ad-hoc identity-probe kernel -- NOT part of the FFT planner/codegen.

Dumps global_uthread_id()/local_uthread_id()/group_id()/num_groups() for
every microthread in a launch, one row per microthread, into DRAM buffers
the host prints back out. Used to empirically pin down the real hardware's
global-id -> (physical unit, local id) mapping for the cooperative
workers_per_fft=8 correctness investigation -- see fft_plan_cooperative.py's
`exclude_full_interleave_chunk` and docs/persistent_vs_cooperative_findings.md.

Does not touch simulator source; this is a pure verification/probe kernel.
"""

from std.sys import argv, size_of

from m2ndp import (
    VECTOR_WIDTH,
    NDPTask,
    PooledRange,
    global_uthread_id,
    local_uthread_id,
    group_id,
    launch_parallel,
)
from m2ndp_host import cxl_alloc

comptime W = VECTOR_WIDTH // size_of[Int32]()

comptime THREADS = 1040
"""4 full 256-wide interleave rounds (8*32) plus a partial 5th, to see both
steady-state wraparound and a ragged tail."""


@fieldwise_init
struct IdDumpParams(Movable):
    var out_global: UnsafePointer[Int32, MutAnyOrigin]
    var out_local: UnsafePointer[Int32, MutAnyOrigin]
    var out_group: UnsafePointer[Int32, MutAnyOrigin]


struct IdDump(NDPTask):
    comptime Params = IdDumpParams

    @staticmethod
    def body():
        ref p = IdDump.params[]
        var g = global_uthread_id()
        p.out_global.store(g, Int32(g))
        p.out_local.store(g, Int32(local_uthread_id()))
        p.out_group.store(g, Int32(group_id()))

    @staticmethod
    def device_main():
        launch_parallel[IdDump.body]()


def main() raises:
    if IdDump.emit_ir_if_asked():
        return

    var dummy_n = THREADS * W
    var dummy = cxl_alloc[Int32](dummy_n)

    var out_global = cxl_alloc[Int32](THREADS)
    var out_local = cxl_alloc[Int32](THREADS)
    var out_group = cxl_alloc[Int32](THREADS)
    for i in range(THREADS):
        out_global[i] = Int32(-1)
        out_local[i] = Int32(-1)
        out_group[i] = Int32(-1)

    var rc = IdDump.launch(
        PooledRange.over(dummy, dummy_n),
        IdDumpParams(out_global, out_local, out_group),
    )
    if rc != 0:
        print("[host] id_dump failed, exit", rc)
        return

    print("global,local,group")
    for i in range(THREADS):
        print(out_global[i], ",", out_local[i], ",", out_group[i])
