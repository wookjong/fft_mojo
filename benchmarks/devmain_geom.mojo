"""device_main reads run geometry through the device-side primitives.

Exercises num_groups() and spad_capacity() read on the device (from the runtime
config, delivered over MMIO) and a peer scratchpad write: device_main writes a
marker into each group's scratchpad through spad_addr, then a kernel on each group
reads its own marker back and reports it, so the round trip is observed.

The host checks the geometry values against the config and that every group saw
the marker device_main placed in its scratchpad.
"""
from m2ndp import (
    NDPTask,
    scratchpad,
    spad_addr,
    num_groups,
    spad_capacity,
    launch_parallel,
    group_id,
    global_uthread_id,
    PooledRange,
)
from m2ndp_host import cxl_alloc, Config


@fieldwise_init
struct DevmainGeomParams(Movable):
    var geom: UnsafePointer[Int32, MutAnyOrigin]  # [num_groups, spad_capacity]
    var seen: UnsafePointer[Int32, MutAnyOrigin]  # per-group: marker the kernel read back


struct DevmainGeom(NDPTask):
    comptime Params = DevmainGeomParams
    comptime mark = scratchpad[1, Int32, name="devmain_geom_mark"]()

    @staticmethod
    def report():
        # Each group's microthreads read the marker device_main wrote into this
        # group's scratchpad and store it out at the group's slot.
        ref p = DevmainGeom.params[]
        p.seen.store(group_id(), DevmainGeom.mark[0])

    @staticmethod
    def device_main():
        ref p = spad_addr(DevmainGeom.params, 0)[]
        var g = num_groups()
        p.geom.store(0, Int32(g))
        p.geom.store(1, Int32(spad_capacity()))
        # Place a distinct marker in each group's scratchpad from the controller.
        for i in range(g):
            spad_addr(DevmainGeom.mark, i).store(0, Int32(0x100 + i))
        launch_parallel[DevmainGeom.report]()  # each group reads its marker back


def main() raises:
    if DevmainGeom.emit_ir_if_asked():
        return

    var cfg = Config.load()
    var groups = cfg.get("num_ndp_units")
    var cap = cfg.get("spad_size")

    var geom = cxl_alloc[Int32](2)
    var seen = cxl_alloc[Int32](groups)
    geom[0] = -1
    geom[1] = -1
    for i in range(groups):
        seen[i] = -1

    # The range must reach every group. A group owns a stride (256 B) at a time,
    # so cover num_groups strides; W(=8) int per packet, 8 packets per stride.
    var n = groups * 64  # groups * (stride 256B / 4B)
    var scratch = cxl_alloc[Int32](n)

    var rc = DevmainGeom.launch(
        PooledRange.over(scratch, n), DevmainGeomParams(geom, seen)
    )
    if rc != 0:
        print("[host] devmain_geom failed, exit", rc)
        return

    if geom[0] != Int32(groups):
        print("[host] devmain_geom num_groups mismatch:", geom[0], "vs", groups)
        return
    if geom[1] != Int32(cap):
        print("[host] devmain_geom spad_capacity mismatch:", geom[1], "vs", cap)
        return
    for i in range(groups):
        if seen[i] != Int32(0x100 + i):
            print("[host] devmain_geom group", i, "marker mismatch:", seen[i])
            return
    print("[host] devmain_geom ok")
