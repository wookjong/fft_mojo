"""device_main writes each group's scratchpad, then reads it back itself.

No kernel: device_main alone writes a distinct marker into every group's
scratchpad through spad_addr and reads each one back, so the round trip to a
unit's scratchpad is observed entirely from the controller. Isolates spad_addr's
peer write+read from any kernel/group_id mapping.
"""
from m2ndp import NDPTask, scratchpad, spad_addr, num_groups, PooledRange
from m2ndp_host import cxl_alloc, Config


@fieldwise_init
struct DevmainPeerParams(Movable):
    var seen: UnsafePointer[Int32, MutAnyOrigin]  # per-group, read back by device_main


struct DevmainPeer(NDPTask):
    comptime Params = DevmainPeerParams
    comptime cell = scratchpad[1, Int32, name="devmain_peer_cell"]()

    @staticmethod
    def device_main():
        ref p = spad_addr(DevmainPeer.params, 0)[]
        var g = num_groups()
        for i in range(g):
            spad_addr(DevmainPeer.cell, i).store(0, Int32(0x100 + i))  # write group i
        for i in range(g):
            p.seen.store(i, spad_addr(DevmainPeer.cell, i).load(0))    # read group i back


def main() raises:
    if DevmainPeer.emit_ir_if_asked():
        return

    var groups = Config.load().get("num_ndp_units")
    var seen = cxl_alloc[Int32](groups)
    for i in range(groups):
        seen[i] = -1

    var n = groups * 8
    var scratch = cxl_alloc[Int32](n)

    var rc = DevmainPeer.launch(
        PooledRange.over(scratch, n), DevmainPeerParams(seen)
    )
    if rc != 0:
        print("[host] devmain_peer failed, exit", rc)
        return

    for i in range(groups):
        if seen[i] != Int32(0x100 + i):
            print("[host] devmain_peer group", i, "mismatch:", seen[i], "vs", 0x100 + i)
            return
    print("[host] devmain_peer ok")
