"""Vector add — port of M2NDP-public examples/benchmarks/vector_add.

Reference kernel (hand-written M²NDP assembly):

    KERNELBODY:
    vsetvli 0, 0, e32, m1, 0
    li x1, spad_addr        ; kernel args arrive through the scratchpad
    li x2, spad_addr + 8
    ld x1, (x1)
    ld x2, (x2)
    add x1, x1, OFFSET      ; other arrays: their base + this µthread's offset
    add x2, x2, OFFSET
    vle32.v v1, (ADDR)      ; this µthread's own chunk
    vle32.v v2, (x1)
    vadd.vv v3, v1, v2
    vse32.v v3, (x2)

Written here with ordinary parameters and an index: arg marshalling and the
id-to-address mapping are the backend's job, not the workload's.

The task is the struct: the kernel and the `device_main` that launches it are
one unit, and neither is separately meaningful.
"""

from std.ffi import external_call
from std.sys import argv, size_of

from m2ndp import NDPTask, PooledRange, global_uthread_id
from m2ndp_host import In, Out

comptime W = 8   # int32 lanes per chunk, one packet's worth


@fieldwise_init
struct VectorAddParams(Movable):
    """What the host passes. The layout is the interface: `main` below hands
    the buffers over in this order, and nothing checks that the two agree."""

    var a: In[Int32]
    var b: In[Int32]
    var c: Out[Int32]


struct VectorAdd(NDPTask):
    # One packet is one chunk: W int32 lanes. Written in terms of W so it
    # cannot drift from what the kernel indexes by.
    comptime Params = VectorAddParams
    comptime packet = W * size_of[Int32]()

    @staticmethod
    def body():
        var i = global_uthread_id() * W
        VectorAdd.params()[].c.ptr.store(
            i, VectorAdd.params()[].a.ptr.load[width=W](i) + VectorAdd.params()[].b.ptr.load[width=W](i)
        )

    @staticmethod
    def device_main(params: UnsafePointer[VectorAddParams, MutAnyOrigin]):
        """The task, as the device runs it: one kernel over the range.

        `device_main` decides the sequence of kernels, so a workload of
        several is one function here rather than a table somewhere else.
        Launches are synchronous -- the call returns when every µthread of
        that kernel has retired -- so the order written is the order that
        happens.

        No size is passed. How many µthreads there are was settled when the
        task was launched over its range; a kernel launch says what to run and
        with what, and nothing about how much.

        Every kernel takes the same one argument: this same parameter block,
        passed straight on. `external_call` allows one signature per symbol
        name, and a kernel taking five buffers -- spmv's -- has to reach the
        same launcher entry as one taking none; handing all of them the block
        is what makes that one signature, with no padding to count and no
        positions to line up. A kernel reads the fields it wants by name.
        See docs/INTERFACE.md.
        """
        external_call["__m2ndp_launch_parallel", NoneType](VectorAdd.body)


# ------------------------------------------------------------ the host
#
#     ./scripts/host-run.sh vector_add
#
# The host owns the data now. It fills the inputs, launches the task, and checks
# the output against a result it computes itself -- so the answer being checked
# does not come from the same place as the answer being produced.
#
# The launch is one line:
#
#     VectorAdd.launch(PooledRange.over(a), VectorAddParams(a, b, c))
#
# Naming the task is the whole of it. The device code is compiled at that point,
# for the target VectorAdd declares. Nothing here says what hardware it runs on:
# the runtime reads config/machine.conf, so running the same program on a
# different machine is a change to that file.


def main() raises:
    if VectorAdd.emit_ir_if_asked():
        return

    # One packet is W lanes; the length has to be a whole number of them,
    # and to divide over whatever core count the machine config names.
    var n = W * 64 * 8

    var a = List[Int32](length=n, fill=0)
    var b = List[Int32](length=n, fill=0)
    var c = List[Int32](length=n, fill=0)
    var expect = List[Int32](length=n, fill=0)

    var state: Int = 20260724
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        a[i] = Int32((state >> 8) % 2000 - 1000)
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        b[i] = Int32((state >> 8) % 2000 - 1000)
        expect[i] = a[i] + b[i]

    var rc = VectorAdd.launch(
        PooledRange.over(a),
        VectorAddParams(a, b, c),
    )
    if rc != 0:
        print("[host] vector_add failed, exit", rc)
        return

    for i in range(n):
        if c[i] != expect[i]:
            print("[host] wrong at", i, ":", c[i], "expected", expect[i])
            return
    print("[host] vector_add ok")
