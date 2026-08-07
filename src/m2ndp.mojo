"""M²NDP PoC primitive library.

M²NDP operations are external symbol calls, which the backend maps onto real
instructions. The symbol names are the interface contract; see
docs/INTERFACE.md.

  __m2ndp_local_uthread_id()  -> i32   index within the group
  __m2ndp_global_uthread_id() -> i32   index across all cores
  __m2ndp_group_id()          -> i32   which group

Atomics are not symbols: they lower to LLVM `atomicrmw`. The scratchpad is not
one either -- it is a global in address space 3.

There is no barrier. µthreads are created and retired by hardware FGMT, so
there is no set to synchronize; the only synchronization point is a kernel
boundary, and launches are synchronous. What would need `__syncthreads()` on a
GPU has to be two kernels here.
"""

from std.atomic import Atomic, Ordering
from std.compile import compile_info
from std.sys import argv
from std.sys.info import CompilationTarget
from std.ffi import external_call
from std.memory import AddressSpace, UnsafePointer
from std.sys import size_of
from std.collections.string.string_slice import _get_kgen_string

# The host-side plumbing a launch needs. It knows nothing about tasks, which is
# what keeps the dependency one-way: the model reaches for the machinery, never
# the other way round.
from m2ndp_host import (
    Config,
    Toolchain,
    _add_export_alias,
    _align_masked_gathers,
    _getenv,
    _mktemp,
    _run,
    cxl_pool,
)


# ---------------------------------------------------------------- tasks
#
# A task is the unit the host launches: its kernels, the scratchpad they
# share, and the `device_main` that decides which of them runs in what order.
# Grouping them in one struct is not organisation -- the parts are not
# separately meaningful, and the compiler assigns scratchpad offsets per
# module on the assumption that one task owns it.
#
# What the host does is launch the task over a memory range. That range is
# what settles how many µthreads there are -- one per packet -- which is why
# no kernel launch inside `device_main` carries a size.


@fieldwise_init
struct PooledRange(Copyable, Movable):
    """The region of the memory pool a task is mapped over.

    The range settles how many microthreads there are: one per packet of it.
    Build one from the data it covers:

        PooledRange.over(samples)       # the whole of a buffer
        PooledRange.of_bytes(n)         # n bytes from the start of the pool
        PooledRange(base=off, size=n)   # part of a larger region
    """

    var base: Int
    """Byte offset into the pool where this task's work starts."""

    var size: Int
    """Bytes of it. Divided by the machine's packet, this is the microthread
    count."""

    @staticmethod
    def over[
        T: Copyable & Movable
    ](data: UnsafePointer[T, MutAnyOrigin], count: Int) -> PooledRange:
        """The whole of a buffer -- what most tasks run over. `base` is the
        buffer's own address, the pool being memory both sides address the
        same way."""
        return PooledRange(Int(data), count * size_of[T]())

    @staticmethod
    def of_bytes[
        T: Copyable & Movable
    ](at: UnsafePointer[T, MutAnyOrigin], n: Int) -> PooledRange:
        """`n` bytes from `at`, for a task whose range is not the length of
        any one buffer."""
        return PooledRange(Int(at), n)


comptime PACKET = 32
"""Bytes of a task's range one microthread is mapped to.

The hardware's granule, not a workload's: a kernel is written against it, so
it appears as a width here rather than a choice a task makes. The simulator
config carries the same number as `packet_size` and `launch` refuses a machine
that disagrees -- kernels compiled for one granule would silently get the wrong
microthread count on another.

A constant rather than read from the config because a SIMD width has to be
known at compile time, which a file read cannot be.
"""


@fieldwise_init
struct Machine(Copyable, Movable):
    """The NDP hardware a run is modelled on -- read from the simulator config,
    not the workload nor a launch argument. Only the packet granule reaches the
    build; the rest of the machine is the device's own config to read.
    """

    var packet: Int

    @staticmethod
    def from_config() raises -> Machine:
        """The machine the environment points at. See `Config` for where."""
        var packet = Config.load().get("packet_size")
        if packet != PACKET:
            # The kernels were compiled against PACKET. A machine with another
            # granule would take the same code and hand each microthread the
            # wrong slice, so say so rather than compute a wrong answer.
            raise Error(
                String("this build's kernels are compiled for packet ")
                + String(PACKET) + ", but the machine says " + String(packet)
            )
        return Machine(packet)


# --------------------------------------------------------------- launching
#
# The kernel is a *parameter*, not an argument: `external_call` takes a
# function only where it is named at the call site, and one passed as a runtime
# argument does not convert -- a declared function's type carries its name. As
# a parameter it keeps that name, and `materialize` hands it on. What comes out
# is `call void @__m2ndp_launch_parallel(ptr @body)`, which is what the backend
# reads to decide a function is a kernel.
#
# `F` is inferred and never written; `ImplicitlyDeletable` is what lets the
# materialized temporary be discarded.


@always_inline
def launch_parallel[F: ImplicitlyDeletable, //, kernel: F]():
    """Run `kernel` over the task's range: one microthread per packet of it.

        launch_parallel[Histogram.body]()

    Spread over the cores by the hardware's mapping, so the kernel keys off
    `global_uthread_id()`. Nothing is passed and no size: the range was settled
    when the task was launched.

    Returns once every microthread has retired, which is the model's only
    synchronization point.
    """
    external_call["__m2ndp_launch_parallel", NoneType](materialize[kernel]())


@always_inline
def launch_serial[F: ImplicitlyDeletable, //, kernel: F]():
    """Run `kernel` once on each core, for work that is per-core rather than
    per-packet.

        launch_serial[Histogram.initialize]()

    Zeroing this core's scratchpad, folding it back out again. That microthread
    is alone on its core, so it walks the whole of it.

    Synchronous, as `launch_parallel` is.
    """
    external_call["__m2ndp_launch_serial", NoneType](materialize[kernel]())


# ---------------------------------------------------------------- device console


@always_inline
def _uart_putc(c: UInt8):
    """Write one byte to the device UART transmit register (the controller
    streams it to the host stdout)."""
    external_call["__m2ndp_putc", NoneType](c)


@fieldwise_init
struct DeviceConsole(Writer):
    """A serial console for `device_main`, over the UART to the host stdout.
    `write` streams to it, reusing the standard formatting so any `Writable`
    works -- Int, Float, String:

        var con = DeviceConsole()
        con.write("launched group ", group_id(), "\n")

    Only `device_main` runs on the controller, which owns the UART; a kernel on
    the cores has no console. The formatting runs in a stack buffer (no device
    allocator); the one runtime symbol it references is defined in sim/device_rt.c.
    """

    def write_bytes(mut self, bytes: Span[Byte, _]):
        for b in bytes:
            _uart_putc(b)

    def write_string(mut self, string: StringSlice):
        self.write_bytes(string.as_bytes())


def _map_address_flags(machine: Machine, range_param: Int) raises -> String:
    """The llc flags that turn an index back into the hardware's mapping.

        M2NDP_MAP_ADDRESS=addr     mapped address where it fits (default)
        M2NDP_MAP_ADDRESS=offset   base + mapped offset only
        M2NDP_MAP_ADDRESS=off      leave indices as written

    A kernel indexes a parameter by the microthread's id, but the hardware
    already handed it the offset that index rebuilds and the address of its own
    chunk. `packet` is what the id scales by, so the backend needs it to
    recognize the stride. `addr` additionally names the parameter the range was
    taken over -- `range_param`, its byte offset in the block, or -1 if none is
    it -- so that parameter's accesses fold to the mapped address outright. See
    docs/INTERFACE.md.
    """
    var mode = _getenv("M2NDP_MAP_ADDRESS")
    if not mode:
        mode = String("addr")
    if mode != "off" and mode != "offset" and mode != "addr":
        raise Error(
            "M2NDP_MAP_ADDRESS must be off, offset or addr, not " + mode
        )
    if mode == "off":
        return String(" -m2ndp-map-address=off")

    var flags = (
        String(" -m2ndp-map-address=") + mode
        + " -m2ndp-packet=" + String(machine.packet)
    )
    if mode == "addr" and range_param >= 0:
        flags += " -m2ndp-range-param=" + String(range_param)
    return flags


def _dump(
    ir: String, ll: String, work: String, tc: Toolchain, flags: String
) raises:
    """Print the device code, if `M2NDP_DUMP` asks for it.

        M2NDP_DUMP=ir    the LLVM a launch hands to llc
        M2NDP_DUMP=asm   what llc makes of it
        M2NDP_DUMP=all   both

    A module is the task and nothing else -- its kernels, its device_main and
    the entry point -- so this is every function of it, in order. The asm is
    compiled with the same `flags` a launch uses, so it is what actually runs.
    """
    var want = _getenv("M2NDP_DUMP")
    if not want:
        return

    if want == "ir" or want == "all":
        print("──── llvm ────")
        print(ir)

    if want == "asm" or want == "all":
        var asm = work + "/task.s"
        if _run(
            tc.llc + " -mtriple=riscv64-unknown-elf -mattr=" + tc.features
            + flags + " " + ll + " -o " + asm
        ) == 0:
            print("──── riscv ────")
            with open(asm, "r") as f:
                print(f.read())


trait NDPTask:
    """What a task has to provide, and what it gets for free.

    Conforming is the whole interface: declare `Params` and `device_main`, and
    the task gains both the runtime entry point the host launches it through
    and the `launch` that reaches it. The granule its kernels are written
    against is not among them -- that is `PACKET`, the hardware's.

    `device_main` says which kernels run and in what order, through
    `launch_parallel` and `launch_serial`. A launch names a kernel and nothing
    else: a kernel takes no arguments and reads the task's parameters from the
    scratchpad with `Self.params`. The backend enforces that, and decides
    what is a kernel by seeing its address reach a launch.
    """

    comptime Params: Movable
    """This task's arguments, declared once for both sides of the launch.

    A struct of the task's own, one pointer per buffer:

        @fieldwise_init
        struct HistogramParams(Movable):
            var samples: UnsafePointer[Int32, MutAnyOrigin]
            var out_hist: UnsafePointer[Int32, MutAnyOrigin]

    Plain pointers, because the host and the device share the pool those
    addresses are in: there is nothing to transfer and so no direction to
    declare. The host allocates from the pool and fills the block; the kernels
    read the same declaration on the device. One declaration for both ends
    leaves them no order to disagree about.
    """

    comptime target: __mlir_type.`!kgen.target` = m2ndp_target()
    """Which machine this task is compiled for.

    Overriding it is the whole of what lowering the same workload onto another
    machine takes:

        struct VectorAdd(NDPTask):
            comptime target = some_other_target()

    It belongs to the task, not the toolchain: a build script deciding it would
    mean the same source meant different things depending on the invocation.
    """

    @staticmethod
    def device_main():
        """Which kernels run, in what order. One per task.

        Takes nothing: a task's parameters are `Self.params`, which the
        host fills and every kernel reads.
        """
        ...

    comptime params = scratchpad[
        1, Self.Params, name="__m2ndp_params", alignment=8
    ]()
    """This task's parameters, where the host puts them and a kernel reads them.

        var chunk = Histogram.params[].samples.load[width=W](i)

    A scratchpad global like any other the task declares, so an access is a
    constant offset from the base -- one instruction, the same as `bins`. The
    launcher writes the block here before running a kernel, which is why a
    kernel takes no arguments.

    Kernels only: the controller has no scratchpad of its own.
    """

    @export
    @staticmethod
    def __m2ndp_rt_launch_task(base: Int, size: Int):
        """The host's launch, arriving on the device.

        Controller code, so it may call and keep a stack. The range comes
        first: nothing can be launched until the microthread count is known.
        The parameters are not passed here -- the launcher already holds them
        and puts them where a kernel reads them.
        """
        # Which region holds the parameters, by name; the backend exports its
        # offset for the launcher and deletes the call.
        Self.params.declare_params()
        external_call["__m2ndp_set_task_range", NoneType](base, size)
        Self.device_main()

    # ------------------------------------------------------------ launching
    #
    # The two below run on the host, not the device. They are here rather than
    # in a free function because launching a task is something a task does --
    # it is the other half of `device_main`, and putting it anywhere else means
    # a workload's reader has to go looking for how it is run.
    #
    # A device build never reaches them: nothing in a compiled task calls
    # either, so neither is instantiated, and the host-only machinery they lean
    # on never has to exist on a core.

    @staticmethod
    def device_ir() -> String:
        """This task's device code, as LLVM IR, ready for our llc.

        Asking for the entry point gets everything it reaches, which is the
        whole task.

        IR rather than assembly: this is the frontend's own LLVM, which has
        never heard of the vendor extension, so none of the M²NDP lowering has
        happened. Assembly here would look finished and not be.

        One repair on the way out: the alignment the frontend drops from a
        masked gather (see `_align_masked_gathers`). Here rather than in
        `launch`, so a build and a launch compile the same text.
        """
        return _align_masked_gathers(
            String(
                compile_info[
                    Self.__m2ndp_rt_launch_task,
                    emission_kind="llvm",
                    target = Self.target,
                ]()
            )
        )

    @staticmethod
    def emit_ir_if_asked() raises -> Bool:
        """Print this task's device code and stop, if asked on the command line.

        `--emit-ir` is how the build gets at the IR that actually ships. A
        whole-module build cannot: a single-source workload has a host `main`
        in it, and compiling that for the device is neither possible nor
        wanted. Asking the task instead gets exactly what a launch compiles.

        Returns whether it printed, so a `main` can begin with

            if Spmv.emit_ir_if_asked(): return
        """
        var a = argv()
        if len(a) > 1 and String(a[1]) == "--emit-ir":
            var ir = Self.device_ir()
            # The launcher calls the entry by its unmangled name. Whether the
            # frontend emits that alongside the mangled definition depends on
            # what else the module holds, so add it here -- as `launch` does for
            # the copy it ships -- and every consumer of `--emit-ir` (the
            # controller-image build among them) links against it.
            var tmp = _mktemp() + "/emit.ll"
            with open(tmp, "w") as f:
                f.write(ir)
            _add_export_alias(ir, tmp)
            with open(tmp, "r") as f:
                print(f.read(), end="")
            return True
        return False

    @staticmethod
    def launch(region: PooledRange, ref params: Self.Params) raises -> Int:
        """Run this task over `region` of the CXL pool with this parameter block.

            _ = Histogram.launch(PooledRange.over(samples, n),
                                 HistogramParams(samples, hist))

        The same block `device_main` is handed: one declaration, so the two
        sides have no order to disagree about. Its fields are plain pointers
        into the pool, which host and device share -- nothing is transferred
        and no parameter carries a direction.

        The device code is compiled here for the target the task declares.
        `region` divided by the machine's packet is the microthread count, and
        the hardware comes from the simulator config (M2NDP_CONFIG).

        Returns the simulator's exit code: 0 finished, 2 launcher error, 3 a
        fault in the target.
        """
        var machine = Machine.from_config()
        var tc = Toolchain()
        var work = _mktemp()
        var pool = cxl_pool()

        # The block goes in the pool, where both sides can see it. Copied
        # bytewise rather than moved: taking the caller's would raise what
        # happens to it if a later step throws.
        var nbytes = size_of[Self.Params]()
        var block = pool[].alloc[UInt8](nbytes)
        var src = UnsafePointer(to=params).bitcast[UInt8]()
        for i in range(nbytes):
            block[i] = src[i]

        var ll = work + "/task.ll"
        var obj = work + "/task.o"
        var elf = work + "/task.elf"

        # Compile for the task's target, finish the lowering with our llc, link
        # against the pre-built launcher, run, download. Each step gates the
        # next, and the working directory goes however far the chain got.
        var ir = Self.device_ir()
        with open(ll, "w") as f:
            f.write(ir)
        _add_export_alias(ir, ll)

        # The mapping the hardware supplies is recovered here, at the one llc a
        # launch runs: the range's base and the parameter block are both known
        # now, and neither is in the compiled-once IR. The block is read as raw
        # words -- neither side has the field types -- and the byte offset of
        # the first pointer equal to the range's base names the parameter it
        # was taken over. See _map_address_flags.
        var range_param = -1
        var words = nbytes // size_of[UInt64]()
        var wp = src.bitcast[UInt64]()
        for w in range(words):
            if wp[w] == UInt64(region.base):
                range_param = w * size_of[UInt64]()
                break
        var mapflags = _map_address_flags(machine, range_param)

        _dump(ir, ll, work, tc, mapflags)

        var rc = _run(
            tc.llc + " -mtriple=riscv64-unknown-elf -mattr=" + tc.features
            + " -relocation-model=pic" + mapflags + " -filetype=obj " + ll + " -o " + obj
        )
        if rc == 0:
            # Link against our controller launcher and run on Detour's timing
            # core: the host staged the pool, m2ndp_run attaches it and runs the
            # task. The range and the parameter block are pool addresses both
            # sides map, so they mean the same to the controller.
            rc = _run("M2NDP_DET='" + tc.det + "' " + tc.link_m2ndp + " " + obj + " " + elf)
            if rc == 0:
                rc = _run(
                    tc.runner + " " + tc.det_config + " " + elf + " "
                    + pool[].path() + " " + String(pool[].base()) + " "
                    + String(pool[].bytes()) + " " + String(region.base) + " "
                    + String(region.size) + " " + String(Int(block)) + " "
                    + String(nbytes)
                )
        # Nothing to download: the task wrote into the caller's pool.
        _ = _run(String("rm -rf ") + work)
        return rc


# ---------------------------------------------------------------- indexing
#
# Two IDs, following Arachne's `GlobalUThreadID()` / `LocalUThreadID()`: one
# across all cores, one within a core. Both come from the hardware in scalar
# registers at spawn, so neither is computed from the other.
#
# `group` is the set of µthreads sharing one scratchpad, i.e. those resident on
# one core.
#
# A kernel indexes its buffers with these rather than being handed the address
# it was mapped to. That mapping is a calling convention and belongs to the
# backend -- surfacing it here would put the convention in every workload.

@always_inline
def local_uthread_id() -> Int:
    """This µthread's index within its group (Arachne `LocalUThreadID()`).

    Counted out as a core receives its µthreads, so it is dense on every core
    whatever the range's spread comes to.
    """
    return Int(external_call["__m2ndp_local_uthread_id", Int32]())


@always_inline
def global_uthread_id() -> Int:
    """This µthread's index across all cores (Arachne `GlobalUThreadID()`).

    Identifies the data this µthread was mapped to.
    """
    return Int(external_call["__m2ndp_global_uthread_id", Int32]())


@always_inline
def group_id() -> Int:
    """Index of this µthread's group."""
    return Int(external_call["__m2ndp_group_id", Int32]())


# The geometry accessors read the same runtime config from either side, but the
# two sides reach it differently: the host runs beside the config file and reads
# it, the device is inside the simulation and reads a value the controller seeded
# into MMIO. `comptime if` picks the branch per compilation target, so the host
# build carries no device symbol and the device IR carries no file access.
@always_inline
def is_ndp() -> Bool:
    """True when compiling for the M2NDP device rather than the host.

    Keys off the `+xm2ndp` vendor feature, which only `m2ndp_target()` carries,
    so it holds whatever the host architecture is -- never `is_x86()`, which a
    non-x86 host would get wrong.
    """
    return CompilationTarget._has_feature["xm2ndp"]()


# The geometry accessors read the same runtime config from either side, reached
# differently: the host runs beside the config file and reads it; the device is
# inside the simulation and reads a value the controller seeded into MMIO. The
# host branch can fail on the file, and a kernel cannot handle an error, so it is
# caught and reported as -1 rather than raised. `comptime if is_ndp()` keeps each
# branch out of the other's build.
@always_inline
def num_groups() -> Int:
    """Number of groups the task spreads across, from the runtime config."""
    comptime if is_ndp():
        return Int(external_call["__m2ndp_num_groups", Int32]())
    else:
        try:
            return Config.load().get("num_ndp_units")
        except:
            return -1


@always_inline
def spad_capacity() -> Int:
    """Usable scratchpad bytes on one unit, from the runtime config.

    Excludes the guard page between units; that spacing is internal, and a
    workload never needs to know it exists.
    """
    comptime if is_ndp():
        return Int(external_call["__m2ndp_spad_capacity", Int32]())
    else:
        try:
            return Config.load().get("spad_size")
        except:
            return -1


# ---------------------------------------------------------------- atomics
#
# With no barrier, atomics are how µthreads combine results. They lower to
# LLVM `atomicrmw` rather than to a symbol, so the backend sees a standard
# instruction. Ordering is RELAXED: accumulation does not need `seq_cst`.
#
# `pop.atomic.rmw` rejects a vector operand, so `atomic_add_lanes` is one
# scalar atomic per lane -- and cannot express the *indexed* case at all, where
# each lane has its own address. See `atomic_add_indexed`.

@always_inline
def atomic_add[
    dtype: DType, address_space: AddressSpace, //
](
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin, address_space=address_space],
    val: Scalar[dtype],
) -> Scalar[dtype]:
    """Atomically add `val` to `ptr[0]`, returning the previous value.

    Works on ordinary memory and on the scratchpad — the address space rides
    along, so this emits `atomicrmw ... ptr` or `atomicrmw ... ptr addrspace(3)`.
    """
    return Atomic.fetch_add[ordering = Ordering.RELAXED](ptr, val)


@always_inline
def atomic_max[
    dtype: DType, address_space: AddressSpace, //
](
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin, address_space=address_space],
    val: Scalar[dtype],
):
    """Atomically raise `ptr[0]` to `val` if `val` is larger.

    The other combine a reduction needs: `softmax` takes a maximum where
    `histogram` takes a sum. Float and integer are different instructions --
    `m2ndp.famomax.w` against `amomax.w` -- and the element type picks between
    them.
    """
    Atomic.max[ordering = Ordering.RELAXED](ptr, val)


@always_inline
def atomic_add_lanes[
    dtype: DType, width: Int, address_space: AddressSpace, //
](
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin, address_space=address_space],
    val: SIMD[dtype, width],
):
    """Atomically add each lane of `val` to consecutive elements at `ptr`.

    **Per-lane atomic, not vector atomic.** Each lane is a separate
    `atomicrmw`; the `width` elements are never updated as one indivisible
    step. Correct for accumulation, wrong if a reader must observe all
    `width` values from the same instant.
    """
    @parameter
    for i in range(width):
        _ = atomic_add(ptr + i, val[i])


@always_inline
def atomic_add_indexed[
    dtype: DType, width: Int, address_space: AddressSpace, //
](
    base: UnsafePointer[Scalar[dtype], MutAnyOrigin, address_space=address_space],
    byte_offsets: SIMD[DType.int32, width],
    val: SIMD[dtype, width],
) -> SIMD[dtype, width]:
    """Atomically add each lane of `val` at `base + byte_offsets[lane]`.

    Indexed, so every lane has its own address -- unlike `atomic_add_lanes`,
    which walks consecutive elements. This is what `histogram` needs, its lanes
    being bin indices. Offsets are in bytes, as in the reference kernel.

    Returns the previous values, one per lane.

    An external symbol rather than an intrinsic, since Mojo's own LLVM cannot
    emit `llvm.riscv.m2ndp.*`; RISCVM2ndpLowerExternalOps rewrites it.
    """
    return external_call[
        _amo_op_prefix[dtype]() + "add_" + _amo_type_suffix[dtype](),
        SIMD[dtype, width],
    ](base, byte_offsets, val)


@always_inline
def _amo_op_prefix[dtype: DType]() -> StaticString:
    """Which vector-atomic family the element type belongs to.

    Integer and float are separate instructions -- `vamoadd` against
    `vfamoadd` -- so the type picks the family as well as the suffix.
    """
    comptime if dtype.is_floating_point():
        return "__m2ndp_vfamo"
    else:
        return "__m2ndp_vamo"


@always_inline
def _amo_type_suffix[dtype: DType]() -> StaticString:
    """The element type as it appears in the vector-atomic symbol names.

    The symbol carries the element type because the frontend cannot overload
    on vector type; the lane count is left to the argument types.
    """
    comptime if dtype == DType.int32 or dtype == DType.uint32:
        return "i32"
    elif dtype == DType.int64 or dtype == DType.uint64:
        return "i64"
    elif dtype == DType.float32:
        return "f32"
    elif dtype == DType.float64:
        return "f64"
    else:
        # Anything else has no M2NDP vector atomic. Returning an empty suffix
        # produces an unresolved `__m2ndp_vamoadd_`, which is a worse error
        # than a compile-time one but is the only spelling available here.
        return ""


# ---------------------------------------------------------------- scratchpad
#
# One shape: a named, statically sized buffer in address space 3, mirroring
# CUDA's `__shared__ T name[N]`. Declare as many as a kernel needs; the
# compiler assigns storage, so offsets never appear in source.
#
# Not used: `external_memory[T, address_space=SHARED]()` (CUDA's
# `extern __shared__`), which emits a fresh global per call site, so two
# accesses to the "same" buffer silently land in different memory.

struct Scratchpad[
    count: Int, type: AnyType, name: StaticString, alignment: Int = 4
](Copyable, Movable, ImplicitlyCopyable):
    """A named per-unit scratchpad region, the CUDA `__shared__` equivalent.

    The name is the region's identity: references to the same name -- in any of
    a task's kernels -- reach one region, and two names never share storage
    whatever their sizes. The handle is empty at runtime and carries only the
    name and shape; `ptr` forms the address on demand.

    `__m2ndp_spad_ptr` is a marker `RISCVM2ndpLowerScratchpad` replaces: it gives
    each name an offset in the `.spad` block and rewrites the call to the
    scratchpad base plus it, so an access is `base + offset`, one load.
    """

    @always_inline
    def __init__(out self):
        pass

    @always_inline
    def ptr(
        self,
    ) -> UnsafePointer[
        Self.type, MutUntrackedOrigin, address_space = AddressSpace.SHARED
    ]:
        """The region's base address on this unit."""
        return external_call[
            "__m2ndp_spad_ptr",
            UnsafePointer[
                Self.type, MutUntrackedOrigin, address_space = AddressSpace.SHARED
            ],
        ](
            _get_kgen_string[Self.name](),
            Self.count * size_of[Self.type](),
            Self.alignment,
        )

    @always_inline
    def __getitem__(
        self,
    ) -> ref [MutUntrackedOrigin, AddressSpace.SHARED] Self.type:
        """The single element, for a one-element region like the parameters."""
        return self.ptr()[]

    @always_inline
    def __getitem__(
        self, i: Int
    ) -> ref [MutUntrackedOrigin, AddressSpace.SHARED] Self.type:
        return self.ptr()[i]

    @always_inline
    def __add__(
        self, n: Int
    ) -> UnsafePointer[
        Self.type, MutUntrackedOrigin, address_space = AddressSpace.SHARED
    ]:
        return self.ptr() + n

    @always_inline
    def store[
        dt: DType, w: Int
    ](self, offset: Int, val: SIMD[dt, w]):
        self.ptr().bitcast[Scalar[dt]]().store(offset, val)

    @always_inline
    def load[
        dt: DType, w: Int
    ](self, offset: Int) -> SIMD[dt, w]:
        return self.ptr().bitcast[Scalar[dt]]().load[width=w](offset)

    @always_inline
    def offset(self) -> Int:
        """Byte offset within the `.spad` block, no base -- the backend replaces
        this with the constant. Legal outside a kernel (it carries no base), so
        `device_main` can form another unit's address of this region."""
        return Int(
            external_call["__m2ndp_spad_offset_by_name", Int64](
                _get_kgen_string[Self.name](),
                Self.count * size_of[Self.type](),
                Self.alignment,
            )
        )

    @always_inline
    def declare_params(self):
        """Mark this region as the parameter block, by name, so the backend
        reserves it and exports the offset the launcher writes it to -- even
        when no kernel reads it, which is why the size travels too."""
        external_call["__m2ndp_declare_params", NoneType](
            _get_kgen_string[Self.name](),
            Self.count * size_of[Self.type](),
            Self.alignment,
        )


@always_inline
def scratchpad[
    count: Int,
    type: AnyType,
    /,
    name: StaticString,
    alignment: Int = 4,
]() -> Scratchpad[count, type, name, alignment]:
    """`count` elements of named scratchpad, shared per unit across launches.

        var tile = scratchpad[64, Float32, name="spmv_tile"]()
        tile[tid] = acc

    Declare several with distinct names and the backend lays them out, so no
    offsets appear in source. See docs/INTERFACE.md and `Scratchpad`.
    """
    return Scratchpad[count, type, name, alignment]()


# The M2NDP scratchpad region in the device address map (detour address_map.h):
# each unit's scratchpad sits at _SPAD_BASE + unit*(spad_capacity() + guard).
# These mirror the non-ASAN map; an ASAN build shifts both by M2NDP_ADDR_OFFSET.
comptime _SPAD_BASE = 0x20000000000  # M2NDP_SPAD_BASE, 2 TiB
comptime _SPAD_GUARD = 0x1000        # M2NDP_GUARD_SIZE, 4 KiB


@always_inline
def _spad_region_base(group: Int) -> Int:
    """Absolute base of `group`'s scratchpad -- its owning unit's region.

    `unit(group)` is identity in the one-group-per-unit model. Matches the
    simulator's `AddressDecoder::scratchpad_base(unit)` so the two agree.
    """
    return _SPAD_BASE + group * (spad_capacity() + _SPAD_GUARD)


@always_inline
def spad_addr[
    count: Int, type: AnyType, name: StaticString, alignment: Int, //
](
    handle: Scratchpad[count, type, name, alignment],
    group: Int,
) -> UnsafePointer[type, MutAnyOrigin]:
    """Absolute address of scratchpad `handle` on `group` (see docs/PRIMITIVES.md).

    device_main cannot index a scratchpad region directly; this forms the
    address instead -- the target unit's region base plus the region's offset --
    so the memory path routes it to that unit.
    """
    var addr = _spad_region_base(group) + handle.offset()
    return UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr).bitcast[type]()


# ---------------------------------------------------------------- target

@always_inline
def m2ndp_target() -> __mlir_type.`!kgen.target`:
    """M²NDP compile target (RISC-V + RVV).

    Built directly rather than through std.sys.info's GPU vendor detection.

    `+xm2ndp` is the vendor extension in our LLVM fork. Mojo's own LLVM does
    not know it and warns on every build, harmlessly:

        '+xm2ndp' is not a recognized feature for this target (ignoring feature)

    The string still reaches the `target-features` attribute verbatim, so the
    marker survives into the IR and our llc picks it up.
    """
    return __mlir_attr[
        `#kgen.target<triple = "riscv64-unknown-elf", `,
        `arch = "generic-rv64", `,
        `features = "+m,+a,+f,+d,+v,+zvl128b,+zfh,+zvfh,+xm2ndp", `,
        `data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
