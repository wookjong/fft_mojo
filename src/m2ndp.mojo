"""M²NDP PoC primitive library.

M²NDP operations are external symbol calls, which the backend maps onto real
instructions. The symbol names are the interface contract; see
docs/INTERFACE.md.

  __m2ndp_local_uthread_id()  -> i32   index within the group
  __m2ndp_global_uthread_id() -> i32   index across all cores
  __m2ndp_group_size()        -> i32   µthreads sharing one scratchpad
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
from std.ffi import external_call
from std.memory import AddressSpace, UnsafePointer
from std.sys import size_of
from std.collections.string.string_slice import _get_kgen_string

# The host-side plumbing a launch needs. It knows nothing about tasks, which is
# what keeps the dependency one-way: the model reaches for the machinery, never
# the other way round.
from m2ndp_host import (
    Arg,
    In,
    Out,
    _ArgRec,
    Config,
    Toolchain,
    _add_export_alias,
    _getenv,
    _mktemp,
    _read_bytes,
    _run,
    _write_bytes,
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
    """Bytes of it. Divided by the task's packet, this is the microthread
    count."""

    @staticmethod
    def over[T: Copyable & Movable](ref data: List[T]) -> PooledRange:
        """The whole of a buffer -- what most tasks run over."""
        return PooledRange(0, len(data) * size_of[T]())

    @staticmethod
    def of_bytes(n: Int) -> PooledRange:
        """`n` bytes from the start of the pool, for a task whose range is not
        the length of any one buffer."""
        return PooledRange(0, n)


@fieldwise_init
struct Machine(Copyable, Movable):
    """The NDP hardware a run is modelled on.

    Not the workload's, and not a launch argument: how many cores exist and
    how finely work is interleaved across them is a property of the machine, so the
    runtime reads it from config/machine.conf and configures itself. A task
    that could name a core count would be saying something it cannot know.

    Not the toolchain either -- where llc and the simulator live is where they
    are installed, which is `Toolchain`'s business.
    """

    var cores: Int
    var interleave: Int

    @staticmethod
    def from_config() raises -> Machine:
        """The machine the environment points at. See `Config` for where."""
        var config = Config.load()
        var cores = config.get("cores")
        var interleave = config.get("interleave")
        if cores <= 0 or interleave <= 0:
            raise Error("cores and interleave must both be positive")
        return Machine(cores, interleave)


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
    is alone on its core, so `local_uthread_id()` is 0 and `group_size()` is 1,
    leaving a strided walk that covers the whole of it.

    Synchronous, as `launch_parallel` is.
    """
    external_call["__m2ndp_launch_serial", NoneType](materialize[kernel]())


trait NDPTask:
    """What a task has to provide, and what it gets for free.

    Conforming is the whole interface: declare `device_main` and the packet
    size its kernels are written against, and the task gains both the runtime
    entry point the host launches it through and the `launch` that reaches it.

    `device_main` says which kernels run and in what order, through
    `launch_parallel` and `launch_serial`. A launch names a kernel and nothing
    else: a kernel takes no arguments and reads the task's parameters from the
    scratchpad with `Self.params()`. The backend enforces that, and decides
    what is a kernel by seeing its address reach a launch.
    """

    comptime Params: Movable
    """This task's arguments, declared once for both sides of the launch.

    A struct of the task's own, one `In` or `Out` field per buffer:

        @fieldwise_init
        struct HistogramParams(Movable):
            var samples: In[Int32]
            var out_hist: Out[Int32]

    The host builds one from its lists and `launch` takes it; the kernels read
    the same declaration on the device. One declaration for both ends leaves
    them no order to disagree about, and direction is in the field types, so a
    caller names its lists and nothing else. See `Arg` in m2ndp_host.mojo.
    """

    comptime packet: Int
    """Bytes of the task's range one microthread is mapped to.

    The granule the parallel kernel is written against, in bytes. The range
    divided by this is the microthread count.

    Required rather than defaulted, and best written in terms of the same
    constant the kernel uses, so the two cannot drift:

        comptime packet = W * size_of[Int32]()
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
    def device_main(params: UnsafePointer[Self.Params, MutAnyOrigin]):
        """Which kernels run, in what order. One per task.

        Arguments arrive the way CUDA's do: one pointer to a block the host
        filled in, since `@export` rejects a parametric function and the entry
        point below therefore has one fixed signature. Typed here, because
        `Params` says what the block is -- the cast happens once, below.
        """
        ...

    @staticmethod
    def params() -> UnsafePointer[Self.Params, MutAnyOrigin]:
        """This task's parameters, where a kernel reads them.

            var chunk = Histogram.params()[].samples.ptr.load[width=W](i)

        The launcher copies the block into every core's scratchpad before
        running a kernel there, so this is a read of the base register and each
        field a constant offset from it -- one instruction, as a scratchpad
        global is. Which is why a kernel needs no arguments.

        Kernels only: the controller has no scratchpad, and is handed the block
        directly.
        """
        return external_call[
            "__m2ndp_task_params", UnsafePointer[Self.Params, MutAnyOrigin]
        ]()

    @export
    @staticmethod
    def __m2ndp_rt_launch_task(
        base: Int,
        size: Int,
        params: UnsafePointer[NoneType, MutAnyOrigin],
    ):
        """The host's launch, arriving on the device.

        Controller code, so it may call and keep a stack. The range comes
        first: nothing can be launched until the microthread count is known.

        The one cast in the system is here. This signature is fixed for every
        task, so what the host fills arrives untyped; doing it once, where the
        untypedness comes from, keeps it out of the kernels.
        """
        external_call["__m2ndp_set_task_range", NoneType](base, size)
        Self.device_main(params.bitcast[Self.Params]())

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
        """
        return String(
            compile_info[
                Self.__m2ndp_rt_launch_task,
                emission_kind="llvm",
                target = Self.target,
            ]()
        )

    @staticmethod
    def emit_ir_if_asked() -> Bool:
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
            print(Self.device_ir())
            return True
        return False

    @staticmethod
    def launch(region: PooledRange, ref params: Self.Params) raises -> Int:
        """Run this task over `region` with this parameter block.

            _ = Histogram.launch(PooledRange.over(samples),
                                 HistogramParams(samples, hist))

        The same block `device_main` is handed. Direction is in `Params`, not
        here, so a caller names its lists and nothing else.

        Naming the task is the rest: the device code is compiled here for the
        target it declares, the inputs uploaded, the task run, the outputs
        downloaded back into the caller's lists. `region` divided by the task's
        packet is the microthread count.

        Nothing says what hardware this runs on -- the runtime reads
        config/machine.conf.

        Returns the simulator's exit code: 0 finished, 2 launcher error, 3 a
        fault in the target.
        """
        var machine = Machine.from_config()
        var tc = Toolchain()
        var work = _mktemp()

        # The block's fields, as records whose element types are forgotten.
        # Mojo has no field reflection, so the count comes from the block's
        # size: every field is an `Arg` and they are all one size.
        var nargs = size_of[Self.Params]() // size_of[_ArgRec]()
        var args = UnsafePointer(to=params).bitcast[_ArgRec]()

        # Upload the inputs, and describe every buffer for the launcher's
        # command line: direction, byte length, and the file it lives in.
        var specs = String("")
        for i in range(nargs):
            var file = work + "/buf" + String(i) + ".bin"
            if not args[i].is_out():
                _write_bytes(file, args[i].data(), args[i].nbytes())
            var dir = String("1") if args[i].is_out() else String("0")
            specs += dir + " " + String(args[i].nbytes()) + " " + file + " "

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

        var rc = _run(
            tc.llc + " -mtriple=riscv64-unknown-elf -mattr=" + tc.features
            + " -filetype=obj " + ll + " -o " + obj
        )
        if rc == 0:
            rc = _run(
                tc.lld + " -T " + tc.link_script + " -e _start "
                + _getenv("M2NDP_COMMON_OBJ") + " " + obj + " -o " + elf
            )
        if rc == 0:
            # The machine parameters and the range lead the command line, then
            # the field size and the buffer specs. The launcher writes each
            # buffer's device address into the block it hands the task, and
            # that size is all it needs to know about a layout only Mojo has.
            rc = _run(
                tc.spike + " " + tc.memory + " --extlib=" + tc.extlib
                + " --extension=m2ndp --isa=" + tc.isa + " " + elf + " "
                + String(machine.cores) + " " + String(machine.interleave) + " "
                + String(Self.packet) + " " + String(region.base) + " "
                + String(region.size) + " " + String(nargs) + " "
                + String(size_of[_ArgRec]()) + " " + specs
            )
        if rc == 0:
            for i in range(nargs):
                if args[i].is_out():
                    _read_bytes(
                        work + "/buf" + String(i) + ".bin",
                        UnsafePointer[UInt8, MutAnyOrigin](
                            unsafe_from_address=args[i].data()
                        ),
                        args[i].nbytes(),
                    )

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

    Doubles as the scratchpad slot index.
    """
    return Int(external_call["__m2ndp_local_uthread_id", Int32]())


@always_inline
def global_uthread_id() -> Int:
    """This µthread's index across all cores (Arachne `GlobalUThreadID()`).

    Identifies the data this µthread was mapped to.
    """
    return Int(external_call["__m2ndp_global_uthread_id", Int32]())


@always_inline
def group_size() -> Int:
    """Number of µthreads sharing one scratchpad."""
    return Int(external_call["__m2ndp_group_size", Int32]())


@always_inline
def group_id() -> Int:
    """Index of this µthread's group."""
    return Int(external_call["__m2ndp_group_id", Int32]())


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
        "__m2ndp_vamoadd_" + _amo_type_suffix[dtype](), SIMD[dtype, width]
    ](base, byte_offsets, val)


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

@always_inline
def scratchpad[
    count: Int,
    type: AnyType,
    /,
    name: StaticString,
    alignment: Int = 4,
]() -> UnsafePointer[
    type, MutUntrackedOrigin, address_space = AddressSpace.SHARED
]:
    """`count` elements of named scratchpad, the CUDA `__shared__` equivalent.

        var tile = scratchpad[64, Float32, name="spmv_tile"]()
        tile[tid] = acc

    Emits an `addrspace(3)` global; declare several and the compiler lays them
    out, so no offsets appear in source.

    This open-codes `pop.global_alloc` rather than calling `stack_allocation`,
    whose promotion is gated on `is_gpu()` -- on a RISC-V triple the
    addrspace(3) alloca falls through to an ordinary stack slot, which is
    per-µthread and so shares nothing.

    The signature mirrors `std._plugin`'s `stack_allocation_fn` hook, so this
    body can move into a plugin overlay once a toolchain ships both a RISC-V
    backend and the plugin selector. See docs/INTERFACE.md.
    """
    return UnsafePointer[
        type, MutUntrackedOrigin, address_space = AddressSpace.SHARED
    ](
        __mlir_op.`pop.global_alloc`[
            name = _get_kgen_string[name](),
            count = count.__mlir_index__(),
            memoryType = __mlir_attr.`#pop<global_alloc_addr_space gpu_shared>`,
            _type = UnsafePointer[
                type, MutUntrackedOrigin, address_space = AddressSpace.SHARED
            ]._mlir_type,
            alignment = alignment.__mlir_index__(),
        ]()
    )


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
        `features = "+m,+a,+f,+d,+v,+zvl128b,+xm2ndp", `,
        `data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
