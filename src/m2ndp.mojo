"""M²NDP PoC primitive library.

There is no M²NDP compiler backend yet, so M²NDP-specific operations are
expressed as external symbol calls. They appear in the LLVM IR as
`declare` + `call`, and the backend team only has to map those symbols onto
real M²NDP intrinsics. The symbol names *are* the interface contract.

Symbol convention:
  __m2ndp_local_uthread_id()  -> i32   index within the group
  __m2ndp_global_uthread_id() -> i32   index across all cores
  __m2ndp_group_size()        -> i32   µthreads sharing one scratchpad
  __m2ndp_group_id()          -> i32   which group

Atomics are not symbols here: they lower to LLVM `atomicrmw`. See the
atomics section for why there is no vector form.

The scratchpad needs no symbol of its own: it is a named global in LLVM
address space 3, which the backend places in M²NDP scratchpad memory.

There is no barrier. µthreads are created and retired by hardware FGMT, so
there is no well-defined set to synchronize; the only synchronization point
is a kernel boundary, and `device_main`'s kernel launches are synchronous.
Anything that would need `__syncthreads()` on a GPU has to be split into
two kernels here.
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
    Buffer,
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

    A task is launched onto a range, and the range is what settles how many
    microthreads there are: one per packet of it. That is the only thing the
    launch needs to be told, and spelling it out as two bare integers made it
    look like an offset into nothing.

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


trait NDPTask:
    """What a task has to provide, and what it gets for free.

    Conforming is the whole interface: declare `device_main` and the packet
    size its kernels are written against, and the task gains both the runtime
    entry point the host launches it through and the `launch` that reaches it.

    What a workload does still write is the kernel launches inside
    `device_main`, spelled out as `external_call` to `__m2ndp_launch_parallel`
    and `__m2ndp_launch_serial`. A library wrapper would be better and is not
    possible here: `external_call` takes only a function named at the call
    site, and refuses one that arrives as a parameter, so a wrapper has no way
    to pass the kernel on. Those symbol names are load-bearing besides -- the
    backend decides which functions are kernels by seeing their addresses
    reach them, so a kernel that is never launched is not one.
    """

    comptime packet: Int
    """Bytes of the task's range one microthread is mapped to.

    The granule the parallel kernel is written against -- eight int32 lanes,
    sixteen samples -- expressed in bytes. It belongs here because it is the
    kernel's, not the machine's: the range comes from the launch, and how many
    microthreads that range is is the range divided by this.

    Required rather than defaulted, and best written in terms of the same
    constant the kernel uses, so the two cannot drift:

        comptime packet = W * size_of[Int32]()
    """

    comptime target: __mlir_type.`!kgen.target` = m2ndp_target()
    """Which machine this task is compiled for.

    Defaults to M²NDP, which is what a task written against this library is
    for. Overriding it is a one-line change in the task, and it is the whole
    of what lowering the same workload onto a different machine takes:

        struct VectorAdd(NDPTask):
            comptime target = some_other_target()

    Belonging to the task rather than to the toolchain is the point. A build
    script deciding it would mean the same source meant different things
    depending on how it was invoked.
    """

    @staticmethod
    def device_main(params: UnsafePointer[NoneType, MutAnyOrigin]):
        """Which kernels run, in what order. One per task.

        Arguments arrive the way CUDA's do: one pointer to a block the host
        filled in, which the task casts to a struct of its own. Not a
        parameter each -- `@export` cannot be applied to a parametric
        function, so the entry point below has one fixed signature, and a
        parameter per argument would cap how many a task could take. A struct
        has no such ceiling and carries names and types rather than positions.
        """
        ...

    @export
    @staticmethod
    def __m2ndp_rt_launch_task(
        base: Int,
        size: Int,
        params: UnsafePointer[NoneType, MutAnyOrigin],
    ):
        """The host's launch, arriving on the device.

        Runs on the controller, not on a core -- the backend recognises the
        `__m2ndp_rt_` marker and gives it the ordinary convention, so it may
        call and keep a stack. See RISCVM2ndpArgInfo.h.

        The range comes first because it is what the machine needs before any
        kernel can be launched: a task is launched over a memory range, and
        that range is what settles how many µthreads there are. The parameter
        block only travels through.
        """
        external_call["__m2ndp_set_task_range", NoneType](base, size)
        Self.device_main(params)

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

        One function is asked for -- the task's entry point -- and what comes
        back is that plus everything it needs: `device_main` inlined into it,
        and a copy of every kernel it launches. That is the whole task, because
        a task is exactly what its entry point reaches.

        IR rather than assembly, though `compile_info` will emit either: this
        is the frontend's own LLVM, which has never heard of the vendor
        extension, so the M²NDP lowering -- scratchpad arguments, the identity
        registers, the indexed vector atomics -- has not happened yet. Asking
        for assembly here produces something that looks finished and is not.
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
    def launch(region: PooledRange, *bufs: Buffer) raises -> Int:
        """Run this task over `region` with these buffers.

            _ = Histogram.launch(PooledRange.over(samples),
                                 Buffer.input(samples), Buffer.output(hist))

        Naming the task is the whole of it. The device code is compiled here,
        for the target the task declares; the buffers are uploaded, the task
        runs, and the outputs are downloaded back into the caller's lists.

        `region` is what the task is mapped over, and dividing it by the
        task's packet is where the microthread count comes from. The parameter
        is not called `range` because that is the builtin a `for` loop needs.

        There is no argument for the machine. How many cores exist and how
        finely work is spread across them is the hardware's, not a caller's,
        so the runtime reads config/machine.conf and configures itself.

        Returns the simulator's exit code: 0 for a run that finished, 2 for a
        launcher error, 3 for a fault in the target. The device-side launcher
        and the host-side stubs are supplied by the build; see sim/host_stubs.c
        and scripts/host-run.sh.
        """
        var machine = Machine.from_config()
        var tc = Toolchain()
        var work = _mktemp()

        # Upload the inputs, and describe every buffer for the launcher's
        # command line: direction, byte length, and the file it lives in.
        var specs = String("")
        for i in range(len(bufs)):
            var file = work + "/buf" + String(i) + ".bin"
            if not bufs[i].is_out:
                _write_bytes(
                    file, Int(bufs[i].data.unsafe_ptr()), bufs[i].nbytes
                )
            var dir = String("1") if bufs[i].is_out else String("0")
            specs += dir + " " + String(bufs[i].nbytes) + " " + file + " "

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
            # the buffer specs.
            rc = _run(
                tc.spike + " " + tc.memory + " --extlib=" + tc.extlib
                + " --extension=m2ndp --isa=" + tc.isa + " " + elf + " "
                + String(machine.cores) + " " + String(machine.interleave) + " "
                + String(Self.packet) + " " + String(region.base) + " "
                + String(region.size) + " " + String(len(bufs)) + " " + specs
            )
        if rc == 0:
            for i in range(len(bufs)):
                if bufs[i].is_out:
                    _read_bytes(
                        work + "/buf" + String(i) + ".bin",
                        bufs[i].out_ptr(), bufs[i].nbytes,
                    )

        _ = _run(String("rm -rf ") + work)
        return rc


# ---------------------------------------------------------------- indexing
#
# Two IDs, following Arachne's `GlobalUThreadID()` / `LocalUThreadID()`:
# one across all cores, one within a single core. Both arrive from the
# hardware — a µthread is handed its identity in scalar registers when
# spawned, so neither is computed from the other.
#
# `group` here means the set of µthreads that share one scratchpad, i.e. the
# µthreads resident on one NDP core: `local_uthread_id()` indexes into it,
# `group_size()` is its size, `group_id()` says which one it is.
#
# Kernels take their buffers as ordinary parameters and index them. On the
# hardware a µthread is instead handed the address it was mapped to, and a
# kernel's arguments arrive through the scratchpad — but that is a calling
# convention, and belongs to the backend and the launch glue. Surfacing it
# (`kernel_arg(0)`, raw byte offsets) would put the convention inside every
# workload and break the rule that `workloads/` survives the backend
# switchover unchanged. Turning `base[id]` back into the address the
# hardware already provided is the compiler's job.

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
# With no barrier, atomics are how µthreads combine results. These lower to
# LLVM `atomicrmw`, not to an M²NDP symbol, so the backend sees a standard
# instruction rather than something to map by name.
#
# Ordering is RELAXED: accumulation does not need the `seq_cst` the stdlib
# defaults to, and a weaker ordering leaves the backend fewer fences to emit.
#
# NOTE (vector atomics): `pop.atomic.rmw` rejects a vector operand outright —
#
#   error: 'pop.atomic.rmw' op operand #0 must be pointer to whose type is an
#   arithmetic dtype, but got '!kgen.pointer<...SIMD<f32, 4>>'
#
# — so `atomic_add_lanes` below is one scalar atomic per lane. That covers the
# contiguous case. What it cannot express is the *indexed* one, where each
# lane has its own address; see `atomic_add_indexed`.

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

    Indexed, so every lane has its own address — unlike `atomic_add_lanes`,
    which walks consecutive elements from one pointer. This is the operation
    `histogram` needs, where the lanes are bin indices.

    Offsets are in bytes, matching the reference kernel, which scales sample
    values with `vmul.vi v, v, 4` before the atomic.

    Returns the previous values, one per lane.

    Emitted as an external symbol rather than an intrinsic: Mojo's own LLVM
    has never heard of `llvm.riscv.m2ndp.*` and cannot be made to emit it.
    Our backend rewrites the call in RISCVM2ndpLowerExternalOps. So the same
    mechanism as the µthread ID symbols, and for the same reason.
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

    Emits `@<name>._gpu_shared_mem = internal addrspace(3) global`. Declare
    several and the compiler lays them out — no offsets in source.

    This open-codes the `pop.global_alloc` path that `std.memory`'s
    `stack_allocation` takes for GPU targets. Calling `stack_allocation`
    directly is NOT equivalent here: its promotion is gated on `is_gpu()`, and
    on a RISC-V triple the addrspace(3) alloca falls through to an ordinary
    stack slot — which is per-µthread, so nothing is actually shared.

    The parameter list deliberately mirrors `std._plugin`'s
    `stack_allocation_fn` hook:

        [count: Int, type: AnyType, /, name: Optional[StringSlice], alignment: Int]
            -> UnsafePointer[type, MutUntrackedOrigin, address_space=address_space]

    so that once a toolchain ships both a RISC-V backend and the plugin
    selector (`stdlib_plugin` on the target attribute), this body can move
    into an `std/_plugin/m2ndp/` overlay unchanged and `stack_allocation`
    itself starts routing here. See docs/INTERFACE.md.
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

    Builds the MLIR target attribute directly rather than going through
    std.sys.info's GPU vendor detection. Once the backend exists, swap
    arch/features for the real M²NDP ones.

    `+xm2ndp` is the vendor extension registered in our LLVM fork. Mojo's
    own LLVM does not know it and says so on every build:

        '+xm2ndp' is not a recognized feature for this target (ignoring feature)

    That warning is expected and harmless. The feature string is passed
    through verbatim into the `target-features` function attribute, so the
    marker survives into the IR and our llc — which does know it — picks it
    up without needing -mattr on the command line.
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
