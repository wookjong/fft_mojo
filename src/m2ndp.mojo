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
from std.ffi import external_call
from std.memory import AddressSpace
from std.collections.string.string_slice import _get_kgen_string


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
