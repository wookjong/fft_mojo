"""M²NDP PoC primitive library.

백엔드가 아직 없으므로, M²NDP 고유 연산은 external symbol 호출로 표현한다.
LLVM IR에는 `declare` + `call`로 나오며, 백엔드팀은 이 심볼을 실제 M²NDP
intrinsic으로 매핑하면 된다. 심볼 이름이 곧 인터페이스 계약이다.

심볼 규약:
  __m2ndp_uthread_id()    -> i32   µthread의 그룹 내 로컬 ID
  __m2ndp_group_id()      -> i32   launch group ID
  __m2ndp_group_size()    -> i32   그룹당 µthread 수
  __m2ndp_grid_size()     -> i32   전체 그룹 수
  __m2ndp_barrier()       -> void  그룹 내 µthread 배리어

스크래치패드는 별도 심볼 없이 LLVM address space로 표현한다(주소공간 3).
백엔드에서 M²NDP scratchpad 주소공간에 매핑하면 된다.
"""

from std.ffi import external_call
from std.memory import AddressSpace
from std.gpu.memory import external_memory


# ---------------------------------------------------------------- 인덱싱

@always_inline
def uthread_id() -> Int:
    """그룹 내 µthread 로컬 ID."""
    return Int(external_call["__m2ndp_uthread_id", Int32]())


@always_inline
def group_id() -> Int:
    """launch group ID."""
    return Int(external_call["__m2ndp_group_id", Int32]())


@always_inline
def group_size() -> Int:
    """그룹당 µthread 수."""
    return Int(external_call["__m2ndp_group_size", Int32]())


@always_inline
def grid_size() -> Int:
    """전체 그룹 수."""
    return Int(external_call["__m2ndp_grid_size", Int32]())


@always_inline
def global_uthread_id() -> Int:
    """전역 µthread 인덱스 = group_id * group_size + uthread_id."""
    return group_id() * group_size() + uthread_id()


# ---------------------------------------------------------------- 동기화

@always_inline
def group_barrier():
    """그룹 내 모든 µthread가 도달할 때까지 대기."""
    external_call["__m2ndp_barrier", NoneType]()


# ---------------------------------------------------------------- 스크래치패드

@always_inline
def scratchpad[
    dtype: DType, alignment: Int = 4
]() -> UnsafePointer[
    Scalar[dtype], MutUntrackedOrigin, address_space = AddressSpace.SHARED
]:
    """그룹이 공유하는 스크래치패드 메모리 포인터.

    LLVM address space 3으로 내려간다(현재 GPU shared와 동일 주소공간).
    백엔드에서 M²NDP scratchpad에 매핑.
    """
    return external_memory[
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ]()


# ---------------------------------------------------------------- 타깃

@always_inline
def m2ndp_target() -> __mlir_type.`!kgen.target`:
    """M²NDP 컴파일 타깃 (RISC-V + RVV).

    std.sys.info의 GPU 벤더 판정을 거치지 않고 MLIR target attribute를
    직접 구성한다. 백엔드가 준비되면 arch/features를 M²NDP 것으로 교체.
    """
    return __mlir_attr[
        `#kgen.target<triple = "riscv64-unknown-elf", `,
        `arch = "generic-rv64", `,
        `features = "+m,+a,+f,+d,+v,+zvl128b", `,
        `data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
