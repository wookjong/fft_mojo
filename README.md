# mojo-m2ndp

Mojo로 M²NDP(RISC-V Vector 기반 µthread GPNDP) 워크로드를 작성하고,
RISC-V/RVV 타깃으로 컴파일해 LLVM IR과 어셈블리를 얻는 PoC.

**백엔드 없이 동작한다.** M²NDP 고유 연산은 external symbol로 표현하므로,
컴파일러 백엔드 작업과 워크로드/라이브러리 작업을 병렬로 진행할 수 있다.
여기서 확정된 심볼 집합이 양쪽의 인터페이스 계약이다.

## Quick start

```bash
git clone <this-repo> && cd mojo-m2ndp
./scripts/setup.sh          # Mojo 툴체인 설치 (./toolchain)
./scripts/build.sh          # 워크로드 -> out/*.ll, out/*.s
./scripts/verify.sh         # 산출물 검증
```

이미 Mojo가 설치돼 있다면 setup을 건너뛰고 경로만 지정한다:

```bash
export MOJO_ROOT=/path/to/modular    # bin/, lib/ 을 가진 디렉토리
./scripts/build.sh
```

## 무엇이 나오는가

`out/vadd_simd.s` (RISC-V 어셈블리):

```asm
.attribute 5, "rv64i2p1_..._v1p0_..._zve32f1p0_zve64d1p0_zvl128b1p0..."

call      __m2ndp_group_id            # M²NDP 인터페이스 심볼
call      __m2ndp_group_size
call      __m2ndp_uthread_id
vsetivli  zero, 4, e32, m1, ta, ma    # RVV: VL=4, SEW=32, LMUL=1
vle32.v   v8, (s2)                    # 벡터 로드
vfadd.vv  v8, v8, v9                  # 벡터 FP 덧셈
vse32.v   v8, (a0)                    # 벡터 스토어
```

`out/spmv.ll` (LLVM IR):

```llvm
target triple = "riscv64-unknown-unknown-elf"
@extern_ptr_syml = external addrspace(3) global [0 x float], align 4

%6  = call i32 @__m2ndp_uthread_id()
      store float %39, ptr addrspace(3) %40      ; 스크래치패드 쓰기
      call void @__m2ndp_barrier()                ; 그룹 배리어
%60 = load float, ptr addrspace(3) %40           ; 스크래치패드 읽기
```

## 구성

```
src/m2ndp.mojo        M²NDP primitive 라이브러리 + 컴파일 타깃 정의
workloads/
  vadd.mojo           µthread 인덱싱만 쓰는 최소 워크로드
  vadd_simd.mojo      SIMD 연산 (RVV 벡터화 확인)
  spmv.mojo           CSR SpMV — indirect access + 스크래치패드 + 배리어 + reduce
scripts/
  setup.sh            Mojo 툴체인 설치
  env.sh              환경변수 (source 해서 사용)
  build.sh            워크로드 컴파일 -> out/
  verify.sh           산출물 검증
docs/INTERFACE.md     백엔드 인터페이스 계약 상세
out/                  생성물 (git 추적 안 함)
```

## 워크로드 작성 방식

```mojo
from m2ndp import uthread_id, group_id, group_size, group_barrier, scratchpad

def spmv_row(values, col_idx, x, row_ptr, y):
    var tile = scratchpad[DType.float32]()   # 그룹 공유 스크래치패드
    var tid = uthread_id()
    var row = group_id()                      # 그룹 하나가 행 하나 담당

    var acc = Float32(0)
    var k = Int(row_ptr[row]) + tid
    while k < Int(row_ptr[row + 1]):
        acc += values[k] * x[Int(col_idx[k])]  # indirect access
        k += group_size()
    tile[tid] = acc
    group_barrier()

    var stride = group_size() // 2             # 스크래치패드 tree reduction
    while stride > 0:
        if tid < stride:
            tile[tid] = tile[tid] + tile[tid + stride]
        group_barrier()
        stride //= 2

    if tid == 0:
        y[row] = tile[0]
```

GPU 커널과 문법이 거의 같다. `std.gpu` 대신 `m2ndp`를 import하고,
warp shuffle 대신 스크래치패드 기반 reduction을 쓴다.

## 백엔드 인터페이스 계약

워크로드 전체가 요구하는 것은 **심볼 4개 + 주소공간 1개**가 전부다.
상세는 [`docs/INTERFACE.md`](docs/INTERFACE.md) 참조.

| 심볼 | 시그니처 | 의미 |
|------|----------|------|
| `__m2ndp_uthread_id` | `i32 ()` | 그룹 내 µthread 로컬 ID |
| `__m2ndp_group_id` | `i32 ()` | launch group ID |
| `__m2ndp_group_size` | `i32 ()` | 그룹당 µthread 수 |
| `__m2ndp_barrier` | `void ()` | 그룹 내 µthread 배리어 |

| 주소공간 | 용도 |
|----------|------|
| `addrspace(3)` | 스크래치패드 (그룹 공유 메모리) |

백엔드가 준비되면 `src/m2ndp.mojo`의 **함수 본체만** 실제 intrinsic으로
교체한다. **워크로드 코드는 수정하지 않는다.**

## 동작 원리

두 가지가 이 PoC를 가능하게 한다.

**1. 커스텀 컴파일 타깃을 직접 구성한다.**
`std.gpu`의 하드웨어 판정(`is_nvidia_gpu()` 등)은 비공개 `std.sys.info`에
묶여 있어 새 하드웨어를 그 체계에 등록할 수 없다. 그러나 커널 컴파일
진입점이 받는 타깃 파라미터는 `!kgen.target` MLIR attribute이고, 이것은
직접 작성할 수 있다. 판정 체계를 통째로 우회한다.

```mojo
def m2ndp_target() -> __mlir_type.`!kgen.target`:
    return __mlir_attr[
        `#kgen.target<triple = "riscv64-unknown-elf", `,
        `arch = "generic-rv64", `,
        `features = "+m,+a,+f,+d,+v,+zvl128b", `,
        ...
    ]
```

`features`의 `+v,+zvl128b`가 RVV를 켠다.

**2. M²NDP 연산은 external symbol로 표현한다.**
Mojo의 `llvm_intrinsic[...]`은 LLVM이 이미 아는 intrinsic만 받는다
(없는 이름은 translation 단계에서 거부). 그래서 백엔드 이전 단계에서는
`external_call`을 쓴다. LLVM IR에 `declare` + `call`로 남아 백엔드
매핑 지점이 명확하다.

## 한계

1. 스크래치패드가 GPU shared와 같은 `addrspace(3)`. 백엔드가 다른 번호를
   쓰면 `src/m2ndp.mojo`의 `scratchpad()`에서 바꾼다.
2. `arch`가 `generic-rv64`. 실제 M²NDP 아키텍처 이름/features/벡터 길이로
   교체 필요.
3. µthread launch는 범위 밖. 이 PoC는 커널 본체만 다룬다. launch
   (그리드/그룹 구성, 디스패치)는 M²NDP 런타임 또는 상위 DSL 담당.
4. `external_call`은 실제 함수 호출로 나간다. 백엔드에서 intrinsic으로
   바꾸면 인라인된 레지스터 읽기가 되어 사라진다.
5. Sleef RVV 초월함수(`exp`/`sqrt` 등) 연결은 미검증.

## 요구 사항

`python3` + `pip`, `unzip`. Linux x86-64에서 검증됨.
