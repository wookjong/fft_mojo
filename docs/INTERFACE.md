# 백엔드 인터페이스 계약

이 문서는 `src/m2ndp.mojo`가 생성하는 LLVM 레벨 인터페이스를 정의한다.
백엔드 팀은 이 심볼들과 주소공간만 처리하면 되며, 워크로드 코드를 볼
필요가 없다.

## 심볼

모든 심볼은 인자 없는 C ABI 함수로 나온다.

| 심볼 | LLVM 시그니처 | 반환 | 의미 |
|------|---------------|------|------|
| `__m2ndp_uthread_id` | `declare i32 @__m2ndp_uthread_id()` | `i32` | 그룹 내 µthread 로컬 인덱스, `[0, group_size)` |
| `__m2ndp_group_id` | `declare i32 @__m2ndp_group_id()` | `i32` | launch group 인덱스, `[0, grid_size)` |
| `__m2ndp_group_size` | `declare i32 @__m2ndp_group_size()` | `i32` | 그룹당 µthread 수 |
| `__m2ndp_grid_size` | `declare i32 @__m2ndp_grid_size()` | `i32` | 전체 그룹 수 (현재 워크로드 미사용) |
| `__m2ndp_barrier` | `declare void @__m2ndp_barrier()` | `void` | 그룹 내 모든 µthread가 도달할 때까지 대기 |

### 전역 인덱스

`global_uthread_id()`는 별도 심볼이 아니라 다음으로 전개된다:

```
group_id() * group_size() + uthread_id()
```

LLVM IR에서는 세 번의 call + `mul`/`add`로 나타난다. 백엔드가 전용
명령을 갖고 있다면 `src/m2ndp.mojo`에서 단일 심볼로 바꿔도 된다.

## 주소공간

| 번호 | 용도 | 생성 형태 |
|------|------|-----------|
| `3` | 스크래치패드 (그룹 공유 메모리) | `@extern_ptr_syml = external addrspace(3) global [0 x T]` |
| `0` | 일반 메모리 (기본) | 평범한 `ptr` |

스크래치패드 접근은 `getelementptr` + `load`/`store`로 나온다:

```llvm
%40 = getelementptr float, ptr addrspace(3) @extern_ptr_syml, i64 %7
      store float %39, ptr addrspace(3) %40, align 4
%60 = load float, ptr addrspace(3) %40, align 4
```

주소공간 3은 GPU shared memory 관례를 그대로 쓴 것이다. M²NDP 백엔드가
다른 번호를 요구하면 `src/m2ndp.mojo`의 `scratchpad()`에서 `address_space`
인자만 바꾸면 된다.

## 컴파일 타깃

현재 설정 (`src/m2ndp.mojo`의 `m2ndp_target()`):

```
triple          = "riscv64-unknown-elf"
arch            = "generic-rv64"
features        = "+m,+a,+f,+d,+v,+zvl128b"
data_layout     = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
index_bit_width = 64
simd_bit_width  = 128
```

실제 M²NDP 아키텍처가 정해지면 `arch`/`features`/`data_layout`을 교체한다.
`features`의 `+v`가 RVV를, `+zvl128b`가 최소 벡터 레지스터 길이를 지정한다.

## 백엔드 준비 후 전환

`external_call` → 실제 intrinsic으로 바꾸는 지점은 `src/m2ndp.mojo`
한 파일이다. 예:

```mojo
# 현재 (백엔드 없음)
def uthread_id() -> Int:
    return Int(external_call["__m2ndp_uthread_id", Int32]())

# 백엔드 준비 후
def uthread_id() -> Int:
    return Int(llvm_intrinsic["llvm.m2ndp.uthread.id", Int32]())
```

워크로드 코드(`workloads/*.mojo`)는 수정하지 않는다.

## 생성물 확인

```bash
./scripts/build.sh
grep -h "declare.*__m2ndp" out/*.ll | sort -u   # 심볼 목록
grep -c "addrspace(3)" out/spmv.ll               # 스크래치패드 사용
```
