# M2NDP FFT execution strategy 지원 검토 — 2026-10-07

**판정: non-cooperative, cooperative, persistent 세 전략의 기본 실행 구현은
존재하고 recursive tree의 leaf별 혼합도 지원한다. 그러나 planner가 성공적으로
만든 후보가 실제 실행 한도를 넘거나, 선택한 설정이 실행까지 전달되지 않는
경로가 남아 있다. 가장 먼저 수정할 것은 plain/cooperative task의 등록 함수 수
검사다.** 이번 검토는 구현 수정 없이 재현 스크립트와 결과를 추가했다.

대상: native search와 recursive planner, GPU baseline의 공통 physical mapping,
세 codegen, spill probe, persistent의 wave/fused/physical 및 부가 실행 옵션.
앞선 GPU baseline planner 감사와 구분해서 **physical task/함수**를 검사했다.

## 기본 지원 상태

| 전략 | 실제 구현과 generic 연결 | 현재 확인한 한계 |
| --- | --- | --- |
| Non-cooperative | `_build_plan` → `_emit_kernel`, recursive host의 round 분할 | 등록 함수 수 검사가 없다. spread 경로는 SIMD launch granule 8만 지원 |
| Cooperative | worker partition → `_emit_cooperative_stage`, recursive host의 round 분할 | 정상 search는 W=1/2/4/8 범위. 직접 factory와 hybrid tail selector는 잘못된 W도 받아들임 |
| Persistent | `make_persistent_leaf_plan` → `emit_persistent_kernel_struct`, 내부 round 반복 | 기본 target에서 scratchpad 16×length 및 preload+stage+writeback 등록 한도 검사. mixed 경로의 target 전달과 부가 옵션 연결이 불완전 |

Persistent 내부의 `wave`, `fused`, `physical` 세 lowering도 각각 구현되어 있다.
W>8, ragged W, 여러 rounds를 처리한다. 따라서 persistent worker 수가 8을
넘거나 8의 약수가 아니라는 이유만으로 미지원으로 분류하면 안 된다.

## 1. [P1] Plain/cooperative 후보가 task 등록 한도를 넘어도 선택된다

관련 코드: [fft_codegen.py:821](../codegen/fft_codegen.py#L821),
[fft_cooperative_codegen.py:264](../codegen/fft_cooperative_codegen.py#L264),
[fft_plan_search.py:795](../planning/search/fft_plan_search.py#L795).
실제 simulator의 `third_party/m2ndp-detour/src/uthread_generator.cc:313`은
`m_ndp_kernels.size() < max_kernel_register`로 등록 가능 여부를 판정한다.
현재 기본 한도는 8이다.

`generate_candidates(729, scratchpad_byte_budget=16384, max_candidates=24)`는
24개를 반환하고, 그중 5개가 한 task에서 9개의 서로 다른 함수를 호출한다.
정렬된 첫 후보도 해당한다. 기본 radix sequence는 `(3,3,3,3,3,3)`이며 FFT
stage 6개에 별도 `_tail` 함수 3개가 더해진다. Cooperative W=8, W=2 후보 및
lane variant에도 같은 문제가 있다. `compute_lanes=4`,
`narrow_middle_stages=True`, `loop_stages=True`인 실제 generic 렌더링 조건으로
확인했다.

Planner와 codegen은 모두 성공하지만 등록 한도 8과 양립하지 않는다. 이번
환경에서 실제 binary hang/crash를 새로 실행한 것은 아니므로 실패 형태까지
단정하지 않는다. **단순 FFT stage 수가 아닌 별도 tail을 포함한 distinct
launch target 수를 검사해야 한다.** Persistent는 이미 이 한도를 확인하므로
세 전략의 공통 execution feasibility 검사가 필요하다.

이 재현은 허용된 16 KiB leaf budget을 사용한다. 기본 4 KiB budget에서는
N=729가 split되므로 이 특정 위반을 재현하지 않는다.

## 2. [P2] Mixed persistent 후보가 전달된 target을 사용하지 않는다

관련 코드: [fft_plan_recursive.py:432](../planning/strategies/fft_plan_recursive.py#L432),
[fft_plan_search.py:396](../planning/search/fft_plan_search.py#L396).

`_build_leaf_kernel`의 persistent 분기는 `make_persistent_leaf_plan`에
`target`을 전달하지 않는다. Recursive API의 scratchpad 인자도 이 분기에
반영되지 않는다. 따라서 기본 target의 32 units, 120 KiB, max_kernel_register=8로
만들어진다. 반면 unsplit persistent 전용 후보 생성은 `target`을 정상 전달한다.

재현:

- `TargetProfile(num_ndp_units=4, spad_capacity_bytes=2048,
  max_kernel_register=4)`로 N=64 search를 호출하면
  `worker_sequence=('persistent',)` 후보가 남는다. 실제 plan은 **32 groups와
  함수 5개**이며 요청된 4 groups/함수 한도 4와 맞지 않는다.
- `make_recursive_transpose_plan(64, persistent_leaf=True,
  spad_capacity_bytes=512, scratchpad_byte_budget=4096)`가 성공하지만
  그 persistent leaf의 두 scratchpad bank는 합계 **1024 bytes**다.

현재 default target만 쓰면 가려지는 문제다. Mixed/recursive persistent에도
standalone과 동일한 target 및 자원 검사를 적용해야 한다.

## 3. [P2] Explicit cooperative worker 선택은 미지원 폭을 받아들인다

관련 코드: [fft_plan_cooperative.py:408](../planning/execution/fft_plan_cooperative.py#L408),
[fft_plan_persistent.py:878](../planning/execution/fft_plan_persistent.py#L878),
[fft_cooperative_codegen.py:85](../codegen/fft_cooperative_codegen.py#L85).

`make_cooperative_leaf_plan`은 W>=1만 검사한다. 현재 8-uthread interleave chunk에서
같은 FFT의 workers가 한 unit의 scratchpad를 공유하려면 W가 8의 약수여야 한다.
하지만 W=3 및 W=16을 직접 지정해도 계획과 codegen이 성공한다.
`make_persistent_tail_hybrid_plan(..., tail_strategy='cooperative_tail',
tail_cooperative_workers=3)`도 같은 경로를 호출한다. N=64, blocks=35에서
tail 3개가 W=3, 총 9개 uthread로 실행되는 계획을 받아들인다.

W=3의 첫 interleave 경계에서 global ID=8인 uthread는 다음 unit의 local ID=0이다.
Emitter는 `logical_fft_id = global_id // 3 = 2`이지만
`worker_id = local_id % 3 = 0`을 계산한다. 해당 FFT의 세 번째 worker는
`global_id % 3 = 2`여야 한다. FFT workers와 scratchpad가 unit 경계에서 분리되므로
현재 cooperative 구현의 매핑 전제를 만족하지 않는다. W=16은 한 FFT 자체가
두 unit에 걸친다.

**일반 search의 `choose_workers_per_fft`와 GPU baseline 공통 mapper는 이 조건을
이미 지킨다.** GPU mapper는 W가 chunk의 약수가 아니면 persistent로 보낸다.
이 발견은 자동 search 전체가 W=3을 선택한다는 뜻이 아니라, 직접 factory와
explicit hybrid-tail selector의 validation이 빠졌다는 뜻이다.

## 4. [P2] Persistent plan의 mode/copy 선택은 generic 실행에 연결되지 않는다

관련 코드: [fft_plan_core.py:542](../planning/core/fft_plan_core.py#L542),
[fft_persistent_codegen.py:735](../codegen/fft_persistent_codegen.py#L735),
[fft_transpose_codegen.py:873](../codegen/fft_transpose_codegen.py#L873).

Plan에 `lowering_mode='wave'`, `copy_mode='vectorized_contiguous'`를 저장해도
default standalone/recursive codegen은 `physical`/`scalar`를 사용한다.
N=64, radices=(4,4,4), W=16, blocks=33에서는 plan이 wave를 의도하지만
default source에는 wave tracker가 없고 stage launch는 6회다. Explicit
`mode='wave'`를 전달하면 tracker가 생기고 12회가 된다.

동일한 recursive plan의 metadata만 physical/scalar로 바꾼 뒤 렌더링하면
소스가 byte-identical하다. Recursive emitter는 `persistent_mode`만 전달하고
`copy_mode` 인자는 제공하지 않는다. 따라서 vectorized copy는 standalone
low-level codegen에 explicit 인자를 주면 작동하지만 generic recursive/probe
경로에서는 선택할 수 없다. 서로 다른 leaf의 lowering mode를 각각 지정하는
기능도 현재 전체 tree에 적용되는 `persistent_mode` 하나로는 표현되지 않는다.

기존 dataclass 문서가 두 필드를 **metadata only**, codegen 인자를 authoritative로
정의하므로 이것을 내부 계약 위반이라고 단정하지 않는다. 다만 planner의 선택이
execution 선택이라는 관점에서는 연결이 미완성이고, plan 기반 cost/report와
실제 렌더링이 달라질 수 있다. Explicit A/B override를 유지하면서 기본값은
각 plan의 선택을 따르거나, 실제 실행 옵션을 공통 resolved configuration으로
확정해 cost/report/codegen에 함께 전달하는 방식이 필요하다.

## 5. [P2] SIMD launch granule 4는 planner가 받지만 기본 실행 경로가 거부한다

관련 코드: [fft_transpose_codegen.py:669](../codegen/fft_transpose_codegen.py#L669),
`make_fft_kernel.py:621`의 `--simd-lanes`.

`make_recursive_transpose_plan(64, simd_lanes=4, scratchpad_byte_budget=4096)`는
성공하지만 default CLI/probe처럼 `spread_across_units=True`로 렌더링하면
`spread_across_units requires simd_lanes=8` 오류가 난다. 이 옵션은 arithmetic
SIMD 폭이 아니라 32-byte microthread pool의 launch granule과 연결되어 있다.
Arithmetic 4 lanes는 `compute_lanes=4`로 정상 지원한다.

이는 codegen에서 명시적으로 거부하는 경로이며 silent corruption이라고
주장하지 않는다. 현재 지원하지 않는 launch granule을 planner/CLI 단계에서
일관되게 거부하거나, 해당 옵션 조합의 지원 범위를 명시해야 한다.

## 구현되어 있으나 search axis로 연결되지 않은 부가 옵션

Cooperative partition의 round-robin/contiguous/balanced-contiguous,
persistent vectorized copy, tail hybrid는 전용 factory와 codegen이 있다.
하지만 현재 `generate_candidates`/`PlanChoices`는 이 세 옵션을 search axis로
생성하지 않는다. 따라서 **기능 자체의 미구현**과 **자동 planner의 선택 미지원**을
구분해야 한다. Normal mixed leaf selection과 persistent 후보 생성은 이미 search에
들어가 있으므로 오래된 문서의 “mixed search 미구현” 설명은 현재 상태와 다르다.

## 검증과 재현

이번 검토에서 다음 기존 suites가 모두 통과했다.

- `verify_fft_execution_invariants`: 세 전략의 같은 입력 비교, forward/inverse,
  split 및 mixed leaf strategy, planner-resolved compute lanes.
- `verify_fft_persistent`: wave/fused/physical, W>8 및 ragged worker,
  여러 rounds, numpy와의 수치 비교.
- `verify_fft_persistent_search`, `verify_fft_cooperative_partition_modes`,
  `verify_fft_persistent_copy_modes`, `verify_fft_persistent_tail_hybrid`.

추가로 N=60/128/216/630/960/1024, batch=1/33, 각 상위 12개 후보의
**144개 계획을 generic codegen으로 렌더링**했다. 이 표본에서는 렌더링 실패와
등록 함수 한도 초과가 없었다. 이 검사는 전체 크기의 지원이나 실제 compiled
binary의 정답을 보장하지 않는다.

재현 스크립트와 machine-readable 결과:

- [audit_fft_execution_support.py](../verification/audit_fft_execution_support.py)
- [fft_execution_support_audit_2026-10-07.json](../verification/fixtures/fft_execution_support_audit_2026-10-07.json)

```powershell
python -m verification.audit_fft_execution_support
```

현재 checkout에 `toolchain/bin/mojo{,.real}`과 simulator 실행 binary가 없고,
Ubuntu WSL PATH에도 Mojo가 없으며 실행 중인 Docker container도 없다. 이번에
compiled binary를 새로 검증하지 못했다. Numeric 검증은 기존 emitted-code Python
translation harness 결과이며, hardware mapping의 한계는 실제 source와 interleave
주소 관계를 별도로 대조했다. 이전 환경에서의 실측 기록은 새 실행 결과로 세지 않았다.

수정 권장 순서: **distinct 등록 함수 수 공통 검사 → recursive persistent target
전달 → cooperative 폭 validation → plan/execution 옵션 연결 → 부가 search axis**.
