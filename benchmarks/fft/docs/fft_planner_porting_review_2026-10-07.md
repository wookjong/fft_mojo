# FFT planner 포팅 충실도 재검토 — 2026-10-07

**현재 GPU baseline planner 전체를 실제 FFT 라이브러리 planner의 충실한
포팅이라고 판단하기는 어렵다. rocFFT-default의 large-1D 수정과 VkFFT의
split/upload 수정은 확인했지만, VkFFT의 FP32 register 선택과 worker/batch
해석에 원본과 다른 동작이 남아 있다. 기본 27개 benchmark 크기 중 17개에서
batch=1의 worker 수가 원본과 다르다.**

검토 대상은 현재 working tree의 `planning/gpu_baseline/{clfft,
rocfft_default,rocfft,vkfft}.py`와 원본 GPU 계획 보존 경로다. 이 문서는 앞선
검토 이후 들어간 변경을 반영하며, 기존 검토의 수정된 문제를 다시 결함으로
세지 않는다. `planning/search`의 native planner는 M2NDP용으로 설계한 별도
search/cost model이다. 기존 라이브러리 planner를 그대로 포팅한 모듈로
분류하지 않는다. 구현 파일은 변경하지 않고 감사 스크립트와 결과를 추가했다.

## 1. [P1] VkFFT FP32 경로가 quad precision register 표를 사용한다

관련 코드: [vkfft.py:309](../planning/gpu_baseline/vkfft.py#L309),
[vkfft.py:940](../planning/gpu_baseline/vkfft.py#L940).

`registers_per_thread_base_table`은 원본 `VkFFTGetRegistersPerThreadQuad`의
표다. 그런데 원본 `VkFFTGetRegistersPerThread`는
`quadDoubleDoublePrecision` 또는 `quadDoubleDoublePrecisionDoubleMemory`가
설정된 경우에만 이 함수를 호출한다. 이 프로젝트의 FP32 경로는 원본 함수의
별도 일반 precision 표를 사용해야 한다.

원본 C++ 두 함수를 컴파일하고 precision flag만 바꾸어 비교했다.

| 입력, batch=1 | 원본 FP32 min registers | 원본 quad min registers | 현재 포트 min registers | 원본 FP32 worker / 현재 worker |
| --- | ---: | ---: | ---: | --- |
| N=12 | 12 | 6 | 6 | 1 / 2 |
| N=15 | 15 | 3 | 3 | 1 / 5 |
| N=20 | 10 | 4 | 4 | 2 / 5 |
| N=60 | 10 | 5 | 5 | 6 / 12 |
| N=120 | 10 | 5 | 5 | 12 / 24 |

포트의 `ceil(length / min_registers)` 식에 다른 precision의 값을 넣으므로
원본과 다른 worker 수를 고른다. 이후 physical mapping과 원본 GPU snapshot
모두 이 값에 의존한다. 따라서 snapshot을 보존하는 구조로 수정했어도
snapshot에 담긴 선택 자체가 FP32 원본의 선택과 다르다. `plan()`과
`plan_m2ndp()`가 같은 helper를 사용하므로 두 경로 모두 해당한다.

기존 `verify_gpu_baseline_source_fidelity.py:832`도 quad 함수를 reference로
삼고 있어 이 오류를 검출하지 못한다. 기존 source-fidelity suite 통과는
**quad 표를 정확히 옮겼다는 증거**이며 FP32에서 그 표를 선택해도 된다는
증거가 아니다.

수정 방향: precision에 맞는 원본 `VkFFTGetRegistersPerThread` 경로를
사용하고, 최종 radix optimization 이후 register 값까지 계획에 보존한다.

## 2. [P1] VkFFT axis swap을 FFT worker/batch 선택의 교환으로 해석한다

관련 코드: [vkfft.py:1133](../planning/gpu_baseline/vkfft.py#L1133),
[vkfft.py:1227](../planning/gpu_baseline/vkfft.py#L1227),
[vkfft.py:1266](../planning/gpu_baseline/vkfft.py#L1266).

원본 AxisBlockSplitter는 shared-memory bank conflict를 줄이기 위해 local
X/Y를 교환하고 `axisSwapped=1`을 기록한다. 교환은 launch 축의 배치를
바꾸며, 한 FFT에 협력하는 thread 수와 workgroup의 FFT 개수를 서로 바꾸는
의미가 아니다. 원본 `vkFFT_Plan_FFT.h:587`은 이 flag에 따라 batch 축을
선택하고, 같은 파일의 787행에서는 dispatch를 위해 축을 다시 교환한다.

포트는 `_postprocess_axis_upload0`에서 두 숫자를 교환한 뒤 flag 없이
`(threads_per_transform, batch)`로 반환한다. `_leaf_result`는 첫 숫자를
그대로 `workers_per_fft`로 전달한다. 원본 C++ postprocessing block을
컴파일한 결과는 다음과 같다. 이 예제는 power-of-two FFT이므로 앞의 잘못된
mixed-radix register 표와 독립적으로 문제를 보여준다.

| 입력 | 원본의 FFT당 worker / workgroup당 FFT | 현재 포트의 worker / FFT |
| --- | --- | --- |
| N=16, batch=16 | 4 / 16 | 16 / 4 |
| N=64, batch=32 | 8 / 16 | 16 / 8 |
| N=128, batch=32 | 16 / 8 | 8 / 16 |

세 포트 결과는 모두 `OK`이며 실제 physical plan의 worker 수도 잘못
해석한 값이다. N=16·batch=16은 원본의 worker=4를 worker=16으로 바꾸어
cooperative 경로 대신 persistent wave 경로까지 선택한다. FFT 정답의
불일치를 주장하는 것이 아니라, upstream planner가 선택한 협력 단위와
성능 비교 대상이 달라진다는 지적이다.

수정 방향: logical threads-per-transform / transforms-per-workgroup과
local X/Y 및 axisSwapped를 별도 필드로 보존한다. M2NDP mapper에는 논리
FFT당 worker 수를 전달한다.

## 3. [P2, 명시된 포팅 범위의 한계] VkFFT mixed-radix 선택은 원본 optimizer가 아니다

관련 코드: [vkfft.py:501](../planning/gpu_baseline/vkfft.py#L501).

`direct_radix_sequence`는 고정 목록을 큰 radix부터 greedy하게 나누고
내림차순으로 정렬한다. 원본은 prime multiplicity와 register 표를 구한 뒤
`VkFFTOptimizeRadixKernels`로 결합 가능한 radix를 결정하고, 최종 register
요구량을 확인하여 첫 stage의 radix 위치도 바꾼다.

| 입력, batch=1 | 원본 FP32 radix 순서 | 현재 포트 radix 순서 |
| --- | --- | --- |
| N=12 | (12,) | (6,2) |
| N=30 | (5,6) | (10,3) |
| N=216 | (6,6,6) | (9,8,3) |
| N=120 | (10,12) | (10,6,2) |

N=30과 N=216은 원본과 포트의 min register 값이 같고, 원본 radix도 현재
codegen이 모두 지원한다. 따라서 잘못된 precision 표만 고치거나 unsupported
radix에 대한 제한만 설명해도 이 불일치는 해결되지 않는다.

이 greedy 대체는 모듈에 이미 fidelity gap으로 적혀 있다. 숨겨진 신규
구현 오류로 분류하지는 않는다. 하지만 `OK` 결과를 동일한 원본 planner의
선택이라고 해석할 수 없는 실질적인 한계다. 기본 27개 크기의 17개에서
radix 순서가 다르며, register/worker 불일치 17개와 그 입력 집합은 완전히
같지 않다.

또한 원본이 선택하는 radix 12/14/15는 현재 `SUPPORTED_RADICES`에 없다.
완전한 포팅을 목표로 하면 butterfly 지원을 추가하거나 해당 원본 계획의
실행 미지원을 표시해야 한다. 다른 radix로 분해한 결과를 원본의 선택으로
보고하는 것은 수학적 동등성과 planner 충실도를 혼동한다.

## 4. [P2, 보고서 정확도] rocFFT SBRR의 half-LDS 선택이 snapshot에서 사라진다

관련 코드: [rocfft_default.py:184](../planning/gpu_baseline/rocfft_default.py#L184),
[logical_plan.py:62](../planning/gpu_baseline/logical_plan.py#L62).

원본 `kernel-generator.py:800`은 일반 `CS_KERNEL_STOCKHAM`에 `half_lds=True`를
기본값으로 적용한다. 단일-radix kernel에는 적용하지 않고, config의 explicit
값이 있으면 그 값을 따른다. N=128의 `(16,8)`과 N=4096의 `(16,16,16)`
generic config에는 explicit 값이 없으므로 두 kernel은 half-LDS를 사용한다.

`DefaultConfig`는 WGS/TPT/factors만 저장한다. GPU snapshot의 `forced`는
없는 `half_lds`를 `False`로 처리하므로 두 입력 모두
`gpu_metadata=(..., ('half_lds', False), ...)`,
`gpu_memory_optimization=None`으로 보고한다. SBCC나 tuned config처럼
원본 half-LDS 선택을 보존하지 못한다.

이 발견은 포팅된 WGS/TPT/radix 값이 다르다는 뜻은 아니다. 이미 검증한
compiled table의 숫자는 맞다. 원본 GPU memory optimization을 설명하는
metadata가 잘못된다는 뜻이다. `DefaultConfig`에 원본 flag를 보존하거나,
미추출 상태를 `None`으로 표현해야 한다. 원본에서 explicit False인 N=64와
default True인 N=128/4096을 구분하는 회귀 검증이 필요하다.

원본 생성 규칙:
[kernel-generator.py](https://github.com/ROCm/rocm-libraries/blob/bee97df517907c771de17189cb867d3c401285ae/projects/rocfft/library/src/device/kernel-generator.py).

## 확인된 수정과 남은 범위

- **rocFFT-default:** pinned `config_sbrr.py`와 포팅된 459개 항목의
  WGS/TPT/radix가 모두 일치했다. generic 원본의 추가 1개는 길이 1이며
  이 프로젝트의 FFT domain 밖이다. SBCC 32개/SBRC 20개 설정도 원본과
  대조했다. WGS/TPB 생성 식은 `kernel-generator.py`와
  `StockhamKernel` constructor의 LDS/occupancy 규칙에 대응한다.
  CC의 SBCC(div1) → SBRC(div0), TRTRT의 recursive row1(div1) → forced
  Stockham row2(div0)도 현재 구현이 원본 순서를 보존한다.
- **VkFFT multi-pass:** 수정된 2/3-pass splitter, 분할 인자 곱 검증,
  descending upload 번호와 stageStartSize에 대한 7개 regression test가
  통과했다. splitter test에는 원본 C++ block에서 캡처한 122개 사례가 있다.
  이번 새 감사의 register/worker 지적은 이 split 수정의 재반박이 아니다.
- **clFFT:** specialization table, Is1DPossible, LDS 기반 threshold와
  large-1D split의 원본 근거가 있다. 다만 block path는 M2NDP에서 일반
  Stockham leaf와 명시적 transpose로 실행한다. 원본 block kernel의
  BlockSizes/WGS/batch/LDS 전체를 표현한 포팅은 아직 완성되지 않았다.
  현재 original GPU snapshot이 해당 값을 `None`/빈 radix로 표시하는 것은
  모르는 값을 추정하지 않는 올바른 보고 방식이다.
- **공통 lowering:** 원본 fused kernel을 여러 M2NDP node로 풀어 실행하는
  비용은 현재 original/physical plan과 overhead에 따로 표시한다.
  예컨대 2개 GPU kernel을 5개 physical node로 실행하는 계획은 배열
  read/write 추정치가 2.5배다. 이 수치는 측정 cycle 비율을 뜻하지 않는다.
- **rocFFT tuned:** production planner와 다른 offline leaf tuner의 search
  space 포팅이다. 기존 독립 candidate reference 검증은 통과했지만 이번에
  새 GPU/M2NDP 실측 winner를 선정하거나 라이브러리 전체를 실행하지 않았다.

rocFFT 원본: [node_factory.cpp](https://github.com/ROCm/rocm-libraries/blob/bee97df517907c771de17189cb867d3c401285ae/projects/rocfft/library/src/node_factory.cpp),
[tree_node_1D.cpp](https://github.com/ROCm/rocm-libraries/blob/bee97df517907c771de17189cb867d3c401285ae/projects/rocfft/library/src/tree_node_1D.cpp),
[SBCC](https://github.com/ROCm/rocm-libraries/blob/bee97df517907c771de17189cb867d3c401285ae/projects/rocfft/library/src/device/kernels/configs/config_sbcc.py),
[SBRC](https://github.com/ROCm/rocm-libraries/blob/bee97df517907c771de17189cb867d3c401285ae/projects/rocfft/library/src/device/kernels/configs/config_sbrc.py).
로컬 `rocFFT` checkout은 다른 SHA였으므로 위 pinned source를 별도로 읽었다.

VkFFT 원본: [Scheduler.h](https://github.com/DTolm/VkFFT/blob/066a17c17068c0f11c9298d848c2976c71fad1c1/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h),
[AxisBlockSplitter.h](https://github.com/DTolm/VkFFT/blob/066a17c17068c0f11c9298d848c2976c71fad1c1/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h),
[Plan_FFT.h](https://github.com/DTolm/VkFFT/blob/066a17c17068c0f11c9298d848c2976c71fad1c1/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h).
clFFT는 로컬 `c59712e136fa6207956af22f5c0e4cee7d05340e` source를 읽었다.

## 검증과 재현

새 감사는 pinned VkFFT host C++를 컴파일하여 32개 크기 × batch 1/16/32 ×
quad register flag off/on의 **192개 사례**를 기록했다. 원본 register 함수와 radix optimizer,
upload-zero AxisBlock postprocessing block을 사용한다. 테스트용 wrapper는
1D C2C, 단일 upload, N≤4096, registerBoost=1, NVIDIA 상수,
`VKFFT_BACKEND=0`을 고정한다. 이 범위에서 Rader와 register thread-cap
scaling이 활성화되지 않는 전제도 검사한다. CUDA backend의 별도 radix
확장은 이번 결과의 범위에 포함하지 않는다.

Quad flag를 켠 행은 포트가 어느 register 표를 가져왔는지 확인하는
대조군이다. FP32 배열 layout과 8-byte complex memory를 고정하므로 완전한
quad-precision FFT 계획으로 해석하지 않는다. Worker/batch 충실도 판정에는
quad flag를 끈 FP32 행을 사용했다.

원본 scheduler는 기존 fixture의 SHA256과 일치하는 LF source를 확인했다.
새 결과 JSON에는 scheduler/AxisBlock/검토한 Python port의 hash도 남겼다.
실제 GPU 및 Mojo/M2NDP binary의 수치 출력이나 cycle을 새로 측정한 결과는
아니다.

- [감사 스크립트](../verification/audit_fft_planner_porting.py)
- [192개 비교 결과](../verification/fixtures/fft_planner_porting_audit_2026-10-07.json)

`benchmarks/fft`에서 실행:

```powershell
python -m verification.audit_fft_planner_porting
python -m unittest verification.verify_gpu_baseline_rocfft_large1d verification.verify_gpu_baseline_vkfft_multipass verification.verify_gpu_original_plan
python -m verification.verify_gpu_baseline_source_fidelity
python -m verification.verify_gpu_baseline_golden
python -m verification.verify_gpu_baseline_m2ndp_adapted
```

기존 unittest **23개**와 source-fidelity/golden/adapted 세 executable suite는 모두
통과했다. 이 통과와 새 감사에서 발견한 불일치는 양립한다. 기존 VkFFT
reference는 잘못 선택한 quad 표와 axis swap 이후의 배열 위치를 정답으로
삼고 있고, golden은 과거 포트의 출력에서 캡처했기 때문이다.

Adapted suite의 Python numerical harness에서는 clFFT N=1048576까지
forward/inverse가 통과했다. 이 검증은 현재 분해로 FFT 정답을 계산할 수
있음을 보여주며, 원본의 radix/worker 결정이나 compiled binary의 정확도를
보장하지 않는다.

우선순위는 **FP32 register 함수 선택 → axis swap의 논리/launch 축 분리 →
원본 mixed-radix optimizer 포팅과 부족한 butterfly 지원**이다. 이 세 항목을
수정한 뒤 기본 27개 크기와 batch sweep의 baseline을 다시 측정해야 현재
VkFFT 결과를 원본 planner 정책의 성능 비교로 해석할 수 있다.
