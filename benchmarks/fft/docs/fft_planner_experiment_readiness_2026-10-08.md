# Planner 자체의 실험 준비도 — 2026-10-08

> 후속 execution/codegen 변경으로 radix 12/14/15와 큰 단일 FFT의 global staging을
> 지원한다. 아래의 미지원 사례와 504개 matrix는 변경 전 기록이다. 현재의
> 684개 지원 matrix와 검증 범위는
> [baseline codegen 지원 추가](fft_baseline_codegen_support_2026-10-08.md)를 참고한다.

**현재 지원 범위를 명시하면 planner를 실험 baseline으로 사용할 수 있다.**
이번 판정은 planner의 원본 선택 재현, 자원 제약, batch·direction 보존,
실행 계획 구조와 수치 일관성을 대상으로 한다. Simulator config와
benchmark runner의 준비 여부는 이 판정에 포함하지 않는다.

## 다시 확인한 근거

- 현재 코드로 planner 수정 회귀 11개가 통과했다. VkFFT의 원본 C++에서
  얻은 register 777건, radix/register 결정 585건, worker/axis 128건과
  Python port를 대조한다. rocFFT generic SBRR 설정 459건과 flag,
  clFFT block의 worker 그룹 재사용도 검사한다.
- Source-fidelity, GPU logical worker의 M2NDP mapping, exactly-once batch
  coverage, execution strategy 구조·수치 일관성 검증을 재실행해 통과했다.
- clFFT / rocFFT-default / VkFFT 각각의 원본·adapted 진입점으로
  N 14개, batch 1/32/128, forward/inverse를 조합한 **504개 계획**을
  확인했다. 지원 status, batch·direction 보존 및 등록 함수 제한을 검사했다.
  미지원은 올바른 결과로 기록하며 임의 fallback을 허용하지 않는다.
- Native 및 세 adapted planner에서 N=64/128/256/512/1024/2048/4096,
  batch=2, forward/inverse의 **56개 수치 case**가 NumPy FFT와 일치했다.
  이것은 planner IR evaluator 검증이며 실제 simulator 측정은 아니다.

원본 대조의 상세 근거는 [planner 수정 기록](fft_planner_corrections_2026-10-08.md),
이번 계획 조합은 [planner matrix](../verification/results/experiment_readiness_2026-10-08/planner_matrix.json),
수치 결과는 [numeric results](../verification/results/experiment_readiness_2026-10-08/planner_numeric.json)에 있다.

## 확인된 실험 범위

1D FP32 C2C에서 다음 N은 clFFT / rocFFT-default / VkFFT의 adapted
planner가 모두 OK를 반환한다. Batch 1/32/128과 forward/inverse를 확인했다.

| N | clFFT adapted | rocFFT-default adapted | VkFFT adapted |
|---|---|---|---|
| 64, 128, 256, 512, 1024, 2048, 4096 | OK | OK | OK |
| 20, 30, 216 | OK | OK | OK |
| 960 | OK | OK | UNSUPPORTED_CURRENT_CODEGEN |
| 8192, batch=1 | OK | OK | OK |
| 8192, batch=32/128 | OK | RESOURCE_INFEASIBLE | OK |

VkFFT의 N=960은 원본이 선택한 radix를 현재 butterfly가 지원하지 않는
경우다. 선택을 바꿔서 성공으로 만들지 않는다. rocFFT의 N=8192,
batch=32/128은 현재 target에서 선택된 single-kernel plan의 scratchpad
footprint가 초과하는 경우다. 원본 GPU의 half-LDS flag를 M2NDP의 memory
절반 절감으로 간주하지 않는다. 해당 point는 지원 불가 및 이유를
기록하며 성능값을 다른 plan으로 대체하지 않는다.

## Baseline으로 해석하는 범위

M2NDP에 맞춘 planner 실험에는 `plan_m2ndp`가 계획을 제공하고, 원본
`plan`과 GPU logical metadata가 정책·자원 입력 변경의 대조군을 제공한다.
Radix, split, logical worker, physical execution strategy 및 추가
transpose/twiddle 단계는 계획과 함께 기록해야 한다.

초기 공통 비교에는 위의 N=64~4096 power-of-two 범위를 사용할 수 있다.
모든 library 기능의 완전 이식을 주장하는 baseline은 아니다. 현재
VkFFT 직접 prime 범위는 2/3/5/7이며, Rader/Bluestein 등 미지원 알고리즘과
등록·자원 한계를 넘는 조합은 status로 구분한다. 비용 모델의 예상값은
planner 선택 근거이며 실제 성능이나 최적성의 증거로 사용하지 않는다.
