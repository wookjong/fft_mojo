# FFT planner 수정 — 2026-10-08

사용자 요청에 따라 수정 범위를 planner와 그 검증·문서로 한정했다.
GPU 측정 runner, benchmark batch/traffic 계산, codegen, compiler,
simulator는 이번 수정에서 변경하지 않았다. 기존 완료 감사는 수정 전
상태의 기록으로 유지한다.

## 반영 내용

| 대상 | 수정 후 동작 |
| --- | --- |
| VkFFT register | FP32에서 ordinary `VkFFTGetRegistersPerThread`를 사용한다. Quad 표는 별도 helper로 분리했다. |
| VkFFT radix | register scaling → 원본 composite optimizer → 최종 register min/max → 첫 stage swap 순서를 따른다. 기존 greedy 선택을 제거했다. |
| VkFFT axis | 논리 TPT/TPB와 물리 local X/Y를 구분한다. axis swap 후에도 논리 worker 수를 보존한다. |
| rocFFT default | generic SBRR 459개 설정의 `half_lds`, `direct_to_from_reg`와 generator 기본값을 보존한다. 단일 radix의 half-LDS 비활성화 규칙도 반영했다. |
| clFFT block | 원본 `BlockSizes`의 WGS, block width, TPT, LDS 및 worker 그룹 재사용 횟수를 GPU logical plan에 기록한다. |
| Native lane/cost | 모든 후보의 unset stage 폭을 dedup/cost 이전에 확정한다. explicit scalar/unnarrowed 폭은 유지한다. |
| Hybrid helper | full/tail의 실제 두 kernel에 lane rewrite와 unit 진단을 적용하고 각각의 block 수를 사용한다. |
| Native target | leaf 정책 budget을 실제 target 한도로 제한한 뒤 분해한다. 작은 scratchpad에서 기본 후보 생성이 불필요하게 중단되지 않는다. Persistent tracker 공간도 자원 진단에 포함한다. |

대표 변경:

- VkFFT N=216: `(6,6,6)`, 최종 minimum register=6, 논리 worker=36.
- VkFFT N=128, batch=4: 논리 TPT=16/TPB=4,
  물리 `local_size=(4,16,1)`, `axis_swapped=True`.
- rocFFT N=64의 half-LDS는 False, N=128/4096은 True.
  GPU 최적화 여부는 M2NDP scratchpad를 자동으로 절반 줄이지 않는다.
- clFFT N=8192: GPU block kernel 두 개의 WGS는 각각 128이며,
  width 16/8, TPT 16/16, LDS 8192/8192 bytes이다.
  첫 kernel은 worker 그룹을 두 번 사용하므로 `WGS=TPT*block_width`로
  추정하면 안 된다.
- Native N=64, target scratchpad/concurrent limit=512 bytes:
  기본 policy budget=4096을 그대로 전달해도 가능한 split 후보를 생성한다.

## 원본 대조와 회귀 검증

VkFFT commit `066a17c17068c0f11c9298d848c2976c71fad1c1`의 원본
host 함수를 C++17로 컴파일했다. Python port의 결과에서 기대값을
만들지 않았다. fixture 생성기는 source SHA-256도 확인한다.

- FP32 base register 표: 777건 일치.
- register scaling/최종 radix·register 결정: 585건 일치.
  single/multi upload, upload 0/1, batch 1/32/256 및 큰 FFT를 포함한다.
- upload-zero 논리 worker/slot와 axis swap: 128건 일치.
- rocFFT generic SBRR: pinned `config_sbrr.py`와 generator 규칙에서
  추출한 459건의 수치 설정과 flag를 대조한다.
- 추가 suite의 11개 테스트는 위 fixture 대조, 미지원 butterfly 보존,
  mixed-radix forward/inverse batch 수치, native 비용 일관성,
  hybrid 실제 task와 작은 target의 동작을 확인한다.

기존 standalone 34개와 추가 회귀 suite 두 개, **총 36개가 최종 통과했다.**
실행 인자가 주석으로 기록되는 새 emitter에 맞춰 기존 동일 코드 검증을
갱신했고, VkFFT N=12/768의 옛 OK 기대값은 원본 radix 12의 명시적
미지원으로 교정했다. 해당 suite는 교정 후 전체 재실행했다.

기존 suite를 포함한 최종 검증 상태는
[suite manifest](../verification/results/planner_fix_2026-10-08/suites.json)에,
VkFFT 원본 대조는
[comparison](../verification/results/planner_fix_2026-10-08/vkfft_comparison.json)에 기록한다.
비용 일관성, hybrid task, 작은 target 및 GPU metadata의 변경 예시는
[planner 결과](../verification/results/planner_fix_2026-10-08/planner_findings.json)에 기록했다.

```powershell
# benchmarks/fft에서
python -m verification.verify_fft_planner_corrections
python -m verification.generate_vkfft_fp32_fixture --compiler C:/MinGW/bin/g++.exe
```

## 비교실험에서의 지원 범위

VkFFT planner port의 직접 prime 범위는 2/3/5/7,
FP32 C2C, register boost=1, portable backend의 pow2 grouping bound=3이다.
CUDA 전용 bound 확장, prime 11/13, Rader/Bluestein까지 이식했다는
뜻은 아니다. 이 범위를 넘어서는 계획은 명시적으로 미지원이다.

원본 optimizer가 radix 12/14/15 등 현재 butterfly에 없는 radix를
선택하면 원본 logical plan을 보존하고 `UNSUPPORTED_CURRENT_CODEGEN`을
반환한다. 실행 가능한 다른 radix로 몰래 바꾸지 않는다. 비교 workload는
각 planner의 status와 원본 계획을 함께 확인해 실행 가능한 교집합에서
선정해야 한다.

큰 FFT의 GPU fused memory operation을 M2NDP의 여러 kernel로 풀어내는
기존 lowering 차이는 남아 있다. planner 선택의 일치와 실제 GPU library
전체 실행 성능의 일치는 별개의 검증 항목이다. 이번 수치 검증은 IR/
emitted-source Python evaluator와 NumPy를 사용했으며, 실제 Mojo compile,
M2NDP simulator cycle, GPU 측정 완료를 의미하지 않는다.
