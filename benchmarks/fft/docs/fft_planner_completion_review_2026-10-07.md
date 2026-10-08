# FFT planner 완료 여부 재검토 — 2026-10-07

**아직 전체 planner가 완료됐다고 판단할 수 없다.** 앞서 진행한 execution
연결 수정 1–6의 회귀 검증은 통과한다. 그러나 GPU 원본 선택과의 불일치,
native 비용 계산의 불일치, hybrid 계획을 빠뜨리는 공통 helper, benchmark의
batch 검증과 DRAM 보고 오류가 남아 있다. 이번 검토에서는 planner/codegen
구현을 수정하지 않고 재현 스크립트와 현재 checkout의 결과를 추가했다.

대상은 `planning/gpu_baseline`, native `planning/search`, 계획을 소비하는
lane rewrite/diagnostics/codegen, `gpu_vs_m2ndp_benchmark.py`다. 수학적으로
정답인 FFT를 만들 수 있는지, 원본 라이브러리가 고른 계획을 재현하는지,
실행할 코드와 비용/보고서가 일치하는지를 구분해서 검사했다.

**우선 수정할 문제**

| 우선순위 | 항목 | 재현 결과와 영향 |
| --- | --- | --- |
| P1 | VkFFT FP32 register 표 | N=120에서 원본 min registers=10/worker=12, 포트는 5/24. 다른 precision의 표가 physical mapping까지 바꿈 |
| P1 | VkFFT axis swap 해석 | N=64, batch=32에서 원본 worker/FFT slots=8/16, 포트=16/8. 원본 local 축 교환을 논리 FFT 협력 단위 교환으로 해석 |
| P1 | Benchmark batch 정답 검증 | N=64, batch=2/33의 forward/inverse 모두 입력 크기 불일치로 `correct=not_available` |
| P1 | Benchmark DRAM 보고 | N=64, batch=33에서 cooperative W=2는 2배, persistent는 약 7.76배 과다 계산 |
| P2 | Native compute lane 비용 | 동일한 생성 코드인데 N=128의 작업량 metric이 20/56, 점수가 2050.4/2054.0으로 달라짐 |
| P2 | Hybrid 공통 helper 누락 | all-scalar rewrite가 실제 full/tail task에 전달되지 않음. unit diagnostics도 두 task를 한 개로 보고 |
| P2 | 작은 target에서 search 중단 | 분해하면 실행 가능한 target에서도 기본 unsplit 후보 생성의 예외로 search 전체가 종료 |
| P2 | VkFFT radix 선택 충실도 | 원본 optimizer 대신 greedy factorization. N=30/216처럼 지원 radix만 쓰는 경우에도 선택이 다름 |
| P2 | rocFFT-default SBRR metadata | N=128/4096의 원본 half-LDS 선택을 snapshot에서 False로 보고 |

P1의 benchmark 두 항목은 planner 알고리즘 자체의 오류와 별개다. 다만
baseline 비교 결과의 correctness와 traffic 해석을 손상하므로 비교 실험을
하기 전에 수정해야 한다. 작은 target 문제는 기본 target의 실행 실패를
주장하는 것이 아니라, custom target에 대한 search의 자원 적응 한계다.

**GPU 원본 선택과의 비교**

VkFFT 로컬 checkout의 HEAD는
`066a17c17068c0f11c9298d848c2976c71fad1c1`이다. 원본 Scheduler와
AxisBlockSplitter의 host C++를 g++로 다시 컴파일했다. Scheduler의 LF
normalized SHA256은
`9b37b786a70cd416be468e637d21bf463ad307e98b6217b2c0149878460e5809`로
pinned fixture와 일치한다. 캡처한 source/binary와 결과는
`verification/results/planner_completion_2026-10-07/`에 있다.

32개 길이 × batch 1/16/32 × 일반/quad register flag의 192개 사례를 기록했다.
96개 FP32 행 중 기본 benchmark domain인 27개 길이의 batch=1만 비교하면
**17개에서 register/worker 수가 다르고, 17개에서 radix 순서가 다르다.**
두 불일치 집합은 완전히 같지 않다. Quad flag 행은 사용한 register 표를
식별하는 대조군이며 전체 quad-precision FFT 실행을 검증한 결과가 아니다.

1. [vkfft.py:309](../planning/gpu_baseline/vkfft.py#L309)의
   `registers_per_thread_base_table`은 `VkFFTGetRegistersPerThreadQuad`다.
   FP32 계획도 이를 사용한다. 원본 일반 precision 함수는 다른 값을 고른다.

   | N, batch=1 | 원본 FP32 min registers | 현재 min registers | 원본 worker | 현재 worker |
   | --- | ---: | ---: | ---: | ---: |
   | 12 | 12 | 6 | 1 | 2 |
   | 15 | 15 | 3 | 1 | 5 |
   | 20 | 10 | 4 | 2 | 5 |
   | 60 | 10 | 5 | 6 | 12 |
   | 120 | 10 | 5 | 12 | 24 |

   기존 `verify_gpu_baseline_source_fidelity.py:832`도 Quad 함수를 reference로
   삼는다. 그 suite 통과는 선택한 표의 전사가 맞다는 증거이며, FP32에서
   그 표를 선택해도 된다는 증거는 아니다. `plan`과 `plan_m2ndp`가 같은
   helper를 쓰므로 두 경로에 해당한다.

2. [vkfft.py:1133](../planning/gpu_baseline/vkfft.py#L1133)는
   `(axis_block0, batch)`를 교환하고 `axisSwapped` 없이 논리
   threads-per-transform/transforms-per-workgroup으로 반환한다.
   Power-of-two 예제도 틀리므로 앞의 mixed-radix register 오류와 독립적이다.
   N=16·batch=16은 원본 4/16 대신 16/4를 사용하고, N=128·batch=32는
   원본 16/8 대신 8/16을 사용한다. local X/Y, axisSwapped와 논리 worker/FFT
   slots를 별도로 보존하고 mapper에 논리 worker 수를 전달해야 한다.

3. [vkfft.py:501](../planning/gpu_baseline/vkfft.py#L501)의 greedy radix
   대체는 이미 코드에 명시된 fidelity gap이다. 원본/현재 순서는
   N=30에서 `(5,6)/(10,3)`, N=216에서 `(6,6,6)/(9,8,3)`이다.
   두 경우는 원본 radix도 현재 codegen에서 지원한다. 따라서 미지원
   butterfly만으로 이 차이를 설명할 수 없다. 원본 optimizer와 최종
   register/radix 재조정을 이식해야 같은 planner라고 할 수 있다.
   원본이 선택하는 radix 12/14/15는 butterfly를 추가하거나 원본 계획의
   lowering 미지원으로 명시해야 한다.

기존 감사 스크립트의 raw `different_radices` 요약에는 Python list와 tuple을
직접 비교한 false positive도 있었다. 새 completion 감사는 직렬화된 두
sequence의 값을 비교한다. Power-of-two 10개를 불일치에서 제외한 위 17개가
실제 값의 차이다. 과거 raw 요약의 27개를 그대로 결함 개수로 인용하면 안 된다.

**Native 비용과 실제 실행 계획**

[fft_plan_search.py:862](../planning/search/fft_plan_search.py#L862)는 대부분
후보의 stage `compute_lanes`를 미정으로 남긴다. 반면
[fft_cost_model.py:673](../planning/search/fft_cost_model.py#L673)은 미정을
1 chunk로 계산한다. 실제 고수준 생성 경로는 compute_lanes=4와 middle-stage
narrowing을 적용한다. 미정 후보와 같은 폭을 planner에서 명시한 후보를
같은 옵션으로 렌더링하면 source가 byte 단위까지 같은데 비용이 달라진다.

| N | 두 source | 미정/명시된 physical worker-stage batch metric | 미정/명시된 estimated cost |
| --- | --- | --- | --- |
| 64 | 동일 | 6 / 16 | 1024.9 / 1025.9 |
| 128 | 동일 | 20 / 56 | 2050.4 / 2054.0 |
| 960 | 동일 | 33 / 102 | 76693.8 / 76700.7 |

이는 측정 latency가 잘못됐다는 주장보다 먼저, 동일 코드끼리 비용이
일치해야 한다는 기본 조건을 위반한 것이다. 미정 폭의 기존 후보가 명시된
lane 후보보다 계산량을 적게 평가받는다. 모든 후보의 실제 기본 폭을
metrics/dedup/ranking 전에 결정하고, 동일한 계획을 codegen이 소비해야 한다.
이 감사는 어느 후보가 실측 최적인지를 증명하지 않는다.

[fft_plan_lanes.py:202](../planning/execution/fft_plan_lanes.py#L202)는 leaf의
`kernel`만 rewrite한다. 새 `tail_hybrid`의 persistent/tail plan은 빠진다.
N=64·batch=33의 hybrid에 `apply_all_scalar_lanes_to_plan`을 적용하면:

- root의 보조 kernel 폭은 `[1,1,1]`로 바뀐다.
- 실제 flatten 결과의 두 task 폭은 모두 `[None,None,None]`로 남는다.
- compute_lanes=4로 생성한 full/tail 코드는 적용 전후 동일하다.

따라서 일반 API의 “모든 leaf를 scalar로 실행한다”는 약속이 hybrid에서
성립하지 않는다. 자동 lane sweep의 현재 seed가 hybrid라는 주장은 아니다.
선택된 hybrid를 공통 helper로 변환할 때의 재현이다. 같은 이유로
[fft_unit_utilization.py:268](../planning/diagnostics/fft_unit_utilization.py#L268)도
원래 whole-batch kernel을 읽어 두 task를 한 개로 보고한다. 이 진단은
비용 모델과 분리돼 있으므로 diagnostics 누락을 ranking 오류로 세지 않는다.

[fft_plan_search.py:921](../planning/search/fft_plan_search.py#L921)의 baseline
build는 탐색 전에 예외로 끝날 수 있다. N=64, target의 spad/concurrent cap을
512 byte로 설정하고 기본 leaf budget=4096을 쓰면 unsplit leaf의 1024-byte
요구 때문에 search 전체가 종료한다. 동일 target에 leaf budget=512를
명시하면 40개 후보가 생성되고, baseline split 계획을 렌더링하고 NumPy와
대조했을 때 최대 오차는 `9.61e-9`였다. 계획 자체가 없는 입력은 아니다.
target의 실제 한도로 leaf budget을 제한하거나 자원 예외가 난 후보를
제외하고 가능한 split을 계속 탐색해야 한다.

**비교 실험과 metadata**

[gpu_vs_m2ndp_benchmark.py:337](../gpu_vs_m2ndp_benchmark.py#L337)는 입력을
`n`개만 생성하지만 batch 계획과 현재 numeric runner는 `n*batch`개를
요구한다. N=64·batch=2에서는 `(64,)`를 `(128,)`에 넣는 broadcast 예외,
batch=33에서는 `(2112,)`에 넣는 예외가 난다. Forward/inverse 모두 같다.
batch=1 control은 두 방향 모두 통과한다. 입력 크기를 고치고 reference도
`x.reshape(batch,n)`의 transform axis에 적용해야 한다. 계획의 FFT 계산이
틀린 것과 정답 검증이 실행되지 않은 것을 구분해야 한다.

[gpu_vs_m2ndp_benchmark.py:213](../gpu_vs_m2ndp_benchmark.py#L213)의
`total_uthreads * length`는 cooperative/persistent에서 논리 배열 원소 수가
아니다. N=64·batch=33의 단일 전체 read+write 기대값은 33,792 byte다.

| 실행 계획 | benchmark의 read+write byte | 논리 배열의 read+write byte | 과다 계산 배수 |
| --- | ---: | ---: | ---: |
| Cooperative W=2 | 67,584 | 33,792 | 2.00 |
| All persistent | 262,144 | 33,792 | 7.76 |
| Persistent + noncoop tail | 263,168 | 33,792 | 7.79 |

Native cost model은 이 경우 모두 올바른 33,792를 사용한다. 오류는
benchmark의 별도 static diagnostics 계산에 있다. Task의 논리 원소 수와
transpose replica 수로 계산을 통일해야 한다. 위 기대값은 구조적 최소
traffic이고 memory-controller 실측 counter는 아니다.

[rocfft_default.py:184](../planning/gpu_baseline/rocfft_default.py#L184)의
`DefaultConfig`는 half_lds flag를 저장하지 않고,
[logical_plan.py:62](../planning/gpu_baseline/logical_plan.py#L62)는 이를
False로 바꾼다. N=64의 explicit False와 N=128/4096의 generic True를 모두
False로 보고한다. 원본 generator는 multi-radix Stockham의 기본값을 True로
정하고 single-radix는 제외한다.
[Pinned kernel-generator.py](https://github.com/ROCm/rocm-libraries/blob/bee97df517907c771de17189cb867d3c401285ae/projects/rocfft/library/src/device/kernel-generator.py).
Config의 해당 flag는 기존 full-table 조사
[deepdive](gpu_baseline_rocfft_default_deepdive.md)에 근거한다. 이번 snapshot
재현은 세 입력 모두 `gpu_memory_optimization=None`임을 확인했다. 해당
generator 규칙도 다시 읽었다. WGS/TPT/radix 전사가 틀렸다는 뜻은 아니다.

**완료로 볼 수 있는 범위와 별도 남은 구현**

- Task 등록 함수 수 사전 검사, persistent target 전달, cooperative W 검증,
  persistent mode/copy 전달, spread launch granule 검사, partition/copy/tail
  후보 생성과 root hybrid 렌더링의 기존 회귀 suite는 통과한다.
- rocFFT CC/TRTRT 순서와 block config 적용, VkFFT multi-pass splitter/upload
  순서, 원본 GPU 계획 보존과 lowering overhead의 기존 회귀 suite도 통과한다.
- clFFT block 경로의 원본 WGS/batch/LDS 전체 metadata는 아직 추출되지
  않았다. 모르는 값을 None으로 표시하는 현재 방식은 적절하지만 전체
  라이브러리 planner를 이식했다는 근거가 되지는 않는다.
- 원본 fused GPU kernel을 여러 M2NDP node로 실행하는 lowering은 비용을
  분리해서 보고한다. 실제 fused 구현은 남아 있다. 이는 planner 선택과
  execution 충실도를 별도로 평가해야 하는 이유다.
- Persistent copy 후보의 추가 instruction weight는 기본 0이다. Scalar와
  vector copy를 탐색할 수 있으나 기본 static cost만으로 그 성능 차이를
  고르지는 못한다. Host launch/통신 비용도 전부 실측 보정되지 않았다.
- 기본 `make_fft_kernel`은 계속 기존 heuristic을 쓴다. 후보 search는
  plan-index 선택이나 spill/cycle search 경로에서 활성화된다. 새 후보를
  구현한 사실이 모든 기본 호출에서 자동 선택한다는 뜻은 아니다.
- Bounded search와 root-only hybrid는 명시한 범위다. 모든 옵션의 Cartesian
  product, recursive hybrid, Rader/Bluestein, 모든 GPU architecture/precision
  지원을 이번 구현이 약속하지 않는다.

**검증 기록과 재현**

기존 standalone `verify*.py` **34개 모두 최종 통과했다.** 첫 실행에서
32개가 통과했고, 2개는 180초 제한 시간으로 종료돼 600초 제한으로
재실행했다. `verify_fft_plan`은 141.38초,
`verify_gpu_baseline_m2ndp_adapted`는 487.53초에 통과했다. 최종 상태/시간은
아래 manifest에 기록했다.
Fixture를 생성하는 butterfly helper와 실행 entry point가 없는 helper는
34개에 포함하지 않았다. 일부 suite는 다른 suite를 내부 호출한다.

- [Suite manifest](../verification/results/planner_completion_2026-10-07/suites.json)
- [새 감사 결과](../verification/results/planner_completion_2026-10-07/completion_findings.json)
- [Pinned C++ oracle 결과](../verification/results/planner_completion_2026-10-07/vkfft_current_oracle.json)
- [재현 스크립트](../verification/audit_fft_planner_completion.py)

```powershell
# benchmarks/fft에서, 저장된 oracle과 현재 VkFFT 포트 hash를 확인하고 재현
python -m verification.audit_fft_planner_completion

# pinned host C++ oracle까지 다시 캡처
python -m verification.audit_fft_planner_completion --refresh-vkfft-oracle --compiler C:/MinGW/bin/g++.exe
```

이 감사 스크립트의 exit code 0은 관찰 결과를 정상 기록했다는 뜻이며,
planner correctness certificate가 아니다. 기존 suite 통과와 위 재현된
불일치는 동시에 성립한다. 현재 usable Mojo compiler/M2NDP toolchain을
사용할 수 없어 실제 Mojo compile, simulator correctness, cycle winner
검증은 이번 완료 판정의 근거에 포함하지 않았다.

2026-10-08 사용자 요청에 따라 planner에 한정한 교정을 반영했다.
현재 반영 내용과 원본 대조 결과는
[planner 수정 기록](fft_planner_corrections_2026-10-08.md)을 참고한다.
이 문서와 연결된 2026-10-07 결과는 수정 전 감사 기록이다.

당시 권고한 수정 순서는 비교 결과를 손상하는 batch/traffic 계산을 먼저 바로잡고,
native 폭 결정과 hybrid helper를 통일한 다음, VkFFT precision/axis/radix
원본 선택을 교정하는 것이 좋다. 그 뒤 metadata와 custom target fallback을
정리하고 실제 toolchain으로 후보별 정답과 cycle을 검증해야 한다.
