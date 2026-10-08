# Baseline execution strategy·codegen 지원 추가 — 2026-10-08

기존 baseline의 C2C FP32 경로에서 막혔던 composite butterfly와 큰 단일 FFT의
메모리 staging을 구현했다. GPU planner의 radix, logical worker 수, kernel scheme과
원본 계획 snapshot은 유지한다. 지원 범위는 1D C2C FP32, contiguous batch이다.

## 구현

| 항목 | 현재 동작 |
|---|---|
| VkFFT radix 12 | 한 radix-12 stage 안에서 3×4 Cooley–Tukey butterfly를 생성한다. |
| VkFFT radix 14 | 한 radix-14 stage 안에서 2×7 butterfly를 생성한다. Radix 7은 기존 대칭 DFT를 재사용한다. |
| VkFFT radix 15 | 한 radix-15 stage 안에서 3×5 butterfly를 생성한다. |
| Scratchpad 용량을 넘는 원본 단일 FFT | Persistent global staging으로 원본 radix와 worker 분할을 실행한다. |
| 큰 persistent FFT의 소스 크기 | 기존 loop lowering을 worker별로 연결하고, 증명 가능한 residue 주소를 runtime loop로 압축한다. |
| 큰 loop twiddle 표 | Planner가 정한 FP32 bit pattern을 host에서 복원한다. FFT 수학이나 twiddle 선택을 codegen에서 다시 결정하지 않는다. |

Composite butterfly 내부의 작은 FFT는 stage 추가나 radix sequence 변경을 뜻하지 않는다.
예를 들어 VkFFT의 N=24 선택 `(12, 2)`는 그대로 두 stage이며, radix 12를
`(3, 4)`라는 두 planner stage로 바꾸지 않는다. Forward/inverse 부호와 기존
inverse normalization도 유지한다.

Global staging은 첫 FFT stage가 입력을 직접 읽고 마지막 stage가 출력을 직접 쓴다.
중간 stage에는 별도의 CXL ping-pong bank 두 개를 사용한다. Bank는 활성 physical
group마다 한 slot을 배정하며, 다음 batch round에서 같은 slot을 재사용한다.
단계 완료는 기존 `launch_parallel` 순서로 보장한다. 별도 preload나 최종 데이터
copy 없이, round 종료 phase가 worker별 private counter만 갱신한다.

`PersistentWorkgroupPlan.storage_mode`는 `scratchpad` 또는 `global`이다.
Native persistent factory의 기본값은 scratchpad이고, GPU baseline mapping은
원본 local allocation이 용량을 넘을 때 global staging을 명시적으로 선택한다.
GPU baseline의 큰 persistent leaf에는 `loop_stages`를 활성화한다. Loop 생성은
실제 IR의 모든 load/store offset을 검증한 뒤 수행하며, 불규칙한 worker 분할은
기존 unroll로 처리한다. 논리 worker가 맡은 butterfly 집합은 바뀌지 않는다.

## 검증과 재현

최종 코드 기준 회귀 테스트 **94개**, planner·distinct registration 검사
**684개**, Docker의 실제 compiler/simulator 실행 **5개**가 모두 통과했다.
Simulator 결과는 정확성 확인용이며 현재 config의 성능을 GPU와 비교한 결과는 아니다.

```powershell
python -m unittest discover -s verification -p 'verify*.py' -q
python verification/audit_fft_baseline_support.py
python verification/simulate_fft_baseline_support.py --generate
```

Docker의 기존 Mojo M2NDP 이미지에서 `/fft-work`에 이 디렉터리를 마운트한 뒤:

```sh
python3 /fft-work/verification/simulate_fft_baseline_support.py --run
```

- 모든 지원 radix의 정·역방향 butterfly를 독립 DFT matrix와 대조했다.
- VkFFT N=12/14/15/24/28/60/120/240/768/960을 원본 radix sequence 그대로
  생성 코드 실행 결과와 NumPy FFT로 검증했다.
- Global staging의 in-place 결과, 여러 round, 마지막 partial round,
  ragged logical worker, physical/fused/wave 실행을 검사했다. Phase마다 worker
  실행 순서를 섞고 일반 공유 메모리 읽기를 사용한다.
- N=8192의 VkFFT 원본 단일 FFT와 rocFFT adapted의 큰 SBRR 선택은 원본 radix와
  worker 수를 보존하며 정·역방향 생성 코드가 NumPy FFT와 일치한다.
- 19개 길이 × batch 1/32/128 × 정·역방향 × 세 planner의 original/adapted
  진입점으로 만든 **684개 계획** 모두 OK이고 distinct kernel 등록 검사를 통과했다.
  이는 각 조합의 실제 simulator 성능 측정이나 GPU 성능 비교를 뜻하지 않는다.

실제 compiler/simulator 검증 대상은 radix 12/14/15 batch 3, global N=64·worker 17·
batch 65, global N=8192 batch 2이다. 마지막 케이스는 rocFFT가 batch 33에서 선택한
`(16,4,4,4,8)`·logical worker 512를 유지하고 독립 replica 수만 줄인 정확성 검증이다.
큰 FFT의 실제 검증 입력은 replica별로 진폭을 바꾼 real impulse와 imaginary
quarter-period impulse이며, 모든 출력 bin을 독립적으로 알려진 spectrum과 대조한다.
Batch 33 전체 코드도 컴파일했지만, 전체 simulator 실행은 300초의 harness 제한을
넘어 중단했다. 이를 검증 통과로 집계하지 않는다.

| 실제 compiler/simulator 검증 | 결과 | 내부 phase 실행 수 |
|---|---|---:|
| VkFFT radix 12, batch 3, forward | PASS | 1 |
| VkFFT radix 14, batch 3, inverse | PASS | 1 |
| VkFFT radix 15, batch 3, forward | PASS | 1 |
| Global N=64, logical worker 17, batch 65 | PASS | 12 |
| Global N=8192, logical worker 512, batch 2 | PASS | 6 |

실행 결과와 원본 source/이미지 정보는
[`verification/results/baseline_support_2026-10-08`](../verification/results/baseline_support_2026-10-08/)
에 저장한다. 이미지의 host API 차이만 기존 `MappedDomain` bridge로 처리하며
디바이스 body를 교체하지 않는다.
실제 검증 요약은 [`summary.json`](../verification/results/baseline_support_2026-10-08/summary.json),
planner 검사 결과는 [`planner_matrix.json`](../verification/results/baseline_support_2026-10-08/planner_matrix.json)이다.

## 실험에서 해석해야 하는 차이

Global staging은 rocFFT의 half-LDS 구현과 같은 최적화가 아니다. GPU의 half-LDS
flag는 원본 metadata에 유지되고 `NO_EQUIVALENT` 상태를 유지한다. M2NDP가 큰
단일 FFT를 실행할 수 있는 명시적 storage strategy를 추가한 것이다.

Global leaf의 full-array traffic 추정치는 `16 × N × batch × stage_count` bytes다.
예를 들어 N=8192·batch 128·5 stages는 80 MiB이며, scratchpad leaf의 입출력
16 MiB와 차이가 있다. Twiddle table 읽기, register spill, cache 효과는 이
array traffic 추정치에 포함하지 않는다. `FFT_EXECUTION`에는 storage mode,
workspace bytes, local tracker bytes, phase 수와 실제 arithmetic width를 기록하고,
cost model과 lowering-overhead 보고서에도 같은 array traffic을 반영한다.

실제 compiler는 검증한 커널에서 register spill을 보고했다. N=8192의 stage별
spill frame은 16/320/368/400/320 bytes였으며, 이미지의 compiler는 spill이 DRAM으로
간다고 명시한다. 경고 원문은 실행 결과의 `spill_warnings`에 보존했다.
따라서 실험에서는 global staging traffic과 register spill을 함께 확인해야 한다.

GPU fused transpose/twiddle을 별도 M2NDP pass로 펼치는 기존 multi-pass lowering은
계속 별도의 overhead로 보고한다. 이번 추가로 **해당 범위의 실행 미지원**은
해결했으며, GPU library의 모든 fusion·메모리 최적화를 동일하게 이식했다고
해석해서는 안 된다. R2C/C2R, multidimensional/strided API, Rader/Bluestein은
이번 C2C execution/codegen 변경의 지원 범위에 포함하지 않는다.
