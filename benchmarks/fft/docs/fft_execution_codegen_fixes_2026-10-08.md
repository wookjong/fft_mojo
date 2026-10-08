# FFT execution strategy / code generation 수정

수정 대상은 [이전 검토](fft_execution_codegen_baseline_review_2026-10-07.md)에서 확인한 실행 배치와 코드 생성 문제다. GPU 라이브러리의 radix 선택·planner 정책을 변경하는 작업은 포함하지 않는다.

## 적용 내용

1. **Persistent round / wave tracker를 worker별로 분리했다.** 같은 phase 안에서 worker 0이 공유 counter를 먼저 갱신해 다른 worker가 다음 round나 wave를 읽는 의존성을 제거했다. 각 worker는 자기 cell만 읽고 갱신한다. 이후 phase는 기존 `launch_parallel` 완료 순서를 사용한다. 별도 barrier나 tracker 갱신 kernel을 추가하지 않으므로 등록 함수 개수는 기존과 같다.
2. **Host round와 device FFT ID를 공통 launch descriptor로 연결했다.** `launch_fft_base` / `launch_fft_count`가 필요한 경우 Params에 포함된다. host는 전체 입력·출력 포인터와 logical FFT 범위를 전달하고, device는 해당 범위를 계산한다. Scratchpad cap이 작아도 전체 batch를 빠짐없이 처리한다. cap이 interleave chunk보다 작으면 physical µthread를 padding하고 활성 lane의 logical FFT ID를 압축한다.
3. **Cooperative 실행에도 unit 분산 정책을 적용했다.** 같은 descriptor를 standalone, recursive, hybrid tail에 사용한다. worker count는 chunk를 나누는 기존 조건을 지키며, scratchpad slot은 unit별 local ID로 결정한다. `spread_across_units=False`의 보수적 round 분할도 유지된다.
4. **Pool alignment를 target의 mapping stride로 통일했다.** plain·cooperative·persistent·hybrid의 FFT launch pool에 적용한다. Persistent의 256-byte 상수도 제거했다. vectorized copy의 width는 양의 정수로 검증한다.
5. **Tracker 공간과 실제 생성 옵션을 기록한다.** Persistent scratchpad 검사에 worker-private tracker 공간을 더했다. resource 제한이 있는 recursive persistent leaf는 tracker를 위한 공간을 남겨야 한다. `# FFT_EXECUTION` JSON에는 실제 launch descriptor, stage arithmetic width, persistent mode, copy mode, copy width, tracker footprint, round·phase 개수가 기록된다. override는 원래 plan 객체를 변경하지 않는다.

주요 구현:

- [launch_plan.py](../planning/execution/launch_plan.py): logical FFT 범위, physical launch 크기, padding과 round 계산.
- [common.py](../codegen/common.py): alignment, FFT ID, host launch 및 실행 옵션 기록.
- [fft_persistent_codegen.py](../codegen/fft_persistent_codegen.py): private tracker, copy 검증 및 scratchpad 검사.
- [fft_transpose_codegen.py](../codegen/fft_transpose_codegen.py): recursive launch와 공통 descriptor 연결.

## 수치 및 구조 검증

`python -m unittest discover -s verification -p 'verify_*.py'`: **85 tests 통과**. 별도 persistent 수치 검증도 모든 case가 NumPy FFT와 일치했다.

새 [회귀 검증](../verification/verify_fft_execution_codegen_fixes.py)은 NDP unit별 scratchpad를 두고 실제 interleave에 따른 global/local ID를 계산한다. 기존 max_uthread 크기의 가상 group 모델에 기대지 않는다. cap이 작은 plain/cooperative의 forward·inverse, in-place cooperative, 여러 host round, padded tail, loop stage와 tail, custom 512-byte mapping을 검사한다. Persistent 검증은 snapshot 없이 일반 memory read를 사용하며, worker count 3·10·16·17에서 phase마다 worker 순서를 무작위로 바꾼다.

[수정 후 audit fixture](../verification/fixtures/fft_execution_codegen_fixes_2026-10-08.json):

| 검증 항목 | 수정 전 확인 | 수정 후 |
|---|---|---|
| N=64, batch=64, plain cap=1 FFT/unit | 56 FFT 누락 | 누락 0, host 2 launches |
| N=64, batch=64, cooperative W=8, cap=1 FFT/unit | 32 FFT 누락 | 누락 0, host 2 launches |
| N=1024, batch=128, cooperative W=2 또는 8 | host 32 launches, 일부 unit만 사용 | host 1 launch, 32 units |
| Round/wave counter | 공유 cell 순서에 따라 수치 실패 | 일반 memory read와 임의 worker 순서에서 일치 |
| Custom mapping | persistent host 256-byte 상수 | target의 512-byte alignment |
| Copy width 0 / -1 | 생성 허용 | ValueError로 생성 전 거절 |

Tracker 공간은 기본 8-worker group에서 physical/fused 32 bytes, 별도 wave tracker가 필요한 wave mode는 64 bytes다. 예를 들어 N=64는 data 1024 bytes 외에 physical 32 bytes / wave 64 bytes가 필요하며 경계값과 codegen override도 검사했다. Profile의 scratchpad budget에는 기존처럼 Params 등 런타임 공간을 위한 여유가 필요하다.

## Docker 시뮬레이션

기존 `ghcr.io/psal-postech/mojo-m2ndp:main` 이미지(`b8155d935ba7`)로 독립 검증 컨테이너를 사용했다. 환경 설치나 simulator/config 변경은 없다. Checkout은 `PooledRange` API를 쓰고 이미지에는 `MappedDomain` API가 있어, 검증 소스의 host launch 경계에서 `bytes / UTHREAD_BYTES`로 domain count를 변환했다. Device stage는 수정한 generator의 출력을 그대로 사용한다. Host의 독립 DFT 검증에 필요한 libm은 이미지 driver의 임시 복사본에만 링크했다.

소스·로그·case별 결과는 [실행 결과 디렉터리](../verification/results/execution_codegen_fixes_2026-10-08)에 남긴다. 재현용 [스크립트](../verification/simulate_fft_execution_fixes.py)는 host에서 `--generate`, 기존 이미지 안에서 `--run`으로 실행한다.

**8개 case 모두 host의 독립 DFT 검증을 통과했다.** [요약 JSON](../verification/results/execution_codegen_fixes_2026-10-08/summary.json)에 생성 소스 hash, config hash, launch 수, kernel 실행 cycle 및 실제 spill 경고를 함께 저장했다. 재검증 후 host에서 `--summarize`로 요약을 갱신할 수 있다. 검증용 컨테이너는 종료했다.

| 실제 시뮬레이션 case | Config | Host launches | Device phases | 전체 cycles | 결과 |
|---|---|---:|---:|---:|---|
| Plain N=64, B=64, cap=1 FFT/unit | M2NDP | 2 | 6 | 16,374 | Pass |
| Cooperative N=64, B=65, W=8, cap=1 FFT/unit, in-place | M2NDP | 3 | 9 | 12,277 | Pass |
| Persistent physical, N=64, B=33, W=16 | M2NDP | 1 | 10 | 12,320 | Pass |
| Persistent fused, N=64, B=33, W=17 | M2NDP | 1 | 10 | 13,696 | Pass |
| Persistent wave, N=64, B=65, W=10 | M2NDP | 1 | 24 | 19,512 | Pass |
| Persistent wave inverse, N=64, B=57, W=16 | M2NDP_RTX3060 | 1 | 24 | 30,291 | Pass |
| Persistent + cooperative tail, N=64, B=65 | M2NDP | 2 | 13 | 21,281 | Pass |
| Recursive cooperative, N=960, B=2 | M2NDP | 5 | 9 | 19,419 | Pass |

RTX3060 case의 plan은 해당 config에 맞춰 28 units로 생성했다. 나머지 case는 기본 M2NDP의 32 units다. 수치 harness에서는 N=1024, B=128의 cooperative W=2/8도 실제 생성된 loop stage를 실행해 검증했다. Custom 512-byte mapping은 수치 harness와 hybrid 생성 경로에서 확인했으며, 위 simulator 표에는 해당 custom config를 포함하지 않는다.

## 비교 실험에서의 의미

이 수정은 batch 누락, 공유 tracker 순서 의존성, cooperative 분산 누락을 제거해 execution/codegen의 정확한 출발점을 만든다. 위 수치 검증을 통과했다는 사실은 GPU FFT library planner와 동일한 최적화 수준을 달성했다는 뜻이 아니다.

실제 이미지에서는 compiler가 일부 kernel의 register spill을 DRAM으로 처리하며 경고를 남겼다. 따라서 GPU와 성능을 비교할 때는 전체 task 시간과 FFT phase 실행 시간, host launch 수, DRAM traffic, spill을 함께 기록해야 한다. 이번 cycle 값은 서로 다른 batch·strategy의 기능 검증 결과이며 GPU 대비 성능 결과가 아니다. Library와의 비교는 같은 N·batch·precision·direction·layout 조건으로 별도 측정해야 한다.
