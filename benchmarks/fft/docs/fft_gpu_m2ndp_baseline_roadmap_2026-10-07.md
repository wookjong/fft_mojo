# GPU library와 M2NDP FFT 병목 비교를 위한 planner 포팅 기준

검토일: 2026-10-07. 현재 소스와 이전 upstream 대조 결과를 기준으로 작성했다.
이번 작업은 검토·수정 순서 정리이며 FFT 구현이나 실행 harness를 변경하지 않았다.

사용자가 정한 실험은 GPU에서 원본 FFT library를 실행하고, M2NDP에서는
현재 구현 중인 M2NDP 실행 방식 위에 해당 library의 planner를 올리는 것이다.
로컬 M2NDP 실험은 Docker Desktop으로 수행할 수 있다. 제공한 실행 안내의
`mojo-ndp`와 `M2NDP_RTX3060`을 목표 환경으로 삼는다. 실제 GPU 모델과 사용할
library/backend는 아직 명시되지 않았다. 아래 NVIDIA 도구 설명은 NVIDIA GPU인
경우에 적용하며, 다른 GPU라면 해당 backend의 timer/profiler로 교체한다.

## 1. 현재 상태에 대한 판단

현재 구현은 planner 포팅을 검토하고 M2NDP 내부의 planning policy를 비교하는
기반으로 사용할 수 있다. 그러나 GPU 대 M2NDP의 병목을 판별하는 완성된
baseline으로 사용하기에는 다음 사항이 남아 있다.

| 확인한 사실 | 실험에 미치는 영향 |
| --- | --- |
| `gpu_vs_m2ndp_benchmark.py`의 GPU 이름을 가진 planner도 전부 M2NDP에서 실행한다 | 기존 CSV는 실제 GPU library와 M2NDP의 성능 비교가 아니다. 실제 GPU 결과를 별도로 수집·결합해야 한다 |
| 현재 checkout에 `mojo-ndp`, `scripts/ndp-results.py`, `M2NDP_RTX3060` config가 없다 | 제공한 Docker 실행 안내와 로컬 소스의 runtime/config 버전을 먼저 맞춰야 한다 |
| `DEFAULT_TARGET_PROFILE`은 32 units로 고정되어 있다 | 안내의 RTX3060 config가 28 units라면 unit 분배·round 수·feasibility가 달라진다 |
| VkFFT FP32 register table, axis swap 해석, mixed-radix optimizer에 불일치가 남아 있다 | 원본과 다른 radix 및 FFT당 worker 수로 비교할 수 있다 |
| rocFFT SBRR `half_lds` 정보와 clFFT block kernel의 일부 실행 정보가 빠져 있다 | 해당 경로를 포함하면 LDS/fusion/worker 병목 비교의 근거가 부족하다 |
| GPU logical snapshot을 lowering 이전에 보존하고 추가 physical node/traffic 추정을 보고하는 구조는 구현되어 있다 | 이 구조를 유지하면서 실제 GPU plan과 독립적으로 검증하면 된다 |
| 재귀 persistent leaf까지 target을 전달하는 수정과 registration 검사가 최근 구현되어 있다 | 다시 구현할 항목이 아니라 선택한 Docker/config에서 실기능을 확인할 항목이다 |
| 현재 benchmark의 수치 정답 판정은 Python interpreter 기반이고 spill probe는 실제 binary의 정답을 검사하지 않는다 | Python 검증 통과를 실제 timed binary의 정답 검증으로 간주할 수 없다 |
| CSV의 `dram_read_bytes`/`dram_write_bytes`는 정적 추정이다 | 실제 메모리 병목을 주장하려면 controller/cache counter 또는 추가 계측이 필요하다 |

자세한 upstream 대조는 [기존 planner 검토](fft_planner_porting_review_2026-10-07.md)에
기록했다. VkFFT C++ oracle은 특정 single-upload FP32 범위를 검증한 것이며,
실제 GPU의 모든 backend·device branch를 검증한 결과는 아니다.

## 2. baseline에서 보존할 것과 M2NDP에 맞게 바꿀 것

GPU의 실행 방식을 그대로 복제할 필요는 없다. 사용자가 구현 중인 M2NDP
executor를 활용하되, 다음 세 층의 정보를 나누어 보존해야 한다.

| 층 | 보존하거나 기록할 정보 | 허용되는 변경 |
| --- | --- | --- |
| 원본 GPU logical plan | radix 순서, FFT 분할, pass 실행 순서, stride/layout, transpose, twiddle, fusion, FFT당 logical worker, workgroup당 FFT, GPU LDS, specialization flag | 실제 GPU library가 선택한 원본 결과를 기록한다. M2NDP 값으로 덮어쓰지 않는다 |
| M2NDP용 planner 결과 | 적용한 target resource, 분기 결과, 선택한 radix/분할/배치, 원본과 달라진 항목과 이유 | scratchpad·unit 수 등 의미가 대응되는 resource에 따른 재계획을 허용한다 |
| M2NDP physical execution | physical worker, RVV width/LMUL, wave/round, scratchpad, stage 동기화, 실제 task/launch, 주소 mapping | M2NDP executor의 제약과 최적화에 맞게 결정한다 |

`GPU threads_per_transform = M2NDP physical worker`나 `GPU warp = RVV vector`로
직접 치환하지 않는다. GPU workgroup의 worker 협력은 M2NDP에서 여러 wave와
stage launch로 구현될 수 있다. logical 수를 보존하면서 physical 실행을 따로 기록한다.

다음 세 결과를 함께 확보하는 것이 유용하다.

1. **GPU 원본 library:** 같은 FFT 문제를 실제 GPU에서 실행한 결과.
2. **M2NDP 원본 결정 유지:** 가능한 범위에서 GPU logical plan을 유지하여 M2NDP executor에 매핑한 결과.
3. **M2NDP resource 적용:** 같은 planner 알고리즘에 M2NDP resource를 적용한 결과. Native planner는 이후 성능 개선 폭을 확인하는 추가 대조군으로 둔다.

GPU와 M2NDP의 입력 크기·설정이 같아도 hardware 입력 때문에 plan은 달라질 수
있다. 1과 3은 시스템 성능 비교이고, 2와 3은 resource 적응의 효과를 확인한다.
1과 2도 binary/backend가 다르므로 성능 차이를 곧바로 순수 hardware 차이로
해석하지 않는다. 원본 결정 유지 경로가 지원되지 않으면 그 범위를 명시하고,
해당 항목의 planner 효과에 대한 결론을 제한한다.

## 3. 수정 순서와 각 단계의 완료 기준

### 0. Docker 환경과 비교 조건을 고정한다

수정 대상: runtime 연결부, config 전달, `planning/core/target_profile.py`,
benchmark 입력·결과 schema.

- Docker image의 `main` tag만 기록하지 말고 image digest, repository SHA,
  Mojo/LLVM/simulator 버전, 실제 사용한 config와 관련 DRAM/link 파일 hash를 기록한다.
- `/work`의 툴체인을 유지하고 작업 소스는 다른 경로에 mount한다. 현재 FFT
  codegen/host API와 image의 `NDPTask`/`MappedDomain` API가 맞는지 확인한다.
- FFT run에 config를 명시하고, 로그의 resolved Config 경로/hash와 기대값을
  비교한다. 생략하여 default config로 실행된 결과가 본 실험에 섞이지 않게 한다.
- config에서 unit 수, interleave/packet, scratchpad, registration 제한 등
  필요한 hardware 입력을 읽어 target에 전달한다. 실제 clock domain을 사용해
  NDP cycle을 시간으로 변환한다. SIMD width는 compiler 옵션과 emitted RVV도 확인한다.
- scratchpad 전체 용량, runtime 예약분을 뺀 사용 가능 용량, 경험적으로 얻은
  동시 실행 제한을 구분한다. 기존 64 KiB concurrency 가정을 새 config의
  architecture 사실로 복사하지 않는다.
- 최초 범위는 1D FP32 C2C, 동일한 N·batch·방향·normalization·placement·stride로
  고정한다. 내부 SoA/AoS 변환이 필요하면 비용과 수행 위치를 따로 기록한다.

완료 기준: 한 case에서 binary·target·runtime이 같은 config를 사용하고,
28/32 units 같은 차이가 결과 metadata에서 검출된다. timing에 포함할 구간과
입력 데이터의 최초 위치가 명시되어 있다.

### 1. 실제 GPU library 결과와 plan을 확보한다

수정 대상: GPU 실행용 benchmark와 공통 결과 schema. 기존 M2NDP harness에
GPU 실행 target을 추가하거나 별도 runner를 만든 후 같은 case ID로 합친다.

- 우선 실제 GPU에서 사용할 한 library/backend를 주 baseline으로 선정한다.
  세 library의 모든 기능을 먼저 포팅할 필요는 없다. NVIDIA라면 VkFFT CUDA
  backend가 소스 planner 대조에 적합한 선택지이며 cuFFT는 별도 성능 대조군으로 사용할 수 있다.
- library SHA/version, backend, 실제 GPU resource와 plan 생성 옵션을 고정한다.
  rocFFT의 gfx908 table이나 VkFFT의 대표 GPU 상수가 실제 GPU의 선택과 같다고
  가정하지 않는다.
- 원본 plan dump/debug instrumentation과 GPU launch trace를 통해 실행 순서,
  radix·workgroup·batch·fusion을 확보한다. 추출되지 않은 값은 unknown으로 남긴다.
- plan 생성/JIT, 할당, warm-up, transfer를 steady-state FFT 실행 시간과 분리한다.
  timed 실행과 profiling 실행을 나눈다.

완료 기준: 같은 case의 실제 GPU 정답, 실행 시간, 원본 plan, launch trace가
있으며 결과에 `execution_target`과 `planner_origin`이 별도 필드로 들어간다.

### 2. planner 결정의 불일치를 먼저 고친다

수정 대상: `planning/gpu_baseline/vkfft.py`, `rocfft_default.py`,
`logical_plan.py`, 선택한 library의 원본 oracle/fixture.

VkFFT를 주 baseline으로 선택한다면 다음 순서가 적절하다.

1. FP32가 사용하는 `VkFFTGetRegistersPerThread` 표를 정확히 이식한다.
   현재 mixed-radix 경로는 Quad 표를 사용한다. N=20에서는 원본 min register=10,
   worker=2인데 현재는 4와 5다. 기존 검증의 reference도 같은 Quad 표에
   의존하는 경로가 있어 독립적인 원본 검증으로 바꿔야 한다.
2. register 결과에 의존하는 `VkFFTOptimizeRadixKernels`, register 확인,
   첫 stage 재배치를 이식한다. N=30의 (5,6) 대 (10,3), N=216의
   (6,6,6) 대 (9,8,3)은 precision 표만 바꿔서는 해결되지 않는다.
3. 원본이 선택하는 radix 12/14/15 등을 codegen이 처리하게 하거나 해당 plan을
   명시적으로 unsupported 처리한다. 다른 radix로 분해할 경우 원본 logical
   radix와 실제 lowering을 모두 기록하고, 추가 stage 비용을 보고한다.
4. `axisSwapped`, local X/Y, logical threads-per-transform,
   transforms-per-workgroup을 분리한다. N=16, batch=16의 원본 logical
   worker/batch=4/16을 현재는 16/4로 전달한다. 이 오류는 power-of-two에도 해당한다.
5. 실제 GPU backend의 추가 분기와 큰 N의 upload/split 경계까지 대조한다.
   기존 portable C++ oracle의 일치만으로 CUDA backend 전체의 일치를 주장하지 않는다.

rocFFT가 주 baseline이라면 SBRR metadata에 `half_lds` 및 관련 generator
default/override를 보존한다. 현재 N=128,4096 등의 원본 True 정보가 빠져 있다.
clFFT block 경로를 포함한다면 block radix/WGS/TPT/TPB/LDS와 fusion을 원본에서
추출한다. 이 작업은 해당 library/경로를 이번 baseline에 포함할 때 필요하다.

완료 기준: 선언한 support 범위에서 원본 library의 결정과 frozen port의
결정이 일치하고, resource 적용 후 달라진 결정은 변경 이유가 기록된다.
분기 경계·batch 축 변경·mixed-radix·multi-pass case를 포함하여 검증한다.

### 3. planner와 현재 구현 중인 M2NDP executor를 연결하고 실제 정답을 검증한다

수정 대상: 공통 mapper, recursive/leaf plan, codegen, 실제 실행 runner.

- 원본 logical snapshot을 physical lowering 이전에 보존한다. 현재 구조를 유지한다.
- 모든 leaf와 transpose에 선택한 target과 실제 batch/layout을 전달한다.
  최근 구현한 recursive target 전달과 registration 검사를 Docker에서도 확인한다.
- 같은 FFT 안의 stage 의존성, 서로 다른 FFT/batch의 독립성, scratchpad 소유권,
  wave/round의 logical worker 대응을 검사한다.
- host task 수, controller의 stage launch 수, wave 수, parameter-copy launch 수,
  physical plan node 수를 별도로 기록한다. persistent라는 이름만으로 GPU의
  단일 kernel 실행과 같은 동기화 비용이라고 간주하지 않는다.
- 실제 측정하는 compiled binary의 전체 batch 출력을 꺼내 double reference와
  대조한다. Python interpreter 검증은 유지하되 compiled 결과 판정을 추가한다.
- random/impulse/single-frequency, forward/inverse와 normalization, split 및
  batch tail case를 포함한다. 상대 L2 오차와 최대 오차를 함께 기록하고 FP32
  오차 기준을 사전에 정한다. 단일 절대 오차 1e-2만으로 모든 N을 판정하지 않는다.
- spill, register/launch limit, scratchpad overflow, unsupported는 성공한 성능
  결과와 분리한다. runtime return code가 성공인 것만으로 정답이라고 표시하지 않는다.

완료 기준: 실험 case 전체에서 실제 timed binary의 정답과 resource 유효성이
확인되고, 모든 batch가 정확히 한 번 처리된다.

### 4. timing과 counter를 실행 단계에 연결한다

수정 대상: runtime 결과 adapter, `gpu_vs_m2ndp_benchmark.py`,
`analyze_gpu_vs_m2ndp.py`, 필요하면 simulator counter export.

baseline의 첫 성능 측정 전까지 완료해야 한다. 단계 3의 검증 runner와 함께
개발할 수 있다.

| 측정 구간 | GPU | M2NDP |
| --- | --- | --- |
| 전체 FFT device sequence | 동일 stream에서 FFT 전체를 감싸는 device timer. 여러 kernel과 사이 gap을 포함 | 첫 관련 device 실행부터 마지막 완료까지의 시간. 관련 parameter copy와 stage gap을 포함했는지 명시 |
| 연산 kernel 구간 | 개별 FFT/transpose kernel duration | body/stage/wave별 exec, parameter-copy exec를 별도 기록 |
| 호출 지연 | 준비된 plan·resident buffer에 대한 launch부터 completion까지 host timer | host task issued부터 notified까지. runtime이 보고하는 전체 cycle과 경계가 같은지 확인 |
| application 구간 | 필요할 경우 실제 초기 데이터 위치부터 결과 사용 가능 시점까지 | 같은 초기 위치와 완료 조건으로 별도 측정 |

Docker의 wall-clock simulation 소요 시간과 GPU 실행 시간을 비교하지 않는다.
resident FFT를 주 비교로 두고 GPU H2D/D2H와 M2NDP CXL 초기화는 별도의
application 실험에서 초기 위치를 명시하여 비교한다. GPU 계획 생성/JIT를
제외할 때 M2NDP planner/search·compile도 제외한다.

사용자가 제공한 SAXPY 예시의 total=12704 cycles, body=658, params=1778,
gap=603은 이 분리가 필요함을 보여준다. 아직 FFT에서도 같은 비중이라는
증거는 없다. launch timeline의 nested 구간을 중복 합산하지 않는다.

수집할 M2NDP 지표는 unit별 active/idle·완료 시간, vector issue와 active lane,
register spill, dependency wait, SPAD/LDST resource wait, memory/crossbar queue,
L1 및 DRAM transaction/byte, stage gap이다. 현재 source의
`VECTOR_DEPENDENCY_WAIT`, `SPAD_VECTOR_UNIT_FULL`, `LDST_VECTOR_UNIT_FULL`,
`TO_MEM_FULL`, `TO_CROSSBAR_FULL`, `EMPTY_QUEUE` 등은 출발점으로 사용할 수 있다.
필요한 event/counter가 선택한 image에 없으면 정의와 export를 추가한다.
per-stage attribution이 불가능한 경우 aggregate라고 표시하고 최소한 전체
FFT 측정 범위를 확정한다.

현재 `STAT` export에는 일부 issue/SPAD count만 있으며 모든 stall 지표가
있는 것은 아니다. 각 counter의 단위·scope·분모를 확인한다. 여러 wait가
같은 시간에 발생할 수 있으므로 stall count의 합을 총 cycle 분해로 사용하지 않는다.
GPU와 M2NDP 각각의 peak 대비 utilization %도 같은 대역폭 값으로 비교하지 않는다.

완료 기준: 한 case의 시간·launch·counter가 같은 실행 ID와 stage에 연결되고,
추정 traffic과 measured traffic이 별도 필드이다. GPU 반복 측정의 중앙값과
변동폭, simulator seed/cache 초기 조건을 기록한다.

### 5. 중요한 lowering 추가 비용을 측정하고 필요한 부분을 줄인다

수정 대상: strided load/store, transpose, large-twiddle, fusion codegen과 executor.

원본 GPU의 fusion을 모두 동일하게 구현해야 planner 포팅으로 인정되는 것은
아니다. 다만 현재 mapper가 추가한 비용을 감추면 M2NDP 병목 해석이 달라진다.

- 현재의 original/physical snapshot 및 lowering overhead 보고를 유지한다.
- GPU 2 kernels를 M2NDP 5 nodes로 풀어 실행하는 경로는 전체 배열 pass 기준
  read+write 추정이 2.5배인 사례가 있다. 이것은 actual DRAM traffic 또는
  execution time이 2.5배라는 뜻이 아니다.
- 추가 비용이 큰 경로부터 transpose를 leaf load/store와 통합하거나 twiddle과
  합치는 대조 구현을 만든다. scratchpad 내 stage인지 DRAM을 경유하는
  stage인지도 실제 code와 counter로 확인한다.
- 기존 전개 버전과 개선 버전을 같은 input/plan/config로 비교한다. 지원하지
  못한 fusion은 현재 구현 제한인지 M2NDP 제약인지 근거를 구분하여 기록한다.
- 최적화가 비교 구간의 memory pass·layout·수학적 plan을 바꾸었다면 변경을
  기록한다. 성능을 정적 traffic 비율로 사후 보정하지 않는다.

완료 기준: 각 주요 성능 차이에 대해 planner 결정, executor가 추가한 비용,
hardware resource wait 중 어떤 증거가 있는지 분류할 수 있다. 전개 경로만
있어도 초기 진단 baseline은 가능하지만, 추가 비용을 배제하지 못한 경로의
결과로 intrinsic hardware 한계를 단정하지 않는다.

### 6. 병목 가설을 counter와 통제 실험으로 확인한다

| 가설 | 확인할 관측 | 다른 조건을 고정한 대조 실험 |
| --- | --- | --- |
| launch/parameter/synchronization 비용 | total 대비 params·gap·host response 비중 | batch 증가, task/단계 묶음 개선, launch cost만 바꾼 보조 config |
| unit을 충분히 사용하지 못함 | unit별 active 시간·idle·불균형, round 수 | batch와 mapping을 바꾸고 실제 활성 unit 증가 확인 |
| scratchpad service 또는 협력 비용 | SPAD resource wait, stage/wave 시간, 접근 횟수 | worker/wave 배치 변경. bank/latency knob가 timing 모델에 반영되는지 확인 후 하나씩 변경 |
| DRAM/strided/transpose 비용 | measured transaction·BW·cache miss, transpose 시간 | fusion/주소 layout 대조와 memory bandwidth sweep |
| arithmetic 또는 dependency 비용 | FP/vector issue, dependency wait, lane 활용 | memory 조건을 고정하고 NDP compute clock/실행 자원 변경. emitted RVV 확인 |
| register 압박 | emitted spill load/store와 register 제한 | worker당 radix/할당 변경. spill 제거 전후 비교 |

launch 비용을 제거한 config는 민감도 확인용이다. 이를 실제 GPU와 비교하는
주 baseline으로 바꾸지 않는다. unit 수·memory/compute clock 등을 동시에
바꾸면 원인 판별이 어려우므로 한 변수씩 바꾼다. counter만으로 특정 병목을
확정하지 않고 해당 자원 변화에 대한 실행 시간의 반응을 함께 본다.

## 4. 최초 실험 범위

| 범위 | 시작할 case | 목적 |
| --- | --- | --- |
| 작은 single-pass | N=16,64,128,256 | axis/batch 결정, launch 비중 |
| 큰 single-pass | N=1024,4096 및 실제 target의 capacity 경계 | register·scratchpad·wave 압박 |
| mixed-radix | N=20,30,60,120,216,960 중 선택 library가 지원하는 범위 | precision 표와 optimizer·radix 순서 |
| multi-pass | N=8192,16384,32768,65536 및 실제 planner 분기 양옆 | split·transpose·twiddle·fusion 비용 |
| batch | B=1,8,32,128과 saturation 이후의 B | launch amortization과 unit/worker 활용 |

모든 조합을 처음부터 돌릴 필요는 없다. 먼저 각 mechanism을 대표하는
case를 통과시키고 범위를 늘린다. GPU 원본이 선택했지만 현재 M2NDP codegen이
지원하지 않는 plan은 빠진 coverage로 보고한다. 성공 case만 평균에 넣어서
전체 지원 범위의 성능처럼 표시하지 않는다.

평균 대역폭·FP32 throughput·scratchpad service·empty launch·params copy에
대한 microbenchmark도 확보한다. 28 units와 clock을 GPU에 맞췄다는 사실만으로
simulator의 arithmetic/memory/launch 모델이 실제 GPU와 성능상 대응한다고
볼 수는 없다. 결론은 선택한 M2NDP simulator model에 대한 결론이며, calibration
수준과 모델이 구현하지 않는 효과를 명시한다.

## 5. “baseline으로 충분하다”의 완료 판정

**초기 병목 진단 baseline**은 단계 0~4가 충족되면 사용할 수 있다. 선택한
library·backend·FFT 범위가 명시되고, 원본 planner의 결정이 검증되며,
M2NDP용 변경이 기록되고, actual binary 정답과 같은 경계의 실제 GPU/simulated
M2NDP 시간이 있어야 한다. 추가 transpose/fusion 해제 비용이 있으면 결과에
명시한다. 이 단계에서는 현재 포팅 구현의 병목을 찾을 수 있다.

**M2NDP 구조에 병목을 귀속할 baseline**은 여기에 단계 5~6을 더한다.
지배적인 mapping 추가 비용을 측정·통제하고, model의 관련 resource가
실제로 timing에 반영되는지 확인하고, counter와 resource 민감도가 같은
설명을 지지해야 한다. 전체 속도 비율만으로 architecture 병목을 주장하지 않는다.

모든 library, prime FFT, R2C/C2R, multidimensional FFT, 모든 GPU intrinsic을
포팅하는 것은 이번 범위를 완성하기 위한 선행 조건이 아니다. 한 library의
명확한 범위를 위 기준으로 완성하는 것이 우선이다. Native planner의 최고
성능 튜닝은 그 이후 개선 폭을 확인하는 실험으로 진행한다.

## 6. 외부 측정 근거

- GPU FFT 전체의 device 시간은 같은 stream의 event로 측정하고 async 완료를
  확인한다: [CUDA Best Practices — Timing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#timing).
- cuFFT는 plan 초기화 때 JIT를 수행할 수 있으므로 steady-state 실행과 plan
  시간을 분리한다: [cuFFT — Plan Initialization](https://docs.nvidia.com/cuda/cufft/index.html).
- Nsight Compute의 replay/profiling overhead가 실행 시간에 영향을 주므로
  timing run과 counter 수집 run을 나눈다: [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

M2NDP RTX3060 config의 수치와 SAXPY 예시는 사용자가 제공한 실행 안내를
기준으로 적었다. 이번 검토에서 Docker image를 pull/run하거나 실제 GPU/FFT
simulation을 실행하지 않았으며 새 FFT 병목 측정값을 산출하지 않았다.
