# GPU FFT의 NDP unit 병렬 실행

## 기존 구현 확인

이미 구현된 동기화·교환 경로를 확인한 뒤 재사용했다.

- `src/m2ndp.mojo`의 `launch_parallel`: 모든 microthread가 retire한 뒤 반환하는 kernel completion barrier.
- `planning/strategies/fft_plan_recursive.py`: PRE → near leaf → MIDDLE/twiddle → far leaf → POST의 six-step 분해.
- `codegen/fft_transpose_codegen.py`: 단계 사이 CXL real/imag 버퍼와 tiled transpose.
- `planning/execution/launch_plan.py`, `codegen/common.py`: 256-byte 정렬, pool 확대, padded launch와 bounds로 여러 독립 leaf replica를 unit에 배정.
- `planning/execution/fft_plan_persistent.py`: software group당 한 unit, batch/leaf replicas를 group과 round에 배정.

기존 GPU worker-wave lowering은 독립 FFT replica만 여러 unit에 나눴다. Batch 1의 단일 leaf에서는 group 0만 진행하므로 한 unit이 FFT 전체를 실행했다.

## 추가 실행 모드

`--gpu-execution`은 다음 중 하나다. 기본값은 기존 `worker-wave`다.

| 모드 | 작업 배정 | Unit 간 교환·동기화 | GPU 계획과 관계 |
|---|---|---|---|
| `worker-wave` | 한 FFT replica를 한 unit의 8개 worker에 배정 | 기존 local scratchpad 및 phase completion | 기존 실행 방식 |
| `distributed` | leaf 내부 butterfly SIMD chunk를 여러 unit에 배정 | shared CXL SoA ping-pong banks + radix stage마다 synchronous `launch_parallel` | 기존 leaf 경계·radix·주소 mapping·twiddle·scale 유지, worker 배정 변경 |
| `split-leaves` | 작은 leaf들의 독립 replica를 기존 cooperative/persistent group에 배정 | 기존 recursive CXL transpose와 task completion | 분해 정책 변경, 각 leaf의 GPU planner 선택 사용 |

### Distributed

`planning/execution/fft_unit_parallel.py::distribute_leaf`가 기존 GPU leaf를 공통 radix/address plan으로 다시 낮추고 중간 scratchpad bank를 CXL workspace로 바꾼다. 각 bank에는 replica마다 `2 * length` Float32 영역이 있으며 real/imag를 분리해 저장한다. 다른 replica나 ping-pong bank와 겹치지 않는다.

기본 배정은 `work_slot = gid // workers_per_unit + (gid % workers_per_unit) * num_units`다. 첫 작업들이 같은 unit의 연속 8개 thread에 몰리지 않고 서로 다른 unit의 첫 lane에 배정된다. 전체 launch는 `num_units * workers_per_unit`이고, 작업 수가 이보다 많으면 같은 worker가 stride로 처리한다. 작업이 없는 worker는 값을 읽거나 쓰지 않는다.

각 radix stage에서 모든 output element에 writer가 하나만 존재한다. Stage 종료를 기다린 뒤 다음 stage가 이전 bank를 읽는다. 별도의 spin barrier, atomic wait loop 또는 다른 unit의 private scratchpad를 읽는 코드가 없다. Hardware API에서 제공하는 kernel completion을 전역 barrier로 사용한다.

기존 `codegen/lowering.py`의 주소 패턴 검증 및 loop-twiddle table을 재사용한다. Loop로 증명할 수 없는 stage나 partial tail은 기존 정확한 batch/load/store emitter를 사용한다. Non-power-of-two loop period는 unsupported `mulhsu` lowering을 피하도록 unroll한다. 중간 CXL 교환으로 array traffic은 stage당 `16 * total_elems` bytes이며 cache/spill/twiddle traffic은 별도다.

### Leaf당 unit 수와 batch 배정 (2026-10-16)

`--units-per-leaf U`는 physical target을 바꾸지 않고 unit들을 U개씩 묶는다. 실제 28-unit target에서는 launch가 항상 224 microthread이고 decoder 주기도 28이다. `floor(28/U)`개의 그룹이 서로 다른 FFT replica를 동시에 처리하고, 각 그룹은 replica 번호를 그룹 수만큼 증가시키며 다음 FFT를 처리한다. 그룹 안에서는 unit을 먼저 순회하며 butterfly chunk를 배정한다. 예를 들어 U=8이면 세 그룹이 24개 unit을 쓰고 나머지 4개 unit은 계산하지 않는다. U=14, batch=1이면 14개 unit만 계산하며, batch>=2이면 두 FFT가 각각 14개 unit을 사용한다. Unit 번호는 정렬된 pool 시작 주소에 대한 상대 번호다.

생략하면 기존처럼 leaf 하나가 모든 physical unit을 사용한다. `auto`는 `max(1, physical_units // min(replicas, physical_units))`의 batch 기반 휴리스틱이다. 실측 latency 최적값을 선택하는 기능은 아니며, 나머지 unit을 가변 크기 그룹으로 재분배하지 않는다. GPU leaf 경계와 radix, boundary mapping, twiddle, normalization은 바꾸지 않는다.

### 안전한 두 stage 묶기 (2026-10-16)

`--fuse-stages 2`는 실제 SIMD chunk의 load/store 주소로 dependency component를 계산한다. 두 번째 stage의 모든 입력 producer를 같은 component로 묶고, ping-pong 입력 bank를 최종 출력이 덮어쓰는 경우 그 값을 먼저 읽어야 하는 작업까지 함께 묶는다. 서로 다른 component 사이에 RAW/WAR 의존성이 남으면 안 된다.

각 component는 한 worker가 첫 stage와 두 번째 stage 순서로 실행한다. 중간 결과만 그 worker의 private scratchpad slice에 기록하고, phase 경계에서는 기존 CXL workspace를 사용한다. 다른 worker나 unit의 scratchpad를 읽지 않는다. Component의 원래 offset을 조밀한 local offset으로 바꾸되 boundary 주소, twiddle 및 scale을 유지한다.

Component 크기가 `--fusion-max-elements`(기본 256 complex elements)를 넘거나, unit당 8 worker의 합산 local 공간이 target budget을 넘거나, 독립 component 수가 leaf당 unit 수보다 적으면 해당 stage 쌍을 묶지 않는다. 따라서 `--fuse-stages 2`가 모든 stage의 병합을 보장하지 않는다. 현재는 인접한 두 stage만 지원하고 기본값은 1이다. Scratchpad 왕복, 코드 크기와 register pressure가 늘어날 수 있으므로 성능은 측정해야 한다.

`FFT_EXECUTION` 주석은 실제 phase의 stage ID, unit 그룹 수, local scratchpad bytes, phase 수 기준 array traffic 추정을 기록한다. `estimated_dram_bytes` plan 속성은 fusion 전 상한이며, 실제 phase의 추정치는 `estimated_array_bytes`다. Twiddle/cache/spill traffic은 두 추정치에 포함하지 않는다. 등록 kernel 수 검사는 실제 phase 이름과 동일한 SIMD 정책을 사용한다.

### Split leaves

`make_split_leaf_plan`은 기존 recursive topology를 지정한 leaf byte budget으로 생성한 뒤, 각 leaf를 선택한 GPU planner의 `plan`/`plan_m2ndp`로 다시 계획한다. GPU planner가 선택한 radix와 cooperative/worker-wave 실행을 그대로 사용하고, 여러 leaf replica는 기존 launch engine으로 unit에 배정한다. Near/far dependency는 기존 transpose/task 완료 순서로 유지한다.

각 GPU leaf가 inverse FFT를 자체 정규화하는 경우 그 scale을 enclosing recursive plan의 scale로 교체한다. Inverse normalization은 전체 FFT에서 한 번만 적용한다. 변경된 최상위 분해는 원래 GPU baseline과 구분해 기록한다. Leaf planner가 선택한 결과를 숨기거나 실패 시 native leaf로 바꾸지 않는다.

## 실행

FFT checkout에서 생성:

```powershell
python benchmarks/fft/make_fft_kernel.py 1024 --planner gpu-clfft-m2ndp --gpu-execution distributed -o distributed.mojo
python benchmarks/fft/make_fft_kernel.py 1024 --planner gpu-clfft-m2ndp --gpu-execution distributed --units-per-leaf 14 --fuse-stages 2 -o shared.mojo
python benchmarks/fft/make_fft_kernel.py 1024 --planner gpu-clfft-m2ndp --gpu-execution split-leaves --split-leaf-byte-budget 1024 -o split.mojo
```

Workspace 루트에서 기존 고정 이미지/harness로 새 실험:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "& './experiments/m2ndp_fft/scripts/run_benchmark.ps1' -Planner gpu-clfft-m2ndp -GpuExecution distributed -Sizes 256,1024,4096 -ExperimentId clfft-distributed"
powershell -NoProfile -ExecutionPolicy Bypass -Command "& './experiments/m2ndp_fft/scripts/run_benchmark.ps1' -Planner gpu-clfft-m2ndp -GpuExecution distributed -UnitsPerLeaf 14 -FuseStages 2 -Sizes 1024 -ExperimentId clfft-shared-u14-f2"
powershell -NoProfile -ExecutionPolicy Bypass -Command "& './experiments/m2ndp_fft/scripts/run_benchmark.ps1' -Planner gpu-clfft-m2ndp -GpuExecution split-leaves -SplitLeafByteBudget 1024 -Sizes 256,1024,4096 -ExperimentId clfft-split-leaves"
```

직접 생성 CLI의 기본 physical target은 32 units다. 실제 RTX 3060 대응 NDP 비교에는 위 benchmark wrapper를 사용한다. Wrapper는 `M2NDP_RTX3060` config에서 실제 28-unit target을 읽는다. `UnitsPerLeaf`는 physical target 설정의 대체 수단이 아니다. Python API에서는 실제 target을 planning과 codegen에 동일하게 전달해야 한다.

Harness는 execution mode, leaf budget, units per leaf, fusion 설정, 원래 GPU plan 및 split mode의 각 leaf GPU config를 보존하며 다른 설정으로 resume하는 것을 거부한다. `experiments/m2ndp_fft/scripts/analyze_leaf_sharing.py`는 저장된 입력 bytes와 config/toolchain/SIMD/GPU 계획·분해가 같은지 확인하고, 별도 검증과 timing 반복이 모두 성공한 case만 비교한다.

## 검증

`verification/verify_fft_unit_parallel.py`는 실제 emitted stage 코드를 Python으로 번역해 실행한다. Worker 순서를 stage마다 섞고, shared workspace를 NaN으로 초기화하며, 범위 밖 load/store와 중복 writer를 검사한다. 세 GPU planner의 forward/inverse, batch, ragged stage, recursive exchange 및 arithmetic width를 NumPy와 비교한다. Split mode의 inverse normalization도 확인한다.

Fused phase 검증에서는 physical unit마다 private scratchpad를 따로 만들고 초기화되지 않은 load도 거부한다. 28-unit 배정, U=8의 나머지 unit, `auto`의 batch 분배, bank alias 의존성과 recursive inverse를 포함한다. Python 재실행은 실제 FP32 머신 코드 검증을 대신하지 않는다. `validate_leaf_sharing.py --generate`와 이미지 안의 `--run`으로 실제 빌드·시뮬레이션을 수행하고 `--recheck`로 완성된 통계를 재확인할 수 있다. 초기 보고서는 별도 보존한다. Stats reader는 28개 unit의 카운터가 모두 기록되기 전에 참여 수를 판정하지 않는다.

실제 고정 이미지 검증은 `experiments/m2ndp_fft/scripts/validate_unit_parallel.py`로 생성/실행한다. Generated Mojo, plan, source hash, build/run 로그와 simulator stats를 `experiments/m2ndp_fft/results/20261012-unit-parallel`에 저장한다. 결과는 그 디렉터리의 보고서를 참조한다.

이 모드는 실행 선택지이며 자동 cost-based 선택은 추가하지 않았다. Vector issue가 발생한 unit 수는 전체 pipeline 참여 지표이고 시간 가중 utilization은 아니다. 작은 FFT는 butterfly 작업 수 때문에 28개 unit을 모두 쓰지 않을 수 있다.
