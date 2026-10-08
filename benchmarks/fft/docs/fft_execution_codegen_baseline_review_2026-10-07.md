> 2026-10-08: ? ??? ?? ? ?? ?????. ?? ??? ??? ??? ??? [?? ???](fft_execution_codegen_fixes_2026-10-08.md)? ??????. ?? JSON fixture? ?? ??? ????, audit ????? ?? ??? ???? ? fixture? ?????.

# FFT execution strategy·code generation의 현재 상태와 수정 순서

검토일: 2026-10-07. 범위는 `planning/execution`, `codegen`, 이들이 사용하는
plan 구조와 검증 코드다. 환경 구성, GPU library planner 정책, 실제 GPU 측정은
이번 검토 범위에 포함하지 않았다. 구현을 변경하지 않고 재현 audit와 문서를 추가했다.

## 판정

Non-cooperative, cooperative, persistent 및 recursive leaf별 혼합 실행의
기본 구현은 있다. Persistent의 wave/fused/physical과 scalar/vectorized copy,
cooperative partition 세 방식, tail hybrid도 구현되어 있다.

이전 execution 검토의 다섯 항목은 현재 코드에서 수정되었다.

| 이전 항목 | 현재 코드 |
| --- | --- |
| 등록 함수 수 제한 | stage tail/preload/writeback을 포함한 공통 inventory와 renderer 검사가 있음 |
| recursive persistent target 누락 | recursive leaf까지 target이 전달됨 |
| cooperative W=3/16 허용 | interleave chunk의 양의 정수 약수인지 검사함 |
| persistent mode/copy가 metadata에만 존재 | 기본 codegen이 leaf의 선택을 따르고 explicit override를 별도 허용함 |
| launch granule과 arithmetic lane 혼동 | spreading 경로의 granule validation이 앞단에 연결됨 |

따라서 이 다섯 항목을 미구현 목록으로 다시 제시하지 않는다. 다만 아래의
공유 상태·batch launch·unit 활용 문제는 추가로 처리해야 한다.

## 1. [P1] Persistent tracker의 같은 phase 내 읽기·갱신을 분리해야 한다

관련 코드:
[round read](../codegen/fft_persistent_codegen.py#L433),
[round write](../codegen/fft_persistent_codegen.py#L736),
[wave read/write](../codegen/fft_persistent_codegen.py#L405),
[oracle snapshot](../verification/verify_fft_persistent.py#L81).

`round_tracker`는 unit당 한 공유 cell이다. 모든 worker가 phase 시작 부분에서
읽지만 writeback의 worker 0이 같은 cell을 해당 phase 안에서 갱신한다.
`wave_tracker`도 wave stage 안에서 같은 형태로 읽고 갱신된다.

현재 runtime 소스의 `launch_parallel` 계약은 모든 µthread가 retire한 뒤
다음 phase로 진행하는 것이다. 이는 phase 사이의 완료 순서를 보장하지만,
같은 phase에서 다른 worker의 read가 worker 0의 store보다 먼저 수행된다는
보장은 아니다. 해당 계약은 `src/m2ndp.mojo:149`에 명시되어 있다.

검증 harness의 `_FrozenReadPtr`는 모든 worker가 phase 시작 시점의 tracker
값을 읽도록 강제한다. 코드 주석에 이전 실행에서 이 패턴이 동작했다는
기록은 있으나, 재현 가능한 schedule 보장이나 snapshot memory 계약은 없다.
따라서 현재 수치 검증은 일반적인 공유-memory interleaving의 안전성을 검증하지 않는다.

추가 재현은 emitted phase 코드를 그대로 사용하면서 tracker read만 ordinary
shared read로 바꾸었다. 다음 값은 Python ordering model 결과다.

| 재현 | 기존 snapshot oracle | ordinary shared read |
| --- | --- | --- |
| N=64, B=1, W=16, physical; worker 0부터 순차 실행 | 64개 출력, max error 약 2.9e-7 | 8개만 nonzero, max error 약 655.4 |
| 같은 입력, fused/wave | 정상 | 같은 round writeback 누락이 발생 |
| N=1024, B=1, W=16, wave; round는 snapshot 유지, wave만 shared read. 인접 phase에서 worker 0/1 순서를 교대 | 정상 | max error 약 6.19e4 |

이 재현은 **현재 default simulator에서 실제 failure가 관측되었다는 뜻이
아니다.** 구현이 worker 순서에 독립적이라는 주장에 대한 counterexample이다.
같은 phase에 있는 µthread 실행 순서의 가정을 제거하는 것이 우선이다.

수정 방향:

- round/wave 상태를 physical worker별 private cell로 두고 각 worker가 자신의
  cell만 읽고 갱신하게 하거나, state update를 다른 worker가 읽는 phase와 분리한다.
- private wave 상태는 ragged wave의 inactive logical worker에 해당하는 physical
  worker도 다음 wave로 진행하도록 처리한다.
- 별도 phase를 추가한다면 등록 함수 수·실제 launch 수·동기화 비용에 반영한다.
  private cell을 늘린다면 scratchpad footprint에 반영한다.
- phase마다 worker 순서를 바꾸고 읽기·쓰기를 교차하는 검증을 추가한다.
  tracker snapshot을 correctness의 선행 가정으로 요구하지 않게 한다.

완료 기준: physical/fused/wave 모두 B=1·부분 round·여러 round에서 tracker가
각 worker의 처리 대상과 일치하고, 허용된 실행 순서 변화에 따라 출력이 달라지지 않는다.

## 2. [P1] Standalone plain/cooperative도 resource cap에 맞게 batch를 launch해야 한다

관련 코드:
[plain standalone](../codegen/fft_codegen.py#L918),
[cooperative standalone](../codegen/fft_cooperative_codegen.py#L288).

Factory는 scratchpad cap에 따라 `max_uthread`를 줄일 수 있지만 두 standalone
host generator는 전체 pool을 한 번에 launch한다. Stage에는 local ID/FFT slot
guard가 있어 한 unit에 cap보다 많은 µthread가 배치되면 일부 FFT를 건너뛴다.

N=64, radices=(4,4,4), B=64, spad capacity=1024 bytes를 사용하면:

| 전략 | total µthreads / local cap | 생성한 host launch | aligned pool의 현재 mapping과 guard를 적용한 coverage |
| --- | --- | --- | --- |
| Plain | 64 / 1 | 전체 pool 1회 | FFT 8개 처리 대상, 56개 누락 |
| Cooperative W=8 | 512 / 8 | 전체 pool 1회 | FFT 32개 처리 대상, 32개 누락 |

단순히 `total_uthreads > max_uthread`라는 조건만으로 모든 launch가 잘못된 것은
아니다. 여러 units에 배치되어 각 unit의 cap을 지키면 안전할 수 있다. 위 값은
32 units·8-µthread chunk·chunk-aligned pool에서 unit별 local ID를 적용한
구체적인 counterexample이다. 실제 simulator 실행값은 아니다.

Recursive codegen과 최근 hybrid tail host에는 round 분할이 있다. 이 문제를
recursive 경로에서도 같은 batch 누락이 발생한다는 주장으로 확대하지 않는다.

수정 방향: standalone/recursive/hybrid가 같은 launch descriptor를 사용하도록
한다. Descriptor에 round width, logical FFT 범위, pool alignment, worker grouping,
input/output pointer offset을 확정한다. 미지원 standalone 조합은 생성 전에 거부한다.

완료 기준: cap 이하·경계·cap 초과·마지막 부분 batch의 모든 FFT가 정확히 한 번
처리되고, host launch와 device의 ID 해석이 같은 grouping을 사용한다.

## 3. [P2, 병목 비교 전에 필요] Cooperative round가 unit 수를 활용하도록 한다

관련 코드: [round selection](../codegen/fft_transpose_codegen.py#L671).

`_stage_round_size`는 cooperative에 대해서 `spread_across_units=True`여도
`stage.max_uthread`를 그대로 반환한다. 그 값은 한 unit의 capacity에서 얻은
값인데, 현재 host launch의 전체 width로 사용된다.

기본 target, N=1024, radices=(8,8,16), B=128에서는 FFT slots/unit cap=4다.

| W | round µthreads | FFT/host launch | host launches | aligned pool의 사용 units/round |
| --- | --- | --- | --- | --- |
| 2 | 8 | 4 | 32 | 1 / 32 |
| 8 | 32 | 4 | 32 | 4 / 32 |

이는 현재 codegen의 정적 launch/mapping 결과다. 높은 launch 비용과 낮은
unit 활용을 만들 수 있지만, 그 실행 시간 비중을 이번에 측정한 것은 아니다.
이 결과로 전체 M2NDP의 parallelism 한계를 주장하면 executor 정책의 영향을
hardware 영향과 혼동하게 된다.

수정 방향:

- unit별 cap과 전체 launch width를 다른 값으로 둔다.
- cooperative worker group이 한 unit 안에 있고 slot 한도를 지키도록 전체
  round를 구성한다. chunk보다 작은 local cap에서는 단순 배수 확대 대신
  padding/sparse 배치와 logical FFT ID 압축 mapping 등을 함께 검토한다.
- plain의 local cap이 chunk보다 작은 경우도 unit 분산의 같은 제약을 확인한다.
- unit별 실제 배정되는 logical FFT와 µthread 수를 descriptor에서 계산한다.
  유효 worker·dummy worker·idle unit을 별도로 보고한다.

완료 기준: 충분한 B에서 unit 활용이 batch와 함께 늘고, 모든 slot의 scratchpad
주소가 분리되며 FFT group이 unit 경계를 넘지 않는다. 보수적인 단일-unit 실행을
유지한다면 그 제한을 baseline 범위에 명시한다.

## 4. [P2] Target alignment와 vector copy width의 지원 계약을 일관되게 한다

Standalone persistent 및 hybrid의 persistent host block은 pool을 256B로
고정 정렬한다. 반면 persistent factory는 coherent custom target을 받고,
recursive/cooperative host는 target에 따라 alignment를 계산한다.

`interleave_chunk_uthreads=16`, `mapping_stride_bytes=512`, `uthread_bytes=32`
target으로 만든 plan도 standalone source는 `(raw_addr+255)//256*256`을
출력한다. 이 주소가 512B boundary에 있다는 보장은 없다.

관련 코드: [standalone alignment](../codegen/fft_persistent_codegen.py#L963),
[hybrid alignment](../codegen/fft_persistent_codegen.py#L1040).

또한 vectorized copy의 `vector_width=0`과 `-1`이 거부되지 않고 load width,
loop 조건, increment에 그대로 들어간다. 이는 유효한 실행 옵션이 아니다.
어떤 값은 compilation 단계에서 실패할 수도 있으므로 모든 경우의 runtime
failure 형태를 단정하지 않는다.

관련 코드: [copy emitter](../codegen/fft_persistent_codegen.py#L607).

수정 방향: 공통 pool allocation/alignment emitter와 옵션 validation을 사용한다.
copy width는 지원하는 양의 정수 값으로 제한한다. target 변경을 지원하지 않는
경로는 그 조합을 명시적으로 거부한다.

완료 기준: plan이 허용한 target/width를 codegen도 같은 의미로 처리하고,
invalid 값은 소스·파일을 생성하기 전에 거부한다.

## 5. Baseline에 필요한 execution 기록과 그다음 최적화

앞 항목을 해결한 다음 각 leaf의 실제 실행 조건을 하나로 확정하여 생성 코드와
보고가 공유하게 한다. `ResolvedExecution` 같은 구조에 다음 값을 둘 수 있다.

- logical worker, physical µthread/worker grouping, partition, stage별 arithmetic lanes.
- persistent mode/copy/vector width와 explicit override의 적용 결과.
- host task/round 수, distinct registered 함수, stage/tail/preload/writeback 호출 수.
- unit별 FFT/µthread 배정, scratchpad buffer와 tracker의 allocation footprint.
- 단계별 DRAM read/write, scratchpad read/write, 별도 transpose/twiddle 작업.

현재 plan mode/copy 기본값 연결은 수정되어 있다. Explicit codegen override가
plan metadata를 바꾸지 않는 것은 문서화된 설계다. A/B 비교에서는 원래 plan과
함께 resolved 실행 조건도 기록해야 어떤 코드를 비교했는지 재현할 수 있다.

정답·coverage·unit grouping·실행 조건이 확정된 뒤 vectorized preload/writeback,
physical/fused/wave, partition, tail hybrid를 같은 FFT plan에서 대조한다.
Transpose/large-twiddle를 leaf load/store와 통합하는 최적화도 실제 추가 pass와
stage 호출 수를 보고하면서 진행한다. 모든 fusion을 완성하는 것은 초기
execution baseline의 선행 조건이 아니지만 추가 비용의 가시성은 필요하다.

권장 수정 순서:

1. Persistent 공유 tracker의 phase 내 ordering 의존성 제거.
2. Standalone/recursive/hybrid의 batch coverage와 round descriptor 통일.
3. Cooperative 및 작은 local-cap plain 경로의 여러-unit 배치.
4. 공통 alignment·옵션 validation·실제 실행 조건 기록.
5. Copy/partition/mode/fusion 최적화를 한 변수씩 비교.

코드 기준의 완료 판정은 모든 선언된 지원 조합에서 정답, exactly-once batch
coverage, scratchpad 소유권, stage 의존성, registration/resource 제약이 유지되고,
plan 선택과 생성된 실행 조건이 일치하는 것이다.

## 이번 검증

- Registration, recursive target, cooperative worker, persistent option,
  launch support: unittest 35개 통과.
- Execution invariants, cooperative partition, persistent copy, tail hybrid:
  각 executable suite 통과.
- 추가 audit: tracker ordering 12개, wave-only ordering 2개, standalone coverage
  2개, cooperative rounds 2개 및 alignment/invalid width 재현.

[audit script](../verification/audit_fft_execution_codegen.py)와
[결과 JSON](../verification/fixtures/fft_execution_codegen_review_2026-10-07.json)을
추가했다. Source SHA256를 함께 기록했다.

```powershell
python -m verification.audit_fft_execution_codegen
```

검증은 generated-body Python 실행, 소스 구조 및 주소 grouping model이다.
Mojo compilation, simulator 실행, 성능 측정은 이번에 수행하지 않았다.
