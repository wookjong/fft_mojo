현재 radix × execution joint search 구현까지 완료되었다.

다음 단계에서는 새로운 search axis를 추가하지 말고,
현재 joint search가 실제 성능 차이를 제대로 탐색하고 있는지
measurement-driven validation을 먼저 수행하라.

중요:
이번 단계의 목적은 planner architecture를 더 확장하는 것이 아니라

1. candidate knob가 실제 generated plan/code에 영향을 주는지
2. analytical cost model이 measured cycle의 상대적인 순서를 예측하는지
3. joint search가 baseline보다 실제로 더 좋은 plan을 찾을 수 있는지

를 검증하는 것이다.


==================================================
Phase A. Search knob semantic validation
==================================================

현재 representative example에서

workers=(None,2)
workers=(2,2)
workers=(2,None)

후보들의 estimated cost가 동일했고,
measured cycle도 거의 동일했다.

이 결과를 단순히 "cooperative가 이득 없음"이라고 결론내리지 말고,
먼저 worker configuration이 실제 generated plan/code를
의미 있게 변경하고 있는지 검증하라.

각 candidate에 대해 다음을 비교하라.

- leaf별 radix sequence
- workers_per_fft
- cooperative/non-cooperative mode
- scratchpad allocation
- logical block count
- work assignment
- synchronization/barrier insertion
- generated kernel structure
- generated Mojo/IR의 중요한 차이
- static instruction count 가능하면
- spill status

특히 다음 질문에 답하라.

1. `(None,2)`와 `(2,2)`가 정말 서로 다른 실행을 하는가?
2. worker=2가 지정됐지만 leaf 조건 때문에 cooperative path가
   실질적으로 사용되지 않는 경우가 있는가?
3. plan field는 다르지만 codegen 결과가 동일해지는 "dead knob"가 있는가?
4. 그런 candidate가 있다면 search space에서 early reject 또는 canonicalize할 수 있는가?

단순 textual code diff뿐 아니라,
가능하면 normalized/generated-kernel signature를 만들어
실질적으로 같은 code를 생성하는 candidate를 탐지하라.

plan-level dedup과 code-level equivalence는 다른 문제이므로 구분하라.


==================================================
Phase B. Representative benchmark sweep
==================================================

한 N=960 결과만으로 판단하지 말고
다양한 FFT size에서 joint candidate를 실제 측정하라.

가능한 범위에서 다음 유형을 포함하라.

- small FFT
- medium FFT
- large FFT
- single-leaf plan
- multi-leaf plan
- radix-2 중심
- mixed radix
- cooperative가 유리할 가능성이 있는 큰 leaf
- cooperative overhead가 더 클 가능성이 있는 작은 leaf

예를 들면 현재 regression suite 및 지원 범위 안에서
10~20개의 representative N을 선정한다.

각 N에 대해:

1. candidate 생성
2. analytical estimated cost 계산
3. top candidates 및 baseline 실제 simulator cycle 측정
4. spill 여부 기록

결과를 CSV/JSON 또는 Python structure로 남겨
후속 분석이 가능하게 하라.


==================================================
Phase C. Cost model prediction quality 분석
==================================================

중요한 것은 absolute cycle prediction이 아니라
candidate ranking의 품질이다.

각 FFT N에 대해 다음을 계산하라.

- estimated-cost ranking
- measured-cycle ranking
- Spearman rank correlation
- Kendall rank correlation 가능하면
- analytical top-1 candidate의 measured rank
- analytical top-K 안에 measured best가 포함되는지
- baseline cycle
- measured best cycle
- baseline 대비 improvement
- analytical 선택 candidate와 oracle best 사이의 regret

regret은 예를 들어

    (selected_cycles - best_cycles) / best_cycles

로 계산할 수 있다.

특히 다음을 따로 분석하라.

- radix 변화에 대한 prediction
- worker 변화에 대한 prediction
- radix × worker interaction에 대한 prediction

현재 worker configuration이 달라도 estimated cost가 동일해지는
경우가 얼마나 자주 발생하는지도 측정하라.


==================================================
Phase D. Cost component breakdown
==================================================

이미 cost model이

- memory
- compute
- resource
- execution

으로 분리되었으므로 이를 적극 활용하라.

대표 candidate에 대해 다음 형태의 breakdown을 출력하라.

candidate A:
    memory_cost    = ...
    compute_cost   = ...
    resource_cost  = ...
    execution_cost = ...
    total          = ...
    measured_cycle = ...

candidate B:
    ...

그리고 다음을 확인하라.

worker 수가 바뀌었는데도 total cost가 동일하다면

- execution_cost가 실제로 동일한 이유가 무엇인지
- worker utilization
- synchronization
- parallelism
- cooperative overhead

중 어떤 정보가 model에 빠져 있는지 분석하라.

단, measurement와 맞추기 위해 임의의 magic weight를 바로 추가하지 마라.

먼저 모델에 "정보 자체가 없는 것"과
"정보는 있는데 weight가 부정확한 것"을 구분하라.


==================================================
Phase E. Joint search effectiveness
==================================================

각 N에 대해 최소한 다음 세 가지를 비교하라.

A. 기존 baseline planner
B. radix-only / worker-only staged search
C. radix × worker joint search

각 방식이 최종적으로 선택한 plan을 실제 cycle로 비교하라.

목표는

"joint search candidate가 생성된다"

가 아니라

"joint search가 독립적인 staged search에서는 찾지 못하는
더 좋은 조합을 실제로 발견하는 경우가 있는가?"

를 확인하는 것이다.

가능하면 다음과 같은 실제 사례를 찾아라.

radix A + worker 1
radix B + worker 1
radix A + worker 4
radix B + worker 4

중

radix-only search와 worker-only search로는 선택되지 않지만

    radix B + worker 4

같은 interaction 조합이 실제 measured best가 되는 사례.

그런 사례가 없다면 그것 역시 중요한 결과이므로
억지로 만들지 말고 정직하게 보고하라.


==================================================
Phase F. Top-K autotuning 효과 분석
==================================================

기존의

probe_and_rerank_candidates(rank_by_cycles=True)

를 사용하여 analytical top-K만 실측하는 구조가
얼마나 효과적인지 분석하라.

K = 1, 3, 5, 10 등에서

- oracle best 발견 비율
- simulator 측정 횟수
- final regret

을 비교하라.

이를 통해

cheap analytical model
        ↓
top-K pruning
        ↓
actual cycle measurement

구조가 실제로 합리적인지 확인하라.


==================================================
이번 단계에서 하지 말 것
==================================================

아직 다음은 구현하지 마라.

- split × radix × tile × worker 전체 joint search
- compute_lanes plan 승격
- 새로운 cost weight를 무작정 추가
- planner 대규모 rewrite

이번 단계는 현재 구현된 search의
"검증과 characterization"에 집중한다.


==================================================
최종 보고 형식
==================================================

최종적으로 다음을 보고하라.

1. worker/radix knob가 실제 code에 미치는 영향
2. dead/no-op candidate 존재 여부
3. FFT size별 candidate 수
4. estimated vs measured correlation
5. analytical top-K의 oracle-best coverage
6. baseline vs staged vs joint search 성능
7. joint search가 실제 이득을 낸 concrete example
8. joint interaction이 중요하지 않았던 경우
9. cost model이 놓치고 있는 정보
10. 다음 단계에서 무엇을 수정하는 것이 가장 가치 있는지

마지막에는 다음 세 옵션 중 무엇을 다음 작업으로 추천하는지
측정 결과를 근거로 판단하라.

A. cost model 보정
B. split × radix × tile × worker bounded joint search
C. compute_lanes를 plan/search axis로 승격
