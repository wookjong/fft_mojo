현재 M2NDP용 FFT planner를 개선하려고 한다.

목표는 FFT plan 생성 과정을 아래 4단계로 명확히 분리하되,
특히 3번(radix/stage fusion)과 4번(execution strategy)은 독립적으로 결정하지 말고
필요하면 joint search가 가능하도록 설계하는 것이다.

중요:
- 우선 현재 코드베이스의 planner / plan search / cost model / codegen 구조를 충분히 읽어라.
- 기존 구조를 무시하고 새 planner를 처음부터 작성하지 마라.
- 현재 numerical correctness를 깨뜨리지 마라.
- plan generation이 모든 실행 결정을 담당해야 한다.
- codegen은 이미 결정된 plan을 그대로 코드로 변환하는 역할만 해야 한다.
- heuristic을 임의로 박아 넣기보다, 각 선택의 이유가 cost/model/search 구조에 명시적으로 드러나게 만들어라.
- 먼저 현재 구현과 아래 설계 사이의 차이를 분석한 뒤 수정안을 제시하고 구현하라.


# 전체 planning flow

FFT plan 생성은 개념적으로 다음과 같은 순서를 가진다.

--------------------------------------------------
1. Radix decomposition
--------------------------------------------------

예:

N = 1024

→ primitive radices

[2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

이 단계의 목적은 FFT length를 수학적으로 계산 가능한 radix sequence로 분해하는 것이다.

고려사항:

- FFT length factorization
- supported radix
- butterfly implementation availability
- 각 radix sequence가 실제 FFT decomposition으로 유효한지

이 단계에서는 아직

- kernel boundary
- scratchpad
- worker 수
- cooperative execution
- SIMD efficiency
- register pressure

등을 결정하지 않는다.

즉 이 단계는 가능한 한 "mathematical decomposition" 역할에 집중한다.


--------------------------------------------------
2. Kernel partition / kernel boundary search
--------------------------------------------------

primitive radix sequence를 여러 kernel로 나눈다.

예:

[2,2,2,2,2,2,2,2,2,2]

→

kernel0 = [2,2,2,2,2]
kernel1 = [2,2,2,2,2]

여기서 중요한 문제는

"어디에서 DRAM boundary를 만들 것인가?"

이다.

kernel 내부 stage들은 scratchpad에 데이터를 유지하면서 계산할 수 있지만,
kernel boundary를 지나면 일반적으로 DRAM handoff / layout conversion 등이 필요하다.

따라서 kernel partition은 주로 data movement 관점에서 평가한다.

고려해야 할 항목:

- scratchpad capacity
- scratchpad working-set requirement
- DRAM read/write traffic
- kernel boundary count
- kernel launch/boundary overhead
- transpose requirement
- DRAM access stride
- tiled transpose 가능성
- large twiddle
- large twiddle + transpose fusion 가능성
- layout conversion
- vector/coalesced memory access 가능성
- 지나치게 큰 stride에 따른 penalty
- boundary 전후의 address mapping

이 부분은 우선 memory / data-movement 중심 cost model을 사용한다.

예를 들어 개념적으로

partition_cost =
    DRAM_traffic_cost
  + boundary_cost
  + stride_cost
  + transpose_cost
  + large_twiddle_cost
  + scratchpad_penalty

와 같은 형태가 될 수 있다.

단, 위 식을 그대로 하드코딩하라는 뜻은 아니다.
현재 코드와 측정 가능한 정보에 맞게 적절한 모델을 설계하라.

중요한 점은 이 단계에서는 아직
각 kernel 안의 [2,2]를 4로 합칠지,
worker를 몇 개 사용할지 등을 확정하지 않는 것이다.


--------------------------------------------------
3. Intra-kernel radix composition / stage fusion
--------------------------------------------------

kernel boundary가 정해진 뒤,
각 kernel 내부의 primitive radix stage들을 더 큰 specialized radix로 합칠 수 있다.

예:

kernel0 = [2,2,2,2,2]
kernel1 = [2,2,2,2,2]

→ 후보 A

kernel0 = [4,4,2]
kernel1 = [4,4,2]

또는

→ 후보 B

kernel0 = [8,4]
kernel1 = [8,4]

등.

물론 실제 지원되는 radix 및 mathematical equivalence를 만족해야 한다.

이 단계는 주로 computation 관점에서 평가한다.

고려사항:

- arithmetic instruction count
- multiply/add/FMA count
- twiddle instruction cost
- trivial twiddle 제거 가능성
- specialized butterfly efficiency
- SIMD utilization
- vector lane utilization
- tail handling
- register pressure
- live range
- temporary register count
- spill 가능성
- instruction count
- code size
- scratchpad access 횟수
- stage 간 intermediate store/load
- radix가 커졌을 때의 instruction-level efficiency

예:

[2,2] → [4]

가 항상 좋은 것은 아니다.

radix-4가 instruction 수는 줄여도
register pressure가 커져 spill이 발생하면 오히려 느릴 수 있다.

따라서 단순히 "가능한 한 큰 radix로 합친다"는 heuristic은 사용하지 않는다.


--------------------------------------------------
4. Execution strategy selection
--------------------------------------------------

각 kernel을 어떤 방식으로 실행할지도 결정해야 한다.

현재 지원되는 실행 방식들을 코드에서 조사하라.

예를 들어 다음과 같은 parameter가 있을 수 있다.

- non-cooperative execution
- cooperative execution
- workers per FFT
- logical block 수
- tile size
- uthread allocation
- FFTs per NDP unit
- scratchpad partitioning

실제 코드에 존재하지 않는 parameter를 임의로 만들지는 마라.

고려사항:

- FFT leaf size
- logical block count
- available parallelism
- number of FFTs
- worker utilization
- SIMD utilization
- NDP unit utilization
- number of active µthreads
- scratchpad usage
- synchronization overhead
- cooperative communication overhead
- barrier overhead
- reduction / exchange overhead
- register pressure
- DRAM latency hiding
- small FFT에서 parallelism 부족
- 큰 FFT에서 한 µthread가 너무 많은 일을 수행하는 문제

M2NDP는 fine-grained µthread를 사용하며,
같은 NDP unit의 µthread들은 scratchpad를 공유할 수 있다는 점도 고려하라.

그러나 cooperative execution이 항상 더 빠른 것은 아니다.

작은 FFT에서는 synchronization / worker overhead가
parallelism 이득보다 커질 수 있다.

따라서 execution strategy는 실제 performance 특성에 따라 선택되어야 한다.


==================================================
핵심: Step 3과 Step 4는 joint optimization 가능
==================================================

3번과 4번은 완전히 독립적인 문제가 아니다.

radix configuration이 달라지면

- register pressure
- instruction count
- scratchpad access pattern
- amount of work per worker
- SIMD utilization
- synchronization frequency

가 달라진다.

반대로 execution strategy가 달라지면

- 사용할 수 있는 register budget
- scratchpad allocation
- worker utilization
- stage partitioning efficiency

가 달라질 수 있다.

따라서 다음과 같이 joint search를 가능하게 만들어라.

예:

kernel primitive stages:

[2,2,2,2,2]

radix candidates:

A = [4,4,2]
B = [8,4]
C = [2,4,4]
...

execution candidates:

E0 = non-cooperative
E1 = cooperative, workers=2
E2 = cooperative, workers=4
...

평가 대상은

(A,E0)
(A,E1)
(A,E2)
(B,E0)
(B,E1)
...

과 같은 Cartesian product이다.

즉

(radix configuration, execution strategy)

pair 자체를 하나의 candidate plan으로 평가한다.


# 권장 search hierarchy

전체 search space를 무작정 exhaustive search하면 너무 커질 수 있다.

따라서 기본적인 구조는 다음처럼 하는 것이 좋다.

N
 ↓
radix decomposition
 ↓
kernel partition candidates
 ↓
각 kernel에 대해
    radix composition candidates
        ×
    execution strategy candidates
 ↓
candidate scoring
 ↓
top-K / pruning
 ↓
필요하면 measurement
 ↓
best plan

단, 현재 search space 크기가 충분히 작다면 exhaustive enumeration도 허용한다.

search space가 커질 경우 다음 방법을 검토하라.

- dynamic programming
- beam search
- top-K pruning
- branch-and-bound
- Pareto pruning

특히 명백하게 나쁜 후보는 가능한 빨리 제거한다.

예:

- scratchpad capacity 초과
- unsupported radix
- mathematically invalid sequence
- impossible worker configuration
- SIMD requirement 위반
- guaranteed spill
- zero-use worker
- layout incompatibility

등은 scoring 전에 reject한다.


# Analytical model vs measurement

모든 것을 analytical cost model로 정확하게 예측하려 하지 마라.

FFT library에서도 일부 parameter는 실제 benchmark를 통해 선택한다.

따라서 다음과 같은 hybrid approach를 고려한다.

Step 2:
    analytical/model 중심

Step 3:
    analytical/model 중심 + static compiler information 가능

Step 4:
    실제 cycle measurement의 비중을 높임

Step 3 + Step 4 final candidates:
    가능하다면 실제 simulator cycle measurement

즉

cheap model
    ↓
candidate pruning
    ↓
small candidate set
    ↓
actual cycle measurement

형태로 만드는 것이 바람직하다.

특히 execution strategy는

- worker utilization
- synchronization
- latency hiding
- register pressure

등을 정확히 모델링하기 어렵기 때문에
실제 cycle 측정을 통한 autotuning을 허용한다.


# Cost model을 하나로 억지로 합치지 말 것

가능하면 cost model을 의미별로 분리한다.

예:

MemoryCost
- DRAM bytes
- DRAM transactions
- stride
- transpose
- large twiddle
- boundary

ComputeCost
- arithmetic instructions
- twiddle instructions
- butterfly efficiency
- vector utilization

ResourceCost
- registers
- scratchpad
- expected spill
- worker occupancy

ExecutionCost
- worker utilization
- parallelism
- synchronization
- launch/exchange overhead

그리고 마지막 scoring layer에서 합친다.

이렇게 해야 추후 실제 measurement와 비교하면서
각 model을 독립적으로 보정할 수 있다.


# 반드시 확인할 중요한 문제

현재 코드에서 다음 문제를 조사하라.

1. kernel partition과 radix composition이 이미 섞여 있는가?

2. 현재 greedy decomposition이
   미래 execution strategy를 고려하지 않고 너무 일찍 결정을 내려버리는가?

3. radix coalescing이 단순히 큰 radix 우선으로 되어 있는가?

4. cooperative/non-cooperative가
   radix 결정 이후 별도 선택되는 구조인가?

5. execution strategy에 따라 radix candidate 평가값이 달라질 수 있는가?

6. register pressure / spill을 현재 planner가 볼 수 있는가?

7. 실제 simulator cycle 결과를 planner/autotuner가 feedback으로 사용할 수 있는가?

8. plan object가 최종적으로 다음 정보를 모두 표현할 수 있는가?

- kernel boundary
- radix sequence
- input/output mapping
- transpose
- large twiddle
- tile configuration
- execution strategy
- worker count
- scratchpad allocation

부족하다면 plan representation부터 최소한으로 확장하라.


# 매우 중요한 설계 원칙

Planner:

"What to execute"를 전부 결정

Codegen:

"Given this plan, emit the corresponding code"

즉 codegen 내부에서

if FFT size > ...
    choose radix...
if scratchpad ...
    choose workers...

같은 planning decision을 해서는 안 된다.

모든 decision은 plan에 명시되어 있어야 한다.


# 구현 순서

한 번에 모든 것을 뜯어고치지 말고 다음 순서로 진행하라.

Phase 1.
현재 planner/search/cost-model 구조 분석

다음 내용을 먼저 보고하라.

- 현재 planning flow
- 현재 candidate space
- 현재 greedy decision
- 현재 cost model
- 현재 cooperative/non-cooperative 선택 방식
- 위에서 제안한 4단계와의 차이

Phase 2.
search representation 정리

- primitive radix representation
- kernel partition representation
- radix composition representation
- execution strategy representation

을 명확히 분리한다.

Phase 3.
kernel partition search를 정리한다.

memory/data movement cost 중심으로 candidate를 생성/평가한다.

Phase 4.
각 kernel의 radix composition candidate generator를 만든다.

하나의 최종 radix sequence만 반환하지 말고
여러 합법적인 candidate를 만들 수 있어야 한다.

Phase 5.
execution strategy candidate generator를 만든다.

Phase 6.
radix × execution joint search를 구현한다.

Phase 7.
cost model 기반 ranking을 한다.

Phase 8.
가능하다면 top-K candidate에 대해 simulator cycle을 측정하고
최종 선택하는 autotuning path를 추가한다.


# 검증

기존 numerical correctness test를 전부 유지하라.

추가로 최소한 다음을 테스트하라.

1. radix composition candidate가 원래 N의 factorization을 보존하는가
2. unsupported radix를 생성하지 않는가
3. scratchpad capacity를 넘는 kernel이 reject되는가
4. invalid cooperative configuration이 reject되는가
5. 같은 FFT에 대해 여러
   (radix configuration, execution strategy)
   candidate가 실제로 만들어지는가
6. cost가 낮은 candidate가 ranking 상위에 오는가
7. measured-cycle mode가 있다면
   실제 cycle이 가장 작은 후보를 선택하는가
8. 기존 FFT numerical result와 동일한가


# 최종적으로 내가 원하는 결과

단순히 "cost model을 추가했습니다"가 아니다.

최종 구조가 개념적으로 다음처럼 보였으면 한다.

Radix decomposition
        ↓
Kernel boundary search
        ↓
┌─────────────────────────────────────┐
│ per-kernel joint optimization       │
│                                     │
│ radix composition                   │
│         ×                           │
│ execution strategy                  │
│                                     │
│ → candidate plans                   │
└─────────────────────────────────────┘
        ↓
Analytical cost ranking
        ↓
Top-K
        ↓
(optional) simulator measurement
        ↓
Final FFT Plan
        ↓
Codegen

구현 전에는 반드시 현재 코드와 이 구조를 비교해서
"이미 구현된 것 / 수정할 것 / 새로 만들 것"을 구분해서 설명하라.

불필요한 대규모 rewrite는 피하고,
현재 코드에서 재사용할 수 있는 search/cost-model abstraction은 최대한 재사용하라.

마지막 보고에서는 다음을 포함하라.

1. 기존 구조의 문제점
2. 변경한 architecture
3. candidate search space
4. pruning rule
5. 각 cost component의 의미
6. 3번과 4번을 어떻게 joint search하는지
7. measurement/autotuning을 어떻게 연결했는지
8. 변경한 파일/함수
9. correctness test 결과
10. 대표 FFT N에 대해 생성된 candidate 예시
11. 선택된 candidate와 선택 이유
12. 아직 analytical model이 부정확할 가능성이 높은 부분
