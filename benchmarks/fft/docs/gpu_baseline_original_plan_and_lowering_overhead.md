# GPU baseline 검토 3·4번 수정

검토 보고서의 3번(fusion 해제 비용)과 4번(GPU 실행 metadata 오염)을 수정했다.
3번은 보고서가 권고한 원본 계획 보존 및 추가 비용의 별도 보고를 구현한 것이다.
실제 fused M2NDP codegen을 새로 구현한 것은 아니다.

`planning/gpu_baseline/logical_plan.py`는 physical mapping 전에 GPU kernel
sequence를 결정하고 immutable `GPULogicalKernelPlan` tuple을 보존한다.
clFFT single/large-1D, rocFFT-default 및 adapted, VkFFT에 적용한다.
rocFFT tuned는 공통 `map_cooperative_kernel`에서 각 후보의 원래 설정을
slot clipping과 feasibility 검사 전에 보존하고, 선택된 후보의 snapshot을 유지한다.
`reconstruct_gpu_logical_plan`은 이 snapshot을 반환하며 physical tree를 읽지 않는다.

원본의 WGS/TPT/TPB, radix, scheme, fused operation, upload 번호와 stageStartSize는
physical scheduling과 독립적이다. batch=1 때문에 FFT slot이 줄거나 persistent
mapping이 slot을 1로 바꾸어도 원본 GPU 값은 유지한다. GPU LDS와 M2NDP scratchpad는
별도 필드이며 GPU LDS를 physical scratchpad에서 추정하지 않는다.

GPU kernel section과 physical node section을 독립적으로 출력한다. Physical ID는
`physical_node_id`이며 구 API의 `gpu_kernel_id` property는 호환성 alias이다.
두 sequence 사이의 1:1 대응을 의미하지 않는다.

`compute_lowering_overhead`와 benchmark CSV에 다음 항목을 추가했다.

| 항목 | 의미 |
| --- | --- |
| `gpu_kernel_count` | 보존된 원본 GPU kernel 수 |
| `physical_plan_nodes` | M2NDP FFT/transpose node 수 |
| `added_plan_nodes` | 두 수의 차이 |
| `added_transpose_nodes` | 명시적 transpose node 수의 차이 |
| `gpu_full_array_bytes` | 원본 kernel별 FP32 배열 read+write 추정치 합 |
| `m2ndp_full_array_bytes` | physical node별 FP32 배열 read+write 추정치 합 |
| `additional_full_array_bytes` | 두 traffic 추정치의 차이 |
| `full_array_traffic_ratio` | M2NDP 추정치 / GPU 추정치 |

예를 들어 N=8192, batch=1의 clFFT block, rocFFT CC, VkFFT adapted 2-upload는
GPU kernel 2개를 M2NDP node 5개로 실행한다. 추정치는 각각 262,144 bytes와
655,360 bytes이며 추가 393,216 bytes, 비율 2.5이다. batch=3이면 bytes는
3배이고 비율은 같다. VkFFT adapted 3-upload 예제(N=32768, scratchpad=4096)는
원본 3개와 physical 9개로 구분하고 비율 3.0을 보고한다.

이 수치는 complex FP32 원소 8 bytes에 대해 각 pass가 배열 전체를 한 번 읽고
쓴다는 알고리즘 수준의 추정치다. Padding, twiddle table, cache, persistent
내부 stage별 DRAM 왕복과 simulator transaction은 포함하지 않는다. 측정 DRAM
counter나 cycle 비율이 아니며 physical node 수는 실제 host/stage launch 수와도
다르다. 따라서 benchmark의 실제 cycle을 이 비율로 보정하지 않는다.

아직 추출하지 않은 clFFT block kernel의 radix/WGS/TPT/TPB 및 GPU별 실제 LDS는
`None` 또는 빈 radix tuple로 표시한다. 일반 Stockham 설정이나 M2NDP 값으로
대체하지 않는다. tuned rocFFT의 기존 per-FFT `gpu_logical_lds_bytes`는 보존하되
이를 실제 workgroup 전체 LDS와 동일시하지 않는다. Capture에 실패한 경우 원인과
`unavailable`을 표시하고 overhead를 계산하지 않는다.

대상은 기존 baseline과 같은 contiguous, out-of-place, 1D complex FP32 경로다.
rocFFT gfx908 solution map에는 해당 기본 domain의 override가 없음을 기존
source-fidelity 검증이 확인한다. 다른 placement/layout/solution-map domain으로
확장할 때는 source decision capture도 함께 확장해야 한다.

회귀 검증: `python -m verification.verify_gpu_original_plan`.
기존 execution semantics/mapping/report 및 half-LDS 검증도 원본/physical 분리를
기준으로 갱신했다. 실제 binary 출력 검증(검토 5번)은 이번 변경 범위에 포함하지 않는다.
