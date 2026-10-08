# M2NDP FFT GPU baseline planner 검토 — 2026-10-07

현재 checkout의 `planning/gpu_baseline/{clfft,rocfft_default,rocfft,vkfft}.py`, 공통 lowering, 실행 보고서와 benchmark harness를 검토했다. 구현 변경은 하지 않았다.

**판정: 작은 1D FP32 C2C FFT의 GPU 유래 계획 정책을 비교하는 baseline으로는 상당한 근거가 있다. 그러나 큰 FFT까지 기존 GPU 라이브러리의 계획을 충실하게 재현했다고 보기에는 중요한 불일치가 있고, 현재 측정값을 GPU 라이브러리의 최적화 수준을 대표하는 성능 baseline으로 해석하기에는 lowering과 검증의 제약이 있다.**

현재 `gpu_vs_m2ndp_benchmark.py:101`의 기본 27개 크기는 N=12..4096이다. 아래의 큰 FFT planner 불일치는 이 기본 sweep 밖에서 발생한다. 따라서 큰 FFT 문제만으로 기존 작은 크기 측정값 자체가 틀렸다고 결론 내리지는 않는다.

**확인된 장점**

- clFFT의 specialization table, DetermineSizes와 radix 순서를 별도로 구현했다.
- rocFFT의 production planner와 offline tuner를 분리했다. 두 알고리즘을 같은 baseline으로 혼동하지 않는다.
- pinned rocFFT `config_sbrr.py`를 AST로 읽고 대조한 결과, 포팅된 459개 항목의 WGS/TPT/radix는 각각 실제 upstream 항목과 일치했다. upstream의 추가 항목 대부분은 160 KiB LDS용이므로, gfx908 기준에서 빠졌다는 사실만으로 오류라고 판단하지 않았다.
- VkFFT의 register sizing/AxisBlock 관련 규칙을 구체적인 공개 소스에 연결했다.
- 실패 상태를 숨기지 않으며, 논리 worker 수와 물리 lane 수를 구분하는 기반이 있다.
- W>8 및 ragged worker 지원, clFFT/VkFFT adapted planner의 16-byte scratchpad 비용 보정은 현재 코드에서 이미 구현되어 있다. 오래된 문서의 미지원 설명을 현재 결함으로 세지 않았다.

**1. [P1] rocFFT large-1D는 upstream의 하위 FFT 순서와 block kernel 설정을 보존하지 않는다.**

관련 구현: `rocfft_default.py:978`, `rocfft_default.py:1059`, `rocfft_default.py:1071`, `rocfft_default.py:1101`.

upstream `CC1DNode::BuildTree_internal`은 `div_length1`의 SBCC 다음에 `div_length0`의 SBRC를 실행한다. `TRTRT1DNode::BuildTree_internal`도 첫 row FFT가 `div_length1`이고, 두 번째가 `div_length0`이다. 현재 구현은 `near_node=div0`을 먼저 실행하고 `far_node=div1`을 나중에 실행한다. [rocFFT 원문](https://github.com/ROCm/rocm-libraries/blob/bee97df517907c771de17189cb867d3c401285ae/projects/rocfft/library/src/tree_node_1D.cpp)

| 사례 | upstream FFT 순서 | 현재 FFT 순서 |
|---|---|---|
| N=8192, CS_L1D_CC | 64 → 128 | 128 → 64 |
| N=4704, CS_L1D_CC | 96 → 49 | 49 → 96 |
| N=2058, CS_L1D_TRTRT | 42 → 49 | 49 → 42 |

또한 `_rocfft_leaf_kernel`은 모든 leaf를 `SBRR_TABLE`의 일반 Stockham 설정으로 만든다. CC용 `config_sbcc.py`/`config_sbrc.py`의 실제 설정을 사용하지 않는다.

- N=8192의 upstream SBCC64 radix는 `(8,8)`, SBRC128 radix는 `(8,4,4)`이다. 현재 leaf는 각각 일반 SBRR64 `(4,4,4)`, SBRR128 `(16,8)`이다.
- N=4704의 upstream SBCC96 radix는 `(8,3,4)`지만 현재 leaf는 일반 SBRR96 `(6,16)`이다.

[SBCC 설정 원문](https://github.com/ROCm/rocm-libraries/blob/bee97df517907c771de17189cb867d3c401285ae/projects/rocfft/library/src/device/kernels/configs/config_sbcc.py), [SBRC 설정 원문](https://github.com/ROCm/rocm-libraries/blob/bee97df517907c771de17189cb867d3c401285ae/projects/rocfft/library/src/device/kernels/configs/config_sbrc.py).

Numpy와 일치하는 FFT를 계산하는 것은 가능하다. 그러나 이는 upstream이 선택한 동일한 FFT 순서, radix, worker/batch 배치를 실행한다는 증거가 아니다. 원래 계획과 다른 작업 분배의 성능을 측정하게 된다. GPU 논리 계획을 먼저 만들고, 그 순서와 SBCC/SBRC 설정을 유지한 채 물리 lowering을 적용하는 것이 우선이다.

**2. [P1] VkFFT multi-pass split은 실제 scheduler의 조건 분기와 다르고, 3-pass에서는 인자 곱도 깨진다.**

관련 구현: `vkfft.py:540`, `vkfft.py:605`, `vkfft.py:675`, `vkfft.py:730`, `vkfft.py:1173`, `vkfft.py:1282`.

`split_pow2_2pass`는 `maxPow8SharedMemory`를 우선하는 분기를 항상 적용한다. 하지만 upstream의 이 분기는 non-strided axis이면서 `reorderFourStep=false` 또는 Bluestein인 경우에 해당한다. baseline이 전제로 하는 `reorderFourStep=true`, 일반 C2C에서는 strided 한계와 `maxPow8Strided <= 512` 분기를 사용한다. 또 upstream은 power-of-two 입력에 `sharedMemorySizePow2`를 사용하지만, 현재 split은 raw `spad_capacity_bytes`를 사용한다. [VkFFT scheduler 원문](https://github.com/DTolm/VkFFT/blob/066a17c17068c0f11c9298d848c2976c71fad1c1/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h)

동일한 자원 입력을 대응시킨 비교: `sharedMemorySize=122880`, `sharedMemorySizePow2=65536`, `complexSize=8`, `coalescedMemory=32`, `registerBoost=1`, `reorderFourStep=true`, NVIDIA 분기. 아래 upstream 값은 이 조건의 소스 분기를 직접 계산한 결과이며, 실제 GPU 실행 측정값은 아니다.

| N | upstream split | 현재 split |
|---|---|---|
| 65536 | (512,128) | (1024,64) |
| 131072 | (512,256) | (2048,64) |
| 262144 | (512,512) | (4096,64) |

`split_pow2_3pass`는 upstream의 별도 3-pass 알고리즘 대신 2-pass splitter를 두 번 호출한다. 기본 target에서 다음을 재현했다.

| 호출 | 분할 | 분할 인자 곱 | 원래 길이 |
|---|---|---|---|
| complex_size_bytes=8 | (15360,64,17) | 16711680 | 16777216 |
| complex_size_bytes=16 | (7680,64,34) | 16711680 | 16777216 |

그 결과 `vkfft.plan(2**24)`는 길이 17의 Rader 미지원으로, `vkfft.plan_m2ndp(2**24)`는 길이 34에 남은 인자 17의 Rader 미지원으로 반환한다. 순수 power-of-two FFT에 잘못 생성한 인자가 끼어든 것이므로, 정상적인 prime-size 기능 제한으로 설명할 수 없다.

upload 번호도 확인이 필요하다. 현재 코드는 첫 실행 leaf를 `upload_id=0`으로 정하고 증가시키지만, 실제 VkFFT forward 실행은 `numAxisUploads-1`부터 0까지 역순이다. upstream `axis_upload_id==0`용 AxisBlock 조건을 '첫 실행 pass'용으로 해석하면 다른 pass에 적용된다. [실행 순서 원문](https://github.com/DTolm/VkFFT/blob/066a17c17068c0f11c9298d848c2976c71fad1c1/vkFFT/vkFFT/vkFFT_AppManagement/vkFFT_RunApp.h)

split 인자의 곱·정확한 divisibility·각 upload의 자원 상한을 invariant로 확인하고, GPU upload 번호와 실행 순서를 별도로 보존해야 한다.

**3. [P2, 성능 해석] fused 메모리 접근을 풀어서 실행하므로, lowering 비용이 baseline 차이에 포함된다.**

관련 구현: `clfft.py:792`, `rocfft_default.py:1009`, `vkfft.py:1282`.

현재 세 모듈의 large-1D 경로는 공통적으로 `transpose → FFT → transpose+twiddle → FFT → transpose`를 만든다. 실제 clFFT SBCC는 2개 또는 3개 kernel이며, rocFFT CC는 2개 kernel이다. 일반적인 VkFFT 2-upload도 두 FFT upload 안에서 관련 메모리 재배열을 처리한다. [clFFT 원문](https://github.com/clMathLibraries/clFFT/blob/c59712e136fa6207956af22f5c0e4cee7d05340e/src/library/plan.cpp), [rocFFT 원문](https://github.com/ROCm/rocm-libraries/blob/bee97df517907c771de17189cb867d3c401285ae/projects/rocfft/library/src/tree_node_1D.cpp), [VkFFT 실행 원문](https://github.com/DTolm/VkFFT/blob/066a17c17068c0f11c9298d848c2976c71fad1c1/vkFFT/vkFFT/vkFFT_AppManagement/vkFFT_RunApp.h).

예를 들어 N=8192의 clFFT/rocFFT CC는 현재 5개 plan node이다. 각 pass가 전체 배열을 한 번 읽고 쓰는 단순 계산에서는, 2-pass의 `32*N` bytes 대신 5-pass의 `80*N` bytes가 된다. 이는 계획상 배열 traffic이 2.5배라는 뜻이며, 측정 cycle이 2.5배라는 뜻은 아니다. Persistent leaf 내부의 preload/stage/writeback 실행 비용도 별도로 붙는다.

이러한 물리 스케줄 변경은 NDP 포팅에서 허용 가능한 선택일 수 있다. 다만 수치 동등성만으로 성능상 동등한 lowering이라고 주장해서는 안 된다. 최소한 추가 transpose/pass와 traffic을 별도 보고하고, 가능하다면 strided load/store와 twiddle을 FFT leaf에 합친 lowering과 비교해야 한다. 그렇지 않으면 'planner 정책의 차이'와 '포팅 구현의 차이'를 분리하기 어렵다.

**4. [P2] GPU 논리 계획 보고서는 M2NDP 실행 계획을 역으로 읽어 일부 GPU 값을 바꾸거나 잃는다.**

관련 구현: `common.py:428`, `common.py:562`, `common.py:847`, `common.py:887`, `common.py:929`.

`reconstruct_gpu_logical_plan`은 lowering 완료 후의 tree를 순회한다. Cooperative path에서는 실제 배치에 맞춰 축소된 FFT slot 수를 GPU 원래 설정으로 보고하며, persistent path에서는 원래 TPB/WGS를 잃어 `None`으로 보고한다.

| clFFT 사례, batch=1 | raw GPU 설정 (WGS,NT) | GPU 보고서 (WGS,NT) |
|---|---|---|
| N=2 | (64,64) | (1,1) |
| N=16 | (64,16) | (4,1) |
| N=64 | (64,4) | (None,None) |

또 `gpu_shared_memory_bytes`는 M2NDP scratchpad buffer 크기에서 얻고, GPU kernel 개수와 transpose 역할도 M2NDP의 5-node tree에서 얻는다. 따라서 현재 보고서는 upstream 실행 trace를 독립적으로 표현하지 못한다. `half_lds`용 별도 logical LDS 필드는 일부 개선이지만, 전체 메커니즘을 해결하지는 않는다.

실제 배치에 맞춰 물리 slot 수를 줄이는 것은 타당할 수 있다. 문제는 그 값을 GPU가 원래 고른 WGS/NT로 표시하는 것이다. Lowering 전에 원본 GPU kernel별 scheme/radix/WGS/TPT/TPB/LDS/upload/fusion을 별도 IR 또는 metadata로 남겨야 한다.

**5. [P2, 검증] benchmark의 'correct'는 실제 실행 binary의 출력 정확도 검증이 아니다.**

관련 구현: `gpu_vs_m2ndp_benchmark.py:305`, `gpu_vs_m2ndp_benchmark.py:323`, `gpu_vs_m2ndp_benchmark.py:333`, `gpu_vs_m2ndp_benchmark.py:360`, `planning/diagnostics/spill_probe.py:216`, `planning/diagnostics/spill_probe.py:241`.

Harness는 Python의 `run_recursive_plan`을 Numpy와 비교해 `correct`를 정한다. 실제 toolchain은 `probe_spill_free`로 실행하며, 이 함수는 `reference_check=False`이고 INPUT/OUTPUT dump도 비교하지 않는다고 명시한다. 이후 Python oracle 통과와 실제 실행/spill 통과를 조합해 `spill_free_correct`를 기록한다.

그러므로 이 상태는 'Python 계획은 수치적으로 맞고, binary는 실행되며 spill이 없었다'는 증거다. Compiler/codegen/address mapping을 통과한 실제 출력까지 맞았다는 증거는 아니다. 이를 입증하려면 `run_fft_test.sh`의 수치 검증이나 실제 출력 dump를 Numpy FFT와 비교하는 단계를 별도로 수행해야 한다. 튜너의 실측 승자도 최종적으로 같은 검증이 필요하다.

**검증 범위와 재현**

- 기존 GPU baseline 관련 검증 10개 모두 통과: `verify_gpu_baseline`, `verify_gpu_baseline_source_fidelity`, `verify_gpu_baseline_golden`, `verify_gpu_baseline_m2ndp_adapted`, 두 scratchpad regression, rocFFT half-LDS, execution semantics, execution mapping, execution report.
- Adapted 검증은 `benchmarks/fft`에서 `python -u -m verification.verify_gpu_baseline_m2ndp_adapted`로 실행했다. clFFT large-1D는 N=1048576까지 forward/inverse Python 수치 검증을 통과했다.
- 추가 read-only probe로 위 plan node 순서, radix, split 곱, 실패 상태와 GPU 보고서 값을 재현했다.
- 이번 검토에서 Mojo/LLVM/M2NDP-Detour binary를 새로 빌드하거나 cycle을 새로 측정하지 않았다. 실제 성능 영향의 크기는 확정하지 않았다.

`benchmarks/fft`에서 실행할 최소 재현:

```python
from math import prod
from planning.core.target_profile import DEFAULT_TARGET_PROFILE as t
from planning.gpu_baseline import clfft, rocfft_default, vkfft
from planning.gpu_baseline.common import reconstruct_gpu_logical_plan
from planning.strategies.fft_plan_recursive import flatten_recursive_node

r = rocfft_default.plan(8192)
print([(k.length, tuple(s.radix for s in k.stages))
       for k in flatten_recursive_node(r.plan.root) if hasattr(k, "stages")])
# [(128, (16,8)), (64, (4,4,4))]

print(vkfft.split_pow2_2pass(65536, t))  # (1024,64)
f = vkfft.split_pow2_3pass(2**24, t)
print(f, prod(f), 2**24)  # (15360,64,17), 16711680, 16777216
print(vkfft.plan(2**24).diagnostics)

r = clfft.plan(16, batch=1)
print(r.gpu_config.extra)
print(reconstruct_gpu_logical_plan(r)[0])
```

기존 golden 값은 `verify_gpu_baseline_golden.py` 설명대로 upstream 실행이 아니라 이 구현에서 캡처한 값이다. Source-fidelity 테스트도 Python으로 재작성한 reference와 비교하는 부분이 많다. 따라서 테스트 통과가 큰 FFT의 실제 upstream 계획과 일치함을 보장하지 않는다. 원본 upstream 계획 trace를 기대값으로 삼아 큰 split, CC/TRTRT, upload 순서, batch=1 및 여러 batch를 검증하는 테스트가 필요하다.

**권장 조치 순서**

1. rocFFT CC/TRTRT의 FFT 순서 및 CC용 SBCC/SBRC 설정을 바로잡는다.
2. VkFFT의 reorder/vendor/power-of-two 자원 분기, 별도 3-pass split과 upload 번호를 원문대로 구현한다.
3. 원래 GPU 논리 계획을 보존하고, fusion 해제에 따른 pass/traffic 비용을 따로 보고한다.
4. 실제 binary 출력까지 검증한 뒤 baseline 성능 비교를 갱신한다. 기본 27개 작은 크기에 더해 large-1D와 batch sweep을 포함한다.

주 비교에는 `*-m2ndp`를 사용하고 GPU 상수를 유지한 버전은 보조 비교로 구분하는 방향이 적절하다. 다만 VkFFT의 frozen `plan()`도 실제로는 target scratchpad 예산을 사용한다. 예를 들어 N=8192의 `choose_num_passes`는 target 32 KiB에서 2, 64/120 KiB에서 1이다. 세 baseline 모두가 동일한 '고정 GPU 자원' 정의를 따르는 것은 아니므로 명칭과 문서를 맞춰야 한다.

clFFT/VkFFT의 현재 provenance는 upstream SHA가 없는 `master`이다. 이번 검토에서는 GitHub API에서 SHA를 조회하고 master 다운로드와 pinned 다운로드가 byte-identical임을 확인했다: clFFT `c59712e136fa6207956af22f5c0e4cee7d05340e`, VkFFT `066a17c17068c0f11c9298d848c2976c71fad1c1`. rocFFT는 구현에 기록된 `bee97df517907c771de17189cb867d3c401285ae`를 사용했다. 과거 최초 추출 때의 SHA가 기록되어 있지 않으므로, 이번 조회만으로 그 과거 snapshot과의 동일성까지 주장하지는 않는다.
