# 변경 기록: 데이터 배치 최적 경로 채택 — 2026-10-31

`layout_candidates_20261030`에서 측정한 최적 경로를 생성기 옵션으로 채택했다. 이 문서는 무엇을, 왜, 어떻게 바꿨는지와 그 근거를 파일 단위로 기록한다.

## 1. 요약

| 항목 | 내용 |
|---|---|
| 새 옵션 | `layout_rewrite` (API) / `--layout-rewrite` (CLI, benchmark harness). 기본값 `off` |
| 선택 경로 | `layout_rewrite='n8+f8+sp'`. B8은 `batch_window=8`과 함께 쓴다 |
| 정비한 검사 | compact SPAD 할당(live-banks/phase-packed)이면 grouped 소유권의 선언-bank 추정 검사를 최종 allocator 검사로 미룬다. 이로써 W8이 실제로 생성된다 |
| 기본 동작 | 새 옵션을 쓰지 않으면 생성 소스와 거부 동작이 전과 같다(9개 구성 확인) |
| 새 측정 | 없음. 채택 옵션으로 만든 소스가 10/30에 측정한 소스와 바이트 단위로 같다(17개 구성) |

선택 경로의 시간(10/30 측정, 구성당 1회, 28-unit/1640MHz 시뮬레이터):

| 조건 | 이전 선택 경로 | 채택 경로 | 변화 |
|---|---:|---:|---:|
| N16384/B1 | 25.536µs (phase-packed, native4-control) | 19.724µs | −22.8% |
| N16384/B8 | 82.234µs (W4, live-banks, native4-latch-forward) | 46.096µs (W8) | −43.9% |

선택 경로 명령. 공통 옵션은 이전 선택 경로와 같고, 끝의 옵션만 추가·변경했다.

```text
B1: --planner gpu-vkfft-m2ndp --leaf-storage spad --batch-window 4 --boundary-layout grouped
    --boundary-fusion pre --near-group 4 --far-group 1 --leaf-phase-policy far12-private
    --leaf-codegen-schedule far0-short-live --spad-allocation phase-packed
    --far12-codelet native4-control --layout-rewrite n8+f8+sp
B8: (위와 같음) --batch-window 8 --spad-allocation live-banks
    --far12-codelet native4-latch-forward --layout-rewrite n8+f8+sp
```

B1은 batch가 1이라 `--batch-window 4`가 실제로는 window 1로 줄어든다(이전과 같음).

## 2. 파일별 변경

경로는 `candidate/` 기준이다. 전체 diff는 `manifests/changes.patch`, 파일 해시는 `manifests/candidate.json`, 변경 목록은 `manifests/changes.json`에 있다. baseline 373개 파일은 원본과 해시가 같다.

### 2.1 `mojo-m2ndp/benchmarks/fft/codegen/layout_rewrite/` (신규 패키지)

10/30 측정에 쓴 변환 코드(`layout_candidates_20261030/transforms/`)를 생성기 안으로 옮겼다. 측정한 코드와 같은 결과를 내도록 로직은 바꾸지 않았고, import만 패키지 상대 경로로 바꿨다.

| 파일 | 출처 | 내용 |
|---|---|---|
| `phases.py` | `variants.py`의 앞부분(구조 탐색, 주소식 평가, strided store 재작성)과 phase 이름, `strided_phase` | shuffle 전치(T), radix-16(E), W8/조합 처리는 채택하지 않아 옮기지 않았다 |
| `near_lane_major.py` | 같은 이름 파일 | `variant_A`(토큰 `a`)와 `variant_B`. B는 4-lane 결합으로 기각했지만, N8이 MIDDLE 제거와 host 연결에 이를 쓰므로 내부 함수로 남겼다 |
| `wide.py` | 같은 이름 파일 | `variant_N8`(토큰 `n8`), `variant_F8`(토큰 `f8`) |
| `__init__.py` | 신규 | 토큰 해석·검증(`normalize_layout_rewrite`), 지원 구성 검사(`check_layout_rewrite_support`), 적용(`apply_layout_rewrite`) |

토큰별로 생성 코드가 바뀌는 내용:

- **`n8`**: near FFT 8개(r = 8g..8g+7)를 한 tile로 묶고, lane = r % 8로 두 near stage 모두 처리한다.
  - near stage0: 입력 load를 width 8로 바꾼다. 출력은 SPAD `tile_slot*1024 + plane*512 + pos*8`에 vector store 1개로 쓴다(기존: lane별 store 4개).
  - near stage1: worker 하나가 butterfly 하나(b = worker slot)를 8개 FFT에 대해 계산한다. 입력은 `sbase + 64m`에서 읽는다. 출력 k = b + 8j마다 8-packed twiddle(`g*512 + k*8 + lane`, 값 W_N^(r·k))을 곱해 far 입력 `q*N + k*256 + 8g`에 32B로 연속 저장한다.
  - MIDDLE 함수와 그 `launch_parallel` 호출을 지워 window당 phase가 6개에서 5개가 된다.
  - host 쪽: `FFTVkfftLeaf1Params`에 `boundary_twiddle_real/imag_base` 필드를 추가한다. near의 출력 버퍼를 work0에서 work1(far 입력)로 바꾸고, twiddle1을 넘긴다. twiddle1 생성 주소식을 `(r//8)*512 + c*8 + r%8`로 바꾼다. 이 작업은 측정 구간 밖의 host 준비 단계에서 이뤄진다.
  - SPAD 크기는 tile slot당 1024 floats로, 기존 near bank와 같은 영역을 쓴다.
- **`a`**: `n8`의 4-lane판이다. tile = near FFT 4개이고, SPAD는 `tile_slot*512 + plane*256 + pos*4`, near 출력은 work0 `q*N + g*256 + k*4 + lane`이다. MIDDLE은 packed 출력과 4-packed twiddle을 vector로 읽어 곱하고, r 8개를 join해 far 입력에 32B로 저장한다(SPAD 전치 없음).
- **`f8`**: far stage0을 FFT당 job 4개(outer 0..3, 8 butterfly)로 다시 나눈다. 입력과 twiddle load는 width 8이고, 출력 vector마다 strided store 하나(stride 8)로 기존 SPAD 배치(`b*8 + j`)에 쓴다. far stage1+2 codelet과 private scratch는 그대로다.
- **`sn`/`sm`/`sf`/`sp`/`s`**: 해당 phase에서 한 vector의 lane별 scalar SPAD store 묶음(실제 lowering: `vslidedown` + VL=1 `vse`)을 strided store 하나로 바꾼다(`vsoxei` 1개). 배치는 바뀌지 않는다. stride는 생성된 주소식을 무작위 정수 환경 4개에서 평가해 일정함을 증명한 값이다.

적용 순서는 토큰 순서와 상관없이 near(`n8`/`a`) → `f8` → `sn`·`sm`·`sf`·`sp`로 고정한다. 측정한 소스를 만든 순서와 같다.

다음 조합은 거부한다.
- `n8`과 `a`를 함께 쓰는 경우
- `n8`/`a`에 `sn`/`sm`을 더하는 경우(이미 vector store다)
- `f8`과 `sf`를 함께 쓰는 경우
- 같은 토큰을 반복하거나 알 수 없는 토큰을 쓰는 경우

지원 구성(이 밖은 사유와 함께 거부한다):
- task fusion auto, control 없음, leaf SPAD, grouped + PRE, near C4 / far C1
- far12-private, far0-short-live, compute_lanes 4
- spad_allocation live-banks 또는 phase-packed
- forward 변환, N = 16384 (near 길이 64 × near FFT 256개)

변환 단계마다 자기가 바꿀 코드의 모양을 assertion으로 증명한다. 다른 모양이면 `ValueError`로 거부한다.

### 2.2 `codegen/fft_transpose_codegen.py`

- 기존 `generate_recursive_fft_kernels`의 이름을 `_render_recursive_fft_kernels`로 바꿨다. 본문은 그대로다.
- 같은 이름의 공개 래퍼 `generate_recursive_fft_kernels(plan, *, layout_rewrite="off", **options)`를 추가했다.
  - `off`이면 렌더 결과를 그대로 돌려준다.
  - 그 밖에는 실제 시그니처의 기본값과 넘겨받은 옵션을 합쳐 지원 구성을 검사하고, 렌더한 뒤 변환을 적용한다.
- 모든 호출부는 keyword 인자를 쓰므로 호출 방식이 바뀌지 않는다.

### 2.3 `make_fft_kernel.py`

- `_make_gpu_baseline_fft_kernel`과 `make_fft_kernel`에 `layout_rewrite="off"` 인자를 추가하고 생성기까지 전달한다.
- `gpu-vkfft-m2ndp`가 아닌 planner에 `layout_rewrite`를 주면 거부한다.
- CLI에 `--layout-rewrite`(기본 off)를 추가했다.

### 2.4 W8: grouped 소유권 앞단 검사 정비

| 파일 | 변경 |
|---|---|
| `planning/execution/fft_boundary_simd.py` | `make_boundary_simd_plan(..., defer_leaf_budget=False)`. True이면 `ownership.arena_bytes > scratchpad_budget_bytes` 거부를 하지 않는다. docstring에 이유를 적었다 |
| `planning/execution/fft_grouped_boundary.py` | `make_grouped_boundary_plan(..., defer_leaf_budget=False)`를 위로 전달 |
| `planning/execution/fft_boundary_fusion.py` | `make_boundary_fusion_plan(..., defer_leaf_budget=False)`를 grouped 경로로 전달 |
| `planning/execution/task_fusion.py` | 두 호출부(`make_task_fusion_plan`, `resolve_task_execution`)가 `defer_leaf_budget = spad_allocation != 'declared'`를 넘긴다 |

이유:
- 앞단 검사는 소유권이 선언한 bank를 모두 더한 값(W8 near: 76 slot × 2 bank = 77,824B)을 leaf 예산 65,536B와 비교한다.
- compact allocator는 쓰지 않는 bank를 제거하고 phase 간 arena를 재사용한다. W8 B8에서 실제 arena는 12,288 floats(49,152B)이다.
- 그 결과는 `make_task_fusion_plan`의 최종 검사(`reserved > spad_capacity_bytes` 또는 `arena > max_concurrent_scratchpad_bytes`이면 거부)를 그대로 거친다.
- 따라서 compact 할당일 때만 앞단 추정을 최종 검사에 맡긴다.

바뀌지 않은 것:
- declared 할당의 동작. W8 declared 요청은 전과 같은 사유로 거부된다.
- `make_boundary_simd_plan`을 직접 부르는 기존 테스트의 거부 동작.
- 최종 예산 값.
- `batch_window='auto'`의 32KiB 선호. B8 auto는 전과 같은 window를 고른다.

결과적으로 `batch_window=8` + live-banks가 임시 완화 없이 생성되며, 10/30에 측정한 W8 소스와 바이트 단위로 같다.

### 2.5 `experiments/m2ndp_fft/scripts/leaf_storage_benchmark.py` (benchmark harness)

- `--layout-rewrite` 인자를 추가하고 생성 kwargs로 전달한다.
- `metadata.json`의 `implementation.layout_rewrite`에 기록한다.
- resume 시 설정이 달라지면 거부하는 검사에 `layout_rewrite`를 포함했다.
- 이 harness는 `mojo-m2ndp` 저장소 밖(워크스페이스 `experiments/`)에 있으므로 git 커밋에는 들어가지 않는다. 변경본은 이 디렉터리의 `candidate/`와 patch에 보존했다.

## 3. 검증 (`verify_adoption.py` → `verification/adoption.json`, 상태 PASS)

1. **측정 소스와 동일성 (17/17 IDENTICAL)**
   - 대상: N8+F8+Sp(B1), W8+N8+F8+Sp(B8), N8+Sf+Sp(B1·W8 B8), A+Sf+Sp(B1·B8·W8 B8), S(B1·B8·W8 B8), A(B1·B8), N8(B1·B8), F8(B1·B8), W8(B8, 변환 없음).
   - 새 옵션으로 생성한 소스가 측정한 `variant.original.mojo`와 바이트 단위로 같다. 따라서 10/30의 시간, 정답 검증, ELF 자원 기록이 채택 옵션에 그대로 적용된다.
2. **기본 동작 무변경 (9/9 UNCHANGED)**: baseline과 candidate 생성기를 별도 프로세스에서 실행해 비교했다.
   - 이전 B1·B8 최선 경로
   - B8 W4 declared
   - B8 W8 declared(거부 사유 동일)
   - B8 auto window
   - VkFFT N8192 CXL
   - VkFFT legacy SPAD
   - clFFT SPAD W4
   - inverse B1
3. **거부 (8/8)**: declared 할당, inverse, N8192, legacy layout, `n8+a`, `f8+sf`, `n8+sm`, 알 수 없는 토큰.
4. **기존 단위 테스트(candidate)**: `verify_fft_boundary_simd`·`verify_fft_leaf_ownership` 18개 OK(131.9초). `verify_fft_task_fusion`·`verify_fft_batch_windows` 12개 OK(549.0초, `verification/unittest_task_fusion.log`).

정답·race 검증, 실제 빌드, 시뮬레이터 실행은 10/30 작업에서 같은 소스로 이미 마쳤다. 구성마다 Python 실행 screen, 고정 toolchain 빌드, 새 시뮬레이터 세션, NumPy 검증을 거쳤다.

## 4. 측정 출처

선택 경로 시간의 원시 기록:

- B1: `layout_candidates_20261030/candidate/experiments/m2ndp_fft/results/20261030-layout-N8+F8+Sp-n16384-b1/raw_logs/n16384_b1/rep01/attempt001/trial.json` — 32,348 cycles = 19.724µs
- B8: `.../20261030-layout-W8+N8+F8+Sp-n16384-b8/raw_logs/n16384_b8/rep01/attempt001/trial.json` — 75,598 cycles = 46.096µs

phase 분해(µs):

| | near0 | near1 (+MIDDLE) | far0 | far12 | POST | 기타 |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 2.232 | 3.381 | 2.170 | 4.840 | 1.265 | 5.838 |
| B8/W8 | 7.523 | 8.854 | 6.137 | 14.556 | 3.167 | 5.859 |

정적 자원(실제 링크된 IR, 고정 llc):
- near0과 POST: spill 없음
- near1: vector spill 1개(frame 16B)
- far0: 명령 286개, `vsoxei` 16개
- far12: 이전과 같음(frame 32B, scalar spill 3개)

## 5. 바꾸지 않은 것과 한계

- 기본값은 off다. 이전 작업의 관례대로 선택 경로는 명시적 옵션으로 고른다.
- `layout_rewrite`는 렌더한 소스를 다시 쓰는 단계다. planner IR이나 emitter에 배치를 직접 표현한 것이 아니다. 다른 N, inverse, 다른 그룹 구성으로 넓히려면 IR 수준 구현과 새 검증이 필요하며, 지금은 거부한다.
- B1 상위 경로들(19.72–20.39µs)의 차이는 1회 측정 범위 안이다. 선택 근거는 관측 최저값이고, 통계적 유의성은 평가하지 않았다.
- shuffle 전치(T, study의 C·D), radix-16(E), 4-lane MIDDLE 결합(B)은 측정 결과 느려서 채택하지 않았다. 측정 기록은 `layout_candidates_20261030`에 보존했다.

## 6. 커밋

`mojo-m2ndp` 저장소, 브랜치 `fft-layout-adoption-20261031`(main에서 분기, push하지 않음):

1. 측정에 쓴 고정 소스 상태. 10/19–10/21 구현 체인과 그 이전의 미커밋 작업을 `benchmarks/fft`에 반영한다. 루트 작업 트리의 미커밋 파일은 이 체인의 출발점(post-fusion baseline)과 같아서 덮어써도 잃는 변경이 없다.
2. 이 문서의 채택 변경.

커밋 해시는 `verification/commits.json`에 기록한다.
