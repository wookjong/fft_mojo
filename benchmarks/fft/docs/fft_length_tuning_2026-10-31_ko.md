# 변경 기록: 길이별 튜닝을 기본값으로 채택 — 2026-10-31 (2026-11-01 갱신)

`implementations/all_lengths_20261031`에서 측정한 길이별 최적 구성을 생성기 기본 동작으로 채택했다. 이 문서는 무엇을 바꿨는지와 그 근거를 정리한다.

**2026-11-01 갱신:** 병목 분석 제안 실험(round 16–21, `report_r16_ko.md`)의 조건별 최고 구성으로 기본값을 바꿨다.
- leaf 토큰 IX·W16·PL·PB를 추가했고, recursive 경로에는 UX 단계를 추가했다.
- 64를 뺀 16개 조건이 round 15 기본값보다 3.3–11.4% 빠르다.
- 측정에서 손해였거나 시뮬레이터에서 실행되지 않은 토큰(U2, TRH, AOS, SP, four-step)은 옮기지 않았다.

## 1. 요약

| 항목 | 내용 |
|---|---|
| 새 옵션 | `length_tuning` (API `generate_recursive_fft_kernels`, `make_fft_kernel`) / `--length-tuning` (CLI) |
| 기본값 | `auto`. plan의 (N, batch)에 대해 측정한 최적 구성으로 커널을 다시 쓴다. 조건은 렌더 구성이 측정한 구성과 같을 때이고, 그 밖의 구성은 출력이 전과 같다 |
| 끄기 | `length_tuning='off'` / `--length-tuning off`. 출력이 이전과 바이트 단위로 같다 |
| 명시 지정 | leaf spec(예: `D1+TR+R8x4x8+IX+PB`) 또는 `L`, slot 토큰, `UX`(예: `L+A1x2+F12+B2x+B4+UX`). 지원하지 않는 구성이면 사유와 함께 거부한다 |
| 새 측정 | 없음. 기본값으로 만든 소스가 측정한 최적 소스와 바이트 단위로 같다(18개 조건). `off`는 측정한 baseline과 같다 |

## 2. 기본값이 적용되는 범위

모든 조건은 forward 변환이고, target profile은 측정에 쓴 28-unit profile(`MEASURED_TARGET`)이다. 저장소 기본 target(32 unit)에서는 적용되지 않는다.

| 경로 | 측정한 구성(이때만 auto 적용) | 선택 |
|---|---|---|
| N ≤ 4096 (단일 leaf) | VkFFT-M2NDP leaf(`FFTVkfftLeaf0`), distributed 실행 + fuse_stages=1, compute_lanes=4, narrow_middle_stages, spread_across_units, task fusion off | 측정한 (N, batch)는 측정값을 쓴다. 측정하지 않은 batch는 같은 N에서 가장 가까운 측정 batch의 spec을 쓴다(log 거리, 동률이면 큰 batch). 이때 O<P>는 ceil(batch/28) owner로 바꾼다. S<P>는 SPAD arena(28672 floats)를 넘으면 D4로 바꾼다 |
| N ≥ 8192 (2단 fused grouped) | task fusion auto, leaf SPAD, grouped + PRE, near C4 / far C1, far12-private, far0-short-live, compute_lanes 4. B1은 phase-packed + native4-control + window 4, B>1은 live-banks + native4-latch-forward + window 8. layout_rewrite off 또는 n8+f8+sp | 측정한 (8192, 1), (16384, 1), (16384, 8)만 적용한다 |

측정 결과(구성당 1회, 28-unit/1640MHz 시뮬레이터). 시간 단위는 µs이다.

| N × B | 이전 | 기본값(auto) | 변화 | round 15 기본값 | spec |
|---|---:|---:|---:|---:|---|
| 64×1 | 8.632 | 7.402 | −14.2% | 7.412 | O1+IX+PB |
| 64×256 | 21.844 | 7.721 | −64.7% | 7.793 | O10+IX+PB |
| 128×1 | 11.355 | 7.403 | −34.8% | 7.724 | D1+TR+R8x4x4+IX+PB |
| 256×1 | 11.617 | 7.802 | −32.8% | 8.115 | D1+TR+R8x4x8+IX+PB |
| 256×4 | 12.622 | 7.970 | −36.9% | 8.255 | D1+TR+R8x4x8+IX+PB |
| 256×16 | 13.549 | 8.006 | −40.9% | 8.300 | D1+TR+R8x4x8+IX+PB |
| 256×64 | 19.276 | 8.092 | −58.0% | 8.696 | S24+J+TR+R8x4x8+IX+PL+PB |
| 512×1 | 11.960 | 7.987 | −33.2% | 8.532 | D1+TR+R8x4x4x4+IX+PB |
| 1024×1 | 13.554 | 8.560 | −36.8% | 9.040 | D1+TR+R8x4x4x8+IX+PB |
| 1024×4 | 17.905 | 8.586 | −52.0% | 9.212 | D2+TR+R8x4x8x4+IX+PB |
| 1024×16 | 42.715 | 8.925 | −79.1% | 9.673 | S32+TR+R8x8x4x4+IX+PL |
| 1024×64 | 73.329 | 11.821 | −83.9% | 12.931 | S48+J+TR+R8x4x8x4+IX+W16 |
| 2048×1 | 13.409 | 8.765 | −34.6% | 9.216 | D1+TR+R8x4x4x4x4+IX+PB |
| 4096×1 | 16.978 | 9.190 | −45.9% | 9.717 | D1+TR+R8x4x4x8x4+IX+W16+PB |
| 4096×4 | 32.580 | 9.635 | −70.4% | 10.875 | D4+TR+R8x4x8x4x4+IX+PL+PB |
| 8192×1 | 21.913 | 11.184 | −49.0% | 12.557 | L+A1x2+F12+B2x+B4+UX |
| 16384×1 | 19.724 | 13.110 | −33.5% | 14.120 | L+A1x2+F12+B2x+B4+UX |
| 16384×8 | 46.096 | 21.724 | −52.9% | 22.457 | L+A1x6+F12+B4+B3+UX |

16384의 "이전" 값은 layout_rewrite n8+f8+sp를 쓴 기존 선택 경로다. 그 밖의 "이전" 값은 같은 옵션으로 만든 생성기 출력이다.

## 3. 파일별 변경

### 3.1 `codegen/length_tuning/` (신규 패키지)

측정에 쓴 변환 코드(`implementations/all_lengths_20261031/transforms/`)를 옮겼다. 채택 구성이 쓰는 토큰만 남겼고, 로직은 바꾸지 않았다.

| 파일 | 출처 | 내용 |
|---|---|---|
| `__init__.py` | 신규 | `MEASURED` 표, `MEASURED_TARGET`, `select_leaf_spec`, `resolve_length_tuning`(적용 여부와 사유), `apply_length_tuning`, UX(fused task 안의 음이 아닌 index 변수 div/mod를 unsigned로) |
| `leaf.py` | `leafgen.py` | N ≤ 4096 leaf task를 다시 생성한다: 8-lane Stockham pass, S<P>/O<P>/D<G>, TR, R(radix 2/4/8), J, IX(shift/mask index), W16(radix ≤ 4 pass 16-lane), PL(pass 0 연속 store + pass 1 strided load), PB(item당 base pointer). 측정에서 손해였거나 효과가 없었던 H, K, U, E, M, radix 16/32, U2, TRH, SP와, 시뮬레이터가 실행하지 못하는 AOS, four-step은 뺐다 |
| `layout.py` | `layout_general.py` | 토큰 L. n8+f8+sp를 far 길이 N/64로 일반화했다. N=16384에서는 layout_rewrite 결과와 바이트 단위로 같다 |
| `slots.py` | `slots.py` | A1xG(n/f), F12, B2/B2x, B3, B4(n/f). A3n/A3f/B5/A4(측정에서 손해)는 뺐다 |
| `_source.py`, `_codelets.py` | `variants.py`, `far_radix16.py` | phase 탐색, strided store, 상수 twiddle과 radix-4 codelet helper |

변환 단계마다 자기가 바꿀 코드의 모양을 확인한다. 모양이 다르면 `ValueError`로 거부한다.

### 3.2 `codegen/fft_transpose_codegen.py`

- `generate_recursive_fft_kernels(plan, *, layout_rewrite="off", length_tuning="auto", **options)`
- 렌더 → layout_rewrite → length_tuning 순서로 처리한다.
- 적용 여부는 render 기본값과 넘겨받은 옵션을 합친 값으로 `resolve_length_tuning`이 정한다.

### 3.3 `make_fft_kernel.py`

- `_make_gpu_baseline_fft_kernel`, `make_fft_kernel`에 `length_tuning="auto"`를 추가하고 생성기로 전달한다.
- CLI에 `--length-tuning`을 추가했다.
- `auto`/`off`가 아닌 명시 spec은 `gpu-vkfft-m2ndp` planner에서만 허용한다.

### 3.4 `verification/verify_fft_length_tuning.py` (신규)

| 테스트 | 내용 |
|---|---|
| 측정 소스 재현 | 기본값으로 만든 18개 조건의 소스 SHA-256이 측정한 소스와 같다 |
| 16384 layout off | layout_rewrite off에서 시작해도 L 토큰으로 같은 소스에 도달한다 |
| 미측정 구성 | worker-wave, inverse, 32-unit target, 미측정 recursive batch(16384×4), 기본 옵션 호출은 출력이 `off`와 같다 |
| spec 선택 규칙 | nearest batch, O 확장, S → D4 fallback을 확인한다 |
| 명시 spec 검사 | 지원하지 않는 구성과 알 수 없는 토큰을 거부한다 |

## 4. 검증 (새 측정 없음)

- **18개 조건:** 기본값 출력이 측정한 최적 소스와 같고, `off`는 측정한 baseline과 같다(`implementations/all_lengths_adoption_20261031/verify_adoption.py`).
- **leaf 변형:** 정확도 검사를 통과한 leaf 변형 중 남긴 토큰으로만 된 197개(round 1–21)가 모두 같은 소스로 재현된다.
- **미측정 batch:** 기본값이 고르는 구성을 Python 실행 검사기로 확인했다(`verify_unmeasured.py`).
  - 대상은 19개 (N, batch)다. 예: 64×100 → O4+IX+PB, 1024×256 → D4+TR+R8x4x8x4+IX+W16.
  - 모두 상대 오차 2e-7 이하로 PASS했다.
- **기존 테스트:** 모듈 10개, 72개 테스트가 통과한다.
  - boundary_simd 12, leaf_ownership 6, task_fusion 7, batch_windows 5, unit_parallel 11, baseline_codegen_support 8
  - gpu_baseline_vkfft_multipass 7, task_scratchpad 2, boundary_fusion 7, execution_codegen_fixes 7
- **새 테스트:** `verify_fft_length_tuning` 5개가 통과한다.

## 5. 주의

- 측정은 구성당 1회다. 1% 안쪽의 순위 차이는 확정할 수 없다. 64×1(7.412 → 7.402), 64×256(7.793 → 7.721)은 측정 최고를 그대로 썼지만 개선으로 보지 않는다.
- 측정하지 않은 batch의 구성은 정확도만 확인했고, 시간은 측정하지 않았다.
- benchmark harness(`experiments/m2ndp_fft/scripts/`, 저장소 밖)는 바꾸지 않았다.
  - harness는 생성기를 부르므로 기본값이 그대로 적용된다.
  - 다만 metadata에 `length_tuning`이 기록되지 않는다.
  - 기본값이 바뀌기 전에 시작한 실험을 resume하면 새 소스가 생성된다.
