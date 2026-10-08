# Execution 지원 계획 1번: task 등록 함수 수 검사

`planning/execution/kernel_registration.py`에 세 FFT execution strategy의 공통
등록 함수 inventory와 validation을 추가했다. `codegen.lowering.try_build_loop_stage`를
직접 사용하므로 emitter와 동일한 loop 가능 여부 및 tail 판정을 사용한다.
Generated Mojo 문자열이나 FFT stage 수만으로 추정하지 않는다.

- Plain: stage 함수와 loop가 분리한 tail 함수.
- Cooperative: stage 함수와 stage별 공유 tail 함수. 여러 worker의 tail은 한 번만 센다.
- Persistent: preload, stage 함수들, writeback. Rounds와 worker waves의 반복 호출은
  distinct 함수 수를 늘리지 않는다.
- Recursive tree: 한 task씩 검사한다. 여러 task의 함수 수를 합산하지 않는다.
  Physical transpose는 한 함수를 등록한다.

Native search의 `add`에서 cost 계산 및 ranking **전에** 한도 초과 후보를 제외한다.
Generic recursive codegen과 standalone plain/cooperative/persistent codegen에서도
실제 rendering 조건으로 재확인한다. Standalone plain/persistent generator는
`target`을 지정할 수 있고, 생략하면 기존 default target을 사용한다.
GPU 공통 cooperative mapper도 generic 경로의 loop 조건으로 검사하며 초과하면
`RESOURCE_INFEASIBLE`로 반환한다. 원래 GPU config/snapshot은 유지한다.

재현 N=729, leaf budget=16 KiB에서는 stage 6개와 tail 3개가 합쳐져 9개다.
기본 한도 8일 때 이 후보를 search에서 제거하고, 직접 renderer에 넘기면 함수
이름과 한도가 포함된 `KernelRegistrationError`로 거부한다. `loop_stages=False`면
실제 등록 수는 6개이므로 이 검사만으로는 거부하지 않는다. 이 옵션을 자동으로
바꿔 후보를 살리지는 않는다. Loop를 끄는 것이 spill-free라는 뜻도 아니다.

회귀 검증:

```powershell
python -m verification.verify_fft_kernel_registration
```

검증은 emitter의 실제 `launch_parallel` 대상과 inventory를 비교하고, stage-only
한도 초과, 정확히 8개인 경계, custom target의 작은 한도, N=729 search pruning,
persistent rounds/modes 및 GPU refusal의 원본 metadata 보존을 포함한다.
기존 native search, execution invariants, persistent search, GPU golden 및
large-1D/multipass 검증도 실행했다. 실제 binary 검증은 현재 toolchain 부재로
수행하지 못했다.

이번 변경은 실행 지원 계획의 **1번만** 구현한다. Recursive persistent target
전달, cooperative 폭 validation, mode/copy 연결 및 부가 search axis는 후속 단계다.
