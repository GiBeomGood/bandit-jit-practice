# Sprint 8: vmap 기반 다중 에피소드 병렬화

**상태**: 📋 기획 완료  
**생성일**: 2026-04-09

---

## 전체 목표

`jax.lax.scan`으로 에피소드 step 루프를 순수 함수화하고,  
`jax.jit(jax.vmap(...))` 으로 다중 에피소드를 컴파일·병렬 실행한다.

현재: Python for 루프 + mutable OFUL 객체 → 에피소드 순차 실행  
목표: pure JAX 함수 + scan + jit(vmap) → 에피소드 병렬 실행

---

## 완료 조건

- [ ] `run_episode_scan` 함수 추가 (`src/experiment.py`)
- [ ] `run_episodes_vmap` 함수 추가 (`src/experiment.py`) — `jax.jit(jax.vmap(...))` 적용
- [ ] `ExperimentRunner`가 `use_vmap: bool = False` 옵션 지원
- [ ] `tests/test_vmap.py` 추가 (결정성, shape 검증)
- [ ] `scripts/benchmark_vmap.py` 추가 (sequential vs vmap 성능 비교)
- [ ] 기존 테스트 전부 통과
- [ ] Ruff 이슈 0개

---

## Task 목록

### Task 8.1: `jax.lax.scan` 기반 순수 에피소드 함수

**설명**:  
seed 하나를 받아 cumulative regrets를 반환하는 순수 함수 `run_episode_scan`을 `src/experiment.py`에 추가한다. 내부에서 `true_theta`, `contexts`, `noises`를 seed로부터 한 번에 생성하고, step 루프를 `jax.lax.scan`으로 실행한다. step 내부에서는 기존 `jit_oful_select_action`, `jit_oful_update`를 재사용한다.

**입력/의존성**:

- `src/experiment.py`
- `src/algorithms/oful.py` (`jit_oful_select_action`, `jit_oful_update`)

**출력/결과물**:

- `src/experiment.py`에 `run_episode_scan(seed, context_dim, num_arms, num_steps, ...) -> jnp.ndarray` 추가
- 반환: cumulative regrets, shape `(num_steps,)`

**평가 기준**:

1. 동일 seed로 두 번 호출 시 동일한 결과가 나오는가? (결정성)
2. 반환 shape이 `(num_steps,)`인가?
3. cumulative regrets가 단조 증가(non-decreasing)인가?
4. `jit_oful_select_action`, `jit_oful_update`를 재사용하는가? (중복 구현 금지)

**추가 노트**: `noises`는 `jax.random.normal(..., shape=(num_steps,))`으로 미리 전체 생성해 scan `xs`로 전달한다. `t`는 `jnp.arange(num_steps)`를 xs에 포함하면 scan 내부에서 자연스럽게 활용 가능하다.

---

### Task 8.2: `jax.jit(jax.vmap(...))` 기반 다중 에피소드 병렬화

**설명**:  
`run_episode_scan`을 `jax.vmap`으로 배치화한 뒤 `jax.jit`으로 컴파일하는 `run_episodes_vmap`을 추가한다. `ExperimentRunner`에 `use_vmap` 옵션을 추가해 기존 sequential 경로와 선택적으로 사용 가능하게 한다.

```
jit_vmapped = jax.jit(jax.vmap(episode_fn))  # episode_fn: seed만 가변
result = jit_vmapped(seeds)                   # shape: (num_episodes, num_steps)
```

**입력/의존성**:

- Task 8.1 완료
- `src/experiment.py`

**출력/결과물**:

- `run_episodes_vmap(seeds, context_dim, ...) -> jnp.ndarray` 추가, 반환 shape `(num_episodes, num_steps)`
- `ExperimentRunner.__init__`에 `use_vmap: bool = False` 파라미터 추가 (마지막 위치)
- `ExperimentRunner.run()`에 `use_vmap=True` 분기 추가

**평가 기준**:

1. `run_episodes_vmap` 반환 shape이 `(num_episodes, num_steps)`인가?
2. `jax.jit(jax.vmap(...))` 구조로 컴파일이 적용되어 있는가?
3. `ExperimentRunner(use_vmap=True).run()`이 정상 동작하는가?
4. `ExperimentRunner(use_vmap=False).run()`이 기존과 동일하게 동작하는가? (기존 테스트 호환)

**추가 노트**: `run_episode_scan`의 시그니처 중 seed만 vmap 대상(`in_axes=0`)이며, 나머지 파라미터는 `functools.partial`로 고정한다.

---

### Task 8.3: 테스트 및 성능 벤치마크

**설명**:  
`run_episode_scan`, `run_episodes_vmap`의 정확성을 검증하는 테스트와, sequential vs vmap 성능을 비교하는 벤치마크 스크립트를 추가한다.

**입력/의존성**:

- Task 8.1, 8.2 완료
- `configs/benchmark.yaml` (파라미터 로드)

**출력/결과물**:

- `tests/test_vmap.py` 신규 생성
- `scripts/benchmark_vmap.py` 신규 생성

**평가 기준**:

1. 다음 테스트 케이스를 모두 포함하는가?
   - `run_episode_scan`: shape 검증, 결정성, cumulative regrets 단조 증가
   - `run_episodes_vmap`: shape 검증, 서로 다른 seed는 서로 다른 결과
   - `ExperimentRunner(use_vmap=True)`: run 결과 shape 검증
2. `benchmark_vmap.py`가 sequential vs vmap 실행 시간과 speedup을 출력하는가?
3. `benchmark_vmap.py`가 `configs/benchmark.yaml`에서 파라미터를 로드하는가?
4. 기존 테스트 전부 (44/44) 통과하는가?

---

## 실행 순서

```
Task 8.1 → Task 8.2 → Task 8.3
```

---

## 주의사항

- `run_single_episode_sequential`, `OFUL` 클래스는 **수정 금지** (기존 테스트 호환)
- `ExperimentRunner` 기존 생성자는 `use_vmap` 기본값 `False`로 인해 기존 테스트 코드 수정 불필요

**Last Updated**: 2026-04-09
