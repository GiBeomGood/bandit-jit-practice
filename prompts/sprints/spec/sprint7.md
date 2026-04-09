# Sprint 7: Config 관리 & JAX JIT 구조 개선

**상태**: ✅ 완료  
**생성일**: 2026-04-09

---

## 전체 목표

현재 코드의 두 가지 구조적 문제를 해결한다:

1. **설정값이 코드에 하드코딩**되어 있어 재사용성이 낮음 → YAML + OmegaConf 도입
2. **jax.jit이 작은 단위 함수마다 개별 적용**되어 있어 컴파일 구조가 비효율적 → top-level jit으로 통합

추가로 중복 코드 및 재사용성 문제를 정리한다.

---

## 완료 조건

- [ ] `configs/benchmark.yaml`, `configs/test.yaml`, `configs/experiment.yaml` 생성
- [ ] `scripts/benchmark_jit.py`가 `--disable_jit` argparse 플래그로 동작
- [ ] `scripts/benchmark_jit.py`가 config 파일에서 파라미터를 로드
- [ ] `oful.py` 내 개별 `@jax.jit` 제거 → `jit_select_action`, `jit_update` top-level jit 함수로 통합
- [ ] 중복 메서드(`_compute_radius`, `_compute_ellipsoid_norm`) 정리
- [ ] 모든 기존 테스트 통과
- [ ] Ruff 이슈 0개

---

## Task 목록

### Task 7.1: YAML + OmegaConf Config 시스템 도입

**설명**: 현재 `benchmark_jit.py`에 하드코딩된 파라미터들과 `ExperimentRunner`에 직접 전달되는 값들을 YAML 설정 파일로 분리한다. OmegaConf로 로드하되, JAX jit/vmap과 호환되도록 config 값은 Python native type으로 변환해서 사용한다.

**입력/의존성**:

- `scripts/benchmark_jit.py` (현재 하드코딩된 CONTEXT_DIM, NUM_ARMS 등)
- `src/experiment.py` (ExperimentRunner 파라미터)
- `tests/test_experiment.py` (테스트에서 직접 파라미터 전달 중)
- `pyproject.toml` (omegaconf 의존성 추가 필요)

**출력/결과물**:

- `configs/benchmark.yaml` 신규 생성 (benchmark 전용)
- `configs/test.yaml` 신규 생성 (테스트 전용, 빠른 실행)
- `configs/experiment.yaml` 신규 생성 (실험 전용, 충분한 rounds)
- `uv add omegaconf` 실행 결과 반영

**평가 기준**:
1. OmegaConf `DictConfig` 로드 후 Python dict/native type으로 변환하여 JAX 함수에 전달하는가?
2. config 값이 `configs/` 파일 외 코드에 상수로 존재하지 않는가?
3. test.yaml의 값이 빠른 실행에 적합한가? (num_steps ≤ 20, num_episodes ≤ 2)
4. 기존 테스트가 전부 통과하는가? (테스트 코드 자체는 수정 금지)

**추가 노트**: OmegaConf의 `DictConfig`는 jit/vmap에 직접 전달 불가. `OmegaConf.to_container(cfg, resolve=True)`로 Python dict로 변환 후 사용할 것.

---

### Task 7.2: Benchmark 스크립트 argparse 전환

**설명**: 현재 `DISABLE_JIT=1 uv run python scripts/benchmark_jit.py` 방식을 `uv run python scripts/benchmark_jit.py --disable_jit`으로 변경한다. 추가로 `--config` 플래그로 config 파일 경로를 지정할 수 있게 한다 (기본값: `configs/benchmark.yaml`).

**입력/의존성**:

- Task 7.1 완료 (configs/benchmark.yaml 존재해야 함)
- `scripts/benchmark_jit.py` (현재 환경변수 방식)

**출력/결과물**:

- `scripts/benchmark_jit.py` 수정 (argparse + config 로드)
- 실행 예시:
  - `uv run python scripts/benchmark_jit.py` (기본: JIT enabled)
  - `uv run python scripts/benchmark_jit.py --disable_jit`
  - `uv run python scripts/benchmark_jit.py --config configs/benchmark.yaml --disable_jit`

**평가 기준**:
1. `--disable_jit` 플래그 없이 실행 시 JIT ENABLED로 동작하는가?
2. `--disable_jit` 플래그 사용 시 JIT DISABLED로 동작하는가?
3. `jax.config.update("jax_disable_jit", True)`가 JAX import 이후, 다른 코드 import 이전에 호출되는가? (순서 중요)
4. config 파일에서 파라미터를 로드하여 벤치마크에 사용하는가?

---

### Task 7.3: jax.jit 구조 변경 (Top-level JIT 통합)

**설명**: `oful.py`의 5개 함수에 개별 적용된 `@jax.jit`을 제거하고, `select_action`과 `update` 단위로 묶은 두 개의 top-level jit 함수(`jit_oful_select_action`, `jit_oful_update`)로 통합한다. 개별 helper 함수는 순수 함수(pure function)로 유지한다.

**입력/의존성**:

- `src/algorithms/oful.py` (현재 5개 함수에 `@jax.jit` 적용)
- `tests/test_oful.py` (변경 후에도 통과해야 함)

**출력/결과물**:

- `src/algorithms/oful.py` 수정
  - helper 함수 5개에서 `@jax.jit` 제거
  - `jit_oful_select_action(design_matrix_inv, sum_reward_context, contexts, t, context_dim, lambda_, subgaussian_scale, norm_bound, context_bound, delta) → int` 신규 추가 (jit 적용)
  - `jit_oful_update(design_matrix_inv, sum_reward_context, context, reward) → (new_inv, new_sum)` 신규 추가 (jit 적용)
  - `OFUL.select_action()`, `OFUL.update()`는 위 jit 함수를 호출하도록 수정

**평가 기준**:
1. 개별 helper 함수(`compute_theta_hat` 등)에 `@jax.jit` 데코레이터가 없는가?
2. `jit_oful_select_action`, `jit_oful_update`에만 `@jax.jit`이 적용되는가?
3. `jax.config.update("jax_disable_jit", True)` 설정 시 두 jit 함수 모두 비활성화되는가?
4. 기존 테스트 전부 통과하는가?

**추가 노트**: `jit_oful_update`는 JAX array 두 개를 tuple로 반환. `OFUL.update()` 메서드에서 unpacking하여 `self.design_matrix_inv`, `self.sum_reward_context` 갱신.

---

### Task 7.4: 재사용성 & 중복 제거

**설명**: 코드 전반에서 불필요한 중복 메서드와 반복 연산을 제거한다.

**검토 대상 및 지침**:

1. `OFUL._compute_radius()` — `compute_confidence_radius()`와 동일 로직을 래핑. 삭제하고 외부에서 직접 `compute_confidence_radius()` 사용.
2. `OFUL._compute_ellipsoid_norm()` — `compute_ucb_values()` 내 로직과 부분 중복. 테스트에서 사용 여부 확인 후, 사용되지 않으면 삭제.
3. `experiment.py` — `run_single_episode_sequential`과 `ExperimentRunner.run()` 간 중복 여부 재확인. `ExperimentRunner.run()`이 이미 `run_single_episode_sequential`을 호출하면 문제없음.
4. 반복 연산 — 루프 내에서 같은 값을 매번 재계산하는 패턴이 있는지 확인 (`_run_episode_loop` 포함).

**입력/의존성**:

- Task 7.3 완료 후 수행 (jit 구조 확정 후)
- `src/algorithms/oful.py`, `src/experiment.py`

**출력/결과물**:

- `src/algorithms/oful.py` 수정 (불필요 메서드 제거)
- `src/experiment.py` 수정 (필요 시)
- 기존 테스트 전부 통과

**평가 기준**:
1. 삭제한 메서드/함수가 테스트나 다른 코드에서 참조되지 않는가?
2. 삭제 후에도 동등한 기능이 다른 경로로 제공되는가?
3. 루프 내 불필요한 반복 연산이 없는가?
4. 기존 테스트 전부 통과하는가?

---

## 실행 순서 및 의존성

```
Task 7.3 (jit 구조) ─┐
                      ├──→ Task 7.4 (중복 제거)
Task 7.1 (config)  ──┤
                      └──→ Task 7.2 (argparse)
```

- **Phase 1** (독립): Task 7.3, Task 7.1 (순서 무관)
- **Phase 2** (의존): Task 7.4 (7.3 완료 후), Task 7.2 (7.1 완료 후)

**Last Updated**: 2026-04-09
