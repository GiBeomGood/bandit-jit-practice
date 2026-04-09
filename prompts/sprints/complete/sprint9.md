# Sprint 9 완료 요약

**상태**: 완료 (2026-04-09)

## 완료 조건 체크

- Task 9.1: `ContextualLinearBandit`에서 순수 환경 함수 추출 — APPROVED
- Task 9.2: `jit_oful_select_action` / `jit_oful_update` 분해 및 이름 변경 — APPROVED
- Task 9.3: `ExperimentRunner.from_yaml()` 클래스 메서드 추가 — APPROVED
- Task 9.4: `run_episode_scan`을 추출된 순수 함수로 업데이트 — APPROVED
- Task 9.5: 테스트 업데이트 및 `disable_jit` 검증 — APPROVED
- 최종 테스트: 57/57 통과
- Ruff 위반: 0건

---

## 주요 변경 사항

### Task 9.1: 순수 환경 함수 추출

- `src/environments/contextual_linear.py`에 모듈 레벨 순수 함수 3개 추가:
  - `sample_true_theta(key, context_dim, param_norm_bound) -> jnp.ndarray`
  - `sample_contexts(key, num_steps, num_arms, context_dim, context_bound) -> jnp.ndarray`
  - `compute_reward(true_theta, context, noise) -> jnp.ndarray`
- `ContextualLinearBandit.reset()`, `.step()` 은 이 함수들에 위임 — 기존 클래스 API 유지
- 어떤 함수에도 `@jax.jit` 미적용 (JIT는 호출부에서 결정)

### Task 9.2: OFUL 함수 이름 변경 및 JIT 제거

| 이전 | 이후 |
|------|------|
| `jit_oful_select_action` (`@jax.jit` 적용) | `oful_select_action` (JIT 없음) |
| `jit_oful_update` (`@jax.jit` 적용) | `oful_update` (JIT 없음) |

- Sprint 7에서 도입된 함수 레벨 JIT 번들을 해제
- 이제 JIT는 최외곽 `jax.jit(jax.vmap(run_episode_scan))` 단일 진입점에서만 적용
- `src/experiment.py` 임포트 및 호출부 모두 업데이트 완료

### Task 9.3: YAML-First ExperimentRunner

- `ExperimentRunner.from_yaml(config_path: str) -> ExperimentRunner` 클래스 메서드 추가
- 내부적으로 `OmegaConf.load` + `OmegaConf.to_container(resolve=True)` 사용
- 기존 `ExperimentRunner(...)` 생성자 시그니처 완전 유지 (하위 호환성)
- `configs/experiment.yaml`, `configs/test.yaml`, `configs/benchmark.yaml` 에 `use_vmap: false` 필드 추가
- `scripts/benchmark_vmap.py` 가 `ExperimentRunner.from_yaml(args.config)` 방식으로 전환

### Task 9.4: run_episode_scan 코드 중복 제거

- `run_episode_scan` 내 인라인 theta/context 생성 코드 제거
- Task 9.1에서 추출한 `sample_true_theta`, `sample_contexts` 직접 호출로 대체
- 효과: 순차 경로(`ContextualLinearBandit`)와 scan 기반 경로(`run_episode_scan`) 가 동일한 데이터 생성 로직을 공유

### Task 9.5: 테스트 업데이트 및 검증

- `tests/test_oful.py`:
  - 임포트를 `oful_select_action`, `oful_update`로 업데이트
  - `TestDisableJit` 클래스 추가 (3개 테스트): `jax.disable_jit()` 컨텍스트 내에서 올바른 출력 형태 및 타입 검증
- `tests/test_experiment.py`:
  - `TestExperimentRunnerFromYaml` 클래스 추가 (3개 테스트): YAML 로드, 결과 shape, 생성자 호환성 검증
- 전체 57/57 테스트 통과

---

## 핵심 설계 결정

1. **JIT 단일 진입점**: 모든 JIT는 `jax.jit(jax.vmap(run_episode_scan))` 최외곽에서만 적용. 개별 헬퍼 함수에 JIT 없음.
2. **비-JIT 실행**: `jax.disable_jit()` 컨텍스트 매니저 하나로 전체 실행을 비-JIT 전환 가능. 별도 코드 경로 불필요.
3. **하위 호환성 유지**: `from_yaml` 은 추가이지, 기존 생성자 교체가 아님. 테스트 변경 최소화.
4. **코드 공유 달성**: 순수 함수 추출로 sequential 경로와 vmap 경로 간 동일한 데이터 생성 로직 공유.

---

## 후속 고려 사항

- `use_vmap: false` 기본값이 모든 YAML에 추가되었으나, 현재 대부분의 실험이 sequential로 실행됨. vmap 실험 활성화 시 `configs/benchmark.yaml`의 `use_vmap: true` 설정만으로 전환 가능.
- 추후 Thompson Sampling 등 신규 알고리즘 추가 시, 동일한 순수 함수 패턴(`algo_select_action`, `algo_update` 이름 규약)을 따르면 자연스럽게 JIT-first 설계에 통합 가능.

---

**Last Updated**: 2026-04-09
