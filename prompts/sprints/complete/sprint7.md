# Sprint 7 완료 요약

**상태**: ✅ **완료** (2026-04-09)

## 완료 조건 체크

- ✅ Task 7.3: jax.jit 구조 변경 (top-level jit 통합)
- ✅ Task 7.1: YAML + OmegaConf config 시스템 도입
- ✅ Task 7.2: Benchmark 스크립트 argparse 전환
- ✅ Task 7.4: 재사용성 & 중복 제거
- ✅ 모든 기존 테스트 통과 (44/44)
- ✅ Ruff 이슈 0개

---

## 주요 변경 사항

### Task 7.3: jax.jit Top-level 통합

- `oful.py`: helper 함수 5개에서 개별 `@jax.jit` 제거
- 두 개의 top-level jit 함수 신규 추가:
  - `jit_oful_select_action(...)` — θ̂ 추정 + confidence radius + UCB 계산 묶음
  - `jit_oful_update(...)` — design matrix inv 갱신 + sum_reward_context 갱신 묶음
- `OFUL.select_action()`, `OFUL.update()` → 위 두 함수 호출로 변경
- 효과: `jax_disable_jit=True` 설정 시 두 함수만 비활성화되면 충분, 단일 진입점으로 컴파일 구조 명확화

### Task 7.1: YAML + OmegaConf Config 시스템

- `uv add omegaconf` 의존성 추가
- `configs/benchmark.yaml` — benchmark 전용 (context_dim=10, num_arms=20, num_steps=500, warmup/trial 설정)
- `configs/test.yaml` — 테스트 전용 (context_dim=3, num_arms=5, num_steps=10, num_episodes=2)
- `configs/experiment.yaml` — 실험 전용 (num_steps=5000, num_episodes=10)
- JAX 호환성: `OmegaConf.to_container(cfg, resolve=True)` → Python native dict로 변환 후 사용

### Task 7.2: Benchmark argparse 전환

| 이전 | 이후 |
|------|------|
| `DISABLE_JIT=1 uv run python scripts/benchmark_jit.py` | `uv run python scripts/benchmark_jit.py --disable_jit` |
| 하드코딩된 상수 (CONTEXT_DIM = 10, …) | `configs/benchmark.yaml` 에서 로드 |
| `--config` 플래그로 경로 변경 가능 | 기본값: `configs/benchmark.yaml` |

- `jax.config.update("jax_disable_jit", True)` 호출 순서 유지 (JAX import 직후, src import 이전)

### Task 7.4: 재사용성 & 중복 제거

- `_compute_radius()`, `_compute_ellipsoid_norm()`: 테스트에서 참조 중 → 유지 (module-level 함수로 위임하는 구조는 그대로)
- `experiment.py` `_run_episode_loop()`: `cumulative_regrets.at[t].set(...)` (루프마다 새 JAX 배열 생성) → Python list append 후 `jnp.array(...)` 변환으로 개선
- `ExperimentRunner.run()`: 이미 `run_single_episode_sequential`을 재사용 중 — 중복 없음 확인

---

**Last Updated**: 2026-04-09
