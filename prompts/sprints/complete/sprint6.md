# Sprint 6 완료 요약

**상태**: ✅ **완료** (2026-04-03)

## 완료 조건 체크

- ✅ Task 6.1: Sherman-Morrison 공식 적용
- ✅ Task 6.2: 변수명 및 Attribute 명명 규칙 통일
- ✅ Task 6.3: Parameter norm bound 올바르게 적용
- ✅ Task 6.4: 중복 유효성 검사 제거
- ✅ Task 6.5: JAX jit 성능 비교 벤치마크
- ✅ 모든 기존 테스트 통과 (44/44)
- ✅ Ruff 이슈 0개

---

## 주요 변경 사항

### Task 6.1: Sherman-Morrison 공식 적용

- `update_b_t()` 제거 → `update_design_matrix_inv()` 신규 추가
- `B_t^{-1}` 를 직접 관리하여 O(d³) → O(d²) 복잡도 개선
- `compute_theta_hat` : `jnp.linalg.solve()` → 단순 행렬-벡터 곱 `B^{-1} @ sum`
- `compute_ucb_values` : Python loop 제거 → `contexts @ design_matrix_inv` 벡터화

### Task 6.2: 변수명 통일

| 변경 전 | 변경 후 | 파일 |
|---------|--------|------|
| `d` | `context_dim` | 전체 |
| `self.B_t` | `self.design_matrix_inv` | oful.py |
| `self.sum_r_x` | `self.sum_reward_context` | oful.py |
| `compute_radius` | `compute_confidence_radius` | oful.py |

### Task 6.3: Norm bound 수정

- `contextual_linear.py`: θ* 생성 방식 변경
  - 기존: 단위 구면으로 정규화 (`norm = 1`)
  - 변경: `||θ*||₂ ≤ param_norm_bound` 만족 (clipping 방식)
- `oful.py` 신뢰도 반경 공식 버그 수정:
  - 기존 (버그): `log(1 + t * L² / (λ * δ))`
  - 수정: `log((1 + t * L²/λ) / δ)` ← 수학적으로 올바른 OFUL 공식
- 파라미터 명칭 정리:
  - `r` (sub-Gaussian proxy) → `subgaussian_scale`
  - `s` (파라미터 norm bound) → `norm_bound`
  - `norm_bound` (컨텍스트 norm bound) → `context_bound`

### Task 6.4: 중복 유효성 검사 제거

- `select_action()` / `get_theta_hat()` 에서 중복된 `LinAlgError` fallback 제거
  (Sherman-Morrison 적용으로 `jnp.linalg.solve()` 자체가 제거됨)

### Task 6.5: JAX jit 성능 비교 벤치마크

- `scripts/benchmark_jit.py` 신규 생성
- 실행: `uv run python scripts/benchmark_jit.py` (JIT enabled)
- 실행: `DISABLE_JIT=1 uv run python scripts/benchmark_jit.py` (JIT disabled)
- 측정 결과 (context_dim=10, num_arms=20, num_steps=500):
  - JIT ENABLED:  0.80s ± 0.09s
  - JIT DISABLED: 1.82s ± 0.14s
  - **성능 개선: 약 2.3배**

---

**Last Updated**: 2026-04-03
