# Sprint 6: 코드 품질 개선 (개선.md 반영)

**상태**: ✅ **완료** (2026-04-03)

**목표**: Sprint 5의 JAX 최적화(jit)를 유지하면서 코드 품질을 향상시킵니다. Sherman-Morrison 공식으로 복잡도 개선, 명확한 변수명으로 가독성 향상, 올바른 파라미터 norm bound 적용, 중복 코드 제거로 간결성을 확보합니다.

**완료 조건**:
- ✅ Task 6.1: Sherman-Morrison 공식 적용
- ✅ Task 6.2: 변수명 및 Attribute 명명 규칙 통일
- ✅ Task 6.3: Parameter norm bound 올바르게 적용
- ✅ Task 6.4: 중복 유효성 검사 제거
- ✅ Task 6.5: JAX jit 성능 비교 벤치마크
- ✅ 모든 기존 테스트 통과 (44/44)
- ✅ Ruff 이슈 0개

---

## Task 6.1: Sherman-Morrison 공식 적용

**설명**: 설계 행렬(Design Matrix) 역행렬을 Sherman-Morrison 공식으로 최적화하여 `np.linalg.solve()` 제거, O(d³) → O(d²) 복잡도 감소 (예상 20-30% 성능 개선)

**파일**: `src/algorithms/oful.py`의 `update_b_t()` 함수

**입력/의존성**:
- Sprint 5 완료 (jit 적용된 기반)

**출력/결과물**:
- Sherman-Morrison 적용된 `update_b_t_inv()` 함수
- 벤치마크 결과로 성능 개선 검증

**평가 기준**:
- ✅ 계산 결과가 기존 `np.linalg.inv()` 방식과 일치하는가?
- ✅ 벤치마크 상 20-30% 이상 성능 개선이 측정되는가?
- ✅ 수치 안정성 문제는 없는가?
- ✅ 테스트 통과 (44/44)?

**참고**: 수식, 구현 세부사항, 주의사항은 [상세 가이드](../details/task6.1.sherman_morrison.md) 참고

---

## Task 6.2: 변수명 및 Attribute 명명 규칙 통일

**설명**: 수학 표기법(d, K, T, r, s) 대신 명확한 파이썬 변수명(context_dim, num_arms, num_steps, radius_t, norm_bound) 사용으로 PEP 8 준수 및 가독성 향상

**파일**: `src/algorithms/oful.py`, `src/environments/contextual_linear.py`, `src/experiment.py`, `tests/`

**입력/의존성**:
- Sprint 5 완료

**출력/결과물**:
- PEP 8 snake_case 준수, 모든 호출처 업데이트

**변경 대상 (예시)**:

| 현재 | 개선 후 | 설명 |
|------|--------|------|
| `d` | `context_dim` | Context dimension |
| `K` | `num_arms` | Number of arms |
| `T` | `num_steps` | Number of steps |
| `r` | `radius_t` | Confidence radius at time t |
| `s` | `norm_bound` | Parameter norm bound |
| `self.B_t` | `self.design_matrix` | Design matrix |
| `self.V_t` | `self.design_matrix_inv` | Inverse of design matrix |
| `self.sum_r_x` | `self.sum_reward_context` | Sum of reward × context |

**평가 기준**:
- ✅ 모든 변수/attribute가 snake_case로 변경되었는가?
- ✅ 호출처 모두 업데이트되었는가?
- ✅ 테스트 통과 (44/44)?
- ✅ Ruff 이슈 0개?

---

## Task 6.3: Parameter Norm Bound 올바르게 적용

**설명**: OFUL 알고리즘의 true parameter θ를 올바르게 생성 및 사용. 현재 θ를 정규화(norm = 1)하는 것을 수정하여, norm bound `S` (상수)를 올바르게 사용하도록 변경

**파일**: `src/environments/contextual_linear.py`, `src/algorithms/oful.py`

**입력/의존성**:
- Sprint 5 완료
- Task 6.2 권장 (변수명 통일)

**출력/결과물**:
- Parameter norm bound 수학적으로 정확하게 적용
- 신뢰도 반경 계산에 상수 `norm_bound` 사용

**핵심 변경**:
- θ* 생성: `||θ*||₂ < norm_bound`를 만족하도록 생성 (정규화 제거)
- OFUL 신뢰도 반경: 상수 `norm_bound` 사용 (실제 norm 아님)

**평가 기준**:
- ✅ θ*가 norm_bound 이내에서 생성되는가? (정규화 제거)
- ✅ OFUL이 상수 norm_bound를 사용하는가?
- ✅ 신뢰도 반경 계산이 수학적으로 정확한가?
- ✅ 테스트 통과 (44/44)?

**참고**: 수학적 배경, 구현 세부사항은 [상세 가이드](../details/task6.3.norm_bound.md) 참고

---

## Task 6.4: 중복 유효성 검사 제거

**설명**: 같은 검증이 여러 곳에서 반복되는 부분을 제거하여 코드 간결화. 검증은 public 진입점(step, select_action)에만 수행하고, 내부 함수는 신뢰 기반으로 처리

**파일**: `src/experiment.py`, `src/algorithms/oful.py`

**입력/의존성**:
- Sprint 5 완료

**출력/결과물**:
- 중복 검사 제거된 코드
- 불필요한 검증 로직 삭제

**평가 기준**:
- ✅ 중복 검사가 제거되었는가?
- ✅ 필수 검사(공개 인터페이스)는 유지되었는가?
- ✅ 기능 동작이 변경되지 않았는가?
- ✅ 테스트 통과 (44/44)?

---

## Task 6.5: JAX jit 성능 비교 벤치마크

**설명**: JAX의 `DISABLE_JIT` 환경 변수를 활용하여 jit 적용 전후 성능을 비교하는 벤치마크 스크립트 작성. 터미널에서 즉시 실행 가능

**파일**: `scripts/benchmark_jit.py` (신규 생성)

**입력/의존성**:
- Sprint 5 완료 (jit 적용)
- Task 6.1-6.4 완료

**출력/결과물**:
- `scripts/benchmark_jit.py` 벤치마크 스크립트
- jit enabled/disabled 성능 비교 결과 기록

**실행 방법**:
```bash
# jit 활성화
uv run python scripts/benchmark_jit.py

# jit 비활성화
DISABLE_JIT=1 uv run python scripts/benchmark_jit.py
```

**평가 기준**:
- ✅ `DISABLE_JIT` 환경 변수로 제어 가능한가?
- ✅ jit enabled가 disabled보다 빠른가?
- ✅ 성능 개선이 정량화되어 있는가?
- ✅ 측정값이 합리적인가? (일반적으로 5-10배 개선 예상)
- ✅ 테스트 통과 (44/44)?

---

## 예상 효과

| 개선 사항 | 효과 |
|---------|------|
| Sherman-Morrison (Task 6.1) | O(d³) → O(d²) (20-30% 성능 향상) |
| 변수명 통일 (Task 6.2) | 가독성 대폭 향상, 유지보수 용이 |
| Norm bound 수정 (Task 6.3) | 알고리즘 정확성 향상 |
| 중복 제거 (Task 6.4) | 코드 간결화, 버그 감소 |
| 벤치마크 (Task 6.5) | jit 효과 정량화 |

**최종 목표**: 44/44 테스트 통과, Ruff 0 이슈, 전체 성능 1-2배 향상

---

**Last Updated**: 2026-04-03  
**Status**: ⏳ Planning Phase  
**Next**: Sprint 6 Implementation (Developer)
