# TODO: vmap 최적화 구현

**상태**: ⏳ 미처리 (Backlog)

**목표**: JAX `vmap`을 사용하여 여러 episode를 GPU/TPU에서 병렬로 실행하여 성능을 극대화합니다.

---

## 요구사항

### 기술 제약
- **현재 구조**: Python 반복문 + mutable 객체 기반
- **vmap 요구사항**: 순수 JAX 함수 (pure functions) 필수
- **필수 리팩터링**: JAX `scan`을 사용한 episode 루프 순수화

### 구현 범위

#### 1. 순수 함수화 (Pure Function Refactoring)
```python
# 목표: step() 루프를 functional 스타일로 변경
# 현재: 
#   for step in range(num_steps):
#       self.step(context)  # mutable state 변경
# 
# 개선:
#   def step_pure(state, context):
#       return new_state
#   
#   final_state = jax.scan(step_pure, initial_state, contexts)
```

#### 2. vmap 적용
```python
# 목표: 여러 episode 병렬화
# vmapped_run = jax.vmap(run_episode_pure)
# results = vmapped_run(seeds)  # shape: (num_episodes, num_steps)
```

#### 3. GPU/TPU 최적화
- in_axes/out_axes 신중히 설정
- 배치 크기 최적화 (GPU 메모리 고려)
- 성능 측정 및 비교

---

## 입력/의존성

- ✅ Sprint 5 완료 (jit 적용)
- ✅ Sprint 6 완료 (코드 품질 개선)
- 📋 Functional programming 패턴 학습 필요

---

## 출력/결과물

1. **리팩터링된 코드**:
   - `src/experiment.py`: 순수 함수화된 episode 실행 로직
   - `src/algorithms/oful.py`: pure function 버전의 step 로직

2. **벤치마크 결과**:
   - Sequential vs vmap 성능 비교
   - GPU/CPU 병렬 처리 효과 측정

3. **테스트**:
   - 모든 기존 테스트 통과 (44/44)
   - Vmap 결과가 sequential과 동일함을 검증

---

## 예상 성능

| 시나리오 | 개선 효과 |
|---------|---------|
| **100 episodes** | 10-50배 병렬화 |
| **GPU 활용** | SIMD + GPU core 병렬 |
| **1000 episodes** | 수십 배 전체 속도 향상 |

---

## 참고 자료

- JAX `scan` 문서: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
- JAX `vmap` 문서: https://jax.readthedocs.io/en/latest/api.html#vectorization-vmap

---

**Last Updated**: 2026-04-03
**Priority**: Medium (선택사항, 대규모 리팩터링)
**Estimated Effort**: Large (2-3 sprint 규모)
