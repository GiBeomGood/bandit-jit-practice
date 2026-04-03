# Sprint 2: 완료 요약

**목표**: 여러 episode에 걸쳐 OFUL 알고리즘을 실행하고, cumulative regret을 측정하여 numpy 배열로 저장.

**완료 조건**: 작은 파라미터(num_episodes=3, T=20)에서 실험이 동작하고 결과 저장 확인. ✅ **달성**

---

## 완료된 Task 목록

### Task 2.1: Episode Runner 및 결과 저장 ✅

**구현 결과**:
- `src/experiment.py`: `ExperimentRunner` 클래스 구현
- `tests/test_experiment.py`: 13개 테스트 모두 통과

**핵심 기능**:
- 여러 episode에 걸쳐 OFUL 실행
- 각 episode마다 Environment, Algorithm 독립 초기화 (다른 seed)
- Cumulative regret 계산: Σ(θ* · x_t^best - θ* · x_t^selected)
  - Noise-free regret (순수 선택 오류 측정)
  - 항상 non-negative (누적 단조증가)
- 결과를 numpy ndarray로 저장/로드 (.npz 형식)

**테스트 결과**:
```
13 passed in 2.96s
```

**메타데이터 저장**:
- regrets: (num_episodes, T) 형태
- configs: 전체 설정 (d, K, T, L, algo_params, seed)
- metadata: seed_base 등 추가 정보

**하이퍼파라미터 기본값** (OFUL과 일치):
- lambda_: 1.0
- R: 1.0
- S: 1.0
- delta: 0.01

---

## 생성된 파일 목록

### Source Code
```
src/
├── experiment.py              # ExperimentRunner 클래스
└── environment.py 추가        # get_contexts_at_t(t) 메서드 추가
```

### Test Code
```
tests/
├── test_experiment.py         # ExperimentRunner 테스트 (13 tests)
└── 기존 tests/                 # Sprint 1 테스트 유지 (24 tests)
```

### 전체 테스트 커버리지
```
tests/test_environment.py      # 11 tests
tests/test_oful.py            # 13 tests
tests/test_experiment.py       # 13 tests
─────────────────────
합계: 37 passed in 5.42s
```

---

## 주요 개선사항

### 1. Context 조회 메서드 추가
- `environment.py`: `get_contexts_at_t(t)` 메서드
- 목적: step()을 한 번만 호출하여 noise 일관성 보장

### 2. Regret 정의 명확화
- **Noise-free regret**: 순수 선택 오류만 측정
  - regret(t) = (θ* · x_t^best) - (θ* · x_t^selected)
  - noise는 환경의 randomness이지 알고리즘 오류가 아님
- **결과**: cumulative regret이 항상 non-negative (단조증가)

### 3. JAX 적절히 활용
- jax.numpy로 계산 수행
- numpy array로 저장 (메모리 효율성)
- seed 기반 deterministic 실행

---

## JAX 컴파일 최적화 체크리스트

✅ **JAX 기본 사용**:
- jax.numpy 연산 활용
- jax.random seed 관리

⏳ **vmap 병렬화** (선택사항):
- 다음 단계에서 구현 고려
- Episode-level vmap으로 GPU 가속화 가능

⏳ **컴파일 오버헤드 모니터링**:
- 현재는 소규모 파라미터 (num_episodes=2-5, T=10-20)
- 실제 실험에서 scaling시 jit compilation 프로파일링 예정

---

## 코드 품질 지표

- **라인 수**:
  - ExperimentRunner 클래스: ~130 lines
  - Test 코드: ~214 lines
  
- **설계 패턴**:
  - Clean API: run(), save_results(), load_results() (static method)
  - 설정과 실행의 분리
  
- **에러 처리**:
  - 파라미터 검증 (num_episodes, d, K, T)
  - seed 기반 재현성 보장

---

## 다음 단계

### Sprint 3: 결과 시각화 및 통합 테스트
- Visualizer 클래스 구현
- Mean ± 5%-95% quantile 플롯
- PDF 저장
- End-to-end 통합 테스트 (test_integration.py)

---

**Last Updated**: 2026-04-02
**Status**: ✅ Ready for Sprint 3
