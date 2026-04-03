# Sprint 2: 실험 런너 및 Regret 측정

**상태**: ✅ **완료** (2026-04-02)

**목표**: 여러 episode에 걸쳐 OFUL 알고리즘을 실행하고, cumulative regret을 측정하여 numpy 배열로 저장.

**완료 조건**: 작은 파라미터(e.g., num_episodes=3, T=20)에서 실험이 동작하고 결과 저장 확인.

---

## Task 2.1: Episode Runner 및 결과 저장

**설명**:
- `run_experiment()`: 여러 episode에 걸쳐 OFUL을 실행
  - 각 episode마다 Environment와 Algorithm을 리셋
  - 각 time step t에서 action 선택 → reward 받음 → 알고리즘 업데이트
  - 각 episode의 cumulative regret 계산 및 저장
  
- `ExperimentRunner` 클래스: 실험 설정 및 실행 관리
  - 결과를 numpy 배열로 반환 (shape: (num_episodes, T))
  - 메타데이터 저장 (설정, seeds 등)
  
- Regret 계산:
  - Cumulative regret at time T: $R(T) = \sum_{t=1}^{T} (\theta^* \cdot x_{t,a^*} - r_t)$
  - $a^*$ = best arm at time t (from environment)
  - $r_t$ = actual reward from selected action

**입력/의존성**:
- Sprint 1 완료 (ContextualLinearBandit, OFUL)

**출력/결과물**:
- `src/experiment.py`: `ExperimentRunner` 클래스
  - `__init__(num_episodes, env_params, algo_params, seed=None)`
  - `run()` → Dict
    - `regrets`: numpy array, shape (num_episodes, T)
    - `configs`: dict with all hyperparameters
    - `metadata`: dict with seeds, etc.
  - `save_results(path)`: numpy savez 형식으로 저장
  - `load_results(path)`: 결과 로드 (static method)

- `tests/test_experiment.py`: 기본 동작 테스트
  - 작은 파라미터로 실험 실행
  - 결과 형태 확인 (regret이 단조증가 또는 합리적 패턴)

**JAX 사용**:
- 결과는 numpy로 변환 후 저장 (`jax.numpy` → `numpy`)
- Episode 병렬 실행: `jax.vmap` 활용하여 여러 episode을 벡터화된 연산으로 실행
  - 각 episode은 서로 다른 random seed로 독립 실행
  - vmap을 통해 GPU/TPU 가속화 활용

**테스트 파라미터**:
- `num_episodes=3, d=5, K=10, T=20`

**평가 기준**:
- ✅ Episode 반복 동작 (Environment, Algorithm 매번 리셋)
- ✅ Regret 계산 정확 (best arm과의 차이)
- ✅ Numpy 배열 저장/로드 동작
- ✅ 결과 형태: (num_episodes, T)
- ✅ vmap 병렬 실행 (선택적이나 권장)
- ✅ JAX 컴파일 오버헤드 최소화 (불필요한 재컴파일 없음)
- ✅ 테스트 통과

**추가 노트**:
- Regret이 음수가 될 가능성 (early episodes에서 exploration으로 인해) → 절댓값 아님, cumulative이므로 정상
- 메타데이터에 전체 설정 저장하면 재현성 확보
- JAX 컴파일 오버헤드 체크리스트:
  - 각 episode마다 jit 컴파일이 재발생하지 않는지 확인
  - vmap 사용 시 compiled function의 재사용으로 오버헤드 최소화
  - 첫 실행 vs 이후 실행의 시간 차이 로깅 권장

---

## 프로젝트 구조 (Sprint 2 완료 후)

```
src/
├── __init__.py
├── base.py
├── environment.py
├── experiment.py           # ExperimentRunner
└── algorithms/
    ├── __init__.py
    └── oful.py

tests/
├── __init__.py
├── test_environment.py
├── test_oful.py
└── test_experiment.py

results/                    # 실험 결과 저장 (runtime에 생성)
└── *.npz
```

---

**Last Updated**: 2026-04-02
