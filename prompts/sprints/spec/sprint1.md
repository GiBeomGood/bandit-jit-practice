# Sprint 1: ContextualLinearBandit 환경 및 OFUL 알고리즘 구현

**상태**: ✅ **완료** (2026-04-02)

**목표**: Contextual Linear Bandit 환경과 OFUL 알고리즘을 구현하여, 추후 다른 알고리즘(Linear Thompson 등)을 추가할 수 있는 기반 마련.

**완료 조건**: 작은 파라미터에서 OFUL 알고리즘이 동작하고, 테스트 통과.

---

## Task 1.1: Base Classes 설계 및 ContextualLinearBandit 환경 구현

**설명**:
- `Environment` (ABC): 모든 bandit 환경의 추상 인터페이스
- `ContextualLinearBandit`: Environment를 상속받아 contextual linear bandit 구현
  - Time step t마다 context $x_t \in \mathbb{R}^d$를 제공 (미리 샘플링된 (T, K, d) array에서 인덱싱)
  - Reward: $r_t = \theta^* \cdot x_t + \epsilon_t$ (where $\epsilon_t \sim N(0, 1)$)
  - Best arm: $\arg\max_a (\theta^* \cdot \text{arm}_a)$

**입력/의존성**:
- None (프로젝트 초기 구조)

**출력/결과물**:
- `src/base.py`: `Environment` (ABC) 클래스
  - 추상 메서드: `reset()`, `step(t, action)` → (context, reward, best_arm)
  - 메타데이터: `d`, `K`, `T`
  
- `src/environment.py`: `ContextualLinearBandit(Environment)` 클래스
  - `__init__(d, K, T, L, seed=None)`
  - `reset()`: true_theta와 context array 초기화
  - `step(t, action)` → (context_t, reward_t, best_arm_t)
  - Context는 norm ≤ L (균등하게 [-L, L]^d에서 샘플)
  - True theta: norm = 1로 정규화

- `tests/test_environment.py`: 기본 동작 테스트

**JAX 사용**:
- Context array, theta: JAX array로 관리
- Operations: `jax.numpy` 사용

**평가 기준**:
- ✅ Environment ABC 상속 명확함
- ✅ Context는 미리 샘플링된 (T, K, d) array에서 인덱싱
- ✅ Reward 계산: θ* · x_t + Gaussian noise
- ✅ Best arm 계산: argmax_a (θ* · arm_a)
- ✅ reset() 호출 후 새로운 instance 생성
- ✅ 테스트 통과

**추가 노트**:
- Context의 첫 번째 차원: time (T)
- Context의 두 번째 차원: arms (K)
- Context 배열 형태: (T, K, d)

---

## Task 1.2: Algorithm ABC 및 OFUL 알고리즘 구현

**설명**:
- `Algorithm` (ABC): 모든 bandit 알고리즘의 추상 인터페이스
- `OFUL`: Algorithm을 상속받아 OFUL 알고리즘 구현

**OFUL 알고리즘**:
- Action selection: $\arg\max_a \left[ \hat{\theta}_{t-1}^T x_{t,a} + \text{Radius}(t-1) \cdot \|x_{t,a}\|_{B_{t-1}^{-1}} \right]$
- Design matrix: $B_t = \lambda I + \sum_{s=1}^{t} x_{s,a_s} x_{s,a_s}^T$
- Least-square estimate: $\hat{\theta}_t = B_t^{-1} \sum_{s=1}^{t} r_s x_{s,a_s}$
- Confidence radius: $\text{Radius}(t) = R\sqrt{d \log\left(\frac{1 + tL^2/\lambda}{\delta}\right)} + \lambda^{1/2} S$

**입력/의존성**:
- Task 1.1 완료 (Environment 클래스)

**출력/결과물**:
- `src/base.py` 추가: `Algorithm` (ABC) 클래스
  - 추상 메서드: `reset()`, `select_action(contexts)`, `update(context, reward)`
  
- `src/algorithms/oful.py`: `OFUL(Algorithm)` 클래스
  - `__init__(d, lambda_=1.0, R=1.0, S=1.0, L=1.0, delta=0.01, seed=None)`
  - `reset()`: B_t, theta_hat, collected_data 초기화
  - `select_action(contexts)` → action index
    - Input: contexts of shape (K, d) for all arms at time t
    - Output: selected arm index
  - `update(context, reward)`: B_t, data 업데이트
  
- Helper methods:
  - `_compute_ellipsoid_norm(context)` → $\|x\|_{B_{t-1}^{-1}}$
  - `_estimate_theta()` → $\hat{\theta}_t$
  - `_compute_radius(t)` → confidence radius
  
- `tests/test_oful.py`: 기본 동작 테스트

**JAX 사용**:
- 행렬 연산: `jax.numpy.linalg.solve()` (역행렬 계산)
- B_t 관리: JAX array

**하이퍼파라미터 기본값**:
- `lambda_=1.0`: Ridge regularization
- `R=1.0`: Sub-Gaussian variance proxy
- `S=1.0`: Parameter norm bound estimate
- `L=1.0`: Context norm bound (environment와 일치)
- `delta=0.01`: Failure probability

**평가 기준**:
- ✅ Algorithm ABC 상속 명확함
- ✅ Design matrix B_t 정확히 구현 (수치 안정성: solve() 사용)
- ✅ Ellipsoid norm 계산 정확
- ✅ Action selection이 UCB 원리 따름
- ✅ update() 후 theta_hat 업데이트 확인
- ✅ 테스트 통과

**추가 노트**:
- 초기 action: B_t는 람다I만 있으므로, 부트스트랩 필요 가능성 (확인 후 task 내에서 해결)
- 수치 안정성: B_t^{-1} 계산 시 조건수 모니터링 권장

---

## 프로젝트 구조 (Sprint 1 완료 후)

```
src/
├── __init__.py
├── base.py                 # Environment (ABC), Algorithm (ABC)
├── environment.py          # ContextualLinearBandit(Environment)
└── algorithms/
    ├── __init__.py
    └── oful.py            # OFUL(Algorithm)

tests/
├── __init__.py
├── test_environment.py
└── test_oful.py
```

---

**Last Updated**: 2026-04-02
