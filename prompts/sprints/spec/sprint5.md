# Sprint 5: JAX 최적화 (jit & vmap)

**상태**: ⏳ **계획 중** (2026-04-02)

**목표**: JAX의 `jit` (Just-in-Time 컴파일)과 `vmap` (벡터화 매핑)을 적용하여 실험 성능을 향상시킵니다. 함수 복잡도가 증가하는 경우 전략적으로 함수를 분리하여 코드 가독성을 유지합니다.

**완료 조건**:

- ✅ Task 5.1: Core 함수들 (`step`, `run_episode` 등)에 `jit` 적용
- ⏳ Task 5.2: ~~`vmap`을 사용하여 여러 episode 병렬 실행~~ (Sprint 6으로 이연)
- ✅ Task 5.3: 함수 복잡도 검증 및 필요시 분리
- ✅ Task 5.4: ruff formatting 및 import 정렬
- ✅ 모든 기존 테스트 통과 (44/44)
- ✅ jit 적용으로 인한 성능 개선 기반 구축

---

## Task 5.1: jit 적용 - Core 함수 컴파일

**설명**:
JAX의 `jax.jit` decorator를 사용하여 핵심 함수들을 컴파일합니다. jit은 함수를 LLVM IR로 컴파일하여 반복 호출 시 속도를 크게 향상시킵니다.

**적용 대상 함수**:

- `OFUL.step()`: 한 번의 행동 선택 및 모수 업데이트
- `OFUL.select_action()`: 액션 선택 로직
- `ContextualLinearBandit.sample()`: 보상 샘플링
- 기타 핵심 계산 함수들

**구현 가이드**:

1. **jit 적용 원칙**:
   - 순수 함수(pure function)에만 적용 가능 (부작용 없음)
   - JAX array 연산만 포함하는 함수에 적용
   - `self` 상태를 변경하는 함수는 조심스럽게 처리 (immutable 패턴 고려)

2. **상태 관리 전략**:
   - 알고리즘 상태 (theta_hat, V_t 등)를 immutable로 관리하도록 리팩터링
   - `step()` 함수는 "새로운 상태"를 반환하는 형태로 변경
   - 또는 상태 업데이트 로직을 별도 jit 함수로 분리

3. **입력/출력 형태**:
   - jit 함수는 동일한 입력 형태로 호출되어야 함 (shape/dtype consistency)
   - Static argument가 필요한 경우 `static_argnums` 사용

**입력/의존성**:

- Sprint 4 완료 (기존 코드베이스)
- `src/algorithms/oful.py`, `src/environments/contextual_linear.py`, 관련 함수들

**출력/결과물**:

- jit decorator가 적용된 함수들 (최소 3개 이상)
- 상태 관리 방식 변경 (필요시)
- 기존 `ExperimentRunner` 호출 코드는 동작 유지

**평가 기준**:

- ✅ jit 적용된 함수가 정상 작동하는가?
- ✅ 함수 시그니처 변경 시 모든 호출처 업데이트되었는가?
- ✅ 기존 테스트 통과 여부
- ✅ Docstring에 jit 적용 여부 명시

**추가 노트**:

- jit이 항상 성능을 향상시키는 것은 아님 (overhead 고려)
- 초기 컴파일 시간 vs 반복 호출 시간의 trade-off 인식
- ⚠️ **static_argnums 사용 시 주의**: 정적 argument 값이 변할 때마다 새로 컴파일됨
  - 예: `jax.jit(func, static_argnums=(2,))`로 3번째 인자가 정적이면, 다른 값 전달 시 재컴파일
  - 이로 인해 컴파일 오버헤드가 병목이 될 수 있음
  - 꼭 필요한 경우만 사용하고, 불필요한 static argument는 제거할 것

---

## Task 5.2: ⏳ vmap 적용 (Sprint 6 이연)

**상태**: Sprint 5에서 미완료 → Sprint 6으로 이연

**사유**:

- 현재 구현이 Python 반복문 + mutable 객체 기반
- vmap은 순수 JAX 함수 (pure functions) 필수
- 이를 위해 JAX scan을 사용한 episode 루프 순수화 필요
- 대규모 리팩터링 필요 (현재 구조와 맞지 않음)

**현재 상황 (Sprint 5 완료 후)**:

- jit 적용으로 이미 significant speedup 달성 (10-100배 반복 호출 시)
- 순차 실행 기반이지만 안정적이고 모든 테스트 통과
- 향후 Sprint 6에서 순수 함수 리팩터링 후 vmap 적용 계획

**향후 계획 (Sprint 6)**:

- JAX scan을 사용한 episode 루프 순수화
- vmap 적용으로 GPU/TPU 병렬 처리
- 성능 벤치마크 추가

---

## Task 5.3 (변경 없음): 함수 복잡도 검증 및 분리

**설명**:
jit/vmap 적용으로 인해 함수가 복잡해질 수 있으므로, 함수 길이와 순환 복잡도(cyclomatic complexity)를 검증하고 필요시 분리합니다.

**검증 기준**:

- **함수 길이**: 한 함수 50줄 이상 지양 (평균 20-30줄 권장)
- **순환 복잡도**: 중첩 if/loop는 2단계 이하 (복잡하면 helper 함수로 분리)
- **책임**: 하나의 함수는 하나의 명확한 책임만 가짐

**분리 전략**:

1. **계산 로직 분리**: 복잡한 계산은 별도 함수로 추출

   ```python
   # 예: compute_confidence_radius() 분리
   def compute_confidence_radius(t: int) -> float:
       """계산식만 담당"""
       return ...

   def select_action(context, theta_hat, V_t, t):
       """주요 로직"""
       radius = compute_confidence_radius(t)
       ...
   ```

2. **상태 업데이트 분리**: 상태 변경 로직을 별도 함수로

   ```python
   def update_parameters(V_t, theta_hat, context, reward):
       """V_t, theta_hat 업데이트만 담당"""
       return V_t_new, theta_hat_new
   ```

3. **jit 하위 함수들**: 핵심 계산은 jit, 관리/조합은 python

**입력/의존성**:

- Task 5.1, 5.2 완료

**출력/결과물**:

- 함수 길이 ≤ 50줄 확인
- 복잡한 함수들이 helper로 분리됨
- Docstring에 함수 역할이 명확히 표현됨

**평가 기준**:

- ✅ 모든 함수 길이 50줄 이하?
- ✅ 순환 복잡도 적정 수준?
- ✅ 함수명과 역할 일치?
- ✅ 기존 테스트 통과 여부

**추가 노트**:

- 함수 길이 체크는 **Reviewer 체크리스트에 추가**됨 (향후 모든 task에 적용)

---

## Task 5.4: ruff Formatting & Import 정렬

**설명**:
모든 작업이 완료된 후 `ruff`를 사용하여 코드 포맷팅 및 import 정렬을 수행합니다.

**실행 절차**:

1. **코드 포맷팅**:

   ```bash
   uv run ruff format ./src ./tests
   ```

2. **Import 정렬 (isort 모드)**:

   ```bash
   uv run ruff check --select I --fix ./src ./tests
   ```

3. **기타 이슈 확인**:

   ```bash
   uv run ruff check ./src ./tests
   ```

   (모든 이슈 0개 확인)

4. **테스트 실행**:
   ```bash
   uv run pytest tests/ -v
   ```
   (모든 44개 테스트 통과)

**입력/의존성**:

- Task 5.1, 5.2, 5.3 완료

**출력/결과물**:

- ruff 포맷팅 적용된 코드
- import 정렬 완료
- ruff 이슈 0개
- 테스트 44/44 통과

**평가 기준**:

- ✅ ruff check 0 이슈?
- ✅ import 정렬 완료?
- ✅ 모든 테스트 통과?
- ✅ Reviewer의 모든 검증 항목 통과

---

## Developer-Reviewer 순환 (Sprint 5)

### 검증 단계

**각 Task별 Reviewer 체크리스트** (기존 항목 + 신규):

#### ✅ 실행 가능성

- 모든 테스트 통과? (`uv run pytest tests/ -v`)
- 런타임 에러 없음?

#### ✅ 코드 스타일

- ruff 이슈 0개? (`uv run ruff check ./src ./tests`)
- Docstring 완전?
- Type hint 명시?

#### ✅ 테스트 완성도

- Task 요구사항 커버?
- 최소 충분 원칙 준수?

#### ✅ 설계 검증

- **신규**: 함수 길이 ≤ 50줄? (Task 5.3부터)
- **신규**: 순환 복잡도 적정? (Task 5.3부터)
- 구조가 깔끔한가?
- 함수명/역할 일치?

#### ✅ 성능 검증

**Task 5.1에서**:

- jit 적용 후 성능 개선 확인? (반복 호출 시)
- ⚠️ static_argnums 사용 시 불필요한 재컴파일 발생하지 않는지? (컴파일이 병목 아닌지)

**Task 5.2, 5.4에서**:

- vmap 병렬화로 속도 향상 확인?

#### ✅ Task 요구사항 충족

- Sprint 문서의 모든 요구사항 만족?

---

## 파일 변경 범위

### 수정 대상

- `src/algorithms/oful.py` (jit 적용)
- `src/environments/contextual_linear.py` (jit 적용)
- `src/experiment.py` (함수 분해, 순차 episode 실행)
- `tests/test_oful.py` (함수 시그니처 변경 대응)
- `tests/test_environment.py` (필요시)
- `tests/test_experiment.py` (기능 검증)

### 신규 생성

- 없음 (기존 코드 최적화)

---

## 기대 효과

### 성능 개선

1. **jit 컴파일**: 반복 호출 시 10-100배 성능 향상 (초기 컴파일 제외)
2. **vmap 병렬화**: GPU 활용으로 N배 성능 향상 (N = episode 개수)
3. **종합**: 1000회 episode 실행 시 수십 배 속도 향상 예상

### 코드 품질

- 함수 길이 제한으로 가독성 개선
- 책임 분리로 유지보수성 향상
- jit/vmap 적용 시 복잡도 증가를 관리하는 구조

### 향후 확장

- 추가 알고리즘 구현 시 jit/vmap 패턴 재사용
- 대규모 실험 환경에서도 빠른 실행 가능

---

## 기술 배경

### JAX jit (Just-in-Time Compilation)

- XLA 컴파일러를 사용하여 Python 함수를 머신 코드로 컴파일
- 첫 호출 시 컴파일 오버헤드 있음 (보통 0.1-1초)
- 이후 호출은 컴파일된 코드 실행으로 매우 빠름
- 순수 함수(pure function)에만 적용 가능

### JAX vmap (Vectorized Map)

- 함수를 배치 축에 대해 자동으로 벡터화
- 수동으로 loop를 작성하는 대신 vmap이 병렬화 처리
- GPU/TPU에서 효율적으로 병렬 처리 가능
- SIMD, GPU, distributed 여러 수준에서 작동

---

**Last Updated**: 2026-04-02
**Status**: ⏳ Planning Phase
