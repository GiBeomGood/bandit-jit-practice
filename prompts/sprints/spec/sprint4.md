# Sprint 4: 코드 리뷰 및 구조 개선

**상태**: ⏳ **계획 중** (2026-04-02)

**목표**: 기존 코드베이스를 체계적으로 리뷰하고, 구조를 개선하며, 코드 품질을 향상시킴. Developer-Reviewer 순환 프로세스 도입.

**완료 조건**: 
- ✅ Task 4.1: 코드 리뷰 완료 및 구조적 문제점 파악
- ✅ Task 4.2: 파일 구조 개선 (base classes 적절한 위치로 이동)
- ✅ Task 4.3: ruff 모든 이슈 해결
- ✅ 모든 기존 테스트 통과 (44/44)

---

## Task 4.1: 체계적 코드 리뷰

**설명**:
코드베이스 전체에 대한 초기 리뷰를 수행하여 다음을 확인:
- 구조적 문제점 (import path, base class 위치)
- 네이밍 컨벤션 위반 (대문자, 특수문자 등)
- 미사용 임포트, 중복 코드
- API 일관성
- 테스트 커버리지 갭

**리뷰 체크리스트**:
```
□ src/base.py
  - Algorithm ABC: algorithms/ 서브패키지로 이동해야 하는가?
  - Environment ABC: environments/ 서브패키지 생성 후 이동해야 하는가?
  - 향후 확장성 고려

□ src/environment.py
  - ContextualLinearBandit 구현 검토
  - Private 속성명 (_contexts, _true_theta) 일관성 확인
  - 매직 넘버 또는 설정값

□ src/algorithms/oful.py
  - OFUL 구현 검토
  - 변수명 (theta_hat, ucb_values, radius_t 등) 일관성

□ src/experiment.py
  - ExperimentRunner 설계 검토
  - 멀티에피소드 로직 및 regret 계산

□ src/visualization.py
  - Visualizer 정적 메서드 설계
  - matplotlib 사용 패턴

□ 테스트 코드
  - test_integration.py: 미사용 임포트 (Path, ContextualLinearBandit, OFUL)
  - 중복 테스트 케이스
  - 커버리지 갭

□ 전체 패키지
  - __init__.py 파일 구조 및 export
  - 상대/절대 import 일관성
```

---

## Task 4.2: 파일 구조 개선

**설명**:
Base classes를 적절한 서브패키지로 이동하여 구조 개선:

**현재 구조**:
```
src/
├── base.py                  # ← Algorithm과 Environment 혼재
├── environment.py
├── algorithms/
│   ├── __init__.py
│   └── oful.py
```

**목표 구조**:
```
src/
├── __init__.py
├── environments/            # ← 새로 생성
│   ├── __init__.py
│   ├── base.py             # Environment ABC
│   └── contextual_linear.py # ContextualLinearBandit
├── algorithms/              # ← 기존
│   ├── __init__.py
│   ├── base.py             # Algorithm ABC (src/base.py에서 이동)
│   └── oful.py
├── experiment.py
└── visualization.py
```

**마이그레이션 단계**:

1. **environments/ 패키지 생성**
   - `src/environments/__init__.py` 생성
   - `src/base.py`에서 `Environment` ABC 추출 → `src/environments/base.py`
   - `src/environment.py` → `src/environments/contextual_linear.py` 리네임

2. **algorithms/ 패키지 개선**
   - `src/base.py`에서 `Algorithm` ABC 추출 → `src/algorithms/base.py`
   - `src/algorithms/__init__.py` 업데이트

3. **Import 경로 업데이트**
   - `src/experiment.py`: `from src.base import ...` → `from src.environments.base import ...` 및 `from src.algorithms.base import ...`
   - `src/experiment.py`: `from src.environment import ...` → `from src.environments.contextual_linear import ...`
   - `src/algorithms/oful.py`: `from src.base import Algorithm` → `from src.algorithms.base import Algorithm`
   - 테스트 파일들의 import 경로 업데이트

4. **src/base.py 삭제** (더 이상 필요 없음)

**장점**:
- 명확한 구조: 각 서브패키지가 독립적으로 확장 가능
- Future-proof: 향후 `LinearThompson`, `UCB-GLM` 등 알고리즘 추가 용이
- 향후 새로운 환경(예: `BanditWithDelays`, `NonStationaryBandit`) 추가 가능
- IDE 자동완성 및 코드 탐색 개선

**입력/의존성**:
- Task 4.1 완료 (리뷰 결과)

**검증**:
- 모든 import 경로 정확성 확인
- `uv run pytest tests/ -v` 실행: 44개 테스트 모두 통과 확인

---

## Task 4.3: ruff를 통한 코드 품질 검사 및 개선

**설명**:
ruff 도구를 사용하여 코드 품질 문제 확인 및 수정:

**ruff 체크 항목**:
```bash
uv run ruff check ./src ./tests
```

**알려진 이슈**:
- `tests/test_integration.py`: 미사용 임포트
  - `from pathlib import Path` (불필요)
  - `from src.environment import ContextualLinearBandit` (불필요)
  - `from src.algorithms.oful import OFUL` (불필요)

**작업 순서**:
1. `uv run ruff check ./src ./tests` 실행하여 전체 이슈 파악
2. Auto-fix 가능한 이슈: `uv run ruff check ./src ./tests --fix`
3. 수동 수정 필요한 이슈:
   - 네이밍 컨벤션 (대문자 변수명 등)
   - 복잡한 논리 구조
   - 주석 부족

**검증**:
- `uv run ruff check ./src ./tests` 실행하여 0개 이슈 확인
- `uv run pytest tests/ -v` 실행: 44개 테스트 모두 통과 확인

**입력/의존성**:
- Task 4.2 완료 (파일 구조 정렬 후)

---

## Developer-Reviewer 순환 (전체 Sprint에 적용)

**개발 프로세스**:
- 각 Task (4.1~4.3)은 다음과 같이 진행:
  1. **Developer**: 구현 및 검증 (테스트 통과)
  2. **Reviewer**: 구조/품질/일관성 검토
  3. **Developer**: Feedback 반영 (필요 시)
  4. **Finalize**: 완료 확인

**검토 포인트**:
- Task 4.2: 파일 구조 변경사항 및 import 경로 일관성
- Task 4.3: ruff 이슈 수정의 완전성
- 전체: 44개 테스트 통과 여부

---

## 파일 변경 범위

### 신규 생성
- `src/environments/__init__.py`
- `src/environments/base.py` (Environment ABC 이동)
- `src/environments/contextual_linear.py` (기존 environment.py)
- `src/algorithms/base.py` (Algorithm ABC 이동)

### 삭제
- `src/base.py`
- `src/environment.py`

### 수정 (import 경로 업데이트)
- `src/algorithms/oful.py`
- `src/experiment.py`
- `src/visualization.py` (필요시)
- `tests/test_environment.py` → `tests/test_environments/test_contextual_linear.py` (선택사항)
- `tests/test_oful.py` (import 수정)
- `tests/test_integration.py` (import 수정)
- `tests/test_experiment.py` (import 수정)
- `src/environments/__init__.py` (export 추가)
- `src/algorithms/__init__.py` (export 수정)

---

## 기대 효과

1. **구조적 개선**
   - 명확한 패키지 구조로 확장성 개선
   - Base class 분리로 인한 모듈화 강화

2. **코드 품질**
   - ruff 0 이슈 달성
   - 일관된 네이밍 컨벤션
   - 미사용 코드 제거

3. **개발 프로세스**
   - Developer-Reviewer 순환으로 인한 코드 품질 보증
   - 문서화를 통한 투명성

4. **향후 유지보수**
   - 새로운 환경/알고리즘 추가 용이
   - 코드 리뷰 기록 남김
   - 변경 이력 명확화

---

## 다음 단계 (Sprint 5 예상)

1. **추가 알고리즘 구현**
   - `src/algorithms/linear_thompson.py`
   - `src/algorithms/ucb_glm.py`

2. **추가 환경 구현**
   - `src/environments/nonstationary.py`
   - `src/environments/batched.py`

3. **성능 최적화**
   - vmap 병렬화
   - GPU 가속화

---

**Last Updated**: 2026-04-02
**Status**: ⏳ Planning Phase
