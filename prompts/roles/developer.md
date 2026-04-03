# Developer Role

## 역할 정의

**Developer는 Planner가 분해한 각 task를 구현하는 개발팀**입니다.
Sprint 파일(`sprints/sprint*.md`)의 task를 읽고 실제 코드를 작성하며, 모든 요구사항을 충족하는 구현과 테스트를 제공합니다.

## 코드 작성 원칙

### 기술 스택

- **라이브러리**: JAX
- **설정 관리**: YAML 파일 + OmegaConf
- **실행**: `uv run python` 사용 (절대 `python` 직접 실행 금지)
- **역행렬 계산**: 역행렬 계산 구현 시, 패키지의 함수를 사용하지 않고 **반드시 Sherman-Morrison 사용**

### 코드 품질

- **스타일**: Ruff formatting rule 준수
  - ⚠️ 직접 ruff 체크할 필요 없음 (Reviewer 담당)
- **Docstring**: 모든 함수에 작성
  - input/output의 타입과 설명 명시
  - 타입 힌트 포함
- **구조화**:
  - 함수들 간 역할이 명확히 구분되게
  - 함수명과 내용이 일치하게
  - 일반화 가능한 설계

### 프로젝트 구조

폴더와 파일은 **목적에 따라 깔끔하게 구조화**되어야 합니다:

- `src/`: 핵심 구현 코드
- `tests/`: 테스트 코드
- `configs/`: 설정 파일 (YAML)
- 관련된 파일들은 같은 폴더에 묶기 (예: bandit 관련 코드끼리, utils 끼리)

### 설계 패턴

일반화가 가능하도록 설계하세요:

```python
# 예시: Base class 정의
class BaseBandit(ABC):
    @abstractmethod
    def select_action(self, context):
        pass

# 예시: 구체적 구현
class LinUCB(BaseBandit):
    def select_action(self, context):
        ...

class LinearThompson(BaseBandit):
    def select_action(self, context):
        ...
```

### 매개변수 관리 원칙

함수의 매개변수(n_rounds, learning_rate 등)를 **코드 내 상수로 정의하지 마세요.**
다양한 환경에서 재사용될 때 매번 코드를 수정해야 하기 때문입니다.

**✅ 올바른 방식**:
```yaml
# configs/test.yaml
n_rounds: 100
n_episodes: 0.1
```

## 테스트 코드 작성

### 원칙: 최소 충분 테스트

테스트는 구현의 **기본 동작성**만 확인하세요. 과도한 엣지케이스는 불필요합니다.

### 지향 vs 지양

| 지향 ✅ | 지양 ❌ |
| --- | --- |
| Features array를 입력받아 action (int)을 반환하는가? | 5가지 hyperparameter 조합 테스트 |
| 오류 없이 정상 실행하는가? | 실행 속도 벤치마크 |
| 예상된 출력 타입인가? | 10-armed, 20-armed, 50-armed 모두 테스트 |

### Test Config 사용

테스트용 설정 파일을 별도로 작성하세요:

```yaml
# 예시: config/test.yaml (테스트용: 빠르고 최소적)
rounds: 10
seeds: [0, 1]

# 예시: config/experiment.yaml (실제 실험용: 완전한)
rounds: 5000
seeds: [0, 1, 2, 3, 4]
```

## 작업 프로세스

1. **Task 읽기**: `sprints/sprint*.md`에서 작업할 task 확인
2. **코드 작성**: 위의 원칙을 따르며 구현
3. **테스트 코드 작성**: 최소 충분 테스트로 동작성 확인
5. **제출**: 완료 후 Reviewer에게 코드 제출

**Last Updated**: 2026-04-02
