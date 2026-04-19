# Sprint 16 완료 보고서

**완료일**: 2026-04-19

---

## 완료된 태스크 목록 및 주요 구현 결정 사항

### Task 1: 알고리즘 모듈 클래스 기반 staticmethod 인터페이스 마이그레이션

- `src/algorithms/oful.py`: 모듈 수준 자유 함수 5개(`compute_confidence_radius`, `compute_ucb_values`, `oful_update`, `oful_init_carry`, `make_oful_step_fn`)를 `OFUL` 클래스의 `@staticmethod`로 이동. `make_oful_step_fn` → `make_step_fn`, `oful_init_carry` → `make_init_carry`로 표준 인터페이스에 맞게 이름 변경.
- `src/algorithms/lts.py`: `LinearThompsonSampling` 클래스를 신설하여 `lts_update`, `_lts_step`, `make_lts_step_fn` 등 자유 함수를 staticmethod로 이동. `make_init_carry(prng_key, ...)` 추가.
- `src/algorithms/__init__.py`: `make_lts_step_fn` 내보내기를 `LinearThompsonSampling` 클래스 내보내기로 교체.
- `LtsCarry`는 모듈 수준 `NamedTuple`로 유지 — 자유 함수가 아니므로 `^def` 기준에 위배되지 않음.

### Task 2: experiment.py 통합 클래스 기반 인터페이스 적용

- `src/experiment.py`: `from src.algorithms.oful import OFUL` → `from src.algorithms import OFUL` 로 변경하여 패키지 `__init__`을 통한 임포트 경로 통일.
- Task 1에서 이미 모든 로직이 클래스 staticmethod로 이전된 상태였으므로 Task 2 변경은 단일 임포트 경로 수정으로 완료.

---

## 스프린트 중 발생한 설계 변경 또는 주목할 사항

- `LinearThompsonSampling.make_init_carry`는 `prng_key`를 인자로 받는 점에서 `OFUL.make_init_carry`와 인터페이스가 미묘하게 다름. LTS는 초기화 시점부터 PRNG 키가 필요하기 때문. 향후 `experiment.py`에서 LTS를 실제로 연결할 때 이 차이를 처리해야 함.
- 모든 6개 재현성 테스트 통과, `uv run python main.py` 정상 실행 확인.

---

## 이월된 항목

이번 스프린트에서 생성된 `todo/` 파일 없음. 이월 항목 없음.
