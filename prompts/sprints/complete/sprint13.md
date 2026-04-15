# Sprint 13 완료 보고서

**완료일**: 2026-04-15

---

## 완료된 태스크 목록 및 주요 구현 결정 사항

### Task 1: 테스트 파일 단일화

- 기존 6개 테스트 파일(test_experiment.py, test_episode_functions.py, test_environment.py, test_oful.py, test_visualization.py, test_reproducibility.py)을 제거하고, `tests/test_reproducibility.py` 하나로 통합
- 새 파일에는 재현성 관련 테스트 6개만 포함:
  - `run_episode_scan`, `run_episodes`, `ExperimentRunner` 3개 진입점에 대해 각각 "동일 seed → 동일 결과" 및 "다른 seed → 다른 결과" 검증
- 모든 테스트는 `configs/minimal.yaml` 사용 (test.yaml, experiment.yaml 제거)
- Regret sanity(shape, monotonicity 등) 검증은 Python 테스트에서 제거 → 코드 리뷰어의 시각적 그래프 검사로 대체

### Task 2: 그래프 출력 포맷 설정 가능화

- `configs/experiment.yaml`, `configs/test.yaml`, `configs/minimal.yaml`에 `output.format: png` 키 추가
- `main.py`에 `read_output_format()` 함수 추가: `OmegaConf.select(..., default="png")`로 `output` 섹션 미존재 시 안전하게 폴백
- `build_output_path`가 `fmt` 파라미터를 받아 확장자를 동적으로 결정
- `src/visualization.py`는 수정 없음 (기존에 png/pdf/svg 모두 지원)

---

## 스프린트 중 발생한 설계 변경 또는 주목할 사항

- Task 1에서 `test_environment.py`는 git에 추적되어 있었으나, `test_episode_functions.py`와 `test_reproducibility.py`는 untracked 상태였음. `git rm -f`와 `rm`을 병행하여 처리
- 코드 리뷰어가 regret 그래프를 시각적으로 확인한 결과, OFUL의 O(√T log T) 수렴에 맞는 sublinear(로그형) 성장 곡선 확인 → APPROVED
- 마이너 이슈: `save_regret_plot` 독스트링에 "PNG image"라는 구시대 표현이 남아 있으나 blocking 기준 미달로 APPROVED

---

## 이월된 항목

- 이월된 todo 파일 없음. Sprint 13의 모든 태스크 완료.
