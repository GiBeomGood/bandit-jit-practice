# Sprint 12 완료 요약

**완료일**: 2026-04-15

---

## 완료된 태스크 목록 및 주요 구현 결정 사항

### Task 1: `main.py` — 단일 실험 진입점 생성
- 프로젝트 루트에 `main.py` 생성. `--config` CLI 인자로 YAML 경로를 받아 실험 실행
- 출력 파일: `outputs/{config_stem}_regret.png` (예: `outputs/experiment_regret.png`)
- 그래프 타이틀에 config 이름 포함 (예: `OFUL Regret [experiment] (Episodes=10, Horizon=1000)`)
- 잘못된 config 경로 시 `FileNotFoundError` / `ValueError`로 명확한 오류 메시지 출력
- `src/visualization.py`의 `_save_figure`를 단일 포맷 저장 방식으로 변경 (확장자 기반, 미인식 확장자는 `ValueError`)
- 기존 테스트가 multi-format 동작을 검증하고 있어 `tests/test_visualization.py`도 함께 업데이트

### Task 2: 재현성 보장 및 테스트
- `src/experiment.py`는 이미 결정론적 시딩 구조를 갖추고 있어 추가 수정 불필요
- `configs/minimal.yaml` 신규 생성: context_dim=2, num_arms=3, num_steps=20, num_episodes=2, seed=7
- `tests/test_reproducibility.py` 신규 생성:
  - 정상 동작 검증: shape, non-negative, monotone, bounded regret
  - 재현성 검증: 동일 config 두 번 실행 → 완전 동일한 결과
  - 다른 seed → 다른 결과

### Task 3: 미사용 코드 삭제
- `scripts/benchmark_jit.py`, `scripts/benchmark_jit_vmap.py`, `scripts/benchmark_vmap.py` 삭제
- `configs/benchmark.yaml` 삭제
- `tests/test_benchmark_jit_vmap.py`, `tests/test_integration.py` 삭제
- `src/visualization.py`의 `Visualizer` 클래스를 모듈 레벨 함수 두 개(`plot_regret`, `_save_figure`)로 교체

### Task 4: 파일명 정리 (최종)
- `tests/test_vmap.py` → `tests/test_episode_functions.py` 로 이름 변경 (`vmap` 한정자 제거)
- 그 외 `src/`, `configs/` 파일들은 이미 한정자가 없는 이름 사용 중

---

## 스프린트 중 발생한 설계 변경 또는 주목할 사항

- **Task 1 블록 후 재시도**: `_save_figure` 변경이 기존 `test_visualization.py` 계약을 위반하여 BLOCKED 발생. 테스트를 새 단일 포맷 동작에 맞게 업데이트하여 해결
- **Task 3 블록 후 수정**: 개발자가 파일을 삭제 대신 주석 stub으로 비워놓아 BLOCKED 발생. 오케스트레이터가 직접 `git rm -f`로 물리적 삭제 처리
- **Task 4 블록 후 수정**: 타입 힌트 누락으로 BLOCKED 발생. `tests/test_episode_functions.py` 전체 함수에 타입 어노테이션 추가
- **Pylance 진단 수정**: Sprint 완료 후 `test_reproducibility.py`의 `test_regrets_shape`와 `test_regrets_bounded`에서 사용하지 않는 `runner` 파라미터 제거

---

## 이월된 항목

없음. `prompts/sprints/todo/` 에 미완료 항목 없음.
