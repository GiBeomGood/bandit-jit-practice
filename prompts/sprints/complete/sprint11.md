# Sprint 11 완료 보고서

**완료일**: 2026-04-15
**스프린트 목표**: JAX-First 클린업 — 비-JAX 코드 제거, JIT 함수명 정리, 테스트 단순화

---

## 완료된 태스크 목록 및 주요 구현 결정 사항

### Task 11.1: 비-JAX 코드 경로 제거
- `src/experiment.py`에서 `run_single_episode_sequential`, `_run_episode_loop`, `_compute_step_regret` 함수 및 `ExperimentRunner`의 `use_vmap` 파라미터와 조건 분기 완전 제거
- `scripts/benchmark_jit_vmap.py`에서 `run_no_jit` 함수 제거
- `scripts/benchmark_jit.py` JAX scan 단일 타이밍 벤치마크로 재작성
- `scripts/benchmark_vmap.py`에서 `run_sequential` 함수 제거
- `configs/test.yaml`, `configs/experiment.yaml`, `configs/benchmark.yaml`에서 `use_vmap: false` 키 제거

### Task 11.2: 테스트 스위트 단순화
- `tests/test_integration.py`: 7개 테스트 전체 제거 (모두 `use_vmap=False` 비-JAX 경로 대상)
- `tests/test_vmap.py`: `test_experiment_runner_sequential_unchanged` 제거
- `tests/test_experiment.py`: `test_from_yaml_matches_manual_construction` 제거
- 51개 테스트 보존, JAX 기반 경로만 커버

### Task 11.3: 함수명 정리 — 불필요한 "jit"/"vmap" 접사 제거
- `run_episodes_vmap` → `run_episodes` (`src/experiment.py` 및 모든 호출처)
- `run_vmap` → `run_benchmark` (`scripts/benchmark_vmap.py` 내부 함수)
- `run_jit_vmap` → `run_benchmark` (`scripts/benchmark_jit_vmap.py` 내부 함수)
- 지역 변수 `jit_vmapped` → `compiled_fn`, `vmap_times` → `bench_times`
- 테스트 함수명 `test_benchmark_jit_vmap_runs_without_error` → `test_benchmark_jit_script_runs_without_error`

### Task 11.4: E2E 검증
- 추가 코드 변경 불필요 — Tasks 11.1–11.3 이후 코드베이스가 완전히 일관된 상태
- 51/51 테스트 통과, ruff 위반 제로, 벤치마크 스크립트 정상 실행 (speedup 1.06x 출력)

---

## 스프린트 중 발생한 설계 변경 또는 주목할 사항

1. **`run_experiment.py` 부재**: 스펙 Criterion 4에서 `scripts/run_experiment.py`를 언급했으나 해당 파일이 존재하지 않음. 스펙의 "(or equivalent main entry point)" 조항에 따라 `benchmark_jit_vmap.py`를 동등한 진입점으로 판정하여 진행.

2. **Grep Criterion 해석**: Task 11.3 Criterion 2의 `grep -rn "_vmap\|vmap_"` 패턴이 스크립트 *파일명*(`benchmark_jit_vmap.py`) 문자열 리터럴에도 매칭됨. 파일명 자체는 이번 스프린트 범위 밖이므로, 파일명 참조 문자열은 허용으로 판정.

3. **리뷰 사이클**: Task 11.1은 ruff 위반(F541, E402) 3건으로 1회 BLOCKED 후 재리뷰에서 APPROVED. Task 11.3은 지역변수명 및 테스트 함수명 grep 매칭으로 1회 BLOCKED 후 재리뷰에서 APPROVED.

---

## 이월된 항목

- `prompts/sprints/todo/` 디렉토리에 미완료 항목 없음. 이번 스프린트에서 생성된 todo 파일 없음.
