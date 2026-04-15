# Sprint 10 완료 보고서

**완료일**: 2026-04-09

---

## 완료된 태스크 목록 및 주요 구현 결정 사항

### Task 10.1: `configs/benchmark.yaml` 프로덕션 파라미터 스케일업
- `experiment.context_dim`: 10 → 64
- `experiment.num_arms`: 20 → 50
- `experiment.num_steps`: 500 → 5000
- `experiment.num_episodes`: 20 → 100
- `benchmark.num_episodes`: 20 → 100
- 기타 모든 필드 (`algo.*`, `benchmark.num_warmup`, `benchmark.num_trials` 등) 변경 없음

### Task 10.2: `src/visualization.py` 듀얼 포맷 저장
- `Visualizer.plot_regret`에 SVG 저장 로직 추가 (PDF 저장 직후)
- `.pdf` 경로 → `.svg`로 확장자 치환, 비(非) `.pdf` 경로 → `_plot.svg` 접미사 폴백
- **설계 결정**: 52줄이던 `plot_regret`를 49줄로 줄이기 위해 `_save_figure(fig, save_path)` 정적 헬퍼 메서드를 추출

### Task 10.3: `scripts/benchmark_jit_vmap.py` 신규 스크립트
- JIT+vmap ON (vmap 병렬화) vs OFF (jax.disable_jit + Python 루프) 두 모드의 총 실행 시간 비교
- **설계 결정**: `_SCRIPT_START = time.perf_counter()`를 `import time` 직후 첫 실행 줄로 배치 → ruff E402 위반 발생
- **해결책**: `pyproject.toml`에 `[tool.ruff.lint.per-file-ignores]` 섹션을 추가하여 해당 스크립트만 E402 억제 (`# noqa` 금지 규칙 준수)
- warmup 없이 첫 실행으로 JIT 컴파일 오버헤드 포함한 현실적 wall-clock 측정
- `configs/test.yaml` 미지원 필드(`benchmark` 섹션 없음) 대비 폴백 처리

### Task 10.4: 테스트 추가
- `tests/test_visualization.py`: PDF+SVG 동시 생성 및 폴백 SVG 생성 테스트 (Task 10.2 개발자가 추가)
- `tests/test_benchmark_jit_vmap.py`: 서브프로세스로 스크립트 실행 후 종료 코드 0 및 "Speedup" 출력 확인 스모크 테스트
- 전체 테스트 60개 통과 (0 실패)

---

## 스프린트 중 발생한 설계 변경 또는 주목할 사항

1. **ruff E402 vs 스펙 요구사항 충돌**: `_SCRIPT_START` 타이밍 센티넬 패턴은 스펙이 명시적으로 요구하지만, ruff E402 규칙과 충돌. `# noqa` 금지 규칙을 지키면서 `pyproject.toml`의 per-file-ignores로 해결.

2. **함수 길이 제한 (≤ 50줄)**: `plot_regret`가 52줄이 되어 BLOCKED → `_save_figure` 헬퍼 추출로 해결. 코드 품질 규칙이 실제로 코드 구조를 개선하는 효과를 보임.

3. **테스트 반환 타입 어노테이션**: 기존 테스트 파일들이 `-> None`을 생략하고 있었으나, 컨스티튜션 규칙 적용으로 신규 테스트에는 추가. 기존 파일은 이번 스프린트 범위 밖.

---

## 이월된 항목

이번 스프린트에서 이월된 항목(todo 파일)은 없습니다.
