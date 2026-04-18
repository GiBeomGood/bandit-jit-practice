# Sprint 14 완료 요약

## 완료된 태스크 목록 및 주요 구현 결정 사항

### Task 1: YAML 설정 구조 재편
- `algo` 섹션을 `algorithms` 섹션으로 교체; `oful`과 `lts` 하위 섹션 분리
- 공유 값(`subgaussian_scale`, `delta`)은 `algorithms` 레벨에 한 번만 정의, OmegaConf `${}` 보간으로 참조
- `norm_bound`는 `${experiment.context_bound}`로 보간
- `configs/minimal.yaml`도 동일 구조 적용

### Task 2: 다중 알고리즘 실험 코어 리팩터링
- OFUL 클래스 제거, standalone 함수만 유지
- `_ALGO_REGISTRY` dict 기반 제네릭 디스패치 — 알고리즘 추가 시 코드 구조 변경 불필요
- `OfulCarry` NamedTuple로 pytree 자동 등록
- `jnp.cumsum`은 scan 이후 알고리즘별 1회 적용 (step 함수 내부 호출 없음)
- `run()` 반환값: `Dict[str, np.ndarray]` (알고리즘명 키)

### Task 3: Linear Thompson Sampling (LTS) 구현
- `src/algorithms/lts.py` 신규 작성: `compute_exploration_scale`, `sample_posterior_theta`, `lts_select_action`, `lts_update_design_matrix_inv`, `lts_update`
- B⁻¹ 업데이트에 Sherman-Morrison 공식 적용
- `LtsCarry`에 `prng_key` 필드 포함; 스텝마다 key split으로 독립적 PRNG 스트림 유지
- `_ALGO_REGISTRY`에 등록 완료

### Task 4: 시각화 업데이트
- 제목/레이블 생성 로직을 `main.py`에서 `visualization.py`로 완전 이전
- `build_plot_title`, `_plot_algo_curve` 추가
- `plot_regret`이 regrets dict를 제네릭하게 순회 — 알고리즘 추가/제거 시 코드 변경 불필요

### Task 5: 테스트 업데이트
- `tests/test_reproducibility.py` 전면 재작성
- `cfg.algorithms` 사용, `algo_configs` dict 파라미터 반영
- `ALGO_NAMES = ("oful", "lts")`로 per-algorithm 동등성/비동등성 검증

## 스프린트 중 발생한 설계 변경 또는 주목할 사항

- **init_carry_fn 인터페이스 확장**: OFUL 단계에서는 PRNG 키가 불필요했으나 LTS 추가 시 필요해짐. `(context_dim, algo_cfg)` → `(context_dim, algo_cfg, key)` 로 확장; OFUL은 key를 무시하는 래퍼로 레지스트리 균일성 유지.
- **테스트 리뷰 순서**: Task 2~4 리뷰는 Task 5 미완료로 인한 테스트 실패로 일시 BLOCKED됨. Task 5 완료 후 통합 리뷰로 전환해 모든 태스크를 한 번에 APPROVED 처리.

## 이월된 항목

- 이월 항목 없음. `prompts/sprints/todo/` 에 미완료 파일 없음.
