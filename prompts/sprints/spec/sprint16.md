# Sprint 16: Algorithm Class Consolidation

**Status**: Completed (2026-04-19)

## Sprint Goal

Eliminate module-level free functions from all algorithm modules. Each algorithm class
exposes a unified staticmethod interface (`make_init_carry`, `make_step_fn`, `update`),
making `experiment.py` algorithm-agnostic.

## Execution Order

Task 1 (algorithm modules refactor) ──▶ Task 2 (experiment.py update)

---

### Task 1: Migrate algorithm modules to class-based staticmethod interface

For every algorithm in `src/algorithms/` (excluding `base.py`), move all scan-related
free functions into `@staticmethod`s on the algorithm class. Each class must expose
`make_init_carry` and `make_step_fn` as the standard interface entry points.

**Acceptance Criteria**:
- `grep -rn "^def " src/algorithms/` returns no results outside `base.py`
- All algorithm classes expose `make_init_carry(...)` and `make_step_fn(...)` as staticmethods
- `uv run python -m pytest tests/` passes

---

### Task 2: Update experiment.py to use unified class-based interface

Replace all algorithm-specific free function imports with class imports. Logic that
calls `make_init_carry` and `make_step_fn` must go through the class, not free functions.

**Acceptance Criteria**:
- `grep "from src.algorithms" src/experiment.py` shows only class imports
- No algorithm name appears as a function prefix (e.g. `oful_`, `lts_`) in `experiment.py`
- `uv run python main.py` runs without error
- `uv run python -m pytest tests/` passes
