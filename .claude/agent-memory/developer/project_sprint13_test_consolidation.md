---
name: Sprint 13 Test Consolidation
description: tests/ reduced to conftest.py + test_reproducibility.py only; 5 old test files deleted
type: project
---

Sprint 13 Task 1 consolidated all test files into a single `tests/test_reproducibility.py`.

**Why:** Regret sanity checks (shape, sign, monotonicity, boundedness) are now evaluated visually by the code-reviewer inspecting the saved graph image, not via automated tests.

**What was kept:** Two canonical reproducibility concepts, each covered at both the `run_episode_scan`/`run_episodes` level and the `ExperimentRunner` level:
1. Same seed → bit-for-bit identical regrets
2. Different seed → different regrets

**Files deleted via git rm (by orchestrator):**
- `tests/test_experiment.py`
- `tests/test_episode_functions.py`
- `tests/test_environment.py`
- `tests/test_oful.py`
- `tests/test_visualization.py`

All tests use `configs/minimal.yaml` exclusively (never `test.yaml` or `experiment.yaml`).

**How to apply:** Future test additions in this project belong in `test_reproducibility.py` only if they test reproducibility. All other categories (sanity, shape, algorithm correctness) are intentionally excluded.
