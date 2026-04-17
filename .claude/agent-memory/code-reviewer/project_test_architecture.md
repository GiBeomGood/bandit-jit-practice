---
name: Test Suite Architecture After Sprint 11
description: Maps test files to the code paths they cover, and records what was deleted in Sprint 11
type: project
---

After Sprint 11 (Tasks 11.1–11.4 complete), the test suite covers only JAX-based paths:

- `tests/test_vmap.py` — `run_episode_scan`, `run_episodes`, `ExperimentRunner.run()` shape
- `tests/test_experiment.py` — `ExperimentRunner` initialization, configs, save/load, from_yaml
- `tests/test_oful.py` — OFUL algorithm, `oful_select_action`, `oful_update`, disable-jit tests
- `tests/test_environment.py` — `ContextualLinearBandit` environment
- `tests/test_visualization.py` — `Visualizer.plot_regret`
- `tests/test_benchmark_jit_vmap.py` — benchmark script smoke test (51 tests total, all passing)

Deleted in Sprint 11:
- `tests/test_integration.py` contents — all 7 tests targeted `use_vmap=False` (non-JAX) path; file now contains only a comment block
- `test_experiment_runner_sequential_unchanged` from `test_vmap.py`
- `test_from_yaml_matches_manual_construction` from `test_experiment.py`

Deleted source functions (Task 11.1): `run_single_episode_sequential`, `_run_episode_loop`, `_compute_step_regret`, `use_vmap` parameter.

Renamed functions (Task 11.3): `run_episodes_vmap` → `run_episodes` (in `src/experiment.py` and all call sites).

Note: Sprint 11 spec criterion 2 references `scripts/run_experiment.py` which does not exist. The actual main entry point is `scripts/benchmark_jit_vmap.py`. Task 11.4 review accepted `benchmark_jit_vmap.py` as the canonical entry point per the task description.

**Why:** Non-JAX fallback code was removed; JAX (jit + vmap) is now the unconditional execution path.
**How to apply:** When reviewing tests, any test targeting a non-JAX loop path should be flagged as a violation of acceptance criteria.
