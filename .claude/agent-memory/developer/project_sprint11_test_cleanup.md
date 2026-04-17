---
name: Sprint 11 Test Cleanup Decisions
description: Which tests were removed in Task 11.2 and why, to inform Task 11.4 verification.
type: project
---

Sprint 11, Task 11.2 removed the following non-JAX and redundant tests:

1. `tests/test_integration.py` — emptied entirely. All tests used `ExperimentRunner` with default `use_vmap=False`, routing through `run_single_episode_sequential` (Python for-loop, non-JAX). Every assertion was already covered by `test_experiment.py` and `test_visualization.py`.

2. `tests/test_vmap.py` — removed `test_experiment_runner_sequential_unchanged`. Explicitly tested `use_vmap=False`; JAX-path shape already covered by `test_experiment_runner_vmap_shape`.

3. `tests/test_experiment.py` — removed `test_from_yaml_matches_manual_construction`. Explicitly passed `use_vmap=False` and cross-checked results against YAML runner. YAML runner correctness is already validated by `test_from_yaml_returns_correct_shape` and `test_from_yaml_configs_populated`.

**Why:** `run_single_episode_sequential` is a Python-loop non-JAX path that Task 11.1 will delete. Tests for it would break post-cleanup.

**How to apply:** In Task 11.4 verification, expect these tests to be absent. If `run_single_episode_sequential` was actually kept by Task 11.1, no harm — the tests that remain still pass.
