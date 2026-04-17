---
name: Sprint 12 Task 3 Cleanup
description: What was removed/converted in Sprint 12 Task 3; Visualizer class replaced by module-level functions
type: project
---

Removed in Sprint 12 Task 3 (not reachable from main.py or test_reproducibility.py):

**Emptied (stub only):**
- `scripts/benchmark_jit.py`
- `scripts/benchmark_jit_vmap.py`
- `scripts/benchmark_vmap.py`
- `configs/benchmark.yaml`
- `tests/test_benchmark_jit_vmap.py`
- `tests/test_integration.py`

**Converted:**
- `src/visualization.py`: `Visualizer` staticmethod-only class → module-level functions `plot_regret` and `_save_figure`
- `main.py`: updated import from `Visualizer` to `plot_regret`
- `tests/test_visualization.py`: updated all `Visualizer.plot_regret(...)` calls to `plot_regret(...)`

**Why:** Sprint 12 goal is to consolidate around `main.py` as the single entry point, removing all code not reachable from it.

**How to apply:** If adding new visualization or experiment entry points, add module-level functions to `src/visualization.py` — no class wrapper needed.
