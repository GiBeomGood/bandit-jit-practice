---
name: Sprint 12 Task 4 — Rename Files to Remove JIT/vmap/JAX Qualifiers
description: Final task of Sprint 12: rename test_vmap.py to test_episode_functions.py; no file with jit/vmap/jax qualifier survives
type: project
---

Sprint 12 Task 4 is the final task in Sprint 12. The rename scope was:
- `tests/test_vmap.py` → `tests/test_episode_functions.py` (git rm + new file)
- All benchmark scripts (`scripts/benchmark_jit.py`, `scripts/benchmark_jit_vmap.py`, `scripts/benchmark_vmap.py`) were deleted in Task 3
- `tests/test_benchmark_jit_vmap.py` deleted in Task 3
- `configs/benchmark.yaml` deleted in Task 3

After Task 4, the surviving test files are:
- `tests/test_episode_functions.py` (renamed from test_vmap.py)
- `tests/test_experiment.py`
- `tests/test_oful.py`
- `tests/test_environment.py`
- `tests/test_visualization.py`
- `tests/test_reproducibility.py`
- `tests/conftest.py`

Key review check: `jit`/`vmap`/`jax` appearing in Python code bodies is NOT a violation — only filename qualifiers matter. All import paths were clean after the rename.

Task 4 was APPROVED on 2026-04-15 after a re-review. The blocking issue (missing type hints in test_episode_functions.py) was resolved by adding `from typing import Any`, `from omegaconf import DictConfig` imports and annotating all 8 functions. 58/58 tests pass.

**Why:** JIT and vmap are now the universal execution model; having them as filename qualifiers is misleading.
**How to apply:** When reviewing future renames, distinguish between code-level JAX usage (acceptable) and filename qualifiers (must be removed).
