---
name: Sprint 16 Task 2 — experiment.py import update
description: Single-line import fix: OFUL now imported from package __init__ rather than submodule
type: project
---

Task 2 of Sprint 16 was a one-line change: `from src.algorithms.oful import OFUL` → `from src.algorithms import OFUL`. All algorithm logic was already in class staticmethods after Task 1; no free function calls or oful_/lts_ prefixes existed in experiment.py to remove.

**Why:** The sprint goal was to make experiment.py algorithm-agnostic by going through the package __init__ rather than reaching into submodules directly.

**How to apply:** Future tasks adding algorithms should export them via `src/algorithms/__init__.py` and import from there in experiment.py, never directly from submodule paths.
