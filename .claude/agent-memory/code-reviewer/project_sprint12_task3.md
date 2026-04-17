---
name: Sprint 12 Task 3 — Delete Unused Code (APPROVED)
description: Sprint 12 Task 3 was initially BLOCKED (stubs instead of deletions), then re-reviewed and APPROVED after git rm -f
type: project
---

Sprint 12 Task 3 required physical deletion of unreachable files. The developer initially replaced file contents with single-line stubs — ruled BLOCKED. On re-review, all 6 files were physically deleted via `git rm -f` and APPROVED.

**Deleted files (confirmed gone):**
- `tests/test_benchmark_jit_vmap.py`
- `tests/test_integration.py`
- `scripts/benchmark_jit_vmap.py`
- `scripts/benchmark_jit.py`
- `scripts/benchmark_vmap.py`
- `configs/benchmark.yaml`

**Also completed:** `Visualizer` staticmethod-only class replaced by module-level functions in `src/visualization.py`; `main.py` imports `plot_regret` directly.

**Why:** "Delete unused code" means `git rm` — physical removal. Comment stubs still occupy file paths and could confuse future tooling.

**How to apply:** In future reviews, when a spec says "delete", verify the file does not exist at all — not just that it has no active code.
