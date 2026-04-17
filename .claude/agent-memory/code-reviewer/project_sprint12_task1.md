---
name: Sprint 12 Task 1 — main.py Entry Point
description: Records design decisions and acceptance criteria for Sprint 12 Task 1 (single main.py entry point)
type: project
---

Sprint 12 Task 1 created `main.py` as the sole experiment entry point. Approved on re-review after `_save_figure` was fixed to use single-format dispatch (extension → format) and raise `ValueError` for unrecognized extensions.

Key design decisions:
- Output directory: `outputs/` (created at runtime)
- Output filename: `{config_stem}_regret.png` (e.g., `experiment_regret.png`)
- Plot title: `OFUL Regret [{config_stem}] (Episodes=N, Horizon=T)`
- Config validation: `FileNotFoundError` for missing path, `ValueError` for non-YAML extension
- `Visualizer._save_figure` dispatches on `.suffix` from a `fmt_map` dict; raises `ValueError` for unknown extensions

**Why:** Sprint spec required config name reflected in both filename and title; earlier version saved PDF+SVG always which conflicted with main.py using `.png`.

**How to apply:** When reviewing future visualization changes, verify `_save_figure` stays single-format and that output naming convention `{stem}_regret.png` is preserved.
