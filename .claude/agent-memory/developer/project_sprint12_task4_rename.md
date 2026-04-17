---
name: Sprint 12 Task 4 File Rename
description: test_vmap.py renamed to test_episode_functions.py in Sprint 12 Task 4
type: project
---

Sprint 12 Task 4 renamed the only file with a redundant qualifier:

- `tests/test_vmap.py` → `tests/test_episode_functions.py`

The file tests `run_episode_scan` and `run_episodes` (the episode-level scan/vmap functions). Since vmap is now the standard execution model, the `vmap` qualifier was redundant. The new name describes what is tested rather than how it is executed.

The old file `tests/test_vmap.py` was left as a one-line stub (redirect comment) because shell/git tools were unavailable for a true `git mv`. All test functions live in `test_episode_functions.py`.

**Why:** Sprint 12 goal — drop jit/vmap/jax qualifiers from filenames now that JIT+vmap is the universal execution model.

**How to apply:** When checking test coverage, look in `test_episode_functions.py` for scan/vmap function tests. Ignore the stub `test_vmap.py`.
