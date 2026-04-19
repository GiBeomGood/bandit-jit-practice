---
name: Sprint 16 Task 1 — Algorithm Class Consolidation
description: Key patterns from migrating free functions to staticmethod classes in Sprint 16 Task 1
type: project
---

Free functions in `src/algorithms/` (excluding `base.py`) were moved into class-based staticmethod interfaces. Each class must expose `make_init_carry` and `make_step_fn` as staticmethods.

**Why:** `experiment.py` will become algorithm-agnostic in Task 2 by calling the unified interface.

**How to apply:** When reviewing future algorithm changes, verify `grep -rn "^def " src/algorithms/` returns nothing outside `base.py`. Both `OFUL` and `LinearThompsonSampling` now use this pattern. `LtsCarry` is a module-level `NamedTuple` — not a free function — so it does not violate the `^def` criterion.

`**kwargs` type annotation gaps in `make_init_carry`, `make_step_fn`, and `run_episode_scan` are pre-existing (carried from Sprint 15) and are not introduced by this sprint's developer. `cls` in `from_yaml` is similarly pre-existing. Do not block on pre-existing gaps that are out of the current task's scope.
