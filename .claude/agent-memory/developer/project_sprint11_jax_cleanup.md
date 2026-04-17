---
name: Sprint 11 JAX-First Cleanup
description: Sprint 11 removed all non-JAX code paths; JAX (jit+vmap) is now the only supported execution mode.
type: project
---

Sprint 11 completes the JAX-first migration. Task 11.1 removed all non-JAX fallback code.

**Removed from `src/experiment.py`:**
- `_compute_step_regret` — helper for Python-loop runner
- `_run_episode_loop` — Python for-loop episode runner
- `run_single_episode_sequential` — non-JAX sequential runner
- `use_vmap` parameter from `ExperimentRunner.__init__` and `from_yaml`
- `if self.use_vmap / else` branch in `ExperimentRunner.run()`

**`ExperimentRunner.run()` now unconditionally uses `jax.jit(jax.vmap(...))`** via `run_episodes`.

**Scripts updated:**
- `benchmark_jit_vmap.py` — removed `run_no_jit`; `run_jit_vmap` renamed to `run_benchmark`
- `benchmark_jit.py` — benchmarks `run_episode_scan` (JAX scan, single episode)
- `benchmark_vmap.py` — `run_vmap` renamed to `run_benchmark`

**Configs:** Removed `use_vmap: false` from all YAML files (test.yaml, experiment.yaml, benchmark.yaml).

**Task 11.3 renames applied:**
- `run_episodes_vmap` → `run_episodes` (in `src/experiment.py`, all tests, both benchmark scripts)
- `run_vmap` → `run_benchmark` (in `scripts/benchmark_vmap.py`)
- `run_jit_vmap` → `run_benchmark` (in `scripts/benchmark_jit_vmap.py`)

**Why:** JAX is the unconditional default; "jit"/"vmap" affixes in public function names imply non-JAX alternatives exist (they don't).

**How to apply:** `run_single_episode_sequential` no longer exists. All experiment runs go through `run_episodes`. Tests that tested the sequential path were removed in Task 11.2.
