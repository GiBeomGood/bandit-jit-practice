# Sprint 10: Production-Scale Benchmark Config, Dual-Format Plots, and JIT+vmap Timing Comparison

**Status**: Completed (2026-04-09)
**Created**: 2026-04-09

---

## Sprint Goal

Three independent improvements to the benchmark pipeline:
(1) Scale `configs/benchmark.yaml` to production parameters (T=5000, 100 episodes, dim=64, arms=50);
(2) Extend `Visualizer.plot_regret` to save plots in both PDF and SVG formats;
(3) Add a new script `scripts/benchmark_jit_vmap.py` that measures **total wall-clock time from script invocation** for JIT+vmap ON vs JIT+vmap OFF and reports the speedup.

---

## Execution Order / Dependency Graph

```
Task 10.1 (scale benchmark.yaml)     ──┐
Task 10.2 (dual-format plots)        ──┤──> Task 10.4 (tests)
Task 10.3 (jit_vmap timing script)   ──┘
```

- **Phase 1** (fully independent, can be done in any order or in parallel): Task 10.1, 10.2, 10.3
- **Phase 2** (depends on Phase 1): Task 10.4

---

## Task List

---

### Task 10.1: Scale `configs/benchmark.yaml` to Production Parameters

**Description**: Update `configs/benchmark.yaml` to use production-scale values. Currently it uses lightweight values for smoke-test purposes. The new values are:
- `experiment.context_dim`: 10 → **64**
- `experiment.num_arms`: 20 → **50**
- `experiment.num_steps`: 500 → **5000**
- `experiment.num_episodes`: 20 → **100**
- `benchmark.num_episodes`: 20 → **100**

`T=5000` means 5000 rounds per episode — i.e., 5000 bandit action selections per episode.

**Inputs / Dependencies**:
- `configs/benchmark.yaml`
- No prerequisite tasks

**Outputs / Deliverables**:
- `configs/benchmark.yaml` modified with the new values above
- All other fields (`algo.*`, `benchmark.num_warmup`, `benchmark.num_trials`, `experiment.context_bound`, `experiment.use_vmap`) remain unchanged

**Acceptance Criteria**:
1. `uv run python -c "from omegaconf import OmegaConf; c = OmegaConf.load('configs/benchmark.yaml'); assert c.experiment.context_dim == 64"` passes.
2. `c.experiment.num_arms == 50`, `c.experiment.num_steps == 5000`, `c.experiment.num_episodes == 100`, `c.benchmark.num_episodes == 100`.

---

### Task 10.2: Dual-Format Plot Saving in `src/visualization.py`

**Description**: `Visualizer.plot_regret` (line 62 of `src/visualization.py`) currently saves only as PDF:
```python
fig.savefig(save_path, format="pdf", dpi=150, bbox_inches="tight")
```
Add a second save immediately after that writes the same figure as SVG. The SVG path is derived by replacing the `.pdf` extension with `.svg` in `save_path`. If `save_path` does not end in `.pdf`, append `_plot.svg` as fallback.

The function signature and docstring do not change — it always saves both formats.

**Inputs / Dependencies**:
- `src/visualization.py`, `Visualizer.plot_regret` starting at line 14
- No prerequisite tasks

**Outputs / Deliverables**:
- `src/visualization.py` modified:
  - After `fig.savefig(save_path, format="pdf", ...)`, add:
    ```python
    svg_path = save_path[:-4] + ".svg" if save_path.endswith(".pdf") else save_path + "_plot.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    ```
  - `plt.close(fig)` remains last

**Acceptance Criteria**:
1. Calling `Visualizer.plot_regret(results, "/tmp/test_plot.pdf")` produces both `/tmp/test_plot.pdf` and `/tmp/test_plot.svg`.
2. Calling with a path not ending in `.pdf` produces `<path>_plot.svg` as the SVG file.
3. All existing tests in `tests/test_visualization.py` (if any) continue to pass.

---

### Task 10.3: New Script `scripts/benchmark_jit_vmap.py` — Wall-Clock Timing Comparison

**Description**: Create `scripts/benchmark_jit_vmap.py` that measures the **total wall-clock time from Python script invocation** for two modes:
- **JIT+vmap ON**: calls `run_episodes_vmap(seeds, ...)` — this is `jax.jit(jax.vmap(run_episode_scan))`.
- **JIT+vmap OFF**: calls `run_single_episode_sequential` in a Python loop inside `with jax.disable_jit():` — JIT and vmap are both disabled together as a set.

The script records `_SCRIPT_START = time.perf_counter()` **at module level** (top of file, before any function definitions or imports below it) so that "time from invocation" includes Python startup, JAX import, and config loading. For each mode, it reports:
- Computation time (mode-specific stopwatch: `t_end - t_mode_start`)
- Total time from script invocation to mode completion: `t_end - _SCRIPT_START`

After both modes complete, it prints the speedup ratio: `no_jit_compute_time / jit_vmap_compute_time`.

**Key design notes**:
- JIT and vmap are treated as a single set — both on or both off. There is no "JIT only" or "vmap only" mode.
- Use `jax.disable_jit()` context manager (not `jax.config.update("jax_disable_jit", True)`) so that both modes run in the same process without restart.
- No warmup before the JIT+vmap ON run — the first run includes JIT compilation overhead, which is part of the "real" wall-clock cost.
- Call `.block_until_ready()` on JAX output arrays before stopping the timer.
- Accepts `--config` CLI argument, defaulting to `configs/benchmark.yaml`.
- Reads `benchmark.num_episodes`, `experiment.*`, and `algo.*` from config via `OmegaConf`.

**Script structure** (outline — developer fills in function bodies):

```python
"""Benchmark: wall-clock time for JIT+vmap ON vs OFF.

Usage:
    uv run python scripts/benchmark_jit_vmap.py
    uv run python scripts/benchmark_jit_vmap.py --config configs/benchmark.yaml
"""

import time
_SCRIPT_START = time.perf_counter()  # Must be first executable line

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf


def run_jit_vmap(num_episodes, episode_kwargs) -> float:
    """Run all episodes with JIT+vmap. Returns compute time in seconds."""
    from src.experiment import run_episodes_vmap
    seeds = jnp.arange(num_episodes, dtype=jnp.int32)
    t0 = time.perf_counter()
    result = run_episodes_vmap(seeds=seeds, **episode_kwargs)
    result.block_until_ready()
    return time.perf_counter() - t0


def run_no_jit(num_episodes, episode_kwargs) -> float:
    """Run all episodes sequentially with JIT disabled. Returns compute time in seconds."""
    from src.experiment import run_single_episode_sequential
    t0 = time.perf_counter()
    with jax.disable_jit():
        for i in range(num_episodes):
            result = run_single_episode_sequential(seed=i, **episode_kwargs)
        result.block_until_ready()
    return time.perf_counter() - t0


def main() -> None:
    # ... parse args, load config, build episode_kwargs ...
    # Print config summary
    # Run JIT+vmap mode, record t_jit and t_total_jit = perf_counter() - _SCRIPT_START
    # Run no-JIT mode, record t_nojit
    # Print results table: compute time, total-from-invocation time, speedup ratio
    pass


if __name__ == "__main__":
    main()
```

**Inputs / Dependencies**:
- `src/experiment.py` — `run_episodes_vmap`, `run_single_episode_sequential`, `ExperimentRunner.from_yaml`
- `configs/benchmark.yaml` (default config)
- No prerequisite tasks (Tasks 10.1 can be done in parallel — script works with any config values)

**Outputs / Deliverables**:
- New file: `scripts/benchmark_jit_vmap.py`
- Sample output format (approximate):
  ```
  ============================================================
  Benchmark: JIT+vmap ON vs OFF — wall-clock from invocation
  ============================================================
  Config: context_dim=64, num_arms=50, num_steps=5000, num_episodes=100
  ------------------------------------------------------------
  [JIT+vmap ON]
    Compute time:          12.34s
    Time from invocation:  15.67s  (includes Python startup + JAX import)

  [JIT+vmap OFF (jax.disable_jit)]
    Compute time:         234.56s
    Time from invocation: 237.89s

  Speedup (no-JIT / JIT+vmap): 19.01x
  ============================================================
  ```

**Acceptance Criteria**:
1. `uv run python scripts/benchmark_jit_vmap.py --config configs/test.yaml` completes without error and prints a speedup ratio.
2. `_SCRIPT_START = time.perf_counter()` is the first executable line in the file (before any `import` statements below it).
3. The printed "Time from invocation" for JIT+vmap ON is greater than the "Compute time" (startup overhead is captured).
4. No warmup runs — both modes are first-run measurements.
5. Zero ruff violations.

---

### Task 10.4: Tests

**Description**: Add tests to verify Task 10.2 (dual-format output) and Task 10.3 (script runs without error on `configs/test.yaml`).

**Inputs / Dependencies**:
- Tasks 10.1, 10.2, 10.3 complete
- `configs/test.yaml` (fast config: 2 episodes, T=10, used for smoke tests)
- `tests/` directory

**Outputs / Deliverables**:
- `tests/test_visualization.py` — add or extend with:
  - Test that `Visualizer.plot_regret(results, "/tmp/test_plot.pdf")` creates both `/tmp/test_plot.pdf` and `/tmp/test_plot.svg`
  - Construct minimal `results` dict: `{"regrets": np.zeros((2, 10)), "configs": {"num_steps": 10}}`
- `tests/test_benchmark_jit_vmap.py` — smoke test:
  - Use `subprocess.run(["uv", "run", "python", "scripts/benchmark_jit_vmap.py", "--config", "configs/test.yaml"], ...)` 
  - Assert return code is 0
  - Assert "Speedup" appears in stdout

**Acceptance Criteria**:
1. `uv run python -m pytest tests/test_visualization.py -v` passes with both PDF and SVG assertions.
2. `uv run python -m pytest tests/test_benchmark_jit_vmap.py -v` passes (subprocess exits 0, "Speedup" in output).
3. `uv run python -m pytest tests/ -v` passes with zero failures.
4. Zero ruff violations.

---

## Additional Notes

### Why `_SCRIPT_START` at module level?

Putting `_SCRIPT_START = time.perf_counter()` before any imports captures the Python startup cost and JAX initialization in the "time from invocation" metric. This reflects what a user actually experiences when typing `python benchmark_jit_vmap.py` in the terminal. The compute-time metric (mode-specific stopwatch) isolates just the algorithm execution.

### Why no warmup in Task 10.3?

The goal is to measure realistic wall-clock cost including JIT compilation overhead. Warmup-then-benchmark isolates the "steady-state" JIT cost (useful for micro-benchmarks), but here we want the total cost a user pays when running the script cold. JIT compilation happens once at first call — that cost is real and should be included.

### What is NOT in scope for Sprint 10

- Modifying `configs/experiment.yaml` or `configs/test.yaml` parameter values
- Adding new bandit algorithms
- Changing `ExperimentRunner` or any core algorithm logic
- Adding warmup or multi-trial averaging to `benchmark_jit_vmap.py`

**Last Updated**: 2026-04-09
