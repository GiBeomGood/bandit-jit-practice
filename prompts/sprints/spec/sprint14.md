# Sprint 14

**Status**: Completed (2026-04-17)

## Sprint Goal

Refactor the experiment infrastructure to support multiple algorithms running in the same episode, implement Linear Thompson Sampling (LTS) alongside OFUL, and address config/code hygiene: kwargs via config dicts, isolated algorithm configs with YAML interpolation, visualization code out of `main.py`, deletion of the OFUL class, pytree-registered dataclasses for per-algorithm JAX carry state, and a single `cumsum` over the full regret array.

---

## Execution Order

```
Task 1 (Config restructure)
    │
    ▼
Task 2 (Multi-algorithm experiment core)
    │
    ├──▶ Task 3 (LTS algorithm)
    │
    └──▶ Task 4 (Visualization update)

Task 5 (Tests update) ◄── depends on Tasks 2, 3, 4
```

Sequential chain: Task 1 → Task 2 → {Task 3, Task 4} → Task 5.

---

## Tasks

### Task 1: Restructure YAML configs for multi-algorithm support and YAML interpolation

**Description**: Restructure the experiment configs so that:

1. The `algo` section is replaced by an `algorithms` section containing one sub-section per algorithm. Each sub-section contains only that algorithm's parameters.
2. Where a value needed in a lower-level section already appears in a higher section, reference it via OmegaConf YAML interpolation (`${...}`) rather than repeating it.
3. The `experiment` section must not contain algorithm parameters.
4. Keep the existing `output.format` key.
5. `configs/minimal.yaml` mirrors the same structure with small values for fast test runs.

**Acceptance Criteria**:

1. `OmegaConf.load("configs/experiment.yaml")` resolves without error.
2. No algorithm parameter value is duplicated as a literal — repeated values use `${}` interpolation.
3. Neither algorithm sub-section references keys from the other algorithm's sub-section.

---

### Task 2: Refactor experiment core for multi-algorithm execution

**Description**: Refactor the experiment to support an arbitrary list of algorithms in the same episode.

1. **Dataclass carry**: Each algorithm's scan state is a pytree-registered dataclass. The scan `carry` is a dict keyed by algorithm name containing these per-algorithm state objects, plus shared environment state.
2. **Shared environment**: Each algorithm independently selects an action from the same contexts. There is no interaction between algorithms.
3. **Array-level cumsum**: The scan step emits instantaneous regrets only. Apply a single `jnp.cumsum` after scan returns the full array — not inside the step function.
4. **Generic dispatch**: Adding a third algorithm later must be a data change only, not a structural change.
5. **kwargs via config**: Algorithm configs are passed as structured dicts from the YAML `algorithms` section.
6. **Remove the OFUL class**: Retain all standalone functions; remove only the class wrapper.
7. **Results shape**: `run()` returns a regrets dict keyed by algorithm name, e.g., `{"oful": array(num_episodes, num_steps), "lts": array(...)}`.

**Acceptance Criteria**:

1. `uv run python main.py --config configs/experiment.yaml` runs to completion.
2. Per-algorithm carry state uses a pytree-registered dataclass, not a raw tuple.
3. `jnp.cumsum` is called exactly once per algorithm per episode after scan, never inside the step function.
4. Results regrets is a dict keyed by algorithm name.
5. The OFUL class is gone; only standalone functions remain.

---

### Task 3: Implement Linear Thompson Sampling (LTS) algorithm

**Description**: Implement LTS following the detail document
(`prompts/sprints/details/task_14_lts.md`). Integrate LTS into the multi-algorithm loop alongside OFUL.

**Acceptance Criteria**:

1. Running `main.py` produces regrets for both `"oful"` and `"lts"` with shape `(num_episodes, num_steps)`.
2. The scan loop structure does not hardcode the number of algorithms.

---

### Task 4: Update visualization to plot multiple algorithms on one graph

**Description**: Move all visualization code and plot title generation out of `main.py` and into the visualization layer. The visualization function accepts the multi-algorithm results dict and draws one labelled curve per algorithm on the same axes, with a shaded confidence band. The default title is derived from experiment parameters inside the visualization layer.

**Acceptance Criteria**:

1. Running `main.py` produces a graph with one labelled curve per algorithm on the same axes.
2. `main.py` contains no plot title construction or algorithm label formatting.
3. Adding or removing an algorithm from the config changes the graph without code changes.

---

### Task 5: Update tests for multi-algorithm results structure

**Description**: Update the reproducibility tests to reflect the new results structure.
Regrets are now a dict keyed by algorithm name; tests must assert per-algorithm.
Extend the same reproducibility tests to cover LTS. No new test categories.

**Acceptance Criteria**:

1. `uv run python -m pytest tests/` passes with zero failures.
2. Same-seed tests assert equality for both OFUL and LTS regret arrays.
3. Different-seed tests assert inequality for at least one algorithm.
4. No test references the old config structure.

---

## Detail Documents

`prompts/sprints/details/task_14_lts.md` — LTS posterior sampling math, pytree-dataclass carry structure, and numerical stability notes. Maintained by the user.
