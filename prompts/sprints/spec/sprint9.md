# Sprint 9: Architecture Refactor — JIT-First Design, YAML Config Centralization, and Function Decomposition

**Status**: Completed (2026-04-09)
**Created**: 2026-04-09

---

## Sprint Goal

Refactor the codebase around three principles:
(1) JIT-first — all code paths assume JIT, non-JIT is achieved only via `jax.disable_jit()`;
(2) YAML-first — `ExperimentRunner` reads config from YAML, no more constructor-level parameter passing;
(3) Decomposed functions — `jit_oful_select_action` and `jit_oful_update` are split into their constituent pure functions, with JIT applied at the call site (via `jax.jit(vmap(fn))` patterns).

> **JAX Constraint Note (read before implementing)**:
> `jax.vmap` operates on pure functions — plain Python callables that accept JAX arrays and return JAX arrays. Python class instances (`Environment`, `Algorithm`) carry mutable Python-side state and cannot be passed through `vmap`. The existing `run_episode_scan` approach (generating all randomness from a seed inside a pure function) is already the correct design for `vmap`. "Reuse `Environment` in vmap" does **not** mean passing `ContextualLinearBandit` instances through `vmap`; instead, it means extracting the data-generation logic from `ContextualLinearBandit.reset()` and `ContextualLinearBandit.step()` into standalone pure functions that `run_episode_scan` can call. See Task 9.1 for details.

---

## Execution Order / Dependency Graph

```
Task 9.1 (pure env functions)   ──┐
Task 9.2 (decompose jit fns)    ──┤──> Task 9.4 (update run_episode_scan)
Task 9.3 (YAML config)          ──┘
                                       ──> Task 9.5 (tests + cleanup)
```

- **Phase 1** (all independent, can be done in any order): Task 9.1, 9.2, 9.3
- **Phase 2** (depends on Phase 1): Task 9.4 (depends on 9.1 + 9.2), Task 9.5 (depends on all)

---

## Task List

---

### Task 9.1: Extract Pure Environment Functions from `ContextualLinearBandit`

**Description**: The logic inside `ContextualLinearBandit.reset()` and `ContextualLinearBandit.step()` contains pure JAX computations (theta generation, context sampling, reward computation) that are currently locked inside a stateful class. Extract these into module-level pure functions in `src/environments/contextual_linear.py`. The class itself stays and delegates to these functions internally — no breaking changes to the class API.

The goal is that `run_episode_scan` (in `src/experiment.py`) can call these pure functions directly instead of duplicating the sampling logic.

**Inputs / Dependencies**:
- `src/environments/contextual_linear.py` — `ContextualLinearBandit.reset()` and `.step()` contain the logic to extract
- No prerequisite Tasks

**Outputs / Deliverables**:
- `src/environments/contextual_linear.py` modified:
  - New function: `sample_true_theta(key, context_dim, param_norm_bound) -> jnp.ndarray`
  - New function: `sample_contexts(key, num_steps, num_arms, context_dim, context_bound) -> jnp.ndarray`
  - New function: `compute_reward(true_theta, context, noise) -> jnp.ndarray`
  - `ContextualLinearBandit.reset()` delegates to `sample_true_theta` and `sample_contexts`
  - `ContextualLinearBandit.step()` delegates to `compute_reward`
- All existing tests continue to pass

**Acceptance Criteria**:
1. `sample_true_theta`, `sample_contexts`, `compute_reward` are importable from `src.environments.contextual_linear`.
2. Each function is a pure function: same inputs always produce same outputs, no side effects.
3. `ContextualLinearBandit.reset()` and `.step()` produce identical outputs before and after the refactor (existing tests pass with no changes).
4. No `@jax.jit` on these new functions — they are building blocks; JIT is applied at call site.

---

### Task 9.2: Decompose `jit_oful_select_action` and `jit_oful_update`

**Description**: In `src/algorithms/oful.py`, `jit_oful_select_action` and `jit_oful_update` are currently pre-bundled JIT functions. The rationale was to reduce compilation overhead, but now that `run_episode_scan` uses `jax.jit(jax.vmap(...))` as the top-level compilation unit, per-function bundling is redundant. Remove the `@jax.jit` decorators from `jit_oful_select_action` and `jit_oful_update`, rename them to `oful_select_action` and `oful_update` to reflect that they are no longer inherently JIT-compiled, and update all call sites.

**JAX Note**: This is safe because the functions will be JIT-compiled at the `jax.jit(jax.vmap(run_episode_scan))` level. When `jax.disable_jit()` is used, the whole scan trace is affected uniformly — no partial JIT leaks.

**Inputs / Dependencies**:
- `src/algorithms/oful.py`
- `src/experiment.py` (imports `jit_oful_select_action`, `jit_oful_update`)
- `tests/test_oful.py`, `tests/test_vmap.py` (may reference old names)
- No prerequisite Tasks

**Outputs / Deliverables**:
- `src/algorithms/oful.py` modified:
  - `jit_oful_select_action` renamed to `oful_select_action`, `@jax.jit` removed
  - `jit_oful_update` renamed to `oful_update`, `@jax.jit` removed
  - `OFUL.select_action()` and `OFUL.update()` updated to call the renamed functions
- `src/experiment.py` updated: import and call sites updated to `oful_select_action`, `oful_update`
- All tests pass

**Acceptance Criteria**:
1. Neither `oful_select_action` nor `oful_update` has `@jax.jit` in `oful.py`.
2. `jit_oful_select_action` and `jit_oful_update` no longer exist anywhere in the codebase (no dead exports).
3. `run_episode_scan` in `src/experiment.py` calls `oful_select_action` and `oful_update`.
4. `jax.disable_jit()` context manager applied around `run_episode_scan` (or its vmap wrapper) produces correct non-JIT execution — confirmed by a test using `jax.disable_jit()`.

---

### Task 9.3: YAML-First `ExperimentRunner` — Load Config from File

**Description**: `ExperimentRunner.__init__` currently accepts all hyperparameters as constructor arguments. Scripts and tests manually pull values from YAML and pass them as keyword args. Instead, add a `from_yaml` class method that takes a YAML path and returns a fully configured `ExperimentRunner` instance. The existing constructor signature must remain unchanged for backward compatibility with current tests.

This also means `scripts/benchmark_vmap.py` and any other scripts should use `ExperimentRunner.from_yaml(path)` rather than manually unpacking config fields.

**Inputs / Dependencies**:
- `src/experiment.py` (`ExperimentRunner` class)
- `configs/experiment.yaml`, `configs/test.yaml`, `configs/benchmark.yaml`
- No prerequisite Tasks

**Outputs / Deliverables**:
- `src/experiment.py` modified:
  - New class method: `ExperimentRunner.from_yaml(config_path: str) -> ExperimentRunner`
  - Uses `OmegaConf.load` + `OmegaConf.to_container` internally
  - Maps `experiment.*` and `algo.*` sections to constructor arguments
- `scripts/benchmark_vmap.py` updated to use `ExperimentRunner.from_yaml(args.config)`
- No existing test changes required (constructor still works)

**Acceptance Criteria**:
1. `ExperimentRunner.from_yaml("configs/experiment.yaml").run()` produces a result dict with correct `regrets` shape.
2. `ExperimentRunner.from_yaml("configs/test.yaml")` produces the same instance as constructing manually with `test.yaml` values.
3. `benchmark_vmap.py` no longer manually unpacks `exp.*` / `algo.*` fields for the runner.
4. All existing `ExperimentRunner(...)` constructor call sites continue to work unchanged.

**Notes**: `configs/experiment.yaml` and `configs/test.yaml` do not currently have a `use_vmap` field. Add `use_vmap: false` to both files so `from_yaml` can read it. Default to `false` if the key is absent.

---

### Task 9.4: Update `run_episode_scan` to Use Extracted Pure Functions

**Description**: After Task 9.1 and 9.2 are complete, update `run_episode_scan` in `src/experiment.py` to use the pure environment functions extracted in Task 9.1 (`sample_true_theta`, `sample_contexts`, `compute_reward`) and the renamed algorithm functions from Task 9.2 (`oful_select_action`, `oful_update`). This ensures that sequential code (`ContextualLinearBandit`) and the scan-based code share the same underlying data generation logic — eliminating the current duplication where `run_episode_scan` re-implements theta/context generation inline.

**Inputs / Dependencies**:
- Task 9.1 complete (`sample_true_theta`, `sample_contexts`, `compute_reward` available)
- Task 9.2 complete (`oful_select_action`, `oful_update` available)
- `src/experiment.py`

**Outputs / Deliverables**:
- `src/experiment.py` modified:
  - `run_episode_scan` imports and calls `sample_true_theta`, `sample_contexts` from `src.environments.contextual_linear`
  - `run_episode_scan` calls `oful_select_action`, `oful_update` (already done if Task 9.2 is done)
  - Inline theta/context generation code removed from `run_episode_scan` body
- All existing tests pass, including `tests/test_vmap.py`

**Acceptance Criteria**:
1. `run_episode_scan` no longer contains inline JAX calls that duplicate `sample_true_theta` or `sample_contexts`.
2. `run_episode_scan` output is numerically identical before and after this change for the same seed (determinism preserved).
3. `run_episodes_vmap` (which calls `run_episode_scan`) continues to work — shape and determinism tests pass.
4. `jax.disable_jit()` wrapping `run_episodes_vmap` runs without error (JIT-first design verified).

---

### Task 9.5: Test Updates and `disable_jit` Verification

**Description**: Add and update tests to cover: (a) `disable_jit` correctness, (b) `ExperimentRunner.from_yaml` behavior, (c) the renamed `oful_select_action`/`oful_update` functions. Also verify no dead imports or stale references remain in the test suite.

**Inputs / Dependencies**:
- Tasks 9.1–9.4 complete
- `tests/test_oful.py`, `tests/test_vmap.py`, `tests/test_experiment.py`

**Outputs / Deliverables**:
- `tests/test_oful.py` updated:
  - Import updated from `jit_oful_select_action` / `jit_oful_update` to `oful_select_action` / `oful_update`
  - New test: `jax.disable_jit()` context manager applied around `oful_select_action` call produces correct output shape and type
- `tests/test_experiment.py` updated or extended:
  - New test: `ExperimentRunner.from_yaml("configs/test.yaml")` produces correct shape result
- All 44+ existing tests pass after renaming
- Zero ruff violations

**Acceptance Criteria**:
1. `tests/test_oful.py` contains no references to `jit_oful_select_action` or `jit_oful_update`.
2. A test using `with jax.disable_jit(): run_episodes_vmap(...)` runs without error and returns correct shape.
3. `ExperimentRunner.from_yaml("configs/test.yaml").run()["regrets"].shape == (num_episodes, num_steps)`.
4. `uv run python -m pytest tests/ -v` passes with zero failures and zero ruff violations.

---

## Additional Notes

### On "JIT-first design" feasibility

`jax.jit` is applied at the outermost `jax.jit(jax.vmap(run_episode_scan))` level. Helper functions like `oful_select_action`, `oful_update`, `sample_true_theta`, etc., do not need their own `@jax.jit` — they are compiled as part of the outer JIT trace. To run without JIT, use `with jax.disable_jit():` wrapping the outermost call. This is fully feasible.

### On `ExperimentRunner.run()` with `use_vmap=False`

When `use_vmap=False`, `run_episode_scan` or `run_single_episode_sequential` is called per episode. These still work correctly under `disable_jit` via the same context manager. No separate non-JIT code path is needed.

### What is NOT in scope for Sprint 9

- Adding new algorithms (Thompson Sampling, LinUCB variants)
- Modifying visualization or plotting code
- Changing the `Algorithm` or `Environment` abstract base class signatures

**Last Updated**: 2026-04-09
