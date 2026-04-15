# Sprint 11: JAX-First Cleanup — Remove Non-JAX Code, Rename JIT Functions, and Simplify Tests

**Status**: Completed (2026-04-15)
**Created**: 2026-04-15

---

## Sprint Goal

The debugging phase that required non-JAX fallback code is complete. JAX (jit + vmap) is confirmed working and is now the unconditional baseline. This sprint removes all non-JAX code paths, renames functions that carry a "jit" or "vmap" prefix/suffix that is now redundant, and strips the test suite down to the minimal set that validates the JAX-based experiment paths. All JAX-based experiment scripts must continue to run correctly after the cleanup.

---

## Execution Order / Dependency Graph

```
Task 11.1 (remove non-JAX code)     ──┐
                                      ├──> Task 11.3 (rename functions)  ──> Task 11.4 (tests)
Task 11.2 (simplify tests)          ──┘
```

- **Phase 1** (independent): Task 11.1 and Task 11.2 can be worked in parallel.
- **Phase 2** (depends on Phase 1): Task 11.3 — renaming is done after dead code is removed and test cleanup scope is known, so rename callsites are in the final state.
- **Phase 3** (depends on Phase 2): Task 11.4 — verify the full test suite passes after all structural changes.

---

## Task List

---

### Task 11.1: Remove Non-JAX Code Paths

**Description**: Identify and delete all functions, branches, and modules that were written for the non-JAX (non-jit, non-vmap) execution path. This includes any Python-loop-only implementations, flag-guarded non-jit branches, and standalone non-jit experiment runners that exist solely as fallback alternatives to the JAX versions. JAX is now the only supported execution mode — no fallback code should remain.

The developer must audit `src/` (and `scripts/` if applicable) to locate all such code. Deletion scope includes:
- Functions that duplicate JAX functions but run in pure Python loops without `jax.jit` or `jax.vmap`.
- Any conditional logic of the form "if use_jit: ... else: <non-JAX path>" — the else branch and the condition itself should be removed.
- Config keys (in YAML files) that gate non-JAX behavior, if any exist.

**Inputs / Dependencies**:
- `src/` directory (all modules)
- `scripts/` directory (all scripts)
- `configs/` directory (YAML files)
- No prerequisite tasks

**Outputs / Deliverables**:
- Modified files in `src/` with non-JAX code removed.
- Modified `scripts/` files with non-JAX code removed.
- Modified `configs/` files with any non-JAX toggle keys removed (e.g., `use_jit: false` options).
- No new files created.

**Acceptance Criteria**:
1. No function in `src/` or `scripts/` implements the same logic as a JAX function but without `jax.jit` or `jax.vmap`.
2. No conditional branch in the codebase selects between a JAX path and a non-JAX path at runtime.
3. `uv run python -m ruff check src/ scripts/` reports zero violations after deletion.
4. `uv run python scripts/run_experiment.py` (or equivalent main entry point) completes without error.

---

### Task 11.2: Simplify the Test Suite

**Description**: Audit all files under `tests/` and remove test cases that:
- Target removed non-JAX code paths (will be obsolete after Task 11.1).
- Are redundant duplicates of other tests that cover the same assertion.
- Were written as sanity checks for non-JAX behavior and no longer serve any purpose.

Keep only tests that directly validate the JAX-based (jit/vmap) experiment code paths. The retained test suite must still provide meaningful coverage: at least one test per major public function in `src/`.

The developer determines which specific tests to delete — this is a judgment call guided by the criterion above.

**Inputs / Dependencies**:
- `tests/` directory (all test files)
- No prerequisite tasks (can proceed in parallel with Task 11.1)

**Outputs / Deliverables**:
- Modified test files in `tests/` with redundant/legacy tests removed.
- Retained tests all pass against the (still-present at this stage) codebase.

**Acceptance Criteria**:
1. Every remaining test in `tests/` exercises a JAX-based code path.
2. No test imports or references a function that has been deleted in Task 11.1.
3. `uv run python -m pytest tests/ -v` passes with zero failures (run after Task 11.1 is also complete).
4. Zero ruff violations in `tests/`.

---

### Task 11.3: Rename Functions — Drop Redundant "jit" / "vmap" Affixes

**Description**: With JAX as the unconditional default, function names that include "jit" or "vmap" as a distinguishing prefix or suffix are now misleading — they imply an alternative exists. Rename these functions to drop the redundant affix. Examples of the pattern to address:
- `run_jit_experiment` → `run_experiment`
- `jit_simulate` → `simulate`
- `run_episodes_vmap` → `run_episodes`
- Any similar pattern found in `src/` or `scripts/`

The developer must perform a complete audit of all public function names in `src/` and `scripts/` and apply the rename wherever the "jit" or "vmap" affix is the only thing distinguishing the function from the concept it represents.

Update all call sites (in `src/`, `scripts/`, and `tests/`) to use the new names. No function body logic changes — this is a rename-only task.

**Inputs / Dependencies**:
- Task 11.1 complete (so renamed functions are in their final post-deletion state)
- Task 11.2 complete (so call sites in tests are in their final post-cleanup state)
- `src/`, `scripts/`, `tests/`

**Outputs / Deliverables**:
- Renamed functions in `src/` and `scripts/`.
- All call sites in `src/`, `scripts/`, and `tests/` updated to the new names.
- No logic changes — diffs should show only identifier substitutions.

**Acceptance Criteria**:
1. No public function in `src/` or `scripts/` contains "jit" or "vmap" in its name as a distinguishing qualifier (i.e., where removing it would not cause a name collision).
2. `grep -rn "run_jit\|jit_simulate\|_vmap\|vmap_" src/ scripts/ tests/` returns zero matches for any renamed pattern.
3. `uv run python -m pytest tests/ -v` passes with zero failures.
4. `uv run python -m ruff check src/ scripts/ tests/` reports zero violations.

---

### Task 11.4: End-to-End Verification

**Description**: After all structural changes are complete, perform a full end-to-end verification: run the main experiment entry point, the benchmark script, and the full test suite. Confirm that no regressions were introduced by the cleanup. Fix any import errors, broken references, or name mismatches that surface during this verification — these are considered bugs in Task 11.1–11.3 to be corrected here.

**Inputs / Dependencies**:
- Tasks 11.1, 11.2, 11.3 all complete
- `configs/test.yaml` (fast smoke-test config)
- `configs/experiment.yaml`
- `configs/benchmark.yaml`

**Outputs / Deliverables**:
- All existing JAX-based experiment scripts run cleanly.
- Full test suite passes.
- Any fixup changes needed are applied to `src/`, `scripts/`, or `tests/`.

**Acceptance Criteria**:
1. `uv run python -m pytest tests/ -v` exits with zero failures and zero errors.
2. The main experiment entry point (e.g., `uv run python scripts/run_experiment.py --config configs/test.yaml`) completes without error.
3. `uv run python scripts/benchmark_jit_vmap.py --config configs/test.yaml` completes and prints a speedup ratio.
4. `uv run python -m ruff check src/ scripts/ tests/` reports zero violations.

---

## Out of Scope

- Adding new bandit algorithms or experiment features.
- Changing any algorithm logic or numerical behavior.
- Modifying configuration parameter values (e.g., `num_steps`, `context_dim`).
- Adding new scripts or new test files beyond fixups required by the cleanup.
- Performance optimization beyond what the rename/cleanup incidentally provides.

---

**Last Updated**: 2026-04-15
