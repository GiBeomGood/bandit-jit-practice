# Sprint 12: Codebase Simplification and Single Entry Point

**Status**: Completed (2026-04-15)

## Sprint Goal

Consolidate the codebase around a single `main.py` entry point that runs bandit experiments (JIT/vmap applied) and saves regret graphs. Remove all code not required by `main.py`, enforce reproducibility via deterministic seeding, and finish by renaming files to drop now-redundant JIT/vmap/JAX qualifiers from filenames.

## Execution Order / Dependency Graph

```
Task 1 (Create main.py)
    └── Task 2 (Reproducibility + tests)
            └── Task 3 (Delete unused code)
                    └── Task 4 (Rename files)
```

All tasks are strictly sequential. No parallel execution.

---

## Tasks

### Task 1: Create `main.py` — Single Experiment Entry Point

**Description**: Create `main.py` at the project root. It must:
- Accept a YAML config file path as its sole input (e.g., via CLI argument or a hardcoded default pointing to a config).
- Run bandit algorithm experiments with JIT and vmap applied.
- Save regret graph(s) as image file(s). The graph title and the output filename must both clearly reflect which YAML config file was used (e.g., derived from the config filename).
- Be the authoritative, single way to run experiments in this project.

**Inputs / Dependencies**:
- Existing source modules under `src/`
- Existing YAML configs under `configs/`

**Outputs / Deliverables**:
- `main.py` at project root
- A designated output directory (e.g., `outputs/` or `results/`) where graph images are saved
- Any config key additions required (e.g., output path, seed) documented in the relevant YAML files

**Acceptance Criteria**:
1. `uv run python main.py --config configs/experiment.yaml` (or equivalent) runs without error and produces a graph image file.
2. The output image filename contains the config name (e.g., `experiment_regret.png` for `experiment.yaml`).
3. The graph title displayed in the image also references the config name.
4. The script accepts at minimum one config file; behavior with a missing or invalid config path raises a clear error.

---

### Task 2: Reproducibility — Deterministic Seeding and Tests

**Description**: Ensure that running `main.py` twice with the same YAML config always produces numerically identical results. Achieve this by fixing a random seed in the config (or derivable from it) and passing it consistently through all JAX random operations. Write tests that verify both correct behavior and reproducibility.

**Inputs / Dependencies**:
- Task 1 must be complete (`main.py` exists and runs)
- Existing test infrastructure under `tests/`

**Outputs / Deliverables**:
- Updated `main.py` and/or `src/` modules to guarantee deterministic execution given a fixed seed
- A test config file (under `configs/`) that is as minimal as possible while still exercising the full code path — small number of arms, short horizon, few algorithms
- Test file(s) under `tests/` covering:
  - Normal operation: regret values are sensible (non-negative, bounded, monotonically non-decreasing cumulative regret)
  - Reproducibility: two independent runs with the same config produce identical regret arrays

**Acceptance Criteria**:
1. `uv run python -m pytest tests/` passes with all new tests green.
2. Two consecutive runs of `main.py` with the same config produce byte-identical or numerically identical regret outputs.
3. Tests use only the minimal test config — they do not import or depend on full-scale experiment configs.
4. No `ruff` checks are added anywhere.

---

### Task 3: Delete Unused Code

**Description**: Remove all source files, modules, scripts, and test files that are not required by `main.py` or its tests. For example, if a class contains only `@staticmethod` methods that are used elsewhere, extract those functions to module level and delete the class wrapper. Leave no orphaned imports or dead code.

**Inputs / Dependencies**:
- Tasks 1 and 2 must be complete (the set of required code is now stable)

**Outputs / Deliverables**:
- Deleted: any `src/` module, `scripts/` file, `tests/` file, or `configs/` file not reachable from `main.py` or the tests written in Task 2
- Modified: any module where a staticmethod-only class is replaced by module-level functions
- Updated: all import statements that referenced the deleted classes or files

**Acceptance Criteria**:
1. `uv run python -m pytest tests/` still passes after deletions.
2. `uv run python main.py` still runs without error.
3. No class in the remaining codebase exists solely to namespace staticmethods.
4. No unused imports remain in any retained file.

---

### Task 4: Rename Files — Remove Redundant JIT/vmap/JAX Qualifiers (Final Task)

**Description**: Since JIT and vmap are the standard execution model for this project, their presence in filenames is now redundant. Rename all remaining source files, test files, config files, and script files that include `jit`, `vmap`, or `jax` in their names where those qualifiers add no distinguishing information. Update all internal references (imports, config paths, test discovery patterns) accordingly.

**Inputs / Dependencies**:
- Task 3 must be complete (only files that will survive are present)

**Outputs / Deliverables**:
- Renamed files with clean, qualifier-free names
- Updated import paths and any string references to filenames within the codebase

**Acceptance Criteria**:
1. No retained filename contains `jit`, `vmap`, or `jax` as a redundant qualifier.
2. `uv run python -m pytest tests/` passes after renames.
3. `uv run python main.py` runs without error after renames.
4. All internal import statements and config path strings reflect the new filenames.

---

## Notes

- No `ruff` checks must be added in any task.
- Tasks must be executed strictly in order: Task 1 → Task 2 → Task 3 → Task 4.
- Task 4 is explicitly the final task; no further structural changes after renaming.
