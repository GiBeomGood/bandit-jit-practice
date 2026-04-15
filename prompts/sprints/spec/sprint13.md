# Sprint 13

**Status**: Completed (2026-04-15)

## Sprint Goal

Reduce the test suite to exactly one file (reproducibility only), removing all other test files, and add a YAML config key that controls the output format of the regret graph. Regret sanity is evaluated visually by the code-reviewer, who inspects the saved graph image and judges whether the curve looks log-like (sublinear growth), not by automated tests.

---

## Execution Order / Dependency Graph

```
Task 1  ──────────────────────────────────────────────────▶  (independent)
Task 2  ──────────────────────────────────────────────────▶  (independent)
```

Tasks 1 and 2 are **fully independent** and can be executed in parallel. Neither modifies files that the other touches.

---

## Tasks

### Task 1: Consolidate tests into one file

**Description**: Delete all existing test files except `conftest.py`. Create exactly one new test file:

- `tests/test_reproducibility.py` — contains only tests whose concern is that two runs with identical config produce bit-for-bit identical results, and that a different seed produces a different result.

Regret sanity (shape, sign, monotonicity, boundedness) is **not tested in Python**. Instead, the code-reviewer evaluates it visually by inspecting the saved graph image — the curve should look log-like (sublinear growth).

Tests that do not belong to the reproducibility category must be deleted without replacement.

**Inputs / Dependencies**:
- No prerequisite tasks.
- Files to read before acting: all files currently under `tests/` (listed below) and `configs/minimal.yaml`.

Current test files and their disposition:

| File | Action |
|---|---|
| `tests/conftest.py` | Keep as-is |
| `tests/test_reproducibility.py` | Source material — migrate relevant tests then delete |
| `tests/test_experiment.py` | Source material — migrate relevant tests then delete |
| `tests/test_episode_functions.py` | Source material — migrate relevant tests then delete |
| `tests/test_environment.py` | Delete (environment unit tests are out of scope for both target categories) |
| `tests/test_oful.py` | Delete (algorithm unit tests are out of scope for both target categories) |
| `tests/test_visualization.py` | Delete (visualization unit tests are out of scope for both target categories) |

**Migration mapping** (which tests go where):

- `tests/test_reproducibility.py` (new):
  - From current `test_reproducibility.py`: `test_reproducibility_same_config`, `test_different_seed_yields_different_regrets`
  - From `test_experiment.py`: `test_determinism_with_seed`, `test_different_seeds_produce_different_regrets`
  - From `test_episode_functions.py`: `test_run_episode_scan_deterministic`, `test_run_episodes_different_seeds_differ`
  - Deduplicate: keep one canonical test per concept (same-seed ⟹ same result; different-seed ⟹ different result), covering both `run_episode_scan`/`run_episodes` and `ExperimentRunner` entry points.

All tests in both new files must use `configs/minimal.yaml` (or equivalent inline small config) — never `configs/test.yaml` or `configs/experiment.yaml`.

**Outputs / Deliverables**:
- `tests/test_reproducibility.py` — new file
- Deleted: `tests/test_experiment.py`, `tests/test_episode_functions.py`, `tests/test_environment.py`, `tests/test_oful.py`, `tests/test_visualization.py`
- `tests/conftest.py` — unchanged

**Acceptance Criteria**:
1. `uv run python -m pytest tests/` passes with zero failures, zero errors.
2. `tests/` contains exactly: `conftest.py`, `test_reproducibility.py` — no other `.py` files.
3. `test_reproducibility.py` contains at minimum: one test asserting same config ⟹ identical regrets, and one test asserting different seed ⟹ different regrets.

---

### Task 2: Add configurable graph output format

**Description**: Add an `output.format` key to each YAML config file that specifies the graph file format (e.g., `png`, `pdf`, `svg`). Update `main.py` to read this key and pass the correct extension when constructing the output path. `src/visualization.py` already supports all three formats via `_save_figure` — no changes are needed there beyond verifying the existing interface is used correctly.

**Inputs / Dependencies**:
- No prerequisite tasks.
- Files to modify: `configs/experiment.yaml`, `configs/test.yaml`, `configs/minimal.yaml`, `main.py`.
- File to read but not modify: `src/visualization.py` (already supports `.png`, `.pdf`, `.svg`).

**Specification**:

- Add a top-level `output` section to each YAML file with a single key `format`. Accepted values: `png`, `pdf`, `svg`. Default: `png`.
- In `main.py`, the `build_output_path` function currently hardcodes `.png`. Change it so that it reads `cfg.output.format` (or a parameter derived from it) and uses that extension instead.
- The stem of the output filename must remain `{config_name}_regret`.
- If `output.format` is absent from the loaded config, `main.py` must fall back to `png` without crashing.

**Outputs / Deliverables**:
- `configs/experiment.yaml` — `output.format: png` added
- `configs/test.yaml` — `output.format: png` added
- `configs/minimal.yaml` — `output.format: png` added
- `main.py` — `build_output_path` (and any callers) updated to use the config-driven format

**Acceptance Criteria**:
1. `uv run python main.py --config configs/experiment.yaml` completes without error and writes a file named `outputs/experiment_regret.png`.
2. Manually changing `output.format` to `pdf` in `configs/experiment.yaml` and re-running produces `outputs/experiment_regret.pdf` instead of `outputs/experiment_regret.png`.
3. Removing the `output` section entirely from a config and re-running falls back to `.png` without raising an exception.
4. `src/visualization.py` is not modified (its interface already handles all three formats).

## Regret Sanity — Visual Review (code-reviewer responsibility)

After Task 2 is implemented, the code-reviewer must run `main.py` to generate the regret graph, then **visually inspect the saved image** using the Read tool (vision). The curve must look log-like: rapid initial decrease in per-step regret, flattening over time (sublinear cumulative growth). If the curve looks clearly wrong (e.g., linear, diverging, or flat from step 0), the reviewer should BLOCK and explain what is visually wrong.
