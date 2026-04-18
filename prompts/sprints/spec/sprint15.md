# Sprint 15: Algorithm File Cleanup and experiment.py Decoupling

## Sprint Goal

Move all algorithm-specific code out of `experiment.py` into `oful.py` and `lts.py`. Eliminate cosmetic noise (`# ---` dividers) and API inconsistencies (manual key extraction, long parameter lists) across the codebase. After this sprint, `experiment.py` contains only orchestration logic.

## Execution Order

```
Task 1 (move algorithm code to algorithm files)
Task 2 (inline trivial functions in oful.py and lts.py) in parallel
Task 3 (strip experiment.py to orchestration only — after Task 1 and 2)
Task 4 (divider removal + kwargs + config unpacking — after Task 3)
```

---

### Task 1: Move all algorithm-specific code from experiment.py into algorithm files

**Description**: Every algorithm's carry dataclass, init function, and step closure currently lives in `experiment.py`. Move each piece into its own algorithm file: OFUL-specific code into `oful.py`, LTS-specific code into `lts.py`. Remove any thin wrapper functions created only to delegate to the moved code. `experiment.py` becomes purely orchestration — it imports from algorithm files and wires them together.

**Inputs / Dependencies**:
- `src/experiment.py` (source of all definitions to move)
- `src/algorithms/oful.py`, `src/algorithms/lts.py` (destinations)

**Outputs / Deliverables**:
- Each algorithm file is self-contained: carry dataclass, init, and step closure all present
- `experiment.py` contains no algorithm-specific definitions; only imports and orchestration logic

**Acceptance Criteria**:
- No carry dataclass, init function, or step closure is defined in `experiment.py`
- All algorithm-specific symbols in `experiment.py` are import-based references
- `uv run python -m pytest tests/` passes

---

### Task 2: Eliminate trivial and docstring-heavy thin functions across algorithm files

**Description**: Both `oful.py` and `lts.py` contain functions whose body is a single expression or a few lines, yet carry multi-line docstrings — the documentation outweighs the code. Inline these into their callers. The rule: if removing the function and writing its body inline produces no loss of clarity and no increase in complexity, inline it. Apply this to every algorithm file.

Examples of functions to inline (not exhaustive — audit both files):
- `lts.py`: `lts_update_design_matrix_inv` (3 lines, called only by `lts_update`), `lts_select_action` (1 line)
- `oful.py`: `update_design_matrix_inv`, `update_sum_reward_context`, `compute_theta_hat` (each a single expression called only by `oful_update`)

**Inputs / Dependencies**:
- `src/algorithms/lts.py`, `src/algorithms/oful.py`

**Outputs / Deliverables**:
- Both files have fewer exported functions; inlined logic appears directly in callers

**Acceptance Criteria**:
- No standalone function exists whose entire body is a single expression or whose docstring is longer than its code
- `uv run python -m pytest tests/` passes

---

### Task 3: Strip experiment.py down to orchestration only

**Description**: After Task 1 and 2, audit `experiment.py` for any remaining algorithm-specific logic (LTS carry dataclass, LTS init, LTS step closure, registry entries for non-orchestration concerns). Move anything that is not registry / episode runner / `ExperimentRunner` into the appropriate algorithm file.

**Inputs / Dependencies**:
- Task 1 and Task 2 completed
- `src/experiment.py`, `src/algorithms/lts.py`, `src/algorithms/oful.py`

**Outputs / Deliverables**:
- `experiment.py` contains only: algorithm registry, `run_episode`, `ExperimentRunner` (or equivalent orchestration class/functions)
- Algorithm files are self-contained

**Acceptance Criteria**:
- No carry dataclass, init function, or step closure defined in `experiment.py`
- `uv run python main.py` runs without error
- `uv run python -m pytest tests/` passes

---

### Task 4: Cosmetic and API cleanup across all files

**Description**: Three targeted cleanups applied across all touched files.

1. **Remove `# ---` dividers**: Delete every `# ---` (any length) separator comment line from all `src/` files.

2. **kwargs for fixed parameters**: Functions that receive fixed hyperparameters (values that do not change during a scan loop) must accept them as `**kwargs`, not as individual positional or keyword arguments. The scan pattern is: time-varying inputs go into `carry`; fixed parameters flow through `**kwargs`. No function call should spell out fixed parameters by name one by one — they should be passed as a single `**kwargs` dict derived from config.

3. **Config-driven inputs**: The config structure (YAML) is the single source of truth for all algorithm hyperparameters. Modifying a hyperparameter must require only a YAML edit — no Python changes. This means: no manual extraction of individual config keys in Python code, no rebuilding of config-like dicts by hand. Config dicts are passed as-is (via `**cfg` or equivalent), and functions consume them through `**kwargs`. The mapping between YAML keys and function parameters must be direct and transparent.

**Inputs / Dependencies**:
- Task 3 completed
- `src/experiment.py`, `src/algorithms/oful.py`, `src/algorithms/lts.py`, `configs/experiment.yaml`, `configs/minimal.yaml`

**Outputs / Deliverables**:
- No `# ---` lines remain in any `src/` file
- Fixed hyperparameters flow from YAML → dict → `**kwargs` without intermediate unpacking
- Adding or changing a hyperparameter requires only a YAML edit and a function body change — not a signature change at call sites

**Acceptance Criteria**:
- `grep -r "# ---" src/` returns no results
- No call site passes fixed hyperparameters as individual named arguments; they are forwarded as `**kwargs`
- `uv run python main.py` runs without error; changing a hyperparameter value in YAML takes effect without any Python edits
