# Bandit Algorithm Implementation with JAX

Theoretical research implementation of multi-armed bandit algorithms with contextual features, optimized using JAX's JIT compilation for efficient computation.

## Overview

This repository implements key algorithms from the multi-armed bandit literature, with a focus on **contextual linear bandits**. The implementation emphasizes numerical stability, computational efficiency via JAX's JIT compilation, and modular design for easy algorithm extensions.

### Implemented Algorithms

- **OFUL** (Optimism in the Face of Uncertainty for Linear bandits)
  - Maintains a confidence ellipsoid around parameter estimates
  - Uses Sherman-Morrison formula for O(d²) inverse design matrix updates
  - Optimistic action selection within the confidence set

## Project Structure

```
src/
├── algorithms/          # Algorithm implementations
│   ├── base.py         # Abstract Algorithm base class
│   └── oful.py         # OFUL algorithm
├── environments/        # Environment/problem definitions
│   ├── base.py         # Abstract Environment base class
│   └── contextual_linear.py  # Contextual Linear Bandit environment
├── experiment.py        # Experiment runner and evaluation utilities
├── visualization.py     # Plotting and result visualization
└── __init__.py

tests/                   # Unit tests
scripts/                 # Experiment scripts and utilities
results/                 # Output results and logs
```

## Installation

### Requirements
- Python ≥ 3.11
- JAX ≥ 0.9.2
- NumPy ≥ 2.4.3
- SciPy ≥ 1.17.1
- Matplotlib ≥ 3.8.0

### Setup with UV

```bash
# Install project with dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## Usage

### Basic Example

```python
from src.algorithms.oful import OFUL
from src.environments.contextual_linear import ContextualLinearBandit

# Initialize environment and algorithm
env = ContextualLinearBandit(num_arms=5, context_dim=10)
algo = OFUL(context_dim=10, lambda_=0.1, delta=0.01)

# Reset for new episode
algo.reset()

# Interaction loop
for t in range(num_steps):
    contexts = env.get_contexts_at_t(t)
    action = algo.select_action(contexts)
    context, reward, best_arm = env.step(t, action)
    algo.update(context, reward)
```

## Development

### Code Quality

- **Formatter & Linter**: [Ruff](https://github.com/astral-sh/ruff)
  - Line length: 120 characters
  - Python target: 3.11
  - Extended linting rules for performance and code quality

```bash
# Format and lint code
ruff format .
ruff check . --fix
```

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Performance

JAX JIT compilation is enabled by default for performance-critical functions. To disable JIT for debugging:

```python
import jax
jax.config.update("jax_disable_jit", True)
```

## References

- Abbasi-Yadkori et al. "Improved Algorithms for Linear Stochastic Bandits" (OFUL algorithm)
- Lattimore & Szepesvári "Bandit Algorithms" (comprehensive bandit theory)

## License

This repository contains AI-generated code implementations for research purposes.

## Notes

- All algorithms use JAX for automatic differentiation and JIT compilation
- Design matrix inversion uses Sherman-Morrison formula to avoid O(d³) complexity
- Confidence radius computation follows the concentration inequalities from the OFUL paper
