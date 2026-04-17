---
name: Sprint 12 Task 2 — Reproducibility and Minimal Config
description: Design decisions for deterministic seeding, minimal test config, and reproducibility test structure
type: project
---

Two runs with the same config yield identical regrets because `ExperimentRunner.run()` derives per-episode seeds as `jnp.arange(num_episodes) + seed_base` — fully deterministic given a fixed `seed` in the YAML.

`configs/minimal.yaml` is the designated minimal test config: context_dim=2, num_arms=3, num_steps=20, num_episodes=2, seed=7.

`tests/test_reproducibility.py` tests: shape, non-negative, monotone, bounded regret, same-config reproducibility (assert_array_equal), different-seed divergence.

Tests import only `configs/minimal.yaml` — no dependency on full-scale experiment configs.

**Why:** Sprint 12 Task 2 requires deterministic execution and a minimal config to avoid slow tests.

**How to apply:** When reviewing Task 3+ tests, confirm they still reference `configs/minimal.yaml` (not experiment.yaml or benchmark.yaml) for reproducibility tests.
