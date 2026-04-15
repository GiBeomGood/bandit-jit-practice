"""Reproducibility tests for bandit experiments.

Asserts that identical configs produce bit-for-bit identical results, and that
different seeds produce different results. Uses configs/minimal.yaml exclusively.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from src.experiment import ExperimentRunner, run_episode_scan, run_episodes

MINIMAL_CONFIG = Path(__file__).parent.parent / "configs" / "minimal.yaml"


def _load_episode_kwargs() -> dict:
    """Return run_episode_scan keyword arguments from the minimal config.

    Returns
    -------
    dict
        Keyword arguments matching the run_episode_scan signature.
    """
    cfg = OmegaConf.load(str(MINIMAL_CONFIG))
    exp = cfg.experiment
    algo = cfg.algo
    return {
        "context_dim": exp.context_dim,
        "num_arms": exp.num_arms,
        "num_steps": exp.num_steps,
        "context_bound": exp.context_bound,
        "lambda_": algo.lambda_,
        "subgaussian_scale": algo.subgaussian_scale,
        "norm_bound": algo.norm_bound,
        "delta": algo.delta,
    }


# ---------------------------------------------------------------------------
# Same seed => identical results
# ---------------------------------------------------------------------------


def test_run_episode_scan_same_seed_identical() -> None:
    """Two calls to run_episode_scan with the same seed return identical arrays."""
    kwargs = _load_episode_kwargs()
    r1 = np.array(run_episode_scan(seed=42, **kwargs))
    r2 = np.array(run_episode_scan(seed=42, **kwargs))
    np.testing.assert_array_equal(r1, r2)


def test_run_episodes_same_seeds_identical() -> None:
    """Two calls to run_episodes with the same seed array return identical arrays."""
    kwargs = _load_episode_kwargs()
    seeds = jnp.array([0, 1], dtype=jnp.int32)
    r1 = np.array(run_episodes(seeds=seeds, **kwargs))
    r2 = np.array(run_episodes(seeds=seeds, **kwargs))
    np.testing.assert_array_equal(r1, r2)


def test_experiment_runner_same_config_identical() -> None:
    """Two ExperimentRunner instances built from the same config yield identical regrets."""
    runner_a = ExperimentRunner.from_yaml(str(MINIMAL_CONFIG))
    runner_b = ExperimentRunner.from_yaml(str(MINIMAL_CONFIG))
    regrets_a = runner_a.run()["regrets"]
    regrets_b = runner_b.run()["regrets"]
    np.testing.assert_array_equal(regrets_a, regrets_b)


# ---------------------------------------------------------------------------
# Different seed => different results
# ---------------------------------------------------------------------------


def test_run_episode_scan_different_seeds_differ() -> None:
    """Different seeds passed to run_episode_scan produce different regret arrays."""
    kwargs = _load_episode_kwargs()
    r1 = np.array(run_episode_scan(seed=0, **kwargs))
    r2 = np.array(run_episode_scan(seed=1, **kwargs))
    assert not np.array_equal(r1, r2), "Different seeds must yield different regrets"


def test_run_episodes_different_seeds_differ() -> None:
    """run_episodes with seeds [0, 1] produces different trajectories per episode."""
    kwargs = _load_episode_kwargs()
    seeds = jnp.array([0, 1], dtype=jnp.int32)
    result = np.array(run_episodes(seeds=seeds, **kwargs))
    assert not np.array_equal(result[0], result[1]), (
        "Different seeds must yield different regrets"
    )


def test_experiment_runner_different_seeds_differ() -> None:
    """ExperimentRunner with different seeds produces different regret trajectories."""
    cfg = OmegaConf.load(str(MINIMAL_CONFIG))
    exp = cfg.experiment
    algo = OmegaConf.to_container(cfg.algo, resolve=True)

    runner_a = ExperimentRunner(
        num_episodes=exp.num_episodes,
        context_dim=exp.context_dim,
        num_arms=exp.num_arms,
        num_steps=exp.num_steps,
        context_bound=exp.context_bound,
        algo_params=algo,
        seed=exp.seed,
    )
    runner_b = ExperimentRunner(
        num_episodes=exp.num_episodes,
        context_dim=exp.context_dim,
        num_arms=exp.num_arms,
        num_steps=exp.num_steps,
        context_bound=exp.context_bound,
        algo_params=algo,
        seed=exp.seed + 1,
    )

    regrets_a = runner_a.run()["regrets"]
    regrets_b = runner_b.run()["regrets"]
    assert not np.array_equal(regrets_a, regrets_b), (
        "Different seeds should produce different regret trajectories"
    )
