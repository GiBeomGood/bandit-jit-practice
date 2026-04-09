"""Tests for jax.lax.scan and jax.vmap episode functions."""

import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from src.experiment import ExperimentRunner, run_episode_scan, run_episodes_vmap


@pytest.fixture
def cfg():
    """Load test configuration."""
    return OmegaConf.load("configs/test.yaml")


@pytest.fixture
def episode_kwargs(cfg):
    """Return common episode keyword arguments from test config."""
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
# run_episode_scan tests
# ---------------------------------------------------------------------------


def test_run_episode_scan_shape(episode_kwargs):
    """run_episode_scan returns array of shape (num_steps,)."""
    result = run_episode_scan(seed=0, **episode_kwargs)
    assert result.shape == (episode_kwargs["num_steps"],)


def test_run_episode_scan_deterministic(episode_kwargs):
    """Same seed produces identical cumulative regrets."""
    r1 = run_episode_scan(seed=42, **episode_kwargs)
    r2 = run_episode_scan(seed=42, **episode_kwargs)
    np.testing.assert_array_equal(np.array(r1), np.array(r2))


def test_run_episode_scan_monotone(episode_kwargs):
    """Cumulative regrets are non-decreasing (regret ≥ 0 at each step)."""
    result = np.array(run_episode_scan(seed=7, **episode_kwargs))
    diffs = np.diff(result)
    assert np.all(diffs >= -1e-6), "Cumulative regrets must be non-decreasing"


# ---------------------------------------------------------------------------
# run_episodes_vmap tests
# ---------------------------------------------------------------------------


def test_run_episodes_vmap_shape(episode_kwargs):
    """run_episodes_vmap returns array of shape (num_episodes, num_steps)."""
    num_episodes = 4
    seeds = jnp.arange(num_episodes, dtype=jnp.int32)
    result = run_episodes_vmap(seeds=seeds, **episode_kwargs)
    assert result.shape == (num_episodes, episode_kwargs["num_steps"])


def test_run_episodes_vmap_different_seeds_differ(episode_kwargs):
    """Different seeds produce different trajectories."""
    seeds = jnp.array([0, 1], dtype=jnp.int32)
    result = run_episodes_vmap(seeds=seeds, **episode_kwargs)
    assert not jnp.allclose(result[0], result[1]), "Different seeds must yield different regrets"


# ---------------------------------------------------------------------------
# ExperimentRunner with use_vmap=True
# ---------------------------------------------------------------------------


def test_experiment_runner_vmap_shape(cfg):
    """ExperimentRunner(use_vmap=True).run() returns regrets of shape (num_episodes, num_steps)."""
    exp = cfg.experiment
    algo = cfg.algo
    runner = ExperimentRunner(
        num_episodes=exp.num_episodes,
        context_dim=exp.context_dim,
        num_arms=exp.num_arms,
        num_steps=exp.num_steps,
        context_bound=exp.context_bound,
        algo_params=OmegaConf.to_container(algo, resolve=True),
        seed=exp.seed,
        use_vmap=True,
    )
    result = runner.run()
    regrets = result["regrets"]
    assert regrets.shape == (exp.num_episodes, exp.num_steps)


def test_experiment_runner_sequential_unchanged(cfg):
    """ExperimentRunner(use_vmap=False).run() still works correctly."""
    exp = cfg.experiment
    algo = cfg.algo
    runner = ExperimentRunner(
        num_episodes=exp.num_episodes,
        context_dim=exp.context_dim,
        num_arms=exp.num_arms,
        num_steps=exp.num_steps,
        context_bound=exp.context_bound,
        algo_params=OmegaConf.to_container(algo, resolve=True),
        seed=exp.seed,
        use_vmap=False,
    )
    result = runner.run()
    regrets = result["regrets"]
    assert regrets.shape == (exp.num_episodes, exp.num_steps)
