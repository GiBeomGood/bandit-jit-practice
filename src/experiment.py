"""Experiment runner for bandit algorithms."""

import functools
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from src.algorithms import OFUL
from src.environments.contextual_linear import (
    sample_contexts,
    sample_true_theta,
)


def run_episode_scan(
    seed: int,
    context_dim: int,
    num_arms: int,
    num_steps: int,
    **kwargs,
) -> jnp.ndarray:
    """Run a single OFUL episode using jax.lax.scan for the step loop.

    Pure function: all randomness is derived from seed. Compatible with jax.vmap.
    All environment and algorithm hyperparameters flow through **kwargs so that
    context_bound reaches OFUL.compute_confidence_radius inside OFUL.make_step_fn.

    Parameters
    ----------
    seed : int
        Random seed for this episode (int or scalar JAX array).
    context_dim : int
        Feature dimension.
    num_arms : int
        Number of arms.
    num_steps : int
        Episode length (time steps).
    **kwargs
        All hyperparameters: context_bound, lambda_, subgaussian_scale,
        norm_bound, delta. Passed as-is to OFUL.make_init_carry and OFUL.make_step_fn.

    Returns
    -------
    jnp.ndarray
        Cumulative regret at each step, shape (num_steps,).
    """
    key = jax.random.PRNGKey(seed)
    key, k_theta, k_ctx, k_noise = jax.random.split(key, 4)

    true_theta = sample_true_theta(k_theta, context_dim, kwargs["norm_bound"])
    contexts = sample_contexts(k_ctx, num_steps, num_arms, context_dim, kwargs["context_bound"])
    noises = jax.random.normal(k_noise, shape=(num_steps,))

    init_carry = OFUL.make_init_carry(context_dim, **kwargs)
    step_fn = OFUL.make_step_fn(context_dim, true_theta, **kwargs)

    xs = (contexts, noises, jnp.arange(num_steps))
    _, cumulative_regrets = jax.lax.scan(step_fn, init_carry, xs)
    return cumulative_regrets


def run_episodes(
    seeds: jnp.ndarray,
    context_dim: int,
    num_arms: int,
    num_steps: int,
    **kwargs,
) -> jnp.ndarray:
    """Run multiple episodes in parallel using jax.jit(jax.vmap(...)).

    Parameters
    ----------
    seeds : jnp.ndarray
        Integer array of seeds, shape (num_episodes,).
    context_dim : int
        Feature dimension.
    num_arms : int
        Number of arms.
    num_steps : int
        Episode length (time steps).
    **kwargs
        All hyperparameters forwarded to run_episode_scan (context_bound,
        lambda_, subgaussian_scale, norm_bound, delta).

    Returns
    -------
    jnp.ndarray
        Shape (num_episodes, num_steps).
    """
    episode_fn = functools.partial(
        run_episode_scan,
        context_dim=context_dim,
        num_arms=num_arms,
        num_steps=num_steps,
        **kwargs,
    )
    compiled_fn = jax.jit(jax.vmap(episode_fn))
    return compiled_fn(seeds)


class ExperimentRunner:
    """Runner for conducting bandit experiments over multiple episodes."""

    def __init__(
        self,
        context_dim: int,
        num_arms: int,
        num_steps: int,
        num_episodes: int,
        context_bound: float = 1.0,
        seed: int = None,
        algo_cfg: Dict[str, Any] = None,
    ):
        """Initialize experiment runner.

        Parameters
        ----------
        context_dim : int
            Feature dimension.
        num_arms : int
            Number of arms.
        num_steps : int
            Episode length (time steps per episode).
        num_episodes : int
            Number of episodes to run.
        context_bound : float
            Context norm bound.
        seed : int, optional
            Random seed for reproducibility.
        algo_cfg : dict, optional
            Algorithm hyperparameters dict passed as-is from the algo config section.
            Keys: lambda_, subgaussian_scale, norm_bound, delta.
        """
        self.num_episodes = num_episodes
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.num_steps = num_steps
        self.context_bound = context_bound
        self.seed = seed
        self.algo_cfg = algo_cfg if algo_cfg is not None else {}

        merged_kwargs = {**self.algo_cfg, "context_bound": context_bound}
        episode_fn = functools.partial(
            run_episode_scan,
            context_dim=context_dim,
            num_arms=num_arms,
            num_steps=num_steps,
            **merged_kwargs,
        )
        self._compiled_run = jax.jit(jax.vmap(episode_fn))

    @classmethod
    def from_yaml(cls, config_path: str) -> "ExperimentRunner":
        """Construct an ExperimentRunner from a YAML config file.

        Reads the ``experiment`` and ``algo`` sections of the YAML and passes
        them as-is. No individual key extraction — the YAML is the single
        source of truth for all hyperparameters.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.

        Returns
        -------
        ExperimentRunner
            Fully configured ``ExperimentRunner`` instance.
        """
        cfg = OmegaConf.load(config_path)
        exp_cfg = OmegaConf.to_container(cfg.experiment, resolve=True)
        algo_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
        return cls(**exp_cfg, algo_cfg=algo_cfg)

    def run(self) -> Dict[str, Any]:
        """Run the experiment over multiple episodes using jax.jit(jax.vmap(...)).

        Returns
        -------
        dict
            Dict with keys:

            - ``regrets``: numpy array of shape (num_episodes, num_steps),
              cumulative regrets for each episode.
            - ``configs``: dict with experiment configurations.
            - ``metadata``: dict with additional info (seeds, etc.).
        """
        seed_base = self.seed if self.seed is not None else 0
        seeds = jnp.arange(self.num_episodes, dtype=jnp.int32) + seed_base
        all_regrets = np.array(self._compiled_run(seeds))

        return {
            "regrets": all_regrets,
            "configs": {
                "num_episodes": self.num_episodes,
                "context_dim": self.context_dim,
                "num_arms": self.num_arms,
                "num_steps": self.num_steps,
                "context_bound": self.context_bound,
                "algo_cfg": self.algo_cfg,
                "seed": self.seed,
            },
            "metadata": {
                "num_episodes_completed": self.num_episodes,
                "seed_base": self.seed,
            },
        }

    def save_results(self, path: str) -> None:
        """Save experiment results to file.

        Parameters
        ----------
        path : str
            Path to save results (should end with .npz).
        """
        result = self.run()
        np.savez(
            path,
            regrets=result["regrets"],
            **result["configs"],
        )

    @staticmethod
    def load_results(path: str) -> Dict[str, Any]:
        """Load experiment results from file.

        Parameters
        ----------
        path : str
            Path to load results from (should end with .npz).

        Returns
        -------
        dict
            Dict with loaded results.
        """
        data = np.load(path, allow_pickle=True)

        regrets = data["regrets"]
        configs = {}
        for key in data.files:
            if key != "regrets":
                value = data[key]
                if isinstance(value, np.ndarray):
                    configs[key] = value.item() if value.ndim == 0 else value
                else:
                    configs[key] = value

        return {
            "regrets": regrets,
            "configs": configs,
        }
