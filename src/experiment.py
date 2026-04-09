"""Experiment runner for bandit algorithms."""

import functools
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from src.algorithms.oful import OFUL, oful_select_action, oful_update
from src.environments.contextual_linear import (
    ContextualLinearBandit,
    sample_contexts,
    sample_true_theta,
)


def _compute_step_regret(contexts_t: jnp.ndarray, action: int, best_arm: int, true_theta: jnp.ndarray) -> float:
    """Compute regret for a single step.

    Args:
        contexts_t: Context vectors for all arms at time t, shape (num_arms, context_dim)
        action: Selected action index
        best_arm: Index of best arm (optimal action)
        true_theta: True parameter vector, shape (context_dim,)

    Returns:
        regret_t: Instantaneous regret for this step
    """
    best_reward = float(jnp.dot(true_theta, contexts_t[best_arm]))
    actual_reward = float(jnp.dot(true_theta, contexts_t[action]))
    return best_reward - actual_reward


def _run_episode_loop(env: ContextualLinearBandit, algo: OFUL, true_theta: jnp.ndarray, num_steps: int) -> jnp.ndarray:
    """Execute episode loop and return cumulative regrets.

    Args:
        env: Environment instance
        algo: Algorithm instance
        true_theta: True parameter vector, shape (context_dim,)
        num_steps: Number of steps in episode

    Returns:
        cumulative_regrets: Array of cumulative regrets for each step, shape (num_steps,)
    """
    cumulative_regret = 0.0
    cumulative_regrets = []

    for t in range(num_steps):
        contexts_t = env.get_contexts_at_t(t)
        action = algo.select_action(contexts_t)
        _, reward, best_arm_t = env.step(t, action)

        regret_t = _compute_step_regret(contexts_t, action, best_arm_t, true_theta)
        cumulative_regret += regret_t
        cumulative_regrets.append(cumulative_regret)

        algo.update(contexts_t[action], reward)

    return jnp.array(cumulative_regrets)


def run_single_episode_sequential(
    seed: int,
    context_dim: int,
    num_arms: int,
    num_steps: int,
    context_bound: float,
    lambda_: float,
    subgaussian_scale: float,
    norm_bound: float,
    delta: float,
) -> jnp.ndarray:
    """Run a single episode and return cumulative regrets.

    Args:
        seed: Random seed for this episode
        context_dim: Feature dimension
        num_arms: Number of arms
        num_steps: Episode length (time steps per episode)
        context_bound: Context norm bound
        lambda_: Ridge regularization parameter for OFUL
        subgaussian_scale: Sub-Gaussian variance proxy for OFUL
        norm_bound: Parameter norm bound for OFUL
        delta: Failure probability for OFUL

    Returns:
        cumulative_regrets: Array of shape (num_steps,) with cumulative regrets
    """
    env = ContextualLinearBandit(
        context_dim=context_dim,
        num_arms=num_arms,
        num_steps=num_steps,
        context_bound=context_bound,
        param_norm_bound=norm_bound,
        seed=seed,
    )
    algo = OFUL(
        context_dim=context_dim,
        lambda_=lambda_,
        subgaussian_scale=subgaussian_scale,
        norm_bound=norm_bound,
        context_bound=context_bound,
        delta=delta,
        seed=seed,
    )

    env.reset()
    algo.reset()

    true_theta = env.get_true_theta()
    return _run_episode_loop(env, algo, true_theta, num_steps)


def run_episode_scan(
    seed: int,
    context_dim: int,
    num_arms: int,
    num_steps: int,
    context_bound: float,
    lambda_: float,
    subgaussian_scale: float,
    norm_bound: float,
    delta: float,
) -> jnp.ndarray:
    """Run a single episode using jax.lax.scan for the step loop.

    Pure function: all randomness is derived from seed. Compatible with jax.vmap.

    Args:
        seed: Random seed for this episode (int or scalar JAX array)
        context_dim: Feature dimension
        num_arms: Number of arms
        num_steps: Episode length (time steps)
        context_bound: Context norm bound
        lambda_: Ridge regularization parameter
        subgaussian_scale: Sub-Gaussian variance proxy
        norm_bound: Parameter norm bound (||θ*|| ≤ norm_bound)
        delta: Failure probability

    Returns:
        cumulative_regrets: Cumulative regret at each step, shape (num_steps,)
    """
    key = jax.random.PRNGKey(seed)
    key, k_theta, k_ctx, k_noise = jax.random.split(key, 4)

    true_theta = sample_true_theta(k_theta, context_dim, norm_bound)
    contexts = sample_contexts(k_ctx, num_steps, num_arms, context_dim, context_bound)
    noises = jax.random.normal(k_noise, shape=(num_steps,))

    # Initial OFUL state: B_0 = λI → B_0^{-1} = (1/λ)I
    init_design_matrix_inv = (1.0 / lambda_) * jnp.eye(context_dim)
    init_sum_reward_context = jnp.zeros(context_dim)
    init_carry = (init_design_matrix_inv, init_sum_reward_context, jnp.array(0.0))

    def step_fn(carry: tuple, x: tuple) -> tuple:
        """Execute one bandit step within the scan loop.

        Args:
            carry: (design_matrix_inv, sum_reward_context, cumulative_regret)
            x: (contexts_t, noise_t, t_idx) for this step

        Returns:
            new_carry: Updated (design_matrix_inv, sum_reward_context, cumulative_regret)
            cumulative_regret: Cumulative regret after this step (scalar)
        """
        design_matrix_inv, sum_reward_context, cumulative_regret = carry
        contexts_t, noise_t, t_idx = x

        action = oful_select_action(
            design_matrix_inv,
            sum_reward_context,
            contexts_t,
            t_idx,
            context_dim,
            lambda_,
            subgaussian_scale,
            norm_bound,
            context_bound,
            delta,
        )

        arm_values = contexts_t @ true_theta  # (num_arms,)
        best_arm = jnp.argmax(arm_values)
        reward = arm_values[action] + noise_t
        regret_t = arm_values[best_arm] - arm_values[action]
        new_cumulative_regret = cumulative_regret + regret_t

        new_dm_inv, new_src = oful_update(design_matrix_inv, sum_reward_context, contexts_t[action], reward)

        return (new_dm_inv, new_src, new_cumulative_regret), new_cumulative_regret

    xs = (contexts, noises, jnp.arange(num_steps))
    _, cumulative_regrets = jax.lax.scan(step_fn, init_carry, xs)
    return cumulative_regrets


def run_episodes_vmap(
    seeds: jnp.ndarray,
    context_dim: int,
    num_arms: int,
    num_steps: int,
    context_bound: float,
    lambda_: float,
    subgaussian_scale: float,
    norm_bound: float,
    delta: float,
) -> jnp.ndarray:
    """Run multiple episodes in parallel using jax.jit(jax.vmap(...)).

    Args:
        seeds: Integer array of seeds, shape (num_episodes,)
        context_dim: Feature dimension
        num_arms: Number of arms
        num_steps: Episode length (time steps)
        context_bound: Context norm bound
        lambda_: Ridge regularization parameter
        subgaussian_scale: Sub-Gaussian variance proxy
        norm_bound: Parameter norm bound
        delta: Failure probability

    Returns:
        cumulative_regrets: Shape (num_episodes, num_steps)
    """
    episode_fn = functools.partial(
        run_episode_scan,
        context_dim=context_dim,
        num_arms=num_arms,
        num_steps=num_steps,
        context_bound=context_bound,
        lambda_=lambda_,
        subgaussian_scale=subgaussian_scale,
        norm_bound=norm_bound,
        delta=delta,
    )
    jit_vmapped = jax.jit(jax.vmap(episode_fn))
    return jit_vmapped(seeds)


class ExperimentRunner:
    """Runner for conducting bandit experiments over multiple episodes."""

    def __init__(
        self,
        num_episodes: int,
        context_dim: int,
        num_arms: int,
        num_steps: int,
        context_bound: float = 1.0,
        algo_params: Dict[str, Any] = None,
        seed: int = None,
        use_vmap: bool = False,
    ):
        """Initialize experiment runner.

        Args:
            num_episodes: Number of episodes to run
            context_dim: Feature dimension
            num_arms: Number of arms
            num_steps: Episode length (time steps per episode)
            context_bound: Context norm bound
            algo_params: Dictionary of algorithm parameters
                (lambda_, subgaussian_scale, norm_bound, delta for OFUL)
            seed: Random seed for reproducibility
            use_vmap: If True, use jax.jit(jax.vmap(...)) for parallel episode execution
        """
        self.num_episodes = num_episodes
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.num_steps = num_steps
        self.context_bound = context_bound
        self.seed = seed
        self.use_vmap = use_vmap

        if algo_params is None:
            algo_params = {}
        self.algo_params = {
            "lambda_": algo_params.get("lambda_", 1.0),
            "subgaussian_scale": algo_params.get("subgaussian_scale", 1.0),
            "norm_bound": algo_params.get("norm_bound", 1.0),
            "delta": algo_params.get("delta", 0.01),
        }

        self.configs = {
            "num_episodes": num_episodes,
            "context_dim": context_dim,
            "num_arms": num_arms,
            "num_steps": num_steps,
            "context_bound": context_bound,
            "algo_params": self.algo_params,
            "seed": seed,
            "use_vmap": use_vmap,
        }

    @classmethod
    def from_yaml(cls, config_path: str) -> "ExperimentRunner":
        """Construct an ExperimentRunner from a YAML config file.

        Reads the ``experiment`` and ``algo`` sections of the YAML. The
        ``experiment`` section must contain ``context_dim``, ``num_arms``,
        ``num_steps``, and ``num_episodes``. Optional keys are
        ``context_bound``, ``seed``, and ``use_vmap`` (defaults to ``false``
        if absent).

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            runner: Fully configured ``ExperimentRunner`` instance.
        """
        cfg = OmegaConf.load(config_path)
        exp = OmegaConf.to_container(cfg.experiment, resolve=True)
        algo = OmegaConf.to_container(cfg.algo, resolve=True)

        return cls(
            num_episodes=exp["num_episodes"],
            context_dim=exp["context_dim"],
            num_arms=exp["num_arms"],
            num_steps=exp["num_steps"],
            context_bound=exp.get("context_bound", 1.0),
            algo_params={
                "lambda_": algo.get("lambda_", 1.0),
                "subgaussian_scale": algo.get("subgaussian_scale", 1.0),
                "norm_bound": algo.get("norm_bound", 1.0),
                "delta": algo.get("delta", 0.01),
            },
            seed=exp.get("seed", None),
            use_vmap=exp.get("use_vmap", False),
        )

    def run(self) -> Dict[str, Any]:
        """Run the experiment over multiple episodes.

        Uses jax.jit(jax.vmap(...)) when use_vmap=True, otherwise sequential loop.

        Returns:
            Dict with keys:
            - regrets: numpy array of shape (num_episodes, num_steps)
                cumulative regrets for each episode
            - configs: dict with experiment configurations
            - metadata: dict with additional info (seeds, etc.)
        """
        seed_base = self.seed if self.seed is not None else 0

        if self.use_vmap:
            seeds = jnp.arange(self.num_episodes, dtype=jnp.int32) + seed_base
            regrets_jax = run_episodes_vmap(
                seeds,
                self.context_dim,
                self.num_arms,
                self.num_steps,
                self.context_bound,
                self.algo_params["lambda_"],
                self.algo_params["subgaussian_scale"],
                self.algo_params["norm_bound"],
                self.algo_params["delta"],
            )
            all_regrets = np.array(regrets_jax)
        else:
            all_regrets = np.zeros((self.num_episodes, self.num_steps))
            for episode in range(self.num_episodes):
                episode_seed = seed_base + episode
                regrets = run_single_episode_sequential(
                    episode_seed,
                    self.context_dim,
                    self.num_arms,
                    self.num_steps,
                    self.context_bound,
                    self.algo_params["lambda_"],
                    self.algo_params["subgaussian_scale"],
                    self.algo_params["norm_bound"],
                    self.algo_params["delta"],
                )
                all_regrets[episode, :] = np.array(regrets)

        return {
            "regrets": all_regrets,
            "configs": self.configs,
            "metadata": {
                "num_episodes_completed": self.num_episodes,
                "seed_base": self.seed,
            },
        }

    def save_results(self, path: str) -> None:
        """Save experiment results to file.

        Args:
            path: Path to save results (should end with .npz)
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

        Args:
            path: Path to load results from (should end with .npz)

        Returns:
            Dict with loaded results
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
