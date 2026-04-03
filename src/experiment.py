"""Experiment runner for bandit algorithms."""

from typing import Any, Dict

import jax.numpy as jnp
import numpy as np

from src.algorithms.oful import OFUL
from src.environments.contextual_linear import ContextualLinearBandit


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
    cumulative_regrets = jnp.zeros(num_steps)

    for t in range(num_steps):
        contexts_t = env.get_contexts_at_t(t)
        action = algo.select_action(contexts_t)
        _, reward, best_arm_t = env.step(t, action)

        regret_t = _compute_step_regret(contexts_t, action, best_arm_t, true_theta)
        cumulative_regret += regret_t
        cumulative_regrets = cumulative_regrets.at[t].set(cumulative_regret)

        algo.update(contexts_t[action], reward)

    return cumulative_regrets


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
        """
        self.num_episodes = num_episodes
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.num_steps = num_steps
        self.context_bound = context_bound
        self.seed = seed

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
        }

    def run(self) -> Dict[str, Any]:
        """Run the experiment over multiple episodes.

        Returns:
            Dict with keys:
            - regrets: numpy array of shape (num_episodes, num_steps)
                cumulative regrets for each episode
            - configs: dict with experiment configurations
            - metadata: dict with additional info (seeds, etc.)
        """
        all_regrets = np.zeros((self.num_episodes, self.num_steps))

        for episode in range(self.num_episodes):
            episode_seed = (self.seed + episode) if self.seed is not None else episode

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
