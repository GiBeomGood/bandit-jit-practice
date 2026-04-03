"""Contextual Linear Bandit environment implementation."""

from typing import Tuple

import jax
import jax.numpy as jnp

from src.environments.base import Environment


class ContextualLinearBandit(Environment):
    """Contextual Linear Bandit environment.

    In this environment:
    - At each time step t, the environment provides contexts x_{t,a} for each arm a
    - The reward is r_t = θ* · x_{t,a_t} + ε_t, where ε_t ~ N(0, 1)
    - The learner observes the context of the selected arm and the reward
    - True parameter θ* satisfies ||θ*||₂ ≤ param_norm_bound
    """

    def __init__(
        self,
        context_dim: int,
        num_arms: int,
        num_steps: int,
        context_bound: float = 1.0,
        param_norm_bound: float = 1.0,
        seed: int = None,
    ):
        """Initialize ContextualLinearBandit environment.

        Args:
            context_dim: Feature dimension
            num_arms: Number of arms
            num_steps: Episode length
            context_bound: Context norm bound (contexts sampled from [-context_bound, context_bound]^d)
            param_norm_bound: Parameter norm bound (||θ*||₂ ≤ param_norm_bound)
            seed: Random seed for reproducibility
        """
        super().__init__(context_dim, num_arms, num_steps)
        self.context_bound = context_bound
        self.param_norm_bound = param_norm_bound
        self.seed = seed
        self.key = jax.random.PRNGKey(seed if seed is not None else 0)

        # Initialize state
        self._contexts = None
        self._true_theta = None
        self._current_t = None

    def reset(self) -> None:
        """Reset environment for a new episode.

        Samples:
        - True parameter θ* from Normal(0, I), clipped to satisfy ||θ*||₂ ≤ param_norm_bound
        - Context array of shape (num_steps, num_arms, context_dim) with values in
          [-context_bound, context_bound]^context_dim
        """
        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)

        # Sample true theta: ||θ*||₂ ≤ param_norm_bound (clip if needed)
        theta_raw = jax.random.normal(subkey1, shape=(self.context_dim,))
        theta_norm = jnp.linalg.norm(theta_raw)
        scale = jnp.minimum(1.0, self.param_norm_bound / (theta_norm + 1e-8))
        self._true_theta = theta_raw * scale

        # Sample context array: (num_steps, num_arms, context_dim)
        contexts_raw = jax.random.uniform(
            subkey2,
            shape=(self.num_steps, self.num_arms, self.context_dim),
            minval=-1.0,
            maxval=1.0,
        )
        self._contexts = contexts_raw * self.context_bound

        self._current_t = 0

    def step(self, t: int, action: int) -> Tuple[jnp.ndarray, float, int]:
        """Execute one step of interaction.

        Args:
            t: Time step (0-indexed)
            action: Selected action/arm index

        Returns:
            contexts_t: Contexts for all arms at time t, shape (num_arms, context_dim)
            reward: Scalar reward from selected action
            best_arm: Index of best arm (maximizes θ* · x_t)
        """
        if self._contexts is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if not (0 <= t < self.num_steps):
            raise ValueError(f"Time step {t} out of range [0, {self.num_steps - 1}]")

        if not (0 <= action < self.num_arms):
            raise ValueError(f"Action {action} out of range [0, {self.num_arms - 1}]")

        contexts_t = self._contexts[t]  # shape (num_arms, context_dim)

        self.key, subkey = jax.random.split(self.key)
        noise = jax.random.normal(subkey, shape=())
        selected_context = contexts_t[action]
        reward = jnp.dot(self._true_theta, selected_context) + noise

        arm_values = jnp.dot(contexts_t, self._true_theta)  # shape (num_arms,)
        best_arm = jnp.argmax(arm_values)

        return contexts_t, float(reward), int(best_arm)

    def get_contexts_at_t(self, t: int) -> jnp.ndarray:
        """Get contexts for all arms at time t without executing action.

        Args:
            t: Time step (0-indexed)

        Returns:
            Contexts for all arms at time t, shape (num_arms, context_dim)
        """
        if self._contexts is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if not (0 <= t < self.num_steps):
            raise ValueError(f"Time step {t} out of range [0, {self.num_steps - 1}]")
        return self._contexts[t]

    def get_true_theta(self) -> jnp.ndarray:
        """Return the true parameter vector θ* with ||θ*||₂ ≤ param_norm_bound.

        Returns:
            true_theta: True parameter vector, shape (context_dim,)
        """
        if self._true_theta is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._true_theta

    def get_context_array(self) -> jnp.ndarray:
        """Return the full context array (num_steps, num_arms, context_dim).

        Returns:
            contexts: Full context array, shape (num_steps, num_arms, context_dim)
        """
        if self._contexts is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._contexts
