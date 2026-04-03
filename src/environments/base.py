"""Abstract base class for bandit environments."""

from abc import ABC, abstractmethod
from typing import Tuple

import jax.numpy as jnp


class Environment(ABC):
    """Abstract base class for bandit environments.

    All bandit environment implementations should inherit from this class
    and implement the required methods.
    """

    def __init__(self, context_dim: int, num_arms: int, num_steps: int):
        """Initialize environment.

        Args:
            context_dim: Feature dimension
            num_arms: Number of arms
            num_steps: Episode length (number of time steps)
        """
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.num_steps = num_steps

    @abstractmethod
    def reset(self) -> None:
        """Reset environment for a new episode.

        This should initialize/resample the true parameter, context array,
        and any other necessary state.
        """
        pass

    @abstractmethod
    def step(self, t: int, action: int) -> Tuple[jnp.ndarray, float, int]:
        """Execute one step of interaction.

        Args:
            t: Time step (0-indexed, should be in [0, num_steps-1])
            action: Selected action/arm index (should be in [0, num_arms-1])

        Returns:
            context: Context vector at time t for all arms, shape (num_arms, context_dim)
            reward: Scalar reward from selected action
            best_arm: Index of best arm at time t (maximizes θ* · x_t)
        """
        pass
