"""Abstract base class for bandit algorithms."""

from abc import ABC, abstractmethod

import jax.numpy as jnp


class Algorithm(ABC):
    """Abstract base class for bandit algorithms.

    All algorithm implementations (OFUL, LinearThompson, etc.) should inherit
    from this class and implement the required methods.
    """

    def __init__(self, context_dim: int, seed: int = None):
        """Initialize algorithm.

        Args:
            context_dim: Feature dimension
            seed: Random seed for reproducibility
        """
        self.context_dim = context_dim
        self.seed = seed

    @abstractmethod
    def reset(self) -> None:
        """Reset algorithm state for a new episode.

        This should initialize internal parameters (e.g., design matrix,
        parameter estimates) to their initial values.
        """
        pass

    @abstractmethod
    def select_action(self, contexts: jnp.ndarray) -> int:
        """Select an action given current contexts.

        Args:
            contexts: Context vectors for all arms, shape (num_arms, context_dim)

        Returns:
            action: Selected arm index in [0, num_arms-1]
        """
        pass

    @abstractmethod
    def update(self, context: jnp.ndarray, reward: float) -> None:
        """Update algorithm state with observed feedback.

        Args:
            context: Context vector of selected arm, shape (context_dim,)
            reward: Observed reward (scalar)
        """
        pass
