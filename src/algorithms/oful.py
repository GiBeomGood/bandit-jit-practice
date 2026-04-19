"""OFUL (Optimism in the Face of Uncertainty for Linear bandits) algorithm."""

from typing import Callable, Optional, Tuple

import jax.numpy as jnp

from src.algorithms.base import Algorithm


class OFUL(Algorithm):
    """OFUL (Optimism in the Face of Uncertainty for Linear Bandits).

    Maintains a confidence ellipsoid around the true parameter estimate
    and selects actions optimistically within the confidence set.
    Uses Sherman-Morrison formula for O(d²) inverse design matrix updates.
    """

    def __init__(
        self,
        context_dim: int,
        lambda_: float = 1.0,
        subgaussian_scale: float = 1.0,
        norm_bound: float = 1.0,
        context_bound: float = 1.0,
        delta: float = 0.01,
        seed: Optional[int] = None,
    ):
        """Initialize OFUL algorithm.

        Parameters
        ----------
        context_dim : int
            Feature dimension.
        lambda_ : float
            Ridge regularization parameter (B_t = λI + Σ x_s x_s^T).
        subgaussian_scale : float
            Sub-Gaussian variance proxy (R in OFUL paper).
        norm_bound : float
            Parameter norm bound (S: ||θ*|| ≤ norm_bound).
        context_bound : float
            Context norm bound (L: ||x|| ≤ context_bound).
        delta : float
            Failure probability (confidence parameter).
        seed : int, optional
            Random seed for reproducibility.
        """
        super().__init__(context_dim, seed)
        self.lambda_ = lambda_
        self.subgaussian_scale = subgaussian_scale
        self.norm_bound = norm_bound
        self.context_bound = context_bound
        self.delta = delta

        self.design_matrix_inv = None
        self.sum_reward_context = None
        self.t = 0

    @staticmethod
    def compute_confidence_radius(t: int, context_dim: int, **kwargs) -> jnp.ndarray:
        """Compute β_t = R*sqrt(d*log((1+t*L²/λ)/δ)) + sqrt(λ)*S.

        Uses 1-indexed t to avoid log(0).

        Parameters
        ----------
        t : int
            Current time step (0-indexed).
        context_dim : int
            Feature dimension d.
        **kwargs
            lambda_ : float
                Ridge regularization parameter.
            subgaussian_scale : float
                Sub-Gaussian variance proxy (R in OFUL paper).
            norm_bound : float
                Parameter norm bound (S: ||θ*|| ≤ norm_bound).
            context_bound : float
                Context norm bound (L: ||x|| ≤ context_bound).
            delta : float
                Failure probability.

        Returns
        -------
        jnp.ndarray
            Confidence radius (scalar).
        """
        t_1indexed = t + 1
        log_arg = (
            1.0 + t_1indexed * (kwargs["context_bound"] ** 2) / kwargs["lambda_"]
        ) / kwargs["delta"]
        return (
            kwargs["subgaussian_scale"] * jnp.sqrt(context_dim * jnp.log(log_arg))
            + jnp.sqrt(kwargs["lambda_"]) * kwargs["norm_bound"]
        )

    @staticmethod
    def compute_ucb_values(
        contexts: jnp.ndarray,
        theta_hat: jnp.ndarray,
        design_matrix_inv: jnp.ndarray,
        radius_t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute mean + radius * ||x||_{B^{-1}} for each arm.

        Parameters
        ----------
        contexts : jnp.ndarray
            Context vectors for all arms, shape (num_arms, context_dim).
        theta_hat : jnp.ndarray
            Parameter estimate, shape (context_dim,).
        design_matrix_inv : jnp.ndarray
            Inverse design matrix B_t^{-1}, shape (context_dim, context_dim).
        radius_t : jnp.ndarray
            Confidence radius (scalar).

        Returns
        -------
        jnp.ndarray
            UCB values for all arms, shape (num_arms,).
        """
        mean_terms = contexts @ theta_hat
        b_inv_x = contexts @ design_matrix_inv
        ellipsoid_norms_sq = jnp.sum(b_inv_x * contexts, axis=-1)
        ellipsoid_norms = jnp.sqrt(jnp.maximum(ellipsoid_norms_sq, 0.0))
        return mean_terms + radius_t * ellipsoid_norms

    @staticmethod
    def update_state(
        design_matrix_inv: jnp.ndarray,
        sum_reward_context: jnp.ndarray,
        context: jnp.ndarray,
        reward: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply Sherman-Morrison rank-1 update and accumulate the reward-context sum.

        Parameters
        ----------
        design_matrix_inv : jnp.ndarray
            Current inverse design matrix B_t^{-1}, shape (context_dim, context_dim).
        sum_reward_context : jnp.ndarray
            Current cumulative reward-context sum, shape (context_dim,).
        context : jnp.ndarray
            Context vector of selected arm, shape (context_dim,).
        reward : jnp.ndarray
            Observed reward (scalar).

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            Updated (design_matrix_inv, sum_reward_context).
        """
        b_inv_x = design_matrix_inv @ context
        denom = 1.0 + context @ b_inv_x
        design_matrix_inv_new = design_matrix_inv - jnp.outer(b_inv_x, b_inv_x) / denom
        sum_reward_context_new = sum_reward_context + reward * context
        return design_matrix_inv_new, sum_reward_context_new

    @staticmethod
    def make_init_carry(context_dim: int, **kwargs) -> tuple:
        """Create the initial carry for the OFUL scan loop.

        Parameters
        ----------
        context_dim : int
            Feature dimension d.
        **kwargs
            lambda_ : float
                Ridge regularization parameter; B_0 = λI so B_0^{-1} = (1/λ)I.

        Returns
        -------
        tuple
            (design_matrix_inv, sum_reward_context, cumulative_regret) where
            design_matrix_inv has shape (d, d), sum_reward_context has shape (d,),
            and cumulative_regret is a scalar 0.0.
        """
        init_design_matrix_inv = (1.0 / kwargs["lambda_"]) * jnp.eye(context_dim)
        init_sum_reward_context = jnp.zeros(context_dim)
        return (init_design_matrix_inv, init_sum_reward_context, jnp.array(0.0))

    @staticmethod
    def make_step_fn(context_dim: int, true_theta: jnp.ndarray, **kwargs) -> Callable:
        """Create the per-step closure for the OFUL scan loop.

        Parameters
        ----------
        context_dim : int
            Feature dimension.
        true_theta : jnp.ndarray
            True parameter vector, shape (context_dim,).
        **kwargs
            lambda_ : float
                Ridge regularization parameter.
            subgaussian_scale : float
                Sub-Gaussian variance proxy (R in OFUL paper).
            norm_bound : float
                Parameter norm bound (S: ||θ*|| ≤ norm_bound).
            context_bound : float
                Context norm bound (L: ||x|| ≤ context_bound).
            delta : float
                Failure probability.

        Returns
        -------
        Callable
            step_fn(carry, x) compatible with jax.lax.scan.
        """
        def step_fn(carry: tuple, x: tuple) -> tuple:
            """Execute one OFUL step: select action via UCB, observe reward, update state."""
            design_matrix_inv, sum_reward_context, cumulative_regret = carry
            contexts_t, noise_t, t_idx = x

            theta_hat = design_matrix_inv @ sum_reward_context
            radius_t = OFUL.compute_confidence_radius(t_idx, context_dim, **kwargs)
            action = jnp.argmax(
                OFUL.compute_ucb_values(contexts_t, theta_hat, design_matrix_inv, radius_t)
            )

            arm_values = contexts_t @ true_theta
            best_arm = jnp.argmax(arm_values)
            reward = arm_values[action] + noise_t
            regret_t = arm_values[best_arm] - arm_values[action]
            new_cumulative_regret = cumulative_regret + regret_t

            new_dm_inv, new_src = OFUL.update_state(
                design_matrix_inv, sum_reward_context, contexts_t[action], reward
            )
            return (new_dm_inv, new_src, new_cumulative_regret), new_cumulative_regret

        return step_fn

    def _check_initialized(self) -> None:
        """Raise RuntimeError if reset() has not been called yet."""
        if self.design_matrix_inv is None:
            raise RuntimeError("Algorithm not initialized. Call reset() first.")

    def reset(self) -> None:
        """Reset algorithm state for a new episode."""
        self.design_matrix_inv = (1.0 / self.lambda_) * jnp.eye(self.context_dim)
        self.sum_reward_context = jnp.zeros(self.context_dim)
        self.t = 0

    def select_action(self, contexts: jnp.ndarray) -> int:
        """Select an action using the OFUL strategy.

        Parameters
        ----------
        contexts : jnp.ndarray
            Context vectors for all arms, shape (num_arms, context_dim).

        Returns
        -------
        int
            Selected arm index in [0, num_arms-1].
        """
        self._check_initialized()
        theta_hat = self.design_matrix_inv @ self.sum_reward_context
        algo_kwargs = {
            "lambda_": self.lambda_,
            "subgaussian_scale": self.subgaussian_scale,
            "norm_bound": self.norm_bound,
            "context_bound": self.context_bound,
            "delta": self.delta,
        }
        radius_t = OFUL.compute_confidence_radius(self.t, self.context_dim, **algo_kwargs)
        ucb_values = OFUL.compute_ucb_values(
            contexts, theta_hat, self.design_matrix_inv, radius_t
        )
        return int(jnp.argmax(ucb_values))

    def update(self, context: jnp.ndarray, reward: float) -> None:
        """Update algorithm state with observed feedback.

        Parameters
        ----------
        context : jnp.ndarray
            Context vector of selected arm, shape (context_dim,).
        reward : float
            Observed reward (scalar).
        """
        self.design_matrix_inv, self.sum_reward_context = OFUL.update_state(
            self.design_matrix_inv,
            self.sum_reward_context,
            context,
            jnp.array(reward),
        )
        self.t += 1

    def _compute_ellipsoid_norm(self, context: jnp.ndarray) -> float:
        """Compute ||x||_{B_t^{-1}} = sqrt(x^T B_t^{-1} x).

        Parameters
        ----------
        context : jnp.ndarray
            Context vector, shape (context_dim,).

        Returns
        -------
        float
            Scalar value ||x||_{B_t^{-1}}.
        """
        b_inv_x = self.design_matrix_inv @ context
        norm_sq = jnp.maximum(context @ b_inv_x, 0.0)
        return float(jnp.sqrt(norm_sq))
