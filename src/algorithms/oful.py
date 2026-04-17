"""OFUL (Optimism in the Face of Uncertainty for Linear bandits) algorithm."""

from typing import Optional, Tuple

import jax.numpy as jnp

from src.algorithms.base import Algorithm


def compute_confidence_radius(
    t: int,
    context_dim: int,
    lambda_: float,
    subgaussian_scale: float,
    norm_bound: float,
    context_bound: float,
    delta: float,
) -> jnp.ndarray:
    """Compute confidence radius at time t.

    β_t = subgaussian_scale * sqrt(context_dim * log((1 + t * context_bound² / λ) / δ))
          + sqrt(λ) * norm_bound

    Args:
        t: Current time step (0-indexed)
        context_dim: Feature dimension
        lambda_: Ridge regularization parameter
        subgaussian_scale: Sub-Gaussian variance proxy (R in OFUL paper)
        norm_bound: Parameter norm bound (S in OFUL paper, ||θ*|| ≤ norm_bound)
        context_bound: Context norm bound (L in OFUL paper, ||x|| ≤ context_bound)
        delta: Failure probability

    Returns:
        radius: Confidence radius (scalar)
    """
    # Use 1-indexed time to avoid log(0)
    t_1indexed = t + 1
    log_arg = (1.0 + t_1indexed * (context_bound**2) / lambda_) / delta
    radius = subgaussian_scale * jnp.sqrt(context_dim * jnp.log(log_arg)) + jnp.sqrt(lambda_) * norm_bound
    return radius


def compute_theta_hat(design_matrix_inv: jnp.ndarray, sum_reward_context: jnp.ndarray) -> jnp.ndarray:
    """Compute parameter estimate θ̂ = B_t^{-1} * sum_reward_context.

    Args:
        design_matrix_inv: Inverse design matrix B_t^{-1}, shape (context_dim, context_dim)
        sum_reward_context: Cumulative sum of reward * context, shape (context_dim,)

    Returns:
        theta_hat: Parameter estimate, shape (context_dim,)
    """
    return design_matrix_inv @ sum_reward_context


def update_design_matrix_inv(design_matrix_inv: jnp.ndarray, context: jnp.ndarray) -> jnp.ndarray:
    """Update inverse design matrix using Sherman-Morrison formula.

    B_{t+1}^{-1} = B_t^{-1} - (B_t^{-1} x x^T B_t^{-1}) / (1 + x^T B_t^{-1} x)

    This avoids explicit matrix inversion (O(d³)) in favor of O(d²) rank-1 update.

    Args:
        design_matrix_inv: Current inverse design matrix B_t^{-1}, shape (context_dim, context_dim)
        context: Context vector x_t, shape (context_dim,)

    Returns:
        design_matrix_inv_new: Updated inverse design matrix, shape (context_dim, context_dim)
    """
    b_inv_x = design_matrix_inv @ context  # shape (context_dim,)
    denom = 1.0 + context @ b_inv_x  # scalar, always > 0 since B^{-1} is PSD
    return design_matrix_inv - jnp.outer(b_inv_x, b_inv_x) / denom


def update_sum_reward_context(
    sum_reward_context: jnp.ndarray, context: jnp.ndarray, reward: jnp.ndarray
) -> jnp.ndarray:
    """Update cumulative reward-context sum.

    Args:
        sum_reward_context: Current sum, shape (context_dim,)
        context: Context vector, shape (context_dim,)
        reward: Observed reward (scalar)

    Returns:
        sum_reward_context_new: Updated sum, shape (context_dim,)
    """
    return sum_reward_context + reward * context


def compute_ucb_values(
    contexts: jnp.ndarray,
    theta_hat: jnp.ndarray,
    design_matrix_inv: jnp.ndarray,
    radius_t: jnp.ndarray,
) -> jnp.ndarray:
    """Compute UCB values for all arms.

    Args:
        contexts: Context vectors for all arms, shape (num_arms, context_dim)
        theta_hat: Parameter estimate, shape (context_dim,)
        design_matrix_inv: Inverse design matrix B_t^{-1}, shape (context_dim, context_dim)
        radius_t: Confidence radius (scalar)

    Returns:
        ucb_values: UCB values for all arms, shape (num_arms,)
    """
    mean_terms = contexts @ theta_hat  # shape (num_arms,)

    # Vectorized ellipsoid norm: ||x_a||_{B^{-1}} = sqrt(x_a^T B^{-1} x_a)
    b_inv_x = contexts @ design_matrix_inv  # shape (num_arms, context_dim), symmetric B^{-1}
    ellipsoid_norms_sq = jnp.sum(b_inv_x * contexts, axis=-1)  # shape (num_arms,)
    ellipsoid_norms = jnp.sqrt(jnp.maximum(ellipsoid_norms_sq, 0.0))

    return mean_terms + radius_t * ellipsoid_norms


def oful_update(
    design_matrix_inv: jnp.ndarray,
    sum_reward_context: jnp.ndarray,
    context: jnp.ndarray,
    reward: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Update OFUL state with observed feedback.

    Applies Sherman-Morrison rank-1 update to the inverse design matrix and
    accumulates the reward-context sum. JIT is applied at the call site.

    Args:
        design_matrix_inv: Current inverse design matrix B_t^{-1}, shape (context_dim, context_dim)
        sum_reward_context: Current cumulative reward-context sum, shape (context_dim,)
        context: Context vector of selected arm, shape (context_dim,)
        reward: Observed reward (scalar)

    Returns:
        design_matrix_inv_new: Updated inverse design matrix, shape (context_dim, context_dim)
        sum_reward_context_new: Updated cumulative sum, shape (context_dim,)
    """
    design_matrix_inv_new = update_design_matrix_inv(design_matrix_inv, context)
    sum_reward_context_new = update_sum_reward_context(sum_reward_context, context, reward)
    return design_matrix_inv_new, sum_reward_context_new


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

        Args:
            context_dim: Feature dimension
            lambda_: Ridge regularization parameter (B_t = λI + Σ x_s x_s^T)
            subgaussian_scale: Sub-Gaussian variance proxy (R in OFUL paper)
            norm_bound: Parameter norm bound (S: ||θ*|| ≤ norm_bound)
            context_bound: Context norm bound (L: ||x|| ≤ context_bound)
            delta: Failure probability (confidence parameter)
            seed: Random seed for reproducibility
        """
        super().__init__(context_dim, seed)
        self.lambda_ = lambda_
        self.subgaussian_scale = subgaussian_scale
        self.norm_bound = norm_bound
        self.context_bound = context_bound
        self.delta = delta

        # State initialized in reset()
        self.design_matrix_inv = None  # B_t^{-1}, shape (context_dim, context_dim)
        self.sum_reward_context = None  # Σ r_s x_s, shape (context_dim,)
        self.t = 0

    def _check_initialized(self) -> None:
        if self.design_matrix_inv is None:
            raise RuntimeError("Algorithm not initialized. Call reset() first.")

    def reset(self) -> None:
        """Reset algorithm state for a new episode."""
        # B_0 = λI → B_0^{-1} = (1/λ) I
        self.design_matrix_inv = (1.0 / self.lambda_) * jnp.eye(self.context_dim)
        self.sum_reward_context = jnp.zeros(self.context_dim)
        self.t = 0

    def select_action(self, contexts: jnp.ndarray) -> int:
        """Select an action using the OFUL strategy.

        Args:
            contexts: Context vectors for all arms, shape (num_arms, context_dim)

        Returns:
            action: Selected arm index in [0, num_arms-1]
        """
        self._check_initialized()

        theta_hat = compute_theta_hat(self.design_matrix_inv, self.sum_reward_context)
        radius_t = compute_confidence_radius(
            self.t, self.context_dim, self.lambda_,
            self.subgaussian_scale, self.norm_bound, self.context_bound, self.delta,
        )
        ucb_values = compute_ucb_values(contexts, theta_hat, self.design_matrix_inv, radius_t)
        return int(jnp.argmax(ucb_values))

    def update(self, context: jnp.ndarray, reward: float) -> None:
        """Update algorithm state with observed feedback.

        Args:
            context: Context vector of selected arm, shape (context_dim,)
            reward: Observed reward (scalar)
        """
        self.design_matrix_inv, self.sum_reward_context = oful_update(
            self.design_matrix_inv,
            self.sum_reward_context,
            context,
            jnp.array(reward),
        )
        self.t += 1

    def get_theta_hat(self) -> jnp.ndarray:
        """Return current parameter estimate θ̂ = B_t^{-1} Σ r_s x_s.

        Returns:
            theta_hat: Current parameter estimate, shape (context_dim,)
        """
        self._check_initialized()
        return compute_theta_hat(self.design_matrix_inv, self.sum_reward_context)

    def _compute_radius(self, t: int) -> float:
        """Compute confidence radius at time t.

        Args:
            t: Current time step (0-indexed)

        Returns:
            radius: Confidence radius (scalar)
        """
        return float(
            compute_confidence_radius(
                t,
                self.context_dim,
                self.lambda_,
                self.subgaussian_scale,
                self.norm_bound,
                self.context_bound,
                self.delta,
            )
        )

    def _compute_ellipsoid_norm(self, context: jnp.ndarray) -> float:
        """Compute ||x||_{B_t^{-1}} = sqrt(x^T B_t^{-1} x).

        Args:
            context: Context vector, shape (context_dim,)

        Returns:
            ellipsoid_norm: Scalar value ||x||_{B_t^{-1}}
        """
        b_inv_x = self.design_matrix_inv @ context
        norm_sq = jnp.maximum(context @ b_inv_x, 0.0)
        return float(jnp.sqrt(norm_sq))
