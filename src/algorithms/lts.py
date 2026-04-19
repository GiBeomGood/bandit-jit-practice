"""Linear Thompson Sampling (LTS) algorithm.

Maintains a Gaussian posterior over the unknown reward parameter mu.
At each step, samples mu_tilde from N(mu_hat, v^2 * B^{-1}) and plays
the arm with the highest predicted reward under that sample.

B is the regularized design matrix (initialized to I_d). The inverse
B^{-1} is maintained via Sherman-Morrison rank-1 updates to avoid
explicit matrix inversion.
"""

from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp


class LtsCarry(NamedTuple):
    """Carry state for the LTS algorithm within a scan loop.

    Attributes
    ----------
    design_matrix_inv : jnp.ndarray
        Inverse of regularized design matrix B^{-1}, shape (context_dim, context_dim).
    cumulative_reward_context : jnp.ndarray
        Cumulative sum f = sum of reward * context, shape (context_dim,).
    mu_hat : jnp.ndarray
        Ridge regression estimate B^{-1} f, shape (context_dim,).
    prng_key : jax.Array
        PRNG key for posterior sampling, updated each step.
    """

    design_matrix_inv: jnp.ndarray
    cumulative_reward_context: jnp.ndarray
    mu_hat: jnp.ndarray
    prng_key: jax.Array


class LinearThompsonSampling:
    """Linear Thompson Sampling bandit algorithm.

    Samples from the Gaussian posterior N(mu_hat, v^2 * B^{-1}) at each step
    and plays the arm with the highest predicted reward under the sample.
    The inverse B^{-1} is maintained via Sherman-Morrison rank-1 updates.
    """

    @staticmethod
    def update_state(
        design_matrix_inv: jnp.ndarray,
        cumulative_reward_context: jnp.ndarray,
        context: jnp.ndarray,
        reward: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update B^{-1} via Sherman-Morrison, f, and mu_hat after observing a reward.

        Parameters
        ----------
        design_matrix_inv : jnp.ndarray
            Current inverse design matrix B^{-1}, shape (context_dim, context_dim).
        cumulative_reward_context : jnp.ndarray
            Current cumulative sum f, shape (context_dim,).
        context : jnp.ndarray
            Context vector of selected arm, shape (context_dim,).
        reward : jnp.ndarray
            Observed reward (scalar).

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            Updated (design_matrix_inv, cumulative_reward_context, mu_hat).
        """
        b_inv_x = design_matrix_inv @ context
        denom = 1.0 + context @ b_inv_x
        dm_inv_new = design_matrix_inv - jnp.outer(b_inv_x, b_inv_x) / denom
        f_new = cumulative_reward_context + reward * context
        mu_hat_new = dm_inv_new @ f_new
        return dm_inv_new, f_new, mu_hat_new

    @staticmethod
    def _step(
        carry: LtsCarry,
        contexts_t: jnp.ndarray,
        true_theta: jnp.ndarray,
        noise_t: jnp.ndarray,
        v: float,
    ) -> tuple:
        """Sample mu_tilde from posterior, select arm, observe reward, update carry.

        Parameters
        ----------
        carry : LtsCarry
            Current algorithm state.
        contexts_t : jnp.ndarray
            Context vectors for all arms at this step, shape (num_arms, context_dim).
        true_theta : jnp.ndarray
            True parameter vector, shape (context_dim,).
        noise_t : jnp.ndarray
            Observation noise for this step (scalar).
        v : float
            Posterior sampling scale: std = v * B^{-1/2}.

        Returns
        -------
        tuple
            (new_carry, instant_regret).
        """
        key, subkey = jax.random.split(carry.prng_key)
        covariance = (v ** 2) * carry.design_matrix_inv
        mu_tilde = jax.random.multivariate_normal(subkey, mean=carry.mu_hat, cov=covariance)
        action = jnp.argmax(contexts_t @ mu_tilde)

        arm_values = contexts_t @ true_theta
        best_arm = jnp.argmax(arm_values)
        reward = arm_values[action] + noise_t
        instant_regret = arm_values[best_arm] - arm_values[action]

        dm_inv_new, f_new, mu_hat_new = LinearThompsonSampling.update_state(
            carry.design_matrix_inv,
            carry.cumulative_reward_context,
            contexts_t[action],
            reward,
        )
        new_carry = LtsCarry(
            design_matrix_inv=dm_inv_new,
            cumulative_reward_context=f_new,
            mu_hat=mu_hat_new,
            prng_key=key,
        )
        return new_carry, instant_regret

    @staticmethod
    def make_init_carry(context_dim: int, prng_key: jax.Array, **kwargs) -> LtsCarry:
        """Create the initial carry for the LTS scan loop.

        Parameters
        ----------
        context_dim : int
            Feature dimension d.
        prng_key : jax.Array
            PRNG key for posterior sampling.
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        LtsCarry
            Initial carry with B^{-1} = I_d, f = 0, mu_hat = 0.
        """
        return LtsCarry(
            design_matrix_inv=jnp.eye(context_dim),
            cumulative_reward_context=jnp.zeros(context_dim),
            mu_hat=jnp.zeros(context_dim),
            prng_key=prng_key,
        )

    @staticmethod
    def make_step_fn(context_dim: int, true_theta: jnp.ndarray, **kwargs) -> Callable:
        """Build a scan-compatible step closure for LTS.

        Computes v = R * sqrt(24 / epsilon * d * ln(1/delta)) and closes over it.

        Parameters
        ----------
        context_dim : int
            Feature dimension d.
        true_theta : jnp.ndarray
            True parameter vector, shape (context_dim,).
        **kwargs
            subgaussian_scale : float
                Sub-Gaussian variance proxy (R).
            epsilon : float
                Approximation accuracy parameter.
            delta : float
                Failure probability.

        Returns
        -------
        Callable
            step_fn(carry, x) compatible with jax.lax.scan.
        """
        v = kwargs["subgaussian_scale"] * jnp.sqrt(
            (24.0 / kwargs["epsilon"]) * context_dim * jnp.log(1.0 / kwargs["delta"])
        )

        def step_fn(carry: LtsCarry, x: tuple) -> tuple:
            """Execute one LTS step: sample theta, select action, observe reward, update state."""
            contexts_t, noise_t, _t_idx = x
            return LinearThompsonSampling._step(carry, contexts_t, true_theta, noise_t, v)

        return step_fn
