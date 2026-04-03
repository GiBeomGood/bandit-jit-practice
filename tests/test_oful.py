"""Tests for OFUL algorithm."""

import jax.numpy as jnp
import numpy as np
import pytest

from src.algorithms.oful import OFUL


class TestOFUL:
    """Test suite for OFUL algorithm."""

    @pytest.fixture
    def algo(self):
        """Create a simple OFUL algorithm for testing."""
        return OFUL(
            context_dim=5,
            lambda_=1.0,
            subgaussian_scale=1.0,
            norm_bound=1.0,
            context_bound=1.0,
            delta=0.01,
            seed=42,
        )

    def test_initialization(self, algo):
        """Test algorithm initialization."""
        assert algo.context_dim == 5
        assert algo.lambda_ == 1.0
        assert algo.subgaussian_scale == 1.0
        assert algo.norm_bound == 1.0
        assert algo.context_bound == 1.0
        assert algo.delta == 0.01

    def test_reset(self, algo):
        """Test reset functionality."""
        algo.reset()

        # design_matrix_inv should be initialized to (1/λ) I
        expected_inv = (1.0 / algo.lambda_) * jnp.eye(5)
        assert jnp.allclose(algo.design_matrix_inv, expected_inv)

        # sum_reward_context should be initialized to zeros
        assert jnp.allclose(algo.sum_reward_context, jnp.zeros(5))

        # Time step should be reset
        assert algo.t == 0

    def test_select_action_output(self, algo):
        """Test that select_action returns valid action."""
        algo.reset()

        contexts = jnp.ones((10, 5))
        action = algo.select_action(contexts)

        assert 0 <= action < 10
        assert isinstance(action, (int, np.integer))

    def test_update_changes_state(self, algo):
        """Test that update changes algorithm state."""
        algo.reset()

        inv_initial = algo.design_matrix_inv.copy()
        sum_initial = algo.sum_reward_context.copy()

        context = jnp.ones(5)
        algo.update(context, 1.0)

        assert not jnp.allclose(algo.design_matrix_inv, inv_initial)
        assert not jnp.allclose(algo.sum_reward_context, sum_initial)
        assert algo.t == 1

    def test_update_design_matrix_inv_correctly(self, algo):
        """Test that Sherman-Morrison inverse update is correct."""
        algo.reset()

        context = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        algo.update(context, 0.5)

        # Verify: new B^{-1} should equal inv(λI + x x^T)
        initial_b = algo.lambda_ * jnp.eye(5)
        new_b = initial_b + jnp.outer(context, context)
        expected_inv = jnp.linalg.inv(new_b)

        assert jnp.allclose(algo.design_matrix_inv, expected_inv, atol=1e-5)

    def test_update_sum_reward_context_correctly(self, algo):
        """Test that sum_reward_context is updated correctly."""
        algo.reset()

        sum_initial = algo.sum_reward_context.copy()

        context1 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        reward1 = 0.5
        algo.update(context1, reward1)

        expected_sum_1 = sum_initial + reward1 * context1
        assert jnp.allclose(algo.sum_reward_context, expected_sum_1)

        context2 = jnp.array([2.0, 3.0, 4.0, 5.0, 1.0])
        reward2 = 1.5
        algo.update(context2, reward2)

        expected_sum_2 = expected_sum_1 + reward2 * context2
        assert jnp.allclose(algo.sum_reward_context, expected_sum_2)

    def test_ellipsoid_norm_computation(self, algo):
        """Test ellipsoid norm computation."""
        algo.reset()

        # For initial design_matrix_inv = (1/λ)I = I (λ=1), ||x||_{B^{-1}} = ||x||_2
        context = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ellipsoid_norm = algo._compute_ellipsoid_norm(context)

        expected_norm = float(jnp.linalg.norm(context))
        assert np.isclose(ellipsoid_norm, expected_norm)

    def test_confidence_radius_increases_with_time(self, algo):
        """Test that confidence radius increases with time."""
        algo.reset()

        radius_0 = algo._compute_radius(0)
        radius_10 = algo._compute_radius(10)
        radius_50 = algo._compute_radius(50)

        assert radius_0 < radius_10 < radius_50

    def test_confidence_radius_formula(self, algo):
        """Test confidence radius computation against formula."""
        algo.reset()

        t = 10
        t_1indexed = t + 1
        expected_radius = (
            algo.subgaussian_scale
            * jnp.sqrt(
                algo.context_dim
                * jnp.log((1.0 + t_1indexed * (algo.context_bound**2) / algo.lambda_) / algo.delta)
            )
            + jnp.sqrt(algo.lambda_) * algo.norm_bound
        )
        computed_radius = algo._compute_radius(t)

        assert np.isclose(computed_radius, float(expected_radius), rtol=1e-5)

    def test_theta_hat_estimation(self, algo):
        """Test theta_hat parameter estimation."""
        algo.reset()

        # For initial state with no observations, theta_hat should be 0
        theta_hat_0 = algo.get_theta_hat()
        assert jnp.allclose(theta_hat_0, jnp.zeros(5))

        # After one update
        context = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
        reward = 2.0
        algo.update(context, reward)

        theta_hat_1 = algo.get_theta_hat()
        assert theta_hat_1[0] != 0

    def test_determinism_with_seed(self):
        """Test that same seed produces same action sequence."""
        algo1 = OFUL(context_dim=5, subgaussian_scale=1.0, norm_bound=1.0, context_bound=1.0, seed=42)
        algo2 = OFUL(context_dim=5, subgaussian_scale=1.0, norm_bound=1.0, context_bound=1.0, seed=42)

        algo1.reset()
        algo2.reset()

        contexts = jnp.ones((10, 5))
        assert algo1.select_action(contexts) == algo2.select_action(contexts)

    def test_without_reset_raises_error(self):
        """Test that using algorithm without reset raises error."""
        algo = OFUL(context_dim=5)
        with pytest.raises(RuntimeError):
            algo.select_action(jnp.ones((10, 5)))

    def test_end_to_end_sequence(self, algo):
        """Test a simple episode of interaction."""
        algo.reset()

        num_arms = 10
        context_dim = 5

        for t in range(5):
            contexts = jnp.ones((num_arms, context_dim)) * (t + 1)
            action = algo.select_action(contexts)
            assert 0 <= action < num_arms

            context_selected = contexts[action]
            reward = float(jnp.sum(context_selected))
            algo.update(context_selected, reward)

            assert algo.t == t + 1
