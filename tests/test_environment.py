"""Tests for ContextualLinearBandit environment."""

import jax.numpy as jnp
import numpy as np
import pytest

from src.environments.contextual_linear import ContextualLinearBandit


class TestContextualLinearBandit:
    """Test suite for ContextualLinearBandit environment."""

    @pytest.fixture
    def env(self):
        """Create a simple environment for testing."""
        return ContextualLinearBandit(context_dim=5, num_arms=10, num_steps=20, context_bound=1.0, seed=42)

    def test_initialization(self, env):
        """Test environment initialization."""
        assert env.context_dim == 5
        assert env.num_arms == 10
        assert env.num_steps == 20
        assert env.context_bound == 1.0

    def test_reset(self, env):
        """Test reset functionality."""
        env.reset()

        # Check that true theta is initialized
        theta = env.get_true_theta()
        assert theta.shape == (5,)

        # Check that theta norm is within param_norm_bound
        theta_norm = float(jnp.linalg.norm(theta))
        assert theta_norm <= env.param_norm_bound + 1e-6

        # Check context array shape
        contexts = env.get_context_array()
        assert contexts.shape == (20, 10, 5)  # (num_steps, num_arms, context_dim)

    def test_contexts_within_bounds(self, env):
        """Test that contexts are within [-L, L]^d."""
        env.reset()
        contexts = env.get_context_array()

        assert jnp.all(contexts >= -1.0) and jnp.all(contexts <= 1.0)

    def test_step_output_shapes(self, env):
        """Test that step returns correct output shapes."""
        env.reset()

        contexts_all, reward, best_arm = env.step(t=0, action=0)

        assert contexts_all.shape == (10, 5)  # (num_arms, context_dim)
        assert isinstance(reward, float) or np.isscalar(reward)
        assert isinstance(best_arm, (int, np.integer))
        assert 0 <= best_arm < 10

    def test_best_arm_consistency(self, env):
        """Test that best arm is correctly identified."""
        env.reset()

        theta = env.get_true_theta()
        contexts = env.get_context_array()

        arm_values = jnp.dot(contexts[0], theta)  # (num_arms,)
        expected_best_arm = int(jnp.argmax(arm_values))

        _, _, returned_best_arm = env.step(t=0, action=0)
        assert returned_best_arm == expected_best_arm

    def test_reward_structure(self, env):
        """Test that reward has correct structure (deterministic + noise)."""
        env.reset()

        theta = env.get_true_theta()
        contexts = env.get_context_array()

        selected_action = 0
        expected_reward_base = float(jnp.dot(theta, contexts[0, selected_action]))

        env.reset()  # Reset to get same theta and contexts
        rewards = []
        for _ in range(10):
            _, reward, _ = env.step(t=0, action=selected_action)
            rewards.append(reward)

        rewards_mean = np.mean(rewards)
        assert np.isclose(rewards_mean, expected_reward_base, atol=0.5)
        assert np.std(rewards) > 0.0

    def test_determinism_with_seed(self):
        """Test that same seed produces same results."""
        env1 = ContextualLinearBandit(context_dim=5, num_arms=10, num_steps=20, seed=42)
        env2 = ContextualLinearBandit(context_dim=5, num_arms=10, num_steps=20, seed=42)

        env1.reset()
        env2.reset()

        assert jnp.allclose(env1.get_true_theta(), env2.get_true_theta())

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        env1 = ContextualLinearBandit(context_dim=5, num_arms=10, num_steps=20, seed=42)
        env2 = ContextualLinearBandit(context_dim=5, num_arms=10, num_steps=20, seed=43)

        env1.reset()
        env2.reset()

        assert not jnp.allclose(env1.get_true_theta(), env2.get_true_theta())

    def test_step_without_reset_raises_error(self):
        """Test that calling step without reset raises error."""
        env = ContextualLinearBandit(context_dim=5, num_arms=10, num_steps=20)
        with pytest.raises(RuntimeError):
            env.step(0, 0)

    def test_invalid_action_raises_error(self, env):
        """Test that invalid action raises error."""
        env.reset()
        with pytest.raises(ValueError):
            env.step(0, 10)  # action >= num_arms

    def test_invalid_time_step_raises_error(self, env):
        """Test that invalid time step raises error."""
        env.reset()
        with pytest.raises(ValueError):
            env.step(20, 0)  # t >= num_steps
