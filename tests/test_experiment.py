"""Tests for experiment runner."""

import os
import tempfile

import numpy as np
import pytest

from src.experiment import ExperimentRunner


class TestExperimentRunner:
    """Test suite for ExperimentRunner."""

    @pytest.fixture
    def runner(self):
        """Create a simple experiment runner for testing."""
        return ExperimentRunner(
            num_episodes=2,
            context_dim=3,
            num_arms=5,
            num_steps=10,
            context_bound=1.0,
            algo_params={
                "lambda_": 1.0,
                "subgaussian_scale": 1.0,
                "norm_bound": 1.0,
                "delta": 0.01,
            },
            seed=42,
        )

    def test_initialization(self, runner):
        """Test runner initialization."""
        assert runner.num_episodes == 2
        assert runner.context_dim == 3
        assert runner.num_arms == 5
        assert runner.num_steps == 10
        assert runner.context_bound == 1.0

    def test_configs_populated(self, runner):
        """Test that configs are properly stored."""
        assert "num_episodes" in runner.configs
        assert "context_dim" in runner.configs
        assert "num_arms" in runner.configs
        assert "num_steps" in runner.configs
        assert "algo_params" in runner.configs

    def test_run_returns_correct_structure(self, runner):
        """Test that run() returns dict with correct keys."""
        result = runner.run()

        assert isinstance(result, dict)
        assert "regrets" in result
        assert "configs" in result
        assert "metadata" in result

    def test_regrets_shape(self, runner):
        """Test that regrets array has correct shape."""
        result = runner.run()

        regrets = result["regrets"]
        assert regrets.shape == (2, 10)  # (num_episodes, num_steps)

    def test_regrets_are_nonnegative_cumulative(self, runner):
        """Test that regrets are non-decreasing (cumulative)."""
        result = runner.run()

        regrets = result["regrets"]
        for episode in range(regrets.shape[0]):
            for t in range(1, regrets.shape[1]):
                assert regrets[episode, t] >= regrets[episode, t - 1]

    def test_determinism_with_seed(self):
        """Test that same seed produces same regrets."""
        runner1 = ExperimentRunner(num_episodes=2, context_dim=3, num_arms=5, num_steps=10, seed=42)
        runner2 = ExperimentRunner(num_episodes=2, context_dim=3, num_arms=5, num_steps=10, seed=42)

        result1 = runner1.run()
        result2 = runner2.run()

        np.testing.assert_allclose(result1["regrets"], result2["regrets"])

    def test_different_seeds_produce_different_regrets(self):
        """Test that different seeds produce different regrets."""
        runner1 = ExperimentRunner(num_episodes=2, context_dim=3, num_arms=5, num_steps=10, seed=42)
        runner2 = ExperimentRunner(num_episodes=2, context_dim=3, num_arms=5, num_steps=10, seed=43)

        result1 = runner1.run()
        result2 = runner2.run()

        assert not np.allclose(result1["regrets"], result2["regrets"])

    def test_save_and_load_results(self):
        """Test saving and loading experiment results."""
        runner = ExperimentRunner(num_episodes=2, context_dim=3, num_arms=5, num_steps=10, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_results.npz")

            runner.save_results(save_path)
            assert os.path.exists(save_path)

            loaded = ExperimentRunner.load_results(save_path)
            assert "regrets" in loaded
            assert "configs" in loaded
            assert loaded["regrets"].shape == (2, 10)

    def test_save_load_preserves_regrets(self):
        """Test that save/load preserves regret values exactly."""
        runner = ExperimentRunner(num_episodes=2, context_dim=3, num_arms=5, num_steps=10, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_results.npz")

            result = runner.run()
            original_regrets = result["regrets"].copy()

            runner.save_results(save_path)

            loaded = ExperimentRunner.load_results(save_path)
            np.testing.assert_allclose(original_regrets, loaded["regrets"])

    def test_configs_preserved_in_save_load(self):
        """Test that configs are preserved through save/load."""
        runner = ExperimentRunner(
            num_episodes=2,
            context_dim=3,
            num_arms=5,
            num_steps=10,
            context_bound=1.5,
            algo_params={"lambda_": 0.5, "subgaussian_scale": 2.0},
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_results.npz")

            runner.save_results(save_path)
            loaded = ExperimentRunner.load_results(save_path)

            assert loaded["configs"]["context_dim"] == 3
            assert loaded["configs"]["num_arms"] == 5
            assert loaded["configs"]["num_steps"] == 10

    def test_default_algo_params(self):
        """Test that default algorithm parameters are set correctly."""
        runner = ExperimentRunner(num_episodes=1, context_dim=2, num_arms=3, num_steps=5)

        assert runner.algo_params["lambda_"] == 1.0
        assert runner.algo_params["subgaussian_scale"] == 1.0
        assert runner.algo_params["norm_bound"] == 1.0
        assert runner.algo_params["delta"] == 0.01

    def test_custom_algo_params(self):
        """Test that custom algorithm parameters are used."""
        custom_params = {
            "lambda_": 0.5,
            "subgaussian_scale": 2.0,
            "norm_bound": 1.5,
            "delta": 0.05,
        }

        runner = ExperimentRunner(num_episodes=1, context_dim=2, num_arms=3, num_steps=5, algo_params=custom_params)

        assert runner.algo_params["lambda_"] == 0.5
        assert runner.algo_params["subgaussian_scale"] == 2.0
        assert runner.algo_params["norm_bound"] == 1.5
        assert runner.algo_params["delta"] == 0.05

    def test_multiple_episodes_independence(self):
        """Test that episodes are independent (different results even with same seed range)."""
        runner = ExperimentRunner(num_episodes=5, context_dim=3, num_arms=5, num_steps=10, seed=42)

        result = runner.run()
        regrets = result["regrets"]

        for i in range(regrets.shape[0] - 1):
            assert not np.allclose(regrets[i], regrets[i + 1])


class TestExperimentRunnerFromYaml:
    """Tests for ExperimentRunner.from_yaml class method."""

    def test_from_yaml_returns_correct_shape(self):
        """ExperimentRunner.from_yaml produces regrets of shape (num_episodes, num_steps)."""
        runner = ExperimentRunner.from_yaml("configs/test.yaml")
        result = runner.run()
        regrets = result["regrets"]
        assert regrets.shape == (runner.num_episodes, runner.num_steps)

    def test_from_yaml_matches_manual_construction(self):
        """from_yaml produces same regrets as manually constructed runner with same params."""
        runner_yaml = ExperimentRunner.from_yaml("configs/test.yaml")
        runner_manual = ExperimentRunner(
            num_episodes=2,
            context_dim=3,
            num_arms=5,
            num_steps=10,
            context_bound=1.0,
            algo_params={
                "lambda_": 1.0,
                "subgaussian_scale": 1.0,
                "norm_bound": 1.0,
                "delta": 0.01,
            },
            seed=42,
            use_vmap=False,
        )
        result_yaml = runner_yaml.run()
        result_manual = runner_manual.run()
        np.testing.assert_allclose(result_yaml["regrets"], result_manual["regrets"])

    def test_from_yaml_configs_populated(self):
        """from_yaml populates runner configs correctly."""
        runner = ExperimentRunner.from_yaml("configs/test.yaml")
        assert runner.num_episodes == 2
        assert runner.context_dim == 3
        assert runner.num_arms == 5
        assert runner.num_steps == 10
