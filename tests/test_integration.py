"""End-to-end integration tests for the bandit pipeline."""

import os
import tempfile

import numpy as np
import pytest

from src.experiment import ExperimentRunner
from src.visualization import Visualizer


class TestIntegration:
    """Integration tests for the complete bandit pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_end_to_end_pipeline(self, temp_dir):
        """Test complete pipeline: experiment -> save -> visualize."""
        num_episodes = 2
        context_dim = 3
        num_arms = 5
        num_steps = 10
        seed = 42

        runner = ExperimentRunner(
            num_episodes=num_episodes,
            context_dim=context_dim,
            num_arms=num_arms,
            num_steps=num_steps,
            context_bound=1.0,
            seed=seed,
        )

        result = runner.run()

        assert "regrets" in result
        assert "configs" in result
        assert result["regrets"].shape == (num_episodes, num_steps)

        results_path = os.path.join(temp_dir, "experiment_results.npz")
        np.savez(
            results_path,
            regrets=result["regrets"],
            **result["configs"],
        )
        assert os.path.exists(results_path)

        loaded = np.load(results_path, allow_pickle=True)
        assert loaded["regrets"].shape == (num_episodes, num_steps)

        pdf_path = os.path.join(temp_dir, "regret_plot.pdf")
        Visualizer.plot_regret(result, pdf_path)
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0

    def test_regret_properties(self, temp_dir):
        """Test that regrets have expected properties."""
        runner = ExperimentRunner(
            num_episodes=3,
            context_dim=4,
            num_arms=6,
            num_steps=15,
            seed=123,
        )

        result = runner.run()
        regrets = result["regrets"]

        for episode in range(regrets.shape[0]):
            assert np.all(regrets[episode] >= 0), f"Episode {episode} has negative regrets"
            diffs = np.diff(regrets[episode])
            assert np.all(diffs >= -1e-6), f"Episode {episode} regrets not monotonic"

    def test_small_parameter_experiment(self):
        """Test with minimal parameters."""
        runner = ExperimentRunner(
            num_episodes=2,
            context_dim=2,
            num_arms=3,
            num_steps=5,
            seed=0,
        )

        result = runner.run()

        assert result["regrets"].shape == (2, 5)
        assert result["configs"]["context_dim"] == 2
        assert result["configs"]["num_arms"] == 3

    def test_deterministic_with_seed(self):
        """Test reproducibility with fixed seed."""
        runner1 = ExperimentRunner(
            num_episodes=2,
            context_dim=3,
            num_arms=4,
            num_steps=8,
            seed=999,
        )
        result1 = runner1.run()

        runner2 = ExperimentRunner(
            num_episodes=2,
            context_dim=3,
            num_arms=4,
            num_steps=8,
            seed=999,
        )
        result2 = runner2.run()

        np.testing.assert_array_almost_equal(result1["regrets"], result2["regrets"])

    def test_visualizer_with_different_episode_counts(self, temp_dir):
        """Test visualization with different episode counts."""
        for num_episodes in [1, 3, 5]:
            runner = ExperimentRunner(
                num_episodes=num_episodes,
                context_dim=3,
                num_arms=4,
                num_steps=10,
                seed=42,
            )
            result = runner.run()

            pdf_path = os.path.join(temp_dir, f"plot_{num_episodes}_episodes.pdf")
            Visualizer.plot_regret(result, pdf_path)
            assert os.path.exists(pdf_path)
            assert os.path.getsize(pdf_path) > 0

    def test_configs_preserved_through_pipeline(self):
        """Test that configs are correctly preserved."""
        runner = ExperimentRunner(
            num_episodes=2,
            context_dim=5,
            num_arms=7,
            num_steps=12,
            context_bound=1.5,
            algo_params={
                "lambda_": 2.0,
                "subgaussian_scale": 1.5,
                "norm_bound": 0.8,
                "delta": 0.05,
            },
            seed=555,
        )

        result = runner.run()
        configs = result["configs"]

        assert configs["num_episodes"] == 2
        assert configs["context_dim"] == 5
        assert configs["num_arms"] == 7
        assert configs["num_steps"] == 12
        assert configs["context_bound"] == 1.5
        assert configs["seed"] == 555
        assert configs["algo_params"]["lambda_"] == 2.0
        assert configs["algo_params"]["delta"] == 0.05

    def test_visualizer_with_custom_title(self, temp_dir):
        """Test visualizer with custom title."""
        runner = ExperimentRunner(
            num_episodes=2,
            context_dim=3,
            num_arms=4,
            num_steps=10,
            seed=42,
        )
        result = runner.run()

        pdf_path = os.path.join(temp_dir, "custom_title.pdf")
        Visualizer.plot_regret(result, pdf_path, title="My Custom Title for OFUL")
        assert os.path.exists(pdf_path)
