"""Visualization utilities for bandit experiments."""

from typing import Any, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Visualizer for experiment results."""

    @staticmethod
    def plot_regret(results: Dict[str, Any], save_path: str, title: str = None) -> None:
        """Plot cumulative regret with mean and quantiles.

        Args:
            results: Dict from ExperimentRunner.run() with keys:
                - regrets: numpy array of shape (num_episodes, T)
                - configs: dict with experiment configuration
            save_path: Path to save PDF (should end with .pdf)
            title: Optional title for the plot
        """
        regrets = results["regrets"]
        configs = results.get("configs", {})

        # Compute statistics
        mean_regret = np.mean(regrets, axis=0)
        q_low = np.percentile(regrets, 5, axis=0)
        q_high = np.percentile(regrets, 95, axis=0)

        # Get parameters for labels
        num_steps = configs.get("num_steps", regrets.shape[1])
        num_episodes = regrets.shape[0]

        # Create figure with academic style
        matplotlib.rcParams["font.family"] = "serif"
        fig, ax = plt.subplots(figsize=(10, 6))

        # Time steps
        t_steps = np.arange(num_steps)

        # Plot mean regret
        ax.plot(t_steps, mean_regret, "b-", linewidth=2, label="Mean Regret")

        # Plot quantile range as shaded area
        ax.fill_between(t_steps, q_low, q_high, alpha=0.3, color="blue", label="5%-95% Quantile")

        # Labels and formatting
        ax.set_xlabel("Time Step $t$", fontsize=12)
        ax.set_ylabel("Cumulative Regret $R(t)$", fontsize=12)

        if title is None:
            title = f"OFUL Algorithm Regret (Episodes={num_episodes}, Horizon={num_steps})"
        ax.set_title(title, fontsize=13)

        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=11)

        # Save as PDF
        fig.tight_layout()
        fig.savefig(save_path, format="pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)
