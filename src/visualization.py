"""Visualization utilities for bandit experiments."""

from pathlib import Path
from typing import Any, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_regret(results: Dict[str, Any], save_path: str, title: str = None) -> None:
    """Plot cumulative regret with mean and quantiles.

    Parameters
    ----------
    results : dict
        Dict from ExperimentRunner.run() with keys:
        - regrets: numpy array of shape (num_episodes, T)
        - configs: dict with experiment configuration
    save_path : str
        Path to save the image (supported extensions: .png, .pdf, .svg).
    title : str, optional
        Title for the plot. Defaults to a generated title.
    """
    regrets = results["regrets"]
    configs = results.get("configs", {})

    mean_regret = np.mean(regrets, axis=0)
    q_low = np.percentile(regrets, 5, axis=0)
    q_high = np.percentile(regrets, 95, axis=0)

    num_steps = configs.get("num_steps", regrets.shape[1])
    num_episodes = regrets.shape[0]

    matplotlib.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(10, 6))

    t_steps = np.arange(num_steps)
    ax.plot(t_steps, mean_regret, "b-", linewidth=2, label="Mean Regret")
    ax.fill_between(t_steps, q_low, q_high, alpha=0.3, color="blue", label="5%-95% Quantile")

    ax.set_xlabel("Time Step $t$", fontsize=12)
    ax.set_ylabel("Cumulative Regret $R(t)$", fontsize=12)

    if title is None:
        title = f"OFUL Algorithm Regret (Episodes={num_episodes}, Horizon={num_steps})"
    ax.set_title(title, fontsize=13)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=11)

    fig.tight_layout()
    _save_figure(fig, save_path)


def _save_figure(fig: Any, save_path: str) -> None:
    """Save figure to the format indicated by the save_path extension.

    Supports .png, .pdf, and .svg. Raises ValueError for unknown extensions.

    Parameters
    ----------
    fig : Any
        Matplotlib figure to save.
    save_path : str
        Destination path including the desired file extension.

    Raises
    ------
    ValueError
        If the file extension is not one of .png, .pdf, or .svg.
    """
    path = Path(save_path)
    suffix = path.suffix.lower()
    fmt_map = {".png": "png", ".pdf": "pdf", ".svg": "svg"}
    if suffix not in fmt_map:
        raise ValueError(
            f"Unrecognized file extension '{suffix}'. "
            f"Supported formats: {list(fmt_map.keys())}"
        )
    fig.savefig(save_path, format=fmt_map[suffix], dpi=150, bbox_inches="tight")
    plt.close(fig)
