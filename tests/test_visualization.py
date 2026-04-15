"""Tests for src/visualization.py."""

import os
import tempfile

import numpy as np

from src.visualization import Visualizer


def _make_results(num_episodes: int = 3, num_steps: int = 10) -> dict:
    """Build a minimal results dict compatible with Visualizer.plot_regret."""
    rng = np.random.default_rng(0)
    regrets = rng.uniform(0, 1, size=(num_episodes, num_steps)).cumsum(axis=1)
    return {"regrets": regrets, "configs": {"num_steps": num_steps}}


def test_plot_regret_pdf_path_creates_both_formats() -> None:
    """Saving with a .pdf path must produce both .pdf and .svg files."""
    results = _make_results()
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "test_plot.pdf")
        Visualizer.plot_regret(results, pdf_path)
        svg_path = os.path.join(tmpdir, "test_plot.svg")
        assert os.path.exists(pdf_path), "PDF file was not created"
        assert os.path.exists(svg_path), "SVG file was not created"


def test_plot_regret_non_pdf_path_creates_fallback_svg() -> None:
    """Saving with a non-.pdf path must produce the _plot.svg fallback."""
    results = _make_results()
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "test_plot")
        Visualizer.plot_regret(results, base_path)
        svg_path = base_path + "_plot.svg"
        assert os.path.exists(svg_path), "Fallback SVG file was not created"
