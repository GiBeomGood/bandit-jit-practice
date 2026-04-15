"""Main entry point for bandit algorithm experiments.

Usage
-----
    uv run python main.py --config configs/experiment.yaml
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from src.experiment import ExperimentRunner
from src.visualization import plot_regret

DEFAULT_OUTPUT_FORMAT = "png"

OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``config`` attribute.
    """
    parser = argparse.ArgumentParser(
        description="Run bandit experiment and save regret graph."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to YAML config file (default: configs/experiment.yaml).",
    )
    return parser.parse_args()


def validate_config_path(config_path: Path) -> None:
    """Raise an error if the config file does not exist or is not a YAML file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML config file.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Config file must be a YAML file (.yaml/.yml): {config_path}")


def read_output_format(config_path: Path) -> str:
    """Read the output.format key from a YAML config, falling back to png.

    Parameters
    ----------
    config_path : Path
        Path to the YAML config file.

    Returns
    -------
    str
        File format string (e.g. ``"png"``, ``"pdf"``, ``"svg"``).
    """
    cfg = OmegaConf.load(config_path)
    return OmegaConf.select(cfg, "output.format", default=DEFAULT_OUTPUT_FORMAT)


def build_output_path(config_path: Path, output_dir: Path, fmt: str) -> Path:
    """Derive the output image path from the config filename and format.

    Parameters
    ----------
    config_path : Path
        Path to the YAML config file.
    output_dir : Path
        Directory where the output image will be saved.
    fmt : str
        File extension without a leading dot (e.g. ``"png"``, ``"pdf"``).

    Returns
    -------
    Path
        Full path for the output image file.
    """
    config_name = config_path.stem
    return output_dir / f"{config_name}_regret.{fmt}"


def build_plot_title(config_path: Path, runner: ExperimentRunner) -> str:
    """Build a descriptive plot title that references the config name.

    Parameters
    ----------
    config_path : Path
        Path to the YAML config file.
    runner : ExperimentRunner
        Configured experiment runner (provides episode and step counts).

    Returns
    -------
    str
        Plot title string.
    """
    config_name = config_path.stem
    return (
        f"OFUL Regret [{config_name}] "
        f"(Episodes={runner.num_episodes}, Horizon={runner.num_steps})"
    )


def save_regret_plot(
    results: dict,
    output_path: Path,
    title: str,
) -> None:
    """Save the regret plot to the given path as a PNG image.

    Parameters
    ----------
    results : dict
        Experiment results dict from ExperimentRunner.run().
    output_path : Path
        Destination path for the PNG image.
    title : str
        Plot title to display in the image.
    """
    plot_regret(results, save_path=str(output_path), title=title)


def main() -> None:
    """Run the bandit experiment and save the regret graph."""
    args = parse_args()
    config_path = Path(args.config)

    validate_config_path(config_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner.from_yaml(str(config_path))
    results = runner.run()

    fmt = read_output_format(config_path)
    output_path = build_output_path(config_path, OUTPUT_DIR, fmt)
    title = build_plot_title(config_path, runner)
    save_regret_plot(results, output_path, title)

    print(f"Regret graph saved to: {output_path}")


if __name__ == "__main__":
    main()
