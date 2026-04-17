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
    parser = argparse.ArgumentParser(description="Run bandit experiment and save regret graph.")
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


def main() -> None:
    """Run the bandit experiment and save the regret graph."""
    args = parse_args()
    config_path = Path(args.config)

    validate_config_path(config_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(config_path)
    fmt = OmegaConf.select(cfg, "output.format", default=DEFAULT_OUTPUT_FORMAT)

    runner = ExperimentRunner.from_yaml(str(config_path))
    results = runner.run()

    output_path = OUTPUT_DIR / f"{config_path.stem}_regret.{fmt}"
    title = f"OFUL Regret [{config_path.stem}] (Episodes={runner.num_episodes}, Horizon={runner.num_steps})"
    plot_regret(results, save_path=str(output_path), title=title)

    print(f"Regret graph saved to: {output_path}")


if __name__ == "__main__":
    main()
