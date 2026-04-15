"""Benchmark: wall-clock time for JIT+vmap ON vs OFF.

Usage:
    uv run python scripts/benchmark_jit_vmap.py
    uv run python scripts/benchmark_jit_vmap.py --config configs/benchmark.yaml
"""

import time

_SCRIPT_START = time.perf_counter()  # Must be first executable line

import argparse
import os
import sys

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf


def _build_episode_kwargs(exp: dict, algo: dict) -> dict:
    """Build keyword arguments shared by both episode runner functions.

    Args:
        exp: Experiment config dict with context_dim, num_arms, num_steps, context_bound
        algo: Algorithm config dict with lambda_, subgaussian_scale, norm_bound, delta

    Returns:
        episode_kwargs: Dict of kwargs accepted by run_episodes_vmap and
            run_single_episode_sequential (excluding seed/seeds)
    """
    return {
        "context_dim": exp["context_dim"],
        "num_arms": exp["num_arms"],
        "num_steps": exp["num_steps"],
        "context_bound": exp["context_bound"],
        "lambda_": algo["lambda_"],
        "subgaussian_scale": algo["subgaussian_scale"],
        "norm_bound": algo["norm_bound"],
        "delta": algo["delta"],
    }


def run_jit_vmap(num_episodes: int, episode_kwargs: dict) -> float:
    """Run all episodes with JIT+vmap. Returns compute time in seconds.

    Args:
        num_episodes: Number of episodes to run in parallel
        episode_kwargs: Keyword arguments for the episode runner (excluding seeds)

    Returns:
        elapsed: Compute time in seconds (does not include script startup)
    """
    from src.experiment import run_episodes_vmap

    seeds = jnp.arange(num_episodes, dtype=jnp.int32)
    t0 = time.perf_counter()
    result = run_episodes_vmap(seeds=seeds, **episode_kwargs)
    result.block_until_ready()
    return time.perf_counter() - t0


def run_no_jit(num_episodes: int, episode_kwargs: dict) -> float:
    """Run all episodes sequentially with JIT disabled. Returns compute time in seconds.

    Args:
        num_episodes: Number of episodes to run sequentially
        episode_kwargs: Keyword arguments for the episode runner (excluding seed)

    Returns:
        elapsed: Compute time in seconds (does not include script startup)
    """
    from src.experiment import run_single_episode_sequential

    t0 = time.perf_counter()
    with jax.disable_jit():
        for i in range(num_episodes):
            result = run_single_episode_sequential(seed=i, **episode_kwargs)
        result.block_until_ready()
    return time.perf_counter() - t0


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        args: Parsed arguments with config (str) field
    """
    parser = argparse.ArgumentParser(
        description="Benchmark JIT+vmap ON vs OFF wall-clock time."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark.yaml",
        help="Path to config YAML (default: configs/benchmark.yaml).",
    )
    return parser.parse_args()


def _load_config(config_path: str) -> tuple:
    """Load and parse config file into component dicts.

    Args:
        config_path: Path to YAML config file

    Returns:
        (exp, algo, num_episodes): Experiment dict, algo dict, and episode count
    """
    cfg = OmegaConf.load(config_path)
    exp = OmegaConf.to_container(cfg.experiment, resolve=True)
    algo = OmegaConf.to_container(cfg.algo, resolve=True)

    if hasattr(cfg, "benchmark") and hasattr(cfg.benchmark, "num_episodes"):
        num_episodes = cfg.benchmark.num_episodes
    else:
        num_episodes = exp["num_episodes"]

    return exp, algo, num_episodes


def _print_header(exp: dict, num_episodes: int) -> None:
    """Print benchmark header and config summary.

    Args:
        exp: Experiment config dict
        num_episodes: Number of episodes to be run
    """
    sep = "=" * 60
    dash = "-" * 60
    print(sep)
    print("Benchmark: JIT+vmap ON vs OFF — wall-clock from invocation")
    print(sep)
    print(
        f"Config: context_dim={exp['context_dim']}, "
        f"num_arms={exp['num_arms']}, "
        f"num_steps={exp['num_steps']}, "
        f"num_episodes={num_episodes}"
    )
    print(dash)


def _print_results(t_jit: float, t_total_jit: float, t_nojit: float, t_total_nojit: float) -> None:
    """Print timing results table and speedup ratio.

    Args:
        t_jit: JIT+vmap compute time in seconds
        t_total_jit: JIT+vmap total time from invocation in seconds
        t_nojit: No-JIT compute time in seconds
        t_total_nojit: No-JIT total time from invocation in seconds
    """
    sep = "=" * 60
    speedup = t_nojit / t_jit if t_jit > 0 else float("inf")

    print("\n[JIT+vmap ON]")
    print(f"  Compute time:          {t_jit:.2f}s")
    print(f"  Time from invocation:  {t_total_jit:.2f}s  (includes Python startup + JAX import)")
    print()
    print("[JIT+vmap OFF (jax.disable_jit)]")
    print(f"  Compute time:         {t_nojit:.2f}s")
    print(f"  Time from invocation: {t_total_nojit:.2f}s")
    print()
    print(f"Speedup (no-JIT / JIT+vmap): {speedup:.2f}x")
    print(sep)


def main() -> None:
    """Run JIT+vmap ON and OFF benchmarks and print timing comparison."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    args = _parse_args()
    exp, algo, num_episodes = _load_config(args.config)
    episode_kwargs = _build_episode_kwargs(exp, algo)

    _print_header(exp, num_episodes)

    t_jit = run_jit_vmap(num_episodes, episode_kwargs)
    t_total_jit = time.perf_counter() - _SCRIPT_START

    t_nojit = run_no_jit(num_episodes, episode_kwargs)
    t_total_nojit = time.perf_counter() - _SCRIPT_START

    _print_results(t_jit, t_total_jit, t_nojit, t_total_nojit)


if __name__ == "__main__":
    main()
