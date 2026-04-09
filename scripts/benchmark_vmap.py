"""Benchmark script: sequential vs jax.vmap episode parallelization.

Usage:
    uv run python scripts/benchmark_vmap.py
    uv run python scripts/benchmark_vmap.py --config configs/benchmark.yaml
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        args: Parsed arguments with config (str) field
    """
    parser = argparse.ArgumentParser(description="Benchmark sequential vs vmap episode execution.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark.yaml",
        help="Path to benchmark config YAML (default: configs/benchmark.yaml).",
    )
    return parser.parse_args()


def run_sequential(
    num_episodes: int,
    episode_kwargs: dict,
    num_warmup: int,
    num_trials: int,
) -> tuple:
    """Benchmark sequential episode execution.

    Args:
        num_episodes: Number of episodes per trial
        episode_kwargs: Keyword arguments for run_single_episode_sequential
        num_warmup: Number of warmup trials (for JIT compilation)
        num_trials: Number of benchmark trials

    Returns:
        times: List of elapsed times in seconds for each trial
    """
    from src.experiment import run_single_episode_sequential

    def run_trial(seed_base: int) -> float:
        """Run one sequential trial and return wall-clock time.

        Args:
            seed_base: Base seed for this trial

        Returns:
            elapsed: Wall-clock time in seconds
        """
        start = time.perf_counter()
        for i in range(num_episodes):
            result = run_single_episode_sequential(seed=seed_base + i, **episode_kwargs)
        result.block_until_ready()
        return time.perf_counter() - start

    for i in range(num_warmup):
        run_trial(seed_base=i * num_episodes)

    times = [run_trial(seed_base=(num_warmup + i) * num_episodes) for i in range(num_trials)]
    return times


def run_vmap(
    num_episodes: int,
    episode_kwargs: dict,
    num_warmup: int,
    num_trials: int,
) -> tuple:
    """Benchmark vmap episode execution.

    Args:
        num_episodes: Number of episodes per trial
        episode_kwargs: Keyword arguments for run_episodes_vmap (excluding seeds)
        num_warmup: Number of warmup trials (for JIT compilation)
        num_trials: Number of benchmark trials

    Returns:
        times: List of elapsed times in seconds for each trial
    """
    from src.experiment import run_episodes_vmap

    def run_trial(seed_base: int) -> float:
        """Run one vmap trial and return wall-clock time.

        Args:
            seed_base: Base seed for this trial

        Returns:
            elapsed: Wall-clock time in seconds
        """
        seeds = jnp.arange(num_episodes, dtype=jnp.int32) + seed_base
        start = time.perf_counter()
        result = run_episodes_vmap(seeds=seeds, **episode_kwargs)
        result.block_until_ready()
        return time.perf_counter() - start

    for i in range(num_warmup):
        run_trial(seed_base=i * num_episodes)

    times = [run_trial(seed_base=(num_warmup + i) * num_episodes) for i in range(num_trials)]
    return times


def print_stats(label: str, times: list) -> float:
    """Print timing statistics and return mean time.

    Args:
        label: Method label for display
        times: List of elapsed times in seconds

    Returns:
        mean_t: Mean elapsed time in seconds
    """
    mean_t = np.mean(times)
    std_t = np.std(times)
    min_t = np.min(times)
    print(f"[{label}]")
    print(f"  Mean:  {mean_t:.4f}s  Std: {std_t:.4f}s  Min: {min_t:.4f}s")
    print(f"  Times: {[f'{t:.4f}' for t in times]}")
    return mean_t


def main() -> None:
    """Run sequential vs vmap benchmark and print results."""
    args = parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    cfg = OmegaConf.load(args.config)
    exp = OmegaConf.to_container(cfg.experiment, resolve=True)
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    bench = OmegaConf.to_container(cfg.benchmark, resolve=True)

    num_episodes = bench["num_episodes"]
    num_warmup = bench["num_warmup"]
    num_trials = bench["num_trials"]

    episode_kwargs = {
        "context_dim": exp["context_dim"],
        "num_arms": exp["num_arms"],
        "num_steps": exp["num_steps"],
        "context_bound": exp["context_bound"],
        "lambda_": algo["lambda_"],
        "subgaussian_scale": algo["subgaussian_scale"],
        "norm_bound": algo["norm_bound"],
        "delta": algo["delta"],
    }

    print("=" * 55)
    print("Benchmark: Sequential vs vmap episode parallelization")
    print("=" * 55)
    print(
        f"Config: context_dim={exp['context_dim']}, "
        f"num_arms={exp['num_arms']}, "
        f"num_steps={exp['num_steps']}, "
        f"num_episodes={num_episodes}"
    )
    print(f"Warmup: {num_warmup} trial(s), Benchmark: {num_trials} trial(s)")
    print("-" * 55)

    print("\nWarming up and benchmarking sequential...")
    seq_times = run_sequential(num_episodes, episode_kwargs, num_warmup, num_trials)

    print("Warming up and benchmarking vmap...")
    vmap_times = run_vmap(num_episodes, episode_kwargs, num_warmup, num_trials)

    print("\n--- Results ---")
    seq_mean = print_stats("Sequential", seq_times)
    print()
    vmap_mean = print_stats("vmap (jit+vmap)", vmap_times)

    speedup = seq_mean / vmap_mean if vmap_mean > 0 else float("inf")
    print(f"\nSpeedup (sequential / vmap): {speedup:.2f}x")


if __name__ == "__main__":
    main()
