"""Benchmark script for JAX jit performance comparison.

Usage:
    # JIT enabled (default), using configs/benchmark.yaml
    uv run python scripts/benchmark_jit.py

    # JIT disabled
    uv run python scripts/benchmark_jit.py --disable_jit

    # Custom config
    uv run python scripts/benchmark_jit.py --config configs/benchmark.yaml --disable_jit
"""

import argparse
import os
import sys
import time

import jax
import numpy as np
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        args: Parsed arguments with disable_jit (bool) and config (str) fields
    """
    parser = argparse.ArgumentParser(description="Benchmark JAX jit performance for OFUL.")
    parser.add_argument(
        "--disable_jit",
        action="store_true",
        help="Disable JAX JIT compilation for comparison.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark.yaml",
        help="Path to benchmark config YAML (default: configs/benchmark.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    """Run benchmark and print results."""
    args = parse_args()

    # Must configure JAX before importing any JAX-compiled code
    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)
        jit_status = "DISABLED"
    else:
        jit_status = "ENABLED"

    # Add project root to path (for src imports)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.experiment import run_single_episode_sequential  # noqa: E402

    # Load config
    cfg = OmegaConf.load(args.config)
    exp = OmegaConf.to_container(cfg.experiment, resolve=True)
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    bench = OmegaConf.to_container(cfg.benchmark, resolve=True)

    episode_kwargs = dict(
        context_dim=exp["context_dim"],
        num_arms=exp["num_arms"],
        num_steps=exp["num_steps"],
        context_bound=exp["context_bound"],
        lambda_=algo["lambda_"],
        subgaussian_scale=algo["subgaussian_scale"],
        norm_bound=algo["norm_bound"],
        delta=algo["delta"],
    )
    num_warmup = bench["num_warmup"]
    num_trials = bench["num_trials"]

    print(f"JAX jit: {jit_status}")
    print(
        f"Config: context_dim={exp['context_dim']}, "
        f"num_arms={exp['num_arms']}, "
        f"num_steps={exp['num_steps']}"
    )
    print(f"Warmup: {num_warmup} trial(s), Benchmark: {num_trials} trial(s)")
    print("-" * 50)

    def run_trial(seed: int) -> float:
        """Run a single benchmark trial and return elapsed time in seconds.

        Args:
            seed: Random seed for the episode

        Returns:
            elapsed: Wall-clock time in seconds
        """
        start = time.perf_counter()
        run_single_episode_sequential(seed=seed, **episode_kwargs)
        return time.perf_counter() - start

    # Warmup: trigger JIT compilation
    print("Warming up...")
    for i in range(num_warmup):
        run_trial(seed=i)

    # Benchmark
    print("Benchmarking...")
    times = [run_trial(seed=num_warmup + i) for i in range(num_trials)]

    mean_t = np.mean(times)
    std_t = np.std(times)
    min_t = np.min(times)

    print(f"\nResults (JIT {jit_status}):")
    print(f"  Mean:  {mean_t:.4f}s")
    print(f"  Std:   {std_t:.4f}s")
    print(f"  Min:   {min_t:.4f}s")
    print(f"  Times: {[f'{t:.4f}' for t in times]}")


if __name__ == "__main__":
    main()
