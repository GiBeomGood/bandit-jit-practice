"""Benchmark script for JAX jit performance comparison.

Usage:
    # JIT enabled (default)
    uv run python scripts/benchmark_jit.py

    # JIT disabled
    DISABLE_JIT=1 uv run python scripts/benchmark_jit.py
"""

import os
import sys
import time

import jax
import numpy as np

# Check DISABLE_JIT environment variable before importing any JAX-compiled code
if os.environ.get("DISABLE_JIT", "0") == "1":
    jax.config.update("jax_disable_jit", True)
    jit_status = "DISABLED"
else:
    jit_status = "ENABLED"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment import run_single_episode_sequential  # noqa: E402

# Benchmark configuration
CONTEXT_DIM = 10
NUM_ARMS = 20
NUM_STEPS = 500
NUM_WARMUP = 2
NUM_TRIALS = 5

EPISODE_KWARGS = dict(
    context_dim=CONTEXT_DIM,
    num_arms=NUM_ARMS,
    num_steps=NUM_STEPS,
    context_bound=1.0,
    lambda_=1.0,
    subgaussian_scale=1.0,
    norm_bound=1.0,
    delta=0.01,
)


def run_trial(seed: int) -> float:
    """Run a single benchmark trial and return elapsed time in seconds.

    Args:
        seed: Random seed for the episode

    Returns:
        elapsed: Wall-clock time in seconds
    """
    start = time.perf_counter()
    run_single_episode_sequential(seed=seed, **EPISODE_KWARGS)
    return time.perf_counter() - start


def main() -> None:
    """Run benchmark and print results."""
    print(f"JAX jit: {jit_status}")
    print(f"Config: context_dim={CONTEXT_DIM}, num_arms={NUM_ARMS}, num_steps={NUM_STEPS}")
    print(f"Warmup: {NUM_WARMUP} trial(s), Benchmark: {NUM_TRIALS} trial(s)")
    print("-" * 50)

    # Warmup: trigger JIT compilation
    print("Warming up...")
    for i in range(NUM_WARMUP):
        run_trial(seed=i)

    # Benchmark
    print("Benchmarking...")
    times = [run_trial(seed=NUM_WARMUP + i) for i in range(NUM_TRIALS)]

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
