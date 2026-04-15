"""Smoke tests for scripts/benchmark_jit_vmap.py."""

import subprocess


def test_benchmark_jit_vmap_runs_without_error() -> None:
    """Script must exit 0 and print a speedup ratio when run with configs/test.yaml."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/benchmark_jit_vmap.py",
            "--config",
            "configs/test.yaml",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Script exited with code {result.returncode}.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "Speedup" in result.stdout, (
        f"'Speedup' not found in stdout.\nstdout: {result.stdout}"
    )
