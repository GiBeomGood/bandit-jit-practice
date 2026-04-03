"""Pytest configuration for bandit-practice tests."""

import sys
from pathlib import Path

# Add src directory to Python path so that `from src.xxx import ...` works
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
