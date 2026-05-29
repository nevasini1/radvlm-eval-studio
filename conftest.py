"""Ensure the repo root is importable so `python -m pytest` works without install."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
