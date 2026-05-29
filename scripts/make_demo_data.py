#!/usr/bin/env python
"""Generate the synthetic demo dataset (images + studies.csv).

Usage:
    python scripts/make_demo_data.py [--n 36] [--seed 7]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radvlm_eval import config  # noqa: E402
from radvlm_eval.data.demo_data import generate_demo_dataset  # noqa: E402
from radvlm_eval.data.load_studies import save_studies_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic demo CXR studies.")
    parser.add_argument("--n", type=int, default=36, help="Number of studies (30-50).")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args()

    config.ensure_dirs()
    studies = generate_demo_dataset(n_studies=args.n, seed=args.seed)
    csv_path = save_studies_csv(studies, config.STUDIES_CSV)

    print(f"Generated {len(studies)} synthetic studies.")
    print(f"  Images: {config.DEMO_IMAGES_DIR}")
    print(f"  CSV:    {csv_path}")
    print("Research demo only. Not for diagnosis or clinical use.")


if __name__ == "__main__":
    main()
