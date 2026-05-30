#!/usr/bin/env python
"""Generate evaluator-derived SFT + DPO training pairs from the demo dataset.

Usage:
    python scripts/make_training_pairs.py [--n-flawed 2] [--seed 13] [--with-similar]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radvlm_eval import config  # noqa: E402
from radvlm_eval.data.load_studies import load_demo_dataset  # noqa: E402
from radvlm_eval.training.format_mlx_dataset import write_all  # noqa: E402
from radvlm_eval.training.pair_builder import build_pairs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report-repair training pairs.")
    parser.add_argument("--n-flawed", type=int, default=2, help="Flawed drafts per study.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--with-similar", action="store_true",
                        help="Include similar-case context (requires a built index).")
    args = parser.parse_args()

    config.ensure_dirs()
    studies = load_demo_dataset()

    index = None
    if args.with_similar:
        try:
            from radvlm_eval.retrieval.index import load_index

            index = load_index()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not load index for similar-case context: {exc}")

    data = build_pairs(studies, n_flawed_per_study=args.n_flawed, seed=args.seed, index=index)
    info = write_all(data)
    stats = data.stats()

    print("Generated evaluator-derived training pairs (synthetic data).")
    print(f"  SFT total: {stats['sft_total']}  "
          f"(train={stats['sft_train']}, valid={stats['sft_valid']}, test={stats['sft_test']})")
    print(f"  DPO total: {stats['dpo_total']}  "
          f"(train={stats['dpo_train']}, valid={stats['dpo_valid']}, test={stats['dpo_test']})")
    print(f"  SFT dir: {info['sft_dir']}")
    print(f"  DPO dir: {info['dpo_dir']}")
    print("Research demo only. Not for diagnosis or clinical use.")


if __name__ == "__main__":
    main()
