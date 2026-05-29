#!/usr/bin/env python
"""Build the retrieval index (embeddings + aligned study metadata).

Usage:
    python scripts/build_index.py --dataset demo --embedding-backend fallback
    python scripts/build_index.py --dataset openi --data-root /path/to/openi
    python scripts/build_index.py --dataset demo --limit 20 --embedding-backend auto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radvlm_eval import config  # noqa: E402
from radvlm_eval.retrieval.index import build_index  # noqa: E402
from radvlm_eval.schemas import Study  # noqa: E402


def _load_dataset(dataset: str, data_root: str | None, limit: int | None) -> list[Study]:
    if dataset == "demo":
        from radvlm_eval.data.load_studies import load_demo_dataset

        studies = load_demo_dataset()
    elif dataset == "openi":
        from radvlm_eval.data.import_openi import import_openi

        if not data_root:
            raise SystemExit("--data-root is required for --dataset openi")
        studies = import_openi(data_root, limit=limit)
    elif dataset == "mimic":
        from radvlm_eval.data.import_mimic import import_mimic

        if not data_root:
            raise SystemExit("--data-root is required for --dataset mimic")
        studies = import_mimic(data_root, limit=limit)
    elif dataset == "chexpert":
        from radvlm_eval.data.import_chexpert import import_chexpert

        if not data_root:
            raise SystemExit("--data-root is required for --dataset chexpert")
        studies = import_chexpert(data_root, limit=limit)
    else:
        raise SystemExit(f"Unknown dataset: {dataset}")

    if limit:
        studies = studies[:limit]
    return studies


def main() -> None:
    parser = argparse.ArgumentParser(description="Build similar-case retrieval index.")
    parser.add_argument(
        "--dataset", choices=["demo", "openi", "mimic", "chexpert"], default="demo"
    )
    parser.add_argument("--data-root", default=None, help="Root folder for real datasets.")
    parser.add_argument("--limit", type=int, default=None, help="Max studies to index.")
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "fallback", "biomedclip"],
        default="auto",
    )
    args = parser.parse_args()

    config.ensure_dirs()
    studies = _load_dataset(args.dataset, args.data_root, args.limit)
    if not studies:
        raise SystemExit("No studies loaded; nothing to index.")

    index = build_index(studies, backend=args.embedding_backend)
    print(f"Indexed {len(index.studies)} studies.")
    print(f"  Backend:   {index.backend}")
    print(f"  Embedding: {index.embeddings.shape}")
    print(f"  Saved to:  {config.INDEX_DIR}")
    print("Research demo only. Not for diagnosis or clinical use.")


if __name__ == "__main__":
    main()
