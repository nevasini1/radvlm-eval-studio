#!/usr/bin/env python
"""Evaluate report repair (before vs after) on held-out synthetic cases.

Usage:
    python scripts/evaluate_report_repair_adapter.py [--limit N] [--adapter-path PATH]

If no trained adapter is present, results are produced via the deterministic
template fallback and clearly labelled as such (numbers are real, not faked).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radvlm_eval import config  # noqa: E402
from radvlm_eval.training.evaluate_adapter import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate report-repair before/after.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--adapter-path", default=None)
    args = parser.parse_args()

    config.ensure_dirs()
    adapter = Path(args.adapter_path) if args.adapter_path else None
    results = evaluate(adapter_path=adapter, limit=args.limit, save=True)

    print(f"Report-repair evaluation over {results['n_cases']} synthetic cases.")
    print(f"  Method: {results['label']}")
    agg = results["aggregate"]
    for k, v in agg.items():
        print(f"  {k:24s} before={v['before']:>3} after={v['after']:>3} (Δ {v['delta']:+d})")
    print(f"  Label F1: {results['label_f1']['before']} -> {results['label_f1']['after']}")
    print(f"  New errors introduced: {results['new_errors_introduced']}")
    print(f"  Saved: {results.get('saved')}")
    print("Research demo only. Not for diagnosis or clinical use.")


if __name__ == "__main__":
    main()
