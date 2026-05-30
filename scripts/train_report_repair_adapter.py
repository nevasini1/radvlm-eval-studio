#!/usr/bin/env python
"""Train (or dry-run) the LoRA report-repair adapter with MLX.

Usage:
    python scripts/train_report_repair_adapter.py --dry-run
    python scripts/train_report_repair_adapter.py --iters 300
    python scripts/train_report_repair_adapter.py --model mlx-community/Qwen3-1.7B-4bit

If mlx-lm is not installed (e.g. non-Apple-Silicon / CI), this prints the exact
command and install instructions and exits 0 — it never crashes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radvlm_eval.training.train_adapter import (  # noqa: E402
    DEFAULT_ADAPTER,
    DEFAULT_DATA,
    DEFAULT_MODEL,
    TrainConfig,
    status_message,
    train,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for report repair (MLX).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--adapter-path", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        model=args.model,
        data=Path(args.data),
        adapter_path=Path(args.adapter_path),
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    print(f"[mlx] {status_message()}")
    print(f"[command] {cfg.command_str()}")
    result = train(cfg, dry_run=args.dry_run)
    print(result["message"])
    print("Research demo only. Not for diagnosis or clinical use.")


if __name__ == "__main__":
    main()
