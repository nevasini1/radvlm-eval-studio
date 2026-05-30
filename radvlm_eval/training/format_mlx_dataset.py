"""Export training pairs to MLX-LM-compatible JSONL.

* SFT  -> chat format: one JSON object per line with a "messages" list.
* DPO  -> preference format: {"prompt", "chosen", "rejected"} per line.

mlx-lm's LoRA trainer reads the chat ("messages") format directly. The DPO files
are exported as DPO-ready JSONL even if no DPO trainer is installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from radvlm_eval import config
from radvlm_eval.training.pair_builder import TrainingData

SFT_SUBDIR = "rad_repair_sft"
DPO_SUBDIR = "rad_repair_dpo"
SPLITS = ["train", "valid", "test"]


def _write_jsonl(rows: List[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def write_sft(data: TrainingData, out_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in SPLITS:
        rows = [{"messages": ex["messages"]} for ex in data.sft[split]]
        counts[split] = _write_jsonl(rows, out_dir / f"{split}.jsonl")
    return counts


def write_dpo(data: TrainingData, out_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in SPLITS:
        rows = [
            {"prompt": ex["prompt"], "chosen": ex["chosen"], "rejected": ex["rejected"]}
            for ex in data.dpo[split]
        ]
        counts[split] = _write_jsonl(rows, out_dir / f"{split}.jsonl")
    return counts


def write_all(data: TrainingData, base_dir: Path | None = None) -> Dict[str, object]:
    base = Path(base_dir) if base_dir else (config.OUTPUTS_DIR / "training")
    sft_dir = base / SFT_SUBDIR
    dpo_dir = base / DPO_SUBDIR
    sft_counts = write_sft(data, sft_dir)
    dpo_counts = write_dpo(data, dpo_dir)
    return {
        "sft_dir": str(sft_dir),
        "dpo_dir": str(dpo_dir),
        "sft_counts": sft_counts,
        "dpo_counts": dpo_counts,
    }
