"""Wrapper around `mlx_lm.lora` for LoRA/QLoRA SFT on Apple Silicon.

The actual training runs only if `mlx-lm` is installed (Apple Silicon + MLX). On
any other machine this prints the install instructions and the exact command,
and never crashes — so the rest of the repo (and CI) works without MLX.

Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
"""

from __future__ import annotations

import importlib.util
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from radvlm_eval import config

DEFAULT_MODEL = "mlx-community/Qwen3-1.7B-4bit"
DEFAULT_DATA = config.OUTPUTS_DIR / "training" / "rad_repair_sft"
DEFAULT_ADAPTER = config.OUTPUTS_DIR / "adapters" / "rad-repair-sft"

INSTALL_HINT = (
    "MLX LoRA training requires Apple Silicon + mlx-lm. Install with:\n"
    '    pip install "mlx-lm[train]"\n'
    "Then re-run this command (drop --dry-run to train)."
)


@dataclass
class TrainConfig:
    model: str = DEFAULT_MODEL
    data: Path = DEFAULT_DATA
    adapter_path: Path = DEFAULT_ADAPTER
    iters: int = 300
    batch_size: int = 1
    learning_rate: float = 1e-5

    def command(self) -> List[str]:
        return [
            "mlx_lm.lora",
            "--model", self.model,
            "--train",
            "--data", str(self.data),
            "--iters", str(self.iters),
            "--batch-size", str(self.batch_size),
            "--learning-rate", str(self.learning_rate),
            "--adapter-path", str(self.adapter_path),
        ]

    def command_str(self) -> str:
        return " ".join(shlex.quote(p) for p in self.command())


def mlx_available() -> bool:
    return importlib.util.find_spec("mlx_lm") is not None


def status_message() -> str:
    return "mlx-lm detected — LoRA training available." if mlx_available() else INSTALL_HINT


def train(cfg: TrainConfig, dry_run: bool = False) -> dict:
    """Run (or dry-run) LoRA SFT. Returns a status dict; never raises on missing MLX."""
    data_ok = Path(cfg.data).exists()
    result = {
        "command": cfg.command_str(),
        "mlx_available": mlx_available(),
        "data_exists": data_ok,
        "dry_run": dry_run,
        "ran": False,
        "message": "",
    }

    if not data_ok:
        result["message"] = (
            f"Training data not found at {cfg.data}. Run "
            "`python scripts/make_training_pairs.py` first."
        )
        return result

    if dry_run:
        result["message"] = "Dry run — command shown but not executed.\n" + cfg.command_str()
        return result

    if not mlx_available():
        result["message"] = INSTALL_HINT
        return result

    Path(cfg.adapter_path).mkdir(parents=True, exist_ok=True)
    try:  # pragma: no cover - only exercised with mlx-lm installed
        proc = subprocess.run(cfg.command(), check=False)
        result["ran"] = proc.returncode == 0
        result["return_code"] = proc.returncode
        result["message"] = (
            f"Training finished (rc={proc.returncode}); adapter at {cfg.adapter_path}"
            if proc.returncode == 0
            else f"Training command exited with rc={proc.returncode}."
        )
    except FileNotFoundError:
        result["message"] = INSTALL_HINT
    return result
