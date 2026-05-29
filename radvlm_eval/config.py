"""Central configuration and path management for RadVLM Eval Studio.

Paths are computed relative to the repository root so that commands work from
anywhere. Everything is local; nothing here reaches the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repository root = two levels up from this file (radvlm_eval/config.py).
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
INDEX_DIR = OUTPUTS_DIR / "index"
DEMO_DIR = DATA_DIR / "demo"
DEMO_IMAGES_DIR = DEMO_DIR / "images"

# Canonical SQLite database used for the audit log and metadata.
DB_PATH = OUTPUTS_DIR / "radvlm.db"

# Canonical CSV produced by data loaders / importers.
STUDIES_CSV = DATA_DIR / "studies.csv"

# Index artifacts.
EMBEDDINGS_NPY = INDEX_DIR / "embeddings.npy"
INDEX_STUDIES_CSV = INDEX_DIR / "studies.csv"
INDEX_META_JSON = INDEX_DIR / "index_meta.json"

SAFETY_BANNER = "Research demo only. Not for diagnosis or clinical use."


@dataclass
class Settings:
    """Runtime settings; overridable via environment variables."""

    data_dir: Path = DATA_DIR
    outputs_dir: Path = OUTPUTS_DIR
    db_path: Path = DB_PATH
    embedding_backend: str = field(
        default_factory=lambda: os.environ.get("RADVLM_EMBEDDING_BACKEND", "auto")
    )
    top_k: int = field(default_factory=lambda: int(os.environ.get("RADVLM_TOP_K", "5")))

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.outputs_dir, INDEX_DIR, DEMO_DIR, DEMO_IMAGES_DIR):
            p.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()


def ensure_dirs() -> None:
    get_settings().ensure_dirs()
