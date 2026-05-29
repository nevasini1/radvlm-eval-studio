"""Build and load the retrieval index (embeddings + aligned study metadata)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from radvlm_eval import config
from radvlm_eval.data.load_studies import save_studies_csv
from radvlm_eval.retrieval.embedder import build_embeddings
from radvlm_eval.schemas import Study


@dataclass
class Index:
    embeddings: np.ndarray
    studies: List[Study]
    backend: str

    @property
    def study_ids(self) -> List[str]:
        return [s.study_id for s in self.studies]


def build_index(
    studies: List[Study],
    backend: str = "auto",
    index_dir: Optional[Path] = None,
) -> Index:
    index_dir = Path(index_dir) if index_dir else config.INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)

    embeddings, backend_used = build_embeddings(studies, backend=backend)

    np.save(index_dir / "embeddings.npy", embeddings)
    save_studies_csv(studies, index_dir / "studies.csv")
    meta = {
        "backend": backend_used,
        "n_studies": len(studies),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "study_ids": [s.study_id for s in studies],
    }
    (index_dir / "index_meta.json").write_text(json.dumps(meta, indent=2))
    return Index(embeddings=embeddings, studies=studies, backend=backend_used)


def index_exists(index_dir: Optional[Path] = None) -> bool:
    index_dir = Path(index_dir) if index_dir else config.INDEX_DIR
    return (index_dir / "embeddings.npy").exists() and (index_dir / "studies.csv").exists()


def load_index(index_dir: Optional[Path] = None) -> Index:
    from radvlm_eval.data.load_studies import load_studies_csv

    index_dir = Path(index_dir) if index_dir else config.INDEX_DIR
    emb_path = index_dir / "embeddings.npy"
    studies_path = index_dir / "studies.csv"
    if not emb_path.exists() or not studies_path.exists():
        raise FileNotFoundError(
            f"No index found in {index_dir}. Run "
            "`python scripts/build_index.py --dataset demo --embedding-backend fallback`."
        )
    embeddings = np.load(emb_path)
    studies = load_studies_csv(studies_path)
    backend = "fallback"
    meta_path = index_dir / "index_meta.json"
    if meta_path.exists():
        backend = json.loads(meta_path.read_text()).get("backend", "fallback")
    return Index(embeddings=embeddings, studies=studies, backend=backend)
