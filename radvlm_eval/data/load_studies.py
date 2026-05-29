"""Load and persist canonical Study collections via CSV."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from radvlm_eval import config
from radvlm_eval.schemas import Study


def save_studies_csv(studies: List[Study], path: Optional[Path] = None) -> Path:
    path = Path(path) if path else config.STUDIES_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([s.to_row() for s in studies])
    df.to_csv(path, index=False)
    return path


def load_studies_csv(path: Optional[Path] = None) -> List[Study]:
    path = Path(path) if path else config.STUDIES_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"No studies CSV at {path}. Run `python scripts/make_demo_data.py` first."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return [Study.from_row(row) for row in df.to_dict(orient="records")]


def load_demo_dataset(data_dir: Optional[Path] = None) -> List[Study]:
    """Load the demo studies CSV, generating it on the fly if absent."""
    csv_path = config.STUDIES_CSV
    if csv_path.exists():
        return load_studies_csv(csv_path)

    # Generate on demand so the app never hard-fails on a fresh checkout.
    from radvlm_eval.data.demo_data import generate_demo_dataset

    studies = generate_demo_dataset(data_dir=data_dir)
    save_studies_csv(studies, csv_path)
    return studies
