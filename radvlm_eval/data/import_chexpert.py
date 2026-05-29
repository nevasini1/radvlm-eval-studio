"""Importer for CheXpert (Stanford ML Group).

Expects a local copy obtained from
https://stanfordmlgroup.github.io/competitions/chexpert/ under the user's own
license. No download is performed here.

CheXpert ships labels and images but NOT original free-text reports. When reports
are unavailable we synthesize a *label-derived* reference summary and mark it
clearly as such — it is NOT an original radiologist report.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from radvlm_eval.data.load_studies import save_studies_csv
from radvlm_eval.schemas import CXR_LABELS, LabelValue, Study, empty_labels

LABEL_DERIVED_NOTICE = "[LABEL-DERIVED SUMMARY — not an original radiologist report]"


class ImporterError(RuntimeError):
    pass


def _instructions(root: Path) -> str:
    return (
        f"Could not find CheXpert train.csv/valid.csv under: {root}\n\n"
        "Expected something like:\n"
        "    <chexpert_root>/train.csv\n"
        "    <chexpert_root>/valid.csv\n"
        "    <chexpert_root>/train/patientXXXXX/studyY/view1_frontal.jpg\n\n"
        "Obtain CheXpert yourself from:\n"
        "    https://stanfordmlgroup.github.io/competitions/chexpert/"
    )


def _label_derived_summary(labels: dict) -> str:
    present = [k for k, v in labels.items() if v == 1 and k != "No Finding"]
    uncertain = [k for k, v in labels.items() if v == -1]
    if labels.get("No Finding") == 1 or (not present and not uncertain):
        body = "No acute cardiopulmonary findings indicated by labels."
    else:
        parts = []
        if present:
            parts.append("Findings indicated: " + ", ".join(present) + ".")
        if uncertain:
            parts.append("Uncertain: " + ", ".join(uncertain) + ".")
        body = " ".join(parts)
    return f"{LABEL_DERIVED_NOTICE} {body}"


def import_chexpert(
    chexpert_root: str | Path,
    output_dir: Optional[str | Path] = None,
    limit: Optional[int] = None,
) -> List[Study]:
    root = Path(chexpert_root).expanduser()
    csv_candidates = [root / "train.csv", root / "valid.csv"]
    csvs = [c for c in csv_candidates if c.exists()]
    if not csvs:
        # also accept nested CheXpert-v1.0 folders
        csvs = [p for p in root.rglob("*.csv") if p.name in ("train.csv", "valid.csv")]
    if not csvs:
        raise ImporterError(_instructions(root))

    studies: List[Study] = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        split = "train" if "train" in csv_path.name else "valid"
        for idx, row in df.iterrows():
            if limit and len(studies) >= limit:
                break
            labels: dict[str, LabelValue] = empty_labels()
            for lab in CXR_LABELS:
                if lab in row and pd.notna(row[lab]):
                    val = float(row[lab])
                    labels[lab] = int(val) if val in (1.0, 0.0, -1.0) else None

            rel_path = str(row.get("Path", "")).strip()
            image_paths = []
            if rel_path:
                # CheXpert paths are usually relative to the dataset parent.
                p = (root / rel_path)
                if not p.exists():
                    p = (root.parent / rel_path)
                image_paths = [str(p)]

            summary = _label_derived_summary(labels)
            studies.append(
                Study(
                    study_id=f"CHEXPERT-{split}-{idx}",
                    image_paths=image_paths,
                    report_text=summary,
                    findings=summary,
                    impression="",
                    indication=None,
                    labels=labels,
                    metadata={
                        "source": "chexpert",
                        "split": split,
                        "view_position": str(row.get("AP/PA", "")),
                        "report_available": False,
                        "reference_is_label_derived": True,
                    },
                )
            )

    if not studies:
        raise ImporterError(_instructions(root))

    if output_dir:
        save_studies_csv(studies, Path(output_dir) / "studies.csv")
    return studies
