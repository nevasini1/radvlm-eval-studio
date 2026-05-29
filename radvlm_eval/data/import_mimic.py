"""Importer for MIMIC-CXR / MIMIC-CXR-JPG (credentialed PhysioNet datasets).

IMPORTANT
---------
This importer NEVER downloads data. MIMIC-CXR / MIMIC-CXR-JPG are credentialed
PhysioNet datasets governed by a Data Use Agreement and CANNOT be redistributed.
The user must obtain them via https://physionet.org/ under their own credentials.

We expect only local files. If reports are unavailable we still import labels
and image paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from radvlm_eval.data.load_studies import save_studies_csv
from radvlm_eval.schemas import CXR_LABELS, Study, empty_labels

REDISTRIBUTION_WARNING = (
    "MIMIC-CXR / MIMIC-CXR-JPG are credentialed PhysioNet datasets. They cannot "
    "be redistributed and must not be committed to any repository. This tool only "
    "reads your local copy obtained under your own PhysioNet DUA."
)


class ImporterError(RuntimeError):
    pass


def _instructions(root: Path) -> str:
    return (
        f"Could not find MIMIC-CXR-JPG metadata/labels under: {root}\n\n"
        "Expected local files such as:\n"
        "    mimic-cxr-2.0.0-metadata.csv\n"
        "    mimic-cxr-2.0.0-chexpert.csv   (labels)\n"
        "    mimic-cxr-2.0.0-split.csv\n"
        "    files/p10/p10000032/s50414267/<dicom_id>.jpg  (images)\n\n"
        "Obtain the data yourself from https://physionet.org/content/mimic-cxr-jpg/\n"
        f"\n{REDISTRIBUTION_WARNING}"
    )


def _find_csv(root: Path, *substrings: str) -> Optional[Path]:
    for p in root.rglob("*.csv"):
        name = p.name.lower()
        if all(s in name for s in substrings):
            return p
    return None


def import_mimic(
    mimic_jpg_root: str | Path,
    mimic_report_root_optional: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    limit: Optional[int] = None,
) -> List[Study]:
    print(f"[mimic] {REDISTRIBUTION_WARNING}")
    root = Path(mimic_jpg_root).expanduser()
    if not root.exists():
        raise ImporterError(_instructions(root))

    meta_csv = _find_csv(root, "metadata")
    labels_csv = _find_csv(root, "chexpert") or _find_csv(root, "negbio")
    split_csv = _find_csv(root, "split")

    if meta_csv is None and labels_csv is None:
        raise ImporterError(_instructions(root))

    meta_df = pd.read_csv(meta_csv) if meta_csv else pd.DataFrame()
    labels_df = pd.read_csv(labels_csv) if labels_csv else pd.DataFrame()
    split_df = pd.read_csv(split_csv) if split_csv else pd.DataFrame()

    # MIMIC keys on subject_id + study_id.
    base = meta_df if not meta_df.empty else labels_df
    if base.empty:
        raise ImporterError(_instructions(root))

    if not labels_df.empty and not meta_df.empty:
        base = meta_df.merge(labels_df, on=["subject_id", "study_id"], how="left")
    if not split_df.empty and "study_id" in base.columns:
        split_cols = [c for c in split_df.columns if c in ("study_id", "split")]
        if "study_id" in split_cols and "split" in split_cols:
            base = base.merge(split_df[["study_id", "split"]], on="study_id", how="left")

    report_root = Path(mimic_report_root_optional).expanduser() if mimic_report_root_optional else None

    studies: List[Study] = []
    seen = set()
    for _, row in base.iterrows():
        sid = str(row.get("study_id", ""))
        subj = str(row.get("subject_id", ""))
        if not sid or sid in seen:
            continue
        seen.add(sid)
        if limit and len(studies) >= limit:
            break

        labels = empty_labels()
        for lab in CXR_LABELS:
            if lab in row and pd.notna(row[lab]):
                val = float(row[lab])
                labels[lab] = int(val) if val in (1.0, 0.0, -1.0) else None

        # Optional report text.
        report_text = ""
        if report_root is not None:
            # MIMIC reports: files/p<subj_prefix>/p<subj>/s<study>.txt
            candidates = list(report_root.rglob(f"s{sid}.txt"))
            if candidates:
                report_text = candidates[0].read_text(errors="ignore").strip()

        # Image paths (JPG) for this study, if present on disk.
        image_paths: List[str] = []
        if subj:
            prefix = f"p{subj[:2]}"
            study_dir = root / "files" / prefix / f"p{subj}" / f"s{sid}"
            if study_dir.exists():
                image_paths = [str(p) for p in sorted(study_dir.glob("*.jpg"))]

        studies.append(
            Study(
                study_id=f"MIMIC-{sid}",
                image_paths=image_paths,
                report_text=report_text,
                findings=report_text,  # not parsed into sections here
                impression="",
                indication=None,
                labels=labels,
                metadata={
                    "source": "mimic",
                    "subject_id": subj,
                    "split": str(row.get("split", "")),
                    "view_position": str(row.get("ViewPosition", "")),
                    "report_available": bool(report_text),
                },
            )
        )

    if not studies:
        raise ImporterError(_instructions(root))

    if output_dir:
        save_studies_csv(studies, Path(output_dir) / "studies.csv")
    return studies
