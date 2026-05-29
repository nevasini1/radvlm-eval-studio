"""Importer for the IU X-Ray / Open-i open chest X-ray collection.

This does NOT download anything. It expects the user to have obtained the data
from https://openi.nlm.nih.gov/ (or the Kaggle mirror) and points at a local
folder. If the layout differs from what we expect, we fail gracefully with
instructions instead of crashing.

Expected (common) layout:
    openi_root/
        images/        # *.png / *.jpg
        reports/       # *.xml radiology reports (NLM/Open-i XML)
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

from radvlm_eval.data.load_studies import save_studies_csv
from radvlm_eval.evaluation.report_labeler import label_report
from radvlm_eval.schemas import Study


class ImporterError(RuntimeError):
    pass


def _instructions(openi_root: Path) -> str:
    return (
        f"Could not find a usable Open-i / IU X-Ray layout under: {openi_root}\n\n"
        "Expected something like:\n"
        "    <openi_root>/images/*.png\n"
        "    <openi_root>/reports/*.xml\n\n"
        "Download the open collection yourself from:\n"
        "    https://openi.nlm.nih.gov/\n"
        "    https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university\n"
        "Then re-run with the correct --openi-root path."
    )


def _parse_openi_xml(xml_path: Path) -> Dict[str, str]:
    """Extract indication/comparison/findings/impression from an Open-i XML report."""
    out = {"indication": "", "comparison": "", "findings": "", "impression": ""}
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return out

    for abst in root.iter("AbstractText"):
        label = (abst.get("Label") or "").strip().lower()
        text = (abst.text or "").strip()
        if not text:
            continue
        if label in out:
            out[label] = text
    # Collect referenced image ids (parentImage id attributes).
    image_ids = [img.get("id") for img in root.iter("parentImage") if img.get("id")]
    out["_image_ids"] = ",".join([i for i in image_ids if i])  # type: ignore[assignment]
    return out


def import_openi(
    openi_root: str | Path,
    output_dir: Optional[str | Path] = None,
    limit: Optional[int] = None,
) -> List[Study]:
    openi_root = Path(openi_root).expanduser()
    images_dir = openi_root / "images"
    reports_dir = openi_root / "reports"

    if not openi_root.exists() or not reports_dir.exists():
        raise ImporterError(_instructions(openi_root))

    report_files = sorted(reports_dir.glob("*.xml"))
    if not report_files:
        raise ImporterError(_instructions(openi_root))

    # Build a quick lookup of available image files by stem.
    image_lookup: Dict[str, Path] = {}
    if images_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in images_dir.glob(ext):
                image_lookup[p.stem] = p

    studies: List[Study] = []
    for i, xml_path in enumerate(report_files):
        if limit and i >= limit:
            break
        parsed = _parse_openi_xml(xml_path)
        findings = parsed["findings"]
        impression = parsed["impression"]
        report_text = "\n\n".join(
            part
            for part in [
                f"INDICATION: {parsed['indication']}" if parsed["indication"] else "",
                f"FINDINGS: {findings}" if findings else "",
                f"IMPRESSION: {impression}" if impression else "",
            ]
            if part
        )

        image_ids = [s for s in parsed.get("_image_ids", "").split(",") if s]
        image_paths: List[str] = []
        for iid in image_ids:
            if iid in image_lookup:
                image_paths.append(str(image_lookup[iid]))

        study_id = re.sub(r"\.xml$", "", xml_path.name)
        labels = label_report(report_text)
        studies.append(
            Study(
                study_id=f"OPENI-{study_id}",
                image_paths=image_paths,
                report_text=report_text,
                findings=findings,
                impression=impression,
                indication=parsed["indication"] or None,
                labels=labels,
                metadata={"source": "openi", "split": "all"},
            )
        )

    if output_dir:
        save_studies_csv(studies, Path(output_dir) / "studies.csv")
    return studies
