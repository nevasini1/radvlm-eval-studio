"""Canonical data schemas shared across the package.

We use dataclasses (no pydantic dependency) to keep the install lightweight and
robust on Apple Silicon. Labels follow the CheXpert convention:

    1    = present
    0    = absent
    -1   = uncertain
    None = not mentioned
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# The 14 standard CXR observations (CheXpert/MIMIC label set).
CXR_LABELS: List[str] = [
    "No Finding",
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
    "Pneumothorax",
    "Lung Opacity",
    "Lung Lesion",
    "Fracture",
    "Support Devices",
    "Enlarged Cardiomediastinum",
    "Pleural Other",
    "Pneumonia",
]

LabelValue = Optional[int]  # 1 / 0 / -1 / None


@dataclass
class Study:
    """A single radiology study in canonical form."""

    study_id: str
    image_paths: List[str] = field(default_factory=list)
    report_text: str = ""
    findings: str = ""
    impression: str = ""
    indication: Optional[str] = None
    labels: Dict[str, LabelValue] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- convenience ----------------------------------------------------
    @property
    def primary_image(self) -> Optional[str]:
        return self.image_paths[0] if self.image_paths else None

    @property
    def patient_id_hash(self) -> str:
        return str(self.metadata.get("patient_id_hash", ""))

    @property
    def split(self) -> str:
        return str(self.metadata.get("split", "demo"))

    @property
    def view_position(self) -> str:
        return str(self.metadata.get("view_position", ""))

    def positive_labels(self) -> List[str]:
        return [k for k, v in self.labels.items() if v == 1]

    def to_row(self) -> Dict[str, Any]:
        """Flatten to a CSV/parquet-friendly row (labels + metadata JSON-encoded)."""
        return {
            "study_id": self.study_id,
            "image_paths": json.dumps(self.image_paths),
            "report_text": self.report_text,
            "findings": self.findings,
            "impression": self.impression,
            "indication": self.indication or "",
            "labels": json.dumps(self.labels),
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Study":
        def _load(val: Any, default: Any) -> Any:
            if val is None or val == "" or (isinstance(val, float)):
                return default
            if isinstance(val, (dict, list)):
                return val
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return default

        return cls(
            study_id=str(row["study_id"]),
            image_paths=_load(row.get("image_paths"), []),
            report_text=str(row.get("report_text") or ""),
            findings=str(row.get("findings") or ""),
            impression=str(row.get("impression") or ""),
            indication=(row.get("indication") or None),
            labels=_load(row.get("labels"), {}),
            metadata=_load(row.get("metadata"), {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def empty_labels() -> Dict[str, LabelValue]:
    """A label dict with every observation set to None (not mentioned)."""
    return {label: None for label in CXR_LABELS}
