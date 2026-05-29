"""RSNA-style structured report scaffolding.

We follow the *spirit* of RSNA RadReport structured templates (Indication /
Findings / Impression / follow-up) without copying any proprietary template text.
Reference: https://www.rsna.org/practice-tools/data-tools-and-standards/radreport-reporting-templates
"""

from __future__ import annotations

from typing import Dict, List

from radvlm_eval.safety import DRAFT_DISCLAIMER
from radvlm_eval.schemas import LabelValue

# Natural-language phrasing for each observation when reported as present.
PRESENT_PHRASES: Dict[str, str] = {
    "Cardiomegaly": "Enlargement of the cardiac silhouette.",
    "Edema": "Findings suggestive of pulmonary edema / vascular congestion.",
    "Consolidation": "Airspace consolidation.",
    "Atelectasis": "Atelectasis / volume loss.",
    "Pleural Effusion": "Pleural effusion.",
    "Pneumothorax": "Pneumothorax.",
    "Lung Opacity": "Pulmonary opacity.",
    "Lung Lesion": "Pulmonary nodule / lesion.",
    "Fracture": "Osseous fracture.",
    "Support Devices": "Support device(s) present (line/tube).",
    "Enlarged Cardiomediastinum": "Widening of the cardiomediastinal contour.",
    "Pleural Other": "Pleural abnormality (thickening/calcification).",
    "Pneumonia": "Findings that may reflect pneumonia in the right clinical setting.",
}

UNCERTAIN_PHRASES: Dict[str, str] = {
    label: f"Possible {label.lower()}; cannot be excluded."
    for label in PRESENT_PHRASES
}

FOLLOWUP_QUESTIONS: Dict[str, str] = {
    "Pneumonia": "Any fever, productive cough, or elevated inflammatory markers?",
    "Pleural Effusion": "Known heart failure, malignancy, or recent procedure?",
    "Lung Lesion": "Prior imaging available for comparison? Smoking history?",
    "Pneumothorax": "Recent trauma, procedure, or acute dyspnea?",
    "Support Devices": "Confirm intended device position from the chart.",
    "Edema": "Fluid status / recent weight change?",
}


def render_structured_report(
    indication: str,
    findings_lines: List[str],
    impression_lines: List[str],
    uncertainty_lines: List[str],
    followups: List[str],
    context_note: str = "",
) -> str:
    """Assemble a structured, RSNA-style draft with mandatory disclaimer."""
    findings = "\n".join(f"- {ln}" for ln in findings_lines) or "- No acute findings identified."
    impression = "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(impression_lines)) or "1. No acute cardiopulmonary abnormality."
    uncertainty = "\n".join(f"- {ln}" for ln in uncertainty_lines) or "- None noted."
    follow = "\n".join(f"- {q}" for q in followups) or "- Clinical correlation as indicated."

    parts = [
        f"⚠️ {DRAFT_DISCLAIMER}",
        "",
        f"INDICATION: {indication or 'Not provided.'}",
        "",
        "FINDINGS:",
        findings,
        "",
        "IMPRESSION:",
        impression,
        "",
        "UNCERTAINTY / LIMITATIONS:",
        uncertainty,
        (
            "- Automated draft generated from labels and similar prior cases; "
            "image not independently interpreted by a physician."
        ),
        "",
        "SUGGESTED FOLLOW-UP QUESTIONS:",
        follow,
    ]
    if context_note:
        parts += ["", f"CONTEXT: {context_note}"]
    return "\n".join(parts)
