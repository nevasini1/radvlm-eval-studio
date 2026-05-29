"""Clinically meaningful error taxonomy comparing a reference vs an AI draft.

Produces structured `ClinicalError` records that surface *why* a draft is wrong
in clinical terms — the kind of signal generic BLEU/ROUGE cannot give you.

Error types:
    missed_finding          reference positive, draft missed it
    hallucinated_finding    draft positive, reference did not have it
    negation_error          direct present/absent contradiction
    uncertainty_mismatch    one side hedges (-1), the other is definite
    laterality_mismatch     left/right disagreement on the same finding
    severity_mismatch       small/mild vs large/severe disagreement
    support_device_mismatch disagreement on lines/tubes/devices
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from radvlm_eval.evaluation.report_labeler import evidence_sentence, label_report
from radvlm_eval.schemas import CXR_LABELS, LabelValue

HIGH_RISK_LABELS = {
    "Pneumothorax",
    "Pneumonia",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
    "Lung Lesion",
    "Enlarged Cardiomediastinum",
}

SEVERITY_TERMS = {
    "trace": 1, "tiny": 1, "minimal": 1, "small": 2, "mild": 2, "minor": 2,
    "moderate": 3, "mid": 3, "large": 4, "severe": 4, "extensive": 4, "marked": 4,
}

LATERALITY = ["left", "right", "bilateral"]


@dataclass
class ClinicalError:
    error_type: str
    finding: str
    severity: str  # low / medium / high
    reference_value: LabelValue
    draft_value: LabelValue
    reference_evidence: Optional[str]
    draft_evidence: Optional[str]
    mitigation: str

    def to_dict(self) -> Dict:
        return {
            "error_type": self.error_type,
            "finding": self.finding,
            "severity": self.severity,
            "reference_value": self.reference_value,
            "draft_value": self.draft_value,
            "reference_evidence": self.reference_evidence,
            "draft_evidence": self.draft_evidence,
            "mitigation": self.mitigation,
        }


def _severity_for(label: str, base: str = "medium") -> str:
    if label in HIGH_RISK_LABELS:
        return "high"
    return base


def _laterality_in(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    low = text.lower()
    for term in LATERALITY:
        if re.search(rf"\b{term}\b", low):
            return term
    return None


def _severity_in(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    low = text.lower()
    found = [score for term, score in SEVERITY_TERMS.items() if re.search(rf"\b{term}\b", low)]
    return max(found) if found else None


def _mk(error_type: str, label: str, ref_v, draft_v, ref_ev, draft_ev, severity, mitigation):
    return ClinicalError(
        error_type=error_type,
        finding=label,
        severity=severity,
        reference_value=ref_v,
        draft_value=draft_v,
        reference_evidence=ref_ev,
        draft_evidence=draft_ev,
        mitigation=mitigation,
    )


def compare_reports(
    reference_labels: Dict[str, LabelValue],
    reference_text: str,
    draft_text: str,
    draft_labels: Optional[Dict[str, LabelValue]] = None,
) -> List[ClinicalError]:
    """Compare a reference (labels + text) against an AI draft report."""
    if draft_labels is None:
        draft_labels = label_report(draft_text)

    errors: List[ClinicalError] = []

    for label in CXR_LABELS:
        ref_v = reference_labels.get(label)
        draft_v = draft_labels.get(label)
        ref_ev = evidence_sentence(reference_text, label)
        draft_ev = evidence_sentence(draft_text, label)

        # Missed finding: reference clearly present, draft not present.
        if ref_v == 1 and draft_v != 1:
            if draft_v == 0:
                errors.append(_mk(
                    "negation_error", label, ref_v, draft_v, ref_ev, draft_ev,
                    _severity_for(label, "high"),
                    f"Draft explicitly negates '{label}' that the reference reports as present. "
                    "Verify against the image before sign-off.",
                ))
            else:
                errors.append(_mk(
                    "missed_finding", label, ref_v, draft_v, ref_ev, draft_ev,
                    _severity_for(label, "medium"),
                    f"Draft omits '{label}'. Add to findings/impression if confirmed.",
                ))
            continue

        # Hallucinated finding: draft present, reference not present.
        if draft_v == 1 and ref_v != 1:
            if ref_v == 0:
                errors.append(_mk(
                    "negation_error", label, ref_v, draft_v, ref_ev, draft_ev,
                    _severity_for(label, "high"),
                    f"Draft asserts '{label}' that the reference explicitly negates. "
                    "Likely hallucination — remove unless confirmed.",
                ))
            else:
                errors.append(_mk(
                    "hallucinated_finding", label, ref_v, draft_v, ref_ev, draft_ev,
                    _severity_for(label, "medium"),
                    f"Draft introduces '{label}' not supported by the reference. Remove or confirm.",
                ))
            continue

        # Uncertainty mismatch: one hedges, the other is definite.
        if {ref_v, draft_v} & {-1} and ref_v != draft_v and ref_v is not None and draft_v is not None:
            errors.append(_mk(
                "uncertainty_mismatch", label, ref_v, draft_v, ref_ev, draft_ev,
                "medium" if label in HIGH_RISK_LABELS else "low",
                f"Certainty for '{label}' differs (uncertain vs definite). Align hedging language.",
            ))
            continue

        # For findings present on both sides, check laterality and severity.
        if ref_v == 1 and draft_v == 1:
            ref_lat, draft_lat = _laterality_in(ref_ev), _laterality_in(draft_ev)
            if ref_lat and draft_lat and ref_lat != draft_lat:
                errors.append(_mk(
                    "laterality_mismatch", label, ref_v, draft_v, ref_ev, draft_ev,
                    "high",
                    f"Laterality differs for '{label}' ({ref_lat} vs {draft_lat}). "
                    "Laterality errors are clinically critical — correct before sign-off.",
                ))
                continue
            ref_sev, draft_sev = _severity_in(ref_ev), _severity_in(draft_ev)
            if ref_sev and draft_sev and ref_sev != draft_sev:
                errors.append(_mk(
                    "severity_mismatch", label, ref_v, draft_v, ref_ev, draft_ev,
                    "low",
                    f"Severity wording for '{label}' differs. Align descriptors.",
                ))

    # Support device specialization: relabel any device disagreement.
    for err in errors:
        if err.finding == "Support Devices" and err.error_type in (
            "missed_finding", "hallucinated_finding", "negation_error"
        ):
            err.error_type = "support_device_mismatch"
            err.mitigation = (
                "Lines/tubes/devices disagreement. Confirm presence and position "
                "(e.g., ET tube/catheter tip) against the image."
            )

    return errors


def diff_edits(
    reference_labels: Dict[str, LabelValue],
    reference_text: str,
    draft_before: str,
    draft_after: str,
) -> Dict:
    """Compare evaluation of a draft before vs after a radiologist edit."""
    before = compare_reports(reference_labels, reference_text, draft_before)
    after = compare_reports(reference_labels, reference_text, draft_after)

    def _key(e: ClinicalError):
        return (e.error_type, e.finding)

    before_keys = {_key(e) for e in before}
    after_keys = {_key(e) for e in after}

    errors_fixed = [e.to_dict() for e in before if _key(e) not in after_keys]
    errors_introduced = [e.to_dict() for e in after if _key(e) not in before_keys]

    text_similarity = SequenceMatcher(None, draft_before, draft_after).ratio()

    return {
        "errors_before": len(before),
        "errors_after": len(after),
        "errors_fixed": errors_fixed,
        "errors_introduced": errors_introduced,
        "n_errors_fixed": len(errors_fixed),
        "n_errors_introduced": len(errors_introduced),
        "text_similarity": round(text_similarity, 3),
    }
