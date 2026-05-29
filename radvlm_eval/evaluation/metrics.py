"""Quantitative metrics comparing reference vs draft labels and text.

All functions are null-safe: missing labels, empty reports, and empty label sets
degrade gracefully to 0.0 / 0 rather than raising.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from radvlm_eval.evaluation.report_labeler import label_report
from radvlm_eval.schemas import CXR_LABELS, LabelValue


def _is_positive(v: LabelValue) -> bool:
    return v == 1


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z]{2,}", (text or "").lower()))


def compute_metrics(
    reference_labels: Dict[str, LabelValue],
    reference_text: str,
    draft_text: str,
    draft_labels: Optional[Dict[str, LabelValue]] = None,
) -> Dict[str, float]:
    if draft_labels is None:
        draft_labels = label_report(draft_text)

    reference_labels = reference_labels or {}
    draft_labels = draft_labels or {}

    # Exact agreement over labels that are mentioned in EITHER side.
    considered = [
        lab for lab in CXR_LABELS
        if reference_labels.get(lab) is not None or draft_labels.get(lab) is not None
    ]
    if considered:
        agree = sum(
            1 for lab in considered
            if reference_labels.get(lab) == draft_labels.get(lab)
        )
        exact_agreement = agree / len(considered)
    else:
        exact_agreement = 0.0

    # Positive-finding precision/recall/F1 (treat label==1 as positive).
    ref_pos = {lab for lab in CXR_LABELS if _is_positive(reference_labels.get(lab))}
    draft_pos = {lab for lab in CXR_LABELS if _is_positive(draft_labels.get(lab))}

    tp = len(ref_pos & draft_pos)
    fp = len(draft_pos - ref_pos)
    fn = len(ref_pos - draft_pos)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    missed = len(ref_pos - draft_pos)
    hallucinated = len(draft_pos - ref_pos)

    # Uncertainty mismatch count.
    uncertain_mismatch = sum(
        1 for lab in CXR_LABELS
        if (reference_labels.get(lab) == -1) != (draft_labels.get(lab) == -1)
        and (reference_labels.get(lab) is not None and draft_labels.get(lab) is not None)
    )

    # Laterality mismatch (text heuristic).
    laterality_mismatch = _laterality_mismatch(reference_text, draft_text)

    # Report length ratio + lexical overlap (Jaccard).
    ref_len = len((reference_text or "").split())
    draft_len = len((draft_text or "").split())
    length_ratio = (draft_len / ref_len) if ref_len else 0.0

    ref_tok, draft_tok = _tokens(reference_text), _tokens(draft_text)
    union = ref_tok | draft_tok
    lexical_overlap = (len(ref_tok & draft_tok) / len(union)) if union else 0.0

    return {
        "exact_label_agreement": round(exact_agreement, 3),
        "positive_precision": round(precision, 3),
        "positive_recall": round(recall, 3),
        "positive_f1": round(f1, 3),
        "missed_finding_count": missed,
        "hallucinated_finding_count": hallucinated,
        "uncertain_mismatch_count": uncertain_mismatch,
        "laterality_mismatch_count": laterality_mismatch,
        "report_length_ratio": round(length_ratio, 3),
        "lexical_overlap": round(lexical_overlap, 3),
    }


def _laterality_mismatch(reference_text: str, draft_text: str) -> int:
    """Count findings where left/right laterality flips between ref and draft."""
    def sides(text: str) -> set:
        low = (text or "").lower()
        out = set()
        if re.search(r"\bleft\b", low):
            out.add("left")
        if re.search(r"\bright\b", low):
            out.add("right")
        return out

    ref_sides, draft_sides = sides(reference_text), sides(draft_text)
    # A flip: one side mentions left-only where the other mentions right-only.
    if ref_sides == {"left"} and draft_sides == {"right"}:
        return 1
    if ref_sides == {"right"} and draft_sides == {"left"}:
        return 1
    return 0


def status_color(metrics: Dict[str, float]) -> str:
    """Traffic-light status for the dashboard."""
    missed = metrics.get("missed_finding_count", 0)
    halluc = metrics.get("hallucinated_finding_count", 0)
    lat = metrics.get("laterality_mismatch_count", 0)
    f1 = metrics.get("positive_f1", 0.0)
    if lat > 0 or missed >= 2 or halluc >= 2:
        return "red"
    if missed or halluc or f1 < 0.6:
        return "yellow"
    return "green"
