"""Generate structured AI *draft* reports.

Three modes:
    A. template   — always works; derives a structured draft from labels.
    B. retrieval  — combines common findings from top similar prior cases.
    C. local_vlm  — optional on-device VLM (mlx-vlm); falls back gracefully.

Every draft is conservative, includes uncertainty/limitations, never claims a
diagnosis, and carries an "AI draft — requires radiologist review" disclaimer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from radvlm_eval.reporting import local_vlm
from radvlm_eval.reporting.templates import (
    FOLLOWUP_QUESTIONS,
    PRESENT_PHRASES,
    UNCERTAIN_PHRASES,
    render_structured_report,
)
from radvlm_eval.retrieval.similar_cases import SimilarCase
from radvlm_eval.schemas import CXR_LABELS, LabelValue, Study


@dataclass
class DraftResult:
    draft_text: str
    mode_used: str
    context_used: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    derived_labels: Dict[str, LabelValue] = field(default_factory=dict)


def _lines_from_labels(labels: Dict[str, LabelValue]):
    findings_lines: List[str] = []
    impression_lines: List[str] = []
    uncertainty_lines: List[str] = []
    followups: List[str] = []

    positives = [l for l in CXR_LABELS if labels.get(l) == 1 and l != "No Finding"]
    uncertains = [l for l in CXR_LABELS if labels.get(l) == -1]

    if not positives and not uncertains:
        findings_lines.append("Lungs clear; no focal consolidation, effusion, or pneumothorax.")
        impression_lines.append("No acute cardiopulmonary abnormality.")
    else:
        for lab in positives:
            findings_lines.append(PRESENT_PHRASES.get(lab, f"{lab}."))
            impression_lines.append(PRESENT_PHRASES.get(lab, f"{lab}.").rstrip("."))
        for lab in uncertains:
            uncertainty_lines.append(UNCERTAIN_PHRASES.get(lab, f"Possible {lab.lower()}."))

    for lab in positives + uncertains:
        if lab in FOLLOWUP_QUESTIONS:
            followups.append(FOLLOWUP_QUESTIONS[lab])

    return findings_lines, impression_lines, uncertainty_lines, followups


def _template_draft(study: Study) -> DraftResult:
    f, i, u, q = _lines_from_labels(study.labels)
    text = render_structured_report(
        indication=study.indication or "",
        findings_lines=f,
        impression_lines=i,
        uncertainty_lines=u,
        followups=q,
        context_note="Draft derived from structured labels (template mode).",
    )
    return DraftResult(
        draft_text=text,
        mode_used="template",
        context_used=["structured labels"],
        limitations=[
            "Template draft reflects labels only; it does not interpret the image.",
        ],
        derived_labels=dict(study.labels),
    )


def _aggregate_labels(cases: List[SimilarCase]) -> Dict[str, LabelValue]:
    """Majority vote: a label is positive if >= half of neighbors call it positive."""
    derived: Dict[str, LabelValue] = {l: None for l in CXR_LABELS}
    if not cases:
        return derived
    pos_counts: Counter = Counter()
    unc_counts: Counter = Counter()
    for c in cases:
        for lab, v in c.labels.items():
            if v == 1:
                pos_counts[lab] += 1
            elif v == -1:
                unc_counts[lab] += 1
    threshold = max(1, len(cases) // 2)
    for lab in CXR_LABELS:
        if pos_counts[lab] >= threshold:
            derived[lab] = 1
        elif unc_counts[lab] >= 1:
            derived[lab] = -1
    return derived


def _retrieval_draft(study: Study, cases: List[SimilarCase]) -> DraftResult:
    derived = _aggregate_labels(cases)
    f, i, u, q = _lines_from_labels(derived)
    context = [f"{c.study_id} (sim={c.score:.2f})" for c in cases]
    text = render_structured_report(
        indication=study.indication or "",
        findings_lines=f,
        impression_lines=i,
        uncertainty_lines=u,
        followups=q,
        context_note="Draft synthesized from common findings across "
        + ", ".join(context[:5]) + " (retrieval mode).",
    )
    return DraftResult(
        draft_text=text,
        mode_used="retrieval",
        context_used=context,
        limitations=[
            "Retrieval draft reflects patterns in similar prior cases, not this image.",
            "Conservative wording used; confirm all findings against the study.",
        ],
        derived_labels=derived,
    )


def generate_draft(
    study: Study,
    mode: str = "template",
    similar_cases: Optional[List[SimilarCase]] = None,
) -> DraftResult:
    mode = (mode or "template").lower()

    if mode == "local_vlm":
        image = study.primary_image
        caption = local_vlm.generate_vlm_findings(image) if image else None
        if caption:
            # Wrap the raw VLM caption in our structured, disclaimered scaffold.
            text = render_structured_report(
                indication=study.indication or "",
                findings_lines=[caption.strip()],
                impression_lines=["See findings; physician interpretation required."],
                uncertainty_lines=["Local VLM output is unverified and may hallucinate."],
                followups=["Confirm all findings against the image."],
                context_note="Generated by local mlx-vlm (local_vlm mode).",
            )
            return DraftResult(
                draft_text=text,
                mode_used="local_vlm",
                context_used=["local mlx-vlm caption"],
                limitations=["Local VLM is experimental and unvalidated."],
                derived_labels={},
            )
        # Fall back, recording why.
        fallback = _retrieval_draft(study, similar_cases or []) if similar_cases else _template_draft(study)
        fallback.limitations.insert(0, local_vlm.status_message())
        fallback.mode_used = f"{fallback.mode_used} (VLM fallback)"
        return fallback

    if mode == "retrieval":
        if not similar_cases:
            res = _template_draft(study)
            res.limitations.insert(0, "No similar cases supplied; fell back to template mode.")
            res.mode_used = "template (retrieval fallback)"
            return res
        return _retrieval_draft(study, similar_cases)

    return _template_draft(study)
