"""Radiologist edit/review orchestration: recompute metrics before/after an edit."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from radvlm_eval.evaluation.error_taxonomy import diff_edits
from radvlm_eval.evaluation.metrics import compute_metrics
from radvlm_eval.schemas import LabelValue, Study
from radvlm_eval.workflow.audit_log import write_audit_entry


@dataclass
class ReviewOutcome:
    study_id: str
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]
    diff: Dict[str, Any]
    audit_id: Optional[int] = None
    case_review: Dict[str, Any] = field(default_factory=dict)


def review_edit(
    study: Study,
    draft_before: str,
    draft_after: str,
    reviewer_action: str = "save_edit",
    notes: str = "",
    persist: bool = True,
) -> ReviewOutcome:
    """Recompute evaluation before/after a radiologist edit and (optionally) log it."""
    ref_labels: Dict[str, LabelValue] = study.labels
    ref_text = study.report_text

    metrics_before = compute_metrics(ref_labels, ref_text, draft_before)
    metrics_after = compute_metrics(ref_labels, ref_text, draft_after)
    diff = diff_edits(ref_labels, ref_text, draft_before, draft_after)

    audit_id = None
    if persist:
        audit_id = write_audit_entry(
            study_id=study.study_id,
            draft_before=draft_before,
            draft_after=draft_after,
            reviewer_action=reviewer_action,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            notes=notes,
        )

    case_review = {
        "study_id": study.study_id,
        "reviewer_action": reviewer_action,
        "draft_before": draft_before,
        "draft_after": draft_after,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "diff": diff,
        "notes": notes,
        "disclaimer": "Research demo only. Not for diagnosis or clinical use.",
    }

    return ReviewOutcome(
        study_id=study.study_id,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        diff=diff,
        audit_id=audit_id,
        case_review=case_review,
    )


def export_case_review_json(outcome: ReviewOutcome) -> str:
    return json.dumps(outcome.case_review, indent=2)
