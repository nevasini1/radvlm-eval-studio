"""Evaluation: weak labeling, clinical error taxonomy, and metrics."""

from radvlm_eval.evaluation.error_taxonomy import (
    ClinicalError,
    compare_reports,
    diff_edits,
)
from radvlm_eval.evaluation.metrics import compute_metrics
from radvlm_eval.evaluation.report_labeler import label_report

__all__ = [
    "ClinicalError",
    "compare_reports",
    "diff_edits",
    "compute_metrics",
    "label_report",
]
