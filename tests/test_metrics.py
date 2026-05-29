"""Metrics are null-safe and behave sensibly on clear cases."""

from radvlm_eval.evaluation.metrics import compute_metrics, status_color
from radvlm_eval.schemas import empty_labels


def test_metrics_handle_empty_labels_safely():
    m = compute_metrics({}, "", "")
    assert m["exact_label_agreement"] == 0.0
    assert m["positive_f1"] == 0.0
    assert m["missed_finding_count"] == 0
    assert m["hallucinated_finding_count"] == 0
    assert m["report_length_ratio"] == 0.0


def test_metrics_perfect_match():
    ref = empty_labels()
    ref["Pleural Effusion"] = 1
    text = "Small left pleural effusion."
    m = compute_metrics(ref, text, text, draft_labels=ref)
    assert m["positive_precision"] == 1.0
    assert m["positive_recall"] == 1.0
    assert m["positive_f1"] == 1.0
    assert m["missed_finding_count"] == 0
    assert m["hallucinated_finding_count"] == 0


def test_metrics_missed_finding_counts():
    ref = empty_labels()
    ref["Pleural Effusion"] = 1
    draft_labels = empty_labels()
    m = compute_metrics(ref, "Pleural effusion.", "Clear.", draft_labels=draft_labels)
    assert m["missed_finding_count"] == 1
    assert m["positive_recall"] == 0.0


def test_status_color_red_on_laterality():
    m = {"laterality_mismatch_count": 1, "missed_finding_count": 0,
         "hallucinated_finding_count": 0, "positive_f1": 0.9}
    assert status_color(m) == "red"


def test_status_color_green_when_clean():
    m = {"laterality_mismatch_count": 0, "missed_finding_count": 0,
         "hallucinated_finding_count": 0, "positive_f1": 0.95}
    assert status_color(m) == "green"
