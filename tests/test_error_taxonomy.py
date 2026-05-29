"""Tests for the rule-based labeler and clinical error taxonomy."""

from radvlm_eval.evaluation.error_taxonomy import compare_reports, diff_edits
from radvlm_eval.evaluation.report_labeler import label_report
from radvlm_eval.schemas import empty_labels


def test_negation_no_pneumothorax():
    labels = label_report("No pneumothorax. The lungs are clear.")
    assert labels["Pneumothorax"] != 1
    assert labels["Pneumothorax"] == 0


def test_positive_pneumothorax():
    labels = label_report("Moderate right pneumothorax with partial collapse.")
    assert labels["Pneumothorax"] == 1


def test_uncertainty_detected():
    labels = label_report("Possible pneumonia in the right lower lobe.")
    assert labels["Pneumonia"] == -1


def test_missed_finding_detected():
    ref = empty_labels()
    ref["Cardiomegaly"] = 1
    ref_text = "Mild cardiomegaly. Lungs clear."
    draft = "The lungs are clear. Heart size appears within normal limits."
    errors = compare_reports(ref, ref_text, draft)
    kinds = {(e.error_type, e.finding) for e in errors}
    assert ("missed_finding", "Cardiomegaly") in kinds


def test_hallucinated_finding_detected():
    ref = empty_labels()
    ref["No Finding"] = 1
    ref_text = "No acute cardiopulmonary abnormality."
    draft = "There is a moderate right pneumothorax."
    errors = compare_reports(ref, ref_text, draft)
    kinds = {(e.error_type, e.finding) for e in errors}
    assert ("hallucinated_finding", "Pneumothorax") in kinds


def test_negation_error_high_severity():
    ref = empty_labels()
    ref["Pneumothorax"] = 1
    ref_text = "Moderate right pneumothorax."
    draft = "No pneumothorax identified. Lungs clear."
    errors = compare_reports(ref, ref_text, draft)
    ptx = [e for e in errors if e.finding == "Pneumothorax"]
    assert ptx and ptx[0].error_type == "negation_error"
    assert ptx[0].severity == "high"


def test_diff_edits_fixes_error():
    ref = empty_labels()
    ref["Cardiomegaly"] = 1
    ref_text = "Mild cardiomegaly."
    before = "Lungs clear. Heart normal."
    after = "Mild cardiomegaly present. Lungs clear."
    diff = diff_edits(ref, ref_text, before, after)
    assert diff["n_errors_fixed"] >= 1
