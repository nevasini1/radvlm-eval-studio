"""Error injection produces evaluator-detectable flawed drafts."""

import random

from radvlm_eval.evaluation.error_taxonomy import compare_reports
from radvlm_eval.schemas import Study, empty_labels
from radvlm_eval.training.error_augmenter import (
    applicable_error_types,
    build_reference_report,
    inject_error,
)


def _pneumothorax_study():
    labels = empty_labels()
    labels["Pneumothorax"] = 1
    return Study(
        study_id="T-PTX",
        report_text="FINDINGS: Moderate right pneumothorax with partial collapse.\n\nIMPRESSION: Moderate right pneumothorax.",
        findings="Moderate right pneumothorax with partial collapse.",
        impression="Moderate right pneumothorax.",
        indication="trauma",
        labels=labels,
    )


def _effusion_study():
    labels = empty_labels()
    labels["Pleural Effusion"] = 1
    labels["Atelectasis"] = 1
    return Study(
        study_id="T-EFF",
        report_text="FINDINGS: Small left pleural effusion with basilar atelectasis.\n\nIMPRESSION: Small left pleural effusion.",
        findings="Small left pleural effusion with basilar atelectasis.",
        impression="Small left pleural effusion.",
        labels=labels,
    )


def test_build_reference_report_mentions_finding():
    rep = build_reference_report(_pneumothorax_study())
    assert "pneumothorax" in rep.lower()
    assert "FINDINGS" in rep


def test_negation_error_detected():
    s = _pneumothorax_study()
    aug = inject_error(s, "negation_error", random.Random(1))
    assert aug.error_type == "negation_error"
    assert "no pneumothorax" in aug.flawed_draft.lower()
    errors = compare_reports(s.labels, s.report_text, aug.flawed_draft)
    assert any(e.finding == "Pneumothorax" and e.error_type == "negation_error" for e in errors)
    assert any(e.severity == "high" for e in errors)


def test_missed_finding_detected():
    s = _pneumothorax_study()
    aug = inject_error(s, "missed_finding", random.Random(2))
    errors = compare_reports(s.labels, s.report_text, aug.flawed_draft)
    assert any(e.error_type in ("missed_finding", "negation_error") and e.finding == "Pneumothorax" for e in errors)


def test_laterality_mismatch_detected():
    s = _effusion_study()
    assert "laterality_mismatch" in applicable_error_types(s)
    aug = inject_error(s, "laterality_mismatch", random.Random(3))
    assert aug.error_type == "laterality_mismatch"
    errors = compare_reports(s.labels, s.report_text, aug.flawed_draft)
    assert any(e.error_type == "laterality_mismatch" for e in errors)


def test_hallucinated_always_applicable():
    normal = Study(
        study_id="T-NORM",
        report_text="FINDINGS: The lungs are clear.\n\nIMPRESSION: No acute abnormality.",
        findings="The lungs are clear.",
        impression="No acute abnormality.",
        labels={**empty_labels(), "No Finding": 1},
    )
    assert "hallucinated_finding" in applicable_error_types(normal)
    aug = inject_error(normal, "hallucinated_finding", random.Random(4))
    errors = compare_reports(normal.labels, normal.report_text, aug.flawed_draft)
    assert any(e.error_type == "hallucinated_finding" for e in errors)


def test_every_injection_yields_an_error():
    s = _effusion_study()
    rng = random.Random(5)
    for etype in applicable_error_types(s):
        aug = inject_error(s, etype, rng)
        assert aug is not None
        errors = compare_reports(s.labels, s.report_text, aug.flawed_draft)
        assert errors, f"no error detected for injected {etype}"
