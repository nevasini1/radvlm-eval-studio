"""Template-fallback repair works without MLX and reduces clinical errors."""

import random

from radvlm_eval.evaluation.error_taxonomy import compare_reports
from radvlm_eval.schemas import Study, empty_labels
from radvlm_eval.training.error_augmenter import inject_error
from radvlm_eval.training.evaluate_adapter import evaluate
from radvlm_eval.training.report_repair_model import repair_report


def _ptx_study():
    labels = empty_labels()
    labels["Pneumothorax"] = 1
    return Study(
        study_id="R-PTX",
        report_text="FINDINGS: Moderate right pneumothorax.\n\nIMPRESSION: Moderate right pneumothorax.",
        findings="Moderate right pneumothorax.",
        impression="Moderate right pneumothorax.",
        labels=labels,
    )


def test_repair_uses_template_fallback_without_mlx():
    s = _ptx_study()
    aug = inject_error(s, "negation_error", random.Random(1))
    before = compare_reports(s.labels, s.report_text, aug.flawed_draft)
    assert before  # there is at least one error to fix

    result = repair_report(s, aug.flawed_draft)
    assert result["method"] == "template-fallback"

    after = compare_reports(s.labels, s.report_text, str(result["repaired_text"]))
    assert len(after) < len(before)


def test_evaluate_runs_and_reports_fallback(tmp_path):
    from radvlm_eval.data.demo_data import generate_demo_dataset

    studies = generate_demo_dataset(data_dir=tmp_path, n_studies=30, seed=8)
    results = evaluate(studies=studies, save=False, limit=12)
    assert results["n_cases"] == 12
    assert results["method"] == "template-fallback"
    assert results["is_trained_adapter"] is False
    # Repair should not increase total errors in aggregate.
    agg = results["aggregate"]["total_errors"]
    assert agg["after"] <= agg["before"]
