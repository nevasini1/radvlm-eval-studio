"""Audit log writes to and reads from SQLite; review workflow recomputes metrics."""

from pathlib import Path

from radvlm_eval.schemas import Study, empty_labels
from radvlm_eval.workflow.audit_log import list_audit_entries, write_audit_entry
from radvlm_eval.workflow.review import export_case_review_json, review_edit


def test_audit_write_and_read(tmp_path: Path):
    db = tmp_path / "audit.db"
    rid = write_audit_entry(
        study_id="DEMO0001",
        draft_before="before text",
        draft_after="after text",
        reviewer_action="save_edit",
        metrics_before={"positive_f1": 0.5},
        metrics_after={"positive_f1": 0.9},
        notes="edited cardiomegaly",
        db_path=db,
    )
    assert rid >= 1

    entries = list_audit_entries(db_path=db)
    assert len(entries) == 1
    e = entries[0]
    assert e["study_id"] == "DEMO0001"
    assert e["draft_after"] == "after text"
    assert e["metrics_after"]["positive_f1"] == 0.9
    assert e["timestamp"]


def test_audit_filter_by_study(tmp_path: Path):
    db = tmp_path / "audit.db"
    write_audit_entry("A", "x", "y", "save_edit", db_path=db)
    write_audit_entry("B", "x", "y", "mark_reviewed", db_path=db)
    only_a = list_audit_entries(study_id="A", db_path=db)
    assert len(only_a) == 1 and only_a[0]["study_id"] == "A"


def test_review_edit_recomputes_and_logs(tmp_path: Path):
    db = tmp_path / "audit.db"
    labels = empty_labels()
    labels["Cardiomegaly"] = 1
    study = Study(
        study_id="DEMO0002",
        image_paths=["x.png"],
        report_text="Mild cardiomegaly.",
        findings="Mild cardiomegaly.",
        impression="Mild cardiomegaly.",
        labels=labels,
    )
    # Monkeypatch the default DB by passing through write via review's persist.
    import radvlm_eval.workflow.audit_log as al

    orig = al.write_audit_entry

    def _patched(*args, **kwargs):
        kwargs["db_path"] = db
        return orig(*args, **kwargs)

    al.write_audit_entry = _patched
    try:
        import radvlm_eval.workflow.review as review_mod

        review_mod.write_audit_entry = _patched
        outcome = review_edit(
            study,
            draft_before="Lungs clear. Heart normal.",
            draft_after="Mild cardiomegaly present.",
            reviewer_action="save_edit",
        )
    finally:
        al.write_audit_entry = orig

    assert outcome.diff["n_errors_fixed"] >= 1
    js = export_case_review_json(outcome)
    assert "study_id" in js
    entries = list_audit_entries(db_path=db)
    assert len(entries) == 1
