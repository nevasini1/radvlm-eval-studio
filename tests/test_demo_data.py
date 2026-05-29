"""Demo data generation produces valid, committable synthetic studies."""

from pathlib import Path

from radvlm_eval.data.demo_data import generate_demo_dataset
from radvlm_eval.schemas import CXR_LABELS


def test_generate_demo_dataset_count_and_validity(tmp_path: Path):
    studies = generate_demo_dataset(data_dir=tmp_path, n_studies=32, seed=1)
    assert 30 <= len(studies) <= 50

    seen_ids = set()
    for s in studies:
        assert s.study_id and s.study_id not in seen_ids
        seen_ids.add(s.study_id)

        # Each study has at least one rendered image that exists on disk.
        assert s.image_paths
        img = Path(s.image_paths[0])
        if not img.is_absolute():
            from radvlm_eval import config

            img = config.REPO_ROOT / img
        assert img.exists()

        # Report sections present.
        assert s.findings
        assert s.impression
        assert s.indication

        # Labels use only the allowed schema + values.
        for k, v in s.labels.items():
            assert k in CXR_LABELS
            assert v in (1, 0, -1, None)

        # Metadata is synthetic and carries required fields.
        assert s.metadata.get("synthetic") is True
        assert s.patient_id_hash
        assert s.split in {"valid", "test", "train"}


def test_demo_includes_seed_drafts_with_errors(tmp_path: Path):
    studies = generate_demo_dataset(data_dir=tmp_path, n_studies=36, seed=2)
    seeded = [s for s in studies if s.metadata.get("seed_draft")]
    assert seeded, "expected some studies to carry intentionally-flawed seed drafts"
