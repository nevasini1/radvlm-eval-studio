"""Retrieval returns top-k similar studies and excludes the query itself."""

from pathlib import Path

from radvlm_eval.data.demo_data import generate_demo_dataset
from radvlm_eval.retrieval.index import build_index
from radvlm_eval.retrieval.similar_cases import find_similar


def _build(tmp_path: Path):
    studies = generate_demo_dataset(data_dir=tmp_path / "data", n_studies=30, seed=3)
    index = build_index(studies, backend="fallback", index_dir=tmp_path / "index")
    return studies, index


def test_find_similar_topk_and_excludes_self(tmp_path: Path):
    studies, index = _build(tmp_path)
    query = studies[0].study_id
    results = find_similar(query, top_k=5, index=index)

    assert len(results) == 5
    assert all(r.study_id != query for r in results)
    # Scores are sorted descending.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_find_similar_returns_labels_and_paths(tmp_path: Path):
    studies, index = _build(tmp_path)
    results = find_similar(studies[1].study_id, top_k=3, index=index)
    assert len(results) == 3
    for r in results:
        assert isinstance(r.labels, dict)
        assert r.image_path


def test_find_similar_unknown_id_raises(tmp_path: Path):
    _, index = _build(tmp_path)
    import pytest

    with pytest.raises(KeyError):
        find_similar("NOPE-9999", top_k=3, index=index)
