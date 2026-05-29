"""Similar-case retrieval over a prebuilt index using cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from radvlm_eval.retrieval.index import Index, load_index
from radvlm_eval.schemas import LabelValue, Study


@dataclass
class SimilarCase:
    study_id: str
    score: float
    image_path: Optional[str]
    impression: str
    labels: Dict[str, LabelValue]
    study: Study

    def to_dict(self) -> Dict:
        return {
            "study_id": self.study_id,
            "score": round(self.score, 4),
            "image_path": self.image_path,
            "impression": self.impression,
            "positive_labels": self.study.positive_labels(),
        }


def _cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # Embeddings are stored L2-normalized; cosine == dot product.
    q = query
    qn = np.linalg.norm(q)
    if qn:
        q = q / qn
    return matrix @ q


def find_similar(
    study_id: str,
    top_k: int = 5,
    index: Optional[Index] = None,
) -> List[SimilarCase]:
    """Return top-k similar studies excluding the query itself."""
    if index is None:
        index = load_index()

    ids = index.study_ids
    if study_id not in ids:
        raise KeyError(f"study_id '{study_id}' not in index ({len(ids)} studies).")

    qi = ids.index(study_id)
    scores = _cosine_scores(index.embeddings[qi], index.embeddings)

    order = np.argsort(-scores)
    results: List[SimilarCase] = []
    for idx in order:
        if int(idx) == qi:
            continue
        s = index.studies[int(idx)]
        results.append(
            SimilarCase(
                study_id=s.study_id,
                score=float(scores[int(idx)]),
                image_path=s.primary_image,
                impression=s.impression or s.report_text[:120],
                labels=s.labels,
                study=s,
            )
        )
        if len(results) >= top_k:
            break
    return results
