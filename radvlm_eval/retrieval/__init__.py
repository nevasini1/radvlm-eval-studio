"""Retrieval: embeddings, index building, and similar-case lookup."""

from radvlm_eval.retrieval.embedder import build_embeddings, resolve_backend
from radvlm_eval.retrieval.index import build_index, load_index
from radvlm_eval.retrieval.similar_cases import find_similar

__all__ = [
    "build_embeddings",
    "resolve_backend",
    "build_index",
    "load_index",
    "find_similar",
]
