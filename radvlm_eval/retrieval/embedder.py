"""Embedding backends for similar-case retrieval.

Two backends:

* ``fallback`` (always works, no large downloads): a combined feature of a
  normalized grayscale image histogram + TF-IDF report-text vector. Robust on a
  16GB Apple Silicon Mac with no GPU.
* ``biomedclip`` (optional): microsoft/BiomedCLIP via open_clip if installed and
  the weights are available. Falls back automatically if not.

``auto`` tries biomedclip then falls back.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List, Tuple

import numpy as np

from radvlm_eval import config
from radvlm_eval.schemas import Study

IMG_HIST_BINS = 32
TEXT_MAX_FEATURES = 128
TEXT_WEIGHT = 0.7
IMAGE_WEIGHT = 0.3


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _image_histogram(image_path: str) -> np.ndarray:
    """Normalized grayscale histogram; zeros if the image is missing/unreadable."""
    vec = np.zeros(IMG_HIST_BINS, dtype=np.float32)
    if not image_path:
        return vec
    p = Path(image_path)
    if not p.is_absolute():
        p = config.REPO_ROOT / p
    if not p.exists():
        return vec
    try:
        from PIL import Image

        img = Image.open(p).convert("L")
        arr = np.asarray(img, dtype=np.float32).ravel()
        hist, _ = np.histogram(arr, bins=IMG_HIST_BINS, range=(0, 255))
        total = hist.sum()
        if total > 0:
            vec = (hist / total).astype(np.float32)
    except Exception:  # noqa: BLE001 - retrieval must never crash on a bad image
        pass
    return vec


def biomedclip_available() -> bool:
    return (
        importlib.util.find_spec("open_clip") is not None
        and importlib.util.find_spec("torch") is not None
    )


def resolve_backend(requested: str) -> str:
    """Decide the actual backend to use given the request and what's installed."""
    requested = (requested or "auto").lower()
    if requested == "fallback":
        return "fallback"
    if requested == "biomedclip":
        return "biomedclip" if biomedclip_available() else "fallback"
    # auto
    return "biomedclip" if biomedclip_available() else "fallback"


def _fallback_embeddings(studies: List[Study]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [
        " ".join([s.report_text or "", s.findings or "", s.impression or ""]).strip() or "empty"
        for s in studies
    ]
    vectorizer = TfidfVectorizer(max_features=TEXT_MAX_FEATURES, stop_words="english")
    try:
        text_mat = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    except ValueError:
        # e.g. empty vocabulary; fall back to zeros so the pipeline still runs.
        text_mat = np.zeros((len(studies), 1), dtype=np.float32)
    text_mat = _l2_normalize(text_mat)

    img_mat = np.vstack([
        _image_histogram(s.primary_image or "") for s in studies
    ]).astype(np.float32)
    img_mat = _l2_normalize(img_mat)

    combined = np.hstack([TEXT_WEIGHT * text_mat, IMAGE_WEIGHT * img_mat])
    return _l2_normalize(combined)


def _biomedclip_embeddings(studies: List[Study]) -> np.ndarray:  # pragma: no cover
    """Best-effort BiomedCLIP embeddings; raises to trigger fallback on any issue."""
    import open_clip
    import torch
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    tokenizer = open_clip.get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model.eval()

    feats = []
    with torch.no_grad():
        for s in studies:
            text = (s.impression or s.findings or s.report_text or "study")[:256]
            tokens = tokenizer([text])
            txt_feat = model.encode_text(tokens)
            img_path = s.primary_image
            if img_path:
                p = Path(img_path)
                if not p.is_absolute():
                    p = config.REPO_ROOT / p
                img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                img_feat = model.encode_image(img)
                feat = torch.cat([img_feat, txt_feat], dim=-1)
            else:
                feat = torch.cat([txt_feat, txt_feat], dim=-1)
            feats.append(feat.squeeze(0).cpu().numpy())
    mat = np.vstack(feats).astype(np.float32)
    return _l2_normalize(mat)


def build_embeddings(studies: List[Study], backend: str = "auto") -> Tuple[np.ndarray, str]:
    """Return (embedding_matrix, backend_used)."""
    resolved = resolve_backend(backend)
    if resolved == "biomedclip":
        try:
            return _biomedclip_embeddings(studies), "biomedclip"
        except Exception as exc:  # noqa: BLE001
            print(f"[embedder] BiomedCLIP unavailable ({exc}); using fallback.")
            resolved = "fallback"
    return _fallback_embeddings(studies), "fallback"
