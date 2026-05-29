"""Rule-based weak labeler for the 14 CXR observations.

Detects present (1) / absent (0) / uncertain (-1) / not-mentioned (None) using
keyword matching with clause-scoped negation and uncertainty cues.

Limitations (documented intentionally):
- This is a transparent heuristic, not a trained model. It will miss paraphrases
  and complex negation. It is suitable for a demo / weak supervision, NOT for
  clinical labeling.
- For higher fidelity, install CheXbert and use `label_report(..., backend="chexbert")`
  (optional hook; falls back to rules if unavailable).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from radvlm_eval.evaluation.label_schema import (
    LABEL_KEYWORDS,
    NEGATION_CUES,
    UNCERTAINTY_CUES,
)
from radvlm_eval.schemas import CXR_LABELS, LabelValue, empty_labels

_SENTENCE_SPLIT = re.compile(r"[.!?\n]+")
_CLAUSE_SPLIT = re.compile(r"\bbut\b|;|\bhowever\b")


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _clauses(sentence: str) -> List[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(sentence) if c.strip()]


def _has_negation_before(clause: str, keyword_pos: int) -> bool:
    prefix = clause[:keyword_pos]
    return any(cue in prefix for cue in NEGATION_CUES)


def _has_uncertainty(clause: str) -> bool:
    return any(cue in clause for cue in UNCERTAINTY_CUES)


def _merge(existing: LabelValue, new: LabelValue) -> LabelValue:
    """Combine mentions: present(1) > uncertain(-1) > absent(0) > None."""
    order = {None: 0, 0: 1, -1: 2, 1: 3}
    if existing is None:
        return new
    return existing if order[existing] >= order[new] else new


def _rule_label(report_text: str) -> Dict[str, LabelValue]:
    labels: Dict[str, LabelValue] = empty_labels()
    text = report_text.lower()

    for sentence in _sentences(text):
        for clause in _clauses(sentence):
            uncertain = _has_uncertainty(clause)
            for label, keywords in LABEL_KEYWORDS.items():
                for kw in keywords:
                    pos = clause.find(kw)
                    if pos == -1:
                        continue
                    if _has_negation_before(clause, pos):
                        value: LabelValue = 0
                    elif uncertain:
                        value = -1
                    else:
                        value = 1
                    labels[label] = _merge(labels[label], value)
                    break  # one hit per label per clause is enough

    _postprocess_no_finding(labels)
    return labels


def _postprocess_no_finding(labels: Dict[str, LabelValue]) -> None:
    positives = [
        lab for lab in CXR_LABELS
        if lab != "No Finding" and labels.get(lab) == 1
    ]
    if positives:
        # If there are real positive findings, "No Finding" cannot be present.
        labels["No Finding"] = 0 if labels.get("No Finding") == 1 else labels["No Finding"]


def label_report(report_text: str, backend: str = "rules") -> Dict[str, LabelValue]:
    """Label a report. backend='rules' (default) or 'chexbert' (optional)."""
    if backend == "chexbert":
        try:
            return _chexbert_label(report_text)
        except Exception:  # noqa: BLE001 - graceful fallback
            pass
    return _rule_label(report_text)


def _chexbert_label(report_text: str) -> Dict[str, LabelValue]:  # pragma: no cover
    """Optional CheXbert hook. Raises if CheXbert is not installed/available."""
    raise NotImplementedError(
        "CheXbert backend not wired in this demo. Install CheXbert "
        "(https://github.com/stanfordmlgroup/CheXbert) and implement the adapter "
        "here. Falling back to rule-based labeling."
    )


def evidence_sentence(report_text: str, label: str) -> Optional[str]:
    """Return the first original-case sentence mentioning any keyword for `label`."""
    keywords = LABEL_KEYWORDS.get(label, [])
    for sentence in _sentences(report_text):
        low = sentence.lower()
        if any(kw in low for kw in keywords):
            return sentence.strip()
    return None
