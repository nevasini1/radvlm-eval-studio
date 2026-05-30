"""Build evaluator-generated training pairs (SFT + DPO) from demo studies.

For each study we generate a few flawed drafts via `error_augmenter`, run the
existing clinical evaluator to detect the errors, and emit:

* an **SFT** example: chat messages (system, user, assistant=corrected report)
* a **preference** example: {prompt, chosen (corrected), rejected (flawed)}

All studies for a given study_id land in the same split to avoid leakage.

Research demo only. Not for diagnosis or clinical use.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from radvlm_eval.evaluation.error_taxonomy import ClinicalError, compare_reports
from radvlm_eval.schemas import CXR_LABELS, Study
from radvlm_eval.training.error_augmenter import AugmentedDraft, applicable_error_types, inject_error

SYSTEM_PROMPT = (
    "You repair AI-generated chest X-ray draft reports using evaluator-detected "
    "clinical errors. Rewrite conservatively, never invent findings, preserve "
    "uncertainty, and keep the structured format. Research demo only. "
    "Not for diagnosis or clinical use."
)

SPLIT_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}


@dataclass
class TrainingData:
    sft: Dict[str, List[dict]] = field(default_factory=lambda: {"train": [], "valid": [], "test": []})
    dpo: Dict[str, List[dict]] = field(default_factory=lambda: {"train": [], "valid": [], "test": []})

    def stats(self) -> Dict[str, int]:
        return {
            "sft_total": sum(len(v) for v in self.sft.values()),
            "sft_train": len(self.sft["train"]),
            "sft_valid": len(self.sft["valid"]),
            "sft_test": len(self.sft["test"]),
            "dpo_total": sum(len(v) for v in self.dpo.values()),
            "dpo_train": len(self.dpo["train"]),
            "dpo_valid": len(self.dpo["valid"]),
            "dpo_test": len(self.dpo["test"]),
        }


def _labels_summary(study: Study) -> str:
    parts = []
    name = {1: "present", 0: "absent", -1: "uncertain"}
    for lab in CXR_LABELS:
        v = study.labels.get(lab)
        if v in (1, 0, -1):
            parts.append(f"{lab}={name[v]}")
    return ", ".join(parts) if parts else "no labels mentioned"


def _similar_snippets(study: Study, index, k: int) -> str:
    if index is None or k <= 0:
        return "none"
    try:
        from radvlm_eval.retrieval.similar_cases import find_similar

        sims = find_similar(study.study_id, top_k=k, index=index)
        snippets = [f"[{c.study_id}] {c.impression}" for c in sims if c.impression]
        return " | ".join(snippets) if snippets else "none"
    except Exception:  # noqa: BLE001 - retrieval is optional context
        return "none"


def _errors_summary(errors: List[ClinicalError]) -> str:
    if not errors:
        return "no clinically significant errors detected"
    return "; ".join(f"{e.severity} severity {e.error_type} for {e.finding}" for e in errors)


def _user_prompt(study: Study, aug: AugmentedDraft, errors: List[ClinicalError], similar: str) -> str:
    return (
        f"Indication: {study.indication or 'not provided'}\n"
        f"Reference labels: {_labels_summary(study)}\n"
        f"Similar cases: {similar}\n"
        f"Flawed AI draft:\n{aug.flawed_draft}\n"
        f"Evaluator-detected errors: {_errors_summary(errors)}\n"
        "Rewrite the draft conservatively, correcting the detected errors. "
        "Keep FINDINGS and IMPRESSION sections, preserve genuine uncertainty, "
        "and do not introduce unsupported findings."
    )


def _changes_made(errors: List[ClinicalError], aug: AugmentedDraft) -> str:
    if errors:
        return "; ".join(f"corrected {e.error_type} for {e.finding}" for e in errors)
    return f"corrected {aug.error_type} for {aug.target_finding or 'finding'}"


def _chosen_response(aug: AugmentedDraft, errors: List[ClinicalError]) -> str:
    return (
        "AI draft — requires radiologist review.\n"
        f"{aug.correct_report}\n"
        "Uncertainty: Findings require radiologist confirmation against the image.\n"
        f"Changes made: {_changes_made(errors, aug)}.\n"
        "Safety: Research demo only. Not for diagnosis or clinical use."
    )


def _rejected_response(aug: AugmentedDraft) -> str:
    return aug.flawed_draft


def _split_for(idx: int, n: int) -> str:
    # Deterministic contiguous split by position in the shuffled study list.
    train_end = int(n * SPLIT_RATIOS["train"])
    valid_end = train_end + max(1, int(n * SPLIT_RATIOS["valid"])) if n >= 10 else train_end
    if idx < train_end:
        return "train"
    if idx < valid_end:
        return "valid"
    return "test"


def build_pairs(
    studies: List[Study],
    n_flawed_per_study: int = 2,
    seed: int = 13,
    index=None,
    similar_k: int = 2,
) -> TrainingData:
    rng = random.Random(seed)
    order = list(range(len(studies)))
    rng.shuffle(order)

    data = TrainingData()
    n = len(order)

    for pos, study_idx in enumerate(order):
        study = studies[study_idx]
        split = _split_for(pos, n)
        options = applicable_error_types(study)
        rng.shuffle(options)
        chosen_types = options[: max(1, min(n_flawed_per_study, len(options)))]

        similar = _similar_snippets(study, index, similar_k)

        for etype in chosen_types:
            aug = inject_error(study, etype, rng)
            if aug is None:
                continue
            errors = compare_reports(study.labels, study.report_text, aug.flawed_draft)
            user = _user_prompt(study, aug, errors, similar)
            chosen = _chosen_response(aug, errors)
            rejected = _rejected_response(aug)

            data.sft[split].append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": chosen},
                    ],
                    "meta": {
                        "study_id": study.study_id,
                        "error_type": aug.error_type,
                        "target_finding": aug.target_finding,
                        "severity": aug.severity,
                    },
                }
            )
            data.dpo[split].append(
                {
                    "prompt": f"{SYSTEM_PROMPT}\n\n{user}",
                    "chosen": chosen,
                    "rejected": rejected,
                    "meta": {
                        "study_id": study.study_id,
                        "error_type": aug.error_type,
                        "severity": aug.severity,
                    },
                }
            )

    return data
