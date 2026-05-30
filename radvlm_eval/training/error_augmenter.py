"""Inject clinically meaningful errors into a correct synthetic report.

Given a reference `Study`, we build a canonical *correct* report from its labels
(capturing laterality/severity from the reference text), then deliberately
perturb it to produce a flawed draft for one of the evaluator's error classes:

    missed_finding, hallucinated_finding, negation_error, laterality_mismatch,
    uncertainty_mismatch, severity_mismatch, support_device_mismatch

The flawed drafts are constructed so the existing rule-based evaluator
(`evaluation.error_taxonomy.compare_reports`) actually detects the intended class.

Research demo only. Not for diagnosis or clinical use.
"""

from __future__ import annotations

import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from radvlm_eval.evaluation.error_taxonomy import HIGH_RISK_LABELS
from radvlm_eval.evaluation.report_labeler import evidence_sentence
from radvlm_eval.schemas import CXR_LABELS, Study

# Concrete nouns chosen so the rule-based labeler reliably detects them.
NOUNS: Dict[str, str] = {
    "No Finding": "acute cardiopulmonary abnormality",
    "Cardiomegaly": "cardiomegaly",
    "Edema": "pulmonary edema",
    "Consolidation": "consolidation",
    "Atelectasis": "atelectasis",
    "Pleural Effusion": "pleural effusion",
    "Pneumothorax": "pneumothorax",
    "Lung Opacity": "lung opacity",
    "Lung Lesion": "lung nodule",
    "Fracture": "rib fracture",
    "Support Devices": "central venous catheter",
    "Enlarged Cardiomediastinum": "mediastinal widening",
    "Pleural Other": "pleural thickening",
    "Pneumonia": "pneumonia",
}

LATERALIZABLE = {
    "Pneumothorax", "Pleural Effusion", "Atelectasis", "Consolidation",
    "Lung Opacity", "Lung Lesion", "Fracture", "Pneumonia", "Pleural Other",
}

# Ordered low->high so we can flip severity convincingly.
SEVERITY_WORDS = ["trace", "tiny", "minimal", "small", "mild", "moderate", "large", "severe", "extensive", "marked"]
SEVERITY_FLIP = {
    "trace": "large", "tiny": "large", "minimal": "severe", "small": "large",
    "mild": "severe", "moderate": "small", "large": "small", "severe": "mild",
    "extensive": "minimal", "marked": "mild",
}

# Candidate findings to hallucinate (high-salience, lateralizable where possible).
HALLUCINATION_CANDIDATES = ["Pneumothorax", "Pleural Effusion", "Consolidation", "Lung Lesion"]

ALL_ERROR_TYPES = [
    "missed_finding", "hallucinated_finding", "negation_error",
    "laterality_mismatch", "uncertainty_mismatch", "severity_mismatch",
    "support_device_mismatch",
]


@dataclass
class AugmentedDraft:
    flawed_draft: str
    correct_report: str
    error_type: str
    target_finding: str
    severity: str  # low / medium / high
    explanation: str

    def to_dict(self) -> Dict:
        return {
            "flawed_draft": self.flawed_draft,
            "correct_report": self.correct_report,
            "error_type": self.error_type,
            "target_finding": self.target_finding,
            "severity": self.severity,
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# Canonical report construction
# ---------------------------------------------------------------------------
def _modifiers(study: Study, label: str) -> Tuple[Optional[str], Optional[str]]:
    sent = (evidence_sentence(study.report_text, label) or "").lower()
    lat = None
    if re.search(r"\bleft\b", sent):
        lat = "left"
    elif re.search(r"\bright\b", sent):
        lat = "right"
    elif "bilateral" in sent:
        lat = "bilateral"
    sev = next((w for w in SEVERITY_WORDS if re.search(rf"\b{w}\b", sent)), None)
    return lat, sev


def _present_sentence(label: str, lat: Optional[str] = None, sev: Optional[str] = None) -> str:
    noun = NOUNS[label]
    mods: List[str] = []
    if sev:
        mods.append(sev)
    if lat and label in LATERALIZABLE:
        mods.append(lat)
    phrase = " ".join(mods + [noun])
    return phrase[0].upper() + phrase[1:] + " is present."


def _negated_sentence(label: str) -> str:
    return f"No {NOUNS[label]}."


def _uncertain_sentence(label: str, lat: Optional[str] = None) -> str:
    noun = NOUNS[label]
    side = f"{lat} " if (lat and label in LATERALIZABLE) else ""
    return f"Possible {side}{noun}."


def _finding_map(study: Study) -> "OrderedDict[str, Tuple[str, str, Optional[str], Optional[str]]]":
    """label -> (status, sentence, laterality, severity) for the canonical correct report."""
    fmap: "OrderedDict[str, Tuple[str, str, Optional[str], Optional[str]]]" = OrderedDict()
    for label in CXR_LABELS:
        if label == "No Finding":
            continue
        v = study.labels.get(label)
        lat, sev = _modifiers(study, label)
        if v == 1:
            fmap[label] = ("present", _present_sentence(label, lat, sev), lat, sev)
        elif v == -1:
            fmap[label] = ("uncertain", _uncertain_sentence(label, lat), lat, sev)
    return fmap


def _assemble(fmap: "OrderedDict[str, Tuple[str, str, Optional[str], Optional[str]]]") -> str:
    findings = [t[1] for t in fmap.values()]
    if not findings:
        findings = ["The lungs are clear without pneumothorax, effusion, or consolidation."]
    present_nouns = [NOUNS[lab] for lab, (status, *_rest) in fmap.items() if status == "present"]
    if present_nouns:
        impression = " ".join(f"{n[0].upper() + n[1:]}." for n in present_nouns)
    else:
        impression = "No acute cardiopulmonary abnormality."
    return f"FINDINGS: {' '.join(findings)}\n\nIMPRESSION: {impression}"


def build_reference_report(study: Study) -> str:
    """The canonical *correct* structured report derived from reference labels."""
    return _assemble(_finding_map(study))


# ---------------------------------------------------------------------------
# Error injection
# ---------------------------------------------------------------------------
def _severity_of(error_type: str, label: str) -> str:
    high_risk = label in HIGH_RISK_LABELS
    if error_type == "laterality_mismatch":
        return "high"
    if error_type in ("negation_error", "missed_finding", "hallucinated_finding"):
        return "high" if high_risk else "medium"
    if error_type == "support_device_mismatch":
        return "medium"
    if error_type == "uncertainty_mismatch":
        return "medium" if high_risk else "low"
    if error_type == "severity_mismatch":
        return "low"
    return "medium"


def applicable_error_types(study: Study) -> List[str]:
    fmap = _finding_map(study)
    positives = [lab for lab, (s, *_r) in fmap.items() if s == "present"]
    uncertains = [lab for lab, (s, *_r) in fmap.items() if s == "uncertain"]
    types: List[str] = ["hallucinated_finding"]  # always possible
    if positives:
        types += ["missed_finding", "negation_error"]
    if any(lat for lab, (s, sent, lat, sev) in fmap.items() if s == "present" and lat in ("left", "right")):
        types.append("laterality_mismatch")
    if any(sev for lab, (s, sent, lat, sev) in fmap.items() if s == "present" and sev):
        types.append("severity_mismatch")
    if uncertains:
        types.append("uncertainty_mismatch")
    if study.labels.get("Support Devices") == 1:
        types.append("support_device_mismatch")
    return types


def inject_error(
    study: Study,
    error_type: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> Optional[AugmentedDraft]:
    """Produce a flawed draft for `error_type` (or a random applicable one)."""
    rng = rng or random.Random(0)
    fmap = _finding_map(study)
    options = applicable_error_types(study)

    if error_type is None:
        error_type = rng.choice(options)
    elif error_type not in options:
        # Requested type not applicable to this study; fall back gracefully.
        error_type = rng.choice(options)

    correct = _assemble(fmap)
    flawed = OrderedDict(fmap)
    target = ""
    explanation = ""

    positives = [lab for lab, (s, *_r) in fmap.items() if s == "present"]
    uncertains = [lab for lab, (s, *_r) in fmap.items() if s == "uncertain"]

    if error_type == "missed_finding":
        target = rng.choice(positives)
        del flawed[target]
        explanation = f"Reference reports {NOUNS[target]}; flawed draft omits it."

    elif error_type == "negation_error":
        target = rng.choice(positives)
        flawed[target] = ("absent", _negated_sentence(target), None, None)
        explanation = f"Reference reports {NOUNS[target]}; flawed draft explicitly negates it."

    elif error_type == "hallucinated_finding":
        candidates = [c for c in HALLUCINATION_CANDIDATES if c not in fmap]
        if not candidates:
            candidates = [c for c in CXR_LABELS if c not in fmap and c != "No Finding"]
        target = candidates[0] if candidates else "Pneumothorax"
        flawed[target] = ("present", _present_sentence(target), None, None)
        explanation = f"Flawed draft asserts {NOUNS[target]} that the reference does not support."

    elif error_type == "laterality_mismatch":
        cands = [lab for lab, (s, sent, lat, sev) in fmap.items() if s == "present" and lat in ("left", "right")]
        target = rng.choice(cands)
        _, _, lat, sev = fmap[target]
        flipped = "left" if lat == "right" else "right"
        flawed[target] = ("present", _present_sentence(target, flipped, sev), flipped, sev)
        explanation = f"Reference localizes {NOUNS[target]} to the {lat}; flawed draft says {flipped}."

    elif error_type == "severity_mismatch":
        cands = [lab for lab, (s, sent, lat, sev) in fmap.items() if s == "present" and sev]
        target = rng.choice(cands)
        _, _, lat, sev = fmap[target]
        new_sev = SEVERITY_FLIP.get(sev, "large")
        flawed[target] = ("present", _present_sentence(target, lat, new_sev), lat, new_sev)
        explanation = f"Reference severity for {NOUNS[target]} is '{sev}'; flawed draft says '{new_sev}'."

    elif error_type == "uncertainty_mismatch":
        target = rng.choice(uncertains)
        flawed[target] = ("absent", _negated_sentence(target), None, None)
        explanation = (
            f"Reference is uncertain about {NOUNS[target]} (hedged); flawed draft states it is absent."
        )

    elif error_type == "support_device_mismatch":
        target = "Support Devices"
        if target in flawed:
            del flawed[target]
        explanation = "Reference notes a support device; flawed draft omits the line/tube."

    flawed_text = _assemble(flawed)
    return AugmentedDraft(
        flawed_draft=flawed_text,
        correct_report=correct,
        error_type=error_type,
        target_finding=target,
        severity=_severity_of(error_type, target),
        explanation=explanation,
    )
