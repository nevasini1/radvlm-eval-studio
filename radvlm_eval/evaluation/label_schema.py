"""Keyword and negation vocabulary used by the rule-based report labeler.

These maps are intentionally simple and auditable. They are NOT a substitute for
a trained labeler such as CheXbert; see report_labeler.py for documented limits.
"""

from __future__ import annotations

from typing import Dict, List

# Surface phrases that indicate each observation. Lowercased, substring-matched.
LABEL_KEYWORDS: Dict[str, List[str]] = {
    "No Finding": ["no acute", "normal", "no cardiopulmonary", "unremarkable", "clear"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "enlarged cardiac", "cardiac enlargement"],
    "Edema": ["edema", "vascular congestion", "interstitial fluid", "pulmonary congestion"],
    "Consolidation": ["consolidation", "consolidative"],
    "Atelectasis": ["atelectasis", "atelectatic", "volume loss"],
    "Pleural Effusion": ["pleural effusion", "effusion", "pleural fluid"],
    "Pneumothorax": ["pneumothorax", "ptx"],
    "Lung Opacity": ["opacity", "opacities", "airspace disease", "infiltrate"],
    "Lung Lesion": ["mass", "nodule", "lesion"],
    "Fracture": ["fracture", "fractured"],
    "Support Devices": [
        "endotracheal tube", "et tube", "central venous catheter", "central line",
        "picc", "pacemaker", "chest tube", "ng tube", "nasogastric", "catheter",
        "line tip", "sternal wire",
    ],
    "Enlarged Cardiomediastinum": ["mediastinal widening", "widened mediastinum", "enlarged cardiomediastin"],
    "Pleural Other": ["pleural thickening", "pleural calcification", "fibrothorax"],
    "Pneumonia": ["pneumonia", "infectious process", "infection"],
}

# Negation cues that flip a finding to absent (0).
NEGATION_CUES: List[str] = [
    "no ", "no evidence of", "without", "negative for", "absence of", "resolved",
    "free of", "not identified", "no significant", "rule out", "ruled out",
]

# Uncertainty cues that mark a finding uncertain (-1).
UNCERTAINTY_CUES: List[str] = [
    "possible", "possibly", "may represent", "cannot exclude", "cannot be excluded",
    "suspicious for", "questionable", "likely", "probable", "concerning for",
    "indeterminate", "versus", "differential", "could represent", "suggestive of",
    "equivocal",
]

# Laterality terms for laterality-mismatch detection.
LATERALITY_TERMS: List[str] = ["left", "right", "bilateral", "biliateral"]
