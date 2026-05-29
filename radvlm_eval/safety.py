"""Safety notices reused across the UI and generated artifacts.

Every user-facing surface should display a clear non-diagnostic disclaimer.
"""

from __future__ import annotations

SAFETY_SHORT = "Research demo only. Not for diagnosis or clinical use."

SAFETY_LONG = (
    "RadVLM Eval Studio is a research and demonstration prototype. It is **not** "
    "a medical device, has **not** been clinically validated, and must **not** be "
    "used for diagnosis, treatment, or any clinical decision-making. All studies "
    "shipped with this repository are synthetic and contain no patient data or PHI."
)

DRAFT_DISCLAIMER = "AI draft — requires radiologist review. Not for diagnosis."

DATASET_POLICY = (
    "This repository ships only synthetic demo data. Real datasets (IU X-Ray / "
    "Open-i, MIMIC-CXR / MIMIC-CXR-JPG, CheXpert) must be obtained by the user "
    "under their own licenses and Data Use Agreements. Credentialed PhysioNet "
    "datasets cannot be redistributed and are never auto-downloaded by this tool."
)


def banner() -> str:
    return SAFETY_SHORT
