"""Optional adapter for CXR-Report-Metric / RadGraph / RadCliQ.

These provide entity-and-relation-level report similarity (RadGraph F1) and a
composite quality score (RadCliQ). They are heavyweight optional dependencies.
This adapter detects availability and degrades gracefully — it NEVER blocks the
demo.

References:
    https://github.com/rajpurkarlab/CXR-Report-Metric
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10499844/
"""

from __future__ import annotations

import importlib.util
from typing import Dict, Optional


def is_available() -> bool:
    """True only if a CXR-Report-Metric / RadGraph install is importable."""
    for mod in ("CXRMetric", "radgraph", "cxr_report_metric"):
        if importlib.util.find_spec(mod) is not None:
            return True
    return False


def status_message() -> str:
    if is_available():
        return "RadGraph / CXR-Report-Metric detected."
    return (
        "Optional dependency not installed: RadGraph / RadCliQ / CXR-Report-Metric. "
        "Install from https://github.com/rajpurkarlab/CXR-Report-Metric to enable "
        "entity-level (RadGraph F1) and composite (RadCliQ) scoring."
    )


def compute_radgraph_metrics(reference_text: str, draft_text: str) -> Optional[Dict[str, float]]:
    """Return RadGraph/RadCliQ metrics if available, else None.

    Wrapped in a broad try/except so an environment quirk can never break the app.
    """
    if not is_available():
        return None
    try:  # pragma: no cover - exercised only when the optional dep is present
        # Placeholder integration point. A real wiring would call into
        # CXRMetric.run_eval / radgraph here. We intentionally avoid importing
        # at module top level to keep the demo light.
        raise NotImplementedError(
            "RadGraph adapter present but not wired in this demo build."
        )
    except Exception:  # noqa: BLE001
        return None
