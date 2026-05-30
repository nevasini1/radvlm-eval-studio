"""Evaluate report repair (before vs after) on held-out synthetic cases.

For each case we create a flawed draft, score it, repair it (trained adapter if
available, else template fallback), and score the repair. Results are aggregated
and saved. Numbers are real — never fabricated. When no adapter is trained, the
results are clearly labelled as a template fallback.

Research demo only. Not for diagnosis or clinical use.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from radvlm_eval import config
from radvlm_eval.evaluation.error_taxonomy import ClinicalError, compare_reports
from radvlm_eval.evaluation.metrics import compute_metrics
from radvlm_eval.schemas import Study
from radvlm_eval.training.error_augmenter import applicable_error_types, inject_error
from radvlm_eval.training.report_repair_model import repair_report, report_body

EVAL_JSON = config.OUTPUTS_DIR / "training" / "eval_results.json"
EVAL_CSV = config.OUTPUTS_DIR / "training" / "eval_cases.csv"


def _count_by_type(errors: List[ClinicalError], etype: str) -> int:
    return sum(1 for e in errors if e.error_type == etype)


def _high_sev(errors: List[ClinicalError]) -> int:
    return sum(1 for e in errors if e.severity == "high")


def _key(e: ClinicalError):
    return (e.error_type, e.finding)


def evaluate(
    studies: Optional[List[Study]] = None,
    adapter_path: Optional[Path] = None,
    seed: int = 29,
    limit: Optional[int] = None,
    save: bool = True,
) -> Dict:
    if studies is None:
        from radvlm_eval.data.load_studies import load_demo_dataset

        studies = load_demo_dataset()
    if limit:
        studies = studies[:limit]

    rng = random.Random(seed)
    cases: List[Dict] = []
    method_used = "template-fallback"

    agg = {
        "missed_finding": [0, 0],
        "hallucinated_finding": [0, 0],
        "negation_error": [0, 0],
        "laterality_mismatch": [0, 0],
        "high_severity_total": [0, 0],
        "total_errors": [0, 0],
    }
    f1_before_sum = 0.0
    f1_after_sum = 0.0
    new_errors_introduced = 0

    for study in studies:
        options = applicable_error_types(study)
        etype = rng.choice(options)
        aug = inject_error(study, etype, rng)
        if aug is None:
            continue

        flawed_errors = compare_reports(study.labels, study.report_text, aug.flawed_draft)
        flawed_metrics = compute_metrics(study.labels, study.report_text, aug.flawed_draft)

        repair = repair_report(study, aug.flawed_draft, errors=flawed_errors, adapter_path=adapter_path)
        method_used = str(repair["method"])
        # Score the clinical body only (exclude the "Changes made"/"Safety" footer).
        repaired_text = report_body(str(repair["repaired_text"]))

        repaired_errors = compare_reports(study.labels, study.report_text, repaired_text)
        repaired_metrics = compute_metrics(study.labels, study.report_text, repaired_text)

        # Aggregate before/after counts.
        for etype_name in ("missed_finding", "hallucinated_finding", "negation_error", "laterality_mismatch"):
            agg[etype_name][0] += _count_by_type(flawed_errors, etype_name)
            agg[etype_name][1] += _count_by_type(repaired_errors, etype_name)
        agg["high_severity_total"][0] += _high_sev(flawed_errors)
        agg["high_severity_total"][1] += _high_sev(repaired_errors)
        agg["total_errors"][0] += len(flawed_errors)
        agg["total_errors"][1] += len(repaired_errors)

        f1_before_sum += flawed_metrics["positive_f1"]
        f1_after_sum += repaired_metrics["positive_f1"]

        flawed_keys = {_key(e) for e in flawed_errors}
        introduced = [e for e in repaired_errors if _key(e) not in flawed_keys]
        new_errors_introduced += len(introduced)

        cases.append(
            {
                "study_id": study.study_id,
                "injected_error_type": aug.error_type,
                "severity": aug.severity,
                "errors_before": len(flawed_errors),
                "errors_after": len(repaired_errors),
                "f1_before": flawed_metrics["positive_f1"],
                "f1_after": repaired_metrics["positive_f1"],
                "method": method_used,
            }
        )

    n = max(1, len(cases))
    results = {
        "n_cases": len(cases),
        "method": method_used,
        "is_trained_adapter": method_used == "mlx-adapter",
        "label": "trained MLX adapter" if method_used == "mlx-adapter" else "template fallback (rule-based, not a trained model)",
        "aggregate": {
            k: {"before": v[0], "after": v[1], "delta": v[1] - v[0]} for k, v in agg.items()
        },
        "label_f1": {
            "before": round(f1_before_sum / n, 3),
            "after": round(f1_after_sum / n, 3),
        },
        "new_errors_introduced": new_errors_introduced,
        "disclaimer": "Research demo only. Not for diagnosis or clinical use. Synthetic data.",
    }

    if save:
        EVAL_JSON.parent.mkdir(parents=True, exist_ok=True)
        EVAL_JSON.write_text(json.dumps(results, indent=2))
        _write_cases_csv(cases, EVAL_CSV)
        results["saved"] = {"json": str(EVAL_JSON), "csv": str(EVAL_CSV)}

    results["cases"] = cases
    return results


def _write_cases_csv(cases: List[Dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["study_id", "injected_error_type", "severity", "errors_before",
              "errors_after", "f1_before", "f1_after", "method"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in cases:
            writer.writerow({k: c.get(k, "") for k in fields})
