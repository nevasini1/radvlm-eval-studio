"""Report-repair inference: trained MLX adapter if available, else template fallback.

The app and evaluation must never fail because a trained adapter is missing.

* If an adapter exists AND mlx-lm is installed, generate locally with the adapter.
* Otherwise, use a deterministic rule-based repair that corrects the draft toward
  the reference labels. Results from this path are labelled **"template-fallback"**
  (it is an honest rule-based baseline, not a trained model).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List, Optional

from radvlm_eval.evaluation.error_taxonomy import ClinicalError, compare_reports
from radvlm_eval.schemas import Study
from radvlm_eval.training.error_augmenter import build_reference_report
from radvlm_eval.training.train_adapter import DEFAULT_ADAPTER, DEFAULT_MODEL

DISCLAIMER = "AI draft — requires radiologist review."

# Footer lines that summarise the edit but are NOT clinical content. They must be
# excluded when scoring a repair, otherwise the labeler re-detects finding names
# echoed in "Changes made: corrected ... for Pneumothorax".
_FOOTER_MARKERS = ("Changes made:", "Safety:")


def report_body(text: str) -> str:
    """Return the clinical body (disclaimer/findings/impression/uncertainty) only.

    Drops the meta footer ("Changes made:", "Safety:") so scoring reflects the
    report's clinical content, not its edit summary.
    """
    out = []
    for line in (text or "").splitlines():
        if any(line.strip().startswith(m) for m in _FOOTER_MARKERS):
            break
        out.append(line)
    return "\n".join(out)


def _mlx_available() -> bool:
    return importlib.util.find_spec("mlx_lm") is not None


def _template_repair(study: Study, errors: List[ClinicalError]) -> str:
    """Deterministic repair: rebuild the structured report from reference labels."""
    corrected = build_reference_report(study)
    if errors:
        changes = "; ".join(f"corrected {e.error_type} for {e.finding}" for e in errors)
    else:
        changes = "no clinically significant errors detected"
    return (
        f"{DISCLAIMER}\n"
        f"{corrected}\n"
        "Uncertainty: Findings require radiologist confirmation against the image.\n"
        f"Changes made: {changes}.\n"
        "Safety: Research demo only. Not for diagnosis or clinical use."
    )


def _mlx_repair(  # pragma: no cover - only with mlx-lm + a trained adapter
    prompt: str, adapter_path: Path, model: str
) -> Optional[str]:
    try:
        from mlx_lm import generate, load

        mdl, tokenizer = load(model, adapter_path=str(adapter_path))

        # Apply the model's chat template so the prompt is in-distribution; the
        # adapter was trained on system/user/assistant chat turns. Without this
        # the base model degenerates into repetition.
        formatted = prompt
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:  # noqa: BLE001 - fall back to the raw prompt
            formatted = prompt

        # A repetition penalty further guards against runaway loops.
        gen_kwargs = {"max_tokens": 256, "verbose": False}
        try:
            from mlx_lm.sample_utils import make_logits_processors

            gen_kwargs["logits_processors"] = make_logits_processors(repetition_penalty=1.3)
        except Exception:  # noqa: BLE001 - optional across mlx-lm versions
            pass

        out = generate(mdl, tokenizer, prompt=formatted, **gen_kwargs)
        return str(out).strip() or None
    except Exception:  # noqa: BLE001
        return None


def repair_report(
    study: Study,
    flawed_draft: str,
    errors: Optional[List[ClinicalError]] = None,
    adapter_path: Optional[Path] = None,
    model: str = DEFAULT_MODEL,
    prompt: Optional[str] = None,
) -> Dict[str, object]:
    """Repair a flawed draft. Returns {repaired_text, method, changes}."""
    if errors is None:
        errors = compare_reports(study.labels, study.report_text, flawed_draft)

    adapter_path = Path(adapter_path) if adapter_path else DEFAULT_ADAPTER
    changes = [f"corrected {e.error_type} for {e.finding}" for e in errors]

    # Try the trained adapter only if it exists and MLX is installed.
    if prompt and adapter_path.exists() and _mlx_available():
        generated = _mlx_repair(prompt, adapter_path, model)
        if generated:
            return {"repaired_text": generated, "method": "mlx-adapter", "changes": changes}

    return {
        "repaired_text": _template_repair(study, errors),
        "method": "template-fallback",
        "changes": changes,
    }
