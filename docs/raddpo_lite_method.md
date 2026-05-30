# RadDPO-Lite — Method Note

> Research/demonstration prototype only. Not for diagnosis or clinical use. Synthetic data.

## Name and positioning

**Evaluator-generated clinical preference tuning for radiology draft repair.**

This is deliberately *not* framed as a new training algorithm. It is a clean application of
established methods (LoRA/QLoRA SFT, optionally DPO) where the contribution is the
**clinical preference-pair generation pipeline** driven by an existing report evaluator.

## Pipeline

```
Reference Study (synthetic)
        │  canonical correct report  (labels + laterality/severity from text)
        ▼
error_augmenter ──inject one error──▶ flawed draft
        │                                   │
        │                                   ▼
        │                         evaluator (compare_reports)
        │                                   │ detected errors
        ▼                                   ▼
   pair_builder ── SFT: system/user/assistant(corrected) ──┐
        │       ── DPO: {prompt, chosen, rejected} ─────────┤
        ▼                                                    ▼
format_mlx_dataset ── JSONL ──▶ mlx_lm.lora (LoRA SFT)  /  DPO-ready export
        │
        ▼
report_repair_model ── trained adapter OR template fallback
        │
        ▼
evaluate_adapter ── before/after clinical error + F1 deltas
```

## Design choices

- **Errors are injected so the rule-based labeler detects them.** Each error class maps to a
  controlled text edit (e.g. negation → "No <finding>", laterality → swap left/right,
  severity → flip the modifier). This keeps the generated preferences self-consistent with
  the evaluator that scores them.
- **Scoring excludes the meta footer.** The repaired report ends with a "Changes made:" /
  "Safety:" footer for human readers; `report_body()` strips it before labeling so echoed
  finding names don't pollute the metrics.
- **Splits are study-level.** All examples for a given study land in one of
  train/valid/test to avoid leakage.
- **Two training stages.** Stage 1 (SFT) is the safe default and is enough to demonstrate
  small-model training on a Mac. Stage 2 (DPO) is optional: we always export DPO-ready JSONL
  even when no DPO trainer is installed.

## Error taxonomy used for preferences

`missed_finding`, `hallucinated_finding`, `negation_error`, `laterality_mismatch`,
`uncertainty_mismatch`, `severity_mismatch`, `support_device_mismatch` — severity escalated
for high-risk findings (pneumothorax, pneumonia, consolidation, edema, effusion, lesion,
enlarged cardiomediastinum) and for laterality/negation errors.

## What a hiring manager should take away

- Demonstrates the **post-training loop** end to end (data generation → SFT/DPO format →
  LoRA command → before/after evaluation) without overclaiming.
- Shows judgment: trains **text repair**, not a diagnostic image model; keeps everything
  GPU-light; labels fallback results honestly; never commits weights or real data.
- Connects to current radiology literature (DPO for hallucination suppression; RRG-DPO)
  while keeping the implementation appropriately scoped for a 16 GB Mac.

## Optional future extension: abnormality-aware retrieval adapter

A second GPU-light training story (not implemented in this build): train a small MLP adapter
on top of frozen embeddings with a triplet loss — anchor = current case, positive = same
key abnormality/laterality, negative = conflicting — inspired by CLIP-Adapter. This would
improve similar-case retrieval without touching the base encoder. The report-repair adapter
is the priority; this is listed as a clean next step.
