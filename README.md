# 🩻 RadVLM Eval Studio

[![CI](https://github.com/nevasini1/radvlm-eval-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/nevasini1/radvlm-eval-studio/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: research demo](https://img.shields.io/badge/status-research%20demo-orange.svg)](#safety)

**A GPU-light evaluation and clinician-review workbench for radiology VLM draft reports.**

> ⚠️ **Research demo only. Not for diagnosis or clinical use.** Not a medical device, not
> clinically validated, ships **synthetic data only** (no PHI, no weights). The value is the
> **evaluation + workflow architecture**, not clinical model performance.

A draft report can score highly on BLEU/ROUGE while flipping *"no pneumothorax"* → *"pneumothorax"*,
missing an effusion, or getting **laterality** wrong. This tool scores what clinicians care about —
missed findings, hallucinations, negation, laterality — and adds a clinician-in-the-loop review loop.

## Example failure it catches

```text
Reference : "Moderate right pneumothorax."
AI draft  : "No pneumothorax identified."
Detected  : negation_error  (severity: high)  →  flagged RED, fixed on radiologist edit (2 errors → 0)
```

## What it does

| Stage | Description |
|---|---|
| **Study Viewer** | Load a CXR study (image, indication, findings, impression, 14 labels). |
| **Similar Cases** | Retrieve nearest prior cases via cosine similarity. |
| **Draft Report** | Structured RSNA-style draft (template / retrieval; optional local-VLM adapter). |
| **Evaluation** | Clinical error taxonomy + green/yellow/red status (the star of the project). |
| **Review** | Edit, sign off / escalate, see before/after metrics (errors fixed vs introduced). |
| **Audit Log** | Every action persisted to SQLite with timestamps + metrics. |
| **Training Lab** | RadDPO-Lite — train a tiny LoRA adapter to *repair* flawed drafts (see below). |

**Runs immediately on synthetic data — no datasets, API keys, or CUDA.** Lightweight retrieval
(image histogram + TF-IDF) and a rule-based weak labeler work by default. BiomedCLIP, MLX-VLM,
RadGraph/RadCliQ, and CheXbert are **optional adapters with graceful fallback** — clean seams, not
fully wired in this build.

## Screenshots

**Evaluation dashboard** — a draft saying *"no pneumothorax"* on a confirmed pneumothorax is flagged
**RED** as a high-severity `negation_error`, with reference vs draft evidence side by side.

![Evaluation dashboard](docs/screenshots/evaluation.png)

| Study Viewer | Similar Cases | Draft Report | Review & Audit |
|---|---|---|---|
| ![](docs/screenshots/study_viewer.png) | ![](docs/screenshots/similar_cases.png) | ![](docs/screenshots/draft_report.png) | ![](docs/screenshots/review.png) |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_demo_data.py
python scripts/build_index.py --dataset demo --embedding-backend fallback
streamlit run app/streamlit_app.py        # or: bash scripts/run_app.sh
```

The app auto-generates demo data + index on first launch. **Tested target:** Apple Silicon MacBook,
16 GB unified memory, no CUDA.

## Evaluation methodology

Scores clinical correctness, not lexical overlap: exact label agreement (14 CheXpert observations),
positive precision/recall/F1, and counts of **missed / hallucinated / negation / laterality /
uncertainty / severity / support-device** errors. Lexical overlap is reported for context only.
Optional RadGraph F1 / RadCliQ via `radgraph_adapter` (degrades gracefully). Labels come from a
transparent rule-based weak labeler (not a substitute for CheXbert).

## Small-model training experiment: RadDPO-Lite

**Evaluator-generated clinical preference tuning for radiology draft repair.** The evaluator
manufactures the training signal: it injects clinical errors, detects them, and emits chosen/rejected
pairs. A tiny LoRA adapter then learns to *repair* flawed drafts. Text-side only — **not** a
diagnostic image model. Stage 1 is LoRA SFT (default); DPO-ready JSONL is also exported.

**Actually trained on a 16 GB Mac (no CUDA):**

| Setting | Value | | Before/after (36 studies) | Flawed | **Trained** | Fallback |
|---|---|---|---|---|---|---|
| Base model | `Qwen3-1.7B-4bit` | | Total clinical errors | 48 | **11** | 3 |
| Method | LoRA SFT, 300 iters | | High-severity errors | 25 | **5** | 0 |
| Examples | ~97 synthetic | | Missed findings | 9 | **0** | 0 |
| **Train loss** | **3.45 → 0.03** | | Negation errors | 5 | **0** | 0 |
| **Val loss** | **5.32 → 0.09** | | Mean positive-label F1 | 0.53 | **0.92** | 0.92 |
| Time / mem / size | ~6 min / 2.3 GB / 19 MB | | New errors introduced | — | **7** | 2 |

The adapter genuinely repairs reports via real inference (e.g. *"No pneumothorax"* →
*"Moderate right pneumothorax is present"*), driving missed/negation errors to zero. It introduces
some new errors — expected for a ~1.7 B model on ~100 synthetic examples. The rule-based **template
fallback** (used when no adapter is present) rebuilds from reference labels as an upper-bound baseline.

```bash
python scripts/make_training_pairs.py            # evaluator-derived SFT + DPO pairs
python scripts/train_report_repair_adapter.py --dry-run   # show LoRA command (works anywhere)
python scripts/train_report_repair_adapter.py --iters 300 # train (Apple Silicon: pip install "mlx-lm[train]")
python scripts/evaluate_report_repair_adapter.py # before/after repair metrics
```

Details: [`docs/training_experiment.md`](docs/training_experiment.md) ·
[`docs/raddpo_lite_method.md`](docs/raddpo_lite_method.md). Adapter weights are never committed.

## Dataset policy

Ships **synthetic data only**. Real datasets must be supplied locally under your own licenses/DUAs —
importers read your local copy and never download or redistribute.

| Dataset | Access | Importer |
|---|---|---|
| IU X-Ray / Open-i | Open ([openi.nlm.nih.gov](https://openi.nlm.nih.gov/)) | `import_openi` |
| MIMIC-CXR(-JPG) | **Credentialed PhysioNet DUA** — never redistributed | `import_mimic` |
| CheXpert | Stanford license | `import_chexpert` |

**Next validation step:** run the same workflow over a small local IU X-Ray / MIMIC-CXR-JPG subset
(100–500 studies) and report failure-mode counts vs RadGraph F1 / RadCliQ — without committing any
real data.

## Why this is relevant to radiology VLM companies

- **Failure-mode analysis over leaderboard scores** — quantifies missed pneumothorax / flipped
  laterality and ranks by severity, not ROUGE.
- **Clinician-in-the-loop** — draft → edit → sign-off with an immutable audit trail (QA / reader-study substrate).
- **GPU-light engineering** — robust retrieval + eval (and a real LoRA train) on a 16 GB MacBook.
- **Responsible dataset governance** — credentialed data never auto-downloaded; no PHI; enforced in code.

## Project layout

```
app/streamlit_app.py        # Streamlit UI (8 tabs)
radvlm_eval/
  data/        # demo generator + IU X-Ray / MIMIC / CheXpert importers
  retrieval/   # embedder (fallback + BiomedCLIP), index, similar_cases
  reporting/   # draft_generator, templates, local_vlm
  evaluation/  # report_labeler, error_taxonomy, metrics, radgraph_adapter
  training/    # RadDPO-Lite: error_augmenter, pair_builder, train/evaluate adapter
  workflow/ storage/   # audit_log, review · sqlite
scripts/  tests/  docs/
```

## Testing & CI

`python -m pytest` covers demo-data validity, retrieval, negation labeling, missed/hallucinated
detection, null-safe metrics, audit log, and training pairs. [GitHub Actions](.github/workflows/ci.yml)
runs the full acceptance path (make data → build index → training pairs → pytest) on Python 3.11 & 3.12.

## Future work

Real CheXbert · RadGraph/RadCliQ · local MLX-VLM drafting · DICOM viewer · CT/head-CT · calibration & reader-study simulation.

## Safety & License

**Research only. Not for clinical use.** Not validated. No patient data. MIT — see [LICENSE](LICENSE);
not a medical device.
