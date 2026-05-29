# Technical Memo — RadVLM Eval Studio

> Research/demonstration prototype only. Not a medical device. Not for clinical use.

## 1. Problem

Radiology vision-language models (VLMs) can draft chest X-ray reports, but the bottleneck
for safe deployment is **evaluation and review**, not generation. The clinically dangerous
errors — a missed pneumothorax, a hallucinated effusion, a flipped laterality, an inverted
negation — are precisely the ones that lexical metrics (BLEU/ROUGE) fail to penalize. A
draft can be fluent, high-overlap, and *wrong*.

This project builds the evaluation + clinician-review layer that a radiology VLM product
needs, in a form that runs on commodity hardware.

## 2. Constraints

- Runs on an **Apple Silicon MacBook, 16 GB unified memory, no CUDA**.
- **No large-model training**; **no cloud API key**; **no auto-download** of credentialed data.
- **No committed** medical images, real reports, PHI, weights, or dataset files.
- Works **end-to-end on synthetic data immediately**.
- Optional dependencies (BiomedCLIP, mlx-vlm, RadGraph, CheXbert) must **degrade gracefully**.

## 3. Architecture

```
                ┌───────────────┐      ┌────────────────┐
  demo_data ───▶│ Study (canon) │────▶ │  Retrieval     │── similar cases ─┐
  importers ───▶│  schema       │      │  (embed+index) │                  │
                └───────────────┘      └────────────────┘                  ▼
                       │                                          ┌──────────────────┐
                       │                                          │ Draft Generator  │
                       │                                          │ template/retrieval/VLM
                       ▼                                          └──────────────────┘
                ┌───────────────┐      ┌────────────────┐                  │
                │ Report Labeler │◀────│   Evaluation   │◀── draft + ref ──┘
                │ (rule-based)   │     │ taxonomy+metrics│
                └───────────────┘      └────────────────┘
                                               │
                                       ┌────────────────┐     ┌──────────────┐
                                       │  Review (edit) │────▶│ SQLite audit │
                                       │ before/after   │     │     log      │
                                       └────────────────┘     └──────────────┘
```

Each layer is a small, typed Python module under `radvlm_eval/`. The Streamlit app is a
thin presentation layer over these libraries (everything is unit-testable without the UI).

## 4. Data schema

Canonical `Study` (dataclass, no pydantic dependency):

```
study_id: str
image_paths: list[str]
report_text: str
findings: str
impression: str
indication: str | None
labels: dict[str, int | None]   # 1 present / 0 absent / -1 uncertain / None not-mentioned
metadata: dict                  # patient_id_hash, split, view_position, source, ...
```

Labels follow the 14-observation CheXpert/MIMIC set. Studies persist to a single CSV
(`data/studies.csv`) with JSON-encoded list/dict columns, keeping I/O dependency-free.

## 5. Retrieval method

- **Fallback (default, always available):** 32-bin normalized grayscale image histogram
  ⊕ TF-IDF report-text vector (max 128 features), each L2-normalized, concatenated with
  text/image weights (0.7 / 0.3), then L2-normalized. Cosine similarity = dot product over
  the stored matrix. No FAISS (avoids Mac install friction); NumPy `argsort` is plenty at
  demo scale.
- **BiomedCLIP (optional):** `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` via
  `open_clip` + `torch` when installed; concatenates image + text embeddings. Any failure
  (missing package, no weights) automatically falls back.
- Artifacts: `outputs/index/embeddings.npy`, `studies.csv`, `index_meta.json`.

## 6. Draft-generation method

Three modes, all producing **structured, RSNA-style, conservative** drafts with a mandatory
"AI draft — requires radiologist review" disclaimer and explicit uncertainty/limitations:

- **template** — derives Findings / Impression / Uncertainty / follow-up questions from the
  study's structured labels. Always works.
- **retrieval** — majority-votes labels across the top similar prior cases, then renders the
  same scaffold. Conservative wording.
- **local_vlm** — optional on-device caption via `mlx-vlm`; wrapped in the structured,
  disclaimered scaffold. Falls back to retrieval/template when unavailable.

We follow the *spirit* of RSNA RadReport structured templating without copying any
proprietary template text.

## 7. Evaluation methodology

- **Weak labeler** (`report_labeler.py`): keyword + clause-scoped negation/uncertainty
  detection over the 14 observations. Sentences are split, then split again on
  `but`/`;`/`however` so negation scope is local. Mentions merge with precedence
  present > uncertain > absent. Transparent and auditable; **not** a trained model.
- **Metrics** (`metrics.py`): exact label agreement, positive precision/recall/F1,
  missed/hallucinated counts, uncertainty- and laterality-mismatch counts, report-length
  ratio, lexical overlap. All functions are **null-safe** (empty labels/text → 0.0/0).
- **Status**: green/yellow/red traffic light; escalates to **red** on any laterality error,
  ≥2 missed/hallucinated findings, or (in the UI) any single high-severity clinical error.

## 8. Error taxonomy

`error_taxonomy.py` compares reference labels+text against the draft and emits structured
`ClinicalError` records (type, finding, severity, reference/draft evidence sentence,
suggested mitigation):

| Type | Trigger |
|---|---|
| `missed_finding` | reference positive, draft not positive |
| `hallucinated_finding` | draft positive, reference not positive |
| `negation_error` | direct present↔absent contradiction (high severity) |
| `uncertainty_mismatch` | one side hedges (-1), other is definite |
| `laterality_mismatch` | left/right disagreement in the evidence sentences (high severity) |
| `severity_mismatch` | small/mild vs large/severe wording differs |
| `support_device_mismatch` | lines/tubes/devices disagreement |

High-risk findings (pneumothorax, pneumonia, consolidation, edema, effusion, lesion,
enlarged cardiomediastinum) raise severity. `diff_edits` re-evaluates a draft before vs
after an edit to report **errors fixed** and **errors introduced**.

## 9. Safety and dataset governance

- Every user-facing surface shows "Research demo only. Not for diagnosis or clinical use."
- Drafts carry "AI draft — requires radiologist review" and never claim a diagnosis.
- Synthetic demo images are abstract placeholders; demo data is flagged `synthetic: true`.
- Importers **never download**. They read a local copy, validate layout, and print
  instructions if files are missing. The MIMIC importer prints a non-redistribution
  warning. CheXpert label-derived summaries are explicitly marked as *not* original reports.
- `.gitignore` blocks `data/*`, `outputs/*`, `*.db`, `*.npy`, `*.parquet`, and weights.

## 10. Limitations

- The weak labeler is a heuristic and will miss paraphrases / complex negation; it is not
  CheXbert.
- Synthetic images are not anatomy; retrieval similarity on the demo set is dominated by
  text/labels.
- No model is trained or clinically validated. The contribution is the **evaluation +
  workflow architecture**, not model performance.
- RadGraph/RadCliQ and the MLX VLM are integration points, not fully wired in this build.

## 11. What I would improve with more compute / data

- Replace the rule-based labeler with **CheXbert**; add **RadGraph F1 / RadCliQ**.
- Real **BiomedCLIP** index over IU X-Ray / MIMIC-CXR-JPG, with approximate NN at scale.
- A genuine **local MLX VLM** drafting backend and latency/quality benchmarking.
- **Calibration** (confidence vs accuracy) and a **reader-study** simulation harness.
- DICOM ingestion + windowing; extension to CT / head-CT triage workflows.
