# 🩻 RadVLM Eval Studio

**A GPU-light evaluation and clinician-review workbench for radiology VLM draft reports.**

> ⚠️ **Research demo only. Not for diagnosis or clinical use.** This is not a medical
> device, has not been clinically validated, and ships with **synthetic data only** —
> no patient data, no PHI, no model weights.

---

## Why this matters

Radiology report generation models are improving fast, but **good text is not the same
as a clinically correct report**. A generated draft can score highly on BLEU/ROUGE while
flipping *"no pneumothorax"* into *"pneumothorax,"* missing a small pleural effusion, or
getting the **laterality** (left vs right) wrong. Those are the errors that actually harm
patients — and lexical metrics are blind to them.

RadVLM Eval Studio is built around the part that real deployment depends on: **clinically
meaningful evaluation, a clinician-in-the-loop review workflow, and failure-mode analysis** —
not large-model training. It runs end-to-end on a laptop.

## What it does

1. **Study Viewer** — load a chest X-ray study (image, indication, findings, impression, labels).
2. **Similar Case Retrieval** — find the most similar prior cases via cosine similarity.
3. **Draft Report Generation** — produce a structured, conservative, RSNA-style draft
   (template / retrieval / optional local VLM modes) with mandatory uncertainty + disclaimers.
4. **Clinical Evaluation** — compare the draft against the reference and surface a
   **clinical error taxonomy**: missed findings, hallucinations, negation errors,
   laterality/uncertainty/severity mismatches — with green/yellow/red status.
5. **Radiologist Review** — edit the draft, sign off or escalate, and see **before/after**
   metrics (errors fixed vs introduced).
6. **Audit Log** — every action is persisted to SQLite with timestamps and metrics.

## Features

- ✅ Runs **immediately** on synthetic demo data — no datasets, no API keys, no CUDA.
- ✅ **Clinical error taxonomy**, not just BLEU/ROUGE.
- ✅ Lightweight **retrieval** (image histogram + TF-IDF), with optional **BiomedCLIP**.
- ✅ Three **draft-generation** modes incl. optional on-device **MLX VLM**.
- ✅ **Clinician edit + sign-off** workflow with before/after diff.
- ✅ **SQLite audit log** + exportable case-review JSON.
- ✅ **Responsible dataset handling** — importers for IU X-Ray, MIMIC-CXR, CheXpert that
  read *your local copy* and never download or redistribute credentialed data.
- ✅ Graceful degradation: every optional dependency is optional.

## Screenshots

> Captured from the live app running on synthetic demo data.

**Evaluation dashboard** — the star of the project. A seeded draft that says *"no
pneumothorax"* on a study with a confirmed pneumothorax is flagged **RED** as a
high-severity `negation_error`, with the reference and draft evidence sentences side by side.

![Evaluation dashboard](docs/screenshots/evaluation.png)

| Study Viewer | Similar Cases |
|---|---|
| ![Study Viewer](docs/screenshots/study_viewer.png) | ![Similar Cases](docs/screenshots/similar_cases.png) |

| Draft Report | Review & Audit Log |
|---|---|
| ![Draft Report](docs/screenshots/draft_report.png) | ![Review & Audit Log](docs/screenshots/review.png) |

<details>
<summary>Overview tab</summary>

![Overview](docs/screenshots/overview.png)

</details>

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/make_demo_data.py
python scripts/build_index.py --dataset demo --embedding-backend fallback
streamlit run app/streamlit_app.py
```

Or just run `bash scripts/run_app.sh`.

The app auto-generates demo data + index on first launch if they are missing, so
`streamlit run app/streamlit_app.py` works on a fresh checkout too.

## Hardware

Tested target: **Apple Silicon MacBook, 16 GB unified memory, no CUDA GPU.** The default
retrieval backend uses a normalized image histogram + TF-IDF text vector, so nothing large
is downloaded. BiomedCLIP and a local MLX VLM are **optional**, **off by default**, and
the app degrades gracefully when they are absent.

## Dataset policy

This repository ships **only synthetic demo data**. Real datasets must be obtained by the
user under their own licenses / Data Use Agreements:

| Dataset | Access | Importer |
|---|---|---|
| IU X-Ray / Open-i | Open collection ([openi.nlm.nih.gov](https://openi.nlm.nih.gov/)) | `import_openi` |
| MIMIC-CXR / MIMIC-CXR-JPG | **Credentialed PhysioNet DUA** — never auto-downloaded or redistributed | `import_mimic` |
| CheXpert | Stanford license | `import_chexpert` |

Importers validate the local folder layout and print clear instructions if files are
missing. MIMIC/PhysioNet data **cannot be redistributed** and is never committed.

## Evaluation methodology

Instead of relying on lexical overlap, the evaluator scores what clinicians care about:

- **Exact label agreement** across the 14 CheXpert observations.
- **Positive-finding precision / recall / F1.**
- **Missed-finding** and **hallucinated-finding** counts.
- **Negation errors** (present↔absent contradictions).
- **Laterality mismatches** (left vs right) — clinically critical.
- **Uncertainty mismatches** (definite vs "possible/cannot exclude").
- **Severity mismatches** and **support-device** disagreements.
- Report length ratio + lexical overlap (for context, never as the primary signal).
- **Optional** entity-level metrics: RadGraph F1 / RadCliQ via `radgraph_adapter`
  (degrades to "optional dependency not installed" — never blocks the demo).

Labels are produced by a transparent **rule-based weak labeler** with clause-scoped
negation/uncertainty detection. It is intentionally auditable and documented as *not* a
substitute for a trained labeler such as **CheXbert** (optional hook provided).

## Why this is relevant to radiology VLM companies

Teams shipping radiology VLM **drafting/review** products (e.g. the space companies like
Cognita / Radiology Partners are building in) live or die on four things this project is
built around:

- **Failure-mode analysis over leaderboard scores.** The product risk is a *missed
  pneumothorax* or a *flipped laterality*, not a lower ROUGE. This tool quantifies exactly
  those clinical error classes and ranks them by severity.
- **Clinician-in-the-loop review.** Real deployment is draft → radiologist edit → sign-off.
  This implements that loop with before/after evaluation and an immutable audit trail —
  the substrate for monitoring, QA, and reader studies.
- **GPU-light engineering.** Not every workload needs an H100. Robust retrieval +
  evaluation that runs on a 16 GB MacBook shows pragmatic systems thinking and makes the
  eval harness cheap to run in CI.
- **Responsible dataset governance.** Credentialed data (MIMIC/PhysioNet) is never
  auto-downloaded or redistributed; synthetic data is clearly labeled; PHI never enters the
  repo. This is table stakes for clinical ML and is enforced in code, not just docs.

## Portfolio talking points

- Built a practical **evaluation framework**, not just another classifier.
- Designed for **clinician-in-the-loop** review with a full audit trail.
- Runs **locally on limited hardware** (Apple Silicon, no GPU).
- Handles **dataset governance** responsibly (DUAs, no redistribution, no PHI).
- Surfaces **clinically meaningful failure modes** that lexical metrics miss.

## Project layout

```
radvlm-eval-studio/
  app/streamlit_app.py          # Streamlit UI (7 tabs)
  radvlm_eval/
    config.py  schemas.py  safety.py
    data/        # demo generator + IU X-Ray / MIMIC / CheXpert importers
    retrieval/   # embedder (fallback + BiomedCLIP), index, similar_cases
    reporting/   # draft_generator, templates, local_vlm (mlx-vlm)
    evaluation/  # report_labeler, error_taxonomy, metrics, radgraph_adapter
    workflow/    # audit_log, review (before/after metrics)
    storage/     # sqlite
  scripts/       # make_demo_data, build_index, run_app.sh
  tests/         # pytest (demo data, retrieval, taxonomy, metrics, audit log)
  docs/          # technical_memo.md, demo_script.md
```

## Testing

```bash
python -m pytest
```

Covers: demo-data validity, retrieval top-k / self-exclusion, negation labeling
(`"No pneumothorax"` ≠ positive), missed/hallucinated-finding detection, null-safe
metrics, and SQLite audit read/write.

## Future work

- Wire in real **CheXbert** for labeling.
- Wire in **RadGraph / RadCliQ** entity-level metrics.
- Add a real **local MLX VLM** backend for on-device drafting.
- Add a **DICOM** viewer and windowing.
- Extend to **CT / head-CT** workflows.
- Add **calibration** analysis and **reader-study** simulation.

## Safety

**Research only. Not for clinical use.** Not validated. No patient data included.
See the **About / Safety** tab in the app and `LICENSE`.

## License

MIT — see [LICENSE](LICENSE). Research/demonstration prototype only; **not a medical device**.
