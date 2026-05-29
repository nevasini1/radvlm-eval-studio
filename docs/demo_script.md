# Demo Script — RadVLM Eval Studio (3–5 minutes)

> Research/demonstration prototype only. Not for diagnosis or clinical use.

**Setup before recording**

```bash
python scripts/make_demo_data.py
python scripts/build_index.py --dataset demo --embedding-backend fallback
streamlit run app/streamlit_app.py
```

---

### 0:00 — Mission (20s)

> "Radiology VLMs can draft reports, but the hard part is **evaluating** them the way a
> radiologist would. A draft can read beautifully and still miss a pneumothorax or flip
> left and right. RadVLM Eval Studio is a GPU-light workbench for exactly that —
> clinically meaningful evaluation and a clinician-in-the-loop review workflow. Everything
> you'll see runs locally on a 16 GB MacBook with synthetic data. It's a research demo, not
> a medical device."

Point to the persistent **"Research demo only"** banner.

### 0:20 — Study Viewer (40s)

- Open the **Study Viewer** tab. In the sidebar, select **`DEMO0005`** (a pneumothorax case).
- Show the synthetic image, the **indication**, **findings**, **impression**, and the
  **reference labels** table.

> "Each study has a structured report and the 14 standard CXR observation labels — present,
> absent, uncertain, or not mentioned."

### 1:00 — Similar Case Retrieval (40s)

- Go to **Similar Cases**. Show the top-k neighbors with similarity scores, thumbnails, and
  **shared findings**.

> "We retrieve the most similar prior cases with cosine similarity over a lightweight
> embedding — image histogram plus TF-IDF — so no GPU or big download is needed. There's an
> optional BiomedCLIP backend that swaps in transparently."

- Tick **"Use for draft"** on one or two neighbors.

### 1:40 — Generate a Draft (40s)

- Go to **Draft Report**. Choose **retrieval** mode, click **Generate draft**.

> "The draft is structured RSNA-style, deliberately conservative, and always carries an
> 'AI draft — requires radiologist review' disclaimer plus an explicit uncertainty section.
> It never claims a diagnosis."

- Now click **"Load seeded flawed draft."**

> "For the demo I'll load a draft with realistic errors so we can see the evaluation work."

### 2:20 — Evaluation: the star (60s)

- Go to **Evaluation**.

> "Here's the difference from a leaderboard. The status is **RED**."

- Walk the **metric cards**: F1, missed findings, hallucinated, laterality errors.
- Show the **reference-vs-draft label matrix**, then the **clinical error table**.
- Read the **failure-mode summary** aloud, e.g.:
  - *"Negation mismatch: the reference says moderate right pneumothorax; the draft says no
    pneumothorax — high severity."*

- Expand **"Why generic BLEU/ROUGE is insufficient."**

> "Flipping 'no pneumothorax' to 'pneumothorax' changes one token but inverts the meaning.
> Lexical metrics miss that; this taxonomy catches it and ranks it by clinical severity."

### 3:20 — Radiologist Edit (50s)

- Go to **Review & Audit Log**. Edit the draft text to correct the pneumothorax (e.g. add
  *"Moderate right pneumothorax with partial collapse."*).
- Click **Save edit**.

> "Now I'm the radiologist. I fix the draft and sign off."

- Show **before vs after**: errors before, errors after, **errors fixed** vs **introduced**,
  and the text-similarity delta.

### 4:10 — Audit Log + close (40s)

- Show the **Audit history** table: timestamp, study, action, F1 before/after, notes.
- Optionally click **Export case JSON**.

> "Every action is written to a local SQLite audit log — the foundation for QA, monitoring,
> and reader studies. That's what safe deployment of a drafting model actually requires:
> not just generation, but clinically meaningful evaluation, a human in the loop, and a
> defensible trail. And it all runs on a laptop, with responsible handling of real
> datasets when you bring your own."

### Close (10s)

> "Research demo only — not for clinical use. Thanks for watching."
