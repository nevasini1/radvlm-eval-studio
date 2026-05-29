"""RadVLM Eval Studio — Streamlit UI.

A GPU-light radiology VLM evaluation and clinician-review prototype.
Research demo only. Not for diagnosis or clinical use.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

# Make the package importable when launched via `streamlit run app/streamlit_app.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from radvlm_eval import config, safety  # noqa: E402
from radvlm_eval.data.demo_data import generate_demo_dataset  # noqa: E402
from radvlm_eval.data.load_studies import save_studies_csv  # noqa: E402
from radvlm_eval.evaluation.error_taxonomy import compare_reports, diff_edits  # noqa: E402
from radvlm_eval.evaluation.metrics import compute_metrics, status_color  # noqa: E402
from radvlm_eval.evaluation.radgraph_adapter import status_message as radgraph_status  # noqa: E402
from radvlm_eval.evaluation.report_labeler import label_report  # noqa: E402
from radvlm_eval.reporting import local_vlm  # noqa: E402
from radvlm_eval.reporting.draft_generator import generate_draft  # noqa: E402
from radvlm_eval.retrieval.index import build_index, index_exists, load_index  # noqa: E402
from radvlm_eval.retrieval.similar_cases import find_similar  # noqa: E402
from radvlm_eval.schemas import CXR_LABELS, Study  # noqa: E402
from radvlm_eval.workflow.audit_log import list_audit_entries  # noqa: E402
from radvlm_eval.workflow.review import export_case_review_json, review_edit  # noqa: E402

st.set_page_config(page_title="RadVLM Eval Studio", page_icon="🩻", layout="wide")

LABEL_DISPLAY = {1: "✅ present", 0: "⛔ absent", -1: "❓ uncertain", None: "– n/a"}
STATUS_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴"}


# ---------------------------------------------------------------------------
# Data / index loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing demo data and index…")
def _ensure_ready():
    """Generate demo data + index on first run so the app always loads."""
    config.ensure_dirs()
    if not config.STUDIES_CSV.exists():
        studies = generate_demo_dataset()
        save_studies_csv(studies, config.STUDIES_CSV)
    if not index_exists():
        from radvlm_eval.data.load_studies import load_studies_csv

        studies = load_studies_csv(config.STUDIES_CSV)
        build_index(studies, backend="fallback")
    return load_index()


def _resolve_image(path: str) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = config.REPO_ROOT / p
    return p if p.exists() else None


def _study_by_id(studies: List[Study], sid: str) -> Study:
    return next(s for s in studies if s.study_id == sid)


def _label_matrix(ref_labels: Dict, draft_labels: Dict) -> pd.DataFrame:
    rows = []
    for lab in CXR_LABELS:
        r, d = ref_labels.get(lab), draft_labels.get(lab)
        if r is None and d is None:
            continue
        rows.append(
            {
                "Observation": lab,
                "Reference": LABEL_DISPLAY[r],
                "Draft": LABEL_DISPLAY[d],
                "Match": "✓" if r == d else "✗",
            }
        )
    return pd.DataFrame(rows)


def _safety_banner():
    st.warning(f"⚠️ {safety.SAFETY_SHORT}", icon="⚠️")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
index = _ensure_ready()
studies = index.studies
study_ids = index.study_ids

# Session state defaults
st.session_state.setdefault("drafts", {})          # study_id -> current draft text
st.session_state.setdefault("draft_meta", {})      # study_id -> meta
st.session_state.setdefault("context_ids", {})     # study_id -> list of context ids

st.title("🩻 RadVLM Eval Studio")
st.caption(
    "A GPU-light evaluation and clinician-review workbench for radiology VLM draft reports."
)
_safety_banner()

# Sidebar
with st.sidebar:
    st.header("Study")
    selected_id = st.selectbox("Select study", study_ids, index=0)
    study = _study_by_id(studies, selected_id)
    st.markdown(f"**Backend:** `{index.backend}`")
    st.markdown(f"**Studies indexed:** {len(studies)}")
    st.divider()
    st.caption("Optional backends")
    st.caption(f"Local VLM: {'✅' if local_vlm.is_available() else '— not installed'}")
    st.divider()
    st.caption(safety.SAFETY_SHORT)

tabs = st.tabs(
    [
        "Overview",
        "Study Viewer",
        "Similar Cases",
        "Draft Report",
        "Evaluation",
        "Review & Audit Log",
        "About / Safety",
    ]
)


# ---- Overview --------------------------------------------------------------
with tabs[0]:
    st.subheader("Overview")
    st.info(safety.SAFETY_LONG)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Studies", len(studies))
    c2.metric("Index backend", index.backend)
    pos_studies = sum(1 for s in studies if s.positive_labels())
    c3.metric("Studies w/ findings", pos_studies)
    seeded = sum(1 for s in studies if s.metadata.get("seed_draft"))
    c4.metric("Seeded flawed drafts", seeded)

    st.markdown(
        """
**What this is.** A local prototype that loads chest X-ray studies, retrieves
similar prior cases, drafts a structured report, and then **evaluates that draft
the way a radiologist cares about** — missed findings, hallucinations, negation
and laterality errors — not just text overlap. A clinician can edit and sign off,
and every action is written to an audit log.

**Why it matters.** Radiology report generation needs *clinically meaningful*
evaluation. A draft can score high on BLEU/ROUGE while inverting "no pneumothorax"
into "pneumothorax." This tool surfaces exactly those failure modes.
        """
    )
    st.markdown("##### Quick label distribution across the loaded dataset")
    dist = {lab: sum(1 for s in studies if s.labels.get(lab) == 1) for lab in CXR_LABELS}
    dist_df = pd.DataFrame(
        sorted(dist.items(), key=lambda kv: -kv[1]), columns=["Observation", "# positive"]
    )
    st.dataframe(dist_df, use_container_width=True, hide_index=True)


# ---- Study Viewer ----------------------------------------------------------
with tabs[1]:
    st.subheader(f"Study Viewer — `{study.study_id}`")
    col_img, col_meta = st.columns([1, 1.3])
    with col_img:
        img = _resolve_image(study.primary_image or "")
        if img:
            st.image(str(img), caption=f"{study.study_id} ({study.view_position})", width=360)
        else:
            st.info("No image available for this study.")
        st.caption("Synthetic placeholder image — no patient data.")
    with col_meta:
        st.markdown(f"**Indication:** {study.indication or '—'}")
        st.markdown(f"**View:** {study.view_position or '—'} &nbsp; | &nbsp; **Split:** {study.split}")
        with st.expander("Findings", expanded=True):
            st.write(study.findings or "—")
        with st.expander("Impression", expanded=True):
            st.write(study.impression or "—")
        st.markdown("**Reference labels**")
        ref_rows = [
            {"Observation": lab, "Value": LABEL_DISPLAY[study.labels.get(lab)]}
            for lab in CXR_LABELS
            if study.labels.get(lab) is not None
        ]
        st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)


# ---- Similar Cases ---------------------------------------------------------
with tabs[2]:
    st.subheader("Similar Cases")
    st.caption("Cosine similarity over the retrieval index (excludes the query study).")
    top_k = st.slider("Top-k", 3, 8, 5)
    similar = find_similar(study.study_id, top_k=top_k, index=index)

    chosen_context: List[str] = []
    for sc in similar:
        cols = st.columns([1, 3, 1.2])
        with cols[0]:
            sim_img = _resolve_image(sc.image_path or "")
            if sim_img:
                st.image(str(sim_img), width=110)
        with cols[1]:
            st.markdown(f"**{sc.study_id}**  ·  similarity `{sc.score:.3f}`")
            st.caption(sc.impression or "—")
            shared = set(study.positive_labels()) & set(sc.study.positive_labels())
            st.markdown(
                "Shared findings: " + (", ".join(sorted(shared)) if shared else "_none_")
            )
        with cols[2]:
            use = st.checkbox("Use for draft", key=f"ctx_{sc.study_id}")
            if use:
                chosen_context.append(sc.study_id)
        st.divider()
    st.session_state["context_ids"][study.study_id] = chosen_context


# ---- Draft Report ----------------------------------------------------------
with tabs[3]:
    st.subheader("Draft Report")
    st.caption(safety.DRAFT_DISCLAIMER)
    mode = st.radio(
        "Generation mode",
        ["template", "retrieval", "local_vlm"],
        horizontal=True,
        help="template always works; retrieval uses similar cases; local_vlm uses mlx-vlm if installed.",
    )
    if mode == "local_vlm":
        st.info(local_vlm.status_message())

    gen_cols = st.columns([1, 1, 2])
    with gen_cols[0]:
        do_generate = st.button("Generate draft", type="primary")
    with gen_cols[1]:
        has_seed = bool(study.metadata.get("seed_draft"))
        use_seed = st.button("Load seeded flawed draft", disabled=not has_seed,
                             help="Demo draft with intentional clinical errors for the Evaluation tab.")

    if do_generate:
        ctx_ids = st.session_state["context_ids"].get(study.study_id, [])
        sim_for_ctx = find_similar(study.study_id, top_k=max(5, len(ctx_ids)), index=index)
        if ctx_ids:
            sim_for_ctx = [c for c in sim_for_ctx if c.study_id in ctx_ids] or sim_for_ctx
        result = generate_draft(study, mode=mode, similar_cases=sim_for_ctx)
        st.session_state["drafts"][study.study_id] = result.draft_text
        st.session_state["draft_meta"][study.study_id] = {
            "mode_used": result.mode_used,
            "context_used": result.context_used,
            "limitations": result.limitations,
        }

    if use_seed and has_seed:
        st.session_state["drafts"][study.study_id] = study.metadata["seed_draft"]
        st.session_state["draft_meta"][study.study_id] = {
            "mode_used": "seeded demo draft (intentional errors)",
            "context_used": ["demo seed"],
            "limitations": ["This draft contains deliberate errors to demonstrate evaluation."],
        }

    current_draft = st.session_state["drafts"].get(study.study_id)
    if current_draft:
        meta = st.session_state["draft_meta"].get(study.study_id, {})
        st.success(f"Draft ready (mode: {meta.get('mode_used', '?')})")
        st.code(current_draft, language="markdown")
        with st.expander("Context used"):
            st.write(meta.get("context_used", []))
        with st.expander("Limitations"):
            for lim in meta.get("limitations", []):
                st.markdown(f"- {lim}")
    else:
        st.info("Generate a draft (or load the seeded flawed draft) to continue.")


# ---- Evaluation ------------------------------------------------------------
with tabs[4]:
    st.subheader("Evaluation")
    draft_text = st.session_state["drafts"].get(study.study_id)
    if not draft_text:
        # Auto-generate a template draft so the tab is never empty.
        draft_text = generate_draft(study, mode="template").draft_text
        st.caption("No draft generated yet — showing a template draft for evaluation.")

    draft_labels = label_report(draft_text)
    metrics = compute_metrics(study.labels, study.report_text, draft_text, draft_labels)
    errors = compare_reports(study.labels, study.report_text, draft_text, draft_labels)
    color = status_color(metrics)
    # A single high-severity clinical error (e.g. negation/laterality on a
    # critical finding) should escalate the overall status to RED.
    if any(e.severity == "high" for e in errors):
        color = "red"

    st.markdown(f"### {STATUS_EMOJI[color]} Overall status: **{color.upper()}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Positive F1", metrics["positive_f1"])
    m2.metric("Missed findings", metrics["missed_finding_count"])
    m3.metric("Hallucinated", metrics["hallucinated_finding_count"])
    m4.metric("Laterality errors", metrics["laterality_mismatch_count"])
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Precision", metrics["positive_precision"])
    m6.metric("Recall", metrics["positive_recall"])
    m7.metric("Label agreement", metrics["exact_label_agreement"])
    m8.metric("Length ratio", metrics["report_length_ratio"])

    st.markdown("#### Reference vs draft labels")
    st.dataframe(_label_matrix(study.labels, draft_labels), use_container_width=True, hide_index=True)

    st.markdown("#### Clinical error table")
    if errors:
        err_df = pd.DataFrame([
            {
                "Severity": e.severity,
                "Type": e.error_type,
                "Finding": e.finding,
                "Reference evidence": e.reference_evidence or "—",
                "Draft evidence": e.draft_evidence or "—",
                "Suggested mitigation": e.mitigation,
            }
            for e in sorted(errors, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x.severity])
        ])
        st.dataframe(err_df, use_container_width=True, hide_index=True)

        st.markdown("#### Failure-mode summary")
        for e in errors:
            if e.error_type in ("missed_finding", "support_device_mismatch"):
                st.error(f"Missed: {e.finding} — {e.mitigation}")
            elif e.error_type == "hallucinated_finding":
                st.error(f"Hallucinated {e.finding}: {e.mitigation}")
            elif e.error_type == "negation_error":
                st.error(f"Negation mismatch on {e.finding}: {e.mitigation}")
            elif e.error_type == "laterality_mismatch":
                st.error(f"Laterality mismatch on {e.finding}: {e.mitigation}")
            else:
                st.warning(f"{e.error_type} on {e.finding}: {e.mitigation}")
    else:
        st.success("No clinically significant discrepancies detected between draft and reference.")

    with st.expander("Why generic BLEU/ROUGE is insufficient for radiology reports"):
        st.markdown(
            """
Lexical metrics (BLEU, ROUGE, METEOR) reward surface word overlap. In radiology
this is dangerous: a draft that flips **"no pneumothorax"** to **"pneumothorax"**
changes a single token yet inverts the clinical meaning — while still scoring
highly on n-gram overlap. Conversely, a perfectly correct report phrased
differently from the reference can score *low*.

What actually matters clinically is whether the **findings** are right: did we
miss a real finding, hallucinate one that isn't there, get the **laterality**
(left vs right) wrong, or mis-state **certainty** (definite vs "possible")? This
tool scores those directly via a label-level taxonomy, and exposes optional
entity-level metrics (RadGraph F1 / RadCliQ) for deeper analysis.
            """
        )
        st.caption(radgraph_status())


# ---- Review & Audit Log ----------------------------------------------------
with tabs[5]:
    st.subheader("Review & Audit Log")
    base_draft = st.session_state["drafts"].get(study.study_id)
    if not base_draft:
        base_draft = generate_draft(study, mode="template").draft_text

    st.markdown("#### Radiologist edit")
    edited = st.text_area("Edit the AI draft, then choose an action:", value=base_draft, height=320)

    action_cols = st.columns(4)
    save_edit = action_cols[0].button("💾 Save edit")
    mark_reviewed = action_cols[1].button("✅ Mark reviewed")
    escalate = action_cols[2].button("⚠️ Needs escalation")
    export = action_cols[3].button("⬇️ Export case JSON")

    notes = st.text_input("Reviewer notes", value="")

    action = None
    if save_edit:
        action = "save_edit"
    elif mark_reviewed:
        action = "mark_reviewed"
    elif escalate:
        action = "mark_escalation"
    elif export:
        action = "export"

    if action:
        outcome = review_edit(
            study,
            draft_before=base_draft,
            draft_after=edited,
            reviewer_action=action,
            notes=notes,
            persist=True,
        )
        st.session_state["drafts"][study.study_id] = edited

        st.markdown("#### Before vs after")
        d = outcome.diff
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Errors before", d["errors_before"])
        cc2.metric("Errors after", d["errors_after"], delta=d["errors_after"] - d["errors_before"])
        cc3.metric("Text similarity", d["text_similarity"])

        b1, b2 = st.columns(2)
        with b1:
            st.markdown("**Errors fixed**")
            if d["errors_fixed"]:
                st.dataframe(pd.DataFrame(d["errors_fixed"])[["error_type", "finding", "severity"]],
                             use_container_width=True, hide_index=True)
            else:
                st.caption("None")
        with b2:
            st.markdown("**Errors introduced**")
            if d["errors_introduced"]:
                st.dataframe(pd.DataFrame(d["errors_introduced"])[["error_type", "finding", "severity"]],
                             use_container_width=True, hide_index=True)
            else:
                st.caption("None")

        if action == "export":
            st.download_button(
                "Download case_review.json",
                data=export_case_review_json(outcome),
                file_name=f"case_review_{study.study_id}.json",
                mime="application/json",
            )
        st.success(f"Action '{action}' recorded in the audit log (id={outcome.audit_id}).")

    st.markdown("#### Audit history")
    entries = list_audit_entries()
    if entries:
        audit_df = pd.DataFrame([
            {
                "id": e["id"],
                "timestamp": e["timestamp"],
                "study_id": e["study_id"],
                "action": e["reviewer_action"],
                "f1_before": e["metrics_before"].get("positive_f1"),
                "f1_after": e["metrics_after"].get("positive_f1"),
                "notes": e["notes"],
            }
            for e in entries
        ])
        st.dataframe(audit_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No audit entries yet — perform a review action above.")


# ---- About / Safety --------------------------------------------------------
with tabs[6]:
    st.subheader("About / Safety")
    st.error(
        "**Not for diagnosis. Not validated. No patient data included.** "
        "This is a research and demonstration prototype only."
    )
    st.markdown(
        f"""
**Datasets.** {safety.DATASET_POLICY}

**Hardware.** Designed for an Apple Silicon MacBook with 16 GB unified memory and
no CUDA GPU. The default retrieval backend uses lightweight image histograms +
TF-IDF, so nothing large is downloaded. BiomedCLIP and a local MLX VLM are
optional, off by default, and degrade gracefully when absent.

**Using real datasets responsibly.**
- IU X-Ray / Open-i: open collection — `import_openi`.
- MIMIC-CXR / MIMIC-CXR-JPG: credentialed PhysioNet DUA required; never
  redistributed or auto-downloaded — `import_mimic` reads your local copy only.
- CheXpert: obtain under Stanford's license — `import_chexpert`.

**Limitations.** The weak labeler is a transparent heuristic, not CheXbert. The
synthetic images are abstract placeholders, not anatomy. No model is trained or
clinically validated here. The value is the *evaluation and workflow architecture*.
        """
    )
    st.caption(safety.SAFETY_SHORT)
