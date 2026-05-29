"""Synthetic demo dataset generator.

Produces 30-50 fully synthetic chest X-ray "studies" that are safe to commit:
placeholder grayscale images (no patient data), varied reports, CheXpert-style
labels, and a few intentionally flawed seed drafts so the evaluation dashboard
has interesting failure modes to surface.

Everything is deterministic given a seed so tests and demos are reproducible.
"""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from radvlm_eval import config
from radvlm_eval.schemas import CXR_LABELS, LabelValue, Study, empty_labels

try:  # Pillow is a hard dependency, but keep import errors friendly.
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Pillow is required for demo image generation. Install with `pip install Pillow`."
    ) from exc


# ---------------------------------------------------------------------------
# Case archetypes. Each defines reference text + positive/uncertain labels and,
# optionally, a deliberately flawed AI seed draft used to seed the demo so the
# evaluation tab shows realistic failure modes out of the box.
# ---------------------------------------------------------------------------

INDICATIONS = [
    "Shortness of breath.",
    "Cough and fever, rule out pneumonia.",
    "Chest pain.",
    "Routine post-operative check.",
    "Follow-up of known cardiomegaly.",
    "Status post line placement.",
    "Dyspnea on exertion.",
    "Trauma, evaluate for pneumothorax.",
]

VIEWS = ["PA", "AP", "PA and lateral", "AP portable"]


def _pos(*labels: str) -> Dict[str, LabelValue]:
    out = empty_labels()
    for lab in labels:
        out[lab] = 1
    return out


def _with(base: Dict[str, LabelValue], **updates: LabelValue) -> Dict[str, LabelValue]:
    out = dict(base)
    out.update(updates)
    return out


ARCHETYPES: List[Dict] = [
    {
        "key": "normal",
        "findings": (
            "The lungs are clear without focal consolidation, effusion, or "
            "pneumothorax. The cardiomediastinal silhouette is within normal "
            "limits. No acute osseous abnormality."
        ),
        "impression": "No acute cardiopulmonary abnormality.",
        "labels": _pos("No Finding"),
        "seed_draft": None,
    },
    {
        "key": "mild_cardiomegaly",
        "findings": (
            "Mild enlargement of the cardiac silhouette. The lungs are clear "
            "without focal consolidation or effusion. No pneumothorax."
        ),
        "impression": "Mild cardiomegaly. No acute pulmonary process.",
        "labels": _pos("Cardiomegaly"),
        # Seed draft MISSES the cardiomegaly -> missed_finding.
        "seed_draft": {
            "findings": "The lungs are clear. No focal consolidation, effusion, or pneumothorax. Heart size is normal.",
            "impression": "No acute cardiopulmonary abnormality.",
        },
    },
    {
        "key": "left_pleural_effusion",
        "findings": (
            "Small left pleural effusion with associated basilar atelectasis. "
            "No pneumothorax. The right lung is clear. Heart size is normal."
        ),
        "impression": "Small left pleural effusion.",
        "labels": _pos("Pleural Effusion", "Atelectasis"),
        # Seed draft hallucinates a pneumothorax and misses the effusion side.
        "seed_draft": {
            "findings": "There is a right pleural effusion. Possible small apical pneumothorax. Heart size normal.",
            "impression": "Right pleural effusion and possible pneumothorax.",
        },
    },
    {
        "key": "rll_opacity",
        "findings": (
            "Patchy opacity in the right lower lobe, which may represent "
            "pneumonia in the appropriate clinical setting. No pleural effusion "
            "or pneumothorax. Heart size normal."
        ),
        "impression": "Right lower lobe opacity, possibly pneumonia.",
        "labels": _with(_pos("Lung Opacity"), Pneumonia=-1),
        "seed_draft": {
            "findings": "Patchy opacity in the right lower lobe consistent with pneumonia. No effusion or pneumothorax.",
            "impression": "Right lower lobe pneumonia.",
        },
    },
    {
        "key": "support_devices",
        "findings": (
            "Endotracheal tube terminates approximately 4 cm above the carina. "
            "Right internal jugular central venous catheter tip in the lower "
            "SVC. No pneumothorax. Low lung volumes with bibasilar atelectasis."
        ),
        "impression": "Lines and tubes in standard position. Bibasilar atelectasis.",
        "labels": _pos("Support Devices", "Atelectasis"),
        "seed_draft": None,
    },
    {
        "key": "pneumothorax",
        "findings": (
            "Moderate right-sided pneumothorax with partial lung collapse. No "
            "tension features. Left lung clear. No pleural effusion."
        ),
        "impression": "Moderate right pneumothorax.",
        "labels": _pos("Pneumothorax"),
        # Seed draft says NO pneumothorax -> negation / missed finding (high severity).
        "seed_draft": {
            "findings": "No pneumothorax. The lungs are clear bilaterally. No effusion.",
            "impression": "No acute cardiopulmonary abnormality.",
        },
    },
    {
        "key": "pulmonary_edema",
        "findings": (
            "Pulmonary vascular congestion with interstitial edema and small "
            "bilateral pleural effusions. Cardiomegaly is present."
        ),
        "impression": "Pulmonary edema with cardiomegaly.",
        "labels": _pos("Edema", "Cardiomegaly", "Pleural Effusion"),
        "seed_draft": {
            "findings": "Mild pulmonary vascular congestion. Heart size top-normal. No definite effusion.",
            "impression": "Mild congestion.",
        },
    },
    {
        "key": "consolidation",
        "findings": (
            "Dense consolidation in the left upper lobe. Cannot exclude "
            "underlying mass. No pneumothorax. No large effusion."
        ),
        "impression": "Left upper lobe consolidation; recommend follow-up to exclude mass.",
        "labels": _with(_pos("Consolidation", "Lung Opacity"), **{"Lung Lesion": -1}),
        "seed_draft": None,
    },
    {
        "key": "fracture",
        "findings": (
            "Acute fracture of the right lateral fifth rib. No pneumothorax. "
            "Lungs are clear. Heart size normal."
        ),
        "impression": "Right fifth rib fracture without pneumothorax.",
        "labels": _pos("Fracture"),
        # Seed draft misses fracture and hallucinates a pneumothorax.
        "seed_draft": {
            "findings": "Small right pneumothorax. No definite fracture. Lungs clear.",
            "impression": "Small right pneumothorax.",
        },
    },
    {
        "key": "uncertain",
        "findings": (
            "Subtle left base opacity, possibly atelectasis versus early "
            "consolidation. Cannot exclude small left pleural effusion. No "
            "pneumothorax."
        ),
        "impression": "Indeterminate left base opacity; clinical correlation advised.",
        "labels": _with(
            empty_labels(),
            **{"Atelectasis": -1, "Consolidation": -1, "Pleural Effusion": -1},
        ),
        "seed_draft": None,
    },
]


def _hash_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _render_image(path: Path, study_id: str, archetype_key: str, rng: random.Random) -> None:
    """Render a deterministic synthetic grayscale 'chest-like' placeholder image.

    This is abstract art, not anatomy — purely to make the viewer look populated.
    """
    size = 256
    img = Image.new("L", (size, size), color=10)
    draw = ImageDraw.Draw(img)

    # Soft thoracic background: two lighter lung-field ellipses + mediastinum.
    draw.ellipse([35, 40, 120, 210], fill=70)
    draw.ellipse([136, 40, 221, 210], fill=70)
    draw.rectangle([118, 50, 138, 200], fill=45)  # spine/mediastinum
    # Heart silhouette (bigger for cardiomegaly archetypes).
    heart_w = 95 if "cardio" in archetype_key or archetype_key == "pulmonary_edema" else 70
    draw.ellipse([128 - heart_w // 2, 120, 128 + heart_w // 2, 200], fill=55)

    # Archetype-specific cues (still abstract).
    if "effusion" in archetype_key or archetype_key == "pulmonary_edema":
        draw.rectangle([35, 185, 120, 210], fill=95)  # blunted left base
    if archetype_key == "pneumothorax":
        draw.ellipse([150, 45, 218, 120], fill=25)  # dark apical lucency
    if archetype_key in ("rll_opacity", "consolidation"):
        draw.ellipse([60, 130, 110, 185], fill=110)  # focal opacity
    if archetype_key == "support_devices":
        draw.line([128, 50, 128, 150], fill=200, width=2)  # tube

    # Deterministic speckle noise.
    px = img.load()
    for _ in range(2200):
        x = rng.randint(0, size - 1)
        y = rng.randint(0, size - 1)
        base = px[x, y]
        px[x, y] = max(0, min(255, base + rng.randint(-25, 25)))

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _compose_report(indication: str, findings: str, impression: str) -> str:
    return (
        f"INDICATION: {indication}\n\n"
        f"FINDINGS: {findings}\n\n"
        f"IMPRESSION: {impression}"
    )


def generate_demo_dataset(
    data_dir: Optional[Path] = None,
    n_studies: int = 36,
    seed: int = 7,
) -> List[Study]:
    """Generate synthetic studies, render images, and return canonical Study objects.

    Returns the list of studies. Caller is responsible for persisting them
    (see scripts/make_demo_data.py).
    """
    data_dir = Path(data_dir) if data_dir else config.DEMO_DIR
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    n_studies = max(30, min(50, n_studies))
    rng = random.Random(seed)

    studies: List[Study] = []
    splits = ["train"] * 0 + ["valid", "test"]  # demo uses valid/test only

    for i in range(n_studies):
        arch = ARCHETYPES[i % len(ARCHETYPES)]
        study_id = f"DEMO{i:04d}"
        patient_hash = _hash_id(f"patient-{i // 2}")  # some patients have 2 studies
        indication = rng.choice(INDICATIONS)
        view = rng.choice(VIEWS)
        split = rng.choice(splits)

        image_path = images_dir / f"{study_id}.png"
        _render_image(image_path, study_id, arch["key"], rng)

        findings = arch["findings"]
        impression = arch["impression"]
        report_text = _compose_report(indication, findings, impression)

        metadata = {
            "patient_id_hash": patient_hash,
            "split": split,
            "view_position": view,
            "archetype": arch["key"],
            "synthetic": True,
        }
        if arch.get("seed_draft"):
            sd = arch["seed_draft"]
            metadata["seed_draft"] = _compose_report(
                indication, sd["findings"], sd["impression"]
            )

        study = Study(
            study_id=study_id,
            image_paths=[str(image_path.relative_to(config.REPO_ROOT))]
            if image_path.is_relative_to(config.REPO_ROOT)
            else [str(image_path)],
            report_text=report_text,
            findings=findings,
            impression=impression,
            indication=indication,
            labels=dict(arch["labels"]),
            metadata=metadata,
        )
        studies.append(study)

    return studies
