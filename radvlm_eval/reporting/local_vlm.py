"""Optional local VLM adapter (mlx-vlm) for Apple Silicon.

If mlx-vlm is installed the app can optionally caption an image locally. If not,
this module reports unavailability and the caller falls back to template /
retrieval generation. The app must NEVER fail because a local VLM is missing.

References:
    https://github.com/ml-explore/mlx
    https://github.com/Blaizzy/mlx-vlm
"""

from __future__ import annotations

import importlib.util
from typing import Optional


def is_available() -> bool:
    return importlib.util.find_spec("mlx_vlm") is not None


def status_message() -> str:
    if is_available():
        return "mlx-vlm detected — local VLM drafting available."
    return (
        "Local VLM (mlx-vlm) not installed. Install on Apple Silicon with "
        "`pip install mlx-vlm` to enable on-device drafting. Falling back to "
        "template / retrieval mode."
    )


def generate_vlm_findings(
    image_path: str,
    model_id: str = "mlx-community/llava-1.5-7b-4bit",
    prompt: str = "Describe the chest X-ray findings conservatively.",
) -> Optional[str]:
    """Return a raw VLM caption, or None if mlx-vlm is unavailable / errors out."""
    if not is_available():
        return None
    try:  # pragma: no cover - exercised only with mlx-vlm installed
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template

        model, processor = load(model_id)
        formatted = apply_chat_template(processor, model.config, prompt, num_images=1)
        out = generate(model, processor, formatted, [image_path], verbose=False)
        return str(out)
    except Exception:  # noqa: BLE001
        return None
