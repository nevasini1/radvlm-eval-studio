"""RadDPO-Lite: evaluator-generated clinical preference tuning for draft repair.

A small, Mac-friendly *text-side* training experiment. It does NOT train an image
model and makes no diagnostic claims. The evaluator (rule-based labeler + clinical
error taxonomy) generates flawed/corrected report pairs; a tiny LoRA adapter can
then learn to repair drafts. Everything degrades gracefully when MLX is absent.

Research demo only. Not for diagnosis or clinical use.
"""

from radvlm_eval.training.error_augmenter import (
    AugmentedDraft,
    applicable_error_types,
    build_reference_report,
    inject_error,
)
from radvlm_eval.training.pair_builder import TrainingData, build_pairs
from radvlm_eval.training.report_repair_model import repair_report

__all__ = [
    "AugmentedDraft",
    "applicable_error_types",
    "build_reference_report",
    "inject_error",
    "TrainingData",
    "build_pairs",
    "repair_report",
]
