"""MLX dataset export writes valid JSONL with the expected keys."""

import json

from radvlm_eval.data.demo_data import generate_demo_dataset
from radvlm_eval.training.format_mlx_dataset import write_all
from radvlm_eval.training.pair_builder import build_pairs


def test_write_all_creates_valid_jsonl(tmp_path):
    studies = generate_demo_dataset(data_dir=tmp_path / "data", n_studies=30, seed=9)
    data = build_pairs(studies, n_flawed_per_study=2, seed=4)
    info = write_all(data, base_dir=tmp_path / "training")

    sft_train = tmp_path / "training" / "rad_repair_sft" / "train.jsonl"
    dpo_train = tmp_path / "training" / "rad_repair_dpo" / "train.jsonl"
    assert sft_train.exists()
    assert dpo_train.exists()

    # SFT lines parse and contain only "messages".
    with sft_train.open() as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert lines
    for row in lines:
        assert set(row.keys()) == {"messages"}
        assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]

    # DPO lines parse and contain exactly prompt/chosen/rejected.
    with dpo_train.open() as f:
        dlines = [json.loads(line) for line in f if line.strip()]
    assert dlines
    for row in dlines:
        assert set(row.keys()) == {"prompt", "chosen", "rejected"}

    assert info["sft_counts"]["train"] == len(lines)
