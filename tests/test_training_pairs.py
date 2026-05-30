"""Training-pair generation yields well-formed SFT and DPO examples."""

from radvlm_eval.data.demo_data import generate_demo_dataset
from radvlm_eval.training.pair_builder import SYSTEM_PROMPT, build_pairs


def _studies(tmp_path):
    return generate_demo_dataset(data_dir=tmp_path, n_studies=30, seed=11)


def test_build_pairs_produces_examples(tmp_path):
    data = build_pairs(_studies(tmp_path), n_flawed_per_study=2, seed=7)
    stats = data.stats()
    assert stats["sft_total"] > 0
    assert stats["dpo_total"] == stats["sft_total"]
    # Splits should not all be empty for train.
    assert stats["sft_train"] > 0


def test_sft_example_shape(tmp_path):
    data = build_pairs(_studies(tmp_path), n_flawed_per_study=1, seed=7)
    ex = (data.sft["train"] or data.sft["valid"] or data.sft["test"])[0]
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert ex["messages"][0]["content"] == SYSTEM_PROMPT
    assert "requires radiologist review" in ex["messages"][2]["content"].lower()


def test_dpo_example_chosen_differs_from_rejected(tmp_path):
    data = build_pairs(_studies(tmp_path), n_flawed_per_study=1, seed=7)
    ex = (data.dpo["train"] or data.dpo["valid"] or data.dpo["test"])[0]
    assert set(ex.keys()) >= {"prompt", "chosen", "rejected"}
    assert ex["chosen"] != ex["rejected"]
    assert ex["chosen"].strip()
    assert ex["rejected"].strip()
