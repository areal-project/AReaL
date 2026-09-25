# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from datasets import Dataset
from PIL import Image

from examples.vlm_npu.virl39k_grpo import acc_reward, virl39k_reward_fn

from areal.dataset.virl39k import get_virl39k_rl_dataset

# From ViRL39K row 34209: `\left\{` opens a brace that `\right.` never closes, so
# mathruler's extract_boxed_content returns its "None" sentinel for this reference.
UNBALANCED_ANSWER = r"\boxed{\left\{\eqalign{&x+y-4=30\cr&(x-4)-(y-4)=2\cr }\right.}"


def _fake_processor():
    return SimpleNamespace(
        image_processor=SimpleNamespace(image_processor_type="Qwen2VLImageProcessor"),
        tokenizer=SimpleNamespace(
            apply_chat_template=lambda messages, **_: messages[0]["content"]
        ),
    )


def _write_virl39k_parquet(tmp_path, answers):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (32, 32), color="blue").save(tmp_path / "images" / "0.png")
    n = len(answers)
    Dataset.from_dict(
        {
            "question": ["<image>Solve it."] * n,
            "answer": answers,
            "image": [["images/0.png"]] * n,
            "PassRate_32BTrained": [0.5] * n,
            "PassRate_7BBase": [0.5] * n,
            "category": ["math"] * n,
            "source": ["test"] * n,
            "qid": [f"q{i}" for i in range(n)],
        }
    ).to_parquet(tmp_path / "train.parquet")
    return str(tmp_path / "train.parquet")


def test_loader_drops_rows_without_extractable_boxed_answer(tmp_path):
    """A reference whose boxed content cannot be extracted must not be stored
    as the literal "None" gold answer."""
    path = _write_virl39k_parquet(tmp_path, [r"\boxed{A}", UNBALANCED_ANSWER])

    dataset = get_virl39k_rl_dataset(path, "train", _fake_processor())

    assert dataset["answer"] == ["A"]


def test_acc_reward_rejects_unextractable_prediction():
    """A reply without a closed \\boxed{} must never match, even against "None"."""
    assert acc_reward("I'm sorry, I can't help with that.", "None") == 0.0
    assert acc_reward(r"<think>ok</think> \boxed{5", "None") == 0.0
    assert acc_reward(r"<think>ok</think> \boxed{5}", "5") == 1.0


def test_reward_fn_gives_no_accuracy_credit_for_empty_reply():
    reward = virl39k_reward_fn(
        prompt="",
        completions="",
        prompt_ids=[],
        completion_ids=[],
        answer="None",
    )
    assert reward == 0.0
