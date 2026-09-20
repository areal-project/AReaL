# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from PIL import Image

from areal.dataset import clevr_count_70k, geometry3k


class _FakeTokenizer:
    eos_token = "<eos>"

    def __init__(self, add_bos_by_default):
        self._add_bos_by_default = add_bos_by_default

    def encode(self, text, add_special_tokens=True):
        if text == "answer":
            token_ids = [20, 21]
        elif text == "wxyz":
            token_ids = [30, 21, 22, 23]
        elif text.endswith("proo"):
            token_ids = [10, 11, 12, 13]
        elif text.endswith("proowxyz<eos>"):
            token_ids = [10, 11, 12, 50, 21, 22, 23, 99]
        elif text.endswith("question"):
            token_ids = [10, 11]
        elif text.endswith("questionanswer<eos>"):
            token_ids = [10, 11, 20, 21, 99]
        else:
            raise ValueError(f"Unexpected text: {text}")
        if add_special_tokens and self._add_bos_by_default:
            return [1, *token_ids]
        return token_ids


class _FakeProcessor:
    def __init__(self, add_bos_by_default):
        self.tokenizer = _FakeTokenizer(add_bos_by_default)
        self.image_processor = SimpleNamespace(image_processor_type="qwen")

    def __call__(self, **kwargs):
        input_ids = self.tokenizer.encode(kwargs["text"][0])
        image_offset = 2 if self.tokenizer._add_bos_by_default else 1
        input_ids[image_offset:image_offset] = [70, 71]
        return {
            "input_ids": torch.tensor([input_ids]),
            "pixel_values": torch.zeros(1),
        }


@pytest.mark.parametrize(
    ("module", "loader"),
    [
        (geometry3k, geometry3k.get_geometry3k_sft_dataset),
        (clevr_count_70k, clevr_count_70k.get_clevr_count_70k_sft_dataset),
    ],
)
@pytest.mark.parametrize("add_bos_by_default", [False, True])
@pytest.mark.parametrize(
    ("problem", "answer", "expected_token_ids"),
    [
        ("<image>question", "answer", [20, 21, 99]),
        ("<image>proo", "wxyz", [50, 21, 22, 23, 99]),
    ],
)
def test_vlm_sft_loss_mask_includes_full_answer_and_eos(
    monkeypatch,
    module,
    loader,
    add_bos_by_default,
    problem,
    answer,
    expected_token_ids,
):
    """The SFT target includes the answer, EOS, and any boundary-merged token."""
    dataset = Dataset.from_list(
        [
            {
                "images": [Image.new("RGB", (4, 4))],
                "problem": problem,
                "answer": answer,
            }
        ]
    )
    monkeypatch.setattr(module, "load_dataset", lambda **kwargs: dataset)
    monkeypatch.setattr(module, "convert_image", lambda image, *args: image)
    if module is clevr_count_70k:
        monkeypatch.setattr(module.os, "cpu_count", lambda: 1)

    sample = loader(
        path="unused",
        split="train",
        processor=_FakeProcessor(add_bos_by_default),
    )[0]

    input_ids = sample["input_ids"]
    trained_token_ids = [
        token_id
        for token_id, include_in_loss in zip(input_ids, sample["loss_mask"])
        if include_in_loss
    ]
    assert trained_token_ids == expected_token_ids
