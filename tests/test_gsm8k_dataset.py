# SPDX-License-Identifier: Apache-2.0

from datasets import Dataset

import areal.dataset.gsm8k as gsm8k_mod
from areal.dataset.gsm8k import get_gsm8k_sft_dataset


class _ByteMergeTokenizer:
    eos_token = "<eos>"

    def __init__(self):
        self._vocab: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        ids, i = [], 0
        while i < len(text):
            step = 2 if text[i : i + 2] == "ow" else 1
            ids.append(self._vocab.setdefault(text[i : i + step], len(self._vocab)))
            i += step
        return ids


def _loss_mask(monkeypatch, tokenizer, question: str, answer: str):
    sample = Dataset.from_dict({"question": [question], "answer": [answer]})
    monkeypatch.setattr(gsm8k_mod, "load_dataset", lambda *args, **kwargs: sample)
    dataset = get_gsm8k_sft_dataset(path="ignored", split="train", tokenizer=tokenizer)
    row = dataset[0]
    return row["input_ids"], row["loss_mask"]


def test_boundary_merge_token_is_supervised(monkeypatch):
    tok = _ByteMergeTokenizer()
    question, answer = "abco", "wxyz"

    prompt_ids = tok.encode(question)
    full_ids = tok.encode(question + answer + tok.eos_token)
    assert full_ids[: len(prompt_ids)] != prompt_ids

    input_ids, loss_mask = _loss_mask(monkeypatch, tok, question, answer)
    assert loss_mask == [0] * 3 + [1] * (len(input_ids) - 3)
    assert input_ids[:3] == prompt_ids[:3]


def test_no_merge_boundary_unchanged(monkeypatch):
    tok = _ByteMergeTokenizer()
    question, answer = "abc", "xyz"

    prompt_ids = tok.encode(question)
    full_ids = tok.encode(question + answer + tok.eos_token)
    assert full_ids[: len(prompt_ids)] == prompt_ids

    _, loss_mask = _loss_mask(monkeypatch, tok, question, answer)
    assert loss_mask.count(0) == len(prompt_ids)
