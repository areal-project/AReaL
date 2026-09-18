# SPDX-License-Identifier: Apache-2.0

from areal.dataset import gsm8k


class _Dataset:
    def map(self, *_args, **_kwargs):
        return self

    def remove_columns(self, *_args, **_kwargs):
        return self

    def filter(self, *_args, **_kwargs):
        return self


class _Tokenizer:
    eos_token = "<eos>"

    def encode(self, text):
        return list(range(len(text)))


def test_gsm8k_sft_uses_rank_local_in_memory_cache(tmp_path, monkeypatch):
    captured = {}

    def fake_load_dataset(**kwargs):
        captured.update(kwargs)
        return _Dataset()

    monkeypatch.setenv("RANK", "17")
    monkeypatch.setattr(gsm8k, "load_dataset", fake_load_dataset)

    gsm8k.get_gsm8k_sft_dataset(
        path="/data/gsm8k",
        split="train",
        tokenizer=_Tokenizer(),
        cache_dir=str(tmp_path),
    )

    assert captured["cache_dir"] == str(tmp_path / "rank-17")
    assert captured["keep_in_memory"] is True
