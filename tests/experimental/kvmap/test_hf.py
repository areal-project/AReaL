# SPDX-License-Identifier: Apache-2.0
"""Decoding one boundary token from an extracted cache reproduces full-prefill logits, and metrics detect a changed cache."""

import pathlib

import pytest
import torch

from areal.experimental.kvmap.apply import translate_cache
from areal.experimental.kvmap.artifact import identity_artifact
from areal.experimental.kvmap.hf import (
    extract_cache,
    hash_checkpoint_directory,
    model_identity_from_config,
    next_token_logits,
)
from areal.experimental.kvmap.metrics import distribution_metrics

VOCAB = 97
TOKENS = 12


def _tiny_model(seed: int):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_theta=10_000.0,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )
    return Qwen2ForCausalLM(config).eval()


def _batch(seed: int):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (2, TOKENS), generator=generator)
    attention_mask = torch.ones(2, TOKENS, dtype=torch.long)
    return input_ids, attention_mask


def test_boundary_decode_from_extracted_cache_matches_full_prefill():
    model = _tiny_model(0)
    input_ids, attention_mask = _batch(0)
    full = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]

    cache, _ = extract_cache(
        model, input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1]
    )
    boundary = next_token_logits(
        model,
        cache=cache,
        boundary_tokens=input_ids[:, -1],
        cache_lengths=torch.full((2,), TOKENS - 1),
    )

    torch.testing.assert_close(boundary, full, rtol=1e-4, atol=1e-4)


def test_identity_translated_cache_reproduces_native_logits_and_a_foreign_cache_does_not():
    model = _tiny_model(0)
    other = _tiny_model(1)
    input_ids, attention_mask = _batch(1)
    native_cache, positions = extract_cache(
        model, input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1]
    )
    identity = model_identity_from_config(
        model.config, checkpoint_sha256="a" * 64, cache_dtype=torch.float32
    )
    artifact = identity_artifact(source=identity, target=identity, provenance={})
    lengths = torch.full((2,), TOKENS - 1)
    native = next_token_logits(
        model,
        cache=native_cache,
        boundary_tokens=input_ids[:, -1],
        cache_lengths=lengths,
    )

    translated = translate_cache(
        source=native_cache,
        positions=positions,
        artifact=artifact,
        output_dtype=torch.float32,
    )
    mapped = next_token_logits(
        model, cache=translated, boundary_tokens=input_ids[:, -1], cache_lengths=lengths
    )
    foreign_cache, _ = extract_cache(
        other, input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1]
    )
    reused = next_token_logits(
        model,
        cache=foreign_cache,
        boundary_tokens=input_ids[:, -1],
        cache_lengths=lengths,
    )

    same = distribution_metrics(
        native_logits=native, candidate_logits=mapped, selected_tokens=input_ids[:, -1]
    )
    different = distribution_metrics(
        native_logits=native, candidate_logits=reused, selected_tokens=input_ids[:, -1]
    )
    assert float(same.kl_native_to_candidate.max()) < 1e-6
    assert bool(same.top1_agree.all())
    assert float(different.kl_native_to_candidate.min()) > 1e-3


def test_extract_cache_rejects_left_padding():
    model = _tiny_model(0)
    input_ids, attention_mask = _batch(2)
    attention_mask[0, 0] = 0
    with pytest.raises(ValueError, match="right-aligned prefix mask"):
        extract_cache(model, input_ids=input_ids, attention_mask=attention_mask)


def test_model_identity_reads_layout_and_refuses_sliding_window():
    model = _tiny_model(0)
    identity = model_identity_from_config(
        model.config, checkpoint_sha256="a" * 64, cache_dtype=torch.bfloat16
    )
    assert (
        identity.num_layers,
        identity.num_kv_heads,
        identity.head_dim,
        identity.rope_theta,
        identity.cache_dtype,
    ) == (2, 2, 8, 10_000.0, "bfloat16")
    model.config.use_sliding_window = True
    with pytest.raises(ValueError, match="sliding-window"):
        model_identity_from_config(
            model.config, checkpoint_sha256="a" * 64, cache_dtype=torch.bfloat16
        )


def test_checkpoint_hash_is_deterministic_and_changes_with_content(
    tmp_path: pathlib.Path,
):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "README.md").write_text("ignored")
    first = hash_checkpoint_directory(tmp_path)
    assert first == hash_checkpoint_directory(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weight2")
    assert hash_checkpoint_directory(tmp_path) != first
    with pytest.raises(FileNotFoundError, match="not a checkpoint directory"):
        hash_checkpoint_directory(
            tmp_path / "missing" if (tmp_path / "missing").mkdir() is None else tmp_path
        )
