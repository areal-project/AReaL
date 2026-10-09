# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from areal.trainer.mopd.compatibility import validate_mopd_model_compatibility


def _write_checkpoint(path, *, hidden_size: int, vocab: dict[str, int]) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["QwenForCausalLM"],
                "model_type": "qwen",
                "hidden_size": hidden_size,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "vocab_size": len(vocab),
                "torch_dtype": "bfloat16",
                "transformers_version": "test-only",
            }
        ),
        encoding="utf-8",
    )
    (path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": f"template for {path.name}"}),
        encoding="utf-8",
    )


def test_compatibility_allows_heterogeneous_actor_and_matching_teachers(tmp_path):
    """Actor structure may differ while resident teacher checkpoints stay aligned."""
    vocab = {"a": 0, "b": 1}
    actor = tmp_path / "actor"
    teacher_a = tmp_path / "teacher-a"
    teacher_b = tmp_path / "teacher-b"
    _write_checkpoint(actor, hidden_size=8, vocab=vocab)
    _write_checkpoint(teacher_a, hidden_size=32, vocab=vocab)
    _write_checkpoint(teacher_b, hidden_size=32, vocab=vocab)

    fingerprints = validate_mopd_model_compatibility(
        actor,
        {"teacher_a": teacher_a, "teacher_b": teacher_b},
    )

    assert (
        fingerprints["actor"]["architecture_sha256"]
        != fingerprints["teacher_a"]["architecture_sha256"]
    )
    assert (
        fingerprints["teacher_a"]["architecture_sha256"]
        == fingerprints["teacher_b"]["architecture_sha256"]
    )


def test_compatibility_rejects_teacher_architecture_mismatch(tmp_path):
    """One persistent controller cannot load structurally different teachers."""
    vocab = {"a": 0, "b": 1}
    actor = tmp_path / "actor"
    teacher_a = tmp_path / "teacher-a"
    teacher_b = tmp_path / "teacher-b"
    _write_checkpoint(actor, hidden_size=8, vocab=vocab)
    _write_checkpoint(teacher_a, hidden_size=32, vocab=vocab)
    _write_checkpoint(teacher_b, hidden_size=64, vocab=vocab)

    with pytest.raises(ValueError, match="teachers must share one architecture"):
        validate_mopd_model_compatibility(
            actor,
            {"teacher_a": teacher_a, "teacher_b": teacher_b},
        )


def test_compatibility_rejects_token_id_mismatch(tmp_path):
    """Teacher log-probabilities require the actor's exact token-ID mapping."""
    actor = tmp_path / "actor"
    teacher = tmp_path / "teacher"
    _write_checkpoint(actor, hidden_size=8, vocab={"a": 0, "b": 1})
    _write_checkpoint(teacher, hidden_size=32, vocab={"a": 1, "b": 0})

    with pytest.raises(ValueError, match="compatible token-ID mappings"):
        validate_mopd_model_compatibility(actor, {"teacher": teacher})


def test_compatibility_rejects_reserved_actor_teacher_id(tmp_path):
    """A teacher cannot overwrite the actor fingerprint in the result mapping."""
    actor = tmp_path / "actor-model"
    teacher = tmp_path / "teacher-model"
    _write_checkpoint(actor, hidden_size=8, vocab={"a": 0, "b": 1})
    _write_checkpoint(teacher, hidden_size=32, vocab={"a": 1, "b": 0})

    with pytest.raises(ValueError, match="teacher ID 'actor' is reserved"):
        validate_mopd_model_compatibility(actor, {"actor": teacher})
