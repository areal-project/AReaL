# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_NON_MODEL_CONFIG_KEYS = {
    "_name_or_path",
    "dtype",
    "finetuning_task",
    "id2label",
    "label2id",
    "output_attentions",
    "output_hidden_states",
    "problem_type",
    "return_dict",
    "return_dict_in_generate",
    "task_specific_params",
    "tokenizer_class",
    "torch_dtype",
    "transformers_version",
    "use_cache",
}


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing MOPD model metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _architecture_fingerprint(model_path: Path) -> tuple[dict[str, Any], str]:
    config = _read_json(model_path / "config.json")
    if not isinstance(config, dict):
        raise ValueError(f"MOPD config.json must contain an object: {model_path}")
    architecture = {
        key: value for key, value in config.items() if key not in _NON_MODEL_CONFIG_KEYS
    }
    return architecture, _sha256_json(architecture)


def _tokenizer_payload(tokenizer_path: Path) -> dict[str, Any]:
    tokenizer_config_path = tokenizer_path / "tokenizer_config.json"
    tokenizer_config = (
        _read_json(tokenizer_config_path) if tokenizer_config_path.is_file() else {}
    )
    if not isinstance(tokenizer_config, dict):
        raise ValueError(
            "MOPD tokenizer_config.json must contain an object: "
            f"{tokenizer_config_path}"
        )

    vocab_path = tokenizer_path / "vocab.json"
    tokenizer_json_path = tokenizer_path / "tokenizer.json"
    tokenizer_model_path = tokenizer_path / "tokenizer.model"
    if vocab_path.is_file():
        vocabulary = _read_json(vocab_path)
        tokenizer_added_tokens = None
    elif tokenizer_json_path.is_file():
        tokenizer_json = _read_json(tokenizer_json_path)
        try:
            vocabulary = tokenizer_json["model"]["vocab"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Cannot find model.vocab in {tokenizer_json_path}"
            ) from exc
        tokenizer_added_tokens = tokenizer_json.get("added_tokens")
    elif tokenizer_model_path.is_file():
        vocabulary = {
            "sentencepiece_sha256": hashlib.sha256(
                tokenizer_model_path.read_bytes()
            ).hexdigest()
        }
        tokenizer_added_tokens = None
    else:
        raise FileNotFoundError(
            "MOPD tokenizer must contain vocab.json, tokenizer.json, or "
            f"tokenizer.model: {tokenizer_path}"
        )

    return {
        "vocabulary": vocabulary,
        "added_tokens": tokenizer_added_tokens,
        "added_tokens_decoder": tokenizer_config.get("added_tokens_decoder", {}),
    }


def model_fingerprint(
    model_path: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
) -> dict[str, object]:
    """Fingerprint checkpoint structure and token-ID mappings without weights."""
    resolved_model_path = Path(model_path)
    resolved_tokenizer_path = (
        Path(tokenizer_path) if tokenizer_path else resolved_model_path
    )
    architecture, architecture_sha256 = _architecture_fingerprint(resolved_model_path)
    tokenizer = _tokenizer_payload(resolved_tokenizer_path)
    return {
        "path": str(resolved_model_path.resolve()),
        "tokenizer_path": str(resolved_tokenizer_path.resolve()),
        "architecture": architecture,
        "architecture_sha256": architecture_sha256,
        "tokenizer_sha256": _sha256_json(tokenizer),
    }


def validate_mopd_model_compatibility(
    actor_path: str | Path,
    teacher_paths: Mapping[str, str | Path],
    *,
    actor_tokenizer_path: str | Path | None = None,
) -> dict[str, dict[str, object]]:
    """Validate the persistent-teacher model and tokenizer invariants.

    Teachers share one resident controller, so every teacher must have the same
    architecture. The actor may use a different architecture, but all models
    must map tokens to the same IDs because teacher log-probabilities score actor
    trajectories directly.
    """
    if not teacher_paths:
        raise ValueError("MOPD compatibility validation requires at least one teacher")

    if "actor" in teacher_paths:
        raise ValueError("MOPD teacher ID 'actor' is reserved for the actor model")

    actor_fingerprint = model_fingerprint(
        actor_path,
        tokenizer_path=actor_tokenizer_path,
    )
    teacher_fingerprints = {
        teacher_id: model_fingerprint(path)
        for teacher_id, path in teacher_paths.items()
    }
    fingerprints = {"actor": actor_fingerprint, **teacher_fingerprints}

    actor_tokenizer = actor_fingerprint["tokenizer_sha256"]
    tokenizer_mismatches = [
        name
        for name, fingerprint in fingerprints.items()
        if fingerprint["tokenizer_sha256"] != actor_tokenizer
    ]
    if tokenizer_mismatches:
        raise ValueError(
            "MOPD actor and teachers must use compatible token-ID mappings; "
            f"mismatched={tokenizer_mismatches}"
        )

    teacher_ids = list(teacher_paths)
    reference_id = teacher_ids[0]
    reference_architecture = teacher_fingerprints[reference_id]["architecture_sha256"]
    architecture_mismatches = [
        teacher_id
        for teacher_id in teacher_ids[1:]
        if teacher_fingerprints[teacher_id]["architecture_sha256"]
        != reference_architecture
    ]
    if architecture_mismatches:
        raise ValueError(
            "All MOPD teachers must share one architecture because checkpoints "
            "are loaded into a persistent controller; reference="
            f"{reference_id!r}, mismatched={architecture_mismatches}"
        )

    return fingerprints
