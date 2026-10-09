# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, overload

import transformers

import areal.utils.logging as logging
from areal.utils import pkg_version

logger = logging.getLogger("HFUtils")

HF_MODEL_ASSET_FILES = (
    "generation_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
)


def copy_hf_model_assets(
    source_model_path: str | os.PathLike[str],
    save_directory: str | os.PathLike[str],
) -> None:
    """Copy tokenizer, generation, chat-template, and remote-code assets."""
    source_model_path = os.fspath(source_model_path)
    save_directory = os.fspath(save_directory)
    for filename in HF_MODEL_ASSET_FILES:
        source = os.path.join(source_model_path, filename)
        destination = os.path.join(save_directory, filename)
        try:
            shutil.copy(source, destination)
            logger.info(
                "Copied %s from %s to %s", filename, source_model_path, save_directory
            )
        except FileNotFoundError:
            logger.info(
                "%s does not exist in %s; skipping", filename, source_model_path
            )

    for filename in os.listdir(source_model_path):
        is_remote_code = filename.endswith(".py") and filename.startswith(
            ("chat_format", "configuration_", "modeling_", "tokenization_")
        )
        if is_remote_code or filename.startswith("chat_template"):
            shutil.copy(
                os.path.join(source_model_path, filename),
                os.path.join(save_directory, filename),
            )
            logger.info(
                "Copied %s from %s to %s",
                filename,
                source_model_path,
                save_directory,
            )


def save_hf_config(
    config: transformers.PretrainedConfig,
    save_directory: str | os.PathLike[str],
    *,
    source_model_path: str | os.PathLike[str] | None = None,
    source_config: Mapping[str, Any] | None = None,
) -> None:
    """Save a Hugging Face config without losing its runtime ``model_type``.

    ``PretrainedConfig.to_dict()`` serializes the class-level ``model_type``.
    Some remote-code configs instead receive ``model_type`` as an instance field,
    so a normal ``save_pretrained()`` call replaces the valid runtime value with an
    empty string.  Save first, then restore and validate that value in the emitted
    ``config.json``.  A local source config may also supply fields that Transformers
    omits while re-serializing a generic config.
    """
    save_directory = os.fspath(save_directory)
    source_config = (
        load_hf_config_snapshot(source_model_path)
        if source_config is None
        else dict(source_config)
    )

    runtime_model_type = getattr(config, "model_type", None)
    expected_model_type = runtime_model_type or source_config.get("model_type")
    if expected_model_type is not None and not isinstance(expected_model_type, str):
        raise TypeError(
            "Hugging Face config model_type must be a string, got "
            f"{type(expected_model_type).__name__}."
        )

    # Snapshot source fields before save_pretrained() because source and
    # destination may be the same directory.
    config.save_pretrained(save_directory)

    saved_config_path = Path(save_directory) / "config.json"
    with saved_config_path.open() as f:
        saved_config = json.load(f)

    patched_fields: list[str] = []
    if expected_model_type and saved_config.get("model_type") != expected_model_type:
        saved_config["model_type"] = expected_model_type
        patched_fields.append(f"model_type={expected_model_type}")

    if "torch_dtype" not in saved_config and "torch_dtype" in source_config:
        saved_config["torch_dtype"] = source_config["torch_dtype"]
        patched_fields.append(f"torch_dtype={source_config['torch_dtype']}")

    if patched_fields:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=save_directory,
            prefix=".config.json.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(saved_config, tmp, indent=2)
            tmp.write("\n")
            tmp_path = tmp.name
        os.replace(tmp_path, saved_config_path)
        logger.info("Patched config.json: %s", ", ".join(patched_fields))

    with saved_config_path.open() as f:
        persisted_config = json.load(f)
    if (
        expected_model_type
        and persisted_config.get("model_type") != expected_model_type
    ):
        raise RuntimeError(
            "Saved Hugging Face config has an invalid model_type: expected "
            f"{expected_model_type!r}, got {persisted_config.get('model_type')!r}."
        )


def finalize_hf_export(
    config: transformers.PretrainedConfig,
    save_directory: str | os.PathLike[str],
    *,
    source_model_path: str | os.PathLike[str] | None = None,
    source_config: Mapping[str, Any] | None = None,
) -> None:
    """Finalize an HF export with source assets and a validated config."""
    save_directory = os.fspath(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    local_source_path: str | None = None
    if source_model_path is not None:
        candidate = os.fspath(source_model_path)
        if not os.path.isdir(candidate):
            logger.warning(
                "Cannot copy source Hugging Face assets: model path is not a "
                "local directory: %s",
                candidate,
            )
        else:
            local_source_path = candidate
            if os.path.samefile(candidate, save_directory):
                logger.warning(
                    "Skipping source Hugging Face asset copy because the source "
                    "and checkpoint directories are the same: %s",
                    save_directory,
                )
            else:
                copy_hf_model_assets(candidate, save_directory)

    save_hf_config(
        config,
        save_directory,
        source_model_path=local_source_path,
        source_config=source_config,
    )


def load_hf_config_snapshot(
    source_model_path: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    """Read a source config before an exporter can overwrite it in place."""
    if source_model_path is None:
        return {}
    source_config_path = Path(source_model_path) / "config.json"
    if not source_config_path.is_file():
        return {}
    with source_config_path.open() as f:
        return json.load(f)


@overload
def apply_chat_template(
    tokenizer: transformers.PreTrainedTokenizerFast,
    messages: list[dict[str, Any]],
    *,
    tokenize: Literal[True] = ...,
    **kwargs: Any,
) -> list[int]: ...


@overload
def apply_chat_template(
    tokenizer: transformers.PreTrainedTokenizerFast,
    messages: list[dict[str, Any]],
    *,
    tokenize: Literal[False],
    **kwargs: Any,
) -> str: ...


def apply_chat_template(
    tokenizer: transformers.PreTrainedTokenizerFast,
    messages: list[dict[str, Any]],
    *,
    tokenize: bool = True,
    **kwargs: Any,
) -> list[int] | str:
    """Apply chat template, normalising transformers >=5.0 dict return to list[int]."""
    result = tokenizer.apply_chat_template(messages, tokenize=tokenize, **kwargs)
    if tokenize and pkg_version.is_version_greater_or_equal("transformers", "5.0"):
        return list(result["input_ids"])
    return result


@lru_cache(maxsize=8)
def load_hf_tokenizer(
    model_name_or_path: str,
    fast_tokenizer=True,
    padding_side: str | None = None,
) -> transformers.PreTrainedTokenizerFast:
    kwargs = {}
    if padding_side is not None:
        kwargs["padding_side"] = padding_side
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        fast_tokenizer=fast_tokenizer,
        trust_remote_code=True,
        force_download=False,
        **kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@lru_cache(maxsize=8)
def load_hf_processor_and_tokenizer(
    model_name_or_path: str,
    fast_tokenizer=True,
    padding_side: str | None = None,
) -> tuple[transformers.ProcessorMixin | None, transformers.PreTrainedTokenizerFast]:
    """Load a tokenizer and processor from Hugging Face."""
    # NOTE: use the raw type annoation will trigger cuda initialization
    tokenizer = load_hf_tokenizer(model_name_or_path, fast_tokenizer, padding_side)
    try:
        processor = transformers.AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            force_download=False,
            use_fast=True,
        )
    except Exception:
        processor = None
        logger.warning(
            f"Failed to load processor for {model_name_or_path}. "
            "Using tokenizer only. This may cause issues with some models."
        )
    return processor, tokenizer


def download_from_huggingface(
    repo_id: str, filename: str, revision: str = "main", repo_type: str = "dataset"
) -> str:
    """
    Download a file from a HuggingFace Hub repository.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "Please install huggingface_hub to use this function: pip install huggingface_hub"
        )

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        repo_type=repo_type,
    )


def load_hf_or_local_file(path: str) -> str:
    """
    Load a file from a HuggingFace Hub repository or a local file.
    hf://<org>/<repo>/<filename>
    hf://<org>/<repo>@<revision>/<filename>

    e.g,
    hf-dataset://inclusionAI/AReaL-RL-Data/data/boba_106k_0319.jsonl
    =>
    repo_type = dataset
    repo_id = inclusionAI/AReaL-RL-Data
    filename = data/boba_106k_0319.jsonl
    revision = main
    =>
    /root/.cache/huggingface/hub/models--inclusionAI--AReaL-RL-Data/data/boba_106k_0319.jsonl
    """
    path = str(path)
    if path.startswith("hf://") or path.startswith("hf-dataset://"):
        # repo_type = "dataset" if path.startswith("hf-dataset://") else "model"
        hf_path = path.strip().split("://")[1]
        hf_org, hf_repo, filename = hf_path.split("/", 2)
        repo_id = f"{hf_org}/{hf_repo}"
        revision = "main"
        if "@" in repo_id:
            repo_id, revision = repo_id.split("@", 1)
        return download_from_huggingface(repo_id, filename, revision)
    return path
