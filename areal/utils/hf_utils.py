# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, overload

import transformers

import areal.utils.logging as logging
from areal.utils import pkg_version

logger = logging.getLogger("HFUtils")


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
        force_download=True,
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
            force_download=True,
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
