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


#: Fallback media placeholders for processors that declare none. Prefer
#: :func:`vision_pad_tokens`, which asks the model first; these names are only
#: correct for the Qwen family.
VISION_PAD_TOKENS = ("<|image_pad|>", "<|video_pad|>")

#: Processor attributes naming a modality's placeholder, in tokenizer form.
_PAD_TOKEN_ATTRS = ("image_token", "video_token")

#: Processor attributes naming the collapsed marker -- the single token that
#: stands for one media item before expansion. Qwen reuses its pad token here,
#: but Gemma3 does not: its processor rewrites ``boi_token`` into a run built
#: from ``image_token``, so counting pads in a collapsed prompt finds none.
_MEDIA_MARKER_ATTRS = ("boi_token",)


def _is_multimodal_processor(obj: Any) -> bool:
    """Whether ``obj`` is a real processor rather than a bare tokenizer.

    ``AutoProcessor.from_pretrained`` does not raise for a text-only model; it
    returns that model's tokenizer. A tokenizer answers enough of the processor
    API to be mistaken for one, so a text-only deployment would be treated as
    multimodal and run the vision canary against a processor that cannot
    process an image.
    """
    if obj is None or not hasattr(obj, "tokenizer"):
        return False
    return any(
        hasattr(obj, attr)
        for attr in ("image_processor", "video_processor", "image_token")
    )


def _resolve_pad_token(tokenizer: Any, token: str) -> int | None:
    """Return the id of ``token``, or None if this tokenizer has no such token."""
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
        # Unknown tokens come back as None or the unk id, so round-trip.
        if token_id is None or tokenizer.convert_ids_to_tokens(token_id) != token:
            return None
    except Exception:
        return None
    return token_id


def vision_pad_tokens(
    tokenizer: Any | None, *, processor: Any | None = None
) -> tuple[str, ...]:
    """The model's own media placeholders, in tokenizer form.

    Asks the processor first. vLLM builds its ``PromptReplacement`` target from
    ``hf_processor.image_token``, so reading the same attribute makes a sampler
    ban aim at exactly the token vLLM will expand, instead of at a name that
    merely happens to match for one model family.

    Given a tokenizer, names it cannot resolve to a real special token are
    dropped. Passing one to vLLM is not a harmless no-op: ``bad_words`` takes
    strings and vLLM re-encodes them, so a name the vocabulary lacks silently
    prohibits some ordinary sub-word sequence instead of a placeholder.

    Without a tokenizer the names are returned unchecked. Banning the declared
    placeholders is the safer failure: a caller that omits its tokenizer is then
    no less protected than before this check existed.
    """
    declared = [getattr(processor, attr, None) for attr in _PAD_TOKEN_ATTRS]
    names = tuple(token for token in declared if isinstance(token, str) and token)
    names = names or VISION_PAD_TOKENS
    if tokenizer is None:
        return names
    return tuple(
        token for token in names if _resolve_pad_token(tokenizer, token) is not None
    )


def media_marker_token_ids(
    tokenizer: Any, *, processor: Any | None = None
) -> frozenset[int]:
    """Ids of the collapsed markers: one per media item before expansion.

    This is what vLLM's ``PromptReplacement`` matches, so it is what a collapsed
    prompt must carry exactly once per item. Distinct from
    :func:`vision_pad_token_ids`, which names the tokens the *expanded* run is
    built from -- the same token for Qwen, ``boi_token`` vs ``image_token`` for
    Gemma3.
    """
    declared = [getattr(processor, attr, None) for attr in _MEDIA_MARKER_ATTRS]
    names = tuple(token for token in declared if isinstance(token, str) and token)
    if not names:
        # No distinct marker declared: the model collapses to its pad token.
        return vision_pad_token_ids(tokenizer, processor=processor)
    return frozenset(
        token_id
        for token_id in (_resolve_pad_token(tokenizer, token) for token in names)
        if token_id is not None
    )


def vision_pad_token_ids(
    tokenizer: Any, *, processor: Any | None = None
) -> frozenset[int]:
    """Reserved vision-placeholder ids, which must never appear in generated text.

    A prompt's pad-token count is fixed by the processor from the supplied
    images, and both the processor re-render and the VLM forward count those
    positions. A sampled pad token adds a position no image tensor describes,
    so it breaks the embedding merge in training and on the next turn.

    Pass ``processor`` wherever one is in hand, so the response filter and the
    sampler ban agree on what counts as a placeholder.
    """
    return frozenset(
        token_id
        for token_id in (
            _resolve_pad_token(tokenizer, token)
            for token in vision_pad_tokens(tokenizer, processor=processor)
        )
        if token_id is not None
    )


def collapsed_prompt_token_ids(processor: Any, text: str) -> list[int]:
    """Tokenize a rendered prompt without expanding media placeholders.

    vLLM's tokenized multimodal path expands placeholders itself and would
    expand an already-expanded prompt a second time, so the wire carries this
    form while the processor's expanded ``input_ids`` stay authoritative for
    training.

    A processor expands the placeholders inside ``text`` and then hands the
    result to its own tokenizer, so the collapsed form has to go through that
    same tokenizer. ``tokenizer.encode(text, add_special_tokens=False)`` is not
    a safe substitute; it agrees on Qwen3-VL but that is not a contract.

    Never cache this: the caller owns the returned list and the engine appends
    generated tokens to it in place while resuming an interrupted generation.
    A shared list would leak one rollout's output into the next prompt that
    happens to render identically.
    """
    return list(processor.tokenizer([text], padding=False)["input_ids"][0])


@lru_cache(maxsize=8)
def load_hf_processor(
    model_name_or_path: str,
) -> transformers.ProcessorMixin | None:
    """Load a processor from Hugging Face, returning ``None`` for text-only models.

    Unlike :func:`load_hf_processor_and_tokenizer` this never forces a re-download,
    so it is safe to call from short-lived worker processes.
    """
    try:
        processor = transformers.AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception as e:
        logger.info(
            f"No processor available for {model_name_or_path} ({e}). "
            "Treating the model as text-only."
        )
        return None
    if not _is_multimodal_processor(processor):
        logger.info(
            f"AutoProcessor returned {type(processor).__name__} for "
            f"{model_name_or_path}, which cannot process media. "
            "Treating the model as text-only."
        )
        return None
    return processor


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
