# SPDX-License-Identifier: Apache-2.0
"""Read native caches out of a Hugging Face causal LM and decode one boundary token from a given cache.

Boundary convention (see the WP1 lifecycle trace, section 6): a request holds tokens
``t_1..t_n``. The cache covers ``t_1..t_{n-1}`` at positions ``0..n-2``; ``next_token_logits``
feeds ``t_n`` at position ``n-1`` and returns the logits that sample ``t_{n+1}``. This is what
a rebuild computes, so a translated cache can be judged against a native one.
Contract: ``input_ids`` and ``attention_mask`` are ``[batch, tokens]``; left padding is not
supported, ``attention_mask`` must be a right-aligned prefix mask.
"""

import hashlib
import pathlib

import torch

from areal.experimental.kvmap.apply import DenseCache
from areal.experimental.kvmap.artifact import ModelIdentity

HASHED_CHECKPOINT_SUFFIXES = (".safetensors", ".json")


def hash_checkpoint_directory(directory: pathlib.Path) -> str:
    """sha256 over the sorted names and bytes of the weight and config files in a checkpoint."""
    directory = pathlib.Path(directory)
    files = sorted(
        p
        for p in directory.iterdir()
        if p.suffix in HASHED_CHECKPOINT_SUFFIXES and p.is_file()
    )
    if len(files) == 0:
        raise FileNotFoundError(
            f"{directory} holds no {HASHED_CHECKPOINT_SUFFIXES} files; not a checkpoint directory"
        )
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(64 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def model_identity_from_config(
    config, *, checkpoint_sha256: str, cache_dtype: torch.dtype
) -> ModelIdentity:
    """Build the identity from a transformers config, refusing rope scaling and sliding windows."""
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    rope_type = rope_parameters.get("rope_type", "default")
    rope_theta = float(
        rope_parameters.get("rope_theta", getattr(config, "rope_theta", 0.0))
    )
    if rope_theta <= 0:
        raise ValueError("config carries no positive rope_theta")
    if getattr(config, "use_sliding_window", False):
        raise ValueError(
            "sliding-window attention is unsupported by the mapper; disable it or pick another model"
        )
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    return ModelIdentity(
        checkpoint_sha256=checkpoint_sha256,
        architecture=config.architectures[0]
        if config.architectures
        else config.model_type,
        num_layers=int(config.num_hidden_layers),
        num_kv_heads=int(config.num_key_value_heads),
        head_dim=int(head_dim),
        rope_theta=rope_theta,
        rope_type=rope_type,
        cache_dtype=str(cache_dtype).removeprefix("torch."),
    )


@torch.no_grad()
def extract_cache(
    model, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[DenseCache, torch.Tensor]:
    """Prefill and return the native cache with positions ``[batch, tokens]`` (0-based, right-aligned masks)."""
    _assert_prefix_mask(attention_mask)
    positions = (
        torch.arange(input_ids.shape[1], device=input_ids.device)
        .unsqueeze(0)
        .expand_as(input_ids)
    )
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=positions,
        use_cache=True,
    )
    layers = output.past_key_values.layers
    cache = DenseCache(
        keys=tuple(layer.keys.clone() for layer in layers),
        values=tuple(layer.values.clone() for layer in layers),
    )
    return cache, positions


@torch.no_grad()
def next_token_logits(
    model,
    *,
    cache: DenseCache,
    boundary_tokens: torch.Tensor,
    cache_lengths: torch.Tensor,
) -> torch.Tensor:
    """Feed ``boundary_tokens`` ``[batch]`` at position ``cache_lengths`` ``[batch]`` over ``cache`` and return ``[batch, vocab]`` logits."""
    from transformers import DynamicCache

    batch, tokens = cache.batch, cache.tokens
    if tuple(boundary_tokens.shape) != (batch,) or tuple(cache_lengths.shape) != (
        batch,
    ):
        raise ValueError(
            f"boundary_tokens and cache_lengths must be [batch={batch}], got {tuple(boundary_tokens.shape)} and {tuple(cache_lengths.shape)}"
        )
    if bool((cache_lengths > tokens).any()) or bool((cache_lengths < 0).any()):
        raise ValueError(
            f"cache_lengths must lie in [0, {tokens}], got {cache_lengths.tolist()}"
        )
    past = DynamicCache()
    for layer, (k, v) in enumerate(zip(cache.keys, cache.values)):
        past.update(k, v, layer)
    mask = torch.zeros(
        batch, tokens + 1, dtype=torch.long, device=boundary_tokens.device
    )
    mask[:, :tokens] = (
        torch.arange(tokens, device=boundary_tokens.device).unsqueeze(0)
        < cache_lengths.unsqueeze(1)
    ).long()
    mask[:, tokens] = 1
    output = model(
        input_ids=boundary_tokens.unsqueeze(1),
        attention_mask=mask,
        position_ids=cache_lengths.unsqueeze(1),
        past_key_values=past,
        cache_position=torch.full((1,), tokens, device=boundary_tokens.device),
        use_cache=True,
    )
    return output.logits[:, -1, :]


def _assert_prefix_mask(attention_mask: torch.Tensor) -> None:
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must be [batch, tokens], got {tuple(attention_mask.shape)}"
        )
    lengths = attention_mask.sum(dim=1)
    expected = (
        torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
        < lengths.unsqueeze(1)
    ).to(attention_mask.dtype)
    if not torch.equal(attention_mask, expected):
        raise ValueError(
            "attention_mask must be a right-aligned prefix mask (ones then zeros); left padding is unsupported"
        )
