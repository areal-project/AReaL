# SPDX-License-Identifier: Apache-2.0
"""Small random caches and identities shared by the kvmap tests."""

import torch

from areal.experimental.kvmap.apply import DenseCache
from areal.experimental.kvmap.artifact import ModelIdentity

LAYERS = 3
KV_HEADS = 2
HEAD_DIM = 8
BATCH = 4
TOKENS = 32
SOURCE_SHA = "a" * 64
TARGET_SHA = "b" * 64


def identity(sha: str, *, theta: float = 1_000_000.0) -> ModelIdentity:
    return ModelIdentity(
        checkpoint_sha256=sha,
        architecture="Qwen2ForCausalLM",
        num_layers=LAYERS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        rope_theta=theta,
        rope_type="default",
        cache_dtype="float32",
    )


def random_cache(seed: int, *, batch: int = BATCH, tokens: int = TOKENS) -> DenseCache:
    generator = torch.Generator().manual_seed(seed)
    keys = tuple(
        torch.randn(batch, KV_HEADS, tokens, HEAD_DIM, generator=generator)
        for _ in range(LAYERS)
    )
    values = tuple(
        torch.randn(batch, KV_HEADS, tokens, HEAD_DIM, generator=generator)
        for _ in range(LAYERS)
    )
    return DenseCache(keys=keys, values=values)


def positions(batch: int = BATCH, tokens: int = TOKENS) -> torch.Tensor:
    return torch.arange(tokens).unsqueeze(0).expand(batch, tokens).clone()
