"""Unit tests for the diffusion aesthetic scorer CLIP embedding path."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.diffusion_test_utils import assert_tensors_close, load_diffusion_module

AestheticScorer = load_diffusion_module("aesthetic_reward").AestheticScorer


class _FakeBatch(dict):
    def to(self, device: str):
        return _FakeBatch({k: v.to(device) for k, v in self.items()})


class _FakeProcessor:
    def __init__(self, pixel_values: torch.Tensor):
        self.pixel_values = pixel_values

    def __call__(self, images, return_tensors: str):
        assert return_tensors == "pt"
        return _FakeBatch({"pixel_values": self.pixel_values.clone()})


class _RaisingGetImageFeaturesClip:
    def __init__(self, pooler_output: torch.Tensor):
        self.pooler_output = pooler_output
        self.vision_model_calls = 0

    def get_image_features(self, **kwargs):  # pragma: no cover - regression only
        raise AssertionError("score() should not rely on get_image_features()")

    def vision_model(self, *, pixel_values: torch.Tensor):
        self.vision_model_calls += 1
        return SimpleNamespace(pooler_output=self.pooler_output.clone())

    def visual_projection(self, pooled_output: torch.Tensor):
        return pooled_output * 2


class _TupleVisionClip:
    def __init__(self, pooled_output: torch.Tensor):
        self.pooled_output = pooled_output

    def vision_model(self, *, pixel_values: torch.Tensor):
        return ("ignored", self.pooled_output.clone())

    def visual_projection(self, pooled_output: torch.Tensor):
        return pooled_output + 3


class _FakeMLP:
    def __init__(self):
        self.last_input = None

    def __call__(self, embed: torch.Tensor):
        self.last_input = embed.clone()
        return embed.sum(dim=-1, keepdim=True)


def test_aesthetic_score_uses_vision_model_projection_path():
    """Regression: score must not depend on transformers 5.x get_image_features."""
    pixel_values = torch.randn(1, 3, 4, 4)
    pooled_output = torch.tensor([[3.0, 4.0]])
    scorer = AestheticScorer(weights_path="unused", device="cpu")
    scorer._processor = _FakeProcessor(pixel_values)
    scorer._clip = _RaisingGetImageFeaturesClip(pooled_output)
    scorer._mlp = _FakeMLP()

    score = scorer.score(image=object())

    expected_embed = torch.tensor([[0.6, 0.8]])
    assert isinstance(score, float)
    assert scorer._clip.vision_model_calls == 1
    assert_tensors_close(scorer._mlp.last_input, expected_embed)
    assert_tensors_close(torch.tensor(score), expected_embed.sum())


def test_extract_image_embedding_accepts_tuple_vision_outputs():
    """Compatibility: tuple vision outputs still map to projected embeddings."""
    scorer = AestheticScorer(weights_path="unused", device="cpu")
    scorer._clip = _TupleVisionClip(torch.tensor([[1.0, 2.0]]))

    embed = scorer._extract_image_embedding(torch.randn(1, 3, 4, 4))

    assert_tensors_close(embed, torch.tensor([[4.0, 5.0]]))
