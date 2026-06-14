# SPDX-License-Identifier: Apache-2.0
"""Aesthetic reward for diffusion RL post-training.

This implements the LAION aesthetic predictor: a CLIP ViT-L/14 image embedding
fed into a small MLP head that regresses a 1-10 aesthetic score. It is the
canonical reward used by ddpo-pytorch / DDPO for SD1.5 alignment.

Why scoring runs synchronously in the main process (Phase 1 decision):
``areal.api.reward_api.AsyncRewardWrapper`` dispatches reward functions through
a ``ProcessPoolExecutor``, which requires the reward callable *and its captured
state* to be picklable. The aesthetic scorer holds a CLIP backbone plus an MLP
head -- live model objects that are not picklable -- so wrapping it would crash.
For a single-GPU PoC, rollout throughput is not the bottleneck, so we score
inline. Phase 2 can revisit this with a per-process lazy singleton.

The CLIP backbone and MLP weights are loaded lazily on first use and cached on
the scorer instance, so repeated calls do not reload the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from areal.utils import logging

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger("AestheticReward")

# Default LAION aesthetic predictor (linear head over CLIP ViT-L/14 embeddings).
_DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"


class AestheticScorer:
    """LAION aesthetic predictor: CLIP embedding -> MLP -> scalar score.

    Args:
        weights_path: Path to the aesthetic MLP head state-dict
            (``sac+logos+ava1-l14-linearMSE.pth`` style). If ``None`` or the
            file is missing, scoring raises a clear error rather than silently
            returning zeros.
        clip_model: HuggingFace id of the CLIP backbone.
        device: Device to run the scorer on.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        clip_model: str = _DEFAULT_CLIP_MODEL,
        device: str = "cuda",
    ):
        self.weights_path = weights_path
        self.clip_model = clip_model
        self.device = device

        self._clip = None
        self._processor = None
        self._mlp = None

    def _lazy_init(self):
        if self._mlp is not None:
            return

        import os

        if self.weights_path is None or not os.path.isfile(self.weights_path):
            raise FileNotFoundError(
                "Aesthetic predictor weights not found. Provide a valid "
                "`weights_path` pointing to the LAION aesthetic MLP head "
                "(e.g. 'sac+logos+ava1-l14-linearMSE.pth'). "
                f"Got: {self.weights_path!r}. Phase 1 does not silently fall "
                "back to a zero reward, because that would produce a flat, "
                "uninformative training signal."
            )

        import torch
        import torch.nn as nn
        from transformers import CLIPModel, CLIPProcessor

        self._clip = CLIPModel.from_pretrained(self.clip_model).to(self.device)
        self._clip.requires_grad_(False)
        self._processor = CLIPProcessor.from_pretrained(self.clip_model)

        # The LAION aesthetic MLP head over 768-d CLIP embeddings.
        mlp = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        state = torch.load(self.weights_path, map_location="cpu")
        mlp.load_state_dict(state)
        self._mlp = mlp.to(self.device).eval()
        logger.info(
            f"Loaded aesthetic scorer: clip={self.clip_model}, "
            f"weights={self.weights_path}"
        )

    def score(self, image: Image) -> float:
        """Return the aesthetic score (typically 1-10) for a single image."""
        self._lazy_init()

        import torch
        import torch.nn.functional as F

        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embed = self._clip.get_image_features(**inputs)
            # LAION head expects L2-normalized embeddings.
            embed = F.normalize(embed, dim=-1)
            score = self._mlp(embed.float())
        return float(score.squeeze().item())


def make_aesthetic_reward_fn(
    weights_path: str | None = None,
    clip_model: str = _DEFAULT_CLIP_MODEL,
    device: str = "cuda",
):
    """Build a reward function closing over a cached :class:`AestheticScorer`.

    Returns:
        A callable ``reward_fn(prompt, image, **kwargs) -> float`` suitable for
        the diffusion rollout workflow. The scorer is instantiated once and
        reused across calls (lazy backbone load on first invocation).
    """
    scorer = AestheticScorer(
        weights_path=weights_path, clip_model=clip_model, device=device
    )

    def aesthetic_reward_fn(prompt: str, image: Image, **kwargs) -> float:
        return scorer.score(image)

    return aesthetic_reward_fn
