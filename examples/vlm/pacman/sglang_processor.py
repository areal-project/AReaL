# SPDX-License-Identifier: Apache-2.0

"""Imported only by SGLang-enabled rollout processes."""

from typing import Any

import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class AllowedTokensProcessor(CustomLogitProcessor):
    def __call__(
        self, logits: torch.Tensor, custom_param_list: list[dict[str, Any]]
    ) -> torch.Tensor:
        choices = [params["allowed_token_ids"] for params in custom_param_list]
        if not choices or any(not ids for ids in choices):
            raise ValueError("Every action request requires a nonempty candidate set")
        width = max(map(len, choices))
        indices = torch.tensor(
            [ids + [0] * (width - len(ids)) for ids in choices],
            dtype=torch.long,
            device=logits.device,
        )
        values = torch.tensor(
            [[1] * len(ids) + [0] * (width - len(ids)) for ids in choices],
            dtype=torch.int32,
            device=logits.device,
        )
        keep = torch.zeros_like(logits, dtype=torch.int32).scatter_add_(
            1, indices, values
        )
        return logits.masked_fill(keep == 0, -torch.inf)
