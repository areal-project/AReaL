# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Protocol

import torch
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache

from areal.experimental.dta.dta_engine import DTAEngine
from areal.experimental.dta.token_trie import TokenTrie


class KVCacheModel(Protocol):
    """Structural contract for DTA-compatible models."""

    def forward(
        self,
        tokens: torch.LongTensor,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = True,
    ) -> SimpleNamespace: ...


class DTAWrapper:
    """Engine-agnostic facade for DTA forward/backward paths."""

    def __init__(
        self,
        model: KVCacheModel,
        model_config: PretrainedConfig,
        device: torch.device,
        dtype: torch.dtype,
        max_seq_len: int,
        block_size: int,
        is_critic: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.block_size = block_size
        self.is_critic = is_critic
        self._engine = DTAEngine(
            model_config=model_config,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
            is_critic=is_critic,
        )

    @torch.no_grad()
    def run_forward(self, mb_list: Any) -> torch.Tensor:
        input_ids_list = self._extract_input_ids_list_from_mb_list(mb_list)
        max_seq_len = max((ids.numel() for ids in input_ids_list), default=0)
        input_data = [{} for _ in input_ids_list]
        trie = TokenTrie(input_ids_list, input_data, sorted=False)
        trie.forward_permute()

        output = self._engine.forward(model=self.model, token_trie=trie)
        batch_size = len(output)
        if batch_size == 0:
            return torch.zeros((0, 0), dtype=torch.float32, device=self.device)
        output_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=output[0].dtype,
            device=output[0].device,
        )
        for i, seq in enumerate(output):
            seq_len = seq.shape[0]
            output_padded[i, :seq_len] = seq
        return output_padded

    @staticmethod
    def _extract_input_ids(mb_input: dict[str, Any]) -> torch.Tensor:
        if "input_ids" not in mb_input:
            raise ValueError("DTA expects `input_ids` in micro-batch input.")
        input_ids = mb_input["input_ids"]
        if not torch.is_tensor(input_ids) or input_ids.ndim != 1:
            raise ValueError(
                "DTA expects packed 1D `input_ids` in micro-batch input, "
                f"got {type(input_ids)} with ndim="
                f"{getattr(input_ids, 'ndim', 'N/A')}."
            )
        return input_ids

    def _extract_input_ids_list_from_mb_list(self, mb_list: Any) -> list[torch.Tensor]:
        input_ids_list: list[torch.Tensor] = []
        for mb_item in mb_list:
            input_ids_list.append(self._extract_input_ids(mb_item.orig_mb))
        return input_ids_list

    def run_backward_with_scaled_loss(
        self,
        mb_list: Any,
        prepare_mb_inputs_fn: Any,
        loss_fn: Any,
        loss_weight_fn: Any,
        total_loss_weight: torch.Tensor,
        block_size: int | None = None,
    ) -> dict[str, float]:
        input_ids_list = self._extract_input_ids_list_from_mb_list(mb_list)
        per_seq_input_data: list[dict[str, Any]] = []
        for idx, mb_item in enumerate(mb_list):
            _, ctx = prepare_mb_inputs_fn(mb_item)
            mb_input = ctx.mb_input
            # Keep backward input source aligned with forward input source.
            self._extract_input_ids(mb_input)
            if mb_input["input_ids"].shape != input_ids_list[idx].shape:
                raise ValueError(
                    "DTA expects `ctx.mb_input['input_ids']` to align with "
                    "`mb_item.orig_mb['input_ids']` for each micro-batch."
                )
            loss_scale = loss_weight_fn(ctx.mb_input) / total_loss_weight
            if isinstance(loss_scale, torch.Tensor):
                loss_scale = loss_scale.item()
            per_seq_input_data.append({"original": mb_input, "scale": loss_scale})

        if self.is_critic:

            def scaled_loss_fn(
                values: torch.Tensor,
                seq_input_data: dict[str, Any],
                **extra_kwargs: Any,
            ) -> torch.Tensor:
                loss_val = loss_fn(
                    values,
                    seq_input_data["original"],
                    **extra_kwargs,
                )
                return loss_val * seq_input_data["scale"]
        else:

            def scaled_loss_fn(
                logprobs: torch.Tensor,
                entropy: torch.Tensor,
                seq_input_data: dict[str, Any],
                **extra_kwargs: Any,
            ) -> torch.Tensor:
                # Keep current behavior: DTA engine expects one extra position.
                logprobs = torch.cat([logprobs, logprobs.new_zeros(1)], dim=0)
                loss_val = loss_fn(
                    logprobs,
                    entropy,
                    seq_input_data["original"],
                    **extra_kwargs,
                )
                return loss_val * seq_input_data["scale"]

        trie = TokenTrie(input_ids_list, per_seq_input_data, sorted=False)
        trie.backward_permute()

        total_loss = self._engine.backward(
            model=self.model,
            token_trie=trie,
            block_size=block_size or self.block_size,
            loss_fn=scaled_loss_fn,
        )
        return {"dta_loss": float(total_loss)}
