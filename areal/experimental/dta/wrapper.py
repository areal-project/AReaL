# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from transformers.cache_utils import DynamicCache

from areal.api.cli_args import MicroBatchSpec
from areal.engine.core.train_engine import (
    compute_total_loss_weight,
    reorder_and_pad_outputs,
)
from areal.experimental.dta.runner import DTARunner
from areal.experimental.dta.token_trie import TokenTrie
from areal.utils.data import (
    MicroBatchList,
    amend_position_ids,
    pack_tensor_dict,
    split_batch,
    split_padded_tensor_dict_into_mb_list,
)

if TYPE_CHECKING:
    from areal.experimental.engine.archon_engine import ArchonEngine


class KVCacheModel(Protocol):
    """Structural contract for DTA-compatible models."""

    def forward(
        self,
        tokens: torch.LongTensor,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = True,
    ) -> SimpleNamespace: ...


class DTAWrapper:
    """DTA adapter wrapped around ArchonEngine's batch-level APIs.

    DTA is adapted at the train_batch boundary instead of the generic runner
    boundary. The runner contract exposes dense logits to process_output_fn,
    while DTA intentionally keeps backward activation bounded by block_size
    tokens inside DTARunner.
    """

    def validate_compatibility(self) -> None:
        """Validate DTA-only constraints before model creation."""
        config = self.engine.config
        parallel_dims = self.engine.parallel_dims
        model_type = getattr(self.engine.model_config, "model_type", "")

        if model_type in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}:
            raise ValueError(
                "DTA requires model-level KV-cache support via "
                "forward(..., past_key_values=..., use_cache=True). "
                f"model_type={model_type!r} is not supported because Qwen3.5 "
                "hybrid linear_attention layers do not implement DTA-compatible "
                "cache state."
            )

        if config.gradient_checkpointing:
            raise ValueError(
                "ArchonEngine: gradient_checkpointing=True is incompatible with "
                "tree_training_mode='dta'. Disable gradient_checkpointing for DTA."
            )

        if (
            parallel_dims.pp_enabled
            or parallel_dims.cp_enabled
            or parallel_dims.tp_enabled
            or parallel_dims.ep_enabled
            or parallel_dims.etp_enabled
        ):
            raise ValueError(
                "DTA currently supports only data parallelism. "
                "Found unsupported parallel dimensions enabled among "
                "{pp, cp, tp, ep, etp}. "
                f"Current sizes: pp={parallel_dims.pp}, cp={parallel_dims.cp}, "
                f"tp={parallel_dims.tp}, ep={parallel_dims.ep}, etp={parallel_dims.etp}."
            )

    def __init__(
        self,
        engine: ArchonEngine,
    ) -> None:
        self.engine = engine
        self.device = engine.device
        self.block_size = engine.config.dta_block_size
        self.is_critic = engine.config.is_critic
        self.runner = DTARunner(
            model_config=engine.model_config,
            device=engine.device,
            dtype=getattr(torch, engine.config.dtype),
            max_seq_len=engine.config.mb_spec.max_tokens_per_mb,
            is_critic=engine.config.is_critic,
        )

    @property
    def model(self) -> KVCacheModel:
        return self.engine.model

    def apply_zero1(self) -> None:
        """Apply DTA's Zero1 full-replica model setup."""
        model_args = getattr(self.engine.model, "model_args", None)
        if getattr(model_args, "enable_weight_tying", False):
            output = getattr(self.engine.model, "output", None)
            tok_embeddings = getattr(self.engine.model, "tok_embeddings", None)
            if output is not None and tok_embeddings is not None:
                output.weight = tok_embeddings.weight
        self.engine.model_parts = [self.engine.model]

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create DTA's Zero1 optimizer."""
        assert self.engine.optimizer_config is not None
        optimizer_config = self.engine.optimizer_config
        common_kwargs: dict[str, object] = {
            "lr": optimizer_config.lr,
            "weight_decay": optimizer_config.weight_decay,
        }
        if optimizer_config.type == "adam":
            return ZeroRedundancyOptimizer(
                self.engine._get_all_parameters(),
                optimizer_class=torch.optim.AdamW,
                process_group=self.engine.data_parallel_group,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
                eps=optimizer_config.eps,
                fused=True,
                **common_kwargs,
            )
        if optimizer_config.type == "sgd":
            return ZeroRedundancyOptimizer(
                self.engine._get_all_parameters(),
                optimizer_class=torch.optim.SGD,
                process_group=self.engine.data_parallel_group,
                **common_kwargs,
            )
        raise ValueError(
            f"Unsupported optimizer type for Zero1: {optimizer_config.type}"
        )

    def clip_grad_norm(self) -> float:
        """Clip gradients for DTA's Zero1 full-replica training path."""
        assert self.engine.optimizer_config is not None
        grads = [
            p.grad for p in self.engine._get_all_parameters() if p.grad is not None
        ]
        if not grads:
            return 0.0

        device = grads[0].device
        total_sq = torch.zeros((), device=device, dtype=torch.float32)
        for grad in grads:
            total_sq += grad.detach().float().pow(2).sum()

        total_norm = total_sq.sqrt()
        total_norm_value = float(total_norm)
        if not math.isfinite(total_norm_value):
            return total_norm_value

        clip_coef = (
            self.engine.optimizer_config.gradient_clipping / (total_norm + 1e-6)
        ).clamp(max=1.0)
        for grad in grads:
            grad.mul_(clip_coef.to(device=grad.device, dtype=grad.dtype))
        return total_norm_value

    def prepare_mb_list(self, input_: dict[str, Any]) -> MicroBatchList:
        """Build one sequence per microbatch for DTARunner."""
        input_ = amend_position_ids(input_)
        n_seqs = input_["input_ids"].shape[0]
        mb_spec = MicroBatchSpec.new(
            self.engine.config.mb_spec,
            n_mbs=n_seqs,
            granularity=1,
            n_mbs_divisor=1,
            max_tokens_per_mb=self.engine.config.mb_spec.max_tokens_per_mb,
        )
        # Keep DTA per-rank independent: one sequence per microbatch, no
        # cross-rank synced microbatch-count alignment.
        mb_list = split_padded_tensor_dict_into_mb_list(input_, mb_spec, sync_mbs=False)
        assert len(mb_list.mbs) == n_seqs, (
            f"DTA requires one microbatch per sequence, "
            f"expected {n_seqs} microbatches but got {len(mb_list.mbs)}."
        )
        return mb_list

    @torch.no_grad()
    def forward_batch(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[torch.Tensor]], torch.Tensor] = torch.cat,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Forward pass through DTA, matching ArchonEngine.forward_batch."""
        engine = self.engine
        assert engine._initialized

        input_batched, meta = engine._normalize_batch_input(input_)

        cu_seqlens = pack_tensor_dict(input_batched)["cu_seqlens"]
        inferred_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()
        if meta is not None:
            assert isinstance(input_, list)
            if output_seqlens is not None and output_seqlens != inferred_seqlens:
                raise ValueError(
                    f"output_seqlens mismatch for list input: "
                    f"given {output_seqlens}, "
                    f"inferred {inferred_seqlens} from attention_mask valid lengths."
                )
            output_seqlens = inferred_seqlens
        elif output_seqlens is None:
            output_seqlens = inferred_seqlens
        assert output_seqlens is not None

        mb_list = engine._prepare_mb_list(input_batched).to(engine.device)
        engine.logger.info("tree_training_mode='dta' in forward_batch")
        input_ids_list = self._extract_input_ids_list_from_mb_list(mb_list)
        input_data = [{} for _ in input_ids_list]
        trie = TokenTrie(input_ids_list, input_data, sorted=False)
        trie.forward_permute()

        outputs = self.runner.forward(model=self.model, token_trie=trie)
        if not self.is_critic:
            outputs = [
                torch.cat([x, x.new_zeros((1, *x.shape[1:]))], dim=0) for x in outputs
            ]
        res = reorder_and_pad_outputs(outputs, output_seqlens, mb_list, aggregate_fn)
        if meta is None:
            return res
        return split_batch(res, meta)

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

    def train_batch(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        return_loss: bool = False,
    ) -> dict[str, float]:
        """Train on a batch using DTA's block-wise backward implementation."""
        engine = self.engine
        assert engine._initialized
        engine.optimizer_zero_grad()

        input_batched, _ = engine._normalize_batch_input(input_)
        mb_list = engine._prepare_mb_list(input_batched).to(engine.device)
        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, engine.data_parallel_group
        )

        engine.logger.info("tree_training_mode='dta' in train_batch")
        engine.logger.info(f"total_loss_weight: {total_loss_weight}")

        dta_loss = self._backward_with_scaled_loss(
            mb_list=mb_list,
            loss_fn=loss_fn,
            loss_weight_fn=loss_weight_fn,
            total_loss_weight=total_loss_weight,
        )
        engine.logger.info(f"DTA backward loss: {dta_loss}")

        for parameter in engine._get_all_parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, group=engine.data_parallel_group)
        result = engine.optimizer_step()
        if return_loss:
            result["loss"] = dta_loss
        return result

    def _backward_with_scaled_loss(
        self,
        mb_list: MicroBatchList,
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        total_loss_weight: torch.Tensor,
    ) -> float:
        input_ids_list = self._extract_input_ids_list_from_mb_list(mb_list)
        per_seq_input_data: list[dict[str, Any]] = []
        for idx, mb_item in enumerate(mb_list):
            _, ctx = self.engine._prepare_mb_inputs(mb_item)
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

        return float(
            self.runner.backward(
                model=self.model,
                token_trie=trie,
                block_size=self.block_size,
                loss_fn=scaled_loss_fn,
            )
        )
