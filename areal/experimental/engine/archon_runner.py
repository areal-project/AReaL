# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from areal.experimental.models.archon.pipeline_parallel import build_pipeline_schedule
from areal.models.tree_attn.module_archon import TreeAttentionMeta
from areal.utils import logging

if TYPE_CHECKING:
    from torch import nn
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining.schedules import _PipelineSchedule

    from areal.utils.data import MicroBatchItem, MicroBatchList

logger = logging.getLogger("ArchonRunner")


class _NullOutputChunks(list):
    def append(self, item: Any) -> None:
        pass


class ForwardBackwardRunner(ABC):
    """Abstract base for forward/backward execution strategies."""

    @abstractmethod
    def run(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool,
    ) -> list[torch.Tensor | dict[int, torch.Tensor]] | None:
        """Run forward (and optionally backward) pass over microbatches.

        Args:
            mb_list: List of microbatches to process.
            process_output_fn: Function to process model outputs and compute loss.
            forward_only: If True, skip backward pass.

        Returns:
            List of results from process_output_fn, or None if not applicable.
            Results can be tensors or dicts (for tree training).
        """
        ...


class SequentialRunner(ForwardBackwardRunner):
    """Sequential microbatch execution when no pipeline parallelism is used."""

    def __init__(
        self,
        model_parts: list[nn.Module],
        prepare_inputs_fn: Callable[[MicroBatchItem], tuple[dict, Any]],
    ):
        assert len(model_parts) == 1, "SequentialRunner expects exactly 1 model part"
        self.model = model_parts[0]
        self.prepare_inputs_fn = prepare_inputs_fn

    def run(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool,
    ) -> list[torch.Tensor | dict[int, torch.Tensor]]:
        results: list[torch.Tensor | dict[int, torch.Tensor]] = []

        for mb_item in mb_list:
            inputs, ctx = self.prepare_inputs_fn(mb_item)

            tree_attn_meta = None
            if ctx.trie_node is not None:
                padded_size = mb_item.padded_to_length
                assert padded_size is not None
                tree_attn_meta = TreeAttentionMeta.from_trie(
                    ctx.trie_node, padded_size, inputs["input_ids"].device
                )
                # Tree attention uses tree_attn_meta instead of cu_seqlens;
                # create dummy cu_seqlens for model interface compatibility.
                seq_len = inputs["input_ids"].shape[-1]
                cu_seqlens = torch.tensor(
                    [0, seq_len], dtype=torch.int32, device=inputs["input_ids"].device
                )
                max_seqlen = seq_len
            else:
                cu_seqlens = inputs["cu_seqlens"]
                max_seqlen = int(inputs["max_seqlen"])

            logits = self.model(
                inputs["input_ids"],
                inputs["position_ids"],
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                tree_attn_meta=tree_attn_meta,
            )
            logits = logits.squeeze(0)
            del tree_attn_meta

            result = process_output_fn(logits, ctx.to_dict())

            if result is not None:
                if forward_only:
                    # Result can be a tensor or dict (for tree training)
                    if isinstance(result, dict):
                        results.append({k: v.detach() for k, v in result.items()})
                    else:
                        results.append(result.detach())
                else:
                    result.backward()

        return results


class PipelinedRunner(ForwardBackwardRunner):
    """Unified pipeline-parallel runner supporting all schedule types."""

    def __init__(
        self,
        pp_stages: list[PipelineStage],
        prepare_inputs_fn: Callable[[MicroBatchList], tuple],
        pp_schedule: str,
        pp_group_size: int,
        has_first_stage: bool,
        has_last_stage: bool,
    ):
        self.pp_stages = pp_stages
        self.prepare_inputs_fn = prepare_inputs_fn
        self.pp_schedule = pp_schedule
        self.pp_group_size = pp_group_size
        self.has_first_stage = has_first_stage
        self.has_last_stage = has_last_stage

    def _create_schedule(
        self,
        n_microbatches: int,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
    ) -> _PipelineSchedule:
        return build_pipeline_schedule(
            stages=self.pp_stages,
            pp_schedule=self.pp_schedule,
            n_microbatches=n_microbatches,
            pp_degree=self.pp_group_size,
            loss_fn=loss_fn,
        )

    def _get_output_stage(self) -> PipelineStage:
        for stage in self.pp_stages:
            if stage.is_last:
                return stage
        raise RuntimeError("No last stage found in pp_stages")

    def _patch_skip_output_merge(self, schedule: _PipelineSchedule) -> None:
        """Patch schedule to skip output merging, halving memory usage.

        TODO(pytorch-upgrade): PyTorch 2.10+ adds step(return_outputs=False) parameter.
        This is cleaner and avoids saving outputs entirely when not needed.

        HACK: We monkey patch schedule._merge_outputs to skip the torch.cat operation
        that merges all microbatch outputs. This is safe because:
        1. Archon iterates output_chunks directly, not the merged return value
        2. output_chunks contains the same data, just as a list instead of concatenated
        3. Skipping merge avoids allocating a second copy, halving memory usage
        """
        # NOTE: Upgrading PyTorch may resolve this in the future.
        schedule._merge_outputs = lambda output_chunks: None

    def run(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool,
    ) -> list[torch.Tensor] | None:
        if not mb_list:
            if forward_only:
                return None if not self.has_last_stage else []
            else:
                return []

        n_microbatches = len(mb_list)
        batched_args, batched_kwargs, batched_target, contexts = self.prepare_inputs_fn(
            mb_list
        )
        args = batched_args if self.has_first_stage else ()

        if forward_only:
            return self._run_eval(
                n_microbatches, args, batched_kwargs, contexts, process_output_fn
            )
        else:
            return self._run_train(
                n_microbatches,
                args,
                batched_kwargs,
                batched_target,
                contexts,
                process_output_fn,
            )

    def _run_eval(
        self,
        n_microbatches: int,
        args: tuple,
        batched_kwargs: dict[str, Any],
        contexts: list,
        process_output_fn: Callable,
    ) -> list[torch.Tensor] | None:
        schedule = self._create_schedule(n_microbatches, loss_fn=None)
        self._patch_skip_output_merge(schedule)
        schedule.eval(*args, **batched_kwargs)
        if not self.has_last_stage:
            return None
        output_stage = self._get_output_stage()
        results = self._process_outputs(
            output_stage.output_chunks, contexts, process_output_fn
        )
        output_stage.output_chunks.clear()
        return results

    def _run_train(
        self,
        n_microbatches: int,
        args: tuple,
        batched_kwargs: dict[str, Any],
        batched_target: torch.Tensor | None,
        contexts: list,
        process_output_fn: Callable,
    ) -> list[torch.Tensor]:
        pp_loss_fn = self._create_loss_fn(contexts, process_output_fn)
        schedule = self._create_schedule(n_microbatches, loss_fn=pp_loss_fn)
        self._patch_skip_output_merge(schedule)

        # NOTE: Upgrading PyTorch may resolve this in the future.
        # Replace output_chunks with a null list so
        # forward_one_chunk's `output_chunks.append(output)` becomes a no-op.
        # (torch/distributed/pipelining/schedules.py)
        # This lets each microbatch's logits be freed right after its backward,
        # instead of holding all N sets of logits until step() returns.
        output_stage = None
        if self.has_last_stage:
            output_stage = self._get_output_stage()
            output_stage.output_chunks = _NullOutputChunks()

        schedule.step(*args, target=batched_target, **batched_kwargs)

        # Restore normal list so subsequent eval() calls on the same
        # stage can read output_chunks normally.
        if output_stage is not None:
            output_stage.output_chunks = []
        return []

    def _create_loss_fn(
        self,
        contexts: list,
        process_output_fn: Callable,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.has_last_stage:
            ctx_iter = iter(contexts)

            def pp_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                ctx = next(ctx_iter)
                # Squeeze batch dim: outputs (1, seq_len, vocab) -> (seq_len, vocab)
                if pred.ndim == 3:
                    pred = pred.squeeze(0)
                loss = process_output_fn(pred, ctx.to_dict())
                if loss is None:
                    return pred.mean() * 0.0
                return loss
        else:
            # Non-last stage: dummy loss that keeps all elements in computation graph
            # so autograd can compute complete pred.grad for upstream stage
            def pp_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return pred.mean() * 0.0

        return pp_loss_fn

    def _process_outputs(
        self,
        output_chunks: list[torch.Tensor],
        contexts: list,
        process_output_fn: Callable,
    ) -> list[torch.Tensor]:
        results: list[torch.Tensor] = []
        for output, ctx in zip(output_chunks, contexts, strict=True):
            # Squeeze batch dim: outputs (1, seq_len, vocab) -> (seq_len, vocab)
            if output.ndim == 3:
                output = output.squeeze(0)
            result = process_output_fn(output, ctx.to_dict())
            if result is not None:
                results.append(result.detach())
        return results


def create_runner(
    *,
    pp_enabled: bool,
    model_parts: list[nn.Module],
    prepare_inputs_fn: Callable,
    pp_stages: list[PipelineStage] | None = None,
    pp_schedule: str | None = None,
    pp_group_size: int = 1,
    has_first_stage: bool = True,
    has_last_stage: bool = True,
) -> ForwardBackwardRunner:
    """Factory function to create the appropriate runner."""
    if pp_enabled:
        assert pp_stages is not None, "pp_stages required when pp_enabled=True"
        assert pp_schedule is not None, "pp_schedule required when pp_enabled=True"
        return PipelinedRunner(
            pp_stages=pp_stages,
            prepare_inputs_fn=prepare_inputs_fn,
            pp_schedule=pp_schedule,
            pp_group_size=pp_group_size,
            has_first_stage=has_first_stage,
            has_last_stage=has_last_stage,
        )
    else:
        return SequentialRunner(
            model_parts=model_parts,
            prepare_inputs_fn=prepare_inputs_fn,
        )
