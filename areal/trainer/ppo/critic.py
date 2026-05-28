# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any

import torch

from areal.api import TrainEngine
from areal.api.cli_args import MicroBatchSpec, PPOCriticConfig
from areal.experimental.training_service.controller.controller import (
    GatewayTrainController,
)
from areal.infra import TrainController
from areal.infra.rpc.serialization import serialize_value
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils import stats_tracker
from areal.utils.data import (
    batched_call,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.functional import ppo_critic_loss_fn
from areal.utils.perf_tracer import trace_perf


class PPOCritic:
    def __init__(self, config: PPOCriticConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine

    @trace_perf("ppo_critic.compute_values", category="compute")
    @torch.no_grad()
    def compute_values(self, data: list[dict[str, Any]]) -> list[torch.Tensor]:
        return batched_call(self._compute_values, data)

    def _compute_values(self, data: dict[str, Any]) -> torch.Tensor:
        self.engine.eval()
        return self.engine.forward(
            input_=data,
            aggregate_fn=lambda xs: torch.cat([x.squeeze(-1) for x in xs], dim=-1),
        )

    @trace_perf("ppo_critic.ppo_update", category="compute")
    @stats_tracker.scope_func_wrapper("ppo_critic")
    def ppo_update(self, data: list[dict[str, Any]]) -> None:
        batched_call(self._ppo_update, data, unpack=False)

    def _ppo_update(self, data: dict[str, Any]) -> None:
        ########## Logging code starts ##########
        scalars = dict(
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
            eps_clip=self.config.eps_clip,
        )
        stats_tracker.scalar(**scalars)
        ########## Logging code ends ##########

        for key in ["rewards", "tot_rewards", "kl_rewards", "versions"]:
            data.pop(key, None)

        # NOTE: calling engine.train() is critical to enabling gradient checkpointing
        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )
        for mb in mb_inputs.mbs:
            train_stat = self.engine.train_batch(
                mb,
                loss_fn=functools.partial(
                    ppo_loss_fn,
                    eps_clip=self.config.eps_clip,
                ),
                loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
            )
            stats_tracker.scalar(**train_stat)


class PPOCriticController(TrainController):
    def compute_values(self, *args, **kwargs):
        return self._custom_function_call(
            "compute_values", *args, rpc_meta={"broadcast": True}, **kwargs
        )

    def ppo_update(self, *args, **kwargs):
        self._custom_function_call(
            "ppo_update", *args, rpc_meta={"broadcast": True}, **kwargs
        )


class PPOCriticControllerV2(GatewayTrainController):
    def compute_values(self, *args, **kwargs):
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        return self._gateway_post_result("/ppo/critic/compute_values", payload)

    def ppo_update(self, *args, **kwargs) -> None:
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post("/ppo/critic/update", payload)


def ppo_loss_fn(
    value: torch.Tensor,
    input_data: dict,
    eps_clip: float,
):
    """Loss function for critic step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    value = value.squeeze(-1).float()
    old_value = input_data["values"].float()
    target_value = input_data["returns"].float()
    loss_mask = input_data["loss_mask"].bool()

    loss, stat = ppo_critic_loss_fn(
        value=value,
        old_value=old_value,
        target_value=target_value,
        value_eps_clip=eps_clip,
        loss_mask=loss_mask,
    )

    # Log training statistics
    stats_tracker.denominator(
        n_tokens=infer_token_denominator(input_data, value),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
    )

    stats_tracker.stat(
        new_value=value.detach(),
        old_value=old_value,
        target_value=target_value,
        critic_loss=stat["loss"],
        clip_ratio=stat["clip_mask"].float(),
        denominator="n_valid_tokens",
    )

    clip_mask = stat["clip_mask"]
    clipped_new_value = torch.where(clip_mask, value.detach(), 0.0)
    clipped_old_value = torch.where(clip_mask, old_value, 0.0)
    stats_tracker.stat(
        clipped_new_value=clipped_new_value,
        clipped_old_value=clipped_old_value,
        denominator="clipped_tokens",
    )
    return loss
