# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from transformers import PreTrainedTokenizerFast

from areal.api import (
    AsyncRewardWrapper,
    ModelRequest,
    RewardResult,
    RolloutWorkflow,
    normalize_reward_result,
)
from areal.api.cli_args import GenerationHyperparameters
from areal.utils import logging
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import apply_chat_template
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine
    from areal.api.io_struct import ModelResponse

logger = logging.getLogger("RLVRWorkflow")


def default_get_input_ids_fn(
    data: Any,
    tokenizer: PreTrainedTokenizerFast,
    enable_thinking: bool,
) -> list[int]:
    return apply_chat_template(
        tokenizer,
        data,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def default_data_extract_prompt_fn(data: dict[str, Any]) -> Any:
    return data["messages"]


class RLVRWorkflow(RolloutWorkflow):
    """Single-turn reward learning workflow supporting optional thinking tokens."""

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
        get_input_ids_fn: Callable[[Any, PreTrainedTokenizerFast, bool], list[int]]
        | str = default_get_input_ids_fn,
        data_extract_prompt_fn: Callable[[dict[str, Any]], Any]
        | str = default_data_extract_prompt_fn,
    ):
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(self.tokenizer)
            self.tokenizer = tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        # Support string paths for get_input_ids_fn
        if isinstance(get_input_ids_fn, str):
            get_input_ids_fn = import_from_string(get_input_ids_fn)
        self.get_input_ids_fn = get_input_ids_fn
        # Support string paths for data_extract_prompt_fn
        if isinstance(data_extract_prompt_fn, str):
            data_extract_prompt_fn = import_from_string(data_extract_prompt_fn)
        self.data_extract_prompt_fn = data_extract_prompt_fn

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> RewardResult:
        """Decode completion and compute reward.

        Traces reward phase execution for SessionTracer. Decodes output tokens
        to string, calls async reward function, and logs metric to stats tracker.

        Returns
        -------
        RewardResult
            Reward payload.
        """
        completions_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            resp.input_tokens,
            resp.output_tokens,
            **task_data,
        )

        return normalize_reward_result(reward)

    @session_context()
    async def _collect_samples(
        self,
        engine: "InferenceEngine",
        req: ModelRequest,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple["ModelResponse", RewardResult]:
        """Generate one sample and compute its reward.

        Registers a new session for this sample, calls engine.agenerate,
        computes reward, and logs metrics. SessionTracer automatically
        tracks generate and reward phases via @trace_session decorators.

        Returns
        -------
        tuple[ModelResponse, RewardResult]
            Model response and reward payload.
        """
        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)

        reward = await self._compute_rewards(resp, prompt_str, task_data)

        from areal.infra import workflow_context
        from areal.utils import stats_tracker

        tracker = stats_tracker.get(workflow_context.stat_scope())
        tracker.scalar(reward=reward.final_reward)
        if reward.step_rewards is not None:
            tracker.scalar(
                step_reward_count=len(reward.step_rewards),
                step_reward_sum=float(sum(reward.step_rewards)),
            )

        return resp, reward

    def _build_stepwise_reward_tensors(
        self,
        reward: RewardResult,
        resp: "ModelResponse",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert stepwise rewards into completion-aligned tensors.

        Returns tensors of shape ``[seqlen]`` aligned to the full prompt +
        completion sequence. Reward values are injected at the PPO reward
        timesteps that correspond to predicting the completion token indicated
        by ``reward.step_ends``.
        """

        seq_len = resp.input_len + resp.output_len
        step_rewards = torch.zeros(seq_len, dtype=torch.float32)
        step_reward_mask = torch.zeros(seq_len, dtype=torch.bool)

        if reward.step_rewards is None and reward.step_ends is None:
            return step_rewards, step_reward_mask
        if reward.step_rewards is None or reward.step_ends is None:
            raise ValueError(
                "step_rewards and step_ends must either both be provided or both be None"
            )
        if len(reward.step_rewards) != len(reward.step_ends):
            raise ValueError(
                "step_rewards and step_ends must have the same length, got "
                f"{len(reward.step_rewards)} and {len(reward.step_ends)}"
            )

        completion_len = resp.output_len
        for step_reward, step_end in zip(reward.step_rewards, reward.step_ends):
            if step_end <= 0 or step_end > completion_len:
                raise ValueError(
                    f"Invalid step_end={step_end}; expected 1 <= step_end <= {completion_len}"
                )
            seq_index = resp.input_len + step_end - 2
            if seq_index < 0:
                raise ValueError(
                    f"Invalid aligned step index for step_end={step_end}; "
                    "reward injection requires at least one conditioning token"
                )
            step_rewards[seq_index] += float(step_reward)
            step_reward_mask[seq_index] = True

        return step_rewards, step_reward_mask

    async def arun_episode(
        self, engine: "InferenceEngine", data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        # NOTE: load reward function dynamically if given as string
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        input_ids = self.get_input_ids_fn(
            self.data_extract_prompt_fn(data),
            self.tokenizer,
            self.enable_thinking,
        )
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )

        prompt_str = self.tokenizer.decode(input_ids)

        # Generate single response and compute reward
        resp, reward = await self._collect_samples(engine, req, prompt_str, data)

        # Build result tensor dict with batch dim 1
        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions
        step_rewards, step_reward_mask = self._build_stepwise_reward_tensors(
            reward, resp
        )

        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
            "rewards": torch.tensor(reward.final_reward, dtype=torch.float32),
            "step_rewards": step_rewards,
            "step_reward_mask": step_reward_mask,
        }
        return {k: v.unsqueeze(0) for k, v in res.items()}
