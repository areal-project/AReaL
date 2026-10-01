# SPDX-License-Identifier: Apache-2.0
"""Rollout workflow for On-Policy Self-Adaptation.

The external reward is computed for evaluation/monitoring only.
It is NOT used by the OPSA actor/update path.
"""

import uuid
from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import (
    AsyncRewardWrapper,
    InferenceEngine,
    ModelRequest,
    RolloutWorkflow,
)
from areal.api.cli_args import GenerationHyperparameters
from areal.utils import stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import apply_chat_template, load_hf_tokenizer
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)


class SelfAdaptationWorkflow(RolloutWorkflow):
    """Generate OPSA trajectories without using external rewards for training.

    The reward function is evaluated only for monitoring/evaluation.
    The OPSA actor path ignores the reward and uses its own self-adaptation
    signal.
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
    ):
        if isinstance(tokenizer, str):
            tokenizer = load_hf_tokenizer(tokenizer)

        self.tokenizer = tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.enable_thinking = enable_thinking

        # Keep reward function for evaluation/monitoring only.
        self.reward_fn = reward_fn

        # Lazily initialize when reward_fn is a string.
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> float:
        """Compute external reward for evaluation/monitoring only."""

        completions_str = self.tokenizer.decode(resp.output_tokens)

        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            resp.input_tokens,
            resp.output_tokens,
            **task_data,
        )

        return reward

    @session_context()
    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Generate one OPSA trajectory.

        The external reward is computed for evaluation/monitoring only.
        The returned `rewards` field remains a zero placeholder because
        the OPSA actor path does not consume it.
        """

        # ---------------------------------------------------------
        # Lazily load reward function if configured as a string.
        # ---------------------------------------------------------
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        # ---------------------------------------------------------
        # Build prompt using the NEW AReaL chat-template helper.
        # ---------------------------------------------------------
        input_ids = apply_chat_template(
            self.tokenizer,
            data["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        request = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )

        # Keep this consistent with the generated input IDs.
        prompt_str = self.tokenizer.decode(input_ids)

        # ---------------------------------------------------------
        # OPSA rollout
        # ---------------------------------------------------------
        async with atrace_session_phase("generate"):
            response = await engine.agenerate(request)

        # ---------------------------------------------------------
        # External reward:
        #
        # IMPORTANT:
        # This is ONLY for evaluation/monitoring.
        # It does NOT become the OPSA training reward.
        # ---------------------------------------------------------
        reward = await self._compute_rewards(
            response,
            prompt_str,
            data,
        )

        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        # ---------------------------------------------------------
        # Build trajectory
        # ---------------------------------------------------------
        sequence = response.input_tokens + response.output_tokens

        logprobs = [0.0] * response.input_len + response.output_logprobs

        loss_mask = [0] * response.input_len + [1] * response.output_len

        versions = [-1] * response.input_len + response.output_versions

        turn_ids = [-1] * response.input_len + [0] * response.output_len

        trajectory = {
            "input_ids": torch.tensor(
                sequence,
                dtype=torch.int32,
            ),
            "loss_mask": torch.tensor(
                loss_mask,
                dtype=torch.int32,
            ),
            "logprobs": torch.tensor(
                logprobs,
                dtype=torch.float32,
            ),
            "versions": torch.tensor(
                versions,
                dtype=torch.int32,
            ),
            "turn_ids": torch.tensor(
                turn_ids,
                dtype=torch.int32,
            ),
            "attention_mask": torch.ones(
                len(sequence),
                dtype=torch.bool,
            ),
            "rewards": torch.tensor(
                reward,
                dtype=torch.float32,
            ),
            "is_truncated": torch.tensor(
                response.stop_reason == "length",
                dtype=torch.bool,
            ),
        }

        return {key: value.unsqueeze(0) for key, value in trajectory.items()}
