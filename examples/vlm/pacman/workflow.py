# SPDX-License-Identifier: Apache-2.0

"""Native VLM adapter around the pinned release's environment and reward loop."""

import asyncio
import hashlib
import math
import threading
import uuid
from concurrent.futures import Future
from typing import Any

import torch
from areal_pacman.level1.workflow import (
    ModelTurn,
    PacmanImageOnlyWorkflow,
    PacmanNativeVisionWorkflow,
)

from areal.api import ModelRequest, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.utils.data import concat_padded_tensors
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.image import image2base64


class _Episode(PacmanImageOnlyWorkflow):
    """One isolated environment in a worker thread; generation stays on the engine loop."""

    def __init__(
        self, owner: "PacmanWorkflow", engine: Any, loop: asyncio.AbstractEventLoop
    ):
        super().__init__(**owner.options)
        self.owner, self.engine, self.loop = owner, engine, loop
        self.processor, self.tokenizer = owner.processor, owner.tokenizer
        self.turns: dict[str, tuple[dict[str, Any], Any, list[int]]] = {}
        self.cancelled = threading.Event()
        self.pending: Future | None = None

    def cancel(self) -> None:
        self.cancelled.set()
        if self.pending is not None:
            self.pending.cancel()

    @staticmethod
    def _pil_and_chat_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[Any, list[dict[str, Any]]]:
        return PacmanNativeVisionWorkflow._pil_and_chat_messages(messages)

    async def _call_model(
        self, messages: list[dict[str, Any]], **options: Any
    ) -> ModelTurn:
        if self.cancelled.is_set():
            raise asyncio.CancelledError
        image, _, processed, input_ids = PacmanNativeVisionWorkflow._process_messages(
            self, messages
        )
        constraint = options.get("objective_constraint")
        if constraint is not None:
            if constraint.max_new_tokens != 1:
                raise ValueError("The release requires one token per Edward option")
            allowed = list(constraint.allowed_token_ids)
        else:
            allowed = [
                self.action_token_id_by_action[action]
                for action in options["current_open_actions"]
            ]
        gconfig = self.owner.gconfig.new(n_samples=1)
        if len(input_ids) + 1 > gconfig.max_tokens:
            raise ValueError(
                "Multimodal prompt exceeds max_tokens; truncation is forbidden"
            )
        request_id = uuid.uuid4().hex
        request = ModelRequest(
            rid=request_id,
            input_ids=input_ids,
            image_data=image2base64(image),
            gconfig=gconfig,
            tokenizer=self.tokenizer,
            processor=self.processor,
            metadata={"allowed_token_ids": allowed},
        )
        self.pending = asyncio.run_coroutine_threadsafe(
            self.engine.agenerate(request), self.loop
        )
        if self.cancelled.is_set():
            self.pending.cancel()
        try:
            response = await asyncio.wrap_future(self.pending)
        finally:
            self.pending = None
        if response.input_tokens != input_ids:
            raise RuntimeError(
                "Inference input differs from the training processor input"
            )
        if not (
            len(response.output_tokens)
            == len(response.output_logprobs)
            == len(response.output_versions)
            == 1
        ):
            raise RuntimeError(
                "Every decision requires one token, log-probability and version"
            )
        if response.output_tokens[0] not in allowed:
            raise RuntimeError("Sampled action is outside the recorded support")
        completion = self.tokenizer.decode(
            response.output_tokens, skip_special_tokens=True
        )
        if constraint is not None and constraint.option_for_tokens(
            response.output_tokens
        ) != constraint.option_for_completion(completion):
            raise RuntimeError("Edward token and decoded option disagree")
        self.turns[request_id] = processed, response, allowed
        return ModelTurn(
            completion=completion,
            completion_id=request_id,
            messages=messages,
            raw_response={
                "input_len": response.input_len,
                "output_len": response.output_len,
                "stop_reason": response.stop_reason,
            },
            request_extra_body={
                "native_areal_inference": True,
                "enable_thinking": False,
                "allowed_token_ids": allowed,
            },
        )

    def run_in_thread(self, data: dict[str, Any]) -> dict[str, Any] | None:
        return asyncio.run(self.collect(data))

    async def collect(self, data: dict[str, Any]) -> dict[str, Any] | None:
        rewards = await self.run(data)
        if rewards is None and not self.turns and self.last_episode is None:
            return None
        if (
            not isinstance(rewards, dict)
            or not rewards
            or set(rewards) != set(self.turns)
            or self.last_episode is None
        ):
            raise RuntimeError(
                "Complete episode rewards and recorded model decisions disagree"
            )
        if any(not math.isfinite(float(value)) for value in rewards.values()):
            raise ValueError("Completion rewards must be finite")
        episode_kwargs = {}
        if self.owner.options["reward_objective_contract"] == "episode_return_group_v1":
            digest = hashlib.blake2b(
                self.last_episode["trajectory_sample_id"].encode(), digest_size=8
            ).digest()
            episode_kwargs = {
                "rollout_episode_id": int.from_bytes(digest, "big") & ((1 << 63) - 1),
                "rollout_episode_return": float(
                    self.last_episode["total_shaped_reward"]
                ),
                "rollout_episode_group_size": self.owner.gconfig.n_samples,
            }
        samples = []
        for rid, reward in rewards.items():
            processed, response, allowed = self.turns[rid]
            sample = PacmanNativeVisionWorkflow._tensor_sample(
                processed,
                response,
                episode_kwargs.get("rollout_episode_return", reward),
                [],
                [allowed],
                **episode_kwargs,
            )
            support = sample.pop("pacman_allowed_token_ids")
            sample["policy_support"] = torch.roll(support, -1, 1)
            sample["policy_support"][:, -1] = 0
            for name in ("ids", "returns", "group_sizes"):
                old = f"rollout_episode_{name}"
                if old in sample:
                    sample[f"episode_{name}"] = sample.pop(old)
            samples.append(sample)
        return concat_padded_tensors(samples)


class PacmanWorkflow(RolloutWorkflow):
    def __init__(
        self,
        *,
        gconfig: GenerationHyperparameters,
        tokenizer: str,
        options: dict[str, Any],
    ):
        self.gconfig, self.options = gconfig, options
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(tokenizer)
        if self.processor is None:
            raise ValueError("Pacman requires a multimodal processor")

    async def arun_episode(
        self, engine: Any, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        # The release uses synchronous subprocess/environment and artifact I/O.
        # Construct and run it off the inference event loop, with per-episode state.
        episode = await asyncio.to_thread(
            _Episode, self, engine, asyncio.get_running_loop()
        )
        try:
            return await asyncio.to_thread(episode.run_in_thread, data)
        except asyncio.CancelledError:
            episode.cancel()
            raise
