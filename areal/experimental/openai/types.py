# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations  # noqa

from dataclasses import dataclass, field
from enum import Enum

import torch
from openai.types.chat import ChatCompletion
from openai.types.responses.response import Response
from openai.types.responses.response_input_param import ResponseInputParam

from areal.api import ModelResponse
from areal.engine.r3.preprocess import preprocess_routed_experts_batch
from areal.utils import logging

logger = logging.getLogger("TokenLogpReward")


class ApiType(str, Enum):
    """API type for interaction."""

    COMPLETION = "completion"
    RESPONSE = "response"
    NONE = "none"


class InputName(str, Enum):
    """Input name used for logging."""

    MESSAGES = "messages"
    INPUT_DATA = "input_data"
    NONE = "none"


@dataclass
class InteractionWithTokenLogpReward:
    """Internal structure to store completions/responses with their rewards."""

    # Common
    model_response: ModelResponse | None = None
    reward: float | None = None
    parent: InteractionWithTokenLogpReward | None = None
    chat_template_type: str = "hf"
    r3_num_moe_layers: int | None = None
    r3_topk: int | None = None
    _cache: dict[str, torch.Tensor] | None = None

    # Fields used for parent-child relationship resolving
    messages: list[dict] = field(default_factory=list)
    output_message_list: list[dict] | None = None

    # Completion fields (optional for response)
    completion: ChatCompletion | None = None

    # Response fields (optional for completion)
    response: Response | None = None
    input_data: str | ResponseInputParam = field(default_factory=lambda: "")

    # Interaction ID cache (used for deserialization)
    _interaction_id: str | None = None

    @property
    def has_tensor_data(self) -> bool:
        return self.model_response is not None or self._cache is not None

    @property
    def is_completion(self) -> bool:
        return self.completion is not None

    @property
    def is_response(self) -> bool:
        return self.response is not None

    @property
    def api_type(self) -> ApiType:
        """API type (completion/response)."""
        if self.is_completion:
            return ApiType.COMPLETION
        elif self.is_response:
            return ApiType.RESPONSE
        else:
            return ApiType.NONE

    @property
    def input_name_for_logging(self) -> InputName:
        """Input name used for logging."""
        if self.is_completion:
            return InputName.MESSAGES
        elif self.is_response:
            return InputName.INPUT_DATA
        else:
            return InputName.NONE

    @property
    def current_data(self) -> list[dict] | str | ResponseInputParam | None:
        if self.is_completion:
            return self.messages
        elif self.is_response:
            return self.input_data
        else:
            return None

    @property
    def parent_data(self) -> list[dict] | str | ResponseInputParam | None:
        if self.parent is None:
            return None
        return self.parent.current_data

    @property
    def interaction_id(self) -> str | None:
        if self.is_completion:
            return self.completion.id
        elif self.is_response:
            return self.response.id
        elif self._interaction_id is not None:
            return self._interaction_id
        else:
            return None

    @interaction_id.setter
    def interaction_id(self, value):
        if self.is_completion or self.is_response:
            raise ValueError("Cannot set ID for completion or responses")
        self._interaction_id = value

    @property
    def created_at(self) -> float | None:
        if self.is_completion:
            return float(self.completion.created)
        elif self.is_response:
            return float(self.response.created_at)
        else:
            return None

    @property
    def remaining_messages(self) -> list[dict]:
        if self.parent is None:
            return self.messages
        assert self.parent.output_message_list is not None, (
            "Parent output message is not set."
        )
        parent_len = len(self.parent.messages + self.parent.output_message_list)
        return self.messages[parent_len:]

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        if self._cache is not None:
            return self._cache
        resp = self.model_response
        assert resp is not None, "Model response is not set."
        self.seq_tokens = seq = resp.input_tokens + resp.output_tokens
        if self.parent is not None and self.r3_num_moe_layers is not None:
            if self.parent.r3_num_moe_layers is None:
                self.parent.r3_num_moe_layers = self.r3_num_moe_layers
            if self.parent.r3_topk is None:
                self.parent.r3_topk = self.r3_topk
        routed_experts = None
        r3_routing_valid = None
        if self.chat_template_type == "concat" and self.parent is not None:
            parent_res = self.parent.to_tensor_dict()
            parent_logprobs = parent_res["logprobs"].squeeze(0).tolist()
            parent_loss_mask = parent_res["loss_mask"].squeeze(0).tolist()
            parent_versions = parent_res["versions"].squeeze(0).tolist()
            parent_len = len(parent_logprobs)
            assert parent_len == len(parent_loss_mask) == len(parent_versions)
            current_r3 = self._r3_tensor_dict(seq_len=len(seq))
            if resp.input_len > parent_len:
                logprobs = (
                    parent_logprobs
                    + [0.0] * (resp.input_len - parent_len)
                    + resp.output_logprobs
                )
                loss_mask = (
                    parent_loss_mask
                    + [0] * (resp.input_len - parent_len)
                    + [1] * resp.output_len
                )
                versions = (
                    parent_versions
                    + [-1] * (resp.input_len - parent_len)
                    + resp.output_versions
                )
                if current_r3 is not None:
                    parent_routed = parent_res.get("routed_experts")
                    parent_valid = parent_res.get("r3_routing_valid")
                    if parent_routed is not None and parent_valid is not None:
                        current_routed = current_r3["routed_experts"].squeeze(0)
                        routed_experts = torch.cat(
                            [
                                parent_routed.squeeze(0),
                                current_routed[parent_len:],
                            ],
                            dim=0,
                        ).unsqueeze(0)
                        r3_routing_valid = (
                            parent_valid.bool() & current_r3["r3_routing_valid"].bool()
                        )
                    else:
                        routed_experts = current_r3["routed_experts"]
                        r3_routing_valid = torch.zeros(1, dtype=torch.bool)
            else:
                # FIXME: Find out why this happens occasionally
                api_type = self.api_type
                input_name = self.input_name_for_logging
                logger.warning(
                    f"The input length of the child {api_type} ({resp.input_len}) is less than or "
                    f"equal to the length of the parent {api_type} {parent_len}. "
                    f"This should not happen if the {input_name}s are constructed properly. "
                    f"Ignoring the parent {api_type} by masking them out. \n"
                    f"Parent input token ids: {self.parent.model_response.input_tokens}\n"
                    f"Parent output token ids: {self.parent.model_response.output_tokens}\n"
                    f"Child input token ids: {resp.input_tokens}\n"
                    f"Parent input {input_name}: {self.parent_data}\n"
                    f"Child input {input_name}: {self.current_data}",
                )
                logprobs = [0.0] * resp.input_len + resp.output_logprobs
                loss_mask = [0] * resp.input_len + [1] * resp.output_len
                versions = [-1] * resp.input_len + resp.output_versions
                if current_r3 is not None:
                    routed_experts = current_r3["routed_experts"]
                    r3_routing_valid = torch.zeros(1, dtype=torch.bool)
        else:
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions
            current_r3 = self._r3_tensor_dict(seq_len=len(seq))
            if current_r3 is not None:
                routed_experts = current_r3["routed_experts"]
                r3_routing_valid = current_r3["r3_routing_valid"]
        reward = self.reward if self.reward is not None else 0.0
        result = dict(
            # unsqueeze to add an additional batch dimension
            input_ids=torch.tensor(seq).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            # reward
            rewards=torch.tensor([float(reward)]),
        )
        if routed_experts is not None and r3_routing_valid is not None:
            result["routed_experts"] = routed_experts
            result["r3_routing_valid"] = r3_routing_valid
        self._cache = result
        return result

    def _r3_tensor_dict(self, *, seq_len: int) -> dict[str, torch.Tensor] | None:
        resp = self.model_response
        if resp is None or resp.routed_experts is None:
            return None
        if self.r3_num_moe_layers is None or self.r3_topk is None:
            raise ValueError(
                "Interaction received routed_experts but R3 MoE shape is not "
                "configured. Set r3_num_moe_layers and r3_topk before tensor export."
            )
        return preprocess_routed_experts_batch(
            [resp.routed_experts],
            seq_lens=[seq_len],
            num_moe_layers=self.r3_num_moe_layers,
            topk=self.r3_topk,
        )


def configure_r3_interactions(
    interactions: dict[str, InteractionWithTokenLogpReward],
    *,
    num_moe_layers: int | None,
    topk: int | None,
) -> None:
    if num_moe_layers is None or topk is None:
        return

    seen: set[int] = set()

    def _configure(interaction: InteractionWithTokenLogpReward) -> None:
        ident = id(interaction)
        if ident in seen:
            return
        seen.add(ident)
        interaction.r3_num_moe_layers = num_moe_layers
        interaction.r3_topk = topk
        if interaction.parent is not None:
            _configure(interaction.parent)

    for interaction in interactions.values():
        _configure(interaction)


def concat_string_interactions(
    interactions: dict[str, InteractionWithTokenLogpReward],
) -> dict[str, list[dict]]:
    """Concat interactions that lack tensor data (e.g. external API mode).

    Returns a dict with an ``"interactions"`` key containing a list of
    ``{"request": ..., "response": ..., "reward": ...}`` dicts, one per
    interaction.  This is the counterpart of
    :func:`~areal.utils.data.concat_padded_tensors` for string-only
    trajectories.
    """
    return {
        "interactions": [
            {
                "request": v.messages,
                "response": (
                    v.output_message_list[0]["content"] if v.output_message_list else ""
                ),
                "reward": v.reward,
            }
            for v in interactions.values()
        ]
    }
