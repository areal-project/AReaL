# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations  # noqa

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from openai.types.chat import ChatCompletion
from openai.types.responses.response import Response
from openai.types.responses.response_input_param import ResponseInputParam

from areal.api import ModelResponse
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


def _align_mm_token_type_ids(
    prompt_mm_token_type_ids: list[int],
    resp: ModelResponse,
    seq_len: int,
) -> list[int]:
    """Extend prompt-scoped ``mm_token_type_ids`` over the full sequence.

    Output tokens are never multimodal, so they are zero-filled. The prompt
    portion is expected to line up with ``resp.input_len``; a mismatch means the
    prompt was built without the processor, so the values are padded/truncated
    and a warning is emitted rather than producing a shape error downstream.
    """
    mm = list(prompt_mm_token_type_ids)
    if len(mm) != resp.input_len:
        logger.warning(
            f"mm_token_type_ids length ({len(mm)}) does not match the prompt "
            f"length ({resp.input_len}). Padding/truncating to match; vision "
            "position ids may be wrong for this sample."
        )
        mm = (mm + [0] * resp.input_len)[: resp.input_len]
    return mm + [0] * (seq_len - resp.input_len)


@dataclass
class InteractionWithTokenLogpReward:
    """Internal structure to store completions/responses with their rewards."""

    # Common
    model_response: ModelResponse | None = None
    reward: float | None = None
    parent: InteractionWithTokenLogpReward | None = None
    chat_template_type: str = "hf"
    trajectory_metadata: dict[str, str] = field(default_factory=dict)
    _cache: dict[str, torch.Tensor] | None = None

    # Vision payload for VLM training. ``mm_token_type_ids`` covers the prompt
    # only (it is extended with zeros over the output at tensor-dict time);
    # ``multi_modal_input`` holds a single dict for the whole sequence, matching
    # the convention of the non-agent vision workflows.
    mm_token_type_ids: list[int] | None = None
    multi_modal_input: list[dict[str, Any]] | None = None

    # Inference-transport state, live-cache only. The prompt this turn sent to
    # vLLM, with one unexpanded placeholder per media item (identical to the
    # expanded prompt when the turn carried no media). A concat child needs its
    # parent's copy to build its own collapsed prompt, and cannot recover it
    # from the expanded one without assuming placeholders form dense contiguous
    # runs. Deliberately excluded from to_tensor_dict() and from proxy
    # serialization: training only ever sees the expanded prompt, and
    # deserialized interactions are training inputs, never future parents.
    #
    # Must be a snapshot taken before generation. ModelRequest.extend_prompt()
    # appends generated tokens in place after every response, so sharing the
    # list object with the request would grow this prompt and make the next
    # turn splice the parent's output twice.
    collapsed_input_ids: list[int] | None = None

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
        if self.chat_template_type == "concat" and self.parent is not None:
            parent_res = self.parent.to_tensor_dict()
            parent_logprobs = parent_res["logprobs"].squeeze(0).tolist()
            parent_loss_mask = parent_res["loss_mask"].squeeze(0).tolist()
            parent_versions = parent_res["versions"].squeeze(0).tolist()
            parent_len = len(parent_logprobs)
            assert parent_len == len(parent_loss_mask) == len(parent_versions)
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
        else:
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions
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
        if self.multi_modal_input is not None:
            result["multi_modal_input"] = self.multi_modal_input
        if self.mm_token_type_ids is not None:
            result["mm_token_type_ids"] = torch.tensor(
                _align_mm_token_type_ids(self.mm_token_type_ids, resp, len(seq)),
                dtype=torch.long,
            ).unsqueeze(0)
        self._cache = result
        return result


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


def interactions_to_trajectory(
    interactions: dict[str, InteractionWithTokenLogpReward],
) -> dict:
    """Convert proxy interactions to the trajectory format used by trainers."""
    if all(interaction.has_tensor_data for interaction in interactions.values()):
        from areal.utils.data import concat_padded_tensors

        trajectory = concat_padded_tensors(
            [interaction.to_tensor_dict() for interaction in interactions.values()]
        )
    else:
        trajectory = concat_string_interactions(interactions)

    if interactions:
        interaction_values = list(interactions.values())
        common_keys = set(interaction_values[0].trajectory_metadata)
        for interaction in interaction_values[1:]:
            common_keys.intersection_update(interaction.trajectory_metadata)
        for key in common_keys:
            value = interaction_values[0].trajectory_metadata[key]
            if all(
                interaction.trajectory_metadata[key] == value
                for interaction in interaction_values[1:]
            ):
                trajectory.setdefault(key, value)
    return trajectory
