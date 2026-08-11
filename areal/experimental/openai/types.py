# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations  # noqa

from dataclasses import dataclass, field
from enum import Enum

import contextvars
import hashlib
import re
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


# RAO：请求体里原始的 `model` 字段（含 `__dN` 深度后缀）。
# proxy 在丢弃 model 之前 set，interaction 落库时读。见下方 `rao_depth` 的说明。
RAO_REQ_MODEL: contextvars.ContextVar = contextvars.ContextVar(
    "rao_req_model", default=None
)


def rao_iid_hash(interaction_id: str | None) -> int:
    """`interaction_id` → 48 位整数，用于把训练序列**精确**认回它的 rollout 行。

    rollout dump（文本，带 session/interaction id）和 train_batch dump（token，
    带 loss_mask）之间原本**没有共享 key** —— 训练侧只有 token id，谁生成的、
    在树的哪个节点上，全丢了。以前只能靠 `seqlen == n_tokens` 反推，实测
    force50 的 5654 条训练序列里有 602 条歧义（多条 rollout 撞同一个长度）。

    做法沿用 `rao_depth` 趟通的那条路：per-token 常量张量搭训练数据的顺风车。

    ⚠️ **48 位不是随手取的**：
    - 张量装不下字符串，只能塞整数；
    - 用 `dtype=torch.long`（和 `input_ids` 同款）而不是 `rao_depth` 那样的
      float32 —— float32 只有 24 位尾数，48 位的哈希塞进去会被截得面目全非；
    - 48 < 53，所以中途任何一次 `float()` 往返（tap 里就有）都还是精确的。

    ~3 万条序列下碰撞期望约 0.05 条，可以忽略。
    """
    if not interaction_id:
        return 0
    return int.from_bytes(
        hashlib.blake2b(interaction_id.encode(), digest_size=6).digest(), "big"
    )


@dataclass
class InteractionWithTokenLogpReward:
    """Internal structure to store completions/responses with their rewards."""

    # Common
    model_response: ModelResponse | None = None
    reward: float | None = None
    parent: InteractionWithTokenLogpReward | None = None
    chat_template_type: str = "hf"
    _cache: dict[str, torch.Tensor] | None = None

    # Fields used for parent-child relationship resolving
    messages: list[dict] = field(default_factory=list)
    output_message_list: list[dict] | None = None

    # Completion fields (optional for response)
    completion: ChatCompletion | None = None

    # RAO：递归深度。harness 把它编码在 model id 尾部（`<id>__d<N>`）——
    # 因为 sub-agent 的 system prompt 与 root 不同、prefix_matcher 匹配不上、
    # parent 恒为 None，树结构在 AReaL 侧根本推不出来，只能由 harness 显式传。
    # 走 model id 是因为 pi 既没有 --base-url 也不给自定义 header 的口子，
    # 而 model 必然出现在请求体里。换 API key 不行 —— 会把每层拆成不同 session。
    #
    # ⚠️ **不能只读 `completion.model`** —— proxy 在转发前把 `model` 字段整个丢掉了
    #    （`proxy_rollout_server.py`：`areal_client_ignored_args = ["model"]`，
    #    因为 AReaL 的 client 只服务单一模型、不接受 model 参数）。
    #    响应里的 model 名是服务端改写过的，`__dN` 早没了。
    #    实测：真实跑里**每条序列的 rao_depth 都是 0，连 sub-agent 也是**。
    #    （mock 测试没抓到 —— 它只数派生次数，从没验证过深度解析结果。）
    #
    #    修法：proxy 在丢掉 model **之前**写进 `RAO_REQ_MODEL` 这个 ContextVar，
    #    interaction 落库时抄到 `rao_model`。
    #    ⚠️ 必须是 ContextVar 不能是实例属性 —— 同一 session 里多个 sub-agent 是
    #    **并行**请求的，实例属性会串味；ContextVar 在 asyncio 任务间天然隔离。
    rao_model: str | None = None

    @property
    def rao_depth(self) -> int:
        model = (
            self.rao_model
            or getattr(self.completion, "model", None)
            or getattr(self.response, "model", None)
            or ""
        )
        m = re.search(r"__d(\d+)$", str(model))
        return int(m.group(1)) if m else 0

    # Response fields (optional for completion)
    response: Response | None = None
    input_data: str | ResponseInputParam = field(default_factory=lambda: "")

    # Interaction ID cache (used for deserialization)
    _interaction_id: str | None = None
    # 该 interaction 属于哪一次 agent 运行（= 一条 rollout session）。
    # 由 InteractionCache.export_interactions 盖章，供 dump 落盘还原会话归属。
    session_id: str | None = None
    # 父节点的 id。subproc/online 模式下 interaction 要跨进程传回，`parent`
    # 这个对象引用没法序列化，只能带 id 过来。
    parent_interaction_id: str | None = None

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
            # RAO：递归深度。
            #
            # ⚠️ 这里原本是 per-seq 的 `[1]` 形张量（像 `rewards` 一样）。**那样是错的**：
            # `split_padded_tensor_dict_into_mb_list` 只切分**带序列维**的张量，
            # `[B]` 形的会被**整份复制**给每个微批 —— 于是到 loss 现场
            # `rao_depth.numel()` 是整个 batch 的条数，而 `cu_seqlens` 只描述本微批，
            # 两边对不上，`rao_depth_weighting` 的守卫判否后静默跳过，
            # train_batch dump 里也就一直看不到 `rao_depth`。
            # （`rewards` 没暴露这个问题，是因为它在 actor.py:343 算完 advantage 就被 pop 掉了。）
            #
            # 改成 per-token 的 `[1, L]`：跟着 `loss_mask` 一起切分、一起 pack，
            # loss 现场取每条序列的首 token（`dep[cu_seqlens[i]]`）就是该条的深度。
            rao_depth=torch.full((1, len(seq)), float(self.rao_depth)),
            # RAO：这条序列出自哪个 interaction（见 `rao_iid_hash`）。
            # 和 rollout dump 的 `interaction_id` 一 join 就能把"模型说了什么"
            # 和"这段话进没进梯度"接起来。同样是 per-token，理由同上。
            rao_iid=torch.full(
                (1, len(seq)), rao_iid_hash(self.interaction_id), dtype=torch.long
            ),
        )
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
