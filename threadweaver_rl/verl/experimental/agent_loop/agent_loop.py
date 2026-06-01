# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import heapq
import logging
import os
import queue
import random
import threading
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import Future
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _compute_groupwise_length_penalties(
    base_group_ids: list[str],
    base_lengths: list[int],
    base_corrects: list[bool],
    tau: float,
    beta: float,
    eps: float,
    window: float,
    *,
    verbose: bool = False,
    metric_name: str = "response_token_count",
) -> list[float]:
    """
    Compute per-item length penalties within groups, based on dynamic alpha and overlong ratio.

    - Groups are defined by base_group_ids (same length as base_lengths/base_corrects).
    - AccG = mean(correct) per group; alpha = 0 if AccG < tau, else beta*(AccG-tau+eps)/(1-tau+eps).
    - O_i = clip((L_i - L_shortest_correct)/window, 0, 1); if no correct in group, penalties are 0.
    - Returns a list of penalties aligned with inputs.

    Debug prints are enabled when verbose is True.
    """
    penalties = [0.0] * len(base_group_ids)

    # Build groups
    idxs_by_gid: dict[str, list[int]] = defaultdict(list)
    for idx, gid in enumerate(base_group_ids):
        idxs_by_gid[str(gid)].append(idx)

    for gid, idxs in idxs_by_gid.items():
        n = len(idxs)
        if n == 0:
            continue
        acc_g = sum(1 for j in idxs if base_corrects[j]) / n
        if acc_g < tau:
            alpha = 0.0
        else:
            denom = max(1e-12, (1.0 - tau + eps))
            alpha = beta * (acc_g - tau + eps) / denom

        # If no correct items in group, do not penalize
        corr_lens = [base_lengths[j] for j in idxs if base_corrects[j]]
        if not corr_lens:
            if verbose:
                print(
                    f"[LP] gid={gid} metric={metric_name} n={n} correct=0 AccG={acc_g:.3f} alpha={alpha:.4f} (no correct in group, skip penalty)"
                )
            continue
        l_short = min(corr_lens)

        if window <= 0:
            if verbose:
                print(
                    f"[LP] gid={gid} metric={metric_name} n={n} AccG={acc_g:.3f} alpha={alpha:.4f} L_shortest={l_short} window={window} (disabled)"
                )
            continue

        if verbose:
            n_correct = sum(1 for j in idxs if base_corrects[j])
            print(
                f"[LP] gid={gid} metric={metric_name} n={n} correct={n_correct} AccG={acc_g:.3f} "
                f"alpha={alpha:.4f} L_shortest={l_short} window={window}"
            )
        for j in idxs:
            oi = max(0.0, min(1.0, (float(base_lengths[j]) - float(l_short)) / window))
            penalties[j] = alpha * oi
            if verbose:
                print(
                    f"[LP]   idx={j} L_i={base_lengths[j]} O_i={oi:.4f} penalty={penalties[j]:.4f} correct={base_corrects[j]}"
                )

    return penalties


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput: token output
        """
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
        )
        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    generated_blocks: Optional[list[dict[str, Any]]] = None
    """Structured generation blocks (if provided by the backend)."""
    expanded_prompt_ids: Optional[list[list[int]]] = None
    """Prompts used for each low-level engine call (one list per call)."""
    expanded_response_ids: Optional[list[list[int]]] = None
    """Raw completions from each low-level engine call (aligned with expanded_prompt_ids)."""
    expanded_response_mask: Optional[list[list[int]]] = None
    """Binary masks for expanded_response_ids (1 denotes generated token)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


# make hydra.utils.instantiate happy
class _DummyConfig:
    def __init__(self, config: DictConfig) -> None:
        self.config = config


class AgentLoopBase(ABC):
    """An agent loop takes a input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    _class_initialized = False

    def __init__(
        self,
        trainer_config: _DummyConfig,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (_DummyConfig): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process messages.
        """
        self.init_class(config=trainer_config.config, tokenizer=tokenizer, processor=processor, **kwargs)
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.loop = asyncio.get_running_loop()

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, processor: AutoProcessor, **kwargs):
        """This is used to do heavy initialization work that should shared across all instances. It's only called once.

        Args:
            config (DictConfig): trainer config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process multi_modal data.
            **kwargs: extra kwargs from config file passed in by `hydra.utils.instantiate`.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


@ray.remote(num_cpus=1)
class BatchExecutor:
    """Batch executor is used to collect requests into a batch execution"""

    def __init__(self, batch_func, micro_batch_size=1, max_batch_size=None):
        """

        Args:
            batch_func: batch processing function.
            micro_batch_size (int, optional): micro batch size. Defaults to 1.
            max_batch_size: batch size for batching.
        """
        self._q = queue.Queue()
        self._batch_func = batch_func
        self._max_batch = max_batch_size
        self._micro_batch_size = micro_batch_size

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    async def submit_task(self, item):
        """
        Blocking submission, returning Future
        Args:
            item: function input

        Returns:
            fut: function output
        """
        fut = Future()
        self._q.put((item, fut))
        async_fut = asyncio.wrap_future(fut)
        res = await async_fut
        return res

    def _worker_loop(self):
        while True:
            # 1. Fetch a full batch (block until at least one)
            first, first_fut = self._q.get()
            items = [first]
            futs = [first_fut]

            # Take the remaining tasks at once
            while True:
                try:
                    next_item, next_fut = self._q.get_nowait()
                    items.append(next_item)
                    futs.append(next_fut)
                    if self._max_batch and len(items) >= self._max_batch:
                        break
                except queue.Empty:
                    while len(items) % self._micro_batch_size != 0:
                        next_item, next_fut = self._q.get()
                        items.append(next_item)
                        futs.append(next_fut)
                        if self._max_batch and len(items) >= self._max_batch:
                            break
                    break

            try:
                results = self._batch_func(items)
            except Exception as e:
                for f in futs:
                    f.set_exception(e)
            else:
                for f, r in zip(futs, results, strict=False):
                    f.set_result(r)


@ray.remote(num_cpus=1)
class RewardManagerWorker:
    """Reward manager worker to compute reward score asynchronously to overlap with agent loop."""

    def __init__(self, config: DictConfig, local_path: str, rm_executor: BatchExecutor = None) -> None:
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        reward_manager_type = config.reward_model.get("reward_manager_type", None)
        reward_manager_config = config.reward_model.get("config", {})
        reward_manager_server_workers = config.reward_model.get("reward_manager_server_workers")

        server_kwargs: dict[str, Any] = {}
        if reward_manager_server_workers is not None:
            server_kwargs["workers"] = reward_manager_server_workers

        if reward_manager_type == "reward_manager":
            from verl.trainer.reward_manager import RewardManager

            self.reward_manager = RewardManager(tokenizer=tokenizer, num_examine=0, config=reward_manager_config)
        elif reward_manager_type == "reward_manager_with_server":
            try:
                from verl.trainer.reward_manager_with_server import RewardManagerWithServer

                self.reward_manager = RewardManagerWithServer(
                    tokenizer=tokenizer,
                    num_examine=0,
                    config=reward_manager_config,
                    **server_kwargs,
                )
            except RuntimeError as e:
                print("Error in initializing RewardManagerWithServer:", e, " using RewardManager instead.")
                from verl.trainer.reward_manager import RewardManager
                self.reward_manager = RewardManager(tokenizer=tokenizer, num_examine=0, config=reward_manager_config)
        else:
            self.reward_manager = load_reward_manager(
                config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
            )
        self.rm_executor = rm_executor

    def compute_score(
        self,
        data: DataProto,
        **kwargs,
    ) -> dict:
        """Compute reward score for agent loop output.

        Args:
            data: reward function input

        Returns:
            dict: Reward score and reward extra info.
        """
        if self.rm_executor is not None:
            res = ray.get(self.rm_executor.submit_task.remote(data))
            data = data.union(res)

        rm_kwargs = {}
        if data.meta_info:
            rm_kwargs.update(data.meta_info.get("reward_manager_kwargs", {}))
        if kwargs:
            rm_kwargs.update(kwargs)

        result = self.reward_manager(data, return_dict=True, **rm_kwargs)
        reward_tensor = result["reward_tensor"]

        # Recent reward managers may return multiple tensors (e.g. main/secondary).
        if isinstance(reward_tensor, dict):
            if "main_reward_tensor" in reward_tensor:
                reward_tensor = reward_tensor["main_reward_tensor"]
            elif len(reward_tensor) == 1:
                reward_tensor = next(iter(reward_tensor.values()))
            else:
                tensors = [
                    value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                    for value in reward_tensor.values()
                ]
                base = tensors[0]
                tensors = [tensor.to(base.device) for tensor in tensors]
                reward_tensor = torch.stack(tensors, dim=0).sum(dim=0)

        if not isinstance(reward_tensor, torch.Tensor):
            reward_tensor = torch.as_tensor(reward_tensor)

        reward_score = reward_tensor.sum(dim=-1).item()
        reward_extra_info = {k: v[0] for k, v in result.get("reward_extra_info", {}).items()}
        return {"reward_score": reward_score, "reward_extra_info": reward_extra_info}


@ray.remote
class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], rm_executor: BatchExecutor = None
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        self.config = config
        self.server_manager = AsyncLLMServerManager(config, server_handles)
        self.rm_executor = rm_executor

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        self.reward_manager_worker = RewardManagerWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote(self.config, local_path, self.rm_executor)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        base_meta_info = dict(batch.meta_info) if batch.meta_info is not None else {}
        reward_manager_kwargs = dict(base_meta_info.get("reward_manager_kwargs", {}))
        base_meta_info["reward_manager_kwargs"] = reward_manager_kwargs
        
        additional_kwargs = {}
        if "return_generated_blocks" in base_meta_info:
            additional_kwargs.update(return_generated_blocks=base_meta_info["return_generated_blocks"])
        
        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            # Forward meta-level flags (e.g., return_generated_blocks) into per-sample kwargs
            # so downstream AgentLoop implementations can include structured outputs.
            if additional_kwargs:
                kwargs.update(additional_kwargs)
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(
                        sampling_params,
                        trajectory_info[i],
                        meta_info=base_meta_info,
                        reward_manager_kwargs=reward_manager_kwargs,
                        **kwargs,
                    )
                )
            )
        outputs = await asyncio.gather(*tasks)

        return_expanded_sequences = bool(base_meta_info.get("return_expanded_sequences", False))
        return_base_tensors_with_expanded = bool(
            base_meta_info.get("return_base_tensors_with_expanded", False)
        )

        output = self._postprocess(
            outputs,
            return_expanded_sequences=return_expanded_sequences,
            return_base_tensors_with_expanded=return_base_tensors_with_expanded,
        )
        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        meta_info: dict[str, Any] | None = None,
        reward_manager_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)

            meta_info = meta_info or {}
            return_expanded_sequences = bool(meta_info.get("return_expanded_sequences", False))

            # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

            # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
            # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
            # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
            # input_ids: concatenation of prompt + response
            # Mask:
            # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
            # - prompt_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [0,0,0,0,1,1,1,1]
            # - response_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
            # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
            #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
            # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
            #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
            # - position_ids: sequential positions for tokens, starting at 0
            #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

            self.tokenizer.padding_side = "left"
            prompt_output = self.tokenizer.pad(
                {"input_ids": output.prompt_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if prompt_output["input_ids"].dim() == 1:
                prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

            self.tokenizer.padding_side = "right"
            response_output = self.tokenizer.pad(
                {"input_ids": output.response_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if response_output["input_ids"].dim() == 1:
                response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

            response_mask_output = self.tokenizer.pad(
                {"input_ids": output.response_mask},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            if response_mask_output["input_ids"].dim() == 1:
                response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

            assert len(output.response_ids) == len(output.response_mask), (
                f"{agent_name} returned {len(output.response_ids)} response tokens "
                    f"but only {len(output.response_mask)} mask entries; "
                    f"output.response_ids: {output.response_ids}, "
                    f"output.response_mask: {output.response_mask}"
            )

            response_logprobs = None
            if output.response_logprobs is not None:
                pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
                response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

            response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]

            if not torch.all((response_mask == 0) | (response_mask == 1)):
                bad_values = torch.unique(response_mask[(response_mask != 0) & (response_mask != 1)])
                raise ValueError(f"Non-binary response_mask values detected: {bad_values.tolist()} in {response_mask}, with {output.response_mask=}, {response_mask_output['input_ids']=}, and {response_output['attention_mask']=}")

            attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
            input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

            if return_expanded_sequences and "expanded_prompts_tensor" not in output.extra_fields:
                prompt_target = self.config.actor_rollout_ref.rollout.prompt_length
                response_target = self.config.actor_rollout_ref.rollout.response_length
                input_target = prompt_target + response_target
                pad_token_id = self.tokenizer.pad_token_id or 0

                expanded_prompts = output.expanded_prompt_ids or []
                expanded_responses = output.expanded_response_ids or []
                expanded_masks = output.expanded_response_mask or []

                if not expanded_prompts:
                    expanded_prompts = [list(output.prompt_ids)]
                if not expanded_responses:
                    expanded_responses = [list(output.response_ids)]
                if not expanded_masks:
                    expanded_masks = [list(output.response_mask)]

                (
                    padded_prompts,
                    padded_responses,
                    padded_masks,
                    padded_inputs,
                    padded_attentions,
                ) = self._pad_expanded_lists(
                    expanded_prompts,
                    expanded_responses,
                    expanded_masks,
                    prompt_target,
                    response_target,
                    input_target,
                    pad_token_id,
                )

                if padded_prompts:
                    expanded_prompt_tensor = torch.tensor(padded_prompts, dtype=torch.long)
                    expanded_response_tensor = torch.tensor(padded_responses, dtype=torch.long)
                    expanded_mask_tensor = torch.tensor(padded_masks, dtype=torch.long)
                    expanded_input_tensor = torch.tensor(padded_inputs, dtype=torch.long)
                    expanded_attention_tensor = torch.tensor(padded_attentions, dtype=torch.long)

                    output.extra_fields["expanded_prompts_tensor"] = expanded_prompt_tensor
                    output.extra_fields["expanded_responses_tensor"] = expanded_response_tensor
                    output.extra_fields["expanded_masks_tensor"] = expanded_mask_tensor
                    output.extra_fields["expanded_sequence_count"] = expanded_prompt_tensor.shape[0]
                    output.extra_fields["expanded_inputs_tensor"] = expanded_input_tensor
                    output.extra_fields["expanded_attentions_tensor"] = expanded_attention_tensor

                    output.expanded_prompt_ids = padded_prompts
                    output.expanded_response_ids = padded_responses
                    output.expanded_response_mask = padded_masks

            # Handle multi-modal inputs and position_ids calculation
            # Only support Qwen2VLImageProcessor for multi-modal processing currently
            # TODO: support other multi-modal inputs
            multi_modal_inputs = None
            if (
                self.processor is not None
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
            ):
                from verl.models.transformers.qwen2_vl import get_rope_index

                images = output.multi_modal_data.get("image", None)
                current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
                multi_modal_inputs.pop("input_ids", None)
                multi_modal_inputs.pop("attention_mask", None)

                # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
                # because np.array() only keeps the keys for BatchFeature.
                multi_modal_inputs = dict(multi_modal_inputs)

                image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

                position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids.squeeze(0),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask.squeeze(0),
                ).unsqueeze(0)  # (1, 3, seq_len)
            else:
                position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)
            enable_async_reward = (
                self.rm_executor is not None and self.config.reward_model.enable_resource_pool
            ) or not self.config.reward_model.enable
            if (
                self.reward_manager_worker is not None
                and output.reward_score is None
                and enable_async_reward
            ):
                batch = TensorDict(
                    {
                        "prompts": prompt_output["input_ids"],  # [1, prompt_length]
                        "responses": response_output["input_ids"],  # [1, response_length]
                        "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                        "input_ids": input_ids,  # [1, prompt_length + response_length]
                        "position_ids": position_ids,
                    },
                    batch_size=1,
                )
                non_tensor_batch = {
                    **{k: np.array([v]) for k, v in kwargs.items()},
                    "__num_turns__": np.array([output.num_turns]),
                }
                rm_kwargs = dict(reward_manager_kwargs or {})
                data_meta_info = dict(meta_info or {})
                data_meta_info["reward_manager_kwargs"] = rm_kwargs
                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                    meta_info=data_meta_info,
                )
                result = await self.reward_manager_worker.compute_score.remote(data, **rm_kwargs)
                output.reward_score = result["reward_score"]
                output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

            output.extra_fields.setdefault("_base_non_tensor", kwargs)

            return _InternalAgentLoopOutput(
                prompt_ids=prompt_output["input_ids"],
                response_ids=response_output["input_ids"],
                input_ids=input_ids,
                position_ids=position_ids,
                response_mask=response_mask,
                attention_mask=attention_mask,
                response_logprobs=response_logprobs,
                multi_modal_inputs=multi_modal_inputs,
                multi_modal_data=output.multi_modal_data,
                reward_score=output.reward_score,
                num_turns=output.num_turns,
                metrics=output.metrics,
                extra_fields=output.extra_fields,
                expanded_prompt_ids=output.expanded_prompt_ids if return_expanded_sequences else None,
                expanded_response_ids=output.expanded_response_ids if return_expanded_sequences else None,
                expanded_response_mask=output.expanded_response_mask if return_expanded_sequences else None,
            )

    def _pad_expanded_lists(
        self,
        prompt_lists,
        response_lists,
        mask_lists,
        prompt_target,
        response_target,
        input_target,
        pad_token_id,
    ):
        prompt_lists = prompt_lists or []
        response_lists = response_lists or []
        mask_lists = mask_lists or []

        sequence_count = len(prompt_lists)
        if sequence_count == 0:
            return [], [], [], [], []

        aligned_responses = []
        aligned_masks = []
        for idx in range(sequence_count):
            aligned_responses.append(
                list((response_lists[idx] if idx < len(response_lists) else [])[:response_target])
            )
            aligned_masks.append(
                list((mask_lists[idx] if idx < len(mask_lists) else [])[:response_target])
            )

        truncated_prompts = [list(prompt[:prompt_target]) for prompt in prompt_lists]

        original_padding_side = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = "left"
            prompt_padded = self.tokenizer.pad(
                {"input_ids": truncated_prompts},
                padding="max_length",
                max_length=prompt_target,
                return_tensors="pt",
                return_attention_mask=True,
            )

            self.tokenizer.padding_side = "right"
            response_padded = self.tokenizer.pad(
                {"input_ids": aligned_responses},
                padding="max_length",
                max_length=response_target,
                return_tensors="pt",
                return_attention_mask=True,
            )

            mask_padded = self.tokenizer.pad(
                {"input_ids": aligned_masks},
                padding="max_length",
                max_length=response_target,
                return_tensors="pt",
                return_attention_mask=False,
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

        prompt_tensor = prompt_padded["input_ids"]
        prompt_attention = prompt_padded["attention_mask"]
        response_tensor = response_padded["input_ids"]
        response_attention = response_padded["attention_mask"]
        mask_tensor = mask_padded["input_ids"] * response_attention

        inputs_tensor = torch.cat([prompt_tensor, response_tensor], dim=1)
        attention_tensor = torch.cat([prompt_attention, response_attention], dim=1)

        if inputs_tensor.shape[1] > input_target:
            inputs_tensor = inputs_tensor[:, :input_target]
            attention_tensor = attention_tensor[:, :input_target]
        elif inputs_tensor.shape[1] < input_target:
            pad_width = input_target - inputs_tensor.shape[1]
            pad_shape = (inputs_tensor.shape[0], pad_width)
            inputs_tensor = torch.cat(
                [inputs_tensor, inputs_tensor.new_full(pad_shape, pad_token_id)], dim=1
            )
            attention_tensor = torch.cat(
                [attention_tensor, attention_tensor.new_zeros(pad_shape)], dim=1
            )

        prompts = prompt_tensor.tolist()
        responses = response_tensor.tolist()
        masks = mask_tensor.tolist()
        inputs = inputs_tensor.tolist()
        attentions = attention_tensor.tolist()

        return prompts, responses, masks, inputs, attentions

    def _postprocess(
        self,
        inputs: list[_InternalAgentLoopOutput],
        *,
        return_expanded_sequences: bool,
        return_base_tensors_with_expanded: bool,
    ) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""

        print(
            "[AgentLoop] postprocess | batch_size="
            f"{len(inputs)}, return_expanded_sequences={return_expanded_sequences}, "
            f"return_base_tensors_with_expanded={return_base_tensors_with_expanded}"
        )

        def _to_object_array(values: list[Any]) -> np.ndarray:
            """Pack python objects into a 1D numpy object array without inferring extra axes."""
            arr = np.empty(len(values), dtype=object)
            if values:
                arr[:] = values
            return arr

        expanded_cache: list[
            tuple[
                Optional[torch.Tensor],
                Optional[torch.Tensor],
                Optional[torch.Tensor],
                Optional[torch.Tensor],
                Optional[torch.Tensor],
                dict[str, Any],
            ]
        ] = []
        for input_item in inputs:
            exp_prompts_tensor = input_item.extra_fields.pop("expanded_prompts_tensor", None)
            exp_responses_tensor = input_item.extra_fields.pop("expanded_responses_tensor", None)
            exp_masks_tensor = input_item.extra_fields.pop("expanded_masks_tensor", None)
            input_item.extra_fields.pop("expanded_sequence_count", None)
            exp_inputs_tensor = input_item.extra_fields.pop("expanded_inputs_tensor", None)
            exp_attn_tensor = input_item.extra_fields.pop("expanded_attentions_tensor", None)
            base_non_tensor = input_item.extra_fields.pop("_base_non_tensor", {})
            expanded_cache.append(
                (
                    exp_prompts_tensor,
                    exp_responses_tensor,
                    exp_masks_tensor,
                    exp_inputs_tensor,
                    exp_attn_tensor,
                    base_non_tensor,
                )
            )

        # base_seq_uids is the uid for each base seq.
        # In contrast, `uid` is the uid for each prompt. All base seqs corresponding to the same prompt share the same uid.
        base_seq_uids = [str(uuid.uuid4()) for _ in inputs]

        if return_expanded_sequences:
            if os.getenv("VERL_AGENT_DEBUG", "0") == "1":
                print(
                    f"[AgentLoop] flattening expanded sequences | samples={len(inputs)}"
                )
            include_base_tensors = return_base_tensors_with_expanded

            prompt_chunks: list[torch.Tensor] = []
            response_chunks: list[torch.Tensor] = []
            response_mask_chunks: list[torch.Tensor] = []
            attention_chunks: list[torch.Tensor] = []
            input_chunks: list[torch.Tensor] = []
            position_chunks: list[torch.Tensor] = []
            base_prompts_tensor: Optional[torch.Tensor] = None
            base_responses_tensor: Optional[torch.Tensor] = None
            base_response_mask_tensor: Optional[torch.Tensor] = None
            base_attention_mask_tensor: Optional[torch.Tensor] = None
            base_input_ids_tensor: Optional[torch.Tensor] = None
            base_position_ids_tensor: Optional[torch.Tensor] = None

            logprob_chunks: list[torch.Tensor] = []

            expanded_prompt_chunks: list[torch.Tensor] = []
            expanded_response_chunks: list[torch.Tensor] = []
            expanded_mask_chunks: list[torch.Tensor] = []
            expanded_input_chunks: list[torch.Tensor] = []
            expanded_attention_chunks: list[torch.Tensor] = []
            expanded_position_chunks: list[torch.Tensor] = []
            base_field_values: dict[str, list[Any]] = defaultdict(list)

            metrics_list: list[dict[str, Any]] = []
            reward_scores: list[Optional[float]] = []
            num_turns_list: list[int] = []
            multi_modal_inputs_flat: list[Any] = []
            reward_extra_infos: list[dict[str, Any]] = []
            extra_field_values: dict[str, list[Any]] = defaultdict(list)
            total_sequences = 0
            is_last_in_expanded_flags: list[bool] = []

            has_logprobs = any(inp.response_logprobs is not None for inp in inputs)

            for idx, input_item in enumerate(inputs):
                if os.getenv("VERL_AGENT_DEBUG", "0") == "1":
                    cached = expanded_cache[idx][0]
                    count_dbg = None if cached is None else cached.shape[0]
                    print(
                        f"[AgentLoop] sample {idx}: expanded_count={count_dbg}"
                    )
                (
                    exp_prompt_tensor,
                    exp_response_tensor,
                    exp_mask_tensor,
                    exp_input_tensor,
                    exp_attn_tensor,
                    base_fields,
                ) = expanded_cache[idx]
                base_uid = base_seq_uids[idx]

                count = exp_prompt_tensor.shape[0]
                repeat_dims = (count,) + (1,) * (input_item.position_ids.dim() - 1)

                if include_base_tensors:
                    prompt_chunks.append(input_item.prompt_ids.repeat(count, 1))
                    response_chunks.append(input_item.response_ids.repeat(count, 1))
                    response_mask_chunks.append(input_item.response_mask.repeat(count, 1))
                    attention_chunks.append(input_item.attention_mask.repeat(count, 1))
                    input_chunks.append(input_item.input_ids.repeat(count, 1))
                    position_chunks.append(input_item.position_ids.repeat(repeat_dims))

                if has_logprobs:
                    log_probs = input_item.response_logprobs
                    if log_probs is None:
                        log_probs = torch.zeros_like(input_item.response_ids, dtype=torch.float32)
                    logprob_chunks.append(log_probs.repeat(count, 1))

                expanded_prompt_chunks.append(exp_prompt_tensor)
                expanded_response_chunks.append(exp_response_tensor)
                expanded_mask_chunks.append(exp_mask_tensor)
                expanded_input_chunks.append(exp_input_tensor)
                expanded_attention_chunks.append(exp_attn_tensor)

                if exp_input_tensor is not None and exp_attn_tensor is not None:
                    if input_item.position_ids.dim() == 3:
                        from verl.models.transformers.qwen2_vl import get_rope_index

                        multi_modal_inputs = input_item.multi_modal_inputs or {}
                        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                        video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                        second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

                        expanded_positions = []
                        for seq_idx in range(count):
                            position = get_rope_index(
                                self.processor,
                                input_ids=exp_input_tensor[seq_idx],
                                image_grid_thw=image_grid_thw,
                                video_grid_thw=video_grid_thw,
                                second_per_grid_ts=second_per_grid_ts,
                                attention_mask=exp_attn_tensor[seq_idx],
                            ).unsqueeze(0)
                            expanded_positions.append(position)

                        expanded_position_tensor = torch.cat(expanded_positions, dim=0)
                    else:
                        expanded_position_tensor = compute_position_id_with_mask(exp_attn_tensor)

                    expanded_position_chunks.append(expanded_position_tensor)
                else:
                    expanded_position_chunks.append(input_item.position_ids.repeat(repeat_dims))

                for key, value in base_fields.items():
                    base_field_values[key].extend([value] * count)
                base_field_values["base_seq_uid"].extend([base_uid] * count)

                metrics_list.extend([input_item.metrics.model_dump()] * count)
                reward_scores.extend([input_item.reward_score] * count)
                num_turns_list.extend([input_item.num_turns] * count)
                multi_modal_inputs_flat.extend([input_item.multi_modal_inputs] * count)
                total_sequences += count

                if count > 0:
                    is_last_in_expanded_flags.extend([False] * (count - 1) + [True])

                reward_info = input_item.extra_fields.get("reward_extra_info", {})
                reward_extra_infos.extend([reward_info] * count)

                for key, value in input_item.extra_fields.items():
                    if key in {
                        "reward_extra_info",
                        "expanded_prompts_tensor",
                        "expanded_responses_tensor",
                        "expanded_masks_tensor",
                        "expanded_sequence_count",
                    }:
                        continue
                    extra_field_values[key].extend([value] * count)

            if os.getenv("VERL_AGENT_DEBUG", "0") == "1":
                print(
                    f"[AgentLoop] expanded total_seqs={total_sequences} | base_seqs={len(inputs)}"
                )

            if include_base_tensors:
                base_prompts_tensor = torch.cat(prompt_chunks, dim=0)
                base_responses_tensor = torch.cat(response_chunks, dim=0)
                base_response_mask_tensor = torch.cat(response_mask_chunks, dim=0)
                base_attention_mask_tensor = torch.cat(attention_chunks, dim=0)
                base_input_ids_tensor = torch.cat(input_chunks, dim=0)
                base_position_ids_tensor = torch.cat(position_chunks, dim=0)

            prompts_tensor = torch.cat(expanded_prompt_chunks, dim=0)
            responses_tensor = torch.cat(expanded_response_chunks, dim=0)
            response_mask_tensor = torch.cat(expanded_mask_chunks, dim=0)
            inputs_tensor = torch.cat(expanded_input_chunks, dim=0)
            attention_tensor = torch.cat(expanded_attention_chunks, dim=0)
            position_tensor = torch.cat(expanded_position_chunks, dim=0)

            batch_tensors: dict[str, torch.Tensor] = {
                "prompts": prompts_tensor,
                "responses": responses_tensor,
                "response_mask": response_mask_tensor,
                "input_ids": inputs_tensor,
                "attention_mask": attention_tensor,
                "position_ids": position_tensor,
            }

            if include_base_tensors and base_prompts_tensor is not None:
                batch_tensors.update(
                    {
                        "base_prompts": base_prompts_tensor,
                        "base_responses": base_responses_tensor,
                        "base_response_mask": base_response_mask_tensor,
                        "base_input_ids": base_input_ids_tensor,
                        "base_attention_mask": base_attention_mask_tensor,
                        "base_position_ids": base_position_ids_tensor,
                    }
                )
            if has_logprobs and logprob_chunks:
                batch_tensors["rollout_log_probs"] = torch.cat(logprob_chunks, dim=0)

            batch = TensorDict(
                batch_tensors,
                batch_size=(total_sequences,),
            )

            if all(score is not None for score in reward_scores):
                reward_tensor = torch.tensor([
                    score if score is not None else 0.0 for score in reward_scores
                ], dtype=torch.float32)
                valid_mask = torch.tensor([score is not None for score in reward_scores], dtype=torch.bool)

                expanded_rm_scores = torch.zeros_like(response_mask_tensor, dtype=torch.float32)
                expanded_positions = response_mask_tensor.sum(dim=1) - 1
                expanded_positions = expanded_positions.clamp(min=0)
                expanded_rm_scores[torch.arange(expanded_rm_scores.size(0)), expanded_positions] = reward_tensor
                expanded_rm_scores = expanded_rm_scores * valid_mask.view(-1, 1)
                batch["rm_scores"] = expanded_rm_scores

                if include_base_tensors and base_response_mask_tensor is not None:
                    base_count = base_response_mask_tensor.size(0)
                    base_rm_scores = torch.zeros_like(base_response_mask_tensor, dtype=torch.float32)
                    base_positions = base_response_mask_tensor.sum(dim=1) - 1
                    base_positions = base_positions.clamp(min=0)
                    base_rm_scores[torch.arange(base_count), base_positions] = reward_tensor[:base_count]
                    base_rm_scores = base_rm_scores * valid_mask[:base_count].view(-1, 1)
                    batch["base_rm_scores"] = base_rm_scores

            non_tensor_batch: dict[str, np.ndarray] = {
                "__num_turns__": np.array(num_turns_list, dtype=np.int32),
            }

            if len(is_last_in_expanded_flags) != total_sequences:
                raise ValueError(
                    "Mismatch between expanded sequence count and is_last_in_expanded flags."
                )
            non_tensor_batch["is_last_in_expanded"] = np.array(
                is_last_in_expanded_flags, dtype=np.bool_
            )

            reward_extra_keys = sorted({key for info in reward_extra_infos for key in info.keys()})
            for key in reward_extra_keys:
                non_tensor_batch[key] = _to_object_array([info.get(key) for info in reward_extra_infos])

            if any(mmi is not None for mmi in multi_modal_inputs_flat):
                non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_flat, dtype=object)

            for key, values in extra_field_values.items():
                non_tensor_batch[key] = _to_object_array(values)

            for key, values in base_field_values.items():
                non_tensor_batch[key] = _to_object_array(values)

            if "uid" not in non_tensor_batch and "uid" in base_field_values:
                non_tensor_batch["uid"] = _to_object_array(base_field_values["uid"])

            metrics = metrics_list
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}

            batch = DataProto(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
                meta_info=meta_info,
            )
        
            # use torch.save to save batch for debugging
            if os.getenv("VERL_AGENT_DEBUG", "0") == "1":
                pid = os.getpid()
                debug_path_base = os.getenv("VERL_AGENT_DEBUG_PATH", "/tmp/verl_agent_debug.pt")
                path_without_ext, ext = os.path.splitext(debug_path_base)
                debug_path = f"{path_without_ext}_{pid}{ext}"
                print(f"[AgentLoop] saving batch to {debug_path}")
                torch.save(batch, debug_path)

            return batch

        # Fallback to original (unnested) aggregation when expanded sequences are disabled.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)

        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
            "is_last_in_expanded": np.ones(len(inputs), dtype=np.bool_),
        }

        base_field_values: dict[str, list[Any]] = defaultdict(list)
        for (_, _, _, _, _, base_fields) in expanded_cache:
            for key, value in base_fields.items():
                base_field_values[key].append(value)
        base_field_values["base_seq_uid"].extend(base_seq_uids)

        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = sorted({key for info in reward_extra_infos for key in info.keys()})
        for key in reward_extra_keys:
            non_tensor_batch[key] = _to_object_array([info.get(key) for info in reward_extra_infos])

        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            extra_fields[key] = _to_object_array([input.extra_fields.get(key) for input in inputs])

        non_tensor_batch.update(extra_fields)
        for key, values in base_field_values.items():
            non_tensor_batch[key] = _to_object_array(values)

        if "uid" not in non_tensor_batch and "uid" in base_field_values:
            non_tensor_batch["uid"] = _to_object_array(base_field_values["uid"])
        meta_info = {"metrics": [input.metrics.model_dump() for input in inputs], "reward_extra_keys": reward_extra_keys}

        batch = DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )

        # use torch.save to save batch for debugging
        if os.getenv("VERL_AGENT_DEBUG", "0") == "1":
            pid = os.getpid()
            debug_path_base = os.getenv("VERL_AGENT_DEBUG_PATH", "/tmp/verl_agent_debug.pt")
            path_without_ext, ext = os.path.splitext(debug_path_base)
            debug_path = f"{path_without_ext}_{pid}{ext}"
            print(f"[AgentLoop] saving batch to {debug_path}")
            torch.save(batch, debug_path)

        return batch


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group for hybrid mode; None for standalone mode.
        """
        self.config = config
        self.worker_group = worker_group
        self.rm_executor = None
        self.rm_micro_batch_size = None
        if rm_wg:

            def batch_fn(data_list: list[DataProto]) -> list[torch.Tensor]:
                new_data_list = []
                for data in data_list:
                    temp_non_tensor_batch = {"__num_turns__": data.non_tensor_batch["__num_turns__"]}
                    temp_data = DataProto(batch=data.batch, non_tensor_batch=temp_non_tensor_batch)
                    new_data_list.append(temp_data)

                new_batch = DataProto.concat(new_data_list)
                out_data = rm_wg.compute_rm_score(new_batch)
                return out_data.split(1)

            self.rm_executor = BatchExecutor.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(batch_fn, rm_wg.world_size)

            self.rm_micro_batch_size = rm_wg.world_size

        self._initialize_llm_servers()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self):
        rollout_world_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_rank, config=self.config, gpus_per_node=self.config.trainer.n_gpus_per_node
            )
            for replica_rank in range(num_replicas)
        ]
        if self.worker_group:
            self._run_all([server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.server_handles, self.rm_executor)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        if self.rm_micro_batch_size and len(prompts) % self.rm_micro_batch_size != 0:
            raise ValueError(
                f"The length of prompts {len(prompts)} cannot divide the world size of rm_wg {self.rm_micro_batch_size}"
            )
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # Preserve reward model keys for reward computation (similar to PPO trainer)
        reward_model_keys = {"data_source", "reward_model", "extra_info", "uid"} & prompts.non_tensor_batch.keys()
        for key in reward_model_keys:
            if key not in output.non_tensor_batch:
                output.non_tensor_batch[key] = prompts.non_tensor_batch[key]

        # Optionally compute length penalty across the concatenated outputs (global grouping).
        if prompts.meta_info["compute_length_penalty"] and self.config.reward_model.config.length_penalty_enabled:
            self._attach_length_penalty(output)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _attach_length_penalty(self, output: DataProto) -> None:
        """Compute length penalty over the concatenated batch and attach token-level scores.

        This computes group-wise penalties using uid as the grouping id and one base entry per
        base_seq_uid, then broadcasts to expanded sequences.
        """
        lp_cfg = self.config.reward_model.config
        tau = lp_cfg.length_penalty_tau
        beta = lp_cfg.length_penalty_beta
        eps = lp_cfg.length_penalty_epsilon
        window = lp_cfg.length_penalty_window
        length_metric_cfg = lp_cfg.length_penalty_length_metric

        non_tensor = output.non_tensor_batch
        if "response_mask" not in output.batch:
            raise ValueError("response_mask missing from batch; cannot compute length penalty")

        # Collect per-base stats
        base_seq_uid_arr = non_tensor["base_seq_uid"]
        uid_arr = non_tensor["uid"]

        # index rows by base_seq_uid
        indices_by_base: dict[str, list[int]] = {}
        for i, b in enumerate(base_seq_uid_arr.tolist()):
            indices_by_base.setdefault(str(b), []).append(i)

        base_group_ids: list[str] = []
        base_lengths: list[int] = []
        base_corrects: list[bool] = []
        penalty_per_row: list[float] = [0.0] * len(output)

        # Prepare per-base records
        response_mask = output.batch["response_mask"]
        for b_uid, row_indices in indices_by_base.items():
            # The reason for getting these from the first item in the base group is that these are the same within the base group (across expanded sequences).
            # This includes the length metric and correctness: the length is obtained from the base sample so it's the same across the base group.
            index_first_item_in_base_group = row_indices[0]
            # note that the group is the GRPO group, so it should use uid instead of base_seq_uid
            group_id = str(uid_arr[index_first_item_in_base_group])
            # correctness
            correct = bool(non_tensor["correct"][index_first_item_in_base_group])

            # length metric
            length_val = int(non_tensor[length_metric_cfg][index_first_item_in_base_group])
            base_group_ids.append(group_id)
            base_lengths.append(length_val)
            base_corrects.append(correct)

        # Compute penalties per base and broadcast to rows
        debug_verbose = (os.getenv("VERL_AGENT_DEBUG", "0") == "1") or bool(lp_cfg.verbose)
        base_penalties = _compute_groupwise_length_penalties(
            base_group_ids=base_group_ids,
            base_lengths=base_lengths,
            base_corrects=base_corrects,
            tau=tau,
            beta=beta,
            eps=eps,
            window=window,
            verbose=debug_verbose,
            metric_name=length_metric_cfg,
        )

        # Map base penalty to each row
        for pen, (b_uid, row_indices) in zip(base_penalties, indices_by_base.items(), strict=False):
            for idx in row_indices:
                penalty_per_row[idx] = float(pen)

        # Build token-level tensor and attach at last token position per row
        device = response_mask.device
        length_penalty_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        positions = response_mask.sum(dim=1) - 1
        positions = positions.clamp(min=0)
        length_penalty_scores[torch.arange(length_penalty_scores.size(0), device=device), positions] = (
            torch.tensor(penalty_per_row, dtype=torch.float32, device=device)
        )

        output.batch["length_penalty_scores"] = length_penalty_scores

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
