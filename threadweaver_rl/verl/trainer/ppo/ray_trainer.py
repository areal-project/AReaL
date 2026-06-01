# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_reward_extra_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    summarize_reward_extra,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


def _json_default(obj):
    """Handle numpy/torch values for json.dumps."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _build_expansion_broadcast_info(
    non_tensor_batch: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[int]] | None:
    base_seq_uid = non_tensor_batch.get("base_seq_uid")
    last_flags = non_tensor_batch.get("is_last_in_expanded")

    if base_seq_uid is None or last_flags is None:
        return None

    if base_seq_uid.shape[0] != last_flags.shape[0]:
        return None

    base_seq_uid_np = np.asarray(base_seq_uid)
    last_flags_np = np.asarray(last_flags).astype(np.bool_)

    if base_seq_uid_np.ndim != 1 or last_flags_np.ndim != 1:
        return None

    last_indices_np = np.flatnonzero(last_flags_np)
    if last_indices_np.size == 0 or last_indices_np.size >= base_seq_uid_np.shape[0]:
        return None

    last_group_ids = base_seq_uid_np[last_indices_np]
    if set(last_group_ids.tolist()) != set(base_seq_uid_np.tolist()):
        return None

    last_position_map: dict[Any, int] = {}
    for pos, group_id in enumerate(last_group_ids.tolist()):
        last_position_map[group_id] = pos

    try:
        gather_positions = [last_position_map[group_id] for group_id in base_seq_uid_np.tolist()]
    except KeyError:
        return None

    return last_indices_np, gather_positions


def _compute_scalar_advantages_from_last(
    token_level_rewards: torch.Tensor,
    uid_array: np.ndarray,
    norm_by_std: bool,
    broadcast_info: tuple[np.ndarray, list[int]],
    zero_adv_if_all_ge_le: float | None = None,
) -> torch.Tensor:
    """Compute GRPO scalar advantages using only the last expanded trajectories and broadcast them."""

    last_indices_np, gather_pos_list = broadcast_info
    device = token_level_rewards.device

    last_indices = torch.as_tensor(last_indices_np, device=device, dtype=torch.long)
    gather_indices = torch.as_tensor(gather_pos_list, device=device, dtype=torch.long)

    rewards_last = token_level_rewards.index_select(0, last_indices)
    uid_last = np.asarray(uid_array)[last_indices_np]

    scalar_adv_last = core_algos.compute_grpo_outcome_normalized_scores(
        token_level_rewards=rewards_last,
        index=uid_last,
        norm_adv_by_std_in_grpo=norm_by_std,
        zero_adv_if_all_ge_le=zero_adv_if_all_ge_le,
    )

    return scalar_adv_last.index_select(0, gather_indices)


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    broadcast_from_last = bool(config.broadcast_from_last) if config is not None else False

    broadcast_info = _build_expansion_broadcast_info(data.non_tensor_batch) if broadcast_from_last else None

    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        if config and config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_calculation_mask = data.batch["response_mask"]
        if broadcast_info is None:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )
        else:
            zero_thr = None
            if config is not None:
                zero_thr = config.get("grpo_zero_adv_if_all_ge_le", None)
            scalar_advantages = _compute_scalar_advantages_from_last(
                token_level_rewards=data.batch["token_level_rewards"],
                uid_array=data.non_tensor_batch["uid"],
                norm_by_std=norm_adv_by_std_in_grpo,
                broadcast_info=broadcast_info,
                zero_adv_if_all_ge_le=zero_thr,
            )
            advantages = scalar_advantages.unsqueeze(-1) * grpo_calculation_mask
            returns = advantages
    else:
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        advantages, returns = adv_estimator_fn(**adv_kwargs)

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns

    data.batch["main_advantages"] = advantages
    data.batch["main_returns"] = returns

    second_reward_coef = 0.0
    norm_second_adv = norm_adv_by_std_in_grpo
    if config is not None:
        if hasattr(config, "get"):
            second_reward_coef = config.get("second_reward_coef", 0.0)
            norm_second_adv = config.get("norm_second_adv_by_std_in_grpo", norm_adv_by_std_in_grpo)
        else:
            second_reward_coef = getattr(config, "second_reward_coef", 0.0)
            norm_second_adv = getattr(config, "norm_second_adv_by_std_in_grpo", norm_adv_by_std_in_grpo)

    secondary_rewards = data.batch.get("token_level_secondary_rewards")
    if (
        secondary_rewards is not None
        and second_reward_coef
        and torch.is_tensor(secondary_rewards)
        and torch.any(secondary_rewards != 0)
    ):
        if adv_estimator == AdvantageEstimator.GAE:
            secondary_advantages, secondary_returns = core_algos.compute_gae_advantage_return(
                token_level_rewards=secondary_rewards,
                values=data.batch["values"],
                response_mask=data.batch["response_mask"],
                gamma=gamma,
                lam=lam,
            )
        elif adv_estimator == AdvantageEstimator.GRPO:
            grpo_calculation_mask = data.batch["response_mask"]
            if broadcast_info is None:
                secondary_advantages, secondary_returns = core_algos.compute_grpo_outcome_advantage(
                    token_level_rewards=secondary_rewards,
                    response_mask=grpo_calculation_mask,
                    index=data.non_tensor_batch["uid"],
                    norm_adv_by_std_in_grpo=norm_second_adv,
                )
            else:
                secondary_scalar_adv = _compute_scalar_advantages_from_last(
                    token_level_rewards=secondary_rewards,
                    uid_array=data.non_tensor_batch["uid"],
                    norm_by_std=norm_second_adv,
                    broadcast_info=broadcast_info,
                    zero_adv_if_all_ge_le=None,
                )
                secondary_advantages = secondary_scalar_adv.unsqueeze(-1) * grpo_calculation_mask
                secondary_returns = secondary_advantages
        else:
            adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
            adv_kwargs = {
                "token_level_rewards": secondary_rewards,
                "response_mask": data.batch["response_mask"],
                "config": config,
            }
            if "uid" in data.non_tensor_batch:
                adv_kwargs["index"] = data.non_tensor_batch["uid"]
            secondary_advantages, secondary_returns = adv_estimator_fn(**adv_kwargs)

        secondary_advantages = secondary_advantages * second_reward_coef
        secondary_returns = secondary_returns * second_reward_coef

        data.batch["secondary_advantages"] = secondary_advantages
        data.batch["secondary_returns"] = secondary_returns
        data.batch["advantages"] = data.batch["advantages"] + secondary_advantages
        data.batch["returns"] = data.batch["returns"] + secondary_returns
    else:
        data.batch.pop("secondary_advantages", None)
        data.batch.pop("secondary_returns", None)

    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        # Preserve meta information (e.g., reward manager kwargs) for downstream workers.
        if batch.meta_info is not None:
            gen_batch.meta_info = deepcopy(batch.meta_info)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # Filter out pad tokens before decoding
            input_texts = [self.tokenizer.decode(ids[ids != self.tokenizer.pad_token_id], skip_special_tokens=False) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            reward_manager_kwargs = dict(test_gen_batch.meta_info.get("reward_manager_kwargs", {}))
            reward_manager_kwargs["correctness_as_reward"] = True
            test_gen_batch.meta_info["reward_manager_kwargs"] = reward_manager_kwargs
            test_gen_batch.meta_info["return_expanded_sequences"] = False
            # Always disable length penalty computation during validation
            test_gen_batch.meta_info["compute_length_penalty"] = False
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            gen_batch_size = len(test_gen_batch_padded.batch)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # We don't enable `return_expanded_sequences` for validation rollout, so the output batch size should be the same as input batch size
            assert len(test_output_gen_batch_padded.batch) == gen_batch_size, f"{len(test_output_gen_batch_padded.batch)=}, {gen_batch_size=}"
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            os.environ["GLOBAL_STEP"] = str(self.global_steps)
            result = self.val_reward_fn(test_batch, return_dict=True, correctness_as_reward=True)

            if isinstance(result, dict):
                reward_tensor_dict = result.get("reward_tensor", result)
            else:
                reward_tensor_dict = result

            reward_tensor = None
            if isinstance(reward_tensor_dict, dict):
                if "main_reward_tensor" in reward_tensor_dict:
                    reward_tensor = reward_tensor_dict["main_reward_tensor"]
                elif len(reward_tensor_dict) == 1:
                    reward_tensor = next(iter(reward_tensor_dict.values()))
                else:
                    tensors = [
                        value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                        for value in reward_tensor_dict.values()
                    ]
                    base = tensors[0]
                    tensors = [tensor.to(base.device) for tensor in tensors]
                    reward_tensor = torch.stack(tensors, dim=0).sum(dim=0)
            else:
                reward_tensor = reward_tensor_dict

            if reward_tensor is None:
                raise ValueError("Validation reward function returned no usable reward tensor")

            if not isinstance(reward_tensor, torch.Tensor):
                reward_tensor = torch.as_tensor(reward_tensor)

            reward_tensor_cpu = reward_tensor.detach().cpu()
            if reward_tensor_cpu.ndim == 0:
                scores = [reward_tensor_cpu.item()]
            elif reward_tensor_cpu.ndim == 1:
                scores = reward_tensor_cpu.tolist()
            else:
                scores = reward_tensor_cpu.sum(dim=-1).tolist()

            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if isinstance(result, dict) and "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    if key in {"ground_truth", "error", "details"}:
                        continue
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(scores)))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        sanitized_reward_infos = self._sanitize_validation_metrics(reward_extra_infos_dict, len(sample_scores))

        data_src2var2metric2val = process_validation_metrics(
            data_sources,
            sample_uids,
            sanitized_reward_infos,
        )
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        extras_for_summary = {k: v for k, v in reward_extra_infos_dict.items() if k != "reward"}
        metric_dict.update(summarize_reward_extra(extras_for_summary, prefix="val-extra"))

        return metric_dict

    @staticmethod
    def _sanitize_validation_metrics(
        reward_infos: dict[str, list],
        expected_len: int,
    ) -> dict[str, list[float]]:
        sanitized: dict[str, list[float]] = {}
        for key, values in reward_infos.items():
            cleaned: list[float] = []
            for idx, value in enumerate(values):
                if value is None:
                    cleaned.append(np.nan)
                    continue
                if isinstance(value, dict):
                    cleaned.append(np.nan)
                    continue
                if isinstance(value, (np.ndarray, list, tuple)):
                    try:
                        arr = np.asarray(value, dtype=float)
                    except (TypeError, ValueError):
                        cleaned.append(np.nan)
                        continue
                    cleaned.append(float(np.nanmean(arr)) if arr.size > 0 else np.nan)
                    continue
                if isinstance(value, (np.generic, float, int)):
                    cleaned.append(float(value))
                    continue
                if isinstance(value, bool):
                    cleaned.append(float(value))
                    continue
                # Fallback for unsupported types (e.g., dicts/strings)
                cleaned.append(np.nan)

            # Pad or trim to expected length to keep shape consistent
            if len(cleaned) < expected_len:
                cleaned.extend([np.nan] * (expected_len - len(cleaned)))
            elif len(cleaned) > expected_len:
                cleaned = cleaned[:expected_len]

            sanitized[key] = cleaned

        return sanitized

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _align_batch_with_generated(
        self,
        base_batch: DataProto,
        generated: DataProto,
        exclude_from_base: list[str] = None,
        auto_exclude_from_base: bool = False,
    ) -> DataProto:
        """Repeat base_batch rows to align with flattened expanded outputs before union."""

        def _extract_uid_array(data: DataProto) -> Optional[list]:
            # Prefer non-tensor storage since uid is typically a numpy object array.
            if "uid" in data.non_tensor_batch:
                uid_arr = data.non_tensor_batch["uid"]
                if isinstance(uid_arr, np.ndarray):
                    return uid_arr.tolist()
                return list(uid_arr)
            if data.batch is not None and "uid" in data.batch.keys():
                uid_tensor = data.batch["uid"]
                if isinstance(uid_tensor, torch.Tensor):
                    return uid_tensor.detach().cpu().tolist()
            return None

        exclude_from_base = exclude_from_base or []

        base_uids = _extract_uid_array(base_batch)
        generated_uids = _extract_uid_array(generated)

        indices: Optional[torch.Tensor] = None
        if base_uids is not None and generated_uids is not None:
            if len(base_uids) != len(set(base_uids)):
                raise ValueError("Duplicate uid detected in base batch; cannot align reliably.")

            uid_to_index = {uid: idx for idx, uid in enumerate(base_uids)}
            missing_uids = [uid for uid in generated_uids if uid not in uid_to_index]
            if missing_uids:
                raise ValueError(
                    "Generated batch contains uid(s) not present in base batch: "
                    + ", ".join(map(str, missing_uids))
                )

            indices_list = [uid_to_index[uid] for uid in generated_uids]
            indices = torch.tensor(indices_list, dtype=torch.long)

        if indices is None:
            raise ValueError(
                "Unable to align batches: `uid` field missing from either base or generated data. "
                "Ensure rollouts propagate `uid` for alignment."
            )

        expanded_tensors = base_batch.batch[indices]
        indices_np = indices.numpy()

        if auto_exclude_from_base:
            # exclude any non-tensor keys that are in both base and generated batches
            auto_exclude_set = set(base_batch.non_tensor_batch.keys()) & set(generated.non_tensor_batch.keys())
            if auto_exclude_set:
                # The keys printed include `exclude_from_base` keys too
                print(
                    "These non-tensor keys are in both base and generated batches, and will be excluded from base batch when aligning: "
                    f"{auto_exclude_set}"
                )
                exclude_from_base = list(set(exclude_from_base) | auto_exclude_set)

        expanded_non_tensor = {
            key: val[indices_np]
            for key, val in base_batch.non_tensor_batch.items()
            if key not in exclude_from_base
        }
        expanded_meta = dict(base_batch.meta_info) if base_batch.meta_info is not None else {}
        expanded_batch = DataProto(
            batch=expanded_tensors,
            non_tensor_batch=expanded_non_tensor,
            meta_info=expanded_meta,
        )
        return expanded_batch.union(generated)

    def _compute_last_only_metrics(
        self,
        batch: DataProto,
        timing_raw: dict[str, float],
        n_gpus: int,
    ) -> dict[str, float]:
        """Derive metrics restricted to sequences marked as final within expanded rollouts."""

        if "is_last_in_expanded" not in batch.non_tensor_batch:
            return {}

        is_last_flags = np.asarray(batch.non_tensor_batch["is_last_in_expanded"])
        if is_last_flags.size == 0:
            return {}

        mask_np = is_last_flags.reshape(-1).astype(np.bool_)
        if not np.any(mask_np):
            return {}

        indices_np = np.nonzero(mask_np)[0]
        if indices_np.size == 0:
            return {}

        indices_tensor = torch.from_numpy(indices_np)
        last_batch = batch.select_idxs(indices_tensor)

        # Ensure meta_info mutations do not leak back to the original batch.
        last_batch.meta_info = deepcopy(batch.meta_info) if batch.meta_info is not None else {}
        if last_batch.meta_info:
            for key, value in list(last_batch.meta_info.items()):
                if isinstance(value, list) and len(value) == len(mask_np):
                    last_batch.meta_info[key] = [
                        item for item, keep in zip(value, mask_np) if keep
                    ]

        prefixed_metrics: dict[str, float] = {}

        data_metrics = compute_data_metrics(last_batch, use_critic=self.use_critic)
        prefixed_metrics.update({f"last_only/{key}": value for key, value in data_metrics.items()})

        reward_extra_metrics = compute_reward_extra_metrics(last_batch)
        prefixed_metrics.update({f"last_only/{key}": value for key, value in reward_extra_metrics.items()})

        timing_metrics = compute_timing_metrics(last_batch, timing_raw)
        prefixed_metrics.update({f"last_only/{key}": value for key, value in timing_metrics.items()})

        throughput_metrics = compute_throughout_metrics(last_batch, timing_raw, n_gpus)
        prefixed_metrics.update({f"last_only/{key}": value for key, value in throughput_metrics.items()})

        if "rollout_log_probs" in last_batch.batch:
            from verl.utils.debug.metrics import calculate_debug_metrics

            debug_metrics = calculate_debug_metrics(last_batch)
            prefixed_metrics.update({f"last_only/{key}": value for key, value in debug_metrics.items()})

        return prefixed_metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                return_expanded_sequences = self.config.actor_rollout_ref.rollout.agent_return_expanded_sequences
                return_base_tensors = (
                    self.config.actor_rollout_ref.rollout.agent_return_base_tensors_with_expanded
                )
                # we put these flags in meta_info instead of loading from config since they are always set to False in validate
                gen_batch.meta_info["return_expanded_sequences"] = return_expanded_sequences
                gen_batch.meta_info["return_base_tensors_with_expanded"] = return_base_tensors
                lp_enabled = self.config.reward_model.config.length_penalty_enabled
                gen_batch.meta_info["compute_length_penalty"] = lp_enabled

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        raise NotImplementedError("REMAX is not implemented yet")

                    # repeat to align with repeated responses in rollout
                    if return_expanded_sequences:
                        # `batch` before `_align_batch_with_generated` has unique uid
                        batch = self._align_batch_with_generated(batch, gen_batch_output, exclude_from_base=['data_source', 'extra_info', 'reward_model'], auto_exclude_from_base=True)
                    else:
                        batch = batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                        )
                        batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        # Pad batch to be divisible by world_size for load balancing
                        world_size = self.actor_rollout_wg.world_size
                        batch_size = len(batch.batch)
                        remainder = batch_size % world_size
                        if remainder != 0:
                            pad_size = world_size - remainder
                            # Pad with samples that have all attention_mask = 0 (will be ignored)
                            batch_padded, _ = pad_dataproto_to_divisor(batch, world_size)
                            # Set attention_mask to 0 for padded samples to ignore them
                            batch_padded.batch["attention_mask"][-pad_size:] = 0
                            # Also set response_mask to 0 if it exists
                            if "response_mask" in batch_padded.batch:
                                batch_padded.batch["response_mask"][-pad_size:] = 0
                            self._balance_batch(batch_padded, metrics=metrics)
                            # Use the reordered padded batch (padding will be ignored due to zero attention_mask)
                            batch = batch_padded

                            # This is for debugging purpose only, not for normal usage
                            # Let's randomly zero out half of the samples' attention_mask and response_mask
                            # to debug, and see if balancing still works
                            if os.getenv("VERL_TRAINER_TEST_BALANCE_BATCH", "0") == "1":
                                effective_batch_size = batch_size // 2
                                batch_size = len(batch.batch)
                                indices = np.random.choice(batch_size, effective_batch_size, replace=False)
                                mask = np.zeros(batch_size, dtype=int)
                                mask[indices] = 1
                                batch.batch["attention_mask"] *= torch.tensor(mask, device=batch.batch["attention_mask"].device).unsqueeze(-1)
                                if "response_mask" in batch.batch:
                                    batch.batch["response_mask"] *= torch.tensor(mask, device=batch.batch["response_mask"].device).unsqueeze(-1)
                                print(f"[Trainer] Simulated effective batch size: {effective_batch_size}")
                        else:
                            self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if return_expanded_sequences:
                            assert "rm_scores" in batch.batch.keys(), f"rm_scores must be in batch when return_expanded_sequences is True, but got {batch.batch.keys()}"

                        os.environ["GLOBAL_STEP"] = str(self.global_steps)

                        reward_tensor_dict: dict[str, torch.Tensor]
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch,
                                reward_fn=self.reward_fn,
                                correctness_as_reward=False,
                            )
                        else:
                            reward_tensor_dict, reward_extra_infos_dict = compute_reward(
                                batch,
                                self.reward_fn,
                                correctness_as_reward=False,
                            )

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)

                        if return_expanded_sequences and "is_last_in_expanded" in batch.non_tensor_batch:
                            last_mask_np = batch.non_tensor_batch["is_last_in_expanded"].astype(np.bool_)
                            if last_mask_np.any():
                                last_mask = torch.from_numpy(last_mask_np).to(entropys.device)
                                entropy_last = agg_loss(
                                    loss_mat=entropys[last_mask],
                                    loss_mask=response_masks[last_mask],
                                    loss_agg_mode=loss_agg_mode,
                                )
                                metrics["last_only/actor/entropy"] = entropy_last.detach().item()

                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor_dict, reward_extra_infos_dict = ray.get(future_reward)

                        main_reward_tensor = reward_tensor_dict.get(
                            "main_reward_tensor"
                        )
                        secondary_reward_tensor = reward_tensor_dict.get(
                            "secondary_reward_tensor"
                        )

                        batch.batch["token_level_scores"] = main_reward_tensor
                        if secondary_reward_tensor is not None:
                            batch.batch["token_level_secondary_scores"] = secondary_reward_tensor
                        else:
                            batch.batch.pop("token_level_secondary_scores", None)

                        # Apply length penalty if available (computed in AgentLoop _postprocess for base and duplicated).
                        if "length_penalty_scores" in batch.batch:
                            try:
                                batch.batch["token_level_scores"] = (
                                    batch.batch["token_level_scores"] - batch.batch["length_penalty_scores"]
                                )
                            except Exception as e:
                                print(f"Warning: failed to subtract length_penalty_scores: {e}")

                        if reward_extra_infos_dict:
                            safe_extra = {}
                            for k, v in reward_extra_infos_dict.items():
                                try:
                                    safe_extra[k] = np.array(v)
                                except ValueError:
                                    safe_extra[k] = np.array(v, dtype=object)
                            batch.non_tensor_batch.update(safe_extra)
                            extra_keys = list(reward_extra_infos_dict.keys())
                            if batch.meta_info is None:
                                batch.meta_info = {}
                            stored_keys = batch.meta_info.get("reward_extra_keys", [])
                            merged_keys = sorted(set(stored_keys) | set(extra_keys))
                            batch.meta_info["reward_extra_keys"] = merged_keys

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        if batch.batch.get("token_level_secondary_scores") is not None:
                            batch.batch["token_level_secondary_rewards"] = batch.batch[
                                "token_level_secondary_scores"
                            ]
                        else:
                            batch.batch.pop("token_level_secondary_rewards", None)

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if os.getenv("VERL_AGENT_DEBUG", "0") == "1":
                        # dump the batch for debugging
                        pid = os.getpid()
                        debug_path_base = os.getenv("VERL_AGENT_DEBUG_PATH", "/tmp/verl_agent_debug.pt")
                        path_without_ext, ext = os.path.splitext(debug_path_base)
                        debug_path = f"{path_without_ext}_{pid}_before_update{ext}"
                        print(f"[AgentLoop] saving batch prior to update_actor to {debug_path}")
                        torch.save(batch, debug_path)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_reward_extra_metrics(batch=batch))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                if return_expanded_sequences:
                    metrics.update(
                        self._compute_last_only_metrics(
                            batch=batch,
                            timing_raw=timing_raw,
                            n_gpus=n_gpus,
                        )
                    )

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
