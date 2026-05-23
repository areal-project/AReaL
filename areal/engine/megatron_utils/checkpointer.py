# SPDX-License-Identifier: Apache-2.0

# Modified from VeRL: verl/utils/checkpoint/megatron_checkpoint_manager.py
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

import os
import random

import numpy as np
import torch
import torch.distributed
from megatron.core import dist_checkpointing, mpu, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)

from areal.infra.platforms import current_platform
from areal.utils import logging

logger = logging.getLogger("MegatronCheckpointer")


def log_with_rank(message: str, rank: int, log_only_rank_0: bool = False):
    if not log_only_rank_0 or rank == 0:
        logger.info(f"[Rank {rank}] {message}")


def get_device_name() -> str:
    if current_platform.is_available():
        device = current_platform.device_type
    else:
        device = "cpu"
    return device


def save_dist_checkpointing(sharded_state_dict, ckpt_path, async_save=False):
    validate_sharding_integrity = True
    # Get checkpointing strategies
    save_strategy = get_default_save_sharded_strategy("torch_dist")
    save_strategy = FullyParallelSaveStrategyWrapper(
        save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    # Save model sharded state dicts
    async_save_request = dist_checkpointing.save(
        sharded_state_dict,
        ckpt_path,
        sharded_strategy=save_strategy,
        async_sharded_save=async_save,
        validate_access_integrity=validate_sharding_integrity,
    )

    return async_save_request


def load_dist_checkpointing(sharded_state_dict, ckpt_dir):
    # Get checkpointing strategies
    load_strategy = get_default_load_sharded_strategy(ckpt_dir)
    load_strategy = FullyParallelLoadStrategyWrapper(
        load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    # Load model sharded state dicts
    state_dict = dist_checkpointing.load(
        sharded_state_dict, ckpt_dir, sharded_strategy=load_strategy
    )

    return state_dict


class MegatronCheckpointManager:
    """
    Checkpoint manager for Megatron-LM distributed training.

    This class manages the saving and loading of model checkpoints in a Megatron-LM
    distributed training environment. It handles various aspects of checkpointing
    including model states, optimizer states, learning rate schedulers, and random
    number generator states.

    Key features:
    - Distributed checkpoint saving and loading using Megatron's dist_checkpointing
    - Support for tensor parallel, pipeline parallel, and data parallel configurations
    - Automatic handling of model state dictionaries across multiple pipeline stages
    - Integration with HuggingFace model configurations and tokenizers
    - Random number generator state management for reproducibility
    - Support for both synchronous and asynchronous checkpoint operations

    The manager automatically handles:
    - Directory structure creation based on global steps and process ranks
    - Optimizer and scheduler state persistence
    - CUDA RNG state management for deterministic training
    - Checkpoint cleanup and retention policies

    Args:
        model: The Megatron model instance to checkpoint
        optimizer: The optimizer instance (optional)
        lr_scheduler: The learning rate scheduler instance (optional)

    Attributes:
        model: Reference to the Megatron model being checkpointed
        optimizer: Reference to the optimizer (if provided)
        lr_scheduler: Reference to the learning rate scheduler (if provided)
        rank: Current process rank in the distributed setup

    Example:
        ```python
        checkpoint_manager = MegatronCheckpointManager(
            model=megatron_model,
            optimizer=optimizer,
            lr_scheduler=scheduler
        )

        checkpoint_manager.save_checkpoint(
            local_path="checkpoints/step_1000",
            global_step=1000
        )

        checkpoint_manager.load_checkpoint(
            local_path="checkpoints/step_1000"
        )
        ```
    """

    def __init__(
        self,
        model: torch.nn.ModuleList,
        optimizer,
        lr_scheduler,
        use_distributed_optimizer: bool = True,
        use_checkpoint_opt_param_scheduler: bool = False,
        use_dist_checkpointing: bool = True,
        async_save: bool = False,
        distrib_optim_sharding_type: str = "dp_reshardable",
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.use_distributed_optimizer = use_distributed_optimizer
        assert self.use_distributed_optimizer, (
            "MegatronCheckpointManager now only support distributed optimizer"
        )
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler
        self.rank = torch.distributed.get_rank()
        self.use_dist_checkpointing = use_dist_checkpointing
        self.async_save = async_save
        self.distrib_optim_sharding_type = distrib_optim_sharding_type
        if async_save:
            raise NotImplementedError("Async save not implenmented yet!")

    def get_rng_state(
        self, use_dist_ckpt: bool = True, data_parallel_random_init: bool = False
    ):
        """collect rng state across data parallel ranks"""
        rng_state = {
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
        }

        if get_device_name() != "cpu":
            rng_state[f"{get_device_name()}_rng_state"] = (
                current_platform.get_rng_state()
            )

        rng_state_list = None
        if (
            torch.distributed.is_initialized()
            and mpu.get_data_parallel_world_size() > 1
            and data_parallel_random_init
        ):
            rng_state_list = [None for i in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather_object(
                rng_state_list, rng_state, group=mpu.get_data_parallel_group()
            )
        else:
            rng_state_list = [rng_state]

        if use_dist_ckpt:
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            tp_size = mpu.get_tensor_model_parallel_world_size()
            rng_state_list = ShardedObject(
                "rng_state",
                rng_state_list,
                (pp_size, tp_size),
                (pp_rank, tp_rank),
                replica_id=mpu.get_data_parallel_rank(with_context_parallel=True),
            )

        return rng_state_list

    def get_checkpoint_name(
        self,
        checkpoints_path,
        pipeline_parallel=None,
        tensor_rank=None,
        pipeline_rank=None,
        cp_rank=None,
        expert_parallel=None,
        expert_rank=None,
        return_base_dir=True,
        basename="model.pt",
    ):
        """Determine the directory name for this rank's checkpoint."""
        # Use both the tensor and pipeline MP rank.
        if pipeline_parallel is None:
            pipeline_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        if tensor_rank is None:
            tensor_rank = mpu.get_tensor_model_parallel_rank()
        if pipeline_rank is None:
            pipeline_rank = mpu.get_pipeline_model_parallel_rank()
        if cp_rank is None:
            cp_rank = mpu.get_context_parallel_rank()
        if expert_parallel is None:
            expert_parallel = mpu.get_expert_model_parallel_world_size() > 1
        if expert_rank is None:
            expert_rank = mpu.get_expert_model_parallel_rank()

        # Use both the tensor and pipeline MP rank. If using the distributed
        # optimizer, then the optimizer's path must additionally include the
        # data parallel rank.

        # due to the fact that models are identical across cp ranks, cp rank is not used in the checkpoint path
        if not pipeline_parallel:
            common_path = os.path.join(checkpoints_path, f"mp_rank_{tensor_rank:02d}")
        else:
            common_path = os.path.join(
                checkpoints_path, f"mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}"
            )

        if expert_parallel:
            common_path = common_path + f"_{expert_rank:03d}"

        os.makedirs(common_path, exist_ok=True)

        if return_base_dir:
            return common_path
        return os.path.join(common_path, basename)

    def generate_state_dict(
        self,
        with_model: bool = True,
        with_optimizer: bool = True,
        with_rng: bool = True,
    ):
        # For save dist checkpointing
        state_dict = {}

        # All ranks Save Model to reduce memory pressure
        if with_model:
            # Get sharded state dict, notice that state_dict will collect among dp groups, causing memory pressure
            for vpp_rank, model in enumerate(self.model):
                if len(self.model) > 1:
                    mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                    key = f"model{vpp_rank}" if len(self.model) > 1 else "model"
                else:
                    key = "model"
                if hasattr(model, "module"):
                    model = model.module
                state_dict[key] = model.sharded_state_dict()

        # Optimizer State Dict
        if with_optimizer:
            torch.distributed.barrier()
            # megatron-core >=0.11 removed flattened_range support in
            # ShardedTensor.validate_metadata_integrity(), but the default
            # sharding type (fully_sharded_model_space) still sets
            # flattened_range, causing save/load to fail. Use
            # dp_reshardable which does not rely on flattened_range.
            optimizer_sharded_states = self.optimizer.sharded_state_dict(
                state_dict,
                metadata={
                    "distrib_optim_sharding_type": self.distrib_optim_sharding_type
                },
            )
            state_dict["optimizer"] = optimizer_sharded_states

            if self.lr_scheduler is not None:
                lr_state_dict = self.lr_scheduler.state_dict()
                state_dict["lr_scheduler"] = lr_state_dict

        # RNG States State Dict
        if with_rng:
            torch.distributed.barrier()
            rng_state = self.get_rng_state()
            state_dict["rng_state"] = rng_state

        return state_dict

    def load_rng_states(self, rng_states, data_parallel_random_init=False):
        # access rng_state for data parallel rank
        if data_parallel_random_init:
            rng_states = rng_states[mpu.get_data_parallel_rank()]
        else:
            rng_states = rng_states[0]
        random.setstate(rng_states["random_rng_state"])
        np.random.set_state(rng_states["np_rng_state"])
        torch.set_rng_state(rng_states["torch_rng_state"])

        if get_device_name() != "cpu":
            current_platform.set_rng_state(rng_states[f"{get_device_name()}_rng_state"])

        # Check for empty states array
        if not rng_states["rng_tracker_states"]:
            raise KeyError
        tensor_parallel.get_cuda_rng_tracker().set_states(
            rng_states["rng_tracker_states"]
        )

    def load_checkpoint(
        self,
        local_path: str,
        with_model: bool = True,
        with_optimizer: bool = True,
        with_rng: bool = True,
    ):
        if local_path is not None:
            assert os.path.exists(local_path), (
                f"Checkpoint path {local_path} does not exist."
            )
        dist_checkpoint_path = local_path

        # Get State Dict for loading
        sharded_state_dict = self.generate_state_dict(
            with_model, with_optimizer, with_rng
        )
        # Load Dist Checkpointing
        state_dict = load_dist_checkpointing(
            sharded_state_dict=sharded_state_dict,
            ckpt_dir=dist_checkpoint_path,
        )

        if with_model:
            if self.use_dist_checkpointing:
                assert "model" in state_dict or any(
                    f"model{vpp_rank}" in state_dict
                    for vpp_rank in range(len(self.model))
                ), (
                    f"Model state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
                )
                for vpp_rank, model in enumerate(self.model):
                    if len(self.model) == 1:
                        model_state_dict = state_dict["model"]
                    else:
                        assert f"model{vpp_rank}" in state_dict, (
                            f"model{vpp_rank} not found in state_dict"
                        )
                        model_state_dict = state_dict[f"model{vpp_rank}"]
                    mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                    self.model[vpp_rank].load_state_dict(model_state_dict)
                log_with_rank(
                    f"Loaded sharded model checkpoint from {local_path}",
                    rank=self.rank,
                )
            else:
                raise NotImplementedError("Please use dist checkpointing!")

        if with_optimizer:
            assert "optimizer" in state_dict, (
                f"Optimizer state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
            )
            optimizer_state_dict = state_dict["optimizer"]
            self.optimizer.load_state_dict(optimizer_state_dict)
            log_with_rank(
                f"Loaded optimizer checkpoint from {local_path}",
                rank=self.rank,
            )
            if self.use_checkpoint_opt_param_scheduler:
                assert "lr_scheduler" in state_dict, (
                    f"LR scheduler state dict not found in {state_dict.keys()}. Please check the checkpoint file "
                    f"{local_path}."
                )
                lr_scheduler_state_dict = state_dict["lr_scheduler"]
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                    log_with_rank(
                        f"Loaded LR scheduler checkpoint from {local_path}",
                        rank=self.rank,
                    )

        if with_rng:
            assert "rng_state" in state_dict, (
                f"RNG state dict not found in {state_dict.keys()}. Please check the checkpoint file {local_path}."
            )
            rng_state = state_dict["rng_state"]
            self.load_rng_states(rng_state)
            log_with_rank(f"Loaded RNG states from {local_path}", rank=self.rank)

    def save_checkpoint(
        self,
        local_path: str,
        with_model: bool = True,
        with_optimizer=True,
        with_rng: bool = True,
    ):
        dist_checkpoint_path = local_path

        if self.use_dist_checkpointing:
            # Generate state dict for saving
            state_dict = self.generate_state_dict(with_model, with_optimizer, with_rng)
            # Start Async save if enabled
            async_save_request = save_dist_checkpointing(
                sharded_state_dict=state_dict,
                ckpt_path=dist_checkpoint_path,
                async_save=self.async_save,
            )

            # Synchronize all async save requests
            if not self.async_save:
                assert async_save_request is None, (
                    "Async save request should be None when not using async save."
                )
                torch.distributed.barrier()
        else:
            raise NotImplementedError("Please use dist checkpointing!")
