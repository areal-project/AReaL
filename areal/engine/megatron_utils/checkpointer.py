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

from __future__ import annotations

import copy
import importlib.metadata
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
from megatron.core.dist_checkpointing.strategies.async_utils import (
    AsyncCallsQueue,
    AsyncRequest,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)

from areal.engine.megatron_utils.checkpoint_snapshot import (
    validate_shared_snapshot_capacity,
)
from areal.engine.megatron_utils.gpu_staged_optimizer_checkpoint import (
    ManagedCheckpointTransactionPhase,
    abort_managed_checkpoint_load,
    acknowledge_managed_checkpoint_recovery,
    apply_begin_managed_checkpoint_load,
    apply_managed_optimizer_reset_from_model,
    attach_managed_optimizer_identities,
    begin_managed_async_checkpoint_save,
    begin_managed_checkpoint_replacement,
    bind_managed_async_checkpoint_request,
    build_managed_optimizer_identities,
    build_managed_optimizer_outer_template,
    build_managed_optimizer_tensor_manifest,
    cancel_managed_checkpoint_replacement_configuration,
    complete_managed_async_checkpoint_save,
    configure_managed_checkpoint_snapshots,
    create_managed_checkpoint_load_transaction,
    fail_managed_async_checkpoint_save,
    has_managed_mcore_outer_schema,
    is_managed_optimizer_tensor_checkpoint_key,
    merge_managed_optimizer_tensor_manifests,
    poison_managed_checkpoint_transaction,
    preflight_managed_checkpoint_snapshots,
    prepare_managed_checkpoint_commit,
    prepare_managed_checkpoint_load,
    prepare_managed_checkpoint_recovery,
    prepare_managed_checkpoint_save,
    retry_managed_checkpoint_cleanup,
    validate_managed_checkpoint_load_request,
    validate_managed_optimizer_outer_state,
    validate_managed_optimizer_source_tensor_metadata,
    vote_managed_checkpoint_phase,
)
from areal.engine.megatron_utils.managed_async_checkpoint import (
    ManagedAsyncSaveState,
    ManagedAsyncSaveTransaction,
)
from areal.engine.megatron_utils.managed_async_finalize import (
    ManagedAsyncRecoveryState,
    ManagedAsyncRecoveryToken,
    abort_managed_async_calls,
    finalize_managed_async_calls,
    get_managed_async_worker_recovery,
    preflight_managed_async_finalize,
)
from areal.engine.megatron_utils.managed_async_marker import (
    MANAGED_ASYNC_COMPLETE,
    MANAGED_ASYNC_INCOMPLETE,
    canonical_leaf_identities,
    canonical_ranked_leaf_identities,
    commit_prepared_marker,
    create_incomplete_marker,
    has_managed_async_marker,
    new_checkpoint_id,
    prepare_complete_marker,
    retry_post_commit_cleanup,
    validate_load_marker,
)
from areal.infra.platforms import current_platform
from areal.utils import logging, stats_tracker

logger = logging.getLogger("MegatronCheckpointer")

_MANAGED_ASYNC_INCOMPLETE = MANAGED_ASYNC_INCOMPLETE
_MANAGED_ASYNC_COMPLETE = MANAGED_ASYNC_COMPLETE


def log_with_rank(message: str, rank: int, log_only_rank_0: bool = False):
    if not log_only_rank_0 or rank == 0:
        logger.info(f"[Rank {rank}] {message}")


def get_device_name() -> str:
    if current_platform.is_available():
        device = current_platform.device_type
    else:
        device = "cpu"
    return device


def save_dist_checkpointing(
    sharded_state_dict, ckpt_path, async_save=False
) -> AsyncRequest | None:
    validate_sharding_integrity = True
    # Get checkpointing strategies
    save_strategy = get_default_save_sharded_strategy("torch_dist")
    save_strategy = FullyParallelSaveStrategyWrapper(
        save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    # Save model sharded state dicts. When async_save=True the actual IO is
    # deferred; the returned AsyncRequest must be scheduled by the caller via
    # AsyncCallsQueue.schedule_async_request(). Recent megatron-core versions
    # require an explicit async_strategy when async_sharded_save=True; "mcore"
    # selects the AsyncCallsQueue-backed implementation we use below.
    save_kwargs = dict(
        sharded_state_dict=sharded_state_dict,
        checkpoint_dir=ckpt_path,
        sharded_strategy=save_strategy,
        async_sharded_save=async_save,
        validate_access_integrity=validate_sharding_integrity,
    )
    if async_save:
        save_kwargs["async_strategy"] = "mcore"
    async_save_request = dist_checkpointing.save(**save_kwargs)

    return async_save_request


def load_dist_checkpointing(
    sharded_state_dict,
    ckpt_dir,
    *,
    strict_managed: bool = False,
    allowed_unrequested_prefixes: tuple[str, ...] = (),
    allow_managed_optimizer_tensor_state: bool = False,
):
    # Get checkpointing strategies
    load_strategy = get_default_load_sharded_strategy(ckpt_dir)
    load_strategy = FullyParallelLoadStrategyWrapper(
        load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    # Load model sharded state dicts
    if not strict_managed:
        return dist_checkpointing.load(
            sharded_state_dict, ckpt_dir, sharded_strategy=load_strategy
        )

    from megatron.core.dist_checkpointing.validation import StrictHandling

    state_dict, checkpoint_only_keys, requested_only_keys = dist_checkpointing.load(
        sharded_state_dict,
        ckpt_dir,
        sharded_strategy=load_strategy,
        strict=StrictHandling.RETURN_ALL,
    )
    if requested_only_keys:
        raise KeyError(
            "checkpoint is missing required managed optimizer/template fields: "
            f"{sorted(map(str, requested_only_keys))}"
        )
    disallowed_checkpoint_keys = {
        key
        for key in checkpoint_only_keys
        if not str(key).startswith(allowed_unrequested_prefixes)
        and not (
            allow_managed_optimizer_tensor_state
            and is_managed_optimizer_tensor_checkpoint_key(key)
        )
    }
    if disallowed_checkpoint_keys:
        raise KeyError(
            "checkpoint contains unexpected fields for the requested managed load: "
            f"{sorted(map(str, disallowed_checkpoint_keys))}"
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
        checkpoint_process_group=None,
        managed_checkpoint_enabled: bool = False,
        managed_checkpoint_snapshot_root: str | None = None,
        managed_async_finalize_timeout_seconds: float = 120.0,
    ):
        managed_kind = getattr(optimizer, "managed_checkpoint_format", None)
        fixed_muon_checkpoint = (
            isinstance(managed_kind, str) and managed_kind == "muon_dp_reshard_v2"
        )
        if fixed_muon_checkpoint and not managed_checkpoint_enabled:
            raise RuntimeError(
                "fixed-topology staged Muon requires the managed checkpoint protocol"
            )
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.use_distributed_optimizer = use_distributed_optimizer
        if not self.use_distributed_optimizer and not fixed_muon_checkpoint:
            raise AssertionError(
                "MegatronCheckpointManager requires distributed optimizer or the "
                "fixed-topology staged Muon checkpoint capability"
            )
        if fixed_muon_checkpoint and async_save:
            raise RuntimeError(
                "managed asynchronous checkpoint is not supported for staged Muon"
            )
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler
        if checkpoint_process_group is not None:
            self.rank = torch.distributed.get_rank(checkpoint_process_group)
        elif torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0
        self.use_dist_checkpointing = use_dist_checkpointing
        self.async_save = async_save
        self.checkpoint_process_group = checkpoint_process_group
        self.managed_checkpoint_enabled = managed_checkpoint_enabled
        self.managed_checkpoint_snapshot_root = managed_checkpoint_snapshot_root
        if (
            isinstance(managed_async_finalize_timeout_seconds, bool)
            or managed_async_finalize_timeout_seconds <= 0
        ):
            raise ValueError("managed async finalize timeout must be positive")
        self.managed_async_finalize_timeout_seconds = float(
            managed_async_finalize_timeout_seconds
        )
        self._managed_checkpoint_poisoned_error = None
        self._managed_checkpoint_recovery_transaction = None
        self._managed_checkpoint_cleanup_recovery = None
        self._managed_checkpoint_control_error = None
        self._managed_async_save: ManagedAsyncSaveTransaction | None = None
        self._managed_async_save_error: BaseException | None = None
        self._managed_async_last_state = ManagedAsyncSaveState.IDLE
        self._managed_async_marker_cleanup = None
        self._managed_async_marker_precommit_cleanup = None
        self._managed_async_sequence = 0
        # AsyncCallsQueue manages outstanding background save processes.
        # Created only when async_save is enabled; sync path keeps zero overhead.
        self._async_queue: AsyncCallsQueue | None = (
            AsyncCallsQueue() if async_save else None
        )

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
        is_loading: bool = False,
        *,
        prepare_optimizer_for_save: bool | None = None,
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
            if prepare_optimizer_for_save is None:
                prepare_optimizer_for_save = not is_loading
            if prepare_optimizer_for_save:
                prepare_managed_checkpoint_save(
                    self.optimizer, async_save=self.async_save
                )
            if not getattr(self, "managed_checkpoint_enabled", False):
                torch.distributed.barrier()
            # megatron-core v0.14+ removed flattened_range support (Megatron-LM
            # PR #2126), but the sharded_state_dict default
            # (fully_sharded_model_space) still emits it, so saving optimizer
            # state fails on the pinned 0.17.0. dp_reshardable is upstream's
            # current default. Trade-off: the optimizer state (not the model
            # weights) becomes reshardable only along DP -- load hard-asserts
            # the same bucket layout (per_bucket_numel_unpadded), so save and
            # load must use identical TP/PP. Fine today since recover enforces
            # the same topology; switch to fully_reshardable if cross-topology
            # resume is ever needed (also flattened_range-free, at the cost of
            # gathering optimizer state). is_loading=True pre-allocates
            # exp_avg/exp_avg_sq so the load template requests them --
            # otherwise DCP silently drops the moments on resume.
            optimizer_sharded_states = self.optimizer.sharded_state_dict(
                state_dict,
                is_loading=is_loading,
                metadata={"distrib_optim_sharding_type": "dp_reshardable"},
            )
            if has_managed_mcore_outer_schema(self.optimizer):
                attach_managed_optimizer_identities(
                    self.optimizer,
                    optimizer_sharded_states,
                    self._managed_optimizer_identities(),
                )
            state_dict["optimizer"] = optimizer_sharded_states

            if self.lr_scheduler is not None:
                lr_state_dict = self.lr_scheduler.state_dict()
                state_dict["lr_scheduler"] = lr_state_dict

        # RNG States State Dict
        if with_rng:
            if not getattr(self, "managed_checkpoint_enabled", False):
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

    def _managed_model_parameter_names(self) -> dict[torch.Tensor, str]:
        names: dict[torch.Tensor, str] = {}
        for model_index, model in enumerate(self.model):
            if hasattr(model, "module"):
                model = model.module
            prefix = "model" if len(self.model) == 1 else f"model{model_index}"
            for name, parameter in model.named_parameters():
                stable_name = f"{prefix}.{name}"
                previous = names.setdefault(parameter, stable_name)
                if previous != stable_name:
                    raise RuntimeError(
                        "managed checkpoint parameter has multiple stable model names: "
                        f"{previous!r} and {stable_name!r}"
                    )
        return names

    def _managed_optimizer_identities(self):
        binder = getattr(self.optimizer, "bind_managed_checkpoint_process_group", None)
        if callable(binder):
            binder(self._require_managed_checkpoint_group())
        return build_managed_optimizer_identities(
            self.optimizer, self._managed_model_parameter_names()
        )

    def _require_managed_checkpoint_group(self):
        group = getattr(self, "checkpoint_process_group", None)
        if group is None:
            raise RuntimeError(
                "managed checkpoint requires an explicit all-rank process group"
            )
        group_size = torch.distributed.get_world_size(group)
        get_group_ranks = getattr(torch.distributed, "get_process_group_ranks", None)
        if not callable(get_group_ranks):
            raise RuntimeError(
                "managed checkpoint requires explicit process-group membership API"
            )
        ranks = tuple(get_group_ranks(group))
        world_group = torch.distributed.group.WORLD
        if world_group is None:
            raise RuntimeError(
                "managed checkpoint requires an initialized DCP WORLD group"
            )
        world_ranks = tuple(get_group_ranks(world_group))
        if (
            len(ranks) != group_size
            or ranks != tuple(range(group_size))
            or ranks != world_ranks
        ):
            raise RuntimeError(
                "managed checkpoint process group must explicitly enumerate every "
                "DCP participant: "
                f"group_size={group_size}, ranks={ranks}, world_ranks={world_ranks}"
            )
        return group

    def _vote_managed_phase(
        self,
        phase: str,
        local_error: BaseException | None,
        transaction,
        *,
        details=None,
        require_consistent_details: bool = False,
    ):
        try:
            phase_error = vote_managed_checkpoint_phase(
                self._require_managed_checkpoint_group(),
                phase,
                local_error,
                details=details,
                require_consistent_details=require_consistent_details,
            )
        except BaseException as vote_error:
            if transaction.committed:
                transaction.post_commit_error = vote_error
                self._managed_checkpoint_control_error = (
                    f"{type(vote_error).__name__}: {vote_error}"
                )
            elif transaction is getattr(
                self, "_managed_checkpoint_recovery_transaction", None
            ) and (
                transaction.recovery_owner
                or transaction.phase
                is ManagedCheckpointTransactionPhase.RELOAD_REQUIRED
            ):
                # A recovery vote transport failure cannot revoke the retained
                # generation/action identities or manufacture a fresh poison
                # journal.  The next public call retries this exact authority.
                transaction.poisoned = True
                self._managed_checkpoint_poisoned_error = vote_error
            else:
                poison_managed_checkpoint_transaction(transaction, vote_error)
                self._managed_checkpoint_poisoned_error = vote_error
            raise
        return phase_error

    def _run_managed_phase(self, phase: str, operation, transaction):
        local_error = None
        result = None
        try:
            result = operation()
        except BaseException as error:
            local_error = error
        phase_error = self._vote_managed_phase(phase, local_error, transaction)
        return result, phase_error

    def _raise_if_managed_checkpoint_poisoned(self) -> None:
        poisoned = getattr(self, "_managed_checkpoint_poisoned_error", None)
        if poisoned is not None:
            raise RuntimeError(
                "managed checkpoint manager is poisoned by an incomplete "
                "distributed rollback"
            ) from poisoned

    def _managed_checkpoint_recovery_required(self, control_transaction) -> bool:
        """Make recovery entry uniform across all checkpoint participants."""
        local_required = bool(
            getattr(self, "_managed_checkpoint_poisoned_error", None) is not None
            or getattr(self, "_managed_checkpoint_recovery_transaction", None)
            is not None
            or getattr(control_transaction, "recovery_owner", False)
            or getattr(control_transaction, "reload_generation", None) is not None
        )

        def gather_required() -> bool:
            if (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
            ):
                return local_required
            group = self._require_managed_checkpoint_group()
            required = torch.tensor([int(local_required)], dtype=torch.int64)
            torch.distributed.all_reduce(
                required, op=torch.distributed.ReduceOp.MAX, group=group
            )
            return bool(required.item())

        required, phase_error = self._run_managed_phase(
            "recovery_required", gather_required, control_transaction
        )
        if phase_error is not None:
            self._managed_checkpoint_poisoned_error = phase_error
            raise phase_error from phase_error.local_error
        return bool(required)

    def _retry_managed_checkpoint_recovery(self, *, required: bool = False):
        """Reuse one manager-owned recovery transaction until replacement commit."""
        poison_error = getattr(self, "_managed_checkpoint_poisoned_error", None)
        retained = getattr(self, "_managed_checkpoint_recovery_transaction", None)
        if poison_error is None and retained is None and not required:
            return None
        if poison_error is None:
            self._managed_checkpoint_poisoned_error = RuntimeError(
                "another checkpoint participant requires managed recovery"
            )

        vote_transaction = retained or create_managed_checkpoint_load_transaction(None)

        def discover_recovery():
            transaction = getattr(
                self, "_managed_checkpoint_recovery_transaction", None
            )
            if transaction is not None:
                return transaction
            transaction = create_managed_checkpoint_load_transaction(self.optimizer)
            transaction.recovery_owner = True
            # Publish before any leaf recovery effect or the discovery vote.
            self._managed_checkpoint_recovery_transaction = transaction
            return transaction

        transaction, phase_error = self._run_managed_phase(
            "recovery_discover", discover_recovery, vote_transaction
        )
        if phase_error is not None:
            self._managed_checkpoint_poisoned_error = phase_error
            raise phase_error from phase_error.local_error
        assert transaction is not None
        if transaction.reload_generation is None:
            from areal.engine.megatron_utils.gpu_staged_optimizer_checkpoint import (
                ManagedCheckpointReloadGeneration,
            )

            transaction.reload_generation = ManagedCheckpointReloadGeneration()

        def retry_recovery():
            if transaction.phase is ManagedCheckpointTransactionPhase.RELOAD_REQUIRED:
                return
            prepare_managed_checkpoint_recovery(transaction, retain_authority=True)

        _, phase_error = self._run_managed_phase(
            "recovery_rollback", retry_recovery, transaction
        )
        if phase_error is not None:
            transaction.poisoned = True
            if (
                transaction.phase
                is not ManagedCheckpointTransactionPhase.RELOAD_REQUIRED
            ):
                transaction.phase = ManagedCheckpointTransactionPhase.POISONED
            self._managed_checkpoint_poisoned_error = phase_error
            raise phase_error from phase_error.local_error
        _, phase_error = self._run_managed_phase(
            "recovery_acknowledge",
            lambda: acknowledge_managed_checkpoint_recovery(transaction),
            transaction,
        )
        if phase_error is not None:
            self._managed_checkpoint_poisoned_error = phase_error
            raise phase_error from phase_error.local_error
        return transaction

    def _managed_async_checkpoint_id(self, path: str) -> tuple[str, int]:
        del path
        self._managed_async_sequence += 1
        group = self._require_managed_checkpoint_group()
        rank = torch.distributed.get_rank(group)
        value = [new_checkpoint_id() if rank == 0 else None]
        torch.distributed.broadcast_object_list(value, src=0, group=group)
        checkpoint_id = value[0]
        if not isinstance(checkpoint_id, str) or len(checkpoint_id) != 32:
            raise RuntimeError("managed async checkpoint ID broadcast was invalid")
        return checkpoint_id, self._managed_async_sequence

    def _managed_async_participants(self) -> tuple[tuple[int, ...], str]:
        group = self._require_managed_checkpoint_group()
        ranks = tuple(torch.distributed.get_process_group_ranks(group))
        backend = str(torch.distributed.get_backend(group)).lower()
        if backend != "gloo":
            raise RuntimeError(
                "managed async checkpoint marker requires a WORLD-sized Gloo "
                f"control group, got backend={backend!r}"
            )
        return ranks, backend

    def _managed_async_global_leaf_manifest(self, identities) -> tuple[list[dict], str]:
        group = self._require_managed_checkpoint_group()
        ranks, _ = self._managed_async_participants()
        local_leaves, _ = canonical_leaf_identities(identities)
        gathered = [None] * len(ranks)
        torch.distributed.all_gather_object(
            gathered,
            {
                "rank": torch.distributed.get_rank(group),
                "leaves": local_leaves,
            },
            group=group,
        )
        return canonical_ranked_leaf_identities(gathered)

    @property
    def managed_async_save_state(self) -> str:
        transaction = getattr(self, "_managed_async_save", None)
        if transaction is not None:
            return transaction.state.name
        return getattr(
            self, "_managed_async_last_state", ManagedAsyncSaveState.IDLE
        ).name

    def _create_managed_async_marker(self, transaction) -> None:
        group = self._require_managed_checkpoint_group()
        if torch.distributed.get_rank(group) != 0:
            return
        participants, backend = self._managed_async_participants()
        mcore_version = importlib.metadata.version("megatron-core")
        if mcore_version != "0.17.0":
            raise RuntimeError(
                "managed async checkpoint markers require megatron-core==0.17.0, "
                f"got {mcore_version}"
            )
        try:
            authority = create_incomplete_marker(
                path=transaction.path,
                checkpoint_id=transaction.checkpoint_id,
                logical_call_id=transaction.logical_call_id,
                mcore_async_call_index=transaction.expected_call_idx,
                participant_ranks=participants,
                control_group_backend=backend,
                managed_leaves=transaction.marker_leaves,
                managed_leaves_digest=transaction.marker_leaves_digest,
                mcore_version=mcore_version,
            )
        except BaseException as error:
            transaction.marker_authority = getattr(error, "marker_authority", None)
            raise
        transaction.marker_authority = authority
        transaction.marker_created = True

    def _prepare_managed_async_complete(self, transaction) -> None:
        group = self._require_managed_checkpoint_group()
        if torch.distributed.get_rank(group) != 0:
            return
        authority = transaction.marker_authority
        if authority is None:
            raise RuntimeError("managed async checkpoint marker authority is missing")
        prepare_complete_marker(authority)

    def _commit_managed_async_complete(self, transaction) -> None:
        group = self._require_managed_checkpoint_group()
        if torch.distributed.get_rank(group) != 0:
            return
        authority = transaction.marker_authority
        if authority is None:
            raise RuntimeError("managed async checkpoint marker authority is missing")
        outcome = commit_prepared_marker(authority)
        transaction.marker_committed = outcome.committed
        transaction.marker_cleanup_diagnostic = outcome.cleanup_diagnostic
        if outcome.cleanup_pending:
            self._managed_async_marker_cleanup = authority
        transaction.marker_authority = None

    def _retry_managed_async_marker_cleanup(self) -> None:
        if not hasattr(self, "_managed_async_marker_cleanup"):
            return
        authority = getattr(self, "_managed_async_marker_cleanup", None)
        control = create_managed_checkpoint_load_transaction(self.optimizer)

        def retry_local_cleanup():
            if authority is None:
                return
            group = self._require_managed_checkpoint_group()
            if torch.distributed.get_rank(group) != 0:
                return
            if not retry_post_commit_cleanup(authority):
                raise RuntimeError(
                    "managed async marker post-commit cleanup is still pending: "
                    f"{authority.cleanup_diagnostic}"
                )

        _, phase_error = self._run_managed_phase(
            "async_marker_cleanup_retry", retry_local_cleanup, control
        )
        if phase_error is not None:
            raise phase_error from phase_error.local_error
        self._managed_async_marker_cleanup = None

    def _retry_managed_async_marker_precommit_cleanup(self) -> None:
        if not hasattr(self, "_managed_async_marker_precommit_cleanup"):
            return
        authority = getattr(self, "_managed_async_marker_precommit_cleanup", None)
        control = create_managed_checkpoint_load_transaction(self.optimizer)

        def retry_local_cleanup():
            if authority is None:
                return
            group = self._require_managed_checkpoint_group()
            if torch.distributed.get_rank(group) != 0:
                return
            authority.close()

        _, phase_error = self._run_managed_phase(
            "async_marker_precommit_fd_cleanup", retry_local_cleanup, control
        )
        if phase_error is not None:
            raise phase_error from phase_error.local_error
        self._managed_async_marker_precommit_cleanup = None

    def _validate_managed_async_load_marker(
        self,
        path: str,
        *,
        marker_leaves: list[dict] | None = None,
        marker_digest: str | None = None,
    ) -> None:
        managed = getattr(self, "managed_checkpoint_enabled", False)
        participants = self._managed_async_participants()[0] if managed else None
        validate_load_marker(
            path=path,
            participant_ranks=participants,
            managed_leaves=marker_leaves,
            managed_leaves_digest=marker_digest,
        )

    def _release_managed_async_callbacks(self, transaction) -> None:
        errors = []
        while transaction.completion_callbacks:
            callback = transaction.completion_callbacks.pop(0)
            try:
                callback()
            except BaseException as error:
                errors.append(error)
        if errors:
            primary = errors[0]
            for error in errors[1:]:
                primary.add_note(f"another async completion callback failed: {error!r}")
            raise primary

    def _record_managed_async_failure(self, transaction, error: BaseException) -> None:
        if transaction.marker_committed:
            transaction.state = ManagedAsyncSaveState.COMPLETE
            self._managed_async_last_state = ManagedAsyncSaveState.COMPLETE
            transaction.marker_cleanup_diagnostic = repr(error)
            if transaction.marker_authority is not None:
                self._managed_async_marker_cleanup = transaction.marker_authority
                transaction.marker_authority = None
            transaction.request = None
            self._managed_async_save = None
            return
        transaction.state = ManagedAsyncSaveState.FAILED
        self._managed_async_last_state = ManagedAsyncSaveState.FAILED
        transaction.error = error
        worker_recovery = (
            get_managed_async_worker_recovery(self._async_queue)
            if self._async_queue is not None
            else None
        )
        transaction.worker_recovery = worker_recovery
        self._managed_async_save_error = error
        recovery_token = transaction.recovery_token
        recovery_required = (
            recovery_token is not None
            and recovery_token.state is ManagedAsyncRecoveryState.RECOVERY_REQUIRED
        )
        if recovery_required:
            # The transaction token is the authority.  A publication is only a
            # reconstructible rank-local payload and may legitimately be absent
            # after an injected write/clear fault.
            transaction.worker_recovery = worker_recovery or recovery_token
            return
        if worker_recovery is not None:
            # The request, optimizer source fence, and AWEX residency lease are
            # still authoritative while any rank retains worker recovery.
            # A later wait/teardown retries the same journal before releasing
            # any of these resources.
            return
        transaction.request = None
        authority = transaction.marker_authority
        if authority is not None:
            try:
                authority.close()
            except BaseException as close_error:
                error.add_note(
                    f"managed async marker directory close failed: {close_error!r}"
                )
                self._managed_async_marker_precommit_cleanup = authority
                transaction.marker_authority = None
            else:
                transaction.marker_authority = None
        fail_managed_async_checkpoint_save(transaction.leaves, error)
        try:
            self._release_managed_async_callbacks(transaction)
        except BaseException as callback_error:
            error.add_note(f"async residency release failed: {callback_error!r}")
        transaction.worker_recovery = None

    @staticmethod
    def _require_managed_async_recovery(transaction) -> ManagedAsyncRecoveryToken:
        token = transaction.recovery_token
        if token is None:
            token = ManagedAsyncRecoveryToken()
            transaction.recovery_token = token
        token.require_recovery()
        return token

    def _abort_managed_async_queue(self) -> None:
        queue = self._async_queue
        if queue is None:
            return
        if self._managed_async_save is not None:
            transaction = self._managed_async_save
            abort_managed_async_calls(
                queue,
                self._require_managed_checkpoint_group(),
                recovery_token=self._require_managed_async_recovery(transaction),
            )
            return
        active_calls = getattr(queue, "async_calls", None)
        if active_calls is None:
            raise RuntimeError("MCore 0.17 async queue structure changed")
        errors = []
        while active_calls:
            active = active_calls.popleft()
            caller = getattr(active, "async_caller", None)
            if caller is None:
                errors.append(RuntimeError("MCore async caller is missing"))
                continue
            try:
                caller.close(abort=True)
            except BaseException as error:
                errors.append(error)
        if errors:
            primary = errors[0]
            for error in errors[1:]:
                primary.add_note(f"another async worker abort failed: {error!r}")
            raise primary

    def _finalize_managed_async_save(self, call_idx: int) -> None:
        transaction = self._managed_async_save
        control = create_managed_checkpoint_load_transaction(self.optimizer)

        binding_error = None
        if transaction is None:
            binding_error = RuntimeError(
                "managed async finalize returned a call index without an active "
                f"transaction: actual={call_idx}"
            )
        elif not (transaction.expected_call_idx == transaction.call_idx == call_idx):
            binding_error = RuntimeError(
                "managed async transaction/call index mismatch before source "
                "release: "
                f"expected={transaction.expected_call_idx}, "
                f"bound={transaction.call_idx}, finalized={call_idx}"
            )
        phase_error = self._vote_managed_phase(
            "async_finalize_binding",
            binding_error,
            control,
            details={
                "expected_call_idx": (
                    transaction.expected_call_idx if transaction is not None else None
                ),
                "bound_call_idx": (
                    transaction.call_idx if transaction is not None else None
                ),
                "finalized_call_idx": call_idx,
            },
            require_consistent_details=True,
        )
        if phase_error is not None:
            if transaction is not None:
                self._record_managed_async_failure(transaction, phase_error)
            else:
                self._managed_async_last_state = ManagedAsyncSaveState.FAILED
                self._managed_async_save_error = phase_error
            raise phase_error from phase_error.local_error
        assert transaction is not None

        local_error = None
        try:
            complete_managed_async_checkpoint_save(transaction.leaves)
        except BaseException as error:
            local_error = error
        phase_error = self._vote_managed_phase(
            "async_source_release", local_error, control
        )
        if phase_error is not None:
            self._record_managed_async_failure(transaction, phase_error)
            raise phase_error from phase_error.local_error

        callback_error = None
        try:
            self._release_managed_async_callbacks(transaction)
        except BaseException as error:
            callback_error = error
        phase_error = self._vote_managed_phase(
            "async_residency_release", callback_error, control
        )
        if phase_error is not None:
            self._record_managed_async_failure(transaction, phase_error)
            raise phase_error from phase_error.local_error

        marker_error = None
        try:
            self._prepare_managed_async_complete(transaction)
        except BaseException as error:
            marker_error = error
        phase_error = self._vote_managed_phase(
            "async_complete_marker_prepare", marker_error, control
        )
        if phase_error is not None:
            self._record_managed_async_failure(transaction, phase_error)
            raise phase_error from phase_error.local_error

        # This unanimous vote is the global decision.  The complete marker is
        # durable and validated, while the incomplete marker still fences
        # readers.  No optimizer rollback is permitted beyond this point.
        transaction.marker_commit_decided = True
        marker_error = None
        try:
            self._commit_managed_async_complete(transaction)
        except BaseException as error:
            marker_error = error
        phase_error = self._vote_managed_phase(
            "async_complete_marker_commit", marker_error, control
        )
        if phase_error is not None:
            self._record_managed_async_failure(transaction, phase_error)
            raise phase_error from phase_error.local_error

        transaction.marker_committed = True
        transaction.state = ManagedAsyncSaveState.COMPLETE
        self._managed_async_last_state = ManagedAsyncSaveState.COMPLETE
        transaction.request = None
        self._managed_async_save = None

    def _raise_if_managed_async_failed(self) -> None:
        error = getattr(self, "_managed_async_save_error", None)
        if error is not None:
            raise RuntimeError(
                "managed async checkpoint failed; optimizer remains fail-closed"
            ) from error

    def wait_for_managed_async_mutation(self) -> None:
        """Reach the public MCore finalize boundary before mutating save sources."""
        self.wait_async_saves()
        self._raise_if_managed_async_failed()

    def load_checkpoint(
        self,
        local_path: str,
        with_model: bool = True,
        with_optimizer: bool = True,
        with_rng: bool = True,
    ):
        dist_checkpoint_path = local_path
        managed_protocol = getattr(self, "managed_checkpoint_enabled", False)
        if not managed_protocol:
            # If a prior save to the same directory is still flushing in the
            # background, block until it finishes so we don't load a
            # half-written checkpoint.
            self.wait_async_saves()
            if local_path is not None:
                assert os.path.exists(local_path), (
                    f"Checkpoint path {local_path} does not exist."
                )
                # An ordinary manager must not mistake a failed managed async
                # directory for a complete MCore checkpoint, even if an
                # earlier DCP finalizer happened to write metadata.json.
                self._validate_managed_async_load_marker(local_path)
            managed_load = create_managed_checkpoint_load_transaction(None)
            self._load_checkpoint_state(
                dist_checkpoint_path,
                with_model=with_model,
                with_optimizer=with_optimizer,
                with_rng=with_rng,
                managed_load=managed_load,
                managed_protocol=False,
            )
            return

        # A prior commit is already authoritative. Release its cleanup-only
        # references before interpreting any new request, so protocol errors
        # can never be routed through rollback recovery.
        self._retry_managed_checkpoint_cleanup()
        self._retry_managed_async_marker_precommit_cleanup()
        self._retry_managed_async_marker_cleanup()
        # A managed async source fence is released only after the complete
        # foreground finalization boundary.  This still precedes request
        # comparison, while preserving the sealed cleanup-first invariant.
        self.wait_async_saves()
        self._raise_if_managed_async_failed()
        recovery_control = getattr(
            self, "_managed_checkpoint_recovery_transaction", None
        ) or create_managed_checkpoint_load_transaction(self.optimizer)
        recovery_required = self._managed_checkpoint_recovery_required(recovery_control)
        recovery_transaction = None
        if recovery_required:
            recovery_transaction = self._retry_managed_checkpoint_recovery(
                required=True
            )
        recovery_active = recovery_transaction is not None
        control_transaction = (
            recovery_transaction
            or create_managed_checkpoint_load_transaction(self.optimizer)
        )
        request_error = self._vote_managed_phase(
            "request",
            None,
            control_transaction,
            details={
                "path": os.path.abspath(local_path) if local_path else None,
                "with_model": with_model,
                "with_optimizer": with_optimizer,
                "with_rng": with_rng,
                "recovery_required": recovery_active,
            },
            require_consistent_details=True,
        )
        if request_error is not None:
            raise request_error
        if recovery_active and not (with_model and with_optimizer and with_rng):
            recovery_request_error = RuntimeError(
                "a poisoned managed checkpoint manager can only recover through "
                "a full model+optimizer+RNG checkpoint load"
            )
            ready_error = self._vote_managed_phase(
                "recovery_request",
                recovery_request_error,
                control_transaction,
            )
            assert ready_error is not None
            raise ready_error from ready_error.local_error
        _, request_capability_error = self._run_managed_phase(
            "request_capability",
            lambda: validate_managed_checkpoint_load_request(
                self.optimizer,
                with_model=with_model,
                with_optimizer=with_optimizer,
            ),
            control_transaction,
        )
        if request_capability_error is not None:
            raise request_capability_error from request_capability_error.local_error
        marker_present, marker_probe_error = self._run_managed_phase(
            "async_load_marker_probe",
            lambda: has_managed_async_marker(local_path),
            control_transaction,
        )
        if marker_probe_error is not None:
            raise marker_probe_error from marker_probe_error.local_error
        if marker_present:
            marker_identities, marker_identity_error = self._run_managed_phase(
                "async_load_marker_identity",
                self._managed_optimizer_identities,
                control_transaction,
            )
            if marker_identity_error is not None:
                raise marker_identity_error from marker_identity_error.local_error
            marker_manifest, marker_manifest_error = self._run_managed_phase(
                "async_load_marker_manifest",
                lambda: self._managed_async_global_leaf_manifest(marker_identities),
                control_transaction,
            )
            if marker_manifest_error is not None:
                raise marker_manifest_error from marker_manifest_error.local_error
            marker_leaves, marker_digest = marker_manifest
            _, marker_error = self._run_managed_phase(
                "async_load_marker",
                lambda: self._validate_managed_async_load_marker(
                    local_path,
                    marker_leaves=marker_leaves,
                    marker_digest=marker_digest,
                ),
                control_transaction,
            )
            if marker_error is not None:
                raise marker_error from marker_error.local_error

        ready_error = self._vote_managed_phase("ready", None, control_transaction)
        if ready_error is not None:
            raise ready_error from ready_error.local_error
        path_error = None
        if local_path is not None and not os.path.exists(local_path):
            path_error = FileNotFoundError(
                f"Checkpoint path {local_path} does not exist."
            )
        path_phase_error = self._vote_managed_phase(
            "path", path_error, control_transaction
        )
        if path_phase_error is not None:
            raise path_phase_error from path_phase_error.local_error

        managed_load, phase_error = self._run_managed_phase(
            "discover",
            lambda: recovery_transaction
            or create_managed_checkpoint_load_transaction(
                self.optimizer if with_optimizer or with_model else None
            ),
            control_transaction,
        )
        if phase_error is not None:
            raise phase_error from phase_error.local_error

        replacement_attempt = recovery_transaction is managed_load

        def cancel_replacement_prebegin(original_error: BaseException) -> None:
            if not replacement_attempt:
                return
            _, cancel_error = self._run_managed_phase(
                "replacement_snapshot_cancel",
                lambda: cancel_managed_checkpoint_replacement_configuration(
                    managed_load
                ),
                managed_load,
            )
            if cancel_error is not None:
                original_error.add_note(str(cancel_error))
                self._managed_checkpoint_poisoned_error = cancel_error

        if replacement_attempt:
            _, phase_error = self._run_managed_phase(
                "replacement_begin",
                lambda: begin_managed_checkpoint_replacement(managed_load),
                managed_load,
            )
            if phase_error is not None:
                cancel_replacement_prebegin(phase_error)
                raise phase_error from phase_error.local_error
        identities = None
        if managed_load.leaves:
            identities, phase_error = self._run_managed_phase(
                "snapshot_identity",
                self._managed_optimizer_identities,
                managed_load,
            )
            if phase_error is not None:
                cancel_replacement_prebegin(phase_error)
                raise phase_error from phase_error.local_error
            _, phase_error = self._run_managed_phase(
                "snapshot_configure",
                lambda: configure_managed_checkpoint_snapshots(
                    self.optimizer,
                    identities,
                    parent=getattr(self, "managed_checkpoint_snapshot_root", None),
                    transaction=managed_load,
                ),
                managed_load,
            )
            if phase_error is not None:
                cancel_replacement_prebegin(phase_error)
                raise phase_error from phase_error.local_error
        managed_mcore_optimizer = with_optimizer and has_managed_mcore_outer_schema(
            self.optimizer
        )
        if managed_mcore_optimizer:

            def build_preflight_template():
                nonlocal identities
                if identities is None:
                    identities = self._managed_optimizer_identities()
                return build_managed_optimizer_outer_template(
                    self.optimizer, identities
                )

            outer_template, phase_error = self._run_managed_phase(
                "preflight_template", build_preflight_template, managed_load
            )
            if phase_error is not None:
                cancel_replacement_prebegin(phase_error)
                raise phase_error from phase_error.local_error

            outer_state, phase_error = self._run_managed_phase(
                "preflight_load",
                lambda: load_dist_checkpointing(
                    sharded_state_dict={"optimizer": outer_template},
                    ckpt_dir=dist_checkpoint_path,
                    strict_managed=True,
                    allowed_unrequested_prefixes=(
                        "model",
                        "lr_scheduler",
                        "rng_state",
                    ),
                    allow_managed_optimizer_tensor_state=True,
                ),
                managed_load,
            )
            if phase_error is not None:
                cancel_replacement_prebegin(phase_error)
                raise phase_error from phase_error.local_error

            _, phase_error = self._run_managed_phase(
                "preflight_validate",
                lambda: validate_managed_optimizer_outer_state(
                    self.optimizer, outer_state["optimizer"], identities
                ),
                managed_load,
            )
            if phase_error is not None:
                cancel_replacement_prebegin(phase_error)
                raise phase_error from phase_error.local_error

        sharded_state_dict, phase_error = self._run_managed_phase(
            "full_template",
            lambda: self._build_checkpoint_load_template(
                with_model=with_model,
                with_optimizer=with_optimizer,
                with_rng=with_rng,
                managed_protocol=True,
            ),
            managed_load,
        )
        if phase_error is not None:
            cancel_replacement_prebegin(phase_error)
            raise phase_error from phase_error.local_error

        if managed_mcore_optimizer:
            local_manifest, phase_error = self._run_managed_phase(
                "source_metadata_template",
                lambda: build_managed_optimizer_tensor_manifest(
                    sharded_state_dict["optimizer"]
                ),
                managed_load,
            )
            if phase_error is not None:
                cancel_replacement_prebegin(phase_error)
                raise phase_error from phase_error.local_error

            def gather_tensor_manifests():
                group = self._require_managed_checkpoint_group()
                manifests = [
                    None for _ in range(torch.distributed.get_world_size(group))
                ]
                torch.distributed.all_gather_object(
                    manifests, local_manifest, group=group
                )
                return merge_managed_optimizer_tensor_manifests(manifests)

            tensor_manifest, phase_error = self._run_managed_phase(
                "source_metadata_manifest",
                gather_tensor_manifests,
                managed_load,
            )
            if phase_error is not None:
                cancel_replacement_prebegin(phase_error)
                raise phase_error from phase_error.local_error

            _, phase_error = self._run_managed_phase(
                "source_metadata_validate",
                lambda: validate_managed_optimizer_source_tensor_metadata(
                    dist_checkpoint_path, tensor_manifest
                ),
                managed_load,
            )
            if phase_error is not None:
                cancel_replacement_prebegin(phase_error)
                raise phase_error from phase_error.local_error

        scheduler_snapshot = None
        rng_snapshot = None

        def snapshot_metadata():
            nonlocal scheduler_snapshot, rng_snapshot
            if with_optimizer and self.lr_scheduler is not None:
                scheduler_snapshot = copy.deepcopy(self.lr_scheduler.state_dict())
            if with_rng:
                rng_snapshot = copy.deepcopy(self.get_rng_state(use_dist_ckpt=False))

        _, phase_error = self._run_managed_phase(
            "snapshot", snapshot_metadata, managed_load
        )
        if phase_error is not None:
            cancel_replacement_prebegin(phase_error)
            raise phase_error from phase_error.local_error

        capacity_reports, phase_error = self._run_managed_phase(
            "snapshot_preflight",
            lambda: preflight_managed_checkpoint_snapshots(managed_load),
            managed_load,
        )
        if phase_error is not None:
            cancel_replacement_prebegin(phase_error)
            raise phase_error from phase_error.local_error

        _, phase_error = self._run_managed_phase(
            "snapshot_capacity",
            lambda: validate_shared_snapshot_capacity(
                capacity_reports, self._require_managed_checkpoint_group()
            ),
            managed_load,
        )
        if phase_error is not None:
            cancel_replacement_prebegin(phase_error)
            raise phase_error from phase_error.local_error

        def materialize_snapshot():
            # From this point the retained transaction owns the replacement
            # load attempt rather than the old rollback-recovery authority.
            managed_load.recovery_owner = False
            apply_begin_managed_checkpoint_load(managed_load)

        _, phase_error = self._run_managed_phase(
            "snapshot_materialize", materialize_snapshot, managed_load
        )
        if phase_error is not None:
            self._abort_managed_checkpoint_load_distributed(
                managed_load,
                phase_error,
                scheduler_snapshot,
                rng_snapshot,
                poison_after_rollback=False,
            )
            raise phase_error from phase_error.local_error

        state_dict, phase_error = self._run_managed_phase(
            "dcp_load",
            lambda: self._load_checkpoint_data(
                sharded_state_dict,
                dist_checkpoint_path,
                with_model=with_model,
                with_optimizer=with_optimizer,
                with_rng=with_rng,
                managed_protocol=True,
            ),
            managed_load,
        )
        if phase_error is not None:
            self._abort_managed_checkpoint_load_distributed(
                managed_load,
                phase_error,
                scheduler_snapshot,
                rng_snapshot,
                poison_after_rollback=with_model,
            )
            raise phase_error from phase_error.local_error

        _, phase_error = self._run_managed_phase(
            "dcp_apply",
            lambda: self._apply_checkpoint_state(
                state_dict,
                dist_checkpoint_path,
                with_model=with_model,
                with_optimizer=with_optimizer,
                with_rng=with_rng,
                managed_load=managed_load,
            ),
            managed_load,
        )
        if phase_error is not None:
            self._abort_managed_checkpoint_load_distributed(
                managed_load,
                phase_error,
                scheduler_snapshot,
                rng_snapshot,
                poison_after_rollback=with_model,
            )
            raise phase_error from phase_error.local_error

        _, phase_error = self._run_managed_phase(
            "local_validate",
            lambda: prepare_managed_checkpoint_load(managed_load),
            managed_load,
        )
        if phase_error is not None:
            self._abort_managed_checkpoint_load_distributed(
                managed_load,
                phase_error,
                scheduler_snapshot,
                rng_snapshot,
                poison_after_rollback=with_model,
            )
            raise phase_error from phase_error.local_error

        _, phase_error = self._run_managed_phase(
            "prepare_commit",
            lambda: prepare_managed_checkpoint_commit(managed_load),
            managed_load,
        )
        if phase_error is not None:
            self._abort_managed_checkpoint_load_distributed(
                managed_load,
                phase_error,
                scheduler_snapshot,
                rng_snapshot,
                poison_after_rollback=with_model,
            )
            raise phase_error from phase_error.local_error

        # The unanimous prepare-commit vote is the global decision. This call
        # only mutates coordinator metadata and cannot invoke leaf code.
        from areal.engine.megatron_utils.gpu_staged_optimizer_checkpoint import (
            decide_managed_checkpoint_commit,
        )

        decide_managed_checkpoint_commit(managed_load)
        # The decision is already authoritative. Publish cleanup ownership
        # before any fallible leaf transition or status vote so transport
        # errors cannot lose the only retry journal.
        self._managed_checkpoint_cleanup_recovery = managed_load
        _, cleanup_error = self._run_managed_phase(
            "cleanup",
            lambda: retry_managed_checkpoint_cleanup(managed_load),
            managed_load,
        )
        if cleanup_error is not None:
            raise cleanup_error from cleanup_error.local_error
        self._managed_checkpoint_cleanup_recovery = None
        if (
            getattr(self, "_managed_checkpoint_recovery_transaction", None)
            is managed_load
        ):
            self._managed_checkpoint_recovery_transaction = None
            self._managed_checkpoint_poisoned_error = None
        self._managed_checkpoint_control_error = None

    def _abort_managed_checkpoint_load_distributed(
        self,
        transaction,
        original_error: BaseException,
        scheduler_snapshot,
        rng_snapshot,
        *,
        poison_after_rollback: bool,
    ) -> None:
        rollback_errors: list[BaseException] = []
        if scheduler_snapshot is not None:
            try:
                self.lr_scheduler.load_state_dict(scheduler_snapshot)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        if rng_snapshot is not None:
            try:
                self.load_rng_states(rng_snapshot)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        try:
            abort_managed_checkpoint_load(transaction, original_error, poison=False)
        except BaseException as rollback_error:
            rollback_errors.append(rollback_error)
        if transaction.poisoned and not rollback_errors:
            diagnostics = "; ".join(transaction.rollback_diagnostics)
            rollback_errors.append(
                RuntimeError(
                    "a managed optimizer leaf rollback did not complete"
                    + (f": {diagnostics}" if diagnostics else "")
                )
            )
        if poison_after_rollback:
            rollback_errors.append(
                RuntimeError(
                    "model state may have changed before optimizer rollback completed"
                )
            )
        local_rollback_error = rollback_errors[0] if rollback_errors else None
        for rollback_error in rollback_errors:
            original_error.add_note(
                f"managed checkpoint rollback failure: {rollback_error!r}"
            )
        abort_phase_error = self._vote_managed_phase(
            "abort", local_rollback_error, transaction
        )
        if abort_phase_error is not None or poison_after_rollback:
            poison_error = abort_phase_error or local_rollback_error or original_error
            poison_managed_checkpoint_transaction(transaction, poison_error)
            self._managed_checkpoint_poisoned_error = poison_error
        if abort_phase_error is not None:
            original_error.add_note(str(abort_phase_error))

    def _retry_managed_checkpoint_cleanup(self) -> None:
        transaction = getattr(self, "_managed_checkpoint_cleanup_recovery", None)
        vote_transaction = transaction or create_managed_checkpoint_load_transaction(
            None
        )
        _, cleanup_error = self._run_managed_phase(
            "cleanup_retry",
            (
                (lambda: retry_managed_checkpoint_cleanup(transaction))
                if transaction is not None
                else (lambda: None)
            ),
            vote_transaction,
        )
        if cleanup_error is not None:
            raise cleanup_error from cleanup_error.local_error
        if transaction is not None and not transaction.cleanup_pending:
            self._managed_checkpoint_cleanup_recovery = None
            if (
                getattr(self, "_managed_checkpoint_recovery_transaction", None)
                is transaction
            ):
                self._managed_checkpoint_recovery_transaction = None
                self._managed_checkpoint_poisoned_error = None
            self._managed_checkpoint_control_error = None

    def _load_checkpoint_state(
        self,
        dist_checkpoint_path: str,
        *,
        with_model: bool,
        with_optimizer: bool,
        with_rng: bool,
        managed_load,
        managed_protocol: bool,
    ) -> None:
        """Load one synchronous DCP state inside the managed rollback boundary."""
        sharded_state_dict = self._build_checkpoint_load_template(
            with_model=with_model,
            with_optimizer=with_optimizer,
            with_rng=with_rng,
            managed_protocol=managed_protocol,
        )
        state_dict = self._load_checkpoint_data(
            sharded_state_dict,
            dist_checkpoint_path,
            with_model=with_model,
            with_optimizer=with_optimizer,
            with_rng=with_rng,
            managed_protocol=managed_protocol,
        )
        self._apply_checkpoint_state(
            state_dict,
            dist_checkpoint_path,
            with_model=with_model,
            with_optimizer=with_optimizer,
            with_rng=with_rng,
            managed_load=managed_load,
        )

    def _build_checkpoint_load_template(
        self,
        *,
        with_model: bool,
        with_optimizer: bool,
        with_rng: bool,
        managed_protocol: bool,
    ):
        return self.generate_state_dict(
            with_model,
            with_optimizer,
            with_rng,
            # Managed slabs already exist; False returns direct slab views and
            # avoids MCore's is_loading self-load/dummy-state path.
            is_loading=not managed_protocol,
            # Building a load template must never enter the save lifecycle,
            # even though the managed MCore path deliberately uses
            # is_loading=False to avoid its mutating dummy-state setup.
            prepare_optimizer_for_save=False,
        )

    def _load_checkpoint_data(
        self,
        sharded_state_dict,
        dist_checkpoint_path: str,
        *,
        with_model: bool,
        with_optimizer: bool,
        with_rng: bool,
        managed_protocol: bool,
    ):
        allowed_prefixes = tuple(
            prefix
            for include, prefixes in (
                (with_model, ("model",)),
                (with_optimizer, ("optimizer", "lr_scheduler")),
                (with_rng, ("rng_state",)),
            )
            if not include
            for prefix in prefixes
        )
        if not with_optimizer:
            allowed_prefixes = (*allowed_prefixes, "chained_")
        return load_dist_checkpointing(
            sharded_state_dict=sharded_state_dict,
            ckpt_dir=dist_checkpoint_path,
            strict_managed=managed_protocol,
            allowed_unrequested_prefixes=allowed_prefixes,
        )

    def _apply_checkpoint_state(
        self,
        state_dict,
        dist_checkpoint_path: str,
        *,
        with_model: bool,
        with_optimizer: bool,
        with_rng: bool,
        managed_load,
    ) -> None:
        if with_model:
            if self.use_dist_checkpointing:
                assert "model" in state_dict or any(
                    f"model{vpp_rank}" in state_dict
                    for vpp_rank in range(len(self.model))
                ), (
                    f"Model state dict not found in {state_dict.keys()}. Please check the checkpoint file {dist_checkpoint_path}."
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
                    f"Loaded sharded model checkpoint from {dist_checkpoint_path}",
                    rank=self.rank,
                )
            else:
                raise NotImplementedError("Please use dist checkpointing!")

            if not with_optimizer:
                apply_managed_optimizer_reset_from_model(managed_load)

        if with_optimizer:
            assert "optimizer" in state_dict, (
                f"Optimizer state dict not found in {state_dict.keys()}. Please check the checkpoint file {dist_checkpoint_path}."
            )
            optimizer_state_dict = state_dict["optimizer"]
            self.optimizer.load_state_dict(optimizer_state_dict)
            log_with_rank(
                f"Loaded optimizer checkpoint from {dist_checkpoint_path}",
                rank=self.rank,
            )
            if self.use_checkpoint_opt_param_scheduler:
                assert "lr_scheduler" in state_dict, (
                    f"LR scheduler state dict not found in {state_dict.keys()}. Please check the checkpoint file "
                    f"{dist_checkpoint_path}."
                )
                lr_scheduler_state_dict = state_dict["lr_scheduler"]
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                    log_with_rank(
                        f"Loaded LR scheduler checkpoint from {dist_checkpoint_path}",
                        rank=self.rank,
                    )

        if with_rng:
            assert "rng_state" in state_dict, (
                f"RNG state dict not found in {state_dict.keys()}. Please check the checkpoint file {dist_checkpoint_path}."
            )
            rng_state = state_dict["rng_state"]
            self.load_rng_states(rng_state)
            log_with_rank(
                f"Loaded RNG states from {dist_checkpoint_path}", rank=self.rank
            )

    def save_checkpoint(
        self,
        local_path: str,
        with_model: bool = True,
        with_optimizer=True,
        with_rng: bool = True,
        *,
        async_completion_callback=None,
    ):
        dist_checkpoint_path = local_path
        managed_protocol = getattr(self, "managed_checkpoint_enabled", False)
        control_transaction = create_managed_checkpoint_load_transaction(None)
        if managed_protocol:
            # Cleanup from an earlier irreversible commit precedes request
            # comparison. A failed retry leaves the new state authoritative.
            self._retry_managed_checkpoint_cleanup()
            self._retry_managed_async_marker_precommit_cleanup()
            self._retry_managed_async_marker_cleanup()
            recovery_control = (
                getattr(self, "_managed_checkpoint_recovery_transaction", None)
                or control_transaction
            )
            recovery_required = self._managed_checkpoint_recovery_required(
                recovery_control
            )
            if recovery_required:
                control_transaction = self._retry_managed_checkpoint_recovery(
                    required=True
                )
            else:
                control_transaction = create_managed_checkpoint_load_transaction(
                    self.optimizer
                )
            request_error = self._vote_managed_phase(
                "save_request",
                None,
                control_transaction,
                details={
                    "path": os.path.abspath(local_path),
                    "with_model": with_model,
                    "with_optimizer": with_optimizer,
                    "with_rng": with_rng,
                },
                require_consistent_details=True,
            )
            if request_error is not None:
                raise request_error
            poison_error = getattr(self, "_managed_checkpoint_poisoned_error", None)
            ready_error = self._vote_managed_phase(
                "save_ready", poison_error, control_transaction
            )
            if ready_error is not None:
                raise ready_error from ready_error.local_error
            self._raise_if_managed_async_failed()
        if not self.use_dist_checkpointing:
            raise NotImplementedError("Please use dist checkpointing!")

        # Reap any previously scheduled async saves that have already finished.
        # Non-blocking: this only finalizes completed background processes and
        # writes their metadata.json. Pending saves remain queued. Managed
        # ranks vote before entering another DCP collective.
        if managed_protocol:
            _, reap_error = self._run_managed_phase(
                "save_async_reap",
                self._reap_finished_async_saves,
                control_transaction,
            )
            if reap_error is not None:
                raise reap_error from reap_error.local_error
        else:
            self._reap_finished_async_saves()

        if (
            managed_protocol
            and with_optimizer
            and callable(
                getattr(
                    self.optimizer,
                    "bind_managed_checkpoint_process_group",
                    None,
                )
            )
        ):
            _, schema_error = self._run_managed_phase(
                "save_schema_bind",
                self._managed_optimizer_identities,
                control_transaction,
            )
            if schema_error is not None:
                raise schema_error from schema_error.local_error

        managed_async = None
        if managed_protocol and self.async_save:
            if self._managed_async_save is not None:
                raise RuntimeError(
                    "only one managed asynchronous checkpoint may be outstanding"
                )
            checkpoint_id, logical_call_id = self._managed_async_checkpoint_id(
                local_path
            )
            assert self._async_queue is not None
            current_call_idx = getattr(self._async_queue, "call_idx", None)
            if isinstance(current_call_idx, bool) or not isinstance(
                current_call_idx, int
            ):
                raise RuntimeError(
                    "MCore 0.17 AsyncCallsQueue.call_idx contract changed"
                )
            expected_call_idx = current_call_idx + 1
            binding_error = self._vote_managed_phase(
                "async_marker_request_binding",
                None,
                control_transaction,
                details={
                    "checkpoint_id": checkpoint_id,
                    "logical_call_id": logical_call_id,
                    "mcore_async_call_index": expected_call_idx,
                },
                require_consistent_details=True,
            )
            if binding_error is not None:
                raise binding_error

            identities, identity_error = self._run_managed_phase(
                "async_marker_identity",
                self._managed_optimizer_identities,
                control_transaction,
            )
            if identity_error is not None:
                raise identity_error from identity_error.local_error
            marker_manifest, manifest_error = self._run_managed_phase(
                "async_marker_manifest",
                lambda: self._managed_async_global_leaf_manifest(identities),
                control_transaction,
            )
            if manifest_error is not None:
                raise manifest_error from manifest_error.local_error
            marker_leaves, marker_digest = marker_manifest

            def begin_async_save():
                leaves = (
                    begin_managed_async_checkpoint_save(
                        self.optimizer,
                        checkpoint_id=checkpoint_id,
                        path=os.path.abspath(local_path),
                        control_group=self._require_managed_checkpoint_group(),
                        wait_fn=self.wait_for_managed_async_mutation,
                        identities=identities,
                    )
                    if with_optimizer
                    else ()
                )
                transaction = ManagedAsyncSaveTransaction(
                    checkpoint_id=checkpoint_id,
                    path=os.path.abspath(local_path),
                    leaves=leaves,
                    control_group=self._require_managed_checkpoint_group(),
                    logical_call_id=logical_call_id,
                    expected_call_idx=expected_call_idx,
                    marker_leaves=marker_leaves,
                    marker_leaves_digest=marker_digest,
                )
                if async_completion_callback is not None:
                    transaction.completion_callbacks.append(async_completion_callback)
                self._managed_async_save = transaction
                return transaction

            managed_async, begin_error = self._run_managed_phase(
                "async_fence", begin_async_save, control_transaction
            )
            if begin_error is not None:
                if managed_async is not None:
                    self._record_managed_async_failure(managed_async, begin_error)
                raise begin_error from begin_error.local_error
            _, marker_error = self._run_managed_phase(
                "async_incomplete_marker",
                lambda: self._create_managed_async_marker(managed_async),
                control_transaction,
            )
            if marker_error is not None:
                self._record_managed_async_failure(managed_async, marker_error)
                raise marker_error from marker_error.local_error

        if managed_protocol:
            _, directory_error = self._run_managed_phase(
                "save_directory",
                lambda: os.makedirs(dist_checkpoint_path, exist_ok=True),
                control_transaction,
            )
            if directory_error is not None:
                if managed_async is not None:
                    self._record_managed_async_failure(managed_async, directory_error)
                raise directory_error from directory_error.local_error
            state_dict, template_error = self._run_managed_phase(
                "save_template",
                lambda: self.generate_state_dict(with_model, with_optimizer, with_rng),
                control_transaction,
            )
            if template_error is not None:
                if managed_async is not None:
                    self._record_managed_async_failure(managed_async, template_error)
                raise template_error from template_error.local_error
            async_save_request, save_error = self._run_managed_phase(
                "save_dcp",
                lambda: save_dist_checkpointing(
                    sharded_state_dict=state_dict,
                    ckpt_path=dist_checkpoint_path,
                    async_save=self.async_save,
                ),
                control_transaction,
            )
            if save_error is not None:
                if managed_async is not None:
                    self._record_managed_async_failure(managed_async, save_error)
                raise save_error from save_error.local_error
        else:
            # Generate state dict for saving
            state_dict = self.generate_state_dict(with_model, with_optimizer, with_rng)
            # Start Async save if enabled
            async_save_request = save_dist_checkpointing(
                sharded_state_dict=state_dict,
                ckpt_path=dist_checkpoint_path,
                async_save=self.async_save,
            )

        if self.async_save:
            # Invariant relies on save_dist_checkpointing using "torch_dist" +
            # FullyParallelSaveStrategyWrapper, both of which support async in
            # current megatron-core. A different strategy could legitimately
            # return None here; revisit if the save strategy changes.
            assert async_save_request is not None, (
                "Megatron returned no AsyncRequest despite async_sharded_save=True."
            )
            assert self._async_queue is not None
            if managed_protocol:
                assert managed_async is not None
                try:
                    preflight_managed_async_finalize(
                        self._async_queue,
                        async_save_request,
                        self._require_managed_checkpoint_group(),
                        expected_call_idx=managed_async.expected_call_idx,
                    )
                except BaseException as error:
                    self._record_managed_async_failure(managed_async, error)
                    raise
                call_idx, schedule_error = self._run_managed_phase(
                    "async_schedule",
                    lambda: self._async_queue.schedule_async_request(
                        async_save_request
                    ),
                    control_transaction,
                )
                if schedule_error is not None:
                    abort_error = None
                    try:
                        self._abort_managed_async_queue()
                    except BaseException as error:
                        abort_error = error
                        schedule_error.add_note(f"async worker abort failed: {error!r}")
                    assert managed_async is not None
                    self._record_managed_async_failure(managed_async, schedule_error)
                    if abort_error is not None:
                        self._managed_checkpoint_control_error = str(abort_error)
                    raise schedule_error from schedule_error.local_error
                mismatch = None
                if call_idx != managed_async.expected_call_idx:
                    mismatch = RuntimeError(
                        "MCore async call index changed between marker creation and "
                        f"schedule: expected={managed_async.expected_call_idx}, "
                        f"actual={call_idx}"
                    )
                binding_error = self._vote_managed_phase(
                    "async_schedule_binding",
                    mismatch,
                    control_transaction,
                    details={
                        "expected_call_idx": managed_async.expected_call_idx,
                        "actual_call_idx": call_idx,
                    },
                    require_consistent_details=True,
                )
                if binding_error is not None:
                    try:
                        self._abort_managed_async_queue()
                    except BaseException as error:
                        binding_error.add_note(f"async worker abort failed: {error!r}")
                    self._record_managed_async_failure(managed_async, binding_error)
                    raise binding_error from binding_error.local_error
                managed_async.request = async_save_request
                managed_async.call_idx = call_idx
                managed_async.state = ManagedAsyncSaveState.SAVE_IN_FLIGHT
                _, bind_error = self._run_managed_phase(
                    "async_bind",
                    lambda: bind_managed_async_checkpoint_request(
                        managed_async.leaves, async_save_request, call_idx
                    ),
                    control_transaction,
                )
                if bind_error is not None:
                    try:
                        self._abort_managed_async_queue()
                    except BaseException as error:
                        bind_error.add_note(f"async worker abort failed: {error!r}")
                    self._record_managed_async_failure(managed_async, bind_error)
                    raise bind_error from bind_error.local_error
            else:
                call_idx = self._async_queue.schedule_async_request(async_save_request)
            # schedule_async_request only completes MCore's preload/fork
            # boundary.  Managed CPU slabs remain fenced until the matching
            # call is returned by foreground finalization below.
            stats_tracker.scalar(
                **{
                    "ckpt/async_save_queue_depth": float(
                        self._async_queue.get_num_unfinalized_calls()
                    ),
                }
            )
            log_with_rank(
                f"Scheduled async checkpoint save #{call_idx} to {local_path} "
                f"(queue_depth={self._async_queue.get_num_unfinalized_calls()})",
                rank=self.rank,
                log_only_rank_0=True,
            )
        else:
            assert async_save_request is None, (
                "Async save request should be None when not using async save."
            )
            torch.distributed.barrier(
                group=(
                    self._require_managed_checkpoint_group()
                    if managed_protocol
                    else None
                )
            )

    def _reap_finished_async_saves(self) -> None:
        """Non-blocking finalize of any background save processes that have finished.

        Must be called collectively on all ranks. Ordinary checkpoints retain
        MCore's native queue; managed checkpoints use the pinned 0.17 adapter
        and its explicit Gloo phase votes. Safe when async save is disabled.
        """
        if getattr(self, "_async_queue", None) is None:
            return
        transaction = self._managed_async_save
        finalized = []
        if transaction is not None:
            recovery_token = self._require_managed_async_recovery(transaction)
            try:
                finalized = finalize_managed_async_calls(
                    self._async_queue,
                    self._require_managed_checkpoint_group(),
                    expected_call_idx=transaction.expected_call_idx,
                    bound_call_idx=transaction.call_idx,
                    blocking=False,
                    timeout_seconds=self.managed_async_finalize_timeout_seconds,
                    recovery_token=recovery_token,
                )
            except BaseException as error:
                self._record_managed_async_failure(transaction, error)
                raise
        else:
            # Ordinary async checkpoints retain MCore's native queue semantics.
            finalized = self._async_queue.maybe_finalize_async_calls(blocking=False)
        for call_idx in finalized:
            self._finalize_managed_async_save(call_idx)
            log_with_rank(
                f"Finalized async checkpoint save #{call_idx}",
                rank=self.rank,
                log_only_rank_0=True,
            )

    def wait_async_saves(self) -> None:
        """Block until every previously scheduled async save has finalized.

        Must be called collectively on all ranks. Call before:
        - loading a checkpoint (so prior saves to the same dir are durable),
        - tearing down process groups,
        - exiting the training process.
        """
        if getattr(self, "_async_queue", None) is None:
            return
        # Do NOT early-return when get_num_unfinalized_calls()==0: that count is
        # rank-local. Both the native ordinary path and the managed adapter are
        # collective protocols and must observe an empty queue consistently.
        pending = self._async_queue.get_num_unfinalized_calls()
        if pending > 0:
            log_with_rank(
                f"Waiting for {pending} pending async checkpoint save(s) to finalize",
                rank=self.rank,
                log_only_rank_0=True,
            )
        transaction = self._managed_async_save
        finalized = []
        if transaction is not None:
            recovery_token = self._require_managed_async_recovery(transaction)
            try:
                finalized = finalize_managed_async_calls(
                    self._async_queue,
                    self._require_managed_checkpoint_group(),
                    expected_call_idx=transaction.expected_call_idx,
                    bound_call_idx=transaction.call_idx,
                    blocking=True,
                    timeout_seconds=self.managed_async_finalize_timeout_seconds,
                    recovery_token=recovery_token,
                )
            except BaseException as error:
                self._record_managed_async_failure(transaction, error)
                raise
        else:
            finalized = self._async_queue.maybe_finalize_async_calls(blocking=True)
        for call_idx in finalized:
            self._finalize_managed_async_save(call_idx)
            log_with_rank(
                f"Finalized async checkpoint save #{call_idx}",
                rank=self.rank,
                log_only_rank_0=True,
            )

    def close(self) -> None:
        """Drain all pending async saves. Idempotent; safe to call multiple times."""
        self.wait_async_saves()
        if getattr(self, "managed_checkpoint_enabled", False):
            self._retry_managed_async_marker_precommit_cleanup()
            self._retry_managed_async_marker_cleanup()
        self._async_queue = None
