# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import gc
import math
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.pipelining.schedules import (
    ScheduleDualPipeV,
    ScheduleZBVZeroBubble,
    get_schedule_class,
)
from transformers import (
    AutoConfig,
    PretrainedConfig,
    PreTrainedTokenizerFast,
)

from areal.api import (
    FinetuneSpec,
    ParallelStrategy,
    SaveLoadMeta,
    TrainEngine,
    WeightUpdateMeta,
)
from areal.api.cli_args import MicroBatchSpec
from areal.api.io_struct import DeviceRuntimeInfo
from areal.engine.core.distributed import (
    patch_dist_group_timeout,
    warmup_process_groups,
)
from areal.engine.core.train_engine import (
    aggregate_eval_losses,
    compute_total_loss_weight,
    reorder_and_pad_outputs,
)
from areal.engine.fsdp_utils.grad import fsdp2_clip_grad_norm
from areal.experimental.engine.archon_checkpoint import (
    load_from_dcp,
    load_model_from_hf,
    load_optimizer_state,
    save_model_to_hf,
    save_optimizer_state,
    save_to_dcp,
)
from areal.experimental.engine.archon_runner import create_runner
from areal.experimental.engine.archon_utils import (
    create_lr_scheduler,
    create_optimizer,
    prepare_training_config,
)
from areal.experimental.engine.archon_weight_sync import (
    WeightSyncState,
    init_weight_update_group,
    update_weights_from_disk,
    update_weights_from_distributed,
)
from areal.experimental.models.archon import (
    ArchonParallelDims,
    BaseStateDictAdapter,
    ModelSpec,
    get_model_spec,
    get_supported_model_types,
    is_supported_model,
)
from areal.experimental.models.archon.activation_checkpoint import (
    ActivationCheckpointConfig,
)
from areal.experimental.models.archon.ulysses import (
    ulysses_gather_output,
    ulysses_slice_inputs,
)
from areal.infra.dist_rollout import DistRolloutCoordinator
from areal.infra.platforms import current_platform
from areal.models.tree_attn.functional import (
    _gather_packed_tree_logprobs,
    gather_packed_tree_logprobs_entropy,
    gather_packed_tree_vocab_stats,
    merge_packed_tree_results,
)
from areal.models.tree_attn.tree import TrieNode, build_packed_tree_batch
from areal.utils import logging, perf_tracer, stats_tracker
from areal.utils.constants import DEFAULT_PAGE_SIZE_BYTES, DIST_GROUP_DEFAULT_TIMEOUT
from areal.utils.data import (
    MicroBatchItem,
    MicroBatchList,
    amend_position_ids,
    broadcast_tensor,
    concat_batch,
    pack_tensor_dict,
    pad_mb_list,
    split_batch,
    split_padded_tensor_dict_into_mb_list,
    unsqueeze_mb_list,
)
from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.lock import DistributedLock
from areal.utils.offload import is_tms_enabled, torch_memory_saver

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.pipelining import PipelineStage
    from torchdata.stateful_dataloader import StatefulDataLoader

    from areal.api import InferenceEngine, Scheduler, WorkflowLike
    from areal.api.cli_args import DPOEngineConfig, PerfTracerConfig, TrainEngineConfig
    from areal.experimental.engine.archon_runner import ForwardBackwardRunner


@dataclass
class ArchonTrainContext:
    """Context passed through Archon forward/backward pipeline.

    Attributes:
        mb_input: Original microbatch input.
        labels: Target token ids for loss computation (rolled from input_ids).
            None when using tree training (labels are computed via trie structure).
        pad_length: Batch-level padding added by pad_mb_list.
        trie_node: The root TrieNode for tree training (if applicable).
    """

    mb_input: dict[str, Any]
    labels: torch.Tensor | None = None
    pad_length: int = 0
    trie_node: TrieNode | None = None

    def to_dict(self) -> dict[str, Any]:
        """Shallow dict conversion (avoids ``dataclasses.asdict`` which would
        recurse into TrieNode and hit ``RecursionError``).
        """
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


class ArchonEngine(TrainEngine):
    """Archon Engine is a torch-native training backend."""

    def __init__(self, config: TrainEngineConfig):
        # Configuration (immutable after init)
        self.config = config
        self.optimizer_config = config.optimizer
        self.enable_tree_training = config.enable_tree_training

        # Model Configuration (loaded during __init__)
        self.model_config: PretrainedConfig = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )
        self._validate_model_type()

        self.spec: ModelSpec = get_model_spec(
            getattr(self.model_config, "model_type", "")
        )

        # Core Components (initialized in initialize())
        self.model: nn.Module
        self.tokenizer: PreTrainedTokenizerFast
        self.optimizer: torch.optim.Optimizer
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler
        self.state_dict_adapter: BaseStateDictAdapter | None = None
        self.runner: ForwardBackwardRunner

        # Distributed / Parallelism (initialized in create_process_group())
        self.rank: int
        self.world_size: int
        self.parallel_dims: ArchonParallelDims
        self._world_mesh: DeviceMesh
        self._cpu_group: dist.ProcessGroup
        self.own_global_group = False

        # Pipeline Parallelism (initialized in initialize())
        self.pp_stages: list[PipelineStage] = []
        self.model_parts: list[nn.Module] = []
        self.pp_has_first_stage: bool = True
        self.pp_has_last_stage: bool = True

        # Rollout / Inference Integration
        self._weight_sync_state: WeightSyncState
        self.engine_lock: DistributedLock
        self.rollout_engine: InferenceEngine | None = None
        self.rollout_coordinator: DistRolloutCoordinator | None = None

        # Runtime State (mutable during training)
        self._version: int = 0
        self._initialized = False
        self.is_offload = False
        self._offload_depth: int = 0

    def create_process_group(
        self,
        parallel_strategy: ParallelStrategy | None = None,
    ):
        patch_dist_group_timeout(DIST_GROUP_DEFAULT_TIMEOUT)

        backend = current_platform.communication_backend
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )
            self.own_global_group = True

        self._cpu_group = dist.new_group(
            timeout=DIST_GROUP_DEFAULT_TIMEOUT, backend="gloo"
        )

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.logger = logging.getLogger(f"[Archon Engine Rank {self.rank}]")

        if parallel_strategy is None:
            parallel_strategy = ParallelStrategy()

        self.parallel_dims = ArchonParallelDims(
            dp_shard=parallel_strategy.data_parallel_size,
            tp=parallel_strategy.tensor_parallel_size,
            cp=parallel_strategy.context_parallel_size,
            pp=parallel_strategy.pipeline_parallel_size,
            ep=parallel_strategy.expert_parallel_size,
            etp=parallel_strategy.expert_tensor_parallel_size,
            world_size=self.world_size,
            device_type=current_platform.device_type,
        )

        self._world_mesh = self.parallel_dims.world_mesh

        # Data parallel rank (for data loading)
        dp_mesh = self.parallel_dims.get_mesh("dp")
        self._dp_rank = dp_mesh.get_local_rank() if dp_mesh is not None else 0

        # Pipeline parallel rank
        if self.parallel_dims.pp_enabled:
            self._pp_rank = self.parallel_dims.get_mesh("pp").get_local_rank()
            # Set in _apply_pipeline_parallelism() after pipeline setup
            self._pp_last_stage_rank = None
        else:
            self._pp_rank = 0
            self._pp_last_stage_rank = None

        # Context and model parallel group (pp_cp_tp)
        self._pp_cp_tp_group = self.parallel_dims.get_group("pp_cp_tp")

        # DP head: the rank that holds the batch for this pp_cp_tp group
        self._dp_head = dist.get_process_group_ranks(self._pp_cp_tp_group)[0]

        # Pipeline parallel head: dp_rank=0 and cp/tp rank=0
        cp_rank_is_zero = (
            not self.parallel_dims.cp_enabled
            or self.parallel_dims.get_mesh("cp").get_local_rank() == 0
        )
        tp_rank_is_zero = (
            not self.parallel_dims.tp_enabled
            or self.parallel_dims.get_mesh("tp").get_local_rank() == 0
        )
        self._is_pipeline_parallel_head = (
            self._dp_rank == 0 and cp_rank_is_zero and tp_rank_is_zero
        )

        # Cached parallel groups (None when the dimension is disabled)
        self._tp_group: dist.ProcessGroup | None = (
            self.parallel_dims.get_group("tp")
            if self.parallel_dims.tp_enabled
            else None
        )
        self._cp_group: dist.ProcessGroup | None = (
            self.parallel_dims.get_group("cp")
            if self.parallel_dims.cp_enabled
            else None
        )

        self.logger.info(
            f"Initialized Archon engine with parallel dims: "
            f"pp={self.parallel_dims.pp}, dp_shard={self.parallel_dims.dp_shard}, "
            f"tp={self.parallel_dims.tp}, cp={self.parallel_dims.cp} (Ulysses SP), "
            f"ep={self.parallel_dims.ep}, etp={self.parallel_dims.etp}"
        )

        # Eagerly initialize HCCL/NCCL communicators for the subgroups so
        # that lazy init doesn't race with colocated engines (issue #1099).
        warmup_process_groups(
            self.parallel_dims.world_mesh["dp"].get_group(),
            self._pp_cp_tp_group,
            self._tp_group,
            self._cp_group,
        )

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec, *args, **kwargs):
        """Initialize model, optimizer, and apply parallelism."""
        assert addr is None, "ArchonEngine does not support remote initialization."
        assert ft_spec is not None, "ArchonEngine requires FinetuneSpec to initialize."

        # Initialize weight sync primitives
        self._weight_sync_state = WeightSyncState(self._pp_rank)
        self.engine_lock = DistributedLock("train_engine_lock")

        if is_tms_enabled():
            torch_memory_saver.hook_mode = "preload"

        self._create_device_model()
        self.state_dict_adapter = self._create_state_dict_adapter()

        self.param_dtype = getattr(torch, self.config.dtype)

        # FP8 conversion -- must run on meta device, before parallelism is applied.
        # This assertion covers the training path (Phase 1A): blockwise FP8 matmuls
        # require BF16 master weights. Loading an FP8 checkpoint into a BF16 model
        # (Phase 1B, archon_checkpoint.py) is a separate path and may relax this.
        if self.config.archon.fp8_config.enabled:
            if self.config.dtype != "bfloat16":
                raise ValueError(
                    f"FP8 training requires dtype=bfloat16 (master weights), "
                    f"got {self.config.dtype}"
                )
            from areal.experimental.models.archon.fp8 import (
                enable_fp8_experts,
                enable_fp8_linear,
            )

            fp8_cfg = self.config.archon.fp8_config
            enable_fp8_linear(
                self.model,
                exclude_fqns=set(fp8_cfg.exclude_modules),
                use_triton=fp8_cfg.use_triton,
            )
            if fp8_cfg.include_experts:
                enable_fp8_experts(self.model, use_triton=fp8_cfg.use_triton)

        # NOTE: may mutate self.config.pad_to_maximum and set env vars
        # (CUBLAS_WORKSPACE_CONFIG, NCCL_ALGO, TORCH_COMPILE_DETERMINISTIC).
        ac_config, enable_compile = prepare_training_config(
            config=self.config,
            parallel_dims=self.parallel_dims,
            model_config=self.model_config,
            enable_tree_training=self.enable_tree_training,
            logger=self.logger,
        )

        tik = time.perf_counter()

        self._setup_parallelism(ac_config, enable_compile)

        # Synchronize all ranks after parallelization (especially after torch.compile)
        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

        self.logger.info(
            f"Applied parallelism in {time.perf_counter() - tik:.2f} seconds"
        )

        if self.config.archon.fp8_config.enabled:
            from areal.experimental.models.archon.fp8 import (
                validate_fp8_shard_alignment,
            )

            parts = self.model_parts if self.parallel_dims.pp_enabled else [self.model]
            validate_fp8_shard_alignment(parts)

        self._materialize_and_load_weights()
        self._create_optimizer(ft_spec)

        self.runner = create_runner(
            pp_enabled=self.parallel_dims.pp_enabled,
            model_parts=self.model_parts,
            prepare_inputs_fn=self._prepare_pipelined_mb_inputs
            if self.parallel_dims.pp_enabled
            else self._prepare_mb_inputs,
            pp_stages=self.pp_stages,
            pp_schedule=self.config.archon.pp_schedule,
            pp_group_size=self.parallel_dims.pp,
            has_first_stage=self.pp_has_first_stage,
            has_last_stage=self.pp_has_last_stage,
        )

        self._initialized = True

    @property
    def world_mesh(self) -> DeviceMesh:
        return self._world_mesh

    @property
    def data_parallel_group(self) -> dist.ProcessGroup:
        return self.parallel_dims.world_mesh["dp"].get_group()

    @property
    def data_parallel_rank(self) -> int:
        return self._dp_rank

    @property
    def data_parallel_world_size(self) -> int:
        return self.parallel_dims.dp_shard

    def current_data_parallel_head(self) -> int:
        return self._dp_head

    def is_data_parallel_head(self) -> bool:
        return self.rank == self._dp_head

    @property
    def pipeline_parallel_rank(self) -> int:
        return self._pp_rank

    def is_pipeline_parallel_head(self) -> bool:
        return self._is_pipeline_parallel_head

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        assert self._pp_cp_tp_group is not None
        return self._pp_cp_tp_group

    @property
    def cpu_group(self) -> dist.ProcessGroup:
        return self._cpu_group

    @property
    def initialized(self) -> bool:
        return self._initialized

    def destroy(self):
        """Clean up resources."""
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "model_parts"):
            self.model_parts.clear()
        gc.collect()
        current_platform.empty_cache()
        gc.collect()

        if dist.is_initialized() and self.own_global_group:
            # Pre-destroy synchronization on a CPU (gloo) group so that all
            # ranks leave the NCCL collective phase together. Without this
            # barrier, rank-0 (which owns the TCPStore server) may exit
            # before peers finish their final NCCL abort, causing
            # HeartbeatMonitor background threads on other ranks to observe
            # "recvValue failed" on the already-closed store. This is
            # harmless but produces a noisy stderr backtrace at teardown.
            if getattr(self, "_cpu_group", None) is not None:
                try:
                    dist.barrier(group=self._cpu_group)
                except Exception as e:
                    self.logger.warning(
                        f"pre-destroy CPU barrier failed (ignored): {e}"
                    )
            dist.destroy_process_group()
            # Make destroy() idempotent: if the controller calls destroy
            # more than once (e.g. via cleanup hooks), the second call
            # must not try to destroy already-destroyed groups.
            self.own_global_group = False
        self._initialized = False

    def train(self, mode: bool = True):
        for m in self.model_parts:
            m.train(mode=mode)
        return self

    def set_version(self, version: int):
        self._version = version

    def get_version(self) -> int:
        return self._version

    def optimizer_zero_grad(self):
        assert self.optimizer is not None
        self.optimizer.zero_grad()

    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        assert self.optimizer is not None
        assert self.optimizer_config is not None
        assert self.lr_scheduler is not None

        grad_norm = fsdp2_clip_grad_norm(
            self._get_all_parameters(),
            max_norm=self.optimizer_config.gradient_clipping,
            fsdp_group=self.data_parallel_group,
            tp_group=self._tp_group,
            pp_group=self.parallel_dims.get_group("pp")
            if self.parallel_dims.pp_enabled
            else None,
            offload_params=self.config.archon.offload_params,
        )

        if not math.isfinite(grad_norm):
            self.optimizer_zero_grad()
            update_successful = False
        else:
            self.optimizer.step()
            update_successful = True

        current_lr = self.lr_scheduler.get_last_lr()[0]
        return dict(
            update_successful=float(update_successful),
            grad_norm=float(grad_norm) if grad_norm is not None else float("nan"),
            lr=current_lr,
        )

    def lr_scheduler_step(self):
        assert self.lr_scheduler is not None
        self.lr_scheduler.step()

    def forward_backward_batch(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool = False,
    ) -> list[torch.Tensor] | None:
        """Forward and optionally backward through micro-batches."""
        return self.runner.run(mb_list, process_output_fn, forward_only)

    def train_batch(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        """Train on a batch of data."""
        assert self._initialized
        self.optimizer_zero_grad()

        input_batched, _ = self._normalize_batch_input(input_)

        mb_list = self._prepare_mb_list(input_batched).to(self.device)

        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, self.data_parallel_group
        )

        def process_output(
            logits: torch.Tensor, ctx_dict: dict[str, Any]
        ) -> torch.Tensor:
            ctx = ArchonTrainContext(**ctx_dict)
            return self._compute_logprobs_and_loss(
                logits,
                ctx,
                loss_fn,
                loss_weight_fn,
                total_loss_weight,
                loss_multiplier=self.data_parallel_world_size,
            )

        self.forward_backward_batch(mb_list, process_output, forward_only=False)

        stats = self.optimizer_step()
        stats["num_micro_batches"] = len(mb_list.mbs)
        return stats

    @torch.no_grad()
    def eval_batch(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        """Evaluate on a batch of data."""
        assert self._initialized

        input_batched, _ = self._normalize_batch_input(input_)

        mb_list = self._prepare_mb_list(input_batched).to(self.device)

        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, self.data_parallel_group
        )

        def process_output(
            logits: torch.Tensor, ctx_dict: dict[str, Any]
        ) -> torch.Tensor:
            ctx = ArchonTrainContext(**ctx_dict)
            return self._compute_logprobs_and_loss(
                logits,
                ctx,
                loss_fn,
                loss_weight_fn,
                total_loss_weight,
            )

        losses = self.forward_backward_batch(mb_list, process_output, forward_only=True)

        return aggregate_eval_losses(
            losses if self.pp_has_last_stage else None,
            self.data_parallel_group,
            self.pp_has_last_stage,
            self.parallel_dims.get_group("pp")
            if self.parallel_dims.pp_enabled
            else None,
            self._pp_last_stage_rank,
        )

    @torch.no_grad()
    def forward_batch(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[torch.Tensor]], torch.Tensor] = torch.cat,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Forward pass without gradient computation."""
        assert self._initialized

        input_batched, meta = self._normalize_batch_input(input_)

        if meta is not None:
            assert isinstance(input_, list)
            inferred_seqlens = [d["attention_mask"].shape[-1] for d in input_]
            if output_seqlens is not None and output_seqlens != inferred_seqlens:
                raise ValueError(
                    f"output_seqlens mismatch for list input: "
                    f"given {output_seqlens}, "
                    f"inferred {inferred_seqlens} from attention_mask shapes."
                )
            output_seqlens = inferred_seqlens
        cu_seqlens = pack_tensor_dict(input_batched)["cu_seqlens"]
        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()
        assert output_seqlens is not None
        batch_size = len(output_seqlens)

        mb_list = self._prepare_mb_list(input_batched).to(self.device)

        def process_output(
            logits: torch.Tensor, ctx_dict: dict[str, Any]
        ) -> torch.Tensor:
            ctx = ArchonTrainContext(**ctx_dict)
            return self._compute_forward_result(logits, ctx)

        outputs = self.forward_backward_batch(
            mb_list, process_output, forward_only=True
        )

        if self.pp_has_last_stage:
            assert outputs is not None
            if self.enable_tree_training:
                res = merge_packed_tree_results(outputs, batch_size)
            else:
                res = reorder_and_pad_outputs(
                    outputs, output_seqlens, mb_list, aggregate_fn
                )
        else:
            res = None
        if self.parallel_dims.pp_enabled:
            assert self._pp_last_stage_rank is not None
            res = broadcast_tensor(
                res,
                src_rank=self._pp_last_stage_rank,
                group=self.parallel_dims.get_group("pp"),
            )
        assert res is not None
        if meta is None:
            return res
        return split_batch(res, meta)

    def connect_engine(self, engine: InferenceEngine, meta: WeightUpdateMeta):
        """Connect to an inference engine for rollout."""
        if self.rollout_engine is not None and self.rollout_engine != engine:
            self.logger.warning(
                f"Connected rollout engine changed from {self.rollout_engine} to {engine}."
            )
        self.rollout_engine = engine
        self.rollout_coordinator = DistRolloutCoordinator(
            rollout_engine=engine, train_engine=self
        )

        if meta.type == "xccl" and not self._weight_sync_state.group_initialized:
            init_weight_update_group(
                state=self._weight_sync_state,
                meta=meta,
                engine=self,
            )

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
    ) -> list[dict[str, Any]]:
        """Perform rollout using connected inference engine."""
        self._check_rollout_engine_connected()
        return self.rollout_coordinator.rollout_batch(
            data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            group_size=group_size,
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> list[dict[str, Any]]:
        """Prepare batch from dataloader with rollout."""
        self._check_rollout_engine_connected()
        return self.rollout_coordinator.prepare_batch(
            dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            dynamic_bs=dynamic_bs,
        )

    def clear_batches(self, shard_ids: list[str]) -> None:
        """Drain this worker's client-side RTensor fetch buffer.

        Called via RPC by ``TrainController.clear_batches`` at step end so
        cross-node consumer DP heads release cached tensors. See #1209.
        Upstream ``TrainController.clear_batches`` guards against empty
        input, so ``shard_ids`` is always a non-empty ``list[str]``.
        """
        from areal.infra.rpc.rtensor import clear_fetch_buffer

        clear_fetch_buffer(shard_ids)

    def fetch_buffer_stats(self) -> dict[str, int]:
        """Expose local fetch-buffer stats for post-step drain verification."""
        from areal.infra.rpc.rtensor import fetch_buffer_stats

        return fetch_buffer_stats()

    def update_weights(self, meta: WeightUpdateMeta):
        """Update weights to inference engine."""
        self._check_rollout_engine_connected()
        with self._offload_aware_context():
            if meta.type == "xccl":
                assert self._weight_sync_state.group_initialized
                update_weights_from_distributed(
                    state=self._weight_sync_state,
                    meta=meta,
                    engine=self,
                )
            elif meta.type == "disk":
                update_weights_from_disk(
                    meta=meta,
                    engine=self,
                )
            else:
                raise ValueError(f"Unknown weight update type {meta.type}")

    def save(self, meta: SaveLoadMeta):
        """Save model in HuggingFace or DCP format."""
        with self._offload_aware_context():
            if meta.weight_format == "hf":
                save_model_to_hf(self, meta.path, meta.tokenizer, meta.processor)
            elif meta.weight_format == "dcp":
                save_to_dcp(self, meta.path, meta.with_optim)
            else:
                raise ValueError(f"Unknown weight format {meta.weight_format}.")

            if meta.with_optim and meta.weight_format == "hf":
                save_optimizer_state(self, meta.path)

    def load(self, meta: SaveLoadMeta):
        """Load model from HuggingFace or DCP format."""
        with self._offload_aware_context():
            if meta.weight_format == "hf":
                load_model_from_hf(self, meta.path)
            elif meta.weight_format == "dcp":
                load_from_dcp(self, meta.path, meta.with_optim)
            else:
                raise ValueError(f"Unknown weight format {meta.weight_format}.")

            if meta.with_optim and meta.weight_format == "hf":
                load_optimizer_state(self, meta.path)

    @contextmanager
    def _offload_aware_context(self):
        """Temporarily onload parameters for offload-unsafe operations.

        Reentrant: nested calls increment depth; only the outermost
        call performs actual onload/offload transitions.
        """
        if not self.is_offload:
            yield
            return

        self._offload_depth += 1
        if self._offload_depth == 1:
            self.onload()
        try:
            yield
        finally:
            self._offload_depth -= 1
            if self._offload_depth == 0:
                self.offload()

    def offload(self) -> None:
        """Offload model memory to CPU using torch_memory_saver."""
        if not is_tms_enabled():
            raise RuntimeError(
                "torch_memory_saver requires `enable_offload=True` in yaml config."
            )

        self.get_device_stats().log("before offload model")

        current_platform.clear_memory()
        torch_memory_saver.pause()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after offload model")

        self.is_offload = True

    def onload(self) -> None:
        """Onload model memory from CPU back to GPU using torch_memory_saver."""
        torch_memory_saver.resume()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after onload model")

        self.is_offload = False

    def export_stats(self) -> dict[str, float]:
        assert self._initialized
        with self._offload_aware_context():
            data = stats_tracker.export_all(
                reduce_group=self.data_parallel_group,
            )
        if self.parallel_dims.pp_enabled:
            data_list = [data]
            dist.broadcast_object_list(
                data_list,
                src=self._pp_last_stage_rank,
                group=self.parallel_dims.get_group("pp"),
            )
            data.update(data_list[0])
        return data

    def get_device_stats(self) -> DeviceRuntimeInfo:
        return DeviceRuntimeInfo.get_current()

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        perf_tracer.save(step=step, force=force)

    def config_perf_tracer(
        self, config: PerfTracerConfig, rank: int, role: str
    ) -> None:
        perf_tracer.configure(config, rank=rank, role=role)

    # =========================================================================
    # Internal methods
    # =========================================================================

    def _check_rollout_engine_connected(self) -> None:
        if self.rollout_engine is None or self.rollout_coordinator is None:
            raise RuntimeError(
                "Rollout engine not connected. Call connect_engine()"
                " before using rollout/update_weight methods."
            )

    def _validate_model_type(self) -> None:
        model_type = getattr(self.model_config, "model_type", "")
        if not is_supported_model(model_type):
            supported = ", ".join(sorted(get_supported_model_types()))
            raise ValueError(
                f"Archon Engine does not support model type '{model_type}'. "
                f"Supported model types: {supported}. "
                f"Please use FSDPEngine for unsupported models."
            )

    def _setup_parallelism(
        self,
        ac_config: ActivationCheckpointConfig | None,
        enable_compile: bool,
    ) -> None:
        if self.parallel_dims.pp_enabled:
            self._apply_pipeline_parallelism(ac_config, enable_compile)
        else:
            self._apply_parallelism(ac_config, enable_compile)

    def _apply_pipeline_parallelism(
        self,
        ac_config: ActivationCheckpointConfig | None,
        enable_compile: bool,
    ) -> None:
        """Apply pipeline parallelism using pipelining_fn."""
        if self.spec.pipelining_fn is None:
            raise RuntimeError(
                f"Pipeline Parallel is enabled but {self.spec.name} "
                f"does not support pipelining"
            )

        (
            self.pp_stages,
            self.model_parts,
            self.pp_has_first_stage,
            self.pp_has_last_stage,
        ) = self.spec.pipelining_fn(
            model=self.model,
            device=self.device,
            parallel_dims=self.parallel_dims,
            archon_config=self.config.archon,
            parallelize_fn=self.spec.parallelize_fn,
            param_dtype=self.param_dtype,
            reduce_dtype=torch.float32,
            loss_parallel=True,
            cpu_offload=self.config.archon.offload_params,
            reshard_after_forward_policy=self.config.archon.reshard_after_forward_policy,
            ac_config=ac_config,
            enable_compile=enable_compile,
        )

        # Delete original model to free memory
        del self.model

        # Determine which rank holds the last pipeline stage
        pp_ranks = dist.get_process_group_ranks(self.parallel_dims.get_group("pp"))
        if get_schedule_class(self.config.archon.pp_schedule) in (
            ScheduleZBVZeroBubble,
            ScheduleDualPipeV,
        ):
            # V-style: rank 0 holds stages (0, num_stages-1)
            self._pp_last_stage_rank = pp_ranks[0]
        else:
            # Loop-style: last rank has last stage
            self._pp_last_stage_rank = pp_ranks[-1]

        self.logger.info(
            f"PP enabled: has_first={self.pp_has_first_stage}, "
            f"has_last={self.pp_has_last_stage}"
        )

    def _apply_parallelism(
        self,
        ac_config: ActivationCheckpointConfig | None,
        enable_compile: bool,
    ) -> None:
        """Apply parallelism using parallelize_fn."""
        self.spec.parallelize_fn(
            model=self.model,
            parallel_dims=self.parallel_dims,
            param_dtype=self.param_dtype,
            reduce_dtype=torch.float32,
            loss_parallel=True,
            cpu_offload=self.config.archon.offload_params,
            reshard_after_forward_policy=self.config.archon.reshard_after_forward_policy,
            ac_config=ac_config,
            enable_compile=enable_compile,
        )
        self.model_parts = [self.model]

    def _prepare_mb_inputs(
        self, mb_item: MicroBatchItem
    ) -> tuple[dict[str, Any], ArchonTrainContext]:
        inputs = dict(mb_item.padded_mb)

        # Extract trie_node for tree training (if present)
        trie_node = inputs.pop("trie_node", None)

        # Tree training: labels are derived from trie structure, not torch.roll.
        # (Tree input_ids is 1D packed format, so roll would be wrong anyway.)
        if self.enable_tree_training:
            assert trie_node is not None
            ctx = ArchonTrainContext(
                mb_input=mb_item.orig_mb,
                pad_length=mb_item.padding_length,
                trie_node=trie_node,
            )
        else:
            labels = torch.roll(inputs["input_ids"], shifts=-1, dims=-1)

            if self.parallel_dims.cp_enabled:
                cp_mesh = self.parallel_dims.get_mesh("cp")
                inputs, labels = ulysses_slice_inputs(
                    inputs,
                    labels,
                    cp_mesh.get_local_rank(),
                    self.parallel_dims.cp,
                )

            if labels.ndim == 2 and labels.shape[0] == 1:
                labels = labels.squeeze(0)

            ctx = ArchonTrainContext(
                mb_input=mb_item.orig_mb,
                labels=labels,
                pad_length=mb_item.padding_length,
            )
        return inputs, ctx

    def _prepare_pipelined_mb_inputs(
        self,
        mb_list: MicroBatchList,
    ) -> tuple[tuple, dict, torch.Tensor | None, list[ArchonTrainContext]]:
        """Concatenate microbatch inputs for pipeline scheduler's step()/eval() API."""
        input_ids_list: list[torch.Tensor] = []
        positions_list: list[torch.Tensor] = []
        cu_seqlens_list: list[torch.Tensor] = []
        max_seqlen_list: list[int] = []
        target_list: list[torch.Tensor] = []
        contexts: list[ArchonTrainContext] = []

        def ensure_2d(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0) if t.ndim == 1 else t

        for mb_item in mb_list:
            inputs, ctx = self._prepare_mb_inputs(mb_item)
            contexts.append(ctx)

            input_ids_list.append(ensure_2d(inputs["input_ids"]))
            positions_list.append(ensure_2d(inputs["position_ids"]))
            cu_seqlens_list.append(ensure_2d(inputs["cu_seqlens"]))
            max_seqlen_list.append(int(inputs["max_seqlen"]))

            # For tree training, labels are None (computed via trie structure)
            if self.pp_has_last_stage and ctx.labels is not None:
                target_list.append(ensure_2d(ctx.labels))

        # Pad cu_seqlens to same length using last value to create zero-length sequences
        max_cu_len = max(cs.shape[1] for cs in cu_seqlens_list)
        padded_cu_seqlens = [
            torch.cat([cs, cs[:, -1:].expand(-1, max_cu_len - cs.shape[1])], dim=1)
            if cs.shape[1] < max_cu_len
            else cs
            for cs in cu_seqlens_list
        ]

        batched_args = (
            (torch.cat(input_ids_list, dim=0),) if self.pp_has_first_stage else ()
        )
        batched_kwargs = {
            "positions": torch.cat(positions_list, dim=0),
            "cu_seqlens": torch.cat(padded_cu_seqlens, dim=0),
            "max_seqlen": torch.tensor(max_seqlen_list),
        }
        # For tree training, target_list is empty (labels computed via trie)
        batched_target = (
            torch.cat(target_list, dim=0)
            if self.pp_has_last_stage and target_list
            else None
        )

        return batched_args, batched_kwargs, batched_target, contexts

    def _create_state_dict_adapter(self) -> BaseStateDictAdapter | None:
        return self.spec.state_dict_adapter_class(
            self.model_config, hf_assets_path=self.config.path
        )

    def _get_all_parameters(self) -> list[nn.Parameter]:
        return [p for m in self.model_parts for p in m.parameters()]

    def _get_model_name_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for m in self.model_parts:
            yield from m.named_parameters()

    def _create_device_model(self):
        current_platform.set_device(int(os.environ["LOCAL_RANK"]))
        current_platform.set_numa_affinity(int(os.environ["LOCAL_RANK"]))
        if current_platform.device_type == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(int(os.environ["LOCAL_RANK"]))

        self.tokenizer = load_hf_tokenizer(self.config.path)

        tik = time.perf_counter()

        # Meta device mode: create structure only, no memory allocation
        # Parameters exist only as metadata until materialized after FSDP
        with torch.device("meta"):
            model = self._create_model_structure()
        model = model.to(getattr(torch, self.config.dtype))
        self.model = model

        self.logger.info(
            f"Model structure created on meta device in "
            f"{time.perf_counter() - tik:.2f}s"
        )
        self.get_device_stats().log("after create model structure")

    def _create_model_structure(self) -> nn.Module:
        """Create model structure on meta device without loading weights."""
        # Use tree attention type when tree training is enabled
        attn_type = self.config.archon.attn_type
        if self.enable_tree_training:
            if attn_type != "tree":
                self.logger.warning(
                    f"Tree training enabled, overriding attn_type '{self.config.archon.attn_type}' -> 'tree'"
                )
                attn_type = "tree"
        elif attn_type == "tree":
            self.logger.warning(
                "attn_type is 'tree' but tree training is disabled. Overriding to 'varlen'."
            )
            attn_type = "varlen"

        # Map moe_router_dtype string config to torch.dtype; None means no override
        router_dtype = (
            torch.float32 if self.config.archon.moe_router_dtype == "fp32" else None
        )
        model_args = self.spec.model_args_class.from_hf_config(
            self.model_config,
            is_critic=self.config.is_critic,
            attn_type=attn_type,
            router_dtype=router_dtype,
        )
        return self.spec.model_class(model_args)

    def _materialize_and_load_weights(self):
        """Materialize meta tensors and load weights after FSDP parallelization."""
        if self.config.archon.offload_params:
            init_device = "cpu"
            buffer_device = current_platform.device_type
        else:
            init_device = current_platform.device_type
            buffer_device = init_device

        tik = time.perf_counter()

        for model in self.model_parts:
            model.to_empty(device=init_device)

        if not self.config.init_from_scratch:
            load_model_from_hf(self, self.config.path)
        else:
            with torch.no_grad():
                for model in self.model_parts:
                    model.init_weights()

        for model in self.model_parts:
            model.init_buffers(buffer_device=buffer_device)

        dist.barrier(group=self.cpu_group)

        self.logger.info(
            f"Materialized and loaded weights in {time.perf_counter() - tik:.2f}s"
        )
        self.get_device_stats().log("after materialize and load weights")

    def _create_optimizer(self, ft_spec: FinetuneSpec):
        if self.optimizer_config is None:
            return

        tik = time.perf_counter()

        self.optimizer = create_optimizer(
            self._get_all_parameters(), self.optimizer_config
        )
        self.lr_scheduler = create_lr_scheduler(
            self.optimizer, self.optimizer_config, ft_spec.total_train_steps
        )

        self.logger.info(f"Created optimizer in {time.perf_counter() - tik:.2f}s")

    @staticmethod
    def _normalize_batch_input(
        input_: list[dict[str, Any]] | dict[str, Any],
    ) -> tuple[dict[str, Any], Any | None]:
        if isinstance(input_, list):
            return concat_batch(input_)
        return input_, None

    def _prepare_mb_list(self, input_: dict[str, Any]) -> MicroBatchList:
        assert "attention_mask" in input_ and "input_ids" in input_
        input_ = input_.copy()

        # Tree training path
        # Note: CP/PP incompatibility is validated in initialize().
        if self.enable_tree_training:
            mb_list = build_packed_tree_batch(
                input_,
                mb_spec=self.config.mb_spec,
                pad_to_maximum=self.config.pad_to_maximum,
                dp_group=self.data_parallel_group,
                parallel_size=self.parallel_dims.tp,
            )
            self.logger.info(
                f"Packed tree #microbatch: {len(mb_list)}, microbatch #tokens: {mb_list.group_lens}, "
                f"padded to: {mb_list.padded_to_lengths}, padding lengths: {mb_list.padding_lengths}."
            )
            return mb_list

        input_ = amend_position_ids(input_)

        # Pipeline parallelism requires n_microbatches >= num_total_stages
        if self.parallel_dims.pp_enabled:
            pp_size = self.parallel_dims.pp
            stages_per_rank = len(self.pp_stages)
            num_total_stages = pp_size * stages_per_rank
            n_seqs = input_["attention_mask"].shape[0]
            if n_seqs < num_total_stages:
                raise RuntimeError(
                    f"Pipeline parallelism requires at least {num_total_stages} "
                    f"sequences (pp_size={pp_size} * stages_per_rank="
                    f"{stages_per_rank}), but got {n_seqs}. "
                    f"Increase batch size or reduce PP degree/stages."
                )
            min_n_mbs = num_total_stages
            mb_spec = MicroBatchSpec.new(
                self.config.mb_spec,
                n_mbs=max(min_n_mbs, self.config.mb_spec.n_mbs or 1),
                n_mbs_divisor=pp_size,
            )
        else:
            mb_spec = self.config.mb_spec

        mb_list = split_padded_tensor_dict_into_mb_list(input_, mb_spec)
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]

        # LCM ensures page-aligned memory and exact CP slicing without extra padding.
        model_config = (
            self.model_config.text_config
            if hasattr(self.model_config, "text_config")
            else self.model_config
        )
        page_size = max(
            DEFAULT_PAGE_SIZE_BYTES
            // model_config.hidden_size
            // torch.empty([], dtype=self.param_dtype).element_size(),
            1,
        )
        batch_align_to = math.lcm(page_size, self.parallel_dims.seq_len_divisor)
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=self.config.pad_to_maximum,
            batch_align_to=batch_align_to,
        )

        self.logger.info(
            f"Microbatch #tokens (rank {self.rank}): {mb_list.group_lens}, "
            f"padded to: {mb_list.padded_to_lengths}"
        )

        mb_list = unsqueeze_mb_list(mb_list)

        assert mb_list.padded_mbs is not None
        for i, mb in enumerate(mb_list.mbs):
            mb_list.mbs[i] = dict(**mb)
        for i, mb in enumerate(mb_list.padded_mbs):
            mb_list.padded_mbs[i] = dict(**mb)

        return mb_list

    def _compute_logprobs_and_loss(
        self,
        logits: torch.Tensor,
        ctx: ArchonTrainContext,
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        total_loss_weight: torch.Tensor,
        loss_multiplier: float = 1.0,
    ) -> torch.Tensor:
        """Compute logprobs/entropy and return scaled loss."""
        local_weight = loss_weight_fn(ctx.mb_input)
        if local_weight == 0:
            return logits.mean() * 0.0

        if not self.config.is_critic:
            result = self._gather_actor_train_outputs(logits, ctx)
            if result is None:
                return logits.mean() * 0.0
            logprobs, entropy, vocab_min_logits, vocab_max_logits = result
            loss = loss_fn(
                logprobs,
                entropy,
                ctx.mb_input,
                vocab_min_logits=vocab_min_logits,
                vocab_max_logits=vocab_max_logits,
            )
        else:
            values = self._gather_critic_output(logits, ctx)
            loss = loss_fn(values, ctx.mb_input)

        loss_scale = local_weight / total_loss_weight * loss_multiplier
        return loss * loss_scale

    def _compute_forward_result(
        self,
        logits: torch.Tensor,
        ctx: ArchonTrainContext,
    ) -> torch.Tensor | dict[int, torch.Tensor]:
        """Compute forward output (logprobs or values)."""
        if not self.config.is_critic:
            return self._gather_actor_forward_output(logits, ctx)
        return self._gather_critic_output(logits, ctx)

    def _gather_actor_train_outputs(
        self,
        logits: torch.Tensor,
        ctx: ArchonTrainContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Compute (logprobs, entropy, vocab_min, vocab_max) for actor training."""
        if self.enable_tree_training:
            # Handle dummy trie (empty tree for DP synchronization)
            if ctx.trie_node is None or not ctx.trie_node.all_sequence_ids:
                return None
            vocab_min, vocab_max = gather_packed_tree_vocab_stats(logits, ctx.trie_node)
            logprobs, entropy = gather_packed_tree_logprobs_entropy(
                logits,
                ctx.trie_node,
                ctx.mb_input["input_ids"],
                temperature=self.config.temperature,
                tp_group=self._tp_group,
            )
            return logprobs, entropy, vocab_min, vocab_max

        assert ctx.labels is not None
        logprobs, entropy = gather_logprobs_entropy(
            logits,
            ctx.labels,
            temperature=self.config.temperature,
            tp_group=self._tp_group,
        )
        vocab_min, vocab_max = self._get_vocab_min_max_logits(logits)

        if self._cp_group is not None:
            logprobs = ulysses_gather_output(logprobs, self._cp_group)
            entropy = ulysses_gather_output(entropy, self._cp_group)
            vocab_min = ulysses_gather_output(vocab_min, self._cp_group)
            vocab_max = ulysses_gather_output(vocab_max, self._cp_group)

        if ctx.pad_length > 0:
            logprobs = logprobs[: -ctx.pad_length]
            entropy = entropy[: -ctx.pad_length]
            vocab_min = vocab_min[: -ctx.pad_length]
            vocab_max = vocab_max[: -ctx.pad_length]

        return logprobs, entropy, vocab_min, vocab_max

    def _gather_actor_forward_output(
        self,
        logits: torch.Tensor,
        ctx: ArchonTrainContext,
    ) -> torch.Tensor | dict[int, torch.Tensor]:
        """Compute actor logprobs for forward-only path."""
        if self.enable_tree_training:
            assert ctx.trie_node is not None
            return _gather_packed_tree_logprobs(
                logits,
                ctx.trie_node,
                ctx.mb_input["input_ids"],
                temperature=self.config.temperature,
                tp_group=self._tp_group,
            )

        assert ctx.labels is not None
        result = gather_logprobs(
            logits,
            ctx.labels,
            temperature=self.config.temperature,
            tp_group=self._tp_group,
        )

        if self._cp_group is not None:
            result = ulysses_gather_output(result, self._cp_group)
        if ctx.pad_length > 0:
            result = result[: -ctx.pad_length]

        return result

    def _gather_critic_output(
        self,
        logits: torch.Tensor,
        ctx: ArchonTrainContext,
    ) -> torch.Tensor:
        """Compute critic values with CP gather and pad trimming."""
        values = logits.squeeze(-1)

        if self._cp_group is not None:
            values = ulysses_gather_output(values, self._cp_group)
        if ctx.pad_length > 0:
            values = values[: -ctx.pad_length]

        return values

    def _get_vocab_min_max_logits(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get vocab min/max logits for non-tree training path."""
        vocab_min_logits = logits.detach().min(-1).values.float()
        vocab_max_logits = logits.detach().max(-1).values.float()
        return vocab_min_logits, vocab_max_logits


class ArchonPPOActor(ArchonEngine):
    """PPO Actor implementation using Archon backend."""

    def __init__(self, config):
        from areal.trainer.ppo.actor import PPOActor

        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> list[torch.Tensor] | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> list[dict[str, Any]]:
        return self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.actor.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config, scheduler: Scheduler):
        if config._version == "v2":
            from areal.trainer.ppo.actor import PPOActorControllerV2

            return PPOActorControllerV2(
                train_engine=cls,
                config=config,
                scheduler=scheduler,
            )

        from areal.trainer.ppo.actor import PPOActorController

        return PPOActorController(train_engine=cls, config=config, scheduler=scheduler)


class ArchonPPOCritic(ArchonEngine):
    """PPO Critic implementation using Archon backend."""

    def __init__(self, config):
        from areal.trainer.ppo.critic import PPOCritic

        super().__init__(config)
        self.critic = PPOCritic(config, self)

    @torch.no_grad()
    def compute_values(self, *args, **kwargs) -> torch.Tensor:
        return self.critic.compute_values(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.critic.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config, scheduler: Scheduler):
        if config._version == "v2":
            from areal.trainer.ppo.critic import PPOCriticControllerV2

            return PPOCriticControllerV2(
                train_engine=cls,
                config=config,
                scheduler=scheduler,
            )

        from areal.trainer.ppo.critic import PPOCriticController

        return PPOCriticController(train_engine=cls, config=config, scheduler=scheduler)


class ArchonLMEngine(ArchonEngine):
    """Archon-based LM Engine for SFT training."""

    def __init__(self, config: TrainEngineConfig):
        from areal.trainer.sft.lm_engine import LMEngine

        super().__init__(config)
        self.lm_engine = LMEngine(self)

    def train_lm(self, data):
        return self.lm_engine.train_lm(data)

    def evaluate_lm(self, data):
        return self.lm_engine.evaluate_lm(data)

    @classmethod
    def as_controller(cls, config: TrainEngineConfig, scheduler: Scheduler):
        if config._version == "v2":
            from areal.trainer.sft.lm_engine import LMControllerV2

            return LMControllerV2(
                train_engine=cls,
                config=config,
                scheduler=scheduler,
            )

        from areal.trainer.sft.lm_engine import LMController

        return LMController(train_engine=cls, config=config, scheduler=scheduler)


class ArchonRWEngine(ArchonEngine):
    """Archon-based RW Engine for reward modeling."""

    def __init__(self, config: TrainEngineConfig):
        from copy import deepcopy

        from areal.trainer.rw.rw_engine import RWEngine

        super().__init__(config)
        self.rw_engine = RWEngine(self)
        if self.config.mb_spec.granularity != 2:
            rw_logger = logging.getLogger("RWEngine")
            rw_logger.warning("mb_spec.granularity must be 2 for reward modeling")
            self.config = deepcopy(self.config)
            self.config.mb_spec.granularity = 2

    def train_rw(self, data):
        return self.rw_engine.train_rw(data)

    def evaluate_rw(self, data):
        return self.rw_engine.evaluate_rw(data)

    @classmethod
    def as_controller(cls, config: TrainEngineConfig, scheduler: Scheduler):
        if config._version == "v2":
            from areal.trainer.rw.rw_engine import RWControllerV2

            return RWControllerV2(train_engine=cls, config=config, scheduler=scheduler)

        from areal.trainer.rw.rw_engine import RWController

        return RWController(train_engine=cls, config=config, scheduler=scheduler)


class ArchonDPOEngine(ArchonEngine):
    """Archon-based DPO Engine for direct preference optimization."""

    def __init__(self, config: DPOEngineConfig):
        from copy import deepcopy

        from areal.trainer.dpo.dpo_engine import DPOEngine

        super().__init__(config)
        self.dpo_engine = DPOEngine(self)
        if self.config.mb_spec.granularity != 2:
            dpo_logger = logging.getLogger("DPOEngine")
            dpo_logger.warning("mb_spec.granularity must be 2 for DPO training")
            self.config = deepcopy(self.config)
            self.config.mb_spec.granularity = 2

    def train_dpo(self, data):
        return self.dpo_engine.train_dpo(data)

    def evaluate_dpo(self, data):
        return self.dpo_engine.evaluate_dpo(data)

    def compute_logp(self, data: list[dict[str, Any]]) -> list[torch.Tensor] | None:
        return self.dpo_engine.compute_logp(data)

    @classmethod
    def as_controller(
        cls,
        config: DPOEngineConfig,
        scheduler: Scheduler,
    ):
        if config._version == "v2":
            from areal.trainer.dpo.dpo_engine import DPOControllerV2

            return DPOControllerV2(train_engine=cls, config=config, scheduler=scheduler)

        from areal.trainer.dpo.dpo_engine import DPOController

        return DPOController(train_engine=cls, config=config, scheduler=scheduler)
