# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import gc
import math
import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.nn.functional as dist_F
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.distributed.tensor import DTensor
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    ProcessorMixin,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from areal.api import (
    FinetuneSpec,
    FSDPParallelStrategy,
    InferenceEngine,
    ModelAllocation,
    ParallelStrategy,
    ParamSpec,
    SaveLoadMeta,
    TrainEngine,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.cli_args import OptimizerConfig, PerfTracerConfig, TrainEngineConfig
from areal.api.io_struct import DeviceRuntimeInfo
from areal.engine.core import (
    aggregate_eval_losses,
    compute_total_loss_weight,
    reorder_and_pad_outputs,
)
from areal.engine.core.distributed import (
    init_custom_process_group,
    patch_dist_group_timeout,
    warmup_process_groups,
)
from areal.engine.core.model import (
    disable_dropout_in_model,
    is_gemma3_model,
    is_qwen3_5_model,
    is_qwen3_moe_model,
    is_qwen3_vl_model,
    is_qwen_vl_model,
    is_valid_vision_model,
)
from areal.engine.fsdp_utils import (
    fsdp2_load_full_state_dict,
    get_cosine_schedule_with_warmup,
)
from areal.engine.fsdp_utils.checkpoint import DCPState
from areal.engine.fsdp_utils.grad import fsdp2_clip_grad_norm
from areal.engine.fsdp_utils.optimizer import AnyPrecisionAdamW, PerLayerOptimWrapper
from areal.engine.fsdp_utils.parallel import ParallelHelper, parallelize_model
from areal.infra.dist_rollout import DistRolloutCoordinator
from areal.infra.platforms import current_platform
from areal.models.fsdp.ulysses import (
    set_ulysses_sequence_parallel_group,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
    ulysses_prepare_inputs,
)
from areal.models.transformers.ulyssess_patch import apply_monkey_patch
from areal.models.tree_attn.functional import (
    _gather_packed_tree_logprobs,
    gather_packed_tree_logprobs_entropy,
    gather_packed_tree_vocab_stats,
    merge_packed_tree_results,
)
from areal.models.tree_attn.module import (
    build_tree_attn_kwargs,
    patch_fsdp_for_tree_training,
)
from areal.models.tree_attn.tree import TrieNode, build_packed_tree_batch
from areal.utils import (
    logging,
    name_resolve,
    names,
    perf_tracer,
    pkg_version,
    stats_tracker,
)
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT
from areal.utils.data import (
    MicroBatchItem,
    MicroBatchList,
    amend_position_ids,
    concat_batch,
    is_multi_modal_key,
    pack_tensor_dict,
    pad_mb_list,
    split_batch,
    split_padded_tensor_dict_into_mb_list,
    unsqueeze_mb_list,
)
from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.hf_utils import load_hf_processor_and_tokenizer, load_hf_tokenizer
from areal.utils.network import find_free_ports, format_host_for_url, gethostip
from areal.utils.offload import is_tms_enabled, torch_memory_saver
from areal.utils.perf_tracer import trace_perf, trace_scope
from areal.utils.save_load import get_state_dict_from_repo_id_or_path

if TYPE_CHECKING:
    from areal.api import Scheduler
    from areal.api.cli_args import DPOEngineConfig, PPOActorConfig, PPOCriticConfig


@dataclasses.dataclass
class FSDPTrainContext:
    """Context passed through FSDP forward/backward pipeline.

    Attributes
    ----------
    model_inputs
        The prepared inputs passed to the model (may include Ulysses slicing).
    mb_input
        The original micro-batch dict before any Ulysses transformations.
    pad_length
        Number of padding tokens added for sequence packing.
    ulysses_pad_size
        Extra padding added for Ulysses sequence parallel alignment.
    trie_node
        The root TrieNode for tree training (if applicable).
    """

    model_inputs: dict[str, Any]
    mb_input: dict[str, Any]
    pad_length: int = 0
    ulysses_pad_size: int = 0
    trie_node: TrieNode | None = None

    def to_dict(self) -> dict[str, Any]:
        """Shallow dict conversion (avoids ``dataclasses.asdict`` which would
        recurse into TrieNode and hit ``RecursionError``).
        """
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


@dataclasses.dataclass
class _PendingWeightUpdateBucket:
    handles: list[Any]
    fut: Future[None]
    named_tensors: list[tuple[str, torch.Tensor]]
    stream: torch.cuda.Stream | None = None


_MULTIMODAL_FORWARD_KEYS = ("image_grid_thw", "pixel_values", "video_grid_thw")


def _is_multimodal_payload_key(key: str) -> bool:
    return is_multi_modal_key(key) or key in _MULTIMODAL_FORWARD_KEYS


def _drop_multimodal_payloads(data: dict[str, Any]) -> None:
    for key in list(data.keys()):
        if _is_multimodal_payload_key(key):
            data.pop(key, None)


def _prepare_multimodal_forward_inputs(
    mb: dict[str, Any],
    padded_mb: dict[str, Any],
) -> None:
    """Keep multimodal tensors only in the forward micro-batch.

    ``mb`` is used by loss and bookkeeping callbacks, while ``padded_mb`` is
    passed to the model forward. Large multimodal tensors such as
    ``pixel_values`` should not be duplicated on both sides.
    """

    multi_modal_input = mb.pop("multi_modal_input", None)
    if multi_modal_input is None:
        multi_modal_input = padded_mb.pop("multi_modal_input", None)
    else:
        padded_mb.pop("multi_modal_input", None)

    if multi_modal_input is not None:
        for key in _MULTIMODAL_FORWARD_KEYS:
            values = [item[key] for item in multi_modal_input if key in item]
            if values:
                padded_mb[key] = torch.cat(values, dim=0)

    _drop_multimodal_payloads(mb)


class FSDPEngine(TrainEngine):
    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.optimizer_config = config.optimizer

        self.model: torch.nn.Module
        self.optimizer: torch.optim.Optimizer
        self.tokenizer: PreTrainedTokenizerFast
        self.processor: ProcessorMixin | None = None
        self.model_config: PretrainedConfig
        self._version: int = 0

        self._initialized = False
        self.own_global_group = False
        self._cpu_group: dist.ProcessGroup
        self.weight_update_group_initialized = False
        self.weight_update_group_names: list[str] = []
        self.weight_update_groups: list = []
        self.weight_update_master_addr: str
        self.weight_update_master_port: int

        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )
        self.is_vision_model = is_valid_vision_model(self.model_config.model_type)

        # FSDP-specific initialization
        self.cpu_offload: CPUOffloadPolicy | None = None

        self.rollout_engine: InferenceEngine | None = None
        self.rollout_coordinator: DistRolloutCoordinator | None = None

        self.parallel_helper: ParallelHelper
        self.world_mesh: DeviceMesh

        self.dp_group: dist.ProcessGroup
        self.sp_group: dist.ProcessGroup
        self.mp_group: dist.ProcessGroup

        self.world_size: int
        self.rank: int
        self.dp_head: int
        self.dp_rank: int

        self.is_offload: bool = False
        self._offload_depth: int = 0
        self._per_layer_optim_wrapper: PerLayerOptimWrapper | None = None
        self.enable_tree_training: bool = self.config.enable_tree_training

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        experiment_name: str,
        trial_name: str,
        dp_size: int = 1,
        tp_size: int = 1,
        dtype: str = "bfloat16",
        learning_rate: float | None = None,
        use_lora: bool = False,
        lora_rank: int = 32,
        lora_alpha: int = 16,
        **kwargs,
    ) -> FSDPEngine:
        """Construct an FSDPEngine directly without assembling a TrainEngineConfig.

        Parameters
        ----------
        model : str
            Path to HuggingFace checkpoint or model.
        experiment_name: str
        trial_name: str
        dp_size: int
            Data parallel size, default 1.
        tp_size: int
            Tensor parallel size, default 1.
        dtype : str
            Parameter data type, default 'bfloat16'.
        learning_rate : float | None
            Learning rate. If None, no optimizer is created (inference-only).
        use_lora : bool
            Whether to use LoRA.
        """
        optimizer_config = (
            OptimizerConfig(lr=learning_rate) if learning_rate is not None else None
        )
        config = TrainEngineConfig(
            path=model,
            experiment_name=experiment_name,
            trial_name=trial_name,
            backend=f"fsdp:d{dp_size}t{tp_size}",
            dtype=dtype,
            optimizer=optimizer_config,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            **kwargs,
        )
        return cls(config)

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
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

        # FSDP-specific process group setup
        if parallel_strategy is None:
            if self.config.backend:
                parallel_strategy = ModelAllocation.from_str(
                    self.config.backend
                ).parallel
            else:
                parallel_strategy = ParallelStrategy()

        self.logger = logging.getLogger(f"[FSDPEngine Rank {dist.get_rank()}]")

        parallel_strategy = self._make_parallel_strategy(parallel_strategy)

        self.parallel_helper = ParallelHelper.from_parallel_strategy(parallel_strategy)

        self.logger.info(
            f"Initializing device mesh with parallel dims {str(self.parallel_helper)}."
        )

        self.world_mesh = self.parallel_helper.world_mesh

        self.dp_group = self.world_mesh["dp"].get_group()
        self.sp_group = self.world_mesh["sp"].get_group()

        # Sequence and model parallel group (sp+tp)
        self.mp_group = self.world_mesh["sp_tp"].get_group()

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dp_head = dist.get_process_group_ranks(self.mp_group)[0]
        self.dp_rank = dist.get_rank(self.dp_group)

        self.logger.info(f"Data parallel head {self.dp_head} and rank {self.dp_rank}")

        # Eagerly initialize HCCL/NCCL communicators for the subgroups so
        # that lazy init doesn't race with colocated engines (issue #1099).
        warmup_process_groups(self.dp_group, self.sp_group, self.mp_group)

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec, *args, **kwargs):
        # Initialize distributed enviroments and load model.
        assert addr is None, "FSDPEngine does not support remote initialization."
        assert ft_spec is not None, "FSDPEngine requires FinetuneSpec to initialize."
        if pkg_version.is_version_less("torch", "2.4.0"):
            raise RuntimeError("areal only supports FSDP2, which requires torch>=2.4.0")

        if is_tms_enabled():
            torch_memory_saver.hook_mode = "preload"

        # Create device model
        self._create_device_model()

        if self.enable_tree_training and self.parallel_helper.sp_size > 1:
            raise ValueError(
                "Tree training currently cannot be enabled with sp_size > 1."
            )
        # Monkey patch: replace attention's forward() with Ulysses variant.
        apply_monkey_patch(
            model=self.model,
            ulysses_sp_size=self.parallel_helper.sp_size,
            shard_vision_across_sp=self.config.fsdp.shard_vision_across_sp,
        )
        # Monkey patch: replace attention's forward() with tree attention.
        patch_fsdp_for_tree_training(enable=self.enable_tree_training)

        if self.config.use_lora:
            self._apply_peft_wrapper()

        # sharding_strategy = ShardingStrategy.FULL_SHARD
        # Simple auto wrap policy
        self.cpu_offload = (
            CPUOffloadPolicy() if self.config.fsdp.offload_params else None
        )
        tik = time.perf_counter()

        full_state = None
        need_broadcast = False

        is_llm_cpu_load = (
            self.config.fsdp.memory_efficient_load
            and not self.config.init_from_scratch
            and not self.is_vision_model
        )

        if is_llm_cpu_load or self.config.use_lora:
            need_broadcast = True
            if dist.get_rank() == 0:
                if is_llm_cpu_load:
                    pretrained_state = get_state_dict_from_repo_id_or_path(
                        self.config.path
                    )
                    missing, unexpected = self.model.load_state_dict(
                        pretrained_state, strict=False
                    )
                    if missing:
                        self.logger.warning(
                            f"Missing keys when loading pretrained weights: {missing}"
                        )
                    if unexpected:
                        self.logger.warning(
                            f"Unexpected keys when loading pretrained weights: {unexpected}"
                        )
                    del pretrained_state
                    gc.collect()
                full_state = self.model.state_dict()
            else:
                full_state = {}

        # NOTE: This applies FSDP2 with N-D parallelism (DP+SP+TP)
        parallelize_model(
            self.model,
            config=self.config,
            model_config=self.model_config,
            nd_device_mesh=self.world_mesh,
            parallel_helper=self.parallel_helper,
            cpu_offload=self.cpu_offload,
            wrap_policy=self.config.fsdp.wrap_policy,
        )

        if need_broadcast:
            broadcast_tik = time.perf_counter()
            fsdp2_load_full_state_dict(
                self.model,
                full_state,
                self.cpu_offload,
                tie_word_embeddings=self.model_config.tie_word_embeddings,
            )
            self.logger.info(
                f"Broadcasting model weights took {time.perf_counter() - broadcast_tik:.2f} seconds"
            )

        self.logger.info(
            f"Applying FSDP2 with N-D parallelism for {time.perf_counter() - tik:.2f} seconds"
        )

        self._create_optimizer(ft_spec)

        if self.config.fsdp.per_layer_optim_step:
            if self.optimizer_config.type != "adam":
                raise ValueError(
                    f"per_layer_optim_step only supports 'adam' optimizer, got '{self.optimizer_config.type}'."
                )
            self._per_layer_optim_wrapper = PerLayerOptimWrapper(
                model=self.model,
                optimizer=self.optimizer,
                device_id=self.device,
                prefetch_layers=self.config.fsdp.optim_step_prefetch_layers,
            )

        self._initialized = True

    @property
    def data_parallel_group(self) -> dist.ProcessGroup:
        return self.dp_group

    @property
    def data_parallel_rank(self) -> int:
        return self.dp_rank

    @property
    def data_parallel_world_size(self) -> int:
        return self.parallel_helper.dp_size

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        return self.mp_group

    @property
    def cpu_group(self) -> dist.ProcessGroup:
        assert self._initialized
        return self._cpu_group

    def destroy(self):
        self._initialized = False
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "model"):
            del self.model
        if self._per_layer_optim_wrapper is not None:
            del self._per_layer_optim_wrapper
            self._per_layer_optim_wrapper = None
        gc.collect()
        current_platform.empty_cache()
        gc.collect()
        # NOTE: if `own_global_group` is true, we assume that
        # no communications are needed after `destroy`, so we
        # directly destroy all groups. Otherwise, process group
        # handles still exist and we expect another engine to
        # clean up these groups.
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
                except Exception as e:  # pragma: no cover - best-effort
                    self.logger.warning(
                        f"pre-destroy CPU barrier failed (ignored): {e}"
                    )
            dist.destroy_process_group()
            # Make destroy() idempotent: if the controller calls destroy
            # more than once (e.g. via cleanup hooks), the second call
            # must not try to destroy already-destroyed groups.
            self.own_global_group = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def current_data_parallel_head(self) -> int:
        return self.dp_head

    def is_data_parallel_head(self) -> bool:
        return self.rank == self.dp_head

    def train(self, mode: bool = True):
        assert self.model is not None
        self.model.train(mode=mode)
        return self

    def connect_engine(self, engine: InferenceEngine, meta: WeightUpdateMeta):
        if self.rollout_engine is not None and self.rollout_engine != engine:
            self.logger.warning(
                f"Connected rollout engine changed from {self.rollout_engine} to {engine}."
            )
        self.rollout_engine = engine
        self.rollout_coordinator = DistRolloutCoordinator(
            rollout_engine=engine, train_engine=self
        )

        if meta.type == "xccl" and not self.weight_update_group_initialized:
            self._init_weight_update_from_distributed(meta)
            self.weight_update_group_initialized = True

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
    ) -> list[dict[str, Any]]:
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
        self._check_rollout_engine_connected()
        return self.rollout_coordinator.prepare_batch(
            dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            dynamic_bs=dynamic_bs,
        )

    def update_weights(self, meta: WeightUpdateMeta):
        self._check_rollout_engine_connected()
        with self._offload_aware_context():
            if meta.type == "xccl":
                assert self.weight_update_group_initialized
                self._update_weights_from_distributed(meta)
            elif meta.type == "disk":
                self._update_weights_from_disk(meta)
            else:
                raise ValueError(f"Unknown weight update type {meta.type}")

    def set_version(self, version: int):
        self._version = version

    def get_version(self) -> int:
        return self._version

    def save(self, meta: SaveLoadMeta):
        with self._offload_aware_context():
            if meta.weight_format == "hf":
                self._save_model_to_hf(meta.path, meta.tokenizer, meta.processor)
            elif meta.weight_format == "dcp":
                self._save_to_dcp(meta.path, meta.with_optim)
            else:
                raise ValueError(f"Unknown weight format {meta.weight_format}. ")

            if meta.with_optim and meta.weight_format == "hf":
                self._save_optimizer_state(meta.path)

    def load(self, meta: SaveLoadMeta):
        with self._offload_aware_context():
            if meta.weight_format == "hf":
                self._load_model_from_hf(meta.path)
            elif meta.weight_format == "dcp":
                self._load_from_dcp(meta.path, meta.with_optim)
            else:
                raise ValueError(f"Unknown weight format {meta.weight_format}. ")

            if meta.with_optim and meta.weight_format == "hf":
                self._load_optimizer_state(meta.path)

            # Checkpoint load replaces optimizer state tensor objects, losing
            # pinning and normalization established by PerLayerOptimWrapper.__init__.
            if meta.with_optim and self._per_layer_optim_wrapper is not None:
                self._per_layer_optim_wrapper.refresh_states()

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

    def optimizer_zero_grad(self):
        assert self.optimizer is not None
        self.optimizer.zero_grad()

    def optimizer_step(self):
        assert self.optimizer is not None
        assert self.optimizer_config is not None
        assert self.lr_scheduler is not None

        grad_norm = fsdp2_clip_grad_norm(
            list(self.model.parameters()),
            max_norm=self.optimizer_config.gradient_clipping,
            fsdp_group=self.world_mesh["dp_sp"].get_group(),
            tp_group=self.world_mesh["tp"].get_group(),
            offload_params=self.config.fsdp.offload_params,
        )

        if not math.isfinite(grad_norm):
            self.optimizer_zero_grad()
            update_successful = False
        elif self.config.fsdp.per_layer_optim_step:
            assert self._per_layer_optim_wrapper is not None
            with trace_scope("fsdp_engine.step"):
                self._per_layer_optim_wrapper.step()
            update_successful = True
        else:
            with trace_scope("fsdp_engine.step"):
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
    ) -> None:
        for mb_item in mb_list:
            inputs, ctx = self._prepare_mb_inputs(mb_item)

            # Lazily create tree attention metadata just before forward.
            # The returned dict keys are prefixed with "tree_" to avoid collisions
            # with HuggingFace's own kwargs. The patched _tree_attn_fwd_func in
            # module_fsdp.py reads these keys from the **kwargs that transformers
            # forwards through.
            tree_attn_keys: list[str] = []
            if self.enable_tree_training and ctx.trie_node is not None:
                padded_size = mb_item.padded_to_length
                assert padded_size is not None
                tree_kwargs = build_tree_attn_kwargs(
                    ctx.trie_node, padded_size, self.device
                )
                inputs.update(tree_kwargs)
                tree_attn_keys = list(tree_kwargs.keys())

            with trace_scope("fsdp_engine.forward"):
                outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0)

            # Release tree attention metadata after forward pass
            for key in tree_attn_keys:
                del inputs[key]

            ctx_dict = ctx.to_dict()
            loss = process_output_fn(logits, ctx_dict)

            if not forward_only and loss is not None:
                with trace_scope("fsdp_engine.backward"):
                    loss.backward()

    def train_batch(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        self._ensure_ready()
        self.optimizer_zero_grad()

        input_batched, _ = self._normalize_batch_input(input_)

        # Step 1: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_batched).to(self.device)

        # Step 2: Compute total loss weight
        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, self.dp_group
        )

        # Step 3: Forward-backward using process_output_fn callback
        def process_output(
            logits: torch.Tensor, ctx_dict: dict[str, Any]
        ) -> torch.Tensor:
            ctx = FSDPTrainContext(**ctx_dict)
            return self._compute_logprobs_and_loss(
                logits,
                ctx,
                loss_fn,
                loss_weight_fn,
                total_loss_weight,
                loss_multiplier=self.parallel_helper.dp_size,
            )

        self.forward_backward_batch(mb_list, process_output, forward_only=False)

        # Step 4: Optimizer step
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
        self._ensure_ready()

        input_batched, _ = self._normalize_batch_input(input_)

        # Step 1: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_batched).to(self.device)

        # Step 2: Compute total loss weight
        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, self.dp_group
        )

        # Step 3: Forward using process_output_fn callback, collecting losses
        losses: list[torch.Tensor] = []

        def process_output(
            logits: torch.Tensor, ctx_dict: dict[str, Any]
        ) -> torch.Tensor:
            ctx = FSDPTrainContext(**ctx_dict)
            loss = self._compute_logprobs_and_loss(
                logits,
                ctx,
                loss_fn,
                loss_weight_fn,
                total_loss_weight,
            )
            losses.append(loss.detach())
            return loss

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        # Step 4: Aggregate losses
        return aggregate_eval_losses(losses, self.dp_group)

    @torch.no_grad()
    def forward_batch(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[torch.Tensor]], torch.Tensor] = torch.cat,
    ) -> torch.Tensor | list[torch.Tensor]:
        self._ensure_ready()

        input_batched, meta = self._normalize_batch_input(input_)

        # Step 1: Prepare sequence lengths
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

        # Step 2: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_batched).to(self.device)

        # Step 3: Forward using process_output_fn callback, collecting results
        outputs: list[torch.Tensor] = []

        def process_output(logits: torch.Tensor, ctx_dict: dict[str, Any]) -> None:
            ctx = FSDPTrainContext(**ctx_dict)
            result = self._compute_forward_result(logits, ctx)
            outputs.append(result)
            return None

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        # Step 4: Aggregate and reorder outputs
        if self.enable_tree_training:
            result = merge_packed_tree_results(outputs, batch_size)
        else:
            result = reorder_and_pad_outputs(
                outputs, output_seqlens, mb_list, aggregate_fn
            )

        if meta is None:
            return result
        return split_batch(result, meta)

    def export_stats(self) -> dict[str, float]:
        with self._offload_aware_context():
            return stats_tracker.export_all(
                reduce_group=self.data_parallel_group,
            )

    def offload(self) -> None:
        """Offload model memory to CPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/fsdp_utils/actor.py
        """
        if not is_tms_enabled():
            raise RuntimeError(
                "torch_memory_saver requires `enable_offload=True` in yaml config."
            )

        self.get_device_stats().log("before offload model")

        # Use torch_memory_saver to pause CUDA memory
        current_platform.clear_memory()
        torch_memory_saver.pause()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after offload model")

        self.is_offload = True

    def onload(self) -> None:
        """Onload model memory from CPU back to GPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/fsdp_utils/actor.py
        """

        torch_memory_saver.resume()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after onload model")

        self.is_offload = False

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

    def get_device_stats(self) -> DeviceRuntimeInfo:
        return DeviceRuntimeInfo.get_current()

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        perf_tracer.save(step=step, force=force)

    def config_perf_tracer(
        self, config: PerfTracerConfig, rank: int, role: str
    ) -> None:
        if perf_tracer.is_configured():
            return
        perf_tracer.configure(config, rank=rank, role=role)

    def _make_parallel_strategy(
        self, parallel_strategy: ParallelStrategy
    ) -> FSDPParallelStrategy:
        return FSDPParallelStrategy(
            **dataclasses.asdict(parallel_strategy),
        )

    def _create_llm_actor_or_critic(self):
        dtype = getattr(torch, self.config.dtype)

        if self.config.is_critic:
            model_class = AutoModelForTokenClassification
            model_kwargs = {"num_labels": 1}
        else:
            model_class = AutoModelForCausalLM
            model_kwargs = {}

        common_kwargs = {
            "dtype": dtype,
            "attn_implementation": self.config.attn_impl,
        }
        model_kwargs.update(common_kwargs)

        if self.config.init_from_scratch or self.config.fsdp.memory_efficient_load:
            model = model_class.from_config(
                self.model_config,
                **model_kwargs,
            )
        else:
            model = model_class.from_pretrained(
                pretrained_model_name_or_path=self.config.path,
                trust_remote_code=True,
                **model_kwargs,
            )
        return model

    def _create_device_model(self):
        current_platform.set_device(int(os.environ["LOCAL_RANK"]))
        current_platform.set_numa_affinity(int(os.environ["LOCAL_RANK"]))
        if current_platform.device_type == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(int(os.environ["LOCAL_RANK"]))

        dtype = getattr(torch, self.config.dtype)

        if self.config.fsdp.memory_efficient_load:
            # Only rank 0 loads on CPU; other ranks use meta device (zero memory)
            # to avoid CPU OOM when multiple workers share a node.
            # Weights are broadcast from rank 0 after FSDP sharding in initialize().
            # Note: meta device optimization only applies to LLM (not VLM), because
            # VLM uses from_pretrained() which doesn't support meta device context.
            if not self.is_vision_model and dist.get_rank() != 0:
                loading_device = "meta"
            else:
                loading_device = "cpu"
        else:
            loading_device = current_platform.device_type

        self.get_device_stats().log("before model creation/loading")

        if self.is_vision_model:
            if dtype == torch.float16:
                raise ValueError(
                    "Vision models do not support float16 dtype. Please use bfloat16."
                )
            if self.config.init_from_scratch:
                raise ValueError(
                    "Vision models do not support initialization from scratch. Please use a pretrained model."
                )
            self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
                self.config.path
            )

            tik = time.perf_counter()
            # VLM: Use from_pretrained() on loading_device.
            with torch.device(loading_device):
                model = AutoModelForImageTextToText.from_pretrained(
                    pretrained_model_name_or_path=self.config.path,
                    trust_remote_code=True,
                    dtype=dtype,
                    attn_implementation=self.config.attn_impl,
                )
                if self.config.disable_dropout:
                    disable_dropout_in_model(model)
        else:
            self.tokenizer = load_hf_tokenizer(self.config.path)
            self.processor = None
            tik = time.perf_counter()
            # LLM: Decide between from_config() or from_pretrained() based on memory_efficient_load
            with torch.device(loading_device):
                model = self._create_llm_actor_or_critic()
                if self.config.disable_dropout:
                    disable_dropout_in_model(model)

        self.get_device_stats().log("after model creation/loading")

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if self.config.use_kernels:
            model.use_kernels = True
        self.logger.info(
            f"Model creation and loading time: {time.perf_counter() - tik:.2f}s"
        )
        self.model = model

    def _apply_peft_wrapper(self):
        config = self.config
        if not config.target_modules or config.target_modules == ["all-linear"]:
            target_modules = "all-linear"
        else:
            target_modules = config.target_modules
        peft_config = {
            "task_type": TaskType.CAUSAL_LM,
            "r": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "target_modules": target_modules,
            "bias": "none",
        }
        if self.config.peft_type == "lora":
            peft_config = LoraConfig(**peft_config)
        else:
            raise NotImplementedError()

        self.model.enable_input_require_grads()
        self.model = get_peft_model(
            self.model,
            peft_config,
            autocast_adapter_dtype=False,
        )

        if self.rank == 0:
            self.model.print_trainable_parameters()

    def _create_optimizer(self, ft_spec: FinetuneSpec) -> None:
        if self.optimizer_config is None:
            return
        assert self.model is not None
        # Set up optimizer
        tik = time.perf_counter()
        assert self.optimizer_config.type in [
            "adam",
            "adam_bf16",
            "sgd",
        ], "Only adam/adam_bf16/sgd optimizer is supported in this engine."
        if self.optimizer_config.type in ["sgd", "adam_bf16"]:
            self.logger.warning(
                f"Using the '{self.optimizer_config.type}' optimizer with FSDP may be less stable. Consider using the 'adam' (AdamW) optimizer for improved stability and performance."
            )
        lr = self.optimizer_config.lr
        weight_decay = self.optimizer_config.weight_decay
        beta1 = self.optimizer_config.beta1
        beta2 = self.optimizer_config.beta2
        eps = self.optimizer_config.eps
        if self.optimizer_config.type == "adam":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                eps=eps,
                # VLM with tensor parallelism is incompatible with fused AdamW
                fused=not (self.is_vision_model and self.parallel_helper.tp_enabled),
            )
        elif self.optimizer_config.type == "adam_bf16":
            self.optimizer = AnyPrecisionAdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                eps=eps,
                momentum_dtype="bfloat16",
                variance_dtype="bfloat16",
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        total_train_steps = ft_spec.total_train_steps
        num_warmup_steps = int(
            self.optimizer_config.warmup_steps_proportion * total_train_steps
        )

        if self.optimizer_config.lr_scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps,
                total_train_steps,
                min_lr_ratio=self.optimizer_config.min_lr_ratio,
            )
        elif self.optimizer_config.lr_scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps,
                total_train_steps,
            )
        elif self.optimizer_config.lr_scheduler_type == "constant":
            self.lr_scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps,
            )
        else:
            raise ValueError(
                f"Unknown lr scheduler type {self.optimizer_config.lr_scheduler_type}"
            )
        self.logger.info(f"Create optimizer time: {time.perf_counter() - tik}")

    def _check_rollout_engine_connected(self) -> None:
        """Validate that rollout engine has been connected via connect_engine()."""
        if self.rollout_engine is None or self.rollout_coordinator is None:
            raise RuntimeError(
                "Rollout engine not connected. Call connect_engine()"
                " before using rollout/update_weight methods."
            )

    def _ensure_ready(self) -> None:
        if self.is_offload:
            self.onload()

        if self.parallel_helper.sp_size > 1:
            set_ulysses_sequence_parallel_group(self.sp_group)

    @staticmethod
    def _normalize_batch_input(
        input_: list[dict[str, Any]] | dict[str, Any],
    ) -> tuple[dict[str, Any], Any | None]:
        """Normalize list/dict batch input to a single batched dict.

        Returns ``(batched_input, meta)`` where ``meta`` is non-None only when
        input is list-based and can be used to split forward outputs back into
        per-trajectory results.
        """
        if isinstance(input_, list):
            return concat_batch(input_)
        return input_, None

    def _get_model_name_parameters(
        self, meta: WeightUpdateMeta
    ) -> Iterator[tuple[str, nn.Parameter]]:
        name_params_iterator = self.model.named_parameters()
        if self.is_vision_model and is_qwen_vl_model(self.model_config.model_type):
            for name, value in name_params_iterator:
                if meta.gen_allocation.backend == "sglang":
                    # SGLang 0.5.9 branch
                    # LLM part: "model.language_model.norm.weight" -> "model.norm.weight"
                    # Vision part: "model.visual.blocks.5.mlp.gate_proj.weight" -> "visual.blocks.5.mlp.gate_proj.weight"
                    new_name = name.replace("language_model.", "", 1)
                    if new_name.startswith("model.visual."):
                        new_name = new_name.replace("model.", "", 1)
                    yield new_name, value
                    continue
                # vLLM 0.17.0 branch
                new_name = name.replace("model.", "", 1)
                if new_name.startswith("language_model."):
                    new_name = new_name.replace(
                        "language_model.", "language_model.model.", 1
                    )
                elif new_name.startswith("lm_head."):
                    new_name = f"language_model.{new_name}"
                yield new_name, value
        elif self.is_vision_model and is_gemma3_model(self.model_config.model_type):
            for name, value in name_params_iterator:
                new_name = name.replace("model.", "", 1)
                if new_name.startswith("language_model."):
                    new_name = new_name.replace(
                        "language_model.", "language_model.model.", 1
                    )
                elif new_name.startswith("lm_head."):
                    new_name = new_name.replace(
                        "lm_head.", "language_model.lm_head.", 1
                    )
                yield new_name, value
        else:
            yield from name_params_iterator

    def _get_full_tensor(self, param: nn.Parameter) -> torch.Tensor:
        """Get full tensor from a parameter, handling DTensor and CPU offloaded tensors."""
        tensor = param.data
        if isinstance(tensor, DTensor):
            # For non-offloaded DTensor, directly call full_tensor()
            if tensor.device.type != "cpu":
                return tensor.full_tensor()

            # Handle CPU offloaded DTensor: reconstruct DTensor from local tensor
            temp_dtensor = DTensor.from_local(
                tensor.to_local(),
                device_mesh=tensor.device_mesh,
                placements=tensor.placements,
            )
            return temp_dtensor.full_tensor()
        else:
            if tensor.device.type == "cpu":
                tensor = tensor.to(current_platform.device_type)
            return tensor

    def _update_bucket_weights_from_distributed_async(
        self,
        meta: WeightUpdateMeta,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
        stream: torch.cuda.Stream | None = None,
    ) -> _PendingWeightUpdateBucket | None:
        # Early exit when chunk size is relatively small
        if not named_tensors:
            return None

        param_specs = [
            ParamSpec(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).split("torch.")[1],
            )
            for name, tensor in named_tensors
        ]

        if self.config.use_lora:
            if not self.config.target_modules or self.config.target_modules == [
                "all-linear"
            ]:
                target_modules = "all-linear"
            else:
                target_modules = self.config.target_modules

            meta.peft_config = {
                "r": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "target_modules": target_modules,
                "bias": "none",
            }

        if len(self.weight_update_groups) > 1:
            # Per-PP-rank groups: broadcast to each group sequentially.
            # Each group's NCCL broadcast must complete before the next
            # starts, because sglang's PP event loop processes requests
            # serially within each PP rank.
            for group_name, group in zip(
                self.weight_update_group_names, self.weight_update_groups
            ):
                pp_meta = copy.copy(meta)
                pp_meta.nccl_group_name = group_name
                fut = self.rollout_engine.update_weights_from_distributed(
                    pp_meta, param_specs
                )
                for _, tensor in named_tensors:
                    dist.broadcast(tensor, src=0, group=group, async_op=False)
                fut.result()
            # Return a no-op bucket (all work is already done).
            _done_fut: Future = Future()
            _done_fut.set_result(None)
            return _PendingWeightUpdateBucket(
                handles=[],
                fut=_done_fut,
                named_tensors=named_tensors,
                stream=stream,
            )

        # Single group: original async path
        fut = self.rollout_engine.update_weights_from_distributed(meta, param_specs)

        handles = []
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream())
            context = torch.cuda.stream(stream)
        else:
            context = nullcontext()

        with context:
            for _, tensor in named_tensors:
                handles.append(
                    dist.broadcast(
                        tensor,
                        src=0,
                        group=self.weight_update_groups[0],
                        async_op=True,
                    )
                )

        return _PendingWeightUpdateBucket(
            handles=handles,
            fut=fut,
            named_tensors=named_tensors,
            stream=stream,
        )

    def _wait_pending_weight_update_bucket(
        self, pending_bucket: _PendingWeightUpdateBucket | None
    ):
        if pending_bucket is None:
            return

        for handle in pending_bucket.handles:
            handle.wait()

        pending_bucket.fut.result()
        pending_bucket.named_tensors.clear()

    def _update_bucket_weights_from_distributed(
        self,
        meta: WeightUpdateMeta,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ):
        pending_bucket = self._update_bucket_weights_from_distributed_async(
            meta, named_tensors
        )
        self._wait_pending_weight_update_bucket(pending_bucket)

    def _init_weight_update_from_distributed(self, meta: WeightUpdateMeta):
        assert meta.type == "xccl"
        gen_pp_size = meta.gen_allocation.parallel.pp_size if meta.gen_allocation else 1

        # NOTE: Processes launched with torchrun will set the following env var
        # to True, which blocks creating another TCP store for weight update.
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)

        if gen_pp_size > 1:
            # When the inference side uses PP > 1, we MUST create per-PP-rank
            # NCCL groups (one per PP stage).  sglang's PP scheduler event loop
            # processes requests at PP rank 0 BEFORE forwarding them to PP
            # rank 1.  If a single NCCL group spans all PP ranks, the TCP
            # rendezvous at PP rank 0 blocks forever because PP rank 1 never
            # receives the init request -> deadlock.  Per-PP-rank groups avoid
            # this: each group only requires the trainer + one PP stage's TP
            # workers, so the rendezvous can complete within one PP stage.
            self.logger.info(
                f"gen_pp_size={gen_pp_size} > 1: creating per-PP-rank "
                f"weight update groups to avoid PP event-loop deadlock."
            )
            self._init_per_pp_weight_update_groups(meta, gen_pp_size)
        else:
            # PP == 1: single group spanning all inference workers.
            group_name = "update_weight_group"
            self.weight_update_group_names = [group_name]

            meta.nccl_master_address = self.weight_update_master_addr = gethostip()
            meta.nccl_master_port = self.weight_update_master_port = find_free_ports(1)[
                0
            ]
            meta.nccl_group_name = group_name

            if dist.get_rank() == 0:
                assert meta.gen_allocation is not None
                gen_world_size = meta.gen_allocation.parallel.world_size

                fut = self.rollout_engine.init_weights_update_group(meta)

                init_method = (
                    f"tcp://{format_host_for_url(meta.nccl_master_address)}"
                    f":{meta.nccl_master_port}"
                )
                self.logger.info(
                    f"Initializing weight update group: type={meta.type} "
                    f"init_method={init_method} group={group_name} "
                    f"world_size={gen_world_size + 1}"
                )
                pg = init_custom_process_group(
                    backend=current_platform.communication_backend,
                    world_size=gen_world_size + 1,
                    init_method=init_method,
                    rank=0,
                    group_name=group_name,
                    timeout=DIST_GROUP_DEFAULT_TIMEOUT,
                )
                self.weight_update_groups = [pg]
                self.weight_update_group = pg

                fut.result()
                self.logger.info(f"Weight update group '{group_name}' initialized.")

    def _init_per_pp_weight_update_groups(
        self, meta: WeightUpdateMeta, gen_pp_size: int
    ):
        """Create one NCCL weight-update group per inference PP stage.

        The group name carries a PP-rank suffix (e.g. ``update_weight_group_0``)
        which the inference side uses (Scenario 2 in ``sglang_remote.py``) to
        route the request to the correct PP stage's workers.
        """
        assert meta.gen_allocation is not None
        gen_world_size = meta.gen_allocation.parallel.world_size
        per_pp_world_size = gen_world_size // gen_pp_size

        self.weight_update_group_names = []
        self.weight_update_groups = []

        if dist.get_rank() == 0:
            for pp_rank in range(gen_pp_size):
                group_name = f"update_weight_group_{pp_rank}"
                self.weight_update_group_names.append(group_name)

                pp_meta = copy.copy(meta)
                pp_meta.nccl_master_address = gethostip()
                pp_meta.nccl_master_port = find_free_ports(1)[0]
                pp_meta.nccl_group_name = group_name

                if pp_rank == 0:
                    self.weight_update_master_addr = pp_meta.nccl_master_address
                    self.weight_update_master_port = pp_meta.nccl_master_port

                self.logger.info(
                    f"Initializing per-PP weight update group: "
                    f"pp_rank={pp_rank} group={group_name} "
                    f"world_size={per_pp_world_size + 1}"
                )

                fut = self.rollout_engine.init_weights_update_group(pp_meta)

                init_method = (
                    f"tcp://{format_host_for_url(pp_meta.nccl_master_address)}"
                    f":{pp_meta.nccl_master_port}"
                )
                pg = init_custom_process_group(
                    backend=current_platform.communication_backend,
                    world_size=per_pp_world_size + 1,
                    init_method=init_method,
                    rank=0,
                    group_name=group_name,
                    timeout=DIST_GROUP_DEFAULT_TIMEOUT,
                )
                self.weight_update_groups.append(pg)

                fut.result()
                self.logger.info(
                    f"Per-PP weight update group '{group_name}' "
                    f"(pp_rank={pp_rank}) initialized."
                )

            # Backward compat: set single-group attribute to first group
            self.weight_update_group = self.weight_update_groups[0]
        else:
            # Non rank-0 FSDP ranks do not participate in NCCL groups
            for pp_rank in range(gen_pp_size):
                self.weight_update_group_names.append(f"update_weight_group_{pp_rank}")
            self.weight_update_master_addr = ""
            self.weight_update_master_port = 0

    @trace_perf("fsdp_engine.update_weights_from_distributed", category="comm")
    def _update_weights_from_distributed(self, meta: WeightUpdateMeta):
        """Broadcast parameters with single-pending-bucket pipelining."""

        # Reset weight weight meta with local info
        meta.nccl_master_address = self.weight_update_master_addr
        meta.nccl_master_port = self.weight_update_master_port
        meta.nccl_group_name = (
            self.weight_update_group_names[0]
            if self.weight_update_group_names
            else "update_weight_group"
        )

        main_rank = dist.get_rank() == 0
        if main_rank:
            self.rollout_engine.pause_generation()

        dist.barrier(group=self.cpu_group)

        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024
        broadcast_stream = None

        if (
            main_rank
            and current_platform.device_type == "cuda"
            and torch.cuda.is_available()
        ):
            broadcast_stream = torch.cuda.Stream()

        buffer_size = 0
        named_tensors: list[tuple[str, torch.Tensor]] = []
        pending_bucket: _PendingWeightUpdateBucket | None = None

        if self.config.use_lora:
            # For LoRA, only iterate over trainable LoRA parameters
            param_iterator = (
                (name, param)
                for name, param in self._get_model_name_parameters(meta)
                if param.requires_grad
            )
        else:
            # For full model, iterate over all parameters
            param_iterator = self._get_model_name_parameters(meta)

        try:
            for name, param in param_iterator:
                # Ranks other than 0 only help to get the full tensor
                tensor = self._get_full_tensor(param)
                if not main_rank:
                    continue

                tensor_size = tensor.numel() * tensor.element_size()
                bucket_overflow = (
                    buffer_size > 0
                    and tensor_size + buffer_size > weight_chunked_mem_size
                )
                if bucket_overflow:
                    # Only middle buckets need drain+align before the next all-gather.
                    if pending_bucket is not None:
                        self._wait_pending_weight_update_bucket(pending_bucket)
                        pending_bucket = None

                    pending_bucket = self._update_bucket_weights_from_distributed_async(
                        meta,
                        named_tensors,
                        stream=broadcast_stream,
                    )

                    named_tensors = []
                    buffer_size = 0

                buffer_size += tensor_size
                named_tensors.append((name, tensor))

            if pending_bucket:
                self._wait_pending_weight_update_bucket(pending_bucket)
                pending_bucket = None

            # Process remaining parameters
            if buffer_size > 0:
                self._update_bucket_weights_from_distributed(meta, named_tensors)
        finally:
            if main_rank and pending_bucket is not None:
                self._wait_pending_weight_update_bucket(pending_bucket)
                pending_bucket = None

        dist.barrier(group=self.cpu_group)
        if main_rank:
            self.rollout_engine.continue_generation()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    @trace_perf("fsdp_engine.update_weights_from_disk", category="io")
    def _update_weights_from_disk(self, meta: WeightUpdateMeta):
        fut = Future()

        if dist.get_rank() == 0:
            self.rollout_engine.pause_generation()
            fut = self.rollout_engine.update_weights_from_disk(meta)

        assert meta.path is not None
        self._save_model_to_hf(meta.path, self.tokenizer, self.processor)
        # dist.barrier() are called when _save_model_to_hf finished

        if dist.get_rank() == 0:
            update_name = names.update_weights_from_disk(
                self.config.experiment_name,
                self.config.trial_name,
                self.get_version(),
            )
            name_resolve.add(
                update_name, str(datetime.now().timestamp()), keepalive_ttl=120
            )

            fut.result()
            self.rollout_engine.continue_generation()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def _save_model_to_hf(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerFast | None,
        processor: ProcessorMixin | None,
    ):
        """Save model in HuggingFace format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        os.makedirs(path, exist_ok=True)

        # FSDP2 checkpoint saving
        # Get full state dict with FSDP2
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(self.model, options=options)

        # save huggingface model on rank 0
        if dist.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.model_config.save_pretrained(path)
            if tokenizer is not None and not self.config.use_lora:
                tokenizer.save_pretrained(path)
            if processor is not None and not self.config.use_lora:
                processor.save_pretrained(path)
        dist.barrier(group=self.cpu_group)

    def _load_model_from_hf(self, path: str):
        """Load model from HuggingFace format."""
        if dist.get_rank() == 0:
            full_state = get_state_dict_from_repo_id_or_path(path)
        else:
            full_state = {}

        fsdp2_load_full_state_dict(
            self.model,
            full_state,
            self.cpu_offload,
            tie_word_embeddings=self.model_config.tie_word_embeddings,
        )

    def _save_to_dcp(
        self,
        path: str,
        with_optim: bool,
    ):
        """Save model in PyTorch Distributed Checkpoint (DCP) format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        os.makedirs(path, exist_ok=True)

        dcp_state = DCPState(self.model, self.optimizer if with_optim else None)
        state_dict = {"dcp": dcp_state}
        dcp.save(state_dict, checkpoint_id=path)

    def _load_from_dcp(self, path: str, with_optim: bool):
        """Load model from Distributed Checkpoint (DCP) format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        dcp_state = DCPState(self.model, self.optimizer if with_optim else None)
        state_dict = {"dcp": dcp_state}
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=path,
        )

    def _save_optimizer_state(self, path: str):
        assert self.optimizer is not None
        assert dist.is_initialized()
        rank = dist.get_rank()
        shard_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"
        )
        state_dict = self.optimizer.state_dict()
        torch.save(state_dict, shard_path)
        dist.barrier(group=self.cpu_group)

    def _load_optimizer_state(self, path: str):
        assert self.optimizer is not None
        assert dist.is_initialized()
        rank = dist.get_rank()
        shard_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"
        )
        optimizer_state_dict = torch.load(shard_path, weights_only=False)
        self.optimizer.load_state_dict(optimizer_state_dict)
        dist.barrier(group=self.cpu_group)

    def _prepare_mb_list(self, input_: dict[str, Any]) -> MicroBatchList:
        assert "attention_mask" in input_ and "input_ids" in input_
        input_ = input_.copy()

        # Tree training path
        if self.enable_tree_training:
            mb_list = build_packed_tree_batch(
                input_,
                mb_spec=self.config.mb_spec,
                pad_to_maximum=self.config.pad_to_maximum,
                dp_group=self.data_parallel_group,
                parallel_size=self.parallel_helper.tp_size
                * self.parallel_helper.sp_size,
            )
            self.logger.info(
                f"Packed tree #microbatch: {len(mb_list)}, microbatch #tokens: {mb_list.group_lens}, "
                f"padded to: {mb_list.padded_to_lengths}, padding lengths: {mb_list.padding_lengths}."
            )
            return mb_list

        if is_qwen_vl_model(self.model_config.model_type):
            attn_mask = input_["attention_mask"]
            input_ids = input_["input_ids"]
            # NOTE: Qwen-VL get_rope_index performs indexed assignment where
            # source positions are int64 and position_ids inherits input_ids.dtype.
            # Ensure input_ids uses int64 so destination/source dtypes align and
            # avoid "Index put requires the source and destination dtypes match".
            if input_ids.dtype != torch.long:
                input_ids = input_ids.to(torch.long)
                input_["input_ids"] = input_ids
            image_grid_thw = None
            video_grid_thw = None
            if "multi_modal_input" in input_:
                multi_modal_input = input_["multi_modal_input"]
                image_grid_thw_list = [
                    m["image_grid_thw"]
                    for m in multi_modal_input
                    if "image_grid_thw" in m
                ]
                if image_grid_thw_list:
                    image_grid_thw = torch.cat(image_grid_thw_list)
                video_grid_thw_list = [
                    m["video_grid_thw"]
                    for m in multi_modal_input
                    if "video_grid_thw" in m
                ]
                if video_grid_thw_list:
                    video_grid_thw = torch.cat(video_grid_thw_list)

            position_ids = self.model.model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attn_mask,
                mm_token_type_ids=input_["mm_token_type_ids"],
                inputs_embeds=None,
                past_key_values=None,
            )
            position_ids = torch.einsum("ijk->jki", position_ids)
            input_["position_ids"] = position_ids
        else:
            input_ = amend_position_ids(input_)

        mb_list = split_padded_tensor_dict_into_mb_list(input_, self.config.mb_spec)
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=self.config.pad_to_maximum,
        )
        self.logger.info(
            f"Microbatch #tokens (rank {dist.get_rank()}): {mb_list.group_lens}, "
            f"padded to: {mb_list.padded_to_lengths}, padding lengths: {mb_list.padding_lengths}"
        )
        mb_list = unsqueeze_mb_list(mb_list)
        if is_qwen_vl_model(self.model_config.model_type):
            assert mb_list.padded_mbs is not None
            for mb in mb_list.padded_mbs:
                mb["position_ids"] = torch.einsum("ijk->kij", mb["position_ids"])

        assert mb_list.padded_mbs is not None
        for i, mb in enumerate(mb_list.mbs):
            mb_list.mbs[i] = dict(**mb)
        for i, mb in enumerate(mb_list.padded_mbs):
            mb_list.padded_mbs[i] = dict(**mb)
        for mb, padded_mb in zip(mb_list.mbs, mb_list.padded_mbs):
            mb["max_length_q"] = mb["max_length_k"] = mb["max_seqlen"] = int(
                mb["max_seqlen"]
            )
            padded_mb["max_length_q"] = padded_mb["max_length_k"] = padded_mb[
                "max_seqlen"
            ] = int(padded_mb["max_seqlen"])
            mb["cu_seq_lens_q"] = mb["cu_seq_lens_k"] = mb["cu_seqlens"]
            padded_mb["cu_seq_lens_q"] = padded_mb["cu_seq_lens_k"] = padded_mb[
                "cu_seqlens"
            ]
            mb["use_cache"] = False
            padded_mb["use_cache"] = False
            if (
                is_qwen3_moe_model(self.model_config.model_type)
                or is_qwen3_vl_model(self.model_config.model_type)
                or is_qwen3_5_model(self.model_config.model_type)
            ):
                mb["attention_mask"] = None
                padded_mb["attention_mask"] = None
            else:
                mb["attention_mask"] = dict(full_attention=None, sliding_attention=None)
                padded_mb["attention_mask"] = dict(
                    full_attention=None, sliding_attention=None
                )
            _prepare_multimodal_forward_inputs(mb, padded_mb)
        _drop_multimodal_payloads(mb_list.data)
        return mb_list

    def _prepare_mb_inputs(
        self, mb_item: MicroBatchItem
    ) -> tuple[dict[str, Any], FSDPTrainContext]:
        """Prepare micro-batch inputs with Ulysses sequence parallel handling.

        This method handles Ulysses SP padding and slicing, returning both
        the prepared model inputs and a context object for later processing.
        """
        trie_node = None
        if self.parallel_helper.sp_size > 1:
            input_ids = mb_item.padded_mb["input_ids"]
            position_ids = mb_item.padded_mb.get("position_ids", None)

            if self.is_vision_model:
                (
                    ulysses_input_ids,
                    ulysses_position_ids,
                    ulysses_pad_size,
                ) = ulysses_pad(
                    input_ids, position_ids, sp_size=self.parallel_helper.sp_size
                )
            else:
                # Pad and slice the inputs
                (
                    ulysses_input_ids,
                    ulysses_position_ids,
                    ulysses_pad_size,
                ) = ulysses_pad_and_slice_inputs(
                    input_ids,
                    position_ids,
                    sp_size=self.parallel_helper.sp_size,
                )
            if (
                ulysses_position_ids is not None
                and not ulysses_position_ids.is_contiguous()
            ):
                ulysses_position_ids = ulysses_position_ids.contiguous()

            inputs = ulysses_prepare_inputs(
                mb_item.padded_mb,
                ulysses_input_ids,
                ulysses_position_ids,
                self.parallel_helper.sp_size,
            )
        else:
            inputs = mb_item.padded_mb
            trie_node = inputs.pop("trie_node", None)
            ulysses_pad_size = 0

        ctx = FSDPTrainContext(
            model_inputs=inputs,
            mb_input=mb_item.orig_mb,
            pad_length=mb_item.padding_length,
            ulysses_pad_size=ulysses_pad_size,
            trie_node=trie_node,
        )
        return inputs, ctx

    def _sp_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        gathered = dist_F.all_gather(tensor, group=self.sp_group)
        return torch.cat(gathered, dim=-1)

    def _get_vocab_min_max_logits(
        self,
        logits: torch.Tensor,
        ulysses_pad_size: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vocab_min_logits = logits.detach().min(-1).values.float()
        vocab_max_logits = logits.detach().max(-1).values.float()
        if self.parallel_helper.sp_size > 1:
            vocab_min_logits = self._sp_all_gather(vocab_min_logits)
            vocab_max_logits = self._sp_all_gather(vocab_max_logits)
            if ulysses_pad_size > 0:
                vocab_min_logits = vocab_min_logits[:-ulysses_pad_size]
                vocab_max_logits = vocab_max_logits[:-ulysses_pad_size]
        return vocab_min_logits, vocab_max_logits

    def _compute_logprobs_entropy(
        self,
        logits: torch.Tensor,
        inputs: dict[str, Any],
        ulysses_pad_size: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Try to get rolled_input_ids (if Ulysses SP is enabled)
        labels = inputs.get(
            "rolled_input_ids",
            torch.roll(inputs["input_ids"], shifts=-1, dims=-1),
        )
        # inputs (padded_mbs) has batch dim (1, seq_len), squeeze to match logits (seq_len,)
        if labels.ndim == 2 and labels.shape[0] == 1:
            labels = labels.squeeze(0)
        logprobs, entropy = gather_logprobs_entropy(
            logits,
            labels,
            temperature=self.config.temperature,
            tp_group=self.parallel_helper.tp_group
            if self.parallel_helper.tp_size > 1
            else None,
        )
        if self.parallel_helper.sp_size > 1:
            logprobs = self._sp_all_gather(logprobs)
            entropy = self._sp_all_gather(entropy)
            if ulysses_pad_size > 0:
                logprobs = logprobs[:-ulysses_pad_size]
                entropy = entropy[:-ulysses_pad_size]
        return logprobs, entropy

    def _compute_logprobs(
        self,
        logits: torch.Tensor,
        inputs: dict[str, Any],
        ulysses_pad_size: int = 0,
    ) -> torch.Tensor:
        # Try to get rolled_input_ids (if Ulysses SP is enabled)
        labels = inputs.get(
            "rolled_input_ids",
            torch.roll(inputs["input_ids"], shifts=-1, dims=-1),
        )
        # inputs (padded_mbs) has batch dim (1, seq_len), squeeze to match logits (seq_len,)
        if labels.ndim == 2 and labels.shape[0] == 1:
            labels = labels.squeeze(0)
        logprobs = gather_logprobs(
            logits,
            labels,
            temperature=self.config.temperature,
            tp_group=self.parallel_helper.tp_group
            if self.parallel_helper.tp_size > 1
            else None,
        )
        if self.parallel_helper.sp_size > 1:
            logprobs = self._sp_all_gather(logprobs)
            if ulysses_pad_size > 0:
                logprobs = logprobs[:-ulysses_pad_size]
        return logprobs

    def _compute_values(
        self,
        values: torch.Tensor,
        ulysses_pad_size: int = 0,
    ) -> torch.Tensor:
        if self.parallel_helper.sp_size > 1:
            values = self._sp_all_gather(values)
            if ulysses_pad_size > 0:
                values = values[:-ulysses_pad_size]
        return values

    def _compute_logprobs_and_loss(
        self,
        logits: torch.Tensor,
        ctx: FSDPTrainContext,
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        total_loss_weight: torch.Tensor,
        loss_multiplier: float = 1.0,
    ) -> torch.Tensor:
        """Compute logprobs/entropy and return scaled loss."""
        local_weight = loss_weight_fn(ctx.mb_input)
        if local_weight == 0:
            return logits.mean() * 0.0

        if self.config.is_critic and self.enable_tree_training:
            raise NotImplementedError(
                "Tree training with critic model is not supported yet."
            )
        if not self.config.is_critic:
            if self.enable_tree_training:
                # Handle dummy trie (empty tree for DP synchronization)
                # When trie has no sequences, return zero loss with grad connection
                if ctx.trie_node is None or not ctx.trie_node.all_sequence_ids:
                    # Return zero loss that maintains gradient connection to logits
                    # This ensures backward() works correctly for FSDP synchronization
                    return logits.mean() * 0.0

                # For tree training, use gather_packed_tree_vocab_stats to properly
                # unpack vocab stats from tree structure back to per-sequence format.
                # This is necessary because the logits are in packed tree format where
                # multiple sequences share prefix positions.
                vocab_min_logits, vocab_max_logits = gather_packed_tree_vocab_stats(
                    logits, ctx.trie_node
                )
                logprobs, entropy = gather_packed_tree_logprobs_entropy(
                    logits,
                    ctx.trie_node,
                    ctx.mb_input["input_ids"],
                    temperature=self.config.temperature,
                    tp_group=self.parallel_helper.tp_group
                    if self.parallel_helper.tp_size > 1
                    else None,
                )
            else:
                logprobs, entropy = self._compute_logprobs_entropy(
                    logits, ctx.model_inputs, ctx.ulysses_pad_size
                )
                vocab_min_logits, vocab_max_logits = self._get_vocab_min_max_logits(
                    logits, ctx.ulysses_pad_size
                )
                if ctx.pad_length > 0:
                    logprobs = logprobs[: -ctx.pad_length]
                    entropy = entropy[: -ctx.pad_length]
                    vocab_min_logits = vocab_min_logits[: -ctx.pad_length]
                    vocab_max_logits = vocab_max_logits[: -ctx.pad_length]
            loss = loss_fn(
                logprobs,
                entropy,
                ctx.mb_input,
                vocab_min_logits=vocab_min_logits,
                vocab_max_logits=vocab_max_logits,
            )
        else:
            values = self._compute_values(logits.squeeze(-1), ctx.ulysses_pad_size)
            if ctx.pad_length > 0:
                values = values[: -ctx.pad_length]
            loss = loss_fn(values, ctx.mb_input)

        loss_scale = local_weight / total_loss_weight * loss_multiplier
        return loss * loss_scale

    def _compute_forward_result(
        self,
        logits: torch.Tensor,
        ctx: FSDPTrainContext,
    ) -> torch.Tensor | dict[int, torch.Tensor]:
        """Compute forward output (logprobs or values)."""
        if self.config.is_critic and self.enable_tree_training:
            raise NotImplementedError(
                "Tree training with critic model is not supported yet."
            )
        if not self.config.is_critic:
            if self.enable_tree_training:
                result = _gather_packed_tree_logprobs(
                    logits,
                    ctx.trie_node,
                    ctx.mb_input["input_ids"],
                    temperature=self.config.temperature,
                    tp_group=self.parallel_helper.tp_group
                    if self.parallel_helper.tp_size > 1
                    else None,
                )
                return result
            result = self._compute_logprobs(
                logits, ctx.model_inputs, ctx.ulysses_pad_size
            )
        else:
            result = self._compute_values(logits.squeeze(-1), ctx.ulysses_pad_size)
        if ctx.pad_length > 0:
            result = result[: -ctx.pad_length]
        return result


# =============================================================================
# Algorithm-specific FSDP Engines
# =============================================================================


class FSDPPPOActor(FSDPEngine):
    """PPO Actor implementation using FSDP backend."""

    def __init__(self, config: PPOActorConfig):
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
    def as_controller(cls, config: PPOActorConfig, scheduler: Scheduler):
        if config._version == "v2":
            from areal.trainer.ppo.actor import PPOActorControllerV2

            return PPOActorControllerV2(
                train_engine=cls,
                config=config,
                scheduler=scheduler,
            )

        from areal.trainer.ppo.actor import PPOActorController

        return PPOActorController(train_engine=cls, config=config, scheduler=scheduler)


class FSDPPPOCritic(FSDPEngine):
    """PPO Critic implementation using FSDP backend."""

    def __init__(self, config: PPOCriticConfig):
        from areal.trainer.ppo.critic import PPOCritic

        super().__init__(config)
        self.critic = PPOCritic(config, self)

    @torch.no_grad()
    def compute_values(self, *args, **kwargs) -> torch.Tensor:
        return self.critic.compute_values(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.critic.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config: PPOCriticConfig, scheduler: Scheduler):
        if config._version == "v2":
            from areal.trainer.ppo.critic import PPOCriticControllerV2

            return PPOCriticControllerV2(
                train_engine=cls,
                config=config,
                scheduler=scheduler,
            )

        from areal.trainer.ppo.critic import PPOCriticController

        return PPOCriticController(train_engine=cls, config=config, scheduler=scheduler)


class FSDPLMEngine(FSDPEngine):
    """Language model engine for SFT using FSDP backend."""

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


class FSDPRWEngine(FSDPEngine):
    """Reward model engine using FSDP backend."""

    def __init__(self, config: TrainEngineConfig):
        from copy import deepcopy

        from areal.trainer.rw.rw_engine import RWEngine

        super().__init__(config)
        self.rw_engine = RWEngine(self)
        if self.config.mb_spec.granularity != 2:
            logger = logging.getLogger("RWEngine")
            logger.warning("mb_spec.granularity must be 2 for reward modeling")
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


class FSDPDPOEngine(FSDPEngine):
    """DPO training engine using FSDP backend."""

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
