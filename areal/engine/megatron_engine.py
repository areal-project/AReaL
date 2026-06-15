# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import functools
import gc
import math
import os
import re
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any

import mbridge
import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge as MegatronBridgeAutoBridge
from megatron.bridge.peft.lora import LoRA as MegatronBridgeLoRA
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig as MCoreOptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_model_config
from torch import nn
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PretrainedConfig

import areal.models.mcore.bailing_moe_bridge  # noqa: F401  # register bridge
from areal.api import (
    FinetuneSpec,
    InferenceEngine,
    MegatronParallelStrategy,
    ParallelStrategy,
    ParamSpec,
    SaveLoadMeta,
    TrainEngine,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.cli_args import MicroBatchSpec, PerfTracerConfig, TrainEngineConfig
from areal.api.io_struct import DeviceRuntimeInfo
from areal.engine.core import (
    aggregate_eval_losses,
    compute_total_loss_weight,
    reorder_and_pad_outputs,
)
from areal.engine.core.distributed import (
    init_custom_process_group,
    warmup_process_groups,
)
from areal.engine.core.model import (
    disable_dropout_in_model,
    is_valid_vision_model,
    lang_config,
    requires_padded_seq,
)
from areal.engine.megatron_utils import megatron_bridge_patches  # noqa: F401
from areal.engine.megatron_utils.checkpointer import MegatronCheckpointManager
from areal.engine.megatron_utils.deterministic import set_deterministic_algorithms
from areal.engine.megatron_utils.fp8 import FP8BlockwiseTensorHelper
from areal.engine.megatron_utils.megatron import (
    all_gather_param,
    convert_to_hf,
    get_named_parameters,
    remove_padding,
)
from areal.engine.megatron_utils.megatron_lora import get_vllm_lora_target_modules
from areal.engine.megatron_utils.packed_context_parallel import (
    _is_multi_modal_payload_key,
    extract_vision_from_multi_modal,
    packed_context_parallel_forward,
    reassemble_cp_packed_logprobs,
    split_packed_seqs_for_context_parallel,
)
from areal.engine.megatron_utils.pipeline_parallel import (
    configure_pipeline_layer_splits,
)
from areal.infra.dist_rollout import DistRolloutCoordinator
from areal.infra.platforms import current_platform
from areal.models.mcore.hf_load import load_weights_from_hf_with_mbridge_fast
from areal.models.mcore.hf_save import (
    save_critic_value_head,
    save_weights_to_hf_with_mbridge_fast,
)
from areal.models.mcore.registry import make_hf_and_mcore_config, make_mcore_model
from areal.models.tree_attn.functional import (
    _gather_packed_tree_logprobs,
    gather_packed_tree_logprobs_entropy,
    gather_packed_tree_vocab_stats,
    merge_packed_tree_results,
)
from areal.models.tree_attn.module import (
    build_tree_attn_kwargs,
    patch_bridge_for_tree_training,
)
from areal.models.tree_attn.tree import build_packed_tree_batch
from areal.utils import logging, name_resolve, names, perf_tracer, stats_tracker
from areal.utils.constants import (
    DEFAULT_VECTORIZED_ALIGNMENT_BYTES,
    DIST_GROUP_DEFAULT_TIMEOUT,
)
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
    unpad_logits,
)
from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.hf_utils import load_hf_processor_and_tokenizer, load_hf_tokenizer
from areal.utils.lock import DistributedLock
from areal.utils.network import find_free_ports, format_host_for_url, gethostip
from areal.utils.offload import is_tms_enabled, torch_memory_saver
from areal.utils.perf_tracer import trace_perf, trace_scope
from areal.utils.seeding import get_seed

if TYPE_CHECKING:
    from areal.api import Scheduler
    from areal.api.cli_args import DPOEngineConfig, PPOActorConfig, PPOCriticConfig


# `model.named_modules()` yields LOCAL layer indices on each PP rank, while
# `get_named_parameters` rewrites them to GLOBAL indices via layer_offset. Strip
# the index so the GLU detection set matches across PP ranks. Also strip trailing
# numeric suffixes on `weight`/`bias` so TEGroupedLinear MoE expert weights
# (`weight0`, `weight1`, …, `weight{global_expert_idx}` after expert_offset
# rewriting) collapse to the same canonical form.
_LAYER_IDX_RE = re.compile(r"\.layers\.\d+\.")
_EXPERT_NUM_RE = re.compile(r"\.(weight|bias)\d+$")


def _normalize_glu_param_name(name: str) -> str:
    name = _LAYER_IDX_RE.sub(".layers.", name)
    name = _EXPERT_NUM_RE.sub(r".\1", name)
    return name


class _MegatronModelList(list):
    """List wrapper that exposes module-like helpers for Megatron model chunks."""

    def forward(self, *args, **kwargs) -> Any:
        if len(self) == 1:
            return self[0](*args, **kwargs)
        raise RuntimeError(
            "Direct forward calls are only supported for single-chunk model list."
        )

    def named_parameters(self, *args, **kwargs) -> Iterator[tuple[str, nn.Parameter]]:
        for module in self:
            yield from module.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs) -> Iterator[nn.Parameter]:
        for _, parameter in self.named_parameters(*args, **kwargs):
            yield parameter


class MegatronEngine(TrainEngine):
    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.hf_config: PretrainedConfig
        self.tf_config: TransformerConfig
        self.model: _MegatronModelList | None = None
        self.dtype = getattr(torch, self.config.dtype)
        self.device = None
        self.optimizer_config = config.optimizer
        self.mcore_config = config.megatron
        self.parallel_strategy = None
        self.optimizer = None
        self.lr_scheduler = None
        self.bridge = None
        self.process_group_initialized = False
        self._initialized = False
        self.rollout_engine: InferenceEngine | None = None
        self.rollout_coordinator: DistRolloutCoordinator | None = None
        self.weight_update_group_initialized: bool = False
        self.weight_update_group_name: str
        self.weight_update_master_addr: str
        self.weight_update_master_port: int
        self._version: int = 0
        self.rank: int | None = None
        self.is_pp_head: bool
        self.world_size: int | None = None
        self.rank_generator: mpu.RankGenerator | None = None
        self.checkpointer: MegatronCheckpointManager | None = None
        self.lr_scheduler: OptimizerParamScheduler | None = None
        self.seed: int = 0
        self.own_global_group: bool = False
        self.is_offload: bool = False
        self._offload_depth: int = 0
        self.enable_tree_training: bool = self.config.enable_tree_training
        # FP8 configuration
        self.fp8_config = self.mcore_config.fp8_config
        self.enable_fp8: bool = self.fp8_config is not None
        self.fp8_direct_convert: bool = (
            self.fp8_config.direct_convert if self.enable_fp8 else False
        )
        self.quantization_config: dict[str, int | str | list[str]] | None = None
        self.bridge_cls: str = getattr(self.mcore_config, "bridge_type", "mbridge")
        self.bridge_lora: MegatronBridgeLoRA | None = None
        self.is_vision_model: bool = False
        self.processor = None

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        if parallel_strategy is None:
            parallel_strategy = ParallelStrategy()
        self.parallel_strategy = self._make_parallel_strategy(parallel_strategy)
        backend = current_platform.communication_backend
        if not dist.is_initialized():
            # NOTE: device_id **SHOULD NOT** be passed into init_process_group,
            # otherwise initializing the NCCL weight update group will be wrong!
            dist.init_process_group(
                backend=backend,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )
            # Initialize Megatron parallel states
            # NOTE: we assume all MegatronEngine has the same parallel strategy.
            vpp_size = self.parallel_strategy.virtual_pipeline_parallel_size
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.parallel_strategy.tensor_parallel_size,
                pipeline_model_parallel_size=self.parallel_strategy.pipeline_parallel_size,
                virtual_pipeline_model_parallel_size=vpp_size if vpp_size > 1 else None,
                use_sharp=False,
                order="tp-cp-ep-dp-pp",
                context_parallel_size=self.parallel_strategy.context_parallel_size,
                expert_model_parallel_size=self.parallel_strategy.expert_parallel_size,
                expert_tensor_parallel_size=self.parallel_strategy.expert_tensor_parallel_size,
                distributed_timeout_minutes=int(
                    DIST_GROUP_DEFAULT_TIMEOUT.seconds / 60
                ),
            )
            # Set megatron model parallel seed
            tensor_parallel.model_parallel_cuda_manual_seed(self.seed)
            self.own_global_group = True
        self.logger = logging.getLogger(f"[MegatronEngine Rank {dist.get_rank()}]")
        self._context_and_model_parallel_group = None
        self._init_context_and_model_parallel_group()
        # This is needed for barrier synchronization when models are moved to CPU
        self._cpu_group = dist.new_group(
            timeout=DIST_GROUP_DEFAULT_TIMEOUT, backend="gloo"
        )
        self.process_group_initialized = True

        # Eagerly initialize HCCL/NCCL communicators for the subgroups so
        # that lazy init doesn't race with colocated engines (issue #1099).
        warmup_process_groups(
            self._context_and_model_parallel_group,
            mpu.get_data_parallel_group(),
        )

    def _apply_megatron_bridge_lora(self) -> None:
        assert self.model is not None, "Model must be initialized before applying LoRA."
        assert self.bridge_cls == "megatron-bridge"

        target_modules = list(self.config.target_modules or [])
        if not target_modules or "all-linear" in target_modules:
            # Expand all-linear to explicit Megatron-Bridge linear module targets.
            target_modules = [
                "linear_qkv",
                "linear_proj",
                "linear_fc1",
                "linear_fc2",
            ]
        self.bridge_lora = MegatronBridgeLoRA(
            target_modules=target_modules,
            dim=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=0.0,
        )
        self.model = _MegatronModelList(self.bridge_lora(self.model, training=True))
        self.bridge_lora.set_params_to_save(self.model)

        total_params = sum(param.numel() for param in self.model.parameters())
        trainable_params = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        self.logger.info(
            "Applied Megatron Bridge LoRA: target_modules=%s, rank=%s, alpha=%s, trainable=%s/%s (%.4f%%)",
            target_modules,
            self.config.lora_rank,
            self.config.lora_alpha,
            trainable_params,
            total_params,
            100.0 * trainable_params / max(total_params, 1),
        )

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec, *args, **kwargs):
        try:
            self.seed = get_seed()
        except ValueError:
            self.logger.warning("Seed not set, using default seed 42.")
            self.seed = 42

        assert addr is None, "FSDPEngine does not support remote initialization."

        self._normalize_adam_bf16_config()

        if is_tms_enabled():
            torch_memory_saver.hook_mode = "preload"

        current_platform.set_device(int(os.environ["LOCAL_RANK"]))
        current_platform.set_numa_affinity(int(os.environ["LOCAL_RANK"]))
        self.device = torch.device(int(os.environ["LOCAL_RANK"]))
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.is_pp_head = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
        )
        self.weight_update_group_name = (
            f"update_weight_group_{mpu.get_pipeline_model_parallel_rank()}"
        )
        self.engine_lock = DistributedLock("train_engine_lock")

        if self.config.use_lora and self.bridge_cls != "megatron-bridge":
            raise NotImplementedError(
                "MegatronEngine LoRA POC currently only supports bridge_type='megatron-bridge'. "
                "mbridge does not support LoRA in this path."
            )

        self.tokenizer = load_hf_tokenizer(self.config.path)

        with patch_bridge_for_tree_training(
            self.enable_tree_training and self.bridge_cls == "mbridge"
        ):
            self.bridge = self._build_hf_mcore_bridge()

            self.hf_config, self.tf_config = make_hf_and_mcore_config(
                self.config.path,
                dtype=self.dtype,
                bridge=self.bridge,
                bridge_type=self.bridge_cls,
            )
            self.tf_config = configure_pipeline_layer_splits(
                self.parallel_strategy, self.hf_config, self.tf_config
            )

            self.is_vision_model = is_valid_vision_model(self.hf_config.model_type)
            # GDN/SSM models (e.g. Qwen3.5) reject packed THD input and must run
            # the padded BSHD forward. Derived from model type rather than a
            # config flag so the layout can't be mis-set.
            self.use_padded_seq = requires_padded_seq(self.hf_config.model_type)
            if self.is_vision_model:
                if self.parallel_strategy.context_parallel_size > 1:
                    raise NotImplementedError(
                        "Context parallel (CP > 1) is not supported with VLM models. "
                        f"Got context_parallel_size={self.parallel_strategy.context_parallel_size} "
                        f"for model_type={self.hf_config.model_type}."
                    )
                self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
                    self.config.path
                )
                self.logger.info(
                    f"VLM model detected (type={self.hf_config.model_type}). "
                    f"Loaded processor and tokenizer."
                )

            if self.use_padded_seq and self.parallel_strategy.context_parallel_size > 1:
                raise NotImplementedError(
                    f"Context parallel (CP > 1) is not supported for "
                    f"model_type={self.hf_config.model_type!r}, which requires the "
                    "padded BSHD forward (it operates on [B, S] tensors while the "
                    "CP path packs sequences). "
                    f"Got context_parallel_size={self.parallel_strategy.context_parallel_size}."
                )

            self.quantization_config = getattr(
                self.hf_config, "quantization_config", None
            )

            self._check_and_apply_fp8_config()
            self._validate_fp8_consistency()

            # Warn once if bridge-delegated weight sync was requested but a
            # fallback condition forces the registry conversion path (the
            # dispatch in _update_weights_from_distributed silently falls back).
            if self.mcore_config.use_bridge_for_update_weights:
                fallback_reasons = []
                if self.bridge_cls != "megatron-bridge":
                    fallback_reasons.append(f"bridge_type={self.bridge_cls!r}")
                if self.quantization_config:
                    fallback_reasons.append("FP8/quantized training")
                if self.config.use_lora:
                    fallback_reasons.append("LoRA enabled")
                if fallback_reasons:
                    self.logger.warning(
                        "use_bridge_for_update_weights=True, but live weight sync "
                        "will use the registry conversion path instead because: "
                        f"{', '.join(fallback_reasons)}."
                    )

            with self.device:
                models = make_mcore_model(
                    hf_config=self.hf_config,
                    tf_config=self.tf_config,
                    mcore_config=self.mcore_config,
                    bridge=self.bridge,
                    bridge_type=self.bridge_cls,
                    is_critic=self.config.is_critic,
                    use_lora=self.config.use_lora,
                )

        self.model = _MegatronModelList(models)

        if self.config.use_lora:
            self._apply_megatron_bridge_lora()

        with self.device:
            self._load_model_from_hf(self.config.path)

        # NOTE: Clear high_precision_init_val for FP8 parameters.
        #
        # Background: When using distributed optimizer, Megatron uses
        # high_precision_init_val to initialize optimizer's main parameters.
        # TransformerEngine (TE) provides this via get_high_precision_init_val().
        #
        # Problem with publicly available HF FP8 models:
        # - Megatron sets preserve_high_precision_init_val=True when loading FP8 models
        # - This causes TE (transformer_engine/pytorch/module/base.py) to use the
        #   init_method's random initialization as high_precision_init_val
        # - But for pre-trained HF models, we load actual weights AFTER initialization,
        #   so high_precision_init_val still holds the random init values, not the
        #   loaded weights
        #
        # Solution: Clear high_precision_init_val here after loading HF weights.
        # The optimizer will then use the actual FP8 weights (upcast to high precision)
        # instead of stale random initialization values.
        for model in self.model:
            for _, param in model.named_parameters():
                if hasattr(param, "get_high_precision_init_val"):
                    param.clear_high_precision_init_val()
                    delattr(param, "get_high_precision_init_val")
                    delattr(param, "clear_high_precision_init_val")

        assert self.model, "Megatron models failed to initialize."

        self._glu_fc1_names: set[str] = self._build_glu_fc1_names()

        modules = [m.module if isinstance(m, DDP) else m for m in self.model]
        total_params = sum(
            param.numel() for module in modules for param in module.parameters()
        )
        self.logger.info(
            f"Model parameter count: {total_params / 1e6:.2f}M, pp_stage={mpu.get_pipeline_model_parallel_rank()}, vpp_chunks={len(self.model)}"
        )

        if self.config.disable_dropout:
            for model in self.model:
                disable_dropout_in_model(model)

        primary_model = self.model[0]
        model_config = get_model_config(primary_model)

        # NOTE: It is recommended to set this option to True for RL training on MoE models for stability.
        if self.mcore_config.use_deterministic_algorithms:
            set_deterministic_algorithms(model_config)

        # Set vp_stage for DDP models
        for i, model_chunk in enumerate(self.model):
            if (
                isinstance(model_chunk, DDP)
                and self.mcore_config.virtual_pipeline_parallel_size > 1
            ):
                vp_stage = getattr(model_chunk.module, "vp_stage", None)
                self.logger.info(f"Setting vp_stage {vp_stage} for model chunk {i}.")
                setattr(model_chunk, "vp_stage", vp_stage)

        if self.mcore_config.ddp.overlap_grad_reduce and isinstance(primary_model, DDP):
            model_config.no_sync_func = [
                model_chunk.no_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                model_config.no_sync_func = model_config.no_sync_func[0]

        if (
            self.mcore_config.ddp.overlap_param_gather
            and self.mcore_config.ddp.align_param_gather
        ):
            model_config.param_sync_func = [
                model_chunk.start_param_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                model_config.param_sync_func = model_config.param_sync_func[0]
        model_config.finalize_model_grads_func = finalize_model_grads
        self._create_optimizer(ft_spec)
        self._initialized = True

    def _build_glu_fc1_names(self) -> set[str]:
        """Detect which `linear_fc1` parameters belong to GLU MLPs.

        Compares weight shapes: if ``fc1.weight.shape[0] == 2 * fc2.weight.shape[1]``
        at the TP-local level, the MLP is gated and fc1 needs stride-2 de-interleave
        at TP>1. Shape-based detection is model-agnostic and doesn't rely on config
        flags. Names are stored with layer indices and per-expert numeric suffixes
        stripped so they match across PP ranks (`model.named_modules()` yields LOCAL
        indices while `get_named_parameters` rewrites to GLOBAL via `layer_offset`)
        and across TEGroupedLinear MoE expert weights (`weight0`, `weight1`, ...).
        """
        glu_fc1_names: set[str] = set()
        for model in self.model:
            for mod_name, module in model.named_modules():
                fc1 = getattr(module, "linear_fc1", None)
                fc2 = getattr(module, "linear_fc2", None)
                if fc1 is None or fc2 is None:
                    continue
                # Pick a representative weight: standard linear has `.weight`;
                # TEGroupedLinear has `weight0` per local expert (all experts
                # share the same shape).
                fc1_w = getattr(fc1, "weight", None)
                if fc1_w is None:
                    fc1_w = getattr(fc1, "weight0", None)
                fc2_w = getattr(fc2, "weight", None)
                if fc2_w is None:
                    fc2_w = getattr(fc2, "weight0", None)
                if fc1_w is None or fc2_w is None:
                    continue
                fc1_out = fc1_w.shape[0]
                fc2_in = fc2_w.shape[1] if fc2_w.dim() >= 2 else fc2_w.shape[0]
                if fc1_out != 2 * fc2_in:
                    continue
                # Iterate fc1's direct parameters (recurse=False) and pick
                # `weight`/`bias` plus their grouped-MoE numbered variants.
                for p_name, _ in fc1.named_parameters(recurse=False):
                    base = p_name.rstrip("0123456789")
                    if base not in ("weight", "bias"):
                        continue
                    full_name = (
                        f"{mod_name}.linear_fc1.{p_name}"
                        if mod_name
                        else f"linear_fc1.{p_name}"
                    )
                    glu_fc1_names.add(_normalize_glu_param_name(full_name))
        return glu_fc1_names

    def _build_hf_mcore_bridge(self):
        if self.bridge_cls == "mbridge":
            self.bridge = mbridge.AutoBridge.from_pretrained(
                self.config.path, trust_remote_code=True
            )
            self.bridge.dtype = self.dtype
            if self.config.gradient_checkpointing:
                self.bridge.set_extra_args(
                    recompute_granularity=self.mcore_config.recompute_granularity,
                    recompute_method=self.mcore_config.recompute_method,
                    recompute_num_layers=self.mcore_config.recompute_num_layers,
                    distribute_saved_activations=self.mcore_config.distribute_saved_activations,
                    recompute_modules=self.mcore_config.recompute_modules,
                )
            self.logger.info(
                "Using mbridge to create models and hf model save/load in MegatronEngine."
            )

        elif self.bridge_cls == "megatron-bridge":
            if self.enable_tree_training:
                raise NotImplementedError(
                    "Tree training is not supported with bridge_type='megatron-bridge'."
                )
            self.bridge = MegatronBridgeAutoBridge.from_hf_pretrained(
                self.config.path,
                trust_remote_code=True,
                dtype=self.config.dtype,
            )
            self.logger.info(
                "Using megatron-bridge to create models and hf model save/load in MegatronEngine."
            )

        else:
            self.logger.info(
                "Not using bridge to create models and hf model save/load in MegatronEngine."
            )
            self.bridge = None
        return self.bridge

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def data_parallel_rank(self) -> int:
        assert self.process_group_initialized
        return mpu.get_data_parallel_rank()

    @property
    def data_parallel_world_size(self) -> int:
        assert self.process_group_initialized
        return mpu.get_data_parallel_world_size()

    @property
    def data_parallel_group(self) -> dist.ProcessGroup:
        assert self.process_group_initialized
        return mpu.get_data_parallel_group()

    def current_data_parallel_head(self) -> int:
        """Get the rank of the head of the current data parallel group."""
        assert self.process_group_initialized
        ranks = dist.get_process_group_ranks(self.context_and_model_parallel_group)
        return ranks[0]

    def is_data_parallel_head(self) -> bool:
        assert self.process_group_initialized
        ranks = dist.get_process_group_ranks(self.context_and_model_parallel_group)
        return ranks[0] == self.rank

    @property
    def pipeline_parallel_rank(self) -> int:
        assert self.process_group_initialized
        return mpu.get_pipeline_model_parallel_rank()

    def is_pipeline_parallel_head(self) -> bool:
        assert self.process_group_initialized
        return self.is_pp_head

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        assert self.process_group_initialized
        return self._context_and_model_parallel_group

    @property
    def cpu_group(self) -> dist.ProcessGroup:
        assert self.process_group_initialized
        return self._cpu_group

    def destroy(self):
        self._initialized = False
        self.process_group_initialized = False
        # Drain any pending async checkpoint saves before tearing down process
        # groups; the background save workers issue collectives during finalize.
        if getattr(self, "checkpointer", None) is not None:
            self.checkpointer.close()
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "model"):
            self.model = None
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
            # "recvValue failed" on the already-closed store.
            if getattr(self, "_cpu_group", None) is not None:
                try:
                    dist.barrier(group=self._cpu_group)
                except Exception as e:  # pragma: no cover - best-effort
                    self.logger.warning(
                        f"pre-destroy CPU barrier failed (ignored): {e}"
                    )
            mpu.destroy_model_parallel()
            dist.destroy_process_group()
            self.own_global_group = False

    def train(self, mode: bool = True):
        assert self.model is not None
        for model in self.model:
            model.train(mode=mode)
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
                if meta.with_optim:
                    raise ValueError(
                        "HF format does not support optimizer state saving, please use DCP format instead."
                    )
                self._save_model_to_hf(
                    meta.path,
                    tokenizer=meta.tokenizer,
                    processor=meta.processor,
                    base_model_path=meta.base_model_path,
                )
            elif meta.weight_format == "dcp":
                if self.checkpointer is None:
                    raise NotImplementedError(
                        "DCP checkpoint save is not available for this Megatron configuration "
                        "(e.g., LoRA path without distributed optimizer support). "
                        "Please use weight_format='hf' for adapter/full-model export."
                    )
                self.checkpointer.save_checkpoint(
                    meta.path, with_optimizer=meta.with_optim
                )
            else:
                raise ValueError(f"Unknown weight format {meta.weight_format}. ")

    def load(self, meta: SaveLoadMeta):
        with self._offload_aware_context():
            if meta.weight_format == "hf":
                if meta.with_optim:
                    raise ValueError(
                        "HF format does not support optimizer state loading, please use DCP format instead."
                    )
                self._load_model_from_hf(meta.path)
            elif meta.weight_format == "dcp":
                if self.checkpointer is None:
                    raise NotImplementedError(
                        "DCP checkpoint load is not available for this Megatron configuration "
                        "(e.g., LoRA path without distributed optimizer support). "
                        "Please use weight_format='hf' for adapter/full-model load."
                    )
                self.checkpointer.load_checkpoint(
                    meta.path, with_optimizer=meta.with_optim
                )
            else:
                raise ValueError(f"Unknown weight format {meta.weight_format}. ")

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
        assert self.optimizer is not None, "Optimizer is not initialized."
        self.optimizer.zero_grad()
        for model in self.model:
            model.zero_grad_buffer()

    def optimizer_step(self):
        with trace_scope("megatron_engine.step"):
            update_successful, grad_norm, _ = self.optimizer.step()
        current_lr = self.optimizer.param_groups[0]["lr"]

        return dict(
            update_successful=float(update_successful),
            grad_norm=float(grad_norm) if grad_norm is not None else float("nan"),
            lr=current_lr,
        )

    def lr_scheduler_step(self):
        assert self.lr_scheduler is not None, "LR Scheduler is not initialized."
        self.lr_scheduler.step(1)

    def forward_backward_batch(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool = False,
    ) -> None:
        self._ensure_ready()

        def forward_step(batch_iter, model):
            mb_input: MicroBatchItem = next(batch_iter)

            cu_seqlens = mb_input.padded_mb.get("cu_seqlens", None)

            # Lazily create tree attention metadata just before forward.
            # dense_mask=True because Megatron's gradient checkpointing uses
            # save_for_backward() which can only save torch.Tensor objects;
            # BlockMask is recreated inside PytorchFlexAttention.forward().
            tree_attn_keys: list[str] = []
            if self.enable_tree_training:
                trie_node = mb_input.padded_mb.get("trie_node", None)
                # Ensure trie_node is also in orig_mb for _compute_logprobs_and_loss
                if trie_node is not None and "trie_node" not in mb_input.orig_mb:
                    mb_input.orig_mb["trie_node"] = trie_node
                padded_size = mb_input.padded_to_length
                if trie_node is not None:
                    assert padded_size is not None
                    tree_kwargs = build_tree_attn_kwargs(
                        trie_node,
                        padded_size,
                        mb_input.padded_mb["input_ids"].device,
                        dense_mask=True,
                    )
                    mb_input.padded_mb.update(tree_kwargs)
                    tree_attn_keys = list(tree_kwargs.keys())

            cp_size = mpu.get_context_parallel_world_size()
            cp_local = cp_size > 1

            output = packed_context_parallel_forward(
                model,
                mb_input.padded_mb,
                gather_cp_output=not cp_local,
                is_vision_model=self.is_vision_model,
                use_padded_seq=self.use_padded_seq,
            )

            # Release tree attention metadata after forward pass
            for key in tree_attn_keys:
                del mb_input.padded_mb[key]

            def _process_output(input_, output_):
                loss = process_output_fn(output_, input_)
                if loss is None:
                    loss = torch.tensor(1.0, device=output_.device)
                return loss, {}

            model_vp_stage = getattr(model, "vp_stage", 0)
            if mpu.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=model_vp_stage
            ):
                if cp_local and cu_seqlens is not None:
                    padded_cu_seqlens = mb_input.padded_mb["cu_seqlens"]
                    rolled_ids = torch.roll(
                        mb_input.padded_mb["input_ids"], shifts=-1, dims=-1
                    )
                    cp_labels = split_packed_seqs_for_context_parallel(
                        rolled_ids, padded_cu_seqlens
                    )
                    cp_inputs = dict(mb_input.orig_mb)
                    cp_inputs["_cp_local_labels"] = cp_labels
                    cp_inputs["_cp_padded_cu_seqlens"] = padded_cu_seqlens
                    cp_inputs["_cp_padding_length"] = mb_input.padding_length
                    cp_inputs["_cp_old_cu_seqlens"] = mb_input.old_cu_seqlens
                    return output, functools.partial(_process_output, cp_inputs)
                else:
                    output = unpad_logits(
                        output,
                        padding_length=mb_input.padding_length,
                        cu_seqlens=cu_seqlens,
                        old_cu_seqlens=mb_input.old_cu_seqlens,
                    )
            return output, functools.partial(_process_output, mb_input.orig_mb)

        forward_backward_func = get_forward_backward_func()
        with trace_scope("megatron_engine.forward_backward"):
            if len(self.model) > 1:
                data_iterator = [iter(mb_list) for _ in range(len(self.model))]
            else:
                data_iterator = iter(mb_list)
            forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=data_iterator,
                model=self.model if len(self.model) > 1 else self.model[0],
                num_microbatches=len(mb_list),
                seq_length=mb_list.max_seqlen,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )

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

        # Step 2: Compute total loss weight.
        # Use DP+CP group: after CP all-gather each rank computes the full-sequence
        # loss, so all_gather's backward (reduce_scatter) sums cp_size identical
        # gradients, amplifying by cp_size. Including CP in the weight all-reduce
        # introduces a matching cp_size factor in the denominator, cancelling out.
        total_loss_weight = compute_total_loss_weight(
            mb_list,
            loss_weight_fn,
            mpu.get_data_parallel_group(with_context_parallel=True),
        )

        # Step 3: Forward-backward using Megatron's pipeline function.
        # `len(mb_list)` compensates Megatron Core's `output_tensor /= num_microbatches`
        # applied in the 2-tuple `(loss, {})` branch of
        # `megatron.core.pipeline_parallel.schedules._forward_step_helper`. Our
        # per-microbatch loss is already globally normalized via `w_i / W_total`, so
        # that extra division would shrink every gradient (and thus grad_norm and the
        # effective optimizer step) by `num_microbatches`.
        loss_multiplier = (
            mpu.get_data_parallel_world_size()
            * self.optimizer.get_loss_scale().item()
            * len(mb_list)
        )

        def process_output(
            output: torch.Tensor, inputs: dict[str, Any]
        ) -> torch.Tensor:
            return self._compute_logprobs_and_loss(
                output,
                inputs,
                loss_fn,
                loss_weight_fn,
                total_loss_weight,
                loss_multiplier=loss_multiplier,
            )

        self.forward_backward_batch(
            mb_list,
            process_output,
            forward_only=False,
        )

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

        # Step 2: Compute total loss weight (DP+CP, see train_batch comment).
        total_loss_weight = compute_total_loss_weight(
            mb_list,
            loss_weight_fn,
            mpu.get_data_parallel_group(with_context_parallel=True),
        )

        # Step 3: Forward using Megatron's pipeline function, collecting losses
        losses: list[torch.Tensor] = []

        def process_output(
            output: torch.Tensor, inputs: dict[str, Any]
        ) -> torch.Tensor:
            loss = self._compute_logprobs_and_loss(
                output, inputs, loss_fn, loss_weight_fn, total_loss_weight
            )
            losses.append(loss.detach())
            return loss

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        # Step 4: Aggregate losses
        if mpu.is_pipeline_last_stage():
            return aggregate_eval_losses(
                losses, mpu.get_data_parallel_group(with_context_parallel=True)
            )
        return None

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

        # Step 3: Forward using Megatron's pipeline function, collecting results
        outputs: list[torch.Tensor] = []

        def process_output(output: torch.Tensor, inputs: dict[str, Any]) -> None:
            result = self._compute_forward_result(output, inputs)
            outputs.append(result)
            return None

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        # Step 4: Aggregate, reorder, and broadcast outputs
        res = None
        if mpu.is_pipeline_last_stage():
            if self.enable_tree_training:
                res = merge_packed_tree_results(outputs, batch_size)
            else:
                res = reorder_and_pad_outputs(
                    outputs, output_seqlens, mb_list, aggregate_fn
                )
        res = broadcast_tensor(
            res,
            src_rank=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        if meta is None:
            return res
        return split_batch(res, meta)

    def export_stats(self) -> dict[str, float]:
        with self._offload_aware_context():
            data = stats_tracker.export_all(
                reduce_group=self.data_parallel_group,
            )
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            # Some log info only exist in last pipeline rank
            data_list = [data]
            dist.broadcast_object_list(
                data_list,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )
            data.update(data_list[0])
        return data

    def offload(self) -> None:
        """Offload model memory to CPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/actor.py
        """
        if not is_tms_enabled():
            raise RuntimeError(
                "torch_memory_saver requires `enable_offload=True` in yaml config."
            )

        self.get_device_stats().log("before offload model")

        # Discard gradient buffers via Megatron's native API *before* TMS pause.
        # `DDP.offload_grad_buffers()` releases grad storage in-place
        # (storage().resize_(0)); views like param.main_grad recover automatically
        # on restore. Since grads are recomputed every step, they need no CPU
        # backup. Doing this before pause() means the grad region has no physical
        # memory left for TMS to back up.
        if self.mcore_config.disable_grad_buffers_cpu_backup:
            for m in self.model:
                m.offload_grad_buffers(synchronize=False, empty_cache=False)

        current_platform.clear_memory()
        torch_memory_saver.pause()

        # TODO: NCCL offload
        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after offload model")

        self.is_offload = True

    def onload(self) -> None:
        """Onload model memory from CPU back to GPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/actor.py
        """

        torch_memory_saver.resume()

        # Reallocate gradient buffers released in offload(). resize_() restores
        # storage and zeroes it; param.main_grad views become valid again.
        if self.mcore_config.disable_grad_buffers_cpu_backup:
            for m in self.model:
                m.restore_grad_buffers(synchronize=False)

        current_platform.clear_memory()

        # TODO: NCCL onload
        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after onload model")

        self.is_offload = False

    def clear_batches(self, shard_ids: list[str] | None = None) -> None:
        """Drain this worker's client-side RTensor fetch buffer.

        Called via RPC by ``TrainController.clear_batches`` at step end so
        cross-node consumer DP heads release cached tensors. See #1209.
        Non-DP-head ranks receive no positional args via
        ``_call_workers`` (see train_controller.py:575-577) — accept the
        no-args call and noop, since their ``_fetch_buffer`` is empty.
        """
        from areal.infra.rpc.rtensor import clear_fetch_buffer

        if shard_ids:
            clear_fetch_buffer(shard_ids)

    def fetch_buffer_stats(self) -> dict[str, int]:
        """Expose local fetch-buffer stats for post-step drain verification."""
        from areal.infra.rpc.rtensor import fetch_buffer_stats

        return fetch_buffer_stats()

    def _normalize_adam_bf16_config(self) -> None:
        if self.optimizer_config is None or self.optimizer_config.type != "adam_bf16":
            return

        self.logger.info(
            "Detected 'adam_bf16' optimizer with Megatron Engine. "
            "Automatically converting to 'adam' with precision-aware optimizer "
            "and setting exp_avg_dtype/exp_avg_sq_dtype to 'bfloat16'."
        )

        self.optimizer_config.type = "adam"
        self.mcore_config.use_precision_aware_optimizer = True
        self.mcore_config.exp_avg_dtype = "bfloat16"
        self.mcore_config.exp_avg_sq_dtype = "bfloat16"

        if self.dtype != torch.bfloat16:
            self.logger.warning(
                "Overriding dtype from %s to bfloat16 for adam_bf16 optimizer.",
                self.config.dtype,
            )
            self.dtype = torch.bfloat16
            self.config.dtype = "bfloat16"

    def _check_and_apply_fp8_config(self):
        if not self.enable_fp8:
            return
        fp8_config = self.fp8_config
        special_mappings = {"mode": "fp8"}
        # Fields that use the same name in both configs (no prefix needed)
        same_fields = {
            "tp_only_amax_red",
            "first_last_layers_bf16",
            "num_layers_at_start_in_bf16",
            "num_layers_at_end_in_bf16",
        }
        # All other fields get the `fp8_` prefix
        for field in dataclasses.fields(fp8_config):
            fp8_field = field.name
            if fp8_field in special_mappings:
                tf_field = special_mappings[fp8_field]
            elif fp8_field in same_fields:
                tf_field = fp8_field
            else:
                tf_field = f"fp8_{fp8_field}"
            if hasattr(self.tf_config, tf_field):
                setattr(self.tf_config, tf_field, getattr(fp8_config, fp8_field))
            else:
                self.logger.warning(
                    f"Unknown FP8 field in TransformerConfig: {fp8_field}"
                )
        self.logger.info(
            f"FP8 training enabled: mode={fp8_config.mode}, "
            f"recipe={fp8_config.recipe}, "
            f"param={fp8_config.param}"
        )
        # fp8_param_gather is passed from make_mcore_model()

    def _validate_fp8_consistency(self):
        """Validate that FP8 configuration is consistent.

        If either training uses FP8, quantization_config must exist
        and quant_method must be "fp8" (weights must be FP8).
        """
        train_fp8 = self.enable_fp8
        weights_fp8 = (
            self.quantization_config is not None
            and self.quantization_config.get("quant_method", None) == "fp8"
        )

        if train_fp8 and not weights_fp8:
            raise RuntimeError(
                "FP8 configuration error: "
                "If training uses FP8, quantization_config must exist "
                "and quant_method must be 'fp8' (weights must be FP8). "
                f"Training fp8={train_fp8}, "
                f"weights fp8={weights_fp8}, "
                f"quantization_config={self.quantization_config}"
            )

    def get_device_stats(self) -> DeviceRuntimeInfo:
        return DeviceRuntimeInfo.get_current()

    def start_memory_profile(self, max_entries: int = 100000) -> None:
        torch.cuda.memory._record_memory_history(max_entries=max_entries)

    def stop_memory_profile(self, snapshot_dir: str) -> None:
        pp = mpu.get_pipeline_model_parallel_rank()
        dp = mpu.get_data_parallel_rank()
        cp = mpu.get_context_parallel_rank()
        tp = mpu.get_tensor_model_parallel_rank()
        filename = f"snapshot_rank{self.rank:02d}_p{pp}d{dp}c{cp}t{tp}.pickle"
        path = os.path.join(snapshot_dir, filename)
        torch.cuda.memory._dump_snapshot(path)
        torch.cuda.memory._record_memory_history(enabled=None)

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
    ) -> MegatronParallelStrategy:
        base_strategy = dataclasses.asdict(parallel_strategy)
        vpp_size = self.mcore_config.virtual_pipeline_parallel_size
        return MegatronParallelStrategy(
            use_sequence_parallel=parallel_strategy.tensor_parallel_size > 1,
            virtual_pipeline_parallel_size=vpp_size,
            **base_strategy,
        )

    def _init_context_and_model_parallel_group(self) -> None:
        # Initialize context and model parallel groups, which are only used in AReaL
        # for data distribution
        rank_generator = mpu.RankGenerator(
            tp=self.parallel_strategy.tensor_parallel_size,
            ep=1,
            dp=self.parallel_strategy.data_parallel_size,
            pp=self.parallel_strategy.pipeline_parallel_size,
            cp=self.parallel_strategy.context_parallel_size,
            order="tp-cp-ep-dp-pp",
            rank_offset=0,
        )
        context_and_model_parallel_ranks = rank_generator.get_ranks("tp-cp-pp")
        # create context and model_parallel_groups
        for dp_rank, ranks in enumerate(context_and_model_parallel_ranks):
            group = mpu.create_group(
                ranks,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
                pg_options=mpu.get_nccl_options("tp-cp-pp", {}),
                group_desc="CONTEXT_AND_MODEL_PARALLEL_GROUP",
            )
            if dp_rank == mpu.get_data_parallel_rank():
                self._context_and_model_parallel_group = group

    def _create_optimizer(self, ft_spec: FinetuneSpec) -> None:
        if self.optimizer_config is None:
            return
        assert self.model is not None and len(self.model) > 0

        use_distributed_optimizer = (
            False
            if self.config.use_lora
            else self.mcore_config.ddp.use_distributed_optimizer
        )

        assert self.optimizer_config.type in [
            "adam",
            "sgd",
        ], "Only AdamW/sgd optimizer is supported in this engine."
        if self.optimizer_config.type == "sgd":
            self.logger.warning(
                "Using the 'sgd' optimizer with Megatron may be less stable. Consider using the 'adam' (AdamW) optimizer for improved stability."
            )

        # Make megatron optimizer config
        mcore_opt_config = MCoreOptimizerConfig(
            optimizer=self.optimizer_config.type,
            lr=self.optimizer_config.lr,
            min_lr=self.optimizer_config.min_lr_ratio * self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
            bf16=self.dtype is torch.bfloat16,
            fp16=self.dtype is torch.float16,
            adam_beta1=self.optimizer_config.beta1,
            adam_beta2=self.optimizer_config.beta2,
            adam_eps=self.optimizer_config.eps,
            use_distributed_optimizer=use_distributed_optimizer,
            params_dtype=self.dtype,
            clip_grad=self.optimizer_config.gradient_clipping,
            fp8_recipe=(self.fp8_config.recipe if self.enable_fp8 else None),
        )
        mcore_opt_config.overlap_param_gather_with_optimizer_step = (
            self.mcore_config.overlap_param_gather_with_optimizer_step
        )
        mcore_opt_config.use_precision_aware_optimizer = (
            self.mcore_config.use_precision_aware_optimizer
        )
        mcore_opt_config.main_grads_dtype = getattr(
            torch, self.mcore_config.main_grads_dtype
        )
        mcore_opt_config.main_params_dtype = getattr(
            torch, self.mcore_config.main_params_dtype
        )
        mcore_opt_config.exp_avg_dtype = getattr(torch, self.mcore_config.exp_avg_dtype)
        mcore_opt_config.exp_avg_sq_dtype = getattr(
            torch, self.mcore_config.exp_avg_sq_dtype
        )

        self.optimizer = get_megatron_optimizer(mcore_opt_config, self.model)

        warmup_steps_proportion = self.optimizer_config.warmup_steps_proportion
        warmup_steps = int(warmup_steps_proportion * ft_spec.total_train_steps)
        lr_scheduler = OptimizerParamScheduler(
            self.optimizer,
            init_lr=0.0 if warmup_steps_proportion > 0 else self.optimizer_config.lr,
            max_lr=self.optimizer_config.lr,
            min_lr=self.optimizer_config.min_lr_ratio * self.optimizer_config.lr,
            lr_warmup_steps=warmup_steps,
            # Megatron Core treats `lr_decay_steps` as the absolute step at
            # which decay ends (it subtracts `lr_warmup_steps` internally to
            # get the decay-phase length). Previously this was passed as
            # `total_train_steps - warmup_steps`, which caused Megatron to
            # subtract warmup twice and the cosine schedule to reach min_lr
            # ~one full warmup earlier than intended (e.g. lr=0 at step 198
            # for a 220-step run). Pass the raw total so cosine spans
            # [warmup_steps, total_train_steps], matching HF's
            # get_cosine_schedule_with_warmup used by the FSDP engine.
            lr_decay_steps=ft_spec.total_train_steps,
            lr_decay_style=self.optimizer_config.lr_scheduler_type,
            start_wd=self.optimizer_config.weight_decay,
            end_wd=self.optimizer_config.weight_decay,
            wd_incr_steps=ft_spec.total_train_steps,
            wd_incr_style="constant",
        )
        self.lr_scheduler = lr_scheduler

        # MegatronCheckpointManager now only support distributed optimizer which lora does not support
        if not self.config.use_lora:
            self.checkpointer = MegatronCheckpointManager(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                use_distributed_optimizer=use_distributed_optimizer,
                use_checkpoint_opt_param_scheduler=self.mcore_config.use_checkpoint_opt_param_scheduler,
                async_save=self.mcore_config.async_save,
            )

    def _check_rollout_engine_connected(self) -> None:
        """Validate that rollout engine has been connected via connect_engine()."""
        if self.rollout_engine is None or self.rollout_coordinator is None:
            raise RuntimeError(
                "Rollout engine not connected. Call connect_engine()"
                " before using rollout/update_weight methods."
            )

    @staticmethod
    def _normalize_batch_input(
        input_: list[dict[str, Any]] | dict[str, Any],
    ) -> tuple[dict[str, Any], Any | None]:
        if isinstance(input_, list):
            return concat_batch(input_)
        return input_, None

    def _ensure_ready(self) -> None:
        if self.is_offload:
            self.onload()

        if self.model is None:
            raise RuntimeError("Model is not initialized.")

    def _update_bucket_weights_from_distributed(
        self,
        meta: WeightUpdateMeta,
        converted_named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ) -> None:
        # Early exit when chunk size is relatively small
        if not converted_named_tensors:
            return

        self.engine_lock.acquire()

        param_specs = [
            ParamSpec(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).split("torch.")[1],
            )
            for name, tensor in converted_named_tensors
        ]

        if self.config.use_lora:
            meta.peft_config = {
                "r": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "target_modules": get_vllm_lora_target_modules(
                    list(self.config.target_modules or [])
                ),
                "bias": "none",
            }

        fut = self.rollout_engine.update_weights_from_distributed(meta, param_specs)

        handles = []
        for _, param in converted_named_tensors:
            handles.append(
                dist.broadcast(
                    param.data, 0, group=self.weight_update_group, async_op=True
                )
            )
        for handle in handles:
            handle.wait()

        fut.result()

        converted_named_tensors.clear()

        self.engine_lock.release()

    @property
    def _duplicated_param_names(self) -> set[str]:
        """Parameter names whose parent module has parallel_mode='duplicated'.

        These params are replicated (not TP-sharded) but TE incorrectly marks
        them with tensor_model_parallel=True. Cached after first computation.
        """
        if not hasattr(self, "_cached_duplicated_param_names"):
            duplicated = set()
            if self.model is not None:
                for model in self.model:
                    for mod_name, module in model.named_modules():
                        if getattr(module, "parallel_mode", None) == "duplicated":
                            for p_name, _ in module.named_parameters(recurse=False):
                                full = f"{mod_name}.{p_name}" if mod_name else p_name
                                duplicated.add(full)
            self._cached_duplicated_param_names = duplicated
        return self._cached_duplicated_param_names

    def _collect_param(
        self,
        name: str,
        param: nn.Parameter | torch.Tensor,
    ) -> tuple[nn.Parameter | torch.Tensor, int]:
        """Collect and prepare a parameter for conversion.

        This method handles:
        - All-gathering the parameter across tensor parallel ranks
        - Removing padding for vocabulary-related parameters
        - Dequantizing FP8 parameters to bf16s
        - Calculating the parameter size in bytes

        Returns:
            Tuple of (prepared_param, param_size_in_bytes)
        """
        normalized = _normalize_glu_param_name(name)
        is_glu = any(normalized.endswith(glu_name) for glu_name in self._glu_fc1_names)
        param = all_gather_param(
            name,
            param,
            self.fp8_direct_convert,
            quantization_config=self.quantization_config,
            duplicated_param_names=self._duplicated_param_names,
            gated_linear_unit=is_glu,
        )
        param = remove_padding(name, param, lang_config(self.hf_config).vocab_size)

        if isinstance(param, FP8BlockwiseTensorHelper):
            # FP8 is stored as uint8, so element_size is 1 byte
            param_size = param.numel()
        else:
            param_size = param.numel() * param.element_size()

        return param, param_size

    def _impl_update_weight_from_distributed(
        self,
        meta: WeightUpdateMeta,
        name: str,
        param: nn.Parameter | torch.Tensor,
        converted_named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
        buffer_size: int,
        weight_chunked_mem_size: int,
    ) -> int:
        param, param_size = self._collect_param(name, param)

        if not self.is_pipeline_parallel_head():
            return buffer_size

        if buffer_size + param_size > weight_chunked_mem_size:
            self._update_bucket_weights_from_distributed(meta, converted_named_tensors)
            buffer_size = 0

        model_name = self.hf_config.model_type
        if self.config.use_lora:
            model_name = f"{model_name}_lora"

        converted_named_tensors.extend(
            convert_to_hf(
                self.tf_config,
                model_name,
                name,
                param,
                quantization_config=self.quantization_config,
                fp8_direct_convert=self.fp8_direct_convert,
                hf_config=self.hf_config,
            )
        )
        buffer_size += param_size
        return buffer_size

    def _update_bucket_expert_weights_from_distributed(
        self,
        meta: WeightUpdateMeta,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ) -> None:
        """Gather a bucket of MoE expert weights and broadcast them.

        This function handles the distributed update for a bucket of Mixture-of-Experts
        (MoE) parameters. Since expert parameters are sharded across the expert
        parallel group, this function first performs an `all_gather` to collect all
        shards from all expert ranks.

        Once the full expert parameters are reconstructed on the pipeline parallel
        head, it converts them to the HuggingFace format and calls
        `_update_bucket_weights_from_distributed` to perform the actual broadcast
        to the inference engine.
        """

        # Early exit when chunk size is relatively small
        if not named_tensors:
            return

        group = mpu.get_expert_model_parallel_group()
        world_size = mpu.get_expert_model_parallel_world_size()

        names = [name for name, _ in named_tensors]
        all_names: list[list[str]] = [None] * world_size
        dist.all_gather_object(all_names, names, group=group)

        for rank_names in all_names:
            if len(named_tensors) != len(rank_names):
                raise RuntimeError(
                    "Named tensor count mismatch across expert parallel ranks: "
                    f"expected {len(rank_names)} but got {len(named_tensors)}"
                )

        gathered_params = [[] for _ in range(world_size)]
        handles = []
        for idx, (_, tensor) in enumerate(named_tensors):
            params = [
                torch.empty_like(tensor.data, device=current_platform.current_device())
                for _ in range(world_size)
            ]
            handle = dist.all_gather(params, tensor.data, group=group, async_op=True)
            handles.append(handle)
            for ep_rank, rank_names in enumerate(all_names):
                gathered_params[ep_rank].append((rank_names[idx], params[ep_rank]))

        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self.is_pipeline_parallel_head():
            return

        gathered_params = sum(gathered_params, [])

        converted_hf_tensors = []
        for name, param in gathered_params:
            converted_hf_tensors.extend(
                convert_to_hf(
                    self.tf_config,
                    self.hf_config.model_type,
                    name,
                    param,
                    quantization_config=self.quantization_config,
                    fp8_direct_convert=self.fp8_direct_convert,
                    hf_config=self.hf_config,
                )
            )

        self._update_bucket_weights_from_distributed(meta, converted_hf_tensors)

    def _impl_update_expert_weight_from_distributed(
        self,
        meta: WeightUpdateMeta,
        name: str,
        param: nn.Parameter | torch.Tensor,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
        buffer_size: int,
        weight_chunked_mem_size: int,
    ) -> int:
        param, param_size = self._collect_param(name, param)

        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > weight_chunked_mem_size:
            self._update_bucket_expert_weights_from_distributed(meta, named_tensors)
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _init_weight_update_from_distributed(self, meta: WeightUpdateMeta) -> None:
        assert meta.type == "xccl"
        gen_pp_size = meta.gen_allocation.parallel.pp_size if meta.gen_allocation else 1

        # NOTE: Processes launched with torchrun will set the following env var to True,
        # which blocks creating another TCP store for weight update.
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)

        if gen_pp_size > 1:
            # Per-PP-rank weight sync requires a 1:1 mapping between training
            # PP stages and inference (sglang) PP stages, because each training
            # PP head creates exactly the group update_weight_group_{train_pp_rank}
            # and each sglang PP stage joins update_weight_group_{gen_pp_rank}.
            # If the two sizes differ:
            #   * train_pp_size > gen_pp_size: training heads with
            #     train_pp_rank >= gen_pp_size create a group sglang never joins
            #     -> rendezvous hang;
            #   * train_pp_size < gen_pp_size: sglang stages with
            #     gen_pp_rank >= train_pp_size find no training source
            #     -> rendezvous hang.
            # Fail fast here on every rank with a clear error.
            train_pp_size = self.parallel_strategy.pipeline_parallel_size
            if train_pp_size != gen_pp_size:
                raise ValueError(
                    f"Per-PP-rank weight sync requires train_pp_size == gen_pp_size, "
                    f"got train_pp_size={train_pp_size}, gen_pp_size={gen_pp_size}. "
                    f"Set the inference allocation pp_size to match the training "
                    f"pipeline_parallel_size."
                )
            # PP>1: every PP source rank (dp=0, tp=0) creates its own per-PP-rank
            # NCCL group. The group contains only the inference workers at the
            # corresponding PP rank (TP * DP workers) plus one training rank.
            if self.is_pipeline_parallel_head():
                assert meta.gen_allocation is not None

                self.engine_lock.acquire()
                try:
                    meta.nccl_master_address = self.weight_update_master_addr = (
                        gethostip()
                    )
                    meta.nccl_master_port = self.weight_update_master_port = (
                        find_free_ports(1)[0]
                    )
                    meta.nccl_group_name = self.weight_update_group_name

                    fut = self.rollout_engine.init_weights_update_group(meta)

                    per_pp_world_size = (
                        meta.gen_allocation.parallel.world_size // gen_pp_size
                    )
                    init_method = (
                        f"tcp://{format_host_for_url(meta.nccl_master_address)}"
                        f":{meta.nccl_master_port}"
                    )
                    self.logger.info(
                        f"Initializing per-PP-rank weight update group: "
                        f"type={meta.type} init_method={init_method} "
                        f"group={self.weight_update_group_name} "
                        f"per_pp_world_size={per_pp_world_size}"
                    )
                    self.weight_update_group = init_custom_process_group(
                        backend=current_platform.communication_backend,
                        world_size=per_pp_world_size + 1,
                        init_method=init_method,
                        rank=0,
                        group_name=self.weight_update_group_name,
                        timeout=DIST_GROUP_DEFAULT_TIMEOUT,
                    )

                    fut.result()
                finally:
                    self.engine_lock.release()
            else:
                # Non-PP-head ranks do not create NCCL groups.  Set placeholder values so
                # the attributes exist; they are never used for network I/O
                # on non-PP-head ranks.
                self.weight_update_master_addr = ""
                self.weight_update_master_port = 0
        else:
            # PP==1: original behaviour – only the pp_rank=0 head creates
            # a single group spanning all inference workers.
            # Only one PP head exists, so no port race is possible.
            if self.is_pipeline_parallel_head():
                assert meta.gen_allocation is not None

                meta.nccl_master_address = self.weight_update_master_addr = gethostip()
                meta.nccl_master_port = self.weight_update_master_port = (
                    find_free_ports(1)[0]
                )
                meta.nccl_group_name = self.weight_update_group_name

                self.engine_lock.acquire()
                try:
                    fut = self.rollout_engine.init_weights_update_group(meta)

                    gen_world_size = meta.gen_allocation.parallel.world_size
                    init_method = (
                        f"tcp://{format_host_for_url(meta.nccl_master_address)}"
                        f":{meta.nccl_master_port}"
                    )
                    self.logger.info(
                        f"Initializing weight update group: type={meta.type} "
                        f"init_method={init_method} "
                        f"group={self.weight_update_group_name}"
                    )
                    self.weight_update_group = init_custom_process_group(
                        backend=current_platform.communication_backend,
                        world_size=gen_world_size + 1,
                        init_method=init_method,
                        rank=0,
                        group_name=self.weight_update_group_name,
                        timeout=DIST_GROUP_DEFAULT_TIMEOUT,
                    )

                    fut.result()
                finally:
                    self.engine_lock.release()
            else:
                self.weight_update_master_addr = ""
                self.weight_update_master_port = 0

    @trace_perf("megatron_engine.update_weights_from_distributed", category="comm")
    def _update_weights_from_distributed(self, meta: WeightUpdateMeta) -> None:
        # Reset weight weight meta with local info
        meta.nccl_master_address = self.weight_update_master_addr
        meta.nccl_master_port = self.weight_update_master_port
        meta.nccl_group_name = self.weight_update_group_name

        if dist.get_rank() == 0:
            self.rollout_engine.pause_generation()

        dist.barrier(group=self.cpu_group)

        # Bridge delegation: when bridge_type=megatron-bridge and the user opts in,
        # stream HF tensors directly from bridge.export_hf_weights. Falls back to
        # the hand-rolled registry path for FP8 (quant_mapping in megatron-bridge
        # is amax-style, not TE blockwise) and for LoRA (separate adapter export
        # path not yet wired here).
        use_bridge = (
            self.bridge_cls == "megatron-bridge"
            and self.mcore_config.use_bridge_for_update_weights
            and not self.quantization_config
            and not self.config.use_lora
        )
        if use_bridge:
            self._update_weights_via_bridge(meta)
        else:
            self._update_weights_via_registry(meta)

        if dist.get_rank() == 0:
            self.rollout_engine.continue_generation()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def _update_weights_via_registry(self, meta: WeightUpdateMeta) -> None:
        """Hand-rolled conversion path via convert_to_hf registry.

        Used for FP8, LoRA, and models with a converter entry. Iterates this PP
        rank's local params, TP-gathers per param, converts to HF layout, and
        bucket-broadcasts to the rollout engine.
        """
        num_moe_experts = self.tf_config.num_moe_experts
        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024

        buffer_size = 0
        converted_named_tensors = []

        for name, param in get_named_parameters(self.model, num_moe_experts):
            if ".experts." in name and not self.config.use_lora:
                continue
            if self.config.use_lora and (
                ".adapter." not in name or not getattr(param, "requires_grad", False)
            ):
                continue
            buffer_size = self._impl_update_weight_from_distributed(
                meta,
                name,
                param,
                converted_named_tensors,
                buffer_size,
                weight_chunked_mem_size,
            )

        # Only pipeline parallel heads CAN contain named tensors here
        if converted_named_tensors:
            self._update_bucket_weights_from_distributed(meta, converted_named_tensors)
        elif self.config.use_lora and self.is_pipeline_parallel_head():
            self.logger.warning(
                "No tensors were collected for distributed update at version %s.",
                meta.version,
            )

        dist.barrier(group=self.cpu_group)

        buffer_size = 0
        named_tensors = []

        for name, param in get_named_parameters(self.model, num_moe_experts):
            if ".experts." not in name or self.config.use_lora:
                continue
            buffer_size = self._impl_update_expert_weight_from_distributed(
                meta,
                name,
                param,
                named_tensors,
                buffer_size,
                weight_chunked_mem_size,
            )

        if named_tensors:
            # This function will early return if not pipeline parallel head
            self._update_bucket_expert_weights_from_distributed(meta, named_tensors)

        dist.barrier(group=self.cpu_group)

    def _update_weights_via_bridge(self, meta: WeightUpdateMeta) -> None:
        """Delegate live weight sync to megatron-bridge.export_hf_weights.

        Streams (hf_name, hf_tensor) directly from the bridge, which handles
        TP/EP/PP gather and layout transformation internally. Each PP rank
        iterates the global parameter set (vs registry path which iterates only
        local layers); non-PP-heads participate in collectives but do not bucket.
        MoE expert weights are yielded inline by the bridge's grouped-export
        path, so no separate second pass is needed.
        """
        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024
        bucket: list[tuple[str, torch.Tensor]] = []
        bucket_size = 0

        for hf_name, hf_tensor in self.bridge.export_hf_weights(
            self.model,
            cpu=False,
            show_progress=False,
        ):
            if not self.is_pipeline_parallel_head():
                continue
            size = hf_tensor.numel() * hf_tensor.element_size()
            if bucket_size + size > weight_chunked_mem_size:
                self._update_bucket_weights_from_distributed(meta, bucket)
                bucket_size = 0
            bucket.append((hf_name, hf_tensor.contiguous()))
            bucket_size += size

        if bucket:
            self._update_bucket_weights_from_distributed(meta, bucket)

        dist.barrier(group=self.cpu_group)

    @trace_perf("megatron_engine.update_weights_from_disk", category="io")
    def _update_weights_from_disk(self, meta: WeightUpdateMeta) -> None:
        fut = Future()

        if dist.get_rank() == 0:
            self.rollout_engine.pause_generation()
            fut = self.rollout_engine.update_weights_from_disk(meta)

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
        tokenizer: Any | None = None,
        processor: Any | None = None,
        base_model_path: str | None = None,
    ) -> None:
        assert self.model is not None, "Model is not initialized."
        os.makedirs(path, exist_ok=True)

        if self.bridge_cls == "megatron-bridge":
            if self.config.is_critic:
                raise ValueError(
                    "Saving critic model is not supported with megatron-bridge."
                )
            if self.config.use_lora:
                self.bridge.save_hf_adapter(
                    self.model,
                    path=path,
                    peft_config=self.bridge_lora,
                    base_model_name_or_path=base_model_path or self.config.path,
                )
            else:
                self.bridge.save_hf_pretrained(
                    self.model,
                    path,
                    source_path=base_model_path,
                )
        else:
            if self.mcore_config.use_mbridge_save:
                # when loading model using AreaL's fast hf load, the safetensor_io is never set
                if (
                    not hasattr(self.bridge, "safetensor_io")
                    or self.bridge.safetensor_io is None
                ):
                    self.bridge.safetensor_io = self.bridge._get_safetensor_io(
                        self.config.path
                    )
                self.bridge.save_weights(models=self.model, weights_path=path)
            else:
                save_weights_to_hf_with_mbridge_fast(
                    bridge=self.bridge,
                    models=self.model,
                    weights_path=path,
                    base_model_path=base_model_path,
                    max_shard_size_byte=int(3e9),
                    max_workers=None,
                    fp8_direct_convert=self.fp8_direct_convert,
                )

            if self.config.is_critic:
                save_critic_value_head(self.model, path)

        if dist.get_rank() == 0:
            if tokenizer is not None:
                tokenizer.save_pretrained(path)
            if processor is not None:
                processor.save_pretrained(path)

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def _load_model_from_hf(self, path: str) -> None:
        assert self.model is not None, "Model is not initialized."

        if self.bridge_cls == "megatron-bridge":
            if self.config.is_critic:
                raise ValueError(
                    "Loading critic model is not supported with megatron-bridge."
                )
            # megatron-bridge's load path builds shard-index tensors via
            # ``torch.arange(...)`` to index HF weights that live on CPU. Under
            # the caller's ``with self.device:`` (CUDA) context, those indices
            # become CUDA tensors and the CPU-tensor indexing raises
            # ``RuntimeError: indices should be either on cpu or on the same
            # device as the indexed tensor (cpu)`` — triggered by ChunkedMapping
            # for any model with GDN/Mamba-style conv1d weights (e.g. Qwen3.5).
            # Force CPU as the factory-op default here; tensor data assignment
            # to GPU model params is unaffected (handled by .copy_()).
            with torch.device("cpu"):
                self.bridge.load_hf_weights(self.model, hf_path=path)
        else:
            load_weights_from_hf_with_mbridge_fast(
                bridge=self.bridge,
                models=self.model,
                weights_path=path,
                max_workers=None,
                is_critic=self.config.is_critic,
                fp8_direct_convert=self.fp8_direct_convert,
            )

    def _prepare_mb_list(self, input_: dict[str, Any]) -> MicroBatchList:
        assert "attention_mask" in input_ and "input_ids" in input_
        # Parallel sizes
        pp_size = self.parallel_strategy.pipeline_parallel_size
        cp_size = self.parallel_strategy.context_parallel_size
        tp_size = self.parallel_strategy.tensor_parallel_size
        if self.enable_tree_training:
            assert cp_size == 1, (
                "Context parallelism is not supported in tree training."
            )
            mb_list = build_packed_tree_batch(
                input_,
                mb_spec=self.config.mb_spec,
                pad_to_maximum=self.config.pad_to_maximum,
                dp_group=self.data_parallel_group,
                parallel_size=tp_size,
            )
            recommended_min_n_mbs = 2 * pp_size if pp_size > 1 else 1
            self.logger.info(
                f"Packed tree #microbatch: {len(mb_list)}, microbatch #tokens: {mb_list.group_lens}, "
                f"padded to: {mb_list.padded_to_lengths}, padding lengths: {mb_list.padding_lengths}."
            )
            if len(mb_list) < recommended_min_n_mbs:
                self.logger.warning(
                    f"Number of tree micro-batches ({len(mb_list)}) is less than recommended"
                    f" minimum ({recommended_min_n_mbs}) to avoid pipeline bubbles."
                )
            return mb_list
        # Amend position ids (skip for VLM — model computes mRoPE internally)
        if not self.is_vision_model:
            input_ = amend_position_ids(input_)
        # Split the input into micro-batches
        # NOTE: Here we use 2*pp_size in forward to align logprob precision
        # TODO: Performance check
        min_n_mbs = (
            2 * pp_size if pp_size > 1 else 1
        )  # avoid pipeline bubbles in training
        # NOTE: self.config.mb_spec.max_tokens_per_mb determines
        # the expected **total** number of tokens per micro-batch **in the forward pass**.
        # The micro batch list splitted here will be splitted to each
        # context parallel rank, so the total number of tokens per
        # GPU in a forward pass here will be `max_tokens_per_mb / cp_size`.
        mb_spec = MicroBatchSpec.new(
            self.config.mb_spec,
            n_mbs=max(min_n_mbs, self.config.mb_spec.n_mbs),
            n_mbs_divisor=pp_size,
        )
        mb_list = split_padded_tensor_dict_into_mb_list(
            input_,
            mb_spec,
            group=mpu.get_data_parallel_group(),
        )
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
        # NOTE: Pad micro-batches to:
        # 1. Reduce GPU memory fragmentation, pad actual # tokens per mb to integer multiples
        #  of GPU page size or max_tokens_per_mb
        # 2. Align sequence lengths to integer multiples of `align_to_multiple_of=tp_size*cp_size*2`
        #    to satisfy the requirement of Megatron parallelism.
        align_to_multiple_of = tp_size * cp_size * 2 if cp_size > 1 else tp_size
        align_to_multiple_of = (
            math.lcm(align_to_multiple_of, DEFAULT_VECTORIZED_ALIGNMENT_BYTES)
            if self.enable_fp8
            else align_to_multiple_of
        )
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=self.config.pad_to_maximum,
            seq_align_to=align_to_multiple_of,
        )
        self.logger.info(
            f"#microbatch: {len(mb_list.group_lens)}, microbatch #tokens: {mb_list.group_lens}, "
            f"aligned to: {mb_list.align_to_lengths}, padded to: {mb_list.padded_to_lengths}, "
            f"padding lengths: {mb_list.padding_lengths}."
        )
        # Modern model implementations takes a dict as the input.
        # This eliminates a bug of Qwen2.5-VL for transformers<=4.53.1
        for i, mb in enumerate(mb_list.mbs):
            mb_list.mbs[i] = dict(**mb)
        for i, mb in enumerate(mb_list.padded_mbs):
            mb_list.padded_mbs[i] = dict(**mb)
        for mb in mb_list.mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])
        for mb in mb_list.padded_mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])

        # Extract vision data from multi_modal_input into top-level keys.
        # Vision tensors are placed only on padded_mb (forward side); mb (loss
        # side) gets multimodal payloads stripped. Also rebind mb_list.data to
        # a filtered copy so multimodal references are released from the
        # MicroBatchList without mutating the caller's input dict (which may
        # be reused across forward calls — see save/load round-trip test).
        if self.is_vision_model:
            for mb, padded_mb in zip(mb_list.mbs, mb_list.padded_mbs):
                extract_vision_from_multi_modal(mb, padded_mb)
            mb_list.data = {
                k: v
                for k, v in mb_list.data.items()
                if not _is_multi_modal_payload_key(k)
            }

        return mb_list

    def _compute_logprobs_and_loss(
        self,
        output: torch.Tensor,
        inputs: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        total_loss_weight: torch.Tensor,
        loss_multiplier: float = 1.0,
    ) -> torch.Tensor:
        local_weight = loss_weight_fn(inputs)
        if local_weight == 0:
            return output.mean() * 0.0

        if self.config.is_critic and self.enable_tree_training:
            raise NotImplementedError(
                "Tree training with critic model is not supported yet."
            )
        if not self.config.is_critic:
            if self.enable_tree_training:
                # Handle dummy trie (empty tree for DP synchronization)
                # When trie has no sequences, return zero loss with grad connection
                trie_node = inputs.get("trie_node")
                if trie_node is None or not trie_node.all_sequence_ids:
                    # Return zero loss that maintains gradient connection to output
                    # This ensures backward() works correctly for distributed synchronization
                    return output.mean() * 0.0

                # For tree training, use gather_packed_tree_vocab_stats to properly
                # unpack vocab stats from tree structure back to per-sequence format.
                # This is necessary because the logits are in packed tree format where
                # multiple sequences share prefix positions.
                vocab_min_logits, vocab_max_logits = gather_packed_tree_vocab_stats(
                    output, trie_node
                )
                logprobs, entropy = gather_packed_tree_logprobs_entropy(
                    output,
                    trie_node,
                    inputs["input_ids"],
                    temperature=self.config.temperature,
                    tp_group=mpu.get_tensor_model_parallel_group()
                    if mpu.get_tensor_model_parallel_world_size() > 1
                    else None,
                )
            else:
                cp_local_labels = inputs.get("_cp_local_labels")
                cp_padded_cu_seqlens = inputs.get("_cp_padded_cu_seqlens")
                if cp_local_labels is not None:
                    labels = cp_local_labels
                else:
                    labels = torch.roll(inputs["input_ids"], shifts=-1, dims=-1)
                logprobs, entropy = gather_logprobs_entropy(
                    output,
                    labels,
                    temperature=self.config.temperature,
                    tp_group=mpu.get_tensor_model_parallel_group()
                    if mpu.get_tensor_model_parallel_world_size() > 1
                    else None,
                )
                vocab_min_logits = output.detach().min(-1).values.float()
                vocab_max_logits = output.detach().max(-1).values.float()
                if cp_padded_cu_seqlens is not None:
                    logprobs = reassemble_cp_packed_logprobs(
                        logprobs, cp_padded_cu_seqlens
                    )
                    entropy = reassemble_cp_packed_logprobs(
                        entropy, cp_padded_cu_seqlens
                    )
                    vocab_min_logits = reassemble_cp_packed_logprobs(
                        vocab_min_logits, cp_padded_cu_seqlens
                    )
                    vocab_max_logits = reassemble_cp_packed_logprobs(
                        vocab_max_logits, cp_padded_cu_seqlens
                    )
                    cp_padding_length = inputs.get("_cp_padding_length", 0)
                    cp_old_cu_seqlens = inputs.get("_cp_old_cu_seqlens")
                    logprobs = unpad_logits(
                        logprobs,
                        cp_padding_length,
                        cp_padded_cu_seqlens,
                        cp_old_cu_seqlens,
                    )
                    entropy = unpad_logits(
                        entropy,
                        cp_padding_length,
                        cp_padded_cu_seqlens,
                        cp_old_cu_seqlens,
                    )
                    vocab_min_logits = unpad_logits(
                        vocab_min_logits,
                        cp_padding_length,
                        cp_padded_cu_seqlens,
                        cp_old_cu_seqlens,
                    )
                    vocab_max_logits = unpad_logits(
                        vocab_max_logits,
                        cp_padding_length,
                        cp_padded_cu_seqlens,
                        cp_old_cu_seqlens,
                    )
                    inputs = {
                        k: v for k, v in inputs.items() if not k.startswith("_cp_")
                    }
            loss = loss_fn(
                logprobs,
                entropy,
                inputs,
                vocab_min_logits=vocab_min_logits,
                vocab_max_logits=vocab_max_logits,
            )
        else:
            values = output.squeeze(-1)
            loss = loss_fn(values, inputs)

        loss_scale = local_weight / total_loss_weight * loss_multiplier
        return loss * loss_scale

    def _compute_forward_result(
        self,
        output: torch.Tensor,
        inputs: dict[str, Any],
    ) -> torch.Tensor | dict[int, torch.Tensor]:
        if self.config.is_critic and self.enable_tree_training:
            raise NotImplementedError(
                "Tree training with critic model is not supported yet."
            )
        if not self.config.is_critic:
            if self.enable_tree_training:
                logprobs = _gather_packed_tree_logprobs(
                    output,
                    inputs["trie_node"],
                    inputs["input_ids"],
                    temperature=self.config.temperature,
                    tp_group=mpu.get_tensor_model_parallel_group()
                    if mpu.get_tensor_model_parallel_world_size() > 1
                    else None,
                )
                return logprobs
            labels = torch.roll(inputs["input_ids"], shifts=-1, dims=-1)
            logprobs = gather_logprobs(
                output,
                labels,
                temperature=self.config.temperature,
                tp_group=mpu.get_tensor_model_parallel_group()
                if mpu.get_tensor_model_parallel_world_size() > 1
                else None,
            )
            return logprobs
        else:
            values = output.squeeze(-1)
            return values


# =============================================================================
# Algorithm-specific Megatron Engines
# =============================================================================


class MegatronPPOActor(MegatronEngine):
    """PPO Actor implementation using Megatron backend."""

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


class MegatronPPOCritic(MegatronEngine):
    """PPO Critic implementation using Megatron backend."""

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


class MegatronLMEngine(MegatronEngine):
    """Language model engine for SFT using Megatron backend."""

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


class MegatronRWEngine(MegatronEngine):
    """Reward model engine using Megatron backend."""

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


class MegatronDPOEngine(MegatronEngine):
    """DPO training engine using Megatron backend."""

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
