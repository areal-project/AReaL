# SPDX-License-Identifier: Apache-2.0

import copy
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL.Image import Image as ImageObject
from transformers import PreTrainedTokenizerFast

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import GenerationHyperparameters
from areal.infra.platforms import current_platform
from areal.utils import logging

if TYPE_CHECKING:
    from transformers import AutoProcessor

logger = logging.getLogger("IOStruct")


@dataclass
class ModelRequest:
    rid: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_ids: list[int] = field(default_factory=list)
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    # tokenizer is used for encode-decode in the inference engine
    tokenizer: PreTrainedTokenizerFast | None = None

    # vlm
    image_data: list[str] | None = field(default_factory=list)
    processor: Optional["AutoProcessor"] = None

    # vlm+vllm:
    vision_msg_vllm: list | None = None

    def copy(self):
        return ModelRequest(
            rid=self.rid,
            input_ids=self.input_ids.copy(),
            gconfig=self.gconfig.new(),
            metadata=self.metadata.copy(),
            tokenizer=self.tokenizer,
            image_data=self.image_data.copy() if self.image_data is not None else None,
            processor=self.processor,
            vision_msg_vllm=(
                self.vision_msg_vllm.copy()
                if self.vision_msg_vllm is not None
                else None
            ),
        )


@dataclass
class ModelResponse:
    # outputs
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    output_logprobs: list[float] = field(default_factory=list)
    output_versions: list[int] = field(default_factory=list)
    stop_reason: Literal["length", "stop", "tool_calls", "abort"] = "stop"
    # tokenizer is used for encode-decode in the inference engine
    tokenizer: PreTrainedTokenizerFast | None = None

    # vlm
    input_images: list[ImageObject | str] = field(default_factory=list)
    processor: Optional["AutoProcessor"] = None

    # statistics
    latency: float = float("inf")
    ttft: float = float("inf")  # Time to first token
    itl: list[float] = field(default_factory=list)  # List of inter-token latencies

    # MoE routing (only populated when return_routed_experts=True)
    routed_experts: np.ndarray | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_tokens)

    @property
    def output_len(self) -> int:
        return len(self.output_tokens)

    @property
    def end_with_stop(self) -> bool:
        if self.tokenizer is None:
            raise ValueError("tokenizer is None, cannot check end_with_stop")
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        if len(self.output_tokens) == 0:
            return False
        last_token = self.output_tokens[-1]
        return (eos_id is not None and last_token == eos_id) or (
            pad_id is not None and last_token == pad_id
        )

    @property
    def output_tokens_without_stop(self) -> list[int]:
        if self.tokenizer is None:
            raise ValueError("tokenizer is None, cannot get output_tokens_without_stop")
        if self.stop_reason not in ["length", "abort"] and self.output_tokens:
            if not self.end_with_stop:
                raise ValueError(
                    f"output_tokens does not end with eos or pad token, it ends with {self.output_tokens[-1]}, but stop_reason is {self.stop_reason}"
                )
            pad_or_eos_len = 0
            eos_id = self.tokenizer.eos_token_id
            pad_id = self.tokenizer.pad_token_id
            stop_tokens = {eos_id, pad_id}
            stop_tokens.discard(None)
            for tok in reversed(self.output_tokens):
                if tok in stop_tokens:
                    pad_or_eos_len += 1
                else:
                    break
            if pad_or_eos_len == len(self.output_tokens):
                raise ValueError(
                    "All output_tokens are EOS or PAD tokens; cannot strip stop tokens without removing entire output."
                )
            return self.output_tokens[:-pad_or_eos_len]
        return self.output_tokens


@dataclass
class FinetuneSpec:
    total_train_epochs: int
    dataset_size: int
    train_batch_size: int

    @property
    def total_train_steps(self):
        # assuming drop_last
        return self.total_train_epochs * (self.dataset_size // self.train_batch_size)

    @property
    def steps_per_epoch(self):
        return self.dataset_size // self.train_batch_size


@dataclass
class ParamSpec:
    name: str
    shape: tuple
    dtype: str

    @property
    def size(self) -> int:
        """Param bytes"""
        return getattr(torch, self.dtype).itemsize * np.prod(self.shape)


def get_versioned_lora_name(lora_name: str, version: int) -> str:
    """Get versioned LoRA adapter name (e.g., 'lora-v1')."""
    return f"{lora_name}-v{version}"


def detect_image_mime(base64_data: str) -> str:
    """Detect image MIME type from the first bytes of base64-encoded data.

    Examines base64 magic byte prefixes to determine the actual image format.
    """
    if base64_data.startswith("iVBOR"):  # PNG: \x89PNG
        return "image/png"
    if base64_data.startswith("/9j/"):  # JPEG: \xff\xd8\xff
        return "image/jpeg"
    if base64_data.startswith("R0lGOD"):  # GIF: GIF8
        return "image/gif"
    if base64_data.startswith("UklGR"):  # WebP: RIFF
        return "image/webp"
    return "image/jpeg"


@dataclass
class WeightUpdateMeta:
    type: Literal["disk", "xccl", "awex"]
    path: str | None = None
    gen_allocation: ModelAllocation | None = None

    nccl_master_address: str | None = None
    nccl_master_port: int | None = None
    nccl_group_name: str | None = None
    weight_chunked_mem_mb: int = 1024

    use_lora: bool = False
    lora_name: str = ""
    lora_int_id: int = 0
    base_model_name: str = ""
    peft_config: dict = field(default_factory=dict)

    clear_checkpoint_after_load: bool = True

    version: int | None = None

    def with_version(self, version: int) -> "WeightUpdateMeta":
        """Return a copy of this meta with versioned path.

        Changes path from 'weight_update' to 'weight_update_v{version}'.
        """
        if version < 0:
            raise ValueError(f"version must be non-negative, got {version}")
        new_meta = copy.copy(self)
        new_meta.version = version
        if self.path is not None:
            base_dir = os.path.dirname(self.path)
            new_meta.path = os.path.join(base_dir, f"weight_update_v{version}")
        return new_meta

    @classmethod
    def from_disk(
        cls,
        experiment_name: str,
        trial_name: str,
        file_root: str,
        name: str = "default",
        use_lora: bool = False,
        clear_checkpoint_after_load: bool = True,
        lora_name: str = "",
        lora_int_id: int = 1,
        base_model_name: str = "",
    ) -> "WeightUpdateMeta":
        from areal.utils.saver import Saver

        path = os.path.join(
            Saver.get_model_save_root(experiment_name, trial_name, file_root, name),
            "weight_update",
        )
        return cls(
            type="disk",
            path=path,
            use_lora=use_lora,
            clear_checkpoint_after_load=clear_checkpoint_after_load,
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            base_model_name=base_model_name,
        )

    @classmethod
    def from_megatron_xccl(
        cls,
        gen_allocation: ModelAllocation,
        weight_chunked_mem_mb: int = 1024,
        use_lora: bool = False,
        lora_name: str = "",
        lora_int_id: int = 1,
        base_model_name: str = "",
    ):
        return cls(
            type="xccl",
            gen_allocation=gen_allocation,
            weight_chunked_mem_mb=weight_chunked_mem_mb,
            use_lora=use_lora,
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            base_model_name=base_model_name,
        )

    @classmethod
    def from_fsdp_xccl(
        cls,
        gen_allocation: ModelAllocation,
        weight_chunked_mem_mb: int = 1024,
        use_lora: bool = False,
        lora_name: str = "",
        lora_int_id: int = 1,
        base_model_name: str = "",
    ):
        return cls(
            type="xccl",
            gen_allocation=gen_allocation,
            weight_chunked_mem_mb=weight_chunked_mem_mb,
            use_lora=use_lora,
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            base_model_name=base_model_name,
        )

    @classmethod
    def from_awex(
        cls,
        use_lora: bool = False,
        lora_name: str = "",
        lora_int_id: int = 1,
        base_model_name: str = "",
    ):
        return cls(
            type="awex",
            use_lora=use_lora,
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            base_model_name=base_model_name,
        )


@dataclass
class HttpRequest:
    """Represents an HTTP request to be sent to a remote inference server."""

    endpoint: str
    payload: dict[str, Any]
    method: str = "POST"


@dataclass
class HttpGenerationResult:
    """Parsed result from a generation response."""

    output_tokens: list[int]
    output_logprobs: list[float]
    stop_reason: str
    routed_experts: np.ndarray | None = None


@dataclass
class WeightUpdateRequests:
    """Collection of HTTP requests needed for a weight update operation."""

    requests: list[HttpRequest]


@dataclass
class SaveLoadMeta:
    path: str
    weight_format: str
    with_optim: bool
    tokenizer: PreTrainedTokenizerFast | None = None
    processor: Optional["AutoProcessor"] = None
    base_model_path: str | None = None
    naive_distributed: bool = False


@dataclass
class RolloutStat:
    accepted: int = 0
    enqueued: int = 0
    rejected: int = 0
    running: int = 0


@dataclass
class StepInfo:
    epoch: int
    epoch_step: int
    global_step: int
    steps_per_epoch: int

    def next(self):
        return StepInfo(
            epoch=self.epoch + (self.epoch_step == self.steps_per_epoch - 1),
            epoch_step=(
                0
                if self.epoch_step == self.steps_per_epoch - 1
                else self.epoch_step + 1
            ),
            global_step=self.global_step + 1,
            steps_per_epoch=self.steps_per_epoch,
        )


@dataclass
class LocalInfServerInfo:
    """Information about a locally launched inference server."""

    host: str
    port: int
    process: subprocess.Popen | None


@dataclass
class DeviceRuntimeInfo:
    mem_allocated: float
    mem_reserved: float
    mem_used: float
    mem_total: float
    unit: str

    @classmethod
    def get_current(cls, unit: str = "GB"):
        unit_divisors = {"GB": 1024**3, "MB": 1024**2, "KB": 1024}
        if unit not in unit_divisors:
            raise ValueError(
                f"Unsupported unit '{unit}'. Must be one of {list(unit_divisors.keys())}."
            )
        divisor = unit_divisors[unit]

        mem_allocated = current_platform.memory_allocated()
        mem_reserved = current_platform.memory_reserved()
        mem_free, mem_total = current_platform.mem_get_info()
        mem_used = mem_total - mem_free
        return cls(
            mem_allocated=mem_allocated / divisor,
            mem_reserved=mem_reserved / divisor,
            mem_used=mem_used / divisor,
            mem_total=mem_total / divisor,
            unit=unit,
        )

    def log(self, head: str = "", rank: int = 0, precision: int = 2):
        mem_allocated = f"{self.mem_allocated:.{precision}f}"
        mem_reserved = f"{self.mem_reserved:.{precision}f}"
        mem_used = f"{self.mem_used:.{precision}f}"
        mem_total = f"{self.mem_total:.{precision}f}"
        if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
            logger.info(
                f"Memory-Usage {head}: "
                f"memory allocated ({self.unit}): {mem_allocated}, "
                f"memory reserved ({self.unit}): {mem_reserved}, "
                f"device memory used/total ({self.unit}): {mem_used}/{mem_total}"
            )
