# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel  # pyright: ignore[reportMissingImports]


@dataclass
class WeightUpdateConfig:
    """Configuration for the weight update service."""

    # Gateway
    host: str = "0.0.0.0"
    gateway_port: int = 7080

    # Authentication
    admin_api_key: str = "areal-admin-key"

    log_level: str = "warning"

    # NCCL
    comm_backend: str = "nccl"  # "nccl" or "hccl" (NPU)

    # Plan derivation
    enable_debug_mode: bool = False  # Dump plans to JSON, verbose logging

    # Timeouts
    init_timeout_s: int = 300  # NCCL group init timeout
    update_timeout_s: int = 120  # Per-step weight update timeout

    # Performance
    use_batch_isend_irecv: bool = True  # Use dist.batch_isend_irecv


class WeightUpdateResult(BaseModel):
    """Result of a weight update operation."""

    status: str  # "ok" or "error"
    version: int
    duration_ms: float
    error: str | None = None


@dataclass
class PairInfo:
    pair_name: str
    train_worker_urls: list[str]
    inference_worker_urls: list[str]
    train_world_size: int = 0
    inference_world_size: int = 0
    master_addr: str = ""
    master_port: int = 0
    last_version: int = 0

    # Disk-mode fields (used when mode="disk")
    mode: str = "awex"  # "awex" or "disk"
    save_path: str = ""
    use_lora: bool = False
    lora_name: str = ""

    # Colocated mode (training and inference share GPUs)
    colocate: bool = False

    def __post_init__(self):
        if not self.pair_name:
            raise ValueError("pair_name must not be empty")
        if self.mode not in ("awex", "disk"):
            raise ValueError(f"mode must be 'awex' or 'disk', got '{self.mode}'")
