# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0
"""v1 facade for AWEX colocated Megatron weight transfer."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from areal.engine.weight_update.awex.megatron import MegatronColocateBackend
from areal.engine.weight_update.awex.protocol import ColocateKeyspace
from areal.utils.logging import getLogger

if TYPE_CHECKING:
    from areal.engine.megatron_engine import MegatronEngine

logger = getLogger("AwexColocate")


def resolve_physical_gpu_id(relative_gpu_id: int) -> int:
    """Map a CUDA-masked relative device index to its physical GPU id."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible:
        return relative_gpu_id
    try:
        gpu_ids = [int(x) for x in cuda_visible.split(",") if x.strip()]
        return gpu_ids[relative_gpu_id]
    except (ValueError, IndexError):
        return relative_gpu_id


def awex_colocate_timeout_s(default: float = 1800.0) -> float:
    value = os.environ.get("AWEX_COLOCATE_TIMEOUT_S", "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(
            "Invalid AWEX_COLOCATE_TIMEOUT_S=%r; using default %.1fs",
            value,
            default,
        )
        return default


class AwexMegatronAdapter:
    """v1 orchestration facade over the shared Megatron colocate backend."""

    def __init__(self, engine: MegatronEngine):
        self._engine = engine
        self._colocate_backend = MegatronColocateBackend(
            engine,
            physical_gpu_id_resolver=lambda: resolve_physical_gpu_id(
                torch.cuda.current_device()
            ),
            normalize_infer_hf_config=False,
            allow_hdo_optimizer_offload=True,
        )
        # Keep the existing private state handles available to engine hooks and
        # downstream integrations while the backend owns their lifetime.
        self._offloaded_weights = self._colocate_backend.offloaded_weights
        self._released_tags = self._colocate_backend.released_tags
        self._meta_server_addr: str | None = None
        self._meta_server_client = None
        self._transfer_rank: int | None = None
        self._timeout_s = awex_colocate_timeout_s()

    def init_colocate_weight_update(
        self,
        meta_server_addr: str | None = None,
        pair_name: str = "default",
        transfer_rank: int = 0,
        timeout_s: float | None = None,
    ) -> None:
        """Initialize MetaServer connection; conversion remains lazy."""
        del pair_name
        from awex.meta.meta_server import MetaServerClient, start_meta_server

        if not meta_server_addr:
            meta_server_addr = os.environ.get("AWEX_META_SERVER_ADDR", "")
        if not meta_server_addr:
            host, port = start_meta_server()
            meta_server_addr = f"{host}:{port}"
            os.environ["AWEX_META_SERVER_ADDR"] = meta_server_addr
            logger.info("Started MetaServer at %s", meta_server_addr)

        host, port = meta_server_addr.rsplit(":", 1)
        self._meta_server_client = MetaServerClient(host, int(port))
        self._meta_server_addr = meta_server_addr
        self._transfer_rank = transfer_rank
        self._timeout_s = awex_colocate_timeout_s() if timeout_s is None else timeout_s
        self._colocate_backend.configure(
            meta_server_client=self._meta_server_client,
            timeout_s=self._timeout_s,
        )
        if dist.get_rank() == 0:
            self._meta_server_client.put_object(
                ColocateKeyspace.AWEX_TRAIN_INFO,
                {"train_world_size": dist.get_world_size()},
            )
            logger.info(
                "Registered awex_train_info (train_world_size=%d) with MetaServer",
                dist.get_world_size(),
            )

        logger.info(
            "AwexMegatronAdapter initialized: meta_server=%s, transfer_rank=%d",
            meta_server_addr,
            transfer_rank,
        )

    def _lazy_initialize(self) -> None:
        self._colocate_backend.lazy_initialize()

    def _release_grad_memory(self) -> None:
        self._colocate_backend.release_grad_memory()

    @torch.no_grad()
    def execute_colocate_weight_update(self, version: int) -> None:
        self._colocate_backend.execute_weight_update(
            version,
            publish_offloaded_before_payload=True,
            restore_initial_weight_state=False,
            collect_ipc_after_update=False,
            wrap_reader_timeout=False,
        )

    def finish_colocate_weight_update(self, training_world_size: int) -> None:
        """Preserve the v1 finish wait and collective order exactly."""
        del training_world_size
        num_infer_engines = self._colocate_backend.num_infer_engines
        logger.info(
            "Waiting for %d inference engine(s) to signal "
            "finished_weights_update_engines",
            num_infer_engines,
        )
        self._meta_server_client.wait_set_until_size(
            ColocateKeyspace.FINISHED_WEIGHT_UPDATE_ENGINES,
            num_infer_engines,
            timeout=self._timeout_s,
        )
        logger.info("All inference engines finished weights update")

        dist.barrier(group=self._engine.cpu_group)

        if dist.get_rank() == 0:
            self._meta_server_client.delete_if_exists(
                ColocateKeyspace.FINISHED_WEIGHT_UPDATE_ENGINES
            )
            self._meta_server_client.delete_if_exists(
                ColocateKeyspace.ALL_TRAINING_OFFLOADED_WEIGHTS
            )
        logger.info("Cleaned up MetaServer coordination keys")

    @torch.no_grad()
    def _convert_parameters(self) -> dict[str, torch.Tensor]:
        return self._colocate_backend.convert_parameters()

    def release_memory(self, tags: list[str] | None = None) -> None:
        self._colocate_backend.release_memory(tags)

    def resume_memory(self, tags: list[str] | None = None) -> None:
        self._colocate_backend.resume_memory(tags)

    def _offload_model_weights(self) -> None:
        self._colocate_backend._offload_model_weights()

    def _reload_model_weights(self, load_grad: bool = False) -> None:
        self._colocate_backend._reload_model_weights(load_grad)

    def ensure_grad_buffers(self) -> None:
        self._colocate_backend.ensure_grad_buffers()

    def _get_inner_optimizers(self):
        return self._colocate_backend._get_inner_optimizers()

    def _offload_optimizer_states(self) -> None:
        self._colocate_backend._offload_optimizer_states()

    def _reload_optimizer_states(self) -> None:
        self._colocate_backend._reload_optimizer_states()


__all__ = [
    "AwexMegatronAdapter",
    "awex_colocate_timeout_s",
    "resolve_physical_gpu_id",
]
