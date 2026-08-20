# SPDX-License-Identifier: Apache-2.0

"""Legacy facade for the controller-independent SGLang colocate backend."""

from __future__ import annotations

from typing import Any

from areal.engine.weight_update.awex.colocate_protocol import (
    ColocateKeyspace,
    ColocateTopology,
)
from areal.engine.weight_update.awex.colocate_sglang import (
    SGLangColocateBackend,
    SingleInstanceMetaResolver,
    get_awex_infer_hf_config,
    get_router_dtype,
)
from areal.utils.logging import getLogger

logger = getLogger("AwexColocateReader")

# Preserve the legacy helper imports while their implementation lives in the
# controller-independent package.
_get_awex_infer_hf_config = get_awex_infer_hf_config
_get_router_dtype = get_router_dtype


class AwexColocateReader:
    """Bind v1 plugin orchestration to the shared SGLang data plane."""

    def __init__(self, scheduler: Any):
        self._scheduler = scheduler
        self._backend = SGLangColocateBackend(scheduler)
        self._local_gpu_id: int | None = None
        self._released_tags: set[str] = set()

    def get_parallelism(self) -> dict:
        return self._backend.get_parallelism()

    def get_weight_metadata(self):
        return self._backend.get_weight_metadata()

    def initialize(
        self,
        meta_server_addr: str,
        transfer_rank: int,
        infer_world_size: int,
        train_world_size: int,
        local_gpu_id: int,
        timeout_s: float = 300.0,
    ) -> None:
        """Publish inference metadata; native reader creation stays lazy."""
        del timeout_s
        server_args = self._scheduler.server_args
        tp_size = int(getattr(server_args, "tp_size", 1))
        pp_size = int(getattr(server_args, "pp_size", 1))
        topology = ColocateTopology(
            transfer_rank=transfer_rank,
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            instance_world_size=max(1, tp_size * pp_size),
        )
        model = self._scheduler.tp_worker.model_runner.model
        model_runner = self._scheduler.tp_worker.model_runner
        self._backend.initialize(
            meta_server_addr=meta_server_addr,
            topology=topology,
            infer_hf_config=get_awex_infer_hf_config(model, model_runner),
            router_dtype=get_router_dtype(model.config),
            publish_infer_params_meta=False,
        )
        self._local_gpu_id = local_gpu_id
        logger.info(
            "Eager init done: transfer_rank=%d, local_gpu_id=%d, infer_world_size=%d",
            transfer_rank,
            local_gpu_id,
            infer_world_size,
        )

    def update_weights(self, version: int) -> None:
        self._backend.update_weights(version)

    def release_memory(self, tags: list[str] | None = None) -> None:
        from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput

        tags = tags or ["kv_cache"]
        native_tags = [tag for tag in tags if tag not in self._released_tags]
        if native_tags:
            req = ReleaseMemoryOccupationReqInput(tags=native_tags)
            self._scheduler.release_memory_occupation(req)
            self._released_tags.update(native_tags)
        logger.info("release_memory: tags=%s", tags)

    def resume_memory(self, tags: list[str] | None = None) -> None:
        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput

        tags = tags or ["kv_cache"]
        resume_tags = [tag for tag in tags if tag in self._released_tags]
        if resume_tags:
            req = ResumeMemoryOccupationReqInput(tags=resume_tags)
            self._scheduler.resume_memory_occupation(req)
            self._released_tags.difference_update(resume_tags)
        logger.info("resume_memory: tags=%s", tags)

    def wait_for_training_offloaded(self, version: int) -> None:
        """Wait for v1 writer memory handoff before receiving weights."""
        del version
        from areal.engine.weight_update.awex.v1_megatron_adapter import (
            awex_colocate_timeout_s,
        )

        topology = self._backend.topology
        self._backend.meta_server_client.wait_set_until_size(
            ColocateKeyspace.ALL_TRAINING_OFFLOADED_WEIGHTS,
            topology.train_world_size,
            timeout=awex_colocate_timeout_s(),
        )

    def wait_for_weights_ready(
        self, version: int, timeout_s: float | None = None
    ) -> None:
        """Wait for this physical GPU's versioned IPC payload key."""
        from awex.util.common import get_ip_address

        from areal.engine.weight_update.awex.v1_megatron_adapter import (
            awex_colocate_timeout_s,
        )

        if self._local_gpu_id is None:
            raise RuntimeError("AwexColocateReader is not initialized")
        keyspace = ColocateKeyspace(get_ip_address(), self._local_gpu_id)
        self._backend.meta_server_client.wait_key(
            keyspace.serialized_weights(version),
            timeout=awex_colocate_timeout_s() if timeout_s is None else timeout_s,
        )

    def signal_finished_weights_update(self) -> None:
        """Signal once per inference engine after v1 finishes an update."""
        topology = self._backend.topology
        if topology.instance_local_rank != 0:
            return
        self._backend.meta_server_client.add_object_to_set(
            ColocateKeyspace.FINISHED_WEIGHT_UPDATE_ENGINES,
            topology.engine_rank,
        )

    def teardown(self) -> None:
        self._backend.teardown()


__all__ = [
    "AwexColocateReader",
    "SingleInstanceMetaResolver",
    "_get_awex_infer_hf_config",
    "_get_router_dtype",
]
