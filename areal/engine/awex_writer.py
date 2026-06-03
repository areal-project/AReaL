# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from areal.utils import logging

logger = logging.getLogger(__name__)


class AwexMegatronWriterAdapter:
    """Adapter that exposes AReaL MegatronEngine to Awex writer API."""

    def __init__(self, engine, meta):
        if getattr(meta, "use_mindspeed", False) or (meta.comm_backend == "hccl"):
            # Ensure MindSpeed patches are enabled before Awex imports Megatron.
            os.environ.setdefault("AWEX_USE_MINDSPEED", "1")
        try:
            from awex.writer.weights_writer import get_weights_exchange_writer
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "Awex is not available. Install awex or ensure it is on PYTHONPATH."
            ) from exc

        self._engine = engine
        self._get_writer = get_weights_exchange_writer
        self.weights_exchange_writer = None

        self.hf_config = engine.hf_config
        self.model = engine.model
        self.engine_name = "mcore"
        self.global_step = -1

        self.meta_server_addr = meta.meta_server_addr or ""
        self.comm_backend = meta.comm_backend or "file"
        self.enable_debug_mode = meta.enable_debug_mode
        self.enable_colocate_mode = meta.enable_colocate_mode

        self.config = {
            "weights_validation_steps": meta.weights_validation_steps,
            "validate_weights_every_n_steps": meta.validate_weights_every_n_steps,
            "dump_weights_list_for_validation": meta.dump_weights_list_for_validation,
            "dump_weights_dir_for_validation": meta.dump_weights_dir_for_validation,
            "disable_weights_exchange_pipeline": meta.disable_weights_exchange_pipeline,
            "debug_mode_config": meta.debug_mode_config,
        }

        self._export_meta_server_env(self.meta_server_addr)

    def _export_meta_server_env(self, meta_server_addr: str) -> None:
        os.environ["AWEX_META_SERVER_ADDR"] = meta_server_addr or ""
        ip, port = (meta_server_addr or ":").split(":")
        os.environ["AWEX_META_SERVER_IP"] = ip
        os.environ["AWEX_META_SERVER_PORT"] = port

    def initialize(self) -> None:
        if self.weights_exchange_writer is not None:
            return
        self.weights_exchange_writer = self._get_writer(self)
        self.weights_exchange_writer.initialize()
        if self.enable_colocate_mode:
            self.release_memory_occupation()

    def set_global_step(self, global_step: int) -> None:
        # Awex writer uses this for logging and synchronization metadata.
        self.global_step = global_step

    def write_weights(self, **kwargs) -> None:
        if self.weights_exchange_writer is None:
            raise RuntimeError("Awex writer not initialized.")
        self.weights_exchange_writer.write_weights(step_id=self.global_step, **kwargs)
        if self.enable_colocate_mode:
            self.release_memory_occupation()

    def save_hf_checkpoint(self, path: str) -> None:
        self._engine._save_model_to_hf(path)

    def release_memory_occupation(self, tags: list[str] | None = None) -> None:
        if self.enable_colocate_mode:
            logger.warning(
                "Awex colocate mode requested, but MegatronEngine does not "
                "support fine-grained memory release. No-op."
            )

    def resume_memory_occupation(self, tags: list[str] | None = None) -> None:
        if self.enable_colocate_mode:
            logger.warning(
                "Awex colocate mode requested, but MegatronEngine does not "
                "support fine-grained memory resume. No-op."
            )

    def release_grad_memory(self, empty_cache: bool = True) -> None:
        # Placeholder to satisfy Awex TrainingEngine API.
        return None
