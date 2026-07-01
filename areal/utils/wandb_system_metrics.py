# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
from typing import Any

from areal.api.cli_args import BaseExperimentConfig
from areal.infra.rpc.serialization import deserialize_value
from areal.utils import logging

logger = logging.getLogger("WandBSystemMetrics", "system")

_worker_wandb_run: Any | None = None
_worker_wandb_lock = threading.Lock()
_SERVICE_ROLE_SUFFIXES = ("-data", "-guard")


def prepare_wandb_run_identity(config: BaseExperimentConfig | None) -> None:
    """Freeze the W&B run id on the controller so workers resolve the same one."""
    if config is None:
        return
    if not config.stats_logger.wandb.system_metrics.enabled:
        return
    from areal.utils.stats_logger import resolve_wandb_run_id

    resolve_wandb_run_id(config.stats_logger)


def worker_system_metrics_enabled(
    config: BaseExperimentConfig,
    role: str | None,
) -> bool:
    wandb_config = config.stats_logger.wandb
    system_metrics_config = wandb_config.system_metrics
    if not system_metrics_config.enabled:
        return False
    if wandb_config.mode == "disabled":
        return False
    if wandb_config.mode != "shared":
        raise ValueError(
            "stats_logger.wandb.system_metrics.enabled requires "
            "stats_logger.wandb.mode='shared'."
        )
    if role is None or role.endswith(_SERVICE_ROLE_SUFFIXES):
        return False
    roles = system_metrics_config.roles
    return roles is None or role in roles


def init_worker_wandb_system_metrics(
    config: BaseExperimentConfig,
    role: str | None,
    rank: int,
) -> bool:
    global _worker_wandb_run

    if not worker_system_metrics_enabled(config, role):
        return False

    with _worker_wandb_lock:
        if _worker_wandb_run is not None:
            return False

        import wandb

        from areal.utils.stats_logger import StatsLogger, resolve_wandb_run_id

        stats_config = config.stats_logger
        wandb_config = stats_config.wandb
        system_metrics_config = wandb_config.system_metrics

        if wandb_config.wandb_base_url:
            os.environ["WANDB_BASE_URL"] = wandb_config.wandb_base_url
        if wandb_config.wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_config.wandb_api_key

        settings_kwargs: dict[str, Any] = {
            "mode": "shared",
            "x_primary": False,
            "x_label": f"{role or 'worker'}-{rank}",
            "x_update_finish_state": False,
        }
        if system_metrics_config.gpu_device_ids is not None:
            settings_kwargs["x_stats_gpu_device_ids"] = list(
                system_metrics_config.gpu_device_ids
            )

        try:
            _worker_wandb_run = wandb.init(
                mode=wandb_config.mode,
                entity=wandb_config.entity,
                project=wandb_config.project or stats_config.experiment_name,
                name=wandb_config.name or stats_config.trial_name,
                job_type=wandb_config.job_type,
                group=wandb_config.group
                or f"{stats_config.experiment_name}_{stats_config.trial_name}",
                notes=wandb_config.notes,
                tags=wandb_config.tags,
                dir=StatsLogger.get_log_path(stats_config),
                force=True,
                id=resolve_wandb_run_id(stats_config),
                resume="allow",
                settings=wandb.Settings(**settings_kwargs),
            )
        except Exception as exc:  # noqa: BLE001
            _worker_wandb_run = None
            logger.warning(
                "Failed to start worker W&B system metrics client "
                "(role=%s rank=%s): %s",
                role,
                rank,
                exc,
            )
            return False

        logger.info(
            "Initialized worker W&B system metrics client for role=%s rank=%s.",
            role,
            rank,
        )
        return True


def finish_worker_wandb_system_metrics() -> None:
    global _worker_wandb_run
    with _worker_wandb_lock:
        if _worker_wandb_run is None:
            return

        run = _worker_wandb_run
        try:
            run.finish()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Worker W&B run finish failed: %s", exc)
        finally:
            _worker_wandb_run = None


def configure_worker_wandb_system_metrics(data: dict) -> dict[str, Any]:
    config_data = data.get("config")
    if config_data is None:
        raise ValueError("Missing 'config' field in request")

    rank = data.get("rank")
    if rank is None:
        raise ValueError("Missing 'rank' field in request")

    config = deserialize_value(config_data)
    role = data.get("role")
    enabled = init_worker_wandb_system_metrics(config, role=role, rank=rank)
    return {"wandb_system_metrics": "enabled" if enabled else "skipped"}


def register_worker_wandb_system_metrics_hooks(state) -> None:
    state.register_configure_hook(configure_worker_wandb_system_metrics)
    state.register_cleanup_hook(finish_worker_wandb_system_metrics)
