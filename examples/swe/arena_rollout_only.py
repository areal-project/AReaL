"""Run Arena Stream rollouts without constructing a training engine."""

from __future__ import annotations

import argparse
import getpass
import os
import random
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import torch

from examples.swe.arena_client import ArenaOpenAPIClient, infer_llm_protocol
from examples.swe.utils import SWEPPOConfig

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import SGLangConfig, load_expr_config
from areal.engine import RemoteSGLangEngine
from areal.infra import LocalScheduler
from areal.infra.rpc.rtensor import RTensor
from areal.utils import logging
from areal.utils.config_utils import redact_sensitive_config

logger = logging.getLogger("ArenaRolloutOnly")


def _trajectory_reward(trajectory: dict[str, Any]) -> float:
    """Extract the episode reward from a localized proxy trajectory."""
    localized = RTensor.localize(trajectory)
    rewards = localized.get("rewards")
    if torch.is_tensor(rewards):
        return float(rewards.sum().item())

    interactions = localized.get("interactions")
    if isinstance(interactions, list):
        return float(
            sum(
                interaction.get("reward", 0.0)
                for interaction in interactions
                if isinstance(interaction, dict)
            )
        )
    raise ValueError("Rollout trajectory contains neither rewards nor interactions")


def _run_rollout_tasks(
    controller: Any,
    rows: list[dict[str, str]],
    workflow_kwargs: dict[str, Any],
) -> tuple[list[tuple[str, float]], list[str]]:
    """Run queued rollouts while retaining row identity for rejected tasks."""
    submitted: list[tuple[dict[str, str], int]] = []
    failed_data_ids: list[str] = []
    for row in rows:
        data_id = row["data_id"]
        try:
            task_id = controller.submit(
                data=row,
                workflow="examples.swe.arena_agent.ArenaStreamAgentWorkflow",
                workflow_kwargs=workflow_kwargs,
                group_size=1,
            )
        except Exception:
            logger.exception("Failed to submit Arena rollout: data_id=%s", data_id)
            failed_data_ids.append(data_id)
            continue
        submitted.append((row, task_id))

    completed: list[tuple[str, float]] = []
    for row, task_id in submitted:
        data_id = row["data_id"]
        try:
            trajectory = controller.wait_for_task(task_id)
            if trajectory is None:
                logger.warning("Arena rollout rejected: data_id=%s", data_id)
                failed_data_ids.append(data_id)
                continue
            reward = _trajectory_reward(trajectory)
        except Exception:
            logger.exception("Failed to collect Arena rollout: data_id=%s", data_id)
            failed_data_ids.append(data_id)
            continue
        completed.append((data_id, reward))
        logger.info("Collected Arena rollout: data_id=%s, reward=%s", data_id, reward)

    return completed, failed_data_ids


def _init_wandb(config: SWEPPOConfig):
    """Initialize the configured W&B run without starting trainer infrastructure."""
    import wandb

    wandb_config = config.stats_logger.wandb
    log_dir = (
        Path(config.cluster.fileroot)
        / "logs"
        / getpass.getuser()
        / config.experiment_name
        / config.trial_name
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        mode=wandb_config.mode,
        entity=wandb_config.entity,
        project=wandb_config.project or config.experiment_name,
        name=wandb_config.name or config.trial_name,
        job_type="rollout-only",
        group=wandb_config.group or f"{config.experiment_name}_{config.trial_name}",
        notes=wandb_config.notes,
        tags=wandb_config.tags,
        config=redact_sensitive_config(asdict(config)),
        dir=str(log_dir),
        id=f"{config.experiment_name}_{config.trial_name}_rollout",
        resume="allow",
    )


def _run_registry_smoke(
    arena_client: ArenaOpenAPIClient,
    proxy_base_url: str,
    proxy_admin_api_key: str,
    trial_name: str,
) -> tuple[str, str]:
    """Create a proxy session and exercise Arena LLM registration lifecycle."""
    suffix = uuid.uuid4().hex[:10]
    model_prefix = "".join(
        character if character.isalnum() else "-" for character in trial_name.lower()
    ).strip("-")[-40:]
    model_name = f"stream-areal-{model_prefix}-{suffix}"
    deployment_id = str(uuid.uuid4())
    registered_model_id = deployment_id
    admin_headers = {"Authorization": f"Bearer {proxy_admin_api_key}"}

    with httpx.Client(timeout=30.0) as proxy_client:
        grant_response = proxy_client.post(
            f"{proxy_base_url.rstrip('/')}/grant_capacity",
            headers=admin_headers,
        )
        grant_response.raise_for_status()
        start_response = proxy_client.post(
            f"{proxy_base_url.rstrip('/')}/rl/start_session",
            headers=admin_headers,
            json={"task_id": f"registry-smoke-{suffix}"},
        )
        start_response.raise_for_status()
        session_api_key = start_response.json()["api_key"]

        try:
            registered_url, registered_model_id = arena_client.register_llm_proxy(
                model_name=model_name,
                upstream_base_url=proxy_base_url,
                upstream_api_key=session_api_key,
                deployment_id=deployment_id,
            )
            logger.info(
                "Arena LLM registration succeeded: model_name=%s, "
                "model_id=%s, registered_url=%s",
                model_name,
                registered_model_id,
                registered_url,
            )
            return registered_url, registered_model_id
        finally:
            active_exception = sys.exc_info()[0] is not None
            delete_error: Exception | None = None
            try:
                arena_client.delete_llm_proxy(registered_model_id)
                logger.info(
                    "Arena LLM registration deleted: model_id=%s",
                    registered_model_id,
                )
            except Exception as exc:
                delete_error = exc
                logger.error(
                    "Failed to delete Arena LLM registration %s: %s",
                    deployment_id,
                    exc,
                )
            finally:
                end_response = proxy_client.post(
                    f"{proxy_base_url.rstrip('/')}/rl/end_session",
                    headers={"Authorization": f"Bearer {session_api_key}"},
                )
                end_response.raise_for_status()
            if delete_error is not None and not active_exception:
                raise delete_error


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--registry-smoke", action="store_true")
    parser.add_argument("--serve-after-smoke-seconds", type=int, default=0)
    rollout_args, remaining = parser.parse_known_args(args)
    if rollout_args.num_rollouts < 1:
        raise ValueError("--num-rollouts must be positive")

    config, _ = load_expr_config(remaining, SWEPPOConfig)
    if config.scheduler.type != "local":
        raise ValueError("arena_rollout_only.py requires scheduler.type=local")

    log_path = (
        Path(config.cluster.fileroot)
        / "logs"
        / getpass.getuser()
        / config.experiment_name
        / config.trial_name
        / "rollout_only.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.setup_file_logging(str(log_path))

    econfig = config.econfig
    arena_client = ArenaOpenAPIClient(
        base_url=econfig.arena_base_url,
        timeout=econfig.arena_request_timeout,
        poll_interval=econfig.arena_poll_interval,
        request_retries=econfig.arena_request_retries,
    )
    stream_id = ""
    selected_rows: list[dict[str, str]] = []
    if not rollout_args.registry_smoke:
        smoke_data_id = os.getenv("ARENA_ROLLOUT_DATA_ID", "")
        if smoke_data_id:
            stream = arena_client.resolve_stream(econfig.stream_id)
            stream_id = str(stream["stream_id"])
            llm_protocol = infer_llm_protocol(stream)
            selected_rows = [
                {
                    "data_id": smoke_data_id,
                    "stream_id": stream_id,
                    "llm_protocol": llm_protocol,
                }
            ]
            logger.info(
                "Using preselected Arena smoke row: stream_id=%s, data_id=%s, "
                "protocol=%s",
                stream_id,
                smoke_data_id,
                llm_protocol,
            )
        else:
            logger.info("Loading Arena Stream dataset")
            stream = arena_client.resolve_stream(econfig.stream_id)
            stream_id = str(stream["stream_id"])
            llm_protocol = infer_llm_protocol(stream)
            rows = arena_client.get_all_dataset_rows(stream_id, llm_protocol)
            selection_rng = random.Random(config.seed)
            selected_rows = selection_rng.sample(
                rows, k=min(rollout_args.num_rollouts, len(rows))
            )
            if len(selected_rows) < rollout_args.num_rollouts:
                raise ValueError(
                    f"Stream {stream_id!r} has only {len(rows)} rows, but "
                    f"{rollout_args.num_rollouts} were requested"
                )
            logger.info(
                "Loaded %d rows from Arena Stream %s using %s; running %d rollout(s)",
                len(rows),
                stream_id,
                llm_protocol,
                len(selected_rows),
            )

    rollout_alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")
    if rollout_alloc.backend != "sglang":
        raise ValueError(
            f"arena_rollout_only.py requires an SGLang backend, got "
            f"{rollout_alloc.backend!r}"
        )
    server_args = SGLangConfig.build_args(
        sglang_config=config.sglang,
        tp_size=rollout_alloc.parallel.tp_size,
        pp_size=rollout_alloc.parallel.pp_size,
        base_gpu_id=0,
    )
    config.rollout.max_head_offpolicyness = int(1e12)

    # These values are shell-expansion placeholders for Slurm workers. Local
    # workers already inherit the real container environment; keeping the
    # placeholders would overwrite credentials with literal "$..." strings.
    for scheduling_spec in config.rollout.scheduling_spec:
        scheduling_spec.env_vars.pop("ARENA_OPENAPI_BASE", None)
        scheduling_spec.env_vars.pop("ARENA_OPENAPI_TOKEN", None)
        scheduling_spec.env_vars.pop("ARENA_LLM_API_KEY", None)

    scheduler = LocalScheduler(
        gpu_devices=list(range(config.cluster.n_gpus_per_node)),
        exp_config=config,
    )
    controller = RemoteSGLangEngine.as_controller(config.rollout, scheduler)
    wandb_run = _init_wandb(config)
    try:
        controller.initialize(role="arena-rollout-smoke", server_args=server_args)
        controller.start_proxy()
        if rollout_args.registry_smoke:
            agent_config = config.rollout.agent
            if agent_config is None:
                raise ValueError("rollout.agent is required for registry smoke")
            registered_url, deployment_id = _run_registry_smoke(
                arena_client=arena_client,
                proxy_base_url=controller.get_proxy_addr(0),
                proxy_admin_api_key=agent_config.admin_api_key,
                trial_name=config.trial_name,
            )
            wandb_run.log({"registry_smoke/succeeded": 1})
            logger.info(
                "Registry smoke complete: deployment_id=%s, registered_url=%s",
                deployment_id,
                registered_url,
            )
            if rollout_args.serve_after_smoke_seconds > 0:
                logger.info(
                    "Keeping rollout service alive for %d seconds",
                    rollout_args.serve_after_smoke_seconds,
                )
                time.sleep(rollout_args.serve_after_smoke_seconds)
            return
        rollout_results, failed_data_ids = _run_rollout_tasks(
            controller=controller,
            rows=selected_rows,
            workflow_kwargs={
                "econfig": asdict(econfig),
                "gen_args": {
                    "temperature": config.gconfig.temperature,
                    "max_completion_tokens": config.gconfig.max_new_tokens,
                },
                "timeout": econfig.timeout,
            },
        )
        rewards = [reward for _, reward in rollout_results]
        completed = len(rewards)
        failed = len(failed_data_ids)
        mean_reward = sum(rewards) / completed if completed else 0.0
        max_reward = max(rewards, default=0.0)
        reward_one_count = sum(reward >= 1.0 for reward in rewards)
        logger.info(
            "Arena rollout-only completed: stream_id=%s, completed=%d, failed=%d, "
            "mean_reward=%.4f, max_reward=%.4f, reward_one_count=%d, results=%s, "
            "failed_data_ids=%s",
            stream_id,
            completed,
            failed,
            mean_reward,
            max_reward,
            reward_one_count,
            rollout_results,
            failed_data_ids,
        )
        wandb_run.log(
            {
                "rollout/mean_reward": mean_reward,
                "rollout/max_reward": max_reward,
                "rollout/reward_one_count": reward_one_count,
                "rollout/completed": completed,
                "rollout/failed": failed,
            }
        )
    finally:
        try:
            controller.destroy()
        finally:
            scheduler.delete_workers(None)
            wandb_run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
