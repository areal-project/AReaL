"""Rollout-only online example via the inference_service gateway stack."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path


def main(args: list[str]) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--provider-api-key", default=None)
    parser.add_argument("--model", default=None)
    ext_args, remaining = parser.parse_known_args(args)

    from areal.api.cli_args import PPOConfig, load_expr_config
    from areal.experimental.inference_service.controller.controller import (
        RolloutControllerV2,
    )
    from areal.utils import logging
    from areal.utils.environ import is_single_controller

    logger = logging.getLogger("InferenceServiceOnlineTrain")

    config, _ = load_expr_config(remaining, PPOConfig)
    agent_cfg = config.rollout.agent
    if agent_cfg is None or agent_cfg.mode != "online":
        raise ValueError(
            "online_rollout.py requires rollout.agent.mode='online' for inference_service online training."
        )
    if not is_single_controller():
        raise NotImplementedError(
            "online_rollout.py requires single-controller execution (for example: scheduler.type=local)."
        )
    from areal.infra.scheduler.local import LocalScheduler
    from areal.infra.scheduler.slurm import SlurmScheduler

    sched_type = config.scheduler.type
    if sched_type == "local":
        scheduler = LocalScheduler(exp_config=config)
    elif sched_type == "slurm":
        scheduler = SlurmScheduler(exp_config=config)
    else:
        raise NotImplementedError(f"Unknown scheduler type: {sched_type}")

    is_external = ext_args.api_url is not None

    ctrl_config = deepcopy(config.rollout)
    if ctrl_config.dump_to_file:
        # FIXME: dump_to_file is not yet supported in inference service.
        logger.warning(
            "rollout.dump_to_file=true is not yet supported in inference service; forcing dump_to_file=false"
        )
    ctrl_config.dump_to_file = False
    if ext_args.model:
        ctrl_config.model = ext_args.model
    if is_external:
        ctrl_config.api_url = ext_args.api_url
        ctrl_config.provider_api_key = ext_args.provider_api_key
        server_args = None
    else:
        from areal.api.alloc_mode import ModelAllocation

        rollout_alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")
        if rollout_alloc.backend == "sglang":
            server_args = asdict(config.sglang)
        elif rollout_alloc.backend == "vllm":
            server_args = asdict(config.vllm)
        else:
            raise ValueError(f"Unsupported rollout backend: {rollout_alloc.backend}")

    ctrl = RolloutControllerV2(config=ctrl_config, scheduler=scheduler)
    try:
        ctrl.initialize(
            role="rollout",
            server_args=server_args,
        )

        logger.info("Proxy gateway available at %s", ctrl.proxy_gateway_addr)

        # Online mode: pass None for both data and workflow so the
        # controller creates empty-dict placeholders and uses the
        # online InferenceServiceWorkflow (no agent).
        result = ctrl.rollout_batch(
            data=None,
            batch_size=config.train_dataset.batch_size,
            workflow=None,
        )

        if is_external:
            logger.info("Rollout complete (%d trajectories)", len(result))
            for i, traj in enumerate(result):
                for j, interaction in enumerate(traj.get("interactions", [])):
                    request_msgs = interaction.get("request", [])
                    request = (
                        request_msgs[-1].get("content", "") if request_msgs else ""
                    )
                    response = interaction.get("response", "")
                    logger.info(
                        "Trajectory %d, interaction %d:\n"
                        "  request:  %s\n  response: %s",
                        i,
                        j,
                        request[:300],
                        response[:300],
                    )
        else:
            import torch

            from areal.infra.rpc.rtensor import RTensor

            # Localize RTensor references into real torch tensors so we
            # can compute aggregate reward statistics.
            localized_rewards = [RTensor.localize(traj)["rewards"] for traj in result]
            all_rewards = torch.cat(localized_rewards, dim=0)
            logger.info(
                "Rollout complete (%d trajectories), avg_reward=%.4f",
                len(result),
                all_rewards.mean().item(),
            )
    finally:
        ctrl.destroy()
        scheduler.delete_workers(None)


if __name__ == "__main__":
    main(sys.argv[1:])
