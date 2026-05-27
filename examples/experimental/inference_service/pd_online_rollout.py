"""Rollout-only online example with prefill-decode (PD) disaggregation.

Identical to ``online_rollout.py`` except it asserts ``rollout.pd_disaggregation``
is enabled, so misconfigured configs fail early. Uses the same controller stack
(``RolloutControllerV2``) — the gateway transparently dual-dispatches each chat
request to a prefill+decode SGLang pair using a fresh bootstrap triplet.

Hardware: needs at least 2 GPUs (1 prefill + 1 decode) for the inference stack.

Backend: SGLang only. Install a KV-cache transfer engine separately (one of
``mooncake-transfer-engine`` or ``nixl``) — AReaL no longer bundles it.
"""

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

    logger = logging.getLogger("InferenceServicePDOnlineTrain")

    config, _ = load_expr_config(remaining, PPOConfig)
    agent_cfg = config.rollout.agent
    if agent_cfg is None or agent_cfg.mode != "online":
        raise ValueError("pd_online_rollout.py requires rollout.agent.mode='online'.")
    if not getattr(config.rollout, "pd_disaggregation", False):
        raise ValueError(
            "pd_online_rollout.py requires rollout.pd_disaggregation=true. "
            "For non-PD mode use online_rollout.py instead."
        )
    if ext_args.api_url is not None:
        raise ValueError(
            "PD disaggregation does not support external --api-url; "
            "drop --api-url or use online_rollout.py for external providers."
        )
    if not is_single_controller():
        raise NotImplementedError(
            "pd_online_rollout.py requires single-controller execution "
            "(for example: scheduler.type=local)."
        )

    from areal.api.alloc_mode import ModelAllocation
    from areal.infra.scheduler.local import LocalScheduler
    from areal.infra.scheduler.slurm import SlurmScheduler

    sched_type = config.scheduler.type
    if sched_type == "local":
        scheduler = LocalScheduler(exp_config=config)
    elif sched_type == "slurm":
        scheduler = SlurmScheduler(exp_config=config)
    else:
        raise NotImplementedError(f"Unknown scheduler type: {sched_type}")

    rollout_alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")
    if rollout_alloc.backend != "sglang":
        raise ValueError(
            f"PD disaggregation requires sglang backend, got: {rollout_alloc.backend}"
        )
    server_args = asdict(config.sglang)

    ctrl_config = deepcopy(config.rollout)
    if ctrl_config.dump_to_file:
        # FIXME: dump_to_file is not yet supported in inference service.
        logger.warning(
            "rollout.dump_to_file=true is not yet supported in inference service; "
            "forcing dump_to_file=false"
        )
        ctrl_config.dump_to_file = False
    if ext_args.model:
        ctrl_config.model = ext_args.model

    ctrl = RolloutControllerV2(config=ctrl_config, scheduler=scheduler)
    try:
        ctrl.initialize(role="rollout", server_args=server_args)

        logger.info("Proxy gateway available at %s", ctrl.proxy_gateway_addr)
        logger.info("PD prefill addrs: %s", ctrl._prefill_addrs)
        logger.info("PD decode addrs:  %s", ctrl._decode_addrs)

        result = ctrl.rollout_batch(
            data=None,
            batch_size=config.train_dataset.batch_size,
            workflow=None,
        )

        import torch

        from areal.infra.rpc.rtensor import RTensor

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
