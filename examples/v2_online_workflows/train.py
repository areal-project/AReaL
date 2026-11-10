"""Native v2 online PPO: consume trajectories produced by external requests."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from examples.v2_online_workflows.artifacts import write_json


def run_training(
    config, trainer_factory, *, control_dir=None, post_train_timeout=0, **kwargs
):
    if config.rollout._version != "v2" or config.rollout.agent.mode != "online":
        raise ValueError("This entrypoint requires v2 rollout.agent.mode=online")
    if post_train_timeout < 0 or (post_train_timeout and control_dir is None):
        raise ValueError(
            "A nonnegative post-train timeout requires a control directory"
        )
    if control_dir is not None:
        control_dir = Path(control_dir).resolve()
        control_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        with trainer_factory(config, **kwargs) as trainer:
            ready = {
                "gateway_url": trainer.rollout.proxy_gateway_addr,
                "model": config.rollout.model,
                "batch_size": config.train_dataset.batch_size,
                "serving_versions": trainer.rollout.serving_versions(),
            }
            if control_dir is not None:
                write_json(control_dir / "ready.json", ready)
            print("TRAINER_READY " + json.dumps(ready), flush=True)
            # The Trainer consumes native online trajectories. It never submits
            # Arena tasks or drives an Agent; external applications own requests.
            trainer.train()
            finished = {
                "status": "completed",
                "serving_versions": trainer.rollout.serving_versions(),
                "checkpoint_root": config.cluster.fileroot,
            }
            if control_dir is not None:
                write_json(control_dir / "training-finished.json", finished)
            if post_train_timeout:
                deadline = time.monotonic() + post_train_timeout
                release = control_dir / "release.json"
                while not release.exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Post-training release timed out; closing services"
                        )
                    time.sleep(1)
                if (
                    json.loads(release.read_text())["serving_versions"]
                    != finished["serving_versions"]
                ):
                    raise ValueError(
                        "Release does not match the completed model version"
                    )
        if control_dir is not None:
            write_json(control_dir / "closed.json", {"closed": True, **finished})
    except BaseException as exc:
        if control_dir is not None:
            write_json(control_dir / "failure.json", {"error_type": type(exc).__name__})
        raise


def main(args):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--elastic-profile", type=Path)
    parser.add_argument("--elastic-settings", type=Path)
    parser.add_argument(
        "--control-dir",
        type=Path,
        help="New private directory for READY and completion receipts",
    )
    parser.add_argument(
        "--post-train-timeout",
        type=float,
        default=0,
        help="Optionally retain services until release.json, bounded in seconds",
    )
    options, args = parser.parse_known_args(args)
    from areal import PPOTrainer
    from areal.api.cli_args import PPOConfig, load_expr_config

    config, _ = load_expr_config(args, PPOConfig)
    kwargs = {}
    if options.elastic_profile is not None or options.elastic_settings is not None:
        from areal.elastic.infra.adapter import ReconfiguratorAdapter

        kwargs["elastic_reconfigurator"] = ReconfiguratorAdapter(
            profile=None
            if options.elastic_profile is None
            else json.loads(options.elastic_profile.read_text()),
            settings=None
            if options.elastic_settings is None
            else json.loads(options.elastic_settings.read_text()),
        )

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        run_training(
            config,
            PPOTrainer,
            control_dir=options.control_dir,
            post_train_timeout=options.post_train_timeout,
            **kwargs,
        )
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main(sys.argv[1:])
