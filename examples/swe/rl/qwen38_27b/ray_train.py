# SPDX-License-Identifier: Apache-2.0
"""Prepare the dense 27B preset and run it on an existing Ray cluster."""

import argparse
import copy
import os
import secrets
import shlex
import sys
from pathlib import Path

import yaml


def prepare_ray_config(config: dict, repo: Path, env: dict[str, str]) -> dict:
    """Preserve training settings while replacing the Slurm launch boundary."""
    config = copy.deepcopy(config)
    # The grouped Ray path executes each role's wrapper. Equal-replica forks and
    # multi-node TP have different bootstrap paths and are not this recipe.
    if config["allocation_mode"] != "megatron[actor]:d1p4t4c4e1|sglang[rollout]:d16t4":
        raise ValueError("This Ray recipe requires actor d1p4t4c4e1 and rollout d16t4")
    # The single-controller trainer reads these fields, not allocation_mode.
    for role, backend in (
        ("actor", "megatron:d1p4t4c4e1"),
        ("rollout", "sglang:d16t4"),
    ):
        if config[role]["backend"] != backend:
            raise ValueError(f"This Ray recipe requires {role}.backend={backend}")
    if (
        config["actor"].get("scheduling_strategy", {}).get("type", "separation")
        != "separation"
    ):
        raise ValueError("This Ray recipe requires actor separation scheduling")
    if (config["cluster"]["n_nodes"], config["cluster"]["n_gpus_per_node"]) != (8, 8):
        raise ValueError("This Ray recipe requires 8 nodes with 8 GPUs each")
    strategy = config["rollout"]["scheduling_strategy"]
    if (
        strategy["type"] != "colocation"
        or strategy["target"] != "actor"
        or not strategy.get("fork", True)
    ):
        raise ValueError(
            "This Ray recipe requires forked rollout colocation with actor"
        )

    config["scheduler"]["type"] = "ray"
    config["experiment_name"] = "${oc.env:QWEN_EXPERIMENT_NAME}"
    config["trial_name"] = "${oc.env:QWEN_ARENA_TRIAL}"
    config["tokenizer_path"] = "${oc.env:QWEN_MODEL}"
    config["cluster"]["fileroot"] = "${oc.env:QWEN_OUTPUT_ROOT}"
    config["cluster"]["name_resolve"] = {
        "type": "nfs",
        "nfs_record_root": "${cluster.fileroot}/name_resolve/${experiment_name}/${trial_name}",
    }
    config["econfig"]["arena_base_url"] = "${oc.env:ARENA_OPENAPI_BASE}"
    config["actor"]["path"] = "${tokenizer_path}"
    config["rollout"]["tokenizer_path"] = "${tokenizer_path}"
    config["sglang"]["model_path"] = "${tokenizer_path}"
    config["rollout"]["agent"]["admin_api_key"] = "${oc.env:SWE_RL_ADMIN_API_KEY}"

    streams = config["econfig"]["arena_streams"]
    if len(streams) != 1:
        raise ValueError("This recipe expects one Arena stream")
    stream = streams[0]
    if env.get("QWEN_ARENA_STREAM_ID"):
        stream["stream_id"] = env["QWEN_ARENA_STREAM_ID"]
    if env.get("QWEN_ARENA_HARNESS"):
        stream["harness"] = env["QWEN_ARENA_HARNESS"]
    reward = stream.setdefault("expected_reward_ref", {})
    for key in ("key", "version"):
        if env.get(f"QWEN_REWARD_{key.upper()}"):
            reward[key] = env[f"QWEN_REWARD_{key.upper()}"]
    if not stream.get("stream_id") or stream["stream_id"] == "your-stream-id":
        raise ValueError(
            "Set QWEN_ARENA_STREAM_ID or a concrete stream_id in the config"
        )
    if not stream.get("harness") or not all(reward.get(k) for k in ("key", "version")):
        raise ValueError("Pin the Arena harness and expected_reward_ref key/version")

    wrapper = repo / "examples/swe/rl/qwen38_27b/ray_worker.sh"
    for role in ("actor", "rollout"):
        for spec in config[role]["scheduling_spec"]:
            for key in (
                "mount",
                "image",
                "reservation",
                "nodelist",
                "exclude",
                "exclusive",
                "partition",
                "srun_additional_args",
                "additional_bash_cmds",
            ):
                spec.pop(key, None)
            spec["cmd"] = shlex.join(["bash", str(wrapper), role])
            # Use Ray's native image launcher; the role wrapper runs inside it.
            # Keep advanced container/run_options supplied in a custom YAML.
            runtime_env = spec.setdefault("ray_runtime_env", {})
            image = env.get(f"QWEN_RAY_{role.upper()}_IMAGE")
            if image:
                if "container" in runtime_env:
                    raise ValueError(
                        f"{role}: choose either a profile image or runtime_env.container"
                    )
                runtime_env["image_uri"] = image
            role_env = spec.setdefault("env_vars", {})
            # The image/profile supplies role dependencies; do not carry the
            # historical SIF's site-packages or loader paths into a new image.
            for key in (
                "PYTHONPATH",
                "LD_LIBRARY_PATH",
                "CUDNN_PATH",
                "QWEN_RUNTIME_SCRIPT",
            ):
                role_env.pop(key, None)
            role_env.update(
                QWEN_ENV_FILE=env["QWEN_ENV_FILE"],
                QWEN_RAY_PYTHON=env.get(f"QWEN_RAY_{role.upper()}_PYTHON") or "python3",
                QWEN_SUBMITTED_TRIAL=env["QWEN_ARENA_TRIAL"],
                QWEN_SUBMITTED_EXPERIMENT=env["QWEN_EXPERIMENT_NAME"],
                QWEN_SUBMITTED_RUN_DIR=env["QWEN_RUN_DIR"],
                QWEN_SUBMITTED_OUTPUT_ROOT=env["QWEN_OUTPUT_ROOT"],
                QWEN_SUBMITTED_USER=env["QWEN_RUN_USER"],
            )
    return config


def main() -> None:
    recipe = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=recipe / "train_fine_released64_bs16.yaml"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the Ray config without connecting or training",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Require recovery of this trial; stop its old Ray job first",
    )
    args = parser.parse_args()
    os.environ["QWEN_ENV_FILE"] = str(
        Path(os.environ["QWEN_ENV_FILE"]).resolve(strict=True)
    )
    config = prepare_ray_config(
        yaml.safe_load(args.config.read_text()), recipe.parents[3], dict(os.environ)
    )
    run_dir = Path(os.environ["QWEN_RUN_DIR"])
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    config_path = run_dir / "train-ray.yaml"
    content = yaml.safe_dump(config, sort_keys=False)
    if config_path.exists():
        if config_path.read_text() != content:
            raise ValueError("Ray config changed: use a new QWEN_RUN_DIR and trial")
    else:
        with config_path.open("x") as output:
            output.write(content)
        config_path.chmod(0o600)
    if args.prepare_only:
        return

    identity = (
        Path(os.environ["QWEN_RUN_USER"])
        / os.environ["QWEN_EXPERIMENT_NAME"]
        / os.environ["QWEN_ARENA_TRIAL"]
    )
    output_root = Path(os.environ["QWEN_OUTPUT_ROOT"])
    if not args.resume and any(
        (output_root / kind / identity).exists() for kind in ("logs", "checkpoints")
    ):
        raise ValueError(
            "Trial output already exists; use a new trial or explicit --resume"
        )
    os.environ["QWEN_REQUIRE_RECOVER"] = "1" if args.resume else "0"
    os.environ["SWE_RL_ADMIN_API_KEY"] = secrets.token_urlsafe(32)

    import ray

    # Unlike the generic convenience initializer, never fall back to a local
    # cluster when the intended 64-GPU cluster is unreachable.
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), ignore_reinit_error=True)
    try:
        nodes = [
            node
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("GPU", 0) >= 8
        ]
        if len(nodes) < 8:
            raise RuntimeError(
                "Qwen27 requires an existing Ray cluster with eight 8-GPU nodes"
            )
        from examples.swe.rl.qwen38_27b import qwen27_train_rl

        # Keep the dense-model decorator, CPU optimizer worker, and RPC settings.
        sys.argv = [sys.argv[0], "swe", "--config", str(config_path)]
        qwen27_train_rl.main()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
