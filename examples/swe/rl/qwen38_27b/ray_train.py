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


def ray_gpu_bundles(config: dict) -> list[int]:
    """Validate v1 colocation and derive the per-node GPU placement requests."""
    from areal.api.alloc_mode import ModelAllocation, ParallelStrategy, _AllocationMode
    from areal.api.cli_args import ClusterSpecConfig

    allocations = {
        role: ModelAllocation.from_str(config[role]["backend"], name=role)
        for role in ("actor", "rollout")
    }
    # v1 schedules from per-role backends. Catch stale legacy metadata rather
    # than silently launching a different topology from the one in the YAML.
    if config.get("allocation_mode"):
        legacy = _AllocationMode.from_str(config["allocation_mode"])
        for role, allocation in allocations.items():
            if allocation.backend != legacy[
                role
            ].backend or not ParallelStrategy.parallelism_eq(
                allocation.parallel, legacy[role].parallel
            ):
                raise ValueError(f"{role}.backend disagrees with allocation_mode")
    for role, backend in (("actor", "megatron"), ("rollout", "sglang")):
        if allocations[role].backend != backend:
            raise ValueError(f"This dense Qwen entry requires {role}.backend={backend}")
        parallel = allocations[role].parallel
        if any(
            getattr(parallel, dim) < 1
            for dim in ("dp_size", "tp_size", "pp_size", "cp_size", "ep_size")
        ):
            raise ValueError(f"{role} parallel dimensions must be positive")
    if (
        config["actor"].get("scheduling_strategy", {}).get("type", "separation")
        != "separation"
    ):
        raise ValueError("This Ray recipe requires actor separation scheduling")
    strategy = config["rollout"].get("scheduling_strategy", {})
    if (
        strategy.get("type") != "colocation"
        or strategy.get("target") != "actor"
        or not strategy.get("fork", True)
    ):
        raise ValueError(
            "This Ray recipe requires forked rollout colocation with actor"
        )
    if any(spec.get("gpu", 0) != 1 for spec in config["actor"]["scheduling_spec"]):
        raise ValueError("The v1 actor requires one GPU per worker")
    if config["rollout"]["scheduling_spec"][0].get("gpu", 0) <= 0:
        raise ValueError("The rollout scheduling spec must request GPUs")
    actor = allocations["actor"].parallel
    rollout = allocations["rollout"].parallel
    if rollout.cp_size != 1 or rollout.ep_size != 1:
        raise ValueError("SGLang rollout requires CP=EP=1")
    if actor.world_size != rollout.world_size:
        raise ValueError("Colocated actor and rollout must use the same total GPUs")
    cluster = config.get("cluster", {})
    per_node = cluster.get("n_gpus_per_node", ClusterSpecConfig().n_gpus_per_node)
    if not isinstance(per_node, int) or isinstance(per_node, bool) or per_node < 1:
        raise ValueError("cluster.n_gpus_per_node must be a positive integer")
    full, remainder = divmod(actor.world_size, per_node)
    bundles = [per_node] * full + ([remainder] if remainder else [])
    node_limit = cluster.get("n_nodes", len(bundles))
    if (
        not isinstance(node_limit, int)
        or isinstance(node_limit, bool)
        or node_limit < len(bundles)
    ):
        raise ValueError(
            f"cluster.n_nodes must provide at least {len(bundles)} nodes for the actor"
        )
    instance_gpus = rollout.tp_size * rollout.pp_size
    if any(gpus % instance_gpus for gpus in bundles):
        raise ValueError(
            "Each node's actor GPUs must be divisible by rollout TP * PP; cross-node rollout is unsupported"
        )
    if instance_gpus == 1 and any(
        not spec.get("ray_runtime_env") for spec in config["rollout"]["scheduling_spec"]
    ):
        raise ValueError(
            "Single-GPU rollout requires a ray_runtime_env (for example a role image) so Ray runs the rollout wrapper"
        )
    return bundles


def validate_ray_capacity(config: dict, nodes: list[dict]) -> None:
    """Check physical capacity; Ray still controls admission of busy resources."""
    required = sorted(ray_gpu_bundles(config), reverse=True)
    available = sorted(
        (node["Resources"].get("GPU", 0) for node in nodes if node["Alive"]),
        reverse=True,
    )
    # PACK must not place two actor launchers on one physical node: grouped
    # rollout colocation expects one target launcher per node.
    if len(required) > 1 and any(gpus > max(required) for gpus in available):
        raise RuntimeError(
            "Set cluster.n_gpus_per_node to the Ray worker node GPU capacity; "
            "larger nodes can pack multiple actor launchers on one node"
        )
    if len(available) < len(required) or any(
        have < need for have, need in zip(available, required)
    ):
        raise RuntimeError(
            f"Ray cluster needs GPU node bundles {required}; live node GPU capacities: {available}"
        )


def prepare_arena_streams(econfig: dict, env: dict[str, str]) -> None:
    """Keep native stream/reward bindings; legacy overrides target one stream only."""
    streams = econfig.get("arena_streams", [])
    overrides = (
        "QWEN_ARENA_STREAM_ID",
        "QWEN_ARENA_HARNESS",
        "QWEN_REWARD_KEY",
        "QWEN_REWARD_VERSION",
    )
    external = any(
        econfig.get(key) for key in ("arena_streams_file", "arena_streams_yaml_b64")
    )
    if any(env.get(key) for key in overrides) and (len(streams) != 1 or external):
        raise ValueError(
            "Single-stream QWEN_* overrides require exactly one inline stream; configure each stream in the YAML instead"
        )
    if external:
        return  # Native Arena configuration loading validates external definitions.
    if not streams:
        raise ValueError(
            "Configure econfig.arena_streams or an external stream definition"
        )
    for stream in streams:
        if env.get("QWEN_ARENA_STREAM_ID"):
            stream["stream_id"] = env["QWEN_ARENA_STREAM_ID"]
        if env.get("QWEN_ARENA_HARNESS"):
            stream["harness"] = env["QWEN_ARENA_HARNESS"]
        for key in ("key", "version"):
            if env.get(f"QWEN_REWARD_{key.upper()}"):
                stream.setdefault("expected_reward_ref", {})[key] = env[
                    f"QWEN_REWARD_{key.upper()}"
                ]
        if not stream.get("stream_id") or stream["stream_id"] == "your-stream-id":
            raise ValueError(
                "Set a concrete stream_id in the config or QWEN_ARENA_STREAM_ID for a single stream"
            )
        reward = stream.get("expected_reward_ref") or {}
        if stream.get("reward_mapping_file") and any(reward.values()):
            raise ValueError(
                "reward_mapping_file and expected_reward_ref are mutually exclusive"
            )
        if any(reward.values()) and not all(
            reward.get(key) for key in ("key", "version")
        ):
            raise ValueError("Pin expected_reward_ref key/version together")


def prepare_ray_config(
    config: dict, repo: Path, env: dict[str, str], task: str = "swe"
) -> dict:
    """Preserve training settings while replacing the Slurm launch boundary."""
    if task not in ("swe", "gsm8k"):
        raise ValueError("Expected task swe or gsm8k")
    config = copy.deepcopy(config)
    config.setdefault("scheduler", {})["type"] = "ray"
    config["experiment_name"] = "${oc.env:QWEN_EXPERIMENT_NAME}"
    config["trial_name"] = "${oc.env:QWEN_ARENA_TRIAL}"
    config["tokenizer_path"] = "${oc.env:QWEN_MODEL}"
    config["cluster"]["fileroot"] = "${oc.env:QWEN_OUTPUT_ROOT}"
    config["cluster"]["name_resolve"] = {
        "type": "nfs",
        "nfs_record_root": "${cluster.fileroot}/name_resolve/${experiment_name}/${trial_name}",
    }
    config["actor"]["path"] = "${tokenizer_path}"
    config["rollout"]["tokenizer_path"] = "${tokenizer_path}"
    config["sglang"]["model_path"] = "${tokenizer_path}"
    if task == "swe":
        config["econfig"]["arena_base_url"] = "${oc.env:ARENA_OPENAPI_BASE}"
        config["rollout"]["agent"]["admin_api_key"] = "${oc.env:SWE_RL_ADMIN_API_KEY}"
        prepare_arena_streams(config["econfig"], env)
    elif "econfig" in config:
        raise ValueError("Use a GSM8K config without Arena econfig")

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
                QWEN_SUBMITTED_TASK=task,
                QWEN_ENV_FILE=env["QWEN_ENV_FILE"],
                QWEN_RAY_PYTHON=env.get(f"QWEN_RAY_{role.upper()}_PYTHON") or "python3",
                QWEN_SUBMITTED_TRIAL=env["QWEN_ARENA_TRIAL"],
                QWEN_SUBMITTED_EXPERIMENT=env["QWEN_EXPERIMENT_NAME"],
                QWEN_SUBMITTED_RUN_DIR=env["QWEN_RUN_DIR"],
                QWEN_SUBMITTED_OUTPUT_ROOT=env["QWEN_OUTPUT_ROOT"],
                QWEN_SUBMITTED_USER=env["QWEN_RUN_USER"],
                QWEN_SUBMITTED_RECIPE_DIR=env.get(
                    "QWEN_RECIPE_DIR", str(wrapper.parent)
                ),
            )
    bundles = ray_gpu_bundles(config)
    config["cluster"].setdefault("n_nodes", len(bundles))
    return config


def main() -> None:
    recipe = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML field; repeat for multiple fields (OmegaConf dotlist)",
    )
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
    task = os.environ.get("QWEN_TASK", "swe")
    os.environ["QWEN_ENV_FILE"] = str(
        Path(os.environ["QWEN_ENV_FILE"]).resolve(strict=True)
    )
    os.environ["QWEN_RECIPE_DIR"] = str(args.config.resolve(strict=True).parent)
    from omegaconf import OmegaConf

    if any("=" not in item for item in args.overrides):
        parser.error("Each --set requires KEY=VALUE")
    config = OmegaConf.to_container(
        OmegaConf.merge(
            OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)
        ),
        resolve=False,
    )
    if config.get("defaults"):
        raise ValueError("Pass a complete YAML config without Hydra defaults")
    config = prepare_ray_config(config, recipe.parents[3], dict(os.environ), task)
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
    # cluster when the intended training cluster is unreachable.
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), ignore_reinit_error=True)
    try:
        validate_ray_capacity(config, ray.nodes())
        from examples.swe.rl.qwen38_27b import qwen27_train_rl

        # Keep the dense-model decorator, CPU optimizer worker, and RPC settings.
        sys.argv = [sys.argv[0], task, "--config", str(config_path)]
        qwen27_train_rl.main()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
