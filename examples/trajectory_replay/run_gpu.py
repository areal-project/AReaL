# SPDX-License-Identifier: Apache-2.0

"""Run one real single-controller RL capture/replay case on an allocated node.

Never executed locally. This is a training launcher, not a unit-test runner.
Use the minimal/end-to-end wrappers to select the required number of updates.
"""

import argparse
import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--write-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temp", type=Path, required=True)
    parser.add_argument("--version", choices=("v1", "v2"), required=True)
    parser.add_argument("--mode", choices=("off", "dump", "replay"), required=True)
    parser.add_argument("--scope", choices=("rollout", "full"), default="rollout")
    parser.add_argument("--snapshots", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--actor-backend")
    parser.add_argument("--rollout-backend")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--areal", default="areal")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Save and validate the effective configuration without starting workers.",
    )
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    # v1 AWEX transfers through colocated CUDA IPC; v2 supports separation.
    args.actor_backend = args.actor_backend or (
        "megatron:d2" if args.version == "v1" else "megatron:d1"
    )
    args.rollout_backend = args.rollout_backend or (
        "sglang:d2" if args.version == "v1" else "sglang:d1"
    )
    if args.steps < 1 or args.gpus < 2 or args.timeout < 1:
        parser.error("steps/timeout must be positive; this example needs >=2 GPUs")
    root = args.write_root.resolve()
    output = args.output.resolve()
    temporary = args.temp.resolve()
    snapshots = (args.snapshots or output / "snapshots").resolve()
    for path in (output, snapshots):
        if path == root or not path.is_relative_to(root):
            parser.error(f"Expected an isolated path beneath write-root: {path}")
    if not (
        (temporary != root and temporary.is_relative_to(root))
        or (temporary.parent == Path("/tmp") and temporary.name.startswith("rsi-"))
    ):
        parser.error(
            "temp must be isolated beneath write-root or container /tmp/rsi-<task>"
        )
    if args.mode == "replay" and args.snapshots is None:
        parser.error("replay requires --snapshots from a completed capture")
    for path in (args.config, args.model, args.data):
        if not path.exists():
            parser.error(f"Missing input: {path}")
    if os.environ.get("AREAL_SPMD_MODE", "0").lower() not in ("0", "false", ""):
        parser.error("This example requires single-controller mode")
    output.mkdir(parents=True, exist_ok=False)
    temporary.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    env = os.environ.copy()
    env.update(
        AREAL_SPMD_MODE="0",
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(temporary),
        TMP=str(temporary),
        TEMP=str(temporary),
        AREAL_CACHE_DIR=str(cache / "areal"),
        HF_HOME=str(cache / "hf"),
        HF_DATASETS_CACHE=str(cache / "datasets"),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        XDG_CACHE_HOME=str(cache / "xdg"),
        TORCH_HOME=str(cache / "torch"),
        TORCH_EXTENSIONS_DIR=str(cache / "extensions"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        CUDA_CACHE_PATH=str(cache / "cuda"),
        FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"),
        OUTLINES_CACHE_DIR=str(cache / "outlines"),
        NUMBA_CACHE_DIR=str(cache / "numba"),
        CUTE_DSL_CACHE_DIR=str(cache / "cutlass"),
        DG_JIT_CACHE_DIR=str(cache / "deep-gemm"),
        TIKTOKEN_CACHE_DIR=str(cache / "tiktoken"),
        WANDB_DIR=str(output / "wandb"),
        WANDB_CACHE_DIR=str(cache / "wandb"),
        WANDB_CONFIG_DIR=str(output / "wandb-config"),
        WANDB_DATA_DIR=str(output / "wandb-data"),
        WANDB_ARTIFACT_DIR=str(output / "wandb-artifacts"),
    )
    repo = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    settings = {
        "experiment_name": "trajectory-replay",
        "trial_name": output.name,
        "seed": 1,
        "scheduler.type": "local",
        "cluster.n_nodes": 1,
        "cluster.n_gpus_per_node": args.gpus,
        "cluster.fileroot": str(output),
        "cluster.name_resolve.type": "nfs",
        "cluster.name_resolve.nfs_record_root": str(output / "names"),
        "total_train_epochs": 1,
        "total_train_steps": args.steps,
        "enable_offload": False,
        "actor._version": args.version,
        "rollout._version": args.version,
        "actor.backend": args.actor_backend,
        "rollout.backend": args.rollout_backend,
        "actor.path": str(args.model.resolve()),
        "actor.offload": False,
        "actor.scheduling_strategy.type": "separation",
        "rollout.scheduling_strategy.type": "separation",
        "actor.weight_update_mode": "awex",
        "actor.megatron.bridge_type": "megatron-bridge",
        "actor.megatron.use_bridge_for_update_weights": True,
        # The engine schedule uses the full dataloader horizon. With a
        # proportional warmup its first update can otherwise have zero LR.
        "actor.optimizer.warmup_steps": 0,
        "actor.optimizer.warmup_steps_proportion": 0.0,
        "rollout.agent.mode": "inline",
        "rollout.dump_to_file": False,
        "rollout.max_concurrent_rollouts": 8,
        "gconfig.n_samples": 2,
        "gconfig.max_new_tokens": 256,
        "gconfig.max_tokens": 2048,
        "train_dataset.path": str(args.data.resolve()),
        "train_dataset.batch_size": 2,
        "train_dataset.num_workers": 0,
        "valid_dataset": None,
        "ref": None,
        "actor.kl_ctl": 0.0,
        # Keep a nonzero real PPO signal even if this tiny model answers every
        # sampled question incorrectly. All comparison runs use this objective.
        "actor.reward_bias": -0.5,
        "actor.reward_norm": None,
        "actor.adv_norm": None,
        "sglang.model_path": str(args.model.resolve()),
        "sglang.context_length": 2048,
        "sglang.mem_fraction_static": 0.4,
        "sglang.attention_backend": "triton",
        "sglang.enable_dp_attention": False,
        "sglang.disable_cuda_graph": True,
        "sglang.enable_memory_saver": True,
        "saver.freq_epochs": None,
        "saver.freq_steps": 1,
        "saver.freq_secs": None,
        "recover.mode": "disabled",
        "stats_logger.wandb.mode": "disabled",
        "trajectory_debug.mode": args.mode,
        "trajectory_debug.scope": args.scope,
        "trajectory_debug.path": str(snapshots),
    }
    if args.version == "v1":
        settings["rollout.scheduling_strategy.type"] = "colocation"
        settings["rollout.scheduling_strategy.target"] = "actor"
    else:
        # Keep routable v2 services authenticated without recording a secret
        # in command.json or command-line overrides. The loader redacts config.
        env["AREAL_TRAJECTORY_ADMIN_KEY"] = "rsi-trajectory-" + secrets.token_urlsafe(
            32
        )
        settings["actor.admin_api_key"] = "${oc.env:AREAL_TRAJECTORY_ADMIN_KEY}"
        settings["rollout.admin_api_key"] = "${oc.env:AREAL_TRAJECTORY_ADMIN_KEY}"
    # ++ works whether a field is already in the source YAML or only in the
    # dataclass. JSON strings are also quoted Hydra string values.
    overrides = [f"++{key}={json.dumps(value)}" for key, value in settings.items()]
    overrides.extend(args.override)
    command = [
        args.areal,
        "train",
        "run",
        "--config",
        str(args.config.resolve()),
        "--driver",
        "examples.trajectory_replay.train:main",
        *overrides,
    ]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    source_files = [
        path
        for directory in (repo / "areal", repo / "examples" / "trajectory_replay")
        for path in directory.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    hashes = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    hashes["baseline_config"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    (output / "source.json").write_text(json.dumps(hashes, indent=2) + "\n")
    if args.prepare_only:
        # This configuration build uses the normal loader, without model
        # loading, worker creation, or unit-test execution.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from areal.api.cli_args import GRPOConfig, load_expr_config; "
                "load_expr_config(sys.argv[1:], GRPOConfig)",
                "--config",
                str(args.config.resolve()),
                *overrides,
            ],
            cwd=repo,
            env=env,
            check=False,
        )
        (output / "result.json").write_text(
            json.dumps(
                {
                    "status": "configuration_prepared"
                    if result.returncode == 0
                    else "configuration_failed",
                    "returncode": result.returncode,
                    "acceptance": "no training executed",
                },
                indent=2,
            )
            + "\n"
        )
        raise SystemExit(result.returncode)

    # Only IPC and required runtime temporary files use this short directory.
    # Artifacts and all named caches stay beneath write-root.
    # SGLang and multiprocessing need them even when output is on shared storage.
    probe = temporary / f"probe-{os.getpid()}.sock"
    try:
        with socket.socket(socket.AF_UNIX) as sock:
            sock.bind(str(probe))
    except OSError as exc:
        (output / "result.json").write_text(
            json.dumps({"status": "preflight_failed", "reason": str(exc)}, indent=2)
            + "\n"
        )
        raise SystemExit(
            f"Authorized temp directory cannot host UNIX sockets: {exc}"
        ) from exc
    finally:
        probe.unlink(missing_ok=True)
    started = time.monotonic()
    with (output / "console.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (output / "process.json").write_text(
            json.dumps({"pid": process.pid, "pgid": process.pid}) + "\n"
        )
        try:
            result = process.wait(timeout=args.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            result = 124
    (output / "result.json").write_text(
        json.dumps(
            {
                "returncode": result,
                "elapsed_seconds": time.monotonic() - started,
                "acceptance": "pending artifact and metric review",
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(result)


if __name__ == "__main__":
    main()
