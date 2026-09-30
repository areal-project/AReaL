# SPDX-License-Identifier: Apache-2.0
"""Run this run's unit checks and fresh off/on GSM8K training in Slurm."""

import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def run(
    command: list[str], log: Path, env: dict[str, str] | None = None, cwd: Path = ROOT
) -> None:
    with log.open("w") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
            check=True,
        )


def inside(out: Path, environment: dict[str, Any]) -> None:
    from omegaconf import OmegaConf

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "false"
    # The allocation inherits client-side proxies that are unreachable on the
    # compute node. All resources are local and worker RPC must stay direct.
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    env["NO_PROXY"] = env["no_proxy"] = "*"
    env["RSI_VALIDATION_ADMIN_KEY"] = secrets.token_urlsafe(32)
    if environment.get("phase") == "checks":
        import tomllib
        from collections import Counter

        checkout = Path(environment["checkout"]).resolve()
        env["PYTHONPATH"] = str(checkout)
        env["UV_HTTP_TIMEOUT"] = "120"
        # Keep container-wide mirror settings from rewriting repository locks.
        with (checkout / "uv.lock").open("rb") as stream:
            locked = tomllib.load(stream)
        registries = Counter(
            package["source"]["registry"]
            for package in locked["package"]
            if "registry" in package.get("source", {})
        )
        for key in ("UV_INDEX", "UV_INDEX_URL", "UV_EXTRA_INDEX_URL"):
            env.pop(key, None)
        env["UV_DEFAULT_INDEX"] = registries.most_common(1)[0][0]
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_partial_rollout_versions.py",
                "tests/test_ppo_actor_truncation.py",
                "tests/infra/test_remote_inf_engine.py",
                "tests/v2/inference_service/test_inf_bridge.py",
                "tests/test_adv_norm_config.py",
                "tests/test_reward_norm_variable_group.py",
            ],
            out / "regression_tests.log",
            env,
            cwd=checkout,
        )
        all_files_passed = True
        try:
            run(
                [sys.executable, "-m", "pre_commit", "run", "--all-files"],
                out / "pre_commit.log",
                env,
                cwd=checkout,
            )
        except subprocess.CalledProcessError:
            all_files_passed = False
        changed = (
            subprocess.check_output(
                ["git", "diff", "--cached", "--name-only", "-z"], cwd=checkout
            )
            .decode()
            .split("\0")
        )
        run(
            [sys.executable, "-m", "pre_commit", "run", "--files"]
            + [path for path in changed if path],
            out / "pre_commit_changed_files.log",
            env,
            cwd=checkout,
        )
        (out / "feedback.json").write_text(
            json.dumps(
                {
                    "pre_commit_all_files_passed": all_files_passed,
                    "pre_commit_changed_files_passed": True,
                    "training_completed": False,
                }
            )
        )
        return
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_partial_rollout_versions.py",
            "tests/test_ppo_actor_truncation.py",
            "tests/infra/test_remote_inf_engine.py",
            "tests/v2/inference_service/test_inf_bridge.py",
        ],
        out / "pytest.log",
        env,
    )
    if environment.get("phase") == "tests":
        (out / "feedback.json").write_text(
            json.dumps({"unit_tests_passed": True, "training_completed": False})
        )
        return
    for enabled in (False, True):
        label = "enabled" if enabled else "disabled"
        cfg = OmegaConf.load(ROOT / "examples/math/gsm8k_grpo.yaml")
        overrides = {
            "experiment_name": "partial-rollout-validation",
            "trial_name": label,
            "total_train_epochs": 1,
            "total_train_steps": 8,
            "scheduler.type": "local",
            "cluster.fileroot": str(out / label),
            "cluster.name_resolve.nfs_record_root": str(out / label / "name_resolve"),
            "actor.path": environment["model"],
            "actor.mask_stale_tokens": enabled,
            "actor.max_token_version_gap": 1,
            "rollout.enable_partial_rollout": enabled,
            "rollout.agent.admin_api_key": "${oc.env:RSI_VALIDATION_ADMIN_KEY}",
            "rollout.admin_api_key": "${oc.env:RSI_VALIDATION_ADMIN_KEY}",
            "actor.admin_api_key": "${oc.env:RSI_VALIDATION_ADMIN_KEY}",
            "rollout.max_concurrent_rollouts": 64,
            "rollout.max_head_offpolicyness": 4,
            "rollout.dump_to_file": True,
            "train_dataset.path": environment["dataset"],
            "valid_dataset.path": environment["dataset"],
            "train_dataset.batch_size": 8,
            "valid_dataset.batch_size": 8,
            "train_dataset.num_workers": 0,
            "valid_dataset.num_workers": 0,
            "gconfig.n_samples": 2,
            "gconfig.max_new_tokens": 1024,
            "sglang.context_length": 4096,
            "sglang.mem_fraction_static": 0.6,
            "evaluator.freq_epochs": None,
            "saver.freq_epochs": None,
        }
        for key, value in overrides.items():
            OmegaConf.update(cfg, key, value)
        path = out / f"{label}.yaml"
        OmegaConf.save(cfg, path)
        run(
            [sys.executable, "examples/math/gsm8k_rl.py", "--config", str(path)],
            out / f"{label}.log",
            env,
        )
    (out / "feedback.json").write_text(
        json.dumps(
            {
                "training_completed": True,
                "acceptance_verified": False,
                "note": "Inspect actual logs for mask counts and mixed-version trajectories before accepting.",
            },
            indent=2,
        )
    )


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env_file = Path(
        os.environ.get("RSI_VALIDATION_ENV", ROOT / "validation/environment.json")
    )
    try:
        environment = json.loads(env_file.read_text())
        if "--inside" in sys.argv:
            inside(out, environment)
        else:
            command = [
                "srun",
                "--mpi=none",
                f"--jobid={environment['job_id']}",
                "--overlap",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=1",
                "singularity",
                "exec",
                "--nv",
            ]
            for bind in [str(ROOT), str(out), *environment.get("binds", [])]:
                command.extend(["--bind", bind])
            command.extend(
                [
                    environment["image"],
                    environment.get("python", "python3"),
                    str(Path(__file__).resolve()),
                    str(out),
                    "--inside",
                ]
            )
            run(command, out / "slurm.log")
    except Exception as exc:
        (out / "feedback.json").write_text(
            json.dumps({"training_completed": False, "error": str(exc)}, indent=2)
        )
        raise


if __name__ == "__main__":
    main()
