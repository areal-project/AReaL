#!/usr/bin/env python3
"""Submit the current checkout and run a command relative to its root.

Preview: python submit.py --command 'bash examples/path/to/submit.sh'
Submit:  add --submit to upload and submit the job.
With more than one node, the command runs only on the Ray head.

Real submissions require aistudio-common==0.0.28.84 and aii-pypai==0.1.42.19
in the local Python environment. Set AISTUDIO_USERNUMBER and AISTUDIO_TOKEN.
The runtime image must not already have ./AReaL in its initial working directory.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import logging
import os
import re
import secrets
import shlex
import shutil
import sys
import tarfile
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path, PurePosixPath
from unittest.mock import patch

# Optional default command. It runs from the unpacked checkout, after Ray
# is ready for multi-node jobs. --command overrides this value.
COMMAND = ""

# TH206 defaults from the tested submission template.
CLUSTER = "TH206-AIDC"
APP = "areal"
PRIORITY = "high"
DEFAULT_IMAGE = (
    "acr-mg-ant-registry-vpc.ap-southeast-7.cr.aliyuncs.com/"
    "gpu/areal-runtime:dev-sglang-0520"
)
NODE_SPEC = {
    "cpu": 128,
    "memory": 3600000,
    "gpu_num": 8,
    "gpu_type": "b300",
}
VOLUME_NAME = "areal-cpfs"
VOLUME_MOUNT = "/storage"

SOURCE_DIR = "AReaL"
RAY_BOOTSTRAP = "ais_ray_bootstrap.py"
SKIP = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".DS_Store",
}


def launcher_path(repo: Path, value: str) -> Path:
    """Resolve a launcher within the checkout, independent of the caller's cwd."""
    candidate = (repo / value).resolve()
    if not candidate.is_relative_to(repo) or not candidate.is_file():
        raise ValueError("The launcher must be a file inside this AReaL checkout.")
    return candidate.relative_to(repo)


def stage_source(repo: Path, source_root: Path) -> None:
    """Stage working-tree contents inside the archive's fixed AReaL/ directory."""
    staged = source_root / SOURCE_DIR
    shutil.copytree(
        repo, staged, symlinks=True, ignore=lambda _, names: SKIP & set(names)
    )
    # The SDK dereferences symlinks when archiving. Keep it inside this snapshot.
    for path in staged.rglob("*"):
        if path.is_symlink():
            try:
                inside = path.resolve(strict=True).is_relative_to(staged.resolve())
            except (FileNotFoundError, RuntimeError):
                inside = False
            if not inside:
                raise ValueError(
                    f"Symlink leaves the bundle or has no target: {path.relative_to(staged)}"
                )


def startup_command(
    command: str,
    *,
    launcher: Path | None = None,
    job_name: str = "",
    nodes: int = 1,
    rdma: bool = False,
    ray_root: str = "",
    forwarded_env: dict[str, str] | None = None,
) -> str:
    """Install locally on every Pod and run the user command once."""
    env = dict(forwarded_env or {})
    entrypoint = ("bash", "-c", command)
    if nodes > 1:
        env.update(
            AREAL_AIS_JOB_NAME=job_name,
            AREAL_AIS_NODES=str(nodes),
            AREAL_AIS_GPUS_PER_NODE=str(NODE_SPEC["gpu_num"]),
            AREAL_AIS_STORAGE_ROOT=VOLUME_MOUNT,
            AREAL_AIS_RAY_ROOT=ray_root,
            AREAL_AIS_RDMA=str(int(rdma)),
        )
        entrypoint = ("python3", RAY_BOOTSTRAP, "--", command)
    export = (
        [shlex.join(("export", *(f"{k}={v}" for k, v in env.items())))] if env else []
    )
    script = " && ".join(
        (
            *export,
            f"cd ./{SOURCE_DIR}",
            "test -f pyproject.toml",
            *([shlex.join(("test", "-f", str(launcher)))] if launcher else []),
            "uv pip install -e . --no-deps",
            shlex.join(("exec", *entrypoint)),
        )
    )
    return shlex.join(("bash", "-c", script))


def collect_forwarded_env(names: list[str]) -> dict[str, str]:
    """Forward explicit non-secret settings; AIS exposes the command in the Pod spec."""
    result = {}
    for name in names:
        if (
            not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name)
            or name.startswith(("AREAL_AIS_", "AISTUDIO_", "PAI_"))
            or re.search(
                r"TOKEN|SECRET|PASSWORD|CREDENTIAL|(?:API|ACCESS|PRIVATE)_?KEY",
                name,
                re.I,
            )
        ):
            raise ValueError(f"Cannot forward environment variable {name!r}.")
        if name not in os.environ:
            raise ValueError(f"Export {name} before passing it to the launcher.")
        result[name] = os.environ[name]
    return result


def resolve_ray_root(value: str, forwarded_env: dict[str, str]) -> str:
    """Validate a Pod-side shared path without looking it up on the submit host."""
    # Older callers already pass the shared recipe output directory.
    root = value or forwarded_env.get("QWEN_ARENA_OUTPUT", "")
    pod_path = PurePosixPath(root)
    if not pod_path.is_relative_to(VOLUME_MOUNT) or ".." in pod_path.parts:
        raise ValueError(
            f"Set AREAL_RAY_ROOT or --ray-root to a path under {VOLUME_MOUNT}."
        )
    return root


def check_archive(filename: str, launcher: Path | None, *, nodes: int = 1) -> None:
    expected = {f"{SOURCE_DIR}/pyproject.toml"}
    if launcher is not None:
        expected.add(f"{SOURCE_DIR}/{launcher.as_posix()}")
    if nodes > 1:
        expected.add(f"{SOURCE_DIR}/{RAY_BOOTSTRAP}")
    with tarfile.open(filename, "r:gz") as archive:
        missing = expected - {entry.name for entry in archive if entry.isfile()}
    if missing:
        raise ValueError(f"SDK archive is missing: {', '.join(sorted(missing))}")


def submit(
    repo: Path,
    launcher: Path | None,
    *,
    command: str,
    name: str,
    image: str,
    nodes: int,
    user_number: str,
    token: str,
    rdma: bool = False,
    host_network: bool = False,
) -> bool:
    """Build and submit one AIS workflow; report its record ID as soon as available."""
    cwd, output = Path.cwd(), sys.stdout
    log_level = logging.root.manager.disable
    stage, sent, record_id = "staging the checkout", False, None

    def emit(message: str) -> None:
        output.write(message + "\n")
        output.flush()

    try:
        with tempfile.TemporaryDirectory(prefix="areal-ais-") as tmp:
            source_root = Path(tmp) / "source"
            stage_source(repo, source_root)
            stage = "loading the AIS SDK"
            # The legacy SDK may log credentials; only report our own diagnostics.
            with (
                Path(os.devnull).open("w") as sink,
                patch.dict(os.environ),
                redirect_stdout(sink),
                redirect_stderr(sink),
            ):
                os.chdir(tmp)
                logging.disable(logging.CRITICAL)
                os.environ.update(
                    HOME=tmp,
                    LAUNCH_CONTAINER_MODE="dev_local",
                    USERNUMBER=user_number,
                    USER_DMS_API_TOKENS=base64.b64encode(token.encode()).decode(),
                    AISTUDIO_PROJECT_NAME=APP,
                    ENV_ODPS_PROJECT_NAME="dtmodel_dev",
                )
                for key in (
                    "WORKFLOW_ID",
                    "EXECUTIONRECORD_ID",
                    "ENV_NODE_INSTANCE_ID",
                    "ENV_NAMESPACE",
                    "ENV_ODPS_ACCESS_ID",
                    "ENV_ODPS_ACCESS_KEY",
                    "ENV_ODPS_ALIYUN_ID",
                    "POD_NAME",
                ):
                    os.environ.pop(key, None)
                try:
                    importlib.import_module("Crypto")
                except ModuleNotFoundError:
                    # This SDK's crypto package uses lowercase on macOS.
                    sys.modules["Crypto"] = importlib.import_module("crypto")

                import requests
                from aistudio_common.openapi.models import DataStore
                from aistudio_common.rest import job as rest_job
                from aistudio_common.utils import http_utils
                from pypai.conf import ExecConf, KMConf
                from pypai.conf.retry_strategy import RetryPolicy, RetryStrategy
                from pypai.job import PythonJobBuilder
                from pypai.job.python_job import PythonJob
                from pypai.utils import code_repo_utils
                from pypai.utils.python_exec_utils import get_env_conf

                host = http_utils.get_host_url().rstrip("/")
                request = requests.sessions.Session.request
                upload = PythonJob.upload_local_file_to_oss.__func__
                submit_workflow = rest_job.submit_workflow
                uploaded = False

                def report(value: object) -> None:
                    nonlocal record_id
                    record = str(value)
                    if not record.isascii() or not record.isdigit() or int(record) <= 0:
                        raise ValueError("AIS did not return a valid record ID")
                    if record_id is None:
                        record_id = record
                        emit(
                            f"AIS job {record}: {host}/project/job/detail/{record}?training=true"
                        )

                def bounded_request(session, method, url, **kwargs):
                    nonlocal sent, stage
                    kwargs.setdefault("timeout", (5, 20))
                    if url == host + "/api/v1.0/workflow/execute":
                        if sent:
                            raise RuntimeError("Refusing to submit the workflow twice")
                        sent, stage = True, "sending the workflow"
                        kwargs["allow_redirects"] = False
                    return request(session, method, url, **kwargs)

                def checked_upload(job_cls, filename, remove=True):
                    nonlocal uploaded
                    is_source = bool(filename and filename.endswith(".tar.gz"))
                    if is_source:
                        check_archive(filename, launcher, nodes=nodes)
                        emit(f"Source bundle: {Path(filename).stat().st_size:,} bytes")
                    uri = upload(job_cls, filename, remove=remove)
                    if is_source:
                        if not uri or not uri.startswith(f"oss://{bucket}/"):
                            raise ValueError(
                                "Source upload did not return the expected OSS URI"
                            )
                        uploaded = True
                    return uri

                def submit_and_report(*args, **kwargs):
                    if not uploaded:
                        raise RuntimeError("Source bundle was not uploaded")
                    result = submit_workflow(*args, **kwargs)
                    report(result)
                    return result

                with (
                    patch.object(requests.sessions.Session, "request", bounded_request),
                    patch.object(
                        PythonJob,
                        "upload_local_file_to_oss",
                        classmethod(checked_upload),
                    ),
                    patch.object(rest_job, "submit_workflow", submit_and_report),
                    patch.object(rest_job, "submit_callback", lambda _: None),
                    patch.object(
                        code_repo_utils, "get_k8s_job_repo_switch", lambda: False
                    ),
                ):
                    stage = "resolving cluster OSS settings"
                    env = get_env_conf(APP, NODE_SPEC["gpu_type"], CLUSTER)
                    bucket = env.get("IDE_COMMON_OSS_BUCKET") if env else None
                    if not bucket or not env.get("AISTUDIO_SITE_ENUM"):
                        raise RuntimeError("Missing cluster OSS settings")
                    os.environ.update(env)
                    emit(f"Preparing source bundle for {CLUSTER} ({bucket})…")
                    stage = "building and uploading the source bundle"
                    job = PythonJobBuilder(
                        name=name,
                        main_file=str(source_root / SOURCE_DIR / Path(__file__).name),
                        source_root=str(source_root),
                        command=command,
                        runtime="pytorch",
                        cluster=CLUSTER,
                        k8s_app_name=APP,
                        k8s_priority=PRIORITY,
                        master=ExecConf(num=1, **NODE_SPEC),
                        worker=ExecConf(num=nodes - 1, **NODE_SPEC)
                        if nodes > 1
                        else None,
                        km_conf=KMConf(
                            image=image,
                            cluster=CLUSTER,
                            retry_strategy=RetryStrategy(
                                retry_policy=RetryPolicy.NEVER, max_attempt=0
                            ),
                        ),
                        data_stores=[
                            DataStore(volume_name=VOLUME_NAME, mount_point=VOLUME_MOUNT)
                        ],
                        host_network=host_network,
                        rdma=rdma,
                        code_env_init=False,
                    )
                    value, _ = job.get_record_id()
                    report(value)
        return True
    except Exception as exc:
        detail = f"{stage}: {type(exc).__name__}"
        if record_id:
            emit(f"Job submitted; local cleanup failed ({detail}). Do not resubmit.")
        elif sent:
            emit(
                f"Submission result unknown ({detail}). Check AIS for {name} before retrying."
            )
        elif stage == "staging the checkout" and isinstance(exc, ValueError):
            emit(f"Cannot stage source: {exc}. No job was submitted.")
        else:
            emit(f"Submission stopped before sending the workflow ({detail}).")
        return record_id is not None
    finally:
        os.chdir(cwd)
        logging.disable(log_level)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "launcher", nargs="?", help="Shortcut for running a Bash script in the checkout"
    )
    parser.add_argument(
        "--command",
        help="Shell command to run after setup (on the head for multi-node jobs)",
    )
    parser.add_argument(
        "--image", default=os.environ.get("AISTUDIO_IMAGE", DEFAULT_IMAGE)
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Total nodes including the head; more than one enables Ray bootstrap",
    )
    parser.add_argument(
        "--submit", action="store_true", help="Submit to AIS; default is preview"
    )
    parser.add_argument(
        "--rdma", action="store_true", help="Request RDMA devices from AIS"
    )
    parser.add_argument(
        "--host-network", action="store_true", help="Use AIS host networking"
    )
    parser.add_argument(
        "--ray-root",
        default=os.environ.get("AREAL_RAY_ROOT", ""),
        help="Shared CPFS directory for Ray discovery (default: AREAL_RAY_ROOT or forwarded output root)",
    )
    parser.add_argument(
        "--pass-env",
        action="append",
        default=[],
        metavar="NAME",
        help="Forward one exported non-secret variable to the launcher (repeatable)",
    )
    parser.add_argument(
        "--launcher-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Append one Bash launcher argument (repeatable, use = for flags)",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    if args.nodes < 1 or not args.image.strip():
        parser.error("Specify a non-empty image and at least one node.")
    if args.launcher and args.command is not None:
        parser.error("Use either a launcher path or --command.")
    if args.launcher_arg and not args.launcher:
        parser.error(
            "--launcher-arg requires a launcher path; put arguments in --command instead."
        )
    try:
        launcher = launcher_path(repo, args.launcher) if args.launcher else None
        forwarded_env = collect_forwarded_env(args.pass_env)
        ray_root = (
            resolve_ray_root(args.ray_root, forwarded_env) if args.nodes > 1 else ""
        )
    except ValueError as exc:
        parser.error(str(exc))
    if launcher:
        command = shlex.join(
            ("exec", "bash", "--", launcher.as_posix(), *args.launcher_arg)
        )
    else:
        command = (COMMAND if args.command is None else args.command).strip()
        if not command:
            parser.error("Set COMMAND, pass --command, or specify a launcher path.")

    name = f"areal-{secrets.token_hex(4)}"
    sys.stdout.write(
        f"{name}: {CLUSTER}/{APP}, {args.nodes} node(s), "
        f"{args.nodes * NODE_SPEC['gpu_num']} B300, priority={PRIORITY}\n"
        f"Image: {args.image}\nCommand: {command}\n"
        f"Ray bootstrap: {'enabled (command runs on head only)' if args.nodes > 1 else 'disabled'}\n"
        f"RDMA: {args.rdma}; host networking: {args.host_network}\n"
    )
    if not args.submit:
        sys.stdout.write("Preview only. Add --submit to upload and submit.\n")
        return 0

    user_number = os.environ.get("AISTUDIO_USERNUMBER", "").strip()
    token = os.environ.get("AISTUDIO_TOKEN", "").strip()
    if not user_number.isascii() or not user_number.isdigit() or not token:
        parser.error(
            "Set AISTUDIO_USERNUMBER (digits) and AISTUDIO_TOKEN before submitting."
        )
    return (
        0
        if submit(
            repo,
            launcher,
            command=startup_command(
                command,
                launcher=launcher,
                job_name=name,
                nodes=args.nodes,
                rdma=args.rdma,
                ray_root=ray_root,
                forwarded_env=forwarded_env,
            ),
            name=name,
            image=args.image.strip(),
            nodes=args.nodes,
            user_number=user_number,
            token=token,
            rdma=args.rdma,
            host_network=args.host_network,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
