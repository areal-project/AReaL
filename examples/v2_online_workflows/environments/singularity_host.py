# SPDX-License-Identifier: Apache-2.0
"""Host-side Apptainer tool service, using only the Python 3.6.8+ standard library.

Run this module on the compute host, outside the modern training SIF.
Keep this module and process.py compatible with Python 3.6.8, without backports.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socketserver
import subprocess
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from examples.v2_online_workflows.environments.process import run_limited

_MAX_MESSAGE = 128 * 1024 * 1024


def _read(stream):
    row = stream.readline(_MAX_MESSAGE + 1)
    if not row.endswith(b"\n") or len(row) > _MAX_MESSAGE:
        raise ValueError("Incomplete or oversized host message")
    return json.loads(row)


def _unlink(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


class _SIFRuntime:
    """One serialized overlay per task; every call is launched by the host runtime."""

    def __init__(
        self,
        kind,
        settings,
        task_id,
        image_id,
        sif,
        executable,
        overlay_mb,
        proc_mode="default",
        repo=None,
        base_commit=None,
    ):
        self.kind = kind
        if kind == "repository":
            if not re.fullmatch("[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo or ""):
                raise ValueError("Repository must be owner/name")
            if not re.fullmatch("[0-9a-fA-F]{40}", base_commit or ""):
                raise ValueError("Repository base_commit must be a full commit hash")
        self.settings, self.task_id, self.image_id = settings, task_id, image_id
        self.repo, self.base_commit = repo, base_commit
        self.workdir = "/" + repo.split("/")[1] if kind == "repository" else "/work"
        self.directory = settings.root / task_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.container = None
        self.records, self.call_timings = [], []
        self.sif, self.executable, self.overlay_mb = sif, executable, overlay_mb
        self.proc_mode = proc_mode
        self.overlay = self.directory / "overlay.img"
        self.scratch = self.directory / "scratch"
        self.patches = None
        # Do not inherit automatic mounts or *_ENV injections from the job.
        self.host_env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("SINGULARITY_", "APPTAINER_"))
        }

    def _launch(self, *program, pwd="/"):
        command = [
            self.executable,
            "exec",
            "--fakeroot",
            "--containall",
            "--cleanenv",
            "--no-home",
            "--no-mount",
            "hostfs,bind-paths,cwd" + (",proc" if self.proc_mode == "none" else ""),
            "--net",
            "--network",
            "none",
            "--overlay",
            str(self.overlay),
            "--workdir",
            str(self.scratch),
            "--pwd",
            pwd,
            "--env",
            "PYTHONDONTWRITEBYTECODE=1,CUDA_VISIBLE_DEVICES=,_JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false",
        ]
        if self.patches:
            command += ["--bind", str(self.patches) + ":/patches:ro"]
        return [*command, str(self.sif), *program]

    def _command(self, *program):
        return self._launch(
            *program, pwd=self.workdir if self.kind == "repository" else "/work"
        )

    def _control(self, *args):
        if args[0] != "exec":
            raise ValueError("Only container exec is supported by the SIF adapter")
        return self._checked(self._command(*args[2:]))

    def _checked(self, command):
        result = run_limited(
            command,
            env=self.host_env,
            timeout=60,
            output_bytes=1024 * 1024,
            process_group=True,
        )
        if result["status"] != "completed" or result["exit_code"] != 0:
            raise RuntimeError(
                "Singularity initialization/control failed: " + result["output"]
            )
        return result["output"].strip()

    def _start(self):
        if self.container is not None:
            return
        self.scratch.mkdir(exist_ok=True)
        self._checked(
            [
                self.executable,
                "overlay",
                "create",
                "--fakeroot",
                "--sparse",
                "--size",
                str(self.overlay_mb),
                str(self.overlay),
            ]
        )
        self.container = "sif-" + uuid.uuid4().hex
        if self.kind == "python":
            self._checked(self._launch("mkdir", "-p", "/work"))
        else:
            self._control(
                "exec", self.container, "bash", "-lc", "command -v timeout >/dev/null"
            )
            self._control(
                "exec", self.container, "git", "reset", "--hard", self.base_commit
            )
            self._control("exec", self.container, "git", "clean", "-ffd")
            if (
                self._control("exec", self.container, "git", "rev-parse", "HEAD")
                != self.base_commit
            ):
                raise RuntimeError(
                    "SIF repository does not match the frozen base commit"
                )

    def execute(self, value):
        python = self.kind == "python"
        if (
            not isinstance(value, str)
            or len(value.encode()) > (self.settings.max_code_bytes if python else 65536)
            or (not python and (not value.strip()))
        ):
            return json.dumps(
                {
                    "status": "invalid_code" if python else "invalid_command",
                    "error": "Python code is missing or too large"
                    if python
                    else "A bounded command is required",
                }
            )
        if python and len(self.records) >= self.settings.max_calls:
            return json.dumps(
                {"status": "call_limit", "error": "Task tool budget exhausted"}
            )
        started, name = time.time(), None
        try:
            self._start()
            name = self.container
            program = (
                ["python", "-I", "-u", "-c", value]
                if python
                else [
                    "timeout",
                    "--kill-after=5s",
                    str(self.settings.timeout),
                    "bash",
                    "-lc",
                    value,
                ]
            )
            result = run_limited(
                self._command(*program),
                env=self.host_env,
                process_group=True,
                timeout=self.settings.timeout + (0 if python else 10),
                output_bytes=self.settings.max_output_bytes,
                truncate=True,
                separate_stderr=python,
                preserve_tail=python,
            )
            if result["exit_code"] in ((125, 126, 127) if python else (125,)):
                result["status"] = "infrastructure_error"
            if not python and result["exit_code"] == 124:
                result["status"] = "timeout"
            if python and result["status"] != "completed":
                self.close()
                result["environment_reset"] = True
        except (OSError, subprocess.SubprocessError) as exc:
            result = {"status": "infrastructure_error", "error": str(exc)}
        record = {
            "task_id": self.task_id,
            "call": len(self.records) + 1,
            "started": started,
            "finished": time.time(),
            "image_id": self.image_id,
            "container": name,
            "code" if python else "command": value,
            **result,
        }
        self.records.append(record)
        with (self.directory / "tools.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return json.dumps(result, ensure_ascii=False)

    def invoke(self, name, arguments):
        clock = time.perf_counter()
        record = {
            "name": name,
            "started": time.time(),
            "finished": None,
            "seconds": None,
        }
        self.call_timings.append(record)
        try:
            return self.execute(
                arguments.get("code" if self.kind == "python" else "command")
            )
        finally:
            record.update(finished=time.time(), seconds=time.perf_counter() - clock)

    def export_artifacts(self):
        if self.kind == "python":
            return {}
        self._start()
        self._control("exec", self.container, "git", "add", "--intent-to-add", "--all")
        result = run_limited(
            self._command("git", "diff", "--binary", self.base_commit, "--"),
            timeout=60,
            output_bytes=self.settings.max_patch_bytes,
            process_group=True,
            env=self.host_env,
        )
        if result["status"] != "completed" or result["exit_code"] != 0:
            raise RuntimeError("Cannot export the complete candidate patch")
        (self.directory / "candidate.patch").write_text(
            result["output"], encoding="utf-8"
        )
        return {"patch": result["output"], "base_commit": self.base_commit}

    def metadata(self):
        metadata = {
            "kind": self.kind,
            "runtime_task_id": self.task_id,
            "image_id": self.image_id,
            "calls": self.records,
            "timing_calls": self.call_timings,
            "backend": "singularity",
            "sif": str(self.sif),
            "host_worker_capacity": self.host_worker_capacity,
            "proc_mode": self.proc_mode,
            "resource_limits": "Requires site Slurm job cgroup; timeout/output bounded per call; no per-task CPU/RAM cgroup",
        }
        if self.kind == "repository":
            metadata.update(repo=self.repo, base_commit=self.base_commit)
        return metadata

    def close(self):
        self.container = None
        _unlink(self.overlay)
        shutil.rmtree(self.scratch, ignore_errors=True)


class Host:
    """Private job-scoped registry; manifest images are pinned by actual SIF digest."""

    def __init__(
        self,
        root,
        images,
        *,
        executable="singularity",
        workers=4,
        overlay_mb=4096,
        proc_mode="default",
    ):
        if proc_mode not in ("default", "none"):
            raise ValueError("proc_mode must be default or none")
        self.proc_mode = proc_mode
        if workers < 1 or overlay_mb < 64:
            raise ValueError("Positive workers and overlay size >=64 MiB are required")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.executable, self.overlay_mb, self.workers = (
            executable,
            overlay_mb,
            workers,
        )
        self.slots = threading.BoundedSemaphore(workers)
        self.lock = threading.RLock()
        self.tasks, self.aliases, self.images = {}, {}, {}
        for alias, entry in json.loads(images.read_text(encoding="utf-8")).items():
            path = Path(entry["sif"]).resolve(strict=True)
            with path.open("rb") as stream:
                digest = (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                    if hasattr(hashlib, "file_digest")
                    else _sha256(stream)
                )
            if path.suffix != ".sif" or digest != entry["sha256"]:
                raise ValueError("SIF digest does not match its image manifest")
            identity = "sha256:" + digest
            self.aliases[alias] = identity
            self.images[identity] = path

    def dispatch(self, action, **payload):
        if action == "resolve":
            image = payload["image"]
            identity = self.aliases.get(image, image)
            if identity not in self.images:
                raise ValueError("Image is not registered in this host's SIF manifest")
            return identity
        if action == "health":
            return {
                "backend": "singularity",
                "images": len(self.images),
                "pid": os.getpid(),
                "active_tasks": len(self.tasks),
                "proc_mode": self.proc_mode,
            }
        if action == "open":
            return self.open(**payload)
        if action == "grade":
            with self.slots:
                return self.grade(**payload)
        task_id = payload["task_id"]
        with self.lock:
            entry = self.tasks.get(task_id)
            if entry is None and action == "close":
                return {}
            runtime, lock = entry
        with lock:
            if action == "invoke":
                expected = (
                    "math_python" if runtime.kind == "python" else "workspace_exec"
                )
                if payload["name"] != expected:
                    raise ValueError("Unknown environment tool")
                with self.slots:
                    value = runtime.invoke(payload["name"], payload["arguments"])
                return {"value": value, "timing": runtime.call_timings[-1]}
            if action == "metadata":
                return runtime.metadata()
            if action == "export":
                with self.slots:
                    return runtime.export_artifacts()
            if action == "close":
                metadata = runtime.metadata()
                runtime.close()
                with self.lock:
                    self.tasks.pop(task_id, None)
                return metadata
        raise ValueError("Unknown host action")

    def open(self, kind, task_id, image, settings, options):
        if not re.fullmatch("[A-Za-z0-9_-]+", task_id):
            raise ValueError("Task ID must be a safe single directory name")
        defaults = {
            "python": dict(
                timeout=15.0,
                max_output_bytes=65536,
                max_code_bytes=16384,
                memory_mb=512,
                max_calls=16,
                cpus=1.0,
            ),
            "repository": dict(
                timeout=120.0,
                max_output_bytes=65536,
                memory_mb=4096,
                cpus=2.0,
                max_patch_bytes=8 * 1024 * 1024,
            ),
        }[kind]
        limits = {
            **defaults,
            **{
                key: value
                for key, value in settings.items()
                if key not in ("root", "image")
            },
        }
        if limits.keys() != defaults.keys() or min(limits.values()) <= 0:
            raise ValueError("Known positive tool resource limits are required")
        identity = self.dispatch("resolve", image=image)
        runtime = _SIFRuntime(
            kind,
            SimpleNamespace(**limits, root=self.root, image=identity),
            task_id,
            image_id=identity,
            **options,
            sif=self.images[identity],
            executable=self.executable,
            overlay_mb=self.overlay_mb,
            proc_mode=self.proc_mode,
        )
        runtime.host_worker_capacity = self.workers
        with self.lock:
            if task_id in self.tasks:
                raise ValueError("Task already exists")
            self.tasks[task_id] = (runtime, threading.RLock())

    def grade(
        self, task_id, image, repo, base_commit, patch, test_patch, test_cmds, timeout
    ):
        if not 0 < timeout <= 3600:
            raise ValueError("Grading timeout must be within 3600 seconds")
        self.open(
            "repository",
            task_id,
            image,
            {"timeout": timeout, "max_output_bytes": 64 * 1024 * 1024},
            {"repo": repo, "base_commit": base_commit},
        )
        runtime, lock = self.tasks[task_id]
        try:
            with lock:
                runtime.patches = runtime.directory / "patches"
                runtime.patches.mkdir()
                (runtime.patches / "patch.diff").write_text(patch, encoding="utf-8")
                (runtime.patches / "test.diff").write_text(test_patch, encoding="utf-8")
                result = json.loads(
                    runtime.execute(
                        "\n".join(
                            [
                                "set -e",
                                "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/patch.diff",
                                "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/test.diff",
                                *test_cmds,
                            ]
                        )
                    )
                )
                if result["status"] != "completed" or result["exit_code"] in (124, 125):
                    raise RuntimeError(
                        "SIF test execution could not be scored completely"
                    )
                return [result["exit_code"], result["output"]]
        finally:
            self.dispatch("close", task_id=task_id)

    def close(self):
        for task_id in list(self.tasks):
            self.dispatch("close", task_id=task_id)


def _sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            result = {"result": self.server.host.dispatch(**_read(self.rfile))}
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        self.wfile.write(json.dumps(result).encode() + b"\n")


class HostServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path, host):
        self.host = host
        super().__init__(str(path), _Handler)
        Path(path).chmod(0o600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.required = True
    serve = commands.add_parser("serve")
    serve.add_argument("--socket", type=Path, required=True)
    serve.add_argument("--root", type=Path, required=True)
    serve.add_argument("--images", type=Path, required=True)
    serve.add_argument("--executable", default="singularity")
    serve.add_argument("--workers", type=int, default=4)
    serve.add_argument("--overlay-mb", type=int, default=4096)
    serve.add_argument(
        "--proc-mode",
        choices=("default", "none"),
        default="default",
        help="Keep runtime /proc mounts, or disable them for task tools and grading",
    )
    args = parser.parse_args()
    os.umask(0o077)
    args.socket.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    host = Host(
        args.root,
        args.images,
        executable=args.executable,
        workers=args.workers,
        overlay_mb=args.overlay_mb,
        proc_mode=args.proc_mode,
    )

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    bound = False
    try:
        with HostServer(args.socket, host) as server:
            bound = True
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        host.close()
        if bound:
            _unlink(args.socket)


if __name__ == "__main__":
    main()
