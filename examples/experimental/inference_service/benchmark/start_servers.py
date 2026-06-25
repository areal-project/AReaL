"""Launch SGLang servers for benchmarking.

Starts Agent and/or User SGLang instances as background processes on a
single multi-GPU node, splitting GPUs via CUDA_VISIBLE_DEVICES.

Usage:
    python3 examples/experimental/inference_service/benchmark/start_servers.py \
        --model-path /models/Qwen3-235B-A22B-Instruct-2507
    python3 ... --model-path /models/Qwen3-235B --tp 8 --agent-only
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

import requests


def _gpu_count() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.DEVNULL, text=True
        )
        return len(out.strip().splitlines())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0


def _wait_for_server(
    url: str,
    pid: int,
    timeout: int = 300,
    label: str = "server",
    log_path: str = "",
) -> None:
    for i in range(1, timeout + 1):
        try:
            resp = requests.get(url, timeout=2)
            if resp.ok:
                print(f"  ✓ {label} ready ({i}s)")
                return
        except requests.RequestException:
            pass

        try:
            os.kill(pid, 0)
        except OSError:
            msg = f"  ✗ {label} crashed."
            if log_path:
                msg += f" Check {log_path}"
            print(msg)
            sys.exit(1)

        time.sleep(1)

    print(f"  ✗ {label} timed out after {timeout}s")
    sys.exit(1)


def _launch_sglang(
    model_path: str,
    model_name: str,
    tp: int,
    port: int,
    gpu_ids: str,
    log_path: str,
    extra_flags: list[str] | None = None,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--served-model-name",
        model_name,
        "--tp",
        str(tp),
        "--port",
        str(port),
        "--host",
        "0.0.0.0",
        "--context-length",
        "262144",
        "--tool-call-parser",
        "qwen25",
        "--enable-metrics",
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_ids}
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    return proc.pid


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch SGLang servers for benchmark")
    parser.add_argument("--model-path", required=True, help="Path to model weights")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Served model name (default: basename of model-path)",
    )
    parser.add_argument("--agent-port", type=int, default=30000)
    parser.add_argument("--user-port", type=int, default=30001)
    parser.add_argument(
        "--tp", type=int, default=4, help="Tensor parallelism per instance"
    )
    parser.add_argument(
        "--agent-only", action="store_true", help="Only start the agent server"
    )
    parser.add_argument("--log-dir", default="./logs")
    args = parser.parse_args()

    model_name = args.model_name or os.path.basename(args.model_path)
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    num_gpus = _gpu_count()
    if not args.agent_only and args.tp * 2 > num_gpus:
        print(
            f"ERROR: Need {args.tp * 2} GPUs for Agent(TP={args.tp}) + User(TP={args.tp}), "
            f"but only {num_gpus} available. Use --agent-only or --tp {num_gpus // 2}."
        )
        sys.exit(1)

    agent_gpus = ",".join(str(i) for i in range(args.tp))
    agent_log = os.path.join(log_dir, "agent-sglang.log")

    print(f"  Agent SGLang: {model_name}")
    print(f"  CUDA_VISIBLE_DEVICES={agent_gpus}, TP={args.tp}, port={args.agent_port}")

    agent_pid = _launch_sglang(
        args.model_path,
        model_name,
        args.tp,
        args.agent_port,
        agent_gpus,
        agent_log,
        extra_flags=["--disable-radix-cache"],
    )
    print(f"  PID: {agent_pid}, log: {agent_log}")

    pids = [agent_pid]
    user_pid = None

    if not args.agent_only:
        user_gpus = ",".join(str(i) for i in range(args.tp, args.tp * 2))
        user_log = os.path.join(log_dir, "user-sglang.log")

        print(f"\n  User SGLang: {model_name}")
        print(
            f"  CUDA_VISIBLE_DEVICES={user_gpus}, TP={args.tp}, port={args.user_port}"
        )

        user_pid = _launch_sglang(
            args.model_path,
            model_name,
            args.tp,
            args.user_port,
            user_gpus,
            user_log,
        )
        print(f"  PID: {user_pid}, log: {user_log}")
        pids.append(user_pid)

    print("\nWaiting for servers to be ready...")
    _wait_for_server(
        f"http://127.0.0.1:{args.agent_port}/v1/models",
        agent_pid,
        label="Agent SGLang",
        log_path=agent_log,
    )
    if user_pid is not None:
        _wait_for_server(
            f"http://127.0.0.1:{args.user_port}/v1/models",
            user_pid,
            label="User SGLang",
            log_path=user_log,
        )

    pid_str = " ".join(str(p) for p in pids)
    print(f"\n  Servers running. To stop: kill {pid_str}")
    print(f"  Agent: http://127.0.0.1:{args.agent_port}")
    if user_pid is not None:
        print(f"  User:  http://127.0.0.1:{args.user_port}")

    print("\nPress Ctrl+C to stop all servers.")
    try:
        signal.pause()
    except KeyboardInterrupt:
        print("\nStopping servers...")
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass


if __name__ == "__main__":
    main()
