#!/usr/bin/env python3
"""Automated human-in-the-loop (HITL) online RL demo.

Procedure:
  1. Launch ``online_rollout.py`` as a subprocess and wait for the gateway
     address to appear in its log output.
  2. Patch ``~/.zeroclaw/config.toml`` to point at the local gateway
     (restored on exit).
  3. Run ``batch_size`` HITL rounds: ask the model a question via zeroclaw,
     optionally give corrective feedback, then POST a reward to the
     ``/rl/set_reward`` endpoint.
  4. Wait for ``online_rollout.py`` to print a ``Rollout complete`` line
     confirming that all trajectories were collected.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]

# ── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_ACTOR_PATH = "Qwen/Qwen3-0.6B"
DEFAULT_ADMIN_KEY = "sk-test123456"
DEFAULT_REQUEST_TIMEOUT = 3600
DEFAULT_GATEWAY_WAIT_SECS = 600
DEFAULT_QUESTION = "how many r's are in the word strawberry?"
DEFAULT_INFERENCE_BACKEND = "sglang"
CORRECT_ANSWER_RE = re.compile(r"\b3\b|three", re.IGNORECASE)
BATCH_SIZE = 4
ROLLOUT_COMPLETE_WAIT_SECS = 60


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _print_header(title: str) -> None:
    sep = "=" * 64
    print(f"\n{sep}\n  {title}\n{sep}")


# ── Zeroclaw config helpers ────────────────────────────────────────────────


def _patch_zeroclaw_config(
    config_path: Path,
    gateway_addr: str,
    api_key: str,
    model: str | None = None,
) -> Path:
    backup = config_path.with_suffix(".demo_bak")
    shutil.copy2(config_path, backup)

    text = config_path.read_text()
    text = re.sub(
        r'^default_provider\s*=\s*".*"',
        f'default_provider = "custom:{gateway_addr}"',
        text,
        flags=re.MULTILINE,
    )
    if re.search(r"^api_key\s*=", text, re.MULTILINE):
        text = re.sub(
            r'^api_key\s*=\s*".*"',
            f'api_key = "{api_key}"',
            text,
            flags=re.MULTILINE,
        )
    else:
        text = f'api_key = "{api_key}"\n' + text

    if model is not None:
        text = re.sub(
            r'^default_model\s*=\s*".*"',
            f'default_model = "{model}"',
            text,
            flags=re.MULTILINE,
        )

    config_path.write_text(text)
    return backup


def _restore_zeroclaw_config(config_path: Path, backup: Path) -> None:
    if backup.exists():
        shutil.copy2(backup, config_path)
        backup.unlink()
        print("  Restored original zeroclaw config.")


# ── Reward submission ──────────────────────────────────────────────────────


def _set_reward(gateway_addr: str, api_key: str, reward: float) -> None:
    print(f"    Setting reward={reward}")
    resp = requests.post(
        f"{gateway_addr}/rl/set_reward",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={"reward": reward},
        timeout=10,
    )
    resp.raise_for_status()


# ── Single HITL round ─────────────────────────────────────────────────────


def _do_round(
    gateway_addr: str,
    api_key: str,
    question: str,
    label: str,
) -> None:
    session_file = tempfile.mktemp(suffix=".json", prefix="zeroclaw_session_")

    print(f"\n  ── {label} ──")
    print(f"  Q: {question}")

    # First attempt
    try:
        resp_text = _strip_ansi(
            subprocess.check_output(
                [
                    "zeroclaw",
                    "agent",
                    "-m",
                    question,
                    "--session-state-file",
                    session_file,
                ],
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "ZEROCLAW_API_KEY": api_key},
            )
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  zeroclaw failed: {exc}")
        resp_text = ""
    print(f"  A: {resp_text}")

    if CORRECT_ANSWER_RE.search(resp_text):
        print("  ✔ Correct on first try.")
        _set_reward(gateway_addr, api_key, 1.0)
        Path(session_file).unlink(missing_ok=True)
        return

    # Second attempt with corrective feedback
    correction = (
        "That's wrong. The word 'strawberry' contains 3 r's. "
        f"Let me ask once more: {question}"
    )
    print("  ✘ Wrong — giving corrective feedback and asking again ...")
    try:
        resp_text = _strip_ansi(
            subprocess.check_output(
                [
                    "zeroclaw",
                    "agent",
                    "-m",
                    correction,
                    "--session-state-file",
                    session_file,
                ],
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "ZEROCLAW_API_KEY": api_key},
            )
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  zeroclaw failed: {exc}")
        resp_text = ""
    print(f"  A: {resp_text}")

    if CORRECT_ANSWER_RE.search(resp_text):
        print("  ✔ Correct on second try.")
        _set_reward(gateway_addr, api_key, 1.0)
    else:
        print("  ✘ Still wrong after two attempts — setting reward to 0.")
        _set_reward(gateway_addr, api_key, 0.0)

    Path(session_file).unlink(missing_ok=True)


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated HITL online RL demo with zeroclaw"
    )
    parser.add_argument(
        "--actor-path", default=DEFAULT_ACTOR_PATH, help="HuggingFace model path"
    )
    parser.add_argument("--admin-key", default=DEFAULT_ADMIN_KEY, help="Admin API key")
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--gateway-wait",
        type=int,
        default=DEFAULT_GATEWAY_WAIT_SECS,
        help="Seconds to wait for gateway readiness",
    )
    parser.add_argument(
        "--question", default=DEFAULT_QUESTION, help="Question for each HITL round"
    )
    parser.add_argument(
        "--inference-backend",
        choices=("sglang", "vllm"),
        default=DEFAULT_INFERENCE_BACKEND,
        help="Inference backend used by online_rollout.py",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="External API URL (enables external model mode)",
    )
    parser.add_argument(
        "--provider-api-key",
        default=None,
        help="API key for the external provider",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for the gateway controller",
    )
    args = parser.parse_args()

    online_rollout = (
        REPO_ROOT / "examples/experimental/inference_service/online_rollout.py"
    )
    config_yaml = (
        REPO_ROOT / "examples/experimental/inference_service/online_rollout.yaml"
    )
    zeroclaw_config = Path.home() / ".zeroclaw" / "config.toml"

    rollout_log = Path(tempfile.mktemp(suffix=".log", prefix="online_rollout_"))
    rollout_proc: subprocess.Popen | None = None
    zeroclaw_backup: Path | None = None

    def cleanup(signum=None, frame=None):
        _print_header("Cleanup")
        if rollout_proc is not None and rollout_proc.poll() is None:
            print(f"  Stopping online_rollout.py (PID {rollout_proc.pid}) ...")
            rollout_proc.terminate()
            try:
                rollout_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                rollout_proc.kill()
        if zeroclaw_backup is not None:
            _restore_zeroclaw_config(zeroclaw_config, zeroclaw_backup)
        print(f"  Rollout log preserved at: {rollout_log}")

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # ── Step 1: Launch online_rollout.py ──
        _print_header("Step 1: Launch online_rollout.py")
        log_fh = open(rollout_log, "w")
        rollout_cmd = [
            sys.executable,
            str(online_rollout),
            "--config",
            str(config_yaml),
            f"actor.path={args.actor_path}",
            f"rollout.backend={args.inference_backend}:d1",
            f"rollout.admin_api_key={args.admin_key}",
            "rollout._version=v2",
            f"rollout.request_timeout={args.request_timeout}",
        ]
        if args.api_url:
            rollout_cmd.extend(["--api-url", args.api_url])
        if args.provider_api_key:
            rollout_cmd.extend(["--provider-api-key", args.provider_api_key])
        if args.model:
            rollout_cmd.extend(["--model", args.model])
        rollout_proc = subprocess.Popen(
            rollout_cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
        print(f"  PID : {rollout_proc.pid}")
        print(f"  Log : {rollout_log}")
        print(f"  Waiting for gateway address (up to {args.gateway_wait}s) ...")

        gateway_addr = ""
        gateway_re = re.compile(r"Proxy gateway available at (http://\S+)")
        deadline = time.monotonic() + args.gateway_wait
        while time.monotonic() < deadline:
            if rollout_proc.poll() is not None:
                log_fh.close()
                tail = rollout_log.read_text().splitlines()[-40:]
                print("  ERROR: online_rollout.py exited prematurely. Log tail:")
                print("\n".join(tail))
                sys.exit(1)
            log_content = _strip_ansi(rollout_log.read_text())
            match = gateway_re.search(log_content)
            if match:
                gateway_addr = match.group(1)
                break
            time.sleep(1)

        if not gateway_addr:
            log_fh.close()
            tail = rollout_log.read_text().splitlines()[-40:]
            print("  ERROR: Timed out waiting for gateway. Log tail:")
            print("\n".join(tail))
            sys.exit(1)
        print(f"  Gateway: {gateway_addr}")

        # ── Step 2: Patch zeroclaw config ──
        _print_header("Step 2: Update ~/.zeroclaw/config.toml")
        zeroclaw_backup = _patch_zeroclaw_config(
            zeroclaw_config, gateway_addr, args.admin_key, model=args.model
        )
        print("  Done.")

        # ── Steps 3–4: HITL rounds ──
        _print_header(f"Steps 3–4  ({BATCH_SIZE} HITL rounds)")
        for i in range(BATCH_SIZE):
            _do_round(gateway_addr, args.admin_key, args.question, f"Trajectory {i}")

        # ── Step 5: Verify rollout completion ──
        _print_header("Step 5: Check online_rollout output for databatch")
        print("  Waiting for rollout to process ...")
        wait_deadline = time.monotonic() + ROLLOUT_COMPLETE_WAIT_SECS
        found = False
        while time.monotonic() < wait_deadline:
            if "Rollout complete" in rollout_log.read_text():
                found = True
                break
            time.sleep(2)

        print()
        tail = rollout_log.read_text().splitlines()[-40:]
        print("  ── Rollout log (last 40 lines) ──")
        for line in tail:
            print(f"  {line}")
        print()

        if found:
            for line in rollout_log.read_text().splitlines():
                if "Rollout complete" in line:
                    print(f"  ✔ Databatch detected:\n  {line}")
                    break
        else:
            print("  ✘ No 'Rollout complete' message found yet.")
            print("    The rollout may still be collecting trajectories.")

        print("\n  Demo finished.")

    finally:
        cleanup()


if __name__ == "__main__":
    main()
