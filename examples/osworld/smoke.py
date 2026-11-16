"""End-to-end smoke test for the gateway-sandbox-backed DesktopEnv.

Verifies the full reset → step → evaluate → close cycle against a real
OSWorld task without spinning up the trainer. Pass the cluster endpoint and
secret token via env vars (``OSWORLD_SANDBOX_ENDPOINT`` and
``OSWORLD_SANDBOX_TOKEN``).

Run:
    python examples/osworld/smoke.py

Expected output (on a clean run): a real screenshot byte count, a successful
step, and ``evaluate result: 0.0`` (the agent didn't solve the task, but the
evaluator returned a real number — that's what we want to confirm).
"""

from __future__ import annotations

import json
import logging
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OSWORLD_ROOT = os.path.join(REPO_ROOT, "OSWorld")
AREAL_ROOT = os.path.join(REPO_ROOT, "AReaL")

# Ensure the AReaL repo is importable when running this script directly.
sys.path.insert(0, AREAL_ROOT)


def _pick_simple_task(osworld_root: str) -> dict:
    """Pick a task whose setup verbs all live in our supported set."""
    meta = json.load(
        open(os.path.join(osworld_root, "evaluation_examples", "test_small.json"))
    )
    supported = {"launch", "execute", "command", "sleep", "open", "activate_window"}
    for domain, eids in meta.items():
        for eid in eids:
            cfg = json.load(
                open(
                    os.path.join(
                        osworld_root,
                        "evaluation_examples",
                        "examples",
                        domain,
                        f"{eid}.json",
                    )
                )
            )
            verbs = {c.get("type") for c in cfg.get("config", [])}
            if verbs.issubset(supported):
                return cfg
    raise RuntimeError("no suitable task found in test_small.json")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    endpoint = os.environ.get("OSWORLD_SANDBOX_ENDPOINT", "")
    token = os.environ.get("OSWORLD_SANDBOX_TOKEN", "")
    if not endpoint or not token:
        raise SystemExit(
            "OSWORLD_SANDBOX_ENDPOINT and OSWORLD_SANDBOX_TOKEN must be set "
            "to the URL and authentication token of your remote sandbox "
            "gateway before running this smoke test."
        )

    from examples.osworld.workflow.gateway_sandbox import make_sandbox_desktop_env

    task_config = _pick_simple_task(OSWORLD_ROOT)
    print(f"[task] id={task_config.get('id')}")
    print(f"[task] instruction: {task_config.get('instruction')!r}")
    print(
        f"[task] config verbs: {[c.get('type') for c in task_config.get('config', [])]}"
    )

    env = make_sandbox_desktop_env(
        osworld_root=OSWORLD_ROOT,
        cluster_endpoint=endpoint,
        secret_token=token,
        cache_dir="/tmp/areal/sandbox_cache",
        require_a11y_tree=False,
    )

    try:
        print("\n--- reset(task_config) ---")
        obs = env.reset(task_config=task_config)
        print(f"reset ok; screenshot bytes={len(obs.get('screenshot') or b'')}")

        print("\n--- step (no-op pyautogui) ---")
        obs2, reward, done, info = env.step("pyautogui.moveTo(960, 540)", 1.0)
        print(
            f"step ok; reward={reward} done={done} screenshot bytes={len(obs2.get('screenshot') or b'')}"
        )

        print("\n--- evaluate ---")
        try:
            r = env.evaluate()
            print(f"evaluate result: {r}")
        except Exception:
            import traceback

            traceback.print_exc()

    finally:
        print("\n--- close ---")
        env.close()
        print("closed")


if __name__ == "__main__":
    main()
