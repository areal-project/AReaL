# SPDX-License-Identifier: Apache-2.0

"""Long-running supervisor for an ``areal inf`` service.

Spawned (detached) by ``areal inf run``. Holds the v2 ``RolloutControllerV2``
which itself manages the sglang workers, router, gateway, and data-proxies.

Lifecycle:

  1. parse args (--config / --service / --overrides)
  2. ``load_expr_config`` -> PPOConfig
  3. resolve log_dir (same v2 training layout) and rebind stdout/stderr to
     ``{log_dir}/main.log`` so users can `tail -f` it from `areal inf logs`.
  4. ``SlurmScheduler`` (or Local / Ray) per ``config.scheduler.type``
  5. ``RolloutControllerV2(config.rollout, scheduler).initialize(role="rollout")``
     -- this spawns workers, router, gateway, proxies as worker sub-processes
     via the scheduler.  When ``initialize`` returns the stack is healthy.
  6. write ``ServiceState`` + ready marker
  7. install SIGTERM/SIGINT handler that calls ``controller.destroy()`` and
     removes the state file -- this is the SAME teardown path that
     ``PPOTrainer.close()`` uses.
  8. sleep forever; SIGTERM is the only way out.

If ANY step before sleep raises, the supervisor writes a ``failed`` marker,
attempts ``controller.destroy()`` if the controller exists, and exits 1.
``areal inf run`` watches both markers and surfaces the failure to the user
instead of waiting for the launch timeout.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _build_scheduler(config: Any):
    from areal.infra.scheduler.local import LocalScheduler
    from areal.infra.scheduler.ray import RayScheduler
    from areal.infra.scheduler.slurm import SlurmScheduler

    t = config.scheduler.type
    if t == "local":
        return LocalScheduler(exp_config=config)
    if t == "ray":
        return RayScheduler(exp_config=config)
    if t == "slurm":
        return SlurmScheduler(exp_config=config)
    raise SystemExit(f"Unknown scheduler type: {t!r}")


def _build_server_args(config: Any) -> dict:
    from areal.api.alloc_mode import ModelAllocation
    from areal.api.cli_args import SGLangConfig, vLLMConfig

    alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")
    backend = alloc.backend
    if backend == "sglang":
        return SGLangConfig.build_args(
            sglang_config=config.sglang,
            tp_size=alloc.parallel.tp_size,
            pp_size=alloc.parallel.pp_size,
            base_gpu_id=0,
        )
    if backend == "vllm":
        return vLLMConfig.build_args(
            vllm_config=config.vllm,
            tp_size=alloc.parallel.tp_size,
            pp_size=alloc.parallel.pp_size,
        )
    raise SystemExit(f"Unsupported rollout backend for `areal inf`: {backend!r}")


def _redirect_stdio(log_path: Path) -> None:
    """Send all subsequent print / logging output to the main log file.

    Uses ``os.dup2`` so any C-level / subprocess-inherited fd 1/2 also lands
    in the same file. ``line buffering = 1`` because users will be tailing it.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    # python-level streams: rebind so getLogger handlers also pick up the new fd
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)


def _run(name: str, config_path: Path, overrides: list[str]) -> int:
    from areal.experimental.cli.inf_config import load_inference_config
    from areal.experimental.cli.inf_state import (
        ServiceState,
        service_failed_marker,
        service_log_dir_for_config,
        service_ready_marker,
    )
    from areal.experimental.inference_service.controller.controller import (
        RolloutControllerV2,
    )
    from areal.utils.logging import getLogger

    config, resolved_name = load_inference_config(config_path, overrides)
    log_dir = service_log_dir_for_config(config)
    main_log = log_dir / "main.log"
    _redirect_stdio(main_log)

    logger = getLogger("InfSupervisor")
    logger.info("=== inf supervisor starting (pid=%d) ===", os.getpid())
    logger.info("service=%s config=%s log_dir=%s", name, config_path, log_dir)
    if name != resolved_name:
        logger.warning(
            "Service name mismatch: CLI passed %r, config resolves to %r. "
            "Using CLI value.",
            name,
            resolved_name,
        )

    controller: RolloutControllerV2 | None = None
    try:
        logger.info("Building scheduler (type=%s) ...", config.scheduler.type)
        scheduler = _build_scheduler(config)

        logger.info("Initializing RolloutControllerV2 ...")
        controller = RolloutControllerV2(config=config.rollout, scheduler=scheduler)
        server_args = _build_server_args(config)
        controller.initialize(role="rollout", server_args=server_args)
        logger.info(
            "RolloutControllerV2 ready (gateway=%s, router=%s, %d server(s))",
            controller._gateway_addr,
            controller._router_addr,
            len(controller._server_infos),
        )

        state = ServiceState(
            name=name,
            supervisor_pid=os.getpid(),
            config_path=str(config_path),
            log_dir=str(log_dir),
            overrides=list(overrides),
            gateway_addr=controller._gateway_addr or "",
            router_addr=controller._router_addr or "",
            server_addrs=[
                f"http://{info.host}:{info.port}" for info in controller._server_infos
            ],
            created_at=time.time(),
            ready_at=time.time(),
        )
        state.save()
        service_ready_marker(name).write_text("ready\n")
        logger.info("Service %r ready; supervisor pid=%d", name, os.getpid())
    except BaseException as e:
        logger.error("Supervisor init failed: %s", e)
        logger.error(traceback.format_exc())
        # Write failed marker so the parent CLI exits the wait loop quickly.
        try:
            service_failed_marker(name).write_text(f"{type(e).__name__}: {e}\n")
        except Exception:
            pass
        if controller is not None:
            try:
                controller.destroy()
            except Exception:
                logger.error(
                    "controller.destroy() during failure cleanup also failed:\n%s",
                    traceback.format_exc(),
                )
        return 1

    stop_event = {"stop": False}

    def _teardown(signum, _frame):
        if stop_event["stop"]:
            return
        stop_event["stop"] = True
        logger.info("Received signal %d, tearing down ...", signum)
        try:
            controller.destroy()
        except Exception:
            logger.error("controller.destroy() failed:\n%s", traceback.format_exc())
        try:
            state.remove()
        except Exception:
            pass
        logger.info("Teardown complete; supervisor exiting.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _teardown)
    signal.signal(signal.SIGINT, _teardown)

    while not stop_event["stop"]:
        time.sleep(1.0)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="areal-inf-supervisor", add_help=False)
    p.add_argument("--name", required=True)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    args = p.parse_args()
    overrides = args.overrides or []
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    try:
        return _run(args.name, args.config, overrides)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
