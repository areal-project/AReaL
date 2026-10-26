"""Three bounded native GPU acceptance runs; --prepare-only starts no services."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from areal.elastic.control.resource_reconfiguration import ResourceLayout
from areal.utils.atomic import atomic_json


def layouts(case: str, devices: tuple[int, ...]) -> tuple[int, list]:
    if len(devices) != 8 or len(set(devices)) != 8 or any(d < 0 for d in devices):
        raise ValueError("Provide eight distinct device indices from your allocation")
    if case == "actor":
        return 2, [4, 2]
    if case == "rollout":
        return 4, [
            ResourceLayout(4, devices[:4], devices[6:]),
            ResourceLayout(4, devices[:4], devices[4:]),
        ]
    if case == "joint":
        return 4, [
            ResourceLayout(2, devices[:2], devices[2:]),
            ResourceLayout(4, devices[:4], devices[4:]),
        ]
    raise ValueError("Unknown acceptance case")


def prepare(args):
    from omegaconf import OmegaConf

    from examples.task_workflows.benchmarks.dapo_math import load_tasks

    from areal.api.cli_args import PPOConfig, parse_cli_args, to_structured_cfg

    output, model = args.output.absolute(), args.model.absolute()
    if not model.is_dir():
        raise ValueError("Model must be a pinned local directory")
    tasks, manifest = load_tasks(args.tasks)
    if len(tasks) < 12:
        raise ValueError("Acceptance needs twelve unique frozen tasks")
    tasks = tasks[:12]
    initial_dp, targets = layouts(args.case, tuple(args.devices))
    # All substitutions are local to this dedicated coordinator process.
    os.environ.update(
        AREAL_TASK_RUN_ROOT=str(output / "logs"),
        AREAL_TASK_STATE_ROOT=str(output / "state"),
        AREAL_MODEL_PATH=str(model),
        AREAL_CHECKPOINT_ESTIMATED_BYTES=str(args.estimate_gib * 1024**3),
    )
    for name in ("AREAL_INFERENCE_ADMIN_KEY", "AREAL_ACTOR_ADMIN_KEY"):
        os.environ.setdefault(name, os.urandom(32).hex())
    overrides = {
        "experiment_name": "elastic-reconfiguration",
        "trial_name": args.case,
        "actor.backend": f"fsdp:d{initial_dp}",
        "rollout.backend": "sglang:d4",
        "actor.weight_update_mode": "awex",
        "actor_lifecycle.mode": "on_demand",
        "actor_lifecycle.actor_devices": args.devices[:4],
        "actor_lifecycle.rollout_devices": args.devices[4:],
        "actor_lifecycle.idle_timeout": 86400,
        "actor_lifecycle.min_ready_seconds": 0,
        "actor_lifecycle.max_recovery_attempts": 0,
        "actor_lifecycle.checkpoint_keep": 2,
        "actor_lifecycle.checkpoint_free_space_floor": 2 * 1024**3,
        "total_train_steps": 3,
        "total_train_epochs": 1,
        "train_dataset.batch_size": 4,
        "rollout.consumer_batch_size": 4,
        "rollout.max_head_offpolicyness": 2,
        "rollout.deterministic_sampling": True,
        "gconfig.max_new_tokens": args.max_new_tokens,
        "actor.max_new_tokens": args.max_new_tokens,
        "actor.mb_spec.max_tokens_per_mb": args.context_length,
        "sglang.context_length": args.context_length,
        "sglang.mem_fraction_static": 0.35,
        "sglang.disable_cuda_graph": True,
        "recover.mode": "on",
        "recover.freq_steps": 1,
        "recover.freq_epochs": None,
        "recover.freq_secs": None,
        "saver.freq_steps": None,
        "saver.freq_epochs": None,
        "saver.freq_secs": None,
        "evaluator.freq_steps": None,
        "evaluator.freq_epochs": None,
        "evaluator.freq_secs": None,
    }
    raw_config, _ = parse_cli_args(
        [
            "--config",
            str(Path(__file__).parent / "configs/dapo.yaml"),
            *[f"++{key}={json.dumps(value)}" for key, value in overrides.items()],
        ],
    )
    config = OmegaConf.to_object(to_structured_cfg(raw_config, PPOConfig))
    plan = {
        "case": args.case,
        "updates": 3,
        "episodes": 12,
        "batch_size": 4,
        "context_length": args.context_length,
        "max_new_tokens": args.max_new_tokens,
        "checkpoint_estimate_gib": args.estimate_gib,
        "model": str(model),
        "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "ordered_task_ids": [task.task_id for task in tasks],
        "initial_actor_dp": initial_dp,
        "initial_actor_lease": args.devices[:4],
        "initial_rollout_devices": args.devices[4:],
        "targets": [
            dataclasses.asdict(target)
            if isinstance(target, ResourceLayout)
            else {"world_size": target}
            for target in targets
        ],
    }
    plan = json.loads(json.dumps(plan))
    output.mkdir(parents=True, exist_ok=True)
    if any(path.name not in ("plan.json", "config.json") for path in output.iterdir()):
        raise ValueError(
            "Use a fresh acceptance directory; previous evidence is never overwritten"
        )
    previous = output / "plan.json"
    if previous.exists() and json.loads(previous.read_text()) != plan:
        raise ValueError("Prepared acceptance plan differs; use a new output directory")
    atomic_json(previous, plan)
    public_config = dataclasses.asdict(config)

    def redact(value):
        if isinstance(value, dict):
            return {
                key: "<redacted>" if key.endswith("api_key") else redact(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [redact(item) for item in value]
        return value

    atomic_json(output / "config.json", redact(public_config))
    return config, tasks, targets, output


def boundary(manager, version: int, timeout: float, stopped: threading.Event):
    deadline = time.monotonic() + timeout
    while not stopped.is_set():
        snapshot = manager.control_snapshot()
        if snapshot.boundary_ready and snapshot.applied_version == version:
            return
        if snapshot.applied_version > version or time.monotonic() >= deadline:
            raise RuntimeError(f"Expected completed Trainer update {version}")
        stopped.wait(0.1)
    raise RuntimeError("Acceptance coordinator stopped")


def snapshot(manager) -> dict:
    return {
        "weights_version": manager.applied_version,
        "published_version": manager.published_version,
        "cursor": manager.consumer.cursor.copy(),
        "data_progress": dict(manager.data_progress),
        "checkpoint": str(manager.latest_path),
        "identity": manager.identity,
    }


def audit_transition(manager, source: dict, output: Path) -> dict:
    from areal.elastic.control.checkpoint_audit import compare_dcp, compare_runtime

    destination = output / "restored"
    destination.mkdir(parents=True)
    if not isinstance(manager.last_restore_reports, list):
        raise RuntimeError(
            "Acceptance requires actual per-rank checkpoint load reports"
        )
    import shutil

    preserved = output / "source-metadata"
    preserved.mkdir()
    source_path = Path(source["checkpoint"])
    for path in [
        source_path / "manifest.json",
        source_path / "actor-layout.json",
        *source_path.glob("runtime-*.json"),
    ]:
        shutil.copy2(path, preserved / path.name)
    reports = manager.actor.full_checkpoint(
        "save",
        path=str(destination),
        backend="fsdp",
        identity_sha256=manager.identity,
        version=manager.applied_version,
    )
    atomic_json(
        output / "restore.json",
        {
            "plan": manager.last_restore_plan,
            "load_reports": manager.last_restore_reports,
            "save_reports": reports,
        },
    )
    result = {
        "runtime": compare_runtime(
            Path(source["checkpoint"]), destination, manager.last_restore_reports
        ),
        "dcp": compare_dcp(Path(source["checkpoint"]), destination),
    }
    atomic_json(output / "audit.json", result)
    return {
        "runtime": result["runtime"],
        "dcp": {
            key: value for key, value in result["dcp"].items() if key != "fingerprints"
        },
    }


def verify_owners(manager, old_actor, old_rollout, previous_urls):
    from areal.elastic.control.processes import (
        gpu_processes,
        owned_processes,
        resolve_gpu_uuids,
    )

    if old_actor and owned_processes(old_actor):
        raise RuntimeError("Source Actor processes still exist")
    if any(owned_processes(owner) for owner in old_rollout):
        raise RuntimeError("Source Rollout processes still exist")
    rollout = manager.rollout
    observed = [
        tuple(int(device) for device in report["visible_devices"].split(","))
        for report in manager.last_restore_reports
    ]
    if (
        len(observed) != manager.actor_world_size
        or any(
            len(devices) != 1 or devices[0] not in manager.actor_devices
            for devices in observed
        )
        or len(set(observed)) != len(observed)
    ):
        raise RuntimeError("Actual Actor rank devices differ from its lease")
    if manager.resource_config_application is None:
        if rollout.inference_worker_urls != previous_urls:
            raise RuntimeError("Actor-only transition replaced Rollout")
        return {"source_actor_gone": True, "rollout_urls_unchanged": True}
    leases = {
        device: manager.owner
        for device in manager.actor_devices[: manager.actor_world_size]
    }
    leases.update(
        {row["device"]: row["owner"] for row in rollout._managed_replicas.records}
    )
    present = gpu_processes()
    owners = {
        owner: {process.pid for process in owned_processes(owner)}
        for owner in set(leases.values())
    }
    for device, gpu in zip(
        manager.resource_reservation,
        resolve_gpu_uuids(list(manager.resource_reservation)),
    ):
        pids = set(present.get(gpu, ()))
        if pids and (device not in leases or not pids <= owners[leases[device]]):
            raise RuntimeError("GPU occupancy is outside this run's active leases")
        if device in leases and not pids:
            raise RuntimeError("Active GPU lease has no observed compute process")
    return {
        "source_actor_gone": True,
        "source_rollout_gone": True,
        "gpu_leases_verified": True,
    }


def validate_telemetry(state_root: Path, case: str) -> dict:
    path = state_root / "reconfiguration-timing.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    ends = [row for row in rows if row["event"] == "end" and row["status"] == "applied"]
    if {row["request_id"] for row in ends} != {
        f"accept-{case}-1",
        f"accept-{case}-2",
    } or len(ends) != 2:
        raise RuntimeError("Require timing evidence for both applied transitions")
    required = {
        "checkpoint",
        "actor_release",
        "actor_start",
        "actor_restore",
        "awex_connect",
        "publish",
        "application_and_commit",
        "admission_open",
        "generation_resume",
        "total",
    }
    if case != "actor":
        required |= {
            "drain",
            "rollout_stop",
            "placement",
            "rollout_start",
            "proxy_rebind",
        }
    observed = []
    for end in ends:
        current = [row for row in rows if row["transition_id"] == end["transition_id"]]
        phases = {
            row["phase"]
            for row in current
            if row["event"] == "phase_end"
            and row["status"] == "completed"
            and row["seconds"] >= 0
        }
        if not required <= phases:
            raise RuntimeError("Transition phase timing is incomplete")
        version = next(
            row["source"]["weights_version"]
            for row in current
            if row["event"] == "begin"
        )
        calls = [
            proxy["first_call"]
            for row in current
            if row["event"] == "first_inference_snapshot"
            for proxy in row["proxies"]
            if proxy.get("first_call") is not None
            and proxy["first_call"].get("transition_id") == end["transition_id"]
            and proxy["first_call"].get("status") == "completed"
        ]
        verified = []
        for call in calls:
            attempts = call["backend_attempts"]
            if not attempts:
                continue
            if any(
                attempt["status"] != "completed"
                or attempt["version"] != version
                or attempt["input_tokens"] <= 0
                or attempt["output_tokens"] < 0
                or attempt["seconds"] < 0
                for attempt in attempts
            ):
                raise RuntimeError("First-inference backend evidence is inconsistent")
            if call["admitted_to_response_seconds"] < 0 or call[
                "response_ready_monotonic"
            ] < max(attempt["response_monotonic"] for attempt in attempts):
                raise RuntimeError("First-inference response boundary is inconsistent")
            verified.append(call)
        if not verified:
            raise RuntimeError(
                "Transition has no observed first model call with backend timing/lengths"
            )
        observed.append(
            {"transition_id": end["transition_id"], "observed_proxies": len(verified)}
        )
    return {"status": "complete", "transitions": observed}


async def drive(args, manager, tasks, targets, output, stopped):
    import httpx

    from examples.task_workflows.client import ClientSettings, EpisodeClient

    settings = ClientSettings(
        agent_url=args.agent_url,
        inference_url=manager.rollout.proxy_gateway_addr,
        agent_key=os.environ["AREAL_AGENT_ADMIN_KEY"],
        inference_key=manager.config.rollout.admin_api_key,
        model=manager.config.rollout.model,
        mode="on_demand",
        output=output,
        state_root=manager.root.absolute(),
        concurrency=4,
        request_timeout=args.timeout,
        handoff_timeout=args.timeout,
    )
    transitions, all_rows = [], []
    async with httpx.AsyncClient(timeout=args.timeout) as http:
        client = EpisodeClient(settings, http)
        retained = (
            await client._post(
                settings.inference_url,
                "/rl/start_session",
                settings.inference_key,
                {"task_id": "reconfiguration-retained-session"},
            )
        )["sessions"][0]
        incarnation = retained["areal_episode"]["incarnation_id"]
        for batch in range(3):
            before = snapshot(manager)
            rows = await asyncio.gather(
                *[
                    client.run_task(task, credentials=retained if index == 0 else None)
                    for index, task in enumerate(tasks[batch * 4 : batch * 4 + 4])
                ]
            )
            for row in rows:
                client._append_episode(row)
            all_rows.extend(rows)
            if any(
                row["status"] != "completed"
                or row.get("agent_cleanup_error")
                or not row.get("handoff", {}).get("verified")
                for row in rows
            ):
                raise RuntimeError(
                    "A task failed infrastructure or durable handoff checks"
                )
            await asyncio.to_thread(boundary, manager, batch + 1, args.timeout, stopped)
            with manager.control_operation(blocking=True):
                after = snapshot(manager)
                if (
                    after["weights_version"] != before["weights_version"] + 1
                    or after["cursor"]["applied_batches"] != batch + 1
                    or after["data_progress"]["samples"] != (batch + 1) * 4
                ):
                    raise RuntimeError(
                        "Trainer update/cursor/sample continuity mismatch"
                    )
                if any(v != batch + 1 for v in manager.rollout.serving_versions()):
                    raise RuntimeError(
                        "Every serving proxy must confirm the completed version"
                    )
            episode = await client._post(
                settings.inference_url,
                "/rl/current_episode",
                retained["session_api_key"],
                {},
            )
            if (
                episode["incarnation_id"] != incarnation
                or episode["trajectory_id"] != batch + 1
            ):
                raise RuntimeError(
                    "Retained session identity/trajectory counter changed"
                )
            retained = {
                **retained,
                "areal_episode": {
                    key: episode[key] for key in ("incarnation_id", "trajectory_id")
                },
            }
            if batch == 2:
                break
            with manager.control_operation(blocking=True):
                old_actor = manager.owner
                pool = getattr(manager.rollout, "_managed_replicas", None)
                old_rollout = (
                    [record["owner"] for record in pool.records] if pool else []
                )
                previous_urls = manager.rollout.inference_worker_urls
                source = snapshot(manager)
                target = targets[batch]
                request = dict(
                    request_id=f"accept-{args.case}-{batch + 1}",
                    expected_generation=batch,
                    expected_weights_version=batch + 1,
                )
                if isinstance(target, ResourceLayout):
                    receipt = manager.reconfigure_resources(target=target, **request)
                else:
                    receipt = manager.reconfigure_actor(world_size=target, **request)
                # The request may refresh the completed source checkpoint before
                # capturing its durable boundary; that save must remain timed.
                source["checkpoint"] = str(manager.latest_path)
                if receipt["status"] != "applied" or snapshot(manager) != source:
                    raise RuntimeError(
                        "Transition changed training state or did not apply"
                    )
                audit_dir = output / f"transition-{batch + 1}"
                audit_dir.mkdir()
                checked = audit_transition(manager, source, audit_dir)
                ownership = verify_owners(
                    manager, old_actor, old_rollout, previous_urls
                )
                transitions.append(
                    {
                        "receipt": receipt,
                        "source": source,
                        "audit": checked,
                        "ownership": ownership,
                    }
                )
                atomic_json(output / "transitions.json", transitions)
    if manager.inbox.status()["counts"] != {"published": 12}:
        raise RuntimeError(
            "Expected twelve published episodes and no active/rejected episodes"
        )
    return {
        "status": "passed",
        "completed_updates": 3,
        "completed_episodes": len(all_rows),
        "infrastructure_errors": 0,
        "durable_handoffs": len(all_rows),
        "retained_session_trajectories": 3,
        "transitions": len(transitions),
        "correct": sum(row["score"]["reward"] == 1 for row in all_rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("actor", "rollout", "joint"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-url", default=os.environ.get("AREAL_AGENT_URL"))
    parser.add_argument("--devices", type=int, nargs=8, default=list(range(8)))
    parser.add_argument("--estimate-gib", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (
        min(args.timeout, args.estimate_gib, args.context_length, args.max_new_tokens)
        <= 0
        or args.context_length <= args.max_new_tokens
    ):
        parser.error(
            "Require positive budgets with context length exceeding generation length"
        )
    config, tasks, targets, output = prepare(args)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "plan": str(output / "plan.json")}))
        return
    if not args.agent_url or not os.environ.get("AREAL_AGENT_ADMIN_KEY"):
        parser.error("Running requires --agent-url and AREAL_AGENT_ADMIN_KEY")
    from areal import PPOTrainer
    from areal.utils import name_resolve

    name_resolve.reconfigure(config.cluster.name_resolve)
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        AREAL_SPMD_MODE="0",
        AREAL_ELASTIC_RESOURCE_RECONFIGURATION="0" if args.case == "actor" else "1",
    )
    stopped = threading.Event()
    atomic_json(
        output / "launch.json",
        {"pid": os.getpid(), "started": time.time(), "case": args.case},
    )

    def terminate(signum, frame):
        raise RuntimeError(
            "Acceptance driver requested coordinator shutdown; inspect failure.json"
        )

    signal.signal(signal.SIGTERM, terminate)
    trainer = None
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        trainer = PPOTrainer(config)
        manager = trainer._on_demand

        def feeder():
            try:
                return asyncio.run(
                    drive(args, manager, tasks, targets, output, stopped)
                )
            except BaseException as error:
                atomic_json(
                    output / "failure.json",
                    {"type": type(error).__name__, "error": str(error)},
                )
                os.kill(os.getpid(), signal.SIGTERM)
                raise

        future = executor.submit(feeder)
        trainer.train()
        result = future.result(timeout=args.timeout)
        manager.save_checkpoint(reason="acceptance-final")
        manifest = manager.latest_manifest
        if (
            manager.applied_version != 3
            or manifest["global_step"] != 3
            or manifest["data_progress"]["samples"] != 12
        ):
            raise RuntimeError("Final complete Trainer checkpoint is missing")
        result["final_checkpoint"] = str(manager.latest_path)
        result["telemetry"] = validate_telemetry(manager.root, args.case)
        trainer.close()
        trainer = None
        atomic_json(
            output / "result.json", {**result, "trainer_exit": 0, "client_exit": 0}
        )
    except BaseException as error:
        if not (output / "failure.json").exists():
            atomic_json(
                output / "failure.json",
                {"type": type(error).__name__, "error": str(error)},
            )
        raise
    finally:
        stopped.set()
        if trainer is not None:
            trainer.close()
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
