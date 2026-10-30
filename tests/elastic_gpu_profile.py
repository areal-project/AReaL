# SPDX-License-Identifier: Apache-2.0
"""Profile complete elastic save/restore without rollout collection or training.

Creates one full-size UNTRAINED snapshot. Version 1 is only a transport label;
optimizer/scheduler steps remain zero. This is not a resumable SWE training run.
"""

import argparse
import asyncio
import dataclasses
import os
import signal
import sys
import threading
from contextlib import ExitStack
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "dataset", "swe-runtime", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--restores", type=int, default=2)
    args = parser.parse_args()
    if args.restores < 1:
        raise ValueError("restores must be positive")
    os.environ.update(
        CUDA_VISIBLE_DEVICES="", AREAL_IS_SINGLE_CONTROLLER="1", AREAL_SPMD_MODE="0"
    )
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    os.environ["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[1])
        + os.pathsep
        + os.environ.get("PYTHONPATH", "")
    )

    import httpx

    from examples.elastic.run_demo import close_trainer, prepare
    from examples.elastic.telemetry import GPUMonitor
    from tests.elastic_profile import (
        Recorder,
        hook,
        instrument_controller,
        instrument_store,
        phase_breakdown,
        summarize_spans,
    )

    from areal import PPOTrainer
    from areal.api import ModelRequest
    from areal.elastic.control.processes import gpu_processes
    from areal.elastic.policy import LifecyclePolicy
    from areal.elastic.state.checkpoint_store import CheckpointStore
    from areal.infra.utils.concurrent import run_async_task
    from areal.utils.atomic import atomic_json
    from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend

    args.__dict__.update(
        steps=4,
        policy="resident",
        devices=list(range(8)),
        concurrency=2,
        export_style="concat",
        idle_seconds=40,
        max_turns=24,
        max_tokens=4096,
        context_tokens=16384,
        checkpoint_gib=48,
        prepare_only=False,
    )
    config, output, dataset = prepare(args)
    monitor = GPUMonitor(output / "gpu-memory.jsonl", args.devices)
    monitor.start()
    recorder = Recorder(output / "spans", "coordinator")
    hooks = ExitStack()
    instrument_store(hooks, recorder)
    from areal.v2.weight_update.controller.controller import WeightUpdateController

    for name in ("initialize", "connect", "disconnect", "destroy"):
        hook(hooks, WeightUpdateController, name, recorder, "awex." + name)
    result = dict(
        kind="untrained_full_size_snapshot_profile",
        training_updates=0,
        optimizer_updates=0,
        swe_rollouts=0,
        protocol_version_label=1,
        resumable_swe_run=False,
        restores=[],
        clock="single-host monotonic_ns",
    )
    trainer = None

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Profile interrupted by {signum}")

    signal.signal(signal.SIGTERM, stop)
    timer = threading.Timer(1800, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.daemon = True
    timer.start()
    try:
        with recorder.span("bench.inference_and_coordinator_init"):
            trainer = PPOTrainer(
                config,
                train_dataset=dataset,
                elastic_policy=LifecyclePolicy(mode="resident"),
            )
        manager, rollout = trainer._on_demand, trainer.rollout
        original_factory = manager.actor_factory
        generation_hooks = ExitStack()

        def factory(actor_config):
            generation_hooks.close()
            recorder.generation = manager.generation
            for spec in actor_config.scheduling_spec:
                spec.env_vars.update(
                    AREAL_ELASTIC_PROFILE_DIR=str(output / "spans"),
                    AREAL_ELASTIC_PROFILE_GENERATION=str(manager.generation),
                    AREAL_ELASTIC_PROBE_AUDIT_DIR=str(output / "state-audits"),
                )
            actor = original_factory(actor_config)
            actor.train_engine = "tests.elastic_profile_actor.ProfileActor"
            instrument_controller(generation_hooks, actor, recorder)
            return actor

        manager.actor_factory = factory
        with recorder.span("bench.cold_start"):
            manager.ensure_actor()

        async def audit(initialize):
            return await asyncio.gather(
                *[
                    manager.actor._call_worker_engine_endpoint(
                        peer,
                        "/get_device_stats",
                        kwargs={"initialize_optimizer": initialize},
                        timeout=config.actor.setup_timeout,
                    )
                    for peer in manager.actor.train_worker_urls
                ]
            )

        result["initial_state"] = run_async_task(audit, True)
        assert all(
            r["scheduler_steps"] == 0
            and not any(r["optimizer_steps"])
            and r["model_unchanged_by_slot_materialization"]
            for r in result["initial_state"]
        )
        manager.actor.set_version(1)
        store = CheckpointStore(
            output / "untrained-snapshot", keep=1, free_space_floor=2 << 30
        )
        hook(hooks, store, "commit", recorder, "store.commit")

        def write(path):
            with recorder.span("save.worker_full_checkpoint"):
                manager.actor.full_checkpoint(
                    "save",
                    path=str(path),
                    backend="megatron",
                    identity_sha256=manager.identity,
                    version=1,
                )
            with recorder.span("save.coordinator_sidecars"):
                atomic_json(path / "coordinator.json", manager._capture_cpu())
                atomic_json(
                    path / "checkpoint-schedule.json",
                    manager.checkpoint_frequency.state_dict(),
                )
                atomic_json(
                    path / "profiling-kind.json",
                    {
                        "training_updates": 0,
                        "optimizer_updates": 0,
                        "version_1_is_transport_label": True,
                        "resumable_swe_run": False,
                    },
                )

        with recorder.span("bench.save_complete"):
            snapshot = store.commit(
                step=1,
                world_size=4,
                compatibility=manager.compatibility,
                write=write,
                estimated_bytes=48 << 30,
            )
            manager.latest_path, manager.latest_manifest = store.latest(
                manager.compatibility
            )
        assert snapshot == manager.latest_path
        files = [p for p in snapshot.rglob("*") if p.is_file()]
        result["snapshot"] = dict(
            path=str(snapshot),
            file_count=len(files),
            logical_bytes=sum(p.stat().st_size for p in files),
            allocated_bytes=sum(p.stat().st_blocks * 512 for p in files),
        )
        manager.applied_version = 1
        bridge = SGLangBridgeBackend()
        request = bridge.build_generation_request(
            ModelRequest(
                input_ids=trainer.tokenizer.encode("Reply with 42."),
                gconfig=dataclasses.replace(
                    config.gconfig, greedy=True, max_new_tokens=8
                ),
                tokenizer=trainer.tokenizer,
            ),
            with_lora=False,
        )
        peers = list(rollout.inference_worker_urls)

        for index in range(args.restores):
            with recorder.span("bench.release"):
                manager._release()
            assert all(not gpu_processes().get(gpu) for gpu in manager.gpu_uuids)
            with httpx.Client(timeout=60) as client:
                generated = []
                for peer in peers:
                    response = client.post(
                        peer + request.endpoint, json=request.payload
                    )
                    response.raise_for_status()
                    generated.append(
                        len(
                            bridge.parse_generation_response(
                                response.json()
                            ).output_tokens
                        )
                    )
                assert len(generated) == 2 and all(generated)
            recorder.generation = manager.generation + 1
            with recorder.span("bench.restore_complete"):
                manager.ensure_actor()
            restored = run_async_task(audit, False)
            assert {r["rank"]: r["state_sha256"] for r in restored} == {
                r["rank"]: r["state_sha256"] for r in result["initial_state"]
            }
            assert all(r["hf_load_calls"] == 0 for r in restored)
            assert list(rollout.inference_worker_urls) == peers
            result["restores"].append(
                dict(rank_reports=restored, generation_while_actor_off=generated)
            )
            atomic_json(output / "result.json", result)
        result["status"] = "passed"
    except BaseException as error:
        result.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        try:
            if trainer is not None:
                # No training state exists to save at close. Keep the test snapshot.
                trainer._on_demand._release(
                    training_uncertain=trainer._on_demand.ready_since is None
                )
                generation_hooks.close()
                close_trainer(trainer, output)
        except BaseException as error:
            result.update(status="failed", cleanup_error=str(error))
            raise
        finally:
            hooks.close()
            timer.cancel()
            monitor.close()
            result["remaining_gpu_processes"] = gpu_processes()
            if result["remaining_gpu_processes"]:
                result.update(status="failed", cleanup_error="GPU processes remain")
            result["phase_summary"] = summarize_spans(output / "spans")
            result["measured_phase_breakdown"] = phase_breakdown(output / "spans")
            atomic_json(output / "result.json", result)
            if result["remaining_gpu_processes"]:
                raise RuntimeError("Profile left GPU processes alive")


if __name__ == "__main__":
    main()
