# SPDX-License-Identifier: Apache-2.0
"""Eight-GPU A/B lifecycle probe using one existing full Megatron checkpoint.

Normal HF+DCP versus elastic deferred HF+DCP, with the same persistent TP2
inference peers. Audits all logical model/master/moment tensors, then performs a
controlled Adam update. This is not a SWE benchmark or coordinator recovery.
No additional model checkpoint is written. Run as a module with --model,
--dataset, --swe-runtime, --checkpoint and a fresh --output directory.
"""

import argparse
import asyncio
import dataclasses
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "dataset", "swe-runtime", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
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

    from areal import PPOTrainer
    from areal.api import FinetuneSpec, ModelRequest
    from areal.elastic.control.processes import gpu_processes
    from areal.elastic.policy import LifecyclePolicy
    from areal.infra.utils.concurrent import run_async_task
    from areal.utils.atomic import atomic_json, recovery_digest
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
        checkpoint_gib=1,
        prepare_only=False,
    )
    config, output, dataset = prepare(args)
    checkpoint = args.checkpoint.resolve()
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    compatibility = manifest["compatibility"]
    assert compatibility["backend"] == "megatron" and manifest["world_size"] == 4
    assert config.actor.backend == compatibility["actor"]["backend"]
    monitor = GPUMonitor(output / "gpu-memory.jsonl", args.devices)
    monitor.start()
    trainer = None
    result = {
        "kind": "lifecycle_ab_probe",
        "swe_rollouts": 0,
        "checkpoint": str(checkpoint),
        "rounds": [],
    }

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Lifecycle probe interrupted by {signum}")

    signal.signal(signal.SIGTERM, stop)
    timer = threading.Timer(1800, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.daemon = True
    timer.start()
    try:
        trainer = PPOTrainer(
            config,
            train_dataset=dataset,
            elastic_policy=LifecyclePolicy(mode="resident"),
        )
        manager = trainer._on_demand
        # Test-only import of the exact snapshot into fresh services; no claim
        # that a fresh inbox/dataloader can resume an old coordinator journal.
        manager.ft_spec = FinetuneSpec(**compatibility["ft_spec"])
        manager.identity = recovery_digest(compatibility)
        manager.latest_path, manager.latest_manifest = checkpoint, manifest
        manager.applied_version = manifest["global_step"]
        original_factory = manager.actor_factory
        peers = list(trainer.rollout.inference_worker_urls)
        assert len(peers) == 2
        bridge = SGLangBridgeBackend()
        request = bridge.build_generation_request(
            ModelRequest(
                input_ids=trainer.tokenizer.encode("Reply with the number 42."),
                gconfig=dataclasses.replace(
                    config.gconfig, max_new_tokens=8, greedy=True
                ),
                tokenizer=trainer.tokenizer,
            ),
            with_lora=False,
        )

        for mode in ("normal", "elastic", "normal"):

            def factory(actor_config):
                actor_config.scheduling_spec[0].env_vars[
                    "AREAL_ELASTIC_PROBE_AUDIT_DIR"
                ] = str(output / "state-audits")
                actor = original_factory(actor_config)
                actor.train_engine = "tests.elastic_probe_actor.ProbeActor"
                if mode == "normal":
                    initialize, destroy = actor.initialize, actor.destroy

                    def ordinary_initialize(**kwargs):
                        kwargs.pop("restore_from_full_checkpoint", None)
                        return initialize(**kwargs)

                    def ordinary_destroy(**kwargs):
                        kwargs.pop("parallel_child_cleanup", None)
                        return destroy(**kwargs)

                    actor.initialize, actor.destroy = (
                        ordinary_initialize,
                        ordinary_destroy,
                    )
                return actor

            manager.actor_factory = factory
            start = time.monotonic()
            actor = manager.ensure_actor()
            row = {"mode": mode, "startup_seconds": time.monotonic() - start}
            result["rounds"].append(row)

            async def all_ranks(endpoint):
                return await asyncio.gather(
                    *[
                        actor._call_worker_engine_endpoint(
                            peer, endpoint, timeout=config.actor.setup_timeout
                        )
                        for peer in actor.train_worker_urls
                    ]
                )

            row["restored_state"] = run_async_task(all_ranks, "/get_device_stats")
            assert all(
                report["hf_load_calls"] == (0 if mode == "elastic" else 1)
                for report in row["restored_state"]
            )
            actor.update_weights(
                dataclasses.replace(
                    manager.weight_meta, version=manager.applied_version
                )
            )
            row["optimizer_update"] = run_async_task(all_ranks, "/optimizer_step")
            assert all(
                report["update_successful"] == 1 for report in row["optimizer_update"]
            )
            row["updated_state"] = run_async_task(all_ranks, "/get_device_stats")
            start = time.monotonic()
            manager._release()
            row["release_seconds"] = time.monotonic() - start
            assert all(not gpu_processes().get(gpu) for gpu in manager.gpu_uuids)
            assert list(trainer.rollout.inference_worker_urls) == peers
            row["off_actor_generation"] = []
            with httpx.Client(timeout=60) as client:
                for peer in peers:
                    response = client.post(
                        peer + request.endpoint, json=request.payload
                    )
                    response.raise_for_status()
                    generated = bridge.parse_generation_response(response.json())
                    assert generated.output_tokens
                    row["off_actor_generation"].append(len(generated.output_tokens))
            assert trainer.rollout.serving_versions() == [manager.applied_version] * 2
            atomic_json(output / "result.json", result)

        for key in ("restored_state", "updated_state"):
            digests = [
                {r["rank"]: r["state_sha256"] for r in row[key]}
                for row in result["rounds"]
            ]
            assert all(digest == digests[0] for digest in digests), (
                f"{key} differs across restore modes"
            )
        result["status"] = "passed"
        result["state_comparison"] = (
            "All logical model/master/Adam moment tensors plus optimizer metadata; normal versus optimized restore of the same checkpoint, before and after one controlled gradient update. Not a PPO/SWE update."
        )
    except BaseException as error:
        result.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        try:
            if trainer is not None:
                # Probe updates are disposable; preserve the source checkpoint.
                trainer._on_demand._release(
                    training_uncertain=trainer._on_demand.ready_since is None
                )
                close_trainer(trainer, output)
        except BaseException as error:
            result.update(status="failed", cleanup_error=str(error))
            raise
        finally:
            timer.cancel()
            monitor.close()
            result["remaining_gpu_processes"] = gpu_processes()
            if result["remaining_gpu_processes"]:
                result.update(status="failed", cleanup_error="GPU processes remain")
            atomic_json(output / "result.json", result)
            if result["remaining_gpu_processes"]:
                raise RuntimeError("Lifecycle probe left GPU processes alive")


if __name__ == "__main__":
    main()
