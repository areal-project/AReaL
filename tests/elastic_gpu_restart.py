# SPDX-License-Identifier: Apache-2.0

"""Explicit eight-GPU restart protocol check, not an automatic-policy benchmark.

Run with ``python -m tests.elastic_gpu_restart --model ... --dataset ...
--swe-runtime ... --output ...``.
The test deliberately collects two native batches, releases/restores the Actor,
then collects a third native batch and performs a real PPO update. Inference and
its dataset/workflow dispatcher remain alive throughout. No trajectory is replayed.
"""

import argparse
import dataclasses
import json
import os
import signal
import sys
import threading
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "dataset", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--swe-runtime", type=Path, default=os.environ.get("AREAL_SWE_CONFIG")
    )
    parser.add_argument("--checkpoint-gib", type=int, default=64)
    args = parser.parse_args()
    os.environ.update(
        CUDA_VISIBLE_DEVICES="", AREAL_IS_SINGLE_CONTROLLER="1", AREAL_SPMD_MODE="0"
    )
    repo = str(Path(__file__).resolve().parents[1])
    os.environ["PYTHONPATH"] = repo + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )

    import httpx

    from examples.elastic.run_demo import close_trainer, prepare

    from areal import PPOTrainer
    from areal.api import ModelRequest
    from areal.api.alloc_mode import ModelAllocation
    from areal.elastic.control.processes import gpu_processes
    from areal.elastic.policy import LifecyclePolicy
    from areal.infra.rpc.rtensor import RTensor
    from areal.infra.rpc.serialization import serialize_value
    from areal.infra.utils.concurrent import run_async_task
    from areal.utils.atomic import atomic_json
    from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend

    args.__dict__.update(
        steps=3,
        policy="resident",
        devices=list(range(8)),
        concurrency=2,
        export_style="concat",
        idle_seconds=40,
        max_turns=24,
        max_tokens=4096,
        context_tokens=16384,
        prepare_only=False,
    )
    config, output, dataset = prepare(args)
    inference_replicas = ModelAllocation.from_str(
        config.rollout.backend
    ).parallel.dp_size
    atomic_json(
        output / "verification-kind.json",
        {"kind": "forced_restart_protocol_check", "automatic_policy": False},
    )

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Restart check interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    timer = threading.Timer(3600, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.daemon = True
    timer.start()
    trainer = None
    try:
        trainer = PPOTrainer(
            config,
            train_dataset=dataset,
            elastic_policy=LifecyclePolicy(mode="resident"),
        )
        manager = trainer._on_demand
        rollout = trainer.rollout
        peers = list(rollout._inf_addrs)
        assert len(peers) == inference_replicas
        workflow = "examples.elastic.swe.workflow.SWEAgent"
        kwargs = {
            "log_path": str(output / "episodes.jsonl"),
            "max_turns": args.max_turns,
            "max_tokens": args.max_tokens,
            "context_tokens": args.context_tokens,
            "runtime_config": str(args.swe_runtime.resolve()),
        }
        # Keep the original dataset-derived training specification. Bound only the
        # first collection/training segment so the test owns the restart boundary.
        config.total_train_steps = 2
        trainer.train(workflow=workflow, workflow_kwargs=kwargs)
        manager.inbox._thread.join(timeout=10)
        assert not manager.inbox._thread.is_alive()
        assert manager.applied_version == 2
        manager.stop_actor()
        assert manager.actor is None
        occupied = gpu_processes()
        assert all(not occupied.get(gpu) for gpu in manager.gpu_uuids)
        assert rollout.serving_versions() == [2] * inference_replicas
        # A next unconsumed training batch may already be prefetched. Generate
        # fresh tokens on every replica while the Actor is confirmed absent, to
        # verify serving separately from training-data collection.
        bridge = SGLangBridgeBackend()
        request = bridge.build_generation_request(
            ModelRequest(
                input_ids=trainer.tokenizer.encode("Reply with the number 42."),
                gconfig=dataclasses.replace(
                    config.gconfig, max_new_tokens=16, greedy=True
                ),
                tokenizer=trainer.tokenizer,
            ),
            with_lora=False,
        )
        probes = []
        with httpx.Client(timeout=30) as client:
            for peer in peers:
                response = client.post(peer + request.endpoint, json=request.payload)
                response.raise_for_status()
                generated = bridge.parse_generation_response(response.json())
                assert generated.output_tokens
                probes.append(
                    {"peer": peer, "generated_tokens": len(generated.output_tokens)}
                )
        occupied = gpu_processes()
        assert manager.actor is None and all(
            not occupied.get(gpu) for gpu in manager.gpu_uuids
        )
        atomic_json(
            output / "released.json",
            {
                "actor_gpus_empty": True,
                "serving_versions": [2] * inference_replicas,
                "inference_peers": peers,
                "fresh_generation_while_actor_off": probes,
            },
        )

        manager.start_actor()
        assert manager.generation == 2 and manager.actor.get_version() == 2
        assert trainer.rollout is rollout and list(rollout._inf_addrs) == peers
        assert rollout.serving_versions() == [2] * inference_replicas
        config.total_train_steps = 3
        # Collect the next batch through the same native dispatcher/dataloader.
        rows = rollout.prepare_batch(
            dataloader=trainer.train_dataloader,
            workflow=workflow,
            workflow_kwargs=kwargs,
            group_size=config.gconfig.n_samples,
        )
        shards = RTensor.collect_shards(rows)
        rows = RTensor.localize(rows)
        assert len(rows) == config.train_dataset.batch_size
        for row in rows:
            manager.inbox._inspect_payload(row)
            manager.inbox.inspect_group(row)
        manager.inbox.queue.enqueue(
            "inline-2",
            {
                "episode_ids": [f"inline-2-{i}" for i in range(len(rows))],
                "group_size": config.gconfig.n_samples,
                "trajectories": [serialize_value(row) for row in rows],
            },
            policy_version=2,
        )
        for node, shard_ids in shards.items():
            run_async_task(RTensor.clear_node, node, shard_ids)
        trainer._train_step(
            global_step=2,
            steps_per_epoch=len(trainer.train_dataloader),
            workflow=workflow,
            workflow_kwargs=kwargs,
            dynamic_filter_fn=None,
            eval_workflow=None,
            eval_workflow_kwargs=None,
        )
        assert manager.applied_version == 3
        assert rollout.serving_versions() == [3] * inference_replicas
        events = [
            json.loads(line)
            for line in (manager.root / "events.jsonl").read_text().splitlines()
        ]
        restored = [event for event in events if event["state"] == "RESTORED"]
        assert len(restored) == 1 and len(restored[0]["reports"]) == 4
        result = {
            "status": "passed",
            "kind": "forced_restart_protocol_check",
            "automatic_policy": False,
            "updates": 3,
            "data_progress": manager.data_progress,
            "restored_version": 2,
            "restored_rank_reports": restored[0]["reports"],
            "serving_versions": [3] * inference_replicas,
            "inference_peers_preserved": True,
            "fresh_generation_while_actor_off": probes,
            "actor_generation": manager.generation,
            "continuation_batch": "next unconsumed native batch; may include prefetch from before release",
            "optimizer_tensor_equality": "native DCP load; not independently tensor-compared",
        }
    except BaseException as error:
        atomic_json(
            output / "result.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        try:
            if trainer is not None:
                close_trainer(trainer, output)
        except BaseException as error:
            atomic_json(output / "cleanup-error.json", {"error": str(error)})
            raise
        finally:
            timer.cancel()
    atomic_json(output / "result.json", result)


if __name__ == "__main__":
    main()
