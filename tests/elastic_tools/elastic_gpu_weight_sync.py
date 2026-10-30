# SPDX-License-Identifier: Apache-2.0
"""Explicit eight-GPU TP4 -> two TP2 weight-publication check, without SWE rollout.

Randomize serving weights, publish the Actor's base weights, and compare selected
TP0 tensors on both replicas bitwise. The existing debug endpoint exports TP0
only; this is not an exhaustive comparison of all inference weights.
"""

import argparse
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "dataset", "swe-runtime", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    repo = str(Path(__file__).resolve().parents[2])
    os.environ.update(
        CUDA_VISIBLE_DEVICES="", AREAL_IS_SINGLE_CONTROLLER="1", AREAL_SPMD_MODE="0"
    )
    os.environ["PYTHONPATH"] = repo + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )

    import httpx
    import torch

    from examples.elastic.run_demo import close_trainer, prepare
    from examples.elastic.telemetry import GPUMonitor

    from areal import PPOTrainer
    from areal.elastic.policy import LifecyclePolicy
    from areal.infra.rpc.serialization import deserialize_value
    from areal.utils.atomic import atomic_json
    from areal.v2.weight_update.gateway.metadata import merge_parameter_metadata

    args.__dict__.update(
        steps=1,
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
    assert config.rollout.backend == "sglang:d2t2"
    monitor = GPUMonitor(output / "gpu-memory.jsonl", args.devices)
    monitor.start()
    trainer = None
    successful = False
    result = {"kind": "weight_sync_probe", "swe_rollouts": 0, "training_steps": 0}

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Weight sync check interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    timer = threading.Timer(600, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.daemon = True
    timer.start()
    try:
        trainer = PPOTrainer(
            config,
            train_dataset=dataset,
            elastic_policy=LifecyclePolicy(mode="resident"),
        )
        manager = trainer._on_demand
        manager.start_actor()
        actor = manager.actor
        train_peers = actor.train_worker_urls
        infer_peers = trainer.rollout.inference_worker_urls
        assert len(train_peers) == 4 and len(infer_peers) == 2
        names = [
            f"model.layers.0.{part}.weight"
            for part in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
                "input_layernorm",
                "post_attention_layernorm",
            )
        ]
        exports = output / "parameters"
        exports.mkdir()

        def post(url, endpoint, payload=None):
            response = httpx.post(url + endpoint, json=payload or {}, timeout=120)
            response.raise_for_status()
            return response.json()

        with ThreadPoolExecutor(max_workers=4) as pool:
            # Megatron exports involve TP all-gathers: every training rank must enter.
            list(
                pool.map(
                    lambda item: post(
                        item[1],
                        "/awex/debug/get_parameters",
                        {
                            "names": names,
                            "save_path": str(exports / f"train-{item[0]}.pt"),
                        },
                    ),
                    enumerate(train_peers),
                )
            )
            list(
                pool.map(
                    lambda peer: post(peer, "/awex/debug/randomize_parameters"),
                    infer_peers,
                )
            )
            list(
                pool.map(
                    lambda item: post(
                        item[1],
                        "/awex/debug/get_parameters",
                        {
                            "names": names,
                            "save_path": str(exports / f"before-{item[0]}.pt"),
                        },
                    ),
                    enumerate(infer_peers),
                )
            )
        trainer.rollout.pause_generation()
        manager._publication_uncertain = True
        started = time.monotonic()
        actor.update_weights(
            manager.weight_meta.with_version(1), resume_generation=False
        )
        result["publication_seconds"] = time.monotonic() - started
        assert trainer.rollout.serving_versions() == [1, 1]
        manager._publication_uncertain = False
        result["serving_versions"] = [1, 1]
        reference = torch.load(
            exports / "train-0.pt", map_location="cpu", weights_only=True
        )
        assert set(reference) == set(names)
        for rank in range(1, 4):
            replica = torch.load(
                exports / f"train-{rank}.pt", map_location="cpu", weights_only=True
            )
            assert all(torch.equal(reference[name], replica[name]) for name in names)
        checks = []
        for i, peer in enumerate(infer_peers):
            post(
                peer,
                "/awex/debug/get_parameters",
                {"names": names, "save_path": str(exports / f"after-{i}.pt")},
            )
            raw = post(peer, "/awex/report_weight_meta")["meta"]
            metadata = deserialize_value(merge_parameter_metadata(raw))
            before = torch.load(
                exports / f"before-{i}.pt", map_location="cpu", weights_only=True
            )
            after = torch.load(
                exports / f"after-{i}.pt", map_location="cpu", weights_only=True
            )
            for parameter in metadata:
                if parameter.name not in names:
                    continue
                shard = next(s for s in parameter.shards if s.global_rank == 0)
                slices = tuple(
                    slice(offset, offset + width)
                    for offset, width in zip(shard.global_offset, shard.shape)
                )
                expected = reference[parameter.name][slices]
                assert not torch.equal(before[parameter.name], expected)
                assert torch.equal(after[parameter.name], expected), (i, parameter.name)
                checks.append(
                    {
                        "engine": i,
                        "tp_rank": 0,
                        "parameter": parameter.name,
                        "shape": list(expected.shape),
                        "bitwise_equal": True,
                        "changed_from_randomized": True,
                    }
                )
        assert len(checks) == 2 * len(names)
        result.update(
            status="passed",
            tensor_checks=checks,
            limitation="Existing debug endpoint exports TP0 only; CPU planner tests verify coverage of all TP ranks.",
        )
        successful = True
    except BaseException as error:
        result.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        try:
            if trainer is not None:
                # This probe never trains; discard the initial Actor without a checkpoint.
                trainer._on_demand._release(training_uncertain=not successful)
                close_trainer(trainer, output)
        except BaseException as error:
            result.update(
                status="failed", cleanup_error=f"{type(error).__name__}: {error}"
            )
            raise
        finally:
            try:
                monitor.close()
            except BaseException as error:
                result.update(
                    status="failed", monitor_error=f"{type(error).__name__}: {error}"
                )
                raise
            finally:
                timer.cancel()
                atomic_json(output / "result.json", result)


if __name__ == "__main__":
    main()
