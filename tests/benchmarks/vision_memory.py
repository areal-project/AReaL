# SPDX-License-Identifier: Apache-2.0
"""Compare one identical CPU image pipeline using aggregate process-tree PSS.

Requires Linux and a baseline directory containing serialization.py and rtensor.py
from main. RTensor uses an in-memory storage adapter; token RPCs are excluded.
This synthetic pipeline uses real PIL, RPC encoding and Gloo broadcasts, but no
model, optimizer or HTTP server. Results are not whole-training-job savings.
Run each variant in a fresh interpreter; see docs/en/best_practices/cpu_vision_memory.md.
"""

import argparse
import base64
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path

PHASES = (
    "idle",
    "preprocess",
    "compact",
    "encode",
    "transfer",
    "broadcast",
    "resident",
)


def load_baseline(root, name="serialization"):
    spec = importlib.util.spec_from_file_location(
        f"baseline_{name}", str(Path(root) / f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_memory(pids, phase, stop, destination):
    peaks = {name: 0.0 for name in PHASES}
    samples = []
    while not stop.is_set():
        current_phase = PHASES[phase.value]
        total = 0
        complete = True
        for pid in pids:
            if not pid:
                complete = False
                break
            try:
                text = Path(f"/proc/{pid}/smaps_rollup").read_text()
                total += next(
                    int(line.split()[1])
                    for line in text.splitlines()
                    if line.startswith("Pss:")
                )
            except (OSError, StopIteration):
                complete = False
                break
        if complete:
            mib = total / 1024
            peaks[current_phase] = max(peaks[current_phase], mib)
            samples.append((time.monotonic(), current_phase, mib))
        stop.wait(0.02)
    Path(destination).write_text(
        json.dumps({"peak_pss_mib": peaks, "samples": samples})
    )


def actor(rank, args, init_method, connection, pids, phase, release):
    import torch
    import torch.distributed as dist

    from areal.infra.rpc.node_shared_vision import NodeSharedVisionTransport
    from areal.infra.rpc.serialization import SerializedTensor
    from areal.utils.data import broadcast_tensor_container

    torch.set_num_threads(1)
    pids[rank + 1] = os.getpid()
    dist.init_process_group(
        "gloo", rank=rank, world_size=args.ranks, init_method=init_method
    )
    try:
        dist.barrier()
        payload = None
        if rank == 0:
            connection.send("ready")
            wire = connection.recv_bytes()
            envelope = json.loads(wire)
            del wire
            source_tensors = {
                key: SerializedTensor.model_validate(value).to_tensor()
                for key, value in envelope["stored"].items()
            }
            payload = {
                "multi_modal_input": [
                    {"pixel_values": source_tensors[key]} for key in envelope["refs"]
                ]
            }
            del envelope
            phase.value = PHASES.index("broadcast")
        broadcast = broadcast_tensor_container
        if args.variant != "main":
            broadcast = NodeSharedVisionTransport(broadcast)
        result = broadcast(payload, src_rank=0, preserve_tensor_aliases=True)
        # Read every trajectory and compare without a tensor-sized copy.
        hashes = [
            hashlib.sha256(
                memoryview(item["pixel_values"].numpy()).cast("B")
            ).hexdigest()
            for item in result["multi_modal_input"]
        ]
        assert len(set(hashes)) == 1
        digest = hashes[0]
        digests = [None] * args.ranks
        dist.all_gather_object(digests, digest)
        assert len(set(digests)) == 1
        if rank == 0:
            phase.value = PHASES.index("resident")
            connection.send(digest)
        release.wait(120)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("main", "shared", "shared-rpc", "shared-lazy", "all"),
        required=True,
    )
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--source-size", type=int, default=2048)
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--trajectories", type=int, default=12)
    args = parser.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    from areal.infra.processor_cache import ProcessorCallCache
    from areal.infra.rpc import rtensor
    from areal.infra.rpc.serialization import SerializedTensor
    from areal.utils import qwen_vl_image

    torch.set_num_threads(1)
    ctx = mp.get_context("spawn")
    pids = ctx.Array("i", [os.getpid()] + [0] * args.ranks)
    phase = ctx.Value("i", 0)
    stop, release = ctx.Event(), ctx.Event()
    sender, receiver = ctx.Pipe()
    directory = tempfile.TemporaryDirectory()
    workers = [
        ctx.Process(
            target=actor,
            args=(
                rank,
                args,
                f"file://{directory.name}/gloo",
                receiver,
                pids,
                phase,
                release,
            ),
        )
        for rank in range(args.ranks)
    ]
    for worker in workers:
        worker.start()
    if not sender.poll(180):
        raise RuntimeError("Actor initialization timed out")
    assert sender.recv() == "ready"
    monitor_output = args.output + ".samples.json"
    monitor = ctx.Process(
        target=sample_memory, args=(pids, phase, stop, monitor_output)
    )
    monitor.start()
    time.sleep(1)
    try:
        phase.value = PHASES.index("preprocess")
        encoded = []
        for index in range(args.images):
            buffer = BytesIO()
            with Image.new(
                "RGB", (args.source_size, args.source_size), (index + 1, 44, 91)
            ) as image:
                image.save(buffer, format="PNG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode())
        decoded_count = 0

        def decode(value):
            nonlocal decoded_count
            decoded_count += 1
            with Image.open(BytesIO(base64.b64decode(value, validate=True))) as image:
                return image.convert("RGB")

        # Fix policy identically in every variant; avoid model/network downloads.
        qwen_vl_image._image_policy = lambda _: qwen_vl_image._ImagePolicy(
            32, args.resize_size**2, args.resize_size**2, None
        )
        cache, processor = ProcessorCallCache(), object()
        lazy = args.variant in ("shared-lazy", "all")
        # Same history appears twice, representing a changed text prompt.
        for _ in range(2):
            inputs = (
                [lambda value=value: decode(value) for value in encoded]
                if lazy
                else [decode(value) for value in encoded]
            )
            prepared, wire_images = qwen_vl_image.prepare_sglang_qwen_images(
                processor, inputs, cache=cache, image_keys=encoded
            )
            assert wire_images == encoded
            if not lazy:
                for image in inputs:
                    image.close()
            del inputs
        # Synthetic dense float32 vision storage, not a model-specific processor.
        tensor = torch.stack(
            [torch.from_numpy(np.asarray(image).copy()) for image in prepared]
        ).float()
        expected = hashlib.sha256(memoryview(tensor.numpy()).cast("B")).hexdigest()
        phase.value = PHASES.index("compact")

        class MemoryStore:
            def __init__(self):
                self.tensors = {}

            def store(self, value):
                key = str(len(self.tensors))
                self.tensors[key] = value
                return key

        store = MemoryStore()
        module = (
            load_baseline(args.baseline_root, "rtensor")
            if args.variant == "main"
            else rtensor
        )
        previous_backend = module.get_backend
        module.get_backend = lambda: store
        trajectories = [
            {
                "input_ids": torch.tensor([[1, 2, 0]]),
                "attention_mask": torch.tensor([[1, 1, 0]]),
                "multi_modal_input": [{"pixel_values": tensor}],
            }
            for _ in range(args.trajectories)
        ]
        try:
            remote = module.RTensor.remotize(
                trajectories, node_addr="benchmark", preserve_tensor_aliases=True
            )
        finally:
            module.get_backend = previous_backend
        refs = [
            row["multi_modal_input"][0]["pixel_values"].shard.shard_id for row in remote
        ]
        unique_refs = list(dict.fromkeys(refs))
        assert len(unique_refs) == (args.trajectories if args.variant == "main" else 1)
        phase.value = PHASES.index("encode")
        serializer = (
            SerializedTensor
            if args.variant in ("shared-rpc", "all")
            else load_baseline(args.baseline_root).SerializedTensor
        )
        serialized = {
            key: serializer.from_tensor(store.tensors[key]).model_dump()
            for key in unique_refs
        }
        phase.value = PHASES.index("transfer")
        wire = json.dumps({"stored": serialized, "refs": refs}).encode()
        sender.send_bytes(wire)
        del wire, serialized
        if not sender.poll(180):
            raise RuntimeError("Actor broadcast timed out")
        assert sender.recv() == expected
        time.sleep(1)
    finally:
        stop.set()
        monitor.join(15)
        release.set()
        for worker in workers:
            worker.join(30)
            if worker.is_alive():
                worker.terminate()
            if worker.exitcode != 0:
                raise RuntimeError(f"Actor failed: {worker.exitcode}")
        directory.cleanup()
    data = json.loads(Path(monitor_output).read_text())
    peaks = data["peak_pss_mib"]
    result = dict(
        variant=args.variant,
        ranks=args.ranks,
        images=args.images,
        source_size=args.source_size,
        resize_size=args.resize_size,
        tensor_mib=tensor.numel() * tensor.element_size() / 2**20,
        trajectories=args.trajectories,
        unique_image_shards=len(unique_refs),
        original_decodes=decoded_count,
        sha256=expected,
        peak_pss_mib=max(peaks.values()),
        phases=peaks,
        samples=len(data["samples"]),
    )
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
