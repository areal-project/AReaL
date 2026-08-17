# SPDX-License-Identifier: Apache-2.0

"""No-forward 8-GPU checkpoint acceptance for a real Megatron model."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import resource
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.engine import MegatronEngine
from areal.engine.megatron_utils.gpu_staged_optimizer import GPUStagedAdamWConfig
from areal.engine.megatron_utils.gpu_staged_optimizer_checkpoint import (
    iter_managed_optimizers,
)


def _memory() -> dict[str, int]:
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "peak": torch.cuda.max_memory_allocated(),
        "driver_used": total - free,
    }


def _slab_samples(inner: Any) -> dict[str, list[float]]:
    slabs = inner.cpu_slabs
    assert slabs is not None
    offsets = (0, slabs.master.numel() // 2, slabs.master.numel() - 1)
    return {
        name: [float(tensor[offset]) for offset in offsets]
        for name, tensor in (
            ("master", slabs.master),
            ("exp_avg", slabs.exp_avg),
            ("exp_avg_sq", slabs.exp_avg_sq),
        )
    }


def _rss_bytes() -> int:
    statm = Path("/proc/self/statm")
    resident_pages = int(statm.read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _fd_targets() -> dict[int, str]:
    targets: dict[int, str] = {}
    for item in Path("/proc/self/fd").iterdir():
        try:
            targets[int(item.name)] = os.readlink(item)
        except FileNotFoundError:
            continue
    return targets


def _sample_snapshot_bytes(root: Path, stop: threading.Event, peak: list[int]) -> None:
    while not stop.wait(0.25):
        peak[0] = max(peak[0], _tree_bytes(root))
    peak[0] = max(peak[0], _tree_bytes(root))


def _residency_evidence(inner: Any) -> dict[str, Any]:
    slabs = inner.cpu_slabs
    assert slabs is not None
    slab_by_state = {
        "master_param": slabs.master,
        "exp_avg": slabs.exp_avg,
        "exp_avg_sq": slabs.exp_avg_sq,
    }
    assert all(
        slab.device.type == "cpu" and slab.dtype is torch.float32 and slab.is_pinned()
        for slab in slab_by_state.values()
    )
    for layout in inner._layouts:
        state = inner.state[layout.param]
        for state_name, slab in slab_by_state.items():
            view = state[state_name]
            assert view.device.type == "cpu"
            assert view.dtype is torch.float32
            assert view.is_pinned()
            assert (
                view.untyped_storage().data_ptr() == slab.untyped_storage().data_ptr()
            )
    return {
        "owned_numel": slabs.master.numel(),
        "residency": inner.residency,
        "cuda_state_numel": inner.cuda_state_numel,
        "cpu_fp32_pinned": True,
        "state_views_alias_slabs": True,
        "samples": _slab_samples(inner),
        "groups": [
            {"step": group["step"], "lr": group["lr"]} for group in inner.param_groups
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("save", "load"), required=True)
    parser.add_argument("--bucket-size-mb", type=float, default=1.0)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    config = TrainEngineConfig(
        backend=args.backend,
        experiment_name="gpu-staged-checkpoint-acceptance",
        trial_name="gpu-staged-checkpoint-acceptance",
        path=args.model_path,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=16),
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(
            bridge_type="mbridge", async_save=args.mode == "save"
        ),
    )
    engine = MegatronEngine(config)
    engine.configure_gpu_staged_adamw(
        GPUStagedAdamWConfig(
            buffer_count=2,
            bucket_size_mb=args.bucket_size_mb,
            checkpoint_snapshot_root=args.snapshot_root,
        )
    )
    allocation = ModelAllocation.from_str(args.backend)
    engine.create_process_group(parallel_strategy=allocation.parallel)
    engine.initialize(
        addr=None,
        ft_spec=FinetuneSpec(
            total_train_epochs=1,
            dataset_size=4,
            train_batch_size=4,
        ),
    )
    assert engine.checkpointer is not None
    managed = tuple(iter_managed_optimizers(engine.optimizer))
    assert managed
    assert dist.get_world_size() == 8
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_root = Path(args.snapshot_root)
    assert snapshot_root.is_dir() and not snapshot_root.is_symlink()
    assert snapshot_root.stat().st_mode & 0o777 == 0o700
    identity_repr = repr(engine.checkpointer._managed_optimizer_identities())

    if args.mode == "save":
        for inner_index, inner in enumerate(managed):
            inner.drain()
            slabs = inner.cpu_slabs
            assert slabs is not None
            offsets = (0, slabs.master.numel() // 2, slabs.master.numel() - 1)
            for sample_index, offset in enumerate(offsets):
                slabs.exp_avg[offset] = inner_index + sample_index + 0.25
                slabs.exp_avg_sq[offset] = inner_index + sample_index + 0.5
            for group in inner.param_groups:
                group["step"] = 7
        expected = [_residency_evidence(inner) for inner in managed]
        (output_dir / f"expected_rank_{rank}.json").write_text(
            json.dumps({"identity": identity_repr, "optimizers": expected}, indent=2)
            + "\n"
        )
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        memory_before = _memory()
        rss_before = _rss_bytes()
        fd_before = _fd_targets()
        snapshot_peak = [0]
        snapshot_sampler_stop = threading.Event()
        snapshot_sampler = threading.Thread(
            target=_sample_snapshot_bytes,
            args=(snapshot_root, snapshot_sampler_stop, snapshot_peak),
            daemon=True,
        )
        snapshot_sampler.start()
        started = time.perf_counter()
        try:
            engine.checkpointer.save_checkpoint(
                args.checkpoint_dir,
                with_model=False,
                with_optimizer=True,
                with_rng=False,
            )
            scheduled = time.perf_counter()
            assert engine.checkpointer.managed_async_save_state == "SAVE_IN_FLIGHT"
            # The first optimizer mutation must finalize the in-flight save. A
            # no-gradient step is sufficient: it advances optimizer metadata
            # without needing the TE/cuDNN forward path blocked on this host.
            fence_started = time.perf_counter()
            for inner in managed:
                inner.step()
                inner.drain()
            fence_finished = time.perf_counter()
            assert engine.checkpointer.managed_async_save_state == "COMPLETE"
            assert engine.checkpointer._async_queue is not None
            assert engine.checkpointer._async_queue.get_num_unfinalized_calls() == 0
        finally:
            duration = time.perf_counter() - started
            snapshot_sampler_stop.set()
            snapshot_sampler.join()
        memory_after = _memory()
        fd_after = _fd_targets()
        complete_marker = (
            Path(args.checkpoint_dir) / ".areal-managed-async-complete.json"
        )
        incomplete_marker = (
            Path(args.checkpoint_dir) / ".areal-managed-async-incomplete.json"
        )
        assert complete_marker.is_file()
        assert not incomplete_marker.exists()
        marker_payload = json.loads(complete_marker.read_text())
    else:
        expected_payload = json.loads(
            (output_dir / f"expected_rank_{rank}.json").read_text()
        )
        assert identity_repr == expected_payload["identity"]
        initial_pointers = [
            tuple(
                slab.untyped_storage().data_ptr()
                for slab in (
                    inner.cpu_slabs.master,
                    inner.cpu_slabs.exp_avg,
                    inner.cpu_slabs.exp_avg_sq,
                )
            )
            for inner in managed
        ]
        torch.cuda.reset_peak_memory_stats()
        memory_before = _memory()
        rss_before = _rss_bytes()
        snapshot_peak = [0]
        snapshot_sampler_stop = threading.Event()
        snapshot_sampler = threading.Thread(
            target=_sample_snapshot_bytes,
            args=(snapshot_root, snapshot_sampler_stop, snapshot_peak),
            daemon=True,
        )
        snapshot_sampler.start()
        started = time.perf_counter()
        try:
            engine.checkpointer.load_checkpoint(
                args.checkpoint_dir,
                with_model=False,
                with_optimizer=True,
                with_rng=False,
            )
        finally:
            duration = time.perf_counter() - started
            snapshot_sampler_stop.set()
            snapshot_sampler.join()
        memory_after = _memory()
        scheduled = started
        fence_started = started
        fence_finished = started
        fd_before = _fd_targets()
        fd_after = fd_before
        marker_payload = None
        actual = [_residency_evidence(inner) for inner in managed]
        assert actual == expected_payload["optimizers"]
        assert initial_pointers == [
            tuple(
                slab.untyped_storage().data_ptr()
                for slab in (
                    inner.cpu_slabs.master,
                    inner.cpu_slabs.exp_avg,
                    inner.cpu_slabs.exp_avg_sq,
                )
            )
            for inner in managed
        ]

    checkpoint_bytes = sum(
        path.stat().st_size
        for path in Path(args.checkpoint_dir).rglob("*")
        if path.is_file()
    )
    evidence = [_residency_evidence(inner) for inner in managed]
    (output_dir / f"{args.mode}_rank_{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "world_size": dist.get_world_size(),
                "backend": args.backend,
                "mode": args.mode,
                "identity": identity_repr,
                "checkpoint_bytes": checkpoint_bytes,
                "rollback_snapshot_peak_bytes": snapshot_peak[0],
                "rollback_directories_after": len(tuple(snapshot_root.iterdir())),
                "duration_seconds": duration,
                "schedule_seconds": scheduled - started,
                "fence_wait_seconds": fence_finished - fence_started,
                "cuda_memory": {"before": memory_before, "after": memory_after},
                "rss_before": rss_before,
                "rss_after": _rss_bytes(),
                "rss_peak": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "fd_delta": len(fd_after) - len(fd_before),
                "queue_depth": (
                    engine.checkpointer._async_queue.get_num_unfinalized_calls()
                    if engine.checkpointer._async_queue is not None
                    else 0
                ),
                "active_worker_pids": [
                    process.pid for process in multiprocessing.active_children()
                ],
                "complete_marker_payload": marker_payload,
                "optimizers": evidence,
            },
            indent=2,
        )
        + "\n"
    )
    dist.barrier()
    engine.destroy()


if __name__ == "__main__":
    main()
