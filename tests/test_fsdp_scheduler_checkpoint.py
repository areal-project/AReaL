# SPDX-License-Identifier: Apache-2.0

"""Exercise scheduler recovery through the FSDP engine's public checkpoint API."""

import copy
import pickle
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.api import FinetuneSpec, SaveLoadMeta
from areal.api.cli_args import OptimizerConfig
from areal.engine.fsdp_engine import FSDPEngine


@pytest.fixture
def cpu_group(tmp_path: Path) -> Iterator[dist.ProcessGroup]:
    dist.init_process_group(
        "gloo",
        init_method=(tmp_path / "rendezvous").as_uri(),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=30),
    )
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


@pytest.fixture(params=["dcp", "hf"])
def checkpoint_format(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> str:
    if request.param == "hf":
        # Replace only model export/import: optimizer and scheduler use the real
        # engine paths, without a Transformers model or GPU collectives.
        def save_model(engine: FSDPEngine, path: str, *args: Any) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            torch.save(engine.model.state_dict(), Path(path) / "model.pt")

        def load_model(engine: FSDPEngine, path: str) -> None:
            engine.model.load_state_dict(
                torch.load(Path(path) / "model.pt", weights_only=True)
            )

        monkeypatch.setattr(FSDPEngine, "_save_model_to_hf", save_model)
        monkeypatch.setattr(FSDPEngine, "_load_model_from_hf", load_model)
    return request.param


def _engine(
    group: dist.ProcessGroup, schedule: str = "cosine", warmup_steps: int = 0
) -> FSDPEngine:
    engine = FSDPEngine.__new__(FSDPEngine)
    engine._initialized = True
    engine._cpu_group = group
    engine.world_size = dist.get_world_size(group)
    engine.is_offload = False
    engine._per_layer_optim_wrapper = None
    engine.is_vision_model = False
    engine.config = SimpleNamespace(optimizer_dtype="float32")
    engine.optimizer_config = OptimizerConfig(
        lr=1e-3,
        weight_decay=0.0,
        lr_scheduler_type=schedule,
        warmup_steps=warmup_steps,
    )
    engine.logger = Mock()
    engine.model = torch.nn.Linear(2, 1, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        engine.model.weight.fill_(1.0)
    engine._create_optimizer(
        FinetuneSpec(total_train_epochs=1, dataset_size=100, train_batch_size=1)
    )
    return engine


def _advance(engine: FSDPEngine) -> float:
    lr = engine.optimizer.param_groups[0]["lr"]
    engine.optimizer.zero_grad()
    engine.model(torch.ones((1, 2), device="cpu", dtype=torch.float32)).sum().backward()
    engine.optimizer.step()
    engine.lr_scheduler_step()
    return lr


@pytest.mark.parametrize(
    "schedule,warmup_steps", [("cosine", 0), ("linear", 10), ("constant", 10)]
)
@pytest.mark.parametrize("checkpoint_step", [5, 80])
def test_fsdp_resume_preserves_scheduler_and_following_updates(
    cpu_group: dist.ProcessGroup,
    checkpoint_format: str,
    tmp_path: Path,
    schedule: str,
    warmup_steps: int,
    checkpoint_step: int,
) -> None:
    """A fresh engine must match uninterrupted training beyond the first update."""
    original = _engine(cpu_group, schedule, warmup_steps)
    for _ in range(checkpoint_step):
        _advance(original)
    meta = SaveLoadMeta(
        str(tmp_path / "checkpoint"), checkpoint_format, with_optim=True
    )
    original.save(meta)

    resumed = _engine(cpu_group, schedule, warmup_steps)
    resumed.load(meta)

    assert resumed.lr_scheduler.state_dict() == original.lr_scheduler.state_dict()
    for _ in range(3):
        assert _advance(resumed) == _advance(original)
        torch.testing.assert_close(
            resumed.model.weight, original.model.weight, rtol=0, atol=0
        )
        assert resumed.lr_scheduler.state_dict() == original.lr_scheduler.state_dict()


def test_fsdp_legacy_checkpoint_warns_and_keeps_scheduler(
    cpu_group: dist.ProcessGroup, checkpoint_format: str, tmp_path: Path
) -> None:
    """Legacy model/optimizer state remains loadable without inventing progress."""
    original = _engine(cpu_group)
    for _ in range(80):
        _advance(original)
    meta = SaveLoadMeta(
        str(tmp_path / "checkpoint"), checkpoint_format, with_optim=True
    )
    original.save(meta)
    (Path(meta.path) / "lr_scheduler.pt").unlink()
    resumed = _engine(cpu_group)
    scheduler_before = copy.deepcopy(resumed.lr_scheduler.state_dict())

    resumed.load(meta)

    torch.testing.assert_close(
        resumed.model.weight, original.model.weight, rtol=0, atol=0
    )
    torch.testing.assert_close(
        resumed.optimizer.state_dict(), original.optimizer.state_dict(), rtol=0, atol=0
    )
    assert resumed.lr_scheduler.state_dict() == scheduler_before
    resumed.logger.warning.assert_called_once()
    assert "no LR scheduler state" in resumed.logger.warning.call_args.args[0]


@pytest.mark.parametrize("saved_with_optim", [False, True])
def test_fsdp_model_only_load_preserves_optimizer_and_scheduler(
    cpu_group: dist.ProcessGroup,
    checkpoint_format: str,
    tmp_path: Path,
    saved_with_optim: bool,
) -> None:
    """Weight-only operations neither require nor apply scheduler metadata."""
    original = _engine(cpu_group)
    for _ in range(10):
        _advance(original)
    meta = SaveLoadMeta(
        str(tmp_path / "checkpoint"), checkpoint_format, with_optim=saved_with_optim
    )
    original.save(meta)
    scheduler_path = Path(meta.path) / "lr_scheduler.pt"
    assert scheduler_path.exists() == saved_with_optim
    if saved_with_optim:
        scheduler_path.write_bytes(b"invalid scheduler checkpoint")
    resumed = _engine(cpu_group)
    scheduler_before = copy.deepcopy(resumed.lr_scheduler.state_dict())
    optimizer_before = copy.deepcopy(resumed.optimizer.state_dict())

    resumed.load(SaveLoadMeta(meta.path, checkpoint_format, with_optim=False))

    torch.testing.assert_close(
        resumed.model.weight, original.model.weight, rtol=0, atol=0
    )
    assert resumed.optimizer.state_dict() == optimizer_before
    assert resumed.lr_scheduler.state_dict() == scheduler_before
    resumed.logger.warning.assert_not_called()


def test_fsdp_corrupt_scheduler_checkpoint_raises(
    cpu_group: dist.ProcessGroup, checkpoint_format: str, tmp_path: Path
) -> None:
    """A present but corrupt sidecar must not silently restart the scheduler."""
    original = _engine(cpu_group)
    _advance(original)
    meta = SaveLoadMeta(
        str(tmp_path / "checkpoint"), checkpoint_format, with_optim=True
    )
    original.save(meta)
    (Path(meta.path) / "lr_scheduler.pt").write_bytes(b"invalid scheduler checkpoint")
    resumed = _engine(cpu_group)

    with pytest.raises(pickle.UnpicklingError):
        resumed.load(meta)

    resumed.logger.warning.assert_not_called()


def test_fsdp_invalid_checkpoint_format_writes_nothing(
    cpu_group: dist.ProcessGroup, tmp_path: Path
) -> None:
    """Reject invalid formats before writing scheduler metadata."""
    engine = _engine(cpu_group)
    path = tmp_path / "invalid"

    with pytest.raises(ValueError, match="Unknown weight format"):
        engine.save(SaveLoadMeta(str(path), "invalid", with_optim=True))

    assert not path.exists()


def _check_distributed_resume(rank: int, rendezvous: str, path: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        original = _engine(dist.group.WORLD)
        for _ in range(80):
            _advance(original)
        meta = SaveLoadMeta(path, "dcp", with_optim=True)
        scheduler_writes = []
        torch_save = torch.save

        def save(
            obj: Any, file: str | Path | BinaryIO, *args: Any, **kwargs: Any
        ) -> None:
            if file == str(Path(path) / "lr_scheduler.pt.tmp"):
                assert rank == 0
                scheduler_writes.append(file)
                # Let the peer enter DCP first: existing save collectives must
                # prevent it from returning before the sidecar is available.
                time.sleep(0.1)
            return torch_save(obj, file, *args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch, "save", save)
            original.save(meta)

        assert len(scheduler_writes) == (1 if rank == 0 else 0)
        assert (Path(path) / "lr_scheduler.pt").is_file()
        assert not (Path(path) / "lr_scheduler.pt.tmp").exists()
        resumed = _engine(dist.group.WORLD)
        resumed.load(meta)
        assert resumed.lr_scheduler.state_dict() == original.lr_scheduler.state_dict()
        for _ in range(3):
            assert _advance(resumed) == _advance(original)
            torch.testing.assert_close(
                resumed.model.weight, original.model.weight, rtol=0, atol=0
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
def test_fsdp_distributed_save_restores_scheduler_on_every_rank(tmp_path: Path) -> None:
    """One writer publishes scheduler state before either rank can resume."""
    mp.spawn(
        _check_distributed_resume,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path / "checkpoint")),
        nprocs=2,
        join=True,
    )
