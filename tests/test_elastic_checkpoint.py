# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("backend", ["fsdp", "megatron"])
def test_native_full_checkpoint_restores_optimizer_scheduler_and_rng(
    tmp_path, monkeypatch, backend
):
    import random
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import numpy as np
    import torch

    from areal.v2.training_service import checkpoint as native

    class Engine:
        rank, world_size, device = 0, 1, torch.device("cpu")

        def __init__(self):
            self.model = torch.nn.Linear(2, 1)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1, gamma=0.5
            )
            self.version = 1
            self.checkpointer = SimpleNamespace(wait_async_saves=MagicMock())
            self.mcore_config = SimpleNamespace(use_checkpoint_opt_param_scheduler=True)

        def get_version(self):
            return self.version

        def set_version(self, version):
            self.version = version

        def save(self, meta):
            root = Path(meta.path)
            root.mkdir(exist_ok=True)
            (root / ".metadata").write_text("test native tensor payload")
            if backend == "megatron":
                (root / "metadata.json").write_text(
                    json.dumps(
                        {
                            "sharded_backend": "torch_dist",
                            "sharded_backend_version": 1,
                            "common_backend": "torch",
                            "common_backend_version": 1,
                        }
                    )
                )
                (root / "common.pt").write_text("test mcore common state")
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                },
                root / "state.pt",
            )
            torch.rand(1)

        def load(self, meta):
            saved = torch.load(Path(meta.path) / "state.pt", weights_only=True)
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            if backend == "megatron":
                self.lr_scheduler.load_state_dict(saved["lr_scheduler"])

    source = Engine()
    source.model(torch.ones(1, 2)).sum().backward()
    source.optimizer.step()
    source.lr_scheduler.step()
    tracker = SimpleNamespace(
        get_states=lambda: {"model": torch.tensor([1, 2, 3], dtype=torch.uint8)},
        set_states=MagicMock(),
    )
    monkeypatch.setattr(native, "_tracker", lambda: tracker)
    identity = "a" * 64
    metadata = dict(
        path=str(tmp_path), backend=backend, identity_sha256=identity, version=1
    )
    saved = native.save_full_checkpoint(source, **metadata)
    expected = (random.random(), np.random.rand(), torch.rand(4))
    restored = Engine()
    restored.version = 0
    if backend == "megatron":
        scheduler = restored.lr_scheduler

        def restore_additive_scheduler(state):
            # MCore restores by incrementing its existing num_steps. An
            # idempotent PyTorch scheduler would conceal a duplicate load.
            scheduler.load_state_dict(
                {**state, "last_epoch": scheduler.last_epoch + state["last_epoch"]}
            )

        restored.lr_scheduler = SimpleNamespace(
            state_dict=scheduler.state_dict,
            load_state_dict=MagicMock(side_effect=restore_additive_scheduler),
        )
    assert native.inspect_full_checkpoint(restored, loading=True, **metadata) == saved
    assert native.load_full_checkpoint(restored, **metadata) == saved
    actual = (random.random(), np.random.rand(), torch.rand(4))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    assert restored.get_version() == 1
    assert restored.optimizer.param_groups[0]["lr"] == 0.005
    for source_param, restored_param in zip(
        source.model.parameters(), restored.model.parameters()
    ):
        torch.testing.assert_close(source_param, restored_param, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                source.optimizer.state[source_param][key],
                restored.optimizer.state[restored_param][key],
                rtol=0,
                atol=0,
            )
    if backend == "megatron":
        restored.lr_scheduler.load_state_dict.assert_called_once()
        source.checkpointer.wait_async_saves.assert_called_once()
        tracker.set_states.assert_called()
    with pytest.raises(ValueError, match="identity"):
        native.inspect_full_checkpoint(
            restored, **{**metadata, "identity_sha256": "b" * 64}, loading=True
        )
