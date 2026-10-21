# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap

import pytest

from areal.utils import logging


@pytest.mark.slow
@pytest.mark.ci
def test_workflow_dump_path_without_metrics_sdks_creates_directory(tmp_path):
    """Inference-only workers must not import SDKs to resolve a dump directory."""
    script = textwrap.dedent(
        """
        import asyncio
        import getpass
        import importlib.abc
        import json
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        blocked = {"swanlab", "wandb", "tensorboardX"}

        class NoMetricsSDKs(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError(fullname)

        sys.meta_path.insert(0, NoMetricsSDKs())
        original_find_spec = importlib.util.find_spec

        def optional_package_spec(name, *args, **kwargs):
            # Accelerate checks optional SDK availability before importing it.
            # Missing packages return None here, while actual imports still fail.
            if name.split(".")[0] in blocked:
                return None
            return original_find_spec(name, *args, **kwargs)

        importlib.util.find_spec = optional_package_spec
        getpass.getuser = lambda: "test-user"

        from areal.infra.workflow_executor import WorkflowExecutor
        import torch

        executor = object.__new__(WorkflowExecutor)
        executor.config = SimpleNamespace(
            fileroot=sys.argv[1], experiment_name="experiment", trial_name="trial"
        )
        expected = Path(sys.argv[1]) / "logs/test-user/experiment/trial"
        assert executor._get_dump_dir(False) == str(expected / "rollout")
        assert executor._get_dump_dir(True) == str(expected / "eval-rollout")
        assert expected.is_dir()
        assert "areal.utils.stats_logger" not in sys.modules
        assert not blocked.intersection(sys.modules)

        executor._get_tokenizer = lambda: SimpleNamespace(
            decode=lambda ids, skip_special_tokens=False: " ".join(
                str(token) for token in ids
            )
        )
        trajectory = {
            "input_ids": torch.tensor([[10, 20]]),
            "loss_mask": torch.tensor([[0, 1]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "versions": torch.tensor([[0, 0]]),
            "rewards": torch.tensor([0.0]),
        }
        ok, reason = asyncio.run(executor._dump_trajectory(trajectory, 1, False))
        assert ok, reason
        record = json.loads((expected / "rollout/0/1.jsonl").read_text())
        assert record["completion"] == "20"
        assert record["reward"] == 0.0
        assert not blocked.intersection(sys.modules)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("missing", ["experiment_name", "trial_name", "fileroot"])
def test_experiment_log_path_missing_identifier_raises_value_error(tmp_path, missing):
    identifiers = {
        "experiment_name": "experiment",
        "trial_name": "trial",
        "fileroot": str(tmp_path),
    }
    identifiers[missing] = None

    with pytest.raises(ValueError, match="must be provided"):
        logging.get_experiment_log_path(**identifiers)
