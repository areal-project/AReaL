# SPDX-License-Identifier: Apache-2.0

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from areal.api import LocalInfServerInfo
from areal.infra.utils import inference_targets
from areal.infra.utils.exp_metadata import get_metadata_dir


@pytest.fixture
def target_writer(tmp_path):
    def write(*, engine="sglang", role="rollout", hosts=("127.0.0.1",), **overrides):
        kwargs = dict(
            inf_engine=type(f"Remote{engine}Engine", (), {}),
            server_infos=[
                LocalInfServerInfo(host=host, port=8000 + rank, process=None)
                for rank, host in enumerate(hosts)
            ],
            fileroot=str(tmp_path),
            experiment_name="experiment",
            trial_name="trial",
            role=role,
            source="separation",
        )
        kwargs.update(overrides)
        inference_targets.write_inference_targets(**kwargs)
        return (
            Path(get_metadata_dir(str(tmp_path), "experiment", "trial"))
            / f"{engine}_targets.json"
        )

    return write


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
def test_write_targets_supported_engine_exports_file_sd(target_writer, engine):
    """Both backends export HTTP addresses and controller labels."""
    path = target_writer(engine=engine, hosts=("127.0.0.1", "::1", "[::1]", ""))
    groups = json.loads(path.read_text())
    assert [group["targets"] for group in groups] == [
        ["127.0.0.1:8000"],
        ["[::1]:8001"],
        ["[::1]:8002"],
    ]
    assert groups[0]["labels"] == {
        "areal_backend": engine,
        "areal_metrics_path": "/metrics",
        "areal_rank": "0",
        "areal_role": "rollout",
        "areal_deployment_mode": "separation",
    }


def test_write_targets_repeated_role_replaces_only_its_targets(target_writer):
    """Updating one controller preserves targets belonging to another role."""
    target_writer()
    target_writer(role="teacher", hosts=("teacher",))
    path = target_writer(hosts=("replacement",), source="provided")
    groups = json.loads(path.read_text())
    assert len(groups) == 2
    by_role = {group["labels"]["areal_role"]: group for group in groups}
    assert by_role["teacher"]["targets"] == ["teacher:8000"]
    assert by_role["rollout"]["targets"] == ["replacement:8000"]
    assert by_role["rollout"]["labels"]["areal_deployment_mode"] == "provided"


@pytest.mark.parametrize(
    "overrides",
    [
        {"engine": "unsupported"},
        {"hosts": ()},
        {"hosts": ("",)},
        {"fileroot": None},
        {"experiment_name": None},
        {"trial_name": None},
    ],
)
def test_write_targets_missing_inputs_does_not_publish(target_writer, overrides):
    """Unsupported engines and incomplete inputs do not publish discovery files."""
    assert not target_writer(**overrides).exists()


@pytest.mark.parametrize("contents", ["invalid JSON", "{}", '[null, {"labels": null}]'])
def test_write_targets_invalid_existing_file_recovers(target_writer, contents):
    """Malformed previous metadata can be replaced with valid current targets."""
    path = target_writer()
    path.write_text(contents)
    target_writer(hosts=("replacement",))
    assert json.loads(path.read_text())[0]["targets"] == ["replacement:8000"]


def test_write_targets_replace_failure_preserves_previous_file(
    target_writer, monkeypatch
):
    """Publishing failures leave the old file readable and do not stop training."""
    path = target_writer()
    previous = path.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(inference_targets.os, "replace", fail_replace)
        target_writer(hosts=("replacement",))
    assert path.read_bytes() == previous
    target_writer(hosts=("retry",))
    assert json.loads(path.read_text())[0]["targets"] == ["retry:8000"]


def test_write_targets_concurrent_roles_preserves_all_targets(target_writer):
    """Simultaneous controller writers cannot overwrite another role's update."""
    roles = [f"teacher-{rank}" for rank in range(8)]
    with ThreadPoolExecutor(max_workers=len(roles)) as pool:
        paths = list(pool.map(lambda role: target_writer(role=role), roles))
    groups = json.loads(paths[0].read_text())
    assert sorted(group["labels"]["areal_role"] for group in groups) == roles
