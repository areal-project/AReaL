# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest
import torch

from areal.api.cli_args import InferenceEngineConfig
from areal.infra.rpc import rtensor
from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
from areal.infra.rpc.serialization import serialize_value
from areal.infra.workflow_executor import WorkflowExecutor


@pytest.mark.asyncio
@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("with_optional_fields", [False, True])
async def test_dump_preserves_training_references_and_fetches_only_dump_fields(
    monkeypatch, tmp_path, remote, with_optional_fields
):
    """Dump output stays intact without materializing any training references."""
    tensors = {
        "input_ids": torch.tensor([[10, 11, 12, 13]] * 4),
        "loss_mask": torch.tensor([[0, 1, 0, 1]] * 4),
        "attention_mask": torch.ones((4, 4), dtype=torch.bool),
        "rewards": torch.ones(4),
        "logprobs": torch.zeros((4, 4)),
        "pixel_values": torch.ones((4, 3)),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    if with_optional_fields:
        tensors["versions"] = torch.tensor([[0, 2, 0, 3]] * 4)
        tensors["original_rewards"] = torch.full((4,), 2.0)

    backend = MagicMock()
    backend.fetch.side_effect = lambda shards: [
        tensors[shard.shard_id] for shard in shards
    ]
    monkeypatch.setattr(rtensor, "get_backend", lambda: backend)
    fetch_buffer = {}
    monkeypatch.setattr(rtensor, "_fetch_buffer", fetch_buffer)
    values = {
        key: (
            RTensor(
                shard=TensorShardInfo(shard_id=key, node_addr="test.invalid:1234"),
                data=tensor.to("meta"),
            )
            if remote
            else tensor
        )
        for key, tensor in tensors.items()
    }
    image = {key: values[key] for key in ("pixel_values", "image_grid_thw")}
    trajectory = {
        key: value
        for key, value in values.items()
        if key not in ("pixel_values", "image_grid_thw")
    }
    trajectory["multi_modal_input"] = [image] * 4
    before = serialize_value(trajectory)

    executor = WorkflowExecutor(
        config=InferenceEngineConfig(backend="sglang:d1", dump_to_file=True),
        inference_engine=MagicMock(),
        staleness_manager=MagicMock(),
    )
    executor.logger = MagicMock()
    executor.inference_engine.get_version.return_value = 7
    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda ids, **kwargs: " ".join(map(str, ids))
    monkeypatch.setattr(executor, "_get_dump_dir", lambda is_eval: str(tmp_path))
    monkeypatch.setattr(executor, "_get_tokenizer", lambda: tokenizer)

    success, reason = await executor._dump_trajectory(trajectory, 42, False)

    assert success, reason
    assert serialize_value(trajectory) == before
    assert all(item is image for item in trajectory["multi_modal_input"])
    if remote:
        assert all(value.data.is_meta for value in values.values())
        expected_fields = set(tensors) - {
            "pixel_values",
            "image_grid_thw",
            "logprobs",
        }
        backend.fetch.assert_called_once()
        assert {
            shard.shard_id for shard in backend.fetch.call_args.args[0]
        } == expected_fields
        assert set(fetch_buffer) == expected_fields
    else:
        backend.fetch.assert_not_called()

    tail = 3 if with_optional_fields else 7
    head = 2 if with_optional_fields else 7
    version_rle = [[2, 1], [3, 1]] if with_optional_fields else [[7, 2]]
    records = [
        json.loads(line)
        for line in (tmp_path / str(tail) / "42.jsonl").read_text().splitlines()
    ]
    assert len(records) == 4
    for i, record in enumerate(records):
        expected = {
            "task_id": 42,
            "sample_idx": i,
            "seqlen": 4,
            "prompt_len": 1,
            "head_version": head,
            "tail_version": tail,
            "version_rle": version_rle,
            "reward": 1.0,
            "prompt": "10",
            "completion": "11 12 13",
            "segments": [
                {"role": "prompt", "len": 1, "text": "10"},
                {"role": "gen", "len": 1, "text": "11"},
                {"role": "context", "len": 1, "text": "12"},
                {"role": "gen", "len": 1, "text": "13"},
            ],
        }
        if with_optional_fields:
            expected["original_reward"] = 2.0
        assert record == expected
