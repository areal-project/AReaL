import pytest
import torch

import areal.utils.data as data_module
from areal.api.cli_args import MicroBatchSpec
from areal.engine.core.train_engine import (
    compute_microbatch_loss_weight,
    reorder_and_pad_outputs,
)
from areal.trainer.dpo.dpo_engine import _dpo_loss_weight
from areal.trainer.rw.rw_engine import _rw_loss_weight
from areal.utils.data import (
    TRANSPORT_DUMMY_KEY,
    MicroBatchList,
    pack_tensor_dict,
    pad_and_stack_tensors_along_first_dim,
    pad_mb_list,
    pad_sequences_to_tensors,
    reorder_list,
    split_padded_tensor_dict_into_mb_list,
    split_training_batch_into_microbatches,
    unpack_sequence,
)

BS = 16
MAX_ANSWER_LEN = 16
MAX_PROMPT_LEN = 8
VOCAB_SIZE = 100


@pytest.fixture
def mock_padded_data():
    prompt_lens = torch.randint(1, MAX_PROMPT_LEN, size=(BS,))
    answer_lens = torch.randint(1, MAX_ANSWER_LEN, size=(BS,))
    all_data = []
    for prompt_len, ans_len in zip(prompt_lens, answer_lens):
        prompt_len = int(prompt_len)
        ans_len = int(ans_len)
        seq = dict(
            input_ids=torch.randint(0, VOCAB_SIZE, size=(prompt_len + ans_len,)),
            loss_mask=torch.tensor([0] * prompt_len + [1] * ans_len),
            logprobs=torch.randn(prompt_len + ans_len),
            position_ids=torch.arange(prompt_len + ans_len),
        )
        all_data.append(seq)
    return pad_sequences_to_tensors(all_data)


@pytest.mark.parametrize("max_tokens_per_mb", [24, 36, 48, 100])
@pytest.mark.parametrize("n_mbs", [1, 2, 4, 8])
@pytest.mark.parametrize("n_mbs_divisor", [1, 2, 3])
def test_micro_batch_split(mock_padded_data, n_mbs, max_tokens_per_mb, n_mbs_divisor):
    mb_spec = MicroBatchSpec(
        n_mbs=n_mbs, max_tokens_per_mb=max_tokens_per_mb, n_mbs_divisor=n_mbs_divisor
    )

    # Unpad and split to microbatches
    packed_data = pack_tensor_dict(mock_padded_data)
    original_lens = packed_data["cu_seqlens"][1:] - packed_data["cu_seqlens"][:-1]
    assert torch.allclose(
        original_lens.long(), mock_padded_data["attention_mask"].sum(1)
    )
    split_result = split_padded_tensor_dict_into_mb_list(mock_padded_data, mb_spec)
    split_result.mbs = [pack_tensor_dict(mb) for mb in split_result.mbs]
    reordered_lens = [original_lens[i] for i in split_result.forward_indices]

    # assert microbatch split result does not violate requirements
    assert len(split_result.mbs) >= n_mbs
    assert len(split_result.mbs) % n_mbs_divisor == 0

    # test reorder back
    for key in split_result.mbs[0].keys():
        if key in ["cu_seqlens", "max_seqlen"]:
            continue

        # assert microbatch split result does not violate requirements
        for mb in split_result.mbs:
            assert mb[key].shape[0] <= max_tokens_per_mb

        x = torch.cat([mb[key] for mb in split_result.mbs])
        xs = unpack_sequence(x, lens=reordered_lens)
        xs = reorder_list(xs, split_result.backward_indices)
        x = torch.cat(xs)
        assert torch.allclose(x, packed_data[key])
        y = pad_and_stack_tensors_along_first_dim(xs)
        assert torch.allclose(mock_padded_data[key], y)


def _preference_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.arange(6).view(2, 3),
        "attention_mask": torch.ones(2, 3, dtype=torch.bool),
    }


@pytest.mark.parametrize("loss_weight_fn", [_dpo_loss_weight, _rw_loss_weight])
def test_transport_padding_bypasses_objective_weight(loss_weight_fn):
    mb_list = split_padded_tensor_dict_into_mb_list(
        _preference_batch(),
        MicroBatchSpec(n_mbs=2, granularity=2),
        allow_transport_padding=True,
    )
    mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
    semantic_mb, transport_mb = sorted(
        mb_list.mbs, key=lambda mb: TRANSPORT_DUMMY_KEY in mb
    )

    # A model-valid preference pair has non-zero objective weight. The transport
    # marker, rather than objective-specific fields, is what makes it weightless.
    torch.testing.assert_close(
        loss_weight_fn(transport_mb), torch.tensor(1.0), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        compute_microbatch_loss_weight(semantic_mb, loss_weight_fn),
        torch.tensor(1.0),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        compute_microbatch_loss_weight(transport_mb, loss_weight_fn),
        torch.tensor(0.0),
        rtol=0.0,
        atol=0.0,
    )

    pad_mb_list(mb_list)
    assert all(
        TRANSPORT_DUMMY_KEY not in padded_mb for padded_mb in mb_list.padded_mbs or []
    )


def test_noop_packed_padding_preserves_semantic_transport_marker():
    transport_mb = pack_tensor_dict(
        {
            "input_ids": torch.zeros(1, 1, dtype=torch.long),
            "attention_mask": torch.ones(1, 1, dtype=torch.bool),
            TRANSPORT_DUMMY_KEY: True,
        }
    )
    mb_list = MicroBatchList(
        data=transport_mb,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=1),
        mbs=[transport_mb],
        group_lens=[1],
        transport_dummy_count=1,
    )

    pad_mb_list(mb_list, pad_to_maximum=True)

    assert mb_list.padding_lengths == [0]
    assert mb_list.mbs[0][TRANSPORT_DUMMY_KEY] is True
    assert mb_list.padded_mbs is not None
    assert TRANSPORT_DUMMY_KEY not in mb_list.padded_mbs[0]

    callback_called = False

    def _loss_weight(_microbatch):
        nonlocal callback_called
        callback_called = True
        return torch.tensor(1.0)

    torch.testing.assert_close(
        compute_microbatch_loss_weight(mb_list.mbs[0], _loss_weight),
        torch.tensor(0.0),
        rtol=0.0,
        atol=0.0,
    )
    assert callback_called is False


def test_forward_transport_padding_is_removed_from_outputs():
    data = {
        "input_ids": torch.arange(3).view(1, 3),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    mb_list = split_padded_tensor_dict_into_mb_list(
        data,
        MicroBatchSpec(n_mbs=3),
        allow_transport_padding=True,
    )
    outputs = [mb["input_ids"][mb["attention_mask"]].float() for mb in mb_list.mbs]

    result = reorder_and_pad_outputs(outputs, [3], mb_list)

    torch.testing.assert_close(
        result, torch.tensor([[0.0, 1.0, 2.0]]), rtol=0.0, atol=0.0
    )


def test_transport_padding_converges_to_distributed_microbatch_count(monkeypatch):
    monkeypatch.setattr(data_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(data_module.dist, "get_world_size", lambda _group=None: 2)

    def _all_gather_counts(output, local_count, group=None):
        del group
        output[:] = [3, local_count]

    monkeypatch.setattr(data_module.dist, "all_gather_object", _all_gather_counts)

    mb_list = split_padded_tensor_dict_into_mb_list(
        {
            "input_ids": torch.arange(3).view(1, 3),
            "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        },
        MicroBatchSpec(),
        allow_transport_padding=True,
    )

    assert len(mb_list.mbs) == 3
    assert mb_list.transport_dummy_count == 2
    assert sum(TRANSPORT_DUMMY_KEY in mb for mb in mb_list.mbs) == 2


def _training_batch(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.arange(batch_size * 3).view(batch_size, 3),
        "attention_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 1]], dtype=torch.bool).repeat(batch_size, 1),
        "advantages": torch.ones(batch_size, 3),
    }


@pytest.mark.parametrize(
    ("rank", "batch_size", "expected_semantic_steps"),
    [(0, 1, [True, False, False]), (1, 2, [False, True, True])],
)
def test_synchronized_training_schedule_has_semantic_global_member_per_step(
    monkeypatch, rank, batch_size, expected_semantic_steps
):
    monkeypatch.setattr(data_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(data_module.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(data_module.dist, "get_rank", lambda group=None: rank)

    def _all_gather_counts(output, value, group=None):
        del value, group
        output[:] = [1, 2]

    monkeypatch.setattr(data_module.dist, "all_gather_object", _all_gather_counts)

    schedule = split_training_batch_into_microbatches(
        _training_batch(batch_size),
        n_mbs=4,
    )

    assert len(schedule) == 3
    assert [
        TRANSPORT_DUMMY_KEY not in microbatch for microbatch in schedule
    ] == expected_semantic_steps
    assert all(
        bool(microbatch["loss_mask"].any()) == is_semantic
        for microbatch, is_semantic in zip(
            schedule, expected_semantic_steps, strict=True
        )
    )


@pytest.mark.parametrize(
    ("rank", "batch_size", "expected_semantic_steps"),
    [(0, 3, [True, True, True]), (1, 1, [True, False, False])],
)
def test_synchronized_training_schedule_preserves_extra_local_microbatches(
    monkeypatch, rank, batch_size, expected_semantic_steps
):
    monkeypatch.setattr(data_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(data_module.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(data_module.dist, "get_rank", lambda group=None: rank)

    def _all_gather_counts(output, value, group=None):
        del value, group
        output[:] = [3, 1]

    def _split_into_local_microbatches(data, mb_spec, synchronize):
        del synchronize
        microbatches = [
            {key: value[index : index + 1] for key, value in data.items()}
            for index in range(batch_size)
        ]
        return MicroBatchList(
            data=data,
            mb_spec=mb_spec,
            mbs=microbatches,
            group_lens=[1] * batch_size,
        )

    monkeypatch.setattr(data_module.dist, "all_gather_object", _all_gather_counts)
    monkeypatch.setattr(
        data_module,
        "split_padded_tensor_dict_into_mb_list",
        _split_into_local_microbatches,
    )

    schedule = split_training_batch_into_microbatches(
        _training_batch(batch_size),
        n_mbs=2,
    )

    assert len(schedule) == 3
    assert [
        TRANSPORT_DUMMY_KEY not in microbatch for microbatch in schedule
    ] == expected_semantic_steps


def test_tensor_container_skeleton_round_trip():
    item = {
        "input_ids": torch.arange(6, dtype=torch.long).view(2, 3),
        "nested": [torch.ones(2, dtype=torch.bool), {"logprobs": torch.randn(4)}],
        "reward": 1.5,
        "task": "math",
    }

    tensors: list[torch.Tensor] = []
    skeleton = data_module._deconstruct_tensor_container(item, tensors)
    rebuilt = data_module._reconstruct_tensor_container(skeleton, iter(tensors))

    assert torch.equal(rebuilt["input_ids"], item["input_ids"])
    assert torch.equal(rebuilt["nested"][0], item["nested"][0])
    assert torch.equal(rebuilt["nested"][1]["logprobs"], item["nested"][1]["logprobs"])
    assert rebuilt["reward"] == 1.5
    assert rebuilt["task"] == "math"


def test_skeleton_tensor_leaves_follow_depth_first_order():
    tensors: list[torch.Tensor] = []
    items = [
        {"a": torch.zeros(1, dtype=torch.long), "b": [torch.zeros(2)]},
        {"a": torch.zeros(3, dtype=torch.long)},
    ]
    skeletons = [
        data_module._deconstruct_tensor_container(item, tensors) for item in items
    ]

    leaves = data_module._skeleton_tensor_leaves(skeletons)

    assert [leaf.shape for leaf in leaves] == [(1,), (2,), (3,)]
    assert [leaf.dtype for leaf in leaves] == [torch.long, torch.float32, torch.long]
    assert [leaf.numel for leaf in leaves] == [1, 2, 3]
