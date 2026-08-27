import pytest
import torch

from areal.api.cli_args import MicroBatchSpec
from areal.utils.data import (
    MicroBatchList,
    align_mb_list_sequences,
    pack_tensor_dict,
    pad_and_stack_tensors_along_first_dim,
    pad_sequences_to_tensors,
    reorder_list,
    split_padded_tensor_dict_into_mb_list,
    unpack_sequence,
    unpad_logits,
)

BS = 16
MAX_ANSWER_LEN = 16
MAX_PROMPT_LEN = 8
VOCAB_SIZE = 100


@pytest.mark.parametrize(
    ("seq_lens", "seq_align_to", "expected_cu_seqlens"),
    [
        ([14010], 1, [0, 14010]),
        ([7, 130], 8, [0, 8, 144]),
    ],
)
def test_align_mb_list_sequences_does_not_add_batch_row(
    seq_lens, seq_align_to, expected_cu_seqlens
):
    """BSHD sequence alignment must preserve the number of real batch rows."""
    total_length = sum(seq_lens)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    input_ids = torch.arange(total_length)
    mb = {
        "input_ids": input_ids,
        "position_ids": torch.cat([torch.arange(length) for length in seq_lens]),
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max(seq_lens),
    }
    mb_list = MicroBatchList(
        data=mb,
        mb_spec=MicroBatchSpec(),
        mbs=[mb],
        group_lens=[total_length],
    )

    aligned = align_mb_list_sequences(mb_list, seq_align_to=seq_align_to)

    assert aligned.padded_mbs is not None
    assert aligned.old_cu_seqlens_list is not None
    padded_mb = aligned.padded_mbs[0]
    assert padded_mb["cu_seqlens"].tolist() == expected_cu_seqlens
    assert padded_mb["cu_seqlens"].numel() == len(seq_lens) + 1
    assert aligned.padding_lengths == [0]
    assert aligned.padded_to_lengths == [expected_cu_seqlens[-1]]
    assert torch.all(
        (padded_mb["cu_seqlens"][1:] - padded_mb["cu_seqlens"][:-1]) % seq_align_to == 0
    ).item()

    restored_ids = unpad_logits(
        padded_mb["input_ids"],
        padding_length=0,
        cu_seqlens=padded_mb["cu_seqlens"],
        old_cu_seqlens=aligned.old_cu_seqlens_list[0],
    )
    torch.testing.assert_close(restored_ids, input_ids, rtol=0, atol=0)


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
