import pytest
import torch

from areal.api.cli_args import MicroBatchSpec
from areal.utils.data import (
    pack_tensor_dict,
    pad_and_stack_tensors_along_first_dim,
    pad_sequences_to_tensors,
    reorder_list,
    split_padded_tensor_dict_into_mb_list,
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
