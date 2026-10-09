# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from torch import nn

from areal.models.mcore.qwen4_exp_awex_contract import visual_segments


class VocabParallelEmbedding(nn.Module):
    """The SGLang position embedding layout needed by the frozen contract."""

    def __init__(self, start: int = 576, end: int = 1152) -> None:
        super().__init__()
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=start, org_vocab_end_index=end
        )
        self.tp_size = 4
        self.use_presharded_weights = False
        self.org_vocab_size = 2304
        self.num_embeddings = 2304
        self.org_vocab_size_padded = 2304
        self.num_embeddings_padded = 2304
        self.embedding_dim = 1152
        self.num_embeddings_per_partition = 576


@pytest.mark.parametrize("start", [0, 576, 1152, 1728])
def test_visual_position_embedding_uses_its_own_tp_row_shard(start):
    segments, local_shape = visual_segments(
        VocabParallelEmbedding(start=start, end=start + 576),
        "model.visual.pos_embed.weight",
        (2304, 1152),
    )

    assert segments == [(0, start, start + 576, 0)]
    assert local_shape == (576, 1152)


@pytest.mark.parametrize(
    "change",
    [
        {"org_vocab_size_padded": 2368},
        {"use_presharded_weights": True},
        {"num_embeddings_per_partition": 640},
    ],
)
def test_visual_position_embedding_rejects_unverified_layout(change):
    module = VocabParallelEmbedding()
    for name, value in change.items():
        setattr(module, name, value)

    with pytest.raises(ValueError, match="frozen vision position"):
        visual_segments(module, "model.visual.pos_embed.weight", (2304, 1152))


def test_visual_position_embedding_rejects_noncontiguous_shard():
    module = VocabParallelEmbedding(start=577, end=1153)

    with pytest.raises(ValueError, match="frozen vision position shard"):
        visual_segments(module, "model.visual.pos_embed.weight", (2304, 1152))


def test_visual_position_embedding_rejects_other_vision_parameters():
    with pytest.raises(ValueError, match="Unsupported frozen vision module"):
        visual_segments(
            VocabParallelEmbedding(), "model.visual.other.weight", (2304, 1152)
        )
