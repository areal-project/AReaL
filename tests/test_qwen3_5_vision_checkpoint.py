import torch
from torch import nn

from areal.models.mcore.vision_checkpoint import checkpoint_qwen3_5_vision_blocks


class VisionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.forward_calls = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_calls += 1
        cosine, sine = position_embeddings
        return hidden_states * self.weight * cosine + sine + cu_seqlens[-1]


class Qwen3_5VisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([VisionBlock(), VisionBlock()])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cosine = torch.full_like(hidden_states, 0.5)
        sine = torch.full_like(hidden_states, 0.25)
        cu_seqlens = torch.tensor([0, hidden_states.numel()])
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                position_embeddings=(cosine, sine),
                cu_seqlens=cu_seqlens,
            )
        return hidden_states


Qwen3_5VisionModel.__module__ = "transformers.models.qwen3_5.test"


def test_qwen3_5_vision_checkpoint_recomputes_blocks_and_preserves_gradients() -> None:
    reference = Qwen3_5VisionModel()
    checkpointed = Qwen3_5VisionModel()
    checkpointed.load_state_dict(reference.state_dict())

    assert checkpoint_qwen3_5_vision_blocks([checkpointed]) == 2
    assert checkpoint_qwen3_5_vision_blocks([checkpointed]) == 0

    reference_input = torch.tensor([1.0, 2.0], requires_grad=True)
    checkpoint_input = reference_input.detach().clone().requires_grad_()
    reference(reference_input).sum().backward()
    checkpointed(checkpoint_input).sum().backward()

    torch.testing.assert_close(
        checkpoint_input.grad, reference_input.grad, rtol=0, atol=0
    )
    for checkpoint_block, reference_block in zip(
        checkpointed.blocks, reference.blocks, strict=True
    ):
        torch.testing.assert_close(
            checkpoint_block.weight.grad, reference_block.weight.grad, rtol=0, atol=0
        )
        assert checkpoint_block.forward_calls == 2
        assert reference_block.forward_calls == 1

    with torch.no_grad():
        checkpointed(torch.tensor([1.0, 2.0]))
    assert all(block.forward_calls == 3 for block in checkpointed.blocks)
