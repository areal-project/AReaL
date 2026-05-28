import os

import torch
import torch.distributed as dist
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2RotaryEmbedding,
)

from areal.infra.platforms import current_platform
from areal.models.fsdp.ulysses import set_ulysses_sequence_parallel_group
from areal.models.transformers.ulyssess_patch import apply_monkey_patch
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT


def setup_distributed_environment():
    if dist.is_initialized():
        return
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )
    current_platform.set_device(rank)


def run_ulysses_correctness_test():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"{current_platform.device_type}:{rank}")

    seq_len = 2048
    batch_size = 1
    hidden_size = 64

    config = Qwen2Config(
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=seq_len,
        attn_implementation="flash_attention_3",
    )
    attention = Qwen2Attention(config=config, layer_idx=0).to(device).to(torch.bfloat16)
    rotary_emb = Qwen2RotaryEmbedding(config=config)

    if rank == 0:
        state_dict = attention.state_dict()
        state_dict = {
            k: v.cpu() if torch.is_tensor(v) else v for k, v in state_dict.items()
        }
    else:
        state_dict = None
    state_dict = [state_dict]
    dist.broadcast_object_list(state_dict, src=0)
    state_dict = state_dict[0]

    attention.load_state_dict(state_dict)
    attention = attention.to(device)

    if rank == 0:
        hidden_states = torch.randn(
            (batch_size, seq_len, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
    else:
        hidden_states = torch.empty(
            (batch_size, seq_len, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
    dist.broadcast(hidden_states, src=0)

    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(
        0
    )
    position_embeddings = rotary_emb(hidden_states, position_ids)

    output_golden, _ = attention(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )

    sp_group = dist.new_group(
        ranks=list(range(world_size)),
        timeout=DIST_GROUP_DEFAULT_TIMEOUT,
        backend="nccl",
    )
    set_ulysses_sequence_parallel_group(sp_group)
    apply_monkey_patch(attention, ulysses_sp_size=world_size)

    hidden_states_sliced = (
        hidden_states.clone().split(seq_len // world_size, dim=1)[rank].contiguous()
    )
    position_ids_sliced = (
        position_ids.clone().split(seq_len // world_size, dim=1)[rank].contiguous()
    )
    position_embeddings_sliced = rotary_emb(hidden_states_sliced, position_ids_sliced)

    output_sp_sliced, _ = attention(
        hidden_states=hidden_states_sliced,
        attention_mask=None,
        position_ids=position_ids_sliced,
        position_embeddings=position_embeddings_sliced,
        cu_seq_lens_q=torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=device
        ),
        cu_seq_lens_k=torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=device
        ),
        max_length_q=seq_len,
        max_length_k=seq_len,
    )

    output_list = [torch.empty_like(output_sp_sliced) for _ in range(world_size)]
    dist.all_gather(output_list, output_sp_sliced, group=sp_group)

    if rank == 0:
        output_sp_full = torch.cat(output_list, dim=1)

        forward_pass_correct = torch.allclose(output_golden, output_sp_full, atol=1e-3)

        if not forward_pass_correct:
            raise ValueError(
                f"Ulysses SP implementation is wrong! Max difference: {(output_golden - output_sp_full).detach().abs().max().item()}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    setup_distributed_environment()
    run_ulysses_correctness_test()
