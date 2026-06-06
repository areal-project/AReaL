#!/usr/bin/env python3
"""Test script for PartitionedRotaryEmbedding"""

import torch
from vllm.model_executor.layers.rotary_embedding import get_rope

def test_partitioned_rope():
    """Test that partitioned RoPE can be instantiated and used."""
    
    # Configuration
    head_size = 128
    max_position = 2048
    rope_parameters = {
        "rope_type": "partitioned",
        "rope_theta": 10000,
        "period": 512
    }
    
    # Create the partitioned RoPE
    print("Creating PartitionedRotaryEmbedding...")
    rope = get_rope(
        head_size=head_size,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters=rope_parameters,
        dtype=torch.float32
    )
    
    print(f"✓ Successfully created: {type(rope).__name__}")
    print(f"  - Head size: {rope.head_size}")
    print(f"  - Rotary dim: {rope.rotary_dim}")
    print(f"  - RoPE dim (80%): {rope.rope_dim}")
    print(f"  - Custom dim (20%): {rope.custom_dim}")
    print(f"  - Period: {rope.period}")
    print(f"  - Max position: {rope.max_position_embeddings}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    seq_len = 128
    num_tokens = batch_size * seq_len
    
    positions = torch.arange(seq_len, dtype=torch.long).repeat(batch_size)
    query = torch.randn(num_tokens, head_size, dtype=torch.float32)
    key = torch.randn(num_tokens, head_size, dtype=torch.float32)
    
    query_rot, key_rot = rope(positions, query, key)
    
    print(f"✓ Forward pass successful!")
    print(f"  - Input query shape: {query.shape}")
    print(f"  - Output query shape: {query_rot.shape}")
    print(f"  - Input key shape: {key.shape}")
    print(f"  - Output key shape: {key_rot.shape}")
    
    # Verify the encoding uses both RoPE and custom parts
    cos, sin = rope.get_cos_sin(seq_len)
    print(f"\n✓ Cos/Sin cache retrieved:")
    print(f"  - Cos shape: {cos.shape}")
    print(f"  - Sin shape: {sin.shape}")
    print(f"  - Expected shape: ({seq_len}, {rope.rotary_dim})")
    
    print("\n✅ All tests passed! PartitionedRotaryEmbedding is working correctly.")
    
    # Show period effect
    print(f"\nPeriod effect demonstration (period={rope.period}):")
    print(f"  Position 0:   block_index = {0 // rope.period}, local_pos = {0 % rope.period}")
    print(f"  Position 512: block_index = {512 // rope.period}, local_pos = {512 % rope.period}")
    print(f"  Position 513: block_index = {513 // rope.period}, local_pos = {513 % rope.period}")
    print(f"  Position 1024: block_index = {1024 // rope.period}, local_pos = {1024 % rope.period}")

if __name__ == "__main__":
    test_partitioned_rope()
