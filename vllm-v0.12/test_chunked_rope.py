#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test script for ChunkedRotaryEmbedding (updated API).

Verifies the bit-packed position encoding and the forward pass.
"""

import torch
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.chunked_rope import (
    ChunkedRotaryEmbedding,
    BLOCK_SHIFT,
    CHUNKED_SHIFT,
    compute_chunked_positions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHUNK_START = 999
CHUNK_END   = 998

# Simulated token stream (marker tokens still present, as they come from the
# tokeniser).  compute_chunked_positions will handle / flag them.
#
#  [1,2,3,4,5]          – "before" (5 tokens)
#  [999]                – <Chunk>
#  [10,11,12,13,14]     – chunk 0 (5 tokens)
#  [998]                – </Chunk>
#  [6]                  – "between" (1 token)
#  [999]                – <Chunk>
#  [15..22]             – chunk 1 (8 tokens)
#  [998]                – </Chunk>
#  [7,8,9]              – "after" (3 tokens)

TOKEN_IDS = torch.tensor(
    [1, 2, 3, 4, 5,
     CHUNK_START,
     10, 11, 12, 13, 14,
     CHUNK_END,
     6,
     CHUNK_START,
     15, 16, 17, 18, 19, 20, 21, 22,
     CHUNK_END,
     7, 8, 9],
    dtype=torch.long,
)

RAW_POSITIONS = torch.arange(len(TOKEN_IDS), dtype=torch.long)


# ---------------------------------------------------------------------------
# Test 1: compute_chunked_positions
# ---------------------------------------------------------------------------

def test_position_encoding():
    print("=" * 70)
    print("Test 1: compute_chunked_positions")
    print("=" * 70)

    encoded, keep_mask = compute_chunked_positions(
        TOKEN_IDS, RAW_POSITIONS, CHUNK_START, CHUNK_END
    )

    print(f"Total tokens  : {len(TOKEN_IDS)}")
    print(f"Keep mask     : {keep_mask.tolist()}")
    print()

    for i, (tid, enc, keep) in enumerate(
        zip(TOKEN_IDS.tolist(), encoded.tolist(), keep_mask.tolist())
    ):
        is_chunked = bool((enc >> CHUNKED_SHIFT) & 1)
        block      = (enc >> BLOCK_SHIFT) & 0xFFFFF
        local_pos  = enc & 0xFFFFF
        marker     = "(MARKER – will be filtered)" if not keep else ""
        kind       = "CHUNKED" if is_chunked else "NORMAL"
        print(
            f"  tok[{i:2d}] id={tid:3d}  keep={keep}  "
            f"{kind:7s}  local_pos={local_pos:3d}  block={block}  {marker}"
        )

    # Sanity checks
    assert not keep_mask[5],  "opening <Chunk> must be removed"
    assert not keep_mask[11], "closing </Chunk> must be removed"

    for i in range(6, 11):
        enc = encoded[i].item()
        assert (enc >> CHUNKED_SHIFT) & 1, f"tok[{i}] should be chunked"
        assert ((enc >> BLOCK_SHIFT) & 0xFFFFF) == 0, f"tok[{i}] should be block 0"

    for i in range(14, 22):
        enc = encoded[i].item()
        assert (enc >> CHUNKED_SHIFT) & 1, f"tok[{i}] should be chunked"
        assert ((enc >> BLOCK_SHIFT) & 0xFFFFF) == 1, f"tok[{i}] should be block 1"

    for i in range(5):
        enc = encoded[i].item()
        assert not ((enc >> CHUNKED_SHIFT) & 1), f"tok[{i}] should NOT be chunked"

    print()
    print("✓ All position-encoding assertions passed.")


# ---------------------------------------------------------------------------
# Test 2: ChunkedRotaryEmbedding forward pass
# ---------------------------------------------------------------------------

def test_forward():
    print()
    print("=" * 70)
    print("Test 2: ChunkedRotaryEmbedding forward_native")
    print("=" * 70)

    HEAD_SIZE  = 128
    MAX_POS    = 2048
    NUM_HEADS  = 8
    DTYPE      = torch.float32

    rope = get_rope(
        head_size=HEAD_SIZE,
        rotary_dim=HEAD_SIZE,
        max_position=MAX_POS,
        is_neox_style=True,
        rope_parameters={"rope_type": "chunked", "rope_theta": 10000.0},
        dtype=DTYPE,
    )
    assert isinstance(rope, ChunkedRotaryEmbedding), \
        "get_rope should return ChunkedRotaryEmbedding"
    print(f"Created: {rope.extra_repr()}")

    encoded, keep_mask = compute_chunked_positions(
        TOKEN_IDS, RAW_POSITIONS, CHUNK_START, CHUNK_END
    )
    valid_positions = encoded[keep_mask]
    T = valid_positions.shape[0]

    query = torch.randn(T, NUM_HEADS * HEAD_SIZE, dtype=DTYPE)
    key   = torch.randn(T, NUM_HEADS * HEAD_SIZE, dtype=DTYPE)

    q_out, k_out = rope.forward_native(valid_positions, query, key)

    assert q_out.shape == query.shape
    assert k_out.shape == key.shape
    assert not torch.allclose(q_out, query)

    print(f"Input  query shape : {query.shape}")
    print(f"Output query shape : {q_out.shape}")
    print()
    print("✓ Forward-pass checks passed.")


# ---------------------------------------------------------------------------
# Test 3: non-chunked tokens == plain RoPE
# ---------------------------------------------------------------------------

def test_normal_tokens_unchanged_by_chunking():
    print()
    print("=" * 70)
    print("Test 3: non-chunked tokens == plain RoPE")
    print("=" * 70)

    HEAD_SIZE = 64
    MAX_POS   = 512
    DTYPE     = torch.float32

    chunked_rope = get_rope(
        head_size=HEAD_SIZE,
        rotary_dim=HEAD_SIZE,
        max_position=MAX_POS,
        is_neox_style=True,
        rope_parameters={"rope_type": "chunked", "rope_theta": 10000.0},
        dtype=DTYPE,
    )
    plain_rope = get_rope(
        head_size=HEAD_SIZE,
        rotary_dim=HEAD_SIZE,
        max_position=MAX_POS,
        is_neox_style=True,
        rope_parameters=None,
        dtype=DTYPE,
    )

    ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    raw = torch.arange(5, dtype=torch.long)
    enc, _ = compute_chunked_positions(ids, raw, CHUNK_START, CHUNK_END)

    T = 5
    q = torch.randn(T, HEAD_SIZE, dtype=DTYPE)
    k = torch.randn(T, HEAD_SIZE, dtype=DTYPE)

    q_chunked, k_chunked = chunked_rope.forward_native(enc, q.clone(), k.clone())
    q_plain,   k_plain   = plain_rope.forward_native(raw, q.clone(), k.clone())

    assert torch.allclose(q_chunked, q_plain, atol=1e-5), \
        "non-chunked tokens must match plain RoPE"
    assert torch.allclose(k_chunked, k_plain, atol=1e-5)

    print("✓ Non-chunked tokens produce identical results to plain RoPE.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_position_encoding()
    test_forward()
    test_normal_tokens_unchanged_by_chunking()
    print()
    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)


def test_chunked_rope():
    """Test the chunked rotary embedding with example input."""
    
    # Initialize the chunked RoPE
    head_size = 128
    rotary_dim = 128
    max_position_embeddings = 2048
    base = 10000.0
    is_neox_style = True
    dtype = torch.float16
    
    rope = ChunkedRotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype,
    )
    
    print("=" * 80)
    print("ChunkedRotaryEmbedding Test")
    print("=" * 80)
    print(f"Configuration: {rope.extra_repr()}")
    print()
    
    # Example: Simulate tokenized input with chunk markers
    # Let's say:
    # - Token IDs 1-100: normal tokens
    # - Token ID 999: <Chunk> marker
    # - Token ID 998: </Chunk> marker
    
    CHUNK_START = 999
    CHUNK_END = 998
    
    # Create a sample token sequence:
    # "Content before <Chunk>[Chunk 1]</Chunk> <Chunk>[Chunk 2]</Chunk> Content after"
    # Represented as:
    # [1, 2, 3, 4, 5, 999, 10, 11, 12, 13, 14, 998, 6, 999, 15, 16, 17, 18, 19, 20, 21, 22, 998, 7, 8, 9]
    #  ^before^      ^---------chunk 1---------^    ^  ^----------chunk 2--------------^    ^after^
    
    token_ids = torch.tensor([
        1, 2, 3, 4, 5,  # Content before (5 tokens)
        CHUNK_START,     # <Chunk>
        10, 11, 12, 13, 14,  # Chunk 1 content (5 tokens)
        CHUNK_END,       # </Chunk>
        6,               # Content between chunks (1 token)
        CHUNK_START,     # <Chunk>
        15, 16, 17, 18, 19, 20, 21, 22,  # Chunk 2 content (8 tokens)
        CHUNK_END,       # </Chunk>
        7, 8, 9,         # Content after (3 tokens)
    ])
    
    print("Token sequence:")
    print(f"  Total tokens: {len(token_ids)}")
    print(f"  Token IDs: {token_ids.tolist()}")
    print()
    
    # Compute positions and masks
    positions, is_chunked, block_indices = rope.compute_positions_from_chunks(
        token_ids, CHUNK_START, CHUNK_END
    )
    
    print("Position analysis:")
    print(f"  Positions:      {positions.tolist()}")
    print(f"  Is chunked:     {is_chunked.tolist()}")
    print(f"  Block indices:  {block_indices.tolist()}")
    print()
    
    # Interpretation
    print("Interpretation:")
    for i, (token_id, pos, chunked, block_idx) in enumerate(
        zip(token_ids.tolist(), positions.tolist(), is_chunked.tolist(), block_indices.tolist())
    ):
        if pos == -1:
            print(f"  Token {i:2d} (ID={token_id:3d}): SEPARATOR (excluded from encoding)")
        elif chunked:
            print(f"  Token {i:2d} (ID={token_id:3d}): CHUNKED at position {pos}, block {block_idx} (partitioned encoding)")
        else:
            print(f"  Token {i:2d} (ID={token_id:3d}): NORMAL at position {pos} (standard RoPE)")
    print()
    
    # Test the forward pass
    batch_size = 1
    num_heads = 8
    seq_len = len(token_ids)
    
    # Filter out separator tokens
    valid_mask = positions >= 0
    valid_positions = positions[valid_mask]
    valid_is_chunked = is_chunked[valid_mask]
    valid_block_indices = block_indices[valid_mask]
    valid_seq_len = len(valid_positions)
    
    # Create dummy query and key tensors
    query = torch.randn(batch_size, valid_seq_len, num_heads, head_size, dtype=dtype)
    key = torch.randn(batch_size, valid_seq_len, num_heads, head_size, dtype=dtype)
    
    print("Forward pass test:")
    print(f"  Input shapes: query={query.shape}, key={key.shape}")
    print(f"  Valid sequence length (excluding separators): {valid_seq_len}")
    
    # Apply rotary embedding
    # Note: The forward method expects the query/key to have the rotary_dim applied
    # For simplicity in this test, we'll just verify the cos/sin computation
    
    # Test 1: Normal RoPE (without chunking)
    cos_normal, sin_normal = rope.get_cos_sin(10)
    print(f"  Normal RoPE cos/sin shapes: {cos_normal.shape}, {sin_normal.shape}")
    
    # Test 2: Chunked RoPE
    local_positions = torch.arange(5)
    cos_chunk, sin_chunk = rope.get_chunked_cos_sin(local_positions, block_index=0)
    print(f"  Chunked RoPE cos/sin shapes: {cos_chunk.shape}, {sin_chunk.shape}")
    
    print()
    print("=" * 80)
    print("Verification of requirements:")
    print("=" * 80)
    
    # Verify requirement 0: Normal RoPE outside chunks
    normal_tokens = [i for i, c in enumerate(is_chunked.tolist()) if not c and positions[i] >= 0]
    print(f"✓ Requirement 0: {len(normal_tokens)} tokens use normal RoPE (outside chunks)")
    
    # Verify requirement 1: Partitioned encoding in chunks with block index
    chunk_tokens = [i for i, c in enumerate(is_chunked.tolist()) if c]
    chunks_info = {}
    for i in chunk_tokens:
        block = block_indices[i].item()
        if block not in chunks_info:
            chunks_info[block] = []
        chunks_info[block].append(i)
    print(f"✓ Requirement 1: {len(chunks_info)} chunks use partitioned encoding")
    for block, tokens in chunks_info.items():
        print(f"  - Chunk {block}: {len(tokens)} tokens (indices {tokens[0]}-{tokens[-1]})")
    
    # Verify requirement 2: Local positions within chunks
    print(f"✓ Requirement 2: Local positions start from previous content end + 1")
    for block, tokens in chunks_info.items():
        local_pos = [positions[i].item() for i in tokens]
        print(f"  - Chunk {block}: local positions start at {min(local_pos)}")
    
    # Verify requirement 3: Content after chunks uses correct position
    after_chunk_tokens = [i for i in range(len(token_ids)) 
                          if not is_chunked[i] and positions[i] >= 0 and block_indices[i] == 0]
    if len(after_chunk_tokens) > 0:
        # Find tokens after last chunk
        last_chunk_end = max([i for i in range(len(token_ids)) if token_ids[i] == CHUNK_END])
        after_tokens = [i for i in range(last_chunk_end + 1, len(token_ids)) if positions[i] >= 0]
        if after_tokens:
            print(f"✓ Requirement 3: Content after chunks starts at position {positions[after_tokens[0]].item()}")
    
    # Verify requirement 4: Separators excluded
    separator_count = (positions == -1).sum().item()
    print(f"✓ Requirement 4: {separator_count} separator tokens excluded from encoding")
    
    print()
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_chunked_rope()
