# Chunked Rotary Positional Embedding

This implementation provides a hybrid positional encoding scheme for handling chunked content with special encoding requirements.

## Overview

The `ChunkedRotaryEmbedding` class extends the standard Rotary Positional Embedding (RoPE) to support content that is divided into chunks with special markers `<Chunk>` and `</Chunk>`.

## Features

### 1. Dual Encoding Modes

- **Normal RoPE**: Applied to all content outside `<Chunk>...</Chunk>` markers
- **Partitioned RoPE**: Applied to content inside chunks, using:
  - **80% of dimensions**: Standard RoPE with local positions (position within the chunk)
  - **20% of dimensions**: Custom sinusoidal encoding based on block index (chunk number)

### 2. Position Tracking

The encoding handles complex position tracking across chunk boundaries:

```
Input: "Content before <Chunk>[Chunk 1]</Chunk> <Chunk>[Chunk 2]</Chunk> Content after"

Positions:
- "Content before": 0, 1, 2, ..., n (normal RoPE)
- "<Chunk>": EXCLUDED (separator)
- "[Chunk 1]": (n+1), (n+2), ... (partitioned, block_index=0, local positions)
- "</Chunk>": EXCLUDED (separator)
- "<Chunk>": EXCLUDED (separator)
- "[Chunk 2]": (n+1), (n+2), ... (partitioned, block_index=1, local positions)
- "</Chunk>": EXCLUDED (separator)
- "Content after": (n + max_chunk_length + 1), ... (normal RoPE)
```

### 3. Requirements Implementation

#### Requirement 0: Normal RoPE Outside Chunks
All content not enclosed in `<Chunk>...</Chunk>` uses standard RoPE encoding.

#### Requirement 1: Block Index for Chunks
Content inside chunks uses partitioned encoding where the block index equals the chunk number (0-indexed):
- First chunk: block_index = 0
- Second chunk: block_index = 1
- Third chunk: block_index = 2
- etc.

The block index affects the 20% custom encoding dimensions:
```
R_ij = sin(2π * i * j * B / n) / sqrt(n)
```
where B is the block index.

#### Requirement 2: Local Positions Within Chunks
Each chunk's local positions start from where the previous content ended + 1:
- If "Content before" ends at position `i`, the first chunk starts at position `i+1`
- Positions within each chunk grow normally from this starting point

#### Requirement 3: Position After Chunks
Content after all chunks starts at:
```
position = (last_position_before_chunks) + (max_chunk_length) + 1
```

For example, if:
- 10 tokens before chunks (positions 0-9)
- 3 chunks with lengths: 5, 8, 7 tokens
- max_chunk_length = 8

Then content after chunks starts at position: 10 + 8 + 1 = 19

#### Requirement 4: Separator Exclusion
`<Chunk>` and `</Chunk>` tokens are:
- Never sent to the model
- Never counted in positional encoding
- Marked with position = -1 (special separator marker)

## Usage

### Basic Usage

```python
import torch
from vllm.model_executor.layers.rotary_embedding.chunked_rope import ChunkedRotaryEmbedding

# Initialize
rope = ChunkedRotaryEmbedding(
    head_size=128,
    rotary_dim=128,
    max_position_embeddings=2048,
    base=10000.0,
    is_neox_style=True,
    dtype=torch.float16,
    chunk_partition_ratio=0.8,  # 80% RoPE, 20% custom
)

# Define chunk marker token IDs
CHUNK_START_TOKEN_ID = 999
CHUNK_END_TOKEN_ID = 998

# Create token sequence with chunk markers
token_ids = torch.tensor([
    1, 2, 3, 4, 5,              # Content before
    CHUNK_START_TOKEN_ID,       # <Chunk>
    10, 11, 12, 13, 14,         # Chunk 1 content
    CHUNK_END_TOKEN_ID,         # </Chunk>
    6,                          # Content between chunks
    CHUNK_START_TOKEN_ID,       # <Chunk>
    15, 16, 17, 18, 19,         # Chunk 2 content
    CHUNK_END_TOKEN_ID,         # </Chunk>
    7, 8, 9,                    # Content after
])

# Compute positions and masks
positions, is_chunked, block_indices = rope.compute_positions_from_chunks(
    token_ids, CHUNK_START_TOKEN_ID, CHUNK_END_TOKEN_ID
)

# Filter out separators
valid_mask = positions >= 0
valid_positions = positions[valid_mask]
valid_is_chunked = is_chunked[valid_mask]
valid_block_indices = block_indices[valid_mask]

# Apply rotary embedding (in your model)
query_rotated, key_rotated = rope.forward(
    valid_positions, query, key, valid_is_chunked, valid_block_indices
)
```

### Integration with vLLM

To use this in a vLLM model:

1. **Register the embedding class** in your model's `__init__.py`:
```python
from vllm.model_executor.layers.rotary_embedding.chunked_rope import ChunkedRotaryEmbedding
```

2. **Configure in model definition**:
```python
self.rotary_emb = ChunkedRotaryEmbedding(
    head_size=self.head_size,
    rotary_dim=self.rotary_dim,
    max_position_embeddings=config.max_position_embeddings,
    base=config.rope_theta,
    is_neox_style=True,
    dtype=self.dtype,
)
```

3. **Preprocess input** to compute positions:
```python
positions, is_chunked, block_indices = self.rotary_emb.compute_positions_from_chunks(
    input_ids, CHUNK_START_TOKEN_ID, CHUNK_END_TOKEN_ID
)

# Filter out separator tokens before passing to model
valid_mask = positions >= 0
input_ids_filtered = input_ids[valid_mask]
positions_filtered = positions[valid_mask]
is_chunked_filtered = is_chunked[valid_mask]
block_indices_filtered = block_indices[valid_mask]
```

4. **Apply in attention layer**:
```python
q, k = self.rotary_emb.forward(
    positions_filtered, q, k, is_chunked_filtered, block_indices_filtered
)
```

## Implementation Details

### Partitioned Encoding

For chunked content, the encoding is partitioned:

1. **First 80% of dimensions** (RoPE part):
   - Uses local positions within the chunk
   - Standard RoPE formula: `R(pos, i) = [cos(pos/θ^(2i/d)), sin(pos/θ^(2i/d))]`

2. **Last 20% of dimensions** (Custom part):
   - Uses block index (chunk number)
   - Custom formula: `R_ij = sin(2π * i * j * B / n) / sqrt(n)`
   - This creates position-independent but chunk-aware encoding

### Position Computation Algorithm

The `compute_positions_from_chunks` method:

1. Scans through token IDs sequentially
2. Tracks state: `in_chunk`, `chunk_idx`, `chunk_local_pos`
3. For each token:
   - If `<Chunk>`: mark as separator, enter chunk mode
   - If `</Chunk>`: mark as separator, exit chunk mode, record chunk length
   - If in chunk: assign local position, mark as chunked
   - If outside chunk: assign normal incremental position
4. For post-chunk content: position = pre_chunk_end + max_chunk_length + 1

### Cos/Sin Computation

- **Normal RoPE**: Uses cached cos/sin from base class
- **Chunked RoPE**: Computes on-the-fly with:
  - RoPE part using local positions
  - Custom part using block index

## Testing

Run the test script to verify functionality:

```bash
python test_chunked_rope.py
```

This will:
- Create a sample sequence with chunks
- Compute positions and masks
- Verify all requirements are met
- Display detailed position information

## Performance Considerations

1. **Separator Filtering**: Separators should be filtered out before passing to the model to avoid wasted computation
2. **Chunked Encoding**: Only computed for tokens marked as chunked (efficient)
3. **Caching**: Normal RoPE uses cached cos/sin; chunked encoding computed on-demand
4. **Batch Processing**: Can handle different chunk configurations per batch item

## Limitations

1. Assumes chunk markers are properly paired (`<Chunk>` followed by `</Chunk>`)
2. Chunk marker token IDs must be provided externally
3. All chunks in a sequence must use the same partition ratio
4. Maximum sequence length still bounded by `max_position_embeddings`

## Future Enhancements

- [ ] Support for nested chunks
- [ ] Variable partition ratios per chunk
- [ ] Automatic chunk marker detection
- [ ] Optimized CUDA kernels for chunked encoding
- [ ] Support for attention masking based on chunk boundaries

## References

- Original RoPE paper: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- vLLM documentation: [vLLM Positional Encodings](https://docs.vllm.ai/)
