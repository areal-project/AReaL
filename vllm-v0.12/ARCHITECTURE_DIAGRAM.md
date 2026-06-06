# Architecture Diagram: Chunked Rotary Positional Encoding

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input Text with Chunks                        │
│  "Before <Chunk>Content1</Chunk> <Chunk>Content2</Chunk> After" │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Tokenization                                │
│         [tok1, tok2, <C>, tok3, tok4, </C>, ...]                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Position Computation (chunked_rope.py)              │
│    compute_positions_from_chunks(tokens, start_id, end_id)      │
│                                                                   │
│  Returns:                                                         │
│  • positions: [0, 1, -1, 2, 3, -1, ...]                         │
│  • is_chunked: [F, F, -, T, T, -, ...]                          │
│  • block_indices: [0, 0, -, 0, 0, -, ...]                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Filter Separators                              │
│              valid_mask = (positions >= 0)                       │
│         filtered_tokens = tokens[valid_mask]                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Model Forward Pass                             │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Attention Layer                              │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────┐     │  │
│  │  │    Q, K, V Projections                          │     │  │
│  │  └─────────────────┬───────────────────────────────┘     │  │
│  │                    │                                      │  │
│  │                    ▼                                      │  │
│  │  ┌─────────────────────────────────────────────────┐     │  │
│  │  │   ChunkedRotaryEmbedding.forward()              │     │  │
│  │  │                                                   │     │  │
│  │  │   For each token:                                │     │  │
│  │  │   ┌─────────────────────────────────────────┐   │     │  │
│  │  │   │ If is_chunked[i] == False:              │   │     │  │
│  │  │   │   → Use Normal RoPE                      │   │     │  │
│  │  │   │   → All rotary_dim dimensions           │   │     │  │
│  │  │   │                                          │   │     │  │
│  │  │   │ If is_chunked[i] == True:               │   │     │  │
│  │  │   │   → Use Partitioned Encoding            │   │     │  │
│  │  │   │   → 80% dims: RoPE(local_position)      │   │     │  │
│  │  │   │   → 20% dims: Custom(block_index)       │   │     │  │
│  │  │   └─────────────────────────────────────────┘   │     │  │
│  │  └─────────────────┬───────────────────────────────┘     │  │
│  │                    │                                      │  │
│  │                    ▼                                      │  │
│  │  ┌─────────────────────────────────────────────────┐     │  │
│  │  │   Rotated Q, K                                  │     │  │
│  │  └─────────────────┬───────────────────────────────┘     │  │
│  └────────────────────┼──────────────────────────────────────┘  │
│                       │                                          │
│                       ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │   Attention Computation (scaled dot-product)            │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Model Output                                │
│                   Generated Text                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Position Encoding Detail

```
Input Sequence:
┌──────┬──────┬──────┬────────┬──────┬──────┬──────┬─────────┬────────┬──────┬──────┬─────────┬──────┬──────┐
│  A   │  B   │  C   │ <Chunk>│  D   │  E   │  F   │ </Chunk>│ <Chunk>│  G   │  H   │ </Chunk>│  I   │  J   │
└──────┴──────┴──────┴────────┴──────┴──────┴──────┴─────────┴────────┴──────┴──────┴─────────┴──────┴──────┘

Position Computation:
┌──────┬──────┬──────┬────────┬──────┬──────┬──────┬─────────┬────────┬──────┬──────┬─────────┬──────┬──────┐
│  0   │  1   │  2   │   -1   │  3   │  4   │  5   │   -1    │   -1   │  3   │  4   │   -1    │  7   │  8   │
└──────┴──────┴──────┴────────┴──────┴──────┴──────┴─────────┴────────┴──────┴──────┴─────────┴──────┴──────┘

Encoding Type:
┌──────┬──────┬──────┬────────┬──────┬──────┬──────┬─────────┬────────┬──────┬──────┬─────────┬──────┬──────┐
│Normal│Normal│Normal│  SKIP  │Chunk0│Chunk0│Chunk0│  SKIP   │  SKIP  │Chunk1│Chunk1│  SKIP   │Normal│Normal│
└──────┴──────┴──────┴────────┴──────┴──────┴──────┴─────────┴────────┴──────┴──────┴─────────┴──────┴──────┘

After Filtering (positions >= 0):
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  A   │  B   │  C   │  D   │  E   │  F   │  G   │  H   │  I   │  J   │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  0   │  1   │  2   │  3   │  4   │  5   │  3   │  4   │  7   │  8   │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

## Partitioned Encoding Matrix

```
For tokens inside chunks (e.g., token D in chunk 0):

Rotary Dimension (128):
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌───────────────────────────────┐  ┌───────────────────────┐  │
│  │      First 80% (dim 0-102)    │  │  Last 20% (dim 103-127)│ │
│  │                                │  │                        │ │
│  │  Standard RoPE Encoding       │  │  Custom Encoding       │ │
│  │  Uses: Local Position (3)     │  │  Uses: Block Index (0) │ │
│  │                                │  │                        │ │
│  │  cos(pos/θ^(2i/d))            │  │  sin(2πijB/n)/√n      │ │
│  │  sin(pos/θ^(2i/d))            │  │  where B=0            │ │
│  │                                │  │                        │ │
│  └───────────────────────────────┘  └───────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

For tokens outside chunks (e.g., token A):

Rotary Dimension (128):
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │             Full Standard RoPE (dim 0-127)                  │ │
│  │                                                              │ │
│  │  Uses: Absolute Position (0)                                │ │
│  │                                                              │ │
│  │  cos(pos/θ^(2i/d))                                          │ │
│  │  sin(pos/θ^(2i/d))                                          │ │
│  │                                                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Through Attention

```
Token: D (inside chunk 0, local position 3)

┌────────────────────────────────────────┐
│  Hidden State h_D                      │
│  [batch, seq, hidden_dim]              │
└────────────┬───────────────────────────┘
             │
      ┌──────┴──────┬──────────┐
      │             │          │
      ▼             ▼          ▼
   ┌────┐       ┌────┐     ┌────┐
   │ Q  │       │ K  │     │ V  │
   └─┬──┘       └─┬──┘     └────┘
     │            │
     │  Reshape   │
     │            │
     ▼            ▼
   ┌────────────────────┐
   │ [batch, seq,       │
   │  num_heads,        │
   │  head_dim]         │
   └─────┬──────────────┘
         │
         │  Apply ChunkedRotaryEmbedding
         │
         ▼
   ┌─────────────────────────────────┐
   │  For head_dim dimensions:       │
   │                                  │
   │  Dims 0-102 (80%):              │
   │    cos_rope = cos(3/θ^(2i/102))│
   │    sin_rope = sin(3/θ^(2i/102))│
   │    → Standard rotation          │
   │                                  │
   │  Dims 103-127 (20%):            │
   │    cos_custom = sin(2πij·0/25) │
   │    sin_custom = 0               │
   │    → Custom rotation            │
   └─────┬───────────────────────────┘
         │
         ▼
   ┌────────────────────┐
   │  Rotated Q, K      │
   └────────────────────┘
```

## File Organization

```
vllm-releases-v0.12.0/
│
├── vllm/model_executor/layers/rotary_embedding/
│   ├── base.py                    # Base RoPE classes
│   ├── partitioned_rope.py        # Original partitioned implementation
│   └── chunked_rope.py            # ★ NEW: Chunked encoding
│
├── test_chunked_rope.py           # ★ NEW: Test script
├── serve_chunked_llm.py           # ★ NEW: Serving script
├── example_model_integration.py   # ★ NEW: Integration example
│
├── CHUNKED_ROPE_README.md         # ★ NEW: Detailed docs
├── SERVING_WITH_CHUNKED_ROPE.md   # ★ NEW: Serving guide
├── QUICK_START_CHUNKED_ROPE.md    # ★ NEW: Quick reference
├── IMPLEMENTATION_SUMMARY.md      # ★ NEW: Summary
└── ARCHITECTURE_DIAGRAM.md        # ★ NEW: This file
```

## Integration Points

```
┌─────────────────────────────────────────────────────────────┐
│                    Your LLM Model                            │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Tokenizer                                             │ │
│  │  • Add <Chunk> and </Chunk> special tokens            │ │
│  │  • Note token IDs for position computation            │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Config (config.json)                                  │ │
│  │  • use_chunked_rope: true                             │ │
│  │  • chunk_start_token_id: 128256                       │ │
│  │  • chunk_end_token_id: 128257                         │ │
│  │  • chunk_partition_ratio: 0.8                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Model Initialization                                  │ │
│  │  • Load ChunkedRotaryEmbedding instead of RoPE        │ │
│  │  • Store chunk token IDs                              │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Preprocessing (before forward)                        │ │
│  │  • Compute positions from chunks                      │ │
│  │  • Filter separator tokens                            │ │
│  │  • Pass is_chunked, block_indices to forward          │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Attention Forward                                     │ │
│  │  • Apply chunked RoPE with metadata                   │ │
│  │  • Use partitioned encoding for chunked tokens        │ │
│  │  • Use normal RoPE for others                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Performance Considerations

```
┌──────────────────────────────────────────────────────────────┐
│               Computational Overhead                          │
│                                                                │
│  Standard RoPE:                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  O(seq_len * rotary_dim)                             │    │
│  │  • Cached cos/sin lookup                             │    │
│  │  • Simple indexing                                   │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  Chunked RoPE:                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  O(seq_len * rotary_dim) + O(position_computation)  │    │
│  │  • Position computation: O(seq_len)                  │    │
│  │  • Separator filtering: O(seq_len)                   │    │
│  │  • Chunked cos/sin: O(chunked_tokens * rotary_dim)  │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  Optimization Opportunities:                                  │
│  • Cache position computations for repeated patterns          │
│  • Batch process sequences with similar chunk structure       │
│  • CUDA kernels for position computation                      │
│  • Precompute custom encodings for common block indices       │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

This diagram provides a visual overview of the chunked rotary encoding system architecture, data flow, and integration points.
