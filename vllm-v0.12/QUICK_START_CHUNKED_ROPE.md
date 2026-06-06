# Quick Start Guide: Serving LLMs with Chunked Rotary Encoding

## 📋 Summary

This package provides a custom positional encoding scheme for LLMs that handles chunked content with special markers `<Chunk>` and `</Chunk>`. It enables:
- Normal RoPE for content outside chunks
- Partitioned encoding for content inside chunks (80% RoPE + 20% custom)
- Proper position tracking across chunk boundaries

## 🚀 Quick Start

### 1. Files Overview

- **`chunked_rope.py`**: Core implementation of ChunkedRotaryEmbedding
- **`test_chunked_rope.py`**: Test script to verify functionality
- **`serve_chunked_llm.py`**: Example serving script
- **`example_model_integration.py`**: How to integrate into a model
- **`CHUNKED_ROPE_README.md`**: Detailed documentation
- **`SERVING_WITH_CHUNKED_ROPE.md`**: Complete serving guide

### 2. Installation

```bash
# Ensure you have vLLM installed
pip install vllm torch transformers

# Or install from source
cd /path/to/vllm
pip install -e .
```

### 3. Add Chunk Tokens to Tokenizer

```python
from transformers import AutoTokenizer

# Load your tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-model-name")

# Add chunk markers as special tokens
special_tokens = {'additional_special_tokens': ['<Chunk>', '</Chunk>']}
tokenizer.add_special_tokens(special_tokens)

# Save
tokenizer.save_pretrained("your-model-chunked")

# Get token IDs (you'll need these)
chunk_start_id = tokenizer.convert_tokens_to_ids('<Chunk>')
chunk_end_id = tokenizer.convert_tokens_to_ids('</Chunk>')
print(f"<Chunk> ID: {chunk_start_id}")
print(f"</Chunk> ID: {chunk_end_id}")
```

### 4. Update Model Configuration

Add to your model's `config.json`:

```json
{
  "use_chunked_rope": true,
  "chunk_start_token_id": 128256,
  "chunk_end_token_id": 128257,
  "chunk_partition_ratio": 0.8
}
```

### 5. Integrate into Model

**Option A: Modify existing model**

Edit `vllm/model_executor/models/your_model.py`:

```python
from vllm.model_executor.layers.rotary_embedding.chunked_rope import ChunkedRotaryEmbedding

# In your attention layer __init__:
if getattr(config, 'use_chunked_rope', False):
    self.rotary_emb = ChunkedRotaryEmbedding(
        head_size=self.head_dim,
        rotary_dim=self.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
        is_neox_style=True,
        dtype=self.dtype,
        chunk_partition_ratio=getattr(config, 'chunk_partition_ratio', 0.8),
    )
else:
    # Standard RoPE
    self.rotary_emb = get_rope(...)
```

**Option B: Use the example integration**

Copy and adapt `example_model_integration.py` to your model structure.

### 6. Serve the Model

**Method 1: Using the example server**

```bash
python serve_chunked_llm.py \
    --model your-model-chunked \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --interactive
```

**Method 2: Using standard vLLM API**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model your-model-chunked \
    --dtype float16 \
    --port 8000
```

**Method 3: Programmatic**

```python
from vllm import LLM, SamplingParams

llm = LLM(model="your-model-chunked")

prompt = """
Analyze these documents:
<Chunk>Document 1: Content here</Chunk>
<Chunk>Document 2: Content here</Chunk>
What are the key points?
"""

outputs = llm.generate([prompt], SamplingParams(temperature=0.7))
print(outputs[0].outputs[0].text)
```

### 7. Test the Implementation

```bash
# Run the test script
python test_chunked_rope.py

# Expected output:
# ✓ Requirements verified
# ✓ Position encoding correct
# ✓ Separator tokens excluded
```

## 📝 Usage Examples

### Example 1: Document Analysis

```python
prompt = """
Compare the following research papers:

<Chunk>
Paper 1: "Attention Is All You Need" introduced the Transformer architecture.
Key contributions: self-attention mechanism, positional encoding.
</Chunk>

<Chunk>
Paper 2: "BERT" showed bidirectional training improves language understanding.
Key contributions: masked language modeling, pre-training approach.
</Chunk>

Summarize the relationship between these papers.
"""
```

### Example 2: Multi-Document QA

```python
prompt = """
Question: What are the main differences between renewable and non-renewable energy?

Context:
<Chunk>
Renewable Energy: Solar, wind, and hydro power are renewable sources.
They regenerate naturally and have minimal environmental impact.
</Chunk>

<Chunk>
Non-Renewable Energy: Coal, oil, and natural gas are finite resources.
They contribute to pollution and climate change.
</Chunk>

<Chunk>
Economic Considerations: Initial costs for renewable energy are higher,
but operational costs are lower over time.
</Chunk>

Answer:
"""
```

### Example 3: Code Analysis

```python
prompt = """
Review these code snippets:

<Chunk>
# Function 1: Bubble Sort
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
</Chunk>

<Chunk>
# Function 2: Quick Sort
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
</Chunk>

Compare time complexity and when to use each.
"""
```

## 🔍 How It Works

### Position Encoding Scheme

```
Input: "Before <Chunk>Chunk1</Chunk> <Chunk>Chunk2</Chunk> After"
       [tok1 tok2 <C> tok3 tok4 </C> <C> tok5 tok6 tok7 </C> tok8 tok9]

Positions:
- tok1, tok2: Normal RoPE at positions 0, 1
- <C>: EXCLUDED (separator)
- tok3, tok4: Partitioned at positions 2, 3 (block_index=0)
- </C>: EXCLUDED
- <C>: EXCLUDED  
- tok5, tok6, tok7: Partitioned at positions 2, 3, 4 (block_index=1)
- </C>: EXCLUDED
- tok8, tok9: Normal RoPE at positions 6, 7 (= 2 + max_chunk_len + 1)
```

### Partitioned Encoding

For tokens inside chunks:
- **First 80% of dimensions**: Standard RoPE using local position
- **Last 20% of dimensions**: Custom encoding using block index

Formula: `R_ij = sin(2π * i * j * B / n) / sqrt(n)`
where B = block_index (chunk number)

## ⚙️ Configuration Options

### ChunkedRotaryEmbedding Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `head_size` | int | - | Size of attention head |
| `rotary_dim` | int | - | Dimension for rotary encoding |
| `max_position_embeddings` | int | - | Maximum sequence length |
| `base` | float | 10000.0 | Base for frequency computation |
| `is_neox_style` | bool | True | Use NeoX-style rotation |
| `dtype` | torch.dtype | - | Data type for embeddings |
| `chunk_partition_ratio` | float | 0.8 | Ratio for partitioning (0.8 = 80% RoPE) |

### Model Configuration

Add to `config.json`:

```json
{
  "use_chunked_rope": true,
  "chunk_start_token_id": 128256,
  "chunk_end_token_id": 128257,
  "chunk_partition_ratio": 0.8
}
```

## 🐛 Troubleshooting

### Issue: Chunk tokens not recognized
**Solution**: Ensure tokens are added to tokenizer vocabulary
```python
tokenizer.add_special_tokens({'additional_special_tokens': ['<Chunk>', '</Chunk>']})
tokenizer.save_pretrained("path")
```

### Issue: Position encoding errors
**Solution**: Verify chunk markers are properly paired
```python
# Check pairing
assert text.count('<Chunk>') == text.count('</Chunk>')
```

### Issue: Shape mismatches
**Solution**: Filter out separator tokens before model forward pass
```python
valid_mask = positions >= 0
filtered_tokens = tokens[valid_mask]
```

### Issue: Performance degradation
**Solution**: 
- Profile position computation overhead
- Consider caching for repeated patterns
- Use batch processing

## 📚 Additional Resources

- **Full Documentation**: See `CHUNKED_ROPE_README.md`
- **Serving Guide**: See `SERVING_WITH_CHUNKED_ROPE.md`
- **Test Examples**: Run `test_chunked_rope.py`
- **Integration Example**: See `example_model_integration.py`

## 🔗 Next Steps

1. **Test locally**: Run `python test_chunked_rope.py`
2. **Add tokens**: Update your tokenizer with chunk markers
3. **Modify model**: Integrate ChunkedRotaryEmbedding into attention
4. **Update config**: Add chunked encoding parameters
5. **Serve**: Use `serve_chunked_llm.py` or standard vLLM API
6. **Monitor**: Check performance and position encoding correctness

## 💡 Tips

- **Start simple**: Test with small sequences first
- **Validate positions**: Always log and check position arrays
- **Profile performance**: Compare latency with standard RoPE
- **Use validation**: Enable chunk marker pairing validation
- **Test edge cases**: Empty chunks, nested markers, long sequences

## 📞 Support

For issues or questions:
1. Check the test script output for validation
2. Review the detailed documentation in `CHUNKED_ROPE_README.md`
3. Examine the serving guide in `SERVING_WITH_CHUNKED_ROPE.md`
4. Run the example integration to verify setup

---

**Ready to serve your LLM with chunked encoding!** 🚀
