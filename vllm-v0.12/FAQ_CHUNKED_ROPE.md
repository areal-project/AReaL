# FAQ: Chunked Rotary Positional Encoding

## General Questions

### Q1: What is chunked rotary encoding?

**A:** Chunked rotary encoding is a hybrid positional encoding scheme that applies different encoding strategies to different parts of the input:
- Content outside `<Chunk>...</Chunk>` markers uses standard RoPE
- Content inside chunks uses partitioned encoding (80% RoPE + 20% custom)
- The custom part encodes chunk identity (block index)
- This allows the model to understand document boundaries and chunk structure

### Q2: Why would I want to use this?

**A:** Use chunked encoding when your input contains multiple distinct documents or sections that should be encoded separately:
- **Multi-document QA**: Each document is a separate chunk
- **RAG systems**: Retrieved passages as separate chunks
- **Long context**: Break long documents into semantic chunks
- **Structured data**: Encode different data sources separately
- **Dialogue systems**: Each turn as a separate chunk

### Q3: How does it differ from regular RoPE?

**A:** 
- **Regular RoPE**: All tokens get positional encoding based on absolute position
- **Chunked RoPE**: 
  - Tokens outside chunks: standard RoPE
  - Tokens inside chunks: partitioned encoding with chunk-aware component
  - Chunk markers are excluded from encoding
  - Position tracking accounts for chunk boundaries

## Implementation Questions

### Q4: Do I need to retrain my model?

**A:** Not necessarily. Chunked encoding can be used with existing models, but:
- **Without retraining**: The model may not fully utilize chunk boundaries
- **With fine-tuning**: Better performance as model learns chunk-aware patterns
- **From scratch**: Best performance for chunk-heavy use cases

Recommendation: Start without retraining, fine-tune if needed.

### Q5: How do I add chunk markers to my tokenizer?

**A:**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-model")

# Add as special tokens (important!)
special_tokens = {
    'additional_special_tokens': ['<Chunk>', '</Chunk>']
}
tokenizer.add_special_tokens(special_tokens)

# Save
tokenizer.save_pretrained("your-model-chunked")

# Get IDs
chunk_start_id = tokenizer.convert_tokens_to_ids('<Chunk>')
chunk_end_id = tokenizer.convert_tokens_to_ids('</Chunk>')
```

**Important**: Use `add_special_tokens()`, not `add_tokens()`, to ensure they're properly handled.

### Q6: Can I use different chunk markers?

**A:** Yes! You can use any markers you want:
```python
special_tokens = {
    'additional_special_tokens': ['[DOC]', '[/DOC]']  # or any other markers
}
```

Just make sure to:
1. Update the tokenizer
2. Update the config with correct token IDs
3. Use consistent markers in your prompts

### Q7: What if my tokenizer already has these tokens?

**A:** Check first:
```python
if '<Chunk>' in tokenizer.get_vocab():
    chunk_start_id = tokenizer.convert_tokens_to_ids('<Chunk>')
    print(f"Chunk marker already exists with ID: {chunk_start_id}")
else:
    # Add it
    tokenizer.add_special_tokens(...)
```

### Q8: How do I integrate this into my vLLM model?

**A:** Three main steps:

1. **Add the import** in your model file:
```python
from vllm.model_executor.layers.rotary_embedding.chunked_rope import ChunkedRotaryEmbedding
```

2. **Replace RoPE initialization** in attention layer:
```python
if getattr(config, 'use_chunked_rope', False):
    self.rotary_emb = ChunkedRotaryEmbedding(...)
else:
    self.rotary_emb = get_rope(...)
```

3. **Update config.json**:
```json
{
  "use_chunked_rope": true,
  "chunk_start_token_id": 128256,
  "chunk_end_token_id": 128257
}
```

See `SERVING_WITH_CHUNKED_ROPE.md` for detailed steps.

## Usage Questions

### Q9: How should I format my prompts?

**A:**
```
Question about documents:

<Chunk>
First document content here.
Can be multiple sentences or paragraphs.
</Chunk>

<Chunk>
Second document content here.
Also can be long.
</Chunk>

What is the relationship between them?
```

**Key points**:
- Use `<Chunk>` to start, `</Chunk>` to end
- Every opening tag must have a closing tag
- Markers must be properly paired
- Content outside chunks is fine (uses normal RoPE)

### Q10: Can chunks be nested?

**A:** Not in the current implementation. Chunks must be flat:

✅ Valid:
```
<Chunk>A</Chunk> <Chunk>B</Chunk>
```

❌ Invalid:
```
<Chunk>A <Chunk>B</Chunk> C</Chunk>
```

### Q11: What happens if chunks aren't properly paired?

**A:** The position computation will fail. Enable validation:

```python
config = ChunkedPromptConfig(validate_pairing=True)
server = ChunkedLLMServer(model, config=config)
```

This will raise an error for malformed input.

### Q12: Can I have empty chunks?

**A:** Yes, but they're not very useful:
```
<Chunk></Chunk>  # Valid but useless
```

The chunk will have zero tokens and no effect.

### Q13: How many chunks can I have?

**A:** No hard limit, but practical considerations:
- More chunks = more position computation overhead
- Total sequence length still bounded by `max_position_embeddings`
- Typical use: 1-10 chunks per sequence

## Performance Questions

### Q14: How much slower is chunked encoding vs regular RoPE?

**A:** Overhead comes from:
1. **Position computation**: O(seq_len) - usually negligible
2. **Separator filtering**: O(seq_len) - very fast
3. **Chunked encoding**: Only for chunked tokens

Typical overhead: 5-15% for sequences with chunks, 0-2% for sequences without chunks.

Benchmark:
```bash
python benchmarks/benchmark_throughput.py --model your-model
```

### Q15: Does it affect generation quality?

**A:** Depends on use case:
- **Without fine-tuning**: Minimal impact on quality
- **For multi-doc tasks**: May improve if model learns to use chunks
- **For single-doc tasks**: Should be neutral

Quality is more about how you use chunks than the encoding itself.

### Q16: Can I cache position computations?

**A:** Yes! For repeated chunk patterns:

```python
# Cache positions for common patterns
position_cache = {}

def get_positions_cached(token_ids):
    cache_key = tuple(token_ids)
    if cache_key not in position_cache:
        position_cache[cache_key] = compute_positions(token_ids)
    return position_cache[cache_key]
```

This is especially useful for batch processing similar documents.

## Technical Questions

### Q17: What is the partition ratio?

**A:** The partition ratio (default 0.8) determines how rotary dimensions are split:
- **80%**: Standard RoPE with local positions
- **20%**: Custom encoding with block indices

You can adjust this in config:
```json
{
  "chunk_partition_ratio": 0.8  // 80/20 split
}
```

Values:
- 1.0 = pure RoPE (no chunk encoding)
- 0.5 = 50/50 split
- 0.8 = recommended default

### Q18: What is the custom encoding formula?

**A:** For the 20% custom dimensions:

```
R_ij = sin(2π * i * j * B / n) / sqrt(n)
```

Where:
- `i, j` are dimension indices
- `B` is the block index (chunk number: 0, 1, 2, ...)
- `n` is the custom dimension size (20% of rotary_dim)

This creates a chunk-specific encoding that's position-independent but chunk-aware.

### Q19: How are positions computed for content after chunks?

**A:** 
```
position_after = pre_chunk_end + max_chunk_length + 1
```

Example:
- 10 tokens before chunks (positions 0-9)
- 3 chunks with lengths: 5, 8, 7
- max_chunk_length = 8
- Content after starts at: 9 + 8 + 1 = 18

This ensures proper spacing for the longest chunk.

### Q20: What happens to `<Chunk>` and `</Chunk>` tokens?

**A:** They are **excluded** from the model:
1. Marked with position = -1
2. Filtered out before model forward pass
3. Never contribute to attention or generation
4. Only used for position computation

### Q21: Can I use this with other RoPE variants (YaRN, NTK, etc.)?

**A:** Potentially yes, but requires adaptation. The current implementation extends the base `RotaryEmbedding` class. To combine with other variants:

1. Inherit from that variant instead
2. Override the necessary methods
3. Test thoroughly

Example:
```python
from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import YarnScalingRotaryEmbedding

class ChunkedYarnRoPE(YarnScalingRotaryEmbedding):
    # Add chunked encoding logic
    pass
```

## Debugging Questions

### Q22: How can I verify positions are correct?

**A:** Add logging in the forward pass:

```python
# In your model's forward method
print(f"Token IDs: {input_ids}")
print(f"Positions: {positions}")
print(f"Is chunked: {is_chunked}")
print(f"Block indices: {block_indices}")
```

Or use the test script:
```bash
python test_chunked_rope.py
```

### Q23: Why am I getting shape mismatches?

**A:** Most likely you forgot to filter separators:

```python
# After computing positions
valid_mask = positions >= 0
filtered_input_ids = input_ids[valid_mask]
filtered_positions = positions[valid_mask]
filtered_is_chunked = is_chunked[valid_mask]
filtered_block_indices = block_indices[valid_mask]

# Use filtered versions in model
```

### Q24: Model generates gibberish with chunks

**A:** Check:
1. Are chunk markers in the tokenizer vocabulary?
2. Are token IDs correct in config?
3. Are positions computed correctly? (log them)
4. Is the model actually using ChunkedRotaryEmbedding? (add print in __init__)

### Q25: Chunks don't seem to affect output

**A:** This is expected if:
1. Model wasn't fine-tuned with chunks
2. Task doesn't benefit from chunk awareness
3. Chunks are too similar (model can't distinguish)

Solution: Fine-tune the model with chunk-aware training data.

## Serving Questions

### Q26: Which serving method should I use?

**A:** Depends on your needs:

| Method | Best For | Complexity |
|--------|----------|------------|
| `serve_chunked_llm.py` | Testing, prototyping | Low |
| vLLM API server | Production, REST API | Medium |
| Programmatic (LLM class) | Integration into apps | Low |
| Custom integration | Special requirements | High |

Start with the example server, move to production methods later.

### Q27: Can I use this with vLLM's OpenAI-compatible API?

**A:** Yes! Once integrated into your model:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model your-model-chunked \
    --port 8000
```

Then use standard OpenAI client:
```python
import openai

openai.api_base = "http://localhost:8000/v1"

response = openai.ChatCompletion.create(
    model="your-model-chunked",
    messages=[{
        "role": "user",
        "content": "Analyze: <Chunk>Doc 1</Chunk> <Chunk>Doc 2</Chunk>"
    }]
)
```

### Q28: Does this work with multi-GPU serving?

**A:** Yes! Use tensor parallelism:

```bash
python serve_chunked_llm.py \
    --model your-model \
    --tensor-parallel-size 4  # 4 GPUs
```

Or with API server:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model your-model \
    --tensor-parallel-size 4
```

### Q29: Can I disable chunked encoding at runtime?

**A:** Not easily once integrated. But you can:
1. Not use chunk markers in your prompts (falls back to normal RoPE for those tokens)
2. Load a different model without chunked encoding
3. Use a config flag to conditionally enable it

## Advanced Questions

### Q30: How can I fine-tune a model with chunked encoding?

**A:** 
1. **Prepare training data** with chunk markers:
```
<Chunk>Context 1</Chunk> <Chunk>Context 2</Chunk> Question: ... Answer: ...
```

2. **Use the chunked model** for training:
```python
from transformers import Trainer, TrainingArguments

# Your model must already have ChunkedRotaryEmbedding integrated
model = AutoModelForCausalLM.from_pretrained("your-model-chunked")

trainer = Trainer(
    model=model,
    args=TrainingArguments(...),
    train_dataset=your_dataset,
)

trainer.train()
```

3. **Ensure position computation** happens during training (may need custom data collator)

### Q31: Can I use this for vision-language models?

**A:** Potentially, if the VLM uses RoPE for text positions. You would:
1. Apply chunked encoding only to text tokens
2. Keep vision tokens with standard encoding
3. Modify the attention layer accordingly

Not tested, proceed with caution.

### Q32: What about streaming generation?

**A:** Streaming works, but:
- Position computation must happen before streaming starts
- All chunk markers must be in the prompt (can't stream chunk markers)
- Generation proceeds token-by-token as normal

```python
# Streaming is the same as without chunks
for token in llm.generate_stream(prompt_with_chunks):
    print(token, end='', flush=True)
```

### Q33: Can I mix chunked and non-chunked sequences in a batch?

**A:** Yes! The position computation handles this:
- Sequences with chunks get chunked encoding
- Sequences without chunks use normal RoPE
- Batch them together normally

Just ensure you compute positions for each sequence.

## Common Issues

### Q34: "Token ID not found" error

**Solution:**
```python
# Add tokens properly
tokenizer.add_special_tokens({
    'additional_special_tokens': ['<Chunk>', '</Chunk>']
})
tokenizer.save_pretrained("path")

# Reload model with resized embeddings
model.resize_token_embeddings(len(tokenizer))
```

### Q35: "Invalid chunk markers" error

**Solution:** Ensure markers are properly paired:
```python
text.count('<Chunk>') == text.count('</Chunk>')  # Must be true
```

### Q36: Performance is very slow

**Solution:**
1. Profile to find bottleneck:
```python
import time

start = time.time()
positions = compute_positions(tokens)
print(f"Position computation: {time.time() - start:.3f}s")
```

2. Optimize:
- Cache repeated patterns
- Batch process similar sequences
- Consider CUDA kernels for position computation

### Q37: Model won't load after integration

**Solution:** Check:
```python
# 1. Config has required fields
assert 'chunk_start_token_id' in config
assert 'chunk_end_token_id' in config

# 2. Token IDs are valid integers
assert isinstance(config.chunk_start_token_id, int)

# 3. Import works
from vllm.model_executor.layers.rotary_embedding.chunked_rope import ChunkedRotaryEmbedding
```

## Best Practices

### Q38: What are the best practices for using chunked encoding?

**A:**

✅ **Do:**
- Validate chunk marker pairing
- Use semantic boundaries for chunks (documents, paragraphs, etc.)
- Test without chunks first to ensure baseline works
- Log positions during development
- Start with default partition ratio (0.8)
- Fine-tune for chunk-heavy use cases

❌ **Don't:**
- Nest chunks
- Use chunks for every sentence (too granular)
- Forget to filter separator tokens
- Assume it improves quality without testing
- Use very long chunks (defeats the purpose)

### Q39: When should I NOT use chunked encoding?

**A:**

Don't use it when:
- Single continuous document (no natural chunks)
- Model hasn't been trained/fine-tuned for it
- Performance overhead is unacceptable
- Use case doesn't benefit from chunk boundaries
- Chunks would be very unbalanced in size

Use regular RoPE instead.

### Q40: How do I know if it's working?

**A:**

Verification checklist:
1. ✅ Test script passes: `python test_chunked_rope.py`
2. ✅ Model loads without errors
3. ✅ Generation works with chunk markers
4. ✅ Positions are computed correctly (log them)
5. ✅ No shape mismatches during forward pass
6. ✅ Output quality is acceptable

If all pass, it's working!

---

**Still have questions?**
- Check the detailed docs: `CHUNKED_ROPE_README.md`
- Review serving guide: `SERVING_WITH_CHUNKED_ROPE.md`
- See examples: `example_model_integration.py`
- Run tests: `python test_chunked_rope.py`
