# Deployment Checklist: Chunked Rotary Encoding

Use this checklist to deploy your LLM with chunked rotary encoding support.

## ✅ Phase 1: Setup & Verification (15-30 min)

### 1.1 Verify Files
- [ ] `vllm/model_executor/layers/rotary_embedding/chunked_rope.py` exists
- [ ] `test_chunked_rope.py` exists in workspace root
- [ ] All documentation files are present

### 1.2 Test Core Implementation
```bash
cd /Users/zzy/Downloads/vllm-releases-v0.12.0
python test_chunked_rope.py
```

Expected output:
- [ ] ✓ Requirements 0-4 verified
- [ ] ✓ Position encoding correct
- [ ] ✓ Separator tokens excluded
- [ ] ✓ No errors or exceptions

### 1.3 Review Documentation
- [ ] Read `QUICK_START_CHUNKED_ROPE.md`
- [ ] Understand position encoding rules
- [ ] Review example use cases

## ✅ Phase 2: Tokenizer Preparation (30-60 min)

### 2.1 Add Chunk Tokens
```python
from transformers import AutoTokenizer

# Load your tokenizer
tokenizer = AutoTokenizer.from_pretrained("YOUR_MODEL_NAME")

# Add special tokens
special_tokens = {
    'additional_special_tokens': ['<Chunk>', '</Chunk>']
}
num_added = tokenizer.add_special_tokens(special_tokens)

print(f"Added {num_added} special tokens")

# Save
tokenizer.save_pretrained("YOUR_MODEL_NAME_chunked")
```

- [ ] Tokenizer saved with chunk markers
- [ ] Verified tokens were added (num_added == 2)

### 2.2 Get Token IDs
```python
CHUNK_START_ID = tokenizer.convert_tokens_to_ids('<Chunk>')
CHUNK_END_ID = tokenizer.convert_tokens_to_ids('</Chunk>')

print(f"<Chunk> token ID: {CHUNK_START_ID}")
print(f"</Chunk> token ID: {CHUNK_END_ID}")

# Verify they're not UNK token
assert CHUNK_START_ID != tokenizer.unk_token_id
assert CHUNK_END_ID != tokenizer.unk_token_id
```

- [ ] Chunk start token ID: __________
- [ ] Chunk end token ID: __________
- [ ] Both IDs are valid (not UNK token)

### 2.3 Test Tokenization
```python
test_text = "Before <Chunk>Content</Chunk> After"
tokens = tokenizer.encode(test_text)
decoded = tokenizer.decode(tokens)

print(f"Original: {test_text}")
print(f"Tokens: {tokens}")
print(f"Decoded: {decoded}")
```

- [ ] Chunk markers appear in token list
- [ ] Decoding preserves chunk markers

## ✅ Phase 3: Model Configuration (15 min)

### 3.1 Update config.json
Edit your model's `config.json`:

```json
{
  "model_type": "llama",  // your model type
  // ... existing config ...
  
  // Add these lines:
  "use_chunked_rope": true,
  "chunk_start_token_id": YOUR_CHUNK_START_ID,
  "chunk_end_token_id": YOUR_CHUNK_END_ID,
  "chunk_partition_ratio": 0.8
}
```

- [ ] Added `use_chunked_rope: true`
- [ ] Added `chunk_start_token_id` with correct ID
- [ ] Added `chunk_end_token_id` with correct ID
- [ ] Added `chunk_partition_ratio: 0.8`
- [ ] Saved config.json

### 3.2 Resize Token Embeddings (if needed)
If you added new tokens, the model needs to resize:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("YOUR_MODEL_NAME")
model.resize_token_embeddings(len(tokenizer))
model.save_pretrained("YOUR_MODEL_NAME_chunked")
```

- [ ] Token embeddings resized (if applicable)
- [ ] Model saved with new embeddings

## ✅ Phase 4: Model Integration (1-2 hours)

Choose one approach:

### Option A: Modify Existing Model (Recommended)

Find your model file (e.g., `vllm/model_executor/models/llama.py` or similar):

#### 4.1 Add Import
```python
from vllm.model_executor.layers.rotary_embedding.chunked_rope import ChunkedRotaryEmbedding
```

- [ ] Import added to model file

#### 4.2 Modify Attention Layer __init__
```python
# In your attention layer (e.g., LlamaAttention.__init__):

if getattr(config, 'use_chunked_rope', False):
    self.rotary_emb = ChunkedRotaryEmbedding(
        head_size=self.head_dim,
        rotary_dim=self.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=getattr(config, 'rope_theta', 10000.0),
        is_neox_style=True,
        dtype=self.dtype,
        chunk_partition_ratio=getattr(config, 'chunk_partition_ratio', 0.8),
    )
    self.chunk_start_token_id = config.chunk_start_token_id
    self.chunk_end_token_id = config.chunk_end_token_id
else:
    # Keep existing RoPE initialization
    self.rotary_emb = get_rope(...)
```

- [ ] Added conditional ChunkedRotaryEmbedding initialization
- [ ] Stored chunk token IDs in attention layer
- [ ] Preserved backward compatibility (else clause)

#### 4.3 Update Forward Pass
```python
# In attention forward method, update RoPE application:

if hasattr(self, 'chunk_start_token_id') and self.chunk_start_token_id is not None:
    # Extract chunk metadata from input_metadata if available
    # This part depends on your model architecture
    is_chunked = getattr(attn_metadata, 'is_chunked', None)
    block_indices = getattr(attn_metadata, 'block_indices', None)
    
    if is_chunked is not None:
        q, k = self.rotary_emb.forward(
            positions=positions,
            query=q,
            key=k,
            is_chunked=is_chunked,
            block_indices=block_indices,
        )
    else:
        q, k = self.rotary_emb(positions, q, k)
else:
    # Standard RoPE
    q, k = self.rotary_emb(positions, q, k)
```

- [ ] Updated forward pass to use chunked encoding
- [ ] Added fallback for standard RoPE
- [ ] Tested compilation (no syntax errors)

### Option B: Use Example Integration

- [ ] Copy `example_model_integration.py` structure
- [ ] Adapt to your specific model architecture
- [ ] Create new model variant file

## ✅ Phase 5: Testing (1-2 hours)

### 5.1 Unit Tests
```bash
# Test the chunked rope implementation
python test_chunked_rope.py
```

- [ ] All tests pass
- [ ] Position computation correct
- [ ] Encoding verified

### 5.2 Integration Test
Create a simple test script:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="YOUR_MODEL_NAME_chunked")

test_prompt = "Test: <Chunk>Chunk 1</Chunk> <Chunk>Chunk 2</Chunk> End"

try:
    outputs = llm.generate([test_prompt], SamplingParams(max_tokens=10))
    print("✓ Model loads and generates")
    print(f"Output: {outputs[0].outputs[0].text}")
except Exception as e:
    print(f"✗ Error: {e}")
```

- [ ] Model loads successfully
- [ ] No errors during generation
- [ ] Output is reasonable

### 5.3 Position Verification
Add debug logging to verify positions:

```python
# In your model forward pass, add temporary logging:
print(f"Positions: {positions}")
print(f"Is chunked: {is_chunked}")
print(f"Block indices: {block_indices}")
```

- [ ] Positions match expected values
- [ ] Chunked mask is correct
- [ ] Block indices increment properly

## ✅ Phase 6: Deployment (variable)

### 6.1 Choose Serving Method

#### Option A: Standalone Server
```bash
python serve_chunked_llm.py \
    --model YOUR_MODEL_NAME_chunked \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --interactive
```

- [ ] Server starts without errors
- [ ] Can generate with chunk markers
- [ ] Output quality is good

#### Option B: vLLM API Server
```bash
python -m vllm.entrypoints.openai.api_server \
    --model YOUR_MODEL_NAME_chunked \
    --dtype float16 \
    --port 8000
```

- [ ] API server starts
- [ ] Endpoints respond
- [ ] Can send chunked prompts

#### Option C: Programmatic
```python
from vllm import LLM, SamplingParams

llm = LLM(model="YOUR_MODEL_NAME_chunked")

def generate_with_chunks(prompt):
    return llm.generate([prompt], SamplingParams(temperature=0.7))
```

- [ ] Integrated into your application
- [ ] Works with chunk markers
- [ ] Performance is acceptable

### 6.2 Production Checklist
- [ ] Model loads in target environment
- [ ] Latency is acceptable (< XXms per token)
- [ ] Memory usage is reasonable
- [ ] Error handling in place
- [ ] Logging configured
- [ ] Monitoring set up

## ✅ Phase 7: Validation (ongoing)

### 7.1 Functional Tests
Test with various chunk patterns:

```python
test_cases = [
    # Single chunk
    "<Chunk>Content</Chunk> Question?",
    
    # Multiple chunks
    "<Chunk>Doc1</Chunk> <Chunk>Doc2</Chunk> <Chunk>Doc3</Chunk>",
    
    # No chunks (fallback to normal RoPE)
    "Regular prompt without chunks",
    
    # Mixed content
    "Before <Chunk>Middle</Chunk> After",
    
    # Empty chunks
    "<Chunk></Chunk>",
]

for test in test_cases:
    result = llm.generate([test])
    print(f"Input: {test}")
    print(f"Output: {result[0].outputs[0].text}\n")
```

- [ ] Single chunk works
- [ ] Multiple chunks work
- [ ] No chunks works (fallback)
- [ ] Mixed content works
- [ ] Edge cases handled

### 7.2 Performance Benchmarks
```bash
# Benchmark throughput
python benchmarks/benchmark_throughput.py \
    --model YOUR_MODEL_NAME_chunked \
    --input-len 512 \
    --output-len 128
```

- [ ] Throughput measured: ______ tokens/sec
- [ ] Latency measured: ______ ms/token
- [ ] Compared to baseline (standard RoPE): ______%
- [ ] Performance is acceptable

### 7.3 Quality Assessment
- [ ] Output quality with chunks is good
- [ ] Model understands chunk boundaries
- [ ] No degradation vs. standard RoPE for non-chunked input
- [ ] Use case specific validation passed

## ✅ Phase 8: Documentation (optional but recommended)

### 8.1 Internal Documentation
- [ ] Documented chunk token IDs
- [ ] Noted configuration changes
- [ ] Recorded integration points
- [ ] Created usage examples

### 8.2 User Guide
- [ ] How to format prompts with chunks
- [ ] When to use chunked encoding
- [ ] Performance characteristics
- [ ] Troubleshooting guide

## 📊 Summary

### Environment Info
- Model: _______________________
- vLLM version: _________________
- Chunk start token ID: _________
- Chunk end token ID: ___________
- Partition ratio: 0.8

### Test Results
- [ ] Unit tests: PASS / FAIL
- [ ] Integration tests: PASS / FAIL
- [ ] Performance tests: PASS / FAIL
- [ ] Quality tests: PASS / FAIL

### Deployment Status
- [ ] Development: READY / NOT READY
- [ ] Staging: READY / NOT READY  
- [ ] Production: READY / NOT READY

### Notes
_______________________________________
_______________________________________
_______________________________________

## 🆘 Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| Chunk tokens not found | Add to tokenizer, save, reload |
| Token IDs map to UNK | Use `add_special_tokens()` not `add_tokens()` |
| Position errors | Verify chunk markers are paired |
| Shape mismatches | Filter separators before model forward |
| Performance slow | Profile position computation, consider caching |
| Model fails to load | Check config.json has correct token IDs |
| No output improvement | Verify chunks are being detected (add logging) |

## 📞 Resources

- **Quick Start**: `QUICK_START_CHUNKED_ROPE.md`
- **Full Documentation**: `CHUNKED_ROPE_README.md`
- **Serving Guide**: `SERVING_WITH_CHUNKED_ROPE.md`
- **Architecture**: `ARCHITECTURE_DIAGRAM.md`
- **Summary**: `IMPLEMENTATION_SUMMARY.md`
- **Test Script**: `test_chunked_rope.py`
- **Example Server**: `serve_chunked_llm.py`

---

**Date Completed**: _______________
**Deployed By**: __________________
**Status**: ☐ In Progress  ☐ Completed  ☐ Issues
