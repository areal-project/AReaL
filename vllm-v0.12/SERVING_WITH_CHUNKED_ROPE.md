# Serving LLMs with Chunked Rotary Encoding

This guide explains how to serve an LLM using the ChunkedRotaryEmbedding in vLLM.

## Overview

To use chunked encoding when serving, you need to:
1. Modify the model to use ChunkedRotaryEmbedding
2. Add chunk marker tokens to the tokenizer vocabulary
3. Create a custom input processor to handle chunk markers
4. Configure the serving endpoint to preprocess inputs
5. Update the model configuration

## Step-by-Step Integration

### Step 1: Register Chunk Tokens in Tokenizer

First, add special tokens for chunk markers to your tokenizer:

```python
# Example: Extending a tokenizer with chunk markers
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-model-name")

# Add special tokens
special_tokens_dict = {
    'additional_special_tokens': ['<Chunk>', '</Chunk>']
}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

# Save the updated tokenizer
tokenizer.save_pretrained("your-model-name-chunked")

# Get token IDs for later use
CHUNK_START_TOKEN_ID = tokenizer.convert_tokens_to_ids('<Chunk>')
CHUNK_END_TOKEN_ID = tokenizer.convert_tokens_to_ids('</Chunk>')
print(f"Chunk start token ID: {CHUNK_START_TOKEN_ID}")
print(f"Chunk end token ID: {CHUNK_END_TOKEN_ID}")
```

### Step 2: Modify Model to Use ChunkedRotaryEmbedding

You need to modify the model's attention mechanism. Here's how:

#### Option A: Modify Existing Model File

Find your model's attention implementation (usually in `vllm/model_executor/models/your_model.py`):

```python
# Before (standard RoPE):
from vllm.model_executor.layers.rotary_embedding import get_rope

class YourModelAttention(nn.Module):
    def __init__(self, config, ...):
        # ...
        self.rotary_emb = get_rope(
            head_size=self.head_size,
            rotary_dim=self.rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=True,
        )
```

```python
# After (chunked RoPE):
from vllm.model_executor.layers.rotary_embedding.chunked_rope import ChunkedRotaryEmbedding

class YourModelAttention(nn.Module):
    def __init__(self, config, ...):
        # ...
        self.rotary_emb = ChunkedRotaryEmbedding(
            head_size=self.head_size,
            rotary_dim=self.rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=True,
            dtype=self.dtype,
            chunk_partition_ratio=0.8,  # 80% RoPE, 20% custom
        )
        
        # Store chunk token IDs from config
        self.chunk_start_token_id = getattr(config, 'chunk_start_token_id', None)
        self.chunk_end_token_id = getattr(config, 'chunk_end_token_id', None)
```

#### Option B: Create a New Model Variant

Create a new file like `vllm/model_executor/models/your_model_chunked.py`:

```python
from vllm.model_executor.models.your_model import YourModel, YourModelAttention
from vllm.model_executor.layers.rotary_embedding.chunked_rope import ChunkedRotaryEmbedding

class ChunkedYourModelAttention(YourModelAttention):
    """Attention layer with chunked rotary embedding support."""
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        # Replace rotary embedding with chunked version
        self.rotary_emb = ChunkedRotaryEmbedding(
            head_size=self.head_size,
            rotary_dim=self.rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, 'rope_theta', 10000.0),
            is_neox_style=True,
            dtype=self.dtype,
            chunk_partition_ratio=getattr(config, 'chunk_partition_ratio', 0.8),
        )
        
        self.chunk_start_token_id = getattr(config, 'chunk_start_token_id', None)
        self.chunk_end_token_id = getattr(config, 'chunk_end_token_id', None)

class ChunkedYourModel(YourModel):
    """Model variant with chunked rotary embedding."""
    pass
```

### Step 3: Create Input Processor

Create a custom input processor to handle chunk-aware position computation:

```python
# File: vllm/model_executor/models/chunked_input_processor.py

from typing import List, Optional, Tuple
import torch
from vllm.sequence import SequenceData

class ChunkedInputProcessor:
    """Processor for handling chunked inputs with special position encoding."""
    
    def __init__(self, chunk_start_token_id: int, chunk_end_token_id: int):
        self.chunk_start_token_id = chunk_start_token_id
        self.chunk_end_token_id = chunk_end_token_id
    
    def process_sequence(
        self,
        token_ids: List[int],
    ) -> Tuple[List[int], List[int], List[bool], List[int]]:
        """Process a sequence with chunk markers.
        
        Args:
            token_ids: Input token IDs including chunk markers
            
        Returns:
            Tuple of:
            - filtered_token_ids: Token IDs without chunk markers
            - positions: Position indices for each token
            - is_chunked: Boolean mask for chunked tokens
            - block_indices: Block index for each token
        """
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        
        # Import here to avoid circular dependency
        from vllm.model_executor.layers.rotary_embedding.chunked_rope import ChunkedRotaryEmbedding
        
        # Create a temporary instance just for position computation
        # (This is not efficient - ideally this should be done once)
        temp_rope = ChunkedRotaryEmbedding(
            head_size=128,  # Dummy values
            rotary_dim=128,
            max_position_embeddings=32768,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float16,
        )
        
        positions, is_chunked, block_indices = temp_rope.compute_positions_from_chunks(
            token_ids_tensor,
            self.chunk_start_token_id,
            self.chunk_end_token_id,
        )
        
        # Filter out separators
        valid_mask = positions >= 0
        filtered_token_ids = token_ids_tensor[valid_mask].tolist()
        filtered_positions = positions[valid_mask].tolist()
        filtered_is_chunked = is_chunked[valid_mask].tolist()
        filtered_block_indices = block_indices[valid_mask].tolist()
        
        return (
            filtered_token_ids,
            filtered_positions,
            filtered_is_chunked,
            filtered_block_indices,
        )
```

### Step 4: Modify Forward Pass

Update the model's forward pass to use chunked positions:

```python
# In your attention forward method:

def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
    # Add these new parameters:
    is_chunked: Optional[torch.Tensor] = None,
    block_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # ... existing code for Q, K, V projection ...
    
    # Apply rotary embedding with chunked support
    if isinstance(self.rotary_emb, ChunkedRotaryEmbedding) and is_chunked is not None:
        q, k = self.rotary_emb.forward(
            positions=positions,
            query=q,
            key=k,
            is_chunked=is_chunked,
            block_indices=block_indices,
        )
    else:
        # Fallback to standard RoPE
        q, k = self.rotary_emb(positions, q, k)
    
    # ... rest of attention computation ...
```

### Step 5: Update Model Configuration

Add chunk configuration to your model's config file:

```json
// config.json
{
  "model_type": "your_model",
  "chunk_start_token_id": 128256,  // Update with actual token ID
  "chunk_end_token_id": 128257,    // Update with actual token ID
  "chunk_partition_ratio": 0.8,
  "use_chunked_rope": true,
  // ... other config parameters ...
}
```

### Step 6: Create Serving Script

Create a serving script that handles chunk preprocessing:

```python
# serve_chunked_model.py

from vllm import LLM, SamplingParams
from typing import List, Dict, Any
import torch

class ChunkedLLMServer:
    """LLM server with chunked encoding support."""
    
    def __init__(
        self,
        model_name: str,
        chunk_start_token: str = "<Chunk>",
        chunk_end_token: str = "</Chunk>",
        **llm_kwargs,
    ):
        self.llm = LLM(model=model_name, **llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        
        # Get chunk token IDs
        self.chunk_start_token_id = self.tokenizer.convert_tokens_to_ids(chunk_start_token)
        self.chunk_end_token_id = self.tokenizer.convert_tokens_to_ids(chunk_end_token)
        
        print(f"Chunk start token ID: {self.chunk_start_token_id}")
        print(f"Chunk end token ID: {self.chunk_end_token_id}")
    
    def preprocess_prompt(self, prompt: str) -> str:
        """Ensure chunk markers are properly formatted in the prompt."""
        # This is where you can add validation or formatting logic
        return prompt
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams = None,
    ) -> List[str]:
        """Generate completions for prompts with chunk support.
        
        Args:
            prompts: List of prompts with chunk markers
            sampling_params: Sampling parameters
            
        Returns:
            List of generated texts
        """
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        
        # Preprocess prompts
        processed_prompts = [self.preprocess_prompt(p) for p in prompts]
        
        # Generate
        outputs = self.llm.generate(processed_prompts, sampling_params)
        
        # Extract text from outputs
        results = [output.outputs[0].text for output in outputs]
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize server
    server = ChunkedLLMServer(
        model_name="your-model-name-chunked",
        tensor_parallel_size=1,
        dtype="float16",
    )
    
    # Example prompts with chunks
    prompts = [
        "Context: <Chunk>First document chunk</Chunk> <Chunk>Second document chunk</Chunk> Question: What is the relationship?",
        "Analyze these sections: <Chunk>Section 1 content here</Chunk> <Chunk>Section 2 content here</Chunk>",
    ]
    
    # Generate
    results = server.generate(prompts)
    
    for prompt, result in zip(prompts, results):
        print(f"Prompt: {prompt}")
        print(f"Result: {result}")
        print("-" * 80)
```

### Step 7: API Server Integration

For REST API serving with vLLM's OpenAI-compatible API:

```python
# custom_api_server.py

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from typing import Optional
import argparse

def preprocess_messages(messages, chunk_format=True):
    """Preprocess messages to include chunk markers if needed."""
    if not chunk_format:
        return messages
    
    # Your custom logic to add chunk markers
    # For example, detecting document boundaries and wrapping them
    processed_messages = []
    for msg in messages:
        content = msg.get("content", "")
        # Add chunk markers if needed based on your application logic
        # This is application-specific
        processed_messages.append({
            **msg,
            "content": content,
        })
    
    return processed_messages


# Start the server
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    # Run the standard vLLM OpenAI-compatible server
    # The model should already have chunked RoPE integrated
    run_server(
        model=args.model,
        host=args.host,
        port=args.port,
    )
```

## Usage Examples

### Example 1: Basic Serving

```bash
# Start the server
python serve_chunked_model.py
```

```python
# Client code
from vllm import LLM, SamplingParams

llm = LLM(model="your-model-name-chunked")

prompt = """
Please analyze the following documents:
<Chunk>
Document 1: This is the first document with important information about topic A.
</Chunk>

<Chunk>
Document 2: This is the second document with information about topic B.
</Chunk>

Based on these documents, what is the relationship between topics A and B?
"""

outputs = llm.generate([prompt], SamplingParams(temperature=0.7, max_tokens=100))
print(outputs[0].outputs[0].text)
```

### Example 2: OpenAI API Compatible Server

```bash
# Start the API server
python -m vllm.entrypoints.openai.api_server \
    --model your-model-name-chunked \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --port 8000
```

```python
# Client code
import openai

openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

response = openai.ChatCompletion.create(
    model="your-model-name-chunked",
    messages=[
        {
            "role": "user",
            "content": "Analyze: <Chunk>Part 1</Chunk> <Chunk>Part 2</Chunk>"
        }
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
```

### Example 3: Batch Processing with Chunks

```python
from chunked_llm_server import ChunkedLLMServer

server = ChunkedLLMServer("your-model-name-chunked")

# Process multiple documents with chunks
documents = [
    "Doc 1: <Chunk>Section A</Chunk> <Chunk>Section B</Chunk>",
    "Doc 2: <Chunk>Part 1</Chunk> <Chunk>Part 2</Chunk> <Chunk>Part 3</Chunk>",
    "Doc 3: <Chunk>Chapter 1</Chunk> <Chunk>Chapter 2</Chunk>",
]

results = server.generate(documents)

for doc, result in zip(documents, results):
    print(f"Input: {doc}")
    print(f"Output: {result}\n")
```

## Important Considerations

### 1. Token Vocabulary Size
After adding chunk markers, you may need to resize the model's token embeddings:

```python
model.resize_token_embeddings(len(tokenizer))
```

### 2. Position Computation Overhead
The position computation happens for each sequence. For better performance:
- Cache position computations if possible
- Batch process sequences with similar chunk structures
- Consider moving position computation to CUDA kernels

### 3. Attention Masking
You may want to add attention masking to prevent cross-chunk attention:

```python
# In attention layer
if use_chunk_masking:
    # Create mask to prevent attention between different chunks
    mask = create_chunk_attention_mask(block_indices)
    attn_output = attn_output * mask
```

### 4. Training Considerations
If you're fine-tuning with chunked encoding:
- Ensure training data includes chunk markers
- Use the same position computation during training
- Consider curriculum learning (start without chunks, gradually add them)

## Debugging Tips

1. **Verify Token IDs**: Always print and verify chunk token IDs
2. **Check Positions**: Log position arrays to ensure correctness
3. **Validate Shapes**: Ensure tensor shapes match after filtering separators
4. **Test Edge Cases**: Empty chunks, nested markers, malformed input
5. **Profile Performance**: Compare latency with/without chunked encoding

## Performance Optimization

1. **Precompute Positions**: Cache position computations for common patterns
2. **Vectorized Operations**: Use batch processing for position computation
3. **CUDA Kernels**: Implement custom CUDA kernels for chunk-aware attention
4. **KV Cache**: Ensure KV cache works correctly with chunked positions

## Next Steps

1. Test the integration with your specific model
2. Benchmark performance vs standard RoPE
3. Fine-tune the model with chunked training data
4. Deploy and monitor in production
5. Iterate based on application requirements

## Troubleshooting

### Issue: Token IDs Not Found
**Solution**: Ensure special tokens are added before saving tokenizer

### Issue: Position Mismatch
**Solution**: Verify chunk markers are properly paired

### Issue: Performance Degradation
**Solution**: Profile and optimize position computation

### Issue: Attention Errors
**Solution**: Check tensor shapes after filtering separators

## Additional Resources

- vLLM Documentation: https://docs.vllm.ai/
- Rotary Embeddings: https://arxiv.org/abs/2104.09864
- vLLM Model Integration Guide: https://docs.vllm.ai/en/latest/models/adding_model.html
