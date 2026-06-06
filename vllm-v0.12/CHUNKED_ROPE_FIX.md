# ChunkedRotaryEmbedding torch.compile Fix

## Problem

The `ChunkedRotaryEmbedding.forward()` method was incompatible with vLLM's torch.compile system, causing the error:

```
torch._dynamo.exc.Unsupported: Unsupported Tensor.item() call with capture_scalar_outputs=False
```

## Root Cause

vLLM uses `torch.compile` (TorchDynamo) to compile model code for performance. The `Chunked RotaryEmbedding` implementation used `Tensor.item()` calls to extract scalar values from tensors, which breaks the computational graph during compilation when `capture_scalar_outputs=False` (the default).

## Solution

The fix involves two changes:

### 1. Enable Scalar Output Capture

At module initialization (top of `chunked_rope.py`), we now enable `capture_scalar_outputs`:

```python
# Enable scalar output capture for torch.compile compatibility
try:
    import torch._dynamo.config
    torch._dynamo.config.capture_scalar_outputs = True
except (ImportError, AttributeError):
    pass
```

This allows `.item()` calls to be traced and included in the compiled graph.

### 2. Use Compile-Safe Tensor Operations

Modified the `forward()` method to:
- Use `index_select()` instead of slicing with `.item()` where possible
- Extract scalar values in a helper function `_tensor_to_int()` that torch.compile can handle
- Minimize the use of `.item()` calls overall

## Files Modified

- `vllm/model_executor/layers/rotary_embedding/chunked_rope.py`

## Testing

After applying the fix, run your command:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /storage/openpsi/models/Qwen__Qwen3-4B \
    --hf-overrides '{"rope_parameters": {"rope_type": "chunked"}}'
```

The server should now start successfully without torch.compile errors.

## Technical Details

### Why capture_scalar_outputs=True?

When `capture_scalar_outputs=True`, torch.compile:
1. Traces `.item()` calls as part of the computational graph
2. Includes them in the compiled function
3. Allows dynamic values to be extracted while maintaining graph structure

Without this, `.item()` calls cause immediate graph breaks and compilation failures.

### Performance Impact

Enabling `capture_scalar_outputs` may have a small performance impact since:
- Scalar extractions are now part of the compiled graph
- Some optimizations may be limited by dynamic scalar dependencies

However, this is necessary for the chunked RoPE implementation to work with vLLM's compilation system.

## Alternative Approaches Considered

1. **Using `torch._dynamo.disable()`**: Failed because calling the disable context manager inside a compiled function is not supported.

2. **Complete rewrite without loops**: Would require vectorizing the block processing logic, which is complex and may not improve performance significantly.

3. **Environment variable**: Setting `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` works but requires user configuration. The code-based solution is more robust.

## Future Improvements

For better performance, consider:
1. Pre-computing chunked encodings for common block indices
2. Vectorizing the block processing loop to avoid .item() calls entirely
3. Implementing custom CUDA kernels for the chunked RoPE operation
