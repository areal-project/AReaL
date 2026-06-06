# Installing vLLM with PartitionedRotaryEmbedding

This directory contains a modified version of vLLM v0.12.0 that includes `PartitionedRotaryEmbedding`.

## Installation on Target Machine

### Prerequisites
- CUDA toolkit installed (11.8 or later)
- Python 3.8+
- GCC/G++ compiler

### Steps

1. **Transfer this directory to the target machine:**
   ```bash
   # On this machine
   scp -r vllm-releases-v0.12.0 user@target-machine:/path/to/destination/
   
   # Or use rsync for faster transfer
   rsync -avz --progress vllm-releases-v0.12.0/ user@target-machine:/path/to/destination/
   ```

2. **On the target machine, uninstall existing vLLM:**
   ```bash
   pip uninstall vllm -y
   ```

3. **Install dependencies:**
   ```bash
   cd /path/to/vllm-releases-v0.12.0
   pip install -r requirements/requirements-common.txt
   pip install -r requirements/requirements-cuda.txt  # If using CUDA
   ```

4. **Install vLLM from source:**
   ```bash
   # For development (changes to Python files take effect immediately)
   pip install -e .
   
   # OR for production (creates a proper installation)
   pip install .
   ```

   **Note**: This will compile CUDA kernels, which may take 10-30 minutes.

5. **Verify installation:**
   ```bash
   python -c "from vllm.model_executor.layers.rotary_embedding import PartitionedRotaryEmbedding; print('✓ PartitionedRotaryEmbedding available')"
   ```

## Usage

### In Python Code
```python
from vllm import LLM

llm = LLM(
    model="your-model-name",
    hf_overrides={
        "rope_parameters": {
            "rope_type": "partitioned",
            "rope_theta": 10000,
            "period": 512  # Optional, defaults to 512
        }
    }
)
```

### Via Model Config
Edit your model's `config.json`:
```json
{
  "rope_parameters": {
    "rope_type": "partitioned",
    "rope_theta": 10000,
    "period": 512
  }
}
```

## Alternative: Quick Update (Not Recommended)

If you already have vLLM installed and just want to update the Python file:

1. **Find vLLM installation:**
   ```bash
   python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))"
   ```

2. **Copy the file:**
   ```bash
   cp vllm/model_executor/layers/rotary_embedding/partitioned_rope.py \
      /path/to/site-packages/vllm/model_executor/layers/rotary_embedding/
   ```

⚠️ **Warning**: This approach:
- Will be overwritten on vLLM updates
- Doesn't guarantee version compatibility
- May cause subtle bugs if the rest of vLLM expects different behavior

## Verification

Test that PartitionedRotaryEmbedding works:

```python
import torch
from vllm.model_executor.layers.rotary_embedding import get_rope

rope = get_rope(
    head_size=128,
    max_position=2048,
    rope_parameters={
        "rope_type": "partitioned",
        "rope_theta": 10000,
        "period": 512
    }
)

print(f"✓ Created: {type(rope).__name__}")
print(f"  Period: {rope.period}")
print(f"  RoPE dim: {rope.rope_dim} (80%)")
print(f"  Custom dim: {rope.custom_dim} (20%)")
```

## Troubleshooting

### Build fails with CUDA errors
- Ensure CUDA toolkit is installed: `nvcc --version`
- Set CUDA_HOME: `export CUDA_HOME=/usr/local/cuda`

### ImportError after installation
- Check Python path: `python -c "import sys; print(sys.path)"`
- Verify vLLM location: `pip show vllm`

### Changes not taking effect
- If installed with `pip install .` (not `-e`), you need to reinstall after changes
- If using `-e`, only Python changes take effect; C++/CUDA changes need rebuild
