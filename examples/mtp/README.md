# Qwen native MTP SFT

`qwen38_27b_sft.yaml` uses the existing SFT entrypoint and freezes the backbone, shared
embedding, and output head. The model directory must contain the native MTP weights and
matching configuration. The dense Qwen model uses EP=1.

MTP-only configuration validation requires loaded cuDNN >=9.19.0, Megatron-Core 0.19.0
or newer, and Megatron-Bridge 0.6.0 or newer. It rejects missing or older runtimes
before model initialization, including an older cuDNN library loaded by PyTorch.

Requirements for Qwen GDN packed THD, CP, and native MTP training: Megatron-Core 0.19.0
or newer and Megatron-Bridge 0.6.0 or newer, with their compatible CUDA/Transformer
Engine/Transformers runtime. Supply model and dataset paths through environment
variables.

The actual 27B MTP-only smoke test uses Megatron-Core 0.19.0, Megatron-Bridge 0.6.0,
PyTorch 2.9.1 (CUDA 12.9), Transformer Engine 2.14.1, and cuDNN 9.19.0.56. The same test
with cuDNN 9.16.0.29 produced nonfinite attention gradients; use the validated cuDNN
version and verify the loaded libraries before training. A newer Python package alone
does not guarantee that its CUDA libraries are loaded.

The dataset can be a Hugging Face DatasetDict saved with `save_to_disk`, containing
`train` and `test` splits and tokenized `input_ids` and `loss_mask` columns. The mask
must select assistant tokens. Use representative coding continuations from the frozen
target model when adapting its MTP head.

```bash
export MODEL_PATH=/path/to/checkpoint-with-native-mtp
export SFT_DATASET_PATH=/path/to/tokenized-dataset
python examples/math/gsm8k_sft.py \
  --config examples/mtp/qwen38_27b_sft.yaml
```

The recipe allocates 8 GPUs: DP1, PP2, TP2, CP2. For joint backbone and MTP SFT, append
`actor.megatron.mtp_only=false`. Adapt the allocation, cluster, and scheduler settings
to the actual hardware before multinode training. Memory use depends on sequence length,
microbatch size, and optimizer settings.

## Distributed validation

The test entrypoint runs two optimizer steps with unequal packed sequence lengths, full
activation recomputation, and deallocated pipeline outputs. It checks frozen parameters
exactly and requires MTP parameters to update. Its default uses a small model derived
from the supplied Qwen configuration.

```bash
torchrun --standalone --nproc-per-node=8 tests/torchrun/run_qwen_mtp_sft.py \
  --checkpoint "$MODEL_PATH" --tp 2 --cp 2
```

Append `--full` for joint training. Use `--tp 1 --cp 2` on 8 GPUs to exercise DP2.
Append `--real-size --export /path/to/new-export` to load the actual checkpoint, train,
export to Hugging Face, clear model weights, reload the export, and compare every
parameter exactly. An export directory should be new and task-specific. These are smoke
checks, not a measurement of MTP acceptance rate or model quality.

Validated: actual 27B weights with TP2/PP2/CP2 passed two finite optimizer steps in both
MTP-only and joint modes; MTP-only HF export/reload matched every parameter exactly on
all ranks. DP2/TP1/PP2/CP2 passed with the reduced configuration. The smoke inputs
contain two short sequences (64 and 32 tokens); long-context and multinode performance
require separate validation.
