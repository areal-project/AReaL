# Training VLMs with GRPO on NPU:

In this instruction, we will introduce how to train VLMs with GRPO on Ascend NPU.

## Examples

This directory contains examples for training vision-language models on NPU with GRPO:

1. **Qwen2.5-VL-3B** - `qwen2_5_vl_3b_geometry3k_grpo.*` - GRPO training on Geometry3K
   with Qwen2.5-VL-2B model dataset
1. **Qwen2.5-VL-3B (multi-node)** - `qwen2_5_vl_3b_virl39k_grpo_multinode.sh` -
   multi-node GRPO training on ViRL39K dataset
1. **Qwen3-VL-2B** - `qwen3_vl_2b_geometry3k_grpo.*` - GRPO training on Geometry3K
   dataset with Qwen3-VL-2B model

## Running the Examples

### Qwen2.5-VL-3B

```bash
bash examples/vlm_npu/qwen2_5_vl_3b_geometry3k_grpo.sh
```

### Qwen2.5-VL-3B (ViRL39K, multi-node)

```bash
bash examples/vlm_npu/qwen2_5_vl_3b_virl39k_grpo_multinode.sh
```

### Qwen3-VL-2B

```bash
bash examples/vlm_npu/qwen3_vl_2b_geometry3k_grpo.sh
```

The Geometry3K examples use the same dataset and GRPO training configuration, but with
different model architectures. The Qwen3-VL-2B model is smaller and may be more suitable
for environments with limited resources. The ViRL39K multi-node example customizes the
default configuration for multi-node training settings.

## Testing Qwen2.5-VL-3B

### Hardware

The following hardware configuration has been extensively tested:

- **NPU**: 16x NPU per node
- **CPU**: 64 cores per node
- **Memory**: 1TB per node
- **Network**: RoCE 3.2 Tbps
- **Storage**:
  - 1TB local storage for single-node experiments
  - 10TB shared storage (NAS) for distributed experiments

### Key Contributions

- Trained Qwen2.5VL-3B-instruct model upto 70 epochs with (4 cards+ 4 cards) train-infer
  configuration. Took around 19hr to finish full training.
- Trained model is tested with more than one benchmark using VLMEvalKit.

### Results:

We trained Qwen2.5-VL-3B for 70 epochs on Geometry3K and evaluated the checkpoints using
VLMEvalKit on out of distribution tasks such as MathVision, MathVista, and LogicVista.
The training was performed on both NPU and GPU and results are as follows:

| Method     | LogicVista | MathVision_mini | MathVista_mini | Avg.     |
| ---------- | ---------- | --------------- | -------------- | -------- |
| Base Model | 31.0       | 18.3            | 52.3           | 33.8     |
| GRPO-GPU   | 35.4       | 20.9            | 55.9           | **37.4** |
| GRPO-NPU   | 35.3       | 20.5            | 54.7           | **36.8** |

## AReaL vs. verl: Multi-node Training Performance

We test performance of AReaL for large-scale multi-node training with long context
generation and compare this performance with verl synchronous training with the same
training settings. All training and evaluation is done on Ascend NPU.

AReaL asynchronous settings

| Framework | Nodes | N_GPUS Per Node | Train Dataset       | Max Generated Tokens | Max Head Offpolicyness | Batch Size | Allocation Mode |
| --------- | ----- | --------------- | ------------------- | -------------------- | ---------------------- | ---------- | --------------- |
| AReaL     | 3     | 16              | `TIGER-Lab/ViRL39K` | 16384                | 4                      | 528        | vllm:d32+d16    |
| verl      | 3     | 16              | `TIGER-Lab/ViRL39K` | 16384                | -                      | 528        | -               |

### Training setup

We trained Qwen2.5-VL-3B following the settings in the above table:

- **AReaL launcher**: `examples/vlm_npu/qwen2_5_vl_3b_virl39k_grpo_multinode.sh`
- **Dataset**: [`TIGER-Lab/ViRL39K`](https://huggingface.co/datasets/TIGER-Lab/ViRL39K)

### Performance Comparison

We compare the training time and out-of-distribution (OOD) performance of both
frameworks. For OOD evaluation, we use VLMEvalKit and report `Avg@8` accuracy to report
the performance.

| Framework | Method   | Checkpoint | Training Time | LogicVista | MathVision_mini | WeMath   | DynaMath | MathVerse | MMMU_Pro_v | Avg. |
| --------- | -------- | ---------- | ------------- | ---------- | --------------- | -------- | -------- | --------- | ---------- | ---- |
| verl      | GRPO-NPU | Epoch 1    | 6.8 hours     | 33.0       | 18.8            | **19.6** | 32.9     | **31.3**  | **23.5**   | 26.5 |
| AReaL     | GRPO-NPU | Epoch 2    | **4.3 hours** | 33.8       | 20.1            | 17.4     | 34.9     | 29.6      | 22.3       | 26.3 |
| AReaL     | GRPO-NPU | Epoch 3    | **6.6 hours** | **34.1**   | **20.3**        | 18.9     | **35.7** | 30.2      | 22.5       | 27.0 |

Under identical hardware and training configurations, AReaL reaches two training epochs
in 4.3 hours, whereas verl requires 6.8 hours to complete a single epoch. Despite the
shorter wall-clock time, AReaL achieves comparable OOD performance at that stage (Avg@8:
26.3 vs. 26.5). With additional training (Epoch 3), AReaL surpasses verl in overall
accuracy (27.0) while still requiring less total training time (6.6 hours). These
results suggest that AReaL’s asynchronous training strategy improves time-to-performance
efficiency for large-scale, long-context GRPO training without sacrificing downstream
generalization.
