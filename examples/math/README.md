## Hyper-parameters for GSM8K Finetuning on Qwen2.5-1.5b-Instruct

`gsm8k_grpo.yaml` uses `6e-6`, retuned for the current FSDP FP32-master optimizer path.

### Current FP32-Master Recipe

A seed-1 run on 8 NVIDIA A800 GPUs completed the official 10-epoch schedule with the
following held-out evaluation results:

| lr     | weight decay | group size | best eval reward | final eval reward |
| ------ | ------------ | ---------- | ---------------- | ----------------- |
| 6.0E-6 | 0.017        | 4          | **0.78412**      | 0.77767           |

### Historical Pre-FP32-Master Sweep

The results below were collected with BF16 parameter and optimizer-state storage. They
are retained for reference and are not directly comparable with the current recipe.

| lr       | weight decay | group size | max task_reward |
| -------- | ------------ | ---------- | --------------- |
| 1.70E-05 | 0.017        | 4          | **0.79570**     |
| 1.30E-05 | 0.015        | 8          | 0.79355         |
| 1.50E-05 | 0.01         | 4          | 0.79043         |
| 1.50E-05 | 0.02         | 4          | 0.78984         |
| 1.00E-05 | 0.02         | 4          | 0.78311         |
| 1.00E-05 | 0.01         | 8          | 0.78066         |

#### Training Details

- Devices: 8 Nvidia H800 GPUs
- Optimizer: Adam
- LR Scheduler: Constant
- Gradient Clipping: 1.0
- Max_new_tokens: 1024
- Max_head_offpolicyness: 2
- Training Time: ~35 minutes (batchsize 4), ~65 minutes (batchsize 8)
