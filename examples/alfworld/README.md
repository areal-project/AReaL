# ALFWorld GRPO Training

Train a small model (Qwen2.5-7B) on ALFWorld tasks using GRPO with outcome reward (0/1).

## Setup

```bash
pip install alfworld textworld
```

Ensure ALFWorld game data is available at:
```
/storage/openpsi/users/yl/agent-memory/MemRL/data/alfworld/json_2.1.1/
```

## Usage

```bash
# Step 1: Build dataset (converts game files to HuggingFace Dataset)
python examples/alfworld/dataset.py \
    --data_root /storage/openpsi/users/yl/agent-memory/MemRL/data/alfworld/json_2.1.1 \
    --output_dir /tmp/areal/alfworld_dataset \
    --split train

# Step 2: Train (local single-node 8 GPU)
python examples/alfworld/train.py \
    --config examples/alfworld/config.yaml \
    scheduler.type=local
```

## Design

- **Workflow**: Multi-turn ReAct agent interacting with ALFWorld TextWorld env
- **Reward**: Binary outcome (task success=1, failure=0), baseline=0.5
- **Algorithm**: GRPO with 4 samples per task, group normalization
- **Model**: Qwen2.5-7B-Instruct
- **Turn discount**: 0.9 (earlier actions get discounted reward signal)

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| n_samples | 4 | Samples per task for GRPO |
| max_steps | 30 | Max env interaction steps |
| reward_bias | -0.5 | Fixed baseline |
| reward_scaling | 10.0 | Amplify reward signal |
| turn_discount | 0.9 | Backward reward propagation |
| kl_ctl | 0.01 | Light KL penalty |
| lr | 5e-6 | Conservative for 7B |
| eps_clip | 0.2 | Standard PPO clip |
