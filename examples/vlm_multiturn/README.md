# Multi-Turn Agentic VLM Training

This folder contains an implementation for training Vision-Language Models (VLMs) with multi-turn agentic interactions and error recovery capabilities using AReaL's GRPO algorithm.

## Overview

The multi-turn agentic workflow enables VLMs to:
- Process multi-modal inputs (images + text)
- Engage in multi-turn conversations with retry mechanisms
- Learn from failures through automatic feedback injection
- Apply turn-based reward discounting to incentivize correct answers on earlier turns

## Key Features

### Multi-Turn Interaction with Error Recovery
When a model generates an incorrect response (reward < 1), the workflow automatically:
1. Appends failure feedback to the conversation
2. Prompts the model to retry with the context of its previous error
3. Continues up to `max_turns` iterations until success or exhaustion
4. Applies exponential discounting (`turn_discount`) to later-turn rewards

### Flexible Reward Accumulation
The workflow tracks rewards across turns using a "best-so-far" strategy with discounting:
```python
reward = max(previous_reward, current_turn_reward * discount^turn)
```
This encourages the model to learn both correctness and efficiency.

## Files

- `vlm_multiturn_grpo.py` - Main training script
- `vlm_multiturn_grpo.yaml` - GRPO training configuration
- `train_vlm_multiturn.sh` - Example training script for GPU
- `train_vlm_multiturn_npu.sh` - Example training script for NPU
- `README.md` - This file

## Quick Start

### Basic Usage

```bash
bash examples/vlm_multiturn/train_vlm_multiturn.sh
```

Or run directly with custom configuration:

```bash
python3 -m areal.launcher.local \
    examples/vlm_multiturn/vlm_multiturn_grpo.py \
    --config examples/vlm_multiturn/vlm_multiturn_grpo.yaml \
    tokenizer_path=/path/to/model \
    actor.path=/path/to/model \
    max_turns=4 \
    turn_discount=0.95 \
    gconfig.n_samples=4 \
    experiment_name=my_vlm_experiment
```

### Configuration

The workflow adds the following key parameters to standard GRPO training:

#### Multi-Turn Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_turns` | 2 | Maximum number of turns for error recovery |
| `turn_discount` | 0.95 | Discount factor applied to rewards at each subsequent turn |

**Example configuration:**

```yaml
max_turns: 4
turn_discount: 0.95
```

#### Understanding Turn Discount

The `turn_discount` parameter controls how much less valuable later-turn successes are compared to first-turn successes:

- `turn_discount=1.0`: All turns equally valuable (no incentive for speed)
- `turn_discount=0.95`: Turn 2 reward is 95% of turn 1, turn 3 is 90.25%, etc.
- `turn_discount=0.8`: More aggressive discounting, strongly prefers quick success

## Architecture

### VisionMultiTurnWorkflow

Located in `areal/workflow/vision_multiturn.py`, this workflow:

1. **Processes Images**: Uses HuggingFace processor to prepare multi-modal inputs
2. **Iterates Through Turns**:
   - Generates model response
   - Computes reward
   - If reward < 1.0 and turns remain, injects failure feedback
   - Applies turn discount
3. **Returns Training Data**: Properly formatted tensors for GRPO training

Key workflow loop:
```python
for turn in range(max_turns):
    response = generate(current_conversation)
    reward = compute_reward(response)
    
    if reward >= 1.0 or turn == max_turns - 1:
        break
        
    # Inject failure feedback for retry
    current_conversation.append(failure_feedback)
    discount *= turn_discount
```

## Hardware Requirements

The examples are designed to run on:
- **GPU**: 8x GPUs (adjust `cluster.n_gpus_per_node`)
- **NPU**: Ascend NPU (use `train_vlm_multiturn_npu.sh` and run it using **ascend** branch)


