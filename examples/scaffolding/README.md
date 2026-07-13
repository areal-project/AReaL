# Scaffolding Framework Examples for AReaL

This directory contains examples demonstrating how to use the Scaffolding framework with
AReaL for reinforcement learning training.

## Overview

The scaffolding framework provides a modular and extensible way to compose various
methods with RL training. It decouples the inference logic (Controllers) from the
execution backend (Workers), enabling flexible composition of different methods. With
Scaffolding, we can flexibly compose various rollout, reward, and trajectory tracing
methods.

### Key Components

1. **Controller**: Defines the inference-time compute logic (e.g., generation, reward
   computation)
1. **Worker**: Handles the actual execution of tasks (e.g., TRT-LLM, OpenAI API)
1. **ScaffoldingLlm**: Orchestrates controllers and workers together
1. **ScaffoldingWorkflow**: Wraps ScaffoldingLlm as a RolloutWorkflow for AReaL training

### AReaL-Specific Components

The following components are implemented in `examples/scaffolding/`:

- **`CreateWorkerFromEngine`**: Creates a scaffolding Worker from AReaL's
  InferenceEngine (e.g., RemoteSGLangEngine). The returned Worker is similar to
  scaffolding's `OpenaiWorker` but integrated with AReaL's engine.

- **`RLVRRewardController`**: A Controller that computes rewards for generated samples
  using verifiable reward functions (e.g., math answer verification).

- **`PipelineTrajectoryMaker`**: A Controller that composes generation and reward
  controllers into a pipeline that produces training trajectories.

- **`ScaffoldingWorkflow`**: A `RolloutWorkflow` implementation that wraps
  ScaffoldingLlm for integration with AReaL's training pipeline.

## RLVR Example with GSM8K

### Quick Start

```bash
python examples/scaffolding/gsm8k_rlvr_scaffolding.py \
    --config examples/scaffolding/gsm8k_rlvr_scaffolding.yaml
```

### Architecture

The scaffolding workflow follows this pattern from the RFC:

```python
# Step 1: Create Worker from the SGLang engine
rollout_worker = CreateWorkerFromEngine(engine)

# Step 2: Create controllers
rollout_controller = NativeGenerationController()
reward_controller = RLVRRewardController(gsm8k_reward_fn)

# Step 3: Create trajectory maker (composes the controllers)
trajectory_maker = PipelineTrajectoryMaker(rollout_controller, reward_controller)

# Step 4: Create ScaffoldingLlm (orchestrates controllers with workers)
scaffolding_llm = ScaffoldingLlm(
    trajectory_maker,
    {NativeGenerationController.WorkerTag.GENERATION: rollout_worker},
)

# Step 5: Create ScaffoldingWorkflow (wraps as RolloutWorkflow)
scaffolding_workflow = ScaffoldingWorkflow(scaffolding_llm)
```

### Data Flow Diagram

```
                              ┌─────────────────────────────────────────────────┐
                              │              ScaffoldingWorkflow                │
                              │                                                 │
                              │  ┌───────────────────────────────────────────┐  │
                              │  │            ScaffoldingLlm                 │  │
                              │  │                                           │  │
                              │  │  ┌─────────────────────────────────────┐  │  │
                              │  │  │      PipelineTrajectoryMaker        │  │  │
                              │  │  │                                     │  │  │
                              │  │  │  ┌───────────────────────────────┐  │  │  │
Data ─────────────────────────┼──┼──┼──►  NativeGenerationController   │  │  │  │
                              │  │  │  │  (from scaffolding.core)       │  │  │  │
                              │  │  │  └───────────────┬───────────────┘  │  │  │
                              │  │  │                  │                  │  │  │
                              │  │  │                  ▼                  │  │  │
                              │  │  │  ┌───────────────────────────────┐  │  │  │
                              │  │  │  │  RLVRRewardController         │  │  │  │
                              │  │  │  │  (from areal.experimental)    │  │  │  │
                              │  │  │  └───────────────┬───────────────┘  │  │  │
                              │  │  │                  │                  │  │  │
                              │  │  └──────────────────┼──────────────────┘  │  │
                              │  │                     │                     │  │
                              │  └─────────────────────┼─────────────────────┘  │
                              │                        │                        │
                              └────────────────────────┼────────────────────────┘
                                                       │
                                                       ▼ Trajectories
                                         ┌─────────────────────────────┐
                                         │       PPOTrainer            │
                                         │   (GRPO/PPO Training)       │
                                         └─────────────────────────────┘
                                                       │
                         via CreateWorkerFromEngine    │
                                                       ▼
                              ┌─────────────────────────────────────────┐
                              │         RemoteSGLangEngine              │
                              │         (AReaL Inference Backend)       │
                              └─────────────────────────────────────────┘
```

### How It Works

1. **Engine Initialization**: `RemoteSGLangEngine` is initialized with the rollout
   configuration and connected to the model server.

1. **Worker Creation**: `CreateWorkerFromEngine(engine)` wraps the engine into a
   scaffolding-compatible Worker. This allows scaffolding controllers to use AReaL's
   inference backends.

1. **Controller Pipeline**:

   - `NativeGenerationController()`: Handles text generation by yielding
     `GenerationTask` objects to the Worker.
   - `RLVRRewardController(reward_fn)`: Computes rewards for generated samples using the
     provided reward function.
   - `PipelineTrajectoryMaker(gen_ctrl, reward_ctrl)`: Composes these controllers into a
     pipeline that produces training trajectories.

1. **ScaffoldingLlm**: Orchestrates the trajectory maker with the worker, handling the
   async execution of tasks.

1. **ScaffoldingWorkflow**: Wraps the ScaffoldingLlm as a `RolloutWorkflow` that can be
   used directly with AReaL's `PPOTrainer`.

1. **Training**: The trainer calls the workflow to generate trajectories, which are then
   used for GRPO/PPO training.

### Configuration

See `gsm8k_rlvr_scaffolding.yaml` for the full configuration. Key options:

```yaml
# Model configuration
pretrain_path: Qwen/Qwen2.5-3B-Instruct
tokenizer_path: Qwen/Qwen2.5-3B-Instruct

# Generation hyperparameters
gconfig:
  max_new_tokens: 1024
  temperature: 1.0
  top_p: 1.0
  n_samples: 8

# Inference engine configuration
engine:
  type: sglang
  tp: 1
  max_model_len: 4096
```

## Extending the Framework

### Custom Reward Controllers

You can create custom reward controllers by subclassing the base Controller:

```python
from examples.scaffolding._compat import Controller

class CustomRewardController(Controller):
    def __init__(self, reward_fn):
        super().__init__()
        self.reward_fn = reward_fn

    def process(self, tasks, **kwargs):
        # Compute rewards for completed generation tasks
        for task in tasks:
            reward = self.reward_fn(
                prompt=task.input_str,
                completion=task.output_str,
                **kwargs
            )
            task.customized_result_fields["reward"] = reward
        yield tasks
```

### Custom Trajectory Makers

For different RL algorithms, you may need different trajectory formats:

```python
from examples.scaffolding._compat import Controller
import torch

class CustomTrajectoryMaker(Controller):
    def __init__(self, generation_controller, reward_controller):
        super().__init__()
        self.generation_controller = generation_controller
        self.reward_controller = reward_controller

    def process(self, tasks, **kwargs):
        # Run generation
        yield from self.generation_controller.process(tasks, **kwargs)

        # Run reward computation
        yield from self.reward_controller.process(tasks, **kwargs)

        # Build trajectories
        trajectories = []
        for task in tasks:
            trajectory = {
                "input_ids": torch.tensor(task.output_tokens),
                "rewards": torch.tensor(task.customized_result_fields["reward"]),
            }
            trajectories.append(trajectory)
        yield trajectories
```

## References

- [TensorRT-LLM Scaffolding README](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/scaffolding)
- [AReaL Workflow Documentation](../../docs/customization/workflow.md)
- [RFC: Scaffolding Integration](https://github.com/areal-project/AReaL/issues/818)
