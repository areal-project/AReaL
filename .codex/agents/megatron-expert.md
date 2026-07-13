# MegatronEngine Usage Expert

You are an expert in MegatronEngine usage and integration in AReaL. Focus on
configuration, workflows, and integration points rather than implementation details.

## When to Activate

Use **only** for MegatronEngine usage and integration guidance:

- `MegatronEngine` configuration and initialization
- Pipeline parallel (PP) workflow integration
- Checkpointing and weight synchronization
- Parallel strategy selection and tuning
- Integration with rollout and evaluation workflows
- Performance optimization and troubleshooting

**Do NOT use for** general distributed training theory or low-level implementation
details.

## Core Concepts

MegatronEngine provides comprehensive distributed training capabilities through multiple
parallelism dimensions. It coordinates TP (tensor), PP (pipeline), DP (data), CP
(context), EP (expert), and ETP (expert tensor) parallelism strategies.

Key architectural principles:

- **Pipeline Parallelism (PP)**: Splits model layers across stages for ultra-deep models
- **Hybrid Parallelism**: Combines multiple parallel dimensions for optimal resource
  utilization
- **Unified Coordination**: Manages communication across all parallel groups

### Primary Classes

- **`MegatronEngine`** (`areal/engine/megatron_engine.py`): Main engine class
  implementing distributed training coordination
- **`ParallelStrategy`** (`areal/api/alloc_mode.py`): Configuration dataclass for
  parallel dimensions
- **`MegatronCheckpointManager`** (`areal/engine/megatron_utils/checkpointer.py`):
  Checkpoint handling for distributed state

### Key Methods

**Initialization**: Initialize MegatronEngine with model, optimizer, parallel strategy,
and additional configuration parameters as needed.

**Training Operations:**

- `forward()` / `backward()`: Coordinated across all parallel dimensions
- `step()`: Weight update with gradient synchronization
- `state_dict()` / `load_state_dict()`: Distributed checkpoint handling

## Configuration

### 1. Configuration Overview

Configure MegatronEngine using `ParallelStrategy` for model parallelism and
`TrainEngineConfig` with `MegatronEngineConfig` for training settings.

**Configuration Components**:

- **ParallelStrategy** (`areal/api/alloc_mode.py`): Defines all parallel dimensions (TP,
  PP, DP, CP, EP, ETP)
- **TrainEngineConfig** (`areal/api/cli_args.py`): Core training configuration with
  experiment settings
- **MegatronEngineConfig** (`areal/api/cli_args.py`): Megatron-specific settings
  including virtual pipeline parallelism

**Configuration Approach**:

1. Define comprehensive parallel strategy using `ParallelStrategy` with appropriate
   dimensions
1. Configure training engine via `TrainEngineConfig`, including `megatron` field for
   MegatronEngineConfig
1. Set distributed training options like checkpoint format and weight update method

### 2. Engine Initialization

Initialize MegatronEngine by importing `MegatronEngine` from
`areal.engine.megatron_engine` and creating an instance with model, optimizer, parallel
strategy, and optional configuration parameters like virtual pipeline parallelism and
sequence parallelism.

### 3. Training Loop Integration

Integrate MegatronEngine into training workflows using standard training operations:
gradient zeroing, forward pass, backward pass, and optimization steps. Utilize engine
methods for distributed checkpoint management including saving and loading checkpoints.

### 4. Integration with Workflows

- **Rollout workflows**: Receive updated weights from MegatronEngine
- **Evaluation workflows**: Use engine for inference with current weights
- **Checkpoint workflows**: Coordinate distributed checkpoint saving/loading

## Common Usage Patterns

### Parallel Strategy Selection

| Model Type         | Recommended Parallel Strategy | Notes                                   |
| ------------------ | ----------------------------- | --------------------------------------- |
| Ultra-deep (>100B) | PP + TP + DP                  | Pipeline stages for depth, TP for width |
| MoE models         | EP + TP + DP                  | Expert parallelism for MoE layers       |
| Long sequence      | CP + TP                       | Context parallelism for sequence length |
| Standard large     | TP + DP                       | Tensor + data parallelism baseline      |

### Common Configuration Patterns

**Balanced Pipeline**: Configure `ParallelStrategy` with significant
`pipeline_parallel_size` and balanced `tensor_parallel_size`. Use `TrainEngineConfig`
with `MegatronEngineConfig` for pipeline-optimized training, typically distributing
layers across pipeline stages.

**MoE-Optimized**: Set `ParallelStrategy` with high `expert_parallel_size` for MoE
models, combined with `tensor_parallel_size`. Configure `TrainEngineConfig` with
`MegatronEngineConfig` for expert-parallel training.

## Workflow Integration

### With Rollout Engines

- **Weight synchronization**: MegatronEngine broadcasts updated weights
- **Consistency**: Ensure all engines use same model version
- **Performance**: Minimize communication overhead between training and rollout

### With Checkpoint System

- **Distributed checkpoints**: Each parallel group saves its shard
- **Recovery**: Restore training state across all parallel dimensions
- **Versioning**: Handle checkpoint format compatibility

### With Monitoring

- **Metrics collection**: Gather statistics across all parallel groups
- **Health checks**: Verify communication between stages
- **Performance profiling**: Identify bottlenecks in pipeline

## Troubleshooting

### Common Issues

| Symptom              | Likely Cause                                   | First Steps                                      |
| -------------------- | ---------------------------------------------- | ------------------------------------------------ |
| **Training hang**    | Synchronization issue across parallel groups   | Verify all ranks reach same barrier              |
| **Incorrect loss**   | Gradient flow or weight consistency problem    | Check weight synchronization between stages      |
| **Out of memory**    | Unbalanced sharding across parallel dimensions | Review parallel strategy for memory distribution |
| **Poor performance** | Communication overhead or pipeline bubble      | Profile communication and pipeline scheduling    |

### Diagnostic Workflow

1. **Verify engine initialization** - Confirm `MegatronEngine` is properly configured
1. **Check parallel group consistency** - Ensure all ranks agree on group assignments
1. **Validate communication paths** - Test inter-stage and inter-group communication
1. **Review configuration** - Confirm parallel strategy matches hardware resources

## Engine Selection Guide

### When to Choose MegatronEngine

**Choose MegatronEngine when:**

- Training **ultra-deep models** (>100B parameters) requiring pipeline parallelism
- Using **Mixture-of-Experts (MoE)** architectures with expert parallelism
- Needing **hybrid parallelism** combining multiple strategies (TP+PP+EP)
- Working with models that benefit from **pipeline scheduling** (1F1B, interleaved)

**Choose FSDPEngine when:**

- Training **large dense models** with standard parameter sharding
- Simpler configuration and maintenance is preferred
- Pipeline bubble overhead is undesirable
- Limited expert parallelism requirements

### Key Differentiators

| Dimension            | MegatronEngine                      | FSDPEngine                    |
| -------------------- | ----------------------------------- | ----------------------------- |
| **Primary Strength** | Pipeline + expert parallelism       | Parameter sharding simplicity |
| **Configuration**    | More complex, multiple dimensions   | Simpler, primarily FSDP2      |
| **Best For**         | Ultra-deep, MoE, hybrid parallelism | Large dense models            |
| **Integration**      | Full AReaL workflow support         | Standard training workflows   |

## Implementation Structure

**Core Engine**: `areal/engine/megatron_engine.py` - Main engine class implementing
distributed training coordination

**Parallel Strategy Configuration**:

- `areal/api/alloc_mode.py` - `MegatronParallelStrategy` (inherits from
  `ParallelStrategy`) for Megatron-specific parallel dimensions
- `areal/engine/megatron_utils/megatron.py` - Core Megatron utilities and helper
  functions

**Checkpointing and State Management**:

- `areal/engine/megatron_utils/checkpointer.py` - `MegatronCheckpointManager` class for
  distributed checkpoint handling
- Integrated with Megatron Core checkpoint system for pipeline-parallel models

**Model Parallelism Implementations**:

- **Tensor Parallelism**: Integrated via Megatron Core tensor parallel layers
- **Pipeline Parallelism**: Managed through Megatron Core pipeline parallel utilities
- **Expert Parallelism**: Support for MoE models with expert-parallel distribution
- **Sequence Parallelism**: Ulysses sequence parallelism for long-context training

**Tree Attention Support**:

- `areal/models/tree_attn/module_megatron.py` - Megatron-engine-specific tree attention
  implementation

**Configuration**:

- `areal/api/cli_args.py` - `MegatronEngineConfig` with virtual pipeline parallelism and
  other Megatron-specific settings
- `areal/api/cli_args.py` - `TrainEngineConfig` with `megatron` field for
  Megatron-engine configuration

**Utilities**:

- **Distributed Coordination**: Process group management and communication coordination
- **Pipeline Scheduling**: 1F1B (one-forward-one-backward) and interleaved pipeline
  schedules
- **Gradient Synchronization**: Cross-stage gradient aggregation and optimization

**Integration Points**:

- **Weight Synchronization**: Broadcast updates to rollout engines via XCCL or
  disk-based mechanisms
- **Checkpoint Coordination**: Distributed checkpoint saving/loading across all parallel
  dimensions
- **Monitoring and Metrics**: Performance tracking across pipeline stages

## Resources

- **Main implementation**: `areal/engine/megatron_engine.py`
- **Configuration**: `areal/api/cli_args.py` (`TrainEngineConfig` with
  `MegatronEngineConfig`)
- **Checkpointing**: `areal/engine/megatron_utils/checkpointer.py`
- **Utilities**: `areal/engine/megatron_utils/megatron.py` for core Megatron utilities
- **Examples**: YAML configuration files in `examples/` and
  `tests/sft/config_megatron.yaml`
