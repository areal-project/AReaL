# AReaL Roadmap

This roadmap outlines the planned features and improvements for AReaL in the next
quarter. We welcome community feedback and contributions to help shape the future
direction of the project.

**Latest Release:** Check [releases](https://github.com/areal-project/AReaL/releases)
for the most recent version.

## 2026 H2 Roadmap (due December 31, 2026)

[GitHub Issue #1381](https://github.com/areal-project/AReaL/issues/1381).

This roadmap tracks major planned enhancements for the second half of 2026. Items are
organized into two categories:

- **On-going:** Features currently under active development by the core AReaL team
- **Planned but not in progress:** Features that are good to have but currently lacking
  bandwidth

We use `[CC]` to mark items suitable for community contributions. If you're interested
in contributing, please reach out to discuss implementation details.

### Backends

**On-going**

- [ ] Support training the latest large MoE models with the Megatron backend (including
  Kimi 2.5 / GLM 5 / DeepSeek V3 / Qwen 3.6, etc.)
  ([#1372](https://github.com/areal-project/AReaL/pull/1372),
  [#1373](https://github.com/areal-project/AReaL/pull/1373))
- [ ] More distribution strategies for colocated deployment

**Planned but not in progress**

- [ ] Omini model RL support with FSDP backend
- [ ] Proxy server support via RayScheduler (HTTP launcher actor) `[CC]`
- [ ] Weight delta update with `awex`
- [ ] Memory service for self-evolving agent

### Usability

**On-going**

- [ ] Effective AReaL CLI support
  ([#1374](https://github.com/areal-project/AReaL/issues/1374))
- [ ] Profiling toolkit (distributed trace)
- [ ] Online RL training example based on AReaL 2.0 architecture

**Planned but not in progress**

- [ ] More RL post-training paradigms: IcePop Plus algorithm (verified in Ling MoE
  training) and stepwise reward rubrics
- [ ] GUI Agent VLM training example
- [ ] OS-bench VLM training example `[CC]`
- [ ] Diffusion image/video generation model RL post-training `[CC]`
- [ ] Comprehensive LoRA support over the existing policy model training
- [ ] AReaL autopilot (automatic performance-optimized RL deployment suggestions and
  systematic bottleneck analysis)

### Documentation

**Planned but not in progress**

- [ ] Explain benchmarking results from nightly CI and how to extract issues

## 2026 Q2 Roadmap (due July 31, 2026)

[GitHub Issue #1302](https://github.com/areal-project/AReaL/issues/1302).

This roadmap tracks major planned enhancements through July 31, 2026. Items are
organized into two categories:

- **On-going:** Features currently under active development by the core AReaL team
- **Planned but not in progress:** Features that are good to have but currently lacking
  bandwidth

We use `[CC]` to mark items suitable for community contributions. If you're interested
in contributing, please reach out to discuss implementation details.

### Backends

**On-going**

- [ ] Full training example with AReaL 2.0 architecture
- [ ] Deprecate support for SPMD mode (launcher, sglang/vllm server, etc.)
- [ ] Initial support of colocation weight transfer with `awex`

**Planned but not in progress**

- [ ] Migrate primary Megatron integration lib from `mbridge` to `megatron-bridge`
  ([#1260](https://github.com/areal-project/AReaL/issues/1260)) `[CC]`
- [ ] Full support for colocation/separation weight transfer with `awex` as the backend
  (`areal/v2/weight_update/`) `[CC]`
- [ ] Migrate legacy NCCL broadcast weight transfer approach from `areal/engine` into
  `areal/v2/weight_update/` `[CC]`
- [ ] Omini model RL support with FSDP backend
  ([#879](https://github.com/areal-project/AReaL/issues/879)) `[CC]`
- [ ] Support training the latest large MoE models with the Megatron backend, including
  dpsk-v3/v4, Kimi-2.5, GLM-4/5 `[CC]`
- [ ] Native Kubernetes (K8S) scheduler `[CC]`

### Usability

**On-going**

- [ ] Nightly CI workflow for performance benchmarking
  ([#1284](https://github.com/areal-project/AReaL/issues/1284))
- [ ] Refactor unit tests for faster execution

**Planned but not in progress**

- [ ] OS-bench VLM training example `[CC]`
- [ ] Multi-agent training example (single LLM, different prompts, e.g., planner agent
  with sub-agents) `[CC]`
- [ ] Migrate legacy multi-turn agent examples to new API (`agenerate` → `ArealOpenAI`
  or URL-based `AgentWorkflow`) `[CC]`
- [ ] Publish PyPI packages and CLI for running experiments
- [ ] Support distributed training and debugging in Jupyter notebooks
- [ ] Implement controller construction with model-centric API similar to `transformers`

### Documentation

**Planned but not in progress**

- [ ] Explain benchmarking results from nightly CI and how to extract issues
- [ ] Document AReaL 2.0 architecture

## 2026 Q1 Roadmap (due April 30, 2026)

[GitHub Issue #907](https://github.com/areal-project/AReaL/issues/907).

This roadmap tracks major planned enhancements through April 30, 2026. Items are
organized into two categories:

- **On-going:** Features currently under active development by the core AReaL team
- **Planned but not in progress:** Features with concrete implementation plans where we
  welcome community contributions

### Backends

**On-going**

- [ ] ZBPP & ZBPP-V support for the Archon backend
- [ ] FP8 training for Archon

**Planned but not in progress**

- [ ] Support for agentic training with large VLM MoE models (Archon backend)
- [ ] Omini model RL support with FSDP/Archon backend
- [ ] Decoupling agent service from the inference service
- [ ] Online RL training with the proxy server
- [ ] LoRA support for the Archon backend
- [ ] Colocation mode with `awex` as the weight sync engine
- [ ] Multi-LLM training (different agents with different parameters)
- [ ] Auto-scaling inference engines in single-controller mode
- [ ] Elastic weight update setup and acceleration
- [ ] RL training with cross-node vLLM pipeline/context parallelism

### Usability

**On-going**

- [ ] Flatten the import structure of areal modules

**Planned but not in progress**

- [ ] Publishing PyPI packages
- [ ] Support distributed training and debugging in Jupyter notebooks
- [ ] Example of using a generative or critic-like reward model
- [ ] Support directly constructing inference/training engines without config objects
- [ ] Add router in rollout controller for simpler proxy server usage
- [ ] Integrate `aenvironment` for environment handling

### Documentation

**Planned but not in progress**

- [ ] Use case guides: multi-agent training
- [ ] Guide for online proxy mode training

## Historical Roadmaps

### 2025 Q4

[GitHub Issue #542](https://github.com/areal-project/AReaL/issues/542).

**Backends**

Completed:

- Single-controller mode
- Detailed profiling for optimal performance across different scales
- Low-precision RL training (Megatron FP8)
- Data transfer optimization in single-controller mode
- New PyTorch-native backend: Archon

Carried over to Q1 2026:

- Multi-LLM training (different agents with different parameters)
- Auto-scaling inference engines in single-controller mode
- Elastic weight update setup and acceleration
- RL training with cross-node vLLM pipeline/context parallelism

**Usability**

Completed:

- Add CI pipeline to build Docker images upon release
- Wrap training scripts into trainers
- Refactor FSDP/Megatron engine/controller APIs to finer granularity
- Fully respect allocation mode in trainers/training scripts

Carried over to Q1 2026:

- Flatten the import structure of areal modules
- Support distributed training and debugging in Jupyter notebooks
- Example of using a generative or critic-like reward model

Canceled:

- Rename `RemoteSGLang/vLLMEngine` as `SGLang/vLLMEngine`

**Documentation**

Completed:

- Tutorial on how to write efficient async rollout workflows
- Benchmarking and profiling guide
- Use case guides: offline inference, offline evaluation
- AReaL performance tuning guide
  - Device allocation strategies for training and inference
  - Parallelism strategy configuration for training and inference

Carried over to Q1 2026:

- Use case guides: multi-agent training

### 2025 Q3

[GitHub Issue #257](https://github.com/areal-project/AReaL/issues/257).

**Backends**

Completed:

- Megatron training backend support
- SGLang large expert parallelism (EP) inference support
- Remote vLLM inference engine
- Ulysses context parallelism & tensor parallelism for FSDP backend
- End-to-end MoE RL training with large EP inference and Megatron expert parallelism
- Distributed weight resharder for Megatron training backend

Canceled:

- Local SGLang inference engine with inference/training colocation (hybrid engine)
- RL training with SGLang pipeline parallelism

**Usability**

Completed:

- OpenAI-compatible client support
- Support RLOO
- Provide benchmarking configuration examples:
  - DAPO
  - Bradley-Terry reward modeling
  - PPO with critic models
  - REINFORCE++

**Documentation**

Completed:

- OpenAI-compatible client documentation
- Out-of-memory (OOM) troubleshooting guide
- AReaL debugging best practices:
  - LLM server-only debugging - How to launch LLM servers independently and debug agent
    workflows
  - Mock data and torchrun debugging - Creating synthetic data and using `torchrun` for
    algorithm debugging
  - Training-free evaluation experiments - Running evaluations without training or
    additional GPUs

## How to Influence the Roadmap

We value community input! Here's how you can help shape AReaL's future:

### 💡 Propose New Features

1. **Check Existing Issues:** Search
   [issues](https://github.com/areal-project/AReaL/issues) and
   [discussions](https://github.com/areal-project/AReaL/discussions) to see if your idea
   already exists
1. **Create a Feature Request:** Use our
   [feature request template](https://github.com/areal-project/AReaL/issues/new?template=feature.md)
1. **Discuss in GitHub Discussions:** Post in
   [Ideas category](https://github.com/areal-project/AReaL/discussions/categories/ideas)
   for early feedback
1. **Vote on Features:** Use 👍 reactions on issues to show support

### 🛠️ Contribute Implementation

Check our [contribution guide](CONTRIBUTING.md).

## Release Cycle

**Minor Releases:** Bi-weekly - Bug fixes, small improvements, and new features

**Major Releases:** Quarterly - Important milestones and significant changes

## Historical Milestones

Check [our historical milestone summaries since open-source](docs/version_history.md).

## Long-Term Vision

Our vision for AReaL is to become the **go-to framework for training reasoning and
agentic AI systems** that is:

1. **Accessible:** Easy to get started, whether you're a researcher or practitioner
1. **Scalable:** Scales from laptop to 1000+ GPU clusters seamlessly
1. **Flexible:** Supports diverse algorithms, models, and use cases
1. **Performant:** Industry-leading training speed and efficiency
1. **Open:** Fully open-source with transparent development

______________________________________________________________________

**Last Updated:** 2026-06-05

**Questions about the roadmap?** Open a discussion in
[GitHub Discussions](https://github.com/areal-project/AReaL/discussions) or ask in our
[WeChat group](./assets/figures/wechat_qrcode.png).
