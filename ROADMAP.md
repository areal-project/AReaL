# AReaL Roadmap

This roadmap outlines the planned features and improvements for AReaL in the next
quarter. We welcome community feedback and contributions to help shape the future
direction of the project.

**Latest Release:** Check [releases](https://github.com/areal-project/AReaL/releases)
for the most recent version.

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
- [ ] Omni model RL support with FSDP/Archon backend
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

**Last Updated:** 2026-02-06

**Questions about the roadmap?** Open a discussion in
[GitHub Discussions](https://github.com/areal-project/AReaL/discussions) or ask in our
[WeChat group](./assets/figures/wechat_qrcode.png).
