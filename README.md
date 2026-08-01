<h1 align="center">
<em>AReaL</em>: A Large-Scale Asynchronous Reinforcement Learning System
</h1>

<p align="center">
| <a href="https://arxiv.org/pdf/2505.24298"><b>Paper</b></a> | <a href="https://areal-project.github.io/AReaL/"><b>Documentation</b></a> | <a href="https://areal-project.github.io/AReaL/zh/"><b>中文文档</b></a> | <a href="https://deepwiki.com/areal-project/AReaL"><b>Ask DeepWiki</b></a> | <a href="https://huggingface.co/collections/inclusionAI/"><b>🤗 Models & Data</b></a> |
<a href="./assets/figures/wechat_qrcode.png" target="_blank"><img src="./assets/figures/wechat_icon.png" width="20" style="vertical-align: middle;"> <b>WeChat (微信) Group</b></a> |
  <a href="https://www.bestpractices.dev/projects/12770"><img src="https://www.bestpractices.dev/projects/12770/badge"></a>
</p>

<img align="right" alt="ReaL" src="/assets/figures/logo.png" width="20%">

**AReaL** is a reinforcement learning (RL) infrastructure designed to bridge foundation
model training with modern agent-based applications. It was originally developed by
researchers and engineers from Tsinghua IIIS and the AReaL Team at Ant Group.

Built on a **fully asynchronous RL training paradigm**, AReaL is optimized for
efficiency and scalability, making it particularly well-suited for training large-scale
**reasoning and agentic models**.

AReaL’s mission is to make building AI agents **accessible, efficient, and
cost-effective** for a broad community of developers and researchers.

> Like milk tea - customizable, scalable, and enjoyable - we hope AReaL brings both
> flexibility and delight to your AI development experience. Cheers!

**AReaL Highlights**

- ⚡ **Flexibility**: Seamless customization for
  [agentic RL](https://areal-project.github.io/AReaL/en/tutorial/agentic_rl.html) and
  [online RL training](https://areal-ai.io/en/tutorial/online_proxy.html) for
  **black-box agent applications** by simply replacing the `base_url`.
- 📈 **Scalability**: **Stable** fully asynchronous RL training with **industry-leading
  speed**.
- ✨ **Cutting-Edge Performance**: State-of-the-art [math](/blog/AReaL_v0_2.md),
  [coding](/blog/AReaL_v0_3.md), [search](https://github.com/inclusionAI/ASearcher), and
  [customer service](https://arxiv.org/abs/2601.22607) agents.

## 📰 News

**\[2026/07/01\]** 🎉 We release **AReaL 2.0**, a major architectural milestone. This
release refactors AReaL into a **microservice architecture** with independent
[training](./areal/v2/training_service/), [inference](./areal/v2/inference_service/),
[agent](./areal/v2/agent_service/), and [weight-update](./areal/v2/weight_update/)
services, and ships the [Hermes online RL loop](./examples/hermes) and
[end-to-end SWE RL training examples](./examples/swe). Check out our
[technical report](https://arxiv.org/abs/2607.01120).

**\[2026/06/17\]** 🔬 Introducing **KPop** — bidirectional binary KL divergence token
masking. Configured via `rejection_sampling.metric=binary_kl`. Also adding an
[IcePop config](examples/math/gsm8k_icepop.yaml) (importance-ratio-based token masking).
Check out [gsm8k_kpop.yaml](examples/math/gsm8k_kpop.yaml) and
[gsm8k_icepop.yaml](examples/math/gsm8k_icepop.yaml) to get started!

<details>
<summary><b>📋 Previous Releases</b></summary>

**\[2026/04/23\]** 🚀 We’re excited to release our integration with
[Scaffoldings](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/scaffolding)
for agentic RL training - now live in our
[examples](https://github.com/areal-project/AReaL/tree/main/examples/scaffolding)! Huge
shoutout to @narutolhy and @WeiHaocheng for making this happen 🙌. The modular design of
the Scaffoldings enables it to achieve a thorough decoupling of agent execution, reward
calculation, and trajectory acquisition. This enables developers to reuse existing
modules when implementing an agentic RL method, allowing them to focus on their own
innovative modules.

**\[2026/04/18\]** We are thrilled to announce that **AReaL's first Community Biweekly
Meeting** was successfully held! Thank you to everyone who joined us. Meeting materials
are now available
[here](https://github.com/areal-project/AReaL/tree/main/assets/community). Our next
meeting is scheduled for **2026/05/01** and will also be conducted in Chinese;
English-language meetings will be scheduled in the future. We warmly welcome everyone to
participate! See [Community](./assets/community/README.md) for more details.

**\[2026/03/02\]** We provide [a complete example](./examples/openclaw/) to train your
own 🦞 OpenClaw agent by simply replacing the `base_url` and `api_key` with AReaL's RL
service - no complicated dependencies, no code changes, works with any agentic runtime!

**\[2026/02/06\]** We are delighted to introduce **AReaL-SEA**, a self-evolving data
synthesis engine. Combined with RL training on AReaL, the 235B MoE model surpasses GPT 5
and achieves comparable performance with Gemini 3.0 Pro on $\\tau^2$-bench! Check out
the [paper](https://arxiv.org/pdf/2601.22607),
[model](https://huggingface.co/inclusionAI/AReaL-SEA-235B-A22B),
[data](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data), and
[code](https://github.com/areal-project/AReaL/tree/main/examples/tau2).

**\[2026/01/15\]** Congrats to our friends at [CAMEL-AI](https://www.camel-ai.org/) for
open-sourcing [SETA](https://github.com/camel-ai/seta), their terminal agent RL project
trained with AReaL! Check out
[their training workflow](https://github.com/camel-ai/seta/tree/main/training/tbench_areal_workflow)
and the [announcement on X](https://x.com/guohao_li/status/2009678513574408636).

**\[2026/01/01\]** Happy New Year! Thanks to the outstanding contribution from
@HwVanICI, we are excited to officially announce stable support for AReaL training on
**Ascend NPU devices**! The code is actively maintained and continuously updated in the
[`ascend` branch](https://github.com/areal-project/AReaL/tree/ascend). Check out
[our documentation](https://areal-project.github.io/AReaL/en/tutorial/installation_npu.html)
to get started, and feel free to report any issues!

**\[2025/08/30\]** Introducing ASearcher, a state-of-the-art search agent built with
AReaL's end-to-end asynchronous RL training. Check out the [paper](assets/paper.pdf) and
the [open-source repository](https://github.com/inclusionAI/ASearcher)!

**\[2025/07/31\] (AReaL-lite)** We introduce AReaL-lite, a **lightweight** version of
AReaL designed specifically for AI researchers and rapid prototyping. AReaL-lite
features an **algorithm-first** API design that prioritizes ease of use and algorithm
development, while natively supporting **fully asynchronous agentic RL**. With 80% fewer
lines of code, AReaL-lite maintains 90% of AReaL's performance and core functionality.
Check out [our AReaL-lite design documentation](/areal/README.md) and
[the quickstart guide](https://areal-project.github.io/AReaL/en/tutorial/quickstart.html)
to begin your journey with **AReaL-lite**!

**\[2025/06/03\] (v0.3, boba²)** We release **boba²** (double-boba) for fully
asynchronous RL training, which achieves **2.77× speedup while delivering comparable or
superior training performance** compared to synchronous systems. Furthermore,
asynchronous RL significantly simplifies multi-turn agentic RL training setup! Check out
[our v0.3 overview blog](/blog/AReaL_v0_3.md) and the
[research paper](assets/paper.pdf).

**\[2025/03/31\] (v0.2, boba)** Introducing our milestone release—boba! Please call it
A-ReaL-boba! This release features significantly faster training with SGLang support and
state-of-the-art 7B and 32B models for mathematical reasoning. Check out our
[v0.2 technical blog](/blog/AReaL_v0_2.md).

**\[2025/02/24\] (v0.1)** Our initial release includes reproducible results for 1.5B and
7B Large Reasoning Models (LRMs). Check out our
[v0.1 technical blog](/blog/AReaL_v0_1.md).

</details>

## 🚀 Getting Started

First, install the package:

```bash
git clone https://github.com/areal-project/AReaL
cd AReaL
pip install uv
# Install flash-attn pre-built wheel first to avoid compiling from source
# (pick the wheel matching your Python version; see https://github.com/mjun0812/flash-attention-prebuild-wheels/releases)
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl"
uv sync --extra cuda  # installs training packages + SGLang (default inference backend)
# For vLLM instead: cp pyproject.vllm.toml pyproject.toml && cp uv.vllm.lock uv.lock && uv sync --extra cuda
```

Our training scripts automatically download the required dataset (openai/gsm8k) and
model (Qwen/Qwen2-1.5B-Instruct). To run on a single node:

```bash
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml scheduler.type=local
```

If you prefer to run experiments on a Ray cluster, update paths in the YAML file to
point to your shared storage, and run:

```bash
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml \
  cluster.n_nodes=2 cluster.n_gpus_per_node=8 \
  cluster.fileroot=/path/to/nfs \
  scheduler.type=ray
```

For comprehensive setup instructions, see
[our quickstart guide](https://areal-project.github.io/AReaL/en/tutorial/quickstart.html).

## 📚 Examples

### Math & Reasoning

| Task                                                | Description                                                                                  | Performance                                                       |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **[Math](examples/math/)**                          | GSM8K math reasoning with GRPO, PPO, DAPO, REINFORCE, RLOO, LitePPO, DR-GRPO, GSPO, and more | -                                                                 |
| **[Multi-Turn Math](examples/multi_turn_math/)**    | Multi-turn math agent with reward discounting across turns                                   | [Training Curve](examples/multi_turn_math/reward_curve.png)       |
| **[LoRA Math](examples/math/gsm8k_grpo_lora.yaml)** | Parameter-efficient math training with LoRA (SGLang/vLLM backends)                           | -                                                                 |
| **[Countdown](examples/countdown/)**                | Countdown numbers game with custom rewards                                                   | [Training Curve](examples/countdown/countdown_training_curve.png) |

### Agentic RL

| Task                                                     | Description                                                               | Performance                                                                  |
| -------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **[General Agent](examples/agent_workflow/)**            | General agentic training with any agentic frameworks                      | [Guide](docs/tutorial/agentic_rl.md)                                         |
| **[Hermes Online RL](examples/hermes/)**                 | End-to-end Online RL loop with Hermes agent under AReaL 2.0               | [Guide](examples/hermes/README.md)                                           |
| **[Coding Agent RL](examples/swe/)**                     | End-to-end coding-agent RL training with AReaL-SWEAgent/Claude Code Agent | [Guide](examples/swe/README.md)                                              |
| **[Tau2 Customer Service](examples/tau2/)**              | Customer service agent on Tau2-Bench (retail, airline, telecom)           | [Paper](https://arxiv.org/abs/2601.22607)                                    |
| **[Search Agent](examples/search_agent/)**               | End-to-end search agent with Tongyi-DeepResearch workflow                 | [Training Curve](examples/search_agent/tongyi_deepresearch/reward_curve.png) |
| **[Tool-Integrated Reasoning](examples/tir/)**           | Multi-turn tool calling during reasoning (Python executor, calculator)    | [Training Curve](examples/tir/figures/task_reward.png)                       |
| **[OpenAI Agents Integration](examples/openai_agents/)** | Integration with OpenAI Agents SDK for agentic workflows                  | -                                                                            |
| **[CAMEL-AI Integration](examples/camel/)**              | Integration with CAMEL-AI framework for agentic RL                        | -                                                                            |

### Vision-Language Models

| Task                                | Description                                               | Performance                                     |
| ----------------------------------- | --------------------------------------------------------- | ----------------------------------------------- |
| **[VLM](examples/vlm/)**            | Geometry3K and CLEVR Count 70K visual reasoning with GRPO | -                                               |
| **[VLM on NPU](examples/vlm_npu/)** | VLM training on Huawei NPU hardware                       | [Benchmark Results](examples/vlm_npu/README.md) |

### Alignment & Infrastructure

| Task                                            | Description                                           | Performance                                       |
| ----------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------- |
| **[RLHF Reward Modeling](examples/alignment/)** | Bradley-Terry reward modeling on Anthropic HH-RLHF    | [Training Curve](examples/alignment/rw_curve.png) |
| **[SkyPilot Deployment](examples/skypilot/)**   | Cloud deployment with SkyPilot (GCP, AWS, Kubernetes) | [Screenshots](examples/skypilot/README.md)        |

## 🔧 Support Matrix

### 🧠 Algorithms

All RL algorithms support both asynchronous and synchronous versions by setting
`max_head_offpolicyness=0`. See [Asynchronous RL Guide](docs/en/algorithms/async.md).

| Algorithm                | Documentation                                 | Paper                                          | Configuration                                                     |
| ------------------------ | --------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------- |
| **GRPO**                 | [📖 Docs](docs/en/algorithms/grpo_series.md)  | [📄 Paper](https://arxiv.org/pdf/2402.03300)   | [🔗 GSM8K Example](examples/math/gsm8k_grpo.yaml)                 |
| **GSPO**                 | [📖 Docs](docs/en/algorithms/grpo_series.md)  | [📄 Paper](https://arxiv.org/abs/2507.18071)   | [🔗 GSM8K Example](examples/math/gsm8k_gspo.yaml)                 |
| **PPO**                  | [📖 Docs](docs/en/algorithms/grpo_series.md)  | [📄 Paper](https://arxiv.org/pdf/2203.02155)   | [🔗 GSM8K Example](examples/math/gsm8k_ppo.yaml)                  |
| **DAPO**                 | [📖 Docs](docs/en/algorithms/grpo_series.md)  | [📄 Paper](https://arxiv.org/abs/2503.14476)   | [🔗 GSM8K Example](examples/math/gsm8k_dapo_dynamic_bs.yaml)      |
| **LitePPO**              | [📖 Docs](docs/en/algorithms/grpo_series.md)  | [📄 Paper](https://arxiv.org/abs/2508.08221)   | [🔗 GSM8K Example](examples/math/gsm8k_liteppo.yaml)              |
| **Dr.GRPO**              | [📖 Docs](docs/en/algorithms/grpo_series.md)  | [📄 Paper](https://arxiv.org/abs/2503.20783)   | [🔗 GSM8K Example](examples/math/gsm8k_drgrpo.yaml)               |
| **REINFORCE++**          | -                                             | [📄 Paper](https://arxiv.org/pdf/2501.03262)   | [🔗 GSM8K Example](examples/math/gsm8k_reinforce.yaml)            |
| **RLOO**                 | [📖 Docs](docs/en/algorithms/grpo_series.md)  | [📄 Paper](https://arxiv.org/pdf/2402.14740v1) | [🔗 GSM8K Example](examples/math/gsm8k_rloo.yaml)                 |
| **SAPO**                 | [📖 Docs](docs/en/algorithms/grpo_series.md)  | [📄 Paper](https://arxiv.org/abs/2511.20347)   | [🔗 GSM8K Example](examples/math/gsm8k_sapo.yaml)                 |
| **IcePop**               | [📖 Docs](docs/en/algorithms/grpo_series.md)  | [📄 Blog](https://ringtech.notion.site/icepop) | [🔗 GSM8K Example](examples/math/gsm8k_icepop.yaml)               |
| **KPop**                 | [📖 Docs](docs/en/algorithms/grpo_series.md)  | [📄 Blog](https://ringtech.notion.site/kpop)   | [🔗 GSM8K Example](examples/math/gsm8k_kpop.yaml)                 |
| **M2PO**                 | [📖 Docs](docs/algorithms/m2po.md)            | [📄 Paper](https://arxiv.org/abs/2510.01161)   | [🔗 GSM8K Example](examples/math/gsm8k_m2po.yaml)                 |
| **DPO**                  | [📖 Docs](docs/en/algorithms/dpo.md)          | [📄 Paper](https://arxiv.org/abs/2305.18290)   | [🔗 HH-RLHF Example](examples/alignment/hhrlhf_dpo.yaml)          |
| **RLHF Reward Modeling** | -                                             | -                                              | [🔗 RLHF Example](examples/alignment/hhrlhf_rw.yaml)              |
| **SFT**                  | -                                             | -                                              | [🔗 GSM8K Example](examples/math/gsm8k_sft.py)                    |
| **Distillation**         | [📖 Docs](docs/en/algorithms/distillation.md) | [📄 Paper](https://arxiv.org/pdf/2506.02208)   | [🔗 GSM8K Example](examples/distillation/gsm8k_grpo_distill.yaml) |

### Models

| Model Family               | Megatron | PyTorch FSDP | PyTorch Archon | Notes                                                    |
| -------------------------- | -------- | ------------ | -------------- | -------------------------------------------------------- |
| **Qwen2/3**                | ✅       | ✅           | ✅             | -                                                        |
| **Qwen3-MoE**              | ✅       | ✅           | ✅             | -                                                        |
| **Qwen2.5-VL**             | ✅       | ✅           | ❌             | Vision-language model                                    |
| **Qwen3-VL**               | ✅       | ✅           | ❌             | Vision-language model                                    |
| **Gemma 3**                | ❌       | ✅           | ❌             | Vision-language model                                    |
| **Other Hugging Face LLM** | ❌       | ✅           | ❌             | Compatibility depending on the version of `transformers` |

Check the [AI Coding Assistant Guide](docs/reference/ai_assisted_dev.md) and
[Archon Reference](docs/tutorial/archon.md) for how to integrate new models into AReaL.

### Training Backends

| Backend            | DP          | Tensor Parallel | Sequence Parallel within TP | Context Parallel | Pipeline Parallel | Expert Parallel | 1D Sequence Packing | LoRA                             |
| ------------------ | ----------- | --------------- | --------------------------- | ---------------- | ----------------- | --------------- | ------------------- | -------------------------------- |
| **Megatron**       | ✅ (ZeRO-1) | ✅              | ✅                          | ✅               | ✅                | ✅              | ✅                  | ✅ (with vLLM inference backend) |
| **PyTorch FSDP**   | ✅ (FSDP2)  | ✅              | ✅                          | ✅               | ❌                | ❌              | ✅                  | ✅                               |
| **PyTorch Archon** | ✅ (FSDP2)  | ✅              | ✅                          | ✅               | ✅                | ✅              | ✅                  | ❌                               |

### Inference Backends

| Backend    | Tensor Parallel | Context Parallel | Pipeline Parallel | Data Parallel Attention | Expert Parallel |
| ---------- | --------------- | ---------------- | ----------------- | ----------------------- | --------------- |
| **vLLM**   | ✅              | ❓               | ✅                | ❓                      | ❓              |
| **SGLang** | ✅              | ❌               | ❌                | ✅                      | ✅              |

## 📖 Resources

### Tutorial

- [Installation](docs/en/tutorial/installation.md)
- [Quickstart](docs/en/tutorial/quickstart.md)
- [AReaL CLI Guide](docs/en/best_practices/cli_guide.md)
- [Agentic RL](docs/en/tutorial/agentic_rl.md)
- [Evaluation](docs/en/tutorial/eval.md)
- [Large MoE with Megatron](docs/en/tutorial/megatron.md)
- [Large MoE with PyTorch Archon](docs/en/tutorial/archon.md)

### Code Walkthrough

- [Running GRPO on GSM8K dataset](docs/en/tutorial/gsm8k_grpo.md)

### Best Practices

- [Improving Algorithm Performance](docs/en/best_practices/algo_perf.md)
- [Agent Workflow Best Practices](docs/en/best_practices/workflow.md)
- [Debugging](docs/en/best_practices/debugging.md)
- [Handling OOM Issues](docs/en/best_practices/handling_oom.md)
- [Performance Profiling](docs/en/best_practices/perf_profiling.md)

### Customization

- [Customize Dataset](docs/en/customization/dataset.md)
- [Customize Agentic/RVLR Rollout Workflows](docs/en/customization/agent.md)

### Algorithms

- [Asynchronous RL Explained](docs/en/algorithms/async.md)
- [PPO, GRPO, and Related Algorithms](docs/en/algorithms/grpo_series.md)
- [M2PO](docs/en/algorithms/m2po.md)

### Reference

- [CLI Configurations](docs/en/cli_reference.md)
- [LoRA RL](docs/en/reference/lora.md)
- [Checkpointing](docs/en/reference/checkpointing.md)
- [Metrics Tracking](docs/en/reference/metrics_tracking.md)
- [Allocation Mode](docs/en/reference/alloc_mode.md)
- [Rollout Workflow](docs/en/reference/rollout_workflow.md)
- [Agent Workflow](docs/en/reference/agent_workflow.md)
- [AI-Assisted Development](docs/en/reference/ai_assisted_dev.md)

## 🤝 Contributing

We warmly welcome contributions from the community! Whether you're fixing bugs, adding
features, improving documentation, or helping others, your contribution is valued.
Please check our
**[Contributing Guide](https://github.com/areal-project/community/blob/main/CONTRIBUTING.md)**
for detailed information.

```bash
# Fork and clone the repository
git clone https://github.com/YOUR-USERNAME/AReaL
cd AReaL

# Install uv and sync dependencies
pip install uv
# Install flash-attn pre-built wheel to avoid compiling from source
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl"
# Use `--extra cuda` on Linux with CUDA (installs training packages + SGLang)
uv sync --extra cuda --group dev
# For vLLM instead:
# cp pyproject.vllm.toml pyproject.toml && cp uv.vllm.lock uv.lock && uv sync --extra cuda --group dev
# Or without CUDA support
# uv sync --group dev

# Set up pre-commit hooks (formatting, linting, commit message checks)
pre-commit install --install-hooks

# Make changes
git checkout -b feat/gpt-o5
git add .
# `git commit` will automatically check your files and commit messages
git commit -m "feat: implement gpt-o5 training loop"
git push
```

## 🗺️ Future Roadmap

- **[Full Roadmap](ROADMAP.md)**
- **[2026 H2 Roadmap](https://github.com/areal-project/AReaL/issues/1381)**
- **[2026 Q2 Roadmap](https://github.com/areal-project/AReaL/issues/1302)**

AReaL is under active development with planned minor releases weekly and major releases
monthly. We warmly welcome community engagement and contributions. We are also
**actively hiring interns and full-time employees** with open positions in both the US
and China.

## 📢 Community

<img src="./assets/figures/wechat_qrcode.png" width="200" alt="WeChat QR Code">

We warmly welcome everyone to join our community! We have a _**WeChat group**_ for
real-time discussions and a
[GitHub discussion board](https://github.com/areal-project/AReaL/discussions) for
asynchronous conversations.

Check out our [Community Repository](https://github.com/areal-project/community) for
more details and meeting materials.

## 🙏 Acknowledgments

We gratefully acknowledge that major contributors are from the AReaL Team at the
Institute for Interdisciplinary Information Sciences (IIIS), Tsinghua University and Ant
Group.

We have also received invaluable assistance from the following groups (listed
alphabetically):

- The Data Intelligence Lab at Ant Research for their data support

- @HwVanICI for support on vLLM, LoRA, NPU integration, and more

- The [Relaxed System Lab](https://github.com/Relaxed-System-Lab) at HKUST for seamless
  collaboration on numerous system-related aspects

- The [SGLang team](https://github.com/sgl-project/sglang) for supporting custom weight
  update features and their contributions during AReaL-lite development

- The Super Computing Technology (SCT) team at Ant Group for their expertise in
  large-scale cluster operations and maintenance

- Special thanks to @Lyken17 for providing valuable suggestions throughout the API
  design process

We also deeply appreciate all pioneering work from the community, particularly the
[ReaLHF](https://github.com/openpsi-project/ReaLHF) project from OpenPsi Inc. and other
outstanding projects, including but not limited to
[DeepScaleR](https://github.com/agentica-project/deepscaler),
[Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main),
[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF),
[VeRL](https://github.com/volcengine/verl),
[SGLang](https://github.com/sgl-project/sglang), [QwQ](https://github.com/QwenLM/QwQ),
[Light-R1](https://github.com/Qihoo360/Light-R1), and
[DAPO](https://github.com/BytedTsinghua-SIA/DAPO).

## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE).

## 📄 Citation

```bibtex
@inproceedings{mei2025real,
  author       = {Mei, Zhiyu and Fu, Wei and Li, Kaiwei and Wang, Guangju and Zhang, Huanchen and Wu, Yi},
  title        = {ReaL: Efficient RLHF Training of Large Language Models with Parameter Reallocation},
  booktitle    = {Proceedings of the Eighth Conference on Machine Learning and Systems,
                  MLSys 2025, Santa Clara, CA, USA, May 12-15, 2025},
  publisher    = {mlsys.org},
  year         = {2025},
}
```

```bibtex
@misc{fu2025areal,
      title={AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning},
      author={Wei Fu and Jiaxuan Gao and Xujie Shen and Chen Zhu and Zhiyu Mei and Chuyi He and Shusheng Xu and Guo Wei and Jun Mei and Jiashu Wang and Tongkai Yang and Binhang Yuan and Yi Wu},
      year={2025},
      eprint={2505.24298},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.24298},
}
```
