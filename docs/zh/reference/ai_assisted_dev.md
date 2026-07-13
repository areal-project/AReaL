# AI 辅助开发

AReaL 开箱即用提供 [Codex](https://developers.openai.com/codex)、
[OpenCode](https://opencode.ai/) 和
[Claude Code](https://docs.anthropic.com/en/docs/claude-code)
三套一等配置。仓库中包含共享的项目指令、领域专家代理和可复用技能，以保证三套 harness 尽量保持一致。

本文以 Codex 作为仓库级工作流的主线说明；OpenCode 和 Claude Code 保持兼容支持。

## Codex

### 启动 Codex 会话

```bash
cd /path/to/AReaL
codex
```

Codex 会从 `AGENTS.md` 读取项目上下文。在本仓库中，Codex harness 的组织方式是：

- **项目指令**：`AGENTS.md`
- **仓库级 skills**：`.agents/skills/<name>/SKILL.md`
- **自定义 agents**：`.codex/config.toml`，以及 `.codex/agents/<name>.toml` 和
  `.codex/agents/<name>.md`

自定义 agent 先在 `.codex/config.toml` 里注册，再指向各自的 agent TOML：

```toml
[agents.fsdp-expert]
description = "FSDP2 expert for configuration, parallel strategy, weight sync, and integration guidance."
config_file = "./agents/fsdp-expert.toml"
```

每个 agent TOML 再指向自己的 markdown 指令文件：

```toml
name = "fsdp-expert"
model_instructions_file = "./fsdp-expert.md"
```

### Codex 的“可直接执行工作流”

在 AReaL 里，Codex 的“可直接执行工作流”就是放在 `.agents/skills/` 下的**仓库级 skill**。Codex 这里不把 `/`
命令作为主要工作流表面，而是让你请求相应的 skill， 然后由 Codex 端到端执行该流程。

| Skill                   | 用途                                             |
| ----------------------- | ------------------------------------------------ |
| `add-dataset`           | 向 `areal/dataset/` 添加新的数据集加载器         |
| `add-workflow`          | 创建新的 `RolloutWorkflow` 实现                  |
| `add-reward`            | 实现新的奖励函数                                 |
| `add-archon-model`      | 向 Archon 引擎添加新的模型架构                   |
| `add-unit-tests`        | 添加或扩展单元测试                               |
| `debug-distributed`     | 排查卡死、OOM、NCCL 和 launcher 配置问题         |
| `commit-conventions`    | 在提交前准备 Conventional Commit 信息            |
| `review-pr`             | 执行只读 PR 审查工作流并输出风险分析             |
| `create-pr`             | 变基、压缩提交、准备元数据并创建或更新 GitHub PR |
| `translate-doc-zh`      | 将 `docs/en/` 下的文档翻译到 `docs/zh/`          |
| `update-docker-image`   | 更新运行时镜像依赖并驱动 Docker PR 流程          |
| `upgrade-megatron-core` | 审计并升级 Megatron-Core 兼容性                  |
| `upgrade-vllm`          | 审计并升级 vLLM 兼容性                           |

这就是“Codex 在本仓库里可直接执行的工作流是什么”的答案。

### Codex 自定义 agents

Codex 自定义 agents 是在 `.codex/config.toml` 中注册的专用顾问。它和 OpenCode
的自动专家路由不同，属于显式调用的能力表面：你可以直接要求 Codex 咨询某个 agent， 或者依赖仓库指令在合适时机触发它们。

| Agent                  | 用途                                        |
| ---------------------- | ------------------------------------------- |
| `planner`              | 为多文件改动和架构变更制定实现计划          |
| `simple-code-reviewer` | 做快速的变更后风险检查                      |
| `code-verifier`        | 运行定向格式化、lint 和验证命令             |
| `fsdp-expert`          | 提供 FSDP2 配置、并行策略和权重同步指导     |
| `archon-expert`        | 提供 Archon 与 MoE 集成指导                 |
| `megatron-expert`      | 提供 Megatron 流水线并行训练指导            |
| `algorithm-expert`     | 提供 PPO、GRPO、DAPO、reward shaping 等指导 |
| `launcher-expert`      | 提供本地、Ray、Slurm 和推理 launcher 指导   |

### 典型 Codex 会话

```text
> 为多模态评测添加一个新的 rollout workflow

Codex: [加载 add-workflow，检查 areal/workflow/，实现修改]

> 审查这个 PR

Codex: [加载 review-pr，分析 diff，并咨询匹配的 expert agents]

> 创建或更新这个 PR

Codex: [加载 create-pr，检查分支状态，并执行 PR 工作流]
```

## OpenCode 兼容性

### 安装 OpenCode 和 oh-my-opencode

```bash
curl -fsSL https://opencode.ai/install | bash
```

其他安装方式包括 `brew install anomalyco/tap/opencode`、`npm install -g opencode-ai`
或直接下载发布二进制。具体以 OpenCode 官方文档为准。

[oh-my-opencode](https://github.com/code-yeongyu/oh-my-opencode) 仍然是可选增强项， 适合需要更强
OpenCode 编排和工具集成时使用。

### 启动 OpenCode

```bash
cd /path/to/AReaL
opencode
```

OpenCode 会读取 `AGENTS.md`，并从 `.opencode/` 目录发现 agents、commands、skills 和
plugins；在相关场景下它也可以读取 `.claude/skills/`。

### OpenCode 的工作流表面

OpenCode 维持自己的 command 和 agent 体系：

| 命令                | 用途                     |
| ------------------- | ------------------------ |
| `/create-pr`        | 变基、压缩提交并创建 PR  |
| `/review-pr`        | 智能代码审查，带风险分析 |
| `/translate-doc-zh` | 将英文文档翻译为中文     |

OpenCode 相关资产位于 `.opencode/agents/`、`.opencode/command/`、 `.opencode/skills/` 和
`.opencode/package.json`。

## Claude Code 兼容性

Claude Code 继续通过 `CLAUDE.md` 与 `.claude/` 目录提供兼容支持，其中包括 agents、commands、hooks 和 rules。

## 配置文件

```
AReaL/
|-- AGENTS.md                # 项目上下文（自动加载）
|-- .agents/
|   +-- skills/              # Codex 仓库级技能
|-- .codex/
|   |-- config.toml          # 注册自定义 Codex 子代理
|   +-- agents/              # Codex agent 配置和指令文件
|-- .opencode/
|   |-- agents/              # OpenCode expert agents
|   |-- command/             # OpenCode slash commands
|   |-- skills/              # OpenCode skills
|   +-- package.json         # OpenCode plugin 依赖
+-- .claude/
    |-- agents/              # Claude Code agents
    |-- commands/            # Claude Code commands
    |-- hooks/               # Claude Code hooks
    +-- rules/               # Claude Code rules
```

## Harness 对照

AReaL 同时支持三套 harness，但三者暴露可复用工作流的方式不同：

| 概念                 | Codex                                   | OpenCode                                  | Claude Code                                       |
| -------------------- | --------------------------------------- | ----------------------------------------- | ------------------------------------------------- |
| 项目上下文           | `AGENTS.md`                             | `AGENTS.md`                               | `CLAUDE.md`                                       |
| 仓库级工作流         | `.agents/skills/`                       | `.opencode/skills/`、`.opencode/command/` | `.claude/skills/`、`.claude/commands/`            |
| 自定义子代理         | `.codex/config.toml` + `.codex/agents/` | `.opencode/agents/`                       | `.claude/agents/`                                 |
| 主要可执行工作流形态 | 仓库级 skill                            | slash command 或 skill                    | slash command 或 skill                            |
| 代理调度方式         | 显式调用 custom agent                   | `task(subagent_type="...", ...)`          | 自动路由                                          |
| 专家命名             | `fsdp-expert`、`archon-expert`、...     | `fsdp-expert`、`archon-expert`、...       | `fsdp-engine-expert`、`archon-engine-expert`、... |
| 提交辅助             | `commit-conventions` skill              | `commit-conventions` skill                | `commit-conventions` skill + `/gen-commit-msg`    |

Claude Code 还提供一些额外的通用代理：

| 代理                   | 用途                                   |
| ---------------------- | -------------------------------------- |
| `planner`              | 在复杂多文件更改之前创建实现计划       |
| `code-verifier`        | 在代码更改后运行 pre-commit 钩子和测试 |
| `simple-code-reviewer` | 在提交前执行快速代码质量检查           |

Claude Code 配置位于：

```
AReaL/
|-- CLAUDE.md              # 项目上下文和约束
+-- .claude/
    |-- agents/            # 5 个领域专家 + 3 个通用代理（共 8 个）
    |-- skills/           # 引导式工作流（与 OpenCode 共享）
    |-- commands/         # 自动化操作（create-pr、gen-commit-msg、review-pr）
    |-- hooks/            # 预/后操作钩子
    +-- rules/            # 代码质量标准
```

## 贡献

我们欢迎对代码库和 AI 开发配置的贡献：

- **代码贡献**：新功能、bug 修复、文档改进
- **AI 配置贡献**：新增 Codex skills、custom agents，或兼容层改进
- **Codex 配置**：编辑 `.codex/`、`.agents/` 和 `AGENTS.md`
- **OpenCode 配置**：编辑 `.opencode/` 中的文件
- **Claude Code 配置**：编辑 `.claude/` 中的文件

参见 [CONTRIBUTING.md](https://github.com/areal-project/AReaL/blob/main/CONTRIBUTING.md)
获取指南。
