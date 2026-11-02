# AReaL Elastic

Elastic 包含原生训练 Infra、Reconfigurator 和 CPU 实验环境，独立于 AutoPilot。

```text
elastic/
├── contracts.py          # 共用观测、候选、预测及执行接口
├── layout.py             # 执行布局、布局比较与调整方案
├── serving.py            # TTFT／TBT／SLO 统计
├── infra/                # 原生运行、重配置执行、状态与观测
├── reconfigurator/       # Basic／Reference 决策与候选评估
└── simulation/           # CPU 实验执行及工作负载生成
```

Infra 和 Simulation 向 Reconfigurator 提供观测并接收目标布局；所选版本负责预测工作量、评估候选和选择配置。执行后继续采集新布局下的运行表现。

Basic v1 使用旧 JointSimulator；Reference 使用 Evaluator。候选评估与外层实验分别推进自己的运行状态，原生和模拟共用所选决策实现。

原生通过 `PPOTrainer → OnDemandActor` 衔接训练、发布和 checkpoint；初始化后的 Actor 保持驻留。`handoff.mode` 保留
Hot／Cold，原生当前执行 Cold。真实逐 token TTFT／最大 TBT、GPU 动态交接与性能收益另行验收。

AutoPilot 的 revision loop 在这些能力之上提出、评测和激活 Reconfigurator 的新版本。

[Infra](infra/README.md) · [重配置执行](RECONFIGURATION.md) ·
[Reconfigurator](reconfigurator/README.md) · [Simulation](simulation/README.md) ·
[Revision](../autopilot/domains/elastic/README.md) · [项目进度](../../PROGRESS.md)
