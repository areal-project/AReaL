# Online Reference revision

本目录将通用代码修订流程接到 `reference` Reconfigurator：①按真实配置和计时边界冻结历史，分开
calibration、validation、final holdout；②评估完整组件模型 factory，接受后复核可执行预测；③固定该模型，再评估 optimizer
的候选搜索；④将两份源码、评测记录和运行配置打包，由模拟入口或原生 adapter 加载。

```text
ReconfiguratorRevisionCycle.run()
├── # 1. 固定历史与修订上下文；final holdout 不参与修订
│   HistoricalModelEvaluator
│   HistoricalSourceEvaluator
├── # 2. 修订完整 Online 组件模型
│   _stage() → RevisionLoop._run()
│   ├── agent.propose() → build_model(calibration_cases, settings) 源码
│   └── HistoricalSourceEvaluator.evaluate()
│       ├── CandidateRunner.run() → worker.main()
│       │   └── OnlineHistoryPredictor.predict()
│       │       → models.actor() / service() / publication()
│       └── HistoricalModelEvaluator.evaluate() / compare()
│   HistoricalSourceEvaluator.build() / verify_loaded()
├── # 3. 用同一个模型评估和修订 Online 搜索
│   OnlineOptimizerEvaluator
│   _stage() → RevisionLoop._run()
│   └── OnlineOptimizerEvaluator.evaluate()
│       ├── worker.main() → build_optimizer(settings).search()
│       │   └── ReferenceProxy → 父进程 SearchPort
│       │       → Evaluator.evaluate_wave()
│       └── 父进程复算所选配置、SLO、Actor 供给、GPU 成本和查询次数
└── # 4. 导出两份完整组件及其绑定证据
    export_accepted_bundle() → load_bundle()
    ├── build_reconfigurator("reference", optimizer=VerifiedOnlineOptimizer(...))
    ├── 模拟：OnlineSimulation.create(..., models=frozen_truth,
    │                         prediction_models=loaded.models,
    │                         reconfigurator=loaded.reconfigurator)
    └── 原生：RevisionAdapter.prepare_reconfigurator()
        → apply_reconfigurator() → _install()
        → 下一轮使用同一组模型和 optimizer
```

## 修订对象

- **模型**：`build_model(calibration_cases, settings)` 返回完整 Online 组件对象，提供
  `source_ids`、`rollout_config()`、`service()`、`actor()`、`publication()`。历史评测和候选
  Simulator 使用同一个对象。模型可以改变耗时预测；训练语义、事件执行和 GPU 记账由固定代码管理。
- **Optimizer**：`build_optimizer(settings)` 返回
  `search(evaluator, view, forecasts, layouts, preferred=(), trigger=...)`。搜索通过模型端口排序和查询候选，返回端口给出的预测；SLO／Actor
  供给约束和最终选择规则固定。`VerifiedOnlineOptimizer` 在部署时保留同一端口、候选范围及查询预算。

模型接受要求相同已覆盖测量上的误差下降，同时检查覆盖和分组回归。Optimizer 接受使用父进程的预测与成本，要求满足固定质量约束、GPU
成本不回归，并改善成本／查询评分。模型变更后，当前 optimizer 也要在新模型上重新评估。

## 文件分工

- `model_evaluator.py`：历史输入、分区、测量、误差和接受规则。
- `model_adapters.py`、`model_execution.py`：组件探针、受限 CPU worker 和加载复核。
- `optimizer_evaluator.py`：搜索端口、参考评分及部署包装。
- `cycle.py`：有预算、可恢复的模型→optimizer 编排。
- `bundle.py`、`loaders.py`：源码、校准身份、上下文与接受证据绑定，以及统一装配。
- `native.py`：原生已发布更新边界的预加载、探测和原子激活；保留运行历史和训练状态。
- `online.py`：从已提交历史启动一个后台 CPU 修订任务；调用方提供 `problem_builder`，明确如何将已有测量映射成评测输入。
- `contracts.py`、`worker.py`：factory 协议、JSON DTO 和父进程模型 RPC。

`basic` 为 Joint 基线 Basic v1；当前修订接入仍作用于 `reference`，后续再从 Basic 验证 revision。通用
proposer、LLM、执行器和版本存储仍在 `autopilot/revision/`。原生逐 token 指标采集另列待办，本轮不修改推理通路。

[项目进度](../../../../PROGRESS.md) · [CPU 修订示例](../../../../examples/autopilot/revising.py)
