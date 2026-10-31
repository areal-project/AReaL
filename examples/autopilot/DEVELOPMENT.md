# Elastic 双机开发交接

共同仓库：`git@github.com:areal-project/AReaL.git`；集成分支：`lws/feat/elastic`。

| 分支                               | 职责                             |
| ---------------------------------- | -------------------------------- |
| `lws/feat/elastic`                 | 共同集成                         |
| `lws/feat/elastic-reconfig`        | online重配置与共用simulation开发 |
| `lws/feat/elastic-reconfig-inline` | inline模拟、CV触发与双周会结果   |
| `lws/feat/elastic-envarena`        | 公司Arena环境与任务工作流适配    |
| `feat/elastic-demo`                | 基于v2.1.0的独立SWE inline演示   |
| `feat/rl-perf-trace-profile`       | 基于较新上游的训练性能采集       |

旧Elastic／Expert的37个提交已压缩为`55eb9118`，与原`9a9cbfc6`源码树一致；本地及fork远端的旧backup／Expert分支已清理。
本地已移除`fork`远端及旧分支`feature/autopilot-elastic`，仅使用`origin`；个人GitHub仓库保留。inline工作保存到独立分支，日常开发回到重配置分支。
当前分支状态见[进度](../../PROGRESS.md)及其分支清单；实验原始数据和验证归档继续保留。

## 公司电脑接入

在公司端现有 checkout 中先执行 `git status --short`、`git remote -v`。
本机尚未提交的工作先保存到独立工作分支；未推送的提交继续保留在原分支。`origin`应指向共同仓库。若它当前指向个人fork，可执行：

```bash
git remote set-url origin git@github.com:areal-project/AReaL.git
```

首次建立本地集成分支：

```bash
git fetch origin
git switch --track -c lws/feat/elastic origin/lws/feat/elastic
```

若本地分支已存在，改用 `git switch lws/feat/elastic`，核对其跟踪关系后执行 `git pull --ff-only`。
两台机器均通过`origin`同步；并行任务使用各自任务分支。

共享分支已清除误推的修复／撤回两个提交，远端回到`7bab3a80`，源码不变。公司端若仍停在`c382abe0`，需先保存待保留工作，再对齐`origin/lws/feat/elastic`；普通pull不会主动回退本地分支。

新目录可直接克隆：

```bash
git clone --branch lws/feat/elastic git@github.com:areal-project/AReaL.git AReaL
```

## 日常开发与环境

- 开工前 `git pull --ff-only`，完成后提交并 `git push`；两台机器并行处理不同任务时，分别使用任务分支再集成，保留共享分支历史。
- checkout 路径、虚拟环境、模型与 SIF 可沿用。`TASK_CODE_ROOT` 指向当前 checkout；模型、题单、结果及凭据由各机器环境变量配置。
- 公司端继续使用宿主 Python≥3.6.8、训练 SIF 内 Python3.12 和 `srun --mpi=none`；工具沙箱可设置
  `TASK_PROC_MODE=none`。
- 按[Slurm／Singularity步骤](../task_workflows/RUN_SLURM_SINGULARITY.md)先验收 CPU
  工具，再进行16题评测与训练冒烟；[2×2步骤](../task_workflows/RUN_2X2_8GPU.md)覆盖四种组合。
- 固定题单沿用已有 manifest／摘要，不重新抽样；模型、数据、镜像、checkpoint 与完整 trace 留在本机或共享存储。

## 可携带的验证摘要

| 范围              | 已验证                                                                                                  | 待完成                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| native elastic    | 固定布局 FSDP／Megatron 的按需闭环、完整状态恢复与发布已有 GPU 验证                                     | 任意拓扑调整、完整在线 Reference 策略及收益实验                                            |
| 共享任务工作流    | 41项离线正常路径；真实Python3.6.8宿主／3.12客户端的14项跨版本检查；两种proc模式、stdout／stderr分流通过 | Docker／Apptainer命令及模型响应在这些检查中为替身，完整公司环境仍待验收                    |
| 1.7B 常驻 PPO     | 16题／4更新历史冒烟通过                                                                                 | 原batch40正式运行第4次更新OOM，仅3／125更新且无checkpoint；先验证原失败规模与64K完整前后向 |
| trace／simulation | 逐调用时间、usage长度、工具依赖和导出工作量已记录；prefill／decode batch采集默认关闭                    | batch采集GPU兼容性／开销、消费边界关联与Expert的1.7B dense适配                             |

历史详细状态见[迁移前进度快照](https://github.com/areal-project/AReaL/blob/86323615a3a62fc22c7371b1ceaba526ec1ada93/PROGRESS.md)。
其中 `../artifacts` 是个人工作区的外部证据目录，不随 Git 同步；本页摘要可在公司电脑读取，完整证据需另行提供。 新实验保存 Git
commit、运行配置、输入摘要和日志位置；当前状态统一更新根目录 `PROGRESS.md`。
