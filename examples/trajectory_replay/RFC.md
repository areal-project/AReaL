# RFC：训练轨迹批次保存与离线回放

## 0. 身份、范围与授权

版本：0.6；状态：已有 v1/v2 真实 GPU 结果及正常训练基线，已补齐行为证据索引；按 v0.5 流程重新独立核验，尚未完成验收。

| 项目               | 记录                                                                                                          |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| 代码库             | `areal-project/AReaL`，目标 `main`                                                                            |
| 初始基线           | `fa8bd7895d97659b4d919687210f9123f4273139`                                                                    |
| 分支               | `feat/trajectory-dump-replay-final`                                                                           |
| 基线核对           | 初次于 2026-10-28 fetch；2026-10-29 再次核对 `origin/main`，工作区 parent 见下文                              |
| 需求参考           | [PR #1407](https://github.com/areal-project/AReaL/pull/1407)，head `6250a6e8c0ac17336f5f272c2267014f60eb0409` |
| 当前目标 main      | `05246919c0055bc613efc9482a02f77ccb27fcf5`，2026-10-29 fetch 后安全更新                                       |
| 开发规范           | `areal-development-v0.5`                                                                                      |
| RFC / 通用评审规范 | `areal-feature-rfc-v0.5` / `areal-pr-review-v0.5`                                                             |
| 独立评审流程       | `areal-independent-verify-v0.2`，受控只读 Reviewer、逐轮复盘与发布 gate                                       |
| 经验读取与复盘     | `areal-development-learning-v0.1`；RFC、实验和 verify 前读取并记录采用，私有证据单独保存                      |
| 远端执行规范       | 执行规范 v0.2（精确摘要见私有记录），受控入口未修改                                                           |
| 依赖声明           | 仓库声明 SGLang `0.5.10.post1`、Megatron-Core `0.19.0`、Awex `0.9.0`；安装环境另行核实                        |
| 私有执行输入       | allocation、账号、模型/数据/写入路径、入口和运行命令单独保存在操作员证据中，不进入公开 PR                     |

当前开发工作区在最新 main 上重放本任务两项已验证修复，parent 为 `8f0e4b09ee2c9ff999a7adcb0a1e1e16b323e5ba`。历史独立
`fix/rollout-resume-log-guard` 提交 `53bdb54fe95a8f31d6628f136f52d89972ef7c00`，其父提交为独立恢复修复
`c1fcc45b89807a85604a7801f0b13cadf0c2e62a`（#1841）。前者只约束无推理时的最终步日志； 后者只在无推理恢复时还原训练策略版本。真实
Megatron checkpoint/优化器恢复探针确认版本从错误的 0 变为应有的 2， 修复前后全部 311 个权重张量均与源 checkpoint 精确相同；原生 v1
完整离线续训也与连续训练逐元素一致。v2 原生恢复限制保持不变。

用户最新明确恢复版本修复与 replay 是同一最终交付单元。因此最终评审基线为 `main`， 完整候选包含恢复、日志修复及回放功能；发布前将本任务相关改动形成一个
feature commit， 不改写 main 或他人历史。独立 fix 分支和已验证提交保留为开发证据，不要求每项修复单独发布 PR。 全部发布前条件满足后，优先评估现有
#1841 的安全复用方案；该 PR 现已关闭且只含恢复修复，若无法保持独立 feat 分支和完整 main diff，则创建完整 feature PR，重新核对新 head
的 CI。旧的以 fix 分支为 base 的 stacked PR 计划作废，旧报告保留历史身份。 按当前 v0.5 规范，用法错误与 feature 自身修订在
feature 内修正，不再要求五项早期修订的历史流程例外；这不是对旧流程逐项批准的声明，也不授予 Agent 执行 UT 的权限。

用户于 2026-10-29 明确要求中文 RFC 完成后直接开发，不再等待逐份认可，并授权本任务必要配置与共享 trainer 改动。
这是流程授权，不表示用户逐项审阅了旧英文设计。旧 RFC、旧规范摘要和规划证据保留其历史身份，旧批准门槛不再适用。

最新 main 相对初始基线只增加平台探测和 XPU 支持，原 CUDA 选择分支未变，新增 device_of 尚无运行时调用方。 已在原 GPU 容器导入当前源码并实际选择
CudaPlatform；这是兼容性预检，不是新训练。既有 GPU 实验仍绑定其原始源码，结合平台差异与后续修订的专项证据说明复用范围，交由新版独立评审核查。

用户同日进一步授权：容器内 `/tmp` 的本任务独立短目录可保存 UNIX socket 及其必要运行临时文件；
其他候选代码、数据、日志、checkpoint、缓存、快照与报告仍位于原授权根目录。 采用 `TMPDIR/TMP/TEMP` 引导
IPC，显式设置各组件缓存与产物目录；原“无可用授权 socket 位置”仅为历史阻塞。 继续使用原 Job 和受控 helper，启动前核查已有进程与 socket
可用性，不清空 `/tmp`，不提前开启 DP attention。

本功能只支持 **single-controller RL**，它是控制器执行方式，可以管理多 GPU、多训练 worker；继续支持实际 v1 与 v2 路径。
不实现其他控制器执行模式，不增加按 DP head 分片写盘、跨控制器回放和相应测试或验收。启用功能时对不支持的执行方式提前报错。 本任务先交付 trajectory
dump/replay；其完整验收和最新 head 必需 CI 通过后，才在独立分支开始 DP attention。

## 1. 功能与收益

现有可读轨迹日志能帮助查看对话，但不足以恢复实际训练输入。开发者若要复现一次 PPO 更新，通常仍需重新生成 rollout 并保持推理服务。

本功能在正常 SGLang RL 训练中保存实际消费批次；之后从同样初始权重另起离线运行，读取批次进行 advantage 和 PPO 更新，无需启动推理服务。
例如先捕获三个训练步，停止原推理和轨迹存储服务，再在 v1 或 v2 训练控制器上依次执行三个离线更新。

预期收益是新增离线调试能力、减少调试对推理资源和服务存活的依赖。保存会增加磁盘、CPU 内存与 IO 时间。 回放陈旧响应用于调试，不宣称与新的 on-policy RL
等价；快照不是模型/优化器/RNG checkpoint。最小与三步链路已验证无推理回放的支持度收益；开销与恢复对照结果见第 6 节，长期稳定性未验证。

范围内：SGLang、single-controller PPO/GRPO、v1/v2、文本和处理后的多模态张量、默认关闭的独立开关，以及两种快照边界：

- `rollout`：批次收集完成后，辅助模型评分之前。回放仍计算适用的训练侧 critic/reference/teacher 和 prox logp。
- `full`：critic/reference/teacher 评分及 actor prox logp 重算之后，advantage
  之前。回放复用这些结果，省略评分专用模型/服务。

初版不增加自动保留/固定步清理、不读取旧 PR 的未版本化文件、不自动改变拓扑或跨 v1/v2 回放。 不支持在线 agent
或回放时推理评测；需显式关闭验证数据。rollout-engine teacher 只能采用 full 回放，保证不启动任何推理服务。 MOPD
拥有独立物化与教师生命周期，初版启用本功能时明确拒绝，不静默绕过其算法。上述是实现边界而非额外任务。

发布验证范围限定为已实测的无 critic/teacher 的 GRPO，以及 reference 评分的 rollout/full 路径。 critic 与 teacher
的代码分支保留原有训练接入，但尚无本功能组合的专项 GPU 证据，属于未验收组合， 不能据此承诺 PPO critic、训练 teacher 或 rollout teacher
的可用性。两种快照边界的定义不变。

## 2. 当前分支现状

以下为当前精确基线的源码事实，不能当运行通过证据。

| 能力                    | v1             | v2               | 源码与缺口                                                                                   |
| ----------------------- | -------------- | ---------------- | -------------------------------------------------------------------------------------------- |
| 统一 RL trainer         | 已有           | 已有             | `areal/trainer/rl_trainer.py:PPOTrainer`，版本选择不同 controller，训练循环共用              |
| 收集训练批次            | 已有           | 已有             | `TrainController.prepare_batch` / `GatewayTrainController.prepare_batch` 转发真实 rollout    |
| 可读轨迹导出            | 共享实现       | 共享实现         | `WorkflowExecutor._dump_trajectory` 只导出选定 JSONL 字段，刻意不保存 vision 与原始 logprobs |
| 完整批次保存/读取       | 未发现         | 未发现           | 当前无等价训练快照配置、存储和离线入口                                                       |
| 无推理初始化            | 缺失           | 缺失             | `_init_impl` 无条件创建 rollout、连接 actor、准备权重传输                                    |
| 旧 NoOpRollout 可复用性 | 不满足回调契约 | 不兼容类型检查   | v1 需要 callback 地址；v2 强制真实 RolloutControllerV2 并建立 WeightUpdateController         |
| 远程张量与分组          | 已有           | 已有             | `RTensor` 引用含服务地址；`RolloutGroup` 包含逻辑分组，不能直接将引用保存成离线输入          |
| 不传推理服务的恢复      | 可复用基础     | 需按真实支持核查 | `RecoverHandler.load` 可接收 `None`；trainer 后续 staleness/传输逻辑需跳过                   |

现有链路：driver → trainer 初始化 → 训练引擎/推理初始化 → 权重连接 → 收集 → 辅助评分 → prox logp → advantage → PPO
→ 保存/传输 → 清理/评测/统计。 v1 实际使用 `PPOActorController`；v2 使用 `PPOActorControllerV2` 和 gateway
worker，不能仅改两个配置名字就声称两条路径通过。示例的 v1 捕获复用 RLVR/VisionRLVRWorkflow；v2 捕获复用 现有
MathAgent/Geometry3KAgent 的 `run()` 接口和网关导出，因其控制器明确拒绝直接传入 RolloutWorkflow。 两版均在共享
trainer 收集完整批次后保存；回放均跳过各自推理与 agent 生命周期。

已有相关覆盖包括
`test_workflow_executor_dump.py`、`test_split_trajectory_for_dump.py`、`test_trainer_eval_before_train.py`、`test_rtensor.py`。
原可读 dump 不物化训练引用、不读取 vision 的行为保持不变；新增训练快照是单独的显式开销。

## 3. 方案与实施计划

### 3.1 侵入等级与选择

| 方案                                    | 等级 | 判断                                       |
| --------------------------------------- | ---- | ------------------------------------------ |
| 修改外部 SGLang/Megatron/Awex 代码      | I5   | 没有必要性证据，不采用                     |
| 升级依赖/镜像                           | I4   | 没有必要性证据，不采用                     |
| 修改普通训练默认行为                    | I3   | 违反默认关闭要求，不采用                   |
| 运行时替换 controller 方法              | I2   | 仍不能满足真实生命周期且维护风险大，不采用 |
| 新增存储类，显式接入 trainer 的可选分支 | I1   | 采用，无第三方 patch，无新依赖             |

必须修改共享 `PPOTrainer`，因为其初始化已经启动推理；只加 driver 不足以实现无推理回放。
保留原训练计算，不复制整个循环、不实现假推理引擎。共享修改的主要风险是初始化/恢复/统计/offload 的生命周期回归。

### 3.2 配置与文件格式

在 `areal/api/cli_args.py` 新增 `TrajectoryDebugConfig`，仅接入 `PPOConfig.trajectory_debug`，包含
help 与 `__post_init__`：

| 字段    | 类型和默认值                           | 语义                                                     |
| ------- | -------------------------------------- | -------------------------------------------------------- |
| `mode`  | `str = "off"`，枚举 choices 元数据     | `off/dump/replay` 互斥；关闭不做快照 IO/RPC              |
| `path`  | `str \| None = None`                   | 保存默认位于本次 fileroot 下；回放必须显式输入目录       |
| `scope` | `str = "rollout"`，枚举 choices 元数据 | `rollout/full` 保存边界；读取必须匹配                    |
| `steps` | `list[int] \| None = None`             | 保存指定非负 global step；空值表示全部，不做隐式回放循环 |

枚举沿用本仓库 OmegaConf dataclass 的字符串、choices 元数据与构造校验组合；不直接使用 OmegaConf 不支持的 `Literal` 字段。
步骤选择由 trainer 管理；新增 `areal/utils/trajectory.py:TrajectoryStore` 负责编码、文件发布、元数据检查和读取。 每个
global step 保存一个完整 single-controller 批次。文件是版本化元数据、CPU 张量、基本容器；`RolloutGroup` 显式编码再校验重建。
保存版本、scope、global step、actor backend/拓扑、controller 版本和步数信息；回放验证来源兼容，不承诺自动重分区。

复制容器与 RTensor wrapper，使用现有批量 localization 和 alias preservation；不修改活跃训练引用。 保留完整
logprobs、masks、versions、rewards、逻辑分组、嵌套 vision 张量及重复 image 引用。 原引用活到正常 cleanup；仅释放本次快照的
fetch buffer，不删除其他消费者的存储。普通张量放置复用 tensor-container 工具。
未知不透明对象明确报错，不偷偷丢字段。文件通过同目录临时文件发布，不静默覆盖已有步骤。 读取使用
`torch.load(map_location="cpu", weights_only=True)`，不采用不受限 pickle 或动态导入任意类。

示例 overrides（rollout 已实际运行；完整矩阵状态见第 6 节）：

```text
trajectory_debug.mode=dump trajectory_debug.path=<capture-dir> trajectory_debug.scope=rollout
trajectory_debug.mode=replay trajectory_debug.path=<capture-dir> trajectory_debug.scope=rollout valid_dataset=null
```

### 3.3 生命周期

关闭模式保持原顺序，不构造存储、不物化张量、不多发 RPC。保存模式仅在选定步骤/边界增加快照，继续正常权重传输与清理。

回放模式在创建 worker 前校验输入与不支持配置，跳过真实 rollout/eval/proxy/初始推理 LoRA 导出、Awex
推理握手、actor.connect_engine 和传输 meta。 真实 `rollout` 为 `None`，复用原 actor/critic
更新、LR、checkpoint、训练统计和必要训练服务；不使用 NoOpRollout。 full 模式不初始化只做评分的 ref/teacher，但需训练的 critic
仍保留；rollout 模式按原逻辑重新评分。 不把离线 actor 当成与 rollout 共置，不触发推理内存所有权交接。

从快照元数据得到 steps_per_epoch 和文件范围，允许不传 train_dataset；不为回放收集响应或启动数据服务。 按原始 global step
读取，缺少步骤明确失败，不隐式重复上一批。恢复后校验 next step；继续使用现有恢复能力，不新增不同版本的恢复框架。 正常训练统计只在真实 rollout
存在时导出推理统计，trace/offload/resume/eval 同理；每个实际创建的资源都能部分初始化失败清理。 full 捕获在 prox logp 之后；full
回放避免再次重算这些值，仍进行 advantage 与训练。

### 3.4 改动文件、成本与顺序

| 文件                                                | 与任务的直接关系                                     |
| --------------------------------------------------- | ---------------------------------------------------- |
| `areal/api/cli_args.py`                             | 默认关闭配置和约束                                   |
| `areal/utils/trajectory.py`                         | 自包含批次存储与校验                                 |
| `areal/trainer/rl_trainer.py`                       | 两种边界与无推理生命周期，v1/v2 共用                 |
| `tests/test_trajectory_replay.py`、必要既有 fixture | 真正常用路径、默认行为、生命周期回归                 |
| `examples/trajectory_replay/`                       | 本 RFC、可参数化 driver、最小与端到端 GPU 脚本、用法 |
| 中英文 debugging 与生成的 CLI 文档                  | 与新增能力/配置一致                                  |

预计约 8–12 个相关文件、400–750 行实现、150–250 行必要测试；实施准备约数小时，UT 准备与静态检查约一至两小时。 这是不确定的规划估计，不是 Goal
时间/token 预算；远端实验长度根据实际环境确定，不包含本机 UT 执行。 先完成本 RFC，再配置/存储/捕获/回放、准备
UT/脚本、静态检查、真实实验、审查，最后发布与 CI。

### 3.5 Awex 归属

当前无 Awex 协议实现；回放跳过推理权重传输属于 AReaL trainer。若发现需暂放的 Awex 支持代码，记录具体符号、上游版本、可迁移部分、保留适配与移除条件。
不复制两份实现，不由此向 Awex 自动发 PR。

## 4. 开发标准

| ID     | 必须落实的标准                                                                                                      |
| ------ | ------------------------------------------------------------------------------------------------------------------- |
| DEV-01 | 核实最新 main，独立 feat/ 与 fix/，保护用户脏分支，记录精确基线                                                     |
| DEV-02 | 实现前完成中文 RFC，按 2026-10-29 流程授权直接实施，不等待逐份批准，不伪造审阅                                      |
| DEV-03 | 每个修改直接相关；不夹带格式化/重构/其他模式支持                                                                    |
| DEV-04 | 最低必要侵入，复用扩展点，不 monkey-patch，不复制训练循环                                                           |
| DEV-05 | 公共 API/默认行为兼容；开关默认 off；不支持组合显式报错                                                             |
| DEV-06 | 配置在 cli_args 对应 dataclass、有 help/校验；任务内必要配置/trainer 改动已授权，不扩大依赖或资源                   |
| DEV-07 | 遵守 AGENTS、日志/类型/异步/分布式规范；无硬编码私密设置、通配 import、热路径 debug 同步                            |
| DEV-08 | single-controller 下 v1/v2 都实际跑通，不能静态推断代替                                                             |
| DEV-09 | 问题及时记录并按 V1a 归因；用法错误和 feature 缺陷随 feature 修正，仅独立基线缺陷走 fix/ 验证后派生 feat/           |
| DEV-10 | UT 先评估实际触发、普通开发者、概率依据、影响和已有覆盖，克制新增                                                   |
| DEV-11 | 不在本机运行任何 UT；准备最小/端到端 GPU 脚本，注明未本地执行                                                       |
| DEV-12 | 远端只走受控 helper、原授权 allocation、登记资源和可写根，不申请/取消 Job、不绕过入口                               |
| DEV-13 | 具体标注 Awex 迁移归属；当前无外部支持实现                                                                          |
| DEV-14 | 发布前必需验证与完整 review 通过；CI 专属检查标 PENDING_CI；仅 squash 本任务提交为一个，单句 Conventional Commit    |
| DEV-15 | 保存 Pre-CI review PASS 后才执行可能触发 CI 的发布/推送/打标；修复先验证和重新 review，再触发最新 head CI，不 merge |

### 4.0 经验读取与逐轮复盘

RFC 修订、实验设计和每轮 verify 前，分别调用受控 recall，逐项记录采用动作或不适用原因；verify 提供最新 application.json。
按必需行为核对可读证据、配置来源和源码差异，保留原始运行、续训与重分析的区别。proposed 仅是待验证方法，不作为当前候选已通过的依据。 每轮 review
后由受控入口另起只读复盘上下文，先读取完整复盘，再修复或补证。复盘回执、逐轮索引和经验采用的精确路径保存在私有交接中。
共享经验须有独立复核支持，不能降低验收、改变权限或机械增加 UT；没有新实验需要时复用适用的既有结果。

### 4.1 UT 准入

| 场景             | 实际触发/概率               | 影响与已有覆盖                                           | 最小必要覆盖                                      |
| ---------------- | --------------------------- | -------------------------------------------------------- | ------------------------------------------------- |
| 完整批次往返     | 每个回放用户，常规          | 错误字段/分组会破坏更新，旧可读 dump 不覆盖              | 代表性分组+多模态嵌套输入，数值/dtype/shape/alias |
| RTensor 活跃引用 | controller 保存，常规       | 物化原 wrapper 改变后续 RPC，旧 RTensor 测试只有基础能力 | 使用 fake backend，验证副本/去重/缓冲清理         |
| 文件缺失或不兼容 | 中断保存/选错目录，合理可遇 | 避免晚失败或错训练                                       | 参数化关键契约错误，不穷举每个值                  |
| 默认关闭         | 全部现有用户                | 共享 trainer 影响面大                                    | 既有 trainer fixture 顺序/无新增 IO 与服务        |
| 真正无推理回放   | 每个回放用户                | 偷启动服务使功能失效                                     | v1/v2 下禁止推理工厂，实际走训练调用              |
| scope 评分选择   | 辅助评分用户，常规          | 重复/漏算改变训练                                        | rollout/full 捕获点和评分跳过/拒绝                |

优先扩展已有 fixture；使用 tmp_path、合适 marker、明确 rtol/atol。测试代码准备不等于执行通过；不在本地执行 pytest/unittest
或间接测试脚本。

### 4.2 Bug 记录

历史 SSH bridge 错误属于环境故障。按 VERIFIER v0.3 追加归因：非 optional agent、v1 Awex 配对、v2 鉴权与 workflow
接口的早期错误属于用法/示例配置错误；新增属性破坏既有 fixture 和无推理时误报最终步属于 feature 自身缺陷，均随 feature 修正验证，不要求独立 fix
或历史例外。 恢复版本问题有精确 main 的独立探针及既有无推理调用方证据，属于独立基线缺陷，其 fix
分支与验证记录保留。日志修订已经存在的独立分支也保留为历史证据，不据此把它改归为 main 缺陷或要求独立发布 PR。
记录包含触发、预期/实际、日志、base/head、v1/v2、归因及验证；旧例外材料和旧报告仅保留历史身份，不改写原始记录、不伪造旧版复现或逐项授权。

## 5. 具体验收标准

### 5.1 工程与交付门槛

| Gate | 要求                                                                                       | 当前状态                                                                 |
| ---- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| G-01 | DEV-01–15 有证据，授权按新版核查                                                           | 实施中                                                                   |
| G-02 | 本 RFC 范围实现，默认 off 不改变行为，无无关改动                                           | 已实现；默认关闭专项回归及有效成本对照完成                               |
| G-03 | 完整 VERIFIER：因果解释、F1–F7、相关领域/跨域、风格、数值/性能风险、旧测试影响及修复 B1–B8 | 旧轮独立结论 INCOMPLETE；行为证据索引已补齐，待新版完整独立复核          |
| G-04 | pre-commit、CLI 文档和必要静态/文档检查通过，不跳 hooks；UT 由用户/CI 或另有授权方式执行   | 既有完整静态通过；双语文档构建各 6 条告警；完整 feature UT 为 PENDING_CI |
| G-05 | single-controller v1/v2 最小和端到端脚本均有真实结果                                       | 两版最小 rollout 及 9B 多模态三步 full 已验证                            |
| G-06 | 证明服务无依赖回放与真实更新；实测成本/收益/限制，不将预期当结果                           | 已声明短程范围的功能、恢复与匹配保存开销证据完成；不扩展为长期稳定性     |
| G-07 | V1a 归因、独立基线缺陷的修复链与 Awex 清单完整                                             | 恢复修复链保留；用法/feature 修订无需历史例外；无 Awex patch             |
| G-08 | 候选/环境/配置身份、执行边界和私密隔离明确                                                 | 私有记录持续更新                                                         |
| G-09 | 必需发布前验证和完整 review 通过，保存 Pre-CI PASS 后才发布/推送/创建 PR，squash 内容一致  | 既有基线结果及新 main 差异须核验；尚无完整 feature commit；旧 PR 已关闭  |
| G-10 | gh 标签真实触发最新 head 必需 CI，全部通过，不 merge                                       | 完整 feature 新 head 的 PR 后门槛，尚未执行                              |

Markdown 规范不是可执行可信验证器，研发自检不是独立正式验收。通用评审只是其中一项；缺少必需证据保持 INCOMPLETE，不伪造数值成绩或专家咨询。

### 5.2 实际 v1/v2 矩阵与实验协议

按 V1b 分别记录未修改 main 的正常在线最小训练基线，以及候选的默认关闭、捕获和回放结果。基线需绑定精确源码、实际依赖、模型/数据、有效配置、参数更新、Awex
传输和后续 rollout；候选的 off 运行不能冒充 main。已有 main 恢复探针只证明恢复契约，不替代这两版在线基线。 本轮已另行补齐精确初始 main
上两版各两步的正常在线基线：全部运行时源码取自该 main，两个私有运行 fixture 单独记录；实际依赖、标准 loader
的有效配置和导入位置均已核对。两版各步均有非零参数更新、有限 loss/gradient、Awex
传输和版本更新后的后续生成。该结果不追认实施前已留存基线，也不是长期稳定性或性能改善证据。

| 路径/场景       | 入口                                                        | 必须观察到的结果                                                          |
| --------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------- |
| v1 最小         | areal train run → 本例 driver，actor/rollout `_version=v1`  | SGLang 捕获一个批次，停止原服务后离线真实更新                             |
| v2 最小         | 同 driver，actor/rollout `_version=v2`，真实 gateway/worker | 同样结果，日志确认不同服务路径                                            |
| v1 端到端       | 固定文本或 Geometry3K 多模态配置，至少三个连续更新          | capture/full/rollout 支持范围明确，参数变化、有限 loss、清理和 checkpoint |
| v2 端到端       | 相同工作负载族，实际 v2 路径                                | 同上，原存储服务停止后仍可读盘训练                                        |
| 两版默认回归    | 相同模型/数据/seed/拓扑，功能 off                           | 原 rollout、更新、选定 Awex 权重传输及后续 rollout、保存/评测继续正常     |
| 两版 scope 验证 | 相同初始权重，rollout 与 full 捕获/回放                     | 数据不变，辅助评分与 prox 仅在适用边界执行                                |

优先约 9B 的已授权模型，先核实 checkpoint、架构、tokenizer 与引擎兼容；最小脚本可选择更小的已验证兼容模型。 文本使用 GSM8K，多模态使用
Geometry3K。实际模型、split、资源、并行维度、安装版本和镜像保存在私有实验记录。 single-controller 可以使用多 GPU；不缩减为单
GPU，也不据此排除 v1/v2。

启动复用
`/opt/.venv/bin/areal train run --config <yaml> --driver <module:main> key=value...`，经
helper 进入原 allocation 的容器。 源码 CLI 直接调用 driver；使用现有 local scheduler 管理已分配资源内 worker，不修改
scheduler/launcher，不机械沿用参考脚本旧规模/共享输出。 优先 overrides，冻结基线 YAML/hash、全部
overrides、最终解析配置、初始权重/数据和源 tree 身份。 v1 Awex 的 CUDA IPC 使用同卡交替执行，最小示例采用 actor
`megatron:d2`、rollout `sglang:d2`， 使两个训练 worker 和两个推理副本逐卡配对；v2 示例采用分离的
`d1/d1`。短程实验显式关闭学习率 warmup， 避免完整 dataloader 计算出的 warmup 使首步零学习率；baseline、capture、replay
使用相同设置。 所有输出、checkpoint、name-resolve、cache、服务状态在获准根下隔离目录；仅按新增明确授权将 UNIX socket
及必要运行临时文件放在容器 `/tmp` 的本任务独立短目录。旧版无 socket 目录授权的失败保留历史身份。

最小脚本最少一组/实际更新；端到端至少三步，多于三步才能支持更长窗口稳定性声明。
脚本记录依赖、资源、输入、命令、输出、失败判定、预期结果和只清理本轮进程的边界；均注明“未在本地运行”。

### 5.3 正确性、收益与不退化条件

1. 快照 round trip 精确保留支持字段的值、shape、dtype、逻辑分组和嵌套多模态数据；不依赖原存储地址。
1. 原 inference/storage 服务停止后，两版均完成真实 actor/适用 critic 更新、优化器/LR 推进；loss/gradient 有限，非全
   mask/退化零信号输入；用 checkpoint 参数差异核实更新，不只看 step 日志。
1. 回放进程/角色记录中无 SGLang、推理 controller/gateway、agent proxy、rollout teacher、权重交换服务或对旧端点请求；训练
   RPC/gateway 允许且必需。
1. rollout 模式按原流程评分，full 只使用已捕获评分；不修改原 logprobs、mask、versions 或 group 语义。
1. 相同初始权重/优化器、批次顺序、精度、确定性配置下，首个固定批次比较 loss/gradient/参数样本，BF16 初始
   `rtol=1e-3, atol=1e-4`，记录最大差异。若非确定性使协议无效，先说明并修订，不能事后放宽只为过关。
1. 支持度收益是新增的无推理批次调试链路。在线生成与离线回放是不同工作量，不能据耗时差宣称 on-policy 训练吞吐提升。
1. 对 off 与 dump 固定模型/数据/seed/拓扑/长度/总步数/版本，至少两组匹配运行，区分 warmup，记录启动、保存耗时/总步时、文件字节、GPU/CPU
   峰值和波动；据实报告开销。
1. off 不做额外快照 IO/RPC、无新增异常或挂起；每批次结束缓存可排空。三步只构成短程观察，不能证明不存在长期泄漏。

真实评分专项覆盖 reference 的 rollout/full 边界；critic/teacher 组合尚未另做 GPU 实测，
不能把该项静态接入推断为所有辅助模型组合均已验证。v2 原生 GatewayTrainController 暂不支持恢复； 本功能不扩展该既有限制。

本功能是调试支持，不是新增模型架构；正常捕获与回归路径仍核对真实 SGLang rollout、Megatron 更新、所选 Awex 传输与更新后继续生成；磁盘
fallback 不能冒充 Awex。 性能/稳定性/支持度分别报告，硬件相关额外阈值在实验前冻结；没有足够窗口或匹配基线则不声称性能/稳定性收益。

## 6. 验证记录、提交与 PR

截至 2026-10-29 的已核实结果如下。环境与候选摘要、完整命令和原始产物保存在私有证据中；不将操作员资源标识发布到 PR。

| 实验                                        | 实际结果                                                                                            | 证据范围与限制                                                                                                                   |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 初始 main 的 v1/v2 正常在线最小基线         | 两版各两步，SGLang 生成、Megatron 更新、Awex 传输和更新后继续生成均完成；标准配置和源码摘要匹配     | 原始身份为第 0 节初始基线；最新 main 的平台差异另行核查，不冒称重新训练；不使用候选 off 冒充 main，不外推长期稳定性              |
| v1 rollout，文本 0.6B，捕获/回放各一步      | 均正常退出；停止捕获服务后回放仅创建 actor，参数实际变化，loss/gradient 有限                        | 全部 311 个 checkpoint 张量、751632384 个元素逐元素相同，最大差异 0；不是三步或 full 验证                                        |
| v2 rollout，同模型与数据族，捕获/回放各一步 | 真实 gateway/worker/agent 捕获，停服后回放仅创建训练角色，无推理角色                                | 全部 311 个张量逐元素相同；TensorBoard 首步 loss=5.005346298217773、gradient norm=44.12732696533203，捕获与回放完全一致          |
| v1 约 9B 多模态 full，连续三步              | 捕获和停服后回放均完成真实更新                                                                      | 每步全部 760 个保存张量、9409813744 个元素逐元素相同；三步精确 loss/gradient 相同                                                |
| v2 同模型/数据族 full，连续三步             | 捕获生成版本依次0/1/2；停服后回放仅有训练角色                                                       | 前两步全部保存张量相同；第三步最大差异1.9073486328125e-6，零元素超出预定容差；第三步 gradient norm 差约9.16e-5，在同一预定容差内 |
| v1 reference 评分边界                       | rollout 回放创建 actor+ref 并重算评分；full 回放只创建 actor，复用评分                              | 两种 scope 各自全部311个 checkpoint 张量、精确 loss/gradient 与对应捕获一致                                                      |
| 两版默认关闭评测回归                        | 小型真实 held-out 数据上完成初始评测、训练后评测和 checkpoint；无快照                               | v1 另已实际加载带优化器的 checkpoint 并续训下一步；v2 原有 GatewayTrainController 不支持 recover，未宣称扩展支持                 |
| v2 reference 评分边界                       | rollout 回放仅创建 actor/ref 训练角色，重算 ref/prox；full 只创建 actor 训练角色，无两项重算        | 两种 scope 各自全部311个保存张量、精确 loss/gradient 与对应捕获一致；未创建推理角色                                              |
| v2 文本 rollout 三步回放                    | 真实训练 gateway/worker 完成三步，仅有训练角色                                                      | 每步全部 311 个权重张量与对应在线捕获逐元素相同，loss/gradient 有限，版本依次 0/1/2                                              |
| v1 离线恢复                                 | 连续三步与一步保存优化器、重启续训两步全部完成；仅创建 actor                                        | 每步全部 311 个权重张量精确相同，精确 loss/gradient/LR 相同，proximal 版本依次 0/1/2；v2 原生恢复仍不支持                        |
| 两版保存成本对照                            | 每版 off/dump 各两次串行三步；工作量计数匹配，权重差异均在预定容差内                                | v1 greedy 解码、v2 原生确定性采样分别测量，不跨版本比较速度；短程 n=2，不声称吞吐改善                                            |
| 静态/文档与 UT                              | 既有候选完整 pre-commit、AST/shell 检查通过；CLI 内容对应的双语 canonical 文档构建成功，各 6 条告警 | 完整 feature UT 仅准备代码，尚未由用户/CI 执行；最终静态结果绑定冻结候选                                                         |
| PR 与 CI                                    | 完整 feature 尚未发布；#1841 已关闭且只含恢复修复，其 CI 不能覆盖完整候选                           | 先完成发布前条件，核实安全复用条件或创建完整 feature PR，重新核对最新 head 的 CI                                                 |

保存成本的已测范围（文本 0.6B、每次三步，各版本保持相同模型/数据/拓扑/精度/优化器等设置）：

| 指标                           | v1                                                    | v2                                                    |
| ------------------------------ | ----------------------------------------------------- | ----------------------------------------------------- |
| off 整次耗时                   | 207.09–208.13 秒                                      | 229.06–238.77 秒                                      |
| dump 整次耗时                  | 201.82–206.28 秒                                      | 233.41–236.67 秒                                      |
| 单步快照保存耗时               | 0.0208–0.0378 秒；去首步后 0.0208–0.0250 秒           | 0.0203–0.0246 秒                                      |
| 单步文件字节                   | 12,403–32,883                                         | 48,685–52,205                                         |
| 首次更新后的相邻指标提交间隔   | off 14.16–21.03 秒；dump 14.16–21.92 秒               | off 9.16–9.62 秒；dump 9.31–9.65 秒                   |
| 两张分配 GPU 的逐卡峰值（MiB） | 64,552 / 63,525，off/dump 相同                        | 19,530 / 59,638，off/dump 相同                        |
| 进程树 RSS 加总峰值（KiB）     | off 98,383,280–98,595,108；dump 98,295,576–98,335,736 | off 59,168,700–59,232,588；dump 59,129,492–59,276,264 |

GPU 和 RSS 按约一秒采样，RSS 加总包含重复计算的共享页，不是唯一物理内存或 allocator 峰值。相邻指标提交间隔包含训练、保存与日志提交等工作， 不是纯训练
kernel 时间。短样本、不同步长度及启动波动使整次耗时不能解释为 dump 提速；工作量计数相同不等于已证明生成 token ID 逐个相同。
每版两个有效配对均无权重元素超过事先设定的容差，但不宣称在线运行逐元素确定性。原先工作量不匹配的观测仅保留为历史材料，不计入有效配对。

在线捕获与离线回放耗时对应不同工作量，不能将其差值报告为在线训练吞吐提升。当前仅有短程功能证据，尚不能证明长期无泄漏或所有模型/后端组合可用。

每次结果绑定代码/config/环境/模型数据身份、命令、时间、登记进程、日志、退出状态与行为断言；实际故障准确归因。 私有 Job/账号/资源路径与 debug
历史单独留存，不放公开 PR；旧规划、自检和规范记录保留历史身份，另存 v0.5 规范摘要与真实用户授权。

提交前运行 `pre-commit run --all-files`，有 venv 按要求激活；可能改无关文件的全仓检查用隔离副本，只回传本任务修正。 CLI
文档使用仓库生成器，必要正式文档构建使用 `docs/build_all.sh`。静态检查与 UT 区分，不因包缺失擅自安装依赖，不本地跑 UT。

发布前冻结候选做完整 PRE_PR_DIAGNOSTIC，覆盖 trainer/配置、RTensor/RPC/分布式、恢复/保存、推理生命周期、数值、测试和 CI 风险。
由受控 `independent_verify.py run` 启动全新只读 Reviewer 核查证据；开发仍为单 Agent 串行，不虚称调用专家。
这是用户明确授权的验收流程更新，不表示用户批准了当前候选；旧报告中等待宿主安排独立上下文的说明仅属历史。 每轮在私有证据索引记录
PID、review_id、reviewer_thread、候选与规范摘要、events/result/report 路径和 findings 处置。 每轮 verify
先读取适用经验并记录采用；结果产生后自动复盘，发布 gate 同时核对有效复盘回执。旧轮复盘补录不能替代新版独立复核。 FAIL/INCOMPLETE 时独立理解问题，按
V1a 修复或补证、完成受影响验证后用新目录和新上下文再次评审；不修改工具或旧结果，不原样重试求通过。 诊断必须覆盖 main 到完整候选， 包括恢复工具的
SFT/DPO/奖励模型调用方、日志修复及相应测试影响；重新绑定 base、候选摘要与文件清单。 所有发布前必需项通过后才将本任务的恢复、日志和 feature
改动合为一个提交，比较内容身份，加载 commit-conventions。 保留 fix 分支，不 squash main 或他人历史。不提前 draft PR；PR
描述最终行为/用法/收益/必要风险和验证结论，不写试错历史。

完整 feature UT 尚未执行；#1841 当前仅含恢复修复的 head 的 CI 不能替代它。 UT 仍由用户或 CI 执行，未获另行授权不得本地或远端运行。CI
专门执行项保持 PENDING_CI，不要求先通过才能触发自身，也不能用 GPU 实验或静态预测冒充通过；独立评审和 main 基线等其他发布前缺口不能放进
PENDING_CI。

按 V6d，任何可能触发 CI 的推送、创建 PR、首次或重打 safe-to-test、dispatch/rerun 之前，须完成完整候选
review、处理发现的问题并保存带时间的独立 Pre-CI review PASS 与 `ci_allowed=true`。 发布前执行受控
`gate --output <本轮> --base <当前核实基线>`，squash 后再用 `--head <实际提交>` 核对内容等价； 两次 gate
均通过后才能发布。该工具是任务级流程校验，不声称 Runtime 已拦截任何绕过命令。 发布后核对实际 base/head 与完整 diff，再添加标签并核实最新 head
的实际必需 jobs。失败先归因、修复、验证和重新 review，再重试。旧 head 绿灯、skip/取消/排队、只有 lint 均不能替代必需证据；补审不追认历史 CI
顺序，也不取消既有 job；不合并。第一项验收通过后才开始下一项。

最终交付中文 RFC、实现/必要 UT、最小与端到端脚本、文档、独立修复分支链、真实收益/限制和验收覆盖、合格 PR 与最新 CI。 任何必需项未过保持目标未完成。

## 7. 参考与证据边界

| 来源                                                                                               | 版本                                                                             | 支持的事实 / 限制                                                                     |
| -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| PR #1407                                                                                           | base `3cf0dfbd2b0fbeabd6977184980e189d1567747a`，head 见第 0 节，2026-10-28 读取 | 动机、两种 scope、保存策略；仅参考，不继承旧 NoOp 或其 CPU 测试验收                   |
| `areal/trainer/rl_trainer.py`                                                                      | 本 RFC 初始基线                                                                  | 共用生命周期与版本分派，静态事实                                                      |
| `areal/infra/controller/train_controller.py`、`areal/v2/training_service/controller/controller.py` | 同上                                                                             | 不同实际回调/传输契约，运行待验                                                       |
| `areal/infra/workflow_executor.py` 与现有 dump 测试                                                | 同上                                                                             | 文本导出不等于训练快照，原引用行为需保留                                              |
| `areal/infra/rpc/rtensor.py`、`areal/utils/data.py`                                                | 同上                                                                             | 引用本地化、缓冲清理、RolloutGroup 和 tensor-container 复用                           |
| `areal/utils/recover.py`                                                                           | 同上                                                                             | 可不传 inference 的恢复基础，不等于整条离线链路已验证                                 |
| `areal/v2/cli/training/commands/run.py`、现有 local scheduler                                      | 同上                                                                             | driver 入口与已有 worker 管理，不授权额外集群操作                                     |
| `pyproject.toml`                                                                                   | 同上                                                                             | 声明版本不等于实际安装版本                                                            |
| 操作员 v0.5 规范与用户 2026-10-29 指令                                                             | 摘要保存在私有证据，旧版记录保留历史身份                                         | 中文/直接实施/single-controller 授权，独立评审、逐轮复盘与 gate；规范文档不是验证结果 |
