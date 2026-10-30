# Inline SWE elastic Actor demo

## Demo v1

本版本冻结当前可运行机制：原生 SWE inline 闭环、Actor 整组启停、完整状态保存恢复，以及 OpenRouter StepFun
事件决策。保持当前完整服务销毁和持久化 checkpoint；CPU 服务常驻、启动并行化、临时快照及训练后立即决策均留待后续版本。

| 实现位置                                                        | 内容                                                 |
| --------------------------------------------------------------- | ---------------------------------------------------- |
| `areal/elastic/`                                                | 生命周期、控制工具、队列／数据进度及 checkpoint 提交 |
| `areal/trainer/rl_trainer.py`                                   | 原生训练循环接入与常驻基线启动                       |
| `areal/v2/training_service/`、`areal/engine/megatron_engine.py` | worker 生命周期与完整训练状态恢复                    |
| `areal/v2/weight_update/`、`areal/v2/inference_service/`        | AWEX 重连、TP4→两个 TP2 副本同步及 rollout 保留      |
| `examples/elastic/`                                             | 配置、SWE-Gym 负载、StepFun agent、运行入口和观测    |
| `tests/test_elastic*.py`、`tests/elastic_gpu*.py`、`tests/v2/`  | CPU 协议测试、八卡验收与分阶段计时                   |

验证边界：当前多任务 TP2／staleness 0
配置完成常驻基线；自主启停成功证据属于旧单题配置。尚未证明学习或性能收益，也未验收协调端崩溃恢复。原始实验记录及分享材料位于仓库外层，不随代码提交；下方保留结果摘要与复现命令，外层证据链接仅在完整项目目录中可用。

## 当前配置

基于 AReaL v2.1.0，8×H20：Megatron TP4 Actor（GPU 0–3）独立启停，两个 SGLang TP2 副本（GPU
4–5、6–7）常驻。CPU 协调端保持原生 dataset/workflow、dataloader 和 staleness 调度；只重启 Actor worker
group。权重发布会短暂暂停生成。

仅运行真实 SWE 仓库修复任务，无延时注入、外部请求波次或强制释放。先测真实等待；数据持续充足时 Actor 留驻是正常结果。

当前默认使用 SWE-Gym 多任务训练数据，每题采样 4 次，每次更新消费 2 个完整任务组（8 条轨迹）。先使用 `resident` 测量常驻基线。此前单题
Matplotlib 的八卡自动启停已通过，7.4 秒释放窗口只属于旧配置；新配置验收状态见 [PROGRESS.md](../../../PROGRESS.md)。

旧 `d4t1`／staleness 2 的首次 SWE-Gym 运行完成 4 更新／32 轨迹，更新间等待约 0.06／63.4／177.7 秒；一个 MONAI 组产生
`[0,0,1,0]` reward。但并发预取取消时有一条 episode
未完成清理报告，整轮退出验收失败。清理等待逻辑已修复并通过本地回归；该轮不能视为完整通过或资源收益证据。

当前配置为 `d2t2`、group 并发 2、训练 batch 2、`n_samples=4`、staleness 0。每次收齐 8
条再训练，新版本发布后才启动下一批。`resident` 在首批采样开始时立即初始化 Actor，使启动与采样重叠，随后保持常驻；其他策略仍由各自的启停决策控制。

当前 TP2／staleness 0 常驻轮次已完成 4 更新／32 轨迹，两个副本均到 v4，退出后八卡显存归零。Actor 初始化 86.9 秒；四轮等待为
151.6／86.6／190.3／177.1 秒，初始化另计。首轮有两个不完整评分组被拒绝后补采；其中 Dask 的候选补丁运行时错误应计为 0
分，但当前校验尚未正确归类。因此本轮验证运行机制，不代表标准 SWE 评分语义或学习收益，32 条训练 reward 均为
0。详见[验收与限制](../../../artifacts/elastic-inline-20261030/swe-gym-baseline-tp2-s0-002/acceptance.json)。

## 运行

在项目根目录先执行 `cd AReaL`，再运行下列开发命令；本地环境可通过 `source .venv/bin/activate`
激活。项目进度、演示材料和实验记录位于仓库外层。

沿用已工作的 AReaL CUDA／Megatron／SGLang 环境，不在本例重装驱动。需要完整本地 Qwen3-4B-Instruct-2507 模型及足够
checkpoint 磁盘（建议约 130 GiB 可用）。每次使用新输出目录。

checkpoint 包含 optimizer 等完整训练状态，八卡基线实测 DCP 约 45 GiB；配置按每份 64 GiB
保守估计。新快照提交成功后才删旧快照，因此保存时需要容纳两份及余量，最终只保留最新一份。

`actor_lifecycle.mode: on_demand` 启用两项优化：通信及各 rank engine 确认关闭后，并行清理 Actor 子服务；BF16、非量化
Megatron 从完整 checkpoint 恢复时跳过重复 HF 权重加载。首次启动仍加载 HF；模型、FP32 master parameters、Adam
状态、LR、RNG 全部恢复成功后才连接推理服务。未启用该生命周期的训练保持默认行为；checkpoint 保存格式和持久化要求不变。

本次已测量的 4B／TP4 配置使用 `--checkpoint-gib 52`，运行前约 114 GiB 可用空间；该参数是容量预估，不会压缩
checkpoint，换模型后需重新测量。

磁盘有限时，可在常驻基线完整结束后运行
`python -m examples.elastic.retire_checkpoint --run /path/to/run-baseline`，再启动 agent
实验。该命令先归档元数据，再删除该 run 的完整 checkpoint；日志、曲线、原模型保留，已归档 run 不再支持完整续训。

```bash
# 本机交互配置；密钥隐藏输入，文件权限 0600，不进入仓库。
python -m examples.elastic.configure_openrouter
```

默认凭据文件 `~/.config/areal-elastic/openrouter.json`；远端可通过 `AREAL_CONTROL_LLM_CONFIG`
指定私有配置路径。环境变量方式见 [openrouter.env.example](openrouter.env.example)。控制模型为
[stepfun/step-3.7-flash](https://openrouter.ai/stepfun/step-3.7-flash)，支持工具调用。只有 batch
就绪、空闲超时、观测变化／有界唤醒触发 LLM。

分组模式要求线性 `concat` 导出，每组严格校验 4 行；某条 workflow 发生基础设施错误时，原生 dispatcher 拒绝整个组。GRPO
在每题的四条轨迹内归一化 reward；组内全为 0 或全为 1 时 advantage 为 0。token versions 和 staleness 额度保持原生语义。

入口固定使用 `swe/workflow.py`。默认 4 更新、2 个并发任务组（最多 8 个 SWE episode）、16K 上下文、每轮最多 4096 token、24
轮。必须提供 `--swe-runtime` 或 `AREAL_SWE_CONFIG`。本次不改 agent 触发时机，仍为 40
秒空闲事件阈值；`--idle-seconds` 不强制释放，决策后和 checkpoint 后均复查队列。

`--prepare-only` 只解析配置和题库。`--policy manual` 可配合以下独立工具；`idle` 是显式阈值基线，不代表 LLM agent：

```bash
python -m areal.elastic.control status --state-root /path/to/run/state
python -m areal.elastic.control start --state-root /path/to/run/state
python -m areal.elastic.control stop --state-root /path/to/run/state
```

## SWE inline 负载

复用已部署的 AEnvironment sandbox：每条 episode 新建真实任务容器，模型读代码、修改文件并运行测试，最终按 SWE 参考测试评分。SDK
独立进程运行，避免其 Pydantic 版本与训练环境冲突；原生 session、token versions、concat 导出和 batch 调度保持不变。

需要已注册的 sandbox 环境、任务镜像、SDK Python、kubectl 和私有 kubeconfig。按
[runtime.example.json](swe/runtime.example.json) 设置运行路径；文件留在仓库外。当前沿用旧 H20 的
AEnvironment／SWEAgent 部署及缓存镜像；SDK 文件哈希保存在运行环境的
`source-manifest.json`。这些是已有部署的源码副本，不声称是未经修改的上游版本。

使用 [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) 的 `train` split 与官方
[SWE-Bench-Fork](https://github.com/SWE-Gym/SWE-Bench-Fork)；固定数据／harness
revision，保存文件哈希。以下从全体训练任务中按种子均匀抽取 8 题，不按 reward 或耗时筛选。`--repeats 2`
提供两遍数据供调度与失败补充；`n_samples=4` 由原生 workflow 完成，无需复制四份题目来凑组。

```bash
# 预先下载该 revision 的 data/train-00000-of-00001.parquet，
# 并检出对应 SWE-Gym/SWE-Bench-Fork revision。
python -m examples.elastic.swe.prepare_data \
  --gym-parquet /path/to/train-00000-of-00001.parquet \
  --dataset-revision bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb \
  --harness /path/to/SWE-Bench-Fork \
  --harness-revision 242429c188fcfd06aad13fce9a54d450470bf0ac \
  --images /path/to/resolved-images.json \
  --count 8 --seed 1 --repeats 2 --output /path/to/swe-gym.jsonl

# 为 swe-gym.fixtures/ 中每题部署 source_image，映射为运行环境的
# imagePrefix:sandbox_image；记录实际镜像 digest。逐题验证空／gold。
python -m examples.elastic.swe.validate \
  --fixture /path/to/swe-gym.fixtures/TASK.json \
  --runtime /path/to/swe-runtime.json --output /path/to/task-validation

# 同一个 demo 入口，Actor 随首批采样启动后常驻；4 次更新共 32 条训练轨迹。
python -m examples.elastic.run_demo --model /path/to/model \
  --swe-runtime /path/to/swe-runtime.json \
  --dataset /path/to/swe-gym.jsonl --output /path/to/swe-baseline \
  --policy resident --steps 4 --export-style concat --concurrency 2 \
  --context-tokens 16384 --max-tokens 4096 --max-turns 24 \
  --checkpoint-gib 52 --timeout 3600

# 约 114 GiB 可用磁盘时，先归档已完成基线的 checkpoint。
python -m examples.elastic.retire_checkpoint --run /path/to/swe-baseline

# 同一负载，事件触发 OpenRouter StepFun 决策。
python -m examples.elastic.run_demo --model /path/to/model \
  --swe-runtime /path/to/swe-runtime.json \
  --dataset /path/to/swe-gym.jsonl --output /path/to/swe-agent \
  --policy agent --steps 4 --export-style concat --concurrency 2 \
  --context-tokens 16384 --max-tokens 4096 --max-turns 24 \
  --checkpoint-gib 52 --idle-seconds 40 --timeout 3600
```

验证要求空 patch=0、gold patch=1，且每个 sandbox 独立确认删除。实际训练评分当前 worktree，不重复应用模型 patch；SDK
故障或不完整评分会拒绝该 episode。清理完成后才向原生闭环返回 reward，退出时等待预取任务的清理结束。

`resolved-images.json` 为
`[{"task": "instance_id", "image": "registry/repository:tag", "digest": "sha256:..."}]`，由镜像部署时解析并核实。`--images`
将选中任务的 digest 写入相邻 `.images.json`，运行时连同哈希保存在输出目录；实际拉取应使用 digest。

SDK worker 在首次连接前设置 HTTP read timeout，支持超过一分钟的真实测试；外层工具 RPC、评分和 episode 仍各有时限。释放 Actor
时保留全局 SIGCHLD，确保正在运行的 SWE 子进程能正常回收。

gold patch 和 hints 仅保留于环境验证 fixture，不进入训练题库。manifest
记录来源、选题、重复次数和哈希；`swe-episodes/*/report.json` 记录真实修改、评分及分阶段耗时。旧 Verified fixture 仍可通过
`--fixtures` 导入，但不能称为 SWE-Gym 训练数据。

## 历史八卡结果（单题、每题采样 1 次）

Matplotlib 空 patch／gold patch 评分为 0／1，863 项参考测试报告完整。常驻基线完成 4 次更新、16 条轨迹，等待 batch 约
34／122／119 秒；因此 agent 的空闲事件阈值选 40 秒，避开短间隙，不保证每次运行都会释放。

| Agent 运行结果       | 实测                                        |
| -------------------- | ------------------------------------------- |
| StepFun 决策事件     | 3 次：启动、释放、恢复                      |
| 中途释放／恢复       | 各 1 次；4 个 rank 从版本 2 恢复            |
| 训练 GPU 0–3         | 释放时显存均为 0；推理 GPU 4–7 仍驻留       |
| 释放到再次申请 Actor | 7.40 秒；已收集轨迹 8→12                    |
| 保存／释放／恢复启动 | 36.82／38.21／78.57 秒                      |
| 完成结果             | 4 次更新、16 条轨迹；四个推理副本均为版本 4 |
| 退出清理             | 20 个所属 sandbox 均删除，八卡显存归零      |

单条 episode 中位数约 123 秒，其中评分约 80 秒、sandbox 清理约 31 秒；等待不能全部归为模型思考或工具执行。任务模型的 16 条 reward 全为
0，后续部分轨迹直接结束，没有证明修复能力或训练收益。当前 case 适合展示真实闭环与状态恢复；释放窗口很短，尚不足以证明资源或吞吐收益。

证据：[验收记录](../../../artifacts/elastic-inline-20261030/swe-matplotlib-agent-001/acceptance.json)、[时间线](../../../artifacts/elastic-inline-20261030/swe-matplotlib-agent-001/timeline.svg)、[GPU 释放采样](../../../artifacts/elastic-inline-20261030/swe-matplotlib-agent-001/released-gpus.jsonl)、[退出检查](../../../artifacts/elastic-inline-20261030/swe-matplotlib-agent-001/final-cleanup.json)。

## 证据与边界

- `episodes.jsonl`：真实任务轮数、工具次数、生成／工具／奖励耗时。
- `gpu-memory.jsonl`：从初始化前到退出清理后的逐卡显存、GPU／显存利用率，约每秒采样；采样故障保留错误，不能据采样最大值声称捕获了瞬时峰值。
- `observations.jsonl`、`state/events.jsonl`、`state/decisions.jsonl`：队列、Actor 状态、版本、LLM
  决策、checkpoint／启动／释放耗时。
- `counts` 与 `batch_size` 均以任务组为单位；`group_size=4`。`data_progress.samples`
  统计实际训练轨迹，`task_groups` 统计任务组；TRAINING 事件保存 `traj_group_sizes` 和 `group_rewards`，可检查分组与
  reward 信号。
- `result.json`、`timeline.svg`：训练完成与是否实际观察到释放→恢复→继续更新分开报告。结束清理不算弹性周期。`max_actor_idle_seconds`
  包含 LLM／保存耗时；纯数据等待需按观测中的 batch 就绪情况判断。
- 释放先保存 model、optimizer、LR、各 rank RNG 和版本，再确认进程／GPU 资源退出。恢复用原生
  DCP；新增八卡对照已核对普通／优化恢复后的完整模型、主参数持久化表示、Adam moments
  和固定梯度更新结果，详见[验收](../../../artifacts/elastic-inline-20261030/lifecycle-ab-002/acceptance.json)。这是同一
  checkpoint 两条恢复路径的比较，不是保存前原始状态的独立核验。
- 原生 remote tensors 在 CPU 完整保存后清理。保留真实 token versions，记录长尾样本年龄；沿用原生调度额度及 PPO importance
  weighting，不额外按 token 年龄过滤样本。
- 协调进程崩溃恢复、DP／TP 规模变化、性能／收敛收益不在本例验收范围。

## CPU 验证

```bash
python -m pytest -q tests/test_elastic_runtime.py tests/test_elastic_checkpoint.py \
  tests/test_elastic_runtime_state.py tests/test_elastic_demo.py tests/test_elastic_swe.py \
  tests/test_elastic_inline.py tests/test_elastic_profile.py tests/test_staleness_manager.py

python -m pytest -q tests/v2/inference_service/test_controller.py -k offline_group
```

这些检查使用真实 CPU 张量、队列和调度器，以及替代 GPU worker 的协议 fixture，不能代替八卡验收。

GitHub 工作流已添加，尚未推送执行；当前验证状态见 [PROGRESS.md](../../../PROGRESS.md)。

TP4→两个 TP2 副本的权重同步检查（不跑 SWE、不训练）：

```bash
python -m tests.elastic_gpu_weight_sync --model /path/to/model \
  --dataset /path/to/swe-gym.jsonl --swe-runtime /path/to/swe-runtime.json \
  --output /path/to/weight-sync-check
```

该检查先随机化推理权重，再发布 Actor 基础权重，并逐元素核对两个副本的 TP0 参数抽样。现有调试接口只导出
TP0，不代表全部权重核验；`tests/v2/weight_update/test_metadata.py` 使用真实 AWEX planner 检查所有 rank
的分片覆盖及重复写入，需具备可导入 AWEX／Megatron 的环境。

八卡重启协议检查单独运行：

```bash
python -m tests.elastic_gpu_restart --model /path/to/model \
  --dataset /path/to/swe-gym.jsonl --swe-runtime /path/to/swe-runtime.json \
  --output /path/to/restart-check --checkpoint-gib 52
```

测试主动划分两个训练阶段：先训练两步，释放 Actor 后让每个推理副本生成新 token，再恢复 Actor，从同一个原生 dispatcher 收集下一批未消费的
batch 并训练第三步。它验证真实 GPU 资源、DCP 恢复、推理服务保留及继续更新，不代表 agent 自主释放或性能收益。

已有完整 checkpoint 的生命周期对照（不采样 SWE，不另存模型快照）：

```bash
python -m tests.elastic_gpu_lifecycle --model /path/to/model \
  --dataset /path/to/swe-gym.jsonl --swe-runtime /path/to/swe-runtime.json \
  --checkpoint /path/to/state/checkpoints/step-... --output /path/to/lifecycle-check
```

按原路径→优化路径→原路径运行，比较同一 checkpoint 恢复后的完整模型、FP32 master、Adam moments
和一次固定非零梯度更新后的状态摘要；排除显式标记的 optimizer padding。精度感知优化器的主参数包含 int16 低位补偿表示，与 BF16
模型一并核验。状态核验在启动计时之后执行。此检查验证两条恢复路径等价，不代表 SWE 训练收益或协调端崩溃恢复。

该轮仅运行八卡生命周期对照，未重跑 SWE 训练或三步续训协议。对照使用的旧基线 checkpoint 已归档元数据并清理，日志和验收结果保留。

本次对照 `lifecycle-ab-002`：原路径／优化／原路径释放 39.0／13.8／38.4 秒，恢复 92.1／83.0／86.7 秒；HF 加载约 0.4
秒，DCP 加载 6.8–8.7 秒。恢复差值包含启动及 I/O 波动，不能全部归因于 HF 优化。保存路径未改，旧基线为 38.9 秒。

保存／恢复分阶段计时（不训练）：

```bash
python -m tests.elastic_gpu_profile --model /path/to/model \
  --dataset /path/to/swe-gym.jsonl --swe-runtime /path/to/swe-runtime.json \
  --output /path/to/new-profile-run --restores 2
```

测试仅初始化模型和完整 optimizer slots，保存一次约 45 GiB 的未训练快照，再恢复两次；需要至少 50 GiB 可用磁盘。禁止训练及 optimizer
step，不收集 SWE 轨迹。版本 1 只是传输标签，快照不能作为 SWE 续训记录。计时钩子全部在 `tests/`，不改变普通训练路径。

`lifecycle-profile-001` 保存 **34.0 秒**，两次恢复 **88.7／88.1
秒**（[分解](../../../artifacts/elastic-inline-20261030/lifecycle-profile-001/phase-breakdown.json)、[验收](../../../artifacts/elastic-inline-20261030/lifecycle-profile-001/acceptance.json)）：

- 恢复平均：guard 9.6 秒、worker 服务 13.4 秒、引擎构造及分布式／模型初始化 19.1 秒、router／proxy／gateway 23.8
  秒、checkpoint 加载 7.0 秒、AWEX 连接 15.2 秒、其他 0.3 秒。
- 保存关键 rank：GPU→CPU 5.8 秒、序列化／写文件 9.5 秒、writer `fsync` 16.5 秒、写线程派发／等待 1.5 秒、其他 0.6
  秒。CheckpointStore 显式文件／目录 `fsync` 扫描合计仅 0.00036 秒，不含原子发布函数内部的同步。
- 两次恢复的四 rank 模型、主参数持久化表示、Adam moments 摘要均与保存前一致，optimizer／scheduler step 为 0。Actor
  释放期间两个推理副本均生成新 token；结束后八卡显存归零。

分解排除状态审计，重叠 rank 按时间区间合并，不重复相加。此次 optimizer moments
为零，恢复读取刚写入的快照、可能命中页缓存，且计时含插桩开销；用于定位当前路径，不能直接视为旧 83／39 秒的复现或性能提升。原始
trace、实际运行的测试源码及其哈希已归档。
