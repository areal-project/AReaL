# 八卡重配置验收

此入口准备或执行三组正常路径验收，每组使用独立目录、同一固定题单的前12题、batch4和3次真实PPO更新。共9次更新／36个episode。它调用原生PPOTrainer、Agent
Service、SGLang和共享任务客户端；不是正式训练或性能基准。

| `--case`  | Actor DP | Rollout TP1副本 | 原预留池                   |
| --------- | -------- | --------------- | -------------------------- |
| `actor`   | 2→4→2    | 固定4           | Actor前4张，Rollout后4张   |
| `rollout` | 固定4    | 4→2→4           | 同上，缩容时两张卡暂时空置 |
| `joint`   | 4→2→4    | 4→6→4           | 4+4→2+6→4+4                |

目前只有CPU检查，真实GPU运行尚未验收。限定local／纯FSDP DP／SGLang TP1／PP1；不提交Slurm作业、不归还作业预留、不执行故障注入。

## 环境与执行

在已安装当前AReaL及Hermes依赖、能运行Python工具环境的同一节点或容器中执行。沿用[2×2文档](RUN_2X2_8GPU.md)的TaskAgent与工具设置；模型先用已下载的小模型，例如Qwen3-0.6B。固定题单须带相邻`manifest.json`，至少12题。需要8张已分配且空闲的GPU，以及完整checkpoint／审计副本的磁盘空间；`--estimate-gib`是每份完整状态的预估值，运行过程会保留多份状态。

先启动专用CPU Agent Service（已有同类服务可复用）：

```bash
CUDA_VISIBLE_DEVICES="" areal agent run --service reconfig-acceptance \
  --agent examples.task_workflows.service.TaskAgent \
  --num-pairs 1 --admin-api-key "$AREAL_AGENT_ADMIN_KEY"
export AREAL_AGENT_URL="$(areal agent status --service reconfig-acceptance --json \
  | python -c 'import json,sys; print(json.load(sys.stdin)["gateway_url"])')"
```

设置`AREAL_MODEL_PATH`、`RECONFIG_TASKS`、`RECONFIG_RUN_ROOT`及`AREAL_AGENT_ADMIN_KEY`后，先生成配置，**不启动Trainer、SGLang或GPU进程**：

```bash
python -m examples.task_workflows.reconfiguration \
  --case actor --model "$AREAL_MODEL_PATH" --tasks "$RECONFIG_TASKS" \
  --output "$RECONFIG_RUN_ROOT/actor" --prepare-only
```

GPU可用后，去掉`--prepare-only`执行该组：

```bash
python -m examples.task_workflows.reconfiguration \
  --case actor --model "$AREAL_MODEL_PATH" --tasks "$RECONFIG_TASKS" \
  --output "$RECONFIG_RUN_ROOT/actor" --agent-url "$AREAL_AGENT_URL"
```

该组通过后，分别替换为`--case rollout`／`--case joint`及对应独立输出目录。默认物理设备索引0–7；不同分配需显式传`--devices`和8个实际设备索引。每个case应在独立Python进程中运行；入口负责自己的CPU协调器与GPU服务，不另外启动SGLang。不覆盖已有运行目录；失败后保留现场，使用新目录重跑。

默认上下文8192、每次生成最多512
tokens、单次等待900秒；这些是冒烟预算，不验证原1.7B长上下文OOM问题。若题目或模型需要更长输入，可同时调整`--context-length`与`--max-new-tokens`。准确率只报告，不作为通过阈值。

## 核对与输出

- 每次只提交一个batch，必须等Trainer完成并发布后才切换；切换不改变step、权重版本、消费游标或样本计数。下一批须恰好增加一次更新及4个样本。
- 每批第一题复用同一个RL session，刷新trajectory identity，验证凭据、incarnation及轨迹编号保留；Agent
  Service会话仍逐题独立，保持TaskAgent协议。
- 恢复后、下次更新前，额外保存同版本状态，逐个全局DCP叶比较模型／Adam张量及param
  groups；按加载报告核对LR、版本及目标RNG。DP改变时核对派生后的RNG，不要求与旧rank相同。读取以单个全局张量为单位，仍需容纳最大张量和读取缓冲。源manifest／sidecar及逐叶指纹保留；源DCP文件仍按checkpoint保留策略清理，旧路径之后可能不存在。
- 核对旧Actor退出、Rollout保留或旧副本退出，以及实际rank设备／活动GPU租约；不根据客户端完成数推断训练完成。
- 只有3次更新、12个持久交接、两次切换的完整计时及首次后端调用证据、无基础设施错误／拒绝、最终完整checkpoint及Trainer／客户端正常退出，才写`result.json`的`status=passed`。审计和等待耗时包含在整组运行时间里，因此不是吞吐或端到端收益测量。

| 文件                                 | 内容                                         |
| ------------------------------------ | -------------------------------------------- |
| `plan.json`／`config.json`           | 固定题序、输入哈希、目标布局及遮蔽密钥的配置 |
| `episodes.jsonl`                     | 逐题结果、inference／tool调用墙钟时间        |
| `transitions.json`                   | 事务回执、进度不变断言及状态／所有权核查结果 |
| `transition-N/{restore,audit}.json`  | 加载计划／报告、同版本全状态比较             |
| `state/reconfiguration-timing.jsonl` | 稀疏切换阶段及首次推理观测                   |
| `result.json`／`failure.json`        | 最终通过结果或需诊断的异常                   |

## 耗时定义

切换记录包含request／transition身份、源／目标布局、权重版本及消费游标；各阶段使用单调时钟记录起止和耗时：checkpoint、drain、Actor释放（含AWEX断开）、Rollout停止及GPU确认、布局应用、Rollout启动至healthy、proxy重绑、Actor初始化、全状态恢复、AWEX建联、发布确认及恢复准入。

`application_and_commit`包含内部阶段和事务提交；`total`还包含源checkpoint及恢复准入。它们是包含关系，不能与内部阶段直接相加。

每个proxy在切换标记后只观测首个接纳的模型调用，记录准入等待、模型响应就绪、后端每次HTTP
dispatch→完整响应返回、请求ID、输入／输出token数和版本。未接到调用的proxy为`null`，不填零；出错调用保留失败状态。

后端耗时包含排队、GPU计算和网络；proxy耗时还包括编码／解析。`armed_to_response_seconds`还包含请求到达前的空闲，以及本验收的CPU审计等待，不能当作纯预热耗时。现有流式输出在后端完整返回后格式化，以上指标都不是GPU
TTFT，也不细分prefill／decode kernel时间。计时为尽力采集；缺失不会中断正常重配置，但此验收不会将缺测视为通过。
