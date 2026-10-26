# Actor / Rollout calibration

固定Qwen3-4B-Instruct-2507、BF16和输入seed。两侧分别测量，不运行完整agent任务；
模型质量与原1.7B正式训练另行评估。采样目录见`calibration_cases.py`。

## Actor

每个DP／TP布局启动一次，固定优化器／微批规则，逐case改变实际输入。
`b`是每个DP组的训练序列数，每个trajectory组一条sequence；全局batch为`DP × b`，不是固定batch加速比。
模型和Adam状态随测量step更新，不在case之间恢复；每step重新构造相同身份的输入，重算behavior／prox logp。

```bash
torchrun --standalone --nproc_per_node=1 -m examples.profile.actor_calibration \
  --model "$MODEL_PATH" --output "$RUN_DIR" --dp 1 --tp 1 \
  --cases pilot --warmup 1 --repeat 2
```

`pilot`覆盖1K、4K、4×4K和16K；`all`选择8个基础case，也可传逗号分隔的case名称。
改变DP／TP时同步调整进程数和可见GPU。校准布局为TP1下DP1–8，以及DP1下TP2／4／8； DP2×TP4保留验证。首轮不涉及CP／PP／EP。

输出逐rank的`steps-rankN.jsonl`：输入摘要、实际长度、global／local batch、loss token、
微批与padding、CUDA事件／主机计时、显存峰值和optimizer更新状态。 phase可能互相包含，不能直接相加；step关键路径取各rank的最大耗时。
启动、首次建优化器状态和warmup分别保留。OOM保留错误与已完成记录，不将后续未执行点记为成功。

## Rollout

使用已有AReaL SGLang入口启用batch观测；固定PP=DP=1、关闭prefix复用。

```bash
AREAL_ROLLOUT_TRACE_DIR="$TRACE_DIR" \
python -m areal.v2.inference_service.sglang.launch_server \
  --model-path "$MODEL_PATH" --tp-size 1 --dp-size 1 \
  --host 127.0.0.1 --port "$PORT" --disable-radix-cache \
  --dtype bfloat16 --context-length 65536 --mem-fraction-static 0.75

python -m examples.profile.rollout_calibration \
  --url "$INFERENCE_URL" --output "$RUN_DIR" --cases pilot --repeat 1
```

认证使用环境中的`AREAL_INFERENCE_ADMIN_KEY`；不写入结果文件。
`pilot`覆盖4K单请求、16K并发32的prefill，以及1K输入／2K输出并发32的decode。
`all`为16个基础case。输出定长token请求，逐条验证实际输入／输出长度并记录墙钟时间； 客户端并发不等于引擎batch大小。

batch trace记录请求关联、本次query、实际past、attention pairs、阶段和GPU事件时长。
KV字段是forward入口已分配槽位占用，包含本批分配；不等于请求逻辑上下文总和。
观测字段缺失保留null，不以客户端墙钟替代GPU时间。混合到达／大batch验证集尚待补充。
