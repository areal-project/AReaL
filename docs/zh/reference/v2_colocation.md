# V2 AWEX 共卡

V2 共卡允许 Megatron 训练 worker 和 SGLang 推理 worker 共享 GPU。它复用统一的 AWEX adapter 与
converter，由部署方式选择共卡显存交接及权重传输流程。

## 支持范围

| 组件           | 支持的配置                                         |
| -------------- | -------------------------------------------------- |
| 训练           | Megatron，全量权重                                 |
| 推理           | SGLang，开启 `enable_memory_saver: true`           |
| 调度           | Local scheduler，或 Slurm 整节点 GPU 分配          |
| Slurm GPU 编号 | 与 guard 布局一致的连续数字编号                    |
| Checkpoint     | HF 导出及同步／异步 DCP 恢复                       |
| 暂不支持       | LoRA 共卡、FSDP 共卡、vLLM 共卡、Ray v2 guard 共卡 |

Slurm 分配使用 UUID、重排编号或部分节点 GPU 时，会在 fork 训练 guard 前拒绝启动。 现有 v1 共卡及 v2 分卡继续使用各自的部署和传输路径。

设置 `actor._version: v2`、`rollout._version: v2`，选择 Megatron 和 SGLang 后端， 并将 actor 配置为与
rollout guard 共卡。两侧必须覆盖相同的物理 GPU。训练和推理 并行布局可以不同，但每个 SGLang TP/PP 组必须满足支持的节点内布局。 设置
`actor.megatron.async_save: true` 可使 DCP 磁盘写入与训练重叠。

## Allocator 配置

TMS 0.0.9 和 0.0.10 明确拒绝 `expandable_segments:True`。v1/v2 AWEX rollout
启动时使用统一校验，不会自动改写环境变量。 rollout 应独立于 actor 配置，例如在各个 `rollout.scheduling_spec` 条目的
`env_vars` 中设置：

```yaml
PYTORCH_ALLOC_CONF: expandable_segments:False
PYTORCH_CUDA_ALLOC_CONF: expandable_segments:False
```

PyTorch 2.9 优先读取 `PYTORCH_ALLOC_CONF`，但 TMS 仍会直接检查旧变量 `PYTORCH_CUDA_ALLOC_CONF`，因此
rollout 中两个变量均不能开启 expandable segments。 也可以移除该选项；actor 的 allocator 配置可以独立保留。GPU mask
必须按升序连续排列，重排编号会明确报错，不再自动排序。

## 显存交接

初始化时先创建 rollout guard、初始化训练 worker，然后释放训练显存，再加载辅助模型和 SGLang。 需要 offload 的 v2 辅助模型须使用
Megatron，通过显式权重驻留管理器释放显存， 不依赖从推理进程继承的 TMS hook。

每次训练先暂停 rollout 提交并等待生成结束，释放 rollout 权重、KV cache 和 CUDA graph。 ref、critic、teacher 评分在
actor 恢复显存之前执行。随后 actor 训练、保存 checkpoint、 发布权重并提交 policy version；释放训练显存后，才恢复推理显存。

权重传输或版本提交未成功时，生成保持暂停。导出的 CUDA IPC storage 会保留到 reader 确认消费完成；未确认的导出也会阻止再次发布及 adapter
teardown。最后一步训练后， 推理服务仍可用于评估，但 rollout dispatcher 不会启动下一批训练采样。

## 恢复

DCP 恢复沿用不可变 checkpoint generation 和 `LATEST` 发布协议。恢复过程在 共卡训练阶段内加载选定 generation，复用已有
gateway，发布恢复后的权重并提交版本， 然后重新开放生成。checkpoint 损坏或恢复失败不会静默继续训练。

异步保存返回前完成 CPU payload staging，后续权重更新及 offload 不会修改保存快照。后台保存 finalize 成功后才发布 `LATEST`。
正常退出先等待所有训练 rank 的异步保存完成，再销毁 worker 及其父 rollout guard。等待使用
`actor.request_timeout`（critic 使用对应配置），应覆盖最长预期 checkpoint 写入时间。 等待失败或超时会抛错并保留 worker
资源供诊断，不报告成功退出。进程突然终止可能留下未发布的 generation；恢复仍选择最后成功发布的 checkpoint。
