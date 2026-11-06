# FSDP 训练步基准

`areal.tools.benchmark_fsdp` 使用固定的合成 SFT 工作负载，测量真正的 FSDP 训练引擎。
这是一个小型、可复现的测量工具，也是后续性能指导的基础；它不会替正在运行的实验自动选择配置。

## 运行扫描

在仓库根目录运行，并使用安装了项目 CUDA 依赖、FlashAttention 2 且能访问所选 Hugging Face 模型的环境：

```bash
python -m areal.tools.benchmark_fsdp \
  --model-path Qwen/Qwen2.5-0.5B \
  --seq-lens 512,512,512,512 \
  --token-budgets 512,1024 \
  --gradient-checkpointing both \
  --nproc-per-node 1 \
  --warmup-steps 3 \
  --steps 10 \
  --output-dir /tmp/areal-fsdp-benchmark-qwen025
```

每个 sequence length 表示固定全局 batch 中的一行，并对所有试验保持不变。工具只解析一次远程模型版本， 然后针对每种 token budget 与
gradient checkpointing 组合启动全新的 `torchrun` 进程。试验超时或发生 OOM
时会记录结果，并在新的进程组中运行下一种组合。输出目录必须不存在，且必须位于本地模型目录外。可以使用 `--dry-run` 检查配置并 保存计划，而不下载模型或启动
CUDA。默认同时测试开启和关闭 checkpointing；用 `off` 或 `on` 只测试一种。 `--init-from-scratch` 适合本地配置的 smoke
test，不适合生成有代表性的性能数据。

首版支持单机 Linux CUDA FSDP 数据并行，模型限定为采用打包 FlashAttention 2 的稠密 Qwen2、Qwen3 和 Llama
文本模型。TP、上下文/序列并行、MoE、视觉语言模型、CPU 执行和多机运行均不在范围内。全局 batch 大小必须能被 rank 数整除。请选择符合模型上下文长度且适合当前
GPU 显存的 token budget。

## 解读结果

每个试验都会调用真正的 `FSDPEngine.train_batch()` 和 `compute_packed_sft_loss`，覆盖 microbatch 准备、
前向/反向、梯度裁剪和 optimizer step。输入由 `--seed` 确定性生成，并在计时前移动到 GPU。同步后的
墙钟时间不包括模型加载、数据生成、输入拷贝和指标导出，但包含 AReaL 的 loss 监控。若梯度非有限导致 optimizer
更新被跳过，试验会失败，不会把该步计入有效训练结果。

JSON 结果记录解析后的 checkpoint 文件清单、有效 engine 配置、软件和 GPU 版本、每个 rank 的测量结果、
延迟样本、显存峰值及失败详情。Markdown 表格便于快速对比。吞吐量使用有效全局输入 token 数或 loss token 数，除以所有 step 中最慢 rank
墙钟时间的总和。prepared 与 executed microbatch 数会分别报告，因为为了 对齐 FSDP collective，microbatch 较少的
rank 可能还要参与空操作 microbatch。

已分配显存取每个计时训练步后的峰值。已预留显存包括 warmup 和指标导出后保留在缓存分配器中的内存块。 GPU
争用会改变计时结果；比较小幅差异时应控制运行条件。结果只适用于本次工作负载、checkpoint、软件栈 和硬件，不能作为普适最优配置，也不能预测 RL 收敛或 rollout
性能。

如需做更短的安装检查，可使用本地 Qwen2 config/tokenizer 和 `--init-from-scratch`。它仍会执行真实 FSDP
训练步，但权重是随机初始化的，不能代替 checkpoint 基准测试。
