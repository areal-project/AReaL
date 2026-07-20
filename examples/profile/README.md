# Qwen3-30B-A3B SFT Profile

这个目录提供一个端到端 SFT profile 样例，用 Qwen3-30B-A3B 和确定性的 fake 128K-token SFT 数据分别采集 kernel
profile 与 memory profile。

## 文件

- `train_sft_profile.py`: SFT 入口。它不读取外部 JSONL，而是用真实 tokenizer 编码一段结构化 SWE/代码修复对话，再重复截断到
  131072 token。
- `qwen3_30b_a3b_sft_profile.yaml`: 1 节点 8 GPU 的 Megatron MoE profile 配置， 默认使用
  `megatron:(attn:d1p2t2c2|ffn:d1p2e4)`。
- `run_qwen3_30b_a3b_sft_profile.sh`: 一键运行 kernel/memory 两类 profile。
- `postprocess_profile.py`: 生成 kernel Chrome trace 视图和 profile summary。

## 快速运行

在仓库根目录执行：

```bash
MODEL_PATH=/path/to/Qwen3-30B-A3B \
FILEROOT=/path/to/shared/experiments \
bash examples/profile/run_qwen3_30b_a3b_sft_profile.sh
```

默认会对 step 0 和 step 1 分别运行 profile；如果 `PROFILE_KINDS=kernel,memory`， 总共会运行 4 个 trial：

- `kernel`: 打开 `perf_tracer.profile_steps`，在 SFT train step 中启动 PyTorch profiler，产出
  CPU/CUDA/GPU kernel trace。
- `memory`: 打开 `memory_profiler.profile_steps`，产出 PyTorch CUDA allocator snapshot。

默认设置：

```bash
PROFILE_STEPS=0,1
# TOTAL_STEPS is unset by default; each trial uses profile_step + 1.
# PROFILE_RANKS is unset by default; the script computes each PP stage's rank0.
PROFILE_KINDS=kernel,memory
PROFILE_FAKE_SEQ_LEN=131072
PROFILE_FAKE_DATASET_SIZE=8
PROFILE_FAKE_LOSS_START_RATIO=0.5
TRAIN_BATCH_SIZE=4
PROFILE_N_MBS=4
AREAL_LOGPROBS_CHUNK_SIZE=128
LM_HEAD_LOSS_CHUNK_SIZE=0
USE_PRECISION_AWARE_OPTIMIZER=true
MAIN_GRADS_DTYPE=bfloat16
STOP_ON_FAILURE=1
```

只跑 kernel profile：

```bash
PROFILE_KINDS=kernel \
MODEL_PATH=/path/to/Qwen3-30B-A3B \
bash examples/profile/run_qwen3_30b_a3b_sft_profile.sh
```

只跑 memory profile，并采集多个 rank：

```bash
PROFILE_KINDS=memory \
PROFILE_RANKS=0,1,4-7 \
MODEL_PATH=/path/to/Qwen3-30B-A3B \
bash examples/profile/run_qwen3_30b_a3b_sft_profile.sh
```

未设置 `PROFILE_RANKS` 时，脚本会根据 `actor.backend` 动态计算每个 pipeline stage 的 rank0；当前默认
`megatron:(attn:d1p2t2c2|ffn:d1p2e4)` 会解析为 `0,4`。显式设置 `PROFILE_RANKS`
会覆盖该脚本默认值；设为空字符串表示采集全部 rank。

## Fake 128K 数据

`train_sft_profile.py` 的 fake 数据生成逻辑有三个约束：

1. 使用目标模型 tokenizer 编码结构化对话文本，而不是直接填充同一个 token。
1. 重复同一段 token 序列直到固定长度 `PROFILE_FAKE_SEQ_LEN=131072`，保证不同 parallel layout 看到完全相同的 token
   内容。
1. `loss_mask` 默认从 50% 位置开始为 1，前半段作为 prompt/context，后半段作为 assistant target。

这样可以避免真实数据 IO 与动态过滤影响 profile，同时保留长上下文、代码块、JSON 片段和自然语言 target 的 tokenizer 分布。

## 产物

脚本默认把运行侧产物放在 `examples/profile/` 下：

```text
examples/profile/profile_data/<timestamp>_qwen3-30b-a3b_fake128k_sft_profile/
  profile_settings.log
  summary.tsv
  qwen3_30b_a3b_fake128k_kernel_step0_<timestamp>/
    launcher.log
    nvidia_smi.csv
    profile_summary.json
    profile_summary.md
    kernel_traces/master/
      traces-r0.chrome.json
      traces-r0.split_clean.chrome.json
      traces-r0.gpu_only.chrome.json
      traces-r0.cpu_only.chrome.json
      traces-r0.cuda_api_only.chrome.json
  qwen3_30b_a3b_fake128k_memory_step0_<timestamp>/
    launcher.log
    nvidia_smi.csv
    profile_summary.json
    profile_summary.md
    memory_snapshots/step_0/
      snapshot_*.pickle
  qwen3_30b_a3b_fake128k_kernel_step1_<timestamp>/
    launcher.log
    nvidia_smi.csv
    profile_summary.json
    profile_summary.md
    kernel_traces/master/
      traces-r0.chrome.json
      traces-r0.split_clean.chrome.json
      traces-r0.gpu_only.chrome.json
      traces-r0.cpu_only.chrome.json
      traces-r0.cuda_api_only.chrome.json
  qwen3_30b_a3b_fake128k_memory_step1_<timestamp>/
    launcher.log
    nvidia_smi.csv
    profile_summary.json
    profile_summary.md
    memory_snapshots/step_1/
      snapshot_*.pickle
```

AReaL 原始日志仍在 `FILEROOT` 下：

```text
${FILEROOT}/logs/<user>/qwen3-30b-a3b-sft-profile/<trial_name>/
  trainer.log
  perf_tracer/<role>/traces-r*.jsonl
  memory_snapshots/step_<profile_step>/snapshot_*.pickle
```

kernel profile 后处理会在 trace 文件旁生成 Chrome trace 视图，并复制一份到
`profile_data/.../kernel_traces/<role>/`：

```text
traces-r0.chrome.json
traces-r0.split_clean.chrome.json
traces-r0.gpu_only.chrome.json
traces-r0.cpu_only.chrome.json
traces-r0.cuda_api_only.chrome.json
```

用 Chrome `chrome://tracing` 或 Perfetto 打开这些 `.chrome.json` 文件即可查看。默认未设置 `PROFILE_RANKS`
时，PP=2 会生成 rank 0 和 rank 4 两套文件，例如 `traces-r0.*.chrome.json` 与
`traces-r4.*.chrome.json`，memory snapshot 也会分别包含 `snapshot_rank00_p0...pickle` 和
`snapshot_rank04_p1...pickle`。

## 单独后处理

如果 profile 已经跑完，只想重新生成 summary 和 kernel trace views：

```bash
python examples/profile/postprocess_profile.py \
  --profile-kind kernel \
  --profile-step 1 \
  --log-dir /path/to/FILEROOT/logs/<user>/qwen3-30b-a3b-sft-profile/<trial_name> \
  --run-dir examples/profile/profile_data/reprocess/<trial_name> \
  --trainer-log /path/to/trainer.log \
  --nvidia-smi-csv /path/to/nvidia_smi.csv
```

## 注意事项

- kernel profile 与 memory profile 建议分开跑；torch profiler 的 memory 记录会扰动 kernel 时间线，本样例默认对
  kernel trial 设置 `AREAL_TORCH_PROFILER_PROFILE_MEMORY=false`。
- 这个 128K profile 样例默认设置 `actor.megatron.ddp.grad_reduce_in_fp32=false`，用于降低 step 1
  steady-state 的梯度规约显存占用。
- `AREAL_LOGPROBS_CHUNK_SIZE` 控制 logprobs/entropy 计算的序列 chunk，默认 128，用来降低 step 1
  steady-state 的 vocab-parallel 临时显存峰值。
- 默认启用 Megatron precision-aware optimizer，并让 TE Adam 直接消费 BF16 distributed grad
  buffer。主参数和两个 Adam moment 仍为 FP32；这避免在 optimizer step 尾部为每个参数额外创建 FP32 grad。设置
  `USE_PRECISION_AWARE_OPTIMIZER=false MAIN_GRADS_DTYPE=float32` 可恢复原路径。
- `PROFILE_RANKS` 控制采集 rank。未设置时脚本会按 `actor.backend` 计算每个 PP stage 的 rank0；为空表示全部 rank；
  `0,2-4` 表示 rank 0、2、3、4。
- Qwen3-30B-A3B 需要 8 GPU profile 环境；没有对应 GPU/模型权重时只能验证脚本和后处理， 无法实际产出 CUDA trace 或 memory
  snapshot。
