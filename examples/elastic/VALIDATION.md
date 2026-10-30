# Demo 验收

公共运行入口见 [README](README.md)，参考结果与计算口径见 [RESULTS](RESULTS.md)。以下检查独立于项目外层实验资料。

## CPU 回归

在仓库根目录、已准备的 Python 环境内执行：

```bash
python -m pytest -q tests/test_elastic_runtime.py tests/test_elastic_checkpoint.py \
  tests/test_elastic_runtime_state.py tests/test_elastic_demo.py tests/test_elastic_swe.py \
  tests/test_elastic_inline.py tests/test_elastic_progress.py tests/test_elastic_reproduce.py \
  tests/test_staleness_manager.py
```

覆盖队列消费、完整训练状态协议、连续启停、提前恢复、决策期间 batch 到达、评分容错及退出清理。GPU worker 使用协议 fixture；CPU
通过不能替代真实八卡验收。

## 八卡功能检查

需要独占八卡、完整模型及 SWE 环境。仅显式运行下列命令时使用 GPU：

```bash
# 不采样 SWE、不训练；检查 TP4→两组 TP2 权重发布。
python -m tests.elastic_tools.elastic_gpu_weight_sync --model /path/to/model \
  --dataset /path/to/swe-gym.jsonl --swe-runtime /path/to/swe-runtime.json \
  --output /path/to/new-weight-sync-check

# 真实训练三步，在第二步后强制释放／恢复 Actor，再继续训练。
python -m tests.elastic_tools.elastic_gpu_restart --model /path/to/model \
  --dataset /path/to/swe-gym.jsonl --swe-runtime /path/to/swe-runtime.json \
  --output /path/to/new-restart-check --checkpoint-gib 52
```

权重检查核对两个副本的 TP0 参数抽样，不代表全部推理张量核验。启停检查验证完整 DCP 恢复、rollout 保留及续训，不代表 Agent 自主释放或性能收益。

两轮正式运行分别检查：8 更新、64 条训练轨迹、两组推理副本 v8、完整结束及退出码、所属 sandbox 全部删除、Actor 和 GPU 进程退出。elastic
另报告中途实际启停、离线期间 rollout 进展、每次保存／释放／恢复耗时；无周期时如实记录。

保存 `result.json`、`invocation.json`、去敏配置及输入哈希、事件／决策／观测／显存日志。训练 reward 和全部 workflow
的评分异常数量分开统计。协调进程崩溃恢复、DP／TP 动态改规模不在本例范围。

历史实验过程位于项目外层 `artifacts/elastic-inline-20261030/reference-results/`；成本分析和插桩工具位于
`experiments/elastic_lifecycle/`。它们不属于运行 demo 或执行上述检查的依赖。
