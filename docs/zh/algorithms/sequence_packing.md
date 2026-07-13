# 序列打包算法 (Sequence Packing Algorithms)

AReaL 支持在训练期间配置用于微批次 (micro-batch) 分配的序列打包算法。序列打包决定了变长序列如何被分组到微批次中，这直接影响数据并行 (DP) rank 之间的负载均衡以及整体训练吞吐量。

## 支持的算法

| 算法 | Key | 描述 | 复杂度 | 均衡质量 |
|---|---|---|---|---|
| **First Fit Decreasing (FFD)** | `ffd` | 贪心装箱启发式算法。按长度（降序）对序列进行排序，并将每个序列分配到第一个还有剩余容量的桶中。 | O(n log n) | 良好 (Good) |
| **Karmarkar-Karp (KK)** | `kk` | 最大差分法 (Largest Differencing Method)。使用最大堆迭代合并两个最不平衡的部分分区，产生接近最优的均衡效果。 | O(n log n · k) | 极佳 (Excellent) |

## 配置

打包算法由 `MicroBatchSpec` 中的 `packing_algorithm` 字段控制，可以直接在 YAML 配置文件中设置。

### YAML 配置

```yaml
# 在你的实验配置中 (例如：examples/countdown/train_config.yaml)

actor:
  mb_spec:
    max_tokens_per_mb: 8192
    n_mbs: 4
    n_mbs_divisor: 1
    packing_algorithm: kk    # 选项: "ffd" (默认), "kk"
```

### Python API

你也可以通过代码设置该算法：

```python
from areal.api.cli_args import MicroBatchSpec

# 使用 KK 算法
mb_spec = MicroBatchSpec(
    max_tokens_per_mb=8192,
    n_mbs=4,
    packing_algorithm="kk",
)

# 或者更新一个现有的 spec
mb_spec_kk = MicroBatchSpec.new(existing_spec, packing_algorithm="kk")
```

## 何时使用 KK

**KK 的推荐场景：**

- 序列长度变化极大的**大规模 RL 训练**（例如，RLHF，具有开放式生成的 PPO）。KK 显著缩小了负载最重和最轻的 DP rank 之间的差距 (spread)。
- **双峰序列分布 (Bimodal sequence distributions)**，其中极短和极长序列的混合使得贪心打包变得次优。
- **高 DP 并行度**（≥4 个 rank），此时即使是很小的负载不平衡也会因同步屏障导致显著的空闲等待时间。

**何时 FFD 就足够了：**

- 均匀或接近均匀的序列长度。
- 相比均衡度，更关注打包开销的小规模实验。
- 对延迟敏感的推理流水线（FFD 速度略快）。
